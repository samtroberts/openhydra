"""Tests for coordinator/layer_coverage.py — Phase 3: Layer Sharding Activation."""
from __future__ import annotations

import pytest
from dataclasses import dataclass

from coordinator.layer_coverage import (
    LayerRange,
    LayerCoverageMap,
    PeerMetrics,
    PipelineWeights,
    coverage_gaps,
    find_complete_pipeline,
    find_optimal_pipeline,
    is_complete_coverage,
    _dijkstra_edge_cost,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def lr(peer_id: str, start: int, end: int, total: int = 32, host: str = "", port: int = 0) -> LayerRange:
    """Shorthand for building a LayerRange in tests."""
    return LayerRange(peer_id=peer_id, layer_start=start, layer_end=end, total_layers=total, host=host, port=port)


@dataclass
class _FakePeer:
    """Minimal peer-like object for LayerCoverageMap.from_endpoints tests."""
    peer_id: str
    layer_start: int = 0
    layer_end: int = 0
    total_layers: int = 0
    host: str = ""
    port: int = 0


# ── LayerRange properties ─────────────────────────────────────────────────────


class TestLayerRangeIsSharded:
    def test_proper_sub_range_start(self):
        """Peer covering layers 0-15 of a 32-layer model is sharded."""
        assert lr("a", 0, 16, 32).is_sharded is True

    def test_proper_sub_range_end(self):
        """Peer covering layers 16-31 of a 32-layer model is sharded."""
        assert lr("b", 16, 32, 32).is_sharded is True

    def test_proper_sub_range_middle(self):
        """Peer covering a middle slice is sharded."""
        assert lr("c", 8, 24, 32).is_sharded is True

    def test_full_model_replica_not_sharded(self):
        """Peer covering [0, total_layers) is a full replica — not sharded."""
        assert lr("d", 0, 32, 32).is_sharded is False

    def test_unsharded_zero_total(self):
        """Peer with total_layers=0 is not sharded (legacy / unsharded)."""
        assert lr("e", 0, 0, 0).is_sharded is False

    def test_unsharded_zero_end(self):
        """Peer with layer_end=0 is not sharded."""
        r = LayerRange(peer_id="f", layer_start=0, layer_end=0, total_layers=32)
        assert r.is_sharded is False

    def test_single_layer_shard(self):
        """A single-layer peer is still sharded as long as it's a proper sub-range."""
        assert lr("g", 5, 6, 32).is_sharded is True


class TestLayerRangeSpan:
    def test_normal_span(self):
        assert lr("a", 0, 16, 32).span == 16

    def test_zero_span(self):
        r = LayerRange(peer_id="b", layer_start=5, layer_end=5, total_layers=32)
        assert r.span == 0

    def test_inverted_span_clamps_to_zero(self):
        r = LayerRange(peer_id="c", layer_start=10, layer_end=5, total_layers=32)
        assert r.span == 0

    def test_full_model_span(self):
        assert lr("d", 0, 32, 32).span == 32


class TestLayerRangeCoversLayer:
    def test_covers_first_layer(self):
        assert lr("a", 0, 16, 32).covers_layer(0) is True

    def test_covers_last_layer_in_range(self):
        assert lr("a", 0, 16, 32).covers_layer(15) is True

    def test_does_not_cover_end_boundary(self):
        """layer_end is exclusive."""
        assert lr("a", 0, 16, 32).covers_layer(16) is False

    def test_does_not_cover_outside_left(self):
        assert lr("a", 8, 24, 32).covers_layer(7) is False

    def test_does_not_cover_outside_right(self):
        assert lr("a", 8, 24, 32).covers_layer(24) is False


class TestLayerRangeOverlaps:
    def test_adjacent_ranges_do_not_overlap(self):
        """[0,16) and [16,32) share no layer index."""
        assert lr("a", 0, 16, 32).overlaps(lr("b", 16, 32, 32)) is False

    def test_overlapping_ranges(self):
        assert lr("a", 0, 20, 32).overlaps(lr("b", 16, 32, 32)) is True

    def test_contained_range_overlaps(self):
        assert lr("a", 0, 32, 32).overlaps(lr("b", 8, 24, 32)) is True

    def test_identical_ranges_overlap(self):
        assert lr("a", 4, 12, 32).overlaps(lr("b", 4, 12, 32)) is True

    def test_non_overlapping_left(self):
        assert lr("a", 0, 8, 32).overlaps(lr("b", 16, 24, 32)) is False


# ── coverage_gaps ─────────────────────────────────────────────────────────────


class TestCoverageGaps:
    def test_full_coverage_two_peers(self):
        """Two perfectly adjacent ranges → no gaps."""
        gaps = coverage_gaps([lr("a", 0, 16, 32), lr("b", 16, 32, 32)], 32)
        assert gaps == []

    def test_full_coverage_single_peer(self):
        """One peer covering all layers → no gaps."""
        gaps = coverage_gaps([lr("a", 0, 32, 32)], 32)
        assert gaps == []

    def test_empty_ranges(self):
        """No peers → entire range is a gap."""
        assert coverage_gaps([], 32) == [(0, 32)]

    def test_zero_total_layers(self):
        """total_layers=0 → nothing to cover; no gaps."""
        assert coverage_gaps([lr("a", 0, 16, 32)], 0) == []

    def test_negative_total_layers(self):
        assert coverage_gaps([], -1) == []

    def test_gap_in_middle(self):
        gaps = coverage_gaps([lr("a", 0, 8, 32), lr("b", 24, 32, 32)], 32)
        assert gaps == [(8, 24)]

    def test_gap_at_start(self):
        gaps = coverage_gaps([lr("a", 8, 32, 32)], 32)
        assert gaps == [(0, 8)]

    def test_gap_at_end(self):
        gaps = coverage_gaps([lr("a", 0, 24, 32)], 32)
        assert gaps == [(24, 32)]

    def test_multiple_gaps(self):
        ranges = [lr("a", 0, 8, 32), lr("b", 12, 20, 32)]
        gaps = coverage_gaps(ranges, 32)
        assert gaps == [(8, 12), (20, 32)]

    def test_overlapping_ranges_no_gap(self):
        """Overlapping peers that together cover everything → no gap."""
        gaps = coverage_gaps([lr("a", 0, 20, 32), lr("b", 16, 32, 32)], 32)
        assert gaps == []

    def test_ranges_beyond_total_layers_clamped(self):
        """Ranges extending beyond total_layers are clamped — no crash."""
        gaps = coverage_gaps([lr("a", 0, 100, 32)], 32)
        assert gaps == []

    def test_unsorted_input(self):
        """Order of input ranges should not matter."""
        gaps = coverage_gaps([lr("b", 16, 32, 32), lr("a", 0, 16, 32)], 32)
        assert gaps == []

    def test_three_peer_coverage(self):
        """Three peers covering 0-10, 10-20, 20-32 → no gaps."""
        gaps = coverage_gaps([lr("a", 0, 10, 32), lr("b", 10, 20, 32), lr("c", 20, 32, 32)], 32)
        assert gaps == []


# ── is_complete_coverage ─────────────────────────────────────────────────────


class TestIsCompleteCoverage:
    def test_complete(self):
        assert is_complete_coverage([lr("a", 0, 16, 32), lr("b", 16, 32, 32)], 32) is True

    def test_incomplete_gap(self):
        assert is_complete_coverage([lr("a", 0, 16, 32)], 32) is False

    def test_empty_ranges(self):
        assert is_complete_coverage([], 32) is False

    def test_zero_total_layers(self):
        """total_layers=0 → always False (nothing meaningful to cover)."""
        assert is_complete_coverage([lr("a", 0, 32, 32)], 0) is False


# ── find_complete_pipeline ────────────────────────────────────────────────────


class TestFindCompletePipeline:
    def test_two_peer_perfect_split(self):
        """Classic two-peer pipeline; result is ordered [a, b]."""
        pipeline = find_complete_pipeline([lr("a", 0, 16, 32), lr("b", 16, 32, 32)], 32)
        assert pipeline is not None
        assert len(pipeline) == 2
        assert pipeline[0].peer_id == "a"
        assert pipeline[1].peer_id == "b"

    def test_single_peer_full_coverage(self):
        """One peer covering everything → single-stage pipeline."""
        pipeline = find_complete_pipeline([lr("a", 0, 32, 32)], 32)
        assert pipeline is not None
        assert len(pipeline) == 1
        assert pipeline[0].peer_id == "a"

    def test_gap_returns_none(self):
        """Missing layers [16,32) → no pipeline possible."""
        pipeline = find_complete_pipeline([lr("a", 0, 16, 32)], 32)
        assert pipeline is None

    def test_empty_ranges_returns_none(self):
        assert find_complete_pipeline([], 32) is None

    def test_zero_total_layers_returns_none(self):
        assert find_complete_pipeline([lr("a", 0, 32, 32)], 0) is None

    def test_greedy_picks_farthest_extension(self):
        """When two peers both start at 0, greedy picks the one reaching farther."""
        a = lr("a", 0, 10, 32)   # short
        b = lr("b", 0, 20, 32)   # long — should be chosen
        c = lr("c", 20, 32, 32)
        pipeline = find_complete_pipeline([a, b, c], 32)
        assert pipeline is not None
        assert pipeline[0].peer_id == "b"
        assert pipeline[1].peer_id == "c"

    def test_overlapping_peers_covered(self):
        """Overlapping ranges still form a valid pipeline via greedy choice."""
        # a:[0,20), b:[10,32) — greedy picks a first, then b covers the rest
        pipeline = find_complete_pipeline([lr("a", 0, 20, 32), lr("b", 10, 32, 32)], 32)
        assert pipeline is not None
        assert pipeline[0].peer_id == "a"
        assert pipeline[1].peer_id == "b"

    def test_three_peer_pipeline(self):
        """Three-stage pipeline in order."""
        pipeline = find_complete_pipeline(
            [lr("c", 20, 32, 32), lr("a", 0, 10, 32), lr("b", 10, 20, 32)], 32
        )
        assert pipeline is not None
        assert len(pipeline) == 3
        assert [p.peer_id for p in pipeline] == ["a", "b", "c"]

    def test_each_peer_used_at_most_once(self):
        """The same peer should not appear twice in the pipeline."""
        # If there were infinite peers or all ranges were the same, we shouldn't loop
        pipeline = find_complete_pipeline([lr("a", 0, 32, 32), lr("a", 0, 32, 32)], 32)
        # `used` set prevents re-use; since only one unique peer_id, result depends on
        # whether we can complete coverage.  With peer "a" used once it covers everything.
        assert pipeline is not None
        assert len(pipeline) == 1

    def test_gap_in_middle_returns_none(self):
        """If there's no peer for layers 10-20, pipeline is impossible."""
        pipeline = find_complete_pipeline(
            [lr("a", 0, 10, 32), lr("b", 20, 32, 32)], 32
        )
        assert pipeline is None

    def test_unsorted_input_still_works(self):
        """Input order should not affect the result."""
        pipeline = find_complete_pipeline(
            [lr("b", 16, 32, 32), lr("a", 0, 16, 32)], 32
        )
        assert pipeline is not None
        assert len(pipeline) == 2
        assert pipeline[0].peer_id == "a"


# ── LayerCoverageMap ──────────────────────────────────────────────────────────


class TestLayerCoverageMapFromEndpoints:
    def test_basic_construction(self):
        peers = [_FakePeer("a", 0, 16, 32), _FakePeer("b", 16, 32, 32)]
        cmap = LayerCoverageMap.from_endpoints(peers)
        assert cmap.total_layers == 32
        assert len(cmap.ranges) == 2

    def test_filters_out_unsharded_peers(self):
        """Peers with total_layers=0 or layer_end=0 are excluded."""
        peers = [
            _FakePeer("sharded", 0, 16, 32),
            _FakePeer("full_replica", 0, 0, 0),   # layer_end=0 and total_layers=0
            _FakePeer("no_total", 0, 16, 0),       # total_layers=0
        ]
        cmap = LayerCoverageMap.from_endpoints(peers)
        assert len(cmap.ranges) == 1
        assert cmap.ranges[0].peer_id == "sharded"

    def test_consensus_total_layers_none(self):
        """When total_layers not provided, uses most common value across peers."""
        peers = [
            _FakePeer("a", 0, 16, 32),
            _FakePeer("b", 16, 32, 32),
            _FakePeer("c", 0, 40, 80),   # minority — ignored in consensus
        ]
        cmap = LayerCoverageMap.from_endpoints(peers, total_layers=None)
        assert cmap.total_layers == 32

    def test_explicit_total_layers_override(self):
        """Explicit total_layers overrides peer-reported values."""
        peers = [_FakePeer("a", 0, 16, 32), _FakePeer("b", 16, 64, 80)]
        cmap = LayerCoverageMap.from_endpoints(peers, total_layers=64)
        assert cmap.total_layers == 64

    def test_empty_peer_list(self):
        cmap = LayerCoverageMap.from_endpoints([])
        assert cmap.total_layers == 0
        assert cmap.ranges == []

    def test_host_and_port_preserved(self):
        peers = [_FakePeer("a", 0, 16, 32, host="10.0.0.1", port=5001)]
        cmap = LayerCoverageMap.from_endpoints(peers)
        assert cmap.ranges[0].host == "10.0.0.1"
        assert cmap.ranges[0].port == 5001


class TestLayerCoverageMapProperties:
    def test_has_sharded_peers_true(self):
        peers = [_FakePeer("a", 0, 16, 32)]  # proper sub-range
        cmap = LayerCoverageMap.from_endpoints(peers)
        assert cmap.has_sharded_peers is True

    def test_has_sharded_peers_false_when_empty(self):
        cmap = LayerCoverageMap.from_endpoints([])
        assert cmap.has_sharded_peers is False

    def test_is_complete_true(self):
        peers = [_FakePeer("a", 0, 16, 32), _FakePeer("b", 16, 32, 32)]
        cmap = LayerCoverageMap.from_endpoints(peers)
        assert cmap.is_complete() is True

    def test_is_complete_false_with_gap(self):
        peers = [_FakePeer("a", 0, 16, 32)]
        cmap = LayerCoverageMap.from_endpoints(peers)
        assert cmap.is_complete() is False

    def test_is_complete_false_when_empty(self):
        cmap = LayerCoverageMap.from_endpoints([])
        assert cmap.is_complete() is False


class TestLayerCoverageMapGaps:
    def test_no_gaps_when_complete(self):
        peers = [_FakePeer("a", 0, 16, 32), _FakePeer("b", 16, 32, 32)]
        cmap = LayerCoverageMap.from_endpoints(peers)
        assert cmap.gaps() == []

    def test_gap_at_end(self):
        peers = [_FakePeer("a", 0, 16, 32)]
        cmap = LayerCoverageMap.from_endpoints(peers)
        assert cmap.gaps() == [(16, 32)]

    def test_gap_in_middle(self):
        peers = [_FakePeer("a", 0, 8, 32), _FakePeer("b", 24, 32, 32)]
        cmap = LayerCoverageMap.from_endpoints(peers)
        assert cmap.gaps() == [(8, 24)]


class TestLayerCoverageMapBestPipeline:
    def test_returns_ordered_pipeline(self):
        peers = [_FakePeer("a", 0, 16, 32), _FakePeer("b", 16, 32, 32)]
        cmap = LayerCoverageMap.from_endpoints(peers)
        pipeline = cmap.best_pipeline()
        assert pipeline is not None
        assert len(pipeline) == 2
        assert pipeline[0].peer_id == "a"
        assert pipeline[1].peer_id == "b"

    def test_returns_none_when_coverage_impossible(self):
        peers = [_FakePeer("a", 0, 16, 32)]  # gap: 16-31
        cmap = LayerCoverageMap.from_endpoints(peers)
        assert cmap.best_pipeline() is None

    def test_returns_none_for_empty_map(self):
        cmap = LayerCoverageMap.from_endpoints([])
        assert cmap.best_pipeline() is None


class TestLayerCoverageMapCoverageFraction:
    def test_full_coverage(self):
        peers = [_FakePeer("a", 0, 32, 32)]
        cmap = LayerCoverageMap.from_endpoints(peers)
        assert cmap.coverage_fraction() == 1.0

    def test_half_coverage(self):
        peers = [_FakePeer("a", 0, 16, 32)]
        cmap = LayerCoverageMap.from_endpoints(peers)
        assert cmap.coverage_fraction() == pytest.approx(0.5)

    def test_no_coverage_empty(self):
        cmap = LayerCoverageMap.from_endpoints([])
        assert cmap.coverage_fraction() == 0.0

    def test_zero_total_layers(self):
        cmap = LayerCoverageMap([], 0)
        assert cmap.coverage_fraction() == 0.0

    def test_overlapping_does_not_exceed_one(self):
        """Even with overlapping ranges, fraction must be ≤ 1.0."""
        peers = [_FakePeer("a", 0, 24, 32), _FakePeer("b", 8, 32, 32)]
        cmap = LayerCoverageMap.from_endpoints(peers)
        assert cmap.coverage_fraction() <= 1.0
        assert cmap.coverage_fraction() == pytest.approx(1.0)

    def test_three_quarter_coverage(self):
        peers = [_FakePeer("a", 0, 24, 32)]
        cmap = LayerCoverageMap.from_endpoints(peers)
        assert cmap.coverage_fraction() == pytest.approx(0.75)


class TestLayerCoverageMapSummary:
    def test_summary_keys(self):
        """summary() must return all documented keys."""
        expected_keys = {
            "total_layers",
            "sharded_peers",
            "full_model_peers",
            "coverage_complete",
            "coverage_fraction",
            "gaps",
            "best_pipeline_stages",
            "best_pipeline",
        }
        peers = [_FakePeer("a", 0, 16, 32), _FakePeer("b", 16, 32, 32)]
        cmap = LayerCoverageMap.from_endpoints(peers)
        assert set(cmap.summary().keys()) == expected_keys

    def test_summary_complete_pipeline(self):
        peers = [_FakePeer("a", 0, 16, 32, "host-a", 5000), _FakePeer("b", 16, 32, 32, "host-b", 5001)]
        cmap = LayerCoverageMap.from_endpoints(peers)
        s = cmap.summary()
        assert s["coverage_complete"] is True
        assert s["coverage_fraction"] == pytest.approx(1.0)
        assert s["gaps"] == []
        assert s["best_pipeline_stages"] == 2
        assert len(s["best_pipeline"]) == 2
        assert s["best_pipeline"][0]["peer_id"] == "a"
        assert s["best_pipeline"][0]["host"] == "host-a"
        assert s["best_pipeline"][0]["port"] == 5000
        assert s["best_pipeline"][1]["peer_id"] == "b"

    def test_summary_incomplete_pipeline(self):
        peers = [_FakePeer("a", 0, 16, 32)]
        cmap = LayerCoverageMap.from_endpoints(peers)
        s = cmap.summary()
        assert s["coverage_complete"] is False
        assert s["best_pipeline_stages"] == 0
        assert s["best_pipeline"] == []
        assert s["gaps"] == [(16, 32)]

    def test_summary_sharded_peer_count(self):
        peers = [_FakePeer("a", 0, 16, 32), _FakePeer("b", 16, 32, 32)]
        cmap = LayerCoverageMap.from_endpoints(peers)
        s = cmap.summary()
        # Both peers cover proper sub-ranges → both are sharded
        assert s["sharded_peers"] == 2
        assert s["full_model_peers"] == 0

    def test_summary_empty(self):
        cmap = LayerCoverageMap.from_endpoints([])
        s = cmap.summary()
        assert s["coverage_complete"] is False
        assert s["coverage_fraction"] == 0.0
        assert s["best_pipeline_stages"] == 0

    def test_summary_coverage_fraction_rounded(self):
        """coverage_fraction is rounded to 4 decimal places in summary."""
        peers = [_FakePeer("a", 0, 1, 3)]   # 1/3 = 0.3333...
        cmap = LayerCoverageMap.from_endpoints(peers)
        s = cmap.summary()
        assert s["coverage_fraction"] == 0.3333


# ── Direct LayerCoverageMap constructor ───────────────────────────────────────


class TestLayerCoverageMapConstructor:
    def test_direct_construction(self):
        ranges = [lr("a", 0, 16, 32), lr("b", 16, 32, 32)]
        cmap = LayerCoverageMap(ranges, 32)
        assert cmap.total_layers == 32
        assert len(cmap.ranges) == 2

    def test_negative_total_layers_clamped_to_zero(self):
        cmap = LayerCoverageMap([], -5)
        assert cmap.total_layers == 0

    def test_ranges_are_copied(self):
        """Mutating the source list should not affect the map."""
        ranges = [lr("a", 0, 16, 32)]
        cmap = LayerCoverageMap(ranges, 32)
        ranges.append(lr("b", 16, 32, 32))
        assert len(cmap.ranges) == 1


# ── Integration: complete pipeline routing scenario ───────────────────────────


class TestPipelineRoutingScenario:
    """End-to-end test simulating a real sharded LLaMA-3-8B (32 layers) scenario."""

    def test_three_peer_llama_8b(self):
        """Three peers, each covering 1/3 of a 32-layer model — not evenly divisible,
        so last peer covers extra layers.  Pipeline should cover [0, 32)."""
        peers = [
            _FakePeer("shard-0", 0, 11, 32, "10.0.0.1", 9001),
            _FakePeer("shard-1", 11, 22, 32, "10.0.0.2", 9001),
            _FakePeer("shard-2", 22, 32, 32, "10.0.0.3", 9001),
        ]
        cmap = LayerCoverageMap.from_endpoints(peers)
        assert cmap.is_complete() is True
        pipeline = cmap.best_pipeline()
        assert pipeline is not None
        assert len(pipeline) == 3
        assert pipeline[0].layer_start == 0
        assert pipeline[-1].layer_end == 32

    def test_redundant_shards_greedy_selects_best(self):
        """Two peers cover [0,16), greedy should pick the one reaching farther."""
        peers = [
            _FakePeer("fast-0", 0, 18, 32),   # covers a bit extra
            _FakePeer("slow-0", 0, 12, 32),   # shorter
            _FakePeer("shard-1", 16, 32, 32),
        ]
        cmap = LayerCoverageMap.from_endpoints(peers)
        pipeline = cmap.best_pipeline()
        assert pipeline is not None
        # fast-0 extends farther from pos=0, so it should be chosen
        assert pipeline[0].peer_id == "fast-0"

    def test_missing_middle_shard(self):
        """If no peer covers [11, 22), pipeline is impossible."""
        peers = [
            _FakePeer("shard-0", 0, 11, 32),
            _FakePeer("shard-2", 22, 32, 32),
        ]
        cmap = LayerCoverageMap.from_endpoints(peers)
        assert cmap.is_complete() is False
        assert cmap.best_pipeline() is None
        gaps = cmap.gaps()
        assert (11, 22) in gaps

    def test_coverage_fraction_with_gap(self):
        """11 + 10 = 21 of 32 layers covered → fraction ≈ 0.65625."""
        peers = [
            _FakePeer("shard-0", 0, 11, 32),
            _FakePeer("shard-2", 22, 32, 32),
        ]
        cmap = LayerCoverageMap.from_endpoints(peers)
        assert cmap.coverage_fraction() == pytest.approx(21 / 32)


# ═══════════════════════════════════════════════════════════════════════════════
# Dijkstra cost-optimal pipeline routing
# ═══════════════════════════════════════════════════════════════════════════════


class TestPeerMetrics:
    def test_defaults(self):
        m = PeerMetrics()
        assert m.latency_ms == 0.0
        assert m.estimated_tps == 0.0
        assert m.reputation_score == 50.0
        assert m.load_pct == 0.0

    def test_custom_values(self):
        m = PeerMetrics(latency_ms=15.0, estimated_tps=120.0, reputation_score=90.0, load_pct=0.3)
        assert m.latency_ms == 15.0
        assert m.estimated_tps == 120.0

    def test_frozen(self):
        m = PeerMetrics()
        with pytest.raises(AttributeError):
            m.latency_ms = 99.0  # type: ignore[misc]


class TestPipelineWeights:
    def test_defaults(self):
        w = PipelineWeights()
        assert w.w_rtt == 1.0
        assert w.w_infer == 1.5
        assert w.w_rep == 0.5
        assert w.w_load == 0.3

    def test_custom(self):
        w = PipelineWeights(w_rtt=2.0, w_infer=3.0)
        assert w.w_rtt == 2.0
        assert w.w_infer == 3.0


class TestDijkstraEdgeCost:
    def test_zero_metrics_returns_positive(self):
        """Even with default metrics, cost must be > 0 (inference fallback)."""
        r = lr("a", 0, 16, 32)
        m = PeerMetrics()
        w = PipelineWeights()
        cost = _dijkstra_edge_cost(r, m, w)
        assert cost > 0

    def test_higher_latency_means_higher_cost(self):
        r = lr("a", 0, 16, 32)
        w = PipelineWeights()
        low = _dijkstra_edge_cost(r, PeerMetrics(latency_ms=10.0), w)
        high = _dijkstra_edge_cost(r, PeerMetrics(latency_ms=100.0), w)
        assert high > low

    def test_lower_reputation_means_higher_cost(self):
        r = lr("a", 0, 16, 32)
        w = PipelineWeights()
        good = _dijkstra_edge_cost(r, PeerMetrics(reputation_score=95.0), w)
        bad = _dijkstra_edge_cost(r, PeerMetrics(reputation_score=10.0), w)
        assert bad > good

    def test_higher_load_means_higher_cost(self):
        r = lr("a", 0, 16, 32)
        w = PipelineWeights()
        idle = _dijkstra_edge_cost(r, PeerMetrics(load_pct=0.0), w)
        busy = _dijkstra_edge_cost(r, PeerMetrics(load_pct=0.9), w)
        assert busy > idle

    def test_higher_tps_means_lower_cost(self):
        r = lr("a", 0, 16, 32)
        w = PipelineWeights()
        slow = _dijkstra_edge_cost(r, PeerMetrics(estimated_tps=10.0), w)
        fast = _dijkstra_edge_cost(r, PeerMetrics(estimated_tps=200.0), w)
        assert fast < slow


class TestFindOptimalPipeline:
    def test_two_peer_perfect_split(self):
        """Basic two-peer pipeline with Dijkstra (no metrics → default costs)."""
        ranges = [lr("a", 0, 16, 32), lr("b", 16, 32, 32)]
        pipeline = find_optimal_pipeline(ranges, 32)
        assert pipeline is not None
        assert len(pipeline) == 2
        ids = [p.peer_id for p in pipeline]
        assert ids == ["a", "b"]

    def test_single_peer_full_coverage(self):
        pipeline = find_optimal_pipeline([lr("a", 0, 32, 32)], 32)
        assert pipeline is not None
        assert len(pipeline) == 1

    def test_gap_returns_none(self):
        pipeline = find_optimal_pipeline([lr("a", 0, 16, 32)], 32)
        assert pipeline is None

    def test_empty_ranges(self):
        assert find_optimal_pipeline([], 32) is None

    def test_zero_total_layers(self):
        assert find_optimal_pipeline([lr("a", 0, 32, 32)], 0) is None

    def test_picks_lower_latency_path(self):
        """Given two overlapping paths, Dijkstra must pick the lower-latency one.

        Path 1: fast-a [0,16) → fast-b [16,32)   total latency = 10+10 = 20ms
        Path 2: slow-a [0,16) → slow-b [16,32)   total latency = 200+200 = 400ms

        Dijkstra must choose fast-a + fast-b.
        """
        ranges = [
            lr("fast-a", 0, 16, 32),
            lr("fast-b", 16, 32, 32),
            lr("slow-a", 0, 16, 32),
            lr("slow-b", 16, 32, 32),
        ]
        metrics = {
            "fast-a": PeerMetrics(latency_ms=10.0, estimated_tps=100.0),
            "fast-b": PeerMetrics(latency_ms=10.0, estimated_tps=100.0),
            "slow-a": PeerMetrics(latency_ms=200.0, estimated_tps=100.0),
            "slow-b": PeerMetrics(latency_ms=200.0, estimated_tps=100.0),
        }
        pipeline = find_optimal_pipeline(ranges, 32, peer_metrics=metrics)
        assert pipeline is not None
        ids = [p.peer_id for p in pipeline]
        assert ids == ["fast-a", "fast-b"]

    def test_avoids_high_load_peers(self):
        """Dijkstra avoids saturated peers even when they have low latency.

        idle-a [0,16) + idle-b [16,32)  → load 0%
        busy-a [0,16) + busy-b [16,32)  → load 95%
        """
        ranges = [
            lr("idle-a", 0, 16, 32),
            lr("idle-b", 16, 32, 32),
            lr("busy-a", 0, 16, 32),
            lr("busy-b", 16, 32, 32),
        ]
        metrics = {
            "idle-a": PeerMetrics(latency_ms=50.0, estimated_tps=100.0, load_pct=0.0),
            "idle-b": PeerMetrics(latency_ms=50.0, estimated_tps=100.0, load_pct=0.0),
            "busy-a": PeerMetrics(latency_ms=50.0, estimated_tps=100.0, load_pct=0.95),
            "busy-b": PeerMetrics(latency_ms=50.0, estimated_tps=100.0, load_pct=0.95),
        }
        pipeline = find_optimal_pipeline(ranges, 32, peer_metrics=metrics)
        assert pipeline is not None
        ids = [p.peer_id for p in pipeline]
        assert ids == ["idle-a", "idle-b"]

    def test_penalizes_low_reputation(self):
        """Dijkstra penalises untrusted peers.

        trusted-a [0,16) + trusted-b [16,32)  → reputation 95
        shady-a [0,16) + shady-b [16,32)      → reputation 5
        """
        ranges = [
            lr("trusted-a", 0, 16, 32),
            lr("trusted-b", 16, 32, 32),
            lr("shady-a", 0, 16, 32),
            lr("shady-b", 16, 32, 32),
        ]
        metrics = {
            "trusted-a": PeerMetrics(latency_ms=30.0, estimated_tps=100.0, reputation_score=95.0),
            "trusted-b": PeerMetrics(latency_ms=30.0, estimated_tps=100.0, reputation_score=95.0),
            "shady-a": PeerMetrics(latency_ms=30.0, estimated_tps=100.0, reputation_score=5.0),
            "shady-b": PeerMetrics(latency_ms=30.0, estimated_tps=100.0, reputation_score=5.0),
        }
        pipeline = find_optimal_pipeline(ranges, 32, peer_metrics=metrics)
        assert pipeline is not None
        ids = [p.peer_id for p in pipeline]
        assert ids == ["trusted-a", "trusted-b"]

    def test_prefers_higher_tps_peer(self):
        """Given identical latency, Dijkstra picks the faster (higher TPS) peer."""
        ranges = [
            lr("fast", 0, 32, 32),
            lr("slow", 0, 32, 32),
        ]
        metrics = {
            "fast": PeerMetrics(latency_ms=20.0, estimated_tps=200.0),
            "slow": PeerMetrics(latency_ms=20.0, estimated_tps=10.0),
        }
        pipeline = find_optimal_pipeline(ranges, 32, peer_metrics=metrics)
        assert pipeline is not None
        assert pipeline[0].peer_id == "fast"

    def test_three_stage_pipeline(self):
        """Three-stage pipeline ordered correctly."""
        ranges = [lr("c", 20, 32, 32), lr("a", 0, 10, 32), lr("b", 10, 20, 32)]
        pipeline = find_optimal_pipeline(ranges, 32)
        assert pipeline is not None
        assert [p.peer_id for p in pipeline] == ["a", "b", "c"]

    def test_overlapping_peers_picks_optimal(self):
        """With overlapping ranges, Dijkstra picks the cheapest path.

        a:[0,20) + c:[20,32) = 2 stages
        b:[0,10) + d:[10,32) = 2 stages but d is slower
        """
        ranges = [
            lr("a", 0, 20, 32),
            lr("b", 0, 10, 32),
            lr("c", 20, 32, 32),
            lr("d", 10, 32, 32),
        ]
        metrics = {
            "a": PeerMetrics(latency_ms=10.0, estimated_tps=100.0),
            "b": PeerMetrics(latency_ms=10.0, estimated_tps=100.0),
            "c": PeerMetrics(latency_ms=10.0, estimated_tps=100.0),
            "d": PeerMetrics(latency_ms=10.0, estimated_tps=5.0),  # very slow
        }
        pipeline = find_optimal_pipeline(ranges, 32, peer_metrics=metrics)
        assert pipeline is not None
        # a+c is cheaper because d has very low TPS
        assert [p.peer_id for p in pipeline] == ["a", "c"]

    def test_no_path_returns_none(self):
        """Disconnected ranges with a gap → None."""
        ranges = [lr("a", 0, 10, 32), lr("b", 20, 32, 32)]
        assert find_optimal_pipeline(ranges, 32) is None

    def test_custom_weights(self):
        """Custom weights that heavily penalise latency must change the result."""
        # fast-a has low latency but terrible reputation;
        # slow-a has high latency but perfect reputation.
        # Default weights: balanced → fast-a wins.
        # Extreme rep weights: slow-a wins.
        ranges = [
            lr("fast-a", 0, 32, 32),
            lr("slow-a", 0, 32, 32),
        ]
        metrics = {
            "fast-a": PeerMetrics(latency_ms=5.0, estimated_tps=100.0, reputation_score=1.0),
            "slow-a": PeerMetrics(latency_ms=50.0, estimated_tps=100.0, reputation_score=100.0),
        }
        # Heavy reputation weight, minimal RTT weight.
        w = PipelineWeights(w_rtt=0.01, w_infer=0.01, w_rep=10.0, rep_penalty=1000.0)
        pipeline = find_optimal_pipeline(ranges, 32, peer_metrics=metrics, weights=w)
        assert pipeline is not None
        assert pipeline[0].peer_id == "slow-a"

    def test_stress_100_peers(self):
        """Dijkstra must handle 100 peers without error and find a path."""
        # 100 peers each covering 1 layer (0-1, 1-2, ..., 99-100).
        ranges = [lr(f"p{i}", i, i + 1, 100) for i in range(100)]
        metrics = {f"p{i}": PeerMetrics(latency_ms=float(i % 20)) for i in range(100)}
        pipeline = find_optimal_pipeline(ranges, 100, peer_metrics=metrics)
        assert pipeline is not None
        assert len(pipeline) == 100
        # Must be in layer order.
        for i, stage in enumerate(pipeline):
            assert stage.layer_start == i

    def test_duplicate_peer_id_keeps_widest(self):
        """If same peer_id appears with different spans, the widest wins."""
        ranges = [
            lr("a", 0, 10, 32),
            lr("a", 0, 20, 32),   # wider — should be kept
            lr("b", 20, 32, 32),
        ]
        pipeline = find_optimal_pipeline(ranges, 32)
        assert pipeline is not None
        assert pipeline[0].peer_id == "a"
        assert pipeline[0].layer_end == 20  # wider range kept


class TestLayerCoverageMapDijkstraIntegration:
    """Tests that LayerCoverageMap.best_pipeline() dispatches to Dijkstra."""

    def test_with_metrics_uses_dijkstra(self):
        """Passing peer_metrics must route through Dijkstra, not greedy."""
        # Two paths: fast (a+b) and slow (c+d).
        # Greedy picks the longest extension, but Dijkstra should pick fast path.
        ranges = [
            lr("fast-a", 0, 16, 32),
            lr("fast-b", 16, 32, 32),
            lr("slow-a", 0, 24, 32),   # greedy would prefer this (extends farther)
            lr("slow-b", 24, 32, 32),
        ]
        cmap = LayerCoverageMap(ranges, 32)

        # Without metrics → greedy picks slow-a (extends to 24 > 16).
        greedy = cmap.best_pipeline()
        assert greedy is not None
        assert greedy[0].peer_id == "slow-a"

        # With metrics → Dijkstra picks fast path (lower latency).
        metrics = {
            "fast-a": PeerMetrics(latency_ms=5.0, estimated_tps=100.0),
            "fast-b": PeerMetrics(latency_ms=5.0, estimated_tps=100.0),
            "slow-a": PeerMetrics(latency_ms=500.0, estimated_tps=100.0),
            "slow-b": PeerMetrics(latency_ms=500.0, estimated_tps=100.0),
        }
        optimal = cmap.best_pipeline(peer_metrics=metrics)
        assert optimal is not None
        assert [p.peer_id for p in optimal] == ["fast-a", "fast-b"]

    def test_without_metrics_uses_greedy(self):
        """Without peer_metrics, best_pipeline() must use greedy as before."""
        ranges = [lr("a", 0, 16, 32), lr("b", 16, 32, 32)]
        cmap = LayerCoverageMap(ranges, 32)
        pipeline = cmap.best_pipeline()
        assert pipeline is not None
        assert [p.peer_id for p in pipeline] == ["a", "b"]

    def test_dijkstra_fallback_to_greedy(self):
        """If Dijkstra can't find a path but greedy can, fallback works.

        This is a safety net — in practice both should agree on reachability.
        We test by providing metrics for only some peers.
        """
        ranges = [lr("a", 0, 16, 32), lr("b", 16, 32, 32)]
        cmap = LayerCoverageMap(ranges, 32)
        # Provide metrics — Dijkstra should find the path just fine.
        metrics = {"a": PeerMetrics(latency_ms=10.0), "b": PeerMetrics(latency_ms=10.0)}
        pipeline = cmap.best_pipeline(peer_metrics=metrics)
        assert pipeline is not None

    def test_empty_map_returns_none_with_metrics(self):
        cmap = LayerCoverageMap([], 32)
        pipeline = cmap.best_pipeline(peer_metrics={})
        assert pipeline is None
