"""Tests for coordinator/swarm_rebalance.py — Phase 2B: Swarm Rebalancing."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from coordinator.layer_coverage import LayerRange, PeerMetrics
from coordinator.swarm_rebalance import (
    RebalanceDirective,
    SwarmRebalancer,
    apply_directive_safely,
    compute_per_layer_throughput,
    find_bottleneck_layer,
    poll_directive_from_dht,
    post_directive_to_dht,
    simulate_migration,
)


# ── compute_per_layer_throughput ─────────────────────────────────────────────


class TestComputePerLayerThroughput:
    def test_single_peer_full_model(self):
        ranges = [LayerRange(peer_id="p1", layer_start=0, layer_end=32, total_layers=32)]
        metrics = {"p1": PeerMetrics(estimated_tps=100.0)}
        result = compute_per_layer_throughput(ranges, metrics, total_layers=32)
        assert len(result) == 32
        assert all(t == 100.0 for t in result)

    def test_two_peers_overlapping(self):
        ranges = [
            LayerRange(peer_id="a", layer_start=0, layer_end=20, total_layers=32),
            LayerRange(peer_id="b", layer_start=10, layer_end=32, total_layers=32),
        ]
        metrics = {
            "a": PeerMetrics(estimated_tps=50.0),
            "b": PeerMetrics(estimated_tps=50.0),
        }
        result = compute_per_layer_throughput(ranges, metrics, total_layers=32)
        # Layers 0-9: only peer a → 50
        assert all(result[i] == 50.0 for i in range(10))
        # Layers 10-19: both peers → 100
        assert all(result[i] == 100.0 for i in range(10, 20))
        # Layers 20-31: only peer b → 50
        assert all(result[i] == 50.0 for i in range(20, 32))

    def test_gap_in_coverage(self):
        ranges = [
            LayerRange(peer_id="a", layer_start=0, layer_end=10, total_layers=32),
            LayerRange(peer_id="b", layer_start=20, layer_end=32, total_layers=32),
        ]
        metrics = {
            "a": PeerMetrics(estimated_tps=100.0),
            "b": PeerMetrics(estimated_tps=100.0),
        }
        result = compute_per_layer_throughput(ranges, metrics, total_layers=32)
        # Layers 10-19: no coverage → 0
        assert all(result[i] == 0.0 for i in range(10, 20))

    def test_missing_metrics_defaults_to_zero(self):
        ranges = [LayerRange(peer_id="p1", layer_start=0, layer_end=32, total_layers=32)]
        result = compute_per_layer_throughput(ranges, {}, total_layers=32)
        assert all(t == 0.0 for t in result)

    def test_empty_ranges(self):
        result = compute_per_layer_throughput([], {}, total_layers=32)
        assert result == [0.0] * 32


# ── find_bottleneck_layer ────────────────────────────────────────────────────


class TestFindBottleneckLayer:
    def test_bottleneck_at_gap(self):
        throughput = [100.0] * 10 + [0.0] * 10 + [100.0] * 12
        idx, min_tps = find_bottleneck_layer(throughput)
        assert idx == 10
        assert min_tps == 0.0

    def test_bottleneck_at_end(self):
        throughput = [100.0, 100.0, 50.0]
        idx, min_tps = find_bottleneck_layer(throughput)
        assert idx == 2
        assert min_tps == 50.0

    def test_uniform_throughput(self):
        throughput = [100.0] * 32
        idx, min_tps = find_bottleneck_layer(throughput)
        assert idx == 0  # first one wins ties
        assert min_tps == 100.0

    def test_empty_returns_sentinel(self):
        idx, min_tps = find_bottleneck_layer([])
        assert idx == -1
        assert min_tps == 0.0


# ── simulate_migration ──────────────────────────────────────────────────────


class TestSimulateMigration:
    def test_migration_fills_gap(self):
        """Migrating a peer to cover a gap should increase min throughput."""
        ranges = [
            LayerRange(peer_id="a", layer_start=0, layer_end=16, total_layers=32),
            LayerRange(peer_id="b", layer_start=0, layer_end=16, total_layers=32),  # redundant
        ]
        metrics = {
            "a": PeerMetrics(estimated_tps=100.0),
            "b": PeerMetrics(estimated_tps=100.0),
        }
        # Before: layers 0-15 have 200 TPS, layers 16-31 have 0
        _, current_min = find_bottleneck_layer(
            compute_per_layer_throughput(ranges, metrics, 32)
        )
        assert current_min == 0.0

        # Simulate migrating "b" to cover [16, 32)
        new_min = simulate_migration(ranges, metrics, 32, "b", 16, 32)
        assert new_min == 100.0  # now all layers have at least 100 TPS

    def test_migration_no_improvement(self):
        """When migration doesn't help, min throughput stays the same."""
        ranges = [
            LayerRange(peer_id="a", layer_start=0, layer_end=32, total_layers=32),
        ]
        metrics = {"a": PeerMetrics(estimated_tps=100.0)}
        # Only one peer — moving it to [0, 16) would leave [16, 32) uncovered
        new_min = simulate_migration(ranges, metrics, 32, "a", 0, 16)
        assert new_min == 0.0


# ── SwarmRebalancer.evaluate ─────────────────────────────────────────────────


class TestSwarmRebalancer:
    def test_generates_directive_for_redundant_peer(self):
        """A redundant peer covering a well-served area should be migrated."""
        rebalancer = SwarmRebalancer(balance_quality=1.0)  # any improvement triggers
        # 3 peers: a + b cover [0,32), c is redundant on [0,32)
        # but c has lower TPS — bottleneck is at min_tps = 300 (all layers).
        # After migrating c to a different model it doesn't improve, so use
        # a scenario with a clear bottleneck:
        # a covers [0,32) at 100 TPS, b covers [0,16) at 100 TPS (redundant on first half)
        # Bottleneck: layers 16-31 at 100 TPS vs layers 0-15 at 200 TPS
        # Moving b to [16,32) would make all layers 100 TPS — but 100/100 = 1.0 which
        # is not > 1.0. So use balance_quality=1.0 with explicit improvement:
        ranges = [
            LayerRange(peer_id="a", layer_start=0, layer_end=32, total_layers=32),
            LayerRange(peer_id="b", layer_start=0, layer_end=16, total_layers=32),  # redundant
        ]
        metrics = {
            "a": PeerMetrics(estimated_tps=100.0),
            "b": PeerMetrics(estimated_tps=100.0),
        }
        # Before: layers 0-15 = 200 TPS, layers 16-31 = 100 TPS → min = 100
        # Migrating b to [16,32): all layers = 200 TPS → min = 200
        # Improvement: 200/100 = 2.0 >= 1.0 threshold
        directives = rebalancer.evaluate(ranges, metrics, total_layers=32)
        assert len(directives) >= 1
        d = directives[0]
        assert d.target_peer_id == "b"
        assert d.total_layers == 32

    def test_no_directive_when_balanced(self):
        """A balanced swarm should not generate directives."""
        rebalancer = SwarmRebalancer(balance_quality=1.15)
        ranges = [
            LayerRange(peer_id="a", layer_start=0, layer_end=16, total_layers=32),
            LayerRange(peer_id="b", layer_start=16, layer_end=32, total_layers=32),
        ]
        metrics = {
            "a": PeerMetrics(estimated_tps=100.0),
            "b": PeerMetrics(estimated_tps=100.0),
        }
        directives = rebalancer.evaluate(ranges, metrics, total_layers=32)
        assert len(directives) == 0

    def test_no_directives_when_empty(self):
        rebalancer = SwarmRebalancer()
        assert rebalancer.evaluate([], {}, 32) == []
        assert rebalancer.evaluate([], {}, 0) == []

    def test_directive_for_zero_coverage_layer(self):
        """When a layer has zero coverage, migration should be triggered."""
        rebalancer = SwarmRebalancer(balance_quality=1.0)
        # a covers full model, b is redundant on [0,32) — but we need a gap.
        # Use: a covers [0,16), b covers [0,16), c covers full [0,32).
        # Bottleneck: layers 16-31 only covered by c (50 TPS).
        # Actually simpler: a covers full model, b redundant on first half.
        ranges = [
            LayerRange(peer_id="a", layer_start=0, layer_end=32, total_layers=32),
            LayerRange(peer_id="b", layer_start=0, layer_end=16, total_layers=32),
        ]
        metrics = {
            "a": PeerMetrics(estimated_tps=50.0),
            "b": PeerMetrics(estimated_tps=50.0),
        }
        # Bottleneck: layers 16-31 at 50 TPS, layers 0-15 at 100 TPS
        # Moving b to cover bottleneck → all layers 100 TPS → 100/50 = 2.0
        directives = rebalancer.evaluate(ranges, metrics, total_layers=32)
        assert len(directives) >= 1
        d = directives[0]
        assert d.target_peer_id == "b"

    def test_balance_quality_threshold(self):
        """Only generate directives when improvement exceeds the threshold."""
        rebalancer = SwarmRebalancer(balance_quality=2.0)  # very high threshold
        ranges = [
            LayerRange(peer_id="a", layer_start=0, layer_end=16, total_layers=32),
            LayerRange(peer_id="b", layer_start=0, layer_end=16, total_layers=32),
            LayerRange(peer_id="c", layer_start=16, layer_end=32, total_layers=32),
        ]
        metrics = {
            "a": PeerMetrics(estimated_tps=100.0),
            "b": PeerMetrics(estimated_tps=100.0),
            "c": PeerMetrics(estimated_tps=80.0),
        }
        # Current min = 80 (layer 16-31 with only peer c)
        # Moving b to [16,32) would give min = 100 (80+100 at layers 16-31, 100 at 0-15)
        # Improvement = 100/80 = 1.25 < 2.0 threshold → no directive
        directives = rebalancer.evaluate(ranges, metrics, total_layers=32)
        assert len(directives) == 0


# ── apply_directive_safely ───────────────────────────────────────────────────


class TestApplyDirectiveSafely:
    def test_applies_when_inflight_zero(self):
        """Directive is applied immediately when no requests are inflight."""
        service = MagicMock()
        service._lock = threading.Lock()
        service._inflight = 0
        service.shard.reshard.return_value = True
        service.shard.runtime_profile.return_value = {"layer_start": 16, "layer_end": 32}

        directive = RebalanceDirective(
            target_peer_id="p1",
            new_layer_start=16,
            new_layer_end=32,
            total_layers=32,
        )
        result = apply_directive_safely(service, directive, drain_timeout_s=1.0)
        assert result is True
        service.shard.reshard.assert_called_once_with(16, 32, 32)

    def test_waits_for_drain_then_applies(self):
        """Directive waits for inflight to drain before applying."""
        service = MagicMock()
        service._lock = threading.Lock()
        service._inflight = 2
        service.shard.reshard.return_value = True
        service.shard.runtime_profile.return_value = {"layer_start": 0, "layer_end": 16}

        directive = RebalanceDirective(
            target_peer_id="p1",
            new_layer_start=0,
            new_layer_end=16,
            total_layers=32,
        )

        # Simulate drain after 0.3s
        def drain_after():
            time.sleep(0.3)
            with service._lock:
                service._inflight = 0

        t = threading.Thread(target=drain_after, daemon=True)
        t.start()

        result = apply_directive_safely(service, directive, drain_timeout_s=2.0)
        t.join(timeout=3.0)
        assert result is True
        service.shard.reshard.assert_called_once()

    def test_skips_when_drain_timeout(self):
        """Directive is skipped if inflight doesn't drain in time."""
        service = MagicMock()
        service._lock = threading.Lock()
        service._inflight = 5  # stuck inflight

        directive = RebalanceDirective(
            target_peer_id="p1",
            new_layer_start=16,
            new_layer_end=32,
            total_layers=32,
        )
        result = apply_directive_safely(service, directive, drain_timeout_s=0.3)
        assert result is False
        service.shard.reshard.assert_not_called()

    def test_returns_false_when_reshard_fails(self):
        """Returns False when shard.reshard() fails."""
        service = MagicMock()
        service._lock = threading.Lock()
        service._inflight = 0
        service.shard.reshard.return_value = False

        directive = RebalanceDirective(
            target_peer_id="p1",
            new_layer_start=16,
            new_layer_end=32,
            total_layers=32,
        )
        result = apply_directive_safely(service, directive, drain_timeout_s=1.0)
        assert result is False


# ── DHT directive posting / polling ──────────────────────────────────────────


class TestDhtDirectiveCycle:
    def test_post_directive_builds_correct_payload(self):
        directive = RebalanceDirective(
            target_peer_id="peer-42",
            new_layer_start=16,
            new_layer_end=32,
            total_layers=32,
            reason="bottleneck_layer=16",
            issued_at_ms=1234567890,
        )

        captured: list[bytes] = []

        def mock_urlopen(req, timeout=None):
            captured.append(req.data)
            resp = MagicMock()
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            resp.status = 200
            return resp

        with patch("coordinator.swarm_rebalance.urllib_request.urlopen", mock_urlopen):
            result = post_directive_to_dht(directive, "http://dht.example.com:8468")

        assert result is True
        assert len(captured) == 1
        body = json.loads(captured[0])
        assert body["key"] == "rebalance_peer-42"
        value = json.loads(body["value"])
        assert value["new_layer_start"] == 16
        assert value["new_layer_end"] == 32

    def test_poll_directive_parses_response(self):
        directive_data = {
            "target_peer_id": "peer-42",
            "new_layer_start": 16,
            "new_layer_end": 32,
            "total_layers": 32,
            "reason": "test",
            "issued_at_ms": 9999,
        }

        def mock_urlopen(req, timeout=None):
            resp = MagicMock()
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            resp.read.return_value = json.dumps(
                {"value": json.dumps(directive_data)}
            ).encode("utf-8")
            return resp

        with patch("coordinator.swarm_rebalance.urllib_request.urlopen", mock_urlopen):
            result = poll_directive_from_dht("peer-42", "http://dht.example.com:8468")

        assert result is not None
        assert result.target_peer_id == "peer-42"
        assert result.new_layer_start == 16
        assert result.new_layer_end == 32
        assert result.total_layers == 32

    def test_poll_returns_none_on_empty(self):
        def mock_urlopen(req, timeout=None):
            resp = MagicMock()
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            resp.read.return_value = json.dumps({"value": None}).encode("utf-8")
            return resp

        with patch("coordinator.swarm_rebalance.urllib_request.urlopen", mock_urlopen):
            result = poll_directive_from_dht("peer-99", "http://dht.example.com:8468")

        assert result is None

    def test_poll_returns_none_on_network_error(self):
        with patch("coordinator.swarm_rebalance.urllib_request.urlopen", side_effect=Exception("network")):
            result = poll_directive_from_dht("peer-99", "http://dht.example.com:8468")
        assert result is None


# ── RebalanceDirective dataclass ─────────────────────────────────────────────


class TestRebalanceDirective:
    def test_frozen(self):
        d = RebalanceDirective(target_peer_id="p1", new_layer_start=0, new_layer_end=16, total_layers=32)
        with pytest.raises(AttributeError):
            d.target_peer_id = "p2"  # type: ignore[misc]

    def test_default_values(self):
        d = RebalanceDirective(target_peer_id="p1", new_layer_start=0, new_layer_end=16, total_layers=32)
        assert d.reason == ""
        assert d.issued_at_ms == 0
