# Copyright 2026 OpenHydra contributors — Apache 2.0

"""B3 follow-up — pipeline ordering fix.

Two unit-level helpers ship with the fix; each gets its own
``TestCase`` class here:

* :func:`coordinator.inference_service._sort_pipeline_by_layer_start`
  — strict ordering so the peer owning ``[0, …)`` is always stage 0,
  regardless of peer discovery order. Tie-breaks on ``peer_id`` for
  determinism across requests (KV cache reuse).

* :func:`coordinator.inference_service._override_local_peer_layer_range_from_snapshot`
  — rewrite the local peer's layer range in the health list with the
  live assignment from the ``LoopSnapshot``, so a just-resharded
  peer's PeerEndpoint stays in sync with its in-memory shard.

Both helpers are deliberately side-effect-free closures over their
inputs — tests drive them with synthetic PeerEndpoint / PeerHealth
fixtures and a stub snapshot.

Run:  ``pytest tests/test_pipeline_ordering.py -v``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from coordinator.inference_service import (
    _override_local_peer_layer_range_from_snapshot,
    _sort_pipeline_by_layer_start,
)
from coordinator.path_finder import PeerEndpoint, PeerHealth
from peer.swarm_negotiator import ShardAssignment


# ─── Shared fixtures ─────────────────────────────────────────────────────────


def _peer(
    peer_id: str,
    *,
    layer_start: int = 0,
    layer_end: int = 0,
    total_layers: int = 0,
    libp2p: str = "",
    host: str = "10.0.0.1",
    port: int = 50051,
) -> PeerEndpoint:
    return PeerEndpoint(
        peer_id=peer_id,
        host=host,
        port=port,
        model_id="openhydra-qwen3.5-2b",
        operator_id="test",
        runtime_backend="pytorch",
        libp2p_peer_id=libp2p,
        layer_start=layer_start,
        layer_end=layer_end,
        total_layers=total_layers,
    )


def _health(peer: PeerEndpoint) -> PeerHealth:
    return PeerHealth(
        peer=peer, healthy=True, latency_ms=10.0, load_pct=0.1,
        daemon_mode="polite",
    )


# ─── _sort_pipeline_by_layer_start ───────────────────────────────────────────


class TestSortPipelineByLayerStart:
    def test_empty_pipeline_returns_unchanged(self):
        assert _sort_pipeline_by_layer_start([]) == []

    def test_single_peer_returns_unchanged(self):
        peers = [_peer("only", layer_start=12, layer_end=24, total_layers=24)]
        assert _sort_pipeline_by_layer_start(peers) is peers

    def test_already_ordered_preserves_order(self):
        a = _peer("A", layer_start=0, layer_end=12, total_layers=24)
        b = _peer("B", layer_start=12, layer_end=24, total_layers=24)
        sorted_pipe = _sort_pipeline_by_layer_start([a, b])
        assert [p.peer_id for p in sorted_pipe] == ["A", "B"]

    def test_reverse_ordered_pipeline_gets_sorted(self):
        """The headline case: Mac got [12, 24) and GPU1 got [0, 12).
        Mac is first in discovery order (it's the coordinator's local
        peer) but must appear as stage 1."""
        mac = _peer("mac-final", layer_start=12, layer_end=24, total_layers=24)
        gpu = _peer("gpu1-final", layer_start=0, layer_end=12, total_layers=24)
        # Simulate discovery order: Mac first.
        unsorted = [mac, gpu]
        sorted_pipe = _sort_pipeline_by_layer_start(unsorted)
        assert [p.peer_id for p in sorted_pipe] == ["gpu1-final", "mac-final"]
        # Stage 0 must own the embed layers.
        assert sorted_pipe[0].layer_start == 0

    def test_three_stage_pipeline(self):
        a = _peer("A", layer_start=8, layer_end=16, total_layers=24)
        b = _peer("B", layer_start=16, layer_end=24, total_layers=24)
        c = _peer("C", layer_start=0, layer_end=8, total_layers=24)
        sorted_pipe = _sort_pipeline_by_layer_start([a, b, c])
        assert [p.peer_id for p in sorted_pipe] == ["C", "A", "B"]
        # Contiguous tiling check.
        for i in range(len(sorted_pipe) - 1):
            assert sorted_pipe[i].layer_end == sorted_pipe[i + 1].layer_start

    def test_equal_layer_starts_are_stable_by_peer_id(self):
        """Two full-model replicas (layer_start=0 on both) shouldn't
        trigger reordering — the helper detects this case and returns
        the input unchanged."""
        a = _peer("zzz", layer_start=0, layer_end=0, total_layers=0)
        b = _peer("aaa", layer_start=0, layer_end=0, total_layers=0)
        result = _sort_pipeline_by_layer_start([a, b])
        assert result is [a, b] or [p.peer_id for p in result] == ["zzz", "aaa"]

    def test_ties_break_by_peer_id_lex(self):
        """When two peers claim the same layer_start (should be rare
        but possible during overlap transitions), tie-break on peer_id
        for deterministic ordering across requests."""
        a = _peer("zzz", layer_start=0, layer_end=12, total_layers=24)
        b = _peer("aaa", layer_start=0, layer_end=12, total_layers=24)
        c = _peer("mmm", layer_start=12, layer_end=24, total_layers=24)
        sorted_pipe = _sort_pipeline_by_layer_start([a, b, c])
        # "aaa" < "zzz" lex → aaa comes first.
        assert [p.peer_id for p in sorted_pipe] == ["aaa", "zzz", "mmm"]


# ─── _override_local_peer_layer_range_from_snapshot ──────────────────────────


@dataclass
class _FakeSnapshot:
    """Stand-in for :class:`peer.negotiation_loop.LoopSnapshot` with
    just the ``snapshot()`` method the override helper reads."""

    assignment: ShardAssignment | None

    def snapshot(self):
        # Shape matches LoopSnapshot.snapshot() return.
        return ("", 2, self.assignment, 0)


class _FakeDsvc:
    """Minimal stand-in for the discovery_service surface the helper
    reads."""

    def __init__(
        self,
        *,
        self_libp2p: str = "",
        snapshot: _FakeSnapshot | None = None,
    ):
        self._self_libp2p_peer_id = self_libp2p
        self._capacity_snapshot_ref = snapshot


class TestOverrideLocalPeerLayerRange:
    def test_no_snapshot_returns_health_unchanged(self):
        health = [_health(_peer("mac", libp2p="12D3KooWMac"))]
        dsvc = _FakeDsvc(self_libp2p="12D3KooWMac", snapshot=None)
        result = _override_local_peer_layer_range_from_snapshot(
            health=health, discovery_service=dsvc,
        )
        assert result is health

    def test_empty_self_libp2p_returns_health_unchanged(self):
        snap = _FakeSnapshot(
            assignment=ShardAssignment(
                model_id="m", layer_start=12, layer_end=24,
                total_layers=24, source="pick_best_fit",
            )
        )
        dsvc = _FakeDsvc(self_libp2p="", snapshot=snap)
        health = [_health(_peer("mac", libp2p="12D3KooWMac"))]
        assert _override_local_peer_layer_range_from_snapshot(
            health=health, discovery_service=dsvc,
        ) is health

    def test_snapshot_with_none_assignment_returns_unchanged(self):
        snap = _FakeSnapshot(assignment=None)
        dsvc = _FakeDsvc(self_libp2p="12D3KooWMac", snapshot=snap)
        health = [_health(_peer("mac", libp2p="12D3KooWMac"))]
        assert _override_local_peer_layer_range_from_snapshot(
            health=health, discovery_service=dsvc,
        ) is health

    def test_local_peer_layer_range_replaced(self):
        """The headline case: Mac's PeerEndpoint says full-model
        (layer_end=0 by default), snapshot says ``[12, 24)``; after
        the override Mac's health entry carries the correct range."""
        snap = _FakeSnapshot(
            assignment=ShardAssignment(
                model_id="openhydra-qwen3.5-2b",
                layer_start=12, layer_end=24, total_layers=24,
                source="conflict_split",
            )
        )
        dsvc = _FakeDsvc(self_libp2p="12D3KooWMac", snapshot=snap)
        mac = _peer("mac", libp2p="12D3KooWMac")  # no layer fields set
        gpu = _peer(
            "gpu", libp2p="12D3KooWGpu",
            layer_start=0, layer_end=12, total_layers=24,
        )
        health = [_health(mac), _health(gpu)]
        result = _override_local_peer_layer_range_from_snapshot(
            health=health, discovery_service=dsvc,
        )
        assert result is not health  # new list when we replaced
        by_id = {h.peer.peer_id: h.peer for h in result}
        # Mac now has the live layer range.
        assert by_id["mac"].layer_start == 12
        assert by_id["mac"].layer_end == 24
        assert by_id["mac"].total_layers == 24
        # GPU1 untouched (not the local peer).
        assert by_id["gpu"].layer_start == 0
        assert by_id["gpu"].layer_end == 12

    def test_non_matching_libp2p_leaves_everyone_alone(self):
        snap = _FakeSnapshot(
            assignment=ShardAssignment(
                model_id="m", layer_start=12, layer_end=24,
                total_layers=24, source="pick_best_fit",
            )
        )
        # Self libp2p doesn't match any peer — nothing to override.
        dsvc = _FakeDsvc(self_libp2p="12D3KooWUnknown", snapshot=snap)
        health = [
            _health(_peer("mac", libp2p="12D3KooWMac",
                          layer_start=0, layer_end=12, total_layers=24)),
            _health(_peer("gpu", libp2p="12D3KooWGpu",
                          layer_start=12, layer_end=24, total_layers=24)),
        ]
        result = _override_local_peer_layer_range_from_snapshot(
            health=health, discovery_service=dsvc,
        )
        # No replacement → identity-equal to input.
        assert result is health

    def test_degenerate_assignment_is_ignored(self):
        """Snapshot with layer_end <= layer_start must be treated as
        unusable and leave the peer untouched."""
        snap = _FakeSnapshot(
            assignment=ShardAssignment(
                model_id="m", layer_start=12, layer_end=12,  # empty range
                total_layers=24, source="pick_best_fit",
            )
        )
        dsvc = _FakeDsvc(self_libp2p="12D3KooWMac", snapshot=snap)
        health = [_health(_peer("mac", libp2p="12D3KooWMac"))]
        assert _override_local_peer_layer_range_from_snapshot(
            health=health, discovery_service=dsvc,
        ) is health

    def test_end_to_end_integration_with_sort(self):
        """Override + sort composed: the headline scenario in one path.
        Before: Mac's health entry claims full-model (layer_end=0).
        After override: Mac=[12, 24), GPU1=[0, 12).
        After sort: GPU1 is stage 0, Mac is stage 1."""
        snap = _FakeSnapshot(
            assignment=ShardAssignment(
                model_id="openhydra-qwen3.5-2b",
                layer_start=12, layer_end=24, total_layers=24,
                source="conflict_split",
            )
        )
        dsvc = _FakeDsvc(self_libp2p="12D3KooWMac", snapshot=snap)
        mac = _peer("mac-final", libp2p="12D3KooWMac")
        gpu = _peer(
            "gpu1-final", libp2p="12D3KooWGpu",
            layer_start=0, layer_end=12, total_layers=24,
        )
        health = [_health(mac), _health(gpu)]
        overridden = _override_local_peer_layer_range_from_snapshot(
            health=health, discovery_service=dsvc,
        )
        # Simulate pipeline builder: take peers out of health + sort.
        pipeline = _sort_pipeline_by_layer_start(
            [h.peer for h in overridden]
        )
        assert [p.peer_id for p in pipeline] == ["gpu1-final", "mac-final"]
        assert pipeline[0].layer_start == 0
        assert pipeline[0].layer_end == 12
        assert pipeline[1].layer_start == 12
        assert pipeline[1].layer_end == 24
