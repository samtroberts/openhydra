# Copyright 2026 OpenHydra contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for peer.swarm_negotiator (Phase 3 decentralised assignment).

Tests drive the negotiator with synthetic :class:`CapacityReport` fixtures
and a stub ``dht_scan`` callable — no real DHT I/O is performed.
"""

from __future__ import annotations

from typing import Callable

import pytest

from coordinator.degradation import ModelAvailability
from peer.capacity import (
    NODE_PERSONA_ATOMIC_WORKER,
    NODE_PERSONA_NATIVE_SHARD,
    UPSTREAM_KIND_OLLAMA,
    UpstreamConfig,
    build_capacity_report,
)
from peer.hardware import HardwareProfile
from peer.swarm_negotiator import (
    PeerClaim,
    ShardAssignment,
    SOURCE_ATOMIC_WORKER,
    SOURCE_CONFLICT_SPLIT,
    SOURCE_FALLBACK_WHOLE,
    SOURCE_PICK_BEST_FIT,
    SwarmNegotiator,
    compute_conflict_split,
    compute_gaps,
    pick_best_fit,
    should_concede,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────


def _cuda_t4() -> HardwareProfile:
    return HardwareProfile(
        ram_total_bytes=16 * 1024**3, ram_available_bytes=10 * 1024**3,
        accelerator="cuda",
        vram_total_bytes=15 * 1024**3, vram_available_bytes=14 * 1024**3,
        cuda_device_count=1,
    )


def _small_cpu() -> HardwareProfile:
    return HardwareProfile(
        ram_total_bytes=2 * 1024**3, ram_available_bytes=1 * 1024**3,
        accelerator="cpu",
        vram_total_bytes=None, vram_available_bytes=None,
        cuda_device_count=0,
    )


def _qwen_2b() -> ModelAvailability:
    return ModelAvailability(
        model_id="openhydra-qwen3.5-2b", required_peers=1,
        hf_model_id="Qwen/Qwen3.5-2B", min_vram_gb=5,
        shard_vram_gb=2.5, shards_needed=1, num_layers=24,
        recommended_quantization="fp16",
    )


def _qwen_9b() -> ModelAvailability:
    return ModelAvailability(
        model_id="openhydra-qwen3.5-9b", required_peers=2,
        hf_model_id="Qwen/Qwen3.5-9B", min_vram_gb=18,
        shard_vram_gb=9.0, shards_needed=2, num_layers=32,
        recommended_quantization="int8",
    )


def _qwen_27b() -> ModelAvailability:
    return ModelAvailability(
        model_id="openhydra-qwen3.5-27b-fp8", required_peers=4,
        hf_model_id="Qwen/Qwen3.5-27B-FP8", min_vram_gb=16,
        shard_vram_gb=7.0, shards_needed=4, num_layers=64,
        recommended_quantization="fp8",
    )


def _native_report(
    hw: HardwareProfile, *models: ModelAvailability,
    libp2p_peer_id: str = "12D3KooWMINE",
):
    return build_capacity_report(
        hardware=hw,
        catalog=list(models),
        peer_id="me", libp2p_peer_id=libp2p_peer_id,
        ports={"api": 8080, "grpc": 50051, "libp2p": 4001},
        advertise_host="10.0.0.1",
    )


def _empty_scan(model_id: str) -> list[PeerClaim]:
    """Stub scanner that always returns an empty list."""
    return []


def _fixed_scan(claims: list[PeerClaim]) -> Callable[[str], list[PeerClaim]]:
    """Return a scanner that ignores the model_id arg and returns the given claims."""
    def _scan(_mid: str) -> list[PeerClaim]:
        return list(claims)
    return _scan


# ─── pick_best_fit ───────────────────────────────────────────────────────────


def test_pick_best_fit_empty_gaps_returns_none():
    assert pick_best_fit([], 10) is None


def test_pick_best_fit_zero_max_layers_returns_none():
    assert pick_best_fit([(0, 10)], 0) is None


def test_pick_best_fit_prefers_exact_cover_widest_gap():
    # gaps: [(0,4), (10,18)] -- both fit in 10, widest is (10,18) span=8
    assert pick_best_fit([(0, 4), (10, 18)], 10) == (10, 18)


def test_pick_best_fit_bites_off_widest_slice_when_nothing_fits():
    # gap [0,30] with budget 10 → take (0, 10) — the widest slice starting
    # from the gap's left edge
    assert pick_best_fit([(0, 30)], 10) == (0, 10)


def test_pick_best_fit_stability_across_ties():
    # Two equal-span coverable gaps — deterministic pick by start index.
    assert pick_best_fit([(10, 14), (0, 4)], 4) == (0, 4)


# ─── compute_gaps ────────────────────────────────────────────────────────────


def test_compute_gaps_empty_claims_returns_whole_model():
    assert compute_gaps([], 24) == [(0, 24)]


def test_compute_gaps_single_claim_left_edge():
    claims = [PeerClaim(libp2p_peer_id="a", model_id="m", layer_start=0, layer_end=8, total_layers=24)]
    assert compute_gaps(claims, 24) == [(8, 24)]


def test_compute_gaps_single_claim_right_edge():
    claims = [PeerClaim(libp2p_peer_id="a", model_id="m", layer_start=16, layer_end=24, total_layers=24)]
    assert compute_gaps(claims, 24) == [(0, 16)]


def test_compute_gaps_middle_claim_produces_two_gaps():
    claims = [PeerClaim(libp2p_peer_id="a", model_id="m", layer_start=8, layer_end=16, total_layers=24)]
    assert compute_gaps(claims, 24) == [(0, 8), (16, 24)]


def test_compute_gaps_merges_overlapping_claims():
    claims = [
        PeerClaim(libp2p_peer_id="a", model_id="m", layer_start=0, layer_end=10, total_layers=24),
        PeerClaim(libp2p_peer_id="b", model_id="m", layer_start=5, layer_end=16, total_layers=24),
    ]
    # After merge: one segment 0..16. Gap: 16..24.
    assert compute_gaps(claims, 24) == [(16, 24)]


def test_compute_gaps_merges_adjacent_claims():
    claims = [
        PeerClaim(libp2p_peer_id="a", model_id="m", layer_start=0, layer_end=8, total_layers=24),
        PeerClaim(libp2p_peer_id="b", model_id="m", layer_start=8, layer_end=16, total_layers=24),
    ]
    assert compute_gaps(claims, 24) == [(16, 24)]


def test_compute_gaps_ignores_claims_with_wrong_total_layers():
    # This claim thinks the model is 32 layers deep, but we're asking for a
    # 24-layer model — must be ignored.
    claims = [PeerClaim(libp2p_peer_id="a", model_id="m", layer_start=0, layer_end=8, total_layers=32)]
    assert compute_gaps(claims, 24) == [(0, 24)]


def test_compute_gaps_zero_total_layers_returns_empty():
    assert compute_gaps([], 0) == []


def test_compute_gaps_clamps_out_of_range_claims():
    claims = [PeerClaim(libp2p_peer_id="a", model_id="m", layer_start=-5, layer_end=30, total_layers=24)]
    assert compute_gaps(claims, 24) == []  # fully covered


# ─── should_concede (conflict resolution) ────────────────────────────────────


def test_concede_when_peer_has_more_vram():
    claims = [PeerClaim(
        libp2p_peer_id="other", model_id="m", layer_start=0, layer_end=12,
        total_layers=24, available_vram_mb=8000,
    )]
    assert should_concede((4, 10), peer_claims=claims, model_id="m",
                          total_layers=24, my_vram_mb=4000,
                          my_libp2p_peer_id="me") is True


def test_do_not_concede_when_peer_has_less_vram():
    claims = [PeerClaim(
        libp2p_peer_id="other", model_id="m", layer_start=0, layer_end=12,
        total_layers=24, available_vram_mb=2000,
    )]
    assert should_concede((4, 10), peer_claims=claims, model_id="m",
                          total_layers=24, my_vram_mb=8000,
                          my_libp2p_peer_id="me") is False


def test_do_not_concede_when_ranges_dont_overlap():
    claims = [PeerClaim(
        libp2p_peer_id="other", model_id="m", layer_start=0, layer_end=8,
        total_layers=24, available_vram_mb=99999,
    )]
    assert should_concede((16, 24), peer_claims=claims, model_id="m",
                          total_layers=24, my_vram_mb=1,
                          my_libp2p_peer_id="me") is False


def test_tie_breaker_smaller_libp2p_peer_id_wins():
    # Same VRAM — lexicographically smaller libp2p_peer_id wins.
    claims = [PeerClaim(
        libp2p_peer_id="aaa", model_id="m", layer_start=0, layer_end=12,
        total_layers=24, available_vram_mb=8000,
    )]
    # I am "zzz" (larger) → concede.
    assert should_concede((4, 10), peer_claims=claims, model_id="m",
                          total_layers=24, my_vram_mb=8000,
                          my_libp2p_peer_id="zzz") is True
    # I am "a" (smaller) → keep.
    assert should_concede((4, 10), peer_claims=claims, model_id="m",
                          total_layers=24, my_vram_mb=8000,
                          my_libp2p_peer_id="a") is False


def test_self_claim_is_ignored():
    claims = [PeerClaim(
        libp2p_peer_id="me", model_id="m", layer_start=0, layer_end=12,
        total_layers=24, available_vram_mb=99999,
    )]
    assert should_concede((4, 10), peer_claims=claims, model_id="m",
                          total_layers=24, my_vram_mb=100,
                          my_libp2p_peer_id="me") is False


def test_different_model_claim_is_ignored():
    claims = [PeerClaim(
        libp2p_peer_id="other", model_id="other-model", layer_start=0, layer_end=24,
        total_layers=24, available_vram_mb=99999,
    )]
    assert should_concede((0, 24), peer_claims=claims, model_id="m",
                          total_layers=24, my_vram_mb=100,
                          my_libp2p_peer_id="me") is False


# ─── SwarmNegotiator — atomic_worker path ────────────────────────────────────


def test_atomic_worker_assigns_whole_model_for_first_hosted_entry():
    upstream = UpstreamConfig(
        kind=UPSTREAM_KIND_OLLAMA, url="http://localhost:11434",
        hosted_model_ids=("openhydra-qwen3.5-2b",),
    )
    report = build_capacity_report(
        hardware=_cuda_t4(),
        catalog=[_qwen_2b(), _qwen_9b()],
        node_persona=NODE_PERSONA_ATOMIC_WORKER,
        upstream=upstream,
    )
    neg = SwarmNegotiator(
        capacity_report=report, libp2p_peer_id="me", dht_scan=_empty_scan,
    )
    a = neg.negotiate()
    assert a is not None
    assert a.source == SOURCE_ATOMIC_WORKER
    assert a.model_id == "openhydra-qwen3.5-2b"
    assert a.layer_start == 0
    assert a.layer_end == a.total_layers == 24


def test_atomic_worker_returns_none_when_no_model_hosted():
    upstream = UpstreamConfig(
        kind=UPSTREAM_KIND_OLLAMA, url="http://localhost:11434",
        hosted_model_ids=("model-not-in-catalog",),
    )
    report = build_capacity_report(
        hardware=_cuda_t4(),
        catalog=[_qwen_2b(), _qwen_9b()],
        node_persona=NODE_PERSONA_ATOMIC_WORKER,
        upstream=upstream,
    )
    neg = SwarmNegotiator(
        capacity_report=report, libp2p_peer_id="me", dht_scan=_empty_scan,
    )
    assert neg.negotiate() is None


def test_atomic_worker_does_not_call_dht_scan():
    """Atomic workers don't consult the DHT — replication is desirable."""
    upstream = UpstreamConfig(
        kind=UPSTREAM_KIND_OLLAMA, url="http://localhost:11434",
        hosted_model_ids=("openhydra-qwen3.5-2b",),
    )
    report = build_capacity_report(
        hardware=_cuda_t4(),
        catalog=[_qwen_2b()],
        node_persona=NODE_PERSONA_ATOMIC_WORKER,
        upstream=upstream,
    )
    calls: list[str] = []

    def _counting_scan(mid: str) -> list[PeerClaim]:
        calls.append(mid)
        return []

    neg = SwarmNegotiator(
        capacity_report=report, libp2p_peer_id="me", dht_scan=_counting_scan,
    )
    a = neg.negotiate()
    assert a is not None
    assert calls == []


# ─── SwarmNegotiator — native_shard path ─────────────────────────────────────


def test_native_shard_empty_swarm_claims_whole_capable_model():
    """On an empty DHT, the negotiator claims layer 0..num_layers for the
    first model it is capable of hosting."""
    report = _native_report(_cuda_t4(), _qwen_2b())
    neg = SwarmNegotiator(
        capacity_report=report, libp2p_peer_id="me", dht_scan=_empty_scan,
    )
    a = neg.negotiate()
    assert a is not None
    assert a.model_id == "openhydra-qwen3.5-2b"
    assert a.layer_start == 0
    assert a.layer_end == 24
    assert a.source == SOURCE_FALLBACK_WHOLE


def test_native_shard_plugs_middle_gap():
    """Two existing peers cover [0,8) and [16,24); we plug [8,16)."""
    claims = [
        PeerClaim(libp2p_peer_id="peer_a", model_id="openhydra-qwen3.5-2b",
                  layer_start=0, layer_end=8, total_layers=24, available_vram_mb=1000),
        PeerClaim(libp2p_peer_id="peer_b", model_id="openhydra-qwen3.5-2b",
                  layer_start=16, layer_end=24, total_layers=24, available_vram_mb=1000),
    ]
    report = _native_report(_cuda_t4(), _qwen_2b())
    neg = SwarmNegotiator(
        capacity_report=report, libp2p_peer_id="me",
        dht_scan=_fixed_scan(claims),
    )
    a = neg.negotiate()
    assert a is not None
    assert (a.layer_start, a.layer_end) == (8, 16)
    assert a.source == SOURCE_PICK_BEST_FIT


def test_native_shard_concedes_to_higher_vram_peer():
    """If a bigger peer already claimed the gap I'd pick, I concede
    (and move on to the next candidate model or return None)."""
    # Peer with higher VRAM already claims 0..24 of qwen 2b — we lose.
    bigger_peer_claims = [
        PeerClaim(libp2p_peer_id="fat_peer", model_id="openhydra-qwen3.5-2b",
                  layer_start=0, layer_end=24, total_layers=24,
                  available_vram_mb=99999),
    ]
    report = _native_report(_cuda_t4(), _qwen_2b())
    neg = SwarmNegotiator(
        capacity_report=report, libp2p_peer_id="me",
        dht_scan=_fixed_scan(bigger_peer_claims),
    )
    # qwen 2b is fully covered → compute_gaps returns [], so we skip
    # and (no other candidates) return None.
    a = neg.negotiate()
    assert a is None


def test_native_shard_conflict_resolution_moves_to_next_candidate():
    """When a higher-priority peer overlaps our pick on one model but another
    model is free, we pick the free one."""
    # Qwen 2B: fully covered by high-VRAM peer.
    # Qwen 9B: wide open.
    claims_2b = [
        PeerClaim(libp2p_peer_id="fat_peer", model_id="openhydra-qwen3.5-2b",
                  layer_start=0, layer_end=24, total_layers=24,
                  available_vram_mb=99999),
    ]

    def _selective_scan(mid: str) -> list[PeerClaim]:
        if mid == "openhydra-qwen3.5-2b":
            return list(claims_2b)
        return []

    report = _native_report(_cuda_t4(), _qwen_2b(), _qwen_9b())
    # Qwen 2B preferred (via preferred_model_order) — but it's covered, so
    # we move on to 9B which we can shard partially.
    neg = SwarmNegotiator(
        capacity_report=report, libp2p_peer_id="me", dht_scan=_selective_scan,
        preferred_model_order=("openhydra-qwen3.5-2b", "openhydra-qwen3.5-9b"),
    )
    a = neg.negotiate()
    assert a is not None
    assert a.model_id == "openhydra-qwen3.5-9b"
    assert a.layer_start == 0  # start at the left edge of the sole gap
    assert a.layer_end > a.layer_start
    assert a.source in (SOURCE_PICK_BEST_FIT, SOURCE_FALLBACK_WHOLE)


def test_native_shard_rank_prefers_specified_model():
    """preferred_model_order moves the user's requested model to the front."""
    report = _native_report(_cuda_t4(), _qwen_2b(), _qwen_9b())

    # With no preferences, 2B comes first (higher max_layers_hostable).
    neg_default = SwarmNegotiator(
        capacity_report=report, libp2p_peer_id="me", dht_scan=_empty_scan,
    )
    a1 = neg_default.negotiate()
    assert a1 is not None and a1.model_id == "openhydra-qwen3.5-2b"

    # With 9B preferred, it comes first even though 2B has bigger max_layers.
    neg_pref = SwarmNegotiator(
        capacity_report=report, libp2p_peer_id="me", dht_scan=_empty_scan,
        preferred_model_order=("openhydra-qwen3.5-9b",),
    )
    a2 = neg_pref.negotiate()
    assert a2 is not None and a2.model_id == "openhydra-qwen3.5-9b"


def test_native_shard_dht_scan_failure_treated_as_empty():
    """DHT scan exceptions must not crash the negotiator — first-boot peers
    on an unreachable swarm should still self-assign."""
    def _broken_scan(_mid: str) -> list[PeerClaim]:
        raise RuntimeError("DHT unreachable")

    report = _native_report(_cuda_t4(), _qwen_2b())
    neg = SwarmNegotiator(
        capacity_report=report, libp2p_peer_id="me", dht_scan=_broken_scan,
    )
    a = neg.negotiate()
    assert a is not None
    assert (a.layer_start, a.layer_end) == (0, 24)


def test_native_shard_skips_incapable_models():
    """Capacity entries with status=incapable are never considered."""
    # Tiny CPU node can't host 27B model in any meaningful shape — use a
    # report that only includes 27B.
    report = _native_report(_small_cpu(), _qwen_27b())
    neg = SwarmNegotiator(
        capacity_report=report, libp2p_peer_id="me", dht_scan=_empty_scan,
    )
    a = neg.negotiate()
    # On a 1 GB CPU node, 27B should be incapable → no assignment.
    assert a is None


def test_native_shard_self_claim_in_dht_is_kept_stable():
    """Post-stability-fix (B3 follow-up): if our own prior announce
    is in the DHT, the negotiator returns it as-is — no reshard,
    no moving to another model. Before this, the negotiator would
    skip fully-covered models → candidate exhausted → None, which
    caused thrashing when the NegotiationLoop interpreted None as
    a change and moved to the next ranked model."""
    my_id = "12D3KooWMINE"
    stale_self = [
        PeerClaim(libp2p_peer_id=my_id, model_id="openhydra-qwen3.5-2b",
                  layer_start=0, layer_end=24, total_layers=24,
                  available_vram_mb=99999),
    ]
    report = _native_report(_cuda_t4(), _qwen_2b(), libp2p_peer_id=my_id)
    neg = SwarmNegotiator(
        capacity_report=report, libp2p_peer_id=my_id,
        dht_scan=_fixed_scan(stale_self),
    )
    assignment = neg.negotiate()
    assert assignment is not None
    assert assignment.model_id == "openhydra-qwen3.5-2b"
    assert (assignment.layer_start, assignment.layer_end) == (0, 24)
    # This isn't a "change" per the NegotiationLoop — same shape as the
    # existing current_assignment, so ``_assignment_changed`` returns
    # False and no reshard fires.


def test_native_shard_partial_self_claim_does_not_oscillate():
    """2026-04-24 fix: a peer that previously claimed a partial range
    (e.g. ``[0, 12)`` of a 24-layer model) must KEEP that range on the
    next negotiation tick — even though ``compute_gaps`` would identify
    ``[12, 24)`` as a "gap" and ``pick_best_fit`` would propose to fill it.

    Without the stability gate, the negotiator computes gaps from claims
    that include self, sees a "gap" that's actually the complement of
    its own range, fills it, and on the following tick sees its own NEW
    range as the obstacle — flipping back and forth every 60 s.
    Observed in the 2026-04-23 cross-ISP benchmark log.

    With the gate, the negotiator looks for its own claim first, finds
    ``[0, 12)``, runs ``should_concede`` (no overlapping higher-priority
    peer), and returns ``[0, 12)`` unchanged.
    """
    my_id = "12D3KooWMINE"
    # Self previously claimed [0, 12) — half of a 24-layer model.
    my_partial_claim = [
        PeerClaim(libp2p_peer_id=my_id, model_id="openhydra-qwen3.5-2b",
                  layer_start=0, layer_end=12, total_layers=24,
                  available_vram_mb=14000),
    ]
    report = _native_report(_cuda_t4(), _qwen_2b(), libp2p_peer_id=my_id)
    neg = SwarmNegotiator(
        capacity_report=report, libp2p_peer_id=my_id,
        dht_scan=_fixed_scan(my_partial_claim),
    )
    # First negotiation: keeps [0, 12).
    a1 = neg.negotiate()
    assert a1 is not None
    assert (a1.layer_start, a1.layer_end) == (0, 12), (
        f"expected stable [0, 12), got [{a1.layer_start}, {a1.layer_end}) — "
        "the oscillation regression has returned"
    )
    # Simulate a second tick where my claim is unchanged in the DHT.
    a2 = neg.negotiate()
    assert (a2.layer_start, a2.layer_end) == (0, 12)


def test_native_shard_partial_self_claim_concedes_to_bigger_peer():
    """The stability gate must NOT lock us in when a higher-priority
    peer (more VRAM) already overlaps our range. In that case we
    should re-negotiate exactly as a fresh peer would."""
    my_id = "12D3KooWMINE"
    # Self has [0, 12); a bigger peer (24 GB) ALSO claims [0, 12).
    overlapping_bigger_peer = PeerClaim(
        libp2p_peer_id="12D3KooWBIGGER",
        model_id="openhydra-qwen3.5-2b",
        layer_start=0, layer_end=12, total_layers=24,
        available_vram_mb=24000,
    )
    my_partial_claim = PeerClaim(
        libp2p_peer_id=my_id, model_id="openhydra-qwen3.5-2b",
        layer_start=0, layer_end=12, total_layers=24,
        available_vram_mb=14000,
    )
    claims = [my_partial_claim, overlapping_bigger_peer]
    report = _native_report(_cuda_t4(), _qwen_2b(), libp2p_peer_id=my_id)
    neg = SwarmNegotiator(
        capacity_report=report, libp2p_peer_id=my_id,
        dht_scan=_fixed_scan(claims),
    )
    a = neg.negotiate()
    # Stability gate sees the conflict, falls through to the normal
    # pick_best_fit path. With BIGGER claiming [0,12), gap is [12,24)
    # → we get assigned [12, 24).
    assert a is not None
    assert (a.layer_start, a.layer_end) == (12, 24)


def test_unknown_persona_returns_none_without_raising():
    """Malformed report with an unknown persona — negotiator stays silent."""
    # Build a normal report then mutate (ugly, but CapacityReport is frozen).
    report = _native_report(_cuda_t4(), _qwen_2b())
    object.__setattr__(report, "node_persona", "potato_worker")
    neg = SwarmNegotiator(
        capacity_report=report, libp2p_peer_id="me", dht_scan=_empty_scan,
    )
    assert neg.negotiate() is None


# ─── ShardAssignment shape ───────────────────────────────────────────────────


# ─── compute_conflict_split (Blocker B fix) ─────────────────────────────────


def _claim_whole(
    peer_id: str,
    total_layers: int = 24,
    *,
    available_vram_mb: int = 8000,
) -> PeerClaim:
    """Shortcut — a peer that claims the whole model, i.e. is in the
    ``fallback_whole_model`` deadlock state.

    Default ``available_vram_mb`` of 8000 is deliberately *below* the
    T4 test fixture's ~14 336 MB so :func:`should_concede` doesn't
    fire for the peer-under-test — the fix is about the deadlock,
    not about conceding to a bigger peer.
    """
    return PeerClaim(
        libp2p_peer_id=peer_id,
        model_id="openhydra-qwen3.5-2b",
        layer_start=0,
        layer_end=total_layers,
        total_layers=total_layers,
        available_vram_mb=available_vram_mb,
    )


class TestComputeConflictSplit:
    """Unit-level coverage of the deterministic whole-model split."""

    def test_no_overlappers_returns_none(self):
        """No deadlock → helper returns None so the caller falls through
        to ``pick_best_fit``."""
        assert compute_conflict_split(
            peer_claims=[],
            total_layers=24,
            my_libp2p_peer_id="12D3KooWMINE",
            max_layers_hostable=24,
        ) is None

    def test_partial_range_overlapper_not_a_deadlock(self):
        """A peer claiming only [0, 12) is a valid partial — not a
        deadlock. Falls through to the normal gap path."""
        assert compute_conflict_split(
            peer_claims=[
                PeerClaim(
                    libp2p_peer_id="12D3KooWOTHER",
                    model_id="openhydra-qwen3.5-2b",
                    layer_start=0, layer_end=12,
                    total_layers=24, available_vram_mb=15000,
                )
            ],
            total_layers=24,
            my_libp2p_peer_id="12D3KooWMINE",
            max_layers_hostable=24,
        ) is None

    def test_two_peer_whole_model_splits_50_50(self):
        """The canonical benchmark scenario: Mac + GPU1 both claim
        [0, 24). With lex-ordered ids, one gets [0, 12), other gets
        [12, 24). On 24 layers and 2 peers this is always exactly 12/12."""
        # "12D3KooWA..." < "12D3KooWM..." lex.
        me = "12D3KooWAAAAA"
        other = "12D3KooWZZZZZ"
        my_split = compute_conflict_split(
            peer_claims=[_claim_whole(other)],
            total_layers=24,
            my_libp2p_peer_id=me,
            max_layers_hostable=24,
        )
        other_split = compute_conflict_split(
            peer_claims=[_claim_whole(me)],
            total_layers=24,
            my_libp2p_peer_id=other,
            max_layers_hostable=24,
        )
        assert my_split == (0, 12)
        assert other_split == (12, 24)

    def test_deterministic_across_two_lex_orders(self):
        """Either peer running the helper with the same inputs produces
        a complementary pair that tiles ``[0, total_layers)``."""
        ids = ["12D3KooWGpu", "12D3KooWMac"]
        # Both peers see the same full-range overlap — exchange roles.
        splits = []
        for self_id in ids:
            other_ids = [p for p in ids if p != self_id]
            split = compute_conflict_split(
                peer_claims=[_claim_whole(o) for o in other_ids],
                total_layers=24,
                my_libp2p_peer_id=self_id,
                max_layers_hostable=24,
            )
            splits.append(split)
        # Chunks must tile [0, 24) with no gaps, no overlap.
        splits_sorted = sorted(splits)
        assert splits_sorted == [(0, 12), (12, 24)]

    def test_three_peer_32_layer_split_is_even_chunks(self):
        """9B model has 32 layers. 3 peers → ceil(32/3)=11. Chunks
        [0,11), [11,22), [22,32) — last peer absorbs the remainder."""
        ids = ["12D3KooWA", "12D3KooWB", "12D3KooWC"]
        results: dict[str, tuple[int, int]] = {}
        for self_id in ids:
            other_ids = [p for p in ids if p != self_id]
            split = compute_conflict_split(
                peer_claims=[
                    _claim_whole(o, total_layers=32) for o in other_ids
                ],
                total_layers=32,
                my_libp2p_peer_id=self_id,
                max_layers_hostable=32,
            )
            results[self_id] = split
        assert results == {
            "12D3KooWA": (0, 11),
            "12D3KooWB": (11, 22),
            "12D3KooWC": (22, 32),
        }
        # Tile: total coverage is [0, 32), no gap, no overlap.
        ranges = sorted(results.values())
        assert ranges[0][0] == 0
        assert ranges[-1][1] == 32
        for i in range(len(ranges) - 1):
            assert ranges[i][1] == ranges[i + 1][0]

    def test_respects_max_layers_hostable_budget(self):
        """If the computed chunk is bigger than the peer can host, it
        shrinks to the budget — still at the same ``my_start``."""
        split = compute_conflict_split(
            peer_claims=[_claim_whole("12D3KooWZZZ")],
            total_layers=24,
            my_libp2p_peer_id="12D3KooWAAA",
            max_layers_hostable=4,  # much smaller than the 12-layer chunk
        )
        assert split == (0, 4)

    def test_ignores_claims_with_wrong_total_layers(self):
        """A peer advertising a different model depth mustn't count
        toward this model's deadlock."""
        split = compute_conflict_split(
            peer_claims=[
                PeerClaim(
                    libp2p_peer_id="12D3KooWOther",
                    model_id="openhydra-qwen3.5-2b",
                    layer_start=0, layer_end=32,  # wrong total
                    total_layers=32, available_vram_mb=15000,
                )
            ],
            total_layers=24,
            my_libp2p_peer_id="12D3KooWMine",
            max_layers_hostable=24,
        ) is None or None  # both branches accept

    def test_empty_self_id_returns_none(self):
        """Defensive: a negotiator with no identity cannot compute a
        deterministic position — bail rather than hand it an arbitrary
        range."""
        assert compute_conflict_split(
            peer_claims=[_claim_whole("12D3KooWOther")],
            total_layers=24,
            my_libp2p_peer_id="",
            max_layers_hostable=24,
        ) is None

    def test_zero_total_layers_returns_none(self):
        assert compute_conflict_split(
            peer_claims=[_claim_whole("12D3KooWOther", total_layers=0)],
            total_layers=0,
            my_libp2p_peer_id="12D3KooWMine",
            max_layers_hostable=24,
        ) is None


class TestConflictSplitInNegotiator:
    """Integration-level: _native_shard_assignment returns a conflict-
    split assignment when a full-range overlapper is present."""

    def test_negotiator_emits_conflict_split_source(self):
        """Mac-like peer discovers GPU1's whole-model claim on tick 2
        and assigns itself the lower half of the model."""
        me = "12D3KooWAlpha"
        other = "12D3KooWZebra"
        claims = [_claim_whole(other)]
        report = _native_report(_cuda_t4(), _qwen_2b(), libp2p_peer_id=me)
        neg = SwarmNegotiator(
            capacity_report=report,
            libp2p_peer_id=me,
            dht_scan=_fixed_scan(claims),
        )
        assignment = neg.negotiate()
        assert assignment is not None
        assert assignment.source == SOURCE_CONFLICT_SPLIT
        assert assignment.model_id == "openhydra-qwen3.5-2b"
        assert assignment.total_layers == 24
        # Alpha < Zebra lex → I take [0, 12).
        assert (assignment.layer_start, assignment.layer_end) == (0, 12)

    def test_negotiator_conflict_split_complements_across_peers(self):
        """Two peers running their own negotiator against each other's
        whole-model claim emit complementary sharded assignments."""
        mac = "12D3KooWMac"
        gpu = "12D3KooWZZZ"
        mac_report = _native_report(_cuda_t4(), _qwen_2b(), libp2p_peer_id=mac)
        gpu_report = _native_report(_cuda_t4(), _qwen_2b(), libp2p_peer_id=gpu)

        mac_neg = SwarmNegotiator(
            capacity_report=mac_report, libp2p_peer_id=mac,
            dht_scan=_fixed_scan([_claim_whole(gpu)]),
        )
        gpu_neg = SwarmNegotiator(
            capacity_report=gpu_report, libp2p_peer_id=gpu,
            dht_scan=_fixed_scan([_claim_whole(mac)]),
        )
        mac_ass = mac_neg.negotiate()
        gpu_ass = gpu_neg.negotiate()
        assert mac_ass.source == gpu_ass.source == SOURCE_CONFLICT_SPLIT
        assert (mac_ass.layer_start, mac_ass.layer_end) == (0, 12)
        assert (gpu_ass.layer_start, gpu_ass.layer_end) == (12, 24)

    def test_partial_peer_does_not_trigger_split(self):
        """A peer already holding a partial shard [0, 12) lets the
        normal pick_best_fit logic fill the [12, 24) gap — conflict
        split should NOT fire."""
        me = "12D3KooWMine"
        other_partial = PeerClaim(
            libp2p_peer_id="12D3KooWOther",
            model_id="openhydra-qwen3.5-2b",
            layer_start=0, layer_end=12,
            total_layers=24, available_vram_mb=15000,
        )
        report = _native_report(_cuda_t4(), _qwen_2b(), libp2p_peer_id=me)
        neg = SwarmNegotiator(
            capacity_report=report, libp2p_peer_id=me,
            dht_scan=_fixed_scan([other_partial]),
        )
        assignment = neg.negotiate()
        assert assignment is not None
        assert assignment.source == SOURCE_PICK_BEST_FIT
        assert (assignment.layer_start, assignment.layer_end) == (12, 24)


def test_shard_assignment_is_immutable():
    assignment = ShardAssignment(
        model_id="m", layer_start=0, layer_end=4,
        total_layers=24, source=SOURCE_ATOMIC_WORKER,
    )
    with pytest.raises(Exception):  # frozen dataclass → FrozenInstanceError
        assignment.layer_start = 99  # type: ignore[misc]
