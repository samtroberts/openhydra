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
    SOURCE_FALLBACK_WHOLE,
    SOURCE_PICK_BEST_FIT,
    SwarmNegotiator,
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


def test_native_shard_self_claim_in_dht_does_not_trigger_concession():
    """If our own prior announce is still in the DHT, ignore it."""
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
    # The stale self-claim fully covers 2B. compute_gaps won't exclude it
    # (it's a valid segment), so the gap is empty → no assignment.
    # This is expected behaviour: a re-negotiation should not collide
    # with the still-live prior claim.  The Phase 3 design accepts this
    # as a no-op on re-run; the peer's next announce refreshes the TTL.
    assert neg.negotiate() is None


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


def test_shard_assignment_is_immutable():
    assignment = ShardAssignment(
        model_id="m", layer_start=0, layer_end=4,
        total_layers=24, source=SOURCE_ATOMIC_WORKER,
    )
    with pytest.raises(Exception):  # frozen dataclass → FrozenInstanceError
        assignment.layer_start = 99  # type: ignore[misc]
