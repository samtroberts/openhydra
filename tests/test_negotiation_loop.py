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

"""Unit tests for peer.negotiation_loop (Phase 4 — continuous re-negotiation).

Tests drive ``tick_once()`` synchronously with injected fakes for the
capacity builder, negotiator factory, and is_busy probe.  No real
threads are started except in the dedicated ``start/stop`` lifecycle
tests.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

import pytest

from coordinator.degradation import ModelAvailability
from peer.capacity import (
    NODE_PERSONA_NATIVE_SHARD,
    build_capacity_report,
)
from peer.hardware import HardwareProfile
from peer.negotiation_loop import (
    DEFAULT_NEGOTIATION_INTERVAL_S,
    LoopSnapshot,
    MIN_NEGOTIATION_INTERVAL_S,
    NegotiationLoop,
    SKIP_REASON_BUSY,
    SKIP_REASON_CAPACITY_BUILD_FAILED,
    SKIP_REASON_NEGOTIATION_FAILED,
    SKIP_REASON_NONE,
    TickResult,
    _assignment_changed,
)
from peer.swarm_negotiator import (
    PeerClaim,
    ShardAssignment,
    SOURCE_ATOMIC_WORKER,
    SOURCE_FALLBACK_WHOLE,
    SOURCE_PICK_BEST_FIT,
    SwarmNegotiator,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────


def _hw() -> HardwareProfile:
    return HardwareProfile(
        ram_total_bytes=16 * 1024**3, ram_available_bytes=12 * 1024**3,
        accelerator="cuda",
        vram_total_bytes=15 * 1024**3, vram_available_bytes=14 * 1024**3,
        cuda_device_count=1,
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


def _report_for(vram_available_bytes: int | None = None):
    """Build a CapacityReport with a specific VRAM snapshot so we can
    assert that refreshed reports propagate through the loop."""
    hw = HardwareProfile(
        ram_total_bytes=16 * 1024**3, ram_available_bytes=12 * 1024**3,
        accelerator="cuda",
        vram_total_bytes=15 * 1024**3,
        vram_available_bytes=vram_available_bytes if vram_available_bytes is not None
                              else 14 * 1024**3,
        cuda_device_count=1,
    )
    return build_capacity_report(
        hardware=hw,
        catalog=[_qwen_2b(), _qwen_9b()],
        peer_id="me", libp2p_peer_id="12D3KooWME",
        ports={"api": 8080, "grpc": 50051, "libp2p": 4001},
        advertise_host="10.0.0.1",
    )


def _negotiator_from_report(report, dht_claims: list[PeerClaim] | None = None) -> SwarmNegotiator:
    claims = list(dht_claims or [])
    return SwarmNegotiator(
        capacity_report=report,
        libp2p_peer_id="12D3KooWME",
        dht_scan=lambda _mid: list(claims),
    )


def _make_loop(
    *,
    is_busy_fn: Callable[[], bool] | None = None,
    initial_assignment: ShardAssignment | None = None,
    dht_claims: list[PeerClaim] | None = None,
    build_report_fn: Callable[[], object] | None = None,
    make_negotiator_fn: Callable[..., SwarmNegotiator] | None = None,
    snapshot: LoopSnapshot | None = None,
    interval_s: float = 1.0,
) -> tuple[NegotiationLoop, LoopSnapshot]:
    snapshot = snapshot or LoopSnapshot.build(
        capacity_json="",
        capacity_schema_version=0,
        current_assignment=initial_assignment,
    )
    if build_report_fn is None:
        def build_report_fn():
            return _report_for()
    if make_negotiator_fn is None:
        def make_negotiator_fn(report):
            return _negotiator_from_report(report, dht_claims)

    loop = NegotiationLoop(
        build_capacity_report_fn=build_report_fn,
        make_negotiator_fn=make_negotiator_fn,
        snapshot=snapshot,
        initial_assignment=initial_assignment,
        is_busy_fn=is_busy_fn,
        interval_s=interval_s,
    )
    return loop, snapshot


# ─── LoopSnapshot thread-safety basics ───────────────────────────────────────


def test_loop_snapshot_build_sets_timestamp():
    s = LoopSnapshot.build(capacity_json="{}", capacity_schema_version=2)
    _, _, _, ts = s.snapshot()
    assert ts > 0


def test_loop_snapshot_update_replaces_all_fields_atomically():
    s = LoopSnapshot.build(capacity_json="{\"v\":1}", capacity_schema_version=1)
    a = ShardAssignment(model_id="m", layer_start=0, layer_end=4,
                        total_layers=8, source=SOURCE_PICK_BEST_FIT)
    s.update(capacity_json="{\"v\":2}", capacity_schema_version=2, current_assignment=a)
    cj, sv, snap_a, _ = s.snapshot()
    assert cj == "{\"v\":2}"
    assert sv == 2
    assert snap_a == a


def test_loop_snapshot_concurrent_readers_never_see_torn_state():
    """A rapid reader should always see a self-consistent triple, never a mix
    of old + new fields."""
    s = LoopSnapshot.build(capacity_json="v1", capacity_schema_version=1)
    stop = threading.Event()
    errors: list[str] = []

    def _reader():
        while not stop.is_set():
            cj, sv, _a, _ts = s.snapshot()
            # Versioning invariant: cj "vN" ↔ sv == N.
            if cj.startswith("v"):
                try:
                    declared = int(cj[1:])
                except ValueError:
                    continue
                if declared != sv:
                    errors.append(f"torn: {cj!r} sv={sv}")

    r = threading.Thread(target=_reader, daemon=True)
    r.start()
    # Writer flips version 100 times.
    for n in range(2, 102):
        s.update(
            capacity_json=f"v{n}",
            capacity_schema_version=n,
            current_assignment=None,
        )
    stop.set()
    r.join(timeout=2.0)
    assert errors == [], errors[:5]


# ─── interval clamping ───────────────────────────────────────────────────────


def test_interval_clamped_to_minimum():
    """interval_s below MIN_NEGOTIATION_INTERVAL_S must be clamped up."""
    loop, _ = _make_loop(interval_s=0.01)
    assert loop.interval_s == MIN_NEGOTIATION_INTERVAL_S


def test_default_interval_is_60_seconds():
    loop = NegotiationLoop(
        build_capacity_report_fn=lambda: _report_for(),
        make_negotiator_fn=lambda r: _negotiator_from_report(r),
        snapshot=LoopSnapshot.build(),
    )
    assert loop.interval_s == DEFAULT_NEGOTIATION_INTERVAL_S


# ─── idle gating (guardrail #2) ──────────────────────────────────────────────


def test_busy_peer_skips_renegotiation_but_still_refreshes_capacity():
    """When busy: capacity_json still updates but the assignment does not."""
    initial = ShardAssignment(model_id="openhydra-qwen3.5-2b",
                              layer_start=0, layer_end=4, total_layers=24,
                              source=SOURCE_PICK_BEST_FIT)
    # DHT is empty → if we DID re-negotiate, we'd take the whole 2B model
    # and the assignment would flip to layer_end=24.  But we're busy, so
    # it must stay at layer_end=4.
    loop, snap = _make_loop(
        is_busy_fn=lambda: True,
        initial_assignment=initial,
        dht_claims=[],
    )
    result = loop.tick_once()
    assert result.skipped_reason == SKIP_REASON_BUSY
    assert result.capacity_refreshed is True
    assert result.assignment_changed is False
    assert result.new_assignment == initial
    # Assignment in snapshot unchanged too.
    _, _, snap_a, _ = snap.snapshot()
    assert snap_a == initial
    # But capacity_json was updated (length > 0).
    cj, sv, _, _ = snap.snapshot()
    assert len(cj) > 0
    assert sv == 2


def test_idle_peer_renegotiates_and_updates_assignment():
    """Idle peer with better DHT visibility should pick up a bigger range."""
    initial = ShardAssignment(model_id="openhydra-qwen3.5-2b",
                              layer_start=0, layer_end=4, total_layers=24,
                              source=SOURCE_PICK_BEST_FIT)
    loop, snap = _make_loop(
        is_busy_fn=lambda: False,
        initial_assignment=initial,
        dht_claims=[],  # empty DHT → whole 2B available
    )
    result = loop.tick_once()
    assert result.skipped_reason == SKIP_REASON_NONE
    assert result.assignment_changed is True
    assert result.new_assignment is not None
    assert result.new_assignment.layer_end == 24  # full 2B
    assert loop.assignment_change_count == 1
    _, _, snap_a, _ = snap.snapshot()
    assert snap_a is not None and snap_a.layer_end == 24


def test_is_busy_fn_exception_treated_as_busy():
    """If the idle check raises, we conservatively assume busy (safer than
    accidentally reshard-ing on a spurious state)."""
    initial = ShardAssignment(model_id="openhydra-qwen3.5-2b",
                              layer_start=0, layer_end=4, total_layers=24,
                              source=SOURCE_PICK_BEST_FIT)
    def _angry_busy():
        raise RuntimeError("probe exploded")

    loop, _ = _make_loop(is_busy_fn=_angry_busy, initial_assignment=initial)
    result = loop.tick_once()
    assert result.skipped_reason == SKIP_REASON_BUSY
    assert result.new_assignment == initial


# ─── dynamic capacity refresh (guardrail #3) ─────────────────────────────────


def test_capacity_refresh_picks_up_vram_drift():
    """Swap the build_report closure between ticks to simulate VRAM drift
    and verify the snapshot shows the newer numbers."""
    # First tick: full VRAM.
    # Second tick: half VRAM.
    call_count = [0]
    vram_values = [14 * 1024**3, 7 * 1024**3]

    def _drifting_builder():
        vram = vram_values[min(call_count[0], len(vram_values) - 1)]
        call_count[0] += 1
        return _report_for(vram_available_bytes=vram)

    loop, snap = _make_loop(
        build_report_fn=_drifting_builder,
        is_busy_fn=lambda: False,
    )

    loop.tick_once()
    cj_first, _, _, _ = snap.snapshot()
    loop.tick_once()
    cj_second, _, _, _ = snap.snapshot()

    assert cj_first != cj_second, "capacity_json must differ after VRAM drift"
    # Parse both and assert the usable_memory_mb differs.
    import json
    first = json.loads(cj_first)
    second = json.loads(cj_second)
    assert first["hardware"]["usable_memory_mb"] > second["hardware"]["usable_memory_mb"]


def test_capacity_build_failure_keeps_previous_snapshot():
    """When build_report raises, we don't touch the snapshot."""
    snap = LoopSnapshot.build(
        capacity_json="{\"keep\":\"me\"}", capacity_schema_version=2,
    )
    raising_count = [0]

    def _sometimes_raise():
        raising_count[0] += 1
        if raising_count[0] <= 3:
            raise RuntimeError("build failed")
        return _report_for()

    loop = NegotiationLoop(
        build_capacity_report_fn=_sometimes_raise,
        make_negotiator_fn=lambda r: _negotiator_from_report(r),
        snapshot=snap,
        is_busy_fn=lambda: False,
    )
    r1 = loop.tick_once()
    assert r1.skipped_reason == SKIP_REASON_CAPACITY_BUILD_FAILED
    cj, _, _, _ = snap.snapshot()
    assert cj == "{\"keep\":\"me\"}"


# ─── negotiate failure handling ──────────────────────────────────────────────


def test_negotiate_failure_preserves_assignment_and_refreshes_capacity():
    initial = ShardAssignment(model_id="openhydra-qwen3.5-2b",
                              layer_start=8, layer_end=16, total_layers=24,
                              source=SOURCE_PICK_BEST_FIT)

    def _raising_factory(_report):
        class _Broken:
            def negotiate(self):
                raise RuntimeError("DHT timeout")
        return _Broken()

    loop, snap = _make_loop(
        initial_assignment=initial,
        make_negotiator_fn=_raising_factory,
        is_busy_fn=lambda: False,
    )
    r = loop.tick_once()
    assert r.skipped_reason == SKIP_REASON_NEGOTIATION_FAILED
    # Capacity still refreshed.
    assert r.capacity_refreshed is True
    # Assignment preserved.
    _, _, snap_a, _ = snap.snapshot()
    assert snap_a == initial


# ─── no-op ticks when assignment stays stable ────────────────────────────────


def test_stable_assignment_does_not_increment_change_counter():
    """Two ticks with identical DHT state and no conflicts → assignment
    computed the same both times, change_count stays at 1 (first tick's
    initial assignment transitioning from None to first compute)."""
    # Seed the loop with the same assignment it would naturally compute.
    initial = ShardAssignment(
        model_id="openhydra-qwen3.5-2b",
        layer_start=0, layer_end=24, total_layers=24,
        source=SOURCE_FALLBACK_WHOLE,
    )
    loop, _ = _make_loop(
        initial_assignment=initial,
        dht_claims=[],
        is_busy_fn=lambda: False,
    )
    r1 = loop.tick_once()
    r2 = loop.tick_once()
    assert r1.assignment_changed is False, "first tick re-computes same value"
    assert r2.assignment_changed is False
    assert loop.assignment_change_count == 0


# ─── conflict resolution mid-life ────────────────────────────────────────────


def test_higher_vram_peer_arrives_forces_concede_to_different_model():
    """Initially we hold 2B [0,24).  Next tick a bigger peer claims 2B
    fully → we concede and (preferred_model_order empty) take 9B instead."""
    initial = ShardAssignment(
        model_id="openhydra-qwen3.5-2b",
        layer_start=0, layer_end=24, total_layers=24,
        source=SOURCE_FALLBACK_WHOLE,
    )

    # DHT has a bigger peer claiming the full 2B.
    claims = [
        PeerClaim(libp2p_peer_id="FAT_PEER", model_id="openhydra-qwen3.5-2b",
                  layer_start=0, layer_end=24, total_layers=24,
                  available_vram_mb=99999)
    ]

    def _selective_make_negotiator(report):
        # Our own capacity report's VRAM is lower than FAT_PEER's 99999 MB,
        # and 2B appears fully covered → negotiator picks 9B.
        return SwarmNegotiator(
            capacity_report=report,
            libp2p_peer_id="12D3KooWME",
            dht_scan=lambda mid: list(claims) if mid == "openhydra-qwen3.5-2b" else [],
        )

    loop, _ = _make_loop(
        initial_assignment=initial,
        is_busy_fn=lambda: False,
        make_negotiator_fn=_selective_make_negotiator,
    )
    r = loop.tick_once()
    assert r.assignment_changed is True
    assert r.new_assignment is not None
    assert r.new_assignment.model_id == "openhydra-qwen3.5-9b"
    assert loop.assignment_change_count == 1


# ─── thread lifecycle (one real-thread test to exercise start/stop) ─────────


def test_start_then_stop_terminates_cleanly():
    """Lifecycle smoke: start the loop, wait for a tick, stop, join."""
    loop, _ = _make_loop(is_busy_fn=lambda: False, interval_s=5.0)
    t = loop.start()
    # Wait for at least one tick (runs immediately on thread entry).
    for _ in range(40):
        if loop.tick_count > 0:
            break
        time.sleep(0.05)
    assert loop.tick_count >= 1
    loop.stop(join_timeout_s=2.0)
    assert not t.is_alive()


def test_double_start_is_idempotent():
    loop, _ = _make_loop(is_busy_fn=lambda: False, interval_s=5.0)
    t1 = loop.start()
    t2 = loop.start()
    assert t1 is t2
    loop.stop(join_timeout_s=2.0)


# ─── assignment change semantics ────────────────────────────────────────────


def test_assignment_changed_detects_model_swap():
    a = ShardAssignment(model_id="m1", layer_start=0, layer_end=4,
                        total_layers=8, source=SOURCE_PICK_BEST_FIT)
    b = ShardAssignment(model_id="m2", layer_start=0, layer_end=4,
                        total_layers=8, source=SOURCE_PICK_BEST_FIT)
    assert _assignment_changed(a, b) is True


def test_assignment_changed_ignores_source_field():
    """The ``source`` field is advisory — same range + same model = no change,
    even if we arrived at it via a different heuristic path."""
    a = ShardAssignment(model_id="m1", layer_start=0, layer_end=4,
                        total_layers=8, source=SOURCE_PICK_BEST_FIT)
    b = ShardAssignment(model_id="m1", layer_start=0, layer_end=4,
                        total_layers=8, source=SOURCE_FALLBACK_WHOLE)
    assert _assignment_changed(a, b) is False


def test_assignment_changed_detects_layer_start_change():
    a = ShardAssignment(model_id="m", layer_start=0, layer_end=4,
                        total_layers=8, source=SOURCE_PICK_BEST_FIT)
    b = ShardAssignment(model_id="m", layer_start=2, layer_end=4,
                        total_layers=8, source=SOURCE_PICK_BEST_FIT)
    assert _assignment_changed(a, b) is True


def test_assignment_changed_none_to_some_is_change():
    a = None
    b = ShardAssignment(model_id="m", layer_start=0, layer_end=4,
                        total_layers=8, source=SOURCE_ATOMIC_WORKER)
    assert _assignment_changed(a, b) is True
    assert _assignment_changed(b, a) is True


def test_assignment_changed_none_to_none_is_no_change():
    assert _assignment_changed(None, None) is False


# ─── counters & introspection ───────────────────────────────────────────────


def test_counters_increment_correctly_across_mixed_ticks():
    """Run a mixed sequence of tick scenarios and verify counter hygiene."""
    busy_flag = [False]
    claims_box: list[list[PeerClaim]] = [[]]  # mutable so we can swap mid-test
    initial = ShardAssignment(
        model_id="openhydra-qwen3.5-2b",
        layer_start=0, layer_end=24, total_layers=24,
        source=SOURCE_FALLBACK_WHOLE,
    )

    def _factory(report):
        return SwarmNegotiator(
            capacity_report=report,
            libp2p_peer_id="12D3KooWME",
            dht_scan=lambda mid: list(claims_box[0]) if mid == "openhydra-qwen3.5-2b" else [],
        )

    loop, _ = _make_loop(
        initial_assignment=initial,
        is_busy_fn=lambda: busy_flag[0],
        make_negotiator_fn=_factory,
    )

    # Tick 1: idle, no claims → same assignment, no change.
    r1 = loop.tick_once()
    assert r1.skipped_reason == SKIP_REASON_NONE
    assert r1.assignment_changed is False

    # Tick 2: busy → skip.
    busy_flag[0] = True
    r2 = loop.tick_once()
    assert r2.skipped_reason == SKIP_REASON_BUSY
    assert r2.assignment_changed is False

    # Tick 3: idle + FAT peer claims 2B fully → we flip to 9B.
    busy_flag[0] = False
    claims_box[0] = [PeerClaim(
        libp2p_peer_id="FAT_PEER", model_id="openhydra-qwen3.5-2b",
        layer_start=0, layer_end=24, total_layers=24, available_vram_mb=99999,
    )]
    r3 = loop.tick_once()
    assert r3.skipped_reason == SKIP_REASON_NONE
    assert r3.assignment_changed is True

    assert loop.tick_count == 3
    assert loop.skip_count == 1
    assert loop.assignment_change_count == 1
