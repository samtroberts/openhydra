# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Phase 2b — DFlash telemetry surface tests.

Locks the six-metric API the Phase 3 auto-negotiator depends on.
A regression that drops a field or changes its semantics shows up
here before Phase 3 silently makes the wrong topology decision.
"""

from __future__ import annotations

import threading

import pytest

from coordinator.dflash_telemetry import (
    DFlashTelemetry,
    TelemetrySnapshot,
    get_telemetry,
    reset_telemetry,
)


@pytest.fixture(autouse=True)
def _fresh_singleton():
    reset_telemetry()
    yield
    reset_telemetry()


# ── Snapshot starts empty ──────────────────────────────────────────────

def test_snapshot_starts_with_all_none():
    """Before any record_*, every field must be None — Phase 3
    interprets None as 'metric not yet observed; skip this signal'."""
    snap = get_telemetry().snapshot()
    assert snap.draft_inflight_p50_ms is None
    assert snap.draft_ram_mb is None
    assert snap.target_verify_block_p50_ms is None
    assert snap.ring_acceptance_rate_ema is None
    assert snap.peer_gpu_free_ram_mb is None
    assert snap.peer_target_layers_owned is None


# ── Streaming P50 over the latency window ──────────────────────────────

def test_draft_inflight_p50_after_three_observations():
    t = get_telemetry()
    t.record_draft_inflight_ms(10.0)
    t.record_draft_inflight_ms(20.0)
    t.record_draft_inflight_ms(30.0)
    assert t.snapshot().draft_inflight_p50_ms == 20.0


def test_verify_block_p50_after_observations():
    t = get_telemetry()
    for ms in (5.0, 7.0, 9.0, 11.0, 13.0):
        t.record_verify_block_ms(ms)
    assert t.snapshot().target_verify_block_p50_ms == 9.0


def test_latency_window_drops_oldest_after_64():
    """The deque has maxlen=64 — sample 65 should evict sample 1."""
    t = get_telemetry()
    for i in range(65):
        t.record_draft_inflight_ms(float(i))
    # Window now holds samples 1..64 → median is 32.5.
    assert t.snapshot().draft_inflight_p50_ms == 32.5


def test_negative_latency_observations_ignored():
    """Clock skew sends negative deltas occasionally on Linux —
    drop those rather than poisoning the P50."""
    t = get_telemetry()
    t.record_draft_inflight_ms(10.0)
    t.record_draft_inflight_ms(-5.0)        # ignored
    t.record_draft_inflight_ms(20.0)
    assert t.snapshot().draft_inflight_p50_ms == 15.0


# ── EMA acceptance rate ────────────────────────────────────────────────

def test_acceptance_ema_first_observation_seeds():
    """First record sets the EMA to the observed rate exactly —
    subsequent records blend with α=0.1."""
    t = get_telemetry()
    t.record_block_acceptance(accepted_len=8, block_size=16)
    assert abs(t.snapshot().ring_acceptance_rate_ema - 0.5) < 1e-9


def test_acceptance_ema_smooths_observations():
    """α=0.1 → EMA = 0.1*new + 0.9*old. Manual trace verifies."""
    t = get_telemetry()
    t.record_block_acceptance(accepted_len=8, block_size=16)    # 0.5
    # Second observation: 1.0 → EMA = 0.1*1.0 + 0.9*0.5 = 0.55
    t.record_block_acceptance(accepted_len=16, block_size=16)
    assert abs(t.snapshot().ring_acceptance_rate_ema - 0.55) < 1e-9
    # Third: 0.0 → EMA = 0.1*0.0 + 0.9*0.55 = 0.495
    t.record_block_acceptance(accepted_len=0, block_size=16)
    assert abs(t.snapshot().ring_acceptance_rate_ema - 0.495) < 1e-9


def test_acceptance_clamps_to_unit_interval():
    """Defence in depth — accepted_len > block_size shouldn't
    happen but the negotiator depends on ema ∈ [0, 1]."""
    t = get_telemetry()
    t.record_block_acceptance(accepted_len=99, block_size=16)
    assert t.snapshot().ring_acceptance_rate_ema == 1.0


def test_acceptance_zero_block_size_ignored():
    t = get_telemetry()
    t.record_block_acceptance(accepted_len=0, block_size=0)
    assert t.snapshot().ring_acceptance_rate_ema is None


# ── RAM / capacity metrics ─────────────────────────────────────────────

def test_draft_ram_mb_static_after_load():
    t = get_telemetry()
    t.record_draft_ram_mb(1850)
    assert t.snapshot().draft_ram_mb == 1850
    # Subsequent records overwrite (drafter reload).
    t.record_draft_ram_mb(2050)
    assert t.snapshot().draft_ram_mb == 2050


def test_peer_gpu_free_ram_mb_overwrites():
    t = get_telemetry()
    t.record_peer_gpu_free_ram_mb(8192)
    t.record_peer_gpu_free_ram_mb(4096)
    assert t.snapshot().peer_gpu_free_ram_mb == 4096


def test_peer_target_layers_owned_records():
    t = get_telemetry()
    t.record_peer_target_layers_owned(12)
    assert t.snapshot().peer_target_layers_owned == 12


def test_negative_ram_observations_ignored():
    """Defence — record_*_mb with negative input drops silently
    rather than corrupting the snapshot."""
    t = get_telemetry()
    t.record_draft_ram_mb(1000)
    t.record_draft_ram_mb(-1)   # ignored
    assert t.snapshot().draft_ram_mb == 1000


# ── Singleton + reset ──────────────────────────────────────────────────

def test_get_telemetry_returns_same_instance():
    a = get_telemetry()
    b = get_telemetry()
    assert a is b


def test_reset_telemetry_drops_singleton():
    a = get_telemetry()
    a.record_draft_ram_mb(500)
    reset_telemetry()
    b = get_telemetry()
    assert b is not a
    assert b.snapshot().draft_ram_mb is None


def test_explicit_reset_clears_state():
    t = get_telemetry()
    t.record_draft_inflight_ms(50.0)
    t.record_block_acceptance(8, 16)
    t.reset()
    snap = t.snapshot()
    assert snap.draft_inflight_p50_ms is None
    assert snap.ring_acceptance_rate_ema is None


# ── Concurrency ────────────────────────────────────────────────────────

def test_concurrent_record_does_not_race():
    """All record_* methods are mutex-guarded. Sanity: 4 threads
    each pushing 100 observations should produce 400 observations
    in the window (well within the 64-sample maxlen, so the deque
    holds the last 64) without raising or losing samples mid-deque
    operation."""
    t = get_telemetry()
    n_threads = 4
    n_per = 100

    def worker(thread_id: int):
        for i in range(n_per):
            t.record_draft_inflight_ms(float(thread_id * 1000 + i))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    # The deque maxlen is 64. After 400 observations only the last
    # 64 remain. We can't predict exact values (interleaving) but
    # we can confirm the snapshot is well-formed.
    snap = t.snapshot()
    assert snap.draft_inflight_p50_ms is not None
    assert snap.draft_inflight_p50_ms >= 0


def test_telemetry_snapshot_is_immutable():
    """TelemetrySnapshot is a frozen dataclass — concurrent readers
    get a consistent point-in-time view that doesn't tear if a
    writer pushes mid-snapshot."""
    snap = TelemetrySnapshot(draft_ram_mb=1000)
    with pytest.raises(Exception):  # FrozenInstanceError
        snap.draft_ram_mb = 2000   # type: ignore[misc]
