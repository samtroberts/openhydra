# Copyright 2026 OpenHydra contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Phase 2b — DFlash telemetry surface.

The six-metric surface Phase 3's auto-negotiator consumes (Phase 2b
plan §9). All metrics live in one ``DFlashTelemetry`` singleton so
the negotiator can read them with a single import; producers
(``DFlashTopologyADriver``, the runtime hooks, the failover manager)
push updates through the singleton's ``record_*`` methods.

| Metric                        | Source              | Phase 3 use |
|---|---|---|
| ``draft.inflight_p50_ms``     | drafter             | "is the drafter too slow for the link?" |
| ``draft.ram_mb``              | drafter loader      | "can this peer host the drafter?" |
| ``target.verify_block_p50_ms``| coord verify path   | compare against draft latency |
| ``ring.acceptance_rate_ema``  | block-verify result | tune block_size / topology |
| ``peer.gpu_free_ram_mb``      | peer announce       | capacity for promoting drafter location |
| ``peer.target_layers_owned``  | peer announce       | imbalance detection |

Streaming P50 calculation: we use a fixed-size circular buffer per
metric and compute the median over the last N observations. Cheap,
deterministic, and good enough for the negotiator's coarse-grained
decisions (block_size choice, topology flip thresholds). Production
deployments that want tighter accuracy can swap in a tdigest later.

The EMA for ``ring.acceptance_rate_ema`` uses α=0.1 — picks up
sustained shifts within ~10 blocks while smoothing noise from
single-block outliers.

This module is process-local (no swarm broadcast). The negotiator
reads from the local singleton; cross-peer aggregation happens via
the existing peer-announce mechanism, where each peer's announce
already carries a metrics dict that we just extend with these
fields.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import logging
import statistics
import threading
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = [
    "DFlashTelemetry",
    "TelemetrySnapshot",
    "get_telemetry",
    "reset_telemetry",
]


# Window over which we compute streaming P50 latencies. 64 samples
# means a draft taking 12 ms shows up in the P50 within ~12 blocks
# of generation. Smaller windows are noisier; larger windows lag
# regime changes.
_LATENCY_WINDOW = 64

# EMA smoothing factor for acceptance rate. α=0.1 → half-life ≈ 7
# blocks. Tuned to track sustained acceptance shifts (e.g. moving
# from a code prompt to a chat prompt) without overreacting to
# single-block outliers.
_EMA_ALPHA = 0.1


@dataclass(frozen=True)
class TelemetrySnapshot:
    """Point-in-time read of all six metrics.

    Used by the Phase 3 auto-negotiator (which polls this every
    few seconds) and by the ``/v1/internal/telemetry`` HTTP endpoint
    so operators can scrape the values via the existing prometheus-
    compat path.
    """

    draft_inflight_p50_ms: Optional[float] = None
    draft_ram_mb: Optional[int] = None
    target_verify_block_p50_ms: Optional[float] = None
    ring_acceptance_rate_ema: Optional[float] = None
    peer_gpu_free_ram_mb: Optional[int] = None
    peer_target_layers_owned: Optional[int] = None


@dataclass
class _MetricState:
    draft_inflight_ms: deque = field(default_factory=lambda: deque(maxlen=_LATENCY_WINDOW))
    draft_ram_mb: Optional[int] = None
    verify_block_ms: deque = field(default_factory=lambda: deque(maxlen=_LATENCY_WINDOW))
    acceptance_ema: Optional[float] = None
    peer_gpu_free_ram_mb: Optional[int] = None
    peer_target_layers_owned: Optional[int] = None


class DFlashTelemetry:
    """Process-local telemetry singleton.

    All ``record_*`` methods are thread-safe (mutex-guarded) so the
    coord-side block-verify thread, the drafter thread, and the
    peer-announce loop can all push concurrently.
    """

    def __init__(self) -> None:
        self._state = _MetricState()
        self._lock = threading.Lock()

    # ── Drafter metrics ────────────────────────────────────────────

    def record_draft_inflight_ms(self, ms: float) -> None:
        """Per-draft latency, measured around ``Drafter.draft()``.
        Updates the streaming-P50 window."""
        if ms < 0:
            return  # defence — clock skew on a bad system
        with self._lock:
            self._state.draft_inflight_ms.append(float(ms))

    def record_draft_ram_mb(self, mb: int) -> None:
        """Drafter weight footprint. Measured at load time on the
        host that's currently hosting the drafter (coord under
        Topology A, stage-0 under Topology B). Static after load
        — the negotiator reads this once per topology change."""
        if mb < 0:
            return
        with self._lock:
            self._state.draft_ram_mb = int(mb)

    # ── Coord verify metrics ───────────────────────────────────────

    def record_verify_block_ms(self, ms: float) -> None:
        """Per-block latency, measured around the ring verify pass."""
        if ms < 0:
            return
        with self._lock:
            self._state.verify_block_ms.append(float(ms))

    def record_block_acceptance(self, accepted_len: int, block_size: int) -> None:
        """Per-block acceptance rate; folded into the EMA."""
        if block_size <= 0:
            return
        rate = max(0.0, min(1.0, accepted_len / float(block_size)))
        with self._lock:
            current = self._state.acceptance_ema
            if current is None:
                self._state.acceptance_ema = rate
            else:
                self._state.acceptance_ema = (
                    _EMA_ALPHA * rate + (1.0 - _EMA_ALPHA) * current
                )

    # ── Peer-side capacity metrics ────────────────────────────────

    def record_peer_gpu_free_ram_mb(self, mb: int) -> None:
        """Pushed by the local peer's announce loop."""
        if mb < 0:
            return
        with self._lock:
            self._state.peer_gpu_free_ram_mb = int(mb)

    def record_peer_target_layers_owned(self, count: int) -> None:
        """Number of transformer layers this peer's shard owns.
        Updated on reshard."""
        if count < 0:
            return
        with self._lock:
            self._state.peer_target_layers_owned = int(count)

    # ── Snapshot ───────────────────────────────────────────────────

    def snapshot(self) -> TelemetrySnapshot:
        """Atomic read of all six metrics."""
        with self._lock:
            draft_p50 = (
                statistics.median(self._state.draft_inflight_ms)
                if self._state.draft_inflight_ms else None
            )
            verify_p50 = (
                statistics.median(self._state.verify_block_ms)
                if self._state.verify_block_ms else None
            )
            return TelemetrySnapshot(
                draft_inflight_p50_ms=draft_p50,
                draft_ram_mb=self._state.draft_ram_mb,
                target_verify_block_p50_ms=verify_p50,
                ring_acceptance_rate_ema=self._state.acceptance_ema,
                peer_gpu_free_ram_mb=self._state.peer_gpu_free_ram_mb,
                peer_target_layers_owned=self._state.peer_target_layers_owned,
            )

    def reset(self) -> None:
        """Test helper. Clears every metric back to None / empty."""
        with self._lock:
            self._state = _MetricState()


# ── Process-local singleton ────────────────────────────────────────────

_INSTANCE: Optional[DFlashTelemetry] = None
_INSTANCE_LOCK = threading.Lock()


def get_telemetry() -> DFlashTelemetry:
    """Return the process-local singleton, creating it on first call."""
    global _INSTANCE
    if _INSTANCE is None:
        with _INSTANCE_LOCK:
            if _INSTANCE is None:
                _INSTANCE = DFlashTelemetry()
    return _INSTANCE


def reset_telemetry() -> None:
    """Test helper — drop the singleton so each test starts fresh."""
    global _INSTANCE
    with _INSTANCE_LOCK:
        _INSTANCE = None
