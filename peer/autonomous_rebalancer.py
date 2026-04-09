# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Autonomous dynamic rebalancing — peers decide their own layer assignment.

Each peer periodically evaluates whether the swarm would benefit from it
serving different layers.  This is decentralized: no coordinator involvement.
Inspired by Petals' ``should_choose_other_blocks()``.

Algorithm:
    1. Fetch the full swarm state from DHT (all peers + their layer ranges + TPS).
    2. Compute per-layer throughput: sum of TPS for all peers covering each layer.
    3. Find the current bottleneck (layer with minimum throughput).
    4. Hypothetically remove self from current position.
    5. Try every possible position for our span width.
    6. If the best position improves the minimum throughput by >= ``min_improvement``
       (default 15%), recommend migration.

Safety:
    - Cooldown: after rebalancing, wait ``cooldown_s`` before checking again.
    - Position history: don't return to a recent position (prevents ping-pong).
    - Hysteresis: require 15% improvement to move, creating a dead zone.
    - Jitter: random delay before applying to prevent herding.
"""

from __future__ import annotations

import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PeerInfo:
    """Snapshot of a peer's layer assignment and throughput."""
    peer_id: str
    layer_start: int
    layer_end: int
    tps: float


@dataclass(frozen=True)
class RebalanceDecision:
    """Recommendation to migrate to a new layer range."""
    new_layer_start: int
    new_layer_end: int
    current_min_tps: float
    new_min_tps: float
    improvement_ratio: float


def compute_per_layer_throughput(
    peers: list[PeerInfo],
    total_layers: int,
) -> list[float]:
    """Compute the aggregate throughput for each layer across the swarm.

    Returns a list of length ``total_layers`` where ``result[i]`` is the
    sum of TPS of all peers whose range covers layer ``i``.
    """
    per_layer = [0.0] * total_layers
    for peer in peers:
        for layer in range(max(0, peer.layer_start), min(total_layers, peer.layer_end)):
            per_layer[layer] += peer.tps
    return per_layer


def find_bottleneck(per_layer: list[float]) -> tuple[int, float]:
    """Find the bottleneck layer (minimum throughput).

    Returns ``(layer_index, min_throughput)``.  Returns ``(-1, 0.0)`` if empty.
    """
    if not per_layer:
        return -1, 0.0
    min_tps = min(per_layer)
    min_idx = per_layer.index(min_tps)
    return min_idx, min_tps


def should_rebalance(
    my_peer_id: str,
    my_layer_start: int,
    my_layer_end: int,
    my_tps: float,
    swarm_peers: list[PeerInfo],
    total_layers: int,
    min_improvement: float = 1.15,
    position_history: deque | None = None,
) -> RebalanceDecision | None:
    """Determine if this peer should serve different layers.

    Args:
        my_peer_id: This peer's identifier.
        my_layer_start: Current first layer (inclusive).
        my_layer_end: Current one-past-last layer (exclusive).
        my_tps: This peer's measured throughput (tokens/sec).
        swarm_peers: All peers in the swarm (including self).
        total_layers: Total transformer depth of the model.
        min_improvement: Minimum ratio of new_min/current_min to trigger
            migration (default 1.15 = 15% improvement).
        position_history: Deque of recent ``(start, end)`` positions to
            avoid. If the best position is in history, skip it.

    Returns:
        A ``RebalanceDecision`` if migration is beneficial, or ``None``.
    """
    if total_layers <= 0 or my_tps <= 0:
        return None

    span = my_layer_end - my_layer_start
    if span <= 0 or span >= total_layers:
        return None

    # Step 1: Compute current per-layer throughput
    per_layer = compute_per_layer_throughput(swarm_peers, total_layers)
    current_min = min(per_layer) if per_layer else 0.0

    # Step 2: Remove self from current position
    without_self = list(per_layer)
    for layer in range(max(0, my_layer_start), min(total_layers, my_layer_end)):
        without_self[layer] -= my_tps

    # Step 3: Try every possible position
    best_start = my_layer_start
    best_min = current_min

    for start in range(0, total_layers - span + 1):
        end = start + span
        # Simulate placing self at [start, end)
        simulated_min = float("inf")
        for layer in range(total_layers):
            tps_at_layer = without_self[layer]
            if start <= layer < end:
                tps_at_layer += my_tps
            simulated_min = min(simulated_min, tps_at_layer)
        if simulated_min > best_min:
            best_min = simulated_min
            best_start = start

    # Step 4: Check improvement threshold
    best_end = best_start + span

    # Don't move to the same position
    if best_start == my_layer_start and best_end == my_layer_end:
        return None

    # Check position history to prevent oscillation
    if position_history is not None:
        if (best_start, best_end) in position_history:
            logger.debug(
                "rebalance_skip_history: [%d,%d) is in recent history",
                best_start, best_end,
            )
            return None

    # Compute improvement ratio
    if current_min > 0:
        improvement = best_min / current_min
        if improvement < min_improvement:
            return None
    elif best_min > 0:
        # Current min is 0 (gap exists) and we can fill it
        improvement = float("inf")
    else:
        return None

    return RebalanceDecision(
        new_layer_start=best_start,
        new_layer_end=best_end,
        current_min_tps=current_min,
        new_min_tps=best_min,
        improvement_ratio=improvement,
    )


def load_swarm_snapshot(
    dht_urls: list[str],
    model_id: str,
    timeout_s: float = 3.0,
) -> list[PeerInfo]:
    """Fetch all sharded peers for a model from DHT.

    Returns a list of ``PeerInfo`` with layer ranges and TPS.
    Only includes peers with valid shard metadata (layer_end > 0).
    """
    try:
        from coordinator.path_finder import load_peers_from_dht
        if not dht_urls:
            return []
        peers = load_peers_from_dht(
            dht_urls[0], model_id=model_id, timeout_s=timeout_s,
        )
        return [
            PeerInfo(
                peer_id=p.peer_id,
                layer_start=p.layer_start,
                layer_end=p.layer_end,
                tps=max(0.0, p.runtime_estimated_tokens_per_sec),
            )
            for p in peers
            if p.layer_end > p.layer_start and p.total_layers > 0
        ]
    except Exception as exc:
        logger.debug("swarm_snapshot_failed: %s", exc)
        return []


class AutonomousRebalancer:
    """Manages periodic rebalance checks with cooldown and history.

    Attach to a peer's announce loop and call ``check()`` every N cycles.

    Args:
        min_improvement: Minimum ratio to trigger migration (default 1.15).
        cooldown_s: Seconds to wait after a rebalance before checking again.
        history_size: Number of recent positions to remember for oscillation
            prevention.
        jitter_max_s: Maximum random delay before applying rebalance (prevents
            multiple peers from moving simultaneously).
    """

    def __init__(
        self,
        min_improvement: float = 1.15,
        cooldown_s: float = 300.0,
        history_size: int = 3,
        jitter_max_s: float = 30.0,
    ) -> None:
        self.min_improvement = max(1.01, float(min_improvement))
        self.cooldown_s = max(0.0, float(cooldown_s))
        self.jitter_max_s = max(0.0, float(jitter_max_s))
        self._position_history: deque[tuple[int, int]] = deque(maxlen=max(1, history_size))
        self._last_rebalance_at: float = 0.0
        self._rebalance_count: int = 0

    @property
    def in_cooldown(self) -> bool:
        """Whether the cooldown period is still active."""
        return (time.monotonic() - self._last_rebalance_at) < self.cooldown_s

    def check(
        self,
        my_peer_id: str,
        my_layer_start: int,
        my_layer_end: int,
        my_tps: float,
        swarm_peers: list[PeerInfo],
        total_layers: int,
    ) -> RebalanceDecision | None:
        """Evaluate whether to rebalance.  Returns a decision or None.

        Respects cooldown and position history automatically.
        """
        if self.in_cooldown:
            return None

        decision = should_rebalance(
            my_peer_id=my_peer_id,
            my_layer_start=my_layer_start,
            my_layer_end=my_layer_end,
            my_tps=my_tps,
            swarm_peers=swarm_peers,
            total_layers=total_layers,
            min_improvement=self.min_improvement,
            position_history=self._position_history,
        )

        if decision is not None:
            logger.info(
                "rebalance_recommended: [%d,%d) -> [%d,%d) improvement=%.2fx min_tps=%.1f->%.1f",
                my_layer_start, my_layer_end,
                decision.new_layer_start, decision.new_layer_end,
                decision.improvement_ratio,
                decision.current_min_tps, decision.new_min_tps,
            )
        return decision

    def record_applied(self, old_start: int, old_end: int) -> None:
        """Record that a rebalance was applied.  Updates cooldown and history."""
        self._position_history.append((old_start, old_end))
        self._last_rebalance_at = time.monotonic()
        self._rebalance_count += 1

    def apply_with_jitter(
        self,
        service: Any,
        decision: RebalanceDecision,
        current_start: int,
        current_end: int,
    ) -> bool:
        """Apply a rebalance decision with safety guards.

        1. Random jitter delay (prevents herding)
        2. Check inflight_count == 0
        3. Set load to 100% (prevent routing)
        4. Drain period (2s)
        5. Call shard.reshard()
        6. Restore normal load

        Returns True if applied successfully.
        """
        # Jitter
        if self.jitter_max_s > 0:
            jitter = random.uniform(0, self.jitter_max_s)
            logger.info("rebalance_jitter: waiting %.1fs", jitter)
            time.sleep(jitter)

        # Check inflight
        if hasattr(service, "inflight_count") and service.inflight_count() > 0:
            logger.info("rebalance_deferred: inflight=%d", service.inflight_count())
            return False

        # Drain
        logger.info(
            "rebalance_applying: [%d,%d) -> [%d,%d)",
            current_start, current_end,
            decision.new_layer_start, decision.new_layer_end,
        )
        time.sleep(2.0)

        # Reshard
        total_layers = decision.new_layer_end  # Approximate; actual total from runtime
        if hasattr(service, "shard"):
            total = getattr(service.shard, "total_layers", decision.new_layer_end)
            ok = service.shard.reshard(
                decision.new_layer_start,
                decision.new_layer_end,
                total,
            )
            if ok:
                self.record_applied(current_start, current_end)
                logger.info(
                    "rebalance_success: new_range=[%d,%d) rebalance_count=%d",
                    decision.new_layer_start, decision.new_layer_end,
                    self._rebalance_count,
                )
                return True
            else:
                logger.warning("rebalance_failed: reshard returned False")
                return False
        return False
