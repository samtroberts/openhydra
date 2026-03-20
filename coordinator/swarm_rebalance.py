"""Swarm rebalancing engine — dynamic layer-window migration (Phase 2B).

Inspired by Petals' ``block_selection.py``, this module detects throughput
bottlenecks across the layer coverage map and generates migration directives
that tell peers to shift their layer windows to cover under-served layers.

Algorithm
---------
1. Compute per-layer throughput: for each layer index, sum the TPS of all
   peers whose range covers that layer.
2. Identify the **bottleneck layer** — the layer with minimum throughput.
3. For each candidate peer, simulate migrating its layer window to cover the
   bottleneck.  If the migration improves the network's minimum throughput
   by at least ``balance_quality`` (default 1.15×), emit a
   ``RebalanceDirective``.

The coordinator posts directives to the DHT under
``rebalance_{peer_id}`` keys with a short TTL (120 s).  Peers poll for
directives in their announce loop and execute them once inflight requests
have drained.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib import request as urllib_request

from coordinator.layer_coverage import LayerRange, PeerMetrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RebalanceDirective:
    """Instruction for a peer to migrate its layer window.

    Attributes:
        target_peer_id: The peer that should reshard.
        new_layer_start: New inclusive start layer.
        new_layer_end: New exclusive end layer.
        total_layers: Full model depth.
        reason: Human-readable explanation.
        issued_at_ms: Unix timestamp (ms) when the directive was created.
    """

    target_peer_id: str
    new_layer_start: int
    new_layer_end: int
    total_layers: int
    reason: str = ""
    issued_at_ms: int = 0


def compute_per_layer_throughput(
    ranges: list[LayerRange],
    peer_metrics: dict[str, PeerMetrics],
    total_layers: int,
) -> list[float]:
    """Compute the aggregate TPS covering each layer index.

    Args:
        ranges: Layer ranges from all peers in the swarm.
        peer_metrics: Runtime metrics keyed by peer_id.
        total_layers: Full model depth.

    Returns:
        List of length ``total_layers`` where ``result[i]`` is the sum of
        ``estimated_tps`` for all peers whose range covers layer ``i``.
    """
    throughput = [0.0] * total_layers
    for lr in ranges:
        tps = 0.0
        m = peer_metrics.get(lr.peer_id)
        if m is not None:
            tps = max(0.0, m.estimated_tps)
        for layer_idx in range(max(0, lr.layer_start), min(total_layers, lr.layer_end)):
            throughput[layer_idx] += tps
    return throughput


def find_bottleneck_layer(throughput: list[float]) -> tuple[int, float]:
    """Return the layer index with the lowest throughput and its value.

    Args:
        throughput: Per-layer throughput from ``compute_per_layer_throughput``.

    Returns:
        Tuple of ``(layer_index, min_throughput)``.  Returns ``(-1, 0.0)``
        if the throughput list is empty.
    """
    if not throughput:
        return (-1, 0.0)
    min_tps = float("inf")
    min_idx = 0
    for i, tps in enumerate(throughput):
        if tps < min_tps:
            min_tps = tps
            min_idx = i
    return (min_idx, min_tps)


def simulate_migration(
    ranges: list[LayerRange],
    peer_metrics: dict[str, PeerMetrics],
    total_layers: int,
    candidate_peer_id: str,
    new_start: int,
    new_end: int,
) -> float:
    """Simulate migrating *candidate_peer_id* and return the new min throughput.

    Replaces the candidate's current range with ``[new_start, new_end)`` and
    recomputes the per-layer throughput, returning the new minimum.

    Args:
        ranges: Current layer ranges.
        peer_metrics: Runtime metrics keyed by peer_id.
        total_layers: Full model depth.
        candidate_peer_id: Peer to migrate.
        new_start: Proposed new layer_start.
        new_end: Proposed new layer_end.

    Returns:
        The new minimum per-layer throughput after the simulated migration.
    """
    simulated = [
        LayerRange(
            peer_id=lr.peer_id,
            layer_start=new_start if lr.peer_id == candidate_peer_id else lr.layer_start,
            layer_end=new_end if lr.peer_id == candidate_peer_id else lr.layer_end,
            total_layers=total_layers,
        )
        for lr in ranges
    ]
    throughput = compute_per_layer_throughput(simulated, peer_metrics, total_layers)
    _, new_min = find_bottleneck_layer(throughput)
    return new_min


class SwarmRebalancer:
    """Detects throughput bottlenecks and generates migration directives.

    Args:
        balance_quality: Minimum improvement ratio required to trigger
            a migration (e.g. 1.15 means a 15% improvement is needed).
        directive_ttl_s: TTL in seconds for DHT-posted directives.
        min_throughput_for_rebalance: Minimum current throughput below which
            rebalancing is skipped (avoids churn when the swarm is tiny).
    """

    def __init__(
        self,
        balance_quality: float = 1.15,
        directive_ttl_s: int = 120,
        min_throughput_for_rebalance: float = 0.0,
    ) -> None:
        self.balance_quality = max(1.0, float(balance_quality))
        self.directive_ttl_s = max(10, int(directive_ttl_s))
        self.min_throughput_for_rebalance = max(0.0, float(min_throughput_for_rebalance))

    def evaluate(
        self,
        ranges: list[LayerRange],
        peer_metrics: dict[str, PeerMetrics],
        total_layers: int,
    ) -> list[RebalanceDirective]:
        """Evaluate the swarm and generate migration directives.

        Examines all peers and, for each one, checks whether migrating it
        to cover the current bottleneck layer would improve the network's
        minimum throughput by at least ``balance_quality``.

        Args:
            ranges: Current layer ranges from all peers.
            peer_metrics: Runtime metrics keyed by peer_id.
            total_layers: Full model depth.

        Returns:
            List of ``RebalanceDirective`` objects (may be empty if no
            beneficial migrations are found).
        """
        if total_layers <= 0 or not ranges:
            return []

        throughput = compute_per_layer_throughput(ranges, peer_metrics, total_layers)
        bottleneck_idx, current_min = find_bottleneck_layer(throughput)

        if bottleneck_idx < 0:
            return []

        # Skip rebalancing if the swarm is too small to meaningfully rebalance.
        if current_min < self.min_throughput_for_rebalance:
            return []

        directives: list[RebalanceDirective] = []
        now_ms = int(time.time() * 1000)

        for lr in ranges:
            # Don't try to migrate a peer that already covers the bottleneck.
            if lr.covers_layer(bottleneck_idx):
                continue

            # Simulate migrating this peer to centre on the bottleneck.
            # Keep the peer's current span width but shift it to cover the bottleneck.
            span = lr.span
            if span <= 0:
                continue
            new_start = max(0, bottleneck_idx - span // 2)
            new_end = min(total_layers, new_start + span)
            # Adjust start if end was clamped.
            new_start = max(0, new_end - span)

            new_min = simulate_migration(
                ranges, peer_metrics, total_layers,
                lr.peer_id, new_start, new_end,
            )

            # Check if migration improves throughput enough.
            if current_min > 0 and new_min / current_min >= self.balance_quality:
                directives.append(RebalanceDirective(
                    target_peer_id=lr.peer_id,
                    new_layer_start=new_start,
                    new_layer_end=new_end,
                    total_layers=total_layers,
                    reason=f"bottleneck_layer={bottleneck_idx} current_min={current_min:.1f} new_min={new_min:.1f}",
                    issued_at_ms=now_ms,
                ))
            elif current_min == 0 and new_min > 0:
                # Special case: bottleneck has zero coverage.
                directives.append(RebalanceDirective(
                    target_peer_id=lr.peer_id,
                    new_layer_start=new_start,
                    new_layer_end=new_end,
                    total_layers=total_layers,
                    reason=f"zero_coverage_layer={bottleneck_idx} new_min={new_min:.1f}",
                    issued_at_ms=now_ms,
                ))

        return directives


# ── DHT directive posting / polling ──────────────────────────────────────────


def post_directive_to_dht(
    directive: RebalanceDirective,
    dht_url: str,
    ttl_seconds: int = 120,
    timeout_s: float = 2.0,
) -> bool:
    """Post a rebalance directive to the DHT bootstrap node.

    The directive is stored under the key ``rebalance_{peer_id}`` with a
    short TTL so it auto-expires if the peer doesn't poll in time.

    Args:
        directive: The migration directive to post.
        dht_url: DHT bootstrap URL.
        ttl_seconds: Time-to-live for the directive in the DHT.
        timeout_s: HTTP request timeout.

    Returns:
        ``True`` if the directive was posted successfully.
    """
    body = {
        "key": f"rebalance_{directive.target_peer_id}",
        "value": json.dumps(asdict(directive)),
        "ttl_seconds": ttl_seconds,
    }
    payload = json.dumps(body).encode("utf-8")
    try:
        req = urllib_request.Request(
            url=f"{dht_url.rstrip('/')}/store",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib_request.urlopen(req, timeout=timeout_s) as response:
            return response.status == 200
    except Exception as exc:
        logger.debug("post_directive_failed: %s %s", directive.target_peer_id, exc)
        return False


def poll_directive_from_dht(
    peer_id: str,
    dht_url: str,
    timeout_s: float = 2.0,
) -> RebalanceDirective | None:
    """Poll the DHT for a rebalance directive targeting this peer.

    Args:
        peer_id: This peer's identifier.
        dht_url: DHT bootstrap URL to query.
        timeout_s: HTTP request timeout.

    Returns:
        A ``RebalanceDirective`` if one exists, otherwise ``None``.
    """
    try:
        url = f"{dht_url.rstrip('/')}/lookup?key=rebalance_{peer_id}"
        req = urllib_request.Request(url=url, method="GET")
        with urllib_request.urlopen(req, timeout=timeout_s) as response:
            data = json.loads(response.read().decode("utf-8"))
            value = data.get("value")
            if not value:
                return None
            parsed = json.loads(value) if isinstance(value, str) else value
            return RebalanceDirective(
                target_peer_id=str(parsed.get("target_peer_id", "")),
                new_layer_start=int(parsed.get("new_layer_start", 0)),
                new_layer_end=int(parsed.get("new_layer_end", 0)),
                total_layers=int(parsed.get("total_layers", 0)),
                reason=str(parsed.get("reason", "")),
                issued_at_ms=int(parsed.get("issued_at_ms", 0)),
            )
    except Exception as exc:
        logger.debug("poll_directive_failed: %s %s", peer_id, exc)
        return None


def apply_directive_safely(
    service: Any,
    directive: RebalanceDirective,
    drain_timeout_s: float = 30.0,
) -> bool:
    """Apply a rebalance directive to a peer service with safety guards.

    Waits for all inflight requests to drain (up to ``drain_timeout_s``)
    before calling ``shard.reshard()``.  If requests don't drain in time,
    the directive is skipped.

    Args:
        service: The ``PeerService`` instance.
        directive: The migration directive to apply.
        drain_timeout_s: Maximum seconds to wait for inflight drain.

    Returns:
        ``True`` if resharding succeeded, ``False`` if skipped or failed.
    """
    import time as _time

    # Safety guard: wait for inflight requests to drain.
    deadline = _time.monotonic() + drain_timeout_s
    while _time.monotonic() < deadline:
        with service._lock:
            inflight = service._inflight
        if inflight == 0:
            break
        _time.sleep(0.1)
    else:
        with service._lock:
            inflight = service._inflight
        if inflight > 0:
            logger.warning(
                "rebalance_skipped: peer=%s inflight=%d did not drain within %.1fs",
                directive.target_peer_id, inflight, drain_timeout_s,
            )
            return False

    # Apply the reshard.
    success = service.shard.reshard(
        directive.new_layer_start,
        directive.new_layer_end,
        directive.total_layers,
    )
    if success:
        # Update the runtime profile so the next announce cycle broadcasts
        # the new layer range.
        service.runtime_profile = dict(service.shard.runtime_profile())
        logger.info(
            "rebalance_applied: peer=%s new_range=[%d, %d) total=%d reason=%s",
            directive.target_peer_id,
            directive.new_layer_start,
            directive.new_layer_end,
            directive.total_layers,
            directive.reason,
        )
    else:
        logger.warning(
            "rebalance_failed: peer=%s reshard returned False",
            directive.target_peer_id,
        )
    return success
