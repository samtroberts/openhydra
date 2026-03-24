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

"""Layer-range coverage tracking for sharded inference pipelines.

Phase 3: Layer Sharding Activation.

Terminology
-----------
layer_start : int
    Index of the first transformer layer this peer handles (inclusive).
layer_end : int
    Index past the last layer this peer handles (exclusive).
    The peer covers layers in ``[layer_start, layer_end)``.
total_layers : int
    Full depth of the model (e.g. 32 for LLaMA-3-8B, 80 for LLaMA-3-70B).

A peer with ``layer_end == 0`` or ``total_layers == 0`` is considered **not
sharded** — it runs a full-model replica (the pre-Phase-3 default).

A pipeline is *complete* when the union of peer layer ranges covers
``[0, total_layers)`` without gaps.

Coverage algorithm
------------------
``find_complete_pipeline`` uses a greedy interval-covering approach::

    pos = 0
    while pos < total_layers:
        pick the peer whose range starts at or before pos and extends farthest
        advance pos to that peer's layer_end

This is optimal (fewest stages, minimal latency) and runs in O(n log n) after
sorting by ``layer_start``.

Usage example
-------------
::

    from coordinator.layer_coverage import LayerCoverageMap
    from coordinator.path_finder import PeerEndpoint

    cmap = LayerCoverageMap.from_endpoints(healthy_peers)
    if cmap.is_complete():
        pipeline = cmap.best_pipeline()   # ordered list[LayerRange]
        # route the request through pipeline[0] → pipeline[1] → …
"""
from __future__ import annotations

import heapq
import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Data types ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LayerRange:
    """Layer-range announcement from one peer.

    Attributes
    ----------
    peer_id:      Unique peer identifier.
    layer_start:  First transformer layer handled by this peer (inclusive).
    layer_end:    One past the last layer (exclusive); peer covers [start, end).
    total_layers: Full model depth reported by this peer (0 = unknown).
    host:         Network host for gRPC connections.
    port:         Network port for gRPC connections.
    """

    peer_id: str
    layer_start: int       # inclusive
    layer_end: int         # exclusive
    total_layers: int      # full model depth; 0 = unknown / unsharded
    host: str = ""
    port: int = 0

    @property
    def is_sharded(self) -> bool:
        """True when this peer covers a *proper* sub-range of the model.

        A full-model replica (layer_start=0, layer_end=total_layers) or an
        unsharded peer (total_layers=0 / layer_end=0) returns ``False``.
        """
        return (
            self.total_layers > 0
            and self.layer_end > 0
            and (self.layer_start > 0 or self.layer_end < self.total_layers)
        )

    @property
    def span(self) -> int:
        """Number of layers this peer covers (``layer_end - layer_start``)."""
        return max(0, self.layer_end - self.layer_start)

    def covers_layer(self, layer: int) -> bool:
        """Return ``True`` when *layer* falls within ``[layer_start, layer_end)``."""
        return self.layer_start <= layer < self.layer_end

    def overlaps(self, other: LayerRange) -> bool:
        """Return ``True`` when the two ranges share at least one layer index."""
        return self.layer_start < other.layer_end and other.layer_start < self.layer_end


# ── Pure functions ───────────────────────────────────────────────────────────


def coverage_gaps(
    ranges: list[LayerRange],
    total_layers: int,
) -> list[tuple[int, int]]:
    """Return uncovered intervals in ``[0, total_layers)`` as ``(start, end)`` pairs.

    An empty return list means the ranges provide full, gap-free coverage.

    Parameters
    ----------
    ranges:       Layer ranges to inspect (order and overlap do not matter).
    total_layers: Model depth defining the target interval ``[0, total_layers)``.

    Examples
    --------
    >>> coverage_gaps([LayerRange("a", 0, 16, 32), LayerRange("b", 16, 32, 32)], 32)
    []
    >>> coverage_gaps([LayerRange("a", 0, 12, 32)], 32)
    [(12, 32)]
    >>> coverage_gaps([], 32)
    [(0, 32)]
    """
    if total_layers <= 0:
        return []
    if not ranges:
        return [(0, total_layers)]

    # Clamp each interval to [0, total_layers] and sort by start.
    intervals = sorted(
        (max(0, r.layer_start), min(total_layers, r.layer_end))
        for r in ranges
        if r.layer_end > r.layer_start
    )

    gaps: list[tuple[int, int]] = []
    pos = 0
    for start, end in intervals:
        if start > pos:
            gaps.append((pos, start))
        pos = max(pos, end)

    if pos < total_layers:
        gaps.append((pos, total_layers))

    return gaps


def is_complete_coverage(
    ranges: list[LayerRange],
    total_layers: int,
) -> bool:
    """Return ``True`` when *ranges* cover every layer in ``[0, total_layers)``."""
    return total_layers > 0 and not coverage_gaps(ranges, total_layers)


def find_complete_pipeline(
    ranges: list[LayerRange],
    total_layers: int,
) -> list[LayerRange] | None:
    """Greedy search for an ordered list of ranges that cover ``[0, total_layers)``.

    Algorithm
    ---------
    Starting from position 0, at each step we choose the peer whose range
    starts at or *before* the current frontier and extends it the *farthest*.
    This greedy choice is optimal: it minimises the number of pipeline stages
    for contiguous, non-overlapping coverage.

    Parameters
    ----------
    ranges:       Candidate layer ranges (any order; may overlap).
    total_layers: Target coverage depth.

    Returns
    -------
    Ordered ``list[LayerRange]`` from layer 0 → total_layers, or ``None`` if
    complete coverage is impossible with the given ranges.

    Examples
    --------
    >>> ranges = [LayerRange("a", 0, 16, 32), LayerRange("b", 16, 32, 32)]
    >>> find_complete_pipeline(ranges, 32)
    [LayerRange(peer_id='a', ...), LayerRange(peer_id='b', ...)]
    """
    if total_layers <= 0 or not ranges:
        return None

    pipeline: list[LayerRange] = []
    pos = 0
    used: set[str] = set()

    while pos < total_layers:
        # Peers that can advance the frontier: start ≤ pos and end > pos.
        candidates = [
            r for r in ranges
            if r.peer_id not in used
            and r.layer_start <= pos
            and r.layer_end > pos
        ]
        if not candidates:
            return None     # gap: no peer covers position *pos*
        # Greedy: pick the peer that extends the frontier the farthest.
        best = max(candidates, key=lambda r: r.layer_end)
        pipeline.append(best)
        used.add(best.peer_id)
        pos = best.layer_end

    return pipeline


# ── Dijkstra cost-optimal pipeline ────────────────────────────────────────────


@dataclass(frozen=True)
class PeerMetrics:
    """Runtime metrics for one peer, used by Dijkstra cost-optimal routing.

    All fields have sensible defaults so callers can populate only what's
    available from the health subsystem.

    Attributes
    ----------
    latency_ms:       Round-trip time to this peer in milliseconds (0 = unknown).
    estimated_tps:    Estimated tokens/second throughput (0 = unknown).
    reputation_score: Trust score 0–100 (100 = fully trusted, 0 = untrusted).
    load_pct:         Current load as a fraction 0.0–1.0 (0 = idle, 1 = saturated).
    """

    latency_ms: float = 0.0
    estimated_tps: float = 0.0
    reputation_score: float = 50.0
    load_pct: float = 0.0
    # Phase 2A: KV cache availability (0 = unknown or full).
    available_kv_slots: int = 0
    # Phase 2A: Measured RTT (ms) from this peer to downstream peers.
    # Keyed by downstream peer_id; populated from DHT announcement.
    next_hop_rtts: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineWeights:
    """Tunable weights for the Dijkstra edge-cost formula.

    The total cost of routing through a peer is::

        cost = w_rtt   × latency_ms
             + w_infer × infer_time_ms
             + w_rep   × (1.0 - reputation/100) × rep_penalty
             + w_load  × load_pct × load_penalty
             + w_kv    × kv_alloc_delay      (if available_kv_slots <= 0)
             + w_s2s   × next_hop_rtt_ms     (if server-to-server RTT measured)

    where ``infer_time_ms = (span / estimated_tps) * 1000`` when TPS > 0,
    else a conservative fallback.

    Defaults are tuned for a WAN swarm of 10–50 peers where latency and
    inference time dominate; reputation, load, KV pressure, and S2S RTT
    act as tie-breakers and pipeline-topology optimisers.
    """

    w_rtt: float = 1.0
    w_infer: float = 1.5
    w_rep: float = 0.5
    w_load: float = 0.3
    rep_penalty: float = 500.0    # ms-equivalent penalty for a 0-reputation peer
    load_penalty: float = 300.0   # ms-equivalent penalty for a 100%-loaded peer
    unknown_tps_fallback: float = 1.0   # tokens/sec assumed when TPS is unknown
    # Phase 2A: KV cache pressure — penalise peers with no free KV slots.
    w_kv: float = 2.0
    kv_alloc_delay: float = 10_000.0  # ms-equivalent penalty when cache is full
    # Phase 2A: Server-to-server RTT — prefer pipelines where consecutive
    # peers have low measured latency between each other.
    w_s2s: float = 0.8


def _dijkstra_edge_cost(
    lr: LayerRange,
    metrics: PeerMetrics,
    weights: PipelineWeights,
) -> float:
    """Compute the per-peer cost of routing through *lr* given its *metrics*.

    Five cost factors (factors 1–5) in millisecond-equivalent units:

    1. **RTT** — client-to-peer round-trip latency.
    2. **Inference time** — estimated time to process this peer's layer span.
    3. **Reputation** — penalty for low-trust peers.
    4. **Load** — penalty for heavily-loaded peers.
    5. **KV cache pressure** — large penalty when peer has no free KV slots,
       forcing the coordinator to consider alternatives with warm caches.

    Factor 6 (server-to-server RTT) is applied at the graph level in
    ``find_optimal_pipeline`` using the upstream peer's ``next_hop_rtts``.

    Args:
        lr: The layer range this peer covers.
        metrics: Runtime metrics for the peer (latency, TPS, reputation, etc.).
        weights: Tunable weights for each cost factor.

    Returns:
        Positive float: total weighted cost in ms-equivalent units.
    """
    # Factor 1: RTT component.
    rtt_cost = weights.w_rtt * max(0.0, metrics.latency_ms)

    # Factor 2: Inference-time component.
    tps = metrics.estimated_tps if metrics.estimated_tps > 0 else weights.unknown_tps_fallback
    span = max(1, lr.span)
    infer_ms = (span / tps) * 1000.0
    infer_cost = weights.w_infer * infer_ms

    # Factor 3: Reputation penalty.
    rep = max(0.0, min(100.0, metrics.reputation_score))
    rep_cost = weights.w_rep * (1.0 - rep / 100.0) * weights.rep_penalty

    # Factor 4: Load penalty.
    load = max(0.0, min(1.0, metrics.load_pct))
    load_cost = weights.w_load * load * weights.load_penalty

    # Factor 5: KV cache pressure — penalise peers with no free slots.
    kv_cost = 0.0
    if metrics.available_kv_slots <= 0:
        kv_cost = weights.w_kv * weights.kv_alloc_delay

    return rtt_cost + infer_cost + rep_cost + load_cost + kv_cost


def find_optimal_pipeline(
    ranges: list[LayerRange],
    total_layers: int,
    peer_metrics: dict[str, PeerMetrics] | None = None,
    weights: PipelineWeights | None = None,
) -> list[LayerRange] | None:
    """Dijkstra's algorithm for the lowest-cost pipeline covering ``[0, total_layers)``.

    Constructs a layered directed graph:

    * **Source** node at ``layer=0``.
    * One node per peer (identified by ``peer_id``).
    * **Sink** node at ``layer=total_layers``.
    * Edge Source→P exists when ``P.layer_start == 0``.
    * Edge A→B exists when ``A.layer_end >= B.layer_start`` (B can continue
      from where A left off).
    * Edge P→Sink exists when ``P.layer_end >= total_layers``.

    Edge costs are computed by :func:`_dijkstra_edge_cost`.

    Parameters
    ----------
    ranges:       Candidate layer ranges (any order; may overlap).
    total_layers: Target coverage depth.
    peer_metrics: Optional mapping ``peer_id → PeerMetrics``.  Peers not in
                  this dict use default metrics.
    weights:      Optional cost-function weights.

    Returns
    -------
    Ordered ``list[LayerRange]`` forming the cheapest complete pipeline, or
    ``None`` if no path exists.
    """
    if total_layers <= 0 or not ranges:
        return None

    if weights is None:
        weights = PipelineWeights()
    if peer_metrics is None:
        peer_metrics = {}

    _DEFAULT_METRICS = PeerMetrics()

    # Filter to valid, non-duplicate ranges.
    valid: dict[str, LayerRange] = {}
    for r in ranges:
        if r.layer_end > r.layer_start:
            # If same peer_id appears twice, keep the one with wider span.
            existing = valid.get(r.peer_id)
            if existing is None or r.span > existing.span:
                valid[r.peer_id] = r

    if not valid:
        return None

    # Virtual nodes: "__source__" and "__sink__".
    # dist[node] = best cost to reach node; prev[node] = predecessor peer_id.
    INF = float("inf")
    dist: dict[str, float] = {"__source__": 0.0}
    prev: dict[str, str | None] = {"__source__": None}
    for pid in valid:
        dist[pid] = INF
        prev[pid] = None
    dist["__sink__"] = INF
    prev["__sink__"] = None

    # Priority queue: (cost, node_id).
    pq: list[tuple[float, str]] = [(0.0, "__source__")]

    while pq:
        cost, node = heapq.heappop(pq)

        if node == "__sink__":
            break  # Found the cheapest path to the sink.

        if cost > dist[node]:
            continue  # Stale entry.

        if node == "__source__":
            # Edges from source to peers that start at layer 0.
            for pid, r in valid.items():
                if r.layer_start <= 0:
                    m = peer_metrics.get(pid, _DEFAULT_METRICS)
                    edge = _dijkstra_edge_cost(r, m, weights)  # no S2S from source
                    new_cost = cost + edge
                    if new_cost < dist[pid]:
                        dist[pid] = new_cost
                        prev[pid] = "__source__"
                        heapq.heappush(pq, (new_cost, pid))
        else:
            # Edges from peer `node` to successors and possibly the sink.
            r_node = valid[node]

            # Can we reach the sink?
            if r_node.layer_end >= total_layers:
                if cost < dist["__sink__"]:
                    dist["__sink__"] = cost
                    prev["__sink__"] = node
                    heapq.heappush(pq, (cost, "__sink__"))

            # Edges to other peers whose layer_start is reachable.
            # Factor 6 (S2S RTT): the upstream peer (node) measures RTT to
            # each downstream peer and stores it in its next_hop_rtts dict.
            # We pass node's metrics so the cost function can look up the
            # S2S RTT from node → pid.
            node_metrics = peer_metrics.get(node, _DEFAULT_METRICS)
            for pid, r_next in valid.items():
                if pid == node:
                    continue
                if r_next.layer_start <= r_node.layer_end:
                    m = peer_metrics.get(pid, _DEFAULT_METRICS)
                    edge = _dijkstra_edge_cost(r_next, m, weights)
                    # Add S2S RTT from the UPSTREAM node → downstream pid.
                    s2s_rtt = node_metrics.next_hop_rtts.get(pid, 0.0)
                    if s2s_rtt > 0:
                        edge += weights.w_s2s * s2s_rtt
                    new_cost = cost + edge
                    if new_cost < dist[pid]:
                        dist[pid] = new_cost
                        prev[pid] = node
                        heapq.heappush(pq, (new_cost, pid))

    # Reconstruct the path.
    if dist["__sink__"] == INF:
        return None

    path_ids: list[str] = []
    cur: str | None = prev["__sink__"]
    while cur is not None and cur != "__source__":
        path_ids.append(cur)
        cur = prev.get(cur)

    if not path_ids:
        return None

    path_ids.reverse()
    return [valid[pid] for pid in path_ids]


# ── LayerCoverageMap ─────────────────────────────────────────────────────────


class LayerCoverageMap:
    """Organises a fleet of peers by their layer ranges for sharding decisions.

    Usage
    -----
    ::

        cmap = LayerCoverageMap.from_endpoints(healthy_peers)
        if cmap.is_complete():
            pipeline = cmap.best_pipeline()
            # pipeline is an ordered list[LayerRange]: stage0 → stage1 → …

    The map is **read-only** after construction.  Build a new instance whenever
    the peer list changes.

    Parameters
    ----------
    ranges:       Layer ranges from sharded peers only (full-model replicas are
                  excluded; include them separately if desired).
    total_layers: The model depth that defines the coverage target
                  ``[0, total_layers)``.
    """

    def __init__(self, ranges: list[LayerRange], total_layers: int) -> None:
        self.ranges: list[LayerRange] = list(ranges)
        self.total_layers: int = max(0, int(total_layers))

    # ── Constructors ─────────────────────────────────────────────────────────

    @classmethod
    def from_endpoints(
        cls,
        peers: list,          # list[PeerEndpoint] — typed as list to avoid circular import
        total_layers: int | None = None,
    ) -> LayerCoverageMap:
        """Build a :class:`LayerCoverageMap` from ``PeerEndpoint`` objects.

        Only peers with ``total_layers > 0`` and ``layer_end > 0`` are
        included.  If *total_layers* is ``None`` the most commonly reported
        value across the peer set is used (consensus).

        Parameters
        ----------
        peers:        Sequence of ``PeerEndpoint`` (or any object with
                      ``peer_id``, ``layer_start``, ``layer_end``,
                      ``total_layers``, ``host``, ``port`` attributes).
        total_layers: Override for the model depth.  Pass ``None`` to
                      auto-detect from the peer announcements.
        """
        ranges: list[LayerRange] = []
        for p in peers:
            p_total = int(getattr(p, "total_layers", 0) or 0)
            p_end = int(getattr(p, "layer_end", 0) or 0)
            if p_total > 0 and p_end > 0:
                ranges.append(
                    LayerRange(
                        peer_id=str(p.peer_id),
                        layer_start=int(getattr(p, "layer_start", 0) or 0),
                        layer_end=p_end,
                        total_layers=p_total,
                        host=str(getattr(p, "host", "") or ""),
                        port=int(getattr(p, "port", 0) or 0),
                    )
                )

        if total_layers is None:
            # Consensus: pick the most frequently reported total_layers value.
            counts: dict[int, int] = {}
            for r in ranges:
                counts[r.total_layers] = counts.get(r.total_layers, 0) + 1
            total_layers = max(counts, key=lambda k: counts[k]) if counts else 0

        return cls(ranges, int(total_layers))

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def has_sharded_peers(self) -> bool:
        """``True`` when at least one peer reports a proper sub-range."""
        return any(r.is_sharded for r in self.ranges)

    # ── Query API ────────────────────────────────────────────────────────────

    def gaps(self) -> list[tuple[int, int]]:
        """Uncovered layer intervals in ``[0, total_layers)``."""
        return coverage_gaps(self.ranges, self.total_layers)

    def is_complete(self) -> bool:
        """``True`` when the current peer set provides full model coverage."""
        return is_complete_coverage(self.ranges, self.total_layers)

    def best_pipeline(
        self,
        peer_metrics: dict[str, PeerMetrics] | None = None,
        weights: PipelineWeights | None = None,
    ) -> list[LayerRange] | None:
        """Return the best pipeline covering all layers, or ``None``.

        When *peer_metrics* is provided, uses :func:`find_optimal_pipeline`
        (Dijkstra cost-optimal routing).  Falls back to the greedy
        :func:`find_complete_pipeline` if Dijkstra fails or metrics are
        unavailable.

        Parameters
        ----------
        peer_metrics: Optional mapping ``peer_id → PeerMetrics``.
        weights:      Optional cost-function weights.

        Returns
        -------
        Ordered ``list[LayerRange]`` or ``None``.
        """
        if peer_metrics is not None:
            optimal = find_optimal_pipeline(
                self.ranges, self.total_layers, peer_metrics, weights,
            )
            if optimal is not None:
                return optimal
            logger.debug(
                "dijkstra_fallback: optimal pipeline not found, trying greedy"
            )
        return find_complete_pipeline(self.ranges, self.total_layers)

    def coverage_fraction(self) -> float:
        """Fraction of layers covered by at least one peer (0.0 – 1.0)."""
        if self.total_layers <= 0:
            return 0.0
        uncovered = sum(end - start for start, end in self.gaps())
        return max(0.0, 1.0 - uncovered / self.total_layers)

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Diagnostic snapshot suitable for logging and API responses.

        Keys
        ----
        total_layers, sharded_peers, full_model_peers, coverage_complete,
        coverage_fraction, gaps, best_pipeline_stages, best_pipeline.
        """
        pipeline = self.best_pipeline()
        sharded_count = sum(1 for r in self.ranges if r.is_sharded)
        return {
            "total_layers": self.total_layers,
            "sharded_peers": sharded_count,
            "full_model_peers": len(self.ranges) - sharded_count,
            "coverage_complete": self.is_complete(),
            "coverage_fraction": round(self.coverage_fraction(), 4),
            "gaps": self.gaps(),
            "best_pipeline_stages": len(pipeline) if pipeline else 0,
            "best_pipeline": [
                {
                    "peer_id": r.peer_id,
                    "layer_start": r.layer_start,
                    "layer_end": r.layer_end,
                    "host": r.host,
                    "port": r.port,
                }
                for r in (pipeline or [])
            ],
        }
