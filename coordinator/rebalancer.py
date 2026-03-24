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

"""coordinator.rebalancer — dynamic layer rebalancing for sharded inference.

When peers join or leave, the :class:`LayerCoverageMap` may develop gaps —
uncovered layer intervals that prevent sharded inference.  The
:class:`LayerRebalancer` detects these gaps and generates
:class:`RebalanceDirective` objects that instruct adjacent peers to expand
their shard ranges to fill the holes.

Directives are published to the DHT via ``POST /rebalance`` and polled by
peers via ``GET /rebalance?peer_id=X`` during their announce cycle.

Safety guards
-------------
* Peers only apply a directive when ``inflight_count == 0`` (no active requests).
* Load is set to 100% during reshard to prevent routing.
* A 5-second drain period precedes the reshard.
* Directives expire after ``directive_ttl_s`` (default 120s).
* MLX peers silently ignore reshard directives (full-model only).

Typical usage
-------------
::

    from coordinator.rebalancer import LayerRebalancer, RebalanceDirective

    rebalancer = LayerRebalancer()
    directives = rebalancer.compute_directives(cmap, peer_health_list)
    for d in directives:
        publish_to_dht(d)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from coordinator.layer_coverage import LayerCoverageMap, LayerRange, coverage_gaps

logger = logging.getLogger(__name__)


# ── Data types ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RebalanceDirective:
    """Instruction for a peer to expand its shard range.

    Attributes
    ----------
    target_peer_id:    Peer that should reshard.
    new_layer_start:   New inclusive start layer.
    new_layer_end:     New exclusive end layer.
    total_layers:      Full model depth.
    reason:            Human-readable reason for the directive.
    issued_unix_ms:    Timestamp when the directive was created.
    expires_unix_ms:   Timestamp when the directive expires.
    """

    target_peer_id: str
    new_layer_start: int
    new_layer_end: int
    total_layers: int
    reason: str = "gap_fill"
    issued_unix_ms: int = 0
    expires_unix_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_peer_id": self.target_peer_id,
            "new_layer_start": self.new_layer_start,
            "new_layer_end": self.new_layer_end,
            "total_layers": self.total_layers,
            "reason": self.reason,
            "issued_unix_ms": self.issued_unix_ms,
            "expires_unix_ms": self.expires_unix_ms,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RebalanceDirective:
        return cls(
            target_peer_id=str(d.get("target_peer_id", "")),
            new_layer_start=int(d.get("new_layer_start", 0)),
            new_layer_end=int(d.get("new_layer_end", 0)),
            total_layers=int(d.get("total_layers", 0)),
            reason=str(d.get("reason", "gap_fill")),
            issued_unix_ms=int(d.get("issued_unix_ms", 0)),
            expires_unix_ms=int(d.get("expires_unix_ms", 0)),
        )

    @property
    def is_expired(self) -> bool:
        return time.time() * 1000.0 > self.expires_unix_ms


# ── Rebalancer ──────────────────────────────────────────────────────────────


@dataclass
class _PeerCandidate:
    """Internal: a peer that could expand to cover a gap."""

    peer_id: str
    layer_start: int
    layer_end: int
    total_layers: int
    load_pct: float
    available_vram_mb: int
    runtime_backend: str


class LayerRebalancer:
    """Computes rebalance directives to fill gaps in the layer coverage map.

    Parameters
    ----------
    max_load_pct:
        Peers with ``load_pct`` above this threshold are not considered as
        candidates for expansion (default 50.0).
    min_vram_mb:
        Peers with ``available_vram_mb`` below this threshold are not
        considered as candidates for expansion (default 256 MB).
    directive_ttl_s:
        TTL for each directive in seconds (default 120s).
    """

    def __init__(
        self,
        max_load_pct: float = 50.0,
        min_vram_mb: int = 256,
        directive_ttl_s: float = 120.0,
    ) -> None:
        self.max_load_pct = max(0.0, float(max_load_pct))
        self.min_vram_mb = max(0, int(min_vram_mb))
        self.directive_ttl_s = max(1.0, float(directive_ttl_s))

    def compute_directives(
        self,
        cmap: LayerCoverageMap,
        peer_health: list[Any] | None = None,
    ) -> list[RebalanceDirective]:
        """Find gaps in *cmap* and generate directives for adjacent peers.

        Parameters
        ----------
        cmap:
            The current :class:`LayerCoverageMap` built from sharded peers.
        peer_health:
            Optional list of ``PeerHealth`` objects (from
            :meth:`coordinator.engine._discover_for_model`).  Each object
            should have ``.peer`` (with ``peer_id``, ``layer_start``,
            ``layer_end``, ``total_layers``, ``available_vram_mb``,
            ``runtime_backend`` attributes) and ``.load_pct`` attribute.
            When ``None``, candidate info is extracted from ``cmap.ranges``.

        Returns
        -------
        list[RebalanceDirective]
            Ordered list of directives.  Empty if there are no gaps or no
            candidates can fill them.
        """
        gaps = cmap.gaps()
        if not gaps:
            return []

        total_layers = cmap.total_layers
        if total_layers <= 0:
            return []

        # Build candidate pool.
        candidates = self._build_candidates(cmap, peer_health)
        if not candidates:
            logger.debug("rebalancer: no eligible candidates for gap filling")
            return []

        now_ms = int(time.time() * 1000)
        expires_ms = now_ms + int(self.directive_ttl_s * 1000)

        directives: list[RebalanceDirective] = []
        used_peers: set[str] = set()

        for gap_start, gap_end in gaps:
            # Find the best adjacent peer: one whose current range borders
            # this gap (layer_end == gap_start or layer_start == gap_end),
            # has low load, and sufficient VRAM.
            best = self._find_best_candidate(
                gap_start, gap_end, total_layers, candidates, used_peers,
            )
            if best is None:
                logger.debug(
                    "rebalancer: no candidate for gap [%d, %d)", gap_start, gap_end,
                )
                continue

            # Compute the new shard range: expand the peer to cover the gap.
            new_start, new_end = self._compute_expanded_range(
                best, gap_start, gap_end,
            )

            directive = RebalanceDirective(
                target_peer_id=best.peer_id,
                new_layer_start=new_start,
                new_layer_end=new_end,
                total_layers=total_layers,
                reason=f"gap_fill:[{gap_start},{gap_end})",
                issued_unix_ms=now_ms,
                expires_unix_ms=expires_ms,
            )
            directives.append(directive)
            used_peers.add(best.peer_id)
            logger.info(
                "rebalancer: directive peer=%s expand [%d,%d) -> [%d,%d) reason=%s",
                best.peer_id,
                best.layer_start,
                best.layer_end,
                new_start,
                new_end,
                directive.reason,
            )

        return directives

    # ── Internals ───────────────────────────────────────────────────────────

    def _build_candidates(
        self,
        cmap: LayerCoverageMap,
        peer_health: list[Any] | None,
    ) -> list[_PeerCandidate]:
        """Build the candidate pool from peer health or cmap ranges."""
        candidates: list[_PeerCandidate] = []

        if peer_health is not None:
            for h in peer_health:
                p = getattr(h, "peer", h)
                load_pct = float(getattr(h, "load_pct", 0.0) or 0.0)
                if load_pct > self.max_load_pct:
                    continue
                vram = int(getattr(p, "available_vram_mb", 0) or 0)
                if vram < self.min_vram_mb and self.min_vram_mb > 0 and vram > 0:
                    continue
                candidates.append(
                    _PeerCandidate(
                        peer_id=str(p.peer_id),
                        layer_start=int(getattr(p, "layer_start", 0) or 0),
                        layer_end=int(getattr(p, "layer_end", 0) or 0),
                        total_layers=int(getattr(p, "total_layers", 0) or 0),
                        load_pct=load_pct,
                        available_vram_mb=vram,
                        runtime_backend=str(getattr(p, "runtime_backend", "") or ""),
                    )
                )
        else:
            # Fallback: extract from cmap.ranges (no load/vram filtering).
            for r in cmap.ranges:
                candidates.append(
                    _PeerCandidate(
                        peer_id=r.peer_id,
                        layer_start=r.layer_start,
                        layer_end=r.layer_end,
                        total_layers=r.total_layers,
                        load_pct=0.0,
                        available_vram_mb=0,
                        runtime_backend="",
                    )
                )

        return candidates

    def _find_best_candidate(
        self,
        gap_start: int,
        gap_end: int,
        total_layers: int,
        candidates: list[_PeerCandidate],
        used_peers: set[str],
    ) -> _PeerCandidate | None:
        """Find the best peer to expand into the gap."""
        # Score: prefer peers adjacent to the gap, then lower load.
        scored: list[tuple[float, _PeerCandidate]] = []

        for c in candidates:
            if c.peer_id in used_peers:
                continue
            # Skip MLX peers — they run full models, cannot reshard.
            if c.runtime_backend == "mlx":
                continue

            # Adjacency score: how close is this peer to the gap?
            # Direct adjacency (layer_end == gap_start or layer_start == gap_end)
            # gets the best score.
            if c.layer_end == gap_start:
                # Peer ends right where the gap starts — extend rightward.
                adjacency = 0.0
            elif c.layer_start == gap_end:
                # Peer starts right where the gap ends — extend leftward.
                adjacency = 0.0
            elif c.layer_end <= gap_start:
                adjacency = float(gap_start - c.layer_end)
            elif c.layer_start >= gap_end:
                adjacency = float(c.layer_start - gap_end)
            else:
                # Peer overlaps the gap partially — still a candidate.
                adjacency = 0.0

            # Penalize non-adjacent candidates heavily.
            score = adjacency * 100.0 + c.load_pct
            scored.append((score, c))

        if not scored:
            return None

        scored.sort(key=lambda x: x[0])
        return scored[0][1]

    @staticmethod
    def _compute_expanded_range(
        peer: _PeerCandidate,
        gap_start: int,
        gap_end: int,
    ) -> tuple[int, int]:
        """Compute the new [start, end) range after expanding into the gap."""
        new_start = min(peer.layer_start, gap_start)
        new_end = max(peer.layer_end, gap_end)
        return new_start, new_end


# ── DHT publish helper ──────────────────────────────────────────────────────


def publish_directives_to_dht(
    directives: list[RebalanceDirective],
    dht_urls: list[str] | tuple[str, ...],
    timeout_s: float = 3.0,
) -> tuple[int, int]:
    """Publish directives to DHT bootstrap nodes via POST /rebalance.

    Returns
    -------
    (successes, failures)
        Count of successfully published and failed directive×url pairs.
    """
    import json
    import urllib.request

    successes = 0
    failures = 0

    for directive in directives:
        payload = json.dumps(directive.to_dict()).encode("utf-8")
        for url in dht_urls:
            base = url.rstrip("/")
            endpoint = f"{base}/rebalance"
            try:
                req = urllib.request.Request(
                    endpoint,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    if resp.status < 300:
                        successes += 1
                    else:
                        failures += 1
            except Exception as exc:
                logger.debug("rebalancer: publish failed url=%s err=%s", endpoint, exc)
                failures += 1

    return successes, failures


def poll_directives_from_dht(
    peer_id: str,
    dht_urls: list[str] | tuple[str, ...],
    timeout_s: float = 3.0,
) -> list[RebalanceDirective]:
    """Poll DHT bootstrap nodes for directives targeting *peer_id*.

    Returns
    -------
    list[RebalanceDirective]
        Non-expired directives for this peer.
    """
    import json
    import urllib.request

    directives: list[RebalanceDirective] = []
    seen_keys: set[str] = set()

    for url in dht_urls:
        base = url.rstrip("/")
        endpoint = f"{base}/rebalance?peer_id={peer_id}"
        try:
            req = urllib.request.Request(endpoint, method="GET")
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                if resp.status >= 300:
                    continue
                data = json.loads(resp.read().decode("utf-8"))
                for d_dict in data.get("directives", []):
                    d = RebalanceDirective.from_dict(d_dict)
                    key = f"{d.target_peer_id}:{d.new_layer_start}:{d.new_layer_end}"
                    if key not in seen_keys and not d.is_expired:
                        directives.append(d)
                        seen_keys.add(key)
        except Exception as exc:
            logger.debug("rebalancer: poll failed url=%s err=%s", endpoint, exc)

    return directives
