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

"""Capability-aware auto-scaling policy for OpenHydra.

Design spec : plans/auto-scaling-policy.md
Parent plan : docs/beta-launch-strategy.md §4

The AutoScaler evaluates the network on a 5-minute cadence and decides:

  1. Promotions — add a larger model to the active roster when the fleet
     has sufficient capable peers AND users are actually requesting that tier.
  2. Demotions  — remove a model whose effective peer coverage has fallen
     below the safety threshold.
  3. Role assignments — give every peer a role (inference or support) so
     weak nodes earn credits and remain in the network.

Key terminology
---------------
effective_redundancy
    ``capable_peers / shards_needed``, NOT ``total_peers / shards_needed``.
    A peer is *capable* only if its ``available_vram_mb >= model.shard_vram_mb``.
    Peers with ``available_vram_mb == 0`` (unknown) are treated as capable.

hysteresis band
    Promote at 3x, demote at 1.5x, floor at 2x.  The gap prevents oscillation.

cooldown
    15 minutes after any model change.  Stops cascading churn when many peers
    join or leave simultaneously.

demand weighting
    Only promote to a tier that accounts for ≥30% of recent requests.
    Prevents over-allocating capacity to models nobody wants.

MIN_FLEET_SIZE
    Below 5 total peers the scaler stays silent — avoid acting on noise.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from coordinator.request_log import RequestLog, quality_tier_for_model_id
from coordinator.role_assigner import RoleAssignment, assign_role

logger = logging.getLogger(__name__)

# ── Policy constants ───────────────────────────────────────────────────────────
PROMOTE_THRESHOLD: float = 3.0    # effective redundancy required to promote
DEMOTE_THRESHOLD:  float = 1.5    # effective redundancy at which we demote
FLOOR_RATIO:       float = 2.0    # minimum safe redundancy for any active model
MIN_DEMAND_WEIGHT: float = 0.3    # minimum demand fraction to allow promotion

RE_EVALUATE_S:  int = 300   # evaluate every 5 minutes
COOLDOWN_S:     int = 900   # 15-minute cooldown after any change
MIN_FLEET_SIZE: int = 5     # don't act on fewer peers than this


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    """Lightweight view of a catalog model used inside the auto-scaler."""

    model_id:      str
    shard_vram_mb: int    # VRAM required per shard, in MB
    shards_needed: int    # peers needed for a complete pipeline (≥1)
    quality_tier:  str    # "basic" | "standard" | "advanced" | "frontier"
    required_peers: int = 1

    def __post_init__(self) -> None:
        if self.shards_needed < 1:
            raise ValueError(f"shards_needed must be ≥1, got {self.shards_needed}")


@dataclass
class PeerView:
    """Resource snapshot of one peer as visible to the auto-scaler."""

    peer_id:           str
    available_vram_mb: int    # free GPU VRAM in MB; 0 = CPU-only or unknown
    assigned_model_id: str | None = None
    cpu_score:         float = 0.0   # normalised CPU benchmark (0 = unknown)
    disk_free_gb:      float = 0.0   # free disk space in GB (0 = unknown)
    tps:               float = 0.0   # tokens/s reported by the peer


@dataclass
class ScalerResult:
    """Output of one :meth:`AutoScaler.evaluate` run."""

    promoted:         list[str] = field(default_factory=list)  # model_ids added
    demoted:          list[str] = field(default_factory=list)  # model_ids removed
    active_roster:    list[str] = field(default_factory=list)  # full current roster
    role_assignments: list[RoleAssignment] = field(default_factory=list)
    skipped_reason:   str = ""   # non-empty if the run was a no-op


# ── Policy helper functions ────────────────────────────────────────────────────

def effective_redundancy(model: ModelSpec, peers: list[PeerView]) -> float:
    """Compute the effective redundancy for *model* given the current *peers*.

    Only peers with ``available_vram_mb >= model.shard_vram_mb`` (or whose
    VRAM is unknown, i.e. 0) count as capable.

    Returns ``inf`` if ``shards_needed == 0`` (degenerate case).
    """
    if model.shards_needed == 0:
        return float("inf")
    capable = sum(
        1 for p in peers
        # 0 = unknown VRAM → optimistic: assume capable
        if p.available_vram_mb <= 0 or p.available_vram_mb >= model.shard_vram_mb
    )
    return capable / model.shards_needed


def effective_redundancy_after_reassignment(
    existing: ModelSpec,
    new_model: ModelSpec,
    peers: list[PeerView],
) -> float:
    """Conservative post-promotion redundancy for *existing*.

    Two paths depending on whether peers carry assignment state:

    **Assignment-based** (peers have ``assigned_model_id == existing.model_id``):
    Pessimistic worst case — every peer currently serving *existing* that is
    *also* capable of running *new_model* is assumed to be pulled away.
    Peers not serving *existing* are ignored (they are not at risk).

    **Capability-based** (no peers are assigned to *existing* yet):
    Used during initial scaling when no assignment state exists.
    Pessimistically assumes *new_model* will consume exactly
    ``new_model.shards_needed`` peers from the dual-capable pool (peers that
    can run both models). The rest of the capable-for-existing pool remains
    available.

    Returns ``inf`` when ``existing.shards_needed == 0`` (degenerate).
    """
    if existing.shards_needed == 0:
        return float("inf")

    capable_for_new = {
        p.peer_id for p in peers
        if p.available_vram_mb <= 0 or p.available_vram_mb >= new_model.shard_vram_mb
    }
    serving_existing = {
        p.peer_id for p in peers
        if p.assigned_model_id == existing.model_id
    }

    if serving_existing:
        # ── Assignment-based path ─────────────────────────────────────────
        # Peers actively serving existing and capable of new_model are at risk.
        at_risk = serving_existing & capable_for_new
        remaining = len(serving_existing) - len(at_risk)
        return remaining / existing.shards_needed

    # ── Capability-based path (no assignment state) ───────────────────────
    # New_model takes at most shards_needed peers from the dual-capable pool;
    # peers capable only for existing (not new_model) are safe.
    capable_existing = [
        p for p in peers
        if p.available_vram_mb <= 0 or p.available_vram_mb >= existing.shard_vram_mb
    ]
    dual_capable_count = sum(
        1 for p in capable_existing
        if p.available_vram_mb <= 0 or p.available_vram_mb >= new_model.shard_vram_mb
    )
    exclusive_count = len(capable_existing) - dual_capable_count
    taken_by_new = min(dual_capable_count, new_model.shards_needed)
    remaining = exclusive_count + (dual_capable_count - taken_by_new)
    return remaining / existing.shards_needed


def promotion_score(model: ModelSpec, peers: list[PeerView], request_log: RequestLog) -> float:
    """Combined promotion score: effective_redundancy × demand_weight."""
    return effective_redundancy(model, peers) * request_log.demand_weight(model.quality_tier)


def should_promote(
    candidate: ModelSpec,
    active_models: list[ModelSpec],
    peers: list[PeerView],
    request_log: RequestLog,
    recently_changed: set[str],
) -> tuple[bool, str]:
    """Return ``(True, reason)`` if *candidate* can be safely promoted.

    All four gates must pass:

    1. Effective redundancy ≥ PROMOTE_THRESHOLD.
    2. No active model would drop below FLOOR_RATIO after the promotion.
    3. Demand weight for the candidate's tier ≥ MIN_DEMAND_WEIGHT.
    4. The candidate is not in the cooldown blacklist.
    """
    # Gate 1: sufficient capable peers
    ratio = effective_redundancy(candidate, peers)
    if ratio < PROMOTE_THRESHOLD:
        return False, f"ratio={ratio:.2f} < promote_threshold={PROMOTE_THRESHOLD}"

    # Gate 2: floor check — existing models stay above the safety floor
    for active in active_models:
        remaining = effective_redundancy_after_reassignment(active, candidate, peers)
        if remaining < FLOOR_RATIO:
            return False, (
                f"would drop {active.model_id} to {remaining:.2f}x "
                f"(floor={FLOOR_RATIO})"
            )

    # Gate 3: actual demand for this tier
    demand = request_log.demand_weight(candidate.quality_tier)
    if demand < MIN_DEMAND_WEIGHT:
        return False, f"demand={demand:.2f} < min_demand={MIN_DEMAND_WEIGHT}"

    # Gate 4: cooldown
    if candidate.model_id in recently_changed:
        return False, "cooldown_active"

    return True, f"ratio={ratio:.2f} demand={demand:.2f}"


# ── AutoScaler class ───────────────────────────────────────────────────────────

class AutoScaler:
    """Capability-aware auto-scaler for the OpenHydra coordinator.

    Integration with :class:`~coordinator.engine.CoordinatorEngine`::

        # In __init__:
        from coordinator.auto_scaler import AutoScaler, ModelSpec
        from coordinator.request_log import RequestLog
        self._request_log = RequestLog()
        self._auto_scaler = AutoScaler(
            [ModelSpec(...) for item in self.model_catalog if item.shard_vram_mb > 0]
        )

        # On each inference request (record demand):
        self._request_log.record(model_id)

        # In network_status() after _scan_network():
        peer_views = [PeerView(...) for peer in healthy]
        result = self._auto_scaler.maybe_evaluate(peer_views, self._request_log)
        if result:
            self._active_model_roster = result.active_roster

    The scaler is completely passive — it never pushes changes to peers or DHT.
    The coordinator reads ``active_roster`` and passes it to
    :class:`~coordinator.degradation.DegradationPolicy` when selecting models.
    """

    def __init__(self, model_specs: list[ModelSpec]):
        if not model_specs:
            raise ValueError("AutoScaler requires at least one ModelSpec")

        self._specs_by_id: dict[str, ModelSpec] = {m.model_id: m for m in model_specs}

        # Sort from smallest → largest by (tier order, shard_vram_mb).
        _tier_order = {"basic": 0, "standard": 1, "advanced": 2, "frontier": 3}
        self._sorted_specs: list[ModelSpec] = sorted(
            model_specs,
            key=lambda m: (
                _tier_order.get(m.quality_tier, 1),
                m.shard_vram_mb,
            ),
        )

        # Start with the smallest available model.
        self._active_roster: list[str] = [self._sorted_specs[0].model_id]

        # Cooldown tracking: model_id → monotonic timestamp of last change.
        self._recently_changed: dict[str, float] = {}

        self._last_evaluated: float = 0.0
        self._lock = threading.RLock()

        logger.info(
            "auto_scaler_init: %d models  initial_roster=%s",
            len(model_specs), self._active_roster,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def active_roster(self) -> list[str]:
        """Current list of model IDs on the active roster (copy)."""
        with self._lock:
            return list(self._active_roster)

    def maybe_evaluate(
        self,
        peers: list[PeerView],
        request_log: RequestLog,
        force: bool = False,
    ) -> ScalerResult | None:
        """Evaluate the fleet if ``RE_EVALUATE_S`` seconds have elapsed.

        Returns a :class:`ScalerResult` when an evaluation ran, or ``None``
        when the interval has not yet elapsed (and *force* is ``False``).
        """
        now = time.monotonic()
        with self._lock:
            if not force and (now - self._last_evaluated) < RE_EVALUATE_S:
                return None
            result = self._evaluate(peers, request_log, now)
            self._last_evaluated = now
            return result

    def evaluate(
        self,
        peers: list[PeerView],
        request_log: RequestLog,
    ) -> ScalerResult:
        """Force-run an evaluation regardless of the interval timer.

        Always returns a :class:`ScalerResult`.  Useful for tests and for
        on-demand re-evaluation after significant fleet changes.
        """
        now = time.monotonic()
        with self._lock:
            result = self._evaluate(peers, request_log, now)
            self._last_evaluated = now
            return result

    # ── Private implementation ─────────────────────────────────────────────────

    def _evaluate(
        self,
        peers: list[PeerView],
        request_log: RequestLog,
        now: float,
    ) -> ScalerResult:
        result = ScalerResult()

        # Safety guard: ignore tiny fleets.
        if len(peers) < MIN_FLEET_SIZE:
            result.skipped_reason = (
                f"fleet_too_small: {len(peers)} peers < MIN_FLEET_SIZE={MIN_FLEET_SIZE}"
            )
            result.active_roster = list(self._active_roster)
            logger.debug("auto_scaler_skip: %s", result.skipped_reason)
            return result

        # Expire cooldown entries.
        self._recently_changed = {
            mid: ts for mid, ts in self._recently_changed.items()
            if now - ts < COOLDOWN_S
        }
        recently_changed_set: set[str] = set(self._recently_changed)

        active_specs = self._resolve_active_specs()

        # ── Step 1: Demotions ──────────────────────────────────────────────────
        still_active: list[str] = []
        for spec in active_specs:
            ratio = effective_redundancy(spec, peers)
            if ratio < DEMOTE_THRESHOLD:
                result.demoted.append(spec.model_id)
                self._recently_changed[spec.model_id] = now
                logger.info(
                    "auto_scaler_demote: model=%s ratio=%.2f threshold=%.2f",
                    spec.model_id, ratio, DEMOTE_THRESHOLD,
                )
            else:
                still_active.append(spec.model_id)

        self._active_roster = still_active
        active_specs = self._resolve_active_specs()

        # ── Step 2: Best single promotion candidate ───────────────────────────
        best_candidate: ModelSpec | None = None
        best_score:     float = -1.0
        best_reason:    str   = ""

        for spec in self._sorted_specs:
            if spec.model_id in self._active_roster:
                continue
            ok, reason = should_promote(
                spec, active_specs, peers, request_log, recently_changed_set
            )
            if ok:
                score = promotion_score(spec, peers, request_log)
                if score > best_score:
                    best_score = score
                    best_candidate = spec
                    best_reason = reason

        if best_candidate is not None:
            result.promoted.append(best_candidate.model_id)
            self._active_roster.append(best_candidate.model_id)
            self._recently_changed[best_candidate.model_id] = now
            logger.info(
                "auto_scaler_promote: model=%s score=%.2f %s",
                best_candidate.model_id, best_score, best_reason,
            )

        result.active_roster = list(self._active_roster)

        # ── Step 3: Role assignments for all peers ────────────────────────────
        roster_for_roles: list[tuple[str, int]] = [
            (mid, self._specs_by_id[mid].shard_vram_mb)
            for mid in self._active_roster
            if mid in self._specs_by_id
        ]
        for peer in peers:
            assignment = assign_role(
                peer_id=peer.peer_id,
                available_vram_mb=peer.available_vram_mb,
                cpu_score=peer.cpu_score,
                disk_free_gb=peer.disk_free_gb,
                model_roster=roster_for_roles,
            )
            result.role_assignments.append(assignment)

        if result.promoted or result.demoted:
            logger.info(
                "auto_scaler_result: promoted=%s demoted=%s roster=%s",
                result.promoted, result.demoted, result.active_roster,
            )

        return result

    def _resolve_active_specs(self) -> list[ModelSpec]:
        """Return ModelSpec objects for the current active roster."""
        return [
            self._specs_by_id[mid]
            for mid in self._active_roster
            if mid in self._specs_by_id
        ]
