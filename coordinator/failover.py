# Copyright 2026 OpenHydra contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Phase 2b — DFlash failover: draft model registry + drafter promotion.

Two pieces, both transport-agnostic (they consume / emit
``SwarmEvent`` objects via the ``SwarmEventBus`` abstraction from
``coordinator/swarm_events.py``):

1. ``DraftModelRegistry`` — durable record of the swarm's active
   draft model spec. Subscribes to ``register_draft_model`` events,
   stores the latest spec keyed on ``(target_path, draft_path)``.
   Late-joining peers query ``get_active_spec()`` to learn what
   draft weights to load. The registry also acts as the source of
   truth for stage-0's preload check: stage-0 always preloads the
   spec returned by ``get_active_spec()`` regardless of whether it
   is the active drafter (Topology B) or a standby (Topology A
   failover candidate).

2. ``FailoverManager`` — watches coord heartbeat, emits
   ``PromoteDrafter`` when the coord is judged absent, tracks
   ``PromoteDrafter`` events from peers and applies the
   later-timestamp-wins tiebreak. Surfaces an ``active_drafter_id``
   property the rest of the runtime queries when routing PushResult
   callbacks.

Mandate: stage-0 must be able to promote and resume generation
WITHOUT re-prefilling. The KV state lives on the peers (preserved
across the coord crash by virtue of the peers staying up); only the
orchestrator role transfers. The promoting stage-0:
  (a) reads the spec from DraftModelRegistry,
  (b) calls ``DFlashDrafter.reload(cfg)`` if it wasn't already
      hosting the drafter,
  (c) emits ``PromoteDrafter``,
  (d) takes over the verify_block role using the lm_head it owns
      (stage-0 in our default topology owns layer 0..K and has the
      lm_head weights via the existing HeadSampler.borrow_weights
      Phase-5 path when ``runtime_load_full_head=True``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import threading
import time
from typing import Any, Callable, Optional

from coordinator.swarm_events import (
    EVENT_TYPE_PROMOTE_DRAFTER,
    EVENT_TYPE_REGISTER_DRAFT_MODEL,
    PromoteDrafter,
    RegisterDraftModel,
    SwarmEvent,
    SwarmEventBus,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DraftModelRegistry",
    "FailoverManager",
    "FailoverError",
]


class FailoverError(RuntimeError):
    """Raised on failover invariant violations.

    Carries a structured ``code`` field:
      * ``"no_active_spec"`` — failover requested but no
        RegisterDraftModel event has ever been received.
      * ``"self_promote"``   — local peer trying to promote itself
        without first registering.
      * ``"stale_promote"``  — incoming PromoteDrafter with
        unix_ms older than current active_drafter's promote timestamp.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


# ── DraftModelRegistry ──────────────────────────────────────────────────


@dataclass
class _RegistrySnapshot:
    spec: RegisterDraftModel
    from_peer: str
    unix_ms: int


class DraftModelRegistry:
    """Process-local cache of the swarm's RegisterDraftModel events.

    Keeps the latest spec only — re-registrations (e.g. after a
    config change) overwrite. ``later-unix_ms wins`` resolves the
    rare case where two peers register at the same instant.

    Late-joiner replay: the bus delivers historical events when a
    peer first subscribes (transport-dependent — InMemorySwarmEventBus
    delivers nothing past, libp2p gossip replays via the swarm's
    event log). The registry is correct for both because it just
    accumulates whatever it's told.
    """

    def __init__(self, bus: SwarmEventBus) -> None:
        self._bus = bus
        self._lock = threading.Lock()
        self._latest: Optional[_RegistrySnapshot] = None
        self._unsubscribe = bus.subscribe(
            EVENT_TYPE_REGISTER_DRAFT_MODEL, self._on_event,
        )

    def _on_event(self, event: SwarmEvent) -> None:
        if not isinstance(event.payload, RegisterDraftModel):
            return  # defence in depth — bus shouldn't route here otherwise
        with self._lock:
            current = self._latest
            if current is not None and event.unix_ms < current.unix_ms:
                logger.debug(
                    "draft_registry_drop_stale: incoming unix_ms=%d "
                    "current unix_ms=%d",
                    event.unix_ms, current.unix_ms,
                )
                return
            self._latest = _RegistrySnapshot(
                spec=event.payload,
                from_peer=event.from_peer,
                unix_ms=event.unix_ms,
            )
        logger.info(
            "draft_registry_updated: target=%s draft=%s block=%d "
            "backend=%s from_peer=%s",
            event.payload.target_path, event.payload.draft_path,
            event.payload.block_size, event.payload.backend,
            event.from_peer,
        )

    def announce(
        self,
        spec: RegisterDraftModel,
        *,
        from_peer: str,
    ) -> None:
        """Publish ``spec`` so all subscribers (including this
        process's own registry) record it."""
        spec.validate()
        self._bus.publish(spec, from_peer=from_peer)

    def get_active_spec(self) -> Optional[RegisterDraftModel]:
        with self._lock:
            return self._latest.spec if self._latest is not None else None

    def get_active_announcement(self) -> Optional[_RegistrySnapshot]:
        """Returns the full announcement (spec + emitter + timestamp).
        Used by FailoverManager to decide tiebreaks."""
        with self._lock:
            return self._latest

    def close(self) -> None:
        """Drop the bus subscription. Idempotent."""
        unsub = getattr(self, "_unsubscribe", None)
        if unsub is not None:
            try:
                unsub()
            except Exception:  # pragma: no cover
                pass
            self._unsubscribe = None


# ── FailoverManager ─────────────────────────────────────────────────────


@dataclass
class _ActiveDrafter:
    peer_id: str
    unix_ms: int


class FailoverManager:
    """Tracks the swarm's active drafter peer and promotes on coord
    absence.

    Args:
        bus: Swarm event transport.
        local_peer_id: This peer's libp2p id. Used to (a) tag
            outgoing PromoteDrafter events, (b) decide whether
            an incoming PromoteDrafter is targeting us.
        registry: DraftModelRegistry — promotion requires the spec
            to be present so the promoting peer knows what draft
            weights are needed (or are already loaded).
        coord_alive: Callable returning True when the coordinator
            is reachable. Called by ``check_coord()``; the runtime
            schedules the calls (e.g. via the existing health-check
            loop). When this returns False ``absence_threshold_ms``
            consecutive times, the manager promotes.
        absence_threshold_ms: How long the coord must look absent
            before we promote. 5 s default — long enough that a
            transient libp2p reconnect doesn't trigger a thrash;
            short enough that user-visible generation pause is
            tolerable.

    Promotion semantics:
        * On promote, the manager publishes a PromoteDrafter event
          tagged with the current wall-clock unix_ms and our
          local_peer_id.
        * On receiving a PromoteDrafter (including our own, looped
          back through the bus), the manager updates
          ``active_drafter_id`` IF the incoming unix_ms is newer
          than the currently-tracked one. Stale events are dropped
          with a structured log entry.
    """

    def __init__(
        self,
        *,
        bus: SwarmEventBus,
        local_peer_id: str,
        registry: DraftModelRegistry,
        coord_alive: Callable[[], bool] = lambda: True,
        absence_threshold_ms: int = 5_000,
    ) -> None:
        if not local_peer_id:
            raise ValueError("FailoverManager requires non-empty local_peer_id")
        self._bus = bus
        self._local = str(local_peer_id)
        self._registry = registry
        self._coord_alive = coord_alive
        self._absence_threshold_ms = max(100, int(absence_threshold_ms))

        self._lock = threading.Lock()
        self._active: Optional[_ActiveDrafter] = None
        # Wall-clock timestamp the coord was first observed absent
        # in the current absence streak. Reset on next "alive" sample.
        self._absent_since_ms: Optional[int] = None

        self._unsubscribe = bus.subscribe(
            EVENT_TYPE_PROMOTE_DRAFTER, self._on_promote,
        )

    @property
    def active_drafter_id(self) -> Optional[str]:
        with self._lock:
            return self._active.peer_id if self._active is not None else None

    @property
    def is_local_active(self) -> bool:
        return self.active_drafter_id == self._local

    def _on_promote(self, event: SwarmEvent) -> None:
        if not isinstance(event.payload, PromoteDrafter):
            return
        incoming = _ActiveDrafter(
            peer_id=event.payload.to_peer_id,
            unix_ms=int(event.payload.unix_ms),
        )
        with self._lock:
            current = self._active
            if current is not None and incoming.unix_ms < current.unix_ms:
                logger.warning(
                    "failover_stale_promote: incoming peer=%s unix_ms=%d "
                    "current peer=%s unix_ms=%d — dropping",
                    incoming.peer_id, incoming.unix_ms,
                    current.peer_id, current.unix_ms,
                )
                return
            self._active = incoming
        logger.info(
            "failover_active_drafter_updated: peer=%s unix_ms=%d "
            "from_peer=%s",
            incoming.peer_id, incoming.unix_ms, event.from_peer,
        )

    def check_coord(self, *, now_ms: Optional[int] = None) -> bool:
        """Sample coord liveness and emit a promote event if the
        absence streak crosses the threshold.

        Args:
            now_ms: Wall-clock timestamp; default ``time.time()``.
                Pass-through for deterministic tests.

        Returns:
            ``True`` if a promotion was emitted on this call,
            ``False`` otherwise. Subsequent calls won't re-promote
            until the absence streak resets.
        """
        ts = int(now_ms if now_ms is not None else time.time() * 1000)

        if self._coord_alive():
            with self._lock:
                self._absent_since_ms = None
            return False

        with self._lock:
            if self._absent_since_ms is None:
                self._absent_since_ms = ts
                return False
            elapsed = ts - self._absent_since_ms
            if elapsed < self._absence_threshold_ms:
                return False
            # Reset the streak so we don't keep re-promoting on
            # every poll while still absent. A second promotion
            # only happens if the coord recovers, fails again, and
            # the streak rebuilds.
            self._absent_since_ms = None

        return self.promote(now_ms=ts)

    def promote(self, *, now_ms: Optional[int] = None) -> bool:
        """Force-emit a PromoteDrafter (skips the absence check).

        Used (a) by check_coord when the threshold is met, (b) by
        the operator path during planned drafter handoff. Refuses
        to promote if no draft spec has been registered yet — the
        promoted peer would have nothing to load.

        Returns ``True`` if the event was emitted, ``False`` only
        in rare race-y cases where the spec is None despite an
        active subscription (defence in depth).
        """
        spec_announce = self._registry.get_active_announcement()
        if spec_announce is None:
            raise FailoverError(
                "no_active_spec",
                "FailoverManager.promote: no RegisterDraftModel event "
                "has ever been received; cannot promote without "
                "knowing what draft weights to load",
            )

        ts = int(now_ms if now_ms is not None else time.time() * 1000)
        from_peer = spec_announce.from_peer or self._local
        payload = PromoteDrafter(
            from_peer_id=from_peer,
            to_peer_id=self._local,
            unix_ms=ts,
        )
        try:
            self._bus.publish(payload, from_peer=self._local)
        except Exception as exc:  # pragma: no cover — logged
            logger.error(
                "failover_promote_publish_failed: err=%s",
                exc, exc_info=True,
            )
            return False
        logger.info(
            "failover_promote_emitted: from=%s to=%s unix_ms=%d",
            from_peer, self._local, ts,
        )
        return True

    def close(self) -> None:
        unsub = getattr(self, "_unsubscribe", None)
        if unsub is not None:
            try:
                unsub()
            except Exception:  # pragma: no cover
                pass
            self._unsubscribe = None
