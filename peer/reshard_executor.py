# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Track-B3 ``ReshardExecutor`` — the drain-unload-reload FSM.

The :class:`peer.negotiation_loop.NegotiationLoop` detects a desired layer
reassignment every ``interval_s`` (or on a gossip-driven wake) and writes
it into the shared :class:`peer.negotiation_loop.LoopSnapshot`. Up to now
that snapshot update was *advisory only* — the actual model weights on
the peer never re-sharded, and the next ``Forward`` request hit
``shard_layer_mismatch`` once the coordinator's routing caught up. The
"fallback_whole_model" leak that blocked live benchmarks in PR-3 was
exactly that gap.

This module closes it.

Design
------

The executor runs a **5-state FSM** on each accepted proposal:

    IDLE → DRAINING → UNLOADING → LOADING → RESUMING → IDLE

* ``IDLE`` — normal serving. ``propose(new)`` transitions to ``DRAINING``.
* ``DRAINING`` — peer signals intent to stop taking new work via
  ``service.set_resource_budget(should_yield=True, reason="resharding")``.
  The coordinator sees the yield flag and stops routing new requests to
  us. We wait on ``service.inflight_count() == 0`` with a configurable
  timeout (default 120 s, per the master plan — users must not be
  penalised for a reshard event).
* ``UNLOADING`` — acquire the service's reshard lock, drop the current
  ``shard``, free CUDA / MPS / Metal caches, flush KV cache holders.
* ``LOADING`` — construct a fresh ``ModelShard`` with the new
  ``runtime_layer_indices``. Synchronous and potentially slow; peer
  returns ``UNAVAILABLE`` to any in-flight gRPC during this window.
* ``RESUMING`` — clear the yield flag, publish ``RESHARD_ANNOUNCE`` via
  the gossip client (if wired), and re-announce capacity on the next
  DHT tick (handled upstream by the announce loop's snapshot read).

Failure policy (per user directive at the top of the master plan
execution): **stay degraded and log, do not ``execv``**. A failed
``LOADING`` leaves the peer in the degraded ``LOADING_FAILED`` terminal
state until manual recovery; the process does not auto-restart. This
surfaces CUDA / Metal memory leaks instead of masking them.

Feature flag
------------

Off by default via the caller's ``--reshard-executor-enabled`` CLI flag.
When disabled the ``NegotiationLoop`` keeps its existing behaviour (log
``reshard_pending`` and update snapshot only). The executor itself is
a plain Python module; the flag gates only the hookup in
``coordinator/node.py``.

Thread model
------------

* ``propose(new)`` is intentionally synchronous on the caller's thread.
  The ``NegotiationLoop`` tick thread drives a proposal through the
  full FSM before returning; if the loop's ``interval_s`` is shorter
  than a reshard cycle (plausible on 9B models), concurrent proposals
  queue via ``_pending`` and run after the current one completes.
* A per-executor ``threading.RLock`` serialises FSM transitions so two
  wake events racing in from gossip can't interleave states.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from peer.swarm_negotiator import ShardAssignment

logger = logging.getLogger(__name__)


# ─── Public surface ──────────────────────────────────────────────────────────


class ReshardState(str, Enum):
    IDLE = "idle"
    DRAINING = "draining"
    UNLOADING = "unloading"
    LOADING = "loading"
    RESUMING = "resuming"
    LOADING_FAILED = "loading_failed"  # degraded terminal state


# Transition-reason strings emitted in structured logs + test assertions.
REASON_SAME_ASSIGNMENT = "same_assignment_noop"
REASON_DRAIN_TIMEOUT = "drain_timeout_proceeding"
REASON_UNLOAD_FAILED = "unload_failed"
REASON_LOAD_FAILED = "load_failed_staying_degraded"
REASON_GOSSIP_PUBLISH_FAILED = "gossip_publish_failed_nonfatal"
REASON_CONCURRENT_PROPOSAL_QUEUED = "concurrent_proposal_queued"

DEFAULT_DRAIN_TIMEOUT_S = 120.0
# Master plan §"Drain timeout": wait up to 120 s for inflight to complete.
# Under long generation loads on 9B this may not be enough; the executor
# falls through to UNLOADING with the REASON_DRAIN_TIMEOUT log after the
# budget expires, on the premise that the coordinator has already stopped
# routing new work to us and a few token-tails isn't worth blocking the
# whole reshard indefinitely.


@dataclass(frozen=True)
class ReshardResult:
    """The outcome of a single :meth:`ReshardExecutor.propose` call."""

    success: bool
    old_assignment: ShardAssignment | None
    new_assignment: ShardAssignment | None
    final_state: ReshardState
    duration_s: float
    skipped: bool = False
    skip_reason: str = ""
    failure_reason: str = ""
    transitions: tuple[tuple[str, float], ...] = ()  # (state, elapsed_s)


@dataclass
class ReshardStats:
    """Mutable counters for observability (mirrored into
    ``/v1/internal/capacity`` by api_server in a follow-up)."""

    proposals_total: int = 0
    proposals_skipped: int = 0
    reshards_succeeded: int = 0
    reshards_failed: int = 0
    current_state: ReshardState = ReshardState.IDLE
    last_duration_s: float = 0.0
    last_failure_reason: str = ""


# ─── The executor ────────────────────────────────────────────────────────────


class ReshardExecutor:
    """Drain → unload → reload the local ``PeerService.shard`` to match a
    new :class:`ShardAssignment` from the negotiator.

    Parameters
    ----------
    service
        Anything that quacks like :class:`peer.server.PeerService`:

        * ``inflight_count() -> int``
        * ``set_resource_budget(budget)`` (from
          :mod:`peer.daemon_monitor`)
        * ``reload_shard(assignment)`` — performs the actual PyTorch /
          MLX teardown + reload. Must raise on failure.

        Tests inject a stub; production wires the real ``PeerService``.

    drain_timeout_s
        Max wall-clock the DRAINING state will wait on ``inflight_count
        == 0`` before proceeding with UNLOADING regardless. 120 s
        default matches the master plan.

    gossip_publish_fn
        Optional ``callable(event_type: str, data: dict) -> bool``. When
        wired, ``RESUMING`` publishes a ``RESHARD_ANNOUNCE`` event after
        a successful reload. Failure is logged and *not* surfaced as a
        reshard failure — the reshard itself succeeded, just the
        announcement didn't.

    clock_fn
        Injected monotonic clock. Tests pass a deterministic fake.
    """

    def __init__(
        self,
        *,
        service: Any,
        drain_timeout_s: float = DEFAULT_DRAIN_TIMEOUT_S,
        gossip_publish_fn: Callable[[str, dict[str, Any]], bool] | None = None,
        clock_fn: Callable[[], float] = time.monotonic,
        drain_poll_interval_s: float = 0.1,
    ) -> None:
        self._service = service
        self._drain_timeout_s = max(0.0, float(drain_timeout_s))
        self._drain_poll_interval_s = max(0.01, float(drain_poll_interval_s))
        self._publish = gossip_publish_fn
        self._clock = clock_fn

        self._lock = threading.RLock()
        self._current_assignment: ShardAssignment | None = None
        self._state: ReshardState = ReshardState.IDLE
        self._stats = ReshardStats()
        # If a proposal arrives mid-reshard, store it here; we'll apply
        # it immediately after the current cycle hits IDLE.
        self._pending: ShardAssignment | None = None

    # ── introspection ────────────────────────────────────────────────

    def state(self) -> ReshardState:
        with self._lock:
            return self._state

    def current_assignment(self) -> ShardAssignment | None:
        with self._lock:
            return self._current_assignment

    def stats(self) -> ReshardStats:
        with self._lock:
            # Return a snapshot copy so callers can inspect without
            # holding the lock.
            return ReshardStats(
                proposals_total=self._stats.proposals_total,
                proposals_skipped=self._stats.proposals_skipped,
                reshards_succeeded=self._stats.reshards_succeeded,
                reshards_failed=self._stats.reshards_failed,
                current_state=self._state,
                last_duration_s=self._stats.last_duration_s,
                last_failure_reason=self._stats.last_failure_reason,
            )

    # ── public API ──────────────────────────────────────────────────

    def set_initial_assignment(self, assignment: ShardAssignment | None) -> None:
        """Record what the peer believes it is currently serving.

        Called once at boot from :mod:`coordinator.node` so the first
        ``propose`` call can correctly decide whether to no-op.
        """
        with self._lock:
            self._current_assignment = assignment

    def propose(self, new: ShardAssignment) -> ReshardResult:
        """Run the full drain-unload-reload FSM for *new*.

        Synchronous: returns when the peer is back in IDLE (success),
        LOADING_FAILED (degraded), or as a queued no-op (another
        proposal was already mid-flight and has priority).

        A proposal that matches ``current_assignment`` is a cheap
        skipped no-op; the counters record it for observability.
        """
        with self._lock:
            self._stats.proposals_total += 1

            # Idempotency: same shape → no-op.
            if _same_shard(self._current_assignment, new):
                self._stats.proposals_skipped += 1
                logger.debug(
                    "reshard_noop_same_assignment: %s", _fmt(new)
                )
                return ReshardResult(
                    success=True,
                    old_assignment=self._current_assignment,
                    new_assignment=new,
                    final_state=self._state,
                    duration_s=0.0,
                    skipped=True,
                    skip_reason=REASON_SAME_ASSIGNMENT,
                )

            # Coalesce concurrent proposals: if we're already in a
            # non-IDLE state, queue *new* and return immediately.
            if self._state != ReshardState.IDLE:
                self._pending = new
                self._stats.proposals_skipped += 1
                logger.info(
                    "reshard_concurrent_queued: current_state=%s pending=%s",
                    self._state.value, _fmt(new),
                )
                return ReshardResult(
                    success=True,
                    old_assignment=self._current_assignment,
                    new_assignment=new,
                    final_state=self._state,
                    duration_s=0.0,
                    skipped=True,
                    skip_reason=REASON_CONCURRENT_PROPOSAL_QUEUED,
                )

            # Clear the pending slot and take ownership of *new*. If an
            # even newer proposal arrives while we're running, it goes
            # into ``_pending`` and gets picked up at the end.
            self._pending = None
            old = self._current_assignment

        # Run the FSM **outside** the lock so the draining poll doesn't
        # stall other callers. ``_transition()`` re-acquires the lock
        # on every state change so stats / state queries stay consistent.
        result = self._run_fsm(old=old, new=new)

        # Drain any pending proposal that showed up while we were busy.
        queued: ShardAssignment | None
        with self._lock:
            queued = self._pending
            self._pending = None
        if queued is not None:
            logger.info(
                "reshard_draining_pending: next=%s", _fmt(queued)
            )
            # Recurse synchronously — caller already expects an
            # end-to-end block. The pending slot self-limits to one
            # entry so this can't unbounded-recurse.
            self.propose(queued)

        return result

    # ── internals ────────────────────────────────────────────────────

    def _run_fsm(
        self,
        *,
        old: ShardAssignment | None,
        new: ShardAssignment,
    ) -> ReshardResult:
        fsm_start = self._clock()
        transitions: list[tuple[str, float]] = []

        def _tick(state: ReshardState) -> None:
            transitions.append((state.value, self._clock() - fsm_start))
            with self._lock:
                self._state = state
                self._stats.current_state = state
            logger.info(
                "reshard_state: %s old=%s new=%s", state.value, _fmt(old), _fmt(new)
            )

        # ── DRAINING ──
        _tick(ReshardState.DRAINING)
        drain_timed_out = self._drain()

        # ── UNLOADING ──
        _tick(ReshardState.UNLOADING)
        try:
            if hasattr(self._service, "teardown_shard"):
                self._service.teardown_shard()
        except Exception as exc:  # noqa: BLE001
            logger.exception("reshard_unload_failed: %s", exc)
            return self._finalise_failure(
                old=old, new=new,
                fsm_start=fsm_start, transitions=transitions,
                failure_reason=REASON_UNLOAD_FAILED,
                final_state=ReshardState.LOADING_FAILED,
            )

        # ── LOADING ──
        _tick(ReshardState.LOADING)
        try:
            self._service.reload_shard(new)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "reshard_load_failed_staying_degraded: %s — NOT execv-ing, "
                "surface the leak so we can fix it", exc,
            )
            return self._finalise_failure(
                old=old, new=new,
                fsm_start=fsm_start, transitions=transitions,
                failure_reason=f"{REASON_LOAD_FAILED}: {exc}",
                final_state=ReshardState.LOADING_FAILED,
            )

        # ── RESUMING ──
        _tick(ReshardState.RESUMING)
        # Clear the yield flag so the coordinator starts routing to us again.
        _reset_should_yield(self._service)
        # Publish RESHARD_ANNOUNCE — non-fatal on failure.
        if self._publish is not None:
            try:
                self._publish(
                    "RESHARD_ANNOUNCE",
                    {
                        "model_id": str(new.model_id),
                        "layer_start": int(new.layer_start),
                        "layer_end": int(new.layer_end),
                        "total_layers": int(new.total_layers),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("%s: %s", REASON_GOSSIP_PUBLISH_FAILED, exc)

        # ── IDLE ──
        _tick(ReshardState.IDLE)
        with self._lock:
            self._current_assignment = new
            self._stats.reshards_succeeded += 1
            self._stats.last_duration_s = self._clock() - fsm_start
            self._stats.last_failure_reason = ""

        duration = self._clock() - fsm_start
        logger.info(
            "reshard_success: old=%s new=%s duration_s=%.2f "
            "drain_timeout=%s transitions=%d",
            _fmt(old), _fmt(new), duration,
            drain_timed_out, len(transitions),
        )
        return ReshardResult(
            success=True,
            old_assignment=old,
            new_assignment=new,
            final_state=ReshardState.IDLE,
            duration_s=duration,
            transitions=tuple(transitions),
        )

    def _drain(self) -> bool:
        """Signal yield + wait on inflight. Returns True on timeout."""
        try:
            _set_should_yield(self._service, reason="resharding")
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.debug("reshard_set_yield_failed_nonfatal: %s", exc)

        deadline = self._clock() + self._drain_timeout_s
        while True:
            try:
                inflight = int(self._service.inflight_count())
            except Exception:  # noqa: BLE001
                inflight = 0
            if inflight <= 0:
                return False
            if self._clock() >= deadline:
                logger.warning(
                    "reshard_drain_timeout: inflight=%d proceeding with "
                    "unload (master-plan policy: prefer reshard completion "
                    "over indefinite block)", inflight,
                )
                return True
            # Sleep in short steps so the timeout honours any clock skew
            # a fake clock injects.
            end = self._clock() + self._drain_poll_interval_s
            while self._clock() < end:
                time.sleep(min(0.01, end - self._clock()))

    def _finalise_failure(
        self,
        *,
        old: ShardAssignment | None,
        new: ShardAssignment,
        fsm_start: float,
        transitions: list[tuple[str, float]],
        failure_reason: str,
        final_state: ReshardState,
    ) -> ReshardResult:
        duration = self._clock() - fsm_start
        transitions.append((final_state.value, duration))
        with self._lock:
            self._state = final_state
            self._stats.current_state = final_state
            self._stats.reshards_failed += 1
            self._stats.last_duration_s = duration
            self._stats.last_failure_reason = failure_reason
        return ReshardResult(
            success=False,
            old_assignment=old,
            new_assignment=new,
            final_state=final_state,
            duration_s=duration,
            failure_reason=failure_reason,
            transitions=tuple(transitions),
        )


# ─── helpers ─────────────────────────────────────────────────────────────────


def _same_shard(
    a: ShardAssignment | None,
    b: ShardAssignment | None,
) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return (
        a.model_id == b.model_id
        and a.layer_start == b.layer_start
        and a.layer_end == b.layer_end
        and a.total_layers == b.total_layers
    )


def _fmt(a: ShardAssignment | None) -> str:
    if a is None:
        return "none"
    return f"{a.model_id}[{a.layer_start}-{a.layer_end})/{a.total_layers}"


def _set_should_yield(service: Any, *, reason: str) -> None:
    """Flip the service's resource budget to ``should_yield=True`` without
    clobbering any other budget fields the caller already set.

    Uses :class:`peer.daemon_monitor.ResourceBudget` shape: the new
    budget is constructed from the service's current one, or a sensible
    default, with only ``should_yield`` + ``reason`` touched.
    """
    try:
        from peer.daemon_monitor import ResourceBudget
    except Exception:  # pragma: no cover — defensive
        return
    try:
        current = service.resource_budget()
        new = ResourceBudget(
            vram_fraction=float(getattr(current, "vram_fraction", 0.0)),
            cpu_fraction=float(getattr(current, "cpu_fraction", 0.0)),
            should_yield=True,
            reason=reason,
        )
    except Exception:  # noqa: BLE001 — service might not expose getter
        new = ResourceBudget(
            vram_fraction=0.0,
            cpu_fraction=0.0,
            should_yield=True,
            reason=reason,
        )
    service.set_resource_budget(new)


def _reset_should_yield(service: Any) -> None:
    """Inverse of :func:`_set_should_yield`. Used by ``RESUMING``."""
    try:
        from peer.daemon_monitor import ResourceBudget
    except Exception:  # pragma: no cover
        return
    try:
        current = service.resource_budget()
        new = ResourceBudget(
            vram_fraction=float(getattr(current, "vram_fraction", 1.0)),
            cpu_fraction=float(getattr(current, "cpu_fraction", 1.0)),
            should_yield=False,
            reason="resumed_from_reshard",
        )
    except Exception:  # noqa: BLE001
        new = ResourceBudget(
            vram_fraction=1.0,
            cpu_fraction=1.0,
            should_yield=False,
            reason="resumed_from_reshard",
        )
    try:
        service.set_resource_budget(new)
    except Exception:  # pragma: no cover
        pass
