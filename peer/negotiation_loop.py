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

"""Phase 4 continuous re-negotiation — the Heartbeat.

The Phase 3 :class:`peer.swarm_negotiator.SwarmNegotiator` runs once at
boot.  This module wraps it in a periodic background thread so a node
can **self-heal** when:

* A gap opens because a neighbour dropped offline.
* A bigger peer arrives and pushes us off a range we were staking.
* Our own capacity drifts (VRAM pressure from other workloads).

Three strict guardrails from the Phase 4 spec:

1. **Tick loop.**  Default cadence ``60 s``; configurable via constructor
   arg or ``--negotiation-interval-s`` CLI flag.
2. **Active-generation safety lock.**  Never re-assign while the peer has
   in-flight requests — a reshard mid-inference would rip the rug out
   from under a live pipeline.  The loop consults ``is_busy_fn`` each
   tick; when busy, it refreshes the capacity snapshot for the announce
   loop but skips the re-negotiation step.
3. **Dynamic capacity refresh.**  Before each tick the loop calls the
   injected ``build_capacity_report_fn`` so VRAM drift is reflected in
   the next announcement — even on ticks where we don't re-negotiate.

The loop writes its latest state to a :class:`LoopSnapshot` that the
announce loop reads before each DHT broadcast.  That's the only shared
mutable state; a single :class:`threading.Lock` protects it.

**Scope guardrails (what this phase does NOT do):**

* Does NOT tear down a loaded model shard mid-run.  When a re-assignment
  differs from the current assignment, the loop records a
  ``reshard_pending`` log line and updates the broadcast payload; the
  actual model reload is out of scope for Phase 4 (tracked for a future
  ``ReshardExecutor`` that drains requests + reloads weights).
* Does NOT trigger the existing ``LayerRebalancer`` / ``SwarmRebalancer``.
  Those remain the coordinator-side rebalance path; this loop is the
  peer's own initiative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import threading
import time
from typing import Any, Callable

from peer.swarm_negotiator import (
    DhtScanFn,
    ShardAssignment,
    SwarmNegotiator,
)


logger = logging.getLogger(__name__)


# Default tick cadence. 60 s matches the Announcement heartbeat TTL pattern
# in :mod:`peer.dht_announce` and gives the DHT enough time to propagate a
# neighbour's announce before we evaluate against stale data.
DEFAULT_NEGOTIATION_INTERVAL_S = 60.0

# Minimum enforced cadence. Below this we'd risk CPU contention and DHT
# thrash during conflict-resolution races.
MIN_NEGOTIATION_INTERVAL_S = 5.0


# Reasons a tick was skipped. Plain strings for JSON-safety in logs.
SKIP_REASON_BUSY = "busy"
SKIP_REASON_NEGOTIATION_FAILED = "negotiation_failed"
SKIP_REASON_CAPACITY_BUILD_FAILED = "capacity_build_failed"
SKIP_REASON_NONE = ""


@dataclass
class LoopSnapshot:
    """Thread-safe container the negotiation loop writes and the announce
    loop reads.

    The announce loop builds a fresh :class:`peer.dht_announce.Announcement`
    every ~60 s; before each build it calls :meth:`snapshot` on this
    object so every outbound DHT announce carries the **current** capacity
    profile — not the snapshot captured at boot 12 hours ago.
    """

    _capacity_json: str = ""
    _capacity_schema_version: int = 0
    _current_assignment: ShardAssignment | None = None
    _last_updated_unix_ms: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def build(
        cls,
        *,
        capacity_json: str = "",
        capacity_schema_version: int = 0,
        current_assignment: ShardAssignment | None = None,
    ) -> "LoopSnapshot":
        """Preferred constructor — sets the timestamp automatically so
        readers can tell "snapshot from boot" apart from "never populated"."""
        return cls(
            _capacity_json=str(capacity_json or ""),
            _capacity_schema_version=int(capacity_schema_version or 0),
            _current_assignment=current_assignment,
            _last_updated_unix_ms=int(time.time() * 1000),
        )

    def snapshot(self) -> tuple[str, int, ShardAssignment | None, int]:
        """Return ``(capacity_json, schema_version, assignment, last_updated_ms)``."""
        with self._lock:
            return (
                self._capacity_json,
                self._capacity_schema_version,
                self._current_assignment,
                self._last_updated_unix_ms,
            )

    def update(
        self,
        *,
        capacity_json: str,
        capacity_schema_version: int,
        current_assignment: ShardAssignment | None,
    ) -> None:
        """Atomically replace the snapshot — called by the negotiation loop
        after each successful tick."""
        now_ms = int(time.time() * 1000)
        with self._lock:
            self._capacity_json = str(capacity_json or "")
            self._capacity_schema_version = int(capacity_schema_version or 0)
            self._current_assignment = current_assignment
            self._last_updated_unix_ms = now_ms


@dataclass(frozen=True)
class TickResult:
    """Summary of one tick — useful for logging and for unit tests that
    drive the loop step-by-step via :meth:`NegotiationLoop.tick_once`."""

    capacity_refreshed: bool
    capacity_json_bytes: int
    new_assignment: ShardAssignment | None
    assignment_changed: bool
    skipped_reason: str  # "" = ran to completion; otherwise one of SKIP_REASON_*
    timestamp_unix_ms: int


class NegotiationLoop:
    """Periodic re-negotiation background loop.

    Ownership: start exactly one per peer; call :meth:`start` after the
    PeerService is constructed (so ``is_busy_fn`` has something to poll).

    The loop is pure synchronous logic wrapped in a thread — all I/O
    (DHT scans, capacity reports) is injected as callables so unit tests
    can drive ``tick_once()`` deterministically.
    """

    def __init__(
        self,
        *,
        # The thing that mints new CapacityReports on demand.
        # Called at the start of every tick for fresh VRAM/RAM/throughput
        # numbers.  Should never raise — loop catches and logs.
        build_capacity_report_fn: Callable[[], Any],  # -> CapacityReport
        # SwarmNegotiator factory — takes a fresh CapacityReport and returns
        # a negotiator.  We rebuild the negotiator each tick because its
        # capacity_report is passed by reference and the report is frozen.
        make_negotiator_fn: Callable[..., SwarmNegotiator],
        # Snapshot the announce loop reads from before every Announcement.
        snapshot: LoopSnapshot,
        # Initial assignment handed down from Phase 3 boot negotiation.
        # The loop holds this as its "current" assignment until a successful
        # tick changes it.
        initial_assignment: ShardAssignment | None = None,
        # Idle gate: return True while the peer is serving a Forward request.
        # Defaults to "always idle" — the loop still refreshes capacity but
        # will happily re-assign if this is not wired up.
        is_busy_fn: Callable[[], bool] | None = None,
        # Tick cadence in seconds.  Clamped to MIN_NEGOTIATION_INTERVAL_S.
        interval_s: float = DEFAULT_NEGOTIATION_INTERVAL_S,
        # Used by tests to bypass the real ``time.monotonic`` clock.
        clock_fn: Callable[[], float] = time.monotonic,
    ):
        self._build_report = build_capacity_report_fn
        self._make_negotiator = make_negotiator_fn
        self._snapshot = snapshot
        self._current_assignment: ShardAssignment | None = initial_assignment
        self._is_busy = is_busy_fn or (lambda: False)
        self._interval_s = max(MIN_NEGOTIATION_INTERVAL_S, float(interval_s))
        self._clock = clock_fn

        self._stop_event = threading.Event()
        # PR-3 (B1): wake_event forces an immediate re-tick without waiting
        # for ``interval_s``. Used by the gossip client on ``PEER_DEAD`` —
        # when a peer in our routing table just died, we want sub-second
        # re-negotiation, not a 60 s wait. Setting ``_wake_event`` during
        # ``self._stop_event.wait(interval_s)`` causes the wait to return
        # immediately; the loop clears the flag and runs a fresh tick.
        self._wake_event = threading.Event()
        self._thread: threading.Thread | None = None
        # Monotonic counters — handy for introspection + tests.
        self._tick_count = 0
        self._skip_count = 0
        self._assignment_change_count = 0
        self._wake_count = 0

    # ── public lifecycle ───────────────────────────────────────────────

    def start(self, thread_name: str = "openhydra-negotiation") -> threading.Thread:
        """Spin up the background tick thread.  Idempotent — re-calling
        after :meth:`stop` starts a fresh thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.debug("negotiation_loop already running — no-op start()")
            return self._thread
        self._stop_event.clear()
        thread = threading.Thread(
            target=self._run,
            name=thread_name,
            daemon=True,
        )
        thread.start()
        self._thread = thread
        logger.info(
            "negotiation_loop_started: interval=%.1fs initial_assignment=%s",
            self._interval_s,
            (self._current_assignment.model_id + ":"
             + str(self._current_assignment.layer_start) + "-"
             + str(self._current_assignment.layer_end))
            if self._current_assignment is not None else "none",
        )
        return thread

    def wake(self) -> None:
        """Force the next tick to happen immediately, without waiting for
        the ``interval_s`` timer to expire (PR-3 / B1).

        Thread-safe and idempotent: calling ``wake()`` multiple times
        before a single tick has run coalesces into one immediate tick.
        Safe to call from any thread, including from a gossip subscriber
        callback on the :class:`peer.gossip_client.GossipClient` poll
        thread.
        """
        self._wake_event.set()

    def stop(self, join_timeout_s: float = 5.0) -> None:
        """Signal the loop to exit and wait for it to finish one in-flight tick."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=join_timeout_s)
        self._thread = None
        logger.info(
            "negotiation_loop_stopped: ticks=%d skips=%d changes=%d",
            self._tick_count, self._skip_count, self._assignment_change_count,
        )

    # ── counters / introspection (useful for tests + metrics) ─────────

    @property
    def tick_count(self) -> int:
        return int(self._tick_count)

    @property
    def wake_count(self) -> int:
        """How many times :meth:`wake` fired and shortened a tick interval
        (PR-3 observability — handy for verifying gossip-driven reactivity
        in integration tests)."""
        return int(self._wake_count)

    @property
    def skip_count(self) -> int:
        return int(self._skip_count)

    @property
    def assignment_change_count(self) -> int:
        return int(self._assignment_change_count)

    @property
    def current_assignment(self) -> ShardAssignment | None:
        return self._current_assignment

    @property
    def interval_s(self) -> float:
        return float(self._interval_s)

    # ── single-step entry for tests ────────────────────────────────────

    def tick_once(self) -> TickResult:
        """Run exactly one negotiation tick synchronously.  Public for tests;
        production code should use :meth:`start` / :meth:`stop`.

        Tick flow:
            1. Build a fresh CapacityReport (dynamic capacity refresh).
            2. If peer is busy → update snapshot, SKIP re-negotiation.
            3. Build a negotiator for the fresh report, call negotiate().
            4. If result differs from current assignment → log + update
               snapshot; also bump ``assignment_change_count``.
            5. Always update the snapshot with the latest capacity_json.
        """
        self._tick_count += 1
        now_ms = int(time.time() * 1000)

        # Step 1 — refresh capacity.
        try:
            report = self._build_report()
        except Exception as exc:
            # Capacity build failure is benign — we keep broadcasting the
            # previous snapshot.  No assignment change this tick.
            logger.warning(
                "negotiation_tick_capacity_build_failed: %s — "
                "keeping previous snapshot",
                exc,
            )
            self._skip_count += 1
            return TickResult(
                capacity_refreshed=False,
                capacity_json_bytes=len(self._snapshot.snapshot()[0]),
                new_assignment=self._current_assignment,
                assignment_changed=False,
                skipped_reason=SKIP_REASON_CAPACITY_BUILD_FAILED,
                timestamp_unix_ms=now_ms,
            )

        import json as _json
        capacity_json = _json.dumps(report.to_dict())
        schema_version = int(getattr(report, "schema_version", 0))

        # Step 2 — idle gate.
        busy = False
        try:
            busy = bool(self._is_busy())
        except Exception as exc:
            # If the idle check itself raises, treat as busy — safer to
            # skip a tick than to rip the rug out on a spurious state.
            logger.warning(
                "negotiation_tick_is_busy_fn_raised: %s — treating as busy",
                exc,
            )
            busy = True

        if busy:
            # Still push the fresh capacity_json so downstream peers see
            # drift even while we're serving.  We just don't RE-NEGOTIATE.
            self._snapshot.update(
                capacity_json=capacity_json,
                capacity_schema_version=schema_version,
                current_assignment=self._current_assignment,
            )
            self._skip_count += 1
            logger.debug(
                "negotiation_tick_skipped: reason=%s "
                "(capacity refreshed, re-negotiation deferred)",
                SKIP_REASON_BUSY,
            )
            return TickResult(
                capacity_refreshed=True,
                capacity_json_bytes=len(capacity_json),
                new_assignment=self._current_assignment,
                assignment_changed=False,
                skipped_reason=SKIP_REASON_BUSY,
                timestamp_unix_ms=now_ms,
            )

        # Step 3 — re-negotiate against the fresh DHT state.
        try:
            negotiator = self._make_negotiator(report)
            new_assignment = negotiator.negotiate()
        except Exception as exc:
            logger.warning(
                "negotiation_tick_negotiate_failed: %s — "
                "keeping current assignment, refreshing capacity only",
                exc,
            )
            self._snapshot.update(
                capacity_json=capacity_json,
                capacity_schema_version=schema_version,
                current_assignment=self._current_assignment,
            )
            self._skip_count += 1
            return TickResult(
                capacity_refreshed=True,
                capacity_json_bytes=len(capacity_json),
                new_assignment=self._current_assignment,
                assignment_changed=False,
                skipped_reason=SKIP_REASON_NEGOTIATION_FAILED,
                timestamp_unix_ms=now_ms,
            )

        # Step 4 — change detection.
        changed = _assignment_changed(self._current_assignment, new_assignment)
        if changed:
            logger.info(
                "negotiation_reshard_pending: old=%s new=%s "
                "(re-announce will reflect the new range; actual model "
                "reshard is out of scope for Phase 4)",
                _assignment_repr(self._current_assignment),
                _assignment_repr(new_assignment),
            )
            self._current_assignment = new_assignment
            self._assignment_change_count += 1

        # Step 5 — publish the snapshot the announce loop reads from.
        self._snapshot.update(
            capacity_json=capacity_json,
            capacity_schema_version=schema_version,
            current_assignment=self._current_assignment,
        )

        return TickResult(
            capacity_refreshed=True,
            capacity_json_bytes=len(capacity_json),
            new_assignment=self._current_assignment,
            assignment_changed=changed,
            skipped_reason=SKIP_REASON_NONE,
            timestamp_unix_ms=now_ms,
        )

    # ── private loop body ──────────────────────────────────────────────

    def _run(self) -> None:
        """Thread entry-point.  Sleeps ``interval_s`` between ticks; wakes
        early on :meth:`stop` or :meth:`wake`."""
        # Run an immediate first tick so the announce loop sees a fresh
        # snapshot within a few seconds of boot, instead of waiting a
        # full interval.
        while not self._stop_event.is_set():
            try:
                self.tick_once()
            except Exception as exc:  # pragma: no cover — defensive
                logger.exception("negotiation_tick_unhandled: %s", exc)
            # Wait for either:
            #  (a) the interval elapses  — normal cadence
            #  (b) _stop_event is set    — clean shutdown
            #  (c) _wake_event is set    — external wake (PR-3 gossip-driven).
            # Poll both events without racing by combining them into a
            # small-granularity wait.
            waited = 0.0
            step = min(0.25, self._interval_s)
            while waited < self._interval_s:
                if self._stop_event.wait(step):
                    return
                if self._wake_event.is_set():
                    self._wake_event.clear()
                    self._wake_count += 1
                    logger.info(
                        "negotiation_wake: forcing immediate re-tick "
                        "(waited_s=%.2f)", waited
                    )
                    break
                waited += step


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _assignment_changed(
    old: ShardAssignment | None,
    new: ShardAssignment | None,
) -> bool:
    """Semantic equality — two ``ShardAssignment`` values are equivalent if
    they cover the same ``(model_id, layer_start, layer_end, total_layers)``
    tuple.  ``source`` is excluded; it's advisory, not part of the
    effective assignment."""
    if old is None and new is None:
        return False
    if old is None or new is None:
        return True
    return (
        old.model_id != new.model_id
        or old.layer_start != new.layer_start
        or old.layer_end != new.layer_end
        or old.total_layers != new.total_layers
    )


def _assignment_repr(a: ShardAssignment | None) -> str:
    if a is None:
        return "none"
    return f"{a.model_id}[{a.layer_start}-{a.layer_end})/{a.total_layers}"
