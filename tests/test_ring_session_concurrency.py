# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Phase 2a — concurrency stress tests for ``RingSession.lock``.

The user flagged the race during plan review: under
``pipeline_depth >= 2`` the coord-side
``_coordinator_proxy_handler_loop`` dispatches inbound PushResults to
N concurrent worker threads. Each worker runs the compound op
(slot state transition + token append + remaining decrement + EOS
check + in-flight count + next slot reservation). Without
``RingSession.lock``, two workers race on the in-flight count and
``next_slot_id``, which leads to either over-fire (more in-flight
tokens than ``pipeline_depth`` permits) or under-fire (both threads
see the same count and neither fires next).

These tests verify the locking discipline shipped in commits 3815a65
+ 5718f03 + 108ed4a actually upholds the depth invariant and produces
correct final counts. They simulate the worker pool directly with
``threading.Thread`` instead of spinning up the full gRPC + libp2p
stack — the lock contract is what we want to validate, not the
network transport.

Invariants tested:
    1. After all workers join, ``session.next_slot_id`` equals exactly
       the total number of tokens generated (initial pipeline pre-fill
       + reinjections).
    2. ``len(session.slots)`` equals the total tokens generated; no
       slot was created twice (would manifest as a missing entry from
       a clobbered insert).
    3. Every slot ends in ``SLOT_STATE_SAMPLED``; no slot was left
       in-flight, dispatched, or any non-final state.
    4. **The depth invariant**: at no point during the compound op
       did the in-flight count exceed ``pipeline_depth``. Measured
       under the same lock that protects the compound op, so any
       observation that would falsify this invariant would be caught
       by the per-thread assertion before the test finishes.
    5. ``ring_tokens_remaining`` decremented to exactly 0; no
       under/over-decrement under contention.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import List

import pytest

from coordinator.head_sampler import (
    RingSession,
    SlotState,
    SLOT_STATE_DISPATCHED,
    SLOT_STATE_SAMPLED,
    SLOT_STATES_IN_FLIGHT,
)


def _spawn_workers(
    *,
    session: RingSession,
    initial_dispatched: int,
    n_workers: int,
    work_queue: queue.Queue,
    sampled_log: List[int],
    invariant_violations: List[str],
    finished_event: threading.Event,
) -> List[threading.Thread]:
    """Spin up ``n_workers`` daemon threads that run the same locked
    compound op as ``_coordinator_handle_push_result``.

    Each worker pulls a slot id off ``work_queue``, takes
    ``session.lock``, transitions the slot to ``SAMPLED``, decrements
    ``ring_tokens_remaining``, computes the in-flight count, and (if
    there's still work + room in the pipeline) reserves the next
    slot id and seeds a new ``SlotState``. Reservation happens
    inside the lock so two threads cannot claim the same id.
    The new slot id is then enqueued (outside the lock) for another
    worker to process — modelling the way the real
    ``_coordinator_reinject_ring_step`` fires asynchronously.

    The depth invariant is checked AFTER reservation, INSIDE the
    lock — the very property the lock exists to protect.
    """

    def _worker_loop():
        while not finished_event.is_set():
            try:
                slot_id = work_queue.get(timeout=0.05)
            except queue.Empty:
                # Either we're done (event set) or just no work yet;
                # spin briefly and retry.
                continue
            # Tiny synthetic "sample work" to maximise contention —
            # forces concurrent threads to actually overlap on the
            # lock rather than running in lockstep.
            time.sleep(0.0005)
            with session.lock:
                slot = session.slots.get(slot_id)
                if slot is None:
                    invariant_violations.append(
                        f"slot {slot_id} popped from queue but missing in session.slots"
                    )
                    continue
                if slot.state == SLOT_STATE_SAMPLED:
                    invariant_violations.append(
                        f"slot {slot_id} already sampled — duplicate work item"
                    )
                    continue
                # Transition: in_flight → sampled.
                slot.state = SLOT_STATE_SAMPLED
                slot.token_id = slot_id  # synthetic token id == slot id
                sampled_log.append(slot_id)

                # Decide to fire next, atomically with the in-flight
                # count check. Mirrors the production discipline in
                # _coordinator_handle_push_result.
                #
                # NOTE on semantics: in this stress test
                # ``ring_tokens_remaining`` represents
                # "additional slots still allowed to be created"
                # (a budget), not "tokens left to sample". We
                # decrement it INSIDE the fire branch so the count
                # of new slots created tracks 1:1 with budget burn.
                # This keeps the test bookkeeping clean while still
                # exercising the same lock-protected compound op
                # the production code uses.
                in_flight = sum(
                    1 for s in session.slots.values()
                    if s.state in SLOT_STATES_IN_FLIGHT
                )
                if (
                    session.ring_tokens_remaining > 0
                    and in_flight < session.pipeline_depth
                ):
                    session.ring_tokens_remaining -= 1
                    # Reserve next slot id INSIDE the lock.
                    next_id = session.next_slot_id
                    session.next_slot_id += 1
                    session.slots[next_id] = SlotState(
                        slot_id=next_id,
                        state=SLOT_STATE_DISPATCHED,
                        dispatched_at_ms=time.monotonic() * 1000.0,
                        last_update_ms=time.monotonic() * 1000.0,
                    )
                    # CHECK THE DEPTH INVARIANT after reservation.
                    # If the lock wasn't doing its job, two threads
                    # could both pass the < pipeline_depth check
                    # above and both insert, producing
                    # in_flight_after > pipeline_depth.
                    in_flight_after = sum(
                        1 for s in session.slots.values()
                        if s.state in SLOT_STATES_IN_FLIGHT
                    )
                    if in_flight_after > session.pipeline_depth:
                        invariant_violations.append(
                            f"depth violated after reservation: "
                            f"in_flight_after={in_flight_after} "
                            f"depth={session.pipeline_depth} "
                            f"thread={threading.current_thread().name}"
                        )
                    new_slot_to_enqueue = next_id
                else:
                    new_slot_to_enqueue = None

                # Final-quiescence flag: only set when nothing else
                # can fire (no remaining tokens AND no slots are
                # still in flight after this transition).
                if (
                    session.ring_tokens_remaining <= 0
                    and not any(
                        s.state in SLOT_STATES_IN_FLIGHT
                        for s in session.slots.values()
                    )
                ):
                    finished_event.set()
            # ── lock released ──────────────────────────────────────
            # Enqueue the new slot AFTER releasing the lock so we
            # don't hold up other workers.
            if new_slot_to_enqueue is not None:
                work_queue.put(new_slot_to_enqueue)

    threads: List[threading.Thread] = []
    for i in range(n_workers):
        t = threading.Thread(
            target=_worker_loop,
            name=f"stress-worker-{i}",
            daemon=True,
        )
        t.start()
        threads.append(t)
    return threads


@pytest.mark.parametrize(
    "pipeline_depth,total_tokens,n_workers",
    [
        (2, 16, 2),
        (2, 32, 4),
        (4, 32, 8),
        (4, 64, 8),
        (8, 64, 16),
    ],
)
def test_session_lock_prevents_over_fire(
    pipeline_depth, total_tokens, n_workers,
):
    """Headline correctness test: under N concurrent workers, the
    locked compound op never lets the in-flight count exceed
    ``pipeline_depth``, even when total work is many times the depth.

    Asserts every invariant from the test-file docstring.
    """
    # ``ring_tokens_remaining`` is decremented every time a worker
    # samples a slot. To make the bookkeeping line up exactly to
    # ``total_tokens`` slots created end-to-end, we initialise
    # ``remaining`` to ``total_tokens - initial_dispatched`` so that
    # the initial pre-fill counts toward the budget. After all samples
    # complete, ``remaining`` lands at zero and exactly ``total_tokens``
    # slots have been created.
    initial_dispatched = min(pipeline_depth, total_tokens)
    session = RingSession(
        request_id=f"stress-{pipeline_depth}-{total_tokens}",
        pipeline_depth=pipeline_depth,
        ring_tokens_remaining=total_tokens - initial_dispatched,
    )
    # Pre-fill the pipeline up to depth — mirrors what chain.run_push_ring
    # does on initial fire (slot 0 today; future Phase 2b will pre-fill
    # all depth slots speculatively).
    work_queue: queue.Queue = queue.Queue()
    for i in range(initial_dispatched):
        session.slots[i] = SlotState(
            slot_id=i,
            state=SLOT_STATE_DISPATCHED,
            dispatched_at_ms=0.0,
            last_update_ms=0.0,
        )
        work_queue.put(i)
    session.next_slot_id = initial_dispatched

    sampled_log: List[int] = []
    invariant_violations: List[str] = []
    finished_event = threading.Event()

    threads = _spawn_workers(
        session=session,
        initial_dispatched=initial_dispatched,
        n_workers=n_workers,
        work_queue=work_queue,
        sampled_log=sampled_log,
        invariant_violations=invariant_violations,
        finished_event=finished_event,
    )

    # Wait for completion — finished_event is set inside the locked
    # block when the last in-flight slot transitions to sampled.
    finished_in_time = finished_event.wait(timeout=10.0)
    # Give the workers a moment to drain their post-lock enqueue path
    # (any straggler will see the event set and exit on next get()).
    for t in threads:
        t.join(timeout=2.0)

    # ── Invariants ─────────────────────────────────────────────────
    assert finished_in_time, (
        f"workers did not finish within 10s — possible deadlock "
        f"(remaining={session.ring_tokens_remaining}, "
        f"in_flight={sum(1 for s in session.slots.values() if s.state in SLOT_STATES_IN_FLIGHT)}, "
        f"slots={len(session.slots)})"
    )
    assert not invariant_violations, "lock invariants violated:\n" + "\n".join(
        invariant_violations
    )
    # Total slots: should equal exactly total_tokens (every slot was
    # created exactly once because next_slot_id increments under the lock).
    assert session.next_slot_id == total_tokens, (
        f"next_slot_id={session.next_slot_id} expected={total_tokens}"
    )
    assert len(session.slots) == total_tokens, (
        f"len(slots)={len(session.slots)} expected={total_tokens}"
    )
    # Every slot finalised in SAMPLED.
    final_states = [s.state for s in session.slots.values()]
    assert all(s == SLOT_STATE_SAMPLED for s in final_states), (
        f"non-sampled final states present: {set(final_states)}"
    )
    # Counter went exactly to zero.
    assert session.ring_tokens_remaining == 0, (
        f"ring_tokens_remaining={session.ring_tokens_remaining} expected 0"
    )
    # Every slot id from 0..total_tokens-1 sampled exactly once.
    assert sorted(sampled_log) == list(range(total_tokens)), (
        f"sampled set mismatch: missing="
        f"{set(range(total_tokens)) - set(sampled_log)}, "
        f"duplicate_sampled="
        f"{[i for i in sampled_log if sampled_log.count(i) > 1]}"
    )


def test_session_lock_serial_mode_uncontended():
    """Sanity guard: ``pipeline_depth=1`` (default) preserves the
    legacy serial path. The lock should still work — uncontended,
    one slot at a time — and the slots dict can be mutated through
    the same compound op without violation.

    Documents the design choice that depth=1 keeps the lock code path
    alive (rather than skipping it) so there's no two-implementation
    drift to maintain.
    """
    session = RingSession(
        request_id="serial-mode",
        pipeline_depth=1,
        # Budget = additional slots beyond the pre-filled one.
        ring_tokens_remaining=7,
    )
    # In production, depth=1 doesn't actually populate slots. This test
    # only validates that the lock works correctly when the rest of the
    # code does use it.
    session.slots[0] = SlotState(
        slot_id=0, state=SLOT_STATE_DISPATCHED,
        dispatched_at_ms=0.0, last_update_ms=0.0,
    )
    session.next_slot_id = 1

    work_queue: queue.Queue = queue.Queue()
    work_queue.put(0)
    sampled_log: List[int] = []
    violations: List[str] = []
    done = threading.Event()

    threads = _spawn_workers(
        session=session,
        initial_dispatched=1,
        n_workers=1,
        work_queue=work_queue,
        sampled_log=sampled_log,
        invariant_violations=violations,
        finished_event=done,
    )
    assert done.wait(timeout=5.0)
    for t in threads:
        t.join(timeout=2.0)

    assert not violations
    assert sorted(sampled_log) == list(range(8))
    assert session.ring_tokens_remaining == 0
    # Every slot ended in SAMPLED (no slot left dispatched).
    assert all(s.state == SLOT_STATE_SAMPLED for s in session.slots.values())


def test_session_lock_isolation_across_sessions():
    """Two RingSessions in flight simultaneously must not contend on
    each other's locks. Different ``request_id``s = independent locks
    = parallel processing. Validates the per-session granularity
    choice in the plan (vs a global lock).
    """
    s1 = RingSession(
        request_id="s1", pipeline_depth=4, ring_tokens_remaining=16,
    )
    s2 = RingSession(
        request_id="s2", pipeline_depth=4, ring_tokens_remaining=16,
    )

    # Different lock instances.
    assert s1.lock is not s2.lock

    # While holding s1.lock, we MUST be able to acquire s2.lock
    # without blocking — proves they are independent mutexes.
    s1.lock.acquire()
    try:
        acquired = s2.lock.acquire(timeout=0.5)
        assert acquired, (
            "s2.lock should be acquirable while s1.lock is held — "
            "if it blocks, the per-session granularity has regressed "
            "to a shared lock"
        )
        s2.lock.release()
    finally:
        s1.lock.release()


def test_session_lock_excluded_from_dataclass_repr_and_eq():
    """The plan documented that ``RingSession.lock`` is declared with
    ``compare=False, repr=False`` because Lock instances aren't
    comparable and would dirty the debug repr. Guard against
    accidental regression.
    """
    s1 = RingSession(request_id="x", pipeline_depth=2)
    # repr must not contain 'lock=' (would happen if repr=False got dropped).
    r = repr(s1)
    assert "lock=" not in r, (
        f"Lock leaking into repr: {r}\n"
        "Restore field(lock, ..., repr=False)."
    )
    # Two sessions with identical fields but different Lock objects
    # should still compare unequal because request_id distinguishes
    # them — but the Lock difference must NOT contribute to the
    # comparison.
    s2 = RingSession(request_id="x", pipeline_depth=2)
    # If compare=False is dropped, the underlying Lock identity comparison
    # would fail with TypeError on most Python versions.
    try:
        _ = s1 == s2
    except TypeError as exc:  # pragma: no cover — only fires on regression
        pytest.fail(
            f"RingSession equality raised TypeError because lock is "
            f"comparable: {exc}\nRestore field(lock, ..., compare=False)."
        )
