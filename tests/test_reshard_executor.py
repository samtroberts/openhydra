# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Track-B3 ``ReshardExecutor`` tests.

Covers the 5-state FSM end-to-end without any real PyTorch / MLX
teardown: a ``FakeService`` records ``set_resource_budget``,
``inflight_count``, ``teardown_shard`` and ``reload_shard`` calls and
lets tests drive the drain loop deterministically via a mutable
``inflight`` integer.

Coverage:

* Happy-path ``IDLE → DRAINING → UNLOADING → LOADING → RESUMING → IDLE``
  transition sequence.
* Drain waits for inflight to hit zero before proceeding.
* Drain timeout after ``drain_timeout_s`` proceeds anyway
  (master-plan policy: don't block forever on a hung request).
* Idempotent ``propose`` — a proposal that matches the current
  assignment is a skipped no-op with ``skip_reason=same_assignment_noop``.
* Teardown failure transitions to ``LOADING_FAILED`` without reloading.
* Reload failure transitions to ``LOADING_FAILED`` and does **not**
  publish a ``RESHARD_ANNOUNCE`` (master-plan stay-degraded policy).
* Successful reshard publishes ``RESHARD_ANNOUNCE`` via the gossip fn.
* Resource budget is toggled: ``should_yield=True`` during DRAINING,
  ``False`` after RESUMING.
* Concurrent ``propose`` calls coalesce — the second proposal queues
  and runs after the first completes.
* NegotiationLoop integration: when a ``reshard_executor_fn`` is
  wired, the tick that detects a changed assignment calls it.

Run:  ``pytest tests/test_reshard_executor.py -v``
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from peer.reshard_executor import (
    DEFAULT_DRAIN_TIMEOUT_S,
    REASON_CONCURRENT_PROPOSAL_QUEUED,
    REASON_LOAD_FAILED,
    REASON_SAME_ASSIGNMENT,
    REASON_UNLOAD_FAILED,
    ReshardExecutor,
    ReshardState,
    _same_shard,
)
from peer.swarm_negotiator import ShardAssignment


# ─── shared fakes ────────────────────────────────────────────────────────────


@dataclass
class FakeResourceBudget:
    vram_fraction: float = 1.0
    cpu_fraction: float = 1.0
    should_yield: bool = False
    reason: str = "default"


class FakeService:
    """Drop-in for :class:`peer.server.PeerService` for B3 FSM tests."""

    def __init__(
        self,
        *,
        initial_inflight: int = 0,
        teardown_fail: Exception | None = None,
        reload_fail: Exception | None = None,
    ) -> None:
        self.inflight = int(initial_inflight)
        self.teardown_fail = teardown_fail
        self.reload_fail = reload_fail
        self._budget = FakeResourceBudget()
        # Call recording for assertions.
        self.teardown_calls = 0
        self.reload_calls: list[ShardAssignment] = []
        self.budget_history: list[FakeResourceBudget] = []

    # ── PeerService surface ──
    def inflight_count(self) -> int:
        return int(self.inflight)

    def resource_budget(self) -> FakeResourceBudget:
        return FakeResourceBudget(
            vram_fraction=self._budget.vram_fraction,
            cpu_fraction=self._budget.cpu_fraction,
            should_yield=self._budget.should_yield,
            reason=self._budget.reason,
        )

    def set_resource_budget(self, budget: Any) -> None:
        # ``budget`` is :class:`peer.daemon_monitor.ResourceBudget`;
        # snapshot its attrs into our shape.
        self._budget = FakeResourceBudget(
            vram_fraction=float(getattr(budget, "vram_fraction", 0.0)),
            cpu_fraction=float(getattr(budget, "cpu_fraction", 0.0)),
            should_yield=bool(getattr(budget, "should_yield", False)),
            reason=str(getattr(budget, "reason", "")),
        )
        self.budget_history.append(self._budget)

    def teardown_shard(self) -> None:
        self.teardown_calls += 1
        if self.teardown_fail is not None:
            raise self.teardown_fail

    def reload_shard(self, assignment: ShardAssignment) -> None:
        self.reload_calls.append(assignment)
        if self.reload_fail is not None:
            raise self.reload_fail


def _assign(start: int, end: int, source: str = "pick_best_fit") -> ShardAssignment:
    return ShardAssignment(
        model_id="openhydra-qwen3.5-2b",
        layer_start=start,
        layer_end=end,
        total_layers=24,
        source=source,
    )


# ─── happy path ──────────────────────────────────────────────────────────────


class TestHappyPath:
    def test_full_fsm_cycle(self):
        svc = FakeService(initial_inflight=0)
        published: list[tuple[str, dict]] = []
        ex = ReshardExecutor(
            service=svc,
            drain_timeout_s=1.0,
            gossip_publish_fn=lambda t, d: (published.append((t, d)), True)[1],
        )
        ex.set_initial_assignment(_assign(0, 24))

        result = ex.propose(_assign(0, 12))

        assert result.success is True
        assert result.skipped is False
        assert result.final_state == ReshardState.IDLE
        assert result.failure_reason == ""

        # All FSM states touched in order.
        states = [name for name, _ in result.transitions]
        assert states == [
            ReshardState.DRAINING.value,
            ReshardState.UNLOADING.value,
            ReshardState.LOADING.value,
            ReshardState.RESUMING.value,
            ReshardState.IDLE.value,
        ]

        # Service side-effects happened exactly once each.
        assert svc.teardown_calls == 1
        assert len(svc.reload_calls) == 1
        assert svc.reload_calls[0].layer_start == 0
        assert svc.reload_calls[0].layer_end == 12

        # Gossip announce fired with the new assignment shape.
        assert len(published) == 1
        evt_type, data = published[0]
        assert evt_type == "RESHARD_ANNOUNCE"
        assert data["layer_start"] == 0
        assert data["layer_end"] == 12
        assert data["model_id"] == "openhydra-qwen3.5-2b"

        # Current assignment updated post-success.
        cur = ex.current_assignment()
        assert cur is not None
        assert (cur.layer_start, cur.layer_end) == (0, 12)

        # Stats: one successful reshard, zero failures/skips.
        s = ex.stats()
        assert s.proposals_total == 1
        assert s.reshards_succeeded == 1
        assert s.reshards_failed == 0
        assert s.current_state == ReshardState.IDLE

    def test_should_yield_toggles_true_then_false(self):
        svc = FakeService(initial_inflight=0)
        ex = ReshardExecutor(service=svc, drain_timeout_s=1.0)
        ex.set_initial_assignment(_assign(0, 24))
        ex.propose(_assign(0, 12))

        # The FakeService logs every budget update; we expect at least
        # two: one flipping should_yield=True in DRAINING and one
        # flipping it False in RESUMING.
        yield_on = [b for b in svc.budget_history if b.should_yield]
        yield_off = [b for b in svc.budget_history if not b.should_yield]
        assert len(yield_on) >= 1
        assert len(yield_off) >= 1
        assert yield_on[0].reason == "resharding"
        assert yield_off[-1].reason == "resumed_from_reshard"
        # Final state: not yielding.
        assert svc.resource_budget().should_yield is False


# ─── drain behaviour ─────────────────────────────────────────────────────────


class TestDraining:
    def test_drain_waits_for_inflight_zero(self):
        """Drain must block until inflight hits 0, then proceed."""
        svc = FakeService(initial_inflight=3)
        ex = ReshardExecutor(
            service=svc,
            drain_timeout_s=5.0,
            drain_poll_interval_s=0.02,
        )
        ex.set_initial_assignment(_assign(0, 24))

        # Drop the inflight count in a background thread so the drain
        # poll sees the transition and proceeds.
        def _release():
            time.sleep(0.1)
            svc.inflight = 0

        t = threading.Thread(target=_release, daemon=True)
        t.start()
        result = ex.propose(_assign(0, 12))
        t.join()

        assert result.success is True
        assert result.final_state == ReshardState.IDLE
        assert svc.teardown_calls == 1
        # Drain took at least the 0.1s we waited, well under the 5s budget.
        assert 0.05 < result.duration_s < 5.0

    def test_drain_timeout_proceeds_anyway(self):
        """If inflight never hits 0, the drain times out but the FSM
        still proceeds to UNLOADING (users > hung requests)."""
        svc = FakeService(initial_inflight=5)  # never changes
        ex = ReshardExecutor(
            service=svc,
            drain_timeout_s=0.2,
            drain_poll_interval_s=0.02,
        )
        ex.set_initial_assignment(_assign(0, 24))
        result = ex.propose(_assign(0, 12))

        assert result.success is True  # despite the drain timeout
        assert svc.teardown_calls == 1
        assert len(svc.reload_calls) == 1
        # Duration should be at least the drain budget.
        assert result.duration_s >= 0.2


# ─── idempotency ─────────────────────────────────────────────────────────────


class TestIdempotency:
    def test_same_assignment_is_noop(self):
        svc = FakeService()
        ex = ReshardExecutor(service=svc, drain_timeout_s=1.0)
        ex.set_initial_assignment(_assign(0, 12))

        result = ex.propose(_assign(0, 12))
        assert result.skipped is True
        assert result.skip_reason == REASON_SAME_ASSIGNMENT
        assert svc.teardown_calls == 0
        assert svc.reload_calls == []
        s = ex.stats()
        assert s.proposals_skipped == 1
        assert s.reshards_succeeded == 0

    def test_noop_also_when_initial_is_none_and_new_is_none(self):
        ex = ReshardExecutor(service=FakeService(), drain_timeout_s=1.0)
        assert _same_shard(None, None) is True

    def test_non_noop_when_one_is_none(self):
        assert _same_shard(None, _assign(0, 12)) is False
        assert _same_shard(_assign(0, 12), None) is False

    def test_different_source_same_layers_is_noop(self):
        """``source`` field is advisory — layer range identity wins."""
        ex = ReshardExecutor(service=FakeService(), drain_timeout_s=1.0)
        ex.set_initial_assignment(_assign(0, 12, source="fallback_whole_model"))
        result = ex.propose(_assign(0, 12, source="pick_best_fit"))
        assert result.skipped is True


# ─── failure modes ──────────────────────────────────────────────────────────


class TestFailurePaths:
    def test_teardown_failure_halts_fsm(self):
        published: list[tuple[str, dict]] = []
        svc = FakeService(teardown_fail=RuntimeError("cuda oom on del"))
        ex = ReshardExecutor(
            service=svc,
            drain_timeout_s=1.0,
            gossip_publish_fn=lambda t, d: (published.append((t, d)), True)[1],
        )
        ex.set_initial_assignment(_assign(0, 24))

        result = ex.propose(_assign(0, 12))

        assert result.success is False
        assert result.final_state == ReshardState.LOADING_FAILED
        assert REASON_UNLOAD_FAILED in result.failure_reason
        # Reload never attempted.
        assert svc.reload_calls == []
        # No RESHARD_ANNOUNCE.
        assert published == []
        # Current assignment unchanged.
        cur = ex.current_assignment()
        assert cur is not None
        assert cur.layer_end == 24

        s = ex.stats()
        assert s.reshards_failed == 1
        assert s.current_state == ReshardState.LOADING_FAILED

    def test_reload_failure_stays_degraded(self):
        """Master-plan policy: stay degraded, DO NOT execv on reload failure."""
        published: list[tuple[str, dict]] = []
        svc = FakeService(
            reload_fail=RuntimeError("could not allocate 14 GiB on device")
        )
        ex = ReshardExecutor(
            service=svc,
            drain_timeout_s=1.0,
            gossip_publish_fn=lambda t, d: (published.append((t, d)), True)[1],
        )
        ex.set_initial_assignment(_assign(0, 24))
        result = ex.propose(_assign(0, 12))

        assert result.success is False
        assert result.final_state == ReshardState.LOADING_FAILED
        assert REASON_LOAD_FAILED in result.failure_reason
        assert svc.teardown_calls == 1
        assert len(svc.reload_calls) == 1
        # CRITICAL: no announcement — don't tell the swarm we resharded
        # when we didn't. The swarm will route around the failed peer
        # when gRPC errors out.
        assert published == []

    def test_gossip_publish_failure_is_nonfatal(self):
        """RESHARD_ANNOUNCE failing doesn't flip reshard to failed."""
        svc = FakeService()

        def _boom(t, d):
            raise RuntimeError("InsufficientPeers")

        ex = ReshardExecutor(
            service=svc,
            drain_timeout_s=1.0,
            gossip_publish_fn=_boom,
        )
        ex.set_initial_assignment(_assign(0, 24))
        result = ex.propose(_assign(0, 12))

        # Reshard still succeeds despite gossip publish error.
        assert result.success is True
        assert result.final_state == ReshardState.IDLE
        assert svc.teardown_calls == 1
        assert len(svc.reload_calls) == 1


# ─── concurrent proposals ────────────────────────────────────────────────────


class TestConcurrentProposals:
    def test_second_propose_mid_flight_queues(self):
        """While FSM is in DRAINING/UNLOADING/LOADING, a second propose
        is queued and applied after the first completes."""
        svc = FakeService(initial_inflight=1)
        ex = ReshardExecutor(
            service=svc,
            drain_timeout_s=2.0,
            drain_poll_interval_s=0.02,
        )
        ex.set_initial_assignment(_assign(0, 24))

        # Thread 1 starts a reshard; we keep inflight=1 so DRAINING stalls.
        first_result: list = [None]
        def _t1():
            first_result[0] = ex.propose(_assign(0, 12))
        t1 = threading.Thread(target=_t1, daemon=True)
        t1.start()
        # Wait until FSM has actually entered DRAINING before proposing #2.
        deadline = time.monotonic() + 1.0
        while ex.state() == ReshardState.IDLE and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ex.state() != ReshardState.IDLE, "FSM never left IDLE"

        second = ex.propose(_assign(0, 6))
        assert second.skipped is True
        assert second.skip_reason == REASON_CONCURRENT_PROPOSAL_QUEUED

        # Release the drain and let the whole cycle complete.
        svc.inflight = 0
        t1.join(timeout=5.0)
        assert first_result[0] is not None
        assert first_result[0].success is True

        # The queued proposal should have run after — current assignment
        # should reflect *the queued one*, not the first one.
        cur = ex.current_assignment()
        assert cur is not None
        assert (cur.layer_start, cur.layer_end) == (0, 6)
        # Two reload calls: one for [0,12), one for [0,6).
        assert [(r.layer_start, r.layer_end) for r in svc.reload_calls] == [
            (0, 12), (0, 6),
        ]


# ─── NegotiationLoop integration ─────────────────────────────────────────────


class TestNegotiationLoopWiring:
    """The NegotiationLoop must call ``reshard_executor_fn(new)`` when
    ``assignment_changed``. Covered at unit level without running the
    real executor — a stub records what the loop hands to it."""

    def _make_loop(self, *, executor_fn, initial_assignment):
        from peer.negotiation_loop import LoopSnapshot, NegotiationLoop
        from peer.swarm_negotiator import ShardAssignment as _SA

        class _Negotiator:
            def __init__(self, assignments):
                self._a = list(assignments); self._i = 0
            def negotiate(self):
                a = self._a[self._i] if self._i < len(self._a) else self._a[-1]
                self._i += 1
                return a

        class _Cap:
            def to_dict(self):
                return {"schema_version": 2}

        negotiator = _Negotiator([initial_assignment, _assign(0, 12)])
        snapshot = LoopSnapshot.build(
            capacity_json="{}",
            capacity_schema_version=2,
            current_assignment=initial_assignment,
        )
        loop = NegotiationLoop(
            build_capacity_report_fn=lambda: _Cap(),
            make_negotiator_fn=lambda r: negotiator,
            snapshot=snapshot,
            initial_assignment=initial_assignment,
            is_busy_fn=lambda: False,
            interval_s=60.0,
            reshard_executor_fn=executor_fn,
        )
        return loop

    def test_executor_invoked_on_change(self):
        called: list[ShardAssignment] = []
        initial = _assign(0, 24, source="fallback_whole_model")
        loop = self._make_loop(
            executor_fn=lambda new: called.append(new),
            initial_assignment=initial,
        )
        # Boot tick — first stubbed assignment is same as initial → no change.
        loop.tick_once()
        assert called == []
        # Second tick — stubbed assignment flips to [0, 12) → change.
        loop.tick_once()
        assert len(called) == 1
        assert (called[0].layer_start, called[0].layer_end) == (0, 12)

    def test_executor_exception_does_not_derail_loop(self):
        initial = _assign(0, 24, source="fallback_whole_model")
        def _boom(new):
            raise RuntimeError("executor blew up")
        loop = self._make_loop(
            executor_fn=_boom, initial_assignment=initial,
        )
        loop.tick_once()  # no-change
        # Exception is caught inside the loop — must not raise here.
        loop.tick_once()
        # Loop still advances counters.
        assert loop.tick_count == 2
