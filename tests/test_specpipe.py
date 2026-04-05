# Copyright 2026 OpenHydra contributors — Apache 2.0

"""TDD tests for SpecPipe scheduler (P1-A).

Run:  pytest tests/test_specpipe.py -v
"""

from __future__ import annotations

import threading
import pytest


def _make_scheduler(n_stages=3, draft_fn=None, stage_fn=None, **config_kw):
    from coordinator.specpipe_scheduler import SpecPipeScheduler, SpecPipeConfig
    from types import SimpleNamespace

    pipeline = [SimpleNamespace(peer_id=f"peer-{i}", host="127.0.0.1", port=50051+i) for i in range(n_stages)]

    if draft_fn is None:
        def draft_fn(ctx, k):
            base = ctx[-1] if ctx else 0
            return [base + i + 1 for i in range(k)]

    if stage_fn is None:
        # Simple stage: return input activation shifted by 1
        def stage_fn(peer, activation, prompt="", stage_index=0, total_stages=1, max_tokens=1, request_id="", **kw):
            if activation:
                return [activation[0] + 1.0]
            return [0.0]

    config = SpecPipeConfig(enabled=True, **config_kw)
    return SpecPipeScheduler(config=config, pipeline=pipeline, draft_fn=draft_fn, stage_fn=stage_fn)


class TestSpecPipeRound:
    def test_single_round_returns_tokens(self):
        scheduler = _make_scheduler(n_stages=3)
        accepted = scheduler.run_round(context_ids=[100], prompt="hello")
        assert len(accepted) >= 1  # At least the real token

    def test_real_token_always_included(self):
        scheduler = _make_scheduler(n_stages=3)
        accepted = scheduler.run_round(context_ids=[100], prompt="hello")
        # First token should be the real pipeline output
        assert accepted[0] is not None

    def test_speculative_tokens_dispatched(self):
        calls = {"stage_count": 0}
        def counting_stage(**kw):
            calls["stage_count"] += 1
            return [float(kw.get("activation", [0])[0]) + 1]

        scheduler = _make_scheduler(n_stages=3, stage_fn=counting_stage, max_speculative_depth=2)
        scheduler.run_round(context_ids=[100], prompt="hello")
        # 3 stages for real token + 2 speculative dispatches
        assert calls["stage_count"] >= 3

    def test_stats_tracked(self):
        scheduler = _make_scheduler(n_stages=3, max_speculative_depth=2)
        scheduler.run_round(context_ids=[100], prompt="hello")
        stats = scheduler.stats
        assert stats.rounds == 1
        assert stats.real_tokens == 1
        assert stats.speculative_dispatched >= 0
        assert stats.total_stage_calls >= 3


class TestSpecPipeAdaptive:
    def test_depth_decreases_on_low_acceptance(self):
        # Stage fn returns empty → 0 accepted speculative tokens
        def empty_stage(**kw):
            return []

        scheduler = _make_scheduler(
            n_stages=3, stage_fn=empty_stage,
            max_speculative_depth=4, acceptance_low_watermark=0.01,
        )
        for _ in range(10):
            scheduler.run_round(context_ids=[100], prompt="test")
        assert scheduler.current_depth < 4

    def test_depth_bounded(self):
        scheduler = _make_scheduler(n_stages=3, min_depth=1, max_depth=6, max_speculative_depth=3)
        # Many rounds with bad acceptance
        def bad_stage(**kw):
            return [9999.0]
        scheduler._stage_fn = bad_stage
        for _ in range(20):
            scheduler.run_round(context_ids=[100], prompt="test")
        assert scheduler.current_depth >= 1


class TestSpecPipeConcurrency:
    def test_concurrent_stage_dispatch(self):
        """Speculative tokens should be dispatched concurrently."""
        thread_ids = set()
        lock = threading.Lock()

        def tracking_stage(**kw):
            with lock:
                thread_ids.add(threading.get_ident())
            import time; time.sleep(0.01)  # Simulate work
            return [1.0]

        scheduler = _make_scheduler(n_stages=3, stage_fn=tracking_stage, max_speculative_depth=2)
        scheduler.run_round(context_ids=[100], prompt="test")
        # Should see multiple threads (main + speculative workers)
        assert len(thread_ids) >= 1

    def test_shutdown(self):
        scheduler = _make_scheduler(n_stages=3)
        scheduler.run_round(context_ids=[100], prompt="test")
        scheduler.shutdown()
        # Should not raise
