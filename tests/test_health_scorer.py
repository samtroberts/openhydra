from __future__ import annotations

import threading
import time

from coordinator.health_scorer import HealthScorer


def test_health_scorer_penalizes_failures_and_persists(tmp_path):
    path = tmp_path / "health.json"
    scorer = HealthScorer(str(path), flush_interval_s=3600.0)
    reloaded: HealthScorer | None = None
    try:
        scorer.record_ping("peer-a", healthy=True, latency_ms=20.0)
        scorer.record_inference("peer-a", success=True, latency_ms=25.0)
        good = scorer.score("peer-a")

        scorer.record_ping("peer-a", healthy=False, latency_ms=200.0)
        scorer.record_inference("peer-a", success=False)
        bad = scorer.score("peer-a")

        assert bad < good
        assert scorer.flush() is True

        reloaded = HealthScorer(str(path), flush_interval_s=3600.0)
        assert reloaded.score("peer-a") == bad
    finally:
        scorer.close()
        if reloaded is not None:
            reloaded.close()


def test_health_scorer_verification_feedback_affects_score(tmp_path):
    path = tmp_path / "health.json"
    scorer = HealthScorer(str(path), flush_interval_s=3600.0)
    try:
        scorer.record_inference("peer-a", success=True, latency_ms=15.0)
        baseline = scorer.score("peer-a")

        scorer.record_verification("peer-a", success=False)
        after_fail = scorer.score("peer-a")
        assert after_fail < baseline

        scorer.record_verification("peer-a", success=True)
        after_recover = scorer.score("peer-a")
        assert after_recover > after_fail
    finally:
        scorer.close()


def test_health_scorer_unknown_peer_score_has_no_side_effects(tmp_path, monkeypatch):
    path = tmp_path / "health.json"
    scorer = HealthScorer(str(path), flush_interval_s=3600.0)
    try:
        scorer.record_ping("peer-a", healthy=True, latency_ms=12.0)
        scorer.flush()
        before_snapshot = scorer.snapshot()

        write_count = 0
        original_write_text = path.__class__.write_text

        def counting_write_text(self, *args, **kwargs):
            nonlocal write_count
            if self == path:
                write_count += 1
            return original_write_text(self, *args, **kwargs)

        monkeypatch.setattr(path.__class__, "write_text", counting_write_text)
        _ = scorer.score("peer-unknown")
        _ = scorer.scores(["peer-unknown-2", "peer-a"])

        assert write_count == 0
        assert scorer.snapshot() == before_snapshot
        assert "peer-unknown" not in scorer.snapshot()
        assert "peer-unknown-2" not in scorer.snapshot()
    finally:
        scorer.close()


def test_health_scorer_batches_record_writes_under_flush_cycles(tmp_path, monkeypatch):
    path = tmp_path / "health.json"
    tick_events = [threading.Event(), threading.Event()]
    wait_calls = 0
    write_count = 0

    original_wait = HealthScorer._wait_for_flush_tick
    original_write_text = path.__class__.write_text

    def controlled_wait(self: HealthScorer) -> bool:
        nonlocal wait_calls
        if wait_calls < 2:
            event = tick_events[wait_calls]
            wait_calls += 1
            event.wait(timeout=2.0)
            return False
        return True

    def counting_write_text(self, *args, **kwargs):
        nonlocal write_count
        if self == path:
            write_count += 1
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(HealthScorer, "_wait_for_flush_tick", controlled_wait)
    monkeypatch.setattr(path.__class__, "write_text", counting_write_text)
    scorer = HealthScorer(str(path), flush_interval_s=5.0)
    try:
        for _ in range(500):
            scorer.record_ping("peer-a", healthy=True, latency_ms=10.0)
        tick_events[0].set()
        time.sleep(0.05)

        for _ in range(500):
            scorer.record_ping("peer-a", healthy=True, latency_ms=10.0)
        tick_events[1].set()
        time.sleep(0.05)
    finally:
        scorer.close()
        monkeypatch.setattr(HealthScorer, "_wait_for_flush_tick", original_wait)

    assert write_count <= 2


def test_health_scorer_flush_and_close(tmp_path):
    path = tmp_path / "health.json"
    scorer = HealthScorer(str(path), flush_interval_s=3600.0)
    scorer.record_ping("peer-a", healthy=True, latency_ms=7.0)

    assert path.exists() is False
    assert scorer.flush() is True
    assert path.exists() is True
    assert scorer.flush() is False

    scorer.close()
    assert scorer._flush_thread.is_alive() is False
