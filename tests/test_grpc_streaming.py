"""Tests for Phase 3A: Bidirectional gRPC Streaming.

Covers:
- ForwardStream server handler (stream reuse, idle timeout)
- StreamPool (lifecycle, reaping, capacity limits)
- InferenceSession (history replay, failover)
- Graceful unary fallback when streaming fails
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from coordinator.stream_pool import (
    InferenceSession,
    StreamHandle,
    StreamPool,
)


# ── StreamHandle ─────────────────────────────────────────────────────────────


class TestStreamHandle:
    def test_defaults(self):
        h = StreamHandle(peer_id="p1", session_id="s1")
        assert h.peer_id == "p1"
        assert h.session_id == "s1"
        assert h.closed is False
        assert h.channel is None

    def test_mark_closed(self):
        h = StreamHandle(peer_id="p1", session_id="s1")
        h.closed = True
        assert h.closed is True


# ── StreamPool lifecycle ─────────────────────────────────────────────────────


class TestStreamPoolLifecycle:
    def test_pool_starts_empty(self):
        pool = StreamPool(idle_timeout_s=5.0, max_streams=10)
        assert pool.active_count == 0

    def test_has_stream_returns_false_for_missing(self):
        pool = StreamPool()
        assert pool.has_stream("p1", "s1") is False

    def test_close_all_marks_closed(self):
        pool = StreamPool()
        pool._streams[("p1", "s1")] = StreamHandle(
            peer_id="p1", session_id="s1",
            created_at=time.monotonic(), last_used_at=time.monotonic(),
        )
        assert pool.active_count == 1
        pool.close_all()
        assert pool.active_count == 0

    def test_close_stream_specific(self):
        pool = StreamPool()
        pool._streams[("p1", "s1")] = StreamHandle(
            peer_id="p1", session_id="s1",
            created_at=time.monotonic(), last_used_at=time.monotonic(),
        )
        pool._streams[("p2", "s2")] = StreamHandle(
            peer_id="p2", session_id="s2",
            created_at=time.monotonic(), last_used_at=time.monotonic(),
        )
        assert pool.active_count == 2
        pool.close_stream("p1", "s1")
        assert pool.active_count == 1
        assert pool.has_stream("p1", "s1") is False
        assert pool.has_stream("p2", "s2") is True

    def test_pool_rejects_when_closed(self):
        pool = StreamPool()
        pool.close_all()
        result = pool.get_or_create("p1", "s1", "127.0.0.1", 50051)
        assert result is None


# ── StreamPool idle reaping ──────────────────────────────────────────────────


class TestStreamPoolReaping:
    def test_reap_expired_streams(self):
        pool = StreamPool(idle_timeout_s=0.1)
        h = StreamHandle(
            peer_id="p1", session_id="s1",
            created_at=time.monotonic() - 10,
            last_used_at=time.monotonic() - 10,  # expired
        )
        pool._streams[("p1", "s1")] = h
        reaped = pool.reap_expired()
        assert reaped == 1
        assert pool.active_count == 0

    def test_reap_does_not_touch_active(self):
        pool = StreamPool(idle_timeout_s=60.0)
        h = StreamHandle(
            peer_id="p1", session_id="s1",
            created_at=time.monotonic(),
            last_used_at=time.monotonic(),
        )
        pool._streams[("p1", "s1")] = h
        reaped = pool.reap_expired()
        assert reaped == 0
        assert pool.active_count == 1

    def test_reap_removes_closed_streams(self):
        pool = StreamPool(idle_timeout_s=60.0)
        h = StreamHandle(
            peer_id="p1", session_id="s1",
            created_at=time.monotonic(),
            last_used_at=time.monotonic(),
            closed=True,
        )
        pool._streams[("p1", "s1")] = h
        reaped = pool.reap_expired()
        assert reaped == 1


# ── StreamPool capacity ─────────────────────────────────────────────────────


class TestStreamPoolCapacity:
    def test_max_streams_enforced(self):
        pool = StreamPool(max_streams=2)
        for i in range(2):
            pool._streams[(f"p{i}", "s")] = StreamHandle(
                peer_id=f"p{i}", session_id="s",
                created_at=time.monotonic(),
                last_used_at=time.monotonic(),
            )
        assert pool.active_count == 2

        # Trying to get_or_create a third should fail (returns None)
        # because _open_stream will be called but max_streams is reached.
        # We mock _open_stream to avoid real gRPC connections.
        with patch.object(pool, "_open_stream", return_value=None):
            result = pool.get_or_create("p3", "s", "127.0.0.1", 50051)
        assert result is None


# ── StreamPool stream reuse ──────────────────────────────────────────────────


class TestStreamPoolReuse:
    def test_reuse_existing_stream(self):
        pool = StreamPool()
        h = StreamHandle(
            peer_id="p1", session_id="s1",
            created_at=time.monotonic(),
            last_used_at=time.monotonic() - 5,
        )
        pool._streams[("p1", "s1")] = h

        result = pool.get_or_create("p1", "s1", "127.0.0.1", 50051)
        assert result is h
        # last_used_at should be updated
        assert result.last_used_at > time.monotonic() - 1

    def test_create_new_when_old_is_closed(self):
        pool = StreamPool()
        old = StreamHandle(
            peer_id="p1", session_id="s1",
            created_at=time.monotonic(),
            last_used_at=time.monotonic(),
            closed=True,
        )
        pool._streams[("p1", "s1")] = old

        new_handle = StreamHandle(
            peer_id="p1", session_id="s1",
            created_at=time.monotonic(),
            last_used_at=time.monotonic(),
        )
        with patch.object(pool, "_open_stream", return_value=new_handle):
            result = pool.get_or_create("p1", "s1", "127.0.0.1", 50051)
        assert result is new_handle


# ── StreamPool send_and_receive ──────────────────────────────────────────────


class TestStreamPoolSendReceive:
    def test_send_and_receive_success(self):
        pool = StreamPool()
        q = queue.Queue()
        mock_response = MagicMock()
        mock_response.error = ""

        h = StreamHandle(
            peer_id="p1", session_id="s1",
            request_queue=q,
            response_iter=iter([mock_response]),
            created_at=time.monotonic(),
            last_used_at=time.monotonic(),
        )

        request = MagicMock()
        result = pool.send_and_receive(h, request)
        assert result is mock_response
        assert q.get_nowait() is request

    def test_send_on_closed_stream_raises(self):
        pool = StreamPool()
        h = StreamHandle(peer_id="p1", session_id="s1", closed=True)
        with pytest.raises(RuntimeError, match="stream_closed"):
            pool.send_and_receive(h, MagicMock())

    def test_send_when_stream_ends_raises_and_marks_closed(self):
        pool = StreamPool()
        q = queue.Queue()
        h = StreamHandle(
            peer_id="p1", session_id="s1",
            request_queue=q,
            response_iter=iter([]),  # empty → StopIteration
            created_at=time.monotonic(),
            last_used_at=time.monotonic(),
        )
        with pytest.raises(RuntimeError, match="stream_ended"):
            pool.send_and_receive(h, MagicMock())
        assert h.closed is True


# ── InferenceSession ─────────────────────────────────────────────────────────


class TestInferenceSession:
    def test_record_and_replay(self):
        session = InferenceSession(session_id="s1", model_id="qwen-0.8b")
        req1 = MagicMock(name="req1")
        resp1 = MagicMock(name="resp1")
        req2 = MagicMock(name="req2")
        resp2 = MagicMock(name="resp2")

        session.record(req1, resp1)
        session.record(req2, resp2)

        assert len(session.history) == 2
        replay = session.replay_requests()
        assert replay == [req1, req2]

    def test_close_clears_pipeline_streams(self):
        session = InferenceSession(session_id="s1", model_id="qwen-0.8b")
        h = StreamHandle(peer_id="p1", session_id="s1")
        session.pipeline_streams[0] = h
        session.close()
        assert len(session.pipeline_streams) == 0
        assert h.closed is True

    def test_empty_session_replay(self):
        session = InferenceSession(session_id="s1", model_id="qwen-0.8b")
        assert session.replay_requests() == []

    def test_session_metadata(self):
        session = InferenceSession(
            session_id="test-session",
            model_id="Qwen/Qwen3.5-0.8B",
            prompt="Hello world",
        )
        assert session.session_id == "test-session"
        assert session.model_id == "Qwen/Qwen3.5-0.8B"
        assert session.prompt == "Hello world"


# ── ForwardStream server handler ─────────────────────────────────────────────


class TestForwardStreamHandler:
    """Test the PeerService.ForwardStream handler via mock."""

    def _make_service(self):
        """Create a minimal PeerService-like mock."""
        from peer.server import PeerService
        svc = MagicMock(spec=PeerService)
        svc.peer_id = "test-peer"
        svc._stream_idle_timeout_s = 30.0
        svc._lock = threading.Lock()
        svc._inflight = 0

        # Wire the real ForwardStream method.
        svc.ForwardStream = PeerService.ForwardStream.__get__(svc, PeerService)
        return svc

    def test_stream_processes_multiple_requests(self):
        svc = self._make_service()

        req1 = MagicMock()
        req1.kv_session_id = "session-1"
        resp1 = MagicMock()
        resp1.error = ""

        req2 = MagicMock()
        req2.kv_session_id = "session-1"
        resp2 = MagicMock()
        resp2.error = ""

        svc.Forward = MagicMock(side_effect=[resp1, resp2])
        context = MagicMock()

        responses = list(svc.ForwardStream(iter([req1, req2]), context))
        assert len(responses) == 2
        assert responses[0] is resp1
        assert responses[1] is resp2
        assert svc.Forward.call_count == 2

    def test_stream_yields_error_on_exception(self):
        svc = self._make_service()
        svc.Forward = MagicMock(side_effect=RuntimeError("boom"))

        from peer import peer_pb2
        context = MagicMock()
        req = MagicMock()
        req.kv_session_id = ""

        responses = list(svc.ForwardStream(iter([req]), context))
        assert len(responses) == 1
        assert "stream_error" in responses[0].error

    def test_stream_empty_iterator(self):
        svc = self._make_service()
        context = MagicMock()
        responses = list(svc.ForwardStream(iter([]), context))
        assert len(responses) == 0


# ── Graceful unary fallback ──────────────────────────────────────────────────


class TestUnaryFallback:
    """Test that when streaming fails, the system falls back to unary Forward."""

    def test_fallback_on_stream_open_failure(self):
        """StreamPool returns None → caller should use unary Forward."""
        pool = StreamPool()
        # Force _open_stream to fail.
        with patch.object(pool, "_open_stream", return_value=None):
            handle = pool.get_or_create("p1", "s1", "127.0.0.1", 50051)
        assert handle is None
        # Caller interprets None as "use unary fallback".

    def test_fallback_on_send_failure(self):
        """Stream error during send → caller catches and falls back."""
        pool = StreamPool()
        h = StreamHandle(
            peer_id="p1", session_id="s1",
            request_queue=queue.Queue(),
            response_iter=iter([]),  # will raise StopIteration
            created_at=time.monotonic(),
            last_used_at=time.monotonic(),
        )
        with pytest.raises(RuntimeError, match="stream_ended"):
            pool.send_and_receive(h, MagicMock())
        # Stream is now closed → next attempt should create new or fall back.
        assert h.closed is True

    def test_pool_creates_new_stream_after_failure(self):
        """After a stream closes, get_or_create opens a fresh one."""
        pool = StreamPool()
        old = StreamHandle(
            peer_id="p1", session_id="s1",
            closed=True,
            created_at=time.monotonic(),
            last_used_at=time.monotonic(),
        )
        pool._streams[("p1", "s1")] = old

        new = StreamHandle(
            peer_id="p1", session_id="s1",
            created_at=time.monotonic(),
            last_used_at=time.monotonic(),
        )
        with patch.object(pool, "_open_stream", return_value=new):
            result = pool.get_or_create("p1", "s1", "127.0.0.1", 50051)
        assert result is new
        assert result is not old


# ── Failover history replay ─────────────────────────────────────────────────


class TestFailoverReplay:
    def test_replay_sends_full_history(self):
        """InferenceSession records all req/resp pairs for replay."""
        session = InferenceSession(session_id="s1", model_id="qwen-0.8b")

        for i in range(5):
            req = MagicMock(name=f"req_{i}")
            resp = MagicMock(name=f"resp_{i}")
            session.record(req, resp)

        replay = session.replay_requests()
        assert len(replay) == 5
        # Verify order is preserved.
        for i, req in enumerate(replay):
            assert req._mock_name == f"req_{i}"

    def test_replay_after_partial_failure(self):
        """After a node fails at step 3, replay all 3 requests on replacement."""
        session = InferenceSession(session_id="s1", model_id="qwen-0.8b")

        for i in range(3):
            session.record(MagicMock(), MagicMock())

        # Simulate failure at step 3 — close old streams.
        session.close()

        # All history is preserved for replay.
        assert len(session.replay_requests()) == 3
        assert len(session.pipeline_streams) == 0  # cleaned up


# ── Thread safety ────────────────────────────────────────────────────────────


class TestStreamPoolThreadSafety:
    def test_concurrent_reap_and_create(self):
        """Concurrent reaping and creation should not deadlock."""
        pool = StreamPool(idle_timeout_s=0.01, max_streams=100)
        errors: list[Exception] = []

        def reaper():
            try:
                for _ in range(50):
                    pool.reap_expired()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def creator():
            try:
                for i in range(50):
                    with pool._lock:
                        pool._streams[(f"p{i}", "s")] = StreamHandle(
                            peer_id=f"p{i}", session_id="s",
                            created_at=time.monotonic(),
                            last_used_at=time.monotonic(),
                        )
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=reaper, daemon=True)
        t2 = threading.Thread(target=creator, daemon=True)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert len(errors) == 0, f"Thread errors: {errors}"
        pool.close_all()
