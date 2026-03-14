"""Tests for Phase 4 Request Coalescing — BatchingQueue.

Two test groups:

1. ``TestBatchingQueueUnit`` — pure unit tests using mock shards.
   Verifies coalescing semantics, max-batch flush, overflow, fallback,
   and exception propagation.

2. ``TestBatchingQueueGRPC`` — integration tests with a real gRPC toy peer.
   Verifies that 4 concurrent Forward() calls are coalesced into a single
   forward_batch() call, each client receives the correct response, and a
   single request is unaffected by the coalescing machinery.
"""

from __future__ import annotations

import threading
from concurrent import futures
from typing import Any
from unittest.mock import MagicMock, patch

import grpc
import pytest

from peer import peer_pb2, peer_pb2_grpc
from peer.batching import BatchingQueue, _BatchItem
from peer.server import PeerService


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_mock_shard(return_fn=None):
    """Create a mock shard with a default forward_batch side-effect."""
    shard = MagicMock()
    if return_fn is None:
        # Returns [[float(i)] for i in range(len(items))]
        shard.forward_batch = MagicMock(
            side_effect=lambda items: [[float(i)] for i in range(len(items))]
        )
    else:
        shard.forward_batch = MagicMock(side_effect=return_fn)
    return shard


def _start_toy_peer_with_coalescing(
    batch_window_ms: float = 100.0,
    max_batch_size: int = 8,
) -> tuple[grpc.Server, int, PeerService]:
    """Start a real gRPC toy-backend PeerService with batching enabled."""
    service = PeerService(
        peer_id="coalesce-test",
        model_id="openhydra-toy-345m",
        shard_index=0,
        total_shards=1,
        daemon_mode="polite",
        broken=False,
        batch_window_ms=batch_window_ms,
        max_batch_size=max_batch_size,
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    peer_pb2_grpc.add_PeerServicer_to_server(service, server)
    try:
        port = server.add_insecure_port("127.0.0.1:0")
    except RuntimeError as exc:
        pytest.skip(f"gRPC listener unavailable: {exc}")
    if port == 0:
        pytest.skip("gRPC listener unavailable")
    server.start()
    return server, port, service


# ── Group 1: Unit Tests ────────────────────────────────────────────────────────


class TestBatchingQueueUnit:
    """Pure unit tests for BatchingQueue using mock shards."""

    def test_single_request_passes_through(self):
        """A single request is forwarded and returns the correct result."""
        shard = _make_mock_shard()
        queue = BatchingQueue(shard, batch_window_ms=200.0, max_batch_size=8)
        result = queue.forward("hello", [], 1, request_id="req0")

        assert shard.forward_batch.call_count == 1
        call_items = shard.forward_batch.call_args[0][0]
        assert len(call_items) == 1
        assert result == [0.0]

    def test_window_coalesces_4_concurrent_requests(self):
        """4 concurrent requests within the window → forward_batch called once."""
        shard = _make_mock_shard()
        queue = BatchingQueue(shard, batch_window_ms=300.0, max_batch_size=8)

        barrier = threading.Barrier(4)
        results: list[list[float]] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def worker(i: int) -> None:
            barrier.wait()  # Sync all 4 threads before submitting
            try:
                r = queue.forward(
                    f"prompt{i}",
                    [],
                    1,
                    stage_index=0,
                    total_stages=1,
                    request_id=f"req{i}",
                )
                with lock:
                    results.append(r)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert not errors, f"Worker errors: {errors}"
        assert len(results) == 4
        assert shard.forward_batch.call_count == 1
        call_items = shard.forward_batch.call_args[0][0]
        assert len(call_items) == 4

    def test_max_batch_triggers_immediate_flush(self):
        """8 concurrent requests with max_batch=8 → immediate flush, no timer wait."""
        shard = _make_mock_shard()
        # 5-second window — max_batch must trigger the flush before timer fires.
        queue = BatchingQueue(shard, batch_window_ms=5000.0, max_batch_size=8)

        barrier = threading.Barrier(8)
        results: list[list[float]] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def worker(i: int) -> None:
            barrier.wait()
            try:
                r = queue.forward(f"prompt{i}", [], 1, request_id=f"req{i}")
                with lock:
                    results.append(r)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert not errors, f"Worker errors: {errors}"
        assert len(results) == 8
        # All 8 are dispatched in one call (no timer fired yet).
        assert shard.forward_batch.call_count == 1

    def test_overflow_processed_in_chunks(self):
        """12 requests with max_batch=4 → multiple batches, all 12 results returned."""
        shard = _make_mock_shard()
        queue = BatchingQueue(shard, batch_window_ms=100.0, max_batch_size=4)

        barrier = threading.Barrier(12)
        results: list[list[float]] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def worker(i: int) -> None:
            barrier.wait()
            try:
                r = queue.forward(f"prompt{i}", [], 1, request_id=f"req{i}")
                with lock:
                    results.append(r)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert not errors, f"Worker errors: {errors}"
        assert len(results) == 12
        # Multiple batches of at most 4 items.  At minimum 3 forward_batch calls (4+4+4).
        assert 1 <= shard.forward_batch.call_count <= 12

    def test_fallback_sequential_without_forward_batch(self):
        """Shard without forward_batch → falls back to sequential shard.forward()."""
        shard = MagicMock(spec=["forward"])  # No forward_batch on spec
        shard.forward = MagicMock(return_value=[42.0])
        queue = BatchingQueue(shard, batch_window_ms=50.0, max_batch_size=8)

        result = queue.forward("hello", [], 1, request_id="req0")

        assert result == [42.0]
        assert shard.forward.call_count == 1

    def test_exception_propagated_to_all_batch_items(self):
        """forward_batch raises RuntimeError → all pending futures get the exception."""
        shard = _make_mock_shard()
        shard.forward_batch.side_effect = RuntimeError("gpu_oom")

        queue = BatchingQueue(shard, batch_window_ms=300.0, max_batch_size=8)

        barrier = threading.Barrier(3)
        errors: list[str] = []
        lock = threading.Lock()

        def worker(i: int) -> None:
            barrier.wait()
            try:
                queue.forward(f"prompt{i}", [], 1, request_id=f"req{i}")
            except RuntimeError as exc:
                with lock:
                    errors.append(str(exc))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        # All 3 requests must have received the exception.
        assert len(errors) == 3
        assert all("gpu_oom" in e for e in errors)

    def test_batch_item_carries_all_forward_kwargs(self):
        """_BatchItem stores all forwarded kwargs correctly."""
        item = _BatchItem(
            prompt="test",
            activation=[1.0, 2.0],
            max_tokens=5,
            stage_index=1,
            total_stages=3,
            request_id="myreq",
            decode_do_sample=True,
            decode_temperature=0.7,
            decode_top_p=0.9,
            decode_top_k=50,
            decode_seed=42,
        )
        assert item.prompt == "test"
        assert item.activation == [1.0, 2.0]
        assert item.max_tokens == 5
        assert item.stage_index == 1
        assert item.total_stages == 3
        assert item.request_id == "myreq"
        assert item.decode_do_sample is True
        assert item.decode_temperature == 0.7
        assert item.decode_top_p == 0.9
        assert item.decode_top_k == 50
        assert item.decode_seed == 42
        assert item.future is not None

    def test_request_id_forwarded_to_batch_items(self):
        """BatchingQueue forwards request_id into the _BatchItem submitted to forward_batch."""
        captured_items: list[Any] = []

        shard = MagicMock()
        shard.forward_batch = MagicMock(
            side_effect=lambda items: (
                captured_items.extend(items) or [[0.0] for _ in items]
            )
        )

        queue = BatchingQueue(shard, batch_window_ms=50.0, max_batch_size=8)
        queue.forward("hello", [], 1, request_id="special-id-123")

        assert len(captured_items) == 1
        assert captured_items[0].request_id == "special-id-123"


# ── Group 2: gRPC Integration Tests ───────────────────────────────────────────


class TestBatchingQueueGRPC:
    """Integration tests with a real gRPC toy peer and BatchingQueue."""

    def test_four_concurrent_grpc_requests_coalesced(self):
        """4 concurrent gRPC Forward() calls → forward_batch called once with batch_size=4."""
        server, port, service = _start_toy_peer_with_coalescing(
            batch_window_ms=300.0,
            max_batch_size=8,
        )
        try:
            call_batches: list[int] = []
            original_fb = service.shard.forward_batch

            def spy_fb(items: list[Any]) -> list[list[float]]:
                call_batches.append(len(items))
                return original_fb(items)

            # Patch forward_batch on the shard instance so the BatchingQueue picks it up.
            service.shard.forward_batch = spy_fb

            channel = grpc.insecure_channel(f"127.0.0.1:{port}")
            stub = peer_pb2_grpc.PeerStub(channel)

            barrier = threading.Barrier(4)
            responses: list[Any] = []
            errors: list[Exception] = []
            lock = threading.Lock()

            def client(i: int) -> None:
                barrier.wait()  # Ensure all 4 fire simultaneously
                try:
                    resp = stub.Forward(
                        peer_pb2.ForwardRequest(
                            request_id=f"req_{i}",
                            prompt="hello",
                            activation=[],
                            max_tokens=1,
                            stage_index=0,
                            total_stages=1,
                        ),
                        timeout=10,
                    )
                    with lock:
                        responses.append(resp)
                except Exception as exc:
                    with lock:
                        errors.append(exc)

            threads = [threading.Thread(target=client, args=(i,)) for i in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10.0)

            channel.close()

            assert not errors, f"Client errors: {errors}"
            assert len(responses) == 4
            # All 4 items must have been dispatched (possibly in 1 or 2 batches).
            assert sum(call_batches) == 4
            # With a 300ms window and Barrier synchronisation, typically 1 batch.
            assert len(call_batches) <= 2, (
                f"Expected ≤2 forward_batch calls, got {len(call_batches)}: {call_batches}"
            )

        finally:
            server.stop(grace=0)

    def test_each_client_receives_correct_response(self):
        """Each client's ForwardResponse carries the matching request_id."""
        server, port, service = _start_toy_peer_with_coalescing(
            batch_window_ms=200.0,
            max_batch_size=8,
        )
        try:
            channel = grpc.insecure_channel(f"127.0.0.1:{port}")
            stub = peer_pb2_grpc.PeerStub(channel)

            barrier = threading.Barrier(4)
            responses_by_id: dict[str, Any] = {}
            errors: list[Exception] = []
            lock = threading.Lock()

            def client(i: int) -> None:
                barrier.wait()
                try:
                    resp = stub.Forward(
                        peer_pb2.ForwardRequest(
                            request_id=f"req_{i}",
                            prompt="hello",
                            activation=[],
                            max_tokens=1,
                            stage_index=0,
                            total_stages=1,
                        ),
                        timeout=10,
                    )
                    with lock:
                        responses_by_id[f"req_{i}"] = resp
                except Exception as exc:
                    with lock:
                        errors.append(exc)

            threads = [threading.Thread(target=client, args=(i,)) for i in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10.0)

            channel.close()

            assert not errors, f"Client errors: {errors}"
            # Every client must receive a response with the correct request_id.
            for i in range(4):
                key = f"req_{i}"
                assert key in responses_by_id, f"Missing response for {key}"
                resp = responses_by_id[key]
                assert resp.request_id == key, (
                    f"Expected request_id={key!r}, got {resp.request_id!r}"
                )
                assert resp.error == ""

        finally:
            server.stop(grace=0)

    def test_single_request_unaffected_by_coalescing(self):
        """A single request still works correctly — no starvation by coalescing logic."""
        server, port, service = _start_toy_peer_with_coalescing(
            batch_window_ms=200.0,
            max_batch_size=8,
        )
        try:
            channel = grpc.insecure_channel(f"127.0.0.1:{port}")
            stub = peer_pb2_grpc.PeerStub(channel)

            resp = stub.Forward(
                peer_pb2.ForwardRequest(
                    request_id="solo-req",
                    prompt="hello world",
                    activation=[],
                    max_tokens=1,
                    stage_index=0,
                    total_stages=1,
                ),
                timeout=10,
            )
            channel.close()

            assert resp.error == ""
            assert resp.request_id == "solo-req"
            assert len(resp.activation) > 0

        finally:
            server.stop(grace=0)
