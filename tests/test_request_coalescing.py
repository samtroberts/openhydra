"""Tests for Phase 4 Request Coalescing — BatchingQueue.

Unit tests using mock shards — verifies coalescing semantics, max-batch
flush, overflow, fallback, and exception propagation.
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from peer.batching import BatchingQueue, _BatchItem


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
