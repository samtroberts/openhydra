# Copyright 2026 OpenHydra contributors — Apache 2.0

"""TDD tests for coordinator-level continuous batching (P2-C).

Tests that multiple concurrent API requests are coalesced into
batched pipeline passes, amortizing gRPC overhead.

Run:  pytest tests/test_coordinator_batching.py -v
"""

from __future__ import annotations

import threading
import time

import pytest


def _make_queue(batch_window_ms=100, max_batch=4):
    from coordinator.request_batcher import CoordinatorBatchQueue
    return CoordinatorBatchQueue(
        batch_window_ms=batch_window_ms,
        max_batch_size=max_batch,
    )


class TestBasicBatching:
    def test_single_request_returns_result(self):
        queue = _make_queue(batch_window_ms=50)
        result = queue.submit("prompt1", max_tokens=16, model_id="test")
        assert result is not None
        assert "prompt" in result
        assert result["prompt"] == "prompt1"

    def test_batch_collects_concurrent_requests(self):
        queue = _make_queue(batch_window_ms=200, max_batch=4)
        results = []
        errors = []

        def _submit(prompt):
            try:
                r = queue.submit(prompt, max_tokens=16, model_id="test")
                results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_submit, args=(f"p{i}",)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(errors) == 0
        assert len(results) == 3

    def test_max_batch_triggers_immediate_flush(self):
        queue = _make_queue(batch_window_ms=5000, max_batch=2)
        results = []

        def _submit(prompt):
            results.append(queue.submit(prompt, max_tokens=8, model_id="test"))

        # Submit 2 requests — should trigger immediately at max_batch
        t1 = threading.Thread(target=_submit, args=("a",))
        t2 = threading.Thread(target=_submit, args=("b",))
        t1.start()
        t2.start()
        t1.join(timeout=3)
        t2.join(timeout=3)

        assert len(results) == 2


class TestBatchMetadata:
    def test_batch_id_assigned(self):
        queue = _make_queue()
        result = queue.submit("hello", max_tokens=8, model_id="test")
        assert "batch_id" in result

    def test_batch_size_reported(self):
        queue = _make_queue()
        result = queue.submit("hello", max_tokens=8, model_id="test")
        assert "batch_size" in result
        assert result["batch_size"] >= 1
