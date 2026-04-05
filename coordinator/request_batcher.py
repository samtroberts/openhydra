# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Coordinator-level continuous batching (P2-C).

Coalesces concurrent HTTP API requests into batched pipeline passes.
When N clients send requests within a batching window, the coordinator
groups them and amortizes the gRPC pipeline overhead.

Thread-safety: uses the same pattern as peer/batching.py — the lock
is held only for O(1) list operations, never during inference.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _BatchItem:
    """A single request waiting to be batched."""
    prompt: str
    max_tokens: int
    model_id: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    future: concurrent.futures.Future = field(
        default_factory=lambda: concurrent.futures.Future()
    )


class CoordinatorBatchQueue:
    """Collects concurrent API requests and flushes them as a batch.

    Each ``submit()`` call blocks until the batch is flushed (either by
    the timer or by reaching max_batch_size).  The flush callback
    processes all items in the batch together.

    For v1, the "batch processing" simply returns metadata about the
    batch — actual inference batching will be wired in when the
    coordinator's ``infer()`` supports multi-prompt batching.

    Args:
        batch_window_ms: Maximum time to wait for more requests before
            flushing the current batch.
        max_batch_size: Flush immediately when this many requests
            accumulate.
    """

    def __init__(
        self,
        batch_window_ms: float = 50.0,
        max_batch_size: int = 8,
    ) -> None:
        self._window_s = max(0.001, batch_window_ms / 1000.0)
        self._max_batch = max(1, max_batch_size)
        self._pending: list[_BatchItem] = []
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    def submit(
        self,
        prompt: str,
        max_tokens: int = 64,
        model_id: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Submit a request and block until the batch is processed.

        Returns:
            Dict with batch metadata and the original request params.
        """
        item = _BatchItem(
            prompt=prompt,
            max_tokens=max_tokens,
            model_id=model_id,
            kwargs=kwargs,
        )

        with self._lock:
            self._pending.append(item)
            pending_count = len(self._pending)

            if pending_count >= self._max_batch:
                # Max batch reached — flush immediately
                if self._timer is not None:
                    self._timer.cancel()
                    self._timer = None
                batch = list(self._pending)
                self._pending = []
            elif self._timer is None:
                # First item in a new batch — start the timer
                self._timer = threading.Timer(self._window_s, self._timer_flush)
                self._timer.daemon = True
                self._timer.start()
                batch = None
            else:
                batch = None

        if batch is not None:
            self._process_batch(batch)

        # Block until this item's future is resolved
        return item.future.result(timeout=30.0)

    def _timer_flush(self) -> None:
        """Called by the timer thread when the batch window expires."""
        with self._lock:
            batch = list(self._pending)
            self._pending = []
            self._timer = None

        if batch:
            self._process_batch(batch)

    def _process_batch(self, batch: list[_BatchItem]) -> None:
        """Process a batch of items and resolve their futures.

        For v1, returns batch metadata. In v2, this will call
        the coordinator's batched inference endpoint.
        """
        batch_id = str(uuid.uuid4())[:8]
        batch_size = len(batch)

        for item in batch:
            result = {
                "prompt": item.prompt,
                "max_tokens": item.max_tokens,
                "model_id": item.model_id,
                "batch_id": batch_id,
                "batch_size": batch_size,
            }
            if not item.future.done():
                item.future.set_result(result)
