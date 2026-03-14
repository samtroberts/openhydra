"""Request coalescing for concurrent gRPC ForwardRequests.

BatchingQueue collects concurrent forward() calls into a single batch and
dispatches them together via shard.forward_batch(), allowing the GPU to
process multiple requests in one kernel launch instead of sequentially.

Thread-safety contract:
  - The internal lock is held only for the O(1) list-swap, never during
    the actual forward pass.  This prevents contention between the batch
    timer thread and gRPC worker threads.
  - gRPC worker threads block on item.future.result() — this is safe
    because gRPC uses a ThreadPoolExecutor server and blocking is expected.
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class _BatchItem:
    """Holds all parameters for a single forward() call plus a Future for the result."""

    prompt: str
    activation: list[float]
    max_tokens: int
    stage_index: int = 0
    total_stages: int = 1
    request_id: str | None = None
    decode_do_sample: bool | None = None
    decode_temperature: float | None = None
    decode_top_p: float | None = None
    decode_top_k: int | None = None
    decode_seed: int | None = None
    future: concurrent.futures.Future = field(default_factory=concurrent.futures.Future)


class BatchingQueue:
    """Coalesces concurrent forward() calls into a single forward_batch() call.

    Each gRPC worker thread calls forward() which submits a _BatchItem and
    blocks on item.future.result().  A timer fires after batch_window_ms and
    flushes all pending items via shard.forward_batch(batch).  If max_batch_size
    items accumulate before the timer fires, the batch is flushed immediately.

    Args:
        shard: ModelShard (or any object with forward_batch / forward methods).
        batch_window_ms: milliseconds to wait for additional requests before
            flushing (default: 50.0).
        max_batch_size: maximum items per batch; immediate flush when reached
            (default: 8).
    """

    def __init__(
        self,
        shard: Any,
        batch_window_ms: float = 50.0,
        max_batch_size: int = 8,
    ) -> None:
        self._shard = shard
        self._window_s = max(0.0, float(batch_window_ms)) / 1000.0
        self._max_batch = max(1, int(max_batch_size))
        self._lock = threading.Lock()
        self._pending: list[_BatchItem] = []
        self._flush_timer: threading.Timer | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def forward(
        self,
        prompt: str,
        activation: list[float],
        max_tokens: int,
        stage_index: int = 0,
        total_stages: int = 1,
        request_id: str | None = None,
        decode_do_sample: bool | None = None,
        decode_temperature: float | None = None,
        decode_top_p: float | None = None,
        decode_top_k: int | None = None,
        decode_seed: int | None = None,
    ) -> list[float]:
        """Submit a forward request; blocks until the batch result is ready.

        This method is safe to call from multiple gRPC worker threads
        simultaneously.  All callers within the same batch window will
        block together and be resolved by the same forward_batch() call.
        """
        item = _BatchItem(
            prompt=prompt,
            activation=list(activation),
            max_tokens=max_tokens,
            stage_index=stage_index,
            total_stages=total_stages,
            request_id=request_id,
            decode_do_sample=decode_do_sample,
            decode_temperature=decode_temperature,
            decode_top_p=decode_top_p,
            decode_top_k=decode_top_k,
            decode_seed=decode_seed,
        )

        with self._lock:
            self._pending.append(item)
            n = len(self._pending)
            if n == 1:
                # First item in window — start the timer.
                t = threading.Timer(self._window_s, self._flush)
                t.daemon = True
                self._flush_timer = t
                t.start()
            elif n >= self._max_batch:
                # Full batch — cancel the timer and flush immediately.
                if self._flush_timer is not None:
                    self._flush_timer.cancel()
                    self._flush_timer = None
                flush_thread = threading.Thread(target=self._flush, daemon=True)
                flush_thread.start()

        # Block until the batch fires and our future is resolved.
        return list(item.future.result())

    # ── Internal ──────────────────────────────────────────────────────────────

    def _flush(self) -> None:
        """Dequeue up to max_batch_size items and dispatch them.

        Called either by the timer thread or a dedicated daemon thread
        (when max_batch_size is reached).  Safe to call concurrently — each
        invocation takes its own slice of pending items under the lock.
        """
        with self._lock:
            if not self._pending:
                self._flush_timer = None
                return
            batch = self._pending[: self._max_batch]
            self._pending = self._pending[self._max_batch :]
            self._flush_timer = None
            # If there are still items (overflow), kick off a new immediate flush.
            if self._pending:
                t = threading.Timer(0.0, self._flush)
                t.daemon = True
                self._flush_timer = t
                t.start()

        # Dispatch outside the lock so forward() calls can enqueue while we work.
        try:
            results = self._dispatch(batch)
            for item, result in zip(batch, results):
                if not item.future.done():
                    item.future.set_result(list(result))
        except Exception as exc:
            logger.error("batching_queue_flush_error: %s", exc, exc_info=True)
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(exc)

    def _dispatch(self, batch: list[_BatchItem]) -> list[list[float]]:
        """Route a batch to shard.forward_batch() or a sequential fallback.

        The sequential fallback is only used when the shard does not implement
        forward_batch() (e.g. a legacy runtime).  All production runtimes
        (ToyRuntime, PyTorchRuntime, MLXRuntime) implement forward_batch().
        """
        fb = getattr(self._shard, "forward_batch", None)
        if callable(fb):
            return [list(r) for r in fb(batch)]

        # Graceful fallback: sequential per-item forward() for legacy runtimes.
        logger.warning(
            "batching_queue: shard has no forward_batch(); "
            "falling back to sequential forward() for batch of %d",
            len(batch),
        )
        results: list[list[float]] = []
        for item in batch:
            result = self._shard.forward(
                item.prompt,
                item.activation,
                item.max_tokens,
                stage_index=item.stage_index,
                total_stages=item.total_stages,
                request_id=item.request_id,
                decode_do_sample=item.decode_do_sample,
                decode_temperature=item.decode_temperature,
                decode_top_p=item.decode_top_p,
                decode_top_k=item.decode_top_k,
                decode_seed=item.decode_seed,
            )
            results.append(list(result))
        return results
