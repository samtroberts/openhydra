# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Push-mode result receiver — Petals parity Phase A.

When using server-to-server push mode, the coordinator sends the initial
request to the first peer and waits for the last peer to deliver the
final result via the ``PushResult`` gRPC RPC.  This module provides a
thread-safe dict of ``request_id → Future`` that the ``PushResult``
handler in ``peer/server.py`` resolves.

Usage::

    from coordinator.push_receiver import register_push, await_push

    future = register_push(request_id)
    # ... send ForwardRequest with push_mode=True to first peer ...
    response = await_push(request_id, timeout_s=60.0)
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future
from typing import Any

logger = logging.getLogger(__name__)

# Global registry: request_id → Future[ForwardResponse]
# Accessed by PeerService.PushResult() in peer/server.py
_PUSH_RESULTS: dict[str, Future] = {}
_PUSH_LOCK = threading.Lock()


def register_push(request_id: str) -> Future:
    """Register a pending push-mode request.

    Returns a Future that resolves when the last peer in the chain
    delivers the final activation via PushResult.
    """
    future: Future = Future()
    with _PUSH_LOCK:
        _PUSH_RESULTS[request_id] = future
    return future


def await_push(request_id: str, timeout_s: float = 60.0) -> Any:
    """Wait for the push result to arrive.

    Args:
        request_id: The correlation ID matching the ForwardRequest.
        timeout_s: Max seconds to wait.

    Returns:
        The ForwardResponse from the last peer.

    Raises:
        TimeoutError: If the result doesn't arrive within timeout_s.
    """
    with _PUSH_LOCK:
        future = _PUSH_RESULTS.get(request_id)
    if future is None:
        raise RuntimeError(f"push_not_registered: {request_id}")
    try:
        result = future.result(timeout=timeout_s)
        return result
    except Exception:
        # Clean up on failure
        with _PUSH_LOCK:
            _PUSH_RESULTS.pop(request_id, None)
        raise
    finally:
        with _PUSH_LOCK:
            _PUSH_RESULTS.pop(request_id, None)


def cancel_push(request_id: str) -> None:
    """Cancel a pending push request."""
    with _PUSH_LOCK:
        future = _PUSH_RESULTS.pop(request_id, None)
    if future is not None and not future.done():
        future.cancel()


# ── Ring autoregressive token queue ─────────────────────────────────
#
# The ring topology lets tokens circulate peer-to-peer. The last shard
# (same process as coordinator) drops each sampled token into a
# thread-safe queue.Queue. The coordinator's HTTP handler thread reads
# from it with a blocking get(timeout=...) — safe because each request
# runs in its own thread (ThreadingMixIn).

import queue as _queue_mod

_RING_QUEUES: dict[str, _queue_mod.Queue] = {}
_RING_LOCK = threading.Lock()


def register_ring(request_id: str) -> _queue_mod.Queue:
    """Register a ring token queue for real-time streaming.

    Returns a thread-safe Queue that yields token IDs (int) or None (sentinel).
    """
    q: _queue_mod.Queue = _queue_mod.Queue()
    with _RING_LOCK:
        _RING_QUEUES[request_id] = q
    return q


def emit_ring_token(request_id: str, token_id: int | None) -> None:
    """Emit a token from the gRPC/proxy thread into the ring queue.

    Thread-safe: queue.Queue.put_nowait() is safe from any thread.
    ``None`` is the sentinel: generation complete (EOS or max tokens).
    """
    with _RING_LOCK:
        q = _RING_QUEUES.get(request_id)
    if q is not None:
        q.put_nowait(token_id)


def unregister_ring(request_id: str) -> None:
    """Clean up the ring queue after generation completes."""
    with _RING_LOCK:
        _RING_QUEUES.pop(request_id, None)


# ── Phase 2b live-bench Binding #2: block-verify response queues ────────
#
# Separate from the per-token ring queues above. Each block-verify
# round trip registers a queue keyed on (request_id, block_index),
# the multi-peer transport waits on it, and the coord PushResult
# handler routes ``is_hidden_state=True`` + ``block_size > 0``
# responses here. Block-verify responses are payloads, not tokens —
# the queue carries either bytes (the activation_packed wire form)
# or an exception sentinel for transport-side errors.

_DFLASH_BLOCK_QUEUES: dict[tuple[str, int], _queue_mod.Queue] = {}
_DFLASH_BLOCK_LOCK = threading.Lock()


def register_dflash_block(
    request_id: str, block_index: int,
) -> _queue_mod.Queue:
    """Register a queue for a single block-verify round trip.

    The queue receives EXACTLY ONE message: either a tuple
    ``("ok", activation_packed_bytes, block_size)`` on success,
    or ``("err", exception_message)`` on transport failure. The
    transport's verify() blocks on a single get(); whichever side
    of the tuple arrives, it returns / raises accordingly.

    Idempotent: a second register on the same (req, idx) replaces
    the queue. Useful when retrying after timeout — the new wait
    starts from a fresh queue.
    """
    q: _queue_mod.Queue = _queue_mod.Queue()
    key = (str(request_id), int(block_index))
    with _DFLASH_BLOCK_LOCK:
        _DFLASH_BLOCK_QUEUES[key] = q
    return q


def emit_dflash_block_response(
    request_id: str,
    block_index: int,
    activation_packed: bytes,
    block_size: int,
) -> bool:
    """Deliver a block-verify response to the waiting transport.

    Returns ``True`` if a queue was registered for the (req, idx);
    ``False`` if no transport is waiting (late arrival after a
    timeout, or a stray response). Late responses are dropped
    silently rather than queued indefinitely.
    """
    key = (str(request_id), int(block_index))
    with _DFLASH_BLOCK_LOCK:
        q = _DFLASH_BLOCK_QUEUES.get(key)
    if q is None:
        return False
    q.put_nowait(("ok", bytes(activation_packed or b""), int(block_size)))
    return True


def emit_dflash_block_error(
    request_id: str,
    block_index: int,
    error: str,
) -> bool:
    """Surface a transport failure to the waiting transport. Same
    return semantics as ``emit_dflash_block_response``."""
    key = (str(request_id), int(block_index))
    with _DFLASH_BLOCK_LOCK:
        q = _DFLASH_BLOCK_QUEUES.get(key)
    if q is None:
        return False
    q.put_nowait(("err", str(error or "unknown")))
    return True


def unregister_dflash_block(
    request_id: str, block_index: int,
) -> None:
    """Drop the queue registration. Idempotent. Always called by
    the transport in a finally clause so a thrown exception during
    verify() doesn't leak queue state across blocks."""
    key = (str(request_id), int(block_index))
    with _DFLASH_BLOCK_LOCK:
        _DFLASH_BLOCK_QUEUES.pop(key, None)


def unregister_dflash_session(request_id: str) -> int:
    """Drop ALL queues for ``request_id`` regardless of block_index.
    Called at end-of-generation cleanup. Returns the count of
    queues dropped."""
    with _DFLASH_BLOCK_LOCK:
        keys = [k for k in _DFLASH_BLOCK_QUEUES if k[0] == str(request_id)]
        for k in keys:
            _DFLASH_BLOCK_QUEUES.pop(k, None)
    return len(keys)
