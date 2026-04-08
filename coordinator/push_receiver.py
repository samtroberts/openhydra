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
