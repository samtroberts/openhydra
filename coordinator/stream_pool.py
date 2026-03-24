# Copyright 2026 OpenHydra contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Persistent gRPC streaming connection pool — Phase 3A.

Manages bidirectional ``ForwardStream`` connections to peers, reusing
open streams for subsequent tokens in the same inference session.

Architecture
------------
- ``StreamPool`` maintains a dict of ``(peer_id, session_id)`` →
  ``StreamHandle`` entries.
- Each ``StreamHandle`` wraps a ``grpc.StreamStreamMultiCallable`` with
  an ``asyncio.Queue``-based request/response bridge.
- When a stream drops or the peer doesn't support ``ForwardStream``,
  the pool transparently falls back to unary ``Forward()`` calls.
- Idle streams are reaped after ``idle_timeout_s`` (default 30s).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

logger = logging.getLogger(__name__)


@dataclass
class StreamHandle:
    """A managed bidirectional stream to a single peer.

    Attributes:
        peer_id: Remote peer identifier.
        session_id: Inference session this stream serves.
        channel: The gRPC channel (kept alive while the stream is open).
        stub: The gRPC stub for the ``Peer`` service.
        request_queue: Thread-safe queue for outgoing requests.
        response_iter: Iterator of incoming responses.
        created_at: Monotonic timestamp when the stream was opened.
        last_used_at: Monotonic timestamp of the last request/response.
        closed: Whether this stream has been closed.
    """

    peer_id: str
    session_id: str
    channel: Any = None
    stub: Any = None
    request_queue: Any = None
    response_iter: Any = None
    created_at: float = 0.0
    last_used_at: float = 0.0
    closed: bool = False


class StreamPool:
    """Pool of persistent bidirectional gRPC streams to peers.

    Args:
        idle_timeout_s: Seconds of inactivity before a stream is reaped.
        max_streams: Maximum number of concurrent streams.
    """

    def __init__(
        self,
        idle_timeout_s: float = 30.0,
        max_streams: int = 256,
    ) -> None:
        self.idle_timeout_s = max(1.0, float(idle_timeout_s))
        self.max_streams = max(1, int(max_streams))
        self._streams: dict[tuple[str, str], StreamHandle] = {}
        self._lock = threading.Lock()
        self._closed = False

    def get_or_create(
        self,
        peer_id: str,
        session_id: str,
        host: str,
        port: int,
    ) -> StreamHandle | None:
        """Get an existing stream or create a new one.

        Args:
            peer_id: Target peer identifier.
            session_id: Inference session identifier.
            host: Peer gRPC host.
            port: Peer gRPC port.

        Returns:
            A ``StreamHandle`` if successful, ``None`` if the pool is full
            or closed.
        """
        key = (peer_id, session_id)
        with self._lock:
            if self._closed:
                return None

            handle = self._streams.get(key)
            if handle is not None and not handle.closed:
                handle.last_used_at = time.monotonic()
                return handle

            # Evict expired entries before checking capacity.
            self._reap_expired_locked()

            if len(self._streams) >= self.max_streams:
                logger.warning(
                    "stream_pool_full: max=%d peer=%s session=%s",
                    self.max_streams, peer_id, session_id,
                )
                return None

            handle = self._open_stream(peer_id, session_id, host, port)
            if handle is not None:
                self._streams[key] = handle
            return handle

    def _open_stream(
        self,
        peer_id: str,
        session_id: str,
        host: str,
        port: int,
    ) -> StreamHandle | None:
        """Open a new bidirectional ForwardStream to a peer.

        Returns ``None`` if the peer doesn't support streaming or the
        connection fails (triggering unary fallback).
        """
        try:
            import grpc as _grpc
            from peer import peer_pb2_grpc

            channel = _grpc.insecure_channel(f"{host}:{port}")
            stub = peer_pb2_grpc.PeerStub(channel)

            # Create a thread-safe request queue for the bidirectional stream.
            import queue
            request_queue: queue.Queue = queue.Queue()

            def request_generator() -> Iterator:
                """Yield requests from the queue until a sentinel is received."""
                while True:
                    item = request_queue.get()
                    if item is None:  # sentinel — close stream
                        return
                    yield item

            # Open the bidirectional stream.
            response_iter = stub.ForwardStream(request_generator())

            now = time.monotonic()
            handle = StreamHandle(
                peer_id=peer_id,
                session_id=session_id,
                channel=channel,
                stub=stub,
                request_queue=request_queue,
                response_iter=response_iter,
                created_at=now,
                last_used_at=now,
            )
            logger.info(
                "stream_opened: peer=%s session=%s target=%s:%d",
                peer_id, session_id, host, port,
            )
            return handle
        except Exception as exc:
            logger.debug(
                "stream_open_failed: peer=%s session=%s err=%s (falling back to unary)",
                peer_id, session_id, exc,
            )
            return None

    def send_and_receive(
        self,
        handle: StreamHandle,
        request: Any,
        timeout_s: float = 60.0,
    ) -> Any:
        """Send a request on an open stream and receive the response.

        Args:
            handle: The stream handle from ``get_or_create()``.
            request: A ``ForwardRequest`` protobuf message.
            timeout_s: Maximum seconds to wait for a response.

        Returns:
            A ``ForwardResponse`` protobuf message.

        Raises:
            RuntimeError: If the stream is closed or the response times out.
        """
        if handle.closed:
            raise RuntimeError(f"stream_closed: peer={handle.peer_id}")

        try:
            handle.request_queue.put(request)
            handle.last_used_at = time.monotonic()

            # Read the next response from the iterator.
            response = next(handle.response_iter)
            handle.last_used_at = time.monotonic()
            return response
        except StopIteration:
            handle.closed = True
            raise RuntimeError(f"stream_ended: peer={handle.peer_id}")
        except Exception as exc:
            handle.closed = True
            raise RuntimeError(f"stream_error: peer={handle.peer_id} err={exc}")

    def close_stream(self, peer_id: str, session_id: str) -> None:
        """Close a specific stream and release resources.

        Args:
            peer_id: Target peer identifier.
            session_id: Inference session identifier.
        """
        key = (peer_id, session_id)
        with self._lock:
            handle = self._streams.pop(key, None)
        if handle is not None:
            self._close_handle(handle)

    def close_all(self) -> None:
        """Close all streams and mark the pool as closed."""
        with self._lock:
            self._closed = True
            handles = list(self._streams.values())
            self._streams.clear()
        for h in handles:
            self._close_handle(h)
        logger.info("stream_pool_closed: closed=%d streams", len(handles))

    def reap_expired(self) -> int:
        """Close streams that have been idle longer than ``idle_timeout_s``.

        Returns:
            Number of streams reaped.
        """
        with self._lock:
            return self._reap_expired_locked()

    def _reap_expired_locked(self) -> int:
        """Internal reaper — must be called with ``_lock`` held."""
        now = time.monotonic()
        expired_keys = [
            k for k, h in self._streams.items()
            if h.closed or (now - h.last_used_at > self.idle_timeout_s)
        ]
        for key in expired_keys:
            handle = self._streams.pop(key)
            self._close_handle(handle)
        return len(expired_keys)

    def _close_handle(self, handle: StreamHandle) -> None:
        """Close a single stream handle and its gRPC channel."""
        handle.closed = True
        try:
            if handle.request_queue is not None:
                handle.request_queue.put(None)  # sentinel to close generator
        except Exception:
            pass
        try:
            if handle.channel is not None:
                handle.channel.close()
        except Exception:
            pass
        logger.debug("stream_closed: peer=%s session=%s", handle.peer_id, handle.session_id)

    @property
    def active_count(self) -> int:
        """Number of currently active (non-closed) streams."""
        with self._lock:
            return sum(1 for h in self._streams.values() if not h.closed)

    def has_stream(self, peer_id: str, session_id: str) -> bool:
        """Check if an active stream exists for this peer/session pair."""
        with self._lock:
            h = self._streams.get((peer_id, session_id))
            return h is not None and not h.closed


@dataclass
class InferenceSession:
    """Holds open streams + history for failover replay — Phase 3A.

    Maintains the prompt/token history so that if a peer drops mid-stream,
    the coordinator can replay the history on a replacement peer.

    Attributes:
        session_id: Unique session identifier.
        model_id: Model being served.
        prompt: Original prompt text.
        history: List of (request, response) tuples for replay.
        pipeline_streams: Map of stage_index → StreamHandle.
        created_at: Monotonic timestamp.
    """

    session_id: str
    model_id: str
    prompt: str = ""
    history: list[tuple[Any, Any]] = field(default_factory=list)
    pipeline_streams: dict[int, StreamHandle] = field(default_factory=dict)
    created_at: float = field(default_factory=time.monotonic)

    def record(self, request: Any, response: Any) -> None:
        """Record a request/response pair for potential failover replay.

        Args:
            request: The ForwardRequest sent.
            response: The ForwardResponse received.
        """
        self.history.append((request, response))

    def replay_requests(self) -> list[Any]:
        """Return all recorded requests for replay on a replacement peer.

        Returns:
            List of ForwardRequest messages in order.
        """
        return [req for req, _ in self.history]

    def close(self) -> None:
        """Close all pipeline streams for this session."""
        for handle in self.pipeline_streams.values():
            handle.closed = True
            try:
                if handle.request_queue is not None:
                    handle.request_queue.put(None)
            except Exception:
                pass
            try:
                if handle.channel is not None:
                    handle.channel.close()
            except Exception:
                pass
        self.pipeline_streams.clear()
