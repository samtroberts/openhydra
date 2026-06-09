"""Phase 1d: Prompt handler loop for supernode requests over libp2p.

Polls poll_prompt_request() from the Rust P2PNode and dispatches
CBOR-encoded WirePromptRequests to a SupernodeAdapter, streaming
PromptChunks back via respond_prompt().
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

from .adapter import BackendError, PromptRequest, TokenChunk
from .prompt_protocol import (
    METHOD_PROMPT_CANCEL,
    METHOD_PROMPT_REQUEST,
    PromptChunk as WireChunk,
    UsageStats,
    WirePromptRequest,
)

logger = logging.getLogger(__name__)


def _wire_to_adapter_request(wire: WirePromptRequest) -> PromptRequest:
    return PromptRequest(
        request_id=wire.request_id,
        model_id=wire.model_id,
        prompt=wire.prompt,
        messages=wire.messages,
        max_tokens=wire.max_tokens,
        temperature=wire.temperature,
        top_p=wire.top_p,
        top_k=wire.top_k,
        stop=wire.stop,
        stream=wire.stream,
        response_format=wire.response_format,
    )


class PromptHandlerLoop:
    """Receives inbound prompt requests from libp2p and runs inference.

    Spawned as a daemon thread alongside the existing proxy_handler_loop.
    Each inbound request runs adapter.generate() in a background thread
    with its own asyncio event loop (same pattern as ManifestPublisher).
    """

    def __init__(
        self,
        p2p_node: Any,
        adapter: Any,
        stop_event: threading.Event,
    ):
        self._p2p_node = p2p_node
        self._adapter = adapter
        self._stop_event = stop_event
        self._thread: threading.Thread | None = None
        self._active_requests: dict[str, threading.Event] = {}

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="prompt-handler", daemon=True,
        )
        self._thread.start()
        logger.info("prompt_handler_started")

    def stop(self) -> None:
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=5.0)
            self._thread = None
        for cancel_ev in self._active_requests.values():
            cancel_ev.set()
        logger.info("prompt_handler_stopped")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                pending = self._p2p_node.poll_prompt_request(timeout_ms=500)
                if pending is None:
                    continue
                req_id, raw_bytes = pending
                raw = bytes(raw_bytes)
                self._dispatch(req_id, raw)
            except Exception as e:
                if not self._stop_event.is_set():
                    logger.warning("prompt_handler_poll_error: %s", e)
                    time.sleep(1.0)

    def _dispatch(self, req_id: str, raw: bytes) -> None:
        if not raw:
            self._send_error(req_id, "empty_request", retryable=False)
            return

        method = raw[0]
        payload = raw[1:]

        if method == METHOD_PROMPT_REQUEST:
            try:
                wire = WirePromptRequest.from_cbor(payload)
            except Exception as e:
                self._send_error(req_id, f"cbor_decode_failed: {e}", retryable=False)
                return
            cancel_event = threading.Event()
            self._active_requests[wire.request_id] = cancel_event

            def _handle():
                try:
                    self._handle_prompt_request(req_id, wire, cancel_event)
                finally:
                    self._active_requests.pop(wire.request_id, None)

            threading.Thread(target=_handle, daemon=True).start()

        elif method == METHOD_PROMPT_CANCEL:
            try:
                wire = WirePromptRequest.from_cbor(payload)
                cancel_ev = self._active_requests.get(wire.request_id)
                if cancel_ev is not None:
                    cancel_ev.set()
                    logger.info("prompt_cancel request_id=%s", wire.request_id)
                self._p2p_node.respond_prompt(
                    request_id=req_id,
                    data=bytes([METHOD_PROMPT_CANCEL]),
                )
            except Exception as e:
                logger.warning("prompt_cancel_error: %s", e)
                self._send_error(req_id, f"cancel_failed: {e}", retryable=False)
        else:
            self._send_error(req_id, f"unknown_method: 0x{method:02x}", retryable=False)

    def _handle_prompt_request(
        self,
        req_id: str,
        wire: WirePromptRequest,
        cancel_event: threading.Event,
    ) -> None:
        adapter_req = _wire_to_adapter_request(wire)
        loop = asyncio.new_event_loop()
        try:
            token_count = 0
            t0 = time.monotonic()
            first_token_time: float | None = None

            async def _stream():
                nonlocal token_count, first_token_time
                async for chunk in self._adapter.generate(adapter_req):
                    if cancel_event.is_set():
                        logger.info("prompt_cancelled request_id=%s tokens=%d", wire.request_id, token_count)
                        break
                    if chunk.token:
                        if first_token_time is None:
                            first_token_time = time.monotonic()
                        token_count += 1
                    if chunk.finish_reason:
                        return chunk.finish_reason
                return "stop"

            finish_reason = loop.run_until_complete(_stream())
            elapsed = time.monotonic() - t0
            tps = token_count / elapsed if elapsed > 0 else 0.0
            ttft_ms = int((first_token_time - t0) * 1000) if first_token_time else 0

            done_chunk = WireChunk(
                request_id=wire.request_id,
                chunk_type="done",
                finish_reason=finish_reason if not cancel_event.is_set() else "cancelled",
                usage=UsageStats(
                    completion_tokens=token_count,
                    tokens_per_second=round(tps, 2),
                    time_to_first_token_ms=ttft_ms,
                ),
            )
            self._p2p_node.respond_prompt(
                request_id=req_id,
                data=bytes([METHOD_PROMPT_REQUEST]) + done_chunk.to_cbor(),
            )
            logger.info(
                "prompt_done request_id=%s tokens=%d tps=%.1f ttft=%dms",
                wire.request_id, token_count, tps, ttft_ms,
            )

        except BackendError as e:
            logger.warning("prompt_backend_error request_id=%s: %s", wire.request_id, e)
            self._send_error(req_id, str(e), retryable=True)
        except Exception as e:
            logger.error("prompt_handler_crash request_id=%s: %s", wire.request_id, e, exc_info=True)
            self._send_error(req_id, str(e), retryable=False)
        finally:
            loop.close()

    def _send_error(self, req_id: str, error: str, retryable: bool) -> None:
        chunk = WireChunk(
            request_id=req_id,
            chunk_type="error",
            error=error,
            retryable=retryable,
        )
        try:
            self._p2p_node.respond_prompt(
                request_id=req_id,
                data=bytes([METHOD_PROMPT_REQUEST]) + chunk.to_cbor(),
            )
        except Exception as e:
            logger.warning("prompt_error_send_failed: %s", e)
