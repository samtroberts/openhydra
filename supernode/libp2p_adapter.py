"""LibP2PAdapter: routes prompts to a remote supernode via libp2p streams.

Uses the `/openhydra/prompt-stream/1.0.0` protocol for true token-by-token
streaming, avoiding the thread pool exhaustion risk of blocking
request-response calls.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, AsyncIterator

from .adapter import (
    BackendError,
    BackendStatus,
    ModelInfo,
    PromptRequest,
    SupernodeAdapter,
    TokenChunk,
)
from .manifest import SupernodeManifest
from .prompt_protocol import (
    METHOD_PROMPT_REQUEST,
    PromptChunk,
    WirePromptRequest,
)

logger = logging.getLogger(__name__)


class LibP2PAdapter(SupernodeAdapter):
    """SupernodeAdapter that streams prompts to a remote peer over libp2p."""

    def __init__(
        self,
        p2p_node: Any,
        target_peer_id: str,
        manifest: SupernodeManifest,
        origin_peer_id: str = "",
    ):
        self._p2p_node = p2p_node
        self._target_peer_id = target_peer_id
        self._manifest = manifest
        self._origin_peer_id = origin_peer_id

    async def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                model_id=m.model_id,
                model_family=m.model_family,
                parameter_count=m.parameter_count,
                quantization=m.quantization,
                context_length=m.context_length,
                supports_streaming=True,
            )
            for m in self._manifest.models
        ]

    async def generate(self, request: PromptRequest) -> AsyncIterator[TokenChunk]:
        wire = WirePromptRequest(
            request_id=request.request_id or str(uuid.uuid4()),
            model_id=request.model_id,
            prompt=request.prompt,
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop=request.stop,
            stream=request.stream,
            origin_peer_id=self._origin_peer_id,
            response_format=request.response_format,
        )
        payload = bytes([METHOD_PROMPT_REQUEST]) + wire.to_cbor()

        loop = asyncio.get_event_loop()

        stream_id = await loop.run_in_executor(
            None, self._p2p_node.open_prompt_stream, self._target_peer_id, payload,
        )
        logger.info(
            "libp2p_stream_opened peer=%s stream=%s request=%s",
            self._target_peer_id, stream_id, wire.request_id,
        )

        try:
            while True:
                chunk_bytes = await loop.run_in_executor(
                    None, self._p2p_node.poll_prompt_chunk, stream_id, 500,
                )
                if chunk_bytes is None:
                    continue
                if not chunk_bytes:
                    break

                method = chunk_bytes[0]
                chunk = PromptChunk.from_cbor(chunk_bytes[1:])

                if chunk.chunk_type == "token" and chunk.token:
                    yield TokenChunk(token=chunk.token)
                elif chunk.chunk_type == "done":
                    if chunk.token:
                        yield TokenChunk(
                            token=chunk.token,
                            finish_reason=chunk.finish_reason or "stop",
                        )
                    else:
                        yield TokenChunk(
                            token="",
                            finish_reason=chunk.finish_reason or "stop",
                        )
                    return
                elif chunk.chunk_type == "error":
                    raise BackendError(
                        f"remote supernode error: {chunk.error}"
                    )
        finally:
            try:
                self._p2p_node.close_prompt_stream(stream_id)
            except Exception:
                pass

    async def cancel(self, request_id: str) -> None:
        pass

    async def get_status(self) -> BackendStatus:
        return BackendStatus(
            current_load=0.0,
            active_requests=0,
            max_concurrent=self._manifest.max_concurrent_requests,
            gpu_memory_free_mb=0,
            models_loaded=[m.model_id for m in self._manifest.models],
        )

    async def health_check(self) -> bool:
        try:
            connected = self._p2p_node.is_peer_connected(self._target_peer_id)
            return connected and self._manifest.is_fresh()
        except Exception:
            return False

    async def warmup(self, model_id: str) -> bool:
        return True

    def backend_type(self) -> str:
        return f"libp2p-{self._manifest.backend_type}"
