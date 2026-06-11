from __future__ import annotations

import logging
from typing import AsyncIterator, TYPE_CHECKING

from .adapter import (
    PromptRequest,
    TokenChunk,
    ModelInfo,
    BackendStatus,
)
from .attested_base import AttestedRuntime
from .executor_bridge import ExecutorBridge

if TYPE_CHECKING:
    from coordinator.engine import CoordinatorEngine

logger = logging.getLogger(__name__)


class NativeAdapter(AttestedRuntime):
    """Level-3 attested runtime wrapping CoordinatorEngine.infer_stream().

    The engine's sync generator already handles the full inference stack
    (discovery, pipeline, KV, speculative decode). This adapter bridges
    it to the async SupernodeAdapter interface via ExecutorBridge.
    """

    def __init__(
        self,
        peer_id: str,
        private_key,
        model_id: str,
        engine: CoordinatorEngine,
    ):
        super().__init__(peer_id, private_key, model_id)
        self._engine = engine
        self._bridge = ExecutorBridge(max_workers=1)
        self._active_requests: int = 0

    async def warmup(self, model_id: str) -> bool:
        return True

    async def health_check(self) -> bool:
        return self._engine is not None

    async def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                model_id=self._model_id,
                model_family="native",
                parameter_count=0,
                quantization="unknown",
                context_length=4096,
                supports_streaming=True,
            )
        ]

    async def get_status(self) -> BackendStatus:
        return BackendStatus(
            current_load=float(self._active_requests > 0),
            active_requests=self._active_requests,
            max_concurrent=1,
            gpu_memory_free_mb=0,
            models_loaded=[self._model_id],
        )

    async def cancel(self, request_id: str) -> None:
        pass

    async def generate(self, request: PromptRequest) -> AsyncIterator[TokenChunk]:
        prompt = request.prompt or ""
        if request.messages:
            parts = []
            for msg in request.messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"{role}: {content}")
            prompt = "\n".join(parts)

        engine = self._engine

        def _sync_gen():
            self._active_requests += 1
            try:
                result = engine.infer_stream(
                    prompt=prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                )
                stream = result.get("stream")
                if stream is None:
                    text = result.get("response", "")
                    yield TokenChunk(token=text, finish_reason="stop")
                    return
                for token_text in stream:
                    yield TokenChunk(token=str(token_text))
                yield TokenChunk(token="", finish_reason="stop")
            finally:
                self._active_requests -= 1

        async for chunk in self._bridge.stream(_sync_gen):
            yield chunk

    def shutdown(self):
        self._bridge.shutdown()
