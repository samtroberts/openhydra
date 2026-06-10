from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import AsyncIterator

from .adapter import (
    PromptRequest,
    TokenChunk,
    ModelInfo,
    BackendStatus,
)
from .attested_base import AttestedRuntime
from .executor_bridge import ExecutorBridge

logger = logging.getLogger(__name__)

try:
    from llama_cpp import Llama
    _HAS_LLAMA_CPP = True
except ImportError:
    _HAS_LLAMA_CPP = False
    Llama = None


class GGUFRuntime(AttestedRuntime):
    """Level-3 attested runtime for GGUF models via llama-cpp-python."""

    def __init__(
        self,
        peer_id: str,
        private_key,
        model_id: str,
        model_path: str | Path,
        *,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
    ):
        super().__init__(peer_id, private_key, model_id)
        self._model_path = Path(model_path)
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._llm: Llama | None = None
        self._bridge = ExecutorBridge(max_workers=1)
        self._active_requests: int = 0

    async def warmup(self, model_id: str) -> bool:
        if not _HAS_LLAMA_CPP:
            logger.error("llama-cpp-python not installed")
            return False
        try:
            self._llm = Llama(
                model_path=str(self._model_path),
                n_ctx=self._n_ctx,
                n_gpu_layers=self._n_gpu_layers,
                verbose=False,
            )
            self._register_weights(self._model_path.parent)
            logger.info("gguf_warmup model=%s ctx=%d", model_id, self._n_ctx)
            return True
        except Exception:
            logger.exception("gguf_warmup_failed model=%s", model_id)
            return False

    async def health_check(self) -> bool:
        return self._llm is not None

    async def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                model_id=self._model_id,
                model_family="gguf",
                parameter_count=0,
                quantization="unknown",
                context_length=self._n_ctx,
                supports_streaming=True,
            )
        ]

    async def get_status(self) -> BackendStatus:
        return BackendStatus(
            current_load=float(self._active_requests > 0),
            active_requests=self._active_requests,
            max_concurrent=1,
            gpu_memory_free_mb=0,
            models_loaded=[self._model_id] if self._llm else [],
        )

    async def cancel(self, request_id: str) -> None:
        pass

    async def generate(self, request: PromptRequest) -> AsyncIterator[TokenChunk]:
        if self._llm is None:
            raise RuntimeError("GGUF model not loaded — call warmup() first")

        prompt = request.prompt or ""
        if request.messages:
            parts = []
            for msg in request.messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"{role}: {content}")
            prompt = "\n".join(parts)

        llm = self._llm
        token_ids: list[int] = []

        def _sync_gen():
            self._active_requests += 1
            try:
                for output in llm.create_completion(
                    prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    stop=request.stop,
                    stream=True,
                ):
                    choice = output["choices"][0]
                    text = choice.get("text", "")
                    finish = choice.get("finish_reason")
                    yield TokenChunk(token=text, finish_reason=finish)
            finally:
                self._active_requests -= 1

        async for chunk in self._bridge.stream(_sync_gen):
            yield chunk

    def shutdown(self):
        self._bridge.shutdown()
        self._llm = None
