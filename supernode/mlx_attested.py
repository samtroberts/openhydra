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
    import mlx_lm
    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False
    mlx_lm = None


class MLXAttestedRuntime(AttestedRuntime):
    """Level-3 attested runtime for MLX models."""

    def __init__(
        self,
        peer_id: str,
        private_key,
        model_id: str,
        model_path: str | Path,
        *,
        max_tokens_default: int = 512,
    ):
        super().__init__(peer_id, private_key, model_id)
        self._model_path = Path(model_path)
        self._max_tokens_default = max_tokens_default
        self._model = None
        self._tokenizer = None
        self._bridge = ExecutorBridge(max_workers=1)
        self._active_requests: int = 0

    async def warmup(self, model_id: str) -> bool:
        if not _HAS_MLX:
            logger.error("mlx_lm not installed")
            return False
        try:
            self._model, self._tokenizer = mlx_lm.load(str(self._model_path))
            self._register_weights(self._model_path)
            logger.info("mlx_warmup model=%s", model_id)
            return True
        except Exception:
            logger.exception("mlx_warmup_failed model=%s", model_id)
            return False

    async def health_check(self) -> bool:
        return self._model is not None

    async def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                model_id=self._model_id,
                model_family="mlx",
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
            models_loaded=[self._model_id] if self._model else [],
        )

    async def cancel(self, request_id: str) -> None:
        pass

    async def generate(self, request: PromptRequest) -> AsyncIterator[TokenChunk]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("MLX model not loaded — call warmup() first")

        prompt = request.prompt or ""
        if request.messages:
            parts = []
            for msg in request.messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"{role}: {content}")
            prompt = "\n".join(parts)

        model = self._model
        tokenizer = self._tokenizer
        max_tokens = request.max_tokens or self._max_tokens_default

        def _sync_gen():
            self._active_requests += 1
            try:
                count = 0
                for text_chunk in mlx_lm.stream_generate(
                    model, tokenizer, prompt=prompt, max_tokens=max_tokens
                ):
                    count += 1
                    yield TokenChunk(token=text_chunk)
                yield TokenChunk(token="", finish_reason="stop")
            finally:
                self._active_requests -= 1

        async for chunk in self._bridge.stream(_sync_gen):
            yield chunk

    def shutdown(self):
        self._bridge.shutdown()
        self._model = None
        self._tokenizer = None
