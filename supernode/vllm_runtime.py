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

logger = logging.getLogger(__name__)

try:
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False
    AsyncLLMEngine = None
    AsyncEngineArgs = None
    SamplingParams = None


class VLLMRuntime(AttestedRuntime):
    """Level-3 attested runtime for vLLM models.

    Uses vLLM's native AsyncLLMEngine — no ExecutorBridge needed.
    Emits token deltas, not cumulative text.
    """

    def __init__(
        self,
        peer_id: str,
        private_key,
        model_id: str,
        model_path: str | Path,
        *,
        tensor_parallel_size: int = 1,
        max_model_len: int | None = None,
    ):
        super().__init__(peer_id, private_key, model_id)
        self._model_path = str(model_path)
        self._tensor_parallel_size = tensor_parallel_size
        self._max_model_len = max_model_len
        self._engine: AsyncLLMEngine | None = None
        self._active_requests: int = 0

    async def warmup(self, model_id: str) -> bool:
        if not _HAS_VLLM:
            logger.error("vllm not installed")
            return False
        try:
            engine_args = AsyncEngineArgs(
                model=self._model_path,
                tensor_parallel_size=self._tensor_parallel_size,
                max_model_len=self._max_model_len,
            )
            self._engine = AsyncLLMEngine.from_engine_args(engine_args)
            self._register_weights(Path(self._model_path))
            logger.info("vllm_warmup model=%s", model_id)
            return True
        except Exception:
            logger.exception("vllm_warmup_failed model=%s", model_id)
            return False

    async def health_check(self) -> bool:
        return self._engine is not None

    async def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                model_id=self._model_id,
                model_family="vllm",
                parameter_count=0,
                quantization="unknown",
                context_length=self._max_model_len or 4096,
                supports_streaming=True,
            )
        ]

    async def get_status(self) -> BackendStatus:
        return BackendStatus(
            current_load=float(self._active_requests > 0),
            active_requests=self._active_requests,
            max_concurrent=8,
            gpu_memory_free_mb=0,
            models_loaded=[self._model_id] if self._engine else [],
        )

    async def cancel(self, request_id: str) -> None:
        if self._engine is not None:
            await self._engine.abort(request_id)

    async def generate(self, request: PromptRequest) -> AsyncIterator[TokenChunk]:
        if self._engine is None:
            raise RuntimeError("vLLM engine not loaded — call warmup() first")

        prompt = request.prompt or ""
        if request.messages:
            parts = []
            for msg in request.messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"{role}: {content}")
            prompt = "\n".join(parts)

        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop=request.stop,
        )

        self._active_requests += 1
        prev_text = ""
        try:
            async for result in self._engine.generate(
                prompt, sampling_params, request_id=request.request_id
            ):
                output = result.outputs[0]
                delta = output.text[len(prev_text):]
                prev_text = output.text
                if delta:
                    yield TokenChunk(token=delta)
                if output.finish_reason:
                    yield TokenChunk(token="", finish_reason=output.finish_reason)
        finally:
            self._active_requests -= 1

    def shutdown(self):
        self._engine = None
