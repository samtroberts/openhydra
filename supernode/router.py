from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from typing import Any, Iterator

from .adapter import (
    SupernodeAdapter,
    PromptRequest,
    TokenChunk,
    ModelInfo,
    BackendStatus,
    BackendError,
)
from .selector import PromptRouter, ScoredCandidate

logger = logging.getLogger(__name__)


class SupernodeRouter:
    """Routes OpenAI-format requests to registered SupernodeAdapters.

    MVP: single local adapter (Ollama). Phase 1d adds remote dispatch
    via libp2p; this class gains multi-adapter selection at that point.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop | None = None,
        prompt_router: PromptRouter | None = None,
    ):
        self._adapters: dict[str, SupernodeAdapter] = {}
        self._loop = loop
        self._local = threading.local()
        self._model_cache: dict[str, list[ModelInfo]] = {}
        self._prompt_router = prompt_router

    def register_adapter(self, name: str, adapter: SupernodeAdapter) -> None:
        self._adapters[name] = adapter

    def _run(self, coro):
        return asyncio.run(coro)

    # ------------------------------------------------------------------
    # Model listing
    # ------------------------------------------------------------------

    def list_models_openai(self) -> dict[str, Any]:
        models = self._run(self._list_all_models())
        return {
            "object": "list",
            "data": [
                {
                    "id": m.model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "openhydra",
                    "openhydra": {
                        "family": m.model_family,
                        "parameter_count": m.parameter_count,
                        "quantization": m.quantization,
                        "context_length": m.context_length,
                        "supports_streaming": m.supports_streaming,
                    },
                }
                for m in models
            ],
        }

    async def _list_all_models(self) -> list[ModelInfo]:
        all_models: list[ModelInfo] = []
        for name, adapter in self._adapters.items():
            try:
                models = await adapter.list_models()
                self._model_cache[name] = models
                all_models.extend(models)
            except Exception:
                logger.warning("supernode_list_models_failed adapter=%s", name, exc_info=True)
        return all_models

    # ------------------------------------------------------------------
    # Supernode status
    # ------------------------------------------------------------------

    def list_supernodes(self) -> list[dict[str, Any]]:
        return self._run(self._list_supernodes_async())

    async def _list_supernodes_async(self) -> list[dict[str, Any]]:
        results = []
        for name, adapter in self._adapters.items():
            entry: dict[str, Any] = {
                "name": name,
                "backend": adapter.backend_type(),
                "trust_tier": adapter.trust_tier(),
                "integration_level": adapter.integration_level(),
                "healthy": False,
                "status": None,
                "models": [],
            }
            try:
                entry["healthy"] = await adapter.health_check()
                if entry["healthy"]:
                    status = await adapter.get_status()
                    entry["status"] = {
                        "current_load": status.current_load,
                        "active_requests": status.active_requests,
                        "max_concurrent": status.max_concurrent,
                        "gpu_memory_free_mb": status.gpu_memory_free_mb,
                        "models_loaded": status.models_loaded,
                    }
                    models = await adapter.list_models()
                    entry["models"] = [m.model_id for m in models]
            except Exception:
                logger.warning("supernode_status_failed adapter=%s", name, exc_info=True)
            results.append(entry)
        return results

    # ------------------------------------------------------------------
    # Inference — chat completions
    # ------------------------------------------------------------------

    def chat_completion(
        self,
        body: dict[str, Any],
        request_id: str | None = None,
    ) -> dict[str, Any]:
        request_id = request_id or str(uuid.uuid4())
        req = self._body_to_prompt_request(body, request_id)
        return self._run(self._chat_completion_async(req, body))

    def chat_completion_stream(
        self,
        body: dict[str, Any],
        request_id: str | None = None,
    ) -> Iterator[str]:
        request_id = request_id or str(uuid.uuid4())
        req = self._body_to_prompt_request(body, request_id)
        req.stream = True

        async def collect():
            chunks = []
            async for chunk in self._generate(req):
                chunks.append(chunk)
            return chunks

        chunks = self._run(collect())
        for chunk in chunks:
            if chunk.token:
                yield chunk.token

    # ------------------------------------------------------------------
    # Inference — text completions
    # ------------------------------------------------------------------

    def text_completion(
        self,
        body: dict[str, Any],
        request_id: str | None = None,
    ) -> dict[str, Any]:
        request_id = request_id or str(uuid.uuid4())
        req = self._body_to_prompt_request(body, request_id, mode="completion")
        return self._run(self._text_completion_async(req, body))

    def text_completion_stream(
        self,
        body: dict[str, Any],
        request_id: str | None = None,
    ) -> Iterator[str]:
        request_id = request_id or str(uuid.uuid4())
        req = self._body_to_prompt_request(body, request_id, mode="completion")
        req.stream = True

        async def collect():
            chunks = []
            async for chunk in self._generate(req):
                chunks.append(chunk)
            return chunks

        chunks = self._run(collect())
        for chunk in chunks:
            if chunk.token:
                yield chunk.token

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _chat_completion_async(
        self, req: PromptRequest, body: dict[str, Any]
    ) -> dict[str, Any]:
        text_parts: list[str] = []
        token_count = 0
        finish_reason = "stop"
        backend_usage: dict | None = None

        async for chunk in self._generate(req):
            text_parts.append(chunk.token)
            token_count += 1
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
            if chunk.usage:
                backend_usage = chunk.usage

        usage: dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": token_count,
            "total_tokens": token_count,
        }
        if backend_usage:
            if "eval_count" in backend_usage:
                usage["completion_tokens"] = backend_usage["eval_count"]
            if "prompt_eval_count" in backend_usage:
                usage["prompt_tokens"] = backend_usage["prompt_eval_count"]
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            usage["backend"] = backend_usage

        return {
            "id": req.request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "".join(text_parts)},
                "finish_reason": finish_reason,
            }],
            "usage": usage,
        }

    async def _text_completion_async(
        self, req: PromptRequest, body: dict[str, Any]
    ) -> dict[str, Any]:
        text_parts: list[str] = []
        token_count = 0
        finish_reason = "stop"
        backend_usage: dict | None = None

        async for chunk in self._generate(req):
            text_parts.append(chunk.token)
            token_count += 1
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
            if chunk.usage:
                backend_usage = chunk.usage

        usage: dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": token_count,
            "total_tokens": token_count,
        }
        if backend_usage:
            if "eval_count" in backend_usage:
                usage["completion_tokens"] = backend_usage["eval_count"]
            if "prompt_eval_count" in backend_usage:
                usage["prompt_tokens"] = backend_usage["prompt_eval_count"]
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            usage["backend"] = backend_usage

        return {
            "id": req.request_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": req.model_id,
            "choices": [{
                "index": 0,
                "text": "".join(text_parts),
                "finish_reason": finish_reason,
            }],
            "usage": usage,
        }

    async def _generate(self, req: PromptRequest):
        if self._prompt_router is not None:
            async for chunk in self._generate_with_failover(req):
                yield chunk
            return

        adapter = self._select_adapter(req.model_id)
        if adapter is None:
            raise BackendError(f"No adapter available for model '{req.model_id}'")
        async for chunk in adapter.generate(req):
            yield chunk

    async def _generate_with_failover(self, req: PromptRequest):
        """Fail-fast failover (§4.4): pre-first-token resend; post-first-token error."""
        candidates = self._prompt_router.select_with_failover(req.model_id)
        if not candidates:
            raise BackendError(f"No supernode available for model '{req.model_id}'")

        for i, candidate in enumerate(candidates):
            peer_id = candidate.manifest.libp2p_peer_id
            adapter = self._prompt_router._adapters.get(peer_id)
            if adapter is None:
                adapter = self._select_adapter(req.model_id)
            if adapter is None:
                continue

            first_token_sent = False
            try:
                async for chunk in adapter.generate(req):
                    first_token_sent = True
                    yield chunk
                return
            except Exception as e:
                self._prompt_router.record_failure(peer_id)
                if first_token_sent:
                    logger.warning(
                        "supernode_mid_stream_failure peer=%s err=%s",
                        peer_id, e,
                    )
                    yield TokenChunk(token="", finish_reason="error")
                    return
                else:
                    is_last = i == len(candidates) - 1
                    logger.warning(
                        "supernode_pre_token_failure peer=%s retrying=%s err=%s",
                        peer_id, not is_last, e,
                    )
                    if is_last:
                        raise BackendError(
                            f"All {len(candidates)} supernodes failed for '{req.model_id}'"
                        ) from e

    def _select_adapter(self, model_id: str) -> SupernodeAdapter | None:
        if len(self._adapters) == 1:
            return next(iter(self._adapters.values()))
        for name, cached_models in self._model_cache.items():
            if any(m.model_id == model_id for m in cached_models):
                return self._adapters[name]
        if self._adapters:
            return next(iter(self._adapters.values()))
        return None

    def _body_to_prompt_request(
        self,
        body: dict[str, Any],
        request_id: str,
        mode: str = "chat",
    ) -> PromptRequest:
        stop = body.get("stop")
        if isinstance(stop, str):
            stop = [stop]

        if mode == "chat":
            return PromptRequest(
                request_id=request_id,
                model_id=str(body.get("model", "")),
                messages=body.get("messages"),
                max_tokens=int(body.get("max_tokens", 512)),
                temperature=float(body.get("temperature", 0.7)),
                top_p=float(body.get("top_p", 0.9)),
                top_k=int(body.get("top_k", 40)),
                stop=stop,
                stream=bool(body.get("stream", False)),
                response_format=body.get("response_format", {}).get("type", "text")
                if isinstance(body.get("response_format"), dict)
                else str(body.get("response_format", "text")),
            )
        else:
            return PromptRequest(
                request_id=request_id,
                model_id=str(body.get("model", "")),
                prompt=str(body.get("prompt", "")),
                max_tokens=int(body.get("max_tokens", 512)),
                temperature=float(body.get("temperature", 0.7)),
                top_p=float(body.get("top_p", 0.9)),
                top_k=int(body.get("top_k", 40)),
                stop=stop,
                stream=bool(body.get("stream", False)),
                response_format=body.get("response_format", {}).get("type", "text")
                if isinstance(body.get("response_format"), dict)
                else str(body.get("response_format", "text")),
            )
