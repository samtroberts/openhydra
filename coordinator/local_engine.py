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

"""LocalInferenceEngine — fully offline, zero-network inference.

This engine calls ``ModelShard.forward()`` directly as a Python object,
bypassing the entire coordinator service stack (no DiscoveryService,
no PipelineService, no KvAffinityService, no gRPC, no DHT).

It is the backbone of **Local Mode** in the Hybrid Local/Swarm
architecture (v1.1).  Users point Cursor, LM Studio, or Open WebUI at
``localhost:8080`` and it just works — offline, instant, no API key.

Architecture
------------
::

    curl POST /v1/chat/completions
        │
        ▼
    LocalInferenceEngine.chat()
        │  apply chat template
        ▼
    LocalInferenceEngine.infer()
        │  tokenize prompt (cached tokenizer)
        ▼
    ModelShard.forward()           ← direct Python call, NO gRPC
        │  MLX / PyTorch / Toy
        ▼
    decode tokens → OpenAI JSON

Strict Boundary
---------------
This module must **never** import:
- ``coordinator.discovery_service``
- ``coordinator.pipeline_service``
- ``coordinator.inference_service``
- ``coordinator.chain``
- ``grpc`` / ``peer_pb2_grpc``
"""

from __future__ import annotations

import gc
import threading
import time
import uuid
from typing import Any, Generator

from peer.model_shard import ModelShard


class LocalInferenceEngine:
    """Zero-network inference engine that calls ModelShard directly.

    Args:
        model_id: The model identifier (e.g. "openhydra-qwen3.5-0.8b").
        shard: A fully-initialized ``ModelShard`` instance (ToyRuntime,
            PyTorchRuntime, or MLXRuntime).  The engine does NOT own the
            shard lifecycle — the caller (mode switch controller) manages
            loading and unloading.
    """

    def __init__(self, model_id: str, shard: ModelShard) -> None:
        self.model_id = model_id
        self.shard = shard
        self._lock = threading.Lock()

    # ── Model listing ───────────────────────────────────────────────────────

    def list_models(self) -> list[dict[str, Any]]:
        """Return the single loaded model in OpenAI /v1/models format."""
        profile = self.shard.runtime_profile() if self.shard else {}
        return [
            {
                "id": self.model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "openhydra-local",
                "permission": [],
                "root": self.model_id,
                "parent": None,
                "meta": {
                    "backend": profile.get("backend", "unknown"),
                    "quantization": profile.get("quantization_mode", "fp32"),
                },
            }
        ]

    # ── Core inference ──────────────────────────────────────────────────────

    def infer(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run a single-turn completion and return an OpenAI-compatible dict.

        This calls ``ModelShard.forward()`` directly — no gRPC, no DHT,
        no pipeline assembly.
        """
        request_id = str(uuid.uuid4())
        prompt_token_count = self._count_prompt_tokens(prompt)

        # Call shard directly (single stage, full model)
        activation = self.shard.forward(
            prompt=prompt,
            activation=[],
            max_tokens=max_tokens,
            stage_index=0,
            total_stages=1,
            decode_temperature=temperature,
            decode_top_p=top_p,
        )

        # Decode activation to text
        text = ModelShard.decode_text(
            activation,
            max_tokens=max_tokens,
        )

        # Apply stop sequences
        finish_reason = "length"
        if stop:
            for seq in stop:
                idx = text.lower().find(seq.lower())
                if idx >= 0:
                    text = text[:idx].rstrip()
                    finish_reason = "stop"
                    break

        # Estimate completion tokens from the decoded text
        completion_token_count = self._count_completion_tokens(text, max_tokens)

        # Determine finish reason: if we generated fewer tokens than max, it's "stop"
        if finish_reason != "stop" and completion_token_count < max_tokens:
            finish_reason = "stop"

        return self._build_response(
            request_id=request_id,
            content=text,
            finish_reason=finish_reason,
            prompt_tokens=prompt_token_count,
            completion_tokens=completion_token_count,
        )

    def infer_stream(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop: list[str] | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """Streaming variant of infer() that yields SSE-format chunks.

        For ToyRuntime (and PyTorchRuntime without streaming support),
        this generates all tokens at once then yields them one word at a
        time to emulate streaming.  Real MLX streaming will be wired in
        Phase 2.
        """
        request_id = str(uuid.uuid4())

        # Generate full response
        activation = self.shard.forward(
            prompt=prompt,
            activation=[],
            max_tokens=max_tokens,
            stage_index=0,
            total_stages=1,
            decode_temperature=temperature,
            decode_top_p=top_p,
        )

        text = ModelShard.decode_text(activation, max_tokens=max_tokens)

        # Apply stop sequences
        finish_reason = "length"
        if stop:
            for seq in stop:
                idx = text.lower().find(seq.lower())
                if idx >= 0:
                    text = text[:idx].rstrip()
                    finish_reason = "stop"
                    break

        # Determine finish reason
        completion_token_count = self._count_completion_tokens(text, max_tokens)
        if finish_reason != "stop" and completion_token_count < max_tokens:
            finish_reason = "stop"

        # Yield word-by-word chunks
        words = text.split() if text else []
        for i, word in enumerate(words):
            prefix = " " if i > 0 else ""
            is_last = i == len(words) - 1
            yield self._build_stream_chunk(
                request_id=request_id,
                content=prefix + word,
                finish_reason=finish_reason if is_last else None,
            )

        # If no words at all, yield one empty final chunk
        if not words:
            yield self._build_stream_chunk(
                request_id=request_id,
                content="",
                finish_reason=finish_reason,
            )

    # ── Chat interface ──────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop: list[str] | None = None,
    ) -> dict[str, Any]:
        """Chat completion — apply template then delegate to infer()."""
        prompt = self._apply_chat_template(messages)
        return self.infer(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop: list[str] | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """Streaming chat — apply template then delegate to infer_stream()."""
        prompt = self._apply_chat_template(messages)
        yield from self.infer_stream(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def unload(self) -> None:
        """Release the model shard and free memory.

        After this call, the engine is inert — all inference methods will
        fail.  The caller (mode switch controller) is responsible for
        creating a new engine with a new shard.
        """
        with self._lock:
            if self.shard is not None:
                # Release the runtime inside the shard
                runtime = getattr(self.shard, "_runtime", None)
                if runtime is not None:
                    # MLX-specific: release Metal buffer pool
                    try:
                        import mlx.core as mx
                        mx.metal.clear_cache()
                    except (ImportError, AttributeError):
                        pass
                    del runtime
                self.shard = None
            gc.collect()

    # ── Private helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _apply_chat_template(messages: list[dict[str, str]]) -> str:
        """Flatten a messages list into a single prompt string.

        Uses a simple default template.  Real tokenizer chat templates
        (Jinja2) will be wired when we integrate with HuggingFace
        tokenizers in Phase 2.
        """
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(f"User: {content}")
        parts.append("Assistant:")
        return "\n".join(parts)

    @staticmethod
    def _count_prompt_tokens(prompt: str) -> int:
        """Estimate prompt token count.

        Uses a simple word-split heuristic for ToyRuntime.  Real
        tokenizer counting will be wired in Phase 2 when the tokenizer
        is cached at init.
        """
        return len(prompt.split())

    @staticmethod
    def _count_completion_tokens(text: str, max_tokens: int) -> int:
        """Count completion tokens from decoded text.

        For ToyRuntime: count words (matches decode_text's word-join).
        For real runtimes: will use actual token IDs in Phase 2.
        """
        if not text:
            return 0
        return min(len(text.split()), max_tokens)

    def _build_response(
        self,
        request_id: str,
        content: str,
        finish_reason: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> dict[str, Any]:
        """Build an OpenAI ChatCompletion-format response."""
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def _build_stream_chunk(
        self,
        request_id: str,
        content: str,
        finish_reason: str | None,
    ) -> dict[str, Any]:
        """Build an OpenAI ChatCompletion chunk for SSE streaming."""
        return {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": content,
                    },
                    "finish_reason": finish_reason,
                }
            ],
        }
