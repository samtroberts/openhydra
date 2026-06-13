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

"""Pydantic v2 request models for the OpenHydra coordinator HTTP API.

These types mirror the JSON bodies currently parsed by hand in
``coordinator.api_server``.  They are **not yet wired in** — the plan is
to validate incoming requests through these models in a follow-up step
once the test suite confirms they match the existing parsing logic.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# OpenAI-compatible inference endpoints
# ---------------------------------------------------------------------------

class ChatCompletionRequest(BaseModel):
    """POST /v1/chat/completions"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = "openhydra-qwen3.5-0.8b"
    messages: list[dict[str, str]]
    max_tokens: int = 256
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    do_sample: bool | None = None
    stream: bool = False
    pipeline_width: int | None = None
    grounding: bool = True
    priority: bool = False
    client_id: str = "anonymous"
    allow_degradation: bool = True
    session_id: str | None = None
    expert_tags: list[str] | None = None
    expert_layer_indices: list[int] | None = None


class CompletionRequest(BaseModel):
    """POST /v1/completions"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = "openhydra-qwen3.5-0.8b"
    prompt: str
    max_tokens: int = 256
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    do_sample: bool | None = None
    stream: bool = False
    pipeline_width: int | None = None
    grounding: bool = True
    priority: bool = False
    client_id: str = "anonymous"
    allow_degradation: bool = True
    session_id: str | None = None
    expert_tags: list[str] | None = None
    expert_layer_indices: list[int] | None = None


# ---------------------------------------------------------------------------
# Ollama-compatible endpoints
# ---------------------------------------------------------------------------

class OllamaGenerateRequest(BaseModel):
    """POST /api/generate"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = "openhydra-qwen3.5-0.8b"
    prompt: str = ""
    stream: bool = False
    client_id: str = "anonymous"
    options: dict[str, object] | None = None


class OllamaChatRequest(BaseModel):
    """POST /api/chat"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = "openhydra-qwen3.5-0.8b"
    messages: list[dict[str, str]] = []
    stream: bool = False
    client_id: str = "anonymous"
    options: dict[str, object] | None = None
