from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class PromptRequest:
    request_id: str
    model_id: str
    prompt: str | None = None
    messages: list[dict] | None = None
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    stop: list[str] | None = None
    system_prompt: str | None = None
    stream: bool = True
    response_format: str = "text"


@dataclass
class TokenChunk:
    token: str
    token_id: int | None = None
    finish_reason: str | None = None


@dataclass
class ModelInfo:
    model_id: str
    model_family: str
    parameter_count: int
    quantization: str
    context_length: int
    supports_streaming: bool = True
    supports_system_prompt: bool = True


@dataclass
class BackendStatus:
    current_load: float
    active_requests: int
    max_concurrent: int
    gpu_memory_free_mb: int
    models_loaded: list[str] = field(default_factory=list)


class BackendError(Exception):
    pass


class SupernodeAdapter(ABC):

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        ...

    @abstractmethod
    async def generate(self, request: PromptRequest) -> AsyncIterator[TokenChunk]:
        ...

    @abstractmethod
    async def cancel(self, request_id: str) -> None:
        ...

    @abstractmethod
    async def get_status(self) -> BackendStatus:
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        ...

    @abstractmethod
    async def warmup(self, model_id: str) -> bool:
        ...

    def backend_type(self) -> str:
        return self.__class__.__name__.lower().replace("adapter", "").replace("runtime", "")

    def trust_tier(self) -> str:
        return "unverified"

    def integration_level(self) -> int:
        return 1

    def get_weights_hash(self, model_id: str) -> str | None:
        return None

    def sign_output(
        self,
        request: PromptRequest,
        model_id: str,
        output_token_ids: list[int],
        timestamp_ms: int,
    ) -> bytes | None:
        return None
