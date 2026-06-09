from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import cbor2

METHOD_PROMPT_REQUEST = 0x10
METHOD_PROMPT_CANCEL = 0x11
METHOD_MANIFEST_REQUEST = 0x12
METHOD_MANIFEST_RESPONSE = 0x13
METHOD_LOAD_PROBE = 0x14


@dataclass
class WirePromptRequest:
    request_id: str
    model_id: str
    prompt: str | None = None
    messages: list[dict[str, str]] | None = None
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    stop: list[str] | None = None
    stream: bool = True
    session_id: str = ""
    origin_peer_id: str = ""
    hops: int = 0
    min_trust_level: int = 1
    system_prompt: str | None = None
    response_format: str = "text"

    def to_cbor(self) -> bytes:
        return cbor2.dumps(asdict(self), canonical=True)

    @classmethod
    def from_cbor(cls, data: bytes) -> WirePromptRequest:
        d = cbor2.loads(data)
        return cls(**d)


@dataclass
class UsageStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tokens_per_second: float = 0.0
    time_to_first_token_ms: int = 0
    model_weights_hash: str = ""
    output_signature: bytes = b""
    checkpoint_hashes: list[str] = field(default_factory=list)


@dataclass
class PromptChunk:
    request_id: str
    chunk_type: str  # "token" | "done" | "error" | "status"
    token: str = ""
    token_id: int | None = None
    finish_reason: str = ""
    usage: UsageStats | None = None
    error: str = ""
    retryable: bool = False
    status: str = ""
    estimated_wait_s: float = 0.0

    def to_cbor(self) -> bytes:
        d = asdict(self)
        return cbor2.dumps(d, canonical=True)

    @classmethod
    def from_cbor(cls, data: bytes) -> PromptChunk:
        d = cbor2.loads(data)
        usage_data = d.pop("usage", None)
        usage = UsageStats(**usage_data) if usage_data else None
        return cls(usage=usage, **d)
