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
# HYDRA token economy endpoints
# ---------------------------------------------------------------------------

class HydraTransferRequest(BaseModel):
    """POST /v1/hydra/transfer"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    from_client_id: str
    to_client_id: str
    amount: float


class HydraStakeRequest(BaseModel):
    """POST /v1/hydra/stake  and  POST /v1/hydra/unstake"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    client_id: str
    amount: float


class HydraChannelOpenRequest(BaseModel):
    """POST /v1/hydra/channels/open"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    channel_id: str = ""
    payer: str
    payee: str
    deposit: float
    ttl_seconds: int | None = None


class HydraChannelChargeRequest(BaseModel):
    """POST /v1/hydra/channels/charge"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    channel_id: str
    amount: float
    provider_peer_id: str | None = None


class HydraChannelReconcileRequest(BaseModel):
    """POST /v1/hydra/channels/reconcile"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    channel_id: str
    total_spent: float
    nonce: int


class HydraChannelCloseRequest(BaseModel):
    """POST /v1/hydra/channels/close"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    channel_id: str


class HydraGovernanceVoteRequest(BaseModel):
    """POST /v1/hydra/governance/vote"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pubkey: str
    proposal_id: str
    vote: str


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
