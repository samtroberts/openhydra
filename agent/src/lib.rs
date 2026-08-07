// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! OpenHydra agent — the pure-protocol host (M3.1+).
//!
//! OpenHydra does **not** run models. A provider's agent proxies inference to whatever
//! engine that operator already runs locally (Ollama, vLLM, LM Studio, llama.cpp) over
//! the engine's HTTP API, advertises those models to the swarm by their canonical id
//! ([`openhydra_protocol::model_id`]), and streams results back over libp2p. This crate
//! holds the **engine adapters** and (later) the HTTP/SSE gateway.
//!
//! Each engine is wrapped by an [`adapter::EngineAdapter`], gathered under [`adapters`]:
//! [`adapters::ollama::OllamaAdapter`] (Ollama's native API) and
//! [`adapters::openai::OpenAiAdapter`] (any OpenAI-compatible server — vLLM, LM Studio,
//! Exo, …). Adapters keep their *pure* logic (parse the engine's responses → canonical
//! ids) separate from I/O: HTTP is injected via [`adapter::HttpClient`], so detection +
//! mapping are unit-tested against fixtures with no live engine.

pub mod adapter;
pub mod adapters;
/// Opt-in engine autostart (starts LM Studio / Ollama when down). Feature-gated so the lean
/// build carries no process-spawning code.
#[cfg(feature = "engine-autostart")]
pub mod autostart;
pub mod aup;
/// Engine auto-detection (`--engine-kind auto`) + the multi-engine union adapter.
pub mod detect;
pub mod byok;
pub mod consumer;
pub mod gateway;
pub mod hardening;
pub mod http;
pub mod metrics;
pub mod provider;
pub mod ratelimit;
pub mod receipt;
pub mod serve;
/// P0 introspection: the `--status-bind` endpoint (network snapshot + transfer counters).
pub mod status;
pub mod telemetry;
pub mod workpool;

pub use adapter::{
    AdapterError, ChatMessage, DetectedModel, EmbeddingAdapter, EmbeddingResponse, EngineAdapter,
    HttpClient, InferenceRequest, ServeOutcome,
};
pub use adapters::embeddings::{OpenAiEmbeddingAdapter, DEFAULT_OPENAI_EMBEDDINGS_URL};
pub use aup::{AupDecision, AupPolicy};
pub use byok::{ByokConfig, ByokProvider, EmbeddingConfig};
pub use ratelimit::{RateLimitConfig, RateLimiter};
pub use adapters::anthropic::{AnthropicAdapter, DEFAULT_ANTHROPIC_URL};
pub use adapters::comfyui::{ComfyUiAdapter, DEFAULT_COMFYUI_URL};
pub use adapters::exo::{ExoAdapter, DEFAULT_EXO_URL};
pub use adapters::gemini::{GeminiAdapter, DEFAULT_GEMINI_URL};
pub use adapters::llama_cpp::{LlamaCppAdapter, DEFAULT_LLAMACPP_URL};
pub use adapters::ollama::{OllamaAdapter, DEFAULT_OLLAMA_URL};
pub use adapters::openai::{OpenAiAdapter, DEFAULT_LM_STUDIO_URL, DEFAULT_VLLM_URL};
pub use consumer::{
    default_challenge, rank_providers, rank_providers_with_reputation, request_completion,
    select_provider, ConsumerNode, SelectedProvider,
};
pub use gateway::serve_http;
pub use hardening::harden_process;
pub use http::ReqwestClient;
pub use provider::{build_peer_record, handle_serve_inbound, Provider, SERVE_REQUEST};
pub use status::{CreditEntry, EconomyStats, EconomyView, LedgerRow, RepEntry, StatusServer, TransferStats};
pub use serve::{handle_serve_request, ServeChunk, ServeRequest, ServeSummary};

/// An Ollama adapter backed by the live reqwest transport, pointed at `base_url`
/// (e.g. [`DEFAULT_OLLAMA_URL`]). The convenience entry point for an Ollama provider.
pub fn live_ollama(base_url: &str) -> Result<OllamaAdapter<ReqwestClient>, AdapterError> {
    Ok(OllamaAdapter::new(base_url, ReqwestClient::new()?))
}

/// An OpenAI-compatible adapter backed by the live reqwest transport, labelled `name`
/// (e.g. `"vllm"`, `"lm-studio"`). Covers vLLM, LM Studio, Exo, llama.cpp `--api`, and
/// any OpenAI-shaped server at `base_url`.
pub fn live_openai(
    base_url: &str,
    name: &'static str,
) -> Result<OpenAiAdapter<ReqwestClient>, AdapterError> {
    Ok(OpenAiAdapter::new(base_url, name, ReqwestClient::new()?))
}

/// A llama.cpp (`llama-server`) adapter backed by the live reqwest transport, pointed at
/// `base_url` (e.g. [`DEFAULT_LLAMACPP_URL`]).
pub fn live_llamacpp(base_url: &str) -> Result<LlamaCppAdapter<ReqwestClient>, AdapterError> {
    Ok(LlamaCppAdapter::new(base_url, ReqwestClient::new()?))
}

/// A ComfyUI adapter backed by the live reqwest transport, pointed at `base_url`
/// (e.g. [`DEFAULT_COMFYUI_URL`]). Announces Stable-Diffusion checkpoints as models.
pub fn live_comfyui(base_url: &str) -> Result<ComfyUiAdapter<ReqwestClient>, AdapterError> {
    Ok(ComfyUiAdapter::new(base_url, ReqwestClient::new()?))
}

/// A ComfyUI adapter in **BYO-workflow** mode: load the provider's API-format workflow
/// templates from `dir` (each `*.json` with a `%prompt%` marker becomes a model) and inject
/// the prompt at serve time. Makes the adapter model-agnostic (Flux, SDXL, video, …).
pub fn live_comfyui_with_workflows(
    base_url: &str,
    dir: &std::path::Path,
) -> Result<ComfyUiAdapter<ReqwestClient>, AdapterError> {
    let templates = crate::adapters::comfyui::load_workflow_templates(dir)?;
    Ok(ComfyUiAdapter::with_templates(base_url, ReqwestClient::new()?, templates))
}

/// An Exo cluster adapter backed by the live reqwest transport, pointed at the head node's
/// `base_url` (e.g. [`DEFAULT_EXO_URL`]). Detects only placed-and-ready models via `/state`.
pub fn live_exo(base_url: &str) -> Result<ExoAdapter<ReqwestClient>, AdapterError> {
    Ok(ExoAdapter::new(base_url, ReqwestClient::new()?))
}

/// An Anthropic (Claude) BYOK adapter backed by the live reqwest transport — a hosted
/// passthrough backend with the operator's `api_key`. `base_url` is usually
/// [`DEFAULT_ANTHROPIC_URL`].
pub fn live_anthropic(
    base_url: &str,
    api_key: &str,
) -> Result<AnthropicAdapter<ReqwestClient>, AdapterError> {
    Ok(AnthropicAdapter::new(base_url, api_key, ReqwestClient::new()?))
}

/// A Google Gemini BYOK adapter backed by the live reqwest transport, with the operator's
/// `api_key`. `base_url` is usually [`DEFAULT_GEMINI_URL`].
pub fn live_gemini(
    base_url: &str,
    api_key: &str,
) -> Result<GeminiAdapter<ReqwestClient>, AdapterError> {
    Ok(GeminiAdapter::new(base_url, api_key, ReqwestClient::new()?))
}

/// An OpenAI-compatible embeddings BYOK adapter backed by the live reqwest transport, with the
/// operator's `api_key`. `base_url` is usually [`DEFAULT_OPENAI_EMBEDDINGS_URL`] (or a
/// Gemini-OAI-compat / Voyage / local endpoint).
pub fn live_openai_embeddings(
    base_url: &str,
    api_key: &str,
) -> Result<OpenAiEmbeddingAdapter<ReqwestClient>, AdapterError> {
    Ok(OpenAiEmbeddingAdapter::new(base_url, api_key, ReqwestClient::new()?))
}
