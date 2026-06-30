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
pub mod aup;
pub mod consumer;
pub mod gateway;
pub mod hardening;
pub mod http;
pub mod metrics;
pub mod provider;
pub mod receipt;
pub mod serve;
pub mod telemetry;
pub mod workpool;

pub use adapter::{
    AdapterError, ChatMessage, DetectedModel, EngineAdapter, HttpClient, InferenceRequest,
    ServeOutcome,
};
pub use aup::{AupDecision, AupPolicy};
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
