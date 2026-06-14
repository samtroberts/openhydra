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
//! Each engine is wrapped by an [`adapter::EngineAdapter`]. The first is
//! [`ollama::OllamaAdapter`]. Adapters keep their *pure* logic (parse the engine's
//! responses → canonical ids) separate from I/O: HTTP is injected via
//! [`adapter::HttpClient`], so detection + mapping are unit-tested against fixtures with
//! no live engine and no `reqwest`/`tokio` in the dependency tree yet.

pub mod adapter;
pub mod consumer;
pub mod gateway;
pub mod http;
pub mod ollama;
pub mod provider;
pub mod serve;

pub use adapter::{
    AdapterError, ChatMessage, DetectedModel, EngineAdapter, HttpClient, InferenceRequest,
    ServeOutcome,
};
pub use consumer::{request_completion, select_provider, ConsumerNode, SelectedProvider};
pub use gateway::serve_http;
pub use http::ReqwestClient;
pub use ollama::OllamaAdapter;
pub use provider::{build_peer_record, handle_serve_inbound, Provider, SERVE_REQUEST};
pub use serve::{handle_serve_request, ServeChunk, ServeRequest, ServeSummary};

/// An Ollama adapter backed by the live reqwest transport, pointed at `base_url`
/// (e.g. [`ollama::DEFAULT_OLLAMA_URL`]). The convenience entry point for a provider.
pub fn live_ollama(base_url: &str) -> Result<OllamaAdapter<ReqwestClient>, AdapterError> {
    Ok(OllamaAdapter::new(base_url, ReqwestClient::new()?))
}
