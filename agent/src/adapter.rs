// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! The engine-adapter abstraction (BYO-engine).
//!
//! Every supported engine implements [`EngineAdapter`]: it can *detect* the models the
//! engine currently serves (mapped to canonical ids for swarm advertisement) and —
//! later — *serve* an inbound request by proxying it to the engine. HTTP I/O is
//! injected via [`HttpClient`] so the pure detection/mapping logic is testable without a
//! live engine.

use std::fmt;

/// An error from talking to, or interpreting, an engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdapterError {
    /// The HTTP transport failed (connection refused, timeout, non-2xx, …).
    Http(String),
    /// A response could not be parsed into the expected shape.
    Parse(String),
}

impl fmt::Display for AdapterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AdapterError::Http(m) => write!(f, "engine http error: {m}"),
            AdapterError::Parse(m) => write!(f, "engine response parse error: {m}"),
        }
    }
}

impl std::error::Error for AdapterError {}

/// A minimal, synchronous HTTP transport the adapters call. Injected so detection is
/// unit-testable against fixtures; the live `reqwest`-backed implementation lands with
/// the streaming completion proxy (which will add the async surface separately).
pub trait HttpClient {
    /// `GET {url}` → response body as a string.
    fn get(&self, url: &str) -> Result<String, AdapterError>;
    /// `POST {url}` with a JSON `body` → response body as a string.
    fn post_json(&self, url: &str, body: &str) -> Result<String, AdapterError>;
}

/// A model an engine currently serves, ready to advertise to the swarm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetectedModel {
    /// How the *engine* names this model (e.g. Ollama's `"qwen2.5:7b"`) — the handle the
    /// adapter uses to address it when serving. Distinct from the canonical id.
    pub engine_ref: String,
    /// The protocol canonical id `family/params/quant/template_hash`, or `""` when it
    /// can't be determined (e.g. the engine exposes no chat template). An empty id is
    /// advertised as a legacy/uncanonicalised provider — the router still keeps it.
    pub canonical_id: String,
    /// Best-effort family / params / quant as reported by the engine (pre-canonicalisation),
    /// kept for observability and capability records.
    pub family: String,
    pub params: String,
    pub quant: String,
    /// On-disk size in bytes, if the engine reports it (0 otherwise).
    pub size_bytes: u64,
}

/// A wrapper around one local inference engine.
pub trait EngineAdapter {
    /// Short engine name, e.g. `"ollama"`.
    fn engine_name(&self) -> &'static str;

    /// Detect the models the engine currently serves, mapped to canonical ids for
    /// advertisement. Returns an empty list if the engine serves nothing.
    fn detect_models(&self) -> Result<Vec<DetectedModel>, AdapterError>;

    // TODO(M3.1 next): `serve` — proxy an inbound chat/completion request to the engine
    // and stream tokens back. That adds the async transport; detection stays sync.
}
