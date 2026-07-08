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

use serde::{Deserialize, Serialize};

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
    /// `POST {url}` with a JSON `body`, streaming the response as a **lazy** iterator of
    /// raw chunk lines (e.g. Ollama's newline-delimited JSON). The iterator owns its
    /// reader so it can be pulled as tokens arrive — the point of streaming, not a
    /// buffered `Vec`. The live impl wraps `reqwest` in a `BufReader::lines()`; tests
    /// hand back fixture lines.
    fn post_stream(
        &self,
        url: &str,
        body: &str,
    ) -> Result<Box<dyn Iterator<Item = Result<String, AdapterError>>>, AdapterError>;

    /// `GET {url}` → raw response bytes — for engines whose outputs are binary (ComfyUI
    /// images). Default errors so text-only mocks/transports are untouched; the live
    /// transport overrides it.
    fn get_bytes(&self, url: &str) -> Result<Vec<u8>, AdapterError> {
        let _ = url;
        Err(AdapterError::Http("binary GET not supported by this transport".into()))
    }

    /// `GET {url}` with extra request `headers` — e.g. an auth header for a hosted BYOK
    /// backend (Anthropic `x-api-key`, Gemini `x-goog-api-key`). The default drops the
    /// headers and calls [`get`](Self::get); the live transport overrides it so a key is
    /// actually sent. (Local engines need no auth, so existing adapters/mocks are untouched.)
    fn get_with_headers(
        &self,
        url: &str,
        _headers: &[(&str, &str)],
    ) -> Result<String, AdapterError> {
        self.get(url)
    }

    /// `POST {url}` (non-streaming JSON) with extra request `headers` — used by the hosted
    /// embeddings adapters (an `Authorization: Bearer` key). Default delegates to
    /// [`post_json`](Self::post_json); the live transport overrides it to send them.
    fn post_json_with_headers(
        &self,
        url: &str,
        body: &str,
        _headers: &[(&str, &str)],
    ) -> Result<String, AdapterError> {
        self.post_json(url, body)
    }

    /// `POST {url}` streaming with extra request `headers`. Default delegates to
    /// [`post_stream`](Self::post_stream); the live transport overrides it to send them.
    fn post_stream_with_headers(
        &self,
        url: &str,
        body: &str,
        _headers: &[(&str, &str)],
    ) -> Result<Box<dyn Iterator<Item = Result<String, AdapterError>>>, AdapterError> {
        self.post_stream(url, body)
    }
}

/// One chat message in an inference request. Serializable — it crosses the swarm as
/// part of a serve request (see [`crate::serve`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// A request to serve a streaming chat completion from an engine.
#[derive(Debug, Clone, PartialEq)]
pub struct InferenceRequest {
    /// The engine's handle for the model (e.g. Ollama's `"qwen2.5:7b"`) — i.e.
    /// [`DetectedModel::engine_ref`].
    pub model_ref: String,
    pub messages: Vec<ChatMessage>,
    /// Cap on generated tokens, if any.
    pub max_tokens: Option<u32>,
    pub temperature: Option<f64>,
}

/// Raw counters/timings the engine itself reports for one request — Ollama's
/// `total_duration` / `load_duration` / `prompt_eval_*` / `eval_*`. All durations are
/// nanoseconds; 0 where the engine reports nothing. This is the **provider-local ground
/// truth**: the pipeline cannot change these numbers, only add transport overhead on top.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineMetrics {
    /// Whole-request time on the engine (load + prefill + generation + its own overhead).
    pub total_duration_ns: u64,
    /// Time to load the model into memory/VRAM (large on a cold start, ~0 when warm).
    pub load_duration_ns: u64,
    /// Prompt (input) tokens.
    pub prompt_eval_count: u64,
    /// Prefill time — evaluating the prompt before the first output token.
    pub prompt_eval_duration_ns: u64,
    /// Generated (output) tokens.
    pub eval_count: u64,
    /// Generation time for the output tokens (the basis for native gen-TPS).
    pub eval_duration_ns: u64,
}

/// The result of a served stream, after the last token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServeOutcome {
    /// Completion tokens generated — the engine's own count where it reports one (Ollama
    /// `eval_count`), else the number of non-empty content chunks emitted. Feeds the
    /// co-signed receipt's token count.
    pub tokens: u64,
    /// Whether the engine signalled a clean end-of-stream (vs the stream just ending).
    pub done: bool,
    /// The engine's own per-request metrics (Ollama `eval_*`/`prompt_eval_*`/`load_*`).
    pub engine: EngineMetrics,
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

    /// Proxy a streaming chat completion to the engine, invoking `on_delta` with each
    /// text fragment as it arrives, and returning the [`ServeOutcome`] when the stream
    /// ends. `on_delta` is `&mut dyn` (not generic) so the trait stays object-safe — the
    /// gateway holds adapters behind `dyn EngineAdapter`. The caller forwards each delta
    /// to its SSE response and uses `ServeOutcome.tokens` for the receipt.
    fn serve_stream(
        &self,
        request: &InferenceRequest,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<ServeOutcome, AdapterError>;
}

/// The result of an embeddings request: one vector per input, in input order, plus the
/// backend's prompt-token count (0 if it reports none).
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingResponse {
    pub vectors: Vec<Vec<f32>>,
    pub prompt_tokens: u64,
}

/// A backend that produces embedding vectors (non-streaming — distinct from the chat
/// [`EngineAdapter`]). Implemented by the BYOK embeddings adapters.
pub trait EmbeddingAdapter {
    /// Embed each of `inputs` with `model`, returning one vector per input (in order).
    fn embed(&self, model: &str, inputs: &[String]) -> Result<EmbeddingResponse, AdapterError>;
}
