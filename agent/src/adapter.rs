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
use serde_json::Value;

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
///
/// `content` tolerates an explicit JSON `null` on deserialize (OpenAI sends
/// `content: null` on an assistant turn that only made tool calls) → the empty string, so
/// a coding-agent request never 400s on a well-formed tool-call turn.
///
/// `tool_calls` / `tool_call_id` / `name` carry a multi-turn tool exchange across the wire:
/// an assistant turn that requested tools (`tool_calls`), and the `role:"tool"` result
/// messages that answer them (`tool_call_id` + `content`). All three are absent on an
/// ordinary user/assistant message and `skip_serializing_if`-elided, so a plain message
/// serialises exactly as before (older providers see an unchanged shape).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default, deserialize_with = "de_string_or_null")]
    pub content: String,
    /// Assistant turn: the tool calls the model previously requested (OpenAI `tool_calls`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// `role:"tool"` message: which tool call this is the result of (OpenAI `tool_call_id`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Optional author / tool name (OpenAI `name`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    /// A plain text message (no tool-call state) — the common case.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self { role: role.into(), content: content.into(), ..Default::default() }
    }
}

/// Deserialize an OpenAI message `content` into a plain string, accepting every shape a client
/// may send it as:
/// - a string (the common case),
/// - JSON `null` (an assistant tool-call turn) → `""`,
/// - an **array of content parts** (`[{ "type": "text", "text": "…" }, …]`) → the text parts
///   concatenated. This is valid OpenAI spec — modern clients (e.g. Pi) send even plain text this
///   way — and the gateway would otherwise 400 with "invalid type: sequence, expected a string".
///   Non-text parts (image_url, etc.) are dropped: the downstream text engines can't consume them.
///
/// `#[serde(default)]` alone only covers an *absent* key; this also maps present null/array values.
fn de_string_or_null<'de, D>(de: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(flatten_content(&serde_json::Value::deserialize(de)?))
}

/// Flatten an OpenAI `content` value (string | null | content-parts array) to a plain string.
fn flatten_content(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(parts) => parts
            .iter()
            .filter_map(|p| {
                // A content part is `{ "type": "text", "text": "…" }`; keep its text. Some clients
                // send bare strings in the array — keep those too. Anything else (image_url, …) drops.
                p.get("text").and_then(|t| t.as_str()).or_else(|| p.as_str())
            })
            .collect::<Vec<_>>()
            .join(""),
        // null (and any other unexpected shape) → empty.
        _ => String::new(),
    }
}

/// One OpenAI tool call the model requested (`type: "function"`). `arguments` is the raw
/// JSON *string* the model emitted (the OpenAI wire shape) — passed through verbatim so we
/// never reshape or lose what the model produced. Terminal, structured data: it rides the
/// [`ServeOutcome`] return value, not the text `on_delta` stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type", default = "tool_call_type_function")]
    pub kind: String,
    pub function: ToolCallFunction,
}

/// The `function` payload of a [`ToolCall`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    /// JSON-encoded argument object, kept as a string (OpenAI convention).
    #[serde(default)]
    pub arguments: String,
}

fn tool_call_type_function() -> String {
    "function".to_string()
}

/// A request to serve a streaming chat completion from an engine.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct InferenceRequest {
    /// The engine's handle for the model (e.g. Ollama's `"qwen2.5:7b"`) — i.e.
    /// [`DetectedModel::engine_ref`].
    pub model_ref: String,
    pub messages: Vec<ChatMessage>,
    /// Cap on generated tokens, if any.
    pub max_tokens: Option<u32>,
    pub temperature: Option<f64>,
    /// OpenAI `tools` specs, forwarded opaquely to the engine (empty ⇒ no tools). Carried
    /// as raw JSON so the adapter hands the engine exactly what the client sent.
    pub tools: Vec<Value>,
    /// Deterministic thinking-mode control for reasoning models: `Some(false)` disables the
    /// chain-of-thought, `Some(true)` forces it, `None` uses the engine's default. Adapters
    /// that support it map it to their native switch (Ollama's top-level `think`); adapters
    /// with no thinking control ignore it.
    pub think: Option<bool>,
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
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ServeOutcome {
    /// Completion tokens generated — the engine's own count where it reports one (Ollama
    /// `eval_count`), else the number of non-empty content chunks emitted. Feeds the
    /// co-signed receipt's token count.
    pub tokens: u64,
    /// Whether the engine signalled a clean end-of-stream (vs the stream just ending).
    pub done: bool,
    /// The engine's own per-request metrics (Ollama `eval_*`/`prompt_eval_*`/`load_*`).
    pub engine: EngineMetrics,
    /// Tool calls the model requested this turn (OpenAI `tool_calls`), empty when none.
    /// Terminal structured data — carried on the return value, not via `on_delta` (which is
    /// text-only). The provider relays these back as a [`ServeChunk::ToolCalls`] frame.
    pub tool_calls: Vec<ToolCall>,
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

/// Does `s` end in a `.gguf` extension, case-insensitively? Byte-based so it never panics on a
/// non-char boundary — the last 5 bytes of a `.gguf` suffix are ASCII, so slicing at `len-5` is safe.
fn ends_with_gguf(s: &str) -> bool {
    s.len() >= 5 && s.as_bytes()[s.len() - 5..].eq_ignore_ascii_case(b".gguf")
}

/// Normalise an engine handle into a clean, path-free id to advertise (and to key the share policy
/// on). `llama-server` reports the model id as whatever it was launched with — frequently the
/// absolute `-m` path (`/home/alice/models/Qwen3.5-9B-Q4_K_M.gguf`), which would leak the operator's
/// home dir + OS username onto the network and read as an unreadable name. Reduce a genuine
/// filesystem path (a `.gguf` file, or an absolute / home / drive path) to its GGUF basename without
/// the extension; a namespaced logical id (an Ollama tag `llama3.2:1b`, an HF `Qwen/Qwen2.5-7B`) has
/// no such marker and passes through untouched. Idempotent: a clean id normalises to itself.
pub fn normalize_engine_ref(id: &str) -> String {
    let bytes = id.as_bytes();
    let is_windows_drive =
        bytes.get(1) == Some(&b':') && matches!(bytes.get(2), Some(b'/') | Some(b'\\'));
    let looks_like_path =
        ends_with_gguf(id) || id.starts_with('/') || id.starts_with('~') || is_windows_drive;
    if !looks_like_path {
        return id.to_string();
    }
    let base = id.rsplit(['/', '\\']).next().unwrap_or(id);
    if ends_with_gguf(base) {
        base[..base.len() - 5].to_string()
    } else {
        base.to_string()
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_tolerates_explicit_null() {
        // OpenAI sends `content: null` on an assistant turn that only made tool calls — that
        // must deserialize (→ "") rather than 400 the request.
        let m: ChatMessage = serde_json::from_str(r#"{"role":"assistant","content":null}"#).unwrap();
        assert_eq!(m.content, "");
        // An absent `content` key also defaults to empty.
        let m: ChatMessage = serde_json::from_str(r#"{"role":"user"}"#).unwrap();
        assert_eq!(m.content, "");
        // A normal string is preserved unchanged.
        let m: ChatMessage = serde_json::from_str(r#"{"role":"user","content":"hi"}"#).unwrap();
        assert_eq!(m.content, "hi");
    }

    #[test]
    fn content_accepts_openai_content_parts_array() {
        // Modern OpenAI clients (e.g. Pi) send even plain text as a content-parts array. The
        // gateway must flatten the text parts instead of 400ing "invalid type: sequence".
        let m: ChatMessage = serde_json::from_str(
            r#"{"role":"user","content":[{"type":"text","text":"hello "},{"type":"text","text":"world"}]}"#,
        )
        .unwrap();
        assert_eq!(m.content, "hello world");
        // Non-text parts (image_url, …) are dropped; text is kept.
        let m: ChatMessage = serde_json::from_str(
            r#"{"role":"user","content":[{"type":"text","text":"describe"},{"type":"image_url","image_url":{"url":"data:…"}}]}"#,
        )
        .unwrap();
        assert_eq!(m.content, "describe");
        // An empty parts array → empty string (not an error).
        let m: ChatMessage = serde_json::from_str(r#"{"role":"user","content":[]}"#).unwrap();
        assert_eq!(m.content, "");
    }

    #[test]
    fn plain_message_serialises_without_tool_fields() {
        // A plain user message must serialise exactly as before — no null tool_* keys — so
        // older providers see an unchanged shape.
        let v: Value = serde_json::to_value(ChatMessage::new("user", "hi")).unwrap();
        assert_eq!(v, serde_json::json!({ "role": "user", "content": "hi" }));
    }

    #[test]
    fn multi_turn_tool_message_round_trips() {
        // An assistant tool-call turn + a role:"tool" result must round-trip across the wire.
        let assistant = ChatMessage {
            role: "assistant".into(),
            content: String::new(),
            tool_calls: Some(vec![ToolCall {
                id: "call_1".into(),
                kind: "function".into(),
                function: ToolCallFunction { name: "get_weather".into(), arguments: r#"{"city":"SF"}"#.into() },
            }]),
            ..Default::default()
        };
        let tool = ChatMessage {
            role: "tool".into(),
            content: "72F".into(),
            tool_call_id: Some("call_1".into()),
            name: Some("get_weather".into()),
            ..Default::default()
        };
        for m in [assistant, tool] {
            let back: ChatMessage = serde_json::from_str(&serde_json::to_string(&m).unwrap()).unwrap();
            assert_eq!(back, m);
        }
    }

    #[test]
    fn tool_call_uses_the_openai_wire_shape() {
        let tc = ToolCall {
            id: "call_1".into(),
            kind: "function".into(),
            function: ToolCallFunction {
                name: "get_weather".into(),
                arguments: r#"{"city":"SF"}"#.into(),
            },
        };
        let v: Value = serde_json::from_str(&serde_json::to_string(&tc).unwrap()).unwrap();
        assert_eq!(v["type"], "function"); // `kind` serializes back out as `type`
        assert_eq!(v["function"]["name"], "get_weather");
        // `arguments` stays a JSON *string*, not a parsed object (the OpenAI convention).
        assert!(v["function"]["arguments"].is_string());

        // A provider may omit `type` and `arguments`; both default sanely on the way in.
        let back: ToolCall = serde_json::from_str(r#"{"id":"c","function":{"name":"f"}}"#).unwrap();
        assert_eq!(back.kind, "function");
        assert_eq!(back.function.arguments, "");
    }
}
