// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! OpenAI-compatible engine adapter (protocol plan M3.2).
//!
//! One adapter for any engine that speaks the OpenAI HTTP API — vLLM, LM Studio, Exo,
//! llama.cpp's `--api`, LocalAI, and OpenAI-shaped proxies. Detection lists models via
//! `GET /v1/models`; serving streams `POST /v1/chat/completions` as SSE.
//!
//! Unlike Ollama, the OpenAI API exposes neither the chat template nor quant/param
//! metadata, so detected models carry an **empty canonical id** (advertised
//! uncanonicalised — the router still keeps them) and are addressed by the engine's own
//! model id. The completion-token count comes from the final `usage` chunk, requested via
//! `stream_options.include_usage`; the API reports no per-stage timings, so the returned
//! [`EngineMetrics`] durations are 0 (the pipeline measures end-to-end instead).
//!
//! Parsing is pure; HTTP is injected via [`HttpClient`](crate::adapter::HttpClient).

use serde::Deserialize;

use crate::adapter::{
    AdapterError, DetectedModel, EngineAdapter, EngineMetrics, HttpClient, InferenceRequest,
    ServeOutcome,
};

/// Default vLLM endpoint.
pub const DEFAULT_VLLM_URL: &str = "http://127.0.0.1:8000";
/// Default LM Studio endpoint.
pub const DEFAULT_LM_STUDIO_URL: &str = "http://127.0.0.1:1234";

// ── /v1/models (only the fields we use; unknown fields ignored) ──

#[derive(Debug, Default, Deserialize)]
struct ModelsResponse {
    #[serde(default)]
    data: Vec<ModelEntry>,
}

#[derive(Debug, Default, Deserialize)]
struct ModelEntry {
    /// The engine handle, e.g. `"Qwen/Qwen2.5-7B-Instruct"`.
    #[serde(default)]
    id: String,
}

// ── /v1/chat/completions streaming chunk (SSE `data:` lines of JSON) ──

#[derive(Debug, Default, Deserialize)]
struct ChatChunk {
    #[serde(default)]
    choices: Vec<ChunkChoice>,
    /// Present only on the final chunk, and only when `stream_options.include_usage` was
    /// requested (some servers also send it with empty `choices`).
    #[serde(default)]
    usage: Option<Usage>,
    /// Some servers stream an error object instead of a normal chunk.
    #[serde(default)]
    error: Option<serde_json::Value>,
}

#[derive(Debug, Default, Deserialize)]
struct ChunkChoice {
    #[serde(default)]
    delta: Delta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct Delta {
    /// Absent on the role-only first chunk and the final empty delta; `null` on some
    /// servers — `Option<String>` accepts all three.
    #[serde(default)]
    content: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct Usage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
}

/// Build the `/v1/chat/completions` body (always streaming, usage requested).
/// `max_tokens` / `temperature` map straight through; omitted when unset.
fn build_chat_body(req: &InferenceRequest) -> String {
    let messages: Vec<serde_json::Value> = req
        .messages
        .iter()
        .map(|m| serde_json::json!({ "role": m.role, "content": m.content }))
        .collect();
    let mut body = serde_json::json!({
        "model": req.model_ref,
        "messages": messages,
        "stream": true,
        "stream_options": { "include_usage": true },
    });
    if let Some(n) = req.max_tokens {
        body["max_tokens"] = serde_json::json!(n);
    }
    if let Some(t) = req.temperature {
        body["temperature"] = serde_json::json!(t);
    }
    body.to_string()
}

/// Strip the SSE `data:` framing from one line. Returns `None` for non-data lines
/// (comments `:`, blank framing, `event:`/`id:` fields) which the caller skips;
/// `Some("[DONE]")` for the end-of-stream sentinel; `Some(json)` for a chunk payload.
fn sse_payload(line: &str) -> Option<&str> {
    let rest = line.trim_end().strip_prefix("data:")?;
    Some(rest.trim_start())
}

/// Parse one chunk payload; an `error` object becomes an `AdapterError::Http` (the engine
/// refused / failed mid-stream), malformed JSON a parse error.
fn parse_chunk(payload: &str) -> Result<ChatChunk, AdapterError> {
    let chunk: ChatChunk =
        serde_json::from_str(payload).map_err(|e| AdapterError::Parse(e.to_string()))?;
    if let Some(err) = &chunk.error {
        return Err(AdapterError::Http(format!("openai: {err}")));
    }
    Ok(chunk)
}

/// Adapter for any OpenAI-compatible engine, generic over the injected HTTP transport.
pub struct OpenAiAdapter<H: HttpClient> {
    /// Server root (no trailing slash, no `/v1`); endpoints are appended as `/v1/...`.
    base_url: String,
    /// Engine label for [`EngineAdapter::engine_name`] (`"vllm"`, `"lm-studio"`, …).
    name: &'static str,
    http: H,
}

impl<H: HttpClient> OpenAiAdapter<H> {
    /// New adapter against `base_url` with an engine `name`. The base may include or omit a
    /// trailing `/v1` — both normalise to the server root, so callers can pass either the
    /// root or the OpenAI base URL.
    pub fn new(base_url: impl Into<String>, name: &'static str, http: H) -> Self {
        let root = base_url
            .into()
            .trim_end_matches('/')
            .trim_end_matches("/v1")
            .trim_end_matches('/')
            .to_string();
        Self { base_url: root, name, http }
    }

    /// vLLM convenience constructor.
    pub fn vllm(base_url: impl Into<String>, http: H) -> Self {
        Self::new(base_url, "vllm", http)
    }

    /// LM Studio convenience constructor.
    pub fn lm_studio(base_url: impl Into<String>, http: H) -> Self {
        Self::new(base_url, "lm-studio", http)
    }
}

impl<H: HttpClient> EngineAdapter for OpenAiAdapter<H> {
    fn engine_name(&self) -> &'static str {
        self.name
    }

    fn detect_models(&self) -> Result<Vec<DetectedModel>, AdapterError> {
        let json = self.http.get(&format!("{}/v1/models", self.base_url))?;
        let models: ModelsResponse =
            serde_json::from_str(&json).map_err(|e| AdapterError::Parse(e.to_string()))?;
        Ok(models
            .data
            .iter()
            .filter(|m| !m.id.trim().is_empty())
            .map(|m| DetectedModel {
                engine_ref: m.id.clone(),
                // The OpenAI API exposes no chat template or quant → uncanonicalised; the
                // model is still advertised and addressed by its engine id.
                canonical_id: String::new(),
                family: String::new(),
                params: String::new(),
                quant: String::new(),
                size_bytes: 0,
            })
            .collect())
    }

    fn serve_stream(
        &self,
        request: &InferenceRequest,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<ServeOutcome, AdapterError> {
        serve_chat_completions(
            &self.http,
            &format!("{}/v1/chat/completions", self.base_url),
            request,
            on_delta,
        )
    }
}

/// Stream a chat completion from any OpenAI-compatible `chat_url`
/// (`…/v1/chat/completions`). Factored out of [`OpenAiAdapter`] so the llama.cpp adapter
/// — which serves over the same OpenAI route but detects models bespoke-ly — reuses it.
pub(crate) fn serve_chat_completions<H: HttpClient>(
    http: &H,
    chat_url: &str,
    request: &InferenceRequest,
    on_delta: &mut dyn FnMut(&str),
) -> Result<ServeOutcome, AdapterError> {
    let body = build_chat_body(request);
    let lines = http.post_stream(chat_url, &body)?;

    let mut chunk_tokens = 0u64;
    let mut usage: Option<Usage> = None;
    let mut done = false;
    for line in lines {
        let line = line?;
        let Some(payload) = sse_payload(&line) else {
            continue; // SSE comment / framing / non-data field
        };
        if payload == "[DONE]" {
            done = true;
            break;
        }
        let chunk = parse_chunk(payload)?;
        for choice in &chunk.choices {
            if let Some(content) = &choice.delta.content {
                if !content.is_empty() {
                    on_delta(content);
                    chunk_tokens += 1;
                }
            }
            if choice.finish_reason.is_some() {
                // A clean stop; keep reading — the usage chunk may still follow.
                done = true;
            }
        }
        if chunk.usage.is_some() {
            usage = chunk.usage;
        }
    }

    let engine = EngineMetrics {
        // OpenAI exposes counts (when usage is included) but no per-stage timings.
        prompt_eval_count: usage.as_ref().map(|u| u.prompt_tokens).unwrap_or(0),
        eval_count: usage.as_ref().map(|u| u.completion_tokens).unwrap_or(0),
        ..EngineMetrics::default()
    };
    // Prefer the engine's authoritative completion-token count; fall back to the number
    // of non-empty content chunks when the server reports no usage.
    let tokens = usage
        .as_ref()
        .map(|u| u.completion_tokens)
        .filter(|&t| t > 0)
        .unwrap_or(chunk_tokens);
    Ok(ServeOutcome { tokens, done, engine })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::ChatMessage;

    const MODELS_FIXTURE: &str = r#"{
        "object": "list",
        "data": [
            { "id": "Qwen/Qwen2.5-7B-Instruct", "object": "model", "owned_by": "vllm" },
            { "id": "meta-llama/Llama-3.1-8B-Instruct", "object": "model" }
        ]
    }"#;

    /// Inject a canned `/v1/models` body and `/v1/chat/completions` SSE lines (no network).
    #[derive(Default)]
    struct MockHttp {
        models: String,
        stream_lines: Vec<String>,
    }

    impl HttpClient for MockHttp {
        fn get(&self, url: &str) -> Result<String, AdapterError> {
            if url.ends_with("/v1/models") {
                Ok(self.models.clone())
            } else {
                Err(AdapterError::Http(format!("unexpected GET {url}")))
            }
        }
        fn post_json(&self, url: &str, _body: &str) -> Result<String, AdapterError> {
            Err(AdapterError::Http(format!("unexpected POST {url}")))
        }
        fn post_stream(
            &self,
            url: &str,
            _body: &str,
        ) -> Result<Box<dyn Iterator<Item = Result<String, AdapterError>>>, AdapterError> {
            assert!(url.ends_with("/v1/chat/completions"), "stream url: {url}");
            let lines: Vec<Result<String, AdapterError>> =
                self.stream_lines.iter().cloned().map(Ok).collect();
            Ok(Box::new(lines.into_iter()))
        }
    }

    fn req() -> InferenceRequest {
        InferenceRequest {
            model_ref: "Qwen/Qwen2.5-7B-Instruct".into(),
            messages: vec![ChatMessage { role: "user".into(), content: "hi".into() }],
            max_tokens: Some(16),
            temperature: Some(0.0),
        }
    }

    fn serve(lines: &[&str]) -> Result<(String, ServeOutcome), AdapterError> {
        let http = MockHttp {
            stream_lines: lines.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        };
        let adapter = OpenAiAdapter::vllm("http://127.0.0.1:8000", http);
        let mut out = String::new();
        let outcome = adapter.serve_stream(&req(), &mut |d| out.push_str(d))?;
        Ok((out, outcome))
    }

    #[test]
    fn detect_models_lists_ids_uncanonicalised() {
        let http = MockHttp { models: MODELS_FIXTURE.into(), ..Default::default() };
        let adapter = OpenAiAdapter::vllm("http://127.0.0.1:8000", http);
        let models = adapter.detect_models().unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].engine_ref, "Qwen/Qwen2.5-7B-Instruct");
        assert_eq!(models[1].engine_ref, "meta-llama/Llama-3.1-8B-Instruct");
        // No chat template / quant over the OpenAI API → empty canonical id.
        assert!(models.iter().all(|m| m.canonical_id.is_empty()));
        assert_eq!(adapter.engine_name(), "vllm");
    }

    #[test]
    fn serve_stream_concatenates_deltas_and_reports_usage_token_count() {
        let (out, outcome) = serve(&[
            r#"data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"delta":{"content":"Hel"},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"delta":{"content":"lo"},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
            r#"data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2}}"#,
            "data: [DONE]",
        ])
        .unwrap();
        assert_eq!(out, "Hello");
        assert_eq!(outcome.tokens, 2); // authoritative usage count, not chunk count
        assert!(outcome.done);
        assert_eq!(outcome.engine.prompt_eval_count, 10);
        assert_eq!(outcome.engine.eval_count, 2);
    }

    #[test]
    fn serve_stream_falls_back_to_chunk_count_without_usage() {
        let (out, outcome) = serve(&[
            r#"data: {"choices":[{"delta":{"content":"a"},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"delta":{"content":"b"},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"delta":{"content":"c"},"finish_reason":"stop"}]}"#,
            "data: [DONE]",
        ])
        .unwrap();
        assert_eq!(out, "abc");
        assert_eq!(outcome.tokens, 3); // no usage reported → count content chunks
        assert!(outcome.done);
    }

    #[test]
    fn serve_stream_skips_comments_and_blank_lines_and_stops_at_done() {
        let (out, outcome) = serve(&[
            ": keep-alive ping",
            "",
            r#"data: {"choices":[{"delta":{"content":"x"},"finish_reason":null}]}"#,
            "data: [DONE]",
            r#"data: {"choices":[{"delta":{"content":"SHOULD-NOT-APPEAR"}}]}"#,
        ])
        .unwrap();
        assert_eq!(out, "x"); // nothing after [DONE] is read
        assert!(outcome.done);
    }

    #[test]
    fn serve_stream_surfaces_engine_error_chunk() {
        let err = serve(&[r#"data: {"error":{"message":"model not found","type":"NotFoundError"}}"#])
            .unwrap_err();
        assert!(matches!(err, AdapterError::Http(_)), "got {err:?}");
    }

    #[test]
    fn serve_stream_malformed_chunk_is_a_parse_error() {
        let err = serve(&["data: {not valid json"]).unwrap_err();
        assert!(matches!(err, AdapterError::Parse(_)), "got {err:?}");
    }

    #[test]
    fn base_url_normalises_a_trailing_v1() {
        // A caller that passes the OpenAI base URL (…/v1/) must not produce …/v1/v1/models.
        struct UrlAssertHttp;
        impl HttpClient for UrlAssertHttp {
            fn get(&self, url: &str) -> Result<String, AdapterError> {
                assert_eq!(url, "http://host:8000/v1/models");
                Ok(r#"{"data":[]}"#.into())
            }
            fn post_json(&self, _u: &str, _b: &str) -> Result<String, AdapterError> {
                unreachable!()
            }
            fn post_stream(
                &self,
                _u: &str,
                _b: &str,
            ) -> Result<Box<dyn Iterator<Item = Result<String, AdapterError>>>, AdapterError>
            {
                unreachable!()
            }
        }
        let adapter = OpenAiAdapter::new("http://host:8000/v1/", "vllm", UrlAssertHttp);
        adapter.detect_models().unwrap();
    }
}
