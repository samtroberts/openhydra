// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Anthropic (Claude) BYOK adapter — the hosted Messages API.
//!
//! Unlike the local-engine adapters, this calls a **hosted** API with the operator's key, so
//! it is used on the *gateway* side as a passthrough backend, not the swarm-provider side.
//! Detection lists models via `GET /v1/models`; serving streams `POST /v1/messages`
//! (`stream: true`) and parses Anthropic's typed SSE events.
//!
//! Mapping notes: Anthropic takes the system prompt as a top-level `system` field (not a
//! message role) and **requires** `max_tokens`, so `system`-role messages are lifted out and
//! an unset `max_tokens` falls back to [`DEFAULT_MAX_TOKENS`]. Output-token count comes from
//! the `message_delta` event's `usage`; input tokens from `message_start`.
//!
//! Parsing is pure; HTTP + auth headers are injected via [`HttpClient`].

use serde::Deserialize;

use crate::adapter::{
    AdapterError, DetectedModel, EngineAdapter, EngineMetrics, HttpClient, InferenceRequest,
    ServeOutcome,
};

/// Anthropic API root.
pub const DEFAULT_ANTHROPIC_URL: &str = "https://api.anthropic.com";
/// The `anthropic-version` header value the Messages API expects.
const ANTHROPIC_VERSION: &str = "2023-06-01";
/// Fallback when the request sets no `max_tokens` (the API requires the field).
const DEFAULT_MAX_TOKENS: u32 = 4096;

// ── GET /v1/models (only the fields we use) ──
#[derive(Debug, Default, Deserialize)]
struct ModelsResponse {
    #[serde(default)]
    data: Vec<ModelEntry>,
}
#[derive(Debug, Default, Deserialize)]
struct ModelEntry {
    #[serde(default)]
    id: String,
}

// ── /v1/messages SSE events (one JSON object per `data:` line, dispatched on `type`) ──
#[derive(Debug, Default, Deserialize)]
struct StreamEvent {
    #[serde(rename = "type", default)]
    kind: String,
    #[serde(default)]
    delta: Option<EventDelta>,
    /// On `message_delta` — the running/final output-token count.
    #[serde(default)]
    usage: Option<Usage>,
    /// On `message_start` — nests the initial usage (input tokens).
    #[serde(default)]
    message: Option<MessageStart>,
    #[serde(default)]
    error: Option<serde_json::Value>,
}
#[derive(Debug, Default, Deserialize)]
struct EventDelta {
    #[serde(default)]
    text: Option<String>,
}
#[derive(Debug, Default, Deserialize)]
struct Usage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
}
#[derive(Debug, Default, Deserialize)]
struct MessageStart {
    #[serde(default)]
    usage: Option<Usage>,
}

/// Build the `/v1/messages` request body: lift `system`-role messages into the top-level
/// `system` field, map the rest to `user`/`assistant`, and always stream.
fn build_body(model: &str, request: &InferenceRequest) -> String {
    let mut system = String::new();
    let mut messages: Vec<serde_json::Value> = Vec::new();
    for m in &request.messages {
        match m.role.as_str() {
            "system" => {
                if !system.is_empty() {
                    system.push('\n');
                }
                system.push_str(&m.content);
            }
            "assistant" => messages.push(serde_json::json!({ "role": "assistant", "content": m.content })),
            // Anthropic only accepts user/assistant; anything else is treated as user.
            _ => messages.push(serde_json::json!({ "role": "user", "content": m.content })),
        }
    }
    let mut body = serde_json::json!({
        "model": model,
        "max_tokens": request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        "messages": messages,
        "stream": true,
    });
    if !system.is_empty() {
        body["system"] = serde_json::json!(system);
    }
    if let Some(t) = request.temperature {
        body["temperature"] = serde_json::json!(t);
    }
    body.to_string()
}

/// Strip the SSE `data:` framing; `None` for `event:`/comment/blank lines the caller skips.
fn sse_payload(line: &str) -> Option<&str> {
    let rest = line.trim_end().strip_prefix("data:")?;
    Some(rest.trim_start())
}

/// Anthropic (Claude) hosted adapter, generic over the injected HTTP transport.
pub struct AnthropicAdapter<H: HttpClient> {
    base_url: String,
    api_key: String,
    http: H,
}

impl<H: HttpClient> AnthropicAdapter<H> {
    /// New adapter against `base_url` (default [`DEFAULT_ANTHROPIC_URL`]) with the operator's
    /// `api_key`.
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>, http: H) -> Self {
        let root = base_url.into().trim_end_matches('/').to_string();
        Self { base_url: root, api_key: api_key.into(), http }
    }

    fn auth_headers(&self) -> [(&str, &str); 2] {
        [("x-api-key", self.api_key.as_str()), ("anthropic-version", ANTHROPIC_VERSION)]
    }
}

impl<H: HttpClient> EngineAdapter for AnthropicAdapter<H> {
    fn engine_name(&self) -> &'static str {
        "anthropic"
    }

    fn detect_models(&self) -> Result<Vec<DetectedModel>, AdapterError> {
        let json = self
            .http
            .get_with_headers(&format!("{}/v1/models", self.base_url), &self.auth_headers())?;
        let models: ModelsResponse =
            serde_json::from_str(&json).map_err(|e| AdapterError::Parse(e.to_string()))?;
        Ok(models
            .data
            .iter()
            .filter(|m| !m.id.trim().is_empty())
            .map(|m| DetectedModel {
                engine_ref: m.id.clone(),
                canonical_id: String::new(), // hosted: addressed by the API's own model id
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
        let body = build_body(&request.model_ref, request);
        let lines = self.http.post_stream_with_headers(
            &format!("{}/v1/messages", self.base_url),
            &body,
            &self.auth_headers(),
        )?;

        let mut chunk_tokens = 0u64;
        let (mut input, mut output, mut done) = (0u64, 0u64, false);
        for line in lines {
            let line = line?;
            let Some(payload) = sse_payload(&line) else {
                continue;
            };
            let ev: StreamEvent =
                serde_json::from_str(payload).map_err(|e| AdapterError::Parse(e.to_string()))?;
            if let Some(err) = &ev.error {
                return Err(AdapterError::Http(format!("anthropic: {err}")));
            }
            match ev.kind.as_str() {
                "content_block_delta" => {
                    if let Some(text) = ev.delta.as_ref().and_then(|d| d.text.as_ref()) {
                        if !text.is_empty() {
                            on_delta(text);
                            chunk_tokens += 1;
                        }
                    }
                }
                "message_start" => {
                    if let Some(u) = ev.message.as_ref().and_then(|m| m.usage.as_ref()) {
                        input = u.input_tokens;
                    }
                }
                "message_delta" => {
                    if let Some(u) = &ev.usage {
                        output = u.output_tokens;
                    }
                }
                "message_stop" => done = true,
                _ => {}
            }
        }

        let tokens = if output > 0 { output } else { chunk_tokens };
        Ok(ServeOutcome {
            tokens,
            done,
            engine: EngineMetrics { prompt_eval_count: input, eval_count: output, ..EngineMetrics::default() },
            tool_calls: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::ChatMessage;
    use std::cell::RefCell;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage { role: role.to_string(), content: content.to_string(), ..Default::default() }
    }

    /// Canned `/v1/models` + `/v1/messages` SSE; records the last POST body for assertions.
    #[derive(Default)]
    struct MockHttp {
        models: String,
        stream_lines: Vec<String>,
        last_body: RefCell<String>,
    }
    impl HttpClient for MockHttp {
        fn get(&self, _url: &str) -> Result<String, AdapterError> {
            Ok(self.models.clone())
        }
        fn post_json(&self, _url: &str, _body: &str) -> Result<String, AdapterError> {
            Err(AdapterError::Http("unused".into()))
        }
        fn post_stream(
            &self,
            _url: &str,
            body: &str,
        ) -> Result<Box<dyn Iterator<Item = Result<String, AdapterError>>>, AdapterError> {
            *self.last_body.borrow_mut() = body.to_string();
            Ok(Box::new(self.stream_lines.clone().into_iter().map(Ok)))
        }
    }

    fn req(messages: Vec<ChatMessage>) -> InferenceRequest {
        InferenceRequest { model_ref: "claude-3-5-sonnet".into(), messages, max_tokens: None, temperature: None, tools: Vec::new() }
    }

    #[test]
    fn build_body_lifts_system_and_sets_max_tokens() {
        let body = build_body("claude-x", &req(vec![msg("system", "be terse"), msg("user", "hi")]));
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["system"], "be terse");
        assert_eq!(v["max_tokens"], DEFAULT_MAX_TOKENS); // unset → default (API requires it)
        assert_eq!(v["messages"].as_array().unwrap().len(), 1); // system lifted out
        assert_eq!(v["messages"][0]["role"], "user");
        assert_eq!(v["stream"], true);
    }

    #[test]
    fn detect_models_parses_data_ids() {
        let mock = MockHttp {
            models: r#"{"data":[{"id":"claude-3-5-sonnet"},{"id":"claude-3-opus"}]}"#.into(),
            ..Default::default()
        };
        let a = AnthropicAdapter::new(DEFAULT_ANTHROPIC_URL, "sk-test", mock);
        let models = a.detect_models().unwrap();
        assert_eq!(models.iter().map(|m| m.engine_ref.as_str()).collect::<Vec<_>>(),
                   vec!["claude-3-5-sonnet", "claude-3-opus"]);
    }

    #[test]
    fn serve_stream_accumulates_text_and_output_tokens() {
        let lines = [
            r#"event: message_start"#,
            r#"data: {"type":"message_start","message":{"usage":{"input_tokens":9,"output_tokens":1}}}"#,
            r#"data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}"#,
            r#"data: {"type":"content_block_delta","delta":{"type":"text_delta","text":", world"}}"#,
            r#"data: {"type":"message_delta","usage":{"output_tokens":5}}"#,
            r#"data: {"type":"message_stop"}"#,
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let mock = MockHttp { stream_lines: lines, ..Default::default() };
        let a = AnthropicAdapter::new(DEFAULT_ANTHROPIC_URL, "sk-test", mock);
        let mut out = String::new();
        let outcome = a.serve_stream(&req(vec![msg("user", "hi")]), &mut |d| out.push_str(d)).unwrap();
        assert_eq!(out, "Hello, world");
        assert!(outcome.done);
        assert_eq!(outcome.tokens, 5); // authoritative output_tokens from message_delta
        assert_eq!(outcome.engine.prompt_eval_count, 9);
    }

    #[test]
    fn serve_stream_surfaces_error_event() {
        let lines = vec![
            r#"data: {"type":"error","error":{"type":"overloaded_error","message":"slow down"}}"#.to_string(),
        ];
        let mock = MockHttp { stream_lines: lines, ..Default::default() };
        let a = AnthropicAdapter::new(DEFAULT_ANTHROPIC_URL, "sk-test", mock);
        let err = a.serve_stream(&req(vec![msg("user", "hi")]), &mut |_| {}).unwrap_err();
        assert!(matches!(err, AdapterError::Http(_)));
    }
}
