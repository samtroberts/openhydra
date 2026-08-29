// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Google Gemini BYOK adapter — the hosted Generative Language API.
//!
//! A gateway-side passthrough backend (hosted API + operator key), like
//! [`anthropic`](super::anthropic). Detection lists models via `GET /v1beta/models`; serving
//! streams `POST /v1beta/models/{model}:streamGenerateContent?alt=sse`.
//!
//! Mapping notes: Gemini uses `user`/**`model`** roles (not `assistant`) inside `contents`,
//! takes the system prompt as a top-level `systemInstruction`, and nests generation limits
//! under `generationConfig` (`maxOutputTokens`/`temperature`). Output-token count comes from
//! `usageMetadata.candidatesTokenCount`. Auth is the `x-goog-api-key` header (not a URL query
//! param, to keep the key out of logs).
//!
//! Parsing is pure; HTTP + auth headers are injected via [`HttpClient`].

use serde::Deserialize;

use crate::adapter::{
    AdapterError, DetectedModel, EngineAdapter, EngineMetrics, HttpClient, InferenceRequest,
    ServeOutcome,
};

/// Gemini Generative Language API root.
pub const DEFAULT_GEMINI_URL: &str = "https://generativelanguage.googleapis.com";

// ── GET /v1beta/models ──
#[derive(Debug, Default, Deserialize)]
struct ModelsResponse {
    #[serde(default)]
    models: Vec<ModelEntry>,
}
#[derive(Debug, Default, Deserialize)]
struct ModelEntry {
    /// e.g. `"models/gemini-1.5-pro"`; the serve path uses it without the `models/` prefix.
    #[serde(default)]
    name: String,
}

// ── streamGenerateContent SSE chunk (one JSON object per `data:` line) ──
#[derive(Debug, Default, Deserialize)]
struct StreamChunk {
    #[serde(default)]
    candidates: Vec<Candidate>,
    #[serde(default, rename = "usageMetadata")]
    usage: Option<UsageMeta>,
    #[serde(default)]
    error: Option<serde_json::Value>,
}
#[derive(Debug, Default, Deserialize)]
struct Candidate {
    #[serde(default)]
    content: Option<Content>,
    #[serde(default, rename = "finishReason")]
    finish_reason: Option<String>,
}
#[derive(Debug, Default, Deserialize)]
struct Content {
    #[serde(default)]
    parts: Vec<Part>,
}
#[derive(Debug, Default, Deserialize)]
struct Part {
    #[serde(default)]
    text: Option<String>,
}
#[derive(Debug, Default, Deserialize)]
struct UsageMeta {
    #[serde(default, rename = "promptTokenCount")]
    prompt_tokens: u64,
    #[serde(default, rename = "candidatesTokenCount")]
    candidates_tokens: u64,
}

/// Build the `:streamGenerateContent` body: map `assistant`→`model`, lift `system` into
/// `systemInstruction`, and nest token/temperature limits under `generationConfig`.
fn build_body(request: &InferenceRequest) -> String {
    let mut system_parts: Vec<String> = Vec::new();
    let mut contents: Vec<serde_json::Value> = Vec::new();
    for m in &request.messages {
        match m.role.as_str() {
            "system" => system_parts.push(m.content.clone()),
            "assistant" => {
                contents.push(serde_json::json!({ "role": "model", "parts": [{ "text": m.content }] }))
            }
            _ => contents.push(serde_json::json!({ "role": "user", "parts": [{ "text": m.content }] })),
        }
    }
    let mut gen = serde_json::Map::new();
    if let Some(mt) = request.max_tokens {
        gen.insert("maxOutputTokens".into(), serde_json::json!(mt));
    }
    if let Some(t) = request.temperature {
        gen.insert("temperature".into(), serde_json::json!(t));
    }
    let mut body = serde_json::json!({ "contents": contents });
    if !system_parts.is_empty() {
        body["systemInstruction"] = serde_json::json!({ "parts": [{ "text": system_parts.join("\n") }] });
    }
    if !gen.is_empty() {
        body["generationConfig"] = serde_json::Value::Object(gen);
    }
    body.to_string()
}

/// Strip the SSE `data:` framing; `None` for blank/comment lines.
fn sse_payload(line: &str) -> Option<&str> {
    let rest = line.trim_end().strip_prefix("data:")?;
    Some(rest.trim_start())
}

/// Google Gemini hosted adapter, generic over the injected HTTP transport.
pub struct GeminiAdapter<H: HttpClient> {
    base_url: String,
    api_key: String,
    http: H,
}

impl<H: HttpClient> GeminiAdapter<H> {
    /// New adapter against `base_url` (default [`DEFAULT_GEMINI_URL`]) with the operator's
    /// `api_key`.
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>, http: H) -> Self {
        let root = base_url.into().trim_end_matches('/').to_string();
        Self { base_url: root, api_key: api_key.into(), http }
    }

    fn auth_headers(&self) -> [(&str, &str); 1] {
        [("x-goog-api-key", self.api_key.as_str())]
    }
}

impl<H: HttpClient> EngineAdapter for GeminiAdapter<H> {
    fn engine_name(&self) -> &'static str {
        "gemini"
    }

    fn detect_models(&self) -> Result<Vec<DetectedModel>, AdapterError> {
        let json = self
            .http
            .get_with_headers(&format!("{}/v1beta/models", self.base_url), &self.auth_headers())?;
        let models: ModelsResponse =
            serde_json::from_str(&json).map_err(|e| AdapterError::Parse(e.to_string()))?;
        Ok(models
            .models
            .iter()
            .filter(|m| !m.name.trim().is_empty())
            .map(|m| DetectedModel {
                engine_ref: m.name.strip_prefix("models/").unwrap_or(&m.name).to_string(),
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
        let body = build_body(request);
        let url = format!(
            "{}/v1beta/models/{}:streamGenerateContent?alt=sse",
            self.base_url, request.model_ref
        );
        let lines = self.http.post_stream_with_headers(&url, &body, &self.auth_headers())?;

        let mut chunk_tokens = 0u64;
        let (mut prompt, mut output, mut done) = (0u64, 0u64, false);
        for line in lines {
            let line = line?;
            let Some(payload) = sse_payload(&line) else {
                continue;
            };
            let chunk: StreamChunk =
                serde_json::from_str(payload).map_err(|e| AdapterError::Parse(e.to_string()))?;
            if let Some(err) = &chunk.error {
                return Err(AdapterError::Http(format!("gemini: {err}")));
            }
            for cand in &chunk.candidates {
                if let Some(content) = &cand.content {
                    for part in &content.parts {
                        if let Some(text) = &part.text {
                            if !text.is_empty() {
                                on_delta(text);
                                chunk_tokens += 1;
                            }
                        }
                    }
                }
                if cand.finish_reason.is_some() {
                    done = true;
                }
            }
            if let Some(u) = &chunk.usage {
                prompt = u.prompt_tokens;
                output = u.candidates_tokens;
            }
        }

        let tokens = if output > 0 { output } else { chunk_tokens };
        Ok(ServeOutcome {
            tokens,
            done,
            engine: EngineMetrics { prompt_eval_count: prompt, eval_count: output, ..EngineMetrics::default() },
            tool_calls: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::ChatMessage;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage { role: role.to_string(), content: content.to_string(), ..Default::default() }
    }

    #[derive(Default)]
    struct MockHttp {
        models: String,
        stream_lines: Vec<String>,
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
            _body: &str,
        ) -> Result<Box<dyn Iterator<Item = Result<String, AdapterError>>>, AdapterError> {
            Ok(Box::new(self.stream_lines.clone().into_iter().map(Ok)))
        }
    }

    fn req(messages: Vec<ChatMessage>) -> InferenceRequest {
        InferenceRequest { model_ref: "gemini-1.5-pro".into(), messages, max_tokens: Some(256), temperature: None, tools: Vec::new(), think: None }
    }

    #[test]
    fn build_body_maps_roles_and_system_and_genconfig() {
        let body = build_body(&req(vec![msg("system", "be nice"), msg("user", "hi"), msg("assistant", "hello")]));
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["systemInstruction"]["parts"][0]["text"], "be nice");
        assert_eq!(v["contents"].as_array().unwrap().len(), 2);
        assert_eq!(v["contents"][0]["role"], "user");
        assert_eq!(v["contents"][1]["role"], "model"); // assistant → model
        assert_eq!(v["generationConfig"]["maxOutputTokens"], 256);
    }

    #[test]
    fn detect_models_strips_models_prefix() {
        let mock = MockHttp {
            models: r#"{"models":[{"name":"models/gemini-1.5-pro"},{"name":"models/gemini-1.5-flash"}]}"#.into(),
            ..Default::default()
        };
        let g = GeminiAdapter::new(DEFAULT_GEMINI_URL, "k", mock);
        let models = g.detect_models().unwrap();
        assert_eq!(models.iter().map(|m| m.engine_ref.as_str()).collect::<Vec<_>>(),
                   vec!["gemini-1.5-pro", "gemini-1.5-flash"]);
    }

    #[test]
    fn serve_stream_accumulates_text_and_tokens() {
        let lines = [
            r#"data: {"candidates":[{"content":{"parts":[{"text":"Hi"}],"role":"model"}}]}"#,
            r#"data: {"candidates":[{"content":{"parts":[{"text":" there"}],"role":"model"}},{"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":7}}"#,
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        // Note: the finishReason candidate above is separate; cover both shapes.
        let mock = MockHttp { stream_lines: lines, ..Default::default() };
        let g = GeminiAdapter::new(DEFAULT_GEMINI_URL, "k", mock);
        let mut out = String::new();
        let outcome = g.serve_stream(&req(vec![msg("user", "hi")]), &mut |d| out.push_str(d)).unwrap();
        assert_eq!(out, "Hi there");
        assert!(outcome.done);
        assert_eq!(outcome.tokens, 7); // candidatesTokenCount
        assert_eq!(outcome.engine.prompt_eval_count, 4);
    }

    #[test]
    fn serve_stream_surfaces_error() {
        let lines = vec![
            r#"data: {"error":{"code":429,"message":"quota"}}"#.to_string(),
        ];
        let mock = MockHttp { stream_lines: lines, ..Default::default() };
        let g = GeminiAdapter::new(DEFAULT_GEMINI_URL, "k", mock);
        let err = g.serve_stream(&req(vec![msg("user", "hi")]), &mut |_| {}).unwrap_err();
        assert!(matches!(err, AdapterError::Http(_)));
    }
}
