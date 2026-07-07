// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Ollama engine adapter (protocol plan M3.1).
//!
//! Detects the models a local Ollama instance serves and maps each to a protocol
//! canonical id, so a machine running Ollama can advertise + serve them without any
//! OpenHydra model files. Detection reads two endpoints:
//!
//! * `GET  /api/tags`  — the model list, with `details.{family, parameter_size,
//!   quantization_level}` and on-disk `size`.
//! * `POST /api/show`  — per-model detail; we use its `template` (the chat template) to
//!   compute the canonical id's `template_hash`. A model with no template is advertised
//!   with an empty canonical id (legacy/uncanonicalised — the router still keeps it).
//!
//! Parsing + the canonical mapping are pure; HTTP is injected via
//! [`HttpClient`](crate::adapter::HttpClient).

use serde::Deserialize;

use openhydra_protocol::model_id::{canonical_model_id, chat_template_hash, normalize_chat_template};

use crate::adapter::{
    AdapterError, DetectedModel, EngineAdapter, HttpClient, InferenceRequest, ServeOutcome,
};

/// Default local Ollama endpoint.
pub const DEFAULT_OLLAMA_URL: &str = "http://127.0.0.1:11434";

// ── Ollama JSON shapes (only the fields we use; unknown fields ignored) ──

#[derive(Debug, Deserialize)]
struct TagsResponse {
    #[serde(default)]
    models: Vec<TagEntry>,
}

#[derive(Debug, Deserialize)]
struct TagEntry {
    /// The engine handle, e.g. `"qwen2.5:7b"`.
    #[serde(default)]
    name: String,
    #[serde(default)]
    size: u64,
    #[serde(default)]
    details: TagDetails,
}

#[derive(Debug, Default, Deserialize)]
struct TagDetails {
    #[serde(default)]
    family: String,
    #[serde(default)]
    parameter_size: String,
    #[serde(default)]
    quantization_level: String,
}

#[derive(Debug, Default, Deserialize)]
struct ShowResponse {
    #[serde(default)]
    template: String,
}

// `/api/chat` streaming chunk (newline-delimited JSON). We read the message delta, the
// `done` flag, and `eval_count` (present on the final chunk — the engine's own
// completion-token count).
#[derive(Debug, Default, Deserialize)]
struct ChatStreamChunk {
    #[serde(default)]
    message: ChatStreamMessage,
    #[serde(default)]
    done: bool,
    // Engine metrics — all present on the final (`done`) chunk.
    #[serde(default)]
    eval_count: Option<u64>,
    #[serde(default)]
    eval_duration: Option<u64>,
    #[serde(default)]
    total_duration: Option<u64>,
    #[serde(default)]
    load_duration: Option<u64>,
    #[serde(default)]
    prompt_eval_count: Option<u64>,
    #[serde(default)]
    prompt_eval_duration: Option<u64>,
    /// Present when Ollama returns an error mid-stream.
    #[serde(default)]
    error: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct ChatStreamMessage {
    #[serde(default)]
    content: String,
}

/// Build the Ollama `/api/chat` request body (always streaming). `max_tokens`/
/// `temperature` map to Ollama's `options.{num_predict,temperature}`; omitted when unset.
fn build_chat_body(req: &InferenceRequest) -> String {
    let messages: Vec<serde_json::Value> = req
        .messages
        .iter()
        .map(|m| serde_json::json!({ "role": m.role, "content": m.content }))
        .collect();
    let mut options = serde_json::Map::new();
    if let Some(n) = req.max_tokens {
        options.insert("num_predict".into(), serde_json::json!(n));
    }
    if let Some(t) = req.temperature {
        options.insert("temperature".into(), serde_json::json!(t));
    }
    let mut body = serde_json::json!({
        "model": req.model_ref,
        "messages": messages,
        "stream": true,
    });
    if !options.is_empty() {
        body["options"] = serde_json::Value::Object(options);
    }
    body.to_string()
}

/// Parse one `/api/chat` chunk line. An `error` field becomes an `AdapterError::Http`
/// (the engine refused / failed mid-stream); malformed JSON is a parse error.
fn parse_chat_chunk(line: &str) -> Result<ChatStreamChunk, AdapterError> {
    let chunk: ChatStreamChunk =
        serde_json::from_str(line).map_err(|e| AdapterError::Parse(e.to_string()))?;
    if let Some(err) = &chunk.error {
        return Err(AdapterError::Http(format!("ollama: {err}")));
    }
    Ok(chunk)
}

/// Compute the canonical id for one Ollama model, or `""` when it can't be determined.
///
/// `canonical_model_id` normalises family/params (lowercase) and quant internally. We
/// only attempt it with a non-empty family, params, *and* chat template — without a
/// template there is no meaningful `template_hash`, so we leave the id empty rather than
/// hash an empty string into a misleading id.
fn canonical_for(family: &str, params: &str, quant: &str, template: &str) -> String {
    if family.trim().is_empty() || params.trim().is_empty() {
        return String::new();
    }
    if normalize_chat_template(template).is_empty() {
        return String::new(); // no chat template → advertise uncanonicalised
    }
    let template_hash = chat_template_hash(template);
    match canonical_model_id(family, params, quant, &template_hash) {
        Ok(c) => format!("{}/{}/{}/{}", c.family, c.params, c.quant, c.template_hash),
        Err(_) => String::new(),
    }
}

fn detected_from(entry: &TagEntry, template: &str) -> DetectedModel {
    DetectedModel {
        engine_ref: entry.name.clone(),
        canonical_id: canonical_for(
            &entry.details.family,
            &entry.details.parameter_size,
            &entry.details.quantization_level,
            template,
        ),
        family: entry.details.family.clone(),
        params: entry.details.parameter_size.clone(),
        quant: entry.details.quantization_level.clone(),
        size_bytes: entry.size,
    }
}

/// Adapter for a local Ollama instance, generic over the injected HTTP transport.
pub struct OllamaAdapter<H: HttpClient> {
    base_url: String,
    http: H,
}

impl<H: HttpClient> OllamaAdapter<H> {
    /// New adapter against `base_url` (no trailing slash), e.g. [`DEFAULT_OLLAMA_URL`].
    pub fn new(base_url: impl Into<String>, http: H) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            http,
        }
    }

    /// Best-effort fetch of a model's chat template via `/api/show`. A failure or a
    /// missing template yields `""` (the model is still advertised, just uncanonicalised).
    fn fetch_template(&self, model_name: &str) -> String {
        let body = serde_json::json!({ "name": model_name }).to_string();
        match self.http.post_json(&format!("{}/api/show", self.base_url), &body) {
            Ok(json) => serde_json::from_str::<ShowResponse>(&json)
                .map(|s| s.template)
                .unwrap_or_default(),
            Err(_) => String::new(),
        }
    }
}

impl<H: HttpClient> EngineAdapter for OllamaAdapter<H> {
    fn engine_name(&self) -> &'static str {
        "ollama"
    }

    fn detect_models(&self) -> Result<Vec<DetectedModel>, AdapterError> {
        let tags_json = self.http.get(&format!("{}/api/tags", self.base_url))?;
        let tags: TagsResponse =
            serde_json::from_str(&tags_json).map_err(|e| AdapterError::Parse(e.to_string()))?;
        Ok(tags
            .models
            .iter()
            .map(|m| {
                let template = self.fetch_template(&m.name);
                detected_from(m, &template)
            })
            .collect())
    }

    fn serve_stream(
        &self,
        request: &InferenceRequest,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<ServeOutcome, AdapterError> {
        let body = build_chat_body(request);
        let lines = self
            .http
            .post_stream(&format!("{}/api/chat", self.base_url), &body)?;

        let mut chunk_tokens = 0u64;
        let mut eval_count = None;
        let mut engine = crate::adapter::EngineMetrics::default();
        let mut done = false;
        let mut first_token_at: Option<std::time::Instant> = None;
        for line in lines {
            let line = line?;
            if line.trim().is_empty() {
                continue; // keep-alive / blank framing line
            }
            let chunk = parse_chat_chunk(&line)?;
            if !chunk.message.content.is_empty() {
                if first_token_at.is_none() {
                    first_token_at = Some(std::time::Instant::now());
                }
                on_delta(&chunk.message.content);
                chunk_tokens += 1;
            }
            // The metrics all arrive together on the final chunk.
            if chunk.eval_count.is_some() {
                eval_count = chunk.eval_count; // authoritative count
                engine.eval_count = chunk.eval_count.unwrap_or(0);
                engine.eval_duration_ns = chunk.eval_duration.unwrap_or(0);
                engine.total_duration_ns = chunk.total_duration.unwrap_or(0);
                engine.load_duration_ns = chunk.load_duration.unwrap_or(0);
                engine.prompt_eval_count = chunk.prompt_eval_count.unwrap_or(0);
                engine.prompt_eval_duration_ns = chunk.prompt_eval_duration.unwrap_or(0);
            }
            if chunk.done {
                done = true;
                break;
            }
        }
        let tokens = eval_count.unwrap_or(chunk_tokens);
        // Ollama normally reports authoritative eval timing, but occasionally omits
        // eval_duration (or the whole metrics chunk) on an otherwise-normal completion —
        // which would make native gen-TPS a misleading zero. Backfill from the measured
        // decode time (engine is local → first-token→end ≈ decode) and ensure eval_count
        // reflects the tokens actually streamed.
        if engine.eval_duration_ns == 0 && tokens > 0 {
            if let Some(t) = first_token_at {
                engine.eval_duration_ns =
                    std::time::Instant::now().duration_since(t).as_nanos() as u64;
            }
            if engine.eval_count == 0 {
                engine.eval_count = tokens;
            }
        }
        Ok(ServeOutcome { tokens, done, engine })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TAGS_FIXTURE: &str = r#"{
        "models": [
            {
                "name": "qwen2.5:7b",
                "model": "qwen2.5:7b",
                "size": 4683087332,
                "details": {
                    "format": "gguf",
                    "family": "qwen2",
                    "families": ["qwen2"],
                    "parameter_size": "7.6B",
                    "quantization_level": "Q4_K_M"
                }
            },
            {
                "name": "llama3.2:1b",
                "model": "llama3.2:1b",
                "size": 1321098329,
                "details": {
                    "family": "llama",
                    "parameter_size": "1.2B",
                    "quantization_level": "Q8_0"
                }
            }
        ]
    }"#;

    const SHOW_FIXTURE: &str = r#"{
        "template": "{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n<|im_start|>assistant\n",
        "parameters": "stop \"<|im_end|>\""
    }"#;

    /// Inject canned `/api/tags` + `/api/show` bodies and `/api/chat` stream lines (no
    /// network). The chat stream is a lazy iterator over fixture lines.
    #[derive(Default)]
    struct MockHttp {
        tags: String,
        show: String,
        stream_lines: Vec<String>,
    }

    impl HttpClient for MockHttp {
        fn get(&self, url: &str) -> Result<String, AdapterError> {
            if url.ends_with("/api/tags") {
                Ok(self.tags.clone())
            } else {
                Err(AdapterError::Http(format!("unexpected GET {url}")))
            }
        }
        fn post_json(&self, url: &str, _body: &str) -> Result<String, AdapterError> {
            if url.ends_with("/api/show") {
                Ok(self.show.clone())
            } else {
                Err(AdapterError::Http(format!("unexpected POST {url}")))
            }
        }
        fn post_stream(
            &self,
            url: &str,
            _body: &str,
        ) -> Result<Box<dyn Iterator<Item = Result<String, AdapterError>>>, AdapterError> {
            if url.ends_with("/api/chat") {
                Ok(Box::new(self.stream_lines.clone().into_iter().map(Ok)))
            } else {
                Err(AdapterError::Http(format!("unexpected POST {url}")))
            }
        }
    }

    fn adapter(tags: &str, show: &str) -> OllamaAdapter<MockHttp> {
        OllamaAdapter::new(
            DEFAULT_OLLAMA_URL,
            MockHttp { tags: tags.into(), show: show.into(), stream_lines: vec![] },
        )
    }

    fn serve_adapter(lines: &[&str]) -> OllamaAdapter<MockHttp> {
        OllamaAdapter::new(
            DEFAULT_OLLAMA_URL,
            MockHttp {
                stream_lines: lines.iter().map(|s| s.to_string()).collect(),
                ..Default::default()
            },
        )
    }

    fn user_req(prompt: &str) -> InferenceRequest {
        InferenceRequest {
            model_ref: "qwen2.5:7b".into(),
            messages: vec![crate::adapter::ChatMessage { role: "user".into(), content: prompt.into() }],
            max_tokens: Some(128),
            temperature: Some(0.7),
        }
    }

    #[test]
    fn detects_models_with_canonical_ids() {
        let models = adapter(TAGS_FIXTURE, SHOW_FIXTURE).detect_models().unwrap();
        assert_eq!(models.len(), 2);

        let qwen = &models[0];
        assert_eq!(qwen.engine_ref, "qwen2.5:7b"); // engine handle preserved for serving
        assert_eq!(qwen.size_bytes, 4683087332);
        // canonical id: family/params lowercased, quant normalised, 16-hex template hash.
        let parts: Vec<&str> = qwen.canonical_id.split('/').collect();
        assert_eq!(parts.len(), 4, "canonical id should have 4 components: {}", qwen.canonical_id);
        assert_eq!(parts[0], "qwen2");
        assert_eq!(parts[1], "7.6b");
        assert!(!parts[2].is_empty(), "quant component present");
        assert_eq!(parts[3].len(), 16, "template hash is 16 hex chars");
    }

    #[test]
    fn distinct_models_get_distinct_engine_refs() {
        let models = adapter(TAGS_FIXTURE, SHOW_FIXTURE).detect_models().unwrap();
        assert_eq!(models[1].engine_ref, "llama3.2:1b");
        assert_eq!(models[1].family, "llama");
        assert_ne!(models[0].engine_ref, models[1].engine_ref);
    }

    #[test]
    fn no_template_yields_empty_canonical_id_but_still_detected() {
        // /api/show returns no template → uncanonicalised, but the model is still listed
        // (engine_ref/family/quant retained) so the router can keep it as legacy.
        let models = adapter(TAGS_FIXTURE, r#"{"parameters": "x"}"#).detect_models().unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].canonical_id, "");
        assert_eq!(models[0].engine_ref, "qwen2.5:7b");
        assert_eq!(models[0].family, "qwen2");
    }

    #[test]
    fn empty_model_list_is_ok() {
        let models = adapter(r#"{"models": []}"#, SHOW_FIXTURE).detect_models().unwrap();
        assert!(models.is_empty());
    }

    #[test]
    fn malformed_tags_json_is_a_parse_error() {
        let err = adapter("not json", SHOW_FIXTURE).detect_models().unwrap_err();
        assert!(matches!(err, AdapterError::Parse(_)));
    }

    #[test]
    fn canonical_for_is_deterministic_and_template_sensitive() {
        let a = canonical_for("qwen2", "7.6B", "Q4_K_M", "template-A");
        let b = canonical_for("qwen2", "7.6B", "Q4_K_M", "template-A");
        let c = canonical_for("qwen2", "7.6B", "Q4_K_M", "template-B");
        assert_eq!(a, b);
        assert_ne!(a, c, "a different chat template → a different canonical id");
        assert_eq!(canonical_for("", "7.6B", "Q4_K_M", "t"), ""); // missing family
        assert_eq!(canonical_for("qwen2", "7.6B", "Q4_K_M", ""), ""); // missing template
    }

    // ── streaming completion proxy ──

    #[test]
    fn build_chat_body_has_stream_messages_and_options() {
        let body = build_chat_body(&user_req("hi there"));
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["model"], "qwen2.5:7b");
        assert_eq!(v["stream"], true);
        assert_eq!(v["messages"][0]["role"], "user");
        assert_eq!(v["messages"][0]["content"], "hi there");
        assert_eq!(v["options"]["num_predict"], 128);
        assert_eq!(v["options"]["temperature"], 0.7);
    }

    #[test]
    fn serve_stream_concatenates_deltas_and_reports_engine_token_count() {
        let lines = [
            r#"{"message":{"role":"assistant","content":"Hello"},"done":false}"#,
            r#"{"message":{"role":"assistant","content":", "},"done":false}"#,
            r#"{"message":{"role":"assistant","content":"world"},"done":false}"#,
            r#"{"message":{"role":"assistant","content":""},"done":true,"eval_count":7}"#,
        ];
        let mut out = String::new();
        let outcome = serve_adapter(&lines)
            .serve_stream(&user_req("hi"), &mut |d| out.push_str(d))
            .unwrap();
        assert_eq!(out, "Hello, world");
        assert!(outcome.done);
        assert_eq!(outcome.tokens, 7, "uses Ollama's authoritative eval_count");
    }

    #[test]
    fn serve_stream_falls_back_to_chunk_count_without_eval_count() {
        let lines = [
            r#"{"message":{"content":"a"},"done":false}"#,
            r#"{"message":{"content":"b"},"done":false}"#,
            r#"{"message":{"content":""},"done":true}"#, // no eval_count
        ];
        let mut out = String::new();
        let outcome = serve_adapter(&lines)
            .serve_stream(&user_req("hi"), &mut |d| out.push_str(d))
            .unwrap();
        assert_eq!(out, "ab");
        assert_eq!(outcome.tokens, 2); // 2 non-empty content chunks
        // Backfill: with no metrics chunk, eval_count is filled from the streamed tokens so
        // native gen-TPS is computable rather than a degenerate zero.
        assert_eq!(outcome.engine.eval_count, 2);
    }

    #[test]
    fn serve_stream_backfills_eval_duration_when_ollama_omits_it() {
        // The observed quirk: a normal completion with eval_count but no eval_duration.
        // Without the backfill, native gen-TPS would be a misleading zero.
        let lines = [
            r#"{"message":{"content":"a"},"done":false}"#,
            r#"{"message":{"content":"b"},"done":true,"eval_count":2}"#, // no eval_duration
        ];
        let mut out = String::new();
        let outcome = serve_adapter(&lines)
            .serve_stream(&user_req("hi"), &mut |d| out.push_str(d))
            .unwrap();
        assert_eq!(outcome.tokens, 2);
        assert_eq!(outcome.engine.eval_count, 2);
        // eval_duration is now the measured decode time (> 0), not the engine's missing 0.
        assert!(outcome.engine.eval_duration_ns > 0);
    }

    #[test]
    fn serve_stream_stops_at_done_and_skips_blank_lines() {
        let lines = [
            "",
            r#"{"message":{"content":"x"},"done":false}"#,
            r#"{"message":{"content":""},"done":true,"eval_count":1}"#,
            r#"{"message":{"content":"LEAKED"},"done":false}"#, // after done — must be ignored
        ];
        let mut out = String::new();
        serve_adapter(&lines)
            .serve_stream(&user_req("hi"), &mut |d| out.push_str(d))
            .unwrap();
        assert_eq!(out, "x", "nothing after the done chunk is emitted");
    }

    #[test]
    fn serve_stream_surfaces_engine_error_chunk() {
        let lines = [r#"{"error":"model 'ghost' not found"}"#];
        let err = serve_adapter(&lines)
            .serve_stream(&user_req("hi"), &mut |_| {})
            .unwrap_err();
        assert!(matches!(err, AdapterError::Http(_)));
    }

    #[test]
    fn serve_stream_malformed_chunk_is_a_parse_error() {
        let lines = [r#"{"message":{"content":"ok"},"done":false}"#, "not json"];
        let err = serve_adapter(&lines)
            .serve_stream(&user_req("hi"), &mut |_| {})
            .unwrap_err();
        assert!(matches!(err, AdapterError::Parse(_)));
    }

    /// Live end-to-end check against a real Ollama. Ignored by default (CI/this env have
    /// no engine); run manually with `cargo test -p openhydra-agent -- --ignored` on a
    /// machine with Ollama up and ≥1 model pulled.
    #[test]
    #[ignore = "requires a live Ollama at 127.0.0.1:11434 with >=1 model"]
    fn live_smoke_detect_and_serve() {
        let agent = crate::live_ollama(DEFAULT_OLLAMA_URL).unwrap();
        let models = agent.detect_models().unwrap();
        assert!(!models.is_empty(), "no Ollama models found");

        let req = InferenceRequest {
            model_ref: models[0].engine_ref.clone(),
            messages: vec![crate::adapter::ChatMessage {
                role: "user".into(),
                content: "Reply with one word.".into(),
            }],
            max_tokens: Some(16),
            temperature: Some(0.0),
        };
        let mut out = String::new();
        let outcome = agent.serve_stream(&req, &mut |d| out.push_str(d)).unwrap();
        assert!(!out.is_empty(), "engine streamed no text");
        assert!(outcome.tokens > 0);
    }
}
