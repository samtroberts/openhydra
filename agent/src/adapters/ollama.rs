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
    ToolCall, ToolCallFunction,
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
    /// Model capabilities recent Ollama reports, e.g. `["completion","tools","thinking"]`.
    /// We read `"thinking"` to decide whether it's safe to forward the top-level `think`
    /// switch — Ollama `/api/chat` returns a 400 (`"… does not support thinking"`) if `think`
    /// is sent to a model without the capability.
    #[serde(default)]
    capabilities: Vec<String>,
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
    /// Reasoning models (Qwen3-class, DeepSeek-R1, …) return their chain-of-thought here on
    /// recent Ollama — SEPARATE from `content`, which stays empty until (or unless) the final
    /// answer arrives. Serde would silently drop this unknown field, so a model that spends its
    /// whole budget in the thinking phase (the "cleanup/rewrite" case) would look like it
    /// returned nothing. We capture it and re-emit it wrapped in `<think>…</think>` in
    /// [`OllamaAdapter::serve_stream`], mirroring the OpenAI-compat adapter.
    #[serde(default)]
    thinking: Option<String>,
    /// Tool calls the model requested (recent Ollama; may arrive on any chunk, usually the
    /// last). Ollama's shape differs from OpenAI's: no `id`, and `arguments` is a JSON
    /// *object*, not a string — both are normalised in [`OllamaAdapter::serve_stream`].
    #[serde(default)]
    tool_calls: Vec<OllamaToolCall>,
}

#[derive(Debug, Default, Deserialize)]
struct OllamaToolCall {
    /// Present on recent Ollama (≥0.31, live-verified against qwen2.5:7b); absent on older
    /// builds — we use it when non-empty, else synthesise a stable id.
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: OllamaToolCallFunction,
}

#[derive(Debug, Default, Deserialize)]
struct OllamaToolCallFunction {
    #[serde(default)]
    name: String,
    /// Ollama returns the arguments as a structured JSON object (OpenAI uses a string).
    #[serde(default)]
    arguments: serde_json::Value,
}

/// Build the Ollama `/api/chat` request body (always streaming). `max_tokens`/
/// `temperature` map to Ollama's `options.{num_predict,temperature}`; omitted when unset.
///
/// `think` is the *capability-gated* thinking switch — the caller ([`serve_stream`]) passes it
/// through only for a model that advertises the `thinking` capability, so it is never sent to a
/// model that would 400 on it. It is threaded explicitly (not read from `req.think`) so this
/// stays a pure function while the capability probe lives on the adapter.
fn build_chat_body(req: &InferenceRequest, think: Option<bool>) -> String {
    // Map each message, forwarding multi-turn tool state in Ollama's shape: an assistant's
    // `tool_calls` carry `function.{name, arguments}` where **arguments is an object** (the
    // inverse of the string we surface at the gateway — parse it back; fall back to the raw
    // string if it isn't valid JSON), and a `role:"tool"` result carries `tool_name`.
    let messages: Vec<serde_json::Value> = req
        .messages
        .iter()
        .map(|m| {
            let mut msg = serde_json::json!({ "role": m.role, "content": m.content });
            if let Some(tcs) = &m.tool_calls {
                let calls: Vec<serde_json::Value> = tcs
                    .iter()
                    .map(|tc| {
                        let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                            .unwrap_or_else(|_| serde_json::Value::String(tc.function.arguments.clone()));
                        serde_json::json!({ "function": { "name": tc.function.name, "arguments": args } })
                    })
                    .collect();
                msg["tool_calls"] = serde_json::Value::Array(calls);
            }
            if let Some(name) = &m.name {
                msg["tool_name"] = serde_json::json!(name);
            }
            msg
        })
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
    // Ollama's thinking switch is a TOP-LEVEL request key (`think`), not under `options`.
    // `Some(false)` makes a reasoning model answer directly (no chain-of-thought); `Some(true)`
    // forces thinking on. Omitted when `None` so a plain request is byte-identical to before.
    if let Some(think) = think {
        body["think"] = serde_json::json!(think);
    }
    // Forward the caller's OpenAI-shaped `tools` verbatim — Ollama's `/api/chat` accepts the
    // same schema and streams any resulting `message.tool_calls` back (recent Ollama supports
    // tool calls with `stream: true`). Omitted when empty so a plain chat is byte-identical.
    if !req.tools.is_empty() {
        body["tools"] = serde_json::Value::Array(req.tools.clone());
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

    /// Best-effort check (via `/api/show` `capabilities`) of whether `model_name` supports
    /// thinking. Used to gate the top-level `think` switch: Ollama's `/api/chat` returns a 400
    /// for `think` on a model without the capability, so forwarding it unconditionally would
    /// break otherwise-valid requests. Defaults to `false` on any error or a model that doesn't
    /// advertise the capability — the conservative choice, since dropping `think` never breaks a
    /// request (a non-thinking model ignores it anyway) whereas sending it wrongly can 400.
    /// Consulted only when a caller actually set `think`, so plain serves pay no extra call.
    fn supports_thinking(&self, model_name: &str) -> bool {
        let body = serde_json::json!({ "name": model_name }).to_string();
        match self.http.post_json(&format!("{}/api/show", self.base_url), &body) {
            Ok(json) => serde_json::from_str::<ShowResponse>(&json)
                .map(|s| s.capabilities.iter().any(|c| c == "thinking"))
                .unwrap_or(false),
            Err(_) => false,
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
        // Capability-gate the thinking switch: only forward `think` to a model that advertises
        // the `thinking` capability, so a `think:true`/`think:false` request that lands on a
        // non-thinking model doesn't 400 (`"… does not support thinking"`). The probe runs only
        // when a caller actually set `think`, so plain serves make no extra `/api/show` call.
        let think = match request.think {
            Some(v) if self.supports_thinking(&request.model_ref) => Some(v),
            _ => None,
        };
        let body = build_chat_body(request, think);
        let lines = self
            .http
            .post_stream(&format!("{}/api/chat", self.base_url), &body)?;

        let mut chunk_tokens = 0u64;
        let mut eval_count = None;
        let mut engine = crate::adapter::EngineMetrics::default();
        let mut done = false;
        let mut first_token_at: Option<std::time::Instant> = None;
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        // Reasoning models stream chain-of-thought in `message.thinking`, separate from
        // `content`. Wrap it in a single `<think>…</think>` block as it streams: open on the
        // first reasoning fragment, close before the first answer token (or at end if the model
        // only ever reasoned). Mirrors `openai.rs` so the native and OpenAI-compat adapters are
        // symmetric — reasoning that would otherwise be silently dropped is surfaced, and the
        // decode-TPS timer anchors on the first REAL generated token (reasoning included).
        let mut reasoning_open = false;
        for line in lines {
            let line = line?;
            if line.trim().is_empty() {
                continue; // keep-alive / blank framing line
            }
            let mut chunk = parse_chat_chunk(&line)?;
            if let Some(reasoning) = chunk
                .message
                .thinking
                .as_deref()
                .filter(|s| !s.is_empty())
            {
                if first_token_at.is_none() {
                    first_token_at = Some(std::time::Instant::now());
                }
                if !reasoning_open {
                    on_delta("<think>");
                    reasoning_open = true;
                }
                on_delta(reasoning);
                chunk_tokens += 1;
            }
            if !chunk.message.content.is_empty() {
                if first_token_at.is_none() {
                    first_token_at = Some(std::time::Instant::now());
                }
                // Close the reasoning block before the first answer token.
                if reasoning_open {
                    on_delta("</think>\n\n");
                    reasoning_open = false;
                }
                on_delta(&chunk.message.content);
                chunk_tokens += 1;
            }
            // Tool calls can ride any chunk (usually the last). Normalise Ollama's shape to
            // OpenAI's: synthesise a stable `id` (Ollama omits it — coding agents match tool
            // results to calls by id) and render `arguments` as a JSON *string*.
            for otc in std::mem::take(&mut chunk.message.tool_calls) {
                let arguments = if otc.function.arguments.is_null() {
                    String::new()
                } else {
                    otc.function.arguments.to_string()
                };
                let id = otc
                    .id
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| format!("call_{}", tool_calls.len() + 1));
                tool_calls.push(ToolCall {
                    id,
                    kind: "function".to_string(),
                    function: ToolCallFunction { name: otc.function.name, arguments },
                });
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
        // The model produced only reasoning (no answer — e.g. hit the token cap mid-thought):
        // close the block so the consumer/UI renders the thinking instead of a blank bubble.
        if reasoning_open {
            on_delta("</think>");
        }
        // Anomaly: the engine reports it evaluated tokens but we emitted nothing (no content,
        // no reasoning, no tool calls). This is the silent-empty-response failure mode Fix A
        // targets — surface it rather than returning an invisibly blank serve.
        if chunk_tokens == 0
            && tool_calls.is_empty()
            && eval_count.map(|c| c > 0).unwrap_or(false)
        {
            tracing::warn!(
                eval_count = eval_count.unwrap_or(0),
                model = %request.model_ref,
                "ollama serve produced no output despite eval_count>0 (silent empty response)"
            );
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
        Ok(ServeOutcome { tokens, done, engine, tool_calls })
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
        /// Captures the last `/api/chat` request body so tests can assert what the adapter sent
        /// (e.g. that the capability gate did/didn't include the top-level `think` key).
        last_chat_body: std::cell::RefCell<Option<String>>,
        /// Counts `/api/show` calls so tests can assert the capability probe runs only when a
        /// caller set `think` (a plain serve must pay no extra round-trip).
        show_calls: std::cell::RefCell<u32>,
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
                *self.show_calls.borrow_mut() += 1;
                Ok(self.show.clone())
            } else {
                Err(AdapterError::Http(format!("unexpected POST {url}")))
            }
        }
        fn post_stream(
            &self,
            url: &str,
            body: &str,
        ) -> Result<Box<dyn Iterator<Item = Result<String, AdapterError>>>, AdapterError> {
            if url.ends_with("/api/chat") {
                *self.last_chat_body.borrow_mut() = Some(body.to_string());
                Ok(Box::new(self.stream_lines.clone().into_iter().map(Ok)))
            } else {
                Err(AdapterError::Http(format!("unexpected POST {url}")))
            }
        }
    }

    fn adapter(tags: &str, show: &str) -> OllamaAdapter<MockHttp> {
        OllamaAdapter::new(
            DEFAULT_OLLAMA_URL,
            MockHttp { tags: tags.into(), show: show.into(), ..Default::default() },
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

    /// A serve adapter whose `/api/show` returns `show` — lets the capability gate see (or not
    /// see) the `thinking` capability.
    fn serve_adapter_with_show(lines: &[&str], show: &str) -> OllamaAdapter<MockHttp> {
        OllamaAdapter::new(
            DEFAULT_OLLAMA_URL,
            MockHttp {
                show: show.into(),
                stream_lines: lines.iter().map(|s| s.to_string()).collect(),
                ..Default::default()
            },
        )
    }

    fn user_req(prompt: &str) -> InferenceRequest {
        InferenceRequest {
            model_ref: "qwen2.5:7b".into(),
            messages: vec![crate::adapter::ChatMessage { role: "user".into(), content: prompt.into(), ..Default::default() }],
            max_tokens: Some(128),
            temperature: Some(0.7),
            tools: Vec::new(),
            think: None,
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
        let body = build_chat_body(&user_req("hi there"), None);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["model"], "qwen2.5:7b");
        assert_eq!(v["stream"], true);
        assert_eq!(v["messages"][0]["role"], "user");
        assert_eq!(v["messages"][0]["content"], "hi there");
        assert_eq!(v["options"]["num_predict"], 128);
        assert_eq!(v["options"]["temperature"], 0.7);
        assert!(v.get("tools").is_none(), "no tools field on a plain chat");
        assert!(v.get("think").is_none(), "no think field when unset (byte-identical to before)");
    }

    #[test]
    fn build_chat_body_emits_top_level_think_when_set() {
        // `think` is passed explicitly (the capability-gated value the serve loop resolves).
        let req = user_req("2+2?");
        let v: serde_json::Value =
            serde_json::from_str(&build_chat_body(&req, Some(false))).unwrap();
        // Ollama's thinking switch is a TOP-LEVEL key, not under `options`.
        assert_eq!(v["think"], false);
        assert!(v["options"].get("think").is_none(), "think must not be nested under options");

        let v: serde_json::Value =
            serde_json::from_str(&build_chat_body(&req, Some(true))).unwrap();
        assert_eq!(v["think"], true);

        // None ⇒ no key (a plain request, byte-identical to pre-thinking behaviour).
        let v: serde_json::Value = serde_json::from_str(&build_chat_body(&req, None)).unwrap();
        assert!(v.get("think").is_none());
    }

    #[test]
    fn build_chat_body_forwards_tools_verbatim() {
        let mut req = user_req("what's the weather?");
        req.tools = vec![serde_json::json!({
            "type": "function",
            "function": { "name": "get_weather", "parameters": { "type": "object" } }
        })];
        let v: serde_json::Value = serde_json::from_str(&build_chat_body(&req, None)).unwrap();
        assert_eq!(v["tools"][0]["function"]["name"], "get_weather");
    }

    #[test]
    fn build_chat_body_forwards_multi_turn_tool_messages_in_ollama_shape() {
        use crate::adapter::{ChatMessage, ToolCall, ToolCallFunction};
        let mut req = user_req("weather?");
        req.messages = vec![
            ChatMessage {
                role: "assistant".into(),
                content: String::new(),
                tool_calls: Some(vec![ToolCall {
                    id: "call_1".into(),
                    kind: "function".into(),
                    function: ToolCallFunction { name: "get_weather".into(), arguments: r#"{"city":"SF"}"#.into() },
                }]),
                ..Default::default()
            },
            ChatMessage { role: "tool".into(), content: "72F".into(), name: Some("get_weather".into()), ..Default::default() },
        ];
        let v: serde_json::Value = serde_json::from_str(&build_chat_body(&req, None)).unwrap();
        // Ollama wants arguments as an OBJECT (we parse our string back), and no id/type.
        assert_eq!(v["messages"][0]["tool_calls"][0]["function"]["name"], "get_weather");
        assert_eq!(v["messages"][0]["tool_calls"][0]["function"]["arguments"]["city"], "SF");
        assert!(v["messages"][0]["tool_calls"][0]["function"]["arguments"].is_object());
        // Tool result carries tool_name (Ollama's field), not tool_call_id.
        assert_eq!(v["messages"][1]["role"], "tool");
        assert_eq!(v["messages"][1]["tool_name"], "get_weather");
    }

    #[test]
    fn serve_stream_parses_tool_calls_into_openai_shape() {
        // Ollama emits a tool call on the final chunk: no `id`, `arguments` as an object.
        let lines = [
            r#"{"message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"get_weather","arguments":{"city":"SF"}}}]},"done":true,"eval_count":9}"#,
        ];
        let mut out = String::new();
        let outcome = serve_adapter(&lines)
            .serve_stream(&user_req("weather in SF?"), &mut |d| out.push_str(d))
            .unwrap();
        assert!(out.is_empty(), "a tool-call turn streams no text");
        assert_eq!(outcome.tool_calls.len(), 1);
        let tc = &outcome.tool_calls[0];
        assert_eq!(tc.id, "call_1", "a stable id is synthesised (Ollama omits it)");
        assert_eq!(tc.kind, "function");
        assert_eq!(tc.function.name, "get_weather");
        // arguments normalised to a JSON *string* (OpenAI convention).
        let args: serde_json::Value = serde_json::from_str(&tc.function.arguments).unwrap();
        assert_eq!(args["city"], "SF");
    }

    #[test]
    fn serve_stream_mixes_text_then_tool_call() {
        // A model may narrate, then call a tool across two chunks; ids increment per call.
        let lines = [
            r#"{"message":{"content":"Let me check. "},"done":false}"#,
            r#"{"message":{"content":"","tool_calls":[{"function":{"name":"a","arguments":{}}}]},"done":false}"#,
            r#"{"message":{"content":"","tool_calls":[{"function":{"name":"b","arguments":{"x":1}}}]},"done":true,"eval_count":5}"#,
        ];
        let mut out = String::new();
        let outcome = serve_adapter(&lines)
            .serve_stream(&user_req("hi"), &mut |d| out.push_str(d))
            .unwrap();
        assert_eq!(out, "Let me check. ");
        assert_eq!(outcome.tool_calls.len(), 2);
        assert_eq!(outcome.tool_calls[0].id, "call_1");
        assert_eq!(outcome.tool_calls[0].function.name, "a");
        assert_eq!(outcome.tool_calls[1].id, "call_2");
        assert_eq!(outcome.tool_calls[1].function.name, "b");
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
    fn serve_stream_surfaces_thinking_wrapped_before_content() {
        // A Qwen3-class thinking model streams its chain-of-thought in `message.thinking`
        // (content empty) before the answer. Fix A: it must be surfaced, wrapped in a single
        // `<think>…</think>` block, then the answer — not silently dropped.
        let lines = [
            r#"{"message":{"role":"assistant","content":"","thinking":"The user asks 2+2. "},"done":false}"#,
            r#"{"message":{"content":"","thinking":"That is 4."},"done":false}"#,
            r#"{"message":{"content":"The answer is 4."},"done":false}"#,
            r#"{"message":{"content":""},"done":true,"eval_count":12}"#,
        ];
        let mut out = String::new();
        let outcome = serve_adapter(&lines)
            .serve_stream(&user_req("2+2?"), &mut |d| out.push_str(d))
            .unwrap();
        assert_eq!(
            out, "<think>The user asks 2+2. That is 4.</think>\n\nThe answer is 4.",
            "reasoning is wrapped and closed before the answer"
        );
        assert!(outcome.done);
        assert_eq!(outcome.tokens, 12, "uses Ollama's authoritative eval_count");
    }

    // Capability gate for the `think` switch (Fix 2): forward it only to a model that
    // advertises the `thinking` capability, else drop it (a non-thinking model 400s on `think`).
    const SHOW_THINKING: &str = r#"{"template":"t","capabilities":["completion","tools","thinking"]}"#;
    const SHOW_NO_THINKING: &str = r#"{"template":"t","capabilities":["completion","tools"]}"#;

    #[test]
    fn serve_stream_forwards_think_for_thinking_capable_model() {
        let lines = [r#"{"message":{"content":"9"},"done":true,"eval_count":1}"#];
        let adapter = serve_adapter_with_show(&lines, SHOW_THINKING);
        let mut req = user_req("2+2?");
        req.think = Some(false);
        let mut out = String::new();
        adapter.serve_stream(&req, &mut |d| out.push_str(d)).unwrap();
        let body: serde_json::Value =
            serde_json::from_str(adapter.http.last_chat_body.borrow().as_deref().unwrap()).unwrap();
        assert_eq!(body["think"], false, "think forwarded to a thinking-capable model");
        assert_eq!(*adapter.http.show_calls.borrow(), 1, "capability probed once when think set");
    }

    #[test]
    fn serve_stream_drops_think_for_non_thinking_model() {
        // The regression guard: a `think` request that lands on a non-thinking model must NOT
        // reach the engine (Ollama 400s on it) — the gate strips it so the serve still succeeds.
        let lines = [r#"{"message":{"content":"hi"},"done":true,"eval_count":1}"#];
        let adapter = serve_adapter_with_show(&lines, SHOW_NO_THINKING);
        let mut req = user_req("hi");
        req.think = Some(true);
        let mut out = String::new();
        adapter.serve_stream(&req, &mut |d| out.push_str(d)).unwrap();
        let body: serde_json::Value =
            serde_json::from_str(adapter.http.last_chat_body.borrow().as_deref().unwrap()).unwrap();
        assert!(body.get("think").is_none(), "think dropped for a non-thinking model (avoids 400)");
    }

    #[test]
    fn serve_stream_makes_no_capability_probe_when_think_unset() {
        // A plain serve (think:None) must not incur the /api/show capability call — the mock's
        // empty `show` would parse-fail, but we simply never call it. Body carries no think key.
        let lines = [r#"{"message":{"content":"hi"},"done":true,"eval_count":1}"#];
        let adapter = serve_adapter(&lines); // empty show
        let out_req = user_req("hi"); // think defaults to None
        let mut out = String::new();
        adapter.serve_stream(&out_req, &mut |d| out.push_str(d)).unwrap();
        let body: serde_json::Value =
            serde_json::from_str(adapter.http.last_chat_body.borrow().as_deref().unwrap()).unwrap();
        assert!(body.get("think").is_none());
        assert_eq!(out, "hi");
        assert_eq!(*adapter.http.show_calls.borrow(), 0, "no capability probe on a plain serve");
    }

    #[test]
    fn serve_stream_closes_think_block_when_only_reasoning() {
        // The failing "cleanup/rewrite" case: the model spends its whole budget thinking and
        // emits no `content`. The reasoning must still surface (closed `<think>` block) rather
        // than a silent empty response.
        let lines = [
            r#"{"message":{"content":"","thinking":"Let me reconsider… "},"done":false}"#,
            r#"{"message":{"content":"","thinking":"still unsure."},"done":true,"eval_count":8}"#,
        ];
        let mut out = String::new();
        let outcome = serve_adapter(&lines)
            .serve_stream(&user_req("hard question"), &mut |d| out.push_str(d))
            .unwrap();
        assert_eq!(out, "<think>Let me reconsider… still unsure.</think>");
        assert!(outcome.done);
        assert!(!out.is_empty(), "reasoning-only serve is never silently empty");
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
                ..Default::default()
            }],
            max_tokens: Some(16),
            temperature: Some(0.0),
            tools: Vec::new(),
            think: None,
        };
        let mut out = String::new();
        let outcome = agent.serve_stream(&req, &mut |d| out.push_str(d)).unwrap();
        assert!(!out.is_empty(), "engine streamed no text");
        assert!(outcome.tokens > 0);
    }
}
