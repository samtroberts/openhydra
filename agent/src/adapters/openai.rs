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
//! `stream_options.include_usage`. The API reports no per-stage timings, so — since the
//! engine is local — the adapter *measures* them provider-locally (time-to-first-token ≈
//! prefill, first-to-last ≈ decode) to fill [`EngineMetrics`] with a real native gen-TPS
//! instead of a misleading zero.
//!
//! Parsing is pure; HTTP is injected via [`HttpClient`](crate::adapter::HttpClient).

use serde::Deserialize;

use crate::adapter::{
    AdapterError, DetectedModel, EngineAdapter, EngineMetrics, HttpClient, InferenceRequest,
    ServeOutcome, ToolCall, ToolCallFunction,
};

/// Upper bound on distinct tool calls accumulated from one stream — a defensive cap so a
/// buggy/hostile server can't drive an unbounded `index` into a huge allocation (mirrors the
/// bounded-allocation discipline elsewhere in the serve path). Well above any real fan-out.
const MAX_TOOL_CALLS: usize = 64;

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

/// Lenient `Option<String>` deserializer for the reasoning-carrying delta keys. A plain
/// `Option<String>` field *hard-fails* the whole chunk parse (and so the whole serve) if the
/// server sends the key with a non-string value — e.g. `reasoning` as an object or number,
/// which some OpenAI-compat proxies do. These keys are supplementary (chain-of-thought we
/// surface as `<think>`), never load-bearing, so an unexpected shape must degrade to "no
/// reasoning surfaced", not break a serve that would otherwise deliver `content`. Accepts a
/// JSON string (kept), null/absent (`None`), or any other type (ignored → `None`).
fn de_lenient_opt_string<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    Ok(match value {
        Some(serde_json::Value::String(s)) => Some(s),
        _ => None,
    })
}

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
    /// Reasoning models (Qwen3, DeepSeek-R1, …) served over the OpenAI protocol stream their
    /// chain-of-thought here, SEPARATE from `content` (which stays empty until the final
    /// answer). Without reading this the reasoning is silently dropped and a model that spends
    /// its whole budget thinking looks like it returned nothing. We capture it and re-emit it
    /// wrapped in `<think>…</think>` (see the serve loop). Parsed leniently so a non-string
    /// value can't fail the whole serve (see [`de_lenient_opt_string`]).
    #[serde(default, deserialize_with = "de_lenient_opt_string")]
    reasoning_content: Option<String>,
    /// Same chain-of-thought, but under the engine's native `thinking` key. Some OpenAI-compat
    /// builds emit reasoning here (not `reasoning_content`), so a thinking model would otherwise
    /// return an empty `content` with its whole token budget spent invisibly. Treated identically
    /// to [`reasoning_content`] in the serve loop.
    #[serde(default, deserialize_with = "de_lenient_opt_string")]
    thinking: Option<String>,
    /// Same chain-of-thought under a third key: `reasoning`. Ollama's OpenAI-compat `/v1` stream
    /// and OpenRouter emit the reasoning here (verified live against Ollama `/v1` + qwen3), so
    /// without reading it a thinking model served over the OpenAI protocol drops its whole
    /// reasoning phase (empty `content` when the budget is spent thinking). Treated identically
    /// to [`reasoning_content`]/[`thinking`] in the serve loop. Some providers emit `reasoning`
    /// as a structured object rather than a string — parsed leniently so that can't fail the
    /// serve (see [`de_lenient_opt_string`]).
    #[serde(default, deserialize_with = "de_lenient_opt_string")]
    reasoning: Option<String>,
    /// Tool-call fragments (OpenAI streams these incrementally: `id`/`name` on the first
    /// fragment for an `index`, then `arguments` string pieces on following chunks).
    #[serde(default)]
    tool_calls: Option<Vec<DeltaToolCall>>,
}

#[derive(Debug, Default, Deserialize)]
struct DeltaToolCall {
    /// Which tool call this fragment belongs to (stable across the stream).
    #[serde(default)]
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default, rename = "type")]
    kind: Option<String>,
    #[serde(default)]
    function: Option<DeltaToolCallFunction>,
}

#[derive(Debug, Default, Deserialize)]
struct DeltaToolCallFunction {
    #[serde(default)]
    name: Option<String>,
    /// A fragment of the JSON-string arguments — concatenated across chunks in order.
    #[serde(default)]
    arguments: Option<String>,
}

/// Per-`index` accumulator that reassembles a streamed tool call from its fragments.
#[derive(Default)]
struct ToolAcc {
    id: String,
    kind: String,
    name: String,
    arguments: String,
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
    // Map each message, forwarding multi-turn tool state (assistant `tool_calls`, and a
    // `role:"tool"` result's `tool_call_id`/`name`) verbatim — our `ToolCall` already
    // serialises to the exact OpenAI shape (`type`, `function.{name,arguments-string}`), so a
    // vLLM/LM Studio/llama.cpp/Exo backend can continue the tool loop.
    let messages: Vec<serde_json::Value> = req
        .messages
        .iter()
        .map(|m| {
            let mut msg = serde_json::json!({ "role": m.role, "content": m.content });
            if let Some(tcs) = &m.tool_calls {
                msg["tool_calls"] = serde_json::json!(tcs);
            }
            if let Some(id) = &m.tool_call_id {
                msg["tool_call_id"] = serde_json::json!(id);
            }
            if let Some(name) = &m.name {
                msg["name"] = serde_json::json!(name);
            }
            msg
        })
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
    // Forward the caller's OpenAI-shaped `tools` verbatim — vLLM / LM Studio / llama.cpp /
    // Exo all accept the OpenAI schema and stream `tool_calls` deltas back. Omitted when
    // empty so a plain chat body is byte-identical to before.
    if !req.tools.is_empty() {
        body["tools"] = serde_json::Value::Array(req.tools.clone());
    }
    // Thinking control for reasoning models served over the OpenAI protocol (vLLM, llama.cpp,
    // LM Studio). The portable lever these engines share is `chat_template_kwargs.enable_thinking`
    // — the Qwen3 / DeepSeek-R1 chat templates read it to include or skip the reasoning phase.
    // Unlike Ollama's native top-level `think` key (which 400s on a non-thinking model), this is
    // a chat-template kwarg: a template that doesn't reference `enable_thinking` simply ignores
    // it, so it's safe to forward without a capability probe. Omitted when the caller expressed
    // no preference, keeping a plain body byte-identical to before.
    if let Some(think) = req.think {
        body["chat_template_kwargs"] = serde_json::json!({ "enable_thinking": think });
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
    let start = std::time::Instant::now();
    let lines = http.post_stream(chat_url, &body)?;

    let mut chunk_tokens = 0u64;
    let mut usage: Option<Usage> = None;
    let mut done = false;
    let mut tool_acc: Vec<ToolAcc> = Vec::new();
    // The OpenAI stream carries no timings; the engine is local, so measure them here —
    // start→first-token ≈ prefill, first-token→end ≈ decode.
    let mut first_token_at: Option<std::time::Instant> = None;
    // Reasoning models stream chain-of-thought in `delta.reasoning_content`. We wrap it in a
    // single `<think>…</think>` block as it streams: open on the first reasoning fragment, close
    // before the first answer token (or at end if the model only ever reasoned). This surfaces
    // thinking that would otherwise be dropped, matches the inline-`<think>` convention other
    // engines (Ollama) already use, and anchors the decode-TPS timer on the first REAL generated
    // token — reasoning included — instead of a content token that may never arrive.
    let mut reasoning_open = false;
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
            // B-C4: once a finish_reason has been seen, stop emitting/counting content and
            // stop accumulating tool-call fragments. Anything a (buggy or malicious) server
            // sends after the finish must not inflate `chunk_tokens` (the receipt fallback)
            // or graft extra tool calls onto the assistant turn.
            if !done {
                // Reasoning arrives under one of three keys depending on the engine:
                // `reasoning_content` (vLLM / LM Studio / DeepSeek convention), `thinking` (some
                // OpenAI-compat builds), or `reasoning` (Ollama's `/v1` stream, OpenRouter).
                // Accept whichever is present so the chain-of-thought is never dropped — all feed
                // the same single `<think>…</think>` block.
                if let Some(reasoning) = choice
                    .delta
                    .reasoning_content
                    .as_deref()
                    .filter(|s| !s.is_empty())
                    .or(choice.delta.thinking.as_deref().filter(|s| !s.is_empty()))
                    .or(choice.delta.reasoning.as_deref().filter(|s| !s.is_empty()))
                {
                    if !reasoning.is_empty() {
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
                }
                if let Some(content) = &choice.delta.content {
                    if !content.is_empty() {
                        if first_token_at.is_none() {
                            first_token_at = Some(std::time::Instant::now());
                        }
                        // Close the reasoning block before the first answer token.
                        if reasoning_open {
                            on_delta("</think>\n\n");
                            reasoning_open = false;
                        }
                        on_delta(content);
                        chunk_tokens += 1;
                    }
                }
                if let Some(tcs) = &choice.delta.tool_calls {
                    accumulate_tool_calls(&mut tool_acc, tcs);
                }
            }
            if choice.finish_reason.is_some() {
                // A clean stop (content, `tool_calls`, `length`, …); keep reading — the
                // usage chunk may still follow.
                done = true;
            }
        }
        if chunk.usage.is_some() {
            usage = chunk.usage;
        }
    }
    // The model produced only reasoning (no answer — e.g. hit the token cap mid-thought): close
    // the block so the consumer/UI renders the thinking instead of a blank bubble.
    if reasoning_open {
        on_delta("</think>");
    }
    let end = std::time::Instant::now();

    // Prefer the engine's authoritative completion-token count; fall back to the number
    // of non-empty content chunks when the server reports no usage.
    let tokens = usage
        .as_ref()
        .map(|u| u.completion_tokens)
        .filter(|&t| t > 0)
        .unwrap_or(chunk_tokens);

    let total_duration_ns = end.duration_since(start).as_nanos() as u64;
    let (prompt_eval_duration_ns, eval_duration_ns) = match first_token_at {
        Some(t) => (
            t.duration_since(start).as_nanos() as u64,
            end.duration_since(t).as_nanos() as u64,
        ),
        None => (total_duration_ns, 0), // no content streamed → no decode phase
    };
    let engine = EngineMetrics {
        prompt_eval_count: usage.as_ref().map(|u| u.prompt_tokens).unwrap_or(0),
        // Drives native gen-TPS: use the resolved token count so TPS is right even when
        // the server sends no `usage` block.
        eval_count: tokens,
        prompt_eval_duration_ns,
        eval_duration_ns,
        total_duration_ns,
        ..EngineMetrics::default()
    };
    // Reassemble the accumulated fragments into finished OpenAI tool calls. A slot with no
    // function name never got a real fragment (defensive) → dropped; a missing id/type is
    // filled with a synthesised default so the assistant turn is always well-formed.
    let tool_calls: Vec<ToolCall> = tool_acc
        .into_iter()
        .enumerate()
        .filter(|(_, a)| !a.name.is_empty())
        .map(|(i, a)| ToolCall {
            id: if a.id.is_empty() { format!("call_{}", i + 1) } else { a.id },
            kind: if a.kind.is_empty() { "function".to_string() } else { a.kind },
            function: ToolCallFunction { name: a.name, arguments: a.arguments },
        })
        .collect();
    Ok(ServeOutcome { tokens, done, engine, tool_calls })
}

/// Merge one chunk's streamed tool-call fragments into the per-`index` accumulators: set
/// `id`/`type`/`name` from whichever fragment first carries them, and append each
/// `arguments` piece in arrival order. Fragments past [`MAX_TOOL_CALLS`] are ignored (a
/// bound against a hostile `index`).
fn accumulate_tool_calls(acc: &mut Vec<ToolAcc>, fragments: &[DeltaToolCall]) {
    for f in fragments {
        if f.index >= MAX_TOOL_CALLS {
            continue; // out-of-bound index — drop rather than grow the vec unboundedly
        }
        if f.index >= acc.len() {
            acc.resize_with(f.index + 1, ToolAcc::default);
        }
        let slot = &mut acc[f.index];
        if let Some(id) = &f.id {
            if !id.is_empty() {
                slot.id = id.clone();
            }
        }
        if let Some(kind) = &f.kind {
            if !kind.is_empty() {
                slot.kind = kind.clone();
            }
        }
        if let Some(func) = &f.function {
            if let Some(name) = &func.name {
                if !name.is_empty() {
                    slot.name = name.clone();
                }
            }
            if let Some(arguments) = &func.arguments {
                slot.arguments.push_str(arguments);
            }
        }
    }
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
            messages: vec![ChatMessage { role: "user".into(), content: "hi".into(), ..Default::default() }],
            max_tokens: Some(16),
            temperature: Some(0.0),
            tools: Vec::new(),
            think: None,
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
    fn serve_stream_wraps_reasoning_content_in_think_then_emits_answer() {
        // Reasoning models (Qwen3/DeepSeek-R1 via LM Studio/vLLM) stream chain-of-thought in
        // `reasoning_content`, keeping `content` empty until the final answer. Without capture it
        // was silently dropped; now it must be wrapped in a single <think>…</think> block.
        let (out, outcome) = serve(&[
            r#"data: {"choices":[{"delta":{"reasoning_content":"Let me "},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"delta":{"reasoning_content":"think."},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"delta":{"content":"Hi!"},"finish_reason":"stop"}]}"#,
            r#"data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":3}}"#,
            "data: [DONE]",
        ])
        .unwrap();
        assert_eq!(out, "<think>Let me think.</think>\n\nHi!");
        assert_eq!(outcome.tokens, 3);
        assert!(outcome.done);
    }

    #[test]
    fn serve_stream_closes_think_when_model_only_reasoned() {
        // The failure the users hit: the model spent its whole token budget thinking and never
        // emitted an answer. The <think> block must still be closed so the UI shows the thinking
        // instead of a blank bubble (and the reasoning is no longer dropped entirely).
        let (out, _outcome) = serve(&[
            r#"data: {"choices":[{"delta":{"reasoning_content":"thinking forever"},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"delta":{},"finish_reason":"length"}]}"#,
            "data: [DONE]",
        ])
        .unwrap();
        assert_eq!(out, "<think>thinking forever</think>");
    }

    #[test]
    fn serve_stream_wraps_ollama_thinking_key_like_reasoning_content() {
        // Ollama's /v1 stream emits chain-of-thought under the native `thinking` key rather than
        // `reasoning_content`. It must be wrapped identically so it isn't dropped — same output as
        // the `reasoning_content` case.
        let (out, outcome) = serve(&[
            r#"data: {"choices":[{"delta":{"thinking":"Let me "},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"delta":{"thinking":"think."},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"delta":{"content":"Hi!"},"finish_reason":"stop"}]}"#,
            r#"data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":3}}"#,
            "data: [DONE]",
        ])
        .unwrap();
        assert_eq!(out, "<think>Let me think.</think>\n\nHi!");
        assert_eq!(outcome.tokens, 3);
        assert!(outcome.done);
    }

    #[test]
    fn serve_stream_wraps_ollama_v1_reasoning_key_like_reasoning_content() {
        // Ollama's OpenAI-compat /v1 stream (and OpenRouter) emit chain-of-thought under a third
        // key, `reasoning` — verified live against Ollama /v1 + qwen3. It must be wrapped like the
        // other two so a thinking model served over the OpenAI protocol doesn't drop its reasoning.
        let (out, outcome) = serve(&[
            r#"data: {"choices":[{"delta":{"reasoning":"Eight minus "},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"delta":{"reasoning":"three is five."},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"delta":{"content":"5"},"finish_reason":"stop"}]}"#,
            r#"data: {"choices":[],"usage":{"prompt_tokens":8,"completion_tokens":3}}"#,
            "data: [DONE]",
        ])
        .unwrap();
        assert_eq!(out, "<think>Eight minus three is five.</think>\n\n5");
        assert_eq!(outcome.tokens, 3);
        assert!(outcome.done);
    }

    #[test]
    fn serve_stream_tolerates_non_string_reasoning_without_failing_the_serve() {
        // Regression: a server that streams a reasoning key as a JSON object or number (rather
        // than a string) must NOT fail the whole serve — the reasoning is dropped, `content`
        // still flows. A plain `Option<String>` field would `Err(invalid type)` on every such
        // chunk; the lenient deserializer degrades it to `None`.
        let (out, outcome) = serve(&[
            r#"data: {"choices":[{"delta":{"reasoning":{"text":"hidden"},"content":"Answer"},"finish_reason":"stop"}]}"#,
            r#"data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1}}"#,
            "data: [DONE]",
        ])
        .unwrap();
        assert_eq!(out, "Answer", "content survives; the non-string reasoning is dropped, not fatal");
        assert!(outcome.done);

        // Same tolerance for the other two reasoning keys and a numeric value.
        let (out, _) = serve(&[
            r#"data: {"choices":[{"delta":{"reasoning_content":42,"thinking":{"x":1},"content":"Hi"},"finish_reason":"stop"}]}"#,
            "data: [DONE]",
        ])
        .unwrap();
        assert_eq!(out, "Hi");
    }

    #[test]
    fn serve_stream_reasoning_only_v1_response_is_not_silently_empty() {
        // The silent-empty case on the OpenAI path: the model spends its whole budget in the
        // `reasoning` phase (Ollama /v1 truncation) and streams no `content`. The reasoning must
        // still surface (closed <think> block) rather than an empty response.
        let (out, _outcome) = serve(&[
            r#"data: {"choices":[{"delta":{"reasoning":"still working"},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"delta":{},"finish_reason":"length"}]}"#,
            "data: [DONE]",
        ])
        .unwrap();
        assert_eq!(out, "<think>still working</think>");
    }

    #[test]
    fn serve_stream_empty_reasoning_content_does_not_shadow_thinking() {
        // A proxy may stamp an empty `reasoning_content:""` while passing Ollama's native `thinking`
        // through. The empty string must NOT shadow the real thinking — filter empties before the
        // `.or()`, otherwise `Some("")` wins and the chain-of-thought is dropped.
        let (out, _outcome) = serve(&[
            r#"data: {"choices":[{"delta":{"reasoning_content":"","thinking":"real cot"},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"delta":{"content":"Answer"},"finish_reason":"stop"}]}"#,
            "data: [DONE]",
        ])
        .unwrap();
        assert_eq!(out, "<think>real cot</think>\n\nAnswer");
    }

    #[test]
    fn serve_stream_surfaces_thinking_only_ollama_response() {
        // The exact live failure: a thinking-heavy model served through Ollama's /v1 spent its
        // whole budget in `thinking` and returned an empty `content` — so the consumer saw a blank
        // answer. The thinking must now be surfaced, wrapped and closed, instead of dropped.
        let (out, _outcome) = serve(&[
            r#"data: {"choices":[{"delta":{"thinking":"still reasoning"},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"delta":{},"finish_reason":"length"}]}"#,
            "data: [DONE]",
        ])
        .unwrap();
        assert_eq!(out, "<think>still reasoning</think>");
    }

    #[test]
    fn serve_stream_sets_eval_count_and_measured_timings() {
        let (_out, outcome) = serve(&[
            r#"data: {"choices":[{"delta":{"content":"a"},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"delta":{"content":"b"},"finish_reason":"stop"}]}"#,
            "data: [DONE]",
        ])
        .unwrap();
        // native gen-TPS needs a non-zero eval_count even when the server sends no usage.
        assert_eq!(outcome.engine.eval_count, 2);
        assert_eq!(outcome.engine.eval_count, outcome.tokens);
        // Timings are measured provider-locally; these invariants hold regardless of clock.
        assert!(outcome.engine.total_duration_ns >= outcome.engine.eval_duration_ns);
        assert!(outcome.engine.total_duration_ns >= outcome.engine.prompt_eval_duration_ns);
    }

    #[test]
    fn build_chat_body_forwards_tools_verbatim() {
        let mut r = req();
        r.tools = vec![serde_json::json!({
            "type": "function",
            "function": { "name": "get_weather", "parameters": { "type": "object" } }
        })];
        let v: serde_json::Value = serde_json::from_str(&build_chat_body(&r)).unwrap();
        assert_eq!(v["tools"][0]["function"]["name"], "get_weather");
        // A plain request carries no tools field.
        let plain: serde_json::Value = serde_json::from_str(&build_chat_body(&req())).unwrap();
        assert!(plain.get("tools").is_none());
    }

    #[test]
    fn build_chat_body_maps_think_to_chat_template_kwargs() {
        // Fix 3: the OpenAI-protocol engines (vLLM / llama.cpp / LM Studio) get the thinking
        // control as `chat_template_kwargs.enable_thinking` (their portable lever), NOT Ollama's
        // top-level `think`.
        let mut r = req();
        r.think = Some(false);
        let v: serde_json::Value = serde_json::from_str(&build_chat_body(&r)).unwrap();
        assert_eq!(v["chat_template_kwargs"]["enable_thinking"], false);
        assert!(v.get("think").is_none(), "must not use Ollama's native key on the OpenAI protocol");

        r.think = Some(true);
        let v: serde_json::Value = serde_json::from_str(&build_chat_body(&r)).unwrap();
        assert_eq!(v["chat_template_kwargs"]["enable_thinking"], true);

        // No preference ⇒ no kwarg (plain body unchanged).
        let plain: serde_json::Value = serde_json::from_str(&build_chat_body(&req())).unwrap();
        assert!(plain.get("chat_template_kwargs").is_none());
    }

    #[test]
    fn build_chat_body_forwards_multi_turn_tool_messages() {
        use crate::adapter::{ToolCall, ToolCallFunction};
        let mut r = req();
        r.messages = vec![
            ChatMessage::new("user", "weather in SF?"),
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
            ChatMessage { role: "tool".into(), content: "72F".into(), tool_call_id: Some("call_1".into()), ..Default::default() },
        ];
        let v: serde_json::Value = serde_json::from_str(&build_chat_body(&r)).unwrap();
        // Assistant turn keeps OpenAI-shaped tool_calls (arguments stays a string).
        assert_eq!(v["messages"][1]["tool_calls"][0]["id"], "call_1");
        assert_eq!(v["messages"][1]["tool_calls"][0]["type"], "function");
        assert!(v["messages"][1]["tool_calls"][0]["function"]["arguments"].is_string());
        // Tool result carries its tool_call_id.
        assert_eq!(v["messages"][2]["role"], "tool");
        assert_eq!(v["messages"][2]["tool_call_id"], "call_1");
    }

    #[test]
    fn serve_stream_reassembles_streamed_tool_call_fragments() {
        // vLLM/OpenAI stream a tool call as: id+name first, then argument string fragments,
        // then a finish_reason:"tool_calls" chunk. The adapter must reassemble one call with
        // the full argument string.
        let (out, outcome) = serve(&[
            r#"data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"SF\"}"}}]},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
            r#"data: {"choices":[],"usage":{"prompt_tokens":20,"completion_tokens":8}}"#,
            "data: [DONE]",
        ])
        .unwrap();
        assert!(out.is_empty(), "a tool-call turn streams no text content");
        assert_eq!(outcome.tokens, 8);
        assert_eq!(outcome.tool_calls.len(), 1);
        let tc = &outcome.tool_calls[0];
        assert_eq!(tc.id, "call_abc"); // real id preserved (not synthesised)
        assert_eq!(tc.kind, "function");
        assert_eq!(tc.function.name, "get_weather");
        // arguments reassembled from the three fragments into valid JSON.
        let args: serde_json::Value = serde_json::from_str(&tc.function.arguments).unwrap();
        assert_eq!(args["city"], "SF");
    }

    #[test]
    fn serve_stream_reassembles_two_parallel_tool_calls_by_index() {
        let (_out, outcome) = serve(&[
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a","type":"function","function":{"name":"f0","arguments":"{}"}}]},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"b","type":"function","function":{"name":"f1","arguments":"{\"x\":1}"}}]},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
            "data: [DONE]",
        ])
        .unwrap();
        assert_eq!(outcome.tool_calls.len(), 2);
        assert_eq!(outcome.tool_calls[0].function.name, "f0");
        assert_eq!(outcome.tool_calls[1].function.name, "f1");
        assert_eq!(outcome.tool_calls[1].id, "b");
    }

    #[test]
    fn tool_call_fragments_after_finish_are_ignored() {
        // B-C4: a server injecting a tool call after finish_reason must not graft it on.
        let (_out, outcome) = serve(&[
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a","type":"function","function":{"name":"real","arguments":"{}"}}]},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"x","type":"function","function":{"name":"injected","arguments":"{}"}}]},"finish_reason":null}]}"#,
            "data: [DONE]",
        ])
        .unwrap();
        assert_eq!(outcome.tool_calls.len(), 1, "post-finish tool call dropped");
        assert_eq!(outcome.tool_calls[0].function.name, "real");
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
