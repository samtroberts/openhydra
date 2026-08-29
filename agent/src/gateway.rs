// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! The consumer front door: an OpenAI-compatible HTTP/SSE gateway over [`ConsumerNode`].
//!
//! `POST /v1/chat/completions` discovers a provider for the requested model and relays the
//! completion to the client — as Server-Sent Events when `stream: true`, or as a single
//! `chat.completion` JSON object otherwise (the OpenAI default). `GET /health` is a
//! liveness probe.
//!
//! [`ConsumerNode::complete`] is synchronous and blocks on libp2p (`blocking_send`), so the
//! handler runs it on a **plain OS thread** (outside any tokio context, where
//! `blocking_send` is valid) and pipes the deltas back through an unbounded channel.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::status::{EconomyStats, EconomyView};

use axum::body::Body;
use axum::extract::{ConnectInfo, Request};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{extract::State, Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use openhydra_network::handle::NetworkHandle;

/// Global backstop on concurrent generation workers (G3b), independent of the per-identity
/// rate-limiter and always on. Each chat completion spawns a blocking OS thread that ties up
/// a generation for seconds; without a ceiling a flood spread across many identities (each
/// under its own `max_inflight`) could still exhaust threads/fds. A permit is held for the
/// worker's lifetime; when none is free the gateway sheds with `503` rather than pile up
/// threads. Set high so it only ever trips under genuine overload.
const MAX_CONCURRENT_GENERATIONS: usize = 512;

/// SSE keep-alive comment interval — keeps a streaming connection warm through any stall so a
/// client read-timeout can't trip during a cold-model load / prefill / slow provider.
const KEEPALIVE_INTERVAL: std::time::Duration = std::time::Duration::from_secs(15);
/// How often the Anthropic streamer emits a native `ping` while awaiting the first frame (during
/// discovery + prefill), so a client's first-byte timeout survives a slow start without us
/// fabricating a message envelope.
const PING_INTERVAL: std::time::Duration = std::time::Duration::from_secs(10);
/// How long a model lingers in the browse catalog (`/v1/models`, `/models`) after it was last seen
/// live — smooths transient discovery churn. Matches the desktop W2 sticky-TTL.
const MODEL_STICKY_TTL: std::time::Duration = std::time::Duration::from_secs(300);

use crate::adapter::{ChatMessage, EmbeddingAdapter, EngineAdapter, InferenceRequest, ToolCall, ToolCallFunction};
use crate::aup::{AupDecision, AupPolicy};
use crate::byok::{ByokConfig, ByokProvider, EmbeddingConfig};
use crate::consumer::ConsumerNode;
use crate::metrics::Metrics;
use crate::ratelimit::{RateLimitConfig, RateLimiter};
use crate::serve::ServeMetrics;
use crate::slash::{self, SessionModels, SlashCommand};
use openhydra_protocol::store::Store;
use crate::serve::ServeSummary;

#[derive(Clone)]
struct AppState {
    node: Arc<ConsumerNode>,
    /// When set, `/v1/*` requires `Authorization: Bearer <key>`. `None` ⇒ open (the
    /// loopback default).
    api_key: Option<Arc<String>>,
    /// Gateway-side Prometheus metrics (#33), scraped at `/metrics`.
    metrics: Arc<Metrics>,
    /// Acceptable-use policy floor applied to inbound completion requests (default permissive).
    aup: Arc<AupPolicy>,
    /// Ingress DoS rate-limiter (per-identity concurrency + token bucket; default off).
    rate_limiter: Arc<RateLimiter>,
    /// Global backstop on concurrent generation workers ([`MAX_CONCURRENT_GENERATIONS`], G3b);
    /// always on, independent of `rate_limiter`.
    gen_limiter: Arc<Semaphore>,
    /// Honor `X-Forwarded-For` for per-IP keying (only behind a trusted reverse proxy — the
    /// header is client-spoofable). Default `false`: key off the unspoofable socket address.
    trusted_proxy: bool,
    /// BYOK passthrough routing (#34): mapped models call a hosted backend directly instead
    /// of the swarm. Empty by default.
    byok: Arc<ByokConfig>,
    /// BYOK embeddings routing (#34): `/v1/embeddings` for configured models. Empty by default.
    embeddings: Arc<EmbeddingConfig>,
    /// Per-session sticky model selection set via the `/model <id>` slash-command (keyed by the
    /// caller's API-key identity; in-memory, TTL'd, never persisted).
    sessions: Arc<SessionModels>,
    /// Sticky browse-list cache smoothing catalog churn for `/v1/models` + `/models` (routing is
    /// unaffected — it uses the live set).
    model_cache: Arc<ModelStickyCache>,
}

/// The OpenAI chat-completions request fields we honour (others ignored).
#[derive(Debug, Deserialize)]
struct ChatRequest {
    #[serde(default)]
    model: String,
    #[serde(default)]
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f64>,
    /// OpenAI `tools` — function specs the model may call. Forwarded opaquely to the engine;
    /// any resulting `tool_calls` come back on the assistant turn. Empty ⇒ a plain completion.
    /// (`tool_choice` is accepted but not yet forwarded — see A2b.)
    #[serde(default)]
    tools: Vec<Value>,
    /// OpenAI default is `false` → a single JSON object; `true` → an SSE stream.
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stream_options: Option<StreamOptions>,
    /// Ollama-native thinking switch: `false` makes a reasoning model answer directly (no
    /// chain-of-thought), `true` forces thinking on. Highest-precedence thinking control.
    #[serde(default)]
    think: Option<bool>,
    /// vLLM / llama.cpp convention for the same control (`{"enable_thinking": false}`).
    /// Consulted only when top-level `think` is absent.
    #[serde(default)]
    chat_template_kwargs: Option<ChatTemplateKwargs>,
}

/// The subset of OpenAI-compat `chat_template_kwargs` we read — the thinking toggle that
/// vLLM / llama.cpp-server accept. Other keys are ignored.
#[derive(Debug, Deserialize)]
struct ChatTemplateKwargs {
    #[serde(default)]
    enable_thinking: Option<bool>,
}

impl ChatRequest {
    /// Resolve the caller's thinking preference to one internal flag. Ollama-native `think`
    /// wins; otherwise `chat_template_kwargs.enable_thinking`; `None` ⇒ leave the engine's
    /// default. Threaded to the provider and mapped to each engine's native switch.
    fn effective_think(&self) -> Option<bool> {
        self.think
            .or_else(|| self.chat_template_kwargs.as_ref().and_then(|k| k.enable_thinking))
    }
}

#[derive(Debug, Default, Deserialize)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

/// What the worker thread sends back as the completion progresses.
enum GatewayEvent {
    Delta(String),
    Done(Box<ServeSummary>),
    Error(String),
}

static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(1);

fn next_id() -> String {
    format!("chatcmpl-{}", REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed))
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn unix_now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ── OpenAI error mapping ─────────────────────────────────────────────────────

/// Map a `ConsumerNode::complete` error string to an HTTP status + OpenAI error `type`.
/// The error originates as a `String` (the engine/transport surface is stringly-typed),
/// so we classify by substring — coarse but stable, and far better than a blanket 500.
fn classify_error(msg: &str) -> (StatusCode, &'static str) {
    let m = msg.to_ascii_lowercase();
    if m.contains("no provider") {
        // Nobody currently serves this model on the swarm.
        (StatusCode::SERVICE_UNAVAILABLE, "no_provider")
    } else if m.contains("timed out") || m.contains("timeout") {
        (StatusCode::GATEWAY_TIMEOUT, "upstream_timeout")
    } else {
        // Engine/transport failure relayed from the provider.
        (StatusCode::BAD_GATEWAY, "upstream_error")
    }
}

/// An OpenAI-shaped error response: `{ "error": { message, type, code } }` with a status.
fn openai_error(status: StatusCode, message: &str, etype: &str) -> Response {
    (
        status,
        Json(json!({ "error": { "message": message, "type": etype, "code": Value::Null } })),
    )
        .into_response()
}

/// Reject obviously-invalid requests before spending a discovery/route on them. Returns
/// the error response to send when invalid, or `None` when the request is acceptable.
fn validate(req: &ChatRequest) -> Option<Response> {
    if req.model.trim().is_empty() {
        return Some(openai_error(
            StatusCode::BAD_REQUEST,
            "missing required field: model",
            "invalid_request_error",
        ));
    }
    if req.messages.is_empty() {
        return Some(openai_error(
            StatusCode::BAD_REQUEST,
            "missing or empty field: messages",
            "invalid_request_error",
        ));
    }
    None
}

// ── `openhydra/auto` meta-model ──────────────────────────────────────────────
//
// Connector snippets advertise `model: openhydra/auto`, but the router resolves a *literal*
// model id — so `auto` must be turned into a concrete model that actually has a provider
// before dispatch. Resolution: an operator override (`OPENHYDRA_AUTO_MODEL`) if it's live,
// else a deterministic pick from the currently-known set; empty set ⇒ a clear "no models yet".

/// The canonical meta-model id the connectors advertise.
const AUTO_MODEL_ID: &str = "openhydra/auto";

/// Is `model` the auto meta-model (`auto` or `openhydra/auto`, case/space-insensitive)?
fn is_auto_model(model: &str) -> bool {
    let m = model.trim();
    m.eq_ignore_ascii_case("auto") || m.eq_ignore_ascii_case(AUTO_MODEL_ID)
}

/// Pure selection: an operator-preferred model wins *iff* it's currently served; otherwise the
/// alphabetically-first known model (stable/deterministic). `None` when nothing is served.
fn pick_auto_model(known: &[String], preferred: Option<&str>) -> Option<String> {
    if let Some(p) = preferred.map(str::trim).filter(|s| !s.is_empty()) {
        if known.iter().any(|m| m == p) {
            return Some(p.to_string());
        }
    }
    known.iter().min().cloned()
}

/// Resolve the request's `model` to a concrete, served one. `known_models` is a blocking swarm
/// read, so it runs on a blocking thread.
///
/// - `auto` / `openhydra/auto` → a live model via [`pick_auto_model`].
/// - a concrete model that **is** served → itself.
/// - a concrete model that is **not** served → falls back to `auto` iff `fallback_unservable`.
///   The Anthropic `/v1/messages` path sets this: Claude Code can only send `claude-*` ids it
///   validates client-side, so an id OpenHydra doesn't serve means "route with OpenHydra's
///   models." The OpenAI path leaves it `false` (stays strict → a real "no provider" error).
async fn resolve_model(
    state: &AppState,
    requested: &str,
    fallback_unservable: bool,
    session_override: Option<&str>,
) -> Result<String, String> {
    let auto = is_auto_model(requested);
    // A concrete BYOK-mapped model is served directly (never announced on the swarm) — pass it
    // through untouched, and skip the discovery query entirely.
    if !auto && state.byok.provider_for(requested).is_some() {
        return Ok(requested.to_string());
    }
    // Fast path: a concrete model on the strict (OpenAI) surface needs no discovery query.
    // An explicit concrete model in the request always wins over any `/model` session pin.
    if !auto && !fallback_unservable {
        return Ok(requested.to_string());
    }
    let node = state.node.clone();
    let known = tokio::task::spawn_blocking(move || node.known_models())
        .await
        .map_err(|_| "model resolution failed".to_string())?
        .map_err(|e| e)?;
    // A concrete model that's actually served is used as-is (also wins over a session pin).
    if !auto && known.iter().any(|m| m == requested) {
        return Ok(requested.to_string());
    }
    // Now we're picking (request said `auto`, or an unservable id on the Anthropic surface):
    // a `/model <id>` session pin wins next — but only if it's a concrete model currently served
    // (a stale/typo'd pin falls through to the auto default rather than erroring).
    if let Some(sel) = session_override.map(str::trim).filter(|s| !s.is_empty()) {
        if !is_auto_model(sel) && known.iter().any(|m| m == sel) {
            return Ok(sel.to_string());
        }
    }
    // …else the auto default: an operator-preferred model if served, else the alphabetically-first.
    let preferred = std::env::var("OPENHYDRA_AUTO_MODEL").ok();
    pick_auto_model(&known, preferred.as_deref()).ok_or_else(|| {
        "no models available on the network yet — try again once a provider announces, or request \
         a specific model"
            .to_string()
    })
}

// ── Slash-commands (`/models`, `/model <id>`, `/help`) ───────────────────────
//
// Handled by the gateway before dispatch so they work inside every connected coding tool
// (OpenCode/Continue/Claude Code) — none of which offer a model picker natively. The parser
// and reply-rendering are pure (in `crate::slash`); this layer extracts the latest user turn,
// runs the (blocking) model query only when a command needs it, applies the session pin, and
// shapes the reply into the OpenAI or Anthropic wire form. Commands cost no inference and
// settle no receipt.

/// The session identity a `/model` pin is stored under: the caller's API key (Bearer or the
/// Anthropic `x-api-key`), else a single shared `local` session for the open loopback default.
/// Matches how a coding-tool session behaves (one fixed key ⇒ one selection).
fn session_key(headers: &axum::http::HeaderMap) -> String {
    headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .or_else(|| headers.get("x-api-key").and_then(|v| v.to_str().ok()))
        .map(|k| format!("key:{k}"))
        .unwrap_or_else(|| "local".to_string())
}

/// The plain text of the latest user turn, iff it's an ordinary user message (not a tool result
/// or an assistant/tool-call turn) — the only shape a slash-command may live in. `None` leaves
/// the request to route to inference untouched.
fn latest_user_text(messages: &[ChatMessage]) -> Option<&str> {
    let last = messages.last()?;
    if last.role != "user" || last.tool_calls.is_some() || last.tool_call_id.is_some() {
        return None;
    }
    let t = last.content.trim();
    if t.is_empty() {
        None
    } else {
        Some(last.content.as_str())
    }
}

/// The plain text of the latest Anthropic user turn, iff it's an ordinary text message (bare
/// string or text-only blocks). A turn carrying any `tool_result` / `tool_use` / non-text block
/// is not a command and returns `None`.
fn latest_ant_user_text(messages: &[AntMessage]) -> Option<String> {
    let last = messages.last()?;
    if last.role != "user" {
        return None;
    }
    let text = match &last.content {
        AntContent::Text(s) => s.clone(),
        AntContent::Blocks(blocks) => {
            let mut out = String::new();
            for b in blocks {
                match b {
                    AntBlock::Text { text } => out.push_str(text),
                    _ => return None, // tool_result / tool_use / image / … ⇒ not a plain command turn
                }
            }
            out
        }
    };
    if text.trim().is_empty() {
        None
    } else {
        Some(text)
    }
}

/// Query the currently-served models on a blocking thread (empty on error) — the **live** set,
/// used for routing/`auto` resolution (never route to a model no provider currently serves).
async fn known_models_async(state: &AppState) -> Vec<String> {
    let node = state.node.clone();
    tokio::task::spawn_blocking(move || node.known_models())
        .await
        .ok()
        .and_then(Result::ok)
        .unwrap_or_default()
}

/// The **browse** set for `/v1/models` and the `/models` command: the live set, smoothed by the
/// sticky cache so a provider that briefly drops off the network (a flaky-NAT node aging out of
/// `known_peers`) doesn't flicker out of the catalog mid-session. Routing/`auto` still use the
/// live set (`known_models_async`), so a sticky-but-currently-unserved model resolves on-demand
/// or falls back — never routes to a dead peer.
async fn browse_models(state: &AppState) -> Vec<String> {
    let live = known_models_async(state).await;
    state.model_cache.merge(live)
}

/// Sticky browse-list cache: a model stays listed for `ttl` after it was last seen live. Decoupled
/// from routing; mirrors the desktop W2 sticky-TTL for the app's provider list. Smooths transient
/// discovery churn (e.g. a CGNAT provider whose adverts briefly lapse) without ever claiming a
/// model is *serveable* — the catalog is a hint, and on-demand resolution is the source of truth.
struct ModelStickyCache {
    seen: std::sync::Mutex<std::collections::HashMap<String, std::time::Instant>>,
    ttl: Duration,
}

impl ModelStickyCache {
    fn new(ttl: Duration) -> Self {
        Self { seen: std::sync::Mutex::new(std::collections::HashMap::new()), ttl }
    }

    /// Stamp each currently-live model as seen now, evict entries older than `ttl`, and return the
    /// sorted union of live + still-fresh models.
    fn merge(&self, live: Vec<String>) -> Vec<String> {
        let now = std::time::Instant::now();
        let mut seen = self.seen.lock().unwrap();
        for m in live {
            seen.insert(m, now);
        }
        seen.retain(|_, t| now.duration_since(*t) <= self.ttl);
        let mut out: Vec<String> = seen.keys().cloned().collect();
        out.sort();
        out
    }
}

/// Handle a parsed command: fetch the live model set (only when the command needs it), render
/// the reply, and apply any new session pin. Returns the assistant reply text.
async fn handle_slash(state: &AppState, skey: &str, cmd: SlashCommand) -> String {
    let known = match cmd {
        // Browse/validate against the sticky catalog so a momentary churn dip doesn't make `/models`
        // drop a live provider or `/model <id>` reject a just-flickered model.
        SlashCommand::Models(_) | SlashCommand::ModelSet(_) => browse_models(state).await,
        SlashCommand::ModelShow | SlashCommand::Help => Vec::new(),
    };
    let current = state.sessions.get(skey);
    let result = slash::render(&cmd, &known, current.as_deref());
    if let Some(model) = result.set_model {
        state.sessions.set(skey, model);
    }
    result.reply
}

/// A gateway-authored assistant completion (a slash-command reply) in the OpenAI shape: a
/// buffered `chat.completion`, or a minimal SSE stream (role → content → stop → `[DONE]`) when
/// the client asked to stream. Zero `usage`; no `openhydra` telemetry block (no inference ran).
fn command_response_openai(id: &str, model: &str, created: u64, text: String, stream: bool) -> Response {
    if !stream {
        return Json(json!({
            "id": id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": text },
                "finish_reason": "stop",
            }],
            "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 },
        }))
        .into_response();
    }
    let frames = vec![
        stream_chunk(id, model, created, json!({ "role": "assistant" }), None),
        stream_chunk(id, model, created, json!({ "content": text }), None),
        stream_chunk(id, model, created, json!({}), Some("stop")),
        "[DONE]".to_string(),
    ];
    let body = tokio_stream::iter(
        frames
            .into_iter()
            .map(|d| Ok::<Event, std::convert::Infallible>(Event::default().data(d))),
    );
    Sse::new(body).into_response()
}

/// A gateway-authored assistant message (a slash-command reply) in the Anthropic Messages shape:
/// a buffered `message`, or the full SSE event sequence when the client asked to stream. The
/// streamed form keeps the canonical `message_start` → … → `message_stop` order (SDK-safe — no
/// `ping` before `message_start`). Zero `usage`.
fn command_response_anthropic(id: &str, model: &str, text: String, stream: bool) -> Response {
    if !stream {
        return Json(json!({
            "id": id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{ "type": "text", "text": text }],
            "stop_reason": "end_turn",
            "stop_sequence": Value::Null,
            "usage": { "input_tokens": 0, "output_tokens": 0 },
        }))
        .into_response();
    }
    let frames: Vec<(&'static str, Value)> = vec![
        ("message_start", json!({ "type": "message_start", "message": { "id": id, "type": "message", "role": "assistant", "model": model, "content": [], "stop_reason": Value::Null, "stop_sequence": Value::Null, "usage": { "input_tokens": 0, "output_tokens": 0 } } })),
        ("content_block_start", json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "text", "text": "" } })),
        ("content_block_delta", json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "text_delta", "text": text } })),
        ("content_block_stop", json!({ "type": "content_block_stop", "index": 0 })),
        ("message_delta", json!({ "type": "message_delta", "delta": { "stop_reason": "end_turn", "stop_sequence": Value::Null }, "usage": { "output_tokens": 0 } })),
        ("message_stop", json!({ "type": "message_stop" })),
    ];
    let body = tokio_stream::iter(
        frames
            .into_iter()
            .map(|(n, d)| Ok::<Event, std::convert::Infallible>(ant_sse(n, d))),
    );
    Sse::new(body).into_response()
}

// ── Response builders (pure) ─────────────────────────────────────────────────

/// `usage` object: prompt tokens are the engine's count (0 if it reports none),
/// completion tokens are what the pipeline counted.
fn usage_value(summary: &ServeSummary) -> Value {
    let prompt = summary.metrics.engine.prompt_eval_count;
    let completion = summary.tokens;
    json!({
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        // F-C7: saturating add (matches the saturating_sub nearby) so a bogus
        // engine count can't panic the gateway on overflow.
        "total_tokens": prompt.saturating_add(completion),
    })
}

/// The `openhydra` telemetry block: three views of one request (engine ground-truth,
/// pipeline end-to-end, and where the wall-clock went hop-by-hop).
///
/// `ttft` is the measured time to the **first delta** reaching the responder (P1 streaming): on a
/// streaming serve it's a small fraction of `wall` (the first token arrives while the rest
/// generate); on a buffered serve / legacy-fallback the whole completion lands at once, so it
/// equals `wall`. So `ttft_ms ≪ wall_ms` is also the signal that streaming actually engaged.
fn openhydra_block(summary: &ServeSummary, wall: std::time::Duration, ttft: std::time::Duration) -> Value {
    let e = &summary.metrics.engine;
    let r1 = |x: f64| (x * 10.0).round() / 10.0;
    let ms = |ns: u64| ns / 1_000_000;
    let tps = |n: u64, dur_ns: u64| if dur_ns > 0 { n as f64 / (dur_ns as f64 / 1e9) } else { 0.0 };

    let wall_ns = wall.as_nanos() as u64;
    let provider_serve_ns = summary.metrics.provider_serve_ns;
    let net_rtt_ns = summary.proxy_roundtrip_ns.saturating_sub(provider_serve_ns);
    let gateway_ns = wall_ns
        .saturating_sub(summary.proxy_roundtrip_ns)
        .saturating_sub(summary.discover_ns);
    let pipeline_tps = if wall_ns > 0 { summary.tokens as f64 / (wall_ns as f64 / 1e9) } else { 0.0 };

    json!({
        "tokens": summary.tokens,
        "engine": {
            "native_tps": r1(tps(e.eval_count, e.eval_duration_ns)),
            "prefill_tps": r1(tps(e.prompt_eval_count, e.prompt_eval_duration_ns)),
            "prompt_tokens": e.prompt_eval_count,
            "load_ms": ms(e.load_duration_ns),
            "prompt_eval_ms": ms(e.prompt_eval_duration_ns),
            "eval_ms": ms(e.eval_duration_ns),
            "engine_total_ms": ms(e.total_duration_ns),
        },
        "pipeline": {
            "pipeline_tps": r1(pipeline_tps),
            "wall_ms": ms(wall_ns),
            "overhead_ms": ms(wall_ns.saturating_sub(e.total_duration_ns)),
            // P1 streaming: the real time to the first delta (≪ wall when streaming engaged).
            "ttft_ms": ms((ttft.as_nanos() as u64).min(wall_ns)),
        },
        "hops_ms": {
            "discover": ms(summary.discover_ns),
            "proxy_roundtrip": ms(summary.proxy_roundtrip_ns),
            "provider_serve": ms(provider_serve_ns),
            "network_rtt": ms(net_rtt_ns),
            "gateway_overhead": ms(gateway_ns),
        },
    })
}

/// One streaming `chat.completion.chunk` carrying an arbitrary `delta`.
fn stream_chunk(id: &str, model: &str, created: u64, delta: Value, finish: Option<&str>) -> String {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }],
    })
    .to_string()
}

/// The non-streaming `chat.completion` object. When the model requested tool calls, the
/// assistant message carries `tool_calls` with `content: null` and `finish_reason` is
/// `"tool_calls"` (the OpenAI shape a coding agent expects); otherwise it carries the
/// generated text and finishes `"stop"`.
fn completion_object(
    id: &str,
    model: &str,
    created: u64,
    content: &str,
    summary: &ServeSummary,
    wall: std::time::Duration,
    ttft: std::time::Duration,
) -> Value {
    let (message, finish) = if summary.tool_calls.is_empty() {
        (json!({ "role": "assistant", "content": content }), "stop")
    } else {
        (
            json!({ "role": "assistant", "content": Value::Null, "tool_calls": summary.tool_calls }),
            "tool_calls",
        )
    };
    json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish,
        }],
        "usage": usage_value(summary),
        "openhydra": openhydra_block(summary, wall, ttft),
    })
}

// ── Handler ──────────────────────────────────────────────────────────────────

/// Spawn the blocking `complete` on a plain OS thread, returning the event channel and the
/// start instant. The worker forwards each delta, then a terminal `Done`/`Error`.
#[allow(clippy::too_many_arguments)]
fn spawn_worker(
    node: Arc<ConsumerNode>,
    model: String,
    messages: Vec<ChatMessage>,
    max_tokens: Option<u32>,
    temperature: Option<f64>,
    tools: Vec<Value>,
    think: Option<bool>,
    gen_permit: OwnedSemaphorePermit,
) -> (
    tokio::sync::mpsc::UnboundedReceiver<GatewayEvent>,
    std::time::Instant,
) {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<GatewayEvent>();
    let started = std::time::Instant::now();
    std::thread::spawn(move || {
        // Hold the global generation permit (G3b) for the worker's whole lifetime; it is
        // released when this thread exits, freeing the slot for the next completion.
        let _gen_permit = gen_permit;
        let mut on_delta = |d: &str| {
            let _ = tx.send(GatewayEvent::Delta(d.to_string()));
        };
        match node.complete(&model, messages, max_tokens, temperature, tools, think, &mut on_delta) {
            Ok(summary) => {
                let _ = tx.send(GatewayEvent::Done(Box::new(summary)));
            }
            Err(e) => {
                let _ = tx.send(GatewayEvent::Error(e.to_string()));
            }
        }
    });
    (rx, started)
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    body: Result<Json<ChatRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    // Malformed JSON / wrong content-type → 400, OpenAI-shaped (not axum's plain text).
    let Json(req) = match body {
        Ok(j) => j,
        Err(rej) => {
            return openai_error(StatusCode::BAD_REQUEST, &rej.body_text(), "invalid_request_error")
        }
    };
    state.metrics.incr_chat();
    if let Some(resp) = validate(&req) {
        return resp;
    }

    let id = next_id();
    let created = unix_now();
    let want_usage = req.stream_options.as_ref().is_some_and(|o| o.include_usage);
    let stream = req.stream;
    let skey = session_key(&headers);
    // Gateway `/`-commands (`/models`, `/model <id>`, `/help`): handled here (before AUP/dispatch)
    // and answered as an ordinary assistant message — no inference, no receipt. Only a plain user
    // turn whose whole text is a recognized command is intercepted; anything else routes normally.
    if let Some(cmd) = latest_user_text(&req.messages).and_then(slash::parse) {
        let reply = handle_slash(&state, &skey, cmd).await;
        return command_response_openai(&id, &req.model, created, reply, stream);
    }
    // AUP → generation slot → model resolution (`openhydra/auto`; OpenAI stays strict) → BYOK/swarm
    // dispatch, all inside spawn_dispatch so the backstop bounds the discovery query too. A `/model`
    // session pin steers the `auto`/default case; the resolved concrete model is echoed back.
    let session_pin = state.sessions.get(&skey);
    let think = req.effective_think();
    let (model, rx, started) = match spawn_dispatch(
        &state,
        &headers,
        &req.model,
        false,
        session_pin.as_deref(),
        req.messages,
        req.max_tokens,
        req.temperature,
        req.tools,
        think,
    )
    .await
    {
        Ok(v) => v,
        Err(e) => return dispatch_err_openai(e),
    };

    if stream {
        stream_response(id, model, created, want_usage, rx, started, state.metrics.clone())
    } else {
        buffered_response(id, model, created, rx, started, state.metrics.clone()).await
    }
}

/// A dispatch failure that each API surface renders in its own error shape (OpenAI vs Anthropic).
enum DispatchErr {
    /// AUP floor rejected the request.
    Aup(String),
    /// The global generation backstop (G3b) is full.
    Overloaded,
    /// A BYOK-mapped model has no resolvable API key.
    ByokKeyMissing(String),
    /// Model resolution found nothing to serve (e.g. `auto` with no live models).
    NoModels(String),
}

/// Shared route/dispatch used by both `/v1/chat/completions` and `/v1/messages`: apply the AUP
/// floor, acquire a generation slot, **then** resolve the model and spawn the swarm (or BYOK)
/// worker. Resolution runs after the permit so the backstop bounds its (blocking) discovery query,
/// and recognises BYOK models before any swarm fallback. Returns the *resolved* model (for the
/// response echo) + the event channel + start instant, or a typed error the caller shapes.
#[allow(clippy::too_many_arguments)]
async fn spawn_dispatch(
    state: &AppState,
    headers: &axum::http::HeaderMap,
    requested_model: &str,
    fallback_unservable: bool,
    session_override: Option<&str>,
    messages: Vec<ChatMessage>,
    max_tokens: Option<u32>,
    temperature: Option<f64>,
    tools: Vec<Value>,
    think: Option<bool>,
) -> Result<(String, tokio::sync::mpsc::UnboundedReceiver<GatewayEvent>, std::time::Instant), DispatchErr>
{
    // AUP floor: refuse a policy-violating request before spending a discovery/route on it.
    if let AupDecision::Deny(reason) = state.aup.evaluate(&messages, max_tokens) {
        return Err(DispatchErr::Aup(reason));
    }
    // G3b: acquire the global generation slot BEFORE any (blocking) discovery work, so the backstop
    // bounds resolution too.
    let gen_permit = match state.gen_limiter.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            state.metrics.record_rate_limited();
            return Err(DispatchErr::Overloaded);
        }
    };
    // Resolve `openhydra/auto` (and, on the Anthropic surface, an unservable id) to a concrete
    // model. BYOK-mapped ids are recognised inside `resolve_model` so they're never rewritten.
    let model = resolve_model(state, requested_model, fallback_unservable, session_override)
        .await
        .map_err(DispatchErr::NoModels)?;
    // BYOK (#34): a mapped model is served by calling the hosted backend directly, bypassing the
    // swarm. The key is the caller's `X-Provider-Api-Key` if present, else the operator's.
    let (rx, started) = if let Some(provider) = state.byok.provider_for(&model) {
        let caller_key = headers.get("x-provider-api-key").and_then(|v| v.to_str().ok());
        let key = match state.byok.resolve_key(provider, caller_key) {
            Some(k) => k,
            None => return Err(DispatchErr::ByokKeyMissing(model)),
        };
        spawn_byok_worker(
            provider,
            state.byok.base_url(provider).to_string(),
            key,
            model.clone(),
            messages,
            max_tokens,
            temperature,
            gen_permit,
        )
    } else {
        spawn_worker(state.node.clone(), model.clone(), messages, max_tokens, temperature, tools, think, gen_permit)
    };
    Ok((model, rx, started))
}

/// Render a [`DispatchErr`] as an OpenAI-shaped error response.
fn dispatch_err_openai(e: DispatchErr) -> Response {
    match e {
        DispatchErr::Aup(reason) => {
            openai_error(StatusCode::BAD_REQUEST, &reason, "invalid_request_error")
        }
        DispatchErr::Overloaded => openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "server at generation capacity, retry shortly",
            "server_overloaded",
        ),
        DispatchErr::ByokKeyMissing(m) => openai_error(
            StatusCode::BAD_REQUEST,
            &format!("no API key available for BYOK model '{m}'"),
            "invalid_request_error",
        ),
        DispatchErr::NoModels(m) => openai_error(StatusCode::SERVICE_UNAVAILABLE, &m, "no_provider"),
    }
}

/// Like [`spawn_worker`] but serves a BYOK model by calling the hosted backend directly
/// (#34), translating its [`ServeOutcome`](crate::adapter::ServeOutcome) into the same
/// `GatewayEvent` stream the SSE/buffered responders consume. Runs on a plain OS thread (the
/// adapters are blocking).
#[allow(clippy::too_many_arguments)]
fn spawn_byok_worker(
    provider: ByokProvider,
    base_url: String,
    key: String,
    model: String,
    messages: Vec<ChatMessage>,
    max_tokens: Option<u32>,
    temperature: Option<f64>,
    gen_permit: OwnedSemaphorePermit,
) -> (
    tokio::sync::mpsc::UnboundedReceiver<GatewayEvent>,
    std::time::Instant,
) {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<GatewayEvent>();
    let started = std::time::Instant::now();
    std::thread::spawn(move || {
        // Hold the global generation permit (G3b) for the worker's whole lifetime.
        let _gen_permit = gen_permit;
        // BYOK tool-calling (forwarding `tools` to Anthropic/Gemini) is a separate follow-up;
        // the hosted adapters ignore tools today, so a mapped model serves plain text.
        let request = InferenceRequest {
            model_ref: model,
            messages,
            max_tokens,
            temperature,
            tools: Vec::new(),
            // BYOK thinking-control forwarding is a separate follow-up; hosted adapters use
            // their own defaults today.
            think: None,
        };
        let mut on_delta = |d: &str| {
            let _ = tx.send(GatewayEvent::Delta(d.to_string()));
        };
        let result = match provider {
            ByokProvider::Anthropic => crate::live_anthropic(&base_url, &key)
                .and_then(|a| a.serve_stream(&request, &mut on_delta)),
            ByokProvider::Gemini => crate::live_gemini(&base_url, &key)
                .and_then(|a| a.serve_stream(&request, &mut on_delta)),
        };
        match result {
            Ok(outcome) => {
                let summary = ServeSummary {
                    tokens: outcome.tokens,
                    ok: outcome.done,
                    metrics: ServeMetrics {
                        engine: outcome.engine,
                        provider_serve_ns: started.elapsed().as_nanos() as u64,
                    },
                    discover_ns: 0,
                    proxy_roundtrip_ns: 0,
                    tool_calls: outcome.tool_calls,
                };
                let _ = tx.send(GatewayEvent::Done(Box::new(summary)));
            }
            Err(e) => {
                let _ = tx.send(GatewayEvent::Error(e.to_string()));
            }
        }
    });
    (rx, started)
}

/// `stream: true` → SSE. First an assistant-role chunk, then content deltas, then a
/// terminal `stop` chunk (with `usage` when requested) carrying the `openhydra` block, then
/// the OpenAI `[DONE]` sentinel. A mid-flight failure is surfaced as an error chunk.
fn stream_response(
    id: String,
    model: String,
    created: u64,
    want_usage: bool,
    rx: tokio::sync::mpsc::UnboundedReceiver<GatewayEvent>,
    started: std::time::Instant,
    metrics: Arc<Metrics>,
) -> Response {
    let role = {
        let data = stream_chunk(&id, &model, created, json!({ "role": "assistant" }), None);
        tokio_stream::once(Ok::<Event, std::convert::Infallible>(Event::default().data(data)))
    };
    let (id_s, model_s) = (id.clone(), model.clone());
    // P1 streaming: stamp the first delta so `ttft_ms` reflects real time-to-first-token — the
    // whole point of streaming is that this lands far ahead of `wall`.
    let mut ttft: Option<std::time::Duration> = None;
    let body = UnboundedReceiverStream::new(rx).map(move |ev| {
        let data = match ev {
            GatewayEvent::Delta(t) => {
                ttft.get_or_insert_with(|| started.elapsed());
                stream_chunk(&id_s, &model_s, created, json!({ "content": t }), None)
            }
            GatewayEvent::Done(summary) => {
                let wall = started.elapsed();
                log_completion(&summary, wall);
                metrics.record_completion(
                    summary.tokens,
                    wall,
                    summary.discover_ns,
                    summary.proxy_roundtrip_ns,
                );
                // Tool calls (if any) ride the final chunk's delta with finish_reason
                // "tool_calls"; a plain completion finishes "stop" with an empty delta.
                // (Incremental per-argument streaming of tool_calls is a later refinement —
                // delivered as one delta here, which OpenAI clients accumulate fine.)
                let (delta, finish) = if summary.tool_calls.is_empty() {
                    (json!({}), "stop")
                } else {
                    (json!({ "tool_calls": summary.tool_calls }), "tool_calls")
                };
                let mut chunk: Value =
                    serde_json::from_str(&stream_chunk(&id_s, &model_s, created, delta, Some(finish)))
                        .unwrap_or_else(|_| json!({}));
                // No delta (tool-call-only turn) → first "token" is effectively at completion.
                chunk["openhydra"] = openhydra_block(&summary, wall, ttft.unwrap_or(wall));
                if want_usage {
                    chunk["usage"] = usage_value(&summary);
                }
                chunk.to_string()
            }
            GatewayEvent::Error(m) => {
                metrics.record_error();
                let (_, etype) = classify_error(&m);
                json!({
                    "id": id_s, "object": "chat.completion.chunk", "created": created, "model": model_s,
                    "choices": [{ "index": 0, "delta": {}, "finish_reason": "error" }],
                    "error": { "message": m, "type": etype },
                })
                .to_string()
            }
        };
        Ok::<Event, std::convert::Infallible>(Event::default().data(data))
    });
    let done = tokio_stream::once(Ok(Event::default().data("[DONE]")));
    // Keep-alive comments (`:`) during any stall (cold-model load / prefill / a slow provider)
    // keep the connection warm so a client read-timeout can't trip mid-stream; SSE parsers ignore
    // comment lines.
    Sse::new(role.chain(body).chain(done))
        .keep_alive(KeepAlive::new().interval(KEEPALIVE_INTERVAL))
        .into_response()
}

/// `stream: false` (the OpenAI default) → collect the whole completion and return a single
/// `chat.completion` object, or a proper HTTP error if the route failed.
async fn buffered_response(
    id: String,
    model: String,
    created: u64,
    mut rx: tokio::sync::mpsc::UnboundedReceiver<GatewayEvent>,
    started: std::time::Instant,
    metrics: Arc<Metrics>,
) -> Response {
    let mut content = String::new();
    let mut outcome: Option<Result<Box<ServeSummary>, String>> = None;
    // Time the first delta even though the HTTP body is buffered: with P1 streaming the tokens
    // arrive incrementally over the channel, so this is the real generation time-to-first-token.
    let mut ttft: Option<std::time::Duration> = None;
    while let Some(ev) = rx.recv().await {
        match ev {
            GatewayEvent::Delta(t) => {
                ttft.get_or_insert_with(|| started.elapsed());
                content.push_str(&t);
            }
            GatewayEvent::Done(s) => outcome = Some(Ok(s)),
            GatewayEvent::Error(m) => outcome = Some(Err(m)),
        }
    }
    match outcome {
        Some(Ok(summary)) => {
            let wall = started.elapsed();
            log_completion(&summary, wall);
            metrics.record_completion(
                summary.tokens,
                wall,
                summary.discover_ns,
                summary.proxy_roundtrip_ns,
            );
            Json(completion_object(&id, &model, created, &content, &summary, wall, ttft.unwrap_or(wall)))
                .into_response()
        }
        Some(Err(m)) => {
            metrics.record_error();
            let (status, etype) = classify_error(&m);
            openai_error(status, &m, etype)
        }
        None => openai_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "gateway worker produced no result",
            "internal_error",
        ),
    }
}

fn log_completion(summary: &ServeSummary, wall: std::time::Duration) {
    let eval_ns = summary.metrics.engine.eval_duration_ns;
    let native = if eval_ns > 0 {
        summary.metrics.engine.eval_count as f64 / (eval_ns as f64 / 1e9)
    } else {
        0.0
    };
    let pipeline = summary.tokens as f64 / wall.as_secs_f64().max(1e-9);
    tracing::info!(
        tokens = summary.tokens,
        native_tps = native,
        pipeline_tps = pipeline,
        wall_ms = wall.as_millis() as u64,
        provider_serve_ms = summary.metrics.provider_serve_ns / 1_000_000,
        proxy_roundtrip_ms = summary.proxy_roundtrip_ns / 1_000_000,
        "completion done"
    );
}

// ── Anthropic Messages API (`POST /v1/messages`) ─────────────────────────────
//
// A native inbound translation of Anthropic's Messages API onto the same swarm route as
// `/v1/chat/completions`, so Anthropic-SDK tools (Claude Code, etc.) plug in with only a
// `base_url` change — no LiteLLM shim. Requests are converted to the internal OpenAI-shaped
// `ChatMessage` list (system prompt, text, and full tool-use round-trip), routed via
// `spawn_dispatch`, then the response is re-shaped into Anthropic's Messages JSON (buffered)
// or its SSE event sequence (streaming).

/// Anthropic `POST /v1/messages` request fields we honour (others ignored).
#[derive(Debug, Deserialize)]
struct MessagesRequest {
    #[serde(default)]
    model: String,
    #[serde(default)]
    messages: Vec<AntMessage>,
    /// A system prompt — a bare string or an array of text blocks.
    #[serde(default)]
    system: Option<AntSystem>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    stream: bool,
    /// Anthropic tool definitions (`name` / `description` / `input_schema`).
    #[serde(default)]
    tools: Vec<AntTool>,
}

#[derive(Debug, Deserialize)]
struct AntMessage {
    role: String,
    content: AntContent,
}

/// A message's content: either a bare string or a list of typed blocks.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum AntContent {
    Text(String),
    Blocks(Vec<AntBlock>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum AntBlock {
    #[serde(rename = "text")]
    Text { #[serde(default)] text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        #[serde(default)]
        content: AntToolResultContent,
    },
    /// Any other block kind (image, document, …) — carried so deserialization doesn't fail,
    /// but dropped in conversion (text engines can't consume it).
    #[serde(other)]
    Other,
}

/// A `tool_result` block's payload — a string, or blocks we flatten to their text.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum AntToolResultContent {
    Text(String),
    Blocks(Vec<Value>),
}
impl Default for AntToolResultContent {
    fn default() -> Self {
        AntToolResultContent::Text(String::new())
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum AntSystem {
    Text(String),
    Blocks(Vec<Value>),
}

#[derive(Debug, Deserialize)]
struct AntTool {
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    input_schema: Value,
}

/// Flatten an Anthropic system prompt (string or text blocks) to a single string.
fn ant_system_text(sys: &AntSystem) -> String {
    match sys {
        AntSystem::Text(s) => s.clone(),
        AntSystem::Blocks(blocks) => blocks
            .iter()
            .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(""),
    }
}

/// Flatten a `tool_result` payload to a string (OpenAI `role:"tool"` content is text).
fn ant_tool_result_text(content: &AntToolResultContent) -> String {
    match content {
        AntToolResultContent::Text(s) => s.clone(),
        AntToolResultContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(""),
    }
}

/// Convert an Anthropic (system, messages) pair into the internal OpenAI-shaped `ChatMessage`
/// list: the system prompt becomes a leading `system` turn; assistant `tool_use` blocks become
/// `tool_calls`; user `tool_result` blocks become `role:"tool"` messages keyed by id.
fn to_chat_messages(system: Option<&AntSystem>, messages: &[AntMessage]) -> Vec<ChatMessage> {
    let mut out = Vec::new();
    if let Some(sys) = system {
        let s = ant_system_text(sys);
        if !s.is_empty() {
            out.push(ChatMessage::new("system", s));
        }
    }
    for m in messages {
        match &m.content {
            AntContent::Text(s) => out.push(ChatMessage::new(&m.role, s.clone())),
            AntContent::Blocks(blocks) if m.role == "assistant" => {
                let mut text = String::new();
                let mut tool_calls = Vec::new();
                for b in blocks {
                    match b {
                        AntBlock::Text { text: t } => text.push_str(t),
                        AntBlock::ToolUse { id, name, input } => tool_calls.push(ToolCall {
                            id: id.clone(),
                            kind: "function".to_string(),
                            function: ToolCallFunction { name: name.clone(), arguments: input.to_string() },
                        }),
                        _ => {}
                    }
                }
                out.push(ChatMessage {
                    role: "assistant".to_string(),
                    content: text,
                    tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
                    tool_call_id: None,
                    name: None,
                });
            }
            // user (or any non-assistant) turn with blocks: tool_result → `role:"tool"` messages,
            // text → a trailing user message (matching OpenAI's assistant→tool→user ordering).
            AntContent::Blocks(blocks) => {
                let mut text = String::new();
                for b in blocks {
                    match b {
                        AntBlock::Text { text: t } => text.push_str(t),
                        AntBlock::ToolResult { tool_use_id, content } => out.push(ChatMessage {
                            role: "tool".to_string(),
                            content: ant_tool_result_text(content),
                            tool_calls: None,
                            tool_call_id: Some(tool_use_id.clone()),
                            name: None,
                        }),
                        _ => {}
                    }
                }
                if !text.is_empty() {
                    out.push(ChatMessage::new("user", text));
                }
            }
        }
    }
    out
}

/// Anthropic tool defs → OpenAI `tools` (the shape the engine already forwards).
fn to_openai_tools(tools: &[AntTool]) -> Vec<Value> {
    tools
        .iter()
        .map(|t| {
            json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description.clone().unwrap_or_default(),
                    "parameters": t.input_schema,
                }
            })
        })
        .collect()
}

static MSG_COUNTER: AtomicU64 = AtomicU64::new(1);
fn next_msg_id() -> String {
    format!("msg_{}", MSG_COUNTER.fetch_add(1, Ordering::Relaxed))
}

/// An Anthropic-shaped error response: `{ "type":"error", "error": { type, message } }`.
fn anthropic_error(status: StatusCode, message: &str, etype: &str) -> Response {
    (status, Json(json!({ "type": "error", "error": { "type": etype, "message": message } })))
        .into_response()
}

fn dispatch_err_anthropic(e: DispatchErr) -> Response {
    match e {
        DispatchErr::Aup(reason) => anthropic_error(StatusCode::BAD_REQUEST, &reason, "invalid_request_error"),
        DispatchErr::Overloaded => anthropic_error(
            StatusCode::from_u16(529).unwrap_or(StatusCode::SERVICE_UNAVAILABLE),
            "server at generation capacity, retry shortly",
            "overloaded_error",
        ),
        DispatchErr::ByokKeyMissing(m) => anthropic_error(
            StatusCode::BAD_REQUEST,
            &format!("no API key available for BYOK model '{m}'"),
            "invalid_request_error",
        ),
        DispatchErr::NoModels(m) => anthropic_error(StatusCode::SERVICE_UNAVAILABLE, &m, "api_error"),
    }
}

/// The final Anthropic `message` object: text block + one `tool_use` block per tool call.
fn anthropic_message_object(id: &str, model: &str, content: &str, summary: &ServeSummary) -> Value {
    let mut blocks: Vec<Value> = Vec::new();
    if !content.is_empty() {
        blocks.push(json!({ "type": "text", "text": content }));
    }
    for tc in &summary.tool_calls {
        let input: Value = serde_json::from_str(&tc.function.arguments).unwrap_or_else(|_| json!({}));
        blocks.push(json!({ "type": "tool_use", "id": tc.id, "name": tc.function.name, "input": input }));
    }
    let stop_reason = if summary.tool_calls.is_empty() { "end_turn" } else { "tool_use" };
    json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": stop_reason,
        "stop_sequence": Value::Null,
        "usage": {
            "input_tokens": summary.metrics.engine.prompt_eval_count,
            "output_tokens": summary.tokens,
        },
    })
}

async fn messages(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    body: Result<Json<MessagesRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(req) = match body {
        Ok(j) => j,
        Err(rej) => {
            return anthropic_error(StatusCode::BAD_REQUEST, &rej.body_text(), "invalid_request_error")
        }
    };
    state.metrics.incr_chat();
    if req.model.trim().is_empty() {
        return anthropic_error(StatusCode::BAD_REQUEST, "missing required field: model", "invalid_request_error");
    }
    if req.messages.is_empty() {
        return anthropic_error(StatusCode::BAD_REQUEST, "missing or empty field: messages", "invalid_request_error");
    }

    let id = next_msg_id();
    let stream = req.stream;
    let skey = session_key(&headers);
    // Gateway `/`-commands work through Claude Code too (before dispatch; answered as an assistant
    // message). Only a plain-text user turn whose whole content is a command is intercepted.
    if let Some(cmd) = latest_ant_user_text(&req.messages).and_then(|t| slash::parse(&t)) {
        let reply = handle_slash(&state, &skey, cmd).await;
        return command_response_anthropic(&id, &req.model, reply, stream);
    }
    let messages = to_chat_messages(req.system.as_ref(), &req.messages);
    let tools = to_openai_tools(&req.tools);
    // Bridge Claude Code: on this Anthropic surface an unservable id (e.g. the `claude-*` id Claude
    // Code insists on) routes to a live model; `openhydra/auto`, served, and BYOK models resolve as
    // usual. A `/model` session pin steers that bridged/`auto` case. All inside spawn_dispatch (the
    // permit bounds the discovery query). Resolved model echoed back in the response.
    let session_pin = state.sessions.get(&skey);
    let (model, rx, started) = match spawn_dispatch(
        &state,
        &headers,
        &req.model,
        true,
        session_pin.as_deref(),
        messages,
        req.max_tokens,
        req.temperature,
        tools,
        // Anthropic-surface thinking control (`thinking:{type}`) mapping is a separate
        // follow-up; the OpenAI surface carries the Ollama-native `think` switch today.
        None,
    )
    .await
    {
        Ok(v) => v,
        Err(e) => return dispatch_err_anthropic(e),
    };

    if stream {
        anthropic_stream_response(id, model, rx, started, state.metrics.clone())
    } else {
        anthropic_buffered_response(id, model, rx, started, state.metrics.clone()).await
    }
}

async fn anthropic_buffered_response(
    id: String,
    model: String,
    mut rx: tokio::sync::mpsc::UnboundedReceiver<GatewayEvent>,
    started: std::time::Instant,
    metrics: Arc<Metrics>,
) -> Response {
    let mut content = String::new();
    let mut outcome: Option<Result<Box<ServeSummary>, String>> = None;
    while let Some(ev) = rx.recv().await {
        match ev {
            GatewayEvent::Delta(t) => content.push_str(&t),
            GatewayEvent::Done(s) => outcome = Some(Ok(s)),
            GatewayEvent::Error(m) => outcome = Some(Err(m)),
        }
    }
    match outcome {
        Some(Ok(summary)) => {
            let wall = started.elapsed();
            log_completion(&summary, wall);
            metrics.record_completion(summary.tokens, wall, summary.discover_ns, summary.proxy_roundtrip_ns);
            Json(anthropic_message_object(&id, &model, &content, &summary)).into_response()
        }
        Some(Err(m)) => {
            metrics.record_error();
            let (status, _) = classify_error(&m);
            anthropic_error(status, &m, "api_error")
        }
        None => anthropic_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "gateway worker produced no result",
            "api_error",
        ),
    }
}

/// One named Anthropic SSE event.
fn ant_sse(name: &str, data: Value) -> Event {
    Event::default().event(name).data(data.to_string())
}

/// Translate the internal `GatewayEvent` stream into Anthropic's SSE event sequence
/// (`message_start` → `content_block_start` → `content_block_delta`* → `content_block_stop`
/// → `message_delta` → `message_stop`). A worker task fans one `Done` out into the several
/// terminal events Anthropic expects; tool calls become extra `tool_use` blocks.
fn anthropic_stream_response(
    id: String,
    model: String,
    rx: tokio::sync::mpsc::UnboundedReceiver<GatewayEvent>,
    started: std::time::Instant,
    metrics: Arc<Metrics>,
) -> Response {
    // The worker emits `(event_name, json)` frames; convert to axum SSE `Event` only at the boundary
    // so the translation logic stays unit-testable without axum.
    let (etx, erx) = tokio::sync::mpsc::unbounded_channel::<(&'static str, Value)>();
    tokio::spawn(anthropic_stream_worker(id, model, rx, started, metrics, etx));
    let body = UnboundedReceiverStream::new(erx)
        .map(|(name, data)| Ok::<Event, std::convert::Infallible>(ant_sse(name, data)));
    Sse::new(body)
        .keep_alive(KeepAlive::new().interval(KEEPALIVE_INTERVAL))
        .into_response()
}

/// Translate the internal `GatewayEvent` stream into ordered Anthropic SSE frames sent to `out`.
/// Peeks the first frame: an immediate error yields a single `error` frame with no message
/// envelope; otherwise `message_start` → `content_block_start` → deltas → `content_block_stop`
/// → (tool blocks) → `message_delta` → `message_stop`. Factored out so it's testable without axum.
async fn anthropic_stream_worker(
    id: String,
    model: String,
    mut rx: tokio::sync::mpsc::UnboundedReceiver<GatewayEvent>,
    started: std::time::Instant,
    metrics: Arc<Metrics>,
    out: tokio::sync::mpsc::UnboundedSender<(&'static str, Value)>,
) {
    // Wait for the first frame before committing to a message envelope. If the generation errors
    // before producing anything, emit ONLY a clean `error` frame. The pre-first-frame gap is kept
    // warm by the SSE comment keep-alive on the response — NO native `ping` here, because a `ping`
    // before `message_start` makes the official Anthropic SDK `.stream()` accumulator raise
    // "Unexpected event order, got ping before message_start" and abort.
    let first = match rx.recv().await {
        Some(GatewayEvent::Error(m)) => {
            metrics.record_error();
            let _ = out.send(("error", json!({ "type": "error", "error": { "type": "api_error", "message": m } })));
            return;
        }
        Some(ev) => ev, // a Delta or Done — the message really started
        None => return, // stream closed with nothing
    };
    let _ = out.send((
        "message_start",
        json!({
            "type": "message_start",
            "message": {
                "id": id, "type": "message", "role": "assistant", "model": model,
                "content": [], "stop_reason": Value::Null, "stop_sequence": Value::Null,
                "usage": { "input_tokens": 0, "output_tokens": 0 },
            }
        }),
    ));
    let _ = out.send((
        "content_block_start",
        json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "text", "text": "" } }),
    ));
    // Process the first frame, then drain the rest. A native `ping` is emitted only on a mid-content
    // stall — i.e. AFTER `message_start` (canonical + SDK-safe); `biased` makes a ready frame win, so
    // pings never appear once tokens are flowing.
    let mut ev = first;
    'drain: loop {
        match ev {
            GatewayEvent::Delta(t) => {
                let _ = out.send((
                    "content_block_delta",
                    json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "text_delta", "text": t } }),
                ));
            }
            GatewayEvent::Done(summary) => {
                let wall = started.elapsed();
                log_completion(&summary, wall);
                metrics.record_completion(summary.tokens, wall, summary.discover_ns, summary.proxy_roundtrip_ns);
                let _ = out.send(("content_block_stop", json!({ "type": "content_block_stop", "index": 0 })));
                // Each tool call → a `tool_use` block whose arguments stream as one
                // `input_json_delta` (clients accumulate + parse it).
                for (i, tc) in summary.tool_calls.iter().enumerate() {
                    let idx = i + 1;
                    let _ = out.send((
                        "content_block_start",
                        json!({ "type": "content_block_start", "index": idx, "content_block": { "type": "tool_use", "id": tc.id, "name": tc.function.name, "input": {} } }),
                    ));
                    let _ = out.send((
                        "content_block_delta",
                        json!({ "type": "content_block_delta", "index": idx, "delta": { "type": "input_json_delta", "partial_json": tc.function.arguments } }),
                    ));
                    let _ = out.send(("content_block_stop", json!({ "type": "content_block_stop", "index": idx })));
                }
                let stop_reason = if summary.tool_calls.is_empty() { "end_turn" } else { "tool_use" };
                let _ = out.send((
                    "message_delta",
                    json!({ "type": "message_delta", "delta": { "stop_reason": stop_reason, "stop_sequence": Value::Null }, "usage": { "output_tokens": summary.tokens } }),
                ));
                let _ = out.send(("message_stop", json!({ "type": "message_stop" })));
                break 'drain;
            }
            GatewayEvent::Error(m) => {
                metrics.record_error();
                let _ = out.send(("error", json!({ "type": "error", "error": { "type": "api_error", "message": m } })));
                break 'drain;
            }
        }
        // Await the next frame; emit a `ping` on a mid-content stall (biased → a ready frame wins,
        // so no ping once tokens are flowing). Only reached after a `Delta` (Done/Error broke).
        ev = loop {
            tokio::select! {
                biased;
                f = rx.recv() => match f {
                    Some(e) => break e,
                    None => break 'drain, // channel closed without a terminal frame (defensive)
                },
                _ = tokio::time::sleep(PING_INTERVAL) => {
                    let _ = out.send(("ping", json!({ "type": "ping" })));
                }
            }
        };
    }
}

/// `GET /v1/models` — the models this gateway currently knows a provider for (PEX-learned
/// / discovered). Dynamic in a decentralized swarm: empty until gossip arrives, then grows
/// as providers announce. The blocking swarm query runs on a plain OS thread (same reason
/// as `complete` — `blocking_send` needs a non-tokio context).
async fn list_models(State(state): State<AppState>) -> Response {
    state.metrics.incr_models();
    // The sticky browse set (live models, smoothed over transient churn). Routing stays on the live
    // set elsewhere — a listed-but-currently-unserved model resolves on-demand or falls back.
    let models = browse_models(&state).await;
    let mut data: Vec<Value> = Vec::new();
    // Advertise the `openhydra/auto` meta-model first, but only when at least one concrete
    // model can currently back it (else clients would pick a model that can't route).
    if !models.is_empty() {
        data.push(json!({ "id": AUTO_MODEL_ID, "object": "model", "created": 0, "owned_by": "openhydra" }));
    }
    data.extend(
        models
            .iter()
            .map(|m| json!({ "id": m, "object": "model", "created": 0, "owned_by": "openhydra" })),
    );
    Json(json!({ "object": "list", "data": data })).into_response()
}

/// Constant-time byte comparison — avoids leaking the API key length/prefix via timing.
fn constant_time_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Auth middleware for `/v1/*`: when an API key is configured, require a matching
/// `Authorization: Bearer <key>`; otherwise pass through (open, for loopback).
async fn require_api_key(State(state): State<AppState>, req: Request, next: Next) -> Response {
    if let Some(key) = &state.api_key {
        // Accept OpenAI-style `Authorization: Bearer <key>` and Anthropic-style `x-api-key: <key>`
        // (so `/v1/messages` clients like Claude Code authenticate with their native header).
        let presented = req
            .headers()
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .or_else(|| req.headers().get("x-api-key").and_then(|v| v.to_str().ok()));
        let ok = presented.is_some_and(|t| constant_time_eq(t, key));
        if !ok {
            return openai_error(
                StatusCode::UNAUTHORIZED,
                "missing or invalid API key",
                "invalid_request_error",
            );
        }
    }
    next.run(req).await
}

/// The client identity to rate-limit by — most-unforgeable first: the operator-issued API
/// key (validated by `require_api_key` upstream) → the socket peer IP → a trusted-proxy
/// `X-Forwarded-For`. The raw XFF header is client-spoofable, so it is honored only when
/// `trusted_proxy` is set; otherwise an attacker could forge a fresh identity per request.
fn rate_limit_identity(state: &AppState, peer: SocketAddr, req: &Request) -> String {
    if state.api_key.is_some() {
        if let Some(token) = req
            .headers()
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
        {
            return format!("key:{token}");
        }
    }
    if state.trusted_proxy {
        // G3c: take the RIGHTMOST X-Forwarded-For entry, not the leftmost. Behind an appending
        // proxy (nginx `$proxy_add_x_forwarded_for`) the trusted hop appends the peer IP it saw
        // to the right; the leftmost tokens are whatever the client sent and are fully
        // spoofable, so keying off them lets a caller forge a fresh identity per request and
        // bypass per-IP limiting. The rightmost entry is the one our trusted proxy wrote.
        if let Some(ip) = req
            .headers()
            .get("x-forwarded-for")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.split(',').next_back())
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            return format!("ip:{ip}");
        }
    }
    format!("ip:{}", peer.ip())
}

/// Ingress DoS rate-limit middleware: per-identity concurrency cap + token bucket. Runs after
/// `require_api_key` (so it can key off a validated API key); a shed request returns `429`
/// with `Retry-After`. The `_guard` is held across the request and frees the in-flight slot
/// on drop (including cancellation). A no-op when the limiter is inactive.
async fn rate_limit(
    State(state): State<AppState>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    req: Request,
    next: Next,
) -> Response {
    if !state.rate_limiter.is_active() {
        return next.run(req).await;
    }
    let identity = rate_limit_identity(&state, peer, &req);
    match state.rate_limiter.try_acquire(&identity, unix_now_ms()) {
        Ok(guard) => {
            // G3: the in-flight slot must live as long as the *generation*, not just until the
            // handler returns. For `stream:true` the handler returns the SSE `Response`
            // immediately while the worker keeps generating, so dropping the guard here let
            // streaming completions escape `max_inflight` entirely. Move the guard into the
            // response body so it is released only when the body is fully sent (stream drained)
            // or the client disconnects (body dropped) — correct for both streaming and buffered.
            let resp = next.run(req).await;
            let (parts, body) = resp.into_parts();
            let guarded = body.into_data_stream().map(move |chunk| {
                let _hold = &guard; // keep the slot occupied for the body's whole lifetime
                chunk
            });
            Response::from_parts(parts, Body::from_stream(guarded))
        }
        Err(_) => {
            state.metrics.record_rate_limited();
            let mut resp = openai_error(
                StatusCode::TOO_MANY_REQUESTS,
                "rate limit exceeded",
                "rate_limit_exceeded",
            );
            resp.headers_mut()
                .insert(axum::http::header::RETRY_AFTER, axum::http::HeaderValue::from_static("1"));
            resp
        }
    }
}

/// `POST /v1/embeddings` request (the OpenAI shape we honour). `input` is one string or many.
#[derive(Debug, Deserialize)]
struct EmbeddingRequest {
    #[serde(default)]
    model: String,
    #[serde(default)]
    input: EmbedInput,
}
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EmbedInput {
    One(String),
    Many(Vec<String>),
}
impl Default for EmbedInput {
    fn default() -> Self {
        EmbedInput::Many(Vec::new())
    }
}
impl EmbedInput {
    fn into_vec(self) -> Vec<String> {
        match self {
            EmbedInput::One(s) => vec![s],
            EmbedInput::Many(v) => v,
        }
    }
}

/// `POST /v1/embeddings` (#34): a configured BYOK embeddings model is served by the hosted
/// OpenAI-compatible backend (non-streaming). Models that aren't configured are rejected —
/// this gateway has no swarm embeddings path. Key: caller `X-Provider-Api-Key`, else operator.
async fn embeddings(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    body: Result<Json<EmbeddingRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(req) = match body {
        Ok(j) => j,
        Err(rej) => {
            return openai_error(StatusCode::BAD_REQUEST, &rej.body_text(), "invalid_request_error")
        }
    };
    if req.model.trim().is_empty() {
        return openai_error(StatusCode::BAD_REQUEST, "missing required field: model", "invalid_request_error");
    }
    if !state.embeddings.handles(&req.model) {
        return openai_error(
            StatusCode::BAD_REQUEST,
            &format!("'{}' is not a configured embeddings model", req.model),
            "invalid_request_error",
        );
    }
    let caller_key = headers.get("x-provider-api-key").and_then(|v| v.to_str().ok());
    let key = match state.embeddings.resolve_key(caller_key) {
        Some(k) => k,
        None => {
            return openai_error(
                StatusCode::BAD_REQUEST,
                &format!("no API key available for embeddings model '{}'", req.model),
                "invalid_request_error",
            )
        }
    };
    let inputs = req.input.into_vec();
    if inputs.is_empty() {
        return openai_error(StatusCode::BAD_REQUEST, "missing or empty field: input", "invalid_request_error");
    }
    // AUP floor (H): the chat path enforces the operator's acceptable-use limits, but
    // /v1/embeddings skipped it — an operator's size/deny caps couldn't bound embeddings.
    // Reuse the same policy by treating each input string as a message (bounds input count
    // via max-messages, total size via max-prompt-chars, and applies deny-substrings).
    let aup_inputs: Vec<ChatMessage> = inputs
        .iter()
        .map(|t| ChatMessage { role: "user".to_string(), content: t.clone(), ..Default::default() })
        .collect();
    if let AupDecision::Deny(reason) = state.aup.evaluate(&aup_inputs, None) {
        return openai_error(StatusCode::BAD_REQUEST, &reason, "invalid_request_error");
    }
    let base_url = state.embeddings.base_url().to_string();
    let model = req.model.clone();
    let outcome = tokio::task::spawn_blocking(move || {
        crate::live_openai_embeddings(&base_url, &key).and_then(|a| a.embed(&model, &inputs))
    })
    .await;
    match outcome {
        Ok(Ok(resp)) => {
            let data: Vec<Value> = resp
                .vectors
                .iter()
                .enumerate()
                .map(|(i, v)| json!({ "object": "embedding", "index": i, "embedding": v }))
                .collect();
            Json(json!({
                "object": "list",
                "data": data,
                "model": req.model,
                "usage": { "prompt_tokens": resp.prompt_tokens, "total_tokens": resp.prompt_tokens },
            }))
            .into_response()
        }
        Ok(Err(e)) => {
            let (status, etype) = classify_error(&e.to_string());
            openai_error(status, &e.to_string(), etype)
        }
        Err(_) => openai_error(StatusCode::INTERNAL_SERVER_ERROR, "embeddings task failed", "internal_error"),
    }
}

async fn health() -> Response {
    Json(json!({ "status": "ok" })).into_response()
}

/// `GET /metrics` — Prometheus text exposition (#33). Open (no API key), like `/health`, so
/// a scraper needn't hold the gateway key; it exposes only aggregate counters/latencies, no
/// request content. Bind to loopback (the default) if that aggregate is sensitive.
async fn metrics_endpoint(State(state): State<AppState>) -> Response {
    (
        [(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        state.metrics.render_prometheus(),
    )
        .into_response()
}

/// The gateway router over a started swarm node. `api_key` (when `Some`) gates the `/v1/*`
/// routes behind `Authorization: Bearer <key>`; `/health` is always open.
#[allow(clippy::too_many_arguments)]
pub fn router(
    net: NetworkHandle,
    economy: Arc<EconomyStats>,
    stats: Arc<crate::status::TransferStats>,
    api_key: Option<String>,
    store: Option<Store>,
    aup: AupPolicy,
    rate_limit_cfg: RateLimitConfig,
    trusted_proxy: bool,
    byok: ByokConfig,
    embeddings_cfg: EmbeddingConfig,
    self_provider: Option<String>,
    cards: Option<Arc<crate::cards::CardStore>>,
    creds: Option<Arc<crate::swarms::CredentialStore>>,
) -> Router {
    let node = match store {
        Some(s) => {
            // Rehydrate the durable Ledger (recent `used` rows + lifetime totals) so the desktop
            // Ledger view survives a restart instead of resetting. Read before `s` moves into the
            // node. 250 = the status ring cap.
            if let (Ok(rows), Ok((served, used, n))) = (s.recent_ledger_rows(250), s.ledger_totals()) {
                stats.rehydrate_ledger(&rows, served, used, n);
            }
            ConsumerNode::with_store(net, s) // M2.2(a): persisted reputation
        }
        None => ConsumerNode::new(net),
    };
    // #7/#5: completions record per-model consumed tokens + `used` ledger rows into the shared
    // transfer counters the status endpoint serves. #7: mark our own provider so a self-serve
    // settles no receipt and moves no credit.
    let node = node
        .with_stats(stats)
        .with_self_provider(self_provider)
        .with_cards(cards)
        .with_credentials(creds);
    let node = Arc::new(node);
    // Publish the consumer's give-to-get view (earned reputation of providers we've used +
    // give-side credit balances) to the status endpoint every 2s. A plain std thread: the
    // reads are sync mutex snapshots, so it never touches the tokio runtime.
    {
        let node = Arc::clone(&node);
        std::thread::spawn(move || loop {
            let (reputation, credit) = node.economy_snapshot();
            economy.publish(EconomyView::new("consumer", reputation, credit));
            std::thread::sleep(Duration::from_secs(2));
        });
    }
    let state = AppState {
        node,
        api_key: api_key.map(Arc::new),
        metrics: Arc::new(Metrics::new()),
        aup: Arc::new(aup),
        rate_limiter: Arc::new(RateLimiter::new(rate_limit_cfg)),
        gen_limiter: Arc::new(Semaphore::new(MAX_CONCURRENT_GENERATIONS)),
        trusted_proxy,
        byok: Arc::new(byok),
        embeddings: Arc::new(embeddings_cfg),
        sessions: Arc::new(SessionModels::new(slash::DEFAULT_SESSION_TTL)),
        model_cache: Arc::new(ModelStickyCache::new(MODEL_STICKY_TTL)),
    };
    // The `/v1/*` routes are auth-gated then rate-limited; `/health` and `/metrics` stay open
    // for liveness probes and Prometheus scraping. `route_layer`s run outermost-last, so
    // adding `rate_limit` first and `require_api_key` last makes auth run *before* the
    // limiter — the limiter can then key off the validated API key.
    let v1 = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/messages", post(messages))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/models", get(list_models))
        .route_layer(middleware::from_fn_with_state(state.clone(), rate_limit))
        .route_layer(middleware::from_fn_with_state(state.clone(), require_api_key));
    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics_endpoint))
        .merge(v1)
        .with_state(state)
}

/// Run the gateway, blocking. Builds its own multi-thread tokio runtime and serves until
/// the process exits. `bind` is e.g. `"127.0.0.1:8080"`; `api_key` optionally protects
/// `/v1/*`.
#[allow(clippy::too_many_arguments)]
pub fn serve_http(
    net: NetworkHandle,
    economy: Arc<EconomyStats>,
    stats: Arc<crate::status::TransferStats>,
    bind: &str,
    api_key: Option<String>,
    store: Option<Store>,
    aup: AupPolicy,
    rate_limit_cfg: RateLimitConfig,
    trusted_proxy: bool,
    byok: ByokConfig,
    embeddings_cfg: EmbeddingConfig,
    self_provider: Option<String>,
    cards: Option<Arc<crate::cards::CardStore>>,
    creds: Option<Arc<crate::swarms::CredentialStore>>,
) -> std::io::Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(bind).await?;
        // `into_make_service_with_connect_info` surfaces the peer `SocketAddr` to the
        // rate-limit middleware (the unspoofable per-IP key).
        let app = router(net, economy, stats, api_key, store, aup, rate_limit_cfg, trusted_proxy, byok, embeddings_cfg, self_provider, cards, creds)
            .into_make_service_with_connect_info::<SocketAddr>();
        axum::serve(listener, app).await
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::EngineMetrics;

    fn summary(tokens: u64, prompt: u64) -> ServeSummary {
        ServeSummary {
            tokens,
            ok: true,
            metrics: crate::serve::ServeMetrics {
                engine: EngineMetrics { prompt_eval_count: prompt, eval_count: tokens, ..Default::default() },
                provider_serve_ns: 500_000_000,
            },
            proxy_roundtrip_ns: 600_000_000,
            discover_ns: 1_000_000,
            tool_calls: Vec::new(),
        }
    }

    // ── `openhydra/auto` meta-model ──

    #[test]
    fn is_auto_model_matches_aliases() {
        assert!(is_auto_model("auto"));
        assert!(is_auto_model("openhydra/auto"));
        assert!(is_auto_model("  OpenHydra/Auto  "));
        assert!(!is_auto_model("qwen2.5:7b"));
        assert!(!is_auto_model("openhydra/qwen2.5:7b"));
    }

    #[test]
    fn pick_auto_prefers_available_override_else_first() {
        let known = vec!["qwen2.5:7b".to_string(), "llama3.2:1b".to_string()];
        // operator override wins when it's actually served
        assert_eq!(pick_auto_model(&known, Some("qwen2.5:7b")).as_deref(), Some("qwen2.5:7b"));
        // override that isn't served → fall back to the deterministic (alphabetical) pick
        assert_eq!(pick_auto_model(&known, Some("not-served")).as_deref(), Some("llama3.2:1b"));
        // no override → deterministic pick
        assert_eq!(pick_auto_model(&known, None).as_deref(), Some("llama3.2:1b"));
        // nothing served → nothing to pick
        assert_eq!(pick_auto_model(&[], Some("x")), None);
    }

    // ── Anthropic `/v1/messages` translation ──

    #[test]
    fn ant_string_content_and_system_convert() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "m", "max_tokens": 16,
            "system": "be brief",
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .unwrap();
        let msgs = to_chat_messages(req.system.as_ref(), &req.messages);
        assert_eq!(msgs.len(), 2);
        assert_eq!((msgs[0].role.as_str(), msgs[0].content.as_str()), ("system", "be brief"));
        assert_eq!((msgs[1].role.as_str(), msgs[1].content.as_str()), ("user", "hi"));
    }

    #[test]
    fn ant_tool_roundtrip_converts_to_openai_shapes() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "m", "max_tokens": 16,
            "messages": [
                { "role": "user", "content": [{ "type": "text", "text": "weather?" }] },
                { "role": "assistant", "content": [
                    { "type": "text", "text": "checking" },
                    { "type": "tool_use", "id": "tu_1", "name": "get_weather", "input": { "city": "SF" } }
                ] },
                { "role": "user", "content": [
                    { "type": "tool_result", "tool_use_id": "tu_1", "content": "72F" }
                ] }
            ]
        }))
        .unwrap();
        let msgs = to_chat_messages(req.system.as_ref(), &req.messages);
        assert_eq!(msgs[0].role, "user");
        assert_eq!(msgs[1].role, "assistant");
        let tc = msgs[1].tool_calls.as_ref().expect("assistant tool_calls");
        assert_eq!((tc[0].id.as_str(), tc[0].function.name.as_str()), ("tu_1", "get_weather"));
        assert!(tc[0].function.arguments.contains("SF"));
        assert_eq!(msgs[2].role, "tool");
        assert_eq!(msgs[2].tool_call_id.as_deref(), Some("tu_1"));
        assert_eq!(msgs[2].content, "72F");
    }

    #[test]
    fn anthropic_message_object_text_then_tool_use() {
        let mut s = summary(5, 3);
        let obj = anthropic_message_object("msg_1", "m", "hello", &s);
        assert_eq!(obj["type"], "message");
        assert_eq!(obj["content"][0]["type"], "text");
        assert_eq!(obj["content"][0]["text"], "hello");
        assert_eq!(obj["stop_reason"], "end_turn");
        assert_eq!(obj["usage"]["input_tokens"], 3);
        assert_eq!(obj["usage"]["output_tokens"], 5);

        s.tool_calls = vec![ToolCall {
            id: "tu_9".into(),
            kind: "function".into(),
            function: ToolCallFunction { name: "f".into(), arguments: "{\"x\":1}".into() },
        }];
        let obj = anthropic_message_object("msg_2", "m", "", &s);
        assert_eq!(obj["stop_reason"], "tool_use");
        assert_eq!(obj["content"][0]["type"], "tool_use");
        assert_eq!(obj["content"][0]["id"], "tu_9");
        assert_eq!(obj["content"][0]["input"]["x"], 1);
    }

    #[test]
    fn ant_tools_map_to_openai_functions() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "m", "messages": [{ "role": "user", "content": "hi" }],
            "tools": [{ "name": "f", "description": "d", "input_schema": { "type": "object" } }]
        }))
        .unwrap();
        let t = to_openai_tools(&req.tools);
        assert_eq!(t[0]["type"], "function");
        assert_eq!(t[0]["function"]["name"], "f");
        assert_eq!(t[0]["function"]["parameters"]["type"], "object");
    }

    // ── Anthropic streaming translation (the immediate-error branch, deterministically) ──

    /// Drive `anthropic_stream_worker` with a fixed event sequence and return the ordered SSE event
    /// names it emits.
    async fn stream_names(events: Vec<GatewayEvent>) -> Vec<&'static str> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        for e in events {
            tx.send(e).unwrap();
        }
        drop(tx); // close so the worker's drain loop ends
        let (otx, mut orx) = tokio::sync::mpsc::unbounded_channel();
        anthropic_stream_worker(
            "msg_x".into(),
            "m".into(),
            rx,
            std::time::Instant::now(),
            Arc::new(Metrics::new()),
            otx,
        )
        .await;
        let mut names = Vec::new();
        while let Ok((name, _)) = orx.try_recv() {
            names.push(name);
        }
        names
    }

    #[tokio::test]
    async fn stream_immediate_error_has_no_message_envelope() {
        // An error as the FIRST frame → a single `error` frame, never a `message_start`.
        let names = stream_names(vec![GatewayEvent::Error("no provider".into())]).await;
        assert_eq!(names, vec!["error"]);
    }

    #[tokio::test]
    async fn stream_happy_path_is_a_full_message_envelope() {
        let names = stream_names(vec![
            GatewayEvent::Delta("hi".into()),
            GatewayEvent::Done(Box::new(summary(1, 2))),
        ])
        .await;
        assert_eq!(
            names,
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
    }

    #[tokio::test]
    async fn stream_done_with_tool_call_emits_tool_use_block() {
        let mut s = summary(3, 2);
        s.tool_calls = vec![ToolCall {
            id: "tu".into(),
            kind: "function".into(),
            function: ToolCallFunction { name: "f".into(), arguments: "{}".into() },
        }];
        let names = stream_names(vec![GatewayEvent::Done(Box::new(s))]).await;
        assert_eq!(
            names,
            vec![
                "message_start",
                "content_block_start",  // text block (index 0)
                "content_block_stop",
                "content_block_start",  // tool_use block (index 1)
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
    }

    #[tokio::test(start_paused = true)]
    async fn stream_pings_during_a_mid_content_stall_never_before_message_start() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let (otx, mut orx) = tokio::sync::mpsc::unbounded_channel();
        let worker = tokio::spawn(anthropic_stream_worker(
            "id".into(),
            "m".into(),
            rx,
            std::time::Instant::now(),
            Arc::new(Metrics::new()),
            otx,
        ));
        // First token opens the message envelope.
        tx.send(GatewayEvent::Delta("hi".into())).unwrap();
        tokio::task::yield_now().await;
        // Then the model stalls: advance past two ping intervals → native pings fire, AFTER
        // message_start (where the Anthropic protocol allows them).
        tokio::time::advance(std::time::Duration::from_secs(25)).await;
        tokio::task::yield_now().await;
        tx.send(GatewayEvent::Done(Box::new(summary(1, 1)))).unwrap();
        drop(tx);
        worker.await.unwrap();
        let mut names = Vec::new();
        while let Ok((n, _)) = orx.try_recv() {
            names.push(n);
        }
        // The stream must OPEN with message_start — never a ping before it (that breaks the SDK).
        assert_eq!(names.first(), Some(&"message_start"), "{names:?}");
        let start = names.iter().position(|&n| n == "message_start").unwrap();
        let ping = names.iter().position(|&n| n == "ping").expect("expected mid-content pings");
        assert!(ping > start, "ping must come after message_start: {names:?}");
        assert!(names.contains(&"message_stop"));
    }

    #[test]
    fn request_defaults_to_non_streaming() {
        let req: ChatRequest =
            serde_json::from_str(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#).unwrap();
        assert!(!req.stream, "OpenAI default is stream:false");
    }

    #[test]
    fn effective_think_resolves_and_prioritises_native_over_kwargs() {
        // No thinking control ⇒ leave the engine default.
        let bare: ChatRequest =
            serde_json::from_str(r#"{"model":"m","messages":[]}"#).unwrap();
        assert_eq!(bare.effective_think(), None);
        // Ollama-native top-level `think` is honoured.
        let native: ChatRequest =
            serde_json::from_str(r#"{"model":"m","messages":[],"think":false}"#).unwrap();
        assert_eq!(native.effective_think(), Some(false));
        // vLLM/llama.cpp `chat_template_kwargs.enable_thinking` is honoured when `think` is absent.
        let kwargs: ChatRequest = serde_json::from_str(
            r#"{"model":"m","messages":[],"chat_template_kwargs":{"enable_thinking":false}}"#,
        )
        .unwrap();
        assert_eq!(kwargs.effective_think(), Some(false));
        // Native `think` wins over `chat_template_kwargs` when both are present.
        let both: ChatRequest = serde_json::from_str(
            r#"{"model":"m","messages":[],"think":true,"chat_template_kwargs":{"enable_thinking":false}}"#,
        )
        .unwrap();
        assert_eq!(both.effective_think(), Some(true));
    }

    #[test]
    fn validation_rejects_empty_model_and_messages() {
        let r = ChatRequest { model: "".into(), messages: vec![ChatMessage { role: "user".into(), content: "x".into(), ..Default::default() }], max_tokens: None, temperature: None, tools: Vec::new(), stream: false, stream_options: None, think: None, chat_template_kwargs: None };
        assert!(validate(&r).is_some());
        let r = ChatRequest { model: "m".into(), messages: vec![], max_tokens: None, temperature: None, tools: Vec::new(), stream: false, stream_options: None, think: None, chat_template_kwargs: None };
        assert!(validate(&r).is_some());
        let r = ChatRequest { model: "m".into(), messages: vec![ChatMessage { role: "user".into(), content: "x".into(), ..Default::default() }], max_tokens: None, temperature: None, tools: Vec::new(), stream: false, stream_options: None, think: None, chat_template_kwargs: None };
        assert!(validate(&r).is_none());
    }

    #[test]
    fn error_classification_maps_to_status() {
        assert_eq!(classify_error("engine http error: no provider for model 'm'").0, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(classify_error("proxy_forward: proxy_forward timed out after 45s").0, StatusCode::GATEWAY_TIMEOUT);
        assert_eq!(classify_error("engine http error: connection refused").0, StatusCode::BAD_GATEWAY);
    }

    #[test]
    fn completion_object_is_openai_shaped() {
        let s = summary(27, 46);
        let v = completion_object("chatcmpl-1", "unsloth/Llama-3.2-1B-Instruct", 1_700_000_000, "hello world", &s, std::time::Duration::from_millis(500), std::time::Duration::from_millis(80));
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["message"]["role"], "assistant");
        assert_eq!(v["choices"][0]["message"]["content"], "hello world");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        assert_eq!(v["usage"]["prompt_tokens"], 46);
        assert_eq!(v["usage"]["completion_tokens"], 27);
        assert_eq!(v["usage"]["total_tokens"], 73);
        assert!(v["created"].is_u64());
        assert!(v["openhydra"]["tokens"] == 27);
    }

    #[test]
    fn completion_object_emits_tool_calls_when_present() {
        use crate::adapter::{ToolCall, ToolCallFunction};
        let mut s = summary(0, 12);
        s.tool_calls = vec![ToolCall {
            id: "call_1".into(),
            kind: "function".into(),
            function: ToolCallFunction { name: "get_time".into(), arguments: "{}".into() },
        }];
        let v = completion_object("chatcmpl-1", "m", 1, "", &s, std::time::Duration::from_millis(10), std::time::Duration::from_millis(10));
        // The OpenAI shape a coding agent expects: content null, tool_calls set, finish_reason.
        assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(v["choices"][0]["message"]["content"], Value::Null);
        assert_eq!(v["choices"][0]["message"]["tool_calls"][0]["type"], "function");
        assert_eq!(v["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "get_time");
    }

    #[test]
    fn chat_request_parses_tools_and_ignores_unknown_fields() {
        // A coding agent sends `tools` + `tool_choice`; we honour `tools`, and unknown fields
        // (tool_choice, top_p, …) are silently ignored rather than 400-ing the request.
        let req: ChatRequest = serde_json::from_str(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f"}}],"tool_choice":"auto","top_p":0.9}"#,
        )
        .unwrap();
        assert_eq!(req.tools.len(), 1);
        assert_eq!(req.tools[0]["function"]["name"], "f");
    }

    #[test]
    fn stream_chunk_carries_created_and_delta() {
        let v: Value = serde_json::from_str(&stream_chunk("id1", "m", 1_700_000_000, json!({"content":"Hi"}), None)).unwrap();
        assert_eq!(v["object"], "chat.completion.chunk");
        assert_eq!(v["created"], 1_700_000_000u64);
        assert_eq!(v["choices"][0]["delta"]["content"], "Hi");
        assert!(v["choices"][0]["finish_reason"].is_null());
        let stop: Value = serde_json::from_str(&stream_chunk("id1", "m", 1, json!({}), Some("stop"))).unwrap();
        assert_eq!(stop["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn usage_value_sums_tokens() {
        let v = usage_value(&summary(10, 5));
        assert_eq!(v["total_tokens"], 15);
    }

    #[test]
    fn constant_time_eq_matches_only_identical() {
        assert!(constant_time_eq("sk-secret-123", "sk-secret-123"));
        assert!(!constant_time_eq("sk-secret-123", "sk-secret-124"));
        assert!(!constant_time_eq("sk-secret", "sk-secret-123")); // length mismatch
        assert!(!constant_time_eq("", "x"));
    }

    // ── Slash-command wiring (extraction / session key / reply shape) ──

    #[test]
    fn latest_user_text_only_a_plain_user_turn() {
        // The last message is a plain user turn → its text.
        let msgs = vec![ChatMessage::new("system", "s"), ChatMessage::new("user", "/models")];
        assert_eq!(latest_user_text(&msgs), Some("/models"));
        // An assistant turn last → not a command site.
        let msgs = vec![ChatMessage::new("user", "hi"), ChatMessage::new("assistant", "/models")];
        assert_eq!(latest_user_text(&msgs), None);
        // A `role:"tool"` result turn (has tool_call_id) → None.
        let mut tool = ChatMessage::new("tool", "72F");
        tool.tool_call_id = Some("t1".into());
        assert_eq!(latest_user_text(std::slice::from_ref(&tool)), None);
        // Blank content → None.
        assert_eq!(latest_user_text(&[ChatMessage::new("user", "   ")]), None);
        assert_eq!(latest_user_text(&[]), None);
    }

    #[test]
    fn latest_ant_user_text_flattens_text_blocks_only() {
        let bare: MessagesRequest = serde_json::from_value(json!({
            "model": "m", "messages": [{ "role": "user", "content": "/model qwen" }]
        }))
        .unwrap();
        assert_eq!(latest_ant_user_text(&bare.messages).as_deref(), Some("/model qwen"));

        let blocks: MessagesRequest = serde_json::from_value(json!({
            "model": "m", "messages": [{ "role": "user", "content": [{ "type": "text", "text": "/models" }] }]
        }))
        .unwrap();
        assert_eq!(latest_ant_user_text(&blocks.messages).as_deref(), Some("/models"));

        // A tool_result turn is never a command.
        let tr: MessagesRequest = serde_json::from_value(json!({
            "model": "m", "messages": [{ "role": "user", "content": [{ "type": "tool_result", "tool_use_id": "t", "content": "x" }] }]
        }))
        .unwrap();
        assert_eq!(latest_ant_user_text(&tr.messages), None);
    }

    #[test]
    fn session_key_prefers_api_key_else_local() {
        use axum::http::HeaderMap;
        let mut h = HeaderMap::new();
        h.insert(axum::http::header::AUTHORIZATION, "Bearer abc".parse().unwrap());
        assert_eq!(session_key(&h), "key:abc");
        let mut h = HeaderMap::new();
        h.insert("x-api-key", "xyz".parse().unwrap());
        assert_eq!(session_key(&h), "key:xyz");
        assert_eq!(session_key(&HeaderMap::new()), "local");
    }

    #[tokio::test]
    async fn command_response_openai_buffered_shape() {
        let resp = command_response_openai("id1", "openhydra/auto", 123, "hello".into(), false);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["message"]["content"], "hello");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        assert_eq!(v["usage"]["total_tokens"], 0);
    }

    #[test]
    fn model_sticky_cache_smooths_churn() {
        let c = ModelStickyCache::new(Duration::from_secs(300));
        // Two live models.
        assert_eq!(c.merge(vec!["m1".into(), "m2".into()]), vec!["m1".to_string(), "m2".to_string()]);
        // m2 momentarily gone from the live set → still listed (within TTL).
        assert_eq!(c.merge(vec!["m1".into()]), vec!["m1".to_string(), "m2".to_string()]);
        // A new model appears and joins.
        assert_eq!(
            c.merge(vec!["m1".into(), "m3".into()]),
            vec!["m1".to_string(), "m2".to_string(), "m3".to_string()]
        );
        // ttl = 0 → a model absent from the next live set drops immediately (no stickiness).
        let z = ModelStickyCache::new(Duration::ZERO);
        z.merge(vec!["x".into(), "y".into()]);
        std::thread::sleep(Duration::from_millis(2));
        assert_eq!(z.merge(vec!["x".into()]), vec!["x".to_string()]);
    }

    #[tokio::test]
    async fn command_response_anthropic_buffered_shape() {
        let resp = command_response_anthropic("msg1", "claude-x", "hi".into(), false);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["type"], "message");
        assert_eq!(v["content"][0]["text"], "hi");
        assert_eq!(v["stop_reason"], "end_turn");
        assert_eq!(v["usage"]["output_tokens"], 0);
    }
}
