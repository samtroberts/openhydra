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
//! `blocking_send` is valid) and pipes the deltas back through a bounded channel (a client that reads
//! slowly holds the worker back rather than the gateway buffering whatever the provider sends).

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
use tokio_stream::wrappers::ReceiverStream;
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
    /// OpenAI's current name for the cap (`max_tokens` is deprecated there); see
    /// [`ChatRequest::effective_max_tokens`].
    #[serde(default)]
    max_completion_tokens: Option<u32>,
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
    /// The completion-token cap: `max_completion_tokens` or `max_tokens`, the smaller if both are
    /// set. It bounds both the generation and the tokens the receipt bills.
    fn effective_max_tokens(&self) -> Option<u32> {
        match (self.max_tokens, self.max_completion_tokens) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }

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
/// Events buffered between a generation worker and the response that sends them. Bounded: the
/// worker blocks (and stops reading from the provider) while the client is behind.
const STREAM_BACKLOG: usize = 256;
type EventRx = tokio::sync::mpsc::Receiver<GatewayEvent>;

enum GatewayEvent {
    Delta(String),
    Done(Box<ServeSummary>),
    /// The error's text, and its HTTP status and OpenAI error `type` ([`error_class`]).
    Error(String, ErrorClass),
}

/// An error's HTTP status and OpenAI error `type`.
type ErrorClass = (StatusCode, &'static str);

/// The most completion text a non-streaming response is assembled from (round 8 G8-3); past it the
/// request fails rather than the gateway holding whatever an upstream sends. Room for a media
/// result's data URL (round 9 B9-1).
const MAX_BUFFERED_COMPLETION_BYTES: usize = 128 * 1024 * 1024;

fn too_large_to_buffer() -> (String, ErrorClass) {
    ("the completion is too large to return in one response; use streaming".into(), (StatusCode::BAD_GATEWAY, "upstream_error"))
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

/// The HTTP status and OpenAI error `type` of a completion error, by its kind (review round 4 G4-8):
/// a provider's own error is always an upstream error, whatever its text says (it can't pass for a
/// routing refusal or a timeout). Only our own routing and network messages are read, and errors of
/// the older kinds fall back to [`classify_error`].
fn error_class(e: &crate::adapter::AdapterError) -> ErrorClass {
    use crate::adapter::AdapterError;
    match e {
        AdapterError::Route(m) if m.contains("no provider") => (StatusCode::SERVICE_UNAVAILABLE, "no_provider"),
        AdapterError::Route(_) | AdapterError::Provider(_) | AdapterError::Cancelled(_) => {
            (StatusCode::BAD_GATEWAY, "upstream_error")
        }
        AdapterError::Network(m) if m.to_ascii_lowercase().contains("timed out") || m.to_ascii_lowercase().contains("timeout") => {
            (StatusCode::GATEWAY_TIMEOUT, "upstream_timeout")
        }
        AdapterError::Overloaded(_) => (StatusCode::SERVICE_UNAVAILABLE, "upstream_overloaded"),
        AdapterError::Network(_) => (StatusCode::BAD_GATEWAY, "upstream_error"),
        other => classify_error(&other.to_string()),
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
    if req.effective_max_tokens() == Some(0) {
        return Some(openai_error(
            StatusCode::BAD_REQUEST,
            "max_tokens must be at least 1",
            "invalid_request_error",
        ));
    }
    None
}

// ── `openhydra/auto` meta-model ──────────────────────────────────────────────
//
// Connector snippets advertise `model: openhydra/auto`, but the router resolves a *literal*
// model id — so `auto` must be turned into a concrete model that actually has a provider
// before dispatch. Any peer can announce any model name, so the choice is made from sources this
// node has reason to trust: see [`pick_auto_model`].

/// The canonical meta-model id the connectors advertise.
const AUTO_MODEL_ID: &str = "openhydra/auto";

/// Is `model` the auto meta-model (`auto` or `openhydra/auto`, case/space-insensitive)?
fn is_auto_model(model: &str) -> bool {
    let m = model.trim();
    m.eq_ignore_ascii_case("auto") || m.eq_ignore_ascii_case(AUTO_MODEL_ID)
}

/// Pure selection for `openhydra/auto`, most trusted source first:
/// 1. the operator's preferred model (`OPENHYDRA_AUTO_MODEL`), if it is served;
/// 2. a model this machine's own provider serves;
/// 3. a model from an imported card;
/// 4. a model served by a provider this node has earned reputation with (above neutral);
/// 5. a model served by a provider this node has verified at A1 or above (its own audits and
///    checks; a hardware-attested provider needs no history).
///
/// When routing will hard-gate the request at an authenticity floor (`c.floor`), only providers at
/// or above it count, at every step: picking a model whose providers all fall below it would only
/// end in a refusal. Ties break alphabetically. `None` means nothing qualifies: the caller asks for
/// an explicit model. There is no "most providers" step: announcing a model from several peer ids
/// costs nothing, so a count is not trust.
fn pick_auto_model(c: &crate::consumer::AutoCandidates, preferred: Option<&str>) -> Option<String> {
    use openhydra_protocol::model_id::is_valid_model_id;
    use openhydra_protocol::trust::AuthenticityTier;
    use std::collections::BTreeSet;
    let tier_of = |k: &openhydra_network::types::KnownProvider| {
        c.tiers.get(&(k.libp2p_peer_id.clone(), k.model_id.clone())).copied().unwrap_or(AuthenticityTier::SelfReported)
    };
    let providers: Vec<_> = c
        .providers
        .iter()
        .filter(|p| is_valid_model_id(&p.model_id) && c.floor.is_none_or(|f| tier_of(p) >= f))
        .collect();
    let served = |m: &str| providers.iter().any(|k| k.model_id == m);
    // A card names its provider, which routing gates like any other: with a floor, a card counts
    // only when one of its model's providers clears it.
    let cards: BTreeSet<&str> = c
        .card_models
        .iter()
        .map(String::as_str)
        .filter(|m| is_valid_model_id(m) && (c.floor.is_none() || served(m)))
        .collect();
    if let Some(p) = preferred.map(str::trim).filter(|s| !s.is_empty()) {
        if cards.contains(p) || served(p) {
            return Some(p.to_string());
        }
    }
    if let Some(me) = c.self_provider.as_deref().filter(|s| !s.is_empty()) {
        if let Some(own) = providers.iter().filter(|k| k.libp2p_peer_id == me).map(|k| &k.model_id).min() {
            return Some(own.clone());
        }
    }
    if let Some(card) = cards.iter().next() {
        return Some((*card).to_string());
    }
    let neutral = openhydra_protocol::verify::NEUTRAL_REPUTATION;
    let mut best_rep: Option<(f64, &str)> = None;
    for k in &providers {
        let Some(&score) = c.earned.get(&(k.libp2p_peer_id.clone(), k.model_id.clone())) else {
            continue;
        };
        if score > neutral
            && best_rep.map_or(true, |(s, m)| score > s || (score == s && k.model_id.as_str() < m))
        {
            best_rep = Some((score, k.model_id.as_str()));
        }
    }
    if let Some((_, m)) = best_rep {
        return Some(m.to_string());
    }
    let mut best_tier: Option<(AuthenticityTier, &str)> = None;
    for k in &providers {
        let t = tier_of(k);
        if t >= AuthenticityTier::ReputationRedundant
            && best_tier.map_or(true, |(bt, m)| t > bt || (t == bt && k.model_id.as_str() < m))
        {
            best_tier = Some((t, k.model_id.as_str()));
        }
    }
    best_tier.map(|(_, m)| m.to_string())
}

/// The reply when `auto` has nothing it trusts: name the models it could use instead.
fn no_auto_model_message(known: &[String]) -> String {
    if known.is_empty() {
        return "no models available on the network yet — try again once a provider announces, or \
                request a specific model"
            .to_string();
    }
    let mut list: Vec<&str> = known.iter().map(String::as_str).take(10).collect();
    if known.len() > list.len() {
        list.push("…");
    }
    format!(
        "openhydra/auto found no model it can trust (none from your own machine, an imported card, or a \
         provider you've used or verified before). Choose a model explicitly — after one good answer \
         from a provider, auto can pick it; available: {}",
        list.join(", ")
    )
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
    min_authenticity: Option<openhydra_protocol::trust::AuthenticityTier>,
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
    // …else the auto default (see `pick_auto_model` for the order).
    let preferred = std::env::var("OPENHYDRA_AUTO_MODEL").ok();
    let node = state.node.clone();
    let candidates = tokio::task::spawn_blocking(move || node.auto_candidates(min_authenticity))
        .await
        .map_err(|_| "model resolution failed".to_string())??;
    pick_auto_model(&candidates, preferred.as_deref()).ok_or_else(|| no_auto_model_message(&known))
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
    // A rate over fewer than 2 tokens or under 1 ms says nothing (a 1-token reply would show as a
    // million tokens a second): reported as null.
    let rate = |n: u64, dur_ns: u64| (n >= 2 && dur_ns >= 1_000_000).then(|| r1(tps(n, dur_ns)));

    let wall_ns = wall.as_nanos() as u64;
    let provider_serve_ns = summary.metrics.provider_serve_ns;
    // On a streamed serve `proxy_roundtrip_ns` is the time to the provider's first acknowledgement
    // (sent before it generates) — already the network round trip, and shorter than the serve. On a
    // buffered serve it spans the whole serve, so the serve time comes off it.
    let net_rtt_ns = if summary.streamed {
        summary.proxy_roundtrip_ns
    } else {
        summary.proxy_roundtrip_ns.saturating_sub(provider_serve_ns)
    };
    // What's left of the wall time once discovery, the network and the provider's own serve are
    // taken off: on a buffered serve that is wall − discover − round trip; on a streamed one the
    // round trip is only the first acknowledgement, so the serve comes off too (ADVERSARIAL 10).
    let gateway_ns = wall_ns
        .saturating_sub(summary.discover_ns)
        .saturating_sub(net_rtt_ns)
        .saturating_sub(provider_serve_ns);
    let pipeline_tps = if wall_ns > 0 { summary.tokens as f64 / (wall_ns as f64 / 1e9) } else { 0.0 };

    json!({
        "tokens": summary.tokens,
        // The authenticity tier this gateway had established for the provider that served (D2).
        "provider_authenticity": summary.authenticity.map(|t| t.label()),
        "engine": {
            "native_tps": rate(e.eval_count, e.eval_duration_ns),
            "prefill_tps": rate(e.prompt_eval_count, e.prompt_eval_duration_ns),
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

/// The OpenAI `finish_reason` of a completion: `"tool_calls"` when the model requested tools,
/// `"length"` when the provider stopped it short (out of credit, or a stall it took for the
/// client leaving), else `"stop"`.
fn finish_reason(summary: &ServeSummary) -> &'static str {
    if !summary.tool_calls.is_empty() {
        "tool_calls"
    } else if summary.metrics.stopped {
        "length"
    } else {
        "stop"
    }
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
        (json!({ "role": "assistant", "content": content }), finish_reason(summary))
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
/// Most tool calls passed on from one reply, and their largest combined argument size.
const MAX_TOOL_CALLS: usize = 64;
const MAX_TOOL_CALL_BYTES: usize = 256 * 1024;

/// The names of the tools a request offered (OpenAI shape: `{"type":"function","function":{"name":…}}`).
fn offered_tool_names(tools: &[Value]) -> std::collections::HashSet<String> {
    tools
        .iter()
        .filter_map(|t| t.get("function").and_then(|f| f.get("name")).and_then(Value::as_str))
        .map(str::to_string)
        .collect()
}

/// Keep only the tool calls a client can safely act on. The provider is a stranger on the network,
/// and a coding tool runs whatever calls it is handed, so a call is kept only if it names a tool
/// the request offered and its arguments are a JSON object (empty means `{}`); at most
/// [`MAX_TOOL_CALLS`] calls and [`MAX_TOOL_CALL_BYTES`] of arguments. Returns the kept calls and
/// how many were dropped.
fn filter_tool_calls(
    calls: Vec<ToolCall>,
    offered: &std::collections::HashSet<String>,
) -> (Vec<ToolCall>, usize) {
    let total = calls.len();
    let mut kept = Vec::new();
    let mut bytes = 0usize;
    for mut c in calls {
        if kept.len() >= MAX_TOOL_CALLS || !offered.contains(&c.function.name) {
            continue;
        }
        if c.function.arguments.trim().is_empty() {
            c.function.arguments = "{}".into();
        }
        let is_object = serde_json::from_str::<Value>(&c.function.arguments).map(|v| v.is_object()).unwrap_or(false);
        if !is_object || bytes + c.function.arguments.len() > MAX_TOOL_CALL_BYTES {
            continue;
        }
        bytes += c.function.arguments.len();
        kept.push(c);
    }
    let dropped = total - kept.len();
    (kept, dropped)
}

fn spawn_worker(
    node: Arc<ConsumerNode>,
    model: String,
    messages: Vec<ChatMessage>,
    max_tokens: Option<u32>,
    temperature: Option<f64>,
    tools: Vec<Value>,
    think: Option<bool>,
    min_authenticity: Option<openhydra_protocol::trust::AuthenticityTier>,
    gen_permit: OwnedSemaphorePermit,
) -> (
    EventRx,
    std::time::Instant,
) {
    let (tx, rx) = tokio::sync::mpsc::channel::<GatewayEvent>(STREAM_BACKLOG);
    let started = std::time::Instant::now();
    std::thread::spawn(move || {
        // Hold the global generation permit (G3b) for the worker's whole lifetime; it is
        // released when this thread exits, freeing the slot for the next completion.
        let _gen_permit = gen_permit;
        // Set once the client is gone (its receiver dropped): the serve stops reading, closes the
        // stream and pays for what was read, instead of reading (and paying for) the rest (S8).
        let gone = std::sync::atomic::AtomicBool::new(false);
        let mut on_delta = |d: &str| {
            if tx.blocking_send(GatewayEvent::Delta(d.to_string())).is_err() {
                gone.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        };
        // Or gone before any delta (during discovery or prefill): its receiver is already dropped.
        let cancel = || gone.load(std::sync::atomic::Ordering::Relaxed) || tx.is_closed();
        let offered = offered_tool_names(&tools);
        match node.complete(&model, messages, max_tokens, temperature, tools, think, min_authenticity, &cancel, &mut on_delta) {
            Ok(mut summary) => {
                let (kept, dropped) = filter_tool_calls(std::mem::take(&mut summary.tool_calls), &offered);
                if dropped > 0 {
                    tracing::warn!(%model, dropped, kept = kept.len(), "dropped tool calls the request didn't offer or that were malformed");
                }
                summary.tool_calls = kept;
                let _ = tx.blocking_send(GatewayEvent::Done(Box::new(summary)));
            }
            Err(e) => {
                let _ = tx.blocking_send(GatewayEvent::Error(e.to_string(), error_class(&e)));
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
    let max_tokens = req.effective_max_tokens();
    let (model, rx, started) = match spawn_dispatch(
        &state,
        &headers,
        &req.model,
        false,
        session_pin.as_deref(),
        req.messages,
        max_tokens,
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
    /// The [`MIN_AUTHENTICITY_HEADER`] isn't one of `A0`–`A3`.
    BadAuthenticity,
}

/// The request header a client sets to require a minimum authenticity tier (`A0`–`A3`) for its
/// request: providers below it aren't used, and a request no provider qualifies for fails saying so.
pub const MIN_AUTHENTICITY_HEADER: &str = "x-openhydra-min-authenticity";

/// `A0`…`A3` (any case) as an authenticity tier.
///
/// A3 (hardware attestation) isn't accepted yet: nothing feeds it into a provider's tier at the
/// gateway, so a floor of A3 would refuse every request (review round 4 G4-7).
pub fn parse_authenticity_tier(s: &str) -> Option<openhydra_protocol::trust::AuthenticityTier> {
    use openhydra_protocol::trust::AuthenticityTier::*;
    match s.trim().to_ascii_uppercase().as_str() {
        "A0" => Some(SelfReported),
        "A1" => Some(ReputationRedundant),
        "A2" => Some(EnergyFingerprintPossession),
        _ => None,
    }
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
) -> Result<(String, EventRx, std::time::Instant), DispatchErr>
{
    // AUP floor: refuse a policy-violating request before spending a discovery/route on it.
    if let AupDecision::Deny(reason) = state.aup.evaluate(&messages, max_tokens) {
        return Err(DispatchErr::Aup(reason));
    }
    // D2: the client may require a minimum authenticity tier for this request (raising, never
    // lowering, the gateway's own). Not applied to a BYOK model, which no provider of ours serves.
    let min_authenticity = match headers.get(MIN_AUTHENTICITY_HEADER) {
        None => None,
        Some(v) => match v.to_str().ok().and_then(parse_authenticity_tier) {
            Some(t) => Some(t),
            None => return Err(DispatchErr::BadAuthenticity),
        },
    };
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
    let model = resolve_model(state, requested_model, fallback_unservable, session_override, min_authenticity)
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
        spawn_worker(state.node.clone(), model.clone(), messages, max_tokens, temperature, tools, think, min_authenticity, gen_permit)
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
        DispatchErr::BadAuthenticity => openai_error(
            StatusCode::BAD_REQUEST,
            &format!("{MIN_AUTHENTICITY_HEADER} must be one of A0, A1, A2"),
            "invalid_request_error",
        ),
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
    EventRx,
    std::time::Instant,
) {
    let (tx, rx) = tokio::sync::mpsc::channel::<GatewayEvent>(STREAM_BACKLOG);
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
            // The local gateway path does not request a deterministic serve.
            seed: None,
            stop: Default::default(),
        };
        // The client left (its channel closed): the hosted call stops too (round 8 G8-5).
        let stop = request.stop.clone();
        if tx.is_closed() {
            stop.set();
        }
        let mut on_delta = |d: &str| {
            if tx.blocking_send(GatewayEvent::Delta(d.to_string())).is_err() {
                stop.set();
            }
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
                        billing: 0,
                        stopped: false,
                    },
                    discover_ns: 0,
                    proxy_roundtrip_ns: 0,
                    tool_calls: outcome.tool_calls,
                    authenticity: None,
                    streamed: false,
                };
                let _ = tx.blocking_send(GatewayEvent::Done(Box::new(summary)));
            }
            Err(e) => {
                let _ = tx.blocking_send(GatewayEvent::Error(e.to_string(), error_class(&e)));
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
    rx: EventRx,
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
    let body = ReceiverStream::new(rx).map(move |ev| {
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
                    summary.streamed,
                );
                // Tool calls (if any) ride the final chunk's delta with finish_reason
                // "tool_calls"; a plain completion finishes "stop" with an empty delta.
                // (Incremental per-argument streaming of tool_calls is a later refinement —
                // delivered as one delta here, which OpenAI clients accumulate fine.)
                let (delta, finish) = if summary.tool_calls.is_empty() {
                    (json!({}), finish_reason(&summary))
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
            GatewayEvent::Error(m, (_, etype)) => {
                metrics.record_error();
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
    mut rx: EventRx,
    started: std::time::Instant,
    metrics: Arc<Metrics>,
) -> Response {
    let mut content = String::new();
    let mut outcome: Option<Result<Box<ServeSummary>, (String, ErrorClass)>> = None;
    // Time the first delta even though the HTTP body is buffered: with P1 streaming the tokens
    // arrive incrementally over the channel, so this is the real generation time-to-first-token.
    let mut ttft: Option<std::time::Duration> = None;
    while let Some(ev) = rx.recv().await {
        match ev {
            GatewayEvent::Delta(t) => {
                ttft.get_or_insert_with(|| started.elapsed());
                if content.len() + t.len() > MAX_BUFFERED_COMPLETION_BYTES {
                    outcome = Some(Err(too_large_to_buffer()));
                    break;
                }
                content.push_str(&t);
            }
            GatewayEvent::Done(s) => outcome = Some(Ok(s)),
            GatewayEvent::Error(m, class) => outcome = Some(Err((m, class))),
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
                summary.streamed,
            );
            Json(completion_object(&id, &model, created, &content, &summary, wall, ttft.unwrap_or(wall)))
                .into_response()
        }
        Some(Err((m, (status, etype)))) => {
            metrics.record_error();
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
        DispatchErr::BadAuthenticity => anthropic_error(
            StatusCode::BAD_REQUEST,
            &format!("{MIN_AUTHENTICITY_HEADER} must be one of A0, A1, A2"),
            "invalid_request_error",
        ),
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
    json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": anthropic_stop_reason(summary),
        "stop_sequence": Value::Null,
        "usage": {
            "input_tokens": summary.metrics.engine.prompt_eval_count,
            "output_tokens": summary.tokens,
        },
        "openhydra": anthropic_openhydra(summary),
    })
}

/// Anthropic's `stop_reason` for a completion ([`finish_reason`] in its words).
fn anthropic_stop_reason(summary: &ServeSummary) -> &'static str {
    match finish_reason(summary) {
        "tool_calls" => "tool_use",
        "length" => "max_tokens",
        _ => "end_turn",
    }
}

/// The OpenHydra fields an Anthropic-shaped reply carries (clients ignore unknown fields): the tier
/// the serving provider was verified at, as on the OpenAI surface (review round 4 G4-10).
fn anthropic_openhydra(summary: &ServeSummary) -> Value {
    json!({ "provider_authenticity": summary.authenticity.map(|t| t.label()) })
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
    if req.max_tokens == Some(0) {
        return anthropic_error(StatusCode::BAD_REQUEST, "max_tokens must be at least 1", "invalid_request_error");
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
    mut rx: EventRx,
    started: std::time::Instant,
    metrics: Arc<Metrics>,
) -> Response {
    let mut content = String::new();
    let mut outcome: Option<Result<Box<ServeSummary>, (String, ErrorClass)>> = None;
    while let Some(ev) = rx.recv().await {
        match ev {
            GatewayEvent::Delta(t) => {
                if content.len() + t.len() > MAX_BUFFERED_COMPLETION_BYTES {
                    outcome = Some(Err(too_large_to_buffer()));
                    break;
                }
                content.push_str(&t);
            }
            GatewayEvent::Done(s) => outcome = Some(Ok(s)),
            GatewayEvent::Error(m, class) => outcome = Some(Err((m, class))),
        }
    }
    match outcome {
        Some(Ok(summary)) => {
            let wall = started.elapsed();
            log_completion(&summary, wall);
            metrics.record_completion(summary.tokens, wall, summary.discover_ns, summary.proxy_roundtrip_ns, summary.streamed);
            Json(anthropic_message_object(&id, &model, &content, &summary)).into_response()
        }
        Some(Err((m, (status, _)))) => {
            metrics.record_error();
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
    rx: EventRx,
    started: std::time::Instant,
    metrics: Arc<Metrics>,
) -> Response {
    // The worker emits `(event_name, json)` frames; convert to axum SSE `Event` only at the boundary
    // so the translation logic stays unit-testable without axum.
    let (etx, erx) = tokio::sync::mpsc::channel::<(&'static str, Value)>(STREAM_BACKLOG);
    tokio::spawn(anthropic_stream_worker(id, model, rx, started, metrics, etx));
    let body = ReceiverStream::new(erx)
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
    mut rx: EventRx,
    started: std::time::Instant,
    metrics: Arc<Metrics>,
    out: tokio::sync::mpsc::Sender<(&'static str, Value)>,
) {
    // Wait for the first frame before committing to a message envelope. If the generation errors
    // before producing anything, emit ONLY a clean `error` frame. The pre-first-frame gap is kept
    // warm by the SSE comment keep-alive on the response — NO native `ping` here, because a `ping`
    // before `message_start` makes the official Anthropic SDK `.stream()` accumulator raise
    // "Unexpected event order, got ping before message_start" and abort.
    // The client leaving (the SSE body dropped, so `out` closes) ends the worker, which drops `rx`:
    // the serve then sees its client gone, stops reading and settles for what was read (S8; review
    // round 4 M1). Watched while waiting too, so a client that leaves during prefill is seen.
    let first = tokio::select! {
        f = rx.recv() => f,
        _ = out.closed() => return,
    };
    let first = match first {
        Some(GatewayEvent::Error(m, _)) => {
            metrics.record_error();
            let _ = out.send(("error", json!({ "type": "error", "error": { "type": "api_error", "message": m } }))).await;
            return;
        }
        Some(ev) => ev, // a Delta or Done — the message really started
        None => return, // stream closed with nothing
    };
    if out.send((
        "message_start",
        json!({
            "type": "message_start",
            "message": {
                "id": id, "type": "message", "role": "assistant", "model": model,
                "content": [], "stop_reason": Value::Null, "stop_sequence": Value::Null,
                "usage": { "input_tokens": 0, "output_tokens": 0 },
            }
        }),
    )).await.is_err() {
        return;
    }
    if out.send((
        "content_block_start",
        json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "text", "text": "" } }),
    )).await.is_err() {
        return;
    }
    // Process the first frame, then drain the rest. A native `ping` is emitted only on a mid-content
    // stall — i.e. AFTER `message_start` (canonical + SDK-safe); `biased` makes a ready frame win, so
    // pings never appear once tokens are flowing.
    let mut ev = first;
    'drain: loop {
        match ev {
            GatewayEvent::Delta(t) => {
                if out.send((
                    "content_block_delta",
                    json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "text_delta", "text": t } }),
                )).await.is_err() {
                    return;
                }
            }
            GatewayEvent::Done(summary) => {
                let wall = started.elapsed();
                log_completion(&summary, wall);
                metrics.record_completion(summary.tokens, wall, summary.discover_ns, summary.proxy_roundtrip_ns, summary.streamed);
                if out.send(("content_block_stop", json!({ "type": "content_block_stop", "index": 0 }))).await.is_err() {
                    return;
                }
                // Each tool call → a `tool_use` block whose arguments stream as one
                // `input_json_delta` (clients accumulate + parse it).
                for (i, tc) in summary.tool_calls.iter().enumerate() {
                    let idx = i + 1;
                    if out.send((
                        "content_block_start",
                        json!({ "type": "content_block_start", "index": idx, "content_block": { "type": "tool_use", "id": tc.id, "name": tc.function.name, "input": {} } }),
                    )).await.is_err() {
                        return;
                    }
                    if out.send((
                        "content_block_delta",
                        json!({ "type": "content_block_delta", "index": idx, "delta": { "type": "input_json_delta", "partial_json": tc.function.arguments } }),
                    )).await.is_err() {
                        return;
                    }
                    if out.send(("content_block_stop", json!({ "type": "content_block_stop", "index": idx }))).await.is_err() {
                        return;
                    }
                }
                if out.send((
                    "message_delta",
                    json!({ "type": "message_delta", "delta": { "stop_reason": anthropic_stop_reason(&summary), "stop_sequence": Value::Null }, "usage": { "output_tokens": summary.tokens }, "openhydra": anthropic_openhydra(&summary) }),
                )).await.is_err() {
                    return;
                }
                if out.send(("message_stop", json!({ "type": "message_stop" }))).await.is_err() {
                    return;
                }
                break 'drain;
            }
            GatewayEvent::Error(m, _) => {
                metrics.record_error();
                if out.send(("error", json!({ "type": "error", "error": { "type": "api_error", "message": m } }))).await.is_err() {
                    return;
                }
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
                _ = out.closed() => return,
                _ = tokio::time::sleep(PING_INTERVAL) => {
                    if out.send(("ping", json!({ "type": "ping" }))).await.is_err() {
                        return;
                    }
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

/// Is `origin` (a browser's `Origin` header) a page served from this machine's loopback?
fn is_loopback_origin(origin: &str) -> bool {
    let Some(rest) = origin.strip_prefix("http://").or_else(|| origin.strip_prefix("https://")) else {
        return false;
    };
    let host = rest.split('/').next().unwrap_or("");
    let name = match host.rsplit_once(':') {
        Some((n, p)) if !p.contains(']') && p.parse::<u16>().is_ok() => n,
        _ => host,
    };
    name == "127.0.0.1" || name == "[::1]" || name.eq_ignore_ascii_case("localhost")
}

/// Auth middleware for `/v1/*`: when an API key is configured, require a matching
/// `Authorization: Bearer <key>` (or `x-api-key`). Without one (a loopback CLI gateway), refuse
/// requests a web page sends: a browser always names the page's `Origin`, coding tools send none, so
/// a site the user visits can't spend their credit through the open gateway. (With a key, a page
/// can't present it: a cross-site request with that header needs CORS, which the gateway never
/// grants.)
async fn require_api_key(State(state): State<AppState>, req: Request, next: Next) -> Response {
    if state.api_key.is_none() {
        if let Some(origin) = req.headers().get(axum::http::header::ORIGIN) {
            let origin = origin.to_str().unwrap_or("");
            if !is_loopback_origin(origin) {
                tracing::warn!(origin = %origin.escape_debug(), "refused a browser request from another site");
                return openai_error(
                    StatusCode::FORBIDDEN,
                    "requests from web pages are refused; this gateway serves local programs",
                    "forbidden",
                );
            }
        }
    }
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
    provider_attest: Option<Vec<[u8; 32]>>,
    logprob_refs: Vec<crate::prompt_score::LoadedReference>,
    logprob_registry: Option<crate::registry_client::RegistryClient>,
    audit_replay: bool,
    assurance: openhydra_protocol::trust::AssurancePolicy,
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
    // Crypto-rail settlement peg (USDC micro-units per credit unit): opt-in via
    // `OPENHYDRA_USDC_MICROS_PER_CREDIT`. Unset/invalid/zero → the crypto rail stays OFF (no
    // cumulative stamped), so a node never emits mis-denominated vouchers (review F3).
    let settlement_peg = std::env::var("OPENHYDRA_USDC_MICROS_PER_CREDIT")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);
    let node = node
        .with_stats(stats)
        .with_self_provider(self_provider)
        .with_cards(cards)
        .with_credentials(creds)
        .with_settlement_peg(settlement_peg)
        // The on-chain deployment paying receipts settle on (validated at startup in `serve`).
        .with_settlement_chain(crate::settlement_config::from_env(true).ok().flatten());
    // Phase 0 device attestation of providers (opt-in): informational device-trust badge only.
    let node = match provider_attest {
        Some(trusted) => node.with_provider_attestation(trusted),
        None => node,
    };
    // Track A2-Logprob prompt-scoring checks of providers (opt-in; no-op with no reference sets).
    let node = node.with_logprob_checks(logprob_refs, logprob_registry);
    // D1: audits may replay users' greedy requests to public models (see `with_audit_replay`).
    let node = node.with_audit_replay(audit_replay);
    // D2: the buyer's floor for every request (a request can raise it, see MIN_AUTHENTICITY_HEADER).
    let node = node.with_assurance_policy(assurance);
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
/// Is `host` (a `Host` header or URI authority) a loopback name for this gateway's `port`?
/// `127.0.0.1`, `localhost` or `[::1]`, with this port or none.
pub(crate) fn is_loopback_host(host: &str, port: u16) -> bool {
    let host = host.trim();
    let (name, p) = match host.rsplit_once(':') {
        Some((n, p)) if !n.is_empty() && (n.starts_with('[') == n.ends_with(']')) && !p.contains(']') => (n, Some(p)),
        _ => (host, None),
    };
    let name_ok = name == "127.0.0.1" || name == "[::1]" || name.eq_ignore_ascii_case("localhost");
    name_ok && p.map_or(true, |p| p.parse::<u16>().ok() == Some(port))
}

/// Refuse a request to a loopback-bound gateway whose `Host` isn't a loopback name (403).
async fn loopback_host_guard(State(port): State<u16>, req: Request, next: Next) -> Response {
    let host = req
        .headers()
        .get(axum::http::header::HOST)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
        .or_else(|| req.uri().authority().map(|a| a.as_str().to_string()));
    match host {
        Some(h) if !is_loopback_host(&h, port) => {
            tracing::warn!(host = %h.escape_debug(), "refused a request addressed to a non-loopback host name");
            openai_error(StatusCode::FORBIDDEN, "this gateway only answers requests to 127.0.0.1 / localhost", "forbidden")
        }
        _ => next.run(req).await,
    }
}

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
    provider_attest: Option<Vec<[u8; 32]>>,
    logprob_refs: Vec<crate::prompt_score::LoadedReference>,
    logprob_registry: Option<crate::registry_client::RegistryClient>,
    audit_replay: bool,
    assurance: openhydra_protocol::trust::AssurancePolicy,
) -> std::io::Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(bind).await?;
        let local = listener.local_addr()?;
        let mut app = router(net, economy, stats, api_key, store, aup, rate_limit_cfg, trusted_proxy, byok, embeddings_cfg, self_provider, cards, creds, provider_attest, logprob_refs, logprob_registry, audit_replay, assurance);
        // A loopback-only gateway answers only requests addressed to loopback (DNS rebinding:
        // a web page can point its own host name at 127.0.0.1, but not change the Host header the
        // browser sends). A gateway bound to a LAN/public address is reached by other names, so it
        // relies on --api-key instead.
        if local.ip().is_loopback() {
            app = app.layer(middleware::from_fn_with_state(local.port(), loopback_host_guard));
        }
        // `into_make_service_with_connect_info` surfaces the peer `SocketAddr` to the
        // rate-limit middleware (the unspoofable per-IP key).
        let app = app.into_make_service_with_connect_info::<SocketAddr>();
        axum::serve(listener, app).await
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::EngineMetrics;

    #[test]
    fn the_min_authenticity_header_names_a_tier() {
        use openhydra_protocol::trust::AuthenticityTier::*;
        assert_eq!(parse_authenticity_tier("A1"), Some(ReputationRedundant));
        assert_eq!(parse_authenticity_tier(" a2 "), Some(EnergyFingerprintPossession));
        assert_eq!(parse_authenticity_tier("A0"), Some(SelfReported));
        assert_eq!(parse_authenticity_tier("A3"), None, "not reachable yet");
        assert_eq!(parse_authenticity_tier("A4"), None);
        assert_eq!(parse_authenticity_tier("high"), None);
    }

    #[test]
    fn the_response_reports_the_providers_tier() {
        let mut s = summary(10, 5);
        assert!(openhydra_block(&s, std::time::Duration::from_secs(1), std::time::Duration::ZERO)["provider_authenticity"].is_null());
        s.authenticity = Some(openhydra_protocol::trust::AuthenticityTier::ReputationRedundant);
        assert_eq!(
            openhydra_block(&s, std::time::Duration::from_secs(1), std::time::Duration::ZERO)["provider_authenticity"],
            "A1-reputation-redundant"
        );
    }

    #[test]
    fn network_rtt_and_speeds_are_meaningful() {
        let one = std::time::Duration::from_secs(1);
        // Buffered: the round trip spans the serve (600 ms), so the network share is 100 ms.
        let s = summary(10, 5);
        assert_eq!(openhydra_block(&s, one, one)["hops_ms"]["network_rtt"], 100);
        // Streamed: the round trip is the first ack (40 ms), shorter than the serve: that IS the RTT.
        let mut st = summary(10, 5);
        st.streamed = true;
        st.proxy_roundtrip_ns = 40_000_000;
        assert_eq!(openhydra_block(&st, one, one)["hops_ms"]["network_rtt"], 40);
        // The gateway's own share is what's left after discovery, network and serve, on both: a
        // streamed serve no longer reports its whole generation as gateway overhead.
        for b in [openhydra_block(&s, one, one), openhydra_block(&st, one, one)] {
            let h = &b["hops_ms"];
            let rest = 1000 - h["discover"].as_u64().unwrap() - h["network_rtt"].as_u64().unwrap() - h["provider_serve"].as_u64().unwrap();
            assert_eq!(h["gateway_overhead"].as_u64().unwrap(), rest, "{h}");
            assert!(rest < 1000 - h["provider_serve"].as_u64().unwrap());
        }
        // A 1-token reply with no eval time has no meaningful speed: null, not 1,000,000 tok/s.
        let mut t = summary(1, 5);
        t.metrics.engine.eval_duration_ns = 0;
        assert!(openhydra_block(&t, one, one)["engine"]["native_tps"].is_null());
        t.metrics.engine.eval_count = 20;
        t.metrics.engine.eval_duration_ns = 1_000_000_000;
        assert_eq!(openhydra_block(&t, one, one)["engine"]["native_tps"], 20.0);
    }

    fn summary(tokens: u64, prompt: u64) -> ServeSummary {
        ServeSummary {
            tokens,
            ok: true,
            metrics: crate::serve::ServeMetrics {
                engine: EngineMetrics { prompt_eval_count: prompt, eval_count: tokens, ..Default::default() },
                provider_serve_ns: 500_000_000,
                billing: 0,
                stopped: false,
            },
            proxy_roundtrip_ns: 600_000_000,
            discover_ns: 1_000_000,
            tool_calls: Vec::new(),
            authenticity: None,
            streamed: false,
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
    fn loopback_host_names_pass_and_rebound_names_do_not() {
        for ok in ["127.0.0.1:16527", "localhost:16527", "LOCALHOST:16527", "[::1]:16527", "127.0.0.1", "localhost", "[::1]"] {
            assert!(is_loopback_host(ok, 16527), "{ok}");
        }
        for bad in ["evil.example:16527", "evil.example", "127.0.0.1:80", "localhost:1", "127.0.0.1.evil.example:16527", "[::1]:9", "", "localhost.evil:16527"] {
            assert!(!is_loopback_host(bad, 16527), "{bad}");
        }
    }

    #[tokio::test]
    async fn loopback_guard_refuses_a_rebound_host_with_403() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let app = Router::new()
            .route("/health", get(|| async { "ok" }))
            .layer(middleware::from_fn_with_state(port, loopback_host_guard));
        tokio::spawn(async move { axum::serve(listener, app).await });
        let status_for = |host: String| async move {
            let mut s = tokio::net::TcpStream::connect(("127.0.0.1", port)).await.unwrap();
            s.write_all(format!("GET /health HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n").as_bytes())
                .await
                .unwrap();
            let mut out = String::new();
            s.read_to_string(&mut out).await.unwrap();
            out.lines().next().unwrap_or_default().to_string()
        };
        assert!(status_for(format!("evil.example:{port}")).await.contains(" 403 "));
        assert!(status_for(format!("127.0.0.1:{port}")).await.contains(" 200 "));
        assert!(status_for(format!("localhost:{port}")).await.contains(" 200 "));
    }

    /// Serve the real gateway router on a loopback port; returns the port (the node and its temp dir
    /// live for the test).
    async fn serve_test_gateway(api_key: Option<&str>) -> (u16, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let net = NetworkHandle::start(openhydra_network::node::NodeConfig {
            identity_path: dir.path().join("id.key"),
            listen_addrs: vec!["/ip4/127.0.0.1/tcp/0".into()],
            bootstrap_peers: vec![],
            enable_peer_relay: false,
            enable_connection_reversal: false,
            pcp_gateway: None,
        })
        .unwrap();
        let app = router(
            net,
            Arc::new(EconomyStats::default()),
            Arc::new(crate::status::TransferStats::default()),
            api_key.map(str::to_string),
            None,
            AupPolicy::default(),
            RateLimitConfig::disabled(),
            false,
            ByokConfig::empty(),
            EmbeddingConfig::empty(),
            None,
            None,
            None,
            None,
            Vec::new(),
            None,
            false,
            openhydra_protocol::trust::AssurancePolicy::OPEN,
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            axum::serve(listener, app.into_make_service_with_connect_info::<SocketAddr>()).await
        });
        (port, dir)
    }

    /// The status line of `GET /v1/models` with `headers` (each "Name: value").
    async fn models_status(port: u16, headers: &[&str]) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let mut s = tokio::net::TcpStream::connect(("127.0.0.1", port)).await.unwrap();
        let extra: String = headers.iter().map(|h| format!("{h}\r\n")).collect();
        s.write_all(format!("GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n{extra}Connection: close\r\n\r\n").as_bytes())
            .await
            .unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).await.unwrap();
        out.lines().next().unwrap_or_default().to_string()
    }

    #[tokio::test]
    async fn a_keyed_gateway_refuses_requests_without_its_key() {
        let key = format!("oh-{}", "ab".repeat(32));
        let (port, _dir) = serve_test_gateway(Some(&key)).await;
        assert!(models_status(port, &[]).await.contains(" 401 "), "no key");
        assert!(models_status(port, &["Authorization: Bearer oh-local"]).await.contains(" 401 "), "the old placeholder");
        assert!(models_status(port, &[&format!("Authorization: Bearer {key}")]).await.contains(" 200 "));
        assert!(models_status(port, &[&format!("x-api-key: {key}")]).await.contains(" 200 "), "Anthropic-style header");
    }

    #[tokio::test]
    async fn a_keyless_gateway_refuses_web_pages() {
        let (port, _dir) = serve_test_gateway(None).await;
        assert!(models_status(port, &[]).await.contains(" 200 "), "a local tool sends no Origin");
        assert!(models_status(port, &["Origin: https://evil.example"]).await.contains(" 403 "));
        assert!(models_status(port, &["Origin: null"]).await.contains(" 403 "), "sandboxed frame / file page");
        assert!(models_status(port, &["Origin: http://localhost:3000"]).await.contains(" 200 "), "a local dev page");
        assert!(models_status(port, &["Origin: http://127.0.0.1.evil.example"]).await.contains(" 403 "));
    }

    #[test]
    fn loopback_origins_are_only_loopback_pages() {
        for ok in ["http://localhost", "http://localhost:1420", "https://127.0.0.1:8443", "http://[::1]:3000"] {
            assert!(is_loopback_origin(ok), "{ok}");
        }
        for bad in ["null", "tauri://localhost", "http://localhost.evil.example", "http://127.0.0.1.nip.io", "https://evil.example", ""] {
            assert!(!is_loopback_origin(bad), "{bad}");
        }
    }

    fn call(name: &str, args: &str) -> ToolCall {
        ToolCall {
            id: format!("call_{name}"),
            kind: "function".into(),
            function: ToolCallFunction { name: name.into(), arguments: args.into() },
        }
    }

    #[test]
    fn tool_calls_are_kept_only_for_offered_tools_with_object_arguments() {
        let offered = offered_tool_names(&[
            json!({ "type": "function", "function": { "name": "read_file" } }),
            json!({ "type": "function", "function": { "name": "list_dir" } }),
        ]);
        assert_eq!(offered.len(), 2);
        // Nothing offered → nothing passed on.
        let (kept, dropped) = filter_tool_calls(vec![call("bash", "{\"cmd\":\"rm -rf ~\"}")], &Default::default());
        assert!(kept.is_empty());
        assert_eq!(dropped, 1);
        let (kept, dropped) = filter_tool_calls(
            vec![
                call("read_file", "{\"path\":\"a\"}"),
                call("bash", "{\"cmd\":\"x\"}"),  // not offered
                call("list_dir", "[1,2]"),            // not an object
                call("list_dir", "not json"),
                call("list_dir", ""),                 // no arguments → {}
            ],
            &offered,
        );
        assert_eq!(dropped, 3);
        assert_eq!(kept.iter().map(|c| c.function.name.as_str()).collect::<Vec<_>>(), ["read_file", "list_dir"]);
        assert_eq!(kept[1].function.arguments, "{}");
    }

    #[test]
    fn tool_calls_are_capped_in_number_and_size() {
        let offered = offered_tool_names(&[json!({ "type": "function", "function": { "name": "f" } })]);
        let (kept, dropped) = filter_tool_calls((0..100).map(|_| call("f", "{}")).collect(), &offered);
        assert_eq!((kept.len(), dropped), (MAX_TOOL_CALLS, 100 - MAX_TOOL_CALLS));
        let big = format!("{{\"x\":\"{}\"}}", "a".repeat(MAX_TOOL_CALL_BYTES));
        let (kept, _) = filter_tool_calls(vec![call("f", &big), call("f", "{}")], &offered);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].function.arguments, "{}");
    }

    fn kp(model: &str, oh: &str, lib: &str) -> openhydra_network::types::KnownProvider {
        openhydra_network::types::KnownProvider {
            model_id: model.into(),
            openhydra_peer_id: oh.into(),
            libp2p_peer_id: lib.into(),
            public_key: String::new(),
        }
    }

    fn cands(providers: Vec<openhydra_network::types::KnownProvider>) -> crate::consumer::AutoCandidates {
        crate::consumer::AutoCandidates { providers, ..Default::default() }
    }

    #[test]
    fn auto_never_picks_a_model_only_unknown_peers_serve() {
        let c = cands(vec![kp("0", "evil", "12D3Evil"), kp("qwen2.5:7b", "a", "12D3A")]);
        assert_eq!(pick_auto_model(&c, None), None);
        // Several peer ids announcing one model cost nothing: a count is not trust.
        let c = cands(vec![kp("0", "evil", "12D3Evil"), kp("0", "evil2", "12D3Evil2"), kp("0", "evil3", "12D3Evil3")]);
        assert_eq!(pick_auto_model(&c, None), None);
        assert_eq!(pick_auto_model(&cands(vec![]), Some("x")), None);
    }

    #[test]
    fn auto_prefers_override_then_own_then_cards_then_reputation_then_verified() {
        use openhydra_protocol::trust::AuthenticityTier::*;
        let mut c = cands(vec![
            kp("0", "evil", "12D3Evil"),
            kp("0", "evil2", "12D3Evil2"),
            kp("popular", "p1", "12D3P1"),
            kp("popular", "p2", "12D3P2"),
            kp("trusted", "t", "12D3T"),
            kp("verified", "v", "12D3V"),
            kp("mine", "me", "12D3Me"),
        ]);
        c.self_provider = Some("12D3Me".into());
        c.card_models = vec!["carded".into()];
        c.earned.insert(("12D3T".into(), "trusted".into()), 70.0);
        c.tiers.insert(("12D3V".into(), "verified".into()), HardwareAttested);
        // 1. an override that is served wins; one that isn't served is ignored
        assert_eq!(pick_auto_model(&c, Some("popular")).as_deref(), Some("popular"));
        assert_eq!(pick_auto_model(&c, Some("carded")).as_deref(), Some("carded"));
        assert_eq!(pick_auto_model(&c, Some("not-served")).as_deref(), Some("mine"));
        // 2. own provider
        assert_eq!(pick_auto_model(&c, None).as_deref(), Some("mine"));
        // 3. cards
        c.self_provider = None;
        assert_eq!(pick_auto_model(&c, None).as_deref(), Some("carded"));
        // 4. earned reputation above neutral
        c.card_models.clear();
        assert_eq!(pick_auto_model(&c, None).as_deref(), Some("trusted"));
        // …but not at or below neutral, and not from standing earned on another model
        c.earned.insert(("12D3T".into(), "trusted".into()), 50.0);
        c.earned.insert(("12D3T".into(), "other-model".into()), 99.0);
        // 5. a provider this node verified (A1+), highest tier first
        assert_eq!(pick_auto_model(&c, None).as_deref(), Some("verified"));
        c.tiers.insert(("12D3P1".into(), "popular".into()), ReputationRedundant);
        assert_eq!(pick_auto_model(&c, None).as_deref(), Some("verified"));
        c.tiers.remove(&("12D3V".to_string(), "verified".to_string()));
        assert_eq!(pick_auto_model(&c, None).as_deref(), Some("popular"));
        // Nothing earned or verified: no pick, however many providers announce a model.
        c.tiers.clear();
        assert_eq!(pick_auto_model(&c, None), None);
    }

    #[test]
    fn auto_only_picks_what_the_authenticity_floor_will_route_to() {
        use openhydra_protocol::trust::AuthenticityTier::*;
        let mut c = cands(vec![kp("mine", "me", "12D3Me"), kp("carded", "c", "12D3C"), kp("trusted", "t", "12D3T"), kp("strong", "s", "12D3S")]);
        c.self_provider = Some("12D3Me".into());
        c.card_models = vec!["carded".into(), "card-only".into()];
        c.earned.insert(("12D3T".into(), "trusted".into()), 70.0);
        c.tiers.insert(("12D3T".into(), "trusted".into()), ReputationRedundant);
        c.tiers.insert(("12D3S".into(), "strong".into()), EnergyFingerprintPossession);
        assert_eq!(pick_auto_model(&c, None).as_deref(), Some("mine"));
        // At A1, own and card providers below it are skipped (routing would refuse them).
        c.floor = Some(ReputationRedundant);
        assert_eq!(pick_auto_model(&c, None).as_deref(), Some("trusted"));
        assert_eq!(pick_auto_model(&c, Some("mine")).as_deref(), Some("trusted"), "a preferred model below the floor is skipped");
        // A card whose provider clears the floor still comes first.
        c.tiers.insert(("12D3C".into(), "carded".into()), ReputationRedundant);
        assert_eq!(pick_auto_model(&c, None).as_deref(), Some("carded"));
        // At A2 only the strong provider remains.
        c.floor = Some(EnergyFingerprintPossession);
        assert_eq!(pick_auto_model(&c, None).as_deref(), Some("strong"));
        c.floor = Some(HardwareAttested);
        assert_eq!(pick_auto_model(&c, None), None);
    }

    #[test]
    fn no_auto_model_message_lists_what_is_available() {
        assert!(no_auto_model_message(&[]).contains("no models available"));
        let m = no_auto_model_message(&["a".into(), "b".into()]);
        assert!(m.contains("Choose a model") && m.contains("a, b"), "{m}");
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
        let (tx, rx) = tokio::sync::mpsc::channel(events.len().max(1));
        for e in events {
            tx.try_send(e).unwrap();
        }
        drop(tx); // close so the worker's drain loop ends
        let (otx, mut orx) = tokio::sync::mpsc::channel(1024);
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
        let names = stream_names(vec![GatewayEvent::Error("no provider".into(), (StatusCode::SERVICE_UNAVAILABLE, "no_provider"))]).await;
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
        let (tx, rx) = tokio::sync::mpsc::channel(16);
        let (otx, mut orx) = tokio::sync::mpsc::channel(1024);
        let worker = tokio::spawn(anthropic_stream_worker(
            "id".into(),
            "m".into(),
            rx,
            std::time::Instant::now(),
            Arc::new(Metrics::new()),
            otx,
        ));
        // First token opens the message envelope.
        tx.send(GatewayEvent::Delta("hi".into())).await.unwrap_or_else(|_| panic!("send"));
        tokio::task::yield_now().await;
        // Then the model stalls: advance past two ping intervals → native pings fire, AFTER
        // message_start (where the Anthropic protocol allows them).
        tokio::time::advance(std::time::Duration::from_secs(25)).await;
        tokio::task::yield_now().await;
        tx.send(GatewayEvent::Done(Box::new(summary(1, 1)))).await.unwrap_or_else(|_| panic!("send"));
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
    fn the_token_cap_reads_either_openai_name_and_a_zero_cap_is_rejected() {
        let parse = |j: &str| serde_json::from_str::<ChatRequest>(j).unwrap();
        assert_eq!(parse(r#"{"model":"m","messages":[]}"#).effective_max_tokens(), None);
        assert_eq!(parse(r#"{"model":"m","max_tokens":64}"#).effective_max_tokens(), Some(64));
        assert_eq!(parse(r#"{"model":"m","max_completion_tokens":32}"#).effective_max_tokens(), Some(32));
        assert_eq!(
            parse(r#"{"model":"m","max_tokens":64,"max_completion_tokens":32}"#).effective_max_tokens(),
            Some(32)
        );
        let zero = parse(r#"{"model":"m","messages":[{"role":"user","content":"x"}],"max_completion_tokens":0}"#);
        assert_eq!(validate(&zero).map(|r| r.status()), Some(StatusCode::BAD_REQUEST));
        let ok = parse(r#"{"model":"m","messages":[{"role":"user","content":"x"}],"max_tokens":1}"#);
        assert!(validate(&ok).is_none());
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
        let r = ChatRequest { model: "".into(), messages: vec![ChatMessage { role: "user".into(), content: "x".into(), ..Default::default() }], max_tokens: None, max_completion_tokens: None, temperature: None, tools: Vec::new(), stream: false, stream_options: None, think: None, chat_template_kwargs: None };
        assert!(validate(&r).is_some());
        let r = ChatRequest { model: "m".into(), messages: vec![], max_tokens: None, max_completion_tokens: None, temperature: None, tools: Vec::new(), stream: false, stream_options: None, think: None, chat_template_kwargs: None };
        assert!(validate(&r).is_some());
        let r = ChatRequest { model: "m".into(), messages: vec![ChatMessage { role: "user".into(), content: "x".into(), ..Default::default() }], max_tokens: None, max_completion_tokens: None, temperature: None, tools: Vec::new(), stream: false, stream_options: None, think: None, chat_template_kwargs: None };
        assert!(validate(&r).is_none());
    }

    #[test]
    fn error_classification_maps_to_status() {
        assert_eq!(classify_error("engine http error: no provider for model 'm'").0, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(classify_error("proxy_forward: proxy_forward timed out after 45s").0, StatusCode::GATEWAY_TIMEOUT);
        assert_eq!(classify_error("engine http error: connection refused").0, StatusCode::BAD_GATEWAY);
        // Review E1: routing refusals, provider errors and network failures as now worded.
        assert_eq!(classify_error("no provider of 'm' is verified at A1-reputation-redundant or above (1 found below it)"), (StatusCode::SERVICE_UNAVAILABLE, "no_provider"));
        assert_eq!(classify_error("provider: engine http error: ollama: model crashed").0, StatusCode::BAD_GATEWAY);
        assert_eq!(classify_error("network: stream stalled (no progress before deadline)").0, StatusCode::BAD_GATEWAY);
        assert_eq!(classify_error("network: proxy_forward timed out after 45s").0, StatusCode::GATEWAY_TIMEOUT);
        // By kind (review round 4 G4-8): a provider's error can't pass for a routing refusal or a
        // timeout, whatever it says.
        use crate::adapter::AdapterError;
        assert_eq!(error_class(&AdapterError::Provider("no provider for you, timed out".into())), (StatusCode::BAD_GATEWAY, "upstream_error"));
        assert_eq!(error_class(&AdapterError::Route("no provider for model 'm'".into())).0, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(error_class(&AdapterError::Route("all providers failed for 'm'".into())).0, StatusCode::BAD_GATEWAY);
        assert_eq!(error_class(&AdapterError::Network("proxy_forward timed out after 45s".into())).0, StatusCode::GATEWAY_TIMEOUT);
        // Round 8 G8-4: libp2p's "Timeout", whatever its case.
        assert_eq!(error_class(&AdapterError::Network("proxy_forward: proxy outbound: Timeout".into())).0, StatusCode::GATEWAY_TIMEOUT);
        assert_eq!(error_class(&AdapterError::Overloaded("stream shed at open".into())).0, StatusCode::SERVICE_UNAVAILABLE);
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
