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

use axum::extract::{ConnectInfo, Request};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{extract::State, Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;

use openhydra_network::handle::NetworkHandle;

use crate::adapter::ChatMessage;
use crate::aup::{AupDecision, AupPolicy};
use crate::consumer::ConsumerNode;
use crate::metrics::Metrics;
use crate::ratelimit::{RateLimitConfig, RateLimiter};
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
    /// Honor `X-Forwarded-For` for per-IP keying (only behind a trusted reverse proxy — the
    /// header is client-spoofable). Default `false`: key off the unspoofable socket address.
    trusted_proxy: bool,
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
    /// OpenAI default is `false` → a single JSON object; `true` → an SSE stream.
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stream_options: Option<StreamOptions>,
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

// ── Response builders (pure) ─────────────────────────────────────────────────

/// `usage` object: prompt tokens are the engine's count (0 if it reports none),
/// completion tokens are what the pipeline counted.
fn usage_value(summary: &ServeSummary) -> Value {
    let prompt = summary.metrics.engine.prompt_eval_count;
    let completion = summary.tokens;
    json!({
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    })
}

/// The `openhydra` telemetry block: three views of one request (engine ground-truth,
/// pipeline end-to-end, and where the wall-clock went hop-by-hop).
fn openhydra_block(summary: &ServeSummary, wall: std::time::Duration) -> Value {
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
            "ttft_ms": ms(wall_ns), // buffered transport: no token reaches the client until done
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

/// The non-streaming `chat.completion` object.
fn completion_object(
    id: &str,
    model: &str,
    created: u64,
    content: &str,
    summary: &ServeSummary,
    wall: std::time::Duration,
) -> Value {
    json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": content },
            "finish_reason": "stop",
        }],
        "usage": usage_value(summary),
        "openhydra": openhydra_block(summary, wall),
    })
}

// ── Handler ──────────────────────────────────────────────────────────────────

/// Spawn the blocking `complete` on a plain OS thread, returning the event channel and the
/// start instant. The worker forwards each delta, then a terminal `Done`/`Error`.
fn spawn_worker(
    node: Arc<ConsumerNode>,
    model: String,
    messages: Vec<ChatMessage>,
    max_tokens: Option<u32>,
    temperature: Option<f64>,
) -> (
    tokio::sync::mpsc::UnboundedReceiver<GatewayEvent>,
    std::time::Instant,
) {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<GatewayEvent>();
    let started = std::time::Instant::now();
    std::thread::spawn(move || {
        let mut on_delta = |d: &str| {
            let _ = tx.send(GatewayEvent::Delta(d.to_string()));
        };
        match node.complete(&model, messages, max_tokens, temperature, &mut on_delta) {
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
    // AUP floor: refuse a policy-violating request before spending a discovery/route on it.
    if let AupDecision::Deny(reason) = state.aup.evaluate(&req.messages, req.max_tokens) {
        return openai_error(StatusCode::BAD_REQUEST, &reason, "invalid_request_error");
    }

    let id = next_id();
    let created = unix_now();
    let want_usage = req.stream_options.as_ref().is_some_and(|o| o.include_usage);
    let (rx, started) = spawn_worker(
        state.node.clone(),
        req.model.clone(),
        req.messages,
        req.max_tokens,
        req.temperature,
    );

    if req.stream {
        stream_response(id, req.model, created, want_usage, rx, started, state.metrics.clone())
    } else {
        buffered_response(id, req.model, created, rx, started, state.metrics.clone()).await
    }
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
    let body = UnboundedReceiverStream::new(rx).map(move |ev| {
        let data = match ev {
            GatewayEvent::Delta(t) => {
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
                let mut chunk: Value =
                    serde_json::from_str(&stream_chunk(&id_s, &model_s, created, json!({}), Some("stop")))
                        .unwrap_or_else(|_| json!({}));
                chunk["openhydra"] = openhydra_block(&summary, wall);
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
    Sse::new(role.chain(body).chain(done)).into_response()
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
            metrics.record_completion(
                summary.tokens,
                wall,
                summary.discover_ns,
                summary.proxy_roundtrip_ns,
            );
            Json(completion_object(&id, &model, created, &content, &summary, wall)).into_response()
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

/// `GET /v1/models` — the models this gateway currently knows a provider for (PEX-learned
/// / discovered). Dynamic in a decentralized swarm: empty until gossip arrives, then grows
/// as providers announce. The blocking swarm query runs on a plain OS thread (same reason
/// as `complete` — `blocking_send` needs a non-tokio context).
async fn list_models(State(state): State<AppState>) -> Response {
    state.metrics.incr_models();
    let node = state.node.clone();
    let (tx, rx) = tokio::sync::oneshot::channel();
    std::thread::spawn(move || {
        let _ = tx.send(node.known_models());
    });
    match rx.await {
        Ok(Ok(models)) => {
            let data: Vec<Value> = models
                .iter()
                .map(|m| json!({ "id": m, "object": "model", "created": 0, "owned_by": "openhydra" }))
                .collect();
            Json(json!({ "object": "list", "data": data })).into_response()
        }
        Ok(Err(e)) => openai_error(StatusCode::INTERNAL_SERVER_ERROR, &e, "internal_error"),
        Err(_) => openai_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "models query failed",
            "internal_error",
        ),
    }
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
        let presented = req
            .headers()
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "));
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
        if let Some(ip) = req
            .headers()
            .get("x-forwarded-for")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.split(',').next())
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
        Ok(_guard) => next.run(req).await,
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
pub fn router(
    net: NetworkHandle,
    api_key: Option<String>,
    store: Option<Store>,
    aup: AupPolicy,
    rate_limit_cfg: RateLimitConfig,
    trusted_proxy: bool,
) -> Router {
    let node = match store {
        Some(s) => ConsumerNode::with_store(net, s), // M2.2(a): persisted reputation
        None => ConsumerNode::new(net),
    };
    let state = AppState {
        node: Arc::new(node),
        api_key: api_key.map(Arc::new),
        metrics: Arc::new(Metrics::new()),
        aup: Arc::new(aup),
        rate_limiter: Arc::new(RateLimiter::new(rate_limit_cfg)),
        trusted_proxy,
    };
    // The `/v1/*` routes are auth-gated then rate-limited; `/health` and `/metrics` stay open
    // for liveness probes and Prometheus scraping. `route_layer`s run outermost-last, so
    // adding `rate_limit` first and `require_api_key` last makes auth run *before* the
    // limiter — the limiter can then key off the validated API key.
    let v1 = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
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
    bind: &str,
    api_key: Option<String>,
    store: Option<Store>,
    aup: AupPolicy,
    rate_limit_cfg: RateLimitConfig,
    trusted_proxy: bool,
) -> std::io::Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(bind).await?;
        // `into_make_service_with_connect_info` surfaces the peer `SocketAddr` to the
        // rate-limit middleware (the unspoofable per-IP key).
        let app = router(net, api_key, store, aup, rate_limit_cfg, trusted_proxy)
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
        }
    }

    #[test]
    fn request_defaults_to_non_streaming() {
        let req: ChatRequest =
            serde_json::from_str(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#).unwrap();
        assert!(!req.stream, "OpenAI default is stream:false");
    }

    #[test]
    fn validation_rejects_empty_model_and_messages() {
        let r = ChatRequest { model: "".into(), messages: vec![ChatMessage { role: "user".into(), content: "x".into() }], max_tokens: None, temperature: None, stream: false, stream_options: None };
        assert!(validate(&r).is_some());
        let r = ChatRequest { model: "m".into(), messages: vec![], max_tokens: None, temperature: None, stream: false, stream_options: None };
        assert!(validate(&r).is_some());
        let r = ChatRequest { model: "m".into(), messages: vec![ChatMessage { role: "user".into(), content: "x".into() }], max_tokens: None, temperature: None, stream: false, stream_options: None };
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
        let v = completion_object("chatcmpl-1", "unsloth/Llama-3.2-1B-Instruct", 1_700_000_000, "hello world", &s, std::time::Duration::from_millis(500));
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
}
