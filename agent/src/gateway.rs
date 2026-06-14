// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! The consumer front door: an OpenAI-compatible HTTP/SSE gateway over [`ConsumerNode`].
//!
//! `POST /v1/chat/completions` discovers a provider for the requested model, streams the
//! completion over libp2p, and relays it to the client as Server-Sent Events.
//!
//! [`ConsumerNode::complete`] is synchronous and blocks on libp2p (`blocking_send`), so
//! the handler runs it on a **plain OS thread** (outside any tokio context, where
//! `blocking_send` is valid — `spawn_blocking` threads have murkier runtime-context
//! semantics) and pipes the deltas back through an unbounded channel into the async SSE
//! stream.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use axum::response::sse::{Event, Sse};
use axum::response::IntoResponse;
use axum::routing::post;
use axum::{extract::State, Json, Router};
use serde::Deserialize;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;

use openhydra_network::handle::NetworkHandle;

use crate::adapter::ChatMessage;
use crate::consumer::ConsumerNode;

#[derive(Clone)]
struct AppState {
    node: Arc<ConsumerNode>,
}

/// The OpenAI chat-completions request fields we honour (others ignored).
#[derive(Debug, Deserialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f64>,
}

/// What the worker thread sends back as the completion progresses.
enum GatewayEvent {
    Delta(String),
    Done,
    Error(String),
}

static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(1);

/// An OpenAI `chat.completion.chunk`: a content delta, or (with `finish`) the terminator.
fn chunk_json(id: &str, model: &str, content: Option<&str>, finish: Option<&str>) -> String {
    let delta = match content {
        Some(c) => serde_json::json!({ "content": c }),
        None => serde_json::json!({}),
    };
    serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }],
    })
    .to_string()
}

fn error_json(id: &str, model: &str, message: &str) -> String {
    serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{ "index": 0, "delta": {}, "finish_reason": "error" }],
        "error": { "message": message },
    })
    .to_string()
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> impl IntoResponse {
    let id = format!("chatcmpl-{}", REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed));
    let model = req.model.clone();
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<GatewayEvent>();

    let node = state.node.clone();
    let model_for_thread = model.clone();
    std::thread::spawn(move || {
        let mut on_delta = |d: &str| {
            let _ = tx.send(GatewayEvent::Delta(d.to_string()));
        };
        match node.complete(
            &model_for_thread,
            req.messages,
            req.max_tokens,
            req.temperature,
            &mut on_delta,
        ) {
            Ok(_summary) => {
                let _ = tx.send(GatewayEvent::Done);
            }
            Err(e) => {
                let _ = tx.send(GatewayEvent::Error(e.to_string()));
            }
        }
    });

    let id_s = id.clone();
    let model_s = model.clone();
    let body = UnboundedReceiverStream::new(rx).map(move |ev| {
        let data = match ev {
            GatewayEvent::Delta(t) => chunk_json(&id_s, &model_s, Some(&t), None),
            GatewayEvent::Done => chunk_json(&id_s, &model_s, None, Some("stop")),
            GatewayEvent::Error(m) => error_json(&id_s, &model_s, &m),
        };
        Ok::<Event, std::convert::Infallible>(Event::default().data(data))
    });
    // OpenAI clients expect a trailing `data: [DONE]`.
    let done = tokio_stream::once(Ok(Event::default().data("[DONE]")));
    Sse::new(body.chain(done))
}

/// The gateway router over a started swarm node.
pub fn router(net: NetworkHandle) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(AppState { node: Arc::new(ConsumerNode::new(net)) })
}

/// Run the gateway, blocking. Builds its own multi-thread tokio runtime and serves until
/// the process exits. `bind` is e.g. `"127.0.0.1:8080"`.
pub fn serve_http(net: NetworkHandle, bind: &str) -> std::io::Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(bind).await?;
        axum::serve(listener, router(net)).await
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_chunk_is_openai_shaped() {
        let v: serde_json::Value =
            serde_json::from_str(&chunk_json("id1", "qwen2.5:7b", Some("Hi"), None)).unwrap();
        assert_eq!(v["object"], "chat.completion.chunk");
        assert_eq!(v["model"], "qwen2.5:7b");
        assert_eq!(v["choices"][0]["delta"]["content"], "Hi");
        assert!(v["choices"][0]["finish_reason"].is_null());
    }

    #[test]
    fn done_chunk_has_empty_delta_and_stop() {
        let v: serde_json::Value =
            serde_json::from_str(&chunk_json("id1", "m", None, Some("stop"))).unwrap();
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        assert!(v["choices"][0]["delta"].as_object().unwrap().is_empty());
    }

    #[test]
    fn parses_openai_request() {
        let body = r#"{"model":"qwen2.5:7b","messages":[{"role":"user","content":"hi"}],"max_tokens":64,"stream":true}"#;
        let req: ChatRequest = serde_json::from_str(body).unwrap();
        assert_eq!(req.model, "qwen2.5:7b");
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, "user");
        assert_eq!(req.max_tokens, Some(64));
        assert_eq!(req.temperature, None); // absent → None (unknown fields like `stream` ignored)
    }
}
