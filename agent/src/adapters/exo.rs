// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Exo engine adapter (multi-node MLX inference cluster).
//!
//! Like [`llama_cpp`](crate::adapters::llama_cpp), Exo *serves* over the OpenAI route, so
//! [`serve_stream`](ExoAdapter::serve_stream) reuses
//! [`crate::adapters::openai::serve_chat_completions`]. The reason it's a bespoke adapter is
//! **detection**: Exo's `GET /v1/models` returns its whole *downloadable catalog* (dozens of
//! models it could run), not what it can actually serve — a consumer routed to an unplaced
//! model gets a `404 "No instance found"`. The truth is `GET /state`: a model is serveable
//! only when it has a **placed instance** whose assigned **runners are all `RunnerReady`**.
//! So detection reads `/state` and announces exactly those, never the catalog.
//!
//! Fails *closed*: if `/state` is missing/unparseable or readiness can't be confirmed, the
//! adapter announces nothing rather than advertising a model it can't serve.
//!
//! Parsing is pure; HTTP is injected via [`HttpClient`](crate::adapter::HttpClient).

use std::collections::{HashMap, HashSet};

use serde::Deserialize;

use crate::adapter::{
    AdapterError, DetectedModel, EngineAdapter, HttpClient, InferenceRequest, ServeOutcome,
};
use crate::adapters::openai::serve_chat_completions;

/// Default Exo head-node OpenAI/API endpoint.
pub const DEFAULT_EXO_URL: &str = "http://127.0.0.1:52415";

// ── /state (only the fields we use; unknown fields ignored) ──

#[derive(Debug, Default, Deserialize)]
struct State {
    /// `instanceId → { <variant> → instance }` (the variant key is dynamic, e.g.
    /// `MlxRingInstance`), so the inner layer is a map we iterate over.
    #[serde(default)]
    instances: HashMap<String, HashMap<String, Instance>>,
    /// `runnerId → { <state-variant> → … }`; the state variant name (`RunnerReady`,
    /// `RunnerFailed`, `RunnerIdle`, …) is what tells us readiness.
    #[serde(default)]
    runners: HashMap<String, HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Default, Deserialize)]
struct Instance {
    #[serde(rename = "shardAssignments", default)]
    shard_assignments: Option<ShardAssignments>,
}

#[derive(Debug, Default, Deserialize)]
struct ShardAssignments {
    #[serde(rename = "modelId", default)]
    model_id: String,
    /// The runners this instance is assigned to; every one must be ready for it to serve.
    #[serde(rename = "runnerToShard", default)]
    runner_to_shard: HashMap<String, serde_json::Value>,
}

/// A runner is serveable iff it is **up** — either `RunnerReady` (loaded, idle) or
/// `RunnerRunning` (loaded, actively serving a request). Both states can take work; only down
/// states (`RunnerFailed`, `RunnerShuttingDown`, …) are excluded.
///
/// Counting *only* `RunnerReady` was a bug: Exo reports `RunnerRunning` while a runner is
/// mid-request, so a model dropped out of detection **precisely while it was being served**.
/// Combined with the provider's periodic (120s) route-table rebuild, the model flapped in and
/// out of the route table and consumers hit `no local engine serves '<model>' (auto mode)`.
fn runner_ready(
    runners: &HashMap<String, HashMap<String, serde_json::Value>>,
    runner_id: &str,
) -> bool {
    runners
        .get(runner_id)
        .map(|state| state.contains_key("RunnerReady") || state.contains_key("RunnerRunning"))
        .unwrap_or(false)
}

/// The model ids Exo can actually serve: a placed instance with a non-empty runner set whose
/// **every** assigned runner is up (ready or running). De-duplicated (a model could be placed
/// twice).
fn serving_models(state: &State) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for body in state.instances.values() {
        for instance in body.values() {
            let Some(sa) = &instance.shard_assignments else { continue };
            if sa.model_id.trim().is_empty() || sa.runner_to_shard.is_empty() {
                continue; // no model, or no runners assigned → not serveable
            }
            let all_ready =
                sa.runner_to_shard.keys().all(|rid| runner_ready(&state.runners, rid));
            if all_ready && seen.insert(sa.model_id.clone()) {
                out.push(sa.model_id.clone());
            }
        }
    }
    out
}

/// Adapter for an Exo cluster head, generic over the injected HTTP transport.
pub struct ExoAdapter<H: HttpClient> {
    base_url: String,
    http: H,
}

impl<H: HttpClient> ExoAdapter<H> {
    /// New adapter against `base_url`, e.g. [`DEFAULT_EXO_URL`]. A trailing `/v1` is stripped
    /// so callers may pass either the root or the OpenAI base URL.
    pub fn new(base_url: impl Into<String>, http: H) -> Self {
        let root = base_url
            .into()
            .trim_end_matches('/')
            .trim_end_matches("/v1")
            .trim_end_matches('/')
            .to_string();
        Self { base_url: root, http }
    }
}

impl<H: HttpClient> EngineAdapter for ExoAdapter<H> {
    fn engine_name(&self) -> &'static str {
        "exo"
    }

    fn detect_models(&self) -> Result<Vec<DetectedModel>, AdapterError> {
        // /state, not /v1/models: only placed-and-ready instances are serveable.
        let json = self.http.get(&format!("{}/state", self.base_url))?;
        let state: State =
            serde_json::from_str(&json).map_err(|e| AdapterError::Parse(e.to_string()))?;
        let serving = serving_models(&state);
        // F-C8: /state parsed fine but nothing is serveable while instances exist —
        // surface it, since an empty model list otherwise looks identical to "Exo
        // is down" (the usual cause is a placed instance whose runners aren't all
        // `RunnerReady` yet — still warming up or a stuck runner).
        if serving.is_empty() && !state.instances.is_empty() {
            tracing::debug!(
                instances = state.instances.len(),
                "exo: /state has instances but none are serveable (no placed instance has all runners RunnerReady)"
            );
        }
        Ok(serving
            .into_iter()
            .map(|model_id| DetectedModel {
                engine_ref: model_id,
                // Exo/MLX exposes no chat template or quant → advertised uncanonicalised
                // (same as the generic OpenAI adapter), addressed by its engine id.
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
        // Exo serves placed instances over the OpenAI chat route — reuse the shared SSE serve.
        // #6: right after startup a serve can race Exo still *placing/warming* a model's runners
        // (detection only announces `RunnerReady` models, but readiness can lapse between the
        // announce and the request), so Exo answers `404 "No instance found"`. That's why "send
        // it again and it works". Retry that transient a few times with a short backoff so the
        // first request waits out the cold start instead of erroring. Only retry while nothing
        // has been streamed yet — a mid-stream failure can't be safely re-run.
        let url = format!("{}/v1/chat/completions", self.base_url);
        const MAX_ATTEMPTS: u32 = 4;
        const BACKOFF: std::time::Duration = std::time::Duration::from_millis(1500);
        for attempt in 0..MAX_ATTEMPTS {
            let mut emitted = false;
            let mut sink = |d: &str| {
                emitted = true;
                on_delta(d);
            };
            match serve_chat_completions(&self.http, &url, request, &mut sink) {
                Ok(o) => return Ok(o),
                Err(e) => {
                    if is_cold_start(&e) && !emitted && attempt + 1 < MAX_ATTEMPTS {
                        std::thread::sleep(BACKOFF);
                        continue;
                    }
                    return Err(e);
                }
            }
        }
        unreachable!("loop returns on the last attempt")
    }
}

/// #6: does this error look like Exo's cold-start "model not placed/ready yet" (a transient we
/// should retry), versus a genuine failure (surface immediately)? Matches Exo's `404 "No
/// instance found"` and the readiness-race phrasings, and nothing that would hide a real bug.
fn is_cold_start(e: &AdapterError) -> bool {
    let m = e.to_string().to_ascii_lowercase();
    m.contains("no instance")
        || m.contains("404")
        || m.contains("not found")
        || m.contains("not ready")
        || m.contains("placing")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::ChatMessage;

    /// A `/state` fixture mirroring the real Exo shape: one placed `MlxRingInstance` for
    /// `model_id`, assigned to `runners` (each `(id, ready)`), alongside the catalog-only
    /// noise Exo also reports (which detection must ignore).
    fn state_fixture(model_id: &str, runners: &[(&str, bool)]) -> String {
        let runner_to_shard: serde_json::Map<String, serde_json::Value> = runners
            .iter()
            .map(|(id, _)| (id.to_string(), serde_json::json!({ "PipelineShardMetadata": {} })))
            .collect();
        let runners_map: serde_json::Map<String, serde_json::Value> = runners
            .iter()
            .map(|(id, ready)| {
                let state = if *ready { "RunnerReady" } else { "RunnerFailed" };
                (id.to_string(), serde_json::json!({ state: {} }))
            })
            .collect();
        serde_json::json!({
            "instances": {
                "inst-1": {
                    "MlxRingInstance": {
                        "instanceId": "inst-1",
                        "shardAssignments": { "modelId": model_id, "runnerToShard": runner_to_shard },
                    }
                }
            },
            "runners": runners_map,
        })
        .to_string()
    }

    #[derive(Default)]
    struct MockHttp {
        state: String,
        stream_lines: Vec<String>,
    }

    impl HttpClient for MockHttp {
        fn get(&self, url: &str) -> Result<String, AdapterError> {
            if url.ends_with("/state") {
                Ok(self.state.clone())
            } else {
                // A regression guard: detection must NOT hit /v1/models (the catalog).
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

    fn detect(state: String) -> Vec<String> {
        let adapter = ExoAdapter::new(DEFAULT_EXO_URL, MockHttp { state, ..Default::default() });
        let mut ids: Vec<String> =
            adapter.detect_models().unwrap().into_iter().map(|m| m.engine_ref).collect();
        ids.sort();
        ids
    }

    #[test]
    fn announces_only_placed_and_ready_instances() {
        let state = state_fixture(
            "mlx-community/Llama-3.2-1B-Instruct-4bit",
            &[("r1", true), ("r2", true)],
        );
        assert_eq!(detect(state), vec!["mlx-community/Llama-3.2-1B-Instruct-4bit"]);
    }

    #[test]
    fn skips_instance_with_a_non_ready_runner() {
        // r2 failed → the instance can't serve → announce nothing (not the catalog).
        let state = state_fixture("some/model", &[("r1", true), ("r2", false)]);
        assert!(detect(state).is_empty());
    }

    #[test]
    fn announces_a_running_runner_as_serveable() {
        // A runner actively serving reports `RunnerRunning`, not `RunnerReady`. It must still
        // count as serveable — otherwise a model drops out of detection the moment it is used,
        // and the provider's route table loses it until it next goes idle.
        let state = serde_json::json!({
            "instances": {
                "inst-1": { "MlxRingInstance": { "shardAssignments": {
                    "modelId": "mlx-community/Llama-3.2-1B-Instruct-4bit",
                    "runnerToShard": { "r1": {} } } } },
            },
            "runners": { "r1": { "RunnerRunning": {} } },
        })
        .to_string();
        assert_eq!(detect(state), vec!["mlx-community/Llama-3.2-1B-Instruct-4bit"]);
    }

    #[test]
    fn skips_a_shutting_down_assigned_runner() {
        // A `RunnerShuttingDown` runner assigned to the instance is down → not serveable.
        let state = serde_json::json!({
            "instances": {
                "inst-1": { "MlxRingInstance": { "shardAssignments": {
                    "modelId": "m/x", "runnerToShard": { "r1": {} } } } },
            },
            "runners": { "r1": { "RunnerShuttingDown": {} } },
        })
        .to_string();
        assert!(detect(state).is_empty());
    }

    #[test]
    fn empty_state_announces_nothing() {
        let state = serde_json::json!({ "instances": {}, "runners": {} }).to_string();
        assert!(detect(state).is_empty());
    }

    #[test]
    fn dedupes_a_model_placed_twice() {
        // Two ready instances, one a duplicate of the other's model + one distinct → 2 unique.
        let state = serde_json::json!({
            "instances": {
                "a": { "MlxRingInstance": { "shardAssignments": {
                    "modelId": "m/one", "runnerToShard": { "r1": {} } } } },
                "b": { "MlxRingInstance": { "shardAssignments": {
                    "modelId": "m/one", "runnerToShard": { "r2": {} } } } },
                "c": { "MlxRingInstance": { "shardAssignments": {
                    "modelId": "m/two", "runnerToShard": { "r3": {} } } } },
            },
            "runners": {
                "r1": { "RunnerReady": {} },
                "r2": { "RunnerReady": {} },
                "r3": { "RunnerReady": {} },
            },
        })
        .to_string();
        assert_eq!(detect(state), vec!["m/one", "m/two"]);
    }

    #[test]
    fn serve_delegates_to_the_openai_chat_route() {
        let http = MockHttp {
            stream_lines: vec![
                r#"data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}"#.into(),
                r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1}}"#.into(),
                "data: [DONE]".into(),
            ],
            ..Default::default()
        };
        let adapter = ExoAdapter::new(DEFAULT_EXO_URL, http);
        let mut out = String::new();
        let outcome = adapter
            .serve_stream(
                &InferenceRequest {
                    model_ref: "mlx-community/Llama-3.2-1B-Instruct-4bit".into(),
                    messages: vec![ChatMessage { role: "user".into(), content: "hi".into(), ..Default::default() }],
                    max_tokens: None,
                    temperature: None,
                    tools: Vec::new(),
                },
                &mut |d| out.push_str(d),
            )
            .unwrap();
        assert_eq!(out, "hi");
        assert_eq!(outcome.tokens, 1);
        assert!(outcome.done);
        assert_eq!(adapter.engine_name(), "exo");
    }
}
