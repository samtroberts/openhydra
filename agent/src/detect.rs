// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Engine auto-detection (`--engine-kind auto`) and the multi-engine adapter.
//!
//! [`detect_engines`] probes the standard local ports for the engines OpenHydra speaks —
//! each with the engine's *own* adapter, so the readiness check is exactly the fingerprint
//! that adapter uses (`/api/tags`, `/props`, `/v1/models`) and a hit already yields the
//! model list. Probes run concurrently with a short connect timeout so a dead port can't
//! stall startup.
//!
//! [`MultiAdapter`] is the payoff: it *is* an [`EngineAdapter`], so it drops straight into
//! the generic provider with no changes. Its `detect_models` re-runs detection and returns
//! the **union** of every live engine's models (so one node advertises all of them); its
//! `serve_stream` routes each request to the adapter that owns the requested model. Because
//! detection re-runs on every provider re-announce tick, an engine or model started *after*
//! the agent is picked up within one interval — no restart.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use crate::adapter::{AdapterError, DetectedModel, EngineAdapter, InferenceRequest, ServeOutcome};
use crate::adapters::comfyui::{ComfyUiAdapter, DEFAULT_COMFYUI_URL};
use crate::adapters::exo::{ExoAdapter, DEFAULT_EXO_URL};
use crate::adapters::llama_cpp::{LlamaCppAdapter, DEFAULT_LLAMACPP_URL};
use crate::adapters::ollama::{OllamaAdapter, DEFAULT_OLLAMA_URL};
use crate::adapters::openai::{OpenAiAdapter, DEFAULT_LM_STUDIO_URL, DEFAULT_VLLM_URL};
use crate::http::ReqwestClient;

/// Connect timeout for a detection probe. Localhost refuses instantly when a port is dead,
/// so this only bounds the rare case of a silently-dropping port.
const PROBE_CONNECT_TIMEOUT: Duration = Duration::from_secs(1);

/// A live engine found by [`detect_engines`]: its label, base URL, a ready-to-use adapter,
/// and the models it currently serves.
pub struct DetectedEngine {
    /// Short engine label for logs (`"ollama"`, `"lm-studio"`, `"llama.cpp"`, `"vllm"`, `"exo"`).
    pub label: &'static str,
    /// The base URL it answered on.
    pub url: &'static str,
    /// An adapter bound to `url`, reused for serving (built once, no re-detection cost).
    pub adapter: SharedAdapter,
    /// Models the engine serves right now (possibly empty — the server is up but idle).
    pub models: Vec<DetectedModel>,
}

/// An engine adapter shared between the route table and any in-flight serve (an `Arc` so a
/// serve that's running when detection swaps the table keeps its adapter alive).
pub type SharedAdapter = Arc<dyn EngineAdapter + Send + Sync>;

/// One entry in the probe table: how to reach and build one engine's adapter.
struct ProbeSpec {
    label: &'static str,
    url: &'static str,
    build: fn(&str) -> Result<SharedAdapter, AdapterError>,
}

fn probe_specs() -> Vec<ProbeSpec> {
    // ollama (/api/tags) and llama.cpp (/props) have unique fingerprints; the rest share
    // /v1/models and are distinguished by their (distinct) default ports.
    vec![
        ProbeSpec { label: "ollama", url: DEFAULT_OLLAMA_URL, build: build_ollama },
        ProbeSpec { label: "llama.cpp", url: DEFAULT_LLAMACPP_URL, build: build_llamacpp },
        ProbeSpec { label: "lm-studio", url: DEFAULT_LM_STUDIO_URL, build: build_lmstudio },
        ProbeSpec { label: "vllm", url: DEFAULT_VLLM_URL, build: build_vllm },
        ProbeSpec { label: "exo", url: DEFAULT_EXO_URL, build: build_exo },
        ProbeSpec { label: "comfyui", url: DEFAULT_COMFYUI_URL, build: build_comfyui },
    ]
}

fn probe_client() -> Result<ReqwestClient, AdapterError> {
    ReqwestClient::with_connect_timeout(PROBE_CONNECT_TIMEOUT)
}

fn build_ollama(u: &str) -> Result<SharedAdapter, AdapterError> {
    Ok(Arc::new(OllamaAdapter::new(u, probe_client()?)))
}
fn build_llamacpp(u: &str) -> Result<SharedAdapter, AdapterError> {
    Ok(Arc::new(LlamaCppAdapter::new(u, probe_client()?)))
}
fn build_lmstudio(u: &str) -> Result<SharedAdapter, AdapterError> {
    Ok(Arc::new(OpenAiAdapter::new(u, "lm-studio", probe_client()?)))
}
fn build_vllm(u: &str) -> Result<SharedAdapter, AdapterError> {
    Ok(Arc::new(OpenAiAdapter::new(u, "vllm", probe_client()?)))
}
fn build_comfyui(u: &str) -> Result<SharedAdapter, AdapterError> {
    Ok(Arc::new(ComfyUiAdapter::new(u, probe_client()?)))
}
fn build_exo(u: &str) -> Result<SharedAdapter, AdapterError> {
    // Exo needs its own adapter (detects placed-and-ready models via /state, not the
    // /v1/models catalog) — see [`crate::adapters::exo`].
    Ok(Arc::new(ExoAdapter::new(u, probe_client()?)))
}

/// Probe every standard engine port concurrently and return the ones that answered, in the
/// [`probe_specs`] order. An engine is "found" when its adapter's `detect_models` succeeds
/// (server up), even if it currently serves zero models.
pub fn detect_engines() -> Vec<DetectedEngine> {
    let handles: Vec<_> = probe_specs()
        .into_iter()
        .map(|spec| std::thread::spawn(move || probe_one(spec)))
        .collect();
    handles.into_iter().filter_map(|h| h.join().ok().flatten()).collect()
}

fn probe_one(spec: ProbeSpec) -> Option<DetectedEngine> {
    let adapter = (spec.build)(spec.url).ok()?;
    match adapter.detect_models() {
        Ok(models) => Some(DetectedEngine { label: spec.label, url: spec.url, adapter, models }),
        Err(_) => None, // connection refused / wrong service on the port → not this engine
    }
}

/// The routing state built from a set of detected engines: which adapter serves each model
/// id, and the de-duplicated union of models to announce.
struct RouteTable {
    routes: HashMap<String, SharedAdapter>,
    models: Vec<DetectedModel>,
}

/// Build the route table from detected engines. The union is de-duplicated by model id
/// (`engine_ref`); on a collision the **first** engine in detection order wins (so the more
/// canonical engines — ollama, llama.cpp — take precedence over the generic OpenAI ones).
fn build_routes(engines: &[DetectedEngine]) -> RouteTable {
    let mut routes: HashMap<String, SharedAdapter> = HashMap::new();
    let mut models = Vec::new();
    for engine in engines {
        for model in &engine.models {
            if routes.contains_key(&model.engine_ref) {
                continue; // first engine to claim this model id keeps it
            }
            routes.insert(model.engine_ref.clone(), engine.adapter.clone());
            models.push(model.clone());
        }
    }
    RouteTable { routes, models }
}

/// A composite [`EngineAdapter`] over every locally-detected engine: announces the union of
/// their models and routes each serve to the engine that owns the model. Re-detects on every
/// `detect_models` call (the provider calls it on each re-announce tick), so engines/models
/// that appear after startup are absorbed with no restart.
pub struct MultiAdapter {
    table: RwLock<RouteTable>,
}

impl MultiAdapter {
    /// An empty multi-adapter; the first `detect_models` populates it by probing.
    pub fn new() -> Self {
        Self { table: RwLock::new(RouteTable { routes: HashMap::new(), models: Vec::new() }) }
    }

    #[cfg(test)]
    fn from_engines(engines: Vec<DetectedEngine>) -> Self {
        Self { table: RwLock::new(build_routes(&engines)) }
    }
}

impl Default for MultiAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineAdapter for MultiAdapter {
    fn engine_name(&self) -> &'static str {
        "auto"
    }

    fn detect_models(&self) -> Result<Vec<DetectedModel>, AdapterError> {
        let table = build_routes(&detect_engines());
        let models = table.models.clone();
        // F-C6: recover from a poisoned lock instead of silently skipping the
        // update — otherwise we'd return (and announce) `models` while the route
        // table the serve path reads stays stale, a hard-to-debug mismatch.
        *self.table.write().unwrap_or_else(|e| e.into_inner()) = table;
        Ok(models)
    }

    fn serve_stream(
        &self,
        request: &InferenceRequest,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<ServeOutcome, AdapterError> {
        // Resolve the owning adapter under a short read lock (clone the Arc, then release),
        // so a concurrent re-detection swapping the table can't unmap an in-flight serve.
        let adapter =
            self.table.read().ok().and_then(|t| t.routes.get(&request.model_ref).cloned());
        match adapter {
            Some(adapter) => adapter.serve_stream(request, on_delta),
            None => Err(AdapterError::Http(format!(
                "no local engine serves model '{}' (auto mode)",
                request.model_ref
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::{ChatMessage, EngineMetrics};

    /// A fake adapter that emits its own tag as the single delta, so a serve test can tell
    /// *which* engine handled the request.
    struct FakeAdapter {
        tag: &'static str,
    }
    impl EngineAdapter for FakeAdapter {
        fn engine_name(&self) -> &'static str {
            self.tag
        }
        fn detect_models(&self) -> Result<Vec<DetectedModel>, AdapterError> {
            Ok(vec![])
        }
        fn serve_stream(
            &self,
            _request: &InferenceRequest,
            on_delta: &mut dyn FnMut(&str),
        ) -> Result<ServeOutcome, AdapterError> {
            on_delta(self.tag);
            Ok(ServeOutcome { tokens: 1, done: true, engine: EngineMetrics::default(), tool_calls: Vec::new() })
        }
    }

    fn model(engine_ref: &str) -> DetectedModel {
        DetectedModel {
            engine_ref: engine_ref.into(),
            canonical_id: String::new(),
            family: String::new(),
            params: String::new(),
            quant: String::new(),
            size_bytes: 0,
        }
    }

    fn engine(label: &'static str, tag: &'static str, refs: &[&str]) -> DetectedEngine {
        DetectedEngine {
            label,
            url: "http://x",
            adapter: Arc::new(FakeAdapter { tag }),
            models: refs.iter().map(|r| model(r)).collect(),
        }
    }

    fn infer(model_ref: &str) -> InferenceRequest {
        InferenceRequest {
            model_ref: model_ref.into(),
            messages: vec![ChatMessage { role: "user".into(), content: "hi".into(), ..Default::default() }],
            max_tokens: None,
            temperature: None,
            tools: Vec::new(),
            think: None,
        }
    }

    #[test]
    fn build_routes_unions_and_first_engine_wins_on_duplicate() {
        let engines = vec![
            engine("ollama", "A", &["m1", "shared"]),
            engine("lm-studio", "B", &["m2", "shared"]),
        ];
        let table = build_routes(&engines);
        let mut ids: Vec<_> = table.models.iter().map(|m| m.engine_ref.clone()).collect();
        ids.sort();
        assert_eq!(ids, vec!["m1", "m2", "shared"]); // union, "shared" kept once
        assert_eq!(table.routes.len(), 3);
    }

    #[test]
    fn serve_routes_to_the_owning_engine() {
        let multi = MultiAdapter::from_engines(vec![
            engine("ollama", "A", &["m1"]),
            engine("lm-studio", "B", &["m2"]),
        ]);
        let mut served = String::new();
        multi.serve_stream(&infer("m2"), &mut |d| served.push_str(d)).unwrap();
        assert_eq!(served, "B"); // routed to the second engine's adapter
    }

    #[test]
    fn duplicate_model_id_routes_to_the_first_engine() {
        let multi = MultiAdapter::from_engines(vec![
            engine("ollama", "A", &["shared"]),
            engine("lm-studio", "B", &["shared"]),
        ]);
        let mut served = String::new();
        multi.serve_stream(&infer("shared"), &mut |d| served.push_str(d)).unwrap();
        assert_eq!(served, "A");
    }

    #[test]
    fn unknown_model_errors() {
        let multi = MultiAdapter::from_engines(vec![engine("ollama", "A", &["m1"])]);
        let err = multi.serve_stream(&infer("nope"), &mut |_| {}).unwrap_err();
        assert!(matches!(err, AdapterError::Http(_)));
    }

    #[test]
    fn empty_multi_adapter_serves_nothing() {
        let multi = MultiAdapter::new();
        assert!(multi.serve_stream(&infer("x"), &mut |_| {}).is_err());
    }
}
