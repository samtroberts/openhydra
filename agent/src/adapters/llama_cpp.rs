// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! llama.cpp (`llama-server`) engine adapter (protocol plan M3.2).
//!
//! llama.cpp's server speaks the OpenAI API too, so *serving* reuses
//! [`crate::adapters::openai::serve_chat_completions`] over `/v1/chat/completions`. The
//! reason this is a bespoke adapter rather than just pointing the OpenAI one at
//! `llama-server` is **detection**: `GET /props` exposes the loaded model's chat template
//! and GGUF path, so — unlike the bare OpenAI `/v1/models` — we can compute a real
//! protocol canonical id (`family/params/quant/template_hash`) from the filename + the
//! live template.
//!
//! Parsing is pure; HTTP is injected via [`HttpClient`](crate::adapter::HttpClient).

use serde::Deserialize;

use openhydra_protocol::model_id::{canonical_id_from_hf, parse_hf_model_name};

use crate::adapter::{
    normalize_engine_ref, AdapterError, DetectedModel, EngineAdapter, HttpClient, InferenceRequest,
    ServeOutcome,
};
use crate::adapters::openai::serve_chat_completions;

/// Default `llama-server` endpoint.
pub const DEFAULT_LLAMACPP_URL: &str = "http://127.0.0.1:8080";

// ── /props (only the fields we use; unknown fields ignored) ──

#[derive(Debug, Default, Deserialize)]
struct Props {
    /// The loaded model's chat template. Absent on very old builds → uncanonicalised.
    #[serde(default)]
    chat_template: String,
    /// On-disk path of the loaded GGUF, e.g. `/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf`.
    #[serde(default)]
    model_path: String,
}

// ── /v1/models (the id(s) a consumer addresses when serving) ──

#[derive(Debug, Default, Deserialize)]
struct ModelsResponse {
    #[serde(default)]
    data: Vec<ModelEntry>,
}

#[derive(Debug, Default, Deserialize)]
struct ModelEntry {
    #[serde(default)]
    id: String,
}

/// Does `s` look like a GGUF quant tag (`Q4_K_M`, `Q8_0`, `IQ4_XS`, `F16`, `BF16`)?
fn is_quant_token(s: &str) -> bool {
    let u = s.to_ascii_uppercase();
    let b = u.as_bytes();
    (u.starts_with('Q') && b.len() >= 2 && b[1].is_ascii_digit())
        || (u.starts_with("IQ") && b.len() >= 3 && b[2].is_ascii_digit())
        || matches!(u.as_str(), "F16" | "F32" | "BF16" | "FP16" | "FP32")
}

/// Split a GGUF path into an HF-style name + quant tag for canonicalisation. The quant is
/// the trailing `.`- or `-`-delimited token when it looks like one (`…-Instruct-Q4_K_M`
/// or `…-Instruct.Q8_0`); dotted version/family segments (`Qwen2.5`, `v0.2`) are left in
/// the name. Returns an empty quant for a name with no recognisable tag — the caller then
/// advertises the model uncanonicalised.
fn parse_gguf_path(model_path: &str) -> (String, String) {
    let base = model_path.rsplit(['/', '\\']).next().unwrap_or(model_path);
    let stem = base
        .strip_suffix(".gguf")
        .or_else(|| base.strip_suffix(".GGUF"))
        .unwrap_or(base);
    // The quant is the last dot- or dash-delimited segment, if it matches the pattern.
    if let Some((head, last)) = stem.rsplit_once('.') {
        if is_quant_token(last) {
            return (head.to_string(), last.to_string());
        }
    }
    if let Some((head, last)) = stem.rsplit_once('-') {
        if is_quant_token(last) {
            return (head.to_string(), last.to_string());
        }
    }
    (stem.to_string(), String::new())
}

/// Adapter for a local `llama-server`, generic over the injected HTTP transport.
pub struct LlamaCppAdapter<H: HttpClient> {
    base_url: String,
    http: H,
}

impl<H: HttpClient> LlamaCppAdapter<H> {
    /// New adapter against `base_url`, e.g. [`DEFAULT_LLAMACPP_URL`]. A trailing `/v1` is
    /// stripped so callers may pass either the root or the OpenAI base URL.
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

impl<H: HttpClient> EngineAdapter for LlamaCppAdapter<H> {
    fn engine_name(&self) -> &'static str {
        "llama.cpp"
    }

    fn detect_models(&self) -> Result<Vec<DetectedModel>, AdapterError> {
        // /props → the loaded model's template + GGUF path (the canonicalisation source).
        let props_json = self.http.get(&format!("{}/props", self.base_url))?;
        let props: Props =
            serde_json::from_str(&props_json).map_err(|e| AdapterError::Parse(e.to_string()))?;

        let (hf_name, quant) = parse_gguf_path(&props.model_path);
        let (family, params, _variants) = parse_hf_model_name(&hf_name);
        // `parse_hf_model_name` yields `params == "unknown"` when it finds no size token;
        // refuse to mint a misleading `…/unknown/…` id in that case.
        let canonical_id = if params == "unknown" {
            String::new()
        } else {
            match canonical_id_from_hf(&hf_name, &quant, &props.chat_template) {
                Ok(c) => format!("{}/{}/{}/{}", c.family, c.params, c.quant, c.template_hash),
                Err(_) => String::new(),
            }
        };

        // /v1/models → the id(s) a consumer addresses when serving. `llama-server` usually
        // serves one model; map each id, sharing the /props template (single-model case).
        let models_json = self.http.get(&format!("{}/v1/models", self.base_url))?;
        let models: ModelsResponse =
            serde_json::from_str(&models_json).map_err(|e| AdapterError::Parse(e.to_string()))?;
        // Advertise a clean, path-free handle — never the raw `-m` path (privacy + readability).
        // Skip an id that cleans to empty (a pathological `"/"` or bare `".gguf"`), and drop a
        // basename collision (two GGUFs sharing a name in different dirs) with a log rather than
        // silently — a single-model server never hits either, they guard the rare multi-model case.
        let mut refs: Vec<String> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for id in models.data.into_iter().map(|m| m.id) {
            if id.trim().is_empty() {
                continue;
            }
            let clean = normalize_engine_ref(&id);
            if clean.is_empty() {
                continue;
            }
            if !seen.insert(clean.clone()) {
                eprintln!(
                    "openhydra-agent: llama.cpp model '{id}' collides with already-detected '{clean}' — skipping"
                );
                continue;
            }
            refs.push(clean);
        }
        // Fall back to the GGUF basename if /v1/models reported nothing addressable.
        if refs.is_empty() && !props.model_path.is_empty() {
            let clean = normalize_engine_ref(&props.model_path);
            if !clean.is_empty() {
                refs.push(clean);
            }
        }

        Ok(refs
            .into_iter()
            .map(|engine_ref| DetectedModel {
                engine_ref,
                canonical_id: canonical_id.clone(),
                family: family.clone(),
                params: params.clone(),
                quant: quant.clone(),
                size_bytes: 0,
            })
            .collect())
    }

    fn serve_stream(
        &self,
        request: &InferenceRequest,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<ServeOutcome, AdapterError> {
        // llama-server speaks the OpenAI chat route — reuse the shared SSE serve.
        serve_chat_completions(
            &self.http,
            &format!("{}/v1/chat/completions", self.base_url),
            request,
            on_delta,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::ChatMessage;

    const QWEN_TEMPLATE: &str = "{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n<|im_start|>assistant\n";

    fn props_fixture() -> String {
        serde_json::json!({
            "model_path": "/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
            "chat_template": QWEN_TEMPLATE,
            "total_slots": 1,
        })
        .to_string()
    }

    const MODELS_FIXTURE: &str =
        r#"{"object":"list","data":[{"id":"Qwen2.5-7B-Instruct-Q4_K_M","object":"model"}]}"#;

    #[derive(Default)]
    struct MockHttp {
        props: String,
        models: String,
        stream_lines: Vec<String>,
    }

    impl HttpClient for MockHttp {
        fn get(&self, url: &str) -> Result<String, AdapterError> {
            if url.ends_with("/props") {
                Ok(self.props.clone())
            } else if url.ends_with("/v1/models") {
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

    #[test]
    fn parse_gguf_path_extracts_name_and_quant() {
        assert_eq!(
            parse_gguf_path("/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
            ("Qwen2.5-7B-Instruct".into(), "Q4_K_M".into())
        );
        // Dot-delimited quant; dotted family/version segments stay in the name.
        assert_eq!(
            parse_gguf_path("Meta-Llama-3.1-8B-Instruct.Q8_0.gguf"),
            ("Meta-Llama-3.1-8B-Instruct".into(), "Q8_0".into())
        );
        // No recognisable quant tag → empty quant.
        assert_eq!(parse_gguf_path("model.gguf"), ("model".into(), String::new()));
    }

    #[test]
    fn clean_engine_ref_strips_paths_and_extension() {
        // The kastru case: an absolute path id → basename without .gguf (no home dir / username).
        assert_eq!(
            normalize_engine_ref("/home/kastru/models/Qwen3.5-9B-UD-Q4_K_XL.gguf"),
            "Qwen3.5-9B-UD-Q4_K_XL"
        );
        // Windows-style separator + uppercase extension.
        assert_eq!(normalize_engine_ref(r"C:\models\Llama-3.1-8B.GGUF"), "Llama-3.1-8B");
        // Bare filename with extension (no directory).
        assert_eq!(normalize_engine_ref("mistral-7b-Q4_K_M.gguf"), "mistral-7b-Q4_K_M");
        // Already-clean alias → untouched (no path chars, no .gguf).
        assert_eq!(normalize_engine_ref("Qwen2.5-7B-Instruct-Q4_K_M"), "Qwen2.5-7B-Instruct-Q4_K_M");
        // Ollama-style tag → untouched (the colon is not a path separator).
        assert_eq!(normalize_engine_ref("llama3.2:1b"), "llama3.2:1b");
        // HF-style namespaced id has a slash but is NOT a path → untouched (don't drop the org).
        assert_eq!(normalize_engine_ref("Qwen/Qwen2.5-7B-Instruct"), "Qwen/Qwen2.5-7B-Instruct");
        // Home-relative path → basename.
        assert_eq!(normalize_engine_ref("~/models/phi-3-mini.gguf"), "phi-3-mini");
        // Mixed-case extension is stripped case-insensitively (parity with the JS displayModelName).
        assert_eq!(normalize_engine_ref("/models/Phi-3.GGuf"), "Phi-3");
        assert_eq!(normalize_engine_ref("model.GgUf"), "model");
        // Pathological ids that clean to empty (detect_models then skips these).
        assert_eq!(normalize_engine_ref("/"), "");
        assert_eq!(normalize_engine_ref(".gguf"), "");
    }

    #[test]
    fn detect_never_advertises_a_filesystem_path() {
        // A llama-server that reports its model id AS the launch path must not leak that path onto
        // the network — the advertised engine_ref is the clean basename.
        let http = MockHttp {
            props: serde_json::json!({
                "model_path": "/home/kastru/models/Qwen3.5-9B-UD-Q4_K_XL.gguf",
                "chat_template": QWEN_TEMPLATE,
                "total_slots": 1,
            })
            .to_string(),
            models: r#"{"object":"list","data":[{"id":"/home/kastru/models/Qwen3.5-9B-UD-Q4_K_XL.gguf"}]}"#.into(),
            ..Default::default()
        };
        let adapter = LlamaCppAdapter::new(DEFAULT_LLAMACPP_URL, http);
        let models = adapter.detect_models().unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].engine_ref, "Qwen3.5-9B-UD-Q4_K_XL");
        assert!(!models[0].engine_ref.contains('/'), "must not advertise a path");
        assert!(!models[0].engine_ref.contains("kastru"), "must not leak the username");
    }

    #[test]
    fn detect_canonicalises_from_props_template_and_gguf_name() {
        let http = MockHttp {
            props: props_fixture(),
            models: MODELS_FIXTURE.into(),
            ..Default::default()
        };
        let adapter = LlamaCppAdapter::new(DEFAULT_LLAMACPP_URL, http);
        let models = adapter.detect_models().unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].engine_ref, "Qwen2.5-7B-Instruct-Q4_K_M");
        // Real canonical id: family/params from the GGUF name, quant normalised, template
        // hashed from /props.
        assert!(
            models[0].canonical_id.starts_with("qwen2.5/7b/int4/"),
            "canonical_id = {}",
            models[0].canonical_id
        );
        assert_eq!(models[0].family, "qwen2.5");
        assert_eq!(models[0].params, "7b");
        assert_eq!(models[0].quant, "Q4_K_M");
        assert_eq!(adapter.engine_name(), "llama.cpp");
    }

    #[test]
    fn detect_uncanonicalised_without_a_template() {
        // No chat_template in /props → still detected and addressable, just no canonical id.
        let props = serde_json::json!({
            "model_path": "/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        })
        .to_string();
        let http = MockHttp { props, models: MODELS_FIXTURE.into(), ..Default::default() };
        let adapter = LlamaCppAdapter::new(DEFAULT_LLAMACPP_URL, http);
        let models = adapter.detect_models().unwrap();
        assert_eq!(models.len(), 1);
        assert!(models[0].canonical_id.is_empty());
        assert_eq!(models[0].engine_ref, "Qwen2.5-7B-Instruct-Q4_K_M");
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
        let adapter = LlamaCppAdapter::new(DEFAULT_LLAMACPP_URL, http);
        let mut out = String::new();
        let outcome = adapter
            .serve_stream(
                &InferenceRequest {
                    model_ref: "Qwen2.5-7B-Instruct-Q4_K_M".into(),
                    messages: vec![ChatMessage { role: "user".into(), content: "hi".into(), ..Default::default() }],
                    max_tokens: None,
                    temperature: None,
                    tools: Vec::new(),
                    think: None,
                },
                &mut |d| out.push_str(d),
            )
            .unwrap();
        assert_eq!(out, "hi");
        assert_eq!(outcome.tokens, 1);
        assert!(outcome.done);
    }
}
