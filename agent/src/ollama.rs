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

use crate::adapter::{AdapterError, DetectedModel, EngineAdapter, HttpClient};

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

    /// Inject canned `/api/tags` + `/api/show` bodies (no network).
    struct MockHttp {
        tags: String,
        show: String,
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
    }

    fn adapter(tags: &str, show: &str) -> OllamaAdapter<MockHttp> {
        OllamaAdapter::new(
            DEFAULT_OLLAMA_URL,
            MockHttp { tags: tags.into(), show: show.into() },
        )
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
}
