// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! OpenAI-compatible embeddings BYOK adapter (#34).
//!
//! Embeddings are non-streaming (a `POST` returning vectors), so they implement
//! [`EmbeddingAdapter`] rather than the chat [`EngineAdapter`](crate::adapter::EngineAdapter).
//! One adapter covers the whole OpenAI-shaped `/v1/embeddings` family: OpenAI itself, Gemini's
//! OpenAI-compatibility endpoint, Voyage, and any local engine that serves the route — so a
//! single backend type is broadly useful. Auth is `Authorization: Bearer <key>`.
//!
//! Parsing is pure; HTTP + auth are injected via [`HttpClient`].

use serde::Deserialize;

use crate::adapter::{AdapterError, EmbeddingAdapter, EmbeddingResponse, HttpClient};

/// OpenAI embeddings endpoint root (the public hosted API).
pub const DEFAULT_OPENAI_EMBEDDINGS_URL: &str = "https://api.openai.com";

#[derive(Debug, Default, Deserialize)]
struct EmbeddingsResponse {
    #[serde(default)]
    data: Vec<EmbeddingData>,
    #[serde(default)]
    usage: Option<Usage>,
    #[serde(default)]
    error: Option<serde_json::Value>,
}
#[derive(Debug, Default, Deserialize)]
struct EmbeddingData {
    #[serde(default)]
    embedding: Vec<f32>,
    #[serde(default)]
    index: usize,
}
#[derive(Debug, Default, Deserialize)]
struct Usage {
    #[serde(default)]
    prompt_tokens: u64,
}

/// Build the `/v1/embeddings` request body.
fn build_body(model: &str, inputs: &[String]) -> String {
    serde_json::json!({ "model": model, "input": inputs }).to_string()
}

/// Parse a response into vectors ordered by the response's `index` field (some servers do not
/// return them in input order), plus the prompt-token count.
fn parse_response(json: &str) -> Result<EmbeddingResponse, AdapterError> {
    let resp: EmbeddingsResponse =
        serde_json::from_str(json).map_err(|e| AdapterError::Parse(e.to_string()))?;
    if let Some(err) = &resp.error {
        return Err(AdapterError::Http(format!("embeddings: {err}")));
    }
    let mut data = resp.data;
    data.sort_by_key(|d| d.index);
    Ok(EmbeddingResponse {
        vectors: data.into_iter().map(|d| d.embedding).collect(),
        prompt_tokens: resp.usage.map(|u| u.prompt_tokens).unwrap_or(0),
    })
}

/// OpenAI-compatible embeddings adapter, generic over the injected HTTP transport.
pub struct OpenAiEmbeddingAdapter<H: HttpClient> {
    base_url: String,
    api_key: String,
    http: H,
}

impl<H: HttpClient> OpenAiEmbeddingAdapter<H> {
    /// New adapter against `base_url` (root or `…/v1`, both normalise) with the operator's
    /// `api_key`.
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>, http: H) -> Self {
        let root = base_url
            .into()
            .trim_end_matches('/')
            .trim_end_matches("/v1")
            .trim_end_matches('/')
            .to_string();
        Self { base_url: root, api_key: api_key.into(), http }
    }
}

impl<H: HttpClient> EmbeddingAdapter for OpenAiEmbeddingAdapter<H> {
    fn embed(&self, model: &str, inputs: &[String]) -> Result<EmbeddingResponse, AdapterError> {
        let body = build_body(model, inputs);
        let auth = format!("Bearer {}", self.api_key);
        let json = self.http.post_json_with_headers(
            &format!("{}/v1/embeddings", self.base_url),
            &body,
            &[("authorization", auth.as_str())],
        )?;
        parse_response(&json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct MockHttp {
        response: String,
    }
    impl HttpClient for MockHttp {
        fn get(&self, _url: &str) -> Result<String, AdapterError> {
            Err(AdapterError::Http("unused".into()))
        }
        fn post_json(&self, _url: &str, _body: &str) -> Result<String, AdapterError> {
            Ok(self.response.clone())
        }
        fn post_stream(
            &self,
            _url: &str,
            _body: &str,
        ) -> Result<Box<dyn Iterator<Item = Result<String, AdapterError>>>, AdapterError> {
            Err(AdapterError::Http("unused".into()))
        }
    }

    #[test]
    fn build_body_carries_model_and_inputs() {
        let body = build_body("text-embedding-3-small", &["a".into(), "b".into()]);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["model"], "text-embedding-3-small");
        assert_eq!(v["input"], serde_json::json!(["a", "b"]));
    }

    #[test]
    fn parses_vectors_in_index_order_with_usage() {
        // Returned out of order (index 1 before 0); must be reordered to input order.
        let resp = r#"{
            "object":"list",
            "data":[
                {"object":"embedding","index":1,"embedding":[0.3,0.4]},
                {"object":"embedding","index":0,"embedding":[0.1,0.2]}
            ],
            "model":"text-embedding-3-small",
            "usage":{"prompt_tokens":7,"total_tokens":7}
        }"#;
        let parsed = parse_response(resp).unwrap();
        assert_eq!(parsed.vectors, vec![vec![0.1, 0.2], vec![0.3, 0.4]]);
        assert_eq!(parsed.prompt_tokens, 7);
    }

    #[test]
    fn embed_round_trips_through_the_adapter() {
        let mock = MockHttp {
            response: r#"{"data":[{"index":0,"embedding":[1.0,2.0,3.0]}],"usage":{"prompt_tokens":3}}"#.into(),
        };
        let a = OpenAiEmbeddingAdapter::new(DEFAULT_OPENAI_EMBEDDINGS_URL, "sk-x", mock);
        let out = a.embed("text-embedding-3-small", &["hello".into()]).unwrap();
        assert_eq!(out.vectors, vec![vec![1.0, 2.0, 3.0]]);
        assert_eq!(out.prompt_tokens, 3);
    }

    #[test]
    fn error_response_surfaces() {
        assert!(matches!(
            parse_response(r#"{"error":{"message":"bad key","type":"auth"}}"#),
            Err(AdapterError::Http(_))
        ));
    }
}
