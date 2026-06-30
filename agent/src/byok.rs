// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! BYOK (bring-your-own-key) gateway passthrough config (#34).
//!
//! The operator maps specific model names to a hosted backend; a request for a mapped model
//! is served by calling that frontier API directly (Anthropic/Gemini) instead of routing over
//! the swarm. Everything else falls through to the P2P path.
//!
//! Keys are resolved per request: a caller-supplied `X-Provider-Api-Key` header wins (so a
//! user can bring their own), else the operator's configured key. A mapped model with no key
//! available is refused.

use std::collections::{HashMap, HashSet};

/// Which hosted backend a BYOK model routes to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByokProvider {
    Anthropic,
    Gemini,
}

/// The gateway's BYOK routing table + per-backend base URL and operator key.
#[derive(Debug, Clone)]
pub struct ByokConfig {
    /// Exact model name → backend. Configured explicitly by the operator.
    models: HashMap<String, ByokProvider>,
    anthropic_url: String,
    anthropic_key: Option<String>,
    gemini_url: String,
    gemini_key: Option<String>,
}

impl ByokConfig {
    /// Build from the operator's per-backend model lists, base URLs, and keys.
    pub fn new(
        anthropic_models: Vec<String>,
        anthropic_url: String,
        anthropic_key: Option<String>,
        gemini_models: Vec<String>,
        gemini_url: String,
        gemini_key: Option<String>,
    ) -> Self {
        let mut models = HashMap::new();
        for m in anthropic_models {
            models.insert(m, ByokProvider::Anthropic);
        }
        for m in gemini_models {
            models.insert(m, ByokProvider::Gemini);
        }
        Self { models, anthropic_url, anthropic_key, gemini_url, gemini_key }
    }

    /// An empty config (no BYOK models — every request goes to the swarm).
    pub fn empty() -> Self {
        Self {
            models: HashMap::new(),
            anthropic_url: String::new(),
            anthropic_key: None,
            gemini_url: String::new(),
            gemini_key: None,
        }
    }

    /// Whether any model is mapped to a BYOK backend.
    pub fn is_active(&self) -> bool {
        !self.models.is_empty()
    }

    /// The backend for `model`, or `None` if it isn't a BYOK model (→ route over the swarm).
    pub fn provider_for(&self, model: &str) -> Option<ByokProvider> {
        self.models.get(model).copied()
    }

    /// The base URL for `provider`.
    pub fn base_url(&self, provider: ByokProvider) -> &str {
        match provider {
            ByokProvider::Anthropic => &self.anthropic_url,
            ByokProvider::Gemini => &self.gemini_url,
        }
    }

    /// The operator's configured key for `provider`, if any.
    pub fn operator_key(&self, provider: ByokProvider) -> Option<&str> {
        match provider {
            ByokProvider::Anthropic => self.anthropic_key.as_deref(),
            ByokProvider::Gemini => self.gemini_key.as_deref(),
        }
    }

    /// Resolve the key for a request: a caller-supplied `caller_key` wins (bring-your-own),
    /// else the operator's configured key. `None` → the request must be refused.
    pub fn resolve_key(&self, provider: ByokProvider, caller_key: Option<&str>) -> Option<String> {
        caller_key
            .filter(|k| !k.is_empty())
            .map(str::to_string)
            .or_else(|| self.operator_key(provider).map(str::to_string))
    }
}

/// BYOK embeddings routing (#34): the model names served by a single OpenAI-compatible
/// embeddings backend (OpenAI / Gemini-OAI-compat / Voyage / local), plus its base URL and
/// operator key. Separate from the chat [`ByokConfig`] because embeddings are non-streaming
/// and a different endpoint.
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    models: HashSet<String>,
    base_url: String,
    api_key: Option<String>,
}

impl EmbeddingConfig {
    pub fn new(models: Vec<String>, base_url: String, api_key: Option<String>) -> Self {
        Self { models: models.into_iter().collect(), base_url, api_key }
    }

    pub fn empty() -> Self {
        Self { models: HashSet::new(), base_url: String::new(), api_key: None }
    }

    pub fn is_active(&self) -> bool {
        !self.models.is_empty()
    }

    /// Whether `model` is routed to the embeddings backend.
    pub fn handles(&self, model: &str) -> bool {
        self.models.contains(model)
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Caller-supplied key wins (bring-your-own), else the operator's; `None` → refuse.
    pub fn resolve_key(&self, caller_key: Option<&str>) -> Option<String> {
        caller_key
            .filter(|k| !k.is_empty())
            .map(str::to_string)
            .or_else(|| self.api_key.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> ByokConfig {
        ByokConfig::new(
            vec!["claude-3-5-sonnet".into()],
            "https://api.anthropic.com".into(),
            Some("op-anthropic".into()),
            vec!["gemini-1.5-pro".into()],
            "https://generativelanguage.googleapis.com".into(),
            None, // no operator gemini key
        )
    }

    #[test]
    fn empty_is_inactive_and_routes_nothing() {
        let c = ByokConfig::empty();
        assert!(!c.is_active());
        assert_eq!(c.provider_for("claude-3-5-sonnet"), None);
    }

    #[test]
    fn maps_models_to_backends() {
        let c = cfg();
        assert!(c.is_active());
        assert_eq!(c.provider_for("claude-3-5-sonnet"), Some(ByokProvider::Anthropic));
        assert_eq!(c.provider_for("gemini-1.5-pro"), Some(ByokProvider::Gemini));
        assert_eq!(c.provider_for("qwen2.5:7b"), None); // unmapped → swarm
    }

    #[test]
    fn resolve_key_prefers_caller_then_operator() {
        let c = cfg();
        // Caller key wins (bring-your-own).
        assert_eq!(
            c.resolve_key(ByokProvider::Anthropic, Some("caller-key")).as_deref(),
            Some("caller-key")
        );
        // Empty caller key → fall back to the operator key.
        assert_eq!(
            c.resolve_key(ByokProvider::Anthropic, Some("")).as_deref(),
            Some("op-anthropic")
        );
        // No caller key → operator key.
        assert_eq!(c.resolve_key(ByokProvider::Anthropic, None).as_deref(), Some("op-anthropic"));
        // Gemini has no operator key and no caller key → None (must refuse).
        assert_eq!(c.resolve_key(ByokProvider::Gemini, None), None);
        // …but a caller key still works.
        assert_eq!(c.resolve_key(ByokProvider::Gemini, Some("user-gemini")).as_deref(), Some("user-gemini"));
    }

    #[test]
    fn embedding_config_routes_and_resolves_keys() {
        let e = EmbeddingConfig::new(
            vec!["text-embedding-3-small".into()],
            "https://api.openai.com".into(),
            Some("op-key".into()),
        );
        assert!(e.is_active());
        assert!(e.handles("text-embedding-3-small"));
        assert!(!e.handles("qwen2.5"));
        assert_eq!(e.resolve_key(Some("caller")).as_deref(), Some("caller"));
        assert_eq!(e.resolve_key(None).as_deref(), Some("op-key"));
        assert_eq!(EmbeddingConfig::empty().resolve_key(None), None);
    }
}
