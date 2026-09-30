// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Gateway-side client for a reference registry (`openhydra-agent registry serve`).
//!
//! When a gateway has no local reference for a provider's model, it asks the registry for a round
//! pack: never-issued texts for the declared model identity (or its name), signed by the registry.
//! The pack is verified against the **pinned** registry key before any of it is used.
//!
//! Wire (JSON over HTTPS; plain HTTP only to a loopback host):
//! `POST /v1/packs` with `Authorization: Bearer <key>` and
//! `{"identity": {"family", "params", "quant"} | null, "handle": "..."}` →
//! `{"status": "ok", "pack": base64, "signature": hex, "key": hex}` or
//! `{"status": "no_reference" | "exhausted"}`. A bad key is a 401, a rate limit a 429.

use base64::Engine as _;
use openhydra_protocol::logprob::ModelIdentity;
use openhydra_protocol::refpool::{verify_pack, RefPack};
use serde_json::{json, Value};

use crate::adapter::HttpClient;
use crate::http::ReqwestClient;

/// Why the registry didn't yield a usable pack.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    /// The registry has no reference for this model.
    NoReference,
    /// It has references for this model, but no un-issued round left.
    Exhausted,
    /// Transport failure, a refusal (401 / 429), or a reply that wasn't understood.
    Request(String),
    /// The pack failed verification against the pinned key, or didn't decode.
    Invalid(String),
}

/// A configured registry: base URL, this gateway's API key, and the registry's pinned public key.
pub struct RegistryClient {
    base: String,
    api_key: String,
    pinned: [u8; 32],
    http: ReqwestClient,
}

/// Whether `url` may be used: HTTPS anywhere, plain HTTP only to a loopback host (texts must stay
/// secret in transit until they've been used).
pub fn registry_url_allowed(url: &str) -> bool {
    if url.starts_with("https://") {
        return true;
    }
    let Some(rest) = url.strip_prefix("http://") else { return false };
    let host = rest.split(['/', '?']).next().unwrap_or("");
    let host = host.rsplit_once(':').map_or(host, |(h, p)| if p.chars().all(|c| c.is_ascii_digit()) { h } else { host });
    matches!(host, "127.0.0.1" | "localhost" | "[::1]")
}

impl RegistryClient {
    /// `pinned_hex` is the registry's Ed25519 public key (from `registry pubkey`).
    pub fn new(base: &str, api_key: &str, pinned_hex: &str) -> Result<Self, String> {
        let base = base.trim_end_matches('/').to_string();
        if !registry_url_allowed(&base) {
            return Err(format!("registry URL {base} must be https:// (plain http only for localhost)"));
        }
        let pinned: [u8; 32] = hex::decode(pinned_hex.trim())
            .map_err(|e| format!("registry public key: {e}"))?
            .try_into()
            .map_err(|_| "registry public key: expected 32 bytes (64 hex characters)".to_string())?;
        let api_key = api_key.trim().to_string();
        if api_key.is_empty() {
            return Err("registry API key is empty".into());
        }
        Ok(Self { base, api_key, pinned, http: ReqwestClient::new().map_err(|e| e.to_string())? })
    }

    /// Ask for a round pack for the declared identity (preferred) or the model's name.
    pub fn fetch_pack(&self, identity: Option<&ModelIdentity>, handle: &str) -> Result<RefPack, RegistryError> {
        let body = json!({
            "identity": identity.map(|i| json!({"family": i.family, "params": i.params, "quant": i.quant})),
            "handle": handle,
        });
        let auth = format!("Bearer {}", self.api_key);
        let reply = self
            .http
            .post_json_with_headers(&format!("{}/v1/packs", self.base), &body.to_string(), &[("authorization", auth.as_str())])
            .map_err(|e| RegistryError::Request(e.to_string()))?;
        parse_pack_reply(&reply, &self.pinned)
    }

    /// Ask for the rest of a round whose first-stage pack (`first_pack_id`) didn't decide it.
    pub fn continue_pack(&self, first_pack_id: [u8; 16]) -> Result<RefPack, RegistryError> {
        let body = json!({"continue": hex::encode(first_pack_id)});
        let auth = format!("Bearer {}", self.api_key);
        let reply = self
            .http
            .post_json_with_headers(&format!("{}/v1/packs", self.base), &body.to_string(), &[("authorization", auth.as_str())])
            .map_err(|e| RegistryError::Request(e.to_string()))?;
        let pack = parse_pack_reply(&reply, &self.pinned)?;
        Ok(pack)
    }
}

/// Decode a `/v1/packs` reply and verify the pack against `pinned`.
pub fn parse_pack_reply(reply: &str, pinned: &[u8; 32]) -> Result<RefPack, RegistryError> {
    let v: Value = serde_json::from_str(reply).map_err(|e| RegistryError::Request(format!("registry reply: {e}")))?;
    match v.get("status").and_then(Value::as_str) {
        Some("ok") => {}
        Some("no_reference") => return Err(RegistryError::NoReference),
        Some("exhausted") => return Err(RegistryError::Exhausted),
        other => return Err(RegistryError::Request(format!("registry reply status {other:?}"))),
    }
    let field = |k: &str| v.get(k).and_then(Value::as_str).ok_or_else(|| RegistryError::Invalid(format!("missing {k}")));
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(field("pack")?)
        .map_err(|e| RegistryError::Invalid(format!("pack: {e}")))?;
    let fixed = |k: &str, n: usize| -> Result<Vec<u8>, RegistryError> {
        let b = hex::decode(field(k)?).map_err(|e| RegistryError::Invalid(format!("{k}: {e}")))?;
        (b.len() == n).then_some(b).ok_or_else(|| RegistryError::Invalid(format!("{k}: wrong length")))
    };
    let sig: [u8; 64] = fixed("signature", 64)?.try_into().expect("64");
    let signer: [u8; 32] = fixed("key", 32)?.try_into().expect("32");
    verify_pack(&bytes, &sig, &signer, pinned).map_err(|e| RegistryError::Invalid(format!("{e:?}")))
}

/// Encode a signed pack as a `/v1/packs` reply (used by the registry server).
pub fn pack_reply(bytes: &[u8], signature: &[u8; 64], key: &[u8; 32]) -> Value {
    json!({
        "status": "ok",
        "pack": base64::engine::general_purpose::STANDARD.encode(bytes),
        "signature": hex::encode(signature),
        "key": hex::encode(key),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use openhydra_protocol::logprob::{ReferenceSet, ReferenceText};
    use openhydra_protocol::refpool::{sign_pack, RefPool};

    fn pack() -> RefPack {
        let pool = RefPool::open_in_memory().unwrap();
        let set = ReferenceSet {
            model_key: "qwen3.5-9b@q4_k_m".into(),
            identity: ModelIdentity::new("qwen3.5", "9b", "q4_k_m"),
            texts: (0..4)
                .map(|i| {
                    let ids: Vec<u32> = (0..20).map(|k| (i * 50 + k) as u32).collect();
                    ReferenceText { id: format!("t{i}"), logprobs: vec![-0.5; 19], token_ids: ids, band: 0.02 }
                })
                .collect(),
            cutoff: 2.24,
            min_texts: 4,
            early_stop: vec![],
        };
        pool.import(&set, &[], [1; 16], 0).unwrap();
        pool.issue_pack(set.identity.as_ref(), None, 0, "gw", [2; 16], 0).unwrap()
    }

    #[test]
    fn replies_verify_against_the_pinned_key_only() {
        let key = SigningKey::from_bytes(&[7; 32]);
        let pubkey = key.verifying_key().to_bytes();
        let p = pack();
        let (bytes, sig) = sign_pack(&p, &key);
        let reply = pack_reply(&bytes, &sig, &pubkey).to_string();
        assert_eq!(parse_pack_reply(&reply, &pubkey).unwrap(), p);
        let other = SigningKey::from_bytes(&[8; 32]).verifying_key().to_bytes();
        assert!(matches!(parse_pack_reply(&reply, &other), Err(RegistryError::Invalid(e)) if e.contains("WrongKey")));
        // A forger re-signing with its own key and claiming it is still caught by the pin.
        let forger = SigningKey::from_bytes(&[9; 32]);
        let (b2, s2) = sign_pack(&p, &forger);
        let forged = pack_reply(&b2, &s2, &forger.verifying_key().to_bytes()).to_string();
        assert!(matches!(parse_pack_reply(&forged, &pubkey), Err(RegistryError::Invalid(_))));
        assert_eq!(parse_pack_reply(r#"{"status":"exhausted"}"#, &pubkey), Err(RegistryError::Exhausted));
        assert_eq!(parse_pack_reply(r#"{"status":"no_reference"}"#, &pubkey), Err(RegistryError::NoReference));
        assert!(matches!(parse_pack_reply(r#"{"status":"ok"}"#, &pubkey), Err(RegistryError::Invalid(_))));
    }

    #[test]
    fn only_https_or_loopback_http_is_allowed() {
        assert!(registry_url_allowed("https://registry.example.org"));
        assert!(registry_url_allowed("http://127.0.0.1:7870"));
        assert!(registry_url_allowed("http://localhost:7870/"));
        assert!(!registry_url_allowed("http://10.0.0.5:7870"));
        assert!(!registry_url_allowed("http://localhost.evil.com:7870"));
        assert!(!registry_url_allowed("ftp://x"));
        assert!(RegistryClient::new("http://1.2.3.4:1", "k", &"00".repeat(32)).is_err());
        assert!(RegistryClient::new("http://127.0.0.1:1", "k", "abc").is_err());
        assert!(RegistryClient::new("http://127.0.0.1:1", " ", &"00".repeat(32)).is_err());
    }
}
