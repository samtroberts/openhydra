// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! The seed list (soft launch, SL1): the providers OpenHydra itself runs.
//!
//! `openhydra/auto` only picks a model from sources this node has reason to trust (its own
//! machine, cards, providers it has used or verified): any peer can announce any model name. A
//! brand-new user has none of those, so its first `auto` request had nothing to pick. The seed list
//! is one more such source: the public keys of OpenHydra's own providers, signed by the OpenHydra
//! registry key (the key the tokenizer index is signed with, [`PINNED_INDEX_KEY`]), under its own
//! domain so neither signature can be passed off as the other. `auto` picks a model a seed serves
//! once nothing the user chose or earned qualifies, and routing starts a seed a little above an
//! unknown provider (the user's own history overrides both).
//!
//! The list is served whole from the CDN ([`DEFAULT_SEED_URL`]) and refreshed in the background; a
//! fetched list replaces the current one only when it is newer (a higher `sequence`), so an old
//! signed copy can't be replayed. The build carries no seeds: until a signed list arrives, nothing
//! is trusted on its account.
//!
//! [`PINNED_INDEX_KEY`]: crate::tokenizer_index::PINNED_INDEX_KEY

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use base64::Engine as _;
use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};

use crate::tokenizer_index::{read_capped, refresh_delay, write_atomic};

/// Where gateways fetch the list from: the download bucket, published by
/// `.github/workflows/seed-list.yml` from `agent/data/seed-providers.json`.
pub const DEFAULT_SEED_URL: &str = "https://dl.openhydra.co/seeds/index.json";

/// How often a gateway checks for a newer list after a good fetch (a conditional request).
pub const REFRESH_EVERY: Duration = Duration::from_secs(6 * 60 * 60);

/// Signed bytes are prefixed with this, so a seed-list signature can't be passed off as anything
/// else the registry key signs (the tokenizer index, log-prob packs).
const SEED_DOMAIN: &[u8] = b"openhydra/seed-providers/v1\0";

/// The list format this build reads.
pub const SEED_FORMAT: u32 = 1;

/// The largest list accepted (signed envelope).
pub const MAX_LIST_BYTES: usize = 256 << 10;
/// The most seeds a list may name.
pub const MAX_SEEDS: usize = 256;
/// The longest label.
pub const MAX_LABEL_CHARS: usize = 64;

/// One provider OpenHydra runs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Seed {
    /// The provider's Ed25519 public key, 64 lower-case hex digits: what its adverts are signed
    /// with and what routing keys its standing on.
    pub public_key: String,
    /// What it is, for logs and the app ("OpenHydra seed, us-east-1").
    pub label: String,
}

/// The list's signed content.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeedList {
    pub format: u32,
    /// Raised on every publish; a gateway never goes back to a lower one.
    pub sequence: u64,
    pub issued_ms: u64,
    pub seeds: Vec<Seed>,
}

/// The published form: the list bytes, their signature and the signing key.
#[derive(Debug, Serialize, Deserialize)]
struct Envelope {
    list: String,
    signature: String,
    key: String,
}

fn is_hex(s: &str, n: usize) -> bool {
    s.len() == n && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

impl SeedList {
    /// No seeds (what a build starts from).
    pub fn empty() -> Self {
        Self { format: SEED_FORMAT, sequence: 0, issued_ms: 0, seeds: Vec::new() }
    }

    /// Checks every seed, and that no key is listed twice.
    pub fn validate(&self) -> Result<(), String> {
        if self.format != SEED_FORMAT {
            return Err(format!("seed list format {} (this build reads {SEED_FORMAT})", self.format));
        }
        if self.seeds.len() > MAX_SEEDS {
            return Err(format!("more than {MAX_SEEDS} seeds"));
        }
        let mut seen = HashSet::new();
        for s in &self.seeds {
            if !is_hex(&s.public_key, 64) {
                return Err(format!("seed key {:?} isn't 64 lower-case hex digits", s.public_key));
            }
            if VerifyingKey::from_bytes(&hex::decode(&s.public_key).ok().and_then(|b| b.try_into().ok()).unwrap_or([0; 32])).is_err() {
                return Err(format!("seed key {} isn't an Ed25519 key", s.public_key));
            }
            if s.label.chars().count() > MAX_LABEL_CHARS || s.label.chars().any(char::is_control) {
                return Err(format!("seed {} has a label longer than {MAX_LABEL_CHARS} characters or with control characters", s.public_key));
            }
            if !seen.insert(s.public_key.as_str()) {
                return Err(format!("seed {} listed twice", s.public_key));
            }
        }
        Ok(())
    }

    /// The seeds' public keys (lower-case hex).
    pub fn public_keys(&self) -> HashSet<String> {
        self.seeds.iter().map(|s| s.public_key.clone()).collect()
    }

    /// Add `seed` (registry tooling), replacing the label of one already listed.
    pub fn upsert(&mut self, seed: Seed) -> Result<(), String> {
        let seed = Seed { public_key: seed.public_key.trim().to_ascii_lowercase(), label: seed.label.trim().to_string() };
        match self.seeds.iter_mut().find(|s| s.public_key == seed.public_key) {
            Some(s) => s.label = seed.label,
            None => self.seeds.push(seed),
        }
        self.validate()
    }

    /// Remove the seed with this key (registry tooling). Whether one was listed.
    pub fn remove(&mut self, public_key: &str) -> bool {
        let key = public_key.trim().to_ascii_lowercase();
        let before = self.seeds.len();
        self.seeds.retain(|s| s.public_key != key);
        self.seeds.len() != before
    }
}

/// Sign `list` with the registry key: the published JSON.
pub fn sign(list: &SeedList, key: &SigningKey) -> Result<String, String> {
    list.validate()?;
    let bytes = serde_json::to_vec(list).map_err(|e| e.to_string())?;
    let mut msg = SEED_DOMAIN.to_vec();
    msg.extend_from_slice(&bytes);
    let env = Envelope {
        list: base64::engine::general_purpose::STANDARD.encode(&bytes),
        signature: hex::encode(key.sign(&msg).to_bytes()),
        key: hex::encode(key.verifying_key().to_bytes()),
    };
    serde_json::to_string_pretty(&env).map_err(|e| e.to_string())
}

/// Read a published list, checking its signature against the `pinned` key (hex) and every seed.
pub fn verify(published: &str, pinned: &str) -> Result<SeedList, String> {
    if published.len() > MAX_LIST_BYTES {
        return Err("seed list too large".into());
    }
    let env: Envelope = serde_json::from_str(published).map_err(|e| format!("seed list: {e}"))?;
    if !env.key.eq_ignore_ascii_case(pinned) {
        return Err("seed list signed by another key".into());
    }
    let key: [u8; 32] = hex::decode(pinned).ok().and_then(|b| b.try_into().ok()).ok_or("bad pinned key")?;
    let key = VerifyingKey::from_bytes(&key).map_err(|_| "bad pinned key")?;
    let sig: [u8; 64] = hex::decode(&env.signature).ok().and_then(|b| b.try_into().ok()).ok_or("bad signature")?;
    let bytes = base64::engine::general_purpose::STANDARD.decode(&env.list).map_err(|e| format!("seed list: {e}"))?;
    let mut msg = SEED_DOMAIN.to_vec();
    msg.extend_from_slice(&bytes);
    key.verify_strict(&msg, &Signature::from_bytes(&sig)).map_err(|_| "bad signature")?;
    let list: SeedList = serde_json::from_slice(&bytes).map_err(|e| format!("seed list: {e}"))?;
    list.validate()?;
    Ok(list)
}

/// The gateway's current seed list: the newest of none, the last one fetched (kept in `cache`), and
/// what [`SeedStore::refresh`] fetches.
pub struct SeedStore {
    current: RwLock<Arc<SeedList>>,
    pinned: String,
    url: String,
    /// The last fetched list as published (`~/.openhydra/seed-providers.json`), its ETag beside it.
    cache: Option<PathBuf>,
    etag: RwLock<Option<String>>,
}

impl SeedStore {
    /// Start from the cached copy, if it verifies.
    pub fn new(pinned: &str, url: &str, cache: Option<PathBuf>) -> Self {
        let mut current = SeedList::empty();
        let mut etag = None;
        if let Some(path) = &cache {
            if let Ok(text) = std::fs::read_to_string(path) {
                match verify(&text, pinned) {
                    Ok(cached) => {
                        current = cached;
                        etag = std::fs::read_to_string(path.with_extension("etag")).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
                    }
                    Err(e) => tracing::warn!(error = %e, path = %path.display(), "ignoring the cached seed list"),
                }
            }
        }
        Self { current: RwLock::new(Arc::new(current)), pinned: pinned.to_string(), url: url.to_string(), cache, etag: RwLock::new(etag) }
    }

    /// A store holding `list` as is (tests, and a private deployment's empty list).
    pub fn fixed(list: SeedList) -> Self {
        Self { current: RwLock::new(Arc::new(list)), pinned: String::new(), url: String::new(), cache: None, etag: RwLock::new(None) }
    }

    pub fn current(&self) -> Arc<SeedList> {
        Arc::clone(&self.current.read().unwrap_or_else(|e| e.into_inner()))
    }

    /// Adopt a published list if it verifies and is newer than the current one. Whether it was.
    pub fn offer(&self, published: &str, etag: Option<String>) -> Result<bool, String> {
        let list = verify(published, &self.pinned)?;
        let mut cur = self.current.write().unwrap_or_else(|e| e.into_inner());
        if list.sequence <= cur.sequence {
            if list.sequence == cur.sequence && etag.is_some() {
                *self.etag.write().unwrap_or_else(|e| e.into_inner()) = etag;
            }
            return Ok(false);
        }
        if let Some(path) = &self.cache {
            if let Err(e) = write_atomic(path, published.as_bytes()) {
                tracing::warn!(error = %e, "couldn't keep the seed list");
            } else {
                let _ = write_atomic(&path.with_extension("etag"), etag.as_deref().unwrap_or("").as_bytes());
            }
        }
        *self.etag.write().unwrap_or_else(|e| e.into_inner()) = etag;
        tracing::info!(sequence = list.sequence, seeds = list.seeds.len(), "seed list updated");
        *cur = Arc::new(list);
        Ok(true)
    }

    /// Fetch the list (conditionally, by ETag) and adopt it if newer.
    pub fn refresh(&self, client: &reqwest::blocking::Client) -> Result<bool, String> {
        if !crate::registry_client::registry_url_allowed(&self.url) {
            return Err(format!("not fetching the seed list over plain http: {}", self.url));
        }
        let mut req = client.get(&self.url);
        if let Some(tag) = self.etag.read().unwrap_or_else(|e| e.into_inner()).clone() {
            req = req.header(reqwest::header::IF_NONE_MATCH, tag);
        }
        let resp = req.send().map_err(|e| e.to_string())?;
        if resp.status() == reqwest::StatusCode::NOT_MODIFIED {
            return Ok(false);
        }
        if !resp.status().is_success() {
            return Err(format!("seed list: HTTP {}", resp.status()));
        }
        let etag = resp.headers().get(reqwest::header::ETAG).and_then(|v| v.to_str().ok()).map(str::to_string);
        let body = read_capped(resp, MAX_LIST_BYTES as u64)?;
        let text = String::from_utf8(body).map_err(|_| "seed list isn't UTF-8")?;
        self.offer(&text, etag)
    }

    /// Refresh now and then every [`REFRESH_EVERY`], on a background thread; after a failure,
    /// sooner (the tokenizer index's back-off), warning at most once a day while it keeps failing.
    pub fn spawn_refresher(self: &Arc<Self>, client: reqwest::blocking::Client) {
        let store = Arc::clone(self);
        std::thread::Builder::new()
            .name("seed-list".into())
            .spawn(move || {
                let mut failures = 0u32;
                let mut last_warned: Option<std::time::Instant> = None;
                loop {
                    match store.refresh(&client) {
                        Ok(_) => failures = 0,
                        Err(e) => {
                            failures = failures.saturating_add(1);
                            if last_warned.is_none_or(|t| t.elapsed() >= Duration::from_secs(24 * 60 * 60)) {
                                last_warned = Some(std::time::Instant::now());
                                tracing::warn!(error = %e, url = %store.url, sequence = store.current().sequence,
                                    "couldn't refresh the seed list; using the one this gateway has");
                            } else {
                                tracing::debug!(error = %e, "seed list not refreshed");
                            }
                        }
                    }
                    std::thread::sleep(if failures == 0 { REFRESH_EVERY } else { refresh_delay(failures) });
                }
            })
            .ok();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> SigningKey {
        SigningKey::from_bytes(&[7; 32])
    }

    fn pinned() -> String {
        hex::encode(key().verifying_key().to_bytes())
    }

    fn seed(n: u8, label: &str) -> Seed {
        Seed { public_key: hex::encode(SigningKey::from_bytes(&[n; 32]).verifying_key().to_bytes()), label: label.into() }
    }

    fn list(seq: u64, seeds: Vec<Seed>) -> SeedList {
        SeedList { format: SEED_FORMAT, sequence: seq, issued_ms: 1, seeds }
    }

    #[test]
    fn only_the_pinned_key_and_untampered_bytes_verify() {
        let l = list(2, vec![seed(1, "OpenHydra seed, us-east-1")]);
        let published = sign(&l, &key()).unwrap();
        assert_eq!(verify(&published, &pinned()).unwrap(), l);
        // Another key, even with a valid signature of its own, is refused.
        let theirs = sign(&l, &SigningKey::from_bytes(&[8; 32])).unwrap();
        assert!(verify(&theirs, &pinned()).is_err());
        let mut forged: serde_json::Value = serde_json::from_str(&theirs).unwrap();
        forged["key"] = pinned().into();
        assert!(verify(&forged.to_string(), &pinned()).is_err());
        // One changed byte in the signed list.
        let mut env: serde_json::Value = serde_json::from_str(&published).unwrap();
        let mut bytes = base64::engine::general_purpose::STANDARD.decode(env["list"].as_str().unwrap()).unwrap();
        let at = bytes.iter().position(|b| *b == b'u').unwrap();
        bytes[at] = b'v';
        env["list"] = base64::engine::general_purpose::STANDARD.encode(&bytes).into();
        assert!(verify(&env.to_string(), &pinned()).is_err());
        assert!(verify(&"x".repeat(MAX_LIST_BYTES + 1), &pinned()).is_err());
    }

    #[test]
    fn a_tokenizer_index_signature_is_not_a_seed_list_signature() {
        // Same key, other domain: the bytes of one can't be presented as the other.
        let l = list(1, vec![seed(1, "a")]);
        let bytes = serde_json::to_vec(&l).unwrap();
        let mut msg = b"openhydra/tokenizer-index/v1\0".to_vec();
        msg.extend_from_slice(&bytes);
        let env = serde_json::json!({
            "list": base64::engine::general_purpose::STANDARD.encode(&bytes),
            "signature": hex::encode(key().sign(&msg).to_bytes()),
            "key": pinned(),
        });
        assert_eq!(verify(&env.to_string(), &pinned()).unwrap_err(), "bad signature");
    }

    #[test]
    fn seeds_are_checked() {
        assert!(list(1, vec![seed(1, "a"), seed(2, "b")]).validate().is_ok());
        assert!(list(1, vec![seed(1, "a"), seed(1, "b")]).validate().is_err(), "twice");
        assert!(list(1, vec![Seed { public_key: "AB".repeat(32), label: "a".into() }]).validate().is_err(), "upper case");
        assert!(list(1, vec![Seed { public_key: "ab".into(), label: "a".into() }]).validate().is_err(), "short");
        assert!(list(1, vec![seed(1, &"x".repeat(MAX_LABEL_CHARS + 1))]).validate().is_err());
        assert!(list(1, vec![seed(1, "a\nb")]).validate().is_err());
        assert!(SeedList { format: 2, ..list(1, vec![]) }.validate().is_err());
        let mut l = list(1, vec![]);
        let s = seed(3, "first");
        l.upsert(Seed { public_key: s.public_key.to_uppercase(), label: " first ".into() }).unwrap();
        l.upsert(Seed { public_key: s.public_key.clone(), label: "renamed".into() }).unwrap();
        assert_eq!(l.seeds, vec![Seed { public_key: s.public_key.clone(), label: "renamed".into() }]);
        assert!(l.remove(&s.public_key.to_uppercase()));
        assert!(!l.remove(&s.public_key));
    }

    #[test]
    fn the_store_only_moves_forward_and_keeps_what_it_fetched() {
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join("seed-providers.json");
        let store = SeedStore::new(&pinned(), "https://example.invalid/seeds.json", Some(cache.clone()));
        assert!(store.current().seeds.is_empty(), "a build trusts no seed of its own");
        assert!(store.offer(&sign(&list(2, vec![seed(1, "a")]), &key()).unwrap(), Some("\"e2\"".into())).unwrap());
        assert!(!store.offer(&sign(&list(1, vec![seed(2, "old")]), &key()).unwrap(), None).unwrap(), "an older list is not taken");
        assert!(!store.offer(&sign(&list(2, vec![seed(2, "same seq")]), &key()).unwrap(), None).unwrap());
        assert!(store.offer(&sign(&list(3, vec![]), &SigningKey::from_bytes(&[9; 32])).unwrap(), None).is_err());
        assert_eq!(store.current().public_keys(), [seed(1, "a").public_key].into_iter().collect());
        // A restart starts from the kept copy.
        let again = SeedStore::new(&pinned(), "https://example.invalid/seeds.json", Some(cache));
        assert_eq!(again.current().sequence, 2);
        let plain = SeedStore::new(&pinned(), "http://example.com/seeds.json", None);
        assert!(plain.refresh(&reqwest::blocking::Client::new()).unwrap_err().contains("plain http"));
    }
}
