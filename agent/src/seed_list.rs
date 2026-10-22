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

/// The oldest seed list this build accepts, cached or fetched: the sequence of the list committed
/// with it (agent/data/seed-providers.json, which a test holds it equal to), so an older signed list
/// (one still naming a seed since removed, served by a compromised bucket or CDN) can't be what a
/// fresh install starts from. Signing a new list means raising this in the same commit; the
/// seed-list workflow publishes the list on the push to main, before any release tag needs it.
pub const MIN_SEQUENCE: u64 = 1;

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
/// The most models one seed may list.
pub const MAX_MODELS_PER_SEED: usize = 32;

/// One provider OpenHydra runs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Seed {
    /// The provider's Ed25519 public key, 64 lower-case hex digits: what its adverts are signed
    /// with and what routing keys its standing on.
    pub public_key: String,
    /// What it is, for logs and the app ("OpenHydra seed, us-east-1").
    pub label: String,
    /// The models it serves, as it announces them (`qwen3:8b`). A gateway asks the bootstraps for
    /// these as soon as it starts, so a new user has them in the list within seconds instead of after
    /// the next round of announcements (~30–60 s). Not a promise: only what's actually announced is
    /// used.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<String>,
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
            if s.models.len() > MAX_MODELS_PER_SEED {
                return Err(format!("seed {} lists more than {MAX_MODELS_PER_SEED} models", s.public_key));
            }
            if let Some(m) = s.models.iter().find(|m| !openhydra_protocol::model_id::is_valid_model_id(m)) {
                return Err(format!("seed {} lists an invalid model name {m:?}", s.public_key));
            }
        }
        Ok(())
    }

    /// The seeds' public keys (lower-case hex).
    pub fn public_keys(&self) -> HashSet<String> {
        self.seeds.iter().map(|s| s.public_key.clone()).collect()
    }

    /// The models the seeds list, each once, sorted.
    pub fn models(&self) -> Vec<String> {
        let set: std::collections::BTreeSet<&str> = self.seeds.iter().flat_map(|s| s.models.iter().map(String::as_str)).collect();
        set.into_iter().map(str::to_string).collect()
    }

    /// Add `seed` (registry tooling), replacing the label and models of one already listed.
    pub fn upsert(&mut self, seed: Seed) -> Result<(), String> {
        let mut models: Vec<String> = seed.models.iter().map(|m| m.trim().to_string()).filter(|m| !m.is_empty()).collect();
        models.sort();
        models.dedup();
        let seed = Seed { public_key: seed.public_key.trim().to_ascii_lowercase(), label: seed.label.trim().to_string(), models };
        match self.seeds.iter_mut().find(|s| s.public_key == seed.public_key) {
            Some(s) => *s = seed,
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

/// How long to wait before the next fetch after `failures` failed ones in a row (0: a normal
/// refresh). The first retries come quickly: a gateway started before the network was up (an app
/// opened at login) has no seed until one works, so a newcomer's `openhydra/auto` finds nothing.
/// Then the tokenizer index's back-off, never longer than a normal refresh: a seed the registry
/// removed (a compromised one) must drop out within [`REFRESH_EVERY`].
fn retry_delay(failures: u32) -> Duration {
    match failures {
        0 => REFRESH_EVERY,
        1 => Duration::from_secs(30),
        2 => Duration::from_secs(2 * 60),
        n => refresh_delay(n - 1).min(REFRESH_EVERY),
    }
}

/// `list`, unless it's older than `min` (see [`MIN_SEQUENCE`]).
fn current_enough(list: SeedList, min: u64) -> Result<SeedList, String> {
    if list.sequence < min {
        return Err(format!("seed list #{} is older than this release accepts (#{min} or newer)", list.sequence));
    }
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
                match verify(&text, pinned).and_then(|l| current_enough(l, MIN_SEQUENCE)) {
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
        let list = current_enough(verify(published, &self.pinned)?, MIN_SEQUENCE)?;
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
    /// sooner ([`retry_delay`]), warning at most once a day while it keeps failing.
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
                    std::thread::sleep(retry_delay(failures));
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
        Seed { public_key: hex::encode(SigningKey::from_bytes(&[n; 32]).verifying_key().to_bytes()), label: label.into(), models: vec![] }
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
        assert!(list(1, vec![Seed { public_key: "AB".repeat(32), label: "a".into(), models: vec![] }]).validate().is_err(), "upper case");
        assert!(list(1, vec![Seed { public_key: "ab".into(), label: "a".into(), models: vec![] }]).validate().is_err(), "short");
        assert!(list(1, vec![seed(1, &"x".repeat(MAX_LABEL_CHARS + 1))]).validate().is_err());
        assert!(list(1, vec![seed(1, "a\nb")]).validate().is_err());
        assert!(SeedList { format: 2, ..list(1, vec![]) }.validate().is_err());
        let mut l = list(1, vec![]);
        let s = seed(3, "first");
        l.upsert(Seed { public_key: s.public_key.to_uppercase(), label: " first ".into(), models: vec![] }).unwrap();
        l.upsert(Seed { public_key: s.public_key.clone(), label: "renamed".into(), models: vec!["qwen3:8b".into(), " qwen3:8b ".into()] }).unwrap();
        assert_eq!(l.seeds, vec![Seed { public_key: s.public_key.clone(), label: "renamed".into(), models: vec!["qwen3:8b".into()] }]);
        assert_eq!(l.models(), vec!["qwen3:8b".to_string()]);
        let mut bad = list(1, vec![seed(4, "b")]);
        bad.seeds[0].models = vec!["/home/me/model.gguf".into()];
        assert!(bad.validate().is_err(), "a path is not a model name");
        bad.seeds[0].models = (0..=MAX_MODELS_PER_SEED).map(|i| format!("m{i}")).collect();
        assert!(bad.validate().is_err());
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

    #[test]
    fn a_list_older_than_the_release_accepts_is_refused() {
        assert!(current_enough(list(3, vec![]), 3).is_ok());
        assert!(current_enough(list(2, vec![seed(1, "removed")]), 3).unwrap_err().contains("#2 is older"));
        // Fetched or cached, a list below this build's floor never becomes the current one.
        let store = SeedStore::new(&pinned(), "https://example.invalid/seeds.json", None);
        assert!(store.offer(&sign(&list(MIN_SEQUENCE - 1, vec![seed(1, "old")]), &key()).unwrap(), None).is_err());
        assert!(store.current().seeds.is_empty());
        // A cached list below it is ignored at start, as an unverifiable one is.
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join("seed-providers.json");
        std::fs::write(&cache, sign(&list(MIN_SEQUENCE - 1, vec![seed(1, "old")]), &key()).unwrap()).unwrap();
        assert!(SeedStore::new(&pinned(), "https://example.invalid/seeds.json", Some(cache)).current().seeds.is_empty());
    }

    #[test]
    fn a_failed_fetch_is_retried_soon_and_never_later_than_a_normal_refresh() {
        assert_eq!(retry_delay(0), REFRESH_EVERY);
        assert_eq!(retry_delay(1), Duration::from_secs(30));
        assert!((1..20).all(|n| retry_delay(n) <= REFRESH_EVERY && retry_delay(n) <= retry_delay(n + 1)));
    }

    #[test]
    fn the_list_to_publish_is_signed_by_the_pinned_key_and_valid() {
        // What the seed-list workflow publishes: a gateway must accept it as is.
        let list = verify(include_str!("../data/seed-providers.json"), crate::tokenizer_index::PINNED_INDEX_KEY).unwrap();
        // …and it is this build's floor: raised with each list signed, never ahead of it.
        assert_eq!(MIN_SEQUENCE, list.sequence, "set MIN_SEQUENCE to the sequence of agent/data/seed-providers.json");
        assert!(!list.seeds.is_empty());
    }
}
