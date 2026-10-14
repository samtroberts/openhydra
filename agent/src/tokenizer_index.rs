// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! The tokenizer index (W4): which Hugging Face tokenizer counts each model's tokens.
//!
//! A gateway checks a provider's billed token count by tokenizing what it delivered with the
//! model's own tokenizer ([`crate::token_count`]). The index maps model names to a repo pinned at a
//! commit and the sha256 of its `tokenizer.json`, so the file can't change under us and a provider
//! never chooses the tokenizer it is counted with. It is signed by the OpenHydra registry key
//! (pinned here, [`PINNED_INDEX_KEY`]) and served whole from a CDN ([`DEFAULT_INDEX_URL`]): fetching
//! all of it tells OpenHydra nothing about which models a gateway uses. A copy is built into the
//! binary ([`bundled`]) for first launch and offline use; a fetched index replaces it only when it is
//! newer (a higher `sequence`), so an old signed copy can't be replayed.
//!
//! Models with no entry stay unchecked, as before.

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use base64::Engine as _;
use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};

/// The OpenHydra registry public key the index must be signed with.
pub const PINNED_INDEX_KEY: &str = "65d41cba369f39dd8100c273cc3729294f06bc91c84f5d39f5e5c46dd200e034";

/// Where gateways fetch the index from: the download bucket, published by
/// `.github/workflows/tokenizer-index.yml` from `agent/data/tokenizer-index.json`.
pub const DEFAULT_INDEX_URL: &str = "https://dl.openhydra.co/tokenizers/index.json";

/// How often a gateway checks for a newer index (a conditional request: unchanged costs one 304).
pub const REFRESH_EVERY: Duration = Duration::from_secs(24 * 60 * 60);

/// Signed bytes are prefixed with this, so an index signature can't be passed off as anything else
/// the registry key signs (log-prob packs).
const INDEX_DOMAIN: &[u8] = b"openhydra/tokenizer-index/v1\0";

/// The index format this build reads.
pub const INDEX_FORMAT: u32 = 1;

/// The largest index accepted (signed envelope).
pub const MAX_INDEX_BYTES: usize = 4 << 20;
/// The most models an index may list.
pub const MAX_ENTRIES: usize = 20_000;
/// The most names one entry may carry.
pub const MAX_NAMES: usize = 64;
/// The largest `tokenizer.json` an entry may name (the biggest common ones are ~35 MB).
pub const MAX_TOKENIZER_BYTES: u64 = 64 << 20;
/// The largest per-model tool-call allowance an entry may set.
pub const MAX_TOOL_CALL_ALLOWANCE: u64 = 256;

/// The signed copy built into this binary.
const BUNDLED: &str = include_str!("../data/tokenizer-index.json");

/// One model's tokenizer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexEntry {
    /// Model names this tokenizer counts for (lower-case; matched as [`lookup_keys`] describes).
    pub names: Vec<String>,
    /// Hugging Face repo, `org/name`.
    pub repo: String,
    /// The commit `tokenizer.json` is read at (40 hex digits).
    pub revision: String,
    /// sha256 of `tokenizer.json` (64 hex digits): checked after download and on every load.
    pub tokenizer_sha256: String,
    /// Its size in bytes.
    pub tokenizer_size: u64,
    /// The model's tool-call template tokens, when known; the global allowance otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_allowance: Option<u64>,
}

/// The index's signed content.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerIndex {
    pub format: u32,
    /// Raised on every publish; a gateway never goes back to a lower one.
    pub sequence: u64,
    pub issued_ms: u64,
    pub models: Vec<IndexEntry>,
}

/// The published form: the index bytes, their signature and the signing key.
#[derive(Debug, Serialize, Deserialize)]
struct Envelope {
    index: String,
    signature: String,
    key: String,
}

fn is_hex(s: &str, n: usize) -> bool {
    s.len() == n && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// An `org/name` Hugging Face repo id: no `..`, no other separators, nothing to escape in a URL.
pub fn is_valid_repo(repo: &str) -> bool {
    let part = |p: &str| {
        !p.is_empty()
            && p.len() <= 96
            && p.as_bytes()[0].is_ascii_alphanumeric()
            && !p.contains("..")
            && p.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
    };
    matches!(repo.split_once('/'), Some((org, name)) if part(org) && part(name))
}

impl IndexEntry {
    fn validate(&self) -> Result<(), String> {
        if self.names.is_empty() || self.names.len() > MAX_NAMES {
            return Err(format!("{}: 1-{MAX_NAMES} names", self.repo));
        }
        if let Some(n) = self.names.iter().find(|n| !openhydra_protocol::model_id::is_valid_model_id(n) || n.to_lowercase() != **n) {
            return Err(format!("{}: invalid name {n:?} (lower-case model names only)", self.repo));
        }
        if !is_valid_repo(&self.repo) {
            return Err(format!("invalid repo {:?}", self.repo));
        }
        if !is_hex(&self.revision, 40) {
            return Err(format!("{}: revision must be a 40-digit commit", self.repo));
        }
        if !is_hex(&self.tokenizer_sha256, 64) {
            return Err(format!("{}: tokenizer_sha256 must be 64 hex digits", self.repo));
        }
        if self.tokenizer_size == 0 || self.tokenizer_size > MAX_TOKENIZER_BYTES {
            return Err(format!("{}: tokenizer_size out of range", self.repo));
        }
        if self.tool_call_allowance.is_some_and(|a| a > MAX_TOOL_CALL_ALLOWANCE) {
            return Err(format!("{}: tool_call_allowance above {MAX_TOOL_CALL_ALLOWANCE}", self.repo));
        }
        Ok(())
    }

    /// Where `tokenizer.json` is downloaded from, under `hf_base` (`https://huggingface.co`).
    pub fn tokenizer_url(&self, hf_base: &str) -> String {
        format!("{}/{}/resolve/{}/tokenizer.json", hf_base.trim_end_matches('/'), self.repo, self.revision)
    }
}

impl TokenizerIndex {
    /// Checks every entry, and that no name is listed twice.
    pub fn validate(&self) -> Result<(), String> {
        if self.format != INDEX_FORMAT {
            return Err(format!("index format {} (this build reads {INDEX_FORMAT})", self.format));
        }
        if self.models.len() > MAX_ENTRIES {
            return Err(format!("more than {MAX_ENTRIES} models"));
        }
        let mut seen = std::collections::HashSet::new();
        for e in &self.models {
            e.validate()?;
            if let Some(n) = e.names.iter().find(|n| !seen.insert(n.as_str())) {
                return Err(format!("name {n:?} listed twice"));
            }
        }
        Ok(())
    }

    /// Add `entry` (registry tooling): it replaces an entry for the same repo; a name another repo's
    /// entry already has is refused.
    pub fn upsert(&mut self, entry: IndexEntry) -> Result<(), String> {
        entry.validate()?;
        if let Some((n, other)) = self
            .models
            .iter()
            .filter(|e| e.repo != entry.repo)
            .find_map(|e| entry.names.iter().find(|n| e.names.contains(n)).map(|n| (n, e)))
        {
            return Err(format!("{n:?} already names {}", other.repo));
        }
        self.models.retain(|e| e.repo != entry.repo);
        self.models.push(entry);
        self.models.sort_by(|a, b| a.repo.cmp(&b.repo));
        Ok(())
    }

    /// Of `models`, the ones no entry covers (registry tooling: what to add next).
    pub fn missing<'a>(&self, models: impl IntoIterator<Item = &'a str>) -> Vec<&'a str> {
        models.into_iter().filter(|m| self.lookup(m).is_none()).collect()
    }

    /// The entry that counts `model`'s tokens, if any (see [`lookup_keys`]).
    pub fn lookup(&self, model: &str) -> Option<&IndexEntry> {
        // A name given with an `org/` prefix (vLLM, Hugging Face, LM Studio) is a repo name: without
        // its org it may match a sized name (`qwen3-8b`) or a repo the entry names (its own, `phi-4`,
        // or one it lists as `org/name` sharing its file, `deepseek-ai/deepseek-v4-pro`), never a
        // bare family alias (`deepseek-r1`, Ollama's default tag), which names a different model
        // there: `deepseek-ai/DeepSeek-R1` is the 671B model, not the 8B distill Ollama's
        // `deepseek-r1` is (review 2026-10-14, M6).
        let org_given = model.contains('/');
        lookup_keys(model).iter().find_map(|k| {
            let org_stripped = org_given && !k.contains('/');
            self.models.iter().find(|e| {
                let names_repo = |k: &str| {
                    e.repo.rsplit('/').next().is_some_and(|r| r.eq_ignore_ascii_case(k))
                        || e.names.iter().any(|n| n.split_once('/').is_some_and(|(_, r)| r == k))
                };
                e.names.iter().any(|n| n == k && (!org_stripped || !is_bare_alias(n) || names_repo(k)))
            })
        })
    }
}

/// An index name that names a model family without a size or tag (`deepseek-r1`, `qwen3`,
/// `llama3.2`): Ollama's name for the family's default tag.
fn is_bare_alias(name: &str) -> bool {
    !name.contains(':') && !name.split(['-', '_']).skip(1).any(is_size_token)
}

/// A name part that gives a model's size: `8b`, `1.5b`, `30b`, `a3b`, `e4b`, `8x7b`, `135m`.
fn is_size_token(t: &str) -> bool {
    let t = t.strip_prefix(['a', 'e']).filter(|r| r.starts_with(|c: char| c.is_ascii_digit())).unwrap_or(t);
    let t = t.split_once('x').map_or(t, |(n, rest)| if n.bytes().all(|b| b.is_ascii_digit()) && !n.is_empty() { rest } else { t });
    let Some(num) = t.strip_suffix(['b', 'm']) else { return false };
    !num.is_empty() && num.bytes().all(|b| b.is_ascii_digit() || b == b'.') && num.starts_with(|c: char| c.is_ascii_digit())
}

/// A trailing name part that only says how the weights are stored (`q4_k_m`, `fp16`, `4bit`, `gguf`):
/// one model's quantisations share its tokenizer.
fn is_storage_tag(t: &str) -> bool {
    let quant = |p: &str, rest: &str| rest.strip_prefix(p).is_some_and(|r| r.starts_with(|c: char| c.is_ascii_digit()));
    quant("q", t)
        || quant("iq", t)
        || t.strip_suffix("bit").is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
        || matches!(
            t,
            "f16" | "bf16" | "fp16" | "f32" | "fp32" | "fp8" | "fp4" | "mxfp4" | "nvfp4" | "int4" | "int8" | "awq" | "gptq"
                | "gguf" | "mlx" | "ud" | "qat" | "dwq" | "latest"
                // The instruction-tuned and base models of a size share a tokenizer.
                | "instruct" | "it" | "chat"
        )
}

/// The names `model` is looked up under, most specific first: as given (lower-case); without
/// `:latest`; without trailing storage tags (`qwen3:8b-q4_K_M` → `qwen3:8b`, `Qwen3-8B-UD-Q4_K_XL` →
/// `qwen3-8b`); and each of those without an `org/` prefix (`qwen/qwen3-8b` → `qwen3-8b`).
pub fn lookup_keys(model: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut push = |k: &str| {
        let k = k.trim_end_matches(['-', '_', '.', ':']);
        if !k.is_empty() && !out.iter().any(|o| o == k) {
            out.push(k.to_string());
        }
    };
    let full = model.trim().to_lowercase();
    let bare = full.rsplit_once('/').map(|(_, n)| n.to_string());
    for base in std::iter::once(full.clone()).chain(bare) {
        push(&base);
        let mut s = base.strip_suffix(":latest").unwrap_or(&base).to_string();
        push(&s);
        // Drop storage tags from the end, one at a time ("-ud-q4_k_xl": both go).
        for _ in 0..4 {
            let Some(cut) = s.rfind(['-', ':', '.']) else { break };
            if !is_storage_tag(&s[cut + 1..]) {
                break;
            }
            s.truncate(cut);
            push(&s);
        }
    }
    out
}

/// Sign `index` with the registry key: the published JSON.
pub fn sign(index: &TokenizerIndex, key: &SigningKey) -> Result<String, String> {
    index.validate()?;
    sign_unchecked(index, key)
}

fn sign_unchecked(index: &TokenizerIndex, key: &SigningKey) -> Result<String, String> {
    let bytes = serde_json::to_vec(index).map_err(|e| e.to_string())?;
    let mut msg = INDEX_DOMAIN.to_vec();
    msg.extend_from_slice(&bytes);
    let env = Envelope {
        index: base64::engine::general_purpose::STANDARD.encode(&bytes),
        signature: hex::encode(key.sign(&msg).to_bytes()),
        key: hex::encode(key.verifying_key().to_bytes()),
    };
    serde_json::to_string_pretty(&env).map_err(|e| e.to_string())
}

/// Read a published index, checking its signature against the `pinned` key (hex) and every entry.
pub fn verify(published: &str, pinned: &str) -> Result<TokenizerIndex, String> {
    let index = verify_signature(published, pinned)?;
    index.validate()?;
    Ok(index)
}

/// [`verify`] without the content checks: the index the pinned key signed, as published. One that
/// fails them may be newer than this build (a limit raised since): told apart from a bad one.
fn verify_signature(published: &str, pinned: &str) -> Result<TokenizerIndex, String> {
    if published.len() > MAX_INDEX_BYTES {
        return Err("index too large".into());
    }
    let env: Envelope = serde_json::from_str(published).map_err(|e| format!("index: {e}"))?;
    if !env.key.eq_ignore_ascii_case(pinned) {
        return Err("index signed by another key".into());
    }
    let key: [u8; 32] = hex::decode(pinned).ok().and_then(|b| b.try_into().ok()).ok_or("bad pinned key")?;
    let key = VerifyingKey::from_bytes(&key).map_err(|_| "bad pinned key")?;
    let sig: [u8; 64] = hex::decode(&env.signature).ok().and_then(|b| b.try_into().ok()).ok_or("bad signature")?;
    let bytes = base64::engine::general_purpose::STANDARD.decode(&env.index).map_err(|e| format!("index: {e}"))?;
    let mut msg = INDEX_DOMAIN.to_vec();
    msg.extend_from_slice(&bytes);
    key.verify_strict(&msg, &Signature::from_bytes(&sig)).map_err(|_| "bad signature")?;
    serde_json::from_slice(&bytes).map_err(|e| format!("index: {e}"))
}

/// The copy built into this binary (an empty index if it doesn't verify, which a test rules out).
pub fn bundled() -> TokenizerIndex {
    verify(BUNDLED, PINNED_INDEX_KEY).unwrap_or_else(|e| {
        tracing::warn!(error = %e, "the built-in tokenizer index doesn't verify");
        TokenizerIndex { format: INDEX_FORMAT, sequence: 0, issued_ms: 0, models: Vec::new() }
    })
}

/// `~/.openhydra` (`%USERPROFILE%` on Windows).
pub fn openhydra_dir() -> PathBuf {
    crate::gateway_key::default_path().parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from(".openhydra"))
}

/// The gateway's current index: the newest of the built-in copy, the last one fetched (kept in
/// `cache`), and what [`IndexStore::refresh`] fetches.
pub struct IndexStore {
    current: RwLock<Arc<TokenizerIndex>>,
    pinned: String,
    url: String,
    /// The last fetched index as published (`~/.openhydra/tokenizer-index.json`), its ETag beside it.
    cache: Option<PathBuf>,
    etag: RwLock<Option<String>>,
}

impl IndexStore {
    /// Start from the newer of `bundled` and the cached copy.
    pub fn new(bundled: TokenizerIndex, pinned: &str, url: &str, cache: Option<PathBuf>) -> Self {
        let mut current = bundled;
        let mut etag = None;
        if let Some(path) = &cache {
            if let Ok(text) = std::fs::read_to_string(path) {
                match verify(&text, pinned) {
                    Ok(cached) if cached.sequence > current.sequence => {
                        current = cached;
                        etag = std::fs::read_to_string(etag_path(path)).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
                    }
                    Ok(_) => {}
                    Err(e) => tracing::warn!(error = %e, path = %path.display(), "ignoring the cached tokenizer index"),
                }
            }
        }
        Self { current: RwLock::new(Arc::new(current)), pinned: pinned.to_string(), url: url.to_string(), cache, etag: RwLock::new(etag) }
    }

    pub fn current(&self) -> Arc<TokenizerIndex> {
        Arc::clone(&self.current.read().unwrap_or_else(|e| e.into_inner()))
    }

    /// Adopt a published index if it verifies and is newer than the current one. Whether it was.
    pub fn offer(&self, published: &str, etag: Option<String>) -> Result<bool, String> {
        let index = verify_signature(published, &self.pinned)?;
        let mut cur = self.current.write().unwrap_or_else(|e| e.into_inner());
        if index.sequence <= cur.sequence {
            // The one we have: its ETag makes the next fetch a 304 (it was only kept with a newer one).
            if index.sequence == cur.sequence && etag.is_some() {
                *self.etag.write().unwrap_or_else(|e| e.into_inner()) = etag;
            }
            return Ok(false);
        }
        if let Err(e) = index.validate() {
            return Err(format!(
                "tokenizer index #{} needs a newer OpenHydra ({e}); keeping #{} - update OpenHydra to use it",
                index.sequence, cur.sequence
            ));
        }
        if let Some(path) = &self.cache {
            if let Err(e) = write_atomic(path, published.as_bytes()) {
                tracing::warn!(error = %e, "couldn't keep the tokenizer index");
            } else {
                let _ = write_atomic(&etag_path(path), etag.as_deref().unwrap_or("").as_bytes());
            }
        }
        *self.etag.write().unwrap_or_else(|e| e.into_inner()) = etag;
        tracing::info!(sequence = index.sequence, models = index.models.len(), "tokenizer index updated");
        *cur = Arc::new(index);
        Ok(true)
    }

    /// Fetch the index (conditionally, by ETag) and adopt it if newer.
    pub fn refresh(&self, client: &reqwest::blocking::Client) -> Result<bool, String> {
        if !crate::registry_client::registry_url_allowed(&self.url) {
            return Err(format!("not fetching the tokenizer index over plain http: {}", self.url));
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
            return Err(format!("tokenizer index: HTTP {}", resp.status()));
        }
        let etag = resp.headers().get(reqwest::header::ETAG).and_then(|v| v.to_str().ok()).map(str::to_string);
        let body = read_capped(resp, MAX_INDEX_BYTES as u64)?;
        let text = String::from_utf8(body).map_err(|_| "index isn't UTF-8")?;
        self.offer(&text, etag)
    }

    /// Refresh now and then every [`REFRESH_EVERY`], on a background thread; after a failure,
    /// sooner (see [`refresh_delay`]), with a warning at most once a day while it keeps failing
    /// (review 2026-10-14, L13: a failed fetch was retried a day later, logged at debug only).
    pub fn spawn_refresher(self: &Arc<Self>, client: reqwest::blocking::Client) {
        let store = Arc::clone(self);
        std::thread::Builder::new()
            .name("tokenizer-index".into())
            .spawn(move || {
                let mut failures = 0u32;
                let mut last_warned: Option<std::time::Instant> = None;
                loop {
                    match store.refresh(&client) {
                        Ok(_) => failures = 0,
                        Err(e) => {
                            failures = failures.saturating_add(1);
                            let warn = last_warned.is_none_or(|t| t.elapsed() >= REFRESH_EVERY) || e.contains("needs a newer OpenHydra");
                            if warn {
                                last_warned = Some(std::time::Instant::now());
                                tracing::warn!(error = %e, url = %store.url, sequence = store.current().sequence,
                                    "couldn't refresh the tokenizer index; using the one this gateway has");
                            } else {
                                tracing::debug!(error = %e, "tokenizer index not refreshed");
                            }
                        }
                    }
                    std::thread::sleep(refresh_delay(failures));
                }
            })
            .ok();
    }
}

/// How long the refresher waits after `failures` failed fetches in a row: [`REFRESH_EVERY`] after a
/// good one; after a failure 5 min, 15 min, 1 h, 6 h, then daily.
pub fn refresh_delay(failures: u32) -> Duration {
    match failures {
        0 => REFRESH_EVERY,
        1 => Duration::from_secs(5 * 60),
        2 => Duration::from_secs(15 * 60),
        3 => Duration::from_secs(60 * 60),
        4 => Duration::from_secs(6 * 60 * 60),
        _ => REFRESH_EVERY,
    }
}

fn etag_path(cache: &Path) -> PathBuf {
    cache.with_extension("etag")
}

/// Read a response body, refusing one larger than `cap` bytes.
pub fn read_capped(resp: reqwest::blocking::Response, cap: u64) -> Result<Vec<u8>, String> {
    use std::io::Read;
    if resp.content_length().is_some_and(|n| n > cap) {
        return Err("response too large".into());
    }
    let mut body = Vec::new();
    resp.take(cap + 1).read_to_end(&mut body).map_err(|e| e.to_string())?;
    if body.len() as u64 > cap {
        return Err("response too large".into());
    }
    Ok(body)
}

/// Write `bytes` to `path` through a temp file and a rename (never a half-written file).
pub fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub fn entry(names: &[&str], sha: &str, size: u64) -> IndexEntry {
        IndexEntry {
            names: names.iter().map(|s| s.to_string()).collect(),
            repo: "Qwen/Qwen3-8B".into(),
            revision: "a".repeat(40),
            tokenizer_sha256: sha.into(),
            tokenizer_size: size,
            tool_call_allowance: None,
        }
    }

    pub fn index(seq: u64, models: Vec<IndexEntry>) -> TokenizerIndex {
        TokenizerIndex { format: INDEX_FORMAT, sequence: seq, issued_ms: 1, models }
    }

    fn key() -> SigningKey {
        SigningKey::from_bytes(&[7; 32])
    }

    fn pinned() -> String {
        hex::encode(key().verifying_key().to_bytes())
    }

    #[test]
    fn the_built_in_index_verifies_against_the_pinned_key() {
        let text = BUNDLED;
        let idx = verify(text, PINNED_INDEX_KEY).expect("data/tokenizer-index.json must be signed by the pinned registry key");
        // Above the test copies signed while building W4 (#2); covers what the network serves.
        assert!(idx.sequence >= 11, "{}", idx.sequence);
        for model in ["qwen2.5-coder:1.5b", "qwen3:8b-q4_K_M", "llama3.2", "gemma3:4b-it-qat", "gpt-oss:20b", "deepseek-r1:8b-0528-qwen3"] {
            assert!(idx.lookup(model).is_some(), "{model}");
        }
        // #11: Ollama's deepseek-r1:8b was the Llama distill before 2025-05-28, the Qwen3 one since: by
        // name alone it can't be told which tokenizer counts it, so it isn't listed.
        assert!(idx.lookup("deepseek-r1:8b").is_none());
        assert_eq!(bundled(), idx);
    }

    /// The built-in index on names real engines announce (M6): repo names never reach an Ollama alias.
    #[test]
    fn the_built_in_index_resolves_real_engine_names() {
        let idx = bundled();
        let repo = |m: &str| idx.lookup(m).map(|e| e.repo.as_str());
        assert_eq!(repo("deepseek-r1:latest"), Some("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"));
        assert_eq!(repo("deepseek-ai/DeepSeek-R1"), None, "the 671B model has another tokenizer");
        assert_eq!(repo("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"), Some("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"));
        assert_eq!(repo("Qwen/Qwen3-8B"), Some("Qwen/Qwen3-8B"));
        assert_eq!(repo("microsoft/phi-4"), Some("microsoft/phi-4"));
        assert_eq!(repo("phi4:latest"), Some("microsoft/phi-4"));
        assert_eq!(repo("mistralai/Mistral-Nemo-Instruct-2407"), Some("mistralai/Mistral-Nemo-Instruct-2407"));
        assert_eq!(repo("qwen3"), Some("Qwen/Qwen3-8B"), "Ollama's default tag");
        assert_eq!(repo("Qwen/Qwen3"), None);
        // #11: DeepSeek V4 (one tokenizer for Flash, Pro, their dated and Base versions) and V4.1.
        for (name, want) in [
            ("deepseek-ai/DeepSeek-V4-Pro", "deepseek-ai/DeepSeek-V4-Flash"),
            ("deepseek-ai/DeepSeek-V4-Pro-0813", "deepseek-ai/DeepSeek-V4-Flash"),
            ("deepseek-ai/DeepSeek-V4-Flash-0731", "deepseek-ai/DeepSeek-V4-Flash"),
            ("unsloth/DeepSeek-V4-Pro-GGUF", "deepseek-ai/DeepSeek-V4-Flash"),
            ("deepseek-v4-pro:cloud", "deepseek-ai/DeepSeek-V4-Flash"),
            ("deepseek-v4-flash", "deepseek-ai/DeepSeek-V4-Flash"),
            ("deepseek-ai/DeepSeek-V4.1-Flash", "deepseek-ai/DeepSeek-V4.1-Flash"),
        ] {
            assert_eq!(repo(name), Some(want), "{name}");
        }
    }

    #[test]
    fn only_the_pinned_key_and_untampered_bytes_verify() {
        let idx = index(3, vec![entry(&["qwen3-8b", "qwen3:8b"], &"b".repeat(64), 11_000_000)]);
        let published = sign(&idx, &key()).unwrap();
        assert_eq!(verify(&published, &pinned()).unwrap(), idx);
        // Another key (even a valid signature by it) is refused.
        let other = SigningKey::from_bytes(&[8; 32]);
        let theirs = sign(&idx, &other).unwrap();
        assert!(verify(&theirs, &pinned()).is_err());
        // A changed byte in the index, or the envelope claiming the pinned key over another's signature.
        let mut env: serde_json::Value = serde_json::from_str(&published).unwrap();
        let mut bytes = base64::engine::general_purpose::STANDARD.decode(env["index"].as_str().unwrap()).unwrap();
        let at = bytes.iter().position(|b| *b == b'b').unwrap();
        bytes[at] = b'c';
        env["index"] = base64::engine::general_purpose::STANDARD.encode(&bytes).into();
        assert!(verify(&env.to_string(), &pinned()).is_err());
        let mut forged: serde_json::Value = serde_json::from_str(&theirs).unwrap();
        forged["key"] = pinned().into();
        assert!(verify(&forged.to_string(), &pinned()).is_err());
        assert!(verify(&"x".repeat(MAX_INDEX_BYTES + 1), &pinned()).is_err());
    }

    #[test]
    fn entries_are_checked_before_signing_and_after_verifying() {
        let ok = entry(&["qwen3-8b"], &"b".repeat(64), 10);
        for bad in [
            IndexEntry { repo: "../etc".into(), ..ok.clone() },
            IndexEntry { repo: "Qwen/Qwen3-8B/../x".into(), ..ok.clone() },
            IndexEntry { repo: "Qwen/a?b".into(), ..ok.clone() },
            IndexEntry { revision: "main".into(), ..ok.clone() },
            IndexEntry { tokenizer_sha256: "B".repeat(64), ..ok.clone() },
            IndexEntry { tokenizer_size: MAX_TOKENIZER_BYTES + 1, ..ok.clone() },
            IndexEntry { names: vec!["Qwen3-8B".into()], ..ok.clone() },
            IndexEntry { names: vec!["<img>".into()], ..ok.clone() },
            IndexEntry { names: vec![], ..ok.clone() },
            IndexEntry { tool_call_allowance: Some(MAX_TOOL_CALL_ALLOWANCE + 1), ..ok.clone() },
        ] {
            assert!(sign(&index(1, vec![bad.clone()]), &key()).is_err(), "{bad:?}");
        }
        assert!(sign(&index(1, vec![ok.clone(), ok.clone()]), &key()).is_err(), "a name listed twice");
        assert!(sign(&TokenizerIndex { format: 2, ..index(1, vec![]) }, &key()).is_err());
        assert_eq!(ok.tokenizer_url("https://huggingface.co/"), format!("https://huggingface.co/Qwen/Qwen3-8B/resolve/{}/tokenizer.json", "a".repeat(40)));
    }

    #[test]
    fn names_are_matched_across_engines_and_quantisations() {
        let idx = index(1, vec![entry(&["qwen3-8b", "qwen3:8b"], &"b".repeat(64), 10)]);
        for name in [
            "qwen3:8b",
            "qwen3:8b-q4_K_M",
            "qwen3:8b-fp16",
            "Qwen3-8B",
            "qwen/qwen3-8b",
            "Qwen3-8B-Q4_K_M",
            "unsloth/Qwen3-8B-UD-Q4_K_XL",
            "mlx-community/Qwen3-8B-4bit",
            "qwen3-8b-gguf",
        ] {
            assert!(idx.lookup(name).is_some(), "{name}");
        }
        // A different size or family is not the same tokenizer's model.
        for name in ["qwen3:14b", "qwen3-8b-instruct-extra", "llama3.1:8b", "qwen3", ""] {
            assert!(idx.lookup(name).is_none(), "{name}");
        }
        assert_eq!(lookup_keys("qwen3:8b-q4_K_M"), vec!["qwen3:8b-q4_k_m", "qwen3:8b"]);
        assert_eq!(lookup_keys("llama3.2:latest"), vec!["llama3.2:latest", "llama3.2"]);
        // Ollama's instruct spellings are the same model's tokenizer.
        for name in ["qwen3:8b-instruct", "qwen3:8b-instruct-q8_0", "qwen3:8b-it-qat", "qwen3:8b-chat"] {
            assert!(idx.lookup(name).is_some(), "{name}");
        }
    }

    /// Review 2026-10-14 M6: an org-prefixed repo name never matches a bare family alias.
    #[test]
    fn a_repo_name_doesnt_match_a_bare_family_alias() {
        let mut r1 = entry(&["deepseek-r1", "deepseek-r1:8b", "deepseek-r1-0528-qwen3-8b"], &"b".repeat(64), 10);
        r1.repo = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B".into();
        let idx = index(1, vec![r1]);
        // Ollama's names: the alias is meant for them.
        for name in ["deepseek-r1", "deepseek-r1:latest", "deepseek-r1:8b", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF"] {
            assert!(idx.lookup(name).is_some(), "{name}");
        }
        // The full 671B model under its repo names: another tokenizer, so not counted with this one.
        for name in ["deepseek-ai/DeepSeek-R1", "deepseek/deepseek-r1", "unsloth/DeepSeek-R1-UD-IQ1_S", "deepseek-ai/DeepSeek-R1-0528"] {
            assert!(idx.lookup(name).is_none(), "{name}");
        }
        // A repo whose own name has no size is still found by it.
        let mut phi4 = entry(&["phi4", "phi-4"], &"c".repeat(64), 10);
        phi4.repo = "microsoft/phi-4".into();
        let phi = index(1, vec![phi4]);
        for name in ["microsoft/phi-4", "unsloth/phi-4-GGUF", "phi4", "phi4:latest"] {
            assert!(phi.lookup(name).is_some(), "{name}");
        }
        // Repos that share an entry's file are named by their full ids: found by those, and through
        // a quantiser's copy of them, as the entry's own repo is.
        let mut v4 = entry(&["deepseek-v4-flash", "deepseek-ai/deepseek-v4-pro", "deepseek-v4-pro"], &"d".repeat(64), 10);
        v4.repo = "deepseek-ai/DeepSeek-V4-Flash".into();
        let v4 = index(1, vec![v4]);
        for name in ["deepseek-ai/DeepSeek-V4-Pro", "unsloth/DeepSeek-V4-Pro-GGUF", "deepseek-ai/DeepSeek-V4-Flash", "deepseek-v4-pro"] {
            assert!(v4.lookup(name).is_some(), "{name}");
        }
        // What a bare alias is.
        for bare in ["deepseek-r1", "qwen3", "llama3.2", "gemma3", "phi4", "gpt-oss"] {
            assert!(is_bare_alias(bare), "{bare}");
        }
        for sized in ["qwen3-8b", "qwen3:8b", "qwen2.5-1.5b", "qwen3-30b-a3b", "gemma-3n-e4b", "mixtral-8x7b", "smollm2-135m", "deepseek-r1-0528-qwen3-8b"] {
            assert!(!is_bare_alias(sized), "{sized}");
        }
    }

    #[test]
    fn entries_are_added_by_repo_and_names_stay_unique() {
        let mut idx = index(1, vec![]);
        idx.upsert(entry(&["qwen3-8b"], &"b".repeat(64), 10)).unwrap();
        // The same repo again replaces its entry (a new commit, more names).
        idx.upsert(entry(&["qwen3-8b", "qwen3:8b"], &"c".repeat(64), 11)).unwrap();
        assert_eq!(idx.models.len(), 1);
        assert_eq!(idx.models[0].tokenizer_sha256, "c".repeat(64));
        // Another repo can't take a name.
        let other = IndexEntry { repo: "x/qwen3-copy".into(), ..entry(&["qwen3:8b"], &"d".repeat(64), 10) };
        assert!(idx.upsert(other.clone()).unwrap_err().contains("already names"));
        idx.upsert(IndexEntry { names: vec!["qwen3-copy".into()], ..other }).unwrap();
        assert_eq!(idx.missing(["qwen3:8b-q4_K_M", "llama3.1:8b", "qwen3-copy"]), vec!["llama3.1:8b"]);
        idx.validate().unwrap();
    }

    #[test]
    fn the_store_only_moves_forward_and_keeps_what_it_fetched() {
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join("tokenizer-index.json");
        let e = entry(&["qwen3-8b"], &"b".repeat(64), 10);
        let store = IndexStore::new(index(5, vec![]), &pinned(), "https://example.invalid/x", Some(cache.clone()));
        assert_eq!(store.current().sequence, 5);
        // Older or equal: refused (a replayed old copy can't roll the index back).
        assert!(!store.offer(&sign(&index(4, vec![e.clone()]), &key()).unwrap(), None).unwrap());
        assert!(!store.offer(&sign(&index(5, vec![e.clone()]), &key()).unwrap(), None).unwrap());
        assert!(store.offer(&sign(&index(6, vec![e.clone()]), &SigningKey::from_bytes(&[9; 32])).unwrap(), None).is_err());
        assert!(store.offer(&sign(&index(6, vec![e.clone()]), &key()).unwrap(), Some("\"v6\"".into())).unwrap());
        assert!(store.current().lookup("qwen3-8b").is_some());
        // A restart starts from the kept copy (and its ETag), not the older built-in one.
        let again = IndexStore::new(index(5, vec![]), &pinned(), "https://example.invalid/x", Some(cache.clone()));
        assert_eq!(again.current().sequence, 6);
        assert_eq!(again.etag.read().unwrap().as_deref(), Some("\"v6\""));
        // A newer built-in copy (an app update) wins over an older kept one.
        let updated = IndexStore::new(index(9, vec![]), &pinned(), "https://example.invalid/x", Some(cache.clone()));
        assert_eq!(updated.current().sequence, 9);
        // A kept copy that doesn't verify is ignored.
        std::fs::write(&cache, "{}").unwrap();
        assert_eq!(IndexStore::new(index(5, vec![]), &pinned(), "u", Some(cache)).current().sequence, 5);
    }

    /// Review 2026-10-14 L13 and the ETag note.
    #[test]
    fn refreshing_says_when_an_index_is_too_new_and_keeps_the_etag_of_the_one_it_has() {
        let store = IndexStore::new(index(5, vec![]), &pinned(), "https://example.invalid/x", None);
        // The server's copy is the one we have: its ETag is kept, so the next fetch is a 304.
        assert!(!store.offer(&sign(&index(5, vec![]), &key()).unwrap(), Some("\"v5\"".into())).unwrap());
        assert_eq!(store.etag.read().unwrap().as_deref(), Some("\"v5\""));
        // A newer index with an entry past this build's limits (a later release raised one): signed by
        // the pinned key, so not "bad" - it needs a newer OpenHydra, and the current one stays.
        let mut big = entry(&["qwen3-8b"], &"b".repeat(64), 10);
        big.tokenizer_size = MAX_TOKENIZER_BYTES + 1;
        let err = store.offer(&sign_unchecked(&index(7, vec![big]), &key()).unwrap(), None).unwrap_err();
        assert!(err.contains("needs a newer OpenHydra") && err.contains("#7"), "{err}");
        assert_eq!(store.current().sequence, 5);
        // After a failure the refresher looks again soon, then backs off to daily.
        assert_eq!(refresh_delay(0), REFRESH_EVERY);
        assert_eq!(refresh_delay(1), Duration::from_secs(300));
        assert!(refresh_delay(2) < refresh_delay(3) && refresh_delay(3) < refresh_delay(4));
        assert_eq!(refresh_delay(9), REFRESH_EVERY);
    }

    #[test]
    fn plain_http_is_refused_off_loopback() {
        let store = IndexStore::new(index(1, vec![]), &pinned(), "http://registry.example/index.json", None);
        let client = reqwest::blocking::Client::new();
        assert!(store.refresh(&client).unwrap_err().contains("plain http"));
    }
}
