// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Hugging Face tokenizers, fetched on first use (W4).
//!
//! The [tokenizer index](crate::tokenizer_index) names a model's `tokenizer.json` by repo, commit and
//! sha256. The first serve of a model that has an entry starts a download from huggingface.co (over
//! TLS, redirects only to https), the file is hash-checked and kept under
//! `~/.openhydra/tokenizers/<sha256>.json`, and later serves count with it, in process, with HF's
//! `tokenizers` crate. Until the file is here (and if it can't be had: offline, a gated repo) the
//! model stays unchecked, as before; a failed download is retried after [`RETRY_AFTER`].
//!
//! The download tells Hugging Face which models this gateway uses (the Privacy Policy says so);
//! `serve --no-tokenizer-downloads` turns it off.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

use crate::tokenizer_index::{IndexEntry, IndexStore};

/// Where tokenizers are downloaded from.
pub const HF_BASE: &str = "https://huggingface.co";
/// After a failed download, the model stays unchecked this long before another try.
pub const RETRY_AFTER: Duration = Duration::from_secs(60 * 60);
/// Tokenizers kept loaded (each a few tens of MB in memory); the least recently used goes first.
const MAX_LOADED: usize = 4;
/// A download slower than this is given up (and retried later).
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(300);

/// The gateway's tokenizers: the index, the files on disk, the ones loaded.
pub struct HfTokenizers {
    index: Arc<IndexStore>,
    dir: PathBuf,
    hf_base: String,
    client: Option<reqwest::blocking::Client>,
    /// Most recently used last.
    loaded: Mutex<Vec<(String, Arc<Tokenizer>)>>,
    fetching: Mutex<HashSet<String>>,
    failed: Mutex<HashMap<String, Instant>>,
    /// One lock per tokenizer, held while it is read from disk: counts that need the same one at
    /// once wait for one load instead of each reading, hashing and parsing it (review 2026-10-14, M7).
    load_locks: Mutex<HashMap<String, Arc<Mutex<()>>>>,
    /// Tokenizers read from disk so far.
    disk_loads: std::sync::atomic::AtomicUsize,
}

/// What one count needs: the model's entry and where its tokenizer lives.
#[derive(Clone)]
pub struct HfCounter {
    tokenizers: Arc<HfTokenizers>,
    pub entry: IndexEntry,
}

/// Is `url` https, or plain http to this machine (tests, a local mirror)?
fn fetch_allowed(url: &reqwest::Url) -> bool {
    match url.scheme() {
        "https" => true,
        "http" => matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "[::1]")),
        _ => false,
    }
}

/// A client for tokenizer downloads: at most 5 redirects, never to plain http off this machine.
pub fn download_client() -> Result<reqwest::blocking::Client, String> {
    reqwest::blocking::Client::builder()
        .user_agent(concat!("openhydra-agent/", env!("CARGO_PKG_VERSION")))
        .timeout(DOWNLOAD_TIMEOUT)
        .redirect(reqwest::redirect::Policy::custom(|attempt| {
            if attempt.previous().len() >= 5 {
                attempt.error("too many redirects")
            } else if !fetch_allowed(attempt.url()) {
                attempt.error("redirect to plain http")
            } else {
                attempt.follow()
            }
        }))
        .build()
        .map_err(|e| e.to_string())
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

impl HfTokenizers {
    /// `client` is `None` to never download (only tokenizers already on disk are used).
    pub fn new(index: Arc<IndexStore>, dir: PathBuf, hf_base: &str, client: Option<reqwest::blocking::Client>) -> Self {
        Self {
            index,
            dir,
            hf_base: hf_base.to_string(),
            client,
            loaded: Mutex::new(Vec::new()),
            fetching: Mutex::new(HashSet::new()),
            failed: Mutex::new(HashMap::new()),
            load_locks: Mutex::new(HashMap::new()),
            disk_loads: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// `~/.openhydra/tokenizers`.
    pub fn default_dir() -> PathBuf {
        crate::tokenizer_index::openhydra_dir().join("tokenizers")
    }

    pub fn index(&self) -> &Arc<IndexStore> {
        &self.index
    }

    fn path_of(&self, sha: &str) -> PathBuf {
        self.dir.join(format!("{sha}.json"))
    }

    fn is_here(&self, sha: &str) -> bool {
        self.loaded.lock().unwrap_or_else(|e| e.into_inner()).iter().any(|(s, _)| s == sha) || self.path_of(sha).is_file()
    }

    /// A counter for `model`'s tokens, if the index has an entry for it and its tokenizer is here.
    /// One that isn't here yet is fetched in the background (this serve stays unchecked).
    pub fn counter_for(self: &Arc<Self>, model: &str) -> Option<HfCounter> {
        let entry = self.index.current().lookup(model)?.clone();
        if self.is_here(&entry.tokenizer_sha256) {
            return Some(HfCounter { tokenizers: Arc::clone(self), entry });
        }
        self.fetch_in_background(entry);
        None
    }

    fn fetch_in_background(self: &Arc<Self>, entry: IndexEntry) {
        if self.client.is_none() {
            return;
        }
        let sha = entry.tokenizer_sha256.clone();
        if self.failed.lock().unwrap_or_else(|e| e.into_inner()).get(&sha).is_some_and(|t| t.elapsed() < RETRY_AFTER) {
            return;
        }
        if !self.fetching.lock().unwrap_or_else(|e| e.into_inner()).insert(sha.clone()) {
            return;
        }
        let me = Arc::clone(self);
        let spawned = std::thread::Builder::new().name("tokenizer-fetch".into()).spawn(move || {
            match me.fetch(&entry) {
                Ok(()) => tracing::info!(repo = %entry.repo, "tokenizer downloaded; checking billed token counts for its models"),
                Err(e) => {
                    tracing::warn!(repo = %entry.repo, error = %e, "couldn't download a tokenizer; its models stay unchecked for now");
                    me.failed.lock().unwrap_or_else(|e| e.into_inner()).insert(entry.tokenizer_sha256.clone(), Instant::now());
                }
            }
            me.fetching.lock().unwrap_or_else(|e| e.into_inner()).remove(&entry.tokenizer_sha256);
        });
        if spawned.is_err() {
            self.fetching.lock().unwrap_or_else(|e| e.into_inner()).remove(&sha);
        }
    }

    /// Download `entry`'s tokenizer, check its size and hash, keep it on disk.
    pub fn fetch(&self, entry: &IndexEntry) -> Result<(), String> {
        let client = self.client.as_ref().ok_or("downloads are off")?;
        let url = entry.tokenizer_url(&self.hf_base);
        let parsed = reqwest::Url::parse(&url).map_err(|e| e.to_string())?;
        if !fetch_allowed(&parsed) {
            return Err(format!("not downloading over plain http: {url}"));
        }
        let resp = client.get(parsed).send().map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!("HTTP {}", resp.status()));
        }
        let body = crate::tokenizer_index::read_capped(resp, entry.tokenizer_size)?;
        if body.len() as u64 != entry.tokenizer_size || sha256_hex(&body) != entry.tokenizer_sha256 {
            return Err("the downloaded tokenizer doesn't match the index".into());
        }
        // Loads before it's kept: a file the crate can't read is never cached.
        load_bytes(&body)?;
        crate::tokenizer_index::write_atomic(&self.path_of(&entry.tokenizer_sha256), &body).map_err(|e| e.to_string())
    }

    /// The tokenizer with this sha256: loaded already, or read from disk and hash-checked.
    fn load(&self, sha: &str) -> Result<Arc<Tokenizer>, String> {
        if let Some(t) = self.loaded_now(sha) {
            return Ok(t);
        }
        // One load per tokenizer at a time; whoever waited finds it loaded.
        let gate = Arc::clone(self.load_locks.lock().unwrap_or_else(|e| e.into_inner()).entry(sha.to_string()).or_default());
        let _loading = gate.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(t) = self.loaded_now(sha) {
            return Ok(t);
        }
        self.disk_loads.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = self.path_of(sha);
        let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        if sha256_hex(&bytes) != sha {
            // Changed on disk: dropped, fetched again next time.
            let _ = std::fs::remove_file(&path);
            return Err("a cached tokenizer didn't match its hash".into());
        }
        let t = Arc::new(load_bytes(&bytes)?);
        let mut loaded = self.loaded.lock().unwrap_or_else(|e| e.into_inner());
        if !loaded.iter().any(|(s, _)| s == sha) {
            if loaded.len() >= MAX_LOADED {
                loaded.remove(0);
            }
            loaded.push((sha.to_string(), Arc::clone(&t)));
        }
        Ok(t)
    }
}

impl HfTokenizers {
    /// The tokenizer with this sha256 if it is loaded (marked most recently used).
    fn loaded_now(&self, sha: &str) -> Option<Arc<Tokenizer>> {
        let mut loaded = self.loaded.lock().unwrap_or_else(|e| e.into_inner());
        let i = loaded.iter().position(|(s, _)| s == sha)?;
        let hit = loaded.remove(i);
        let t = Arc::clone(&hit.1);
        loaded.push(hit);
        Some(t)
    }
}

/// A tokenizer from `tokenizer.json` bytes, with any truncation or padding it was saved with off (a
/// count needs every token).
fn load_bytes(bytes: &[u8]) -> Result<Tokenizer, String> {
    let mut t = Tokenizer::from_bytes(bytes).map_err(|e| format!("tokenizer: {e}"))?;
    t.with_truncation(None).map_err(|e| format!("tokenizer: {e}"))?;
    t.with_padding(None);
    Ok(t)
}

/// How many tokens `text` is under `t`, counted in pieces of at most `chunk` bytes cut between words.
pub fn count_with(t: &Tokenizer, text: &str, chunk: usize) -> Result<u64, String> {
    let mut n = 0u64;
    for piece in crate::token_count::chunks(text, chunk) {
        if piece.is_empty() {
            continue;
        }
        let enc = t.encode_fast(piece, false).map_err(|e| format!("tokenizer: {e}"))?;
        n += enc.get_ids().len() as u64;
    }
    Ok(n)
}

impl HfCounter {
    /// How many tokens `text` is under the model's tokenizer.
    pub fn count(&self, text: &str) -> Result<u64, String> {
        let t = self.tokenizers.load(&self.entry.tokenizer_sha256)?;
        count_with(&t, text, crate::token_count::COUNT_CHUNK_BYTES)
    }

    /// The model's tool-call template tokens (the index's, else the global allowance).
    pub fn tool_call_allowance(&self) -> u64 {
        self.entry.tool_call_allowance.unwrap_or(crate::serve::TOOL_CALL_TOKEN_ALLOWANCE)
    }
}

/// Build an index entry for `repo` at `revision` (a branch, tag or commit) — for the registry
/// tooling (`registry index add`). Asks the Hugging Face API for the commit the revision points at
/// (and refuses a gated repo: gateways download without an account), downloads `tokenizer.json` at
/// that commit, checks the crate can load it, and records its sha256 and size.
pub fn build_entry(
    client: &reqwest::blocking::Client,
    hf_base: &str,
    repo: &str,
    revision: &str,
    names: &[String],
    tool_call_allowance: Option<u64>,
) -> Result<IndexEntry, String> {
    if !crate::tokenizer_index::is_valid_repo(repo) {
        return Err(format!("{repo:?} isn't an org/name repo id"));
    }
    if revision.is_empty() || !revision.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-')) {
        return Err(format!("revision {revision:?}: a branch, tag or commit"));
    }
    let base = hf_base.trim_end_matches('/');
    let api = reqwest::Url::parse(&format!("{base}/api/models/{repo}/revision/{revision}")).map_err(|e| e.to_string())?;
    if !fetch_allowed(&api) {
        return Err(format!("not fetching over plain http: {api}"));
    }
    let resp = client.get(api).send().map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("{repo}@{revision}: HTTP {}", resp.status()));
    }
    let info: serde_json::Value =
        serde_json::from_slice(&crate::tokenizer_index::read_capped(resp, 16 << 20)?).map_err(|e| format!("model info: {e}"))?;
    let gated = info.get("gated").is_some_and(|g| !matches!(g, serde_json::Value::Bool(false) | serde_json::Value::Null));
    if gated {
        return Err(format!("{repo} is gated: gateways can't download it without an account; use an ungated copy"));
    }
    let commit = info.get("sha").and_then(|v| v.as_str()).ok_or("model info has no commit sha")?.to_ascii_lowercase();
    let mut entry = IndexEntry {
        names: names.iter().map(|n| n.trim().to_lowercase()).collect(),
        repo: repo.to_string(),
        revision: commit,
        tokenizer_sha256: "0".repeat(64),
        tokenizer_size: 1,
        tool_call_allowance,
    };
    let url = reqwest::Url::parse(&entry.tokenizer_url(base)).map_err(|e| e.to_string())?;
    if !fetch_allowed(&url) {
        return Err(format!("not fetching over plain http: {url}"));
    }
    let resp = client.get(url).send().map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("{repo}: tokenizer.json: HTTP {}", resp.status()));
    }
    let body = crate::tokenizer_index::read_capped(resp, crate::tokenizer_index::MAX_TOKENIZER_BYTES)?;
    load_bytes(&body)?;
    entry.tokenizer_sha256 = sha256_hex(&body);
    entry.tokenizer_size = body.len() as u64;
    Ok(entry)
}

/// A tokenizer file's path for `sha` under `dir` (for the registry tooling and tests).
pub fn cached_path(dir: &Path, sha: &str) -> PathBuf {
    dir.join(format!("{sha}.json"))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::tokenizer_index::tests::{entry, index};

    /// A small word-level tokenizer: each known word is one token, anything else `[UNK]`.
    pub fn tiny_tokenizer_json() -> String {
        serde_json::json!({
            "version": "1.0",
            "truncation": {"direction": "Right", "max_length": 2, "strategy": "LongestFirst", "stride": 0},
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "hello": 1, "world": 2, "!": 3}, "unk_token": "[UNK]"}
        })
        .to_string()
    }

    #[test]
    fn counts_that_need_the_same_tokenizer_at_once_load_it_once() {
        use sha2::Digest;
        let json = tiny_tokenizer_json().into_bytes();
        let sha = hex::encode(sha2::Sha256::digest(&json));
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(cached_path(dir.path(), &sha), &json).unwrap();
        let store = Arc::new(IndexStore::new(index(1, vec![entry(&["qwen3:8b"], &sha, json.len() as u64)]), "00", "https://x", None));
        let tk = Arc::new(HfTokenizers::new(store, dir.path().to_path_buf(), "https://huggingface.co", None));
        let counter = tk.counter_for("qwen3:8b").expect("cached");
        let threads: Vec<_> = (0..10)
            .map(|_| {
                let c = counter.clone();
                std::thread::spawn(move || c.count("hello world").unwrap())
            })
            .collect();
        for t in threads {
            assert_eq!(t.join().unwrap(), 2);
        }
        assert_eq!(tk.disk_loads.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    /// A loopback server answering every GET with `body` (or a redirect to `redirect`).
    pub fn serve(body: Vec<u8>, redirect: Option<String>) -> (String, Arc<std::sync::atomic::AtomicUsize>) {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let hits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let h = Arc::clone(&hits);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut s) = stream else { continue };
                let mut buf = [0u8; 4096];
                let _ = s.read(&mut buf);
                h.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let head = match &redirect {
                    Some(to) => format!("HTTP/1.1 302 Found\r\nlocation: {to}\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"),
                    None => format!("HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n", body.len()),
                };
                let _ = s.write_all(head.as_bytes());
                if redirect.is_none() {
                    let _ = s.write_all(&body);
                }
            }
        });
        (base, hits)
    }

    fn store_with(e: IndexEntry) -> Arc<IndexStore> {
        Arc::new(IndexStore::new(index(1, vec![e]), "00", "https://example.invalid", None))
    }

    #[test]
    fn a_tokenizer_is_fetched_once_checked_and_counted_with() {
        let json = tiny_tokenizer_json().into_bytes();
        let sha = sha256_hex(&json);
        let (base, hits) = serve(json.clone(), None);
        let dir = tempfile::tempdir().unwrap();
        let e = entry(&["qwen3-8b", "qwen3:8b"], &sha, json.len() as u64);
        let tk = Arc::new(HfTokenizers::new(store_with(e.clone()), dir.path().to_path_buf(), &base, Some(download_client().unwrap())));
        // First use: not here yet, fetched in the background, this serve unchecked.
        assert!(tk.counter_for("qwen3:8b-q4_K_M").is_none());
        let deadline = Instant::now() + Duration::from_secs(10);
        while !cached_path(dir.path(), &sha).is_file() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        let c = tk.counter_for("qwen3:8b").expect("downloaded");
        // Saved truncation (max_length 2) is off: every token counts.
        assert_eq!(c.count("hello world ! hello").unwrap(), 4);
        assert_eq!(c.count("").unwrap(), 0);
        assert_eq!(c.tool_call_allowance(), crate::serve::TOOL_CALL_TOKEN_ALLOWANCE);
        assert_eq!(hits.load(std::sync::atomic::Ordering::SeqCst), 1);
        // A model the index doesn't list: nothing fetched, unchecked.
        assert!(tk.counter_for("llama3.1:8b").is_none());
        // Pieces add up to the whole.
        let t = load_bytes(&json).unwrap();
        assert_eq!(count_with(&t, &"hello world ".repeat(100), 30).unwrap(), 200);
    }

    #[test]
    fn a_file_that_does_not_match_the_index_is_never_kept() {
        let json = tiny_tokenizer_json().into_bytes();
        let (base, _) = serve(json.clone(), None);
        let dir = tempfile::tempdir().unwrap();
        // The index says another hash (a swapped file upstream, or on the way).
        let e = entry(&["qwen3-8b"], &"c".repeat(64), json.len() as u64);
        let tk = HfTokenizers::new(store_with(e.clone()), dir.path().to_path_buf(), &base, Some(download_client().unwrap()));
        assert!(tk.fetch(&e).unwrap_err().contains("doesn't match"));
        assert!(std::fs::read_dir(dir.path()).unwrap().next().is_none());
        // Larger than the index says: cut off, refused.
        let e = entry(&["qwen3-8b"], &sha256_hex(&json), json.len() as u64 - 1);
        assert!(tk.fetch(&e).is_err());
        // Not a tokenizer at all, though the hash matches: not kept.
        let junk = b"not json".to_vec();
        let (base, _) = serve(junk.clone(), None);
        let tk = HfTokenizers::new(store_with(e.clone()), dir.path().to_path_buf(), &base, Some(download_client().unwrap()));
        let e = entry(&["qwen3-8b"], &sha256_hex(&junk), junk.len() as u64);
        assert!(tk.fetch(&e).is_err());
        assert!(std::fs::read_dir(dir.path()).unwrap().next().is_none());
    }

    #[test]
    fn a_cached_file_changed_on_disk_is_dropped() {
        let json = tiny_tokenizer_json().into_bytes();
        let sha = sha256_hex(&json);
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(cached_path(dir.path(), &sha), b"{\"tampered\":1}").unwrap();
        let e = entry(&["qwen3-8b"], &sha, json.len() as u64);
        let tk = Arc::new(HfTokenizers::new(store_with(e), dir.path().to_path_buf(), HF_BASE, None));
        let c = tk.counter_for("qwen3-8b").expect("a file is there");
        assert!(c.count("hello").is_err());
        assert!(!cached_path(dir.path(), &sha).exists());
        // With downloads off, a missing tokenizer is simply unavailable.
        assert!(tk.counter_for("qwen3-8b").is_none());
    }

    #[test]
    fn redirects_never_go_to_plain_http_off_this_machine() {
        let (base, _) = serve(Vec::new(), Some("http://example.com/tokenizer.json".into()));
        let dir = tempfile::tempdir().unwrap();
        let e = entry(&["qwen3-8b"], &"c".repeat(64), 10);
        let tk = HfTokenizers::new(store_with(e.clone()), dir.path().to_path_buf(), &base, Some(download_client().unwrap()));
        let err = tk.fetch(&e).unwrap_err();
        assert!(err.contains("plain http") || err.contains("redirect"), "{err}");
        let off = HfTokenizers::new(store_with(e.clone()), dir.path().to_path_buf(), "http://example.com", Some(download_client().unwrap()));
        assert!(off.fetch(&e).unwrap_err().contains("plain http"));
    }

    /// A loopback server answering by path: `routes` maps a path to (status, body).
    fn serve_routes(routes: Vec<(String, u16, Vec<u8>)>) -> String {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut s) = stream else { continue };
                let mut buf = [0u8; 4096];
                let n = s.read(&mut buf).unwrap_or(0);
                let req = String::from_utf8_lossy(&buf[..n]).to_string();
                let path = req.split_whitespace().nth(1).unwrap_or("").to_string();
                let (code, body) = routes.iter().find(|(p, _, _)| *p == path).map(|(_, c, b)| (*c, b.clone())).unwrap_or((404, Vec::new()));
                let _ = s.write_all(format!("HTTP/1.1 {code} X\r\ncontent-length: {}\r\nconnection: close\r\n\r\n", body.len()).as_bytes());
                let _ = s.write_all(&body);
            }
        });
        base
    }

    #[test]
    fn an_entry_is_built_from_the_commit_a_revision_points_at() {
        let json = tiny_tokenizer_json().into_bytes();
        let commit = "0123456789abcdef0123456789abcdef01234567";
        let base = serve_routes(vec![
            ("/api/models/Qwen/Qwen3-8B/revision/main".into(), 200, format!("{{\"sha\":\"{commit}\",\"gated\":false}}").into_bytes()),
            (format!("/Qwen/Qwen3-8B/resolve/{commit}/tokenizer.json"), 200, json.clone()),
            ("/api/models/meta-llama/Llama-3.1-8B/revision/main".into(), 200, format!("{{\"sha\":\"{commit}\",\"gated\":\"manual\"}}").into_bytes()),
            ("/api/models/x/broken/revision/main".into(), 200, format!("{{\"sha\":\"{commit}\"}}").into_bytes()),
            (format!("/x/broken/resolve/{commit}/tokenizer.json"), 200, b"not a tokenizer".to_vec()),
        ]);
        let client = download_client().unwrap();
        let names = vec!["Qwen3-8B".to_string(), "qwen3:8b".to_string()];
        let e = build_entry(&client, &base, "Qwen/Qwen3-8B", "main", &names, Some(20)).unwrap();
        assert_eq!(e.revision, commit);
        assert_eq!(e.names, vec!["qwen3-8b", "qwen3:8b"]);
        assert_eq!((e.tokenizer_sha256.clone(), e.tokenizer_size), (sha256_hex(&json), json.len() as u64));
        assert_eq!(e.tool_call_allowance, Some(20));
        // Gated repos, files the crate can't load, bad ids and revisions are refused.
        assert!(build_entry(&client, &base, "meta-llama/Llama-3.1-8B", "main", &names, None).unwrap_err().contains("gated"));
        assert!(build_entry(&client, &base, "x/broken", "main", &names, None).is_err());
        assert!(build_entry(&client, &base, "../x", "main", &names, None).is_err());
        assert!(build_entry(&client, &base, "Qwen/Qwen3-8B", "refs/pr/1", &names, None).is_err());
        assert!(build_entry(&client, &base, "Qwen/Missing", "main", &names, None).unwrap_err().contains("404"));
        assert!(build_entry(&client, "http://example.com", "Qwen/Qwen3-8B", "main", &names, None).unwrap_err().contains("plain http"));
    }

    #[test]
    fn a_failed_download_is_not_retried_at_once() {
        let dir = tempfile::tempdir().unwrap();
        // Nothing listens here: the download fails.
        let e = entry(&["qwen3-8b"], &"c".repeat(64), 10);
        let tk = Arc::new(HfTokenizers::new(store_with(e), dir.path().to_path_buf(), "http://127.0.0.1:9", Some(download_client().unwrap())));
        assert!(tk.counter_for("qwen3-8b").is_none());
        let deadline = Instant::now() + Duration::from_secs(10);
        while tk.failed.lock().unwrap().is_empty() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert_eq!(tk.failed.lock().unwrap().len(), 1);
        assert!(tk.counter_for("qwen3-8b").is_none());
        assert!(tk.fetching.lock().unwrap().is_empty(), "no new download while backing off");
    }
}
