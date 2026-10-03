// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! The digest of a model's weights, as `sha256:<hex>` — what log-prob references are matched to
//! (the check tells weights files apart, not quantisation labels).
//!
//! * A file (a GGUF): its sha256.
//! * A directory (an MLX model): the sha256 of a manifest listing `config.json` and the model's
//!   weights files ([`dir_members`]), sorted by name, one `<name>\0<file sha256 hex>\n` line each —
//!   the config is included because it decides how the weights are computed with.

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use sha2::{Digest, Sha256};

fn file_sha256_hex(path: &Path) -> io::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    io::copy(&mut f, &mut h)?;
    Ok(hex::encode(h.finalize()))
}

/// The files of an MLX model directory that go into its digest, sorted by name: `config.json` and
/// the model's weights — the shards its `model.safetensors.index.json` lists when it has one, else
/// its `model*.safetensors` (else every `*.safetensors`). Other files (a LoRA's
/// `adapters.safetensors`, a leftover original checkpoint) don't change the digest, and a member that
/// can't be read (a broken link of a half-downloaded snapshot) is an error, never left out.
fn dir_members(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let bad = |why: String| io::Error::new(io::ErrorKind::InvalidData, why);
    let mut names: Vec<String> = match std::fs::read(dir.join("model.safetensors.index.json")) {
        Ok(index) => {
            let index: serde_json::Value = serde_json::from_slice(&index).map_err(|e| bad(format!("model.safetensors.index.json: {e}")))?;
            let map = index.get("weight_map").and_then(|m| m.as_object()).ok_or_else(|| bad("index without weight_map".into()))?;
            map.values().filter_map(|v| v.as_str().map(String::from)).collect()
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            let all: Vec<String> = std::fs::read_dir(dir)?
                .filter_map(|e| e.ok()?.file_name().into_string().ok())
                .filter(|n| n.ends_with(".safetensors"))
                .collect();
            let model: Vec<String> = all.iter().filter(|n| n.starts_with("model")).cloned().collect();
            if model.is_empty() { all } else { model }
        }
        Err(e) => return Err(e),
    };
    if names.is_empty() {
        return Err(bad("not an MLX model directory (config.json + *.safetensors)".into()));
    }
    names.push("config.json".into());
    names.sort();
    names.dedup();
    let mut files = Vec::with_capacity(names.len());
    for name in names {
        // Shard names come from a file: plain names in this directory only.
        if name.contains('/') || name.contains('\\') || name == ".." {
            return Err(bad(format!("unexpected weights file name {name:?}")));
        }
        let f = dir.join(&name);
        if !f.is_file() {
            return Err(bad(format!("{name} is missing or unreadable")));
        }
        files.push(f);
    }
    Ok(files)
}

/// The digest of the weights at `path` (file or MLX directory), computed now. For building
/// references; providers use [`weights_digest`], which never blocks.
pub fn digest_now(path: &Path) -> io::Result<String> {
    if path.is_dir() {
        let mut manifest = Sha256::new();
        for f in dir_members(path)? {
            let name = f.file_name().and_then(|n| n.to_str()).unwrap_or("");
            manifest.update(format!("{name}\0{}\n", file_sha256_hex(&f)?).as_bytes());
        }
        Ok(format!("sha256:{}", hex::encode(manifest.finalize())))
    } else {
        Ok(format!("sha256:{}", file_sha256_hex(path)?))
    }
}

/// What a change to the weights shows up as: total size and latest modification time of the
/// file(s) that go into the digest.
fn cache_key(path: &Path) -> Option<(PathBuf, u64, Option<std::time::SystemTime>)> {
    let members = if path.is_dir() { dir_members(path).ok()? } else { vec![path.to_path_buf()] };
    let (mut size, mut latest) = (0u64, None);
    for m in &members {
        let meta = std::fs::metadata(m).ok()?;
        if !meta.is_file() {
            return None;
        }
        size += meta.len();
        latest = latest.max(meta.modified().ok());
    }
    Some((path.to_path_buf(), size, latest))
}

/// The digest of the served weights at `path`, or `""` if they can't be read here (a remote
/// engine, a missing path) **or aren't hashed yet**. Hashing multi-GB weights takes seconds, so it
/// never holds up model detection (and with it the announce): the first call starts it on a
/// background thread and returns `""`; detection re-runs every announce and picks it up then.
/// Cached per process by path, size and modification time, so replaced weights are hashed afresh.
pub fn weights_digest(path: &Path) -> String {
    weights_digest_with(path, |job| {
        std::thread::spawn(job);
    })
}

/// [`weights_digest`] with the way the hashing job is run injected (tests run it inline).
fn weights_digest_with(path: &Path, run: impl FnOnce(Box<dyn FnOnce() + Send>)) -> String {
    type Key = (PathBuf, u64, Option<std::time::SystemTime>);
    /// `None` while a hash is running (so it starts once), `Some("")` if it failed.
    static CACHE: OnceLock<Mutex<HashMap<Key, Option<String>>>> = OnceLock::new();
    if path.as_os_str().is_empty() {
        return String::new();
    }
    let Some(key) = cache_key(path) else { return String::new() };
    let cache = CACHE.get_or_init(Default::default);
    {
        let mut c = cache.lock().unwrap_or_else(|e| e.into_inner());
        match c.get(&key) {
            Some(Some(d)) => return d.clone(),
            Some(None) => return String::new(),
            None => {
                c.insert(key.clone(), None);
            }
        }
    }
    let (owned, lookup) = (path.to_path_buf(), key.clone());
    run(Box::new(move || {
        let started = std::time::Instant::now();
        let digest = digest_now(&owned).unwrap_or_default();
        tracing::info!(bytes = key.1, ms = started.elapsed().as_millis() as u64, ok = !digest.is_empty(), "hashed the served weights");
        cache.lock().unwrap_or_else(|e| e.into_inner()).insert(key, Some(digest));
    }));
    // Done already when run inline; on a thread, usually not yet.
    cache.lock().unwrap_or_else(|e| e.into_inner()).get(&lookup).cloned().flatten().unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("oh-weights-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn a_file_digest_is_its_sha256_hashed_once_in_the_background() {
        let dir = tmp("file");
        let path = dir.join("m.gguf");
        std::fs::write(&path, b"abc").unwrap();
        let abc = "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        assert_eq!(digest_now(&path).unwrap(), abc);
        let inline = |job: Box<dyn FnOnce() + Send>| job();
        assert_eq!(weights_digest_with(&path, inline), abc);
        assert_eq!(weights_digest_with(&path, |_| panic!("cached: no second hash")), abc);
        // In the background: nothing yet (detection doesn't wait), started once, then cached.
        let later = dir.join("later.gguf");
        std::fs::write(&later, b"abc").unwrap();
        let mut job = None;
        assert_eq!(weights_digest_with(&later, |j| job = Some(j)), "");
        assert_eq!(weights_digest_with(&later, |_| panic!("already running")), "");
        job.expect("hash started")();
        assert_eq!(weights_digest(&later), abc);
        std::fs::write(&path, b"abcd").unwrap(); // changed weights (another size) are hashed afresh
        assert_ne!(weights_digest_with(&path, inline), abc);
        assert_eq!(weights_digest(Path::new("")), "");
        assert_eq!(weights_digest(Path::new("/nonexistent/m.gguf")), "");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_mlx_directory_digest_covers_its_config_and_model_weights_only() {
        let dir = tmp("mlx");
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        std::fs::write(dir.join("model-00002-of-00002.safetensors"), b"two").unwrap();
        std::fs::write(dir.join("model-00001-of-00002.safetensors"), b"one").unwrap();
        std::fs::write(dir.join("tokenizer.json"), b"not part of it").unwrap();
        let d = digest_now(&dir).unwrap();
        // The documented manifest, by hand.
        let line = |n: &str, c: &[u8]| format!("{n}\0{}\n", hex::encode(Sha256::digest(c)));
        let manifest = [line("config.json", b"{}"), line("model-00001-of-00002.safetensors", b"one"), line("model-00002-of-00002.safetensors", b"two")].concat();
        assert_eq!(d, format!("sha256:{}", hex::encode(Sha256::digest(manifest.as_bytes()))));
        // The tokenizer doesn't count; the config and the weights do.
        std::fs::write(dir.join("tokenizer.json"), b"changed").unwrap();
        assert_eq!(digest_now(&dir).unwrap(), d);
        // Nor does a stray safetensors file next to the model's (a LoRA adapter).
        std::fs::write(dir.join("adapters.safetensors"), b"lora").unwrap();
        assert_eq!(digest_now(&dir).unwrap(), d);
        // With an index, exactly the shards it lists: the same files, the same digest.
        let index = r#"{"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors", "c": "model-00001-of-00002.safetensors"}}"#;
        std::fs::write(dir.join("model.safetensors.index.json"), index).unwrap();
        assert_eq!(digest_now(&dir).unwrap(), d);
        // A listed shard that can't be read (a broken link of a half-finished download): no digest.
        std::fs::remove_file(dir.join("model-00002-of-00002.safetensors")).unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(dir.join("nowhere"), dir.join("model-00002-of-00002.safetensors")).unwrap();
        assert!(digest_now(&dir).is_err());
        std::fs::remove_file(dir.join("model-00002-of-00002.safetensors")).ok();
        std::fs::write(dir.join("model-00002-of-00002.safetensors"), b"two").unwrap();
        std::fs::write(dir.join("model.safetensors.index.json"), r#"{"weight_map": {"a": "../escape.safetensors"}}"#).unwrap();
        assert!(digest_now(&dir).is_err(), "shard names stay inside the directory");
        std::fs::remove_file(dir.join("model.safetensors.index.json")).unwrap();
        assert_eq!(digest_now(&dir).unwrap(), d);
        std::fs::write(dir.join("config.json"), b"{\"bits\": 8}").unwrap();
        assert_ne!(digest_now(&dir).unwrap(), d);
        // A directory that isn't an MLX model has no digest.
        let empty = tmp("empty");
        assert!(digest_now(&empty).is_err());
        assert_eq!(weights_digest(&empty), "");
        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_dir_all(&empty).ok();
    }
}
