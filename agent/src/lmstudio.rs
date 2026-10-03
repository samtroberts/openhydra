// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Where LM Studio keeps a model's weights: `lms ls --json` maps the model key LM Studio's API
//! serves under to a path below its models folder (`downloadsFolder` in `~/.lmstudio/settings.json`).
//! A GGUF model is one file; an MLX model is a directory. Used by the side-scorer (LM Studio can't
//! score a prompt itself) and for the weights digest references are matched to.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::Deserialize;

/// How long a listing is reused (`lms ls` is a process launch).
const CACHE_FOR: Duration = Duration::from_secs(60);

/// One model LM Studio has on disk.
#[derive(Debug, Clone, PartialEq)]
pub struct LmsModel {
    /// The key LM Studio's API serves it under (the model id in `/v1/models`).
    pub key: String,
    /// Absolute path of its weights: a `.gguf` file, or an MLX model directory.
    pub path: PathBuf,
    /// `gguf` or `safetensors` (MLX).
    pub format: String,
}

#[derive(Deserialize)]
struct Entry {
    #[serde(rename = "modelKey", default)]
    key: String,
    #[serde(default)]
    path: String,
    #[serde(default)]
    format: String,
    #[serde(rename = "type", default)]
    kind: String,
}

/// Parse `lms ls --json` against the models folder `root` (LLMs only).
pub fn parse_listing(json: &str, root: &Path) -> Vec<LmsModel> {
    serde_json::from_str::<Vec<Entry>>(json)
        .unwrap_or_default()
        .into_iter()
        .filter(|e| e.kind == "llm" && !e.key.is_empty() && !e.path.is_empty())
        .map(|e| LmsModel { key: e.key, path: root.join(e.path), format: e.format })
        .collect()
}

fn home() -> Option<PathBuf> {
    std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(PathBuf::from)
}

/// LM Studio's models folder: `downloadsFolder` from its settings, else `~/.lmstudio/models`.
fn models_root() -> Option<PathBuf> {
    let lms = home()?.join(".lmstudio");
    let configured = std::fs::read_to_string(lms.join("settings.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| v.get("downloadsFolder").and_then(|d| d.as_str()).map(PathBuf::from));
    Some(configured.unwrap_or_else(|| lms.join("models")))
}

/// The `lms` CLI: `~/.lmstudio/bin/lms`, else on `PATH`.
fn lms_binary() -> Option<PathBuf> {
    let name = if cfg!(windows) { "lms.exe" } else { "lms" };
    let bundled = home()?.join(".lmstudio").join("bin").join(name);
    if bundled.is_file() {
        return Some(bundled);
    }
    std::env::var_os("PATH").and_then(|p| std::env::split_paths(&p).map(|d| d.join(name)).find(|c| c.is_file()))
}

/// The models LM Studio has on disk (empty if `lms` isn't available), cached for [`CACHE_FOR`].
pub fn models() -> Vec<LmsModel> {
    static CACHE: Mutex<Option<(Instant, Vec<LmsModel>)>> = Mutex::new(None);
    let mut c = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((at, list)) = c.as_ref() {
        if at.elapsed() < CACHE_FOR {
            return list.clone();
        }
    }
    let list = (|| {
        let stdout = run_with_timeout(&lms_binary()?, &["ls", "--json"], LMS_TIMEOUT)?;
        Some(parse_listing(&String::from_utf8_lossy(&stdout), &models_root()?))
    })()
    .unwrap_or_default();
    *c = Some((Instant::now(), list.clone()));
    list
}

/// How long `lms ls` may take. It runs under the cache lock that announces and challenges wait on,
/// and `lms` can hang (e.g. while it wakes the LM Studio service).
const LMS_TIMEOUT: Duration = Duration::from_secs(15);

/// `program args`' stdout if it exits successfully within `timeout`; else it is killed and `None`.
fn run_with_timeout(program: &Path, args: &[&str], timeout: Duration) -> Option<Vec<u8>> {
    use std::io::Read;
    let mut child = std::process::Command::new(program)
        .args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()?;
    // Read on another thread, so a large listing can't fill the pipe and stall the child; the result
    // is awaited with the same deadline (a process it left running could hold the pipe open).
    let mut pipe = child.stdout.take()?;
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = tx.send(pipe.read_to_end(&mut buf).ok().map(|_| buf));
    });
    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(50)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    };
    let out = rx.recv_timeout(deadline.saturating_duration_since(Instant::now()) + Duration::from_secs(1)).ok()??;
    status.success().then_some(out)
}

/// The local weights of the LM Studio model served as `key`, when they exist here.
pub fn weights_path(key: &str) -> Option<PathBuf> {
    models().into_iter().find(|m| m.key == key).map(|m| m.path).filter(|p| p.exists())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_listing_maps_api_keys_to_weights_under_the_models_folder() {
        let json = r#"[
            {"type":"llm","modelKey":"qwen3-0.6b-mlx","path":"lmstudio-community/Qwen3-0.6B-MLX-4bit","format":"safetensors"},
            {"type":"llm","modelKey":"qwen2.5-0.5b-instruct","path":"Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf","format":"gguf"},
            {"type":"embedding","modelKey":"nomic","path":"nomic-ai/x.gguf","format":"gguf"}
        ]"#;
        let m = parse_listing(json, Path::new("/m"));
        assert_eq!(m.len(), 2, "LLMs only");
        assert_eq!(m[0], LmsModel { key: "qwen3-0.6b-mlx".into(), path: "/m/lmstudio-community/Qwen3-0.6B-MLX-4bit".into(), format: "safetensors".into() });
        assert_eq!(m[1].path, Path::new("/m/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf"));
        assert!(parse_listing("not json", Path::new("/m")).is_empty());
    }
}
