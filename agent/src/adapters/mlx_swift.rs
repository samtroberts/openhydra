// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! The built-in Mac engine (ATTESTED_ENGINE_TIER_PLAN_v1 Phase 1a): Apple's MLX, via mlx-swift-lm,
//! running inside this process through the `openhydra-mlx-engine` crate. It serves one MLX model folder
//! (`--model-dir`) one request at a time, answers prompt-scoring challenges itself, and reports the same
//! weights digest an LM Studio provider of that folder would ([`crate::weights`]).
//!
//! The digest covers exactly the weights MLX loads: the engine loads only the files named here (the
//! digest's members plus the config, tokenizer and template files) through a private folder of links,
//! so a stray `*.safetensors` in the model folder (which MLX's loader would otherwise take) is ignored and
//! reported. The digest is computed from the engine's own hashes, taken before and after loading.
//!
//! The context a request may use is bounded by memory, not just the model's: the KV cache a full context
//! needs is worked out from `config.json` and fitted into Metal's recommended working set beside the
//! weights (optionally lowered with `--max-context`).
//!
//! What it proves today: nothing more than any engine's report. The digests are this process's own
//! word until Phase 2 (a hardware chain over the running binary) exists, so nothing here raises a tier
//! or shows a label; the digest is disclosed like every other engine's.
//!
//! Only built with `--features mlx-swift` on Apple Silicon, after `engine-mlx/build.sh`; executables need
//! the AOT kernel library beside them (`engine-mlx/stage.sh`).

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;

use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

use openhydra_mlx_engine::{Engine, EngineError, PINNED_METALLIB_SHA256};
use openhydra_protocol::model_id::{canonical_model_id, chat_template_hash, normalize_chat_template, parse_hf_model_name};

use crate::adapter::{AdapterError, ChatMessage, DetectedModel, EngineAdapter, EngineMetrics, InferenceRequest, ServeOutcome, ToolCall};

/// How often a serve checks the caller's stop signal while the engine generates.
const STOP_POLL: Duration = Duration::from_millis(20);

/// Files besides the weights that the engine needs (config, tokenizer, template), loaded when present.
const SUPPORT_FILES: [&str; 12] = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "model.safetensors.index.json",
];

/// Memory kept free beside the weights and the KV cache (activations, a prefill step's logits, MLX's
/// buffer cache), and the share of Metal's recommended working set the engine plans to use.
const MEMORY_HEADROOM: u64 = 1 << 30;
const WORKING_SET_SHARE: f64 = 0.9;

/// The least context worth serving; a model whose memory budget is smaller is refused.
const MIN_CONTEXT: u64 = 2048;

/// Requests are refused when their JSON exceeds this many bytes per allowed context token (the engine
/// applies the same bound).
const MAX_BYTES_PER_CONTEXT_TOKEN: u64 = 8;

/// The files besides the weights that decide how the engine reads and prompts the model, in
/// `config_sha256`. Docs and other leftovers in the folder don't change it.
const CONFIG_FILES: [&str; 8] = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
];

#[derive(Debug, Deserialize)]
struct ManifestFile {
    name: String,
    size: u64,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    files: Vec<ManifestFile>,
    metallib_sha256: String,
    #[serde(default)]
    model_type: String,
    vocab_size: u64,
    context_length: u64,
    #[serde(default)]
    gpu_working_set: u64,
    #[serde(default)]
    mlx_swift: String,
    #[serde(default)]
    mlx_swift_lm: String,
}

#[derive(Debug, Deserialize)]
struct GenerateResult {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
    #[serde(default)]
    stop_reason: String,
    #[serde(default)]
    tool_calls: Vec<Value>,
    #[serde(default)]
    prompt_ms: f64,
    #[serde(default)]
    generate_ms: f64,
}

/// One loaded MLX model folder.
pub struct MlxSwiftAdapter {
    engine: Engine,
    model: DetectedModel,
    vocab_size: u64,
    /// The context a request may use: the model's, bounded by the memory budget and `--max-context`.
    pub context_length: u64,
    /// `*.safetensors` in the folder that aren't the model's own weights, and so weren't loaded.
    pub ignored_weights: Vec<String>,
    /// SHA-256 over the files that decide how the model is read and prompted ([`CONFIG_FILES`]), the
    /// engine and library versions, and the AOT kernel library — reported data, like the weights digest.
    pub config_sha256: String,
}

impl MlxSwiftAdapter {
    /// Load `dir` into the engine. `name` is the handle consumers address it by; by default the Hugging
    /// Face id the folder came from (`org/name` from a hub cache path or an LM Studio models folder), else
    /// the folder's name.
    pub fn load(dir: &Path, name: Option<&str>, max_context: Option<u64>) -> Result<Self, String> {
        let dir = dir.canonicalize().map_err(|e| format!("{}: {e}", dir.display()))?;
        // The AOT kernel library, before MLX loads it (and again, from the engine's own reading, after).
        for path in metallib_paths()? {
            check_metallib(&path)?;
        }
        let members = member_names(&dir)?;
        let expected: BTreeSet<String> = members.iter().filter(|n| n.ends_with(".safetensors")).cloned().collect();
        let ignored_weights: Vec<String> = all_safetensors(&dir)?.difference(&expected).cloned().collect();
        let index_sha = std::fs::read(dir.join("model.safetensors.index.json")).ok().map(|b| hex::encode(Sha256::digest(&b)));
        let files = load_list(&dir, &members);

        let engine = Engine::load(&dir, &files).map_err(|e| e.to_string())?;
        let manifest: Manifest =
            serde_json::from_str(engine.manifest()).map_err(|e| format!("engine manifest: {e}"))?;
        if manifest.metallib_sha256 != PINNED_METALLIB_SHA256 {
            return Err(metallib_mismatch(&manifest.metallib_sha256));
        }
        // The index that named the members must be the one the engine hashed, unchanged since.
        let loaded_index = manifest.files.iter().find(|f| f.name == "model.safetensors.index.json").map(|f| f.sha256.clone());
        if loaded_index != index_sha || member_names(&dir)? != members {
            return Err("the model's weights index changed while loading".into());
        }

        let weight_digest = digest_from_manifest(&members, &manifest)?;
        let template = effective_chat_template(&dir);
        let config_sha256 = config_digest(&manifest, &template);
        let by_name: HashMap<&str, u64> = manifest.files.iter().map(|f| (f.name.as_str(), f.size)).collect();
        let size_bytes: u64 = members.iter().filter_map(|m| by_name.get(m.as_str())).sum();
        let config: Value = std::fs::read(dir.join("config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default();
        let budget = context_budget(&config, size_bytes, manifest.gpu_working_set);
        let context_length = [Some(manifest.context_length), budget, max_context].into_iter().flatten().min().unwrap_or(0);
        if context_length < MIN_CONTEXT {
            return Err(if max_context == Some(context_length) {
                format!("--max-context {context_length} is below the {MIN_CONTEXT} tokens the built-in engine needs")
            } else if budget == Some(context_length) {
                format!(
                    "this Mac's GPU memory leaves room for a {context_length}-token context beside the model; at \
                     least {MIN_CONTEXT} is needed (a smaller model, or a lower quantisation)"
                )
            } else {
                format!("the model's context is {context_length} tokens; the built-in engine needs at least {MIN_CONTEXT}")
            });
        }
        let engine_ref = name.map(str::to_string).unwrap_or_else(|| model_name(&dir));
        let (family, params, quant, canonical_id) = canonicalise(&dir, &engine_ref, &template);
        let model = DetectedModel {
            engine_ref,
            canonical_id,
            family: if family.is_empty() { manifest.model_type.clone() } else { family },
            params,
            quant,
            size_bytes,
            weight_digest,
        };
        Ok(MlxSwiftAdapter {
            engine,
            model,
            vocab_size: manifest.vocab_size,
            context_length,
            ignored_weights,
            config_sha256,
        })
    }

    /// The model this adapter serves.
    pub fn model(&self) -> &DetectedModel {
        &self.model
    }
}

/// `mlx.metallib` where MLX may look for it first: beside the executable as invoked and beside its real
/// path (they differ when it runs through a symlink). Both must hold the pinned AOT kernel library.
fn metallib_paths() -> Result<Vec<PathBuf>, String> {
    let exe = std::env::current_exe().map_err(|e| format!("cannot locate this executable: {e}"))?;
    let mut paths = vec![exe.with_file_name("mlx.metallib")];
    if let Ok(real) = exe.canonicalize() {
        let p = real.with_file_name("mlx.metallib");
        if !paths.contains(&p) {
            paths.push(p);
        }
    }
    Ok(paths)
}

/// Refuse an AOT kernel library other than the one this binary was built with.
fn check_metallib(path: &Path) -> Result<(), String> {
    let bytes = std::fs::read(path).map_err(|_| metallib_mismatch(""))?;
    let digest = hex::encode(Sha256::digest(&bytes));
    if digest == PINNED_METALLIB_SHA256 {
        Ok(())
    } else {
        Err(metallib_mismatch(&digest))
    }
}

fn metallib_mismatch(found: &str) -> String {
    format!(
        "the Metal kernel library beside this binary (mlx.metallib, sha256 {}) is not the one it was built with \
         ({PINNED_METALLIB_SHA256}); reinstall, or stage it with engine-mlx/stage.sh",
        if found.is_empty() { "missing" } else { found },
    )
}

/// The digest's members ([`crate::weights::dir_members`]) by file name.
fn member_names(dir: &Path) -> Result<Vec<String>, String> {
    let members = crate::weights::dir_members(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    Ok(members.iter().filter_map(|m| m.file_name().and_then(|n| n.to_str()).map(String::from)).collect())
}

/// Every `*.safetensors` under `dir` as MLX's loader sees them (recursively, hidden files included),
/// relative to `dir`.
fn all_safetensors(dir: &Path) -> Result<BTreeSet<String>, String> {
    fn walk(root: &Path, at: &Path, out: &mut BTreeSet<String>) -> std::io::Result<()> {
        for entry in std::fs::read_dir(at)? {
            let entry = entry?;
            let path = entry.path();
            if entry.file_type()?.is_dir() {
                walk(root, &path, out)?;
            } else if path.extension().is_some_and(|x| x == "safetensors") {
                if let Ok(rel) = path.strip_prefix(root) {
                    out.insert(rel.to_string_lossy().into_owned());
                }
            }
        }
        Ok(())
    }
    let mut out = BTreeSet::new();
    walk(dir, dir, &mut out).map_err(|e| format!("{}: {e}", dir.display()))?;
    Ok(out)
}

/// What the engine loads: the digest's members plus the support files present.
fn load_list(dir: &Path, members: &[String]) -> Vec<String> {
    let mut files: BTreeSet<String> = members.iter().cloned().collect();
    for name in SUPPORT_FILES {
        if dir.join(name).is_file() {
            files.insert(name.to_string());
        }
    }
    files.into_iter().collect()
}

/// The largest context whose KV cache fits Metal's recommended working set beside the weights, from the
/// model's `config.json` (or its `text_config`): 2 (K and V) × KV heads × head size × 2 bytes per
/// attention layer per token. Hybrid models count only their full-attention layers. `None` when the
/// config doesn't say enough or the engine reported no working set.
fn context_budget(config: &Value, weights_bytes: u64, working_set: u64) -> Option<u64> {
    if working_set == 0 {
        return None;
    }
    let text = config.get("text_config").filter(|t| t.is_object()).unwrap_or(config);
    let get = |key: &str| text.get(key).and_then(Value::as_u64);
    let layers = get("num_hidden_layers")?;
    let heads = get("num_attention_heads")?;
    let kv_heads = get("num_key_value_heads").unwrap_or(heads);
    let head_dim = get("head_dim").or_else(|| Some(get("hidden_size")? / heads.max(1)))?;
    let attention_layers = match (text.get("layer_types").and_then(Value::as_array), get("full_attention_interval")) {
        (Some(types), _) => types.iter().filter(|t| t.as_str() != Some("linear_attention")).count() as u64,
        (None, Some(every)) if every > 0 => layers / every,
        _ => layers,
    };
    let per_token = attention_layers * 2 * kv_heads * head_dim * 2;
    if per_token == 0 {
        return None;
    }
    let usable = (working_set as f64 * WORKING_SET_SHARE) as u64;
    Some(usable.saturating_sub(weights_bytes).saturating_sub(MEMORY_HEADROOM) / per_token)
}

/// The [`crate::weights`] digest of the folder, from the engine's own hashes of the files it loaded
/// (taken before and after loading), so the reported digest is of what was loaded, not a second read.
fn digest_from_manifest(members: &[String], manifest: &Manifest) -> Result<String, String> {
    let by_name: HashMap<&str, &str> = manifest.files.iter().map(|f| (f.name.as_str(), f.sha256.as_str())).collect();
    let mut digest = Sha256::new();
    for name in members {
        let sha = by_name.get(name.as_str()).ok_or_else(|| format!("the engine did not load {name}"))?;
        digest.update(format!("{name}\0{sha}\n").as_bytes());
    }
    Ok(format!("sha256:{}", hex::encode(digest.finalize())))
}

/// [`MlxSwiftAdapter::config_sha256`]: `<name>\0<sha256>\n` per present config file, then the engine,
/// library versions, the AOT kernel library and the chat template the engine uses.
fn config_digest(manifest: &Manifest, template: &str) -> String {
    let by_name: HashMap<&str, &str> = manifest.files.iter().map(|f| (f.name.as_str(), f.sha256.as_str())).collect();
    let mut digest = Sha256::new();
    for name in CONFIG_FILES {
        if let Some(sha) = by_name.get(name) {
            digest.update(format!("{name}\0{sha}\n").as_bytes());
        }
    }
    digest.update(format!("engine\0mlx-swift {} mlx-swift-lm {}\n", manifest.mlx_swift, manifest.mlx_swift_lm).as_bytes());
    digest.update(format!("metallib\0{}\n", manifest.metallib_sha256).as_bytes());
    digest.update(format!("chat_template\0{}\n", hex::encode(Sha256::digest(template.as_bytes()))).as_bytes());
    format!("sha256:{}", hex::encode(digest.finalize()))
}

/// `org/name` from a Hugging Face cache path (`…/models--org--name/snapshots/<rev>`) or an LM Studio
/// models folder (`…/models/org/name`), else the folder's name.
fn model_name(dir: &Path) -> String {
    for part in dir.components().rev() {
        let part = part.as_os_str().to_string_lossy();
        if let Some(rest) = part.strip_prefix("models--") {
            if let Some((org, name)) = rest.split_once("--") {
                return format!("{org}/{name}");
            }
        }
    }
    let name = dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let mut up = dir.ancestors().skip(1);
    if let (Some(org), Some(models)) = (up.next(), up.next()) {
        if models.file_name().is_some_and(|m| m == "models") {
            if let Some(org) = org.file_name() {
                return format!("{}/{name}", org.to_string_lossy());
            }
        }
    }
    name
}

/// The chat template the engine renders with, by swift-transformers' rule: `chat_template.jinja` when it
/// exists (and nothing from `chat_template.json` even when it isn't UTF-8), else a string in
/// `chat_template.json`, else `tokenizer_config.json`'s (a list of named templates uses `default`; with
/// tools the engine picks `tool_use`, which the canonical id doesn't follow).
fn effective_chat_template(dir: &Path) -> String {
    let jinja = dir.join("chat_template.jinja");
    let read_json = |file: &str| -> Option<Value> { serde_json::from_slice(&std::fs::read(dir.join(file)).ok()?).ok() };
    if jinja.exists() {
        if let Ok(t) = std::fs::read_to_string(&jinja) {
            return t;
        }
    } else if let Some(t) = read_json("chat_template.json").and_then(|v| v.get("chat_template")?.as_str().map(str::to_string)) {
        return t;
    }
    match read_json("tokenizer_config.json").and_then(|v| v.get("chat_template").cloned()) {
        Some(Value::String(t)) => t,
        Some(Value::Array(named)) => named
            .iter()
            .find(|t| t.get("name").and_then(Value::as_str) == Some("default"))
            .and_then(|t| t.get("template")?.as_str().map(str::to_string))
            .unwrap_or_default(),
        _ => String::new(),
    }
}

/// Family, params, quant and the canonical id (empty when any part is unknown), as the other adapters
/// derive them: family and params from the model's name, quant from `config.json`'s quantisation (or
/// its dtype), and the chat template the engine uses.
fn canonicalise(dir: &Path, engine_ref: &str, template: &str) -> (String, String, String, String) {
    let (family, params, _variants) = parse_hf_model_name(engine_ref);
    let config: Value = std::fs::read(dir.join("config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();
    let quant = config
        .get("quantization")
        .and_then(|q| q.get("bits"))
        .and_then(Value::as_u64)
        .map(|bits| format!("{bits}bit"))
        .or_else(|| config.get("torch_dtype").or_else(|| config.get("dtype")).and_then(Value::as_str).map(str::to_string))
        .unwrap_or_default();
    let canonical = if family.is_empty() || params.is_empty() || params == "unknown" || quant.is_empty()
        || normalize_chat_template(template).is_empty()
    {
        String::new()
    } else {
        match canonical_model_id(&family, &params, &quant, &chat_template_hash(template)) {
            Ok(c) => format!("{}/{}/{}/{}", c.family, c.params, c.quant, c.template_hash),
            Err(_) => String::new(),
        }
    };
    (family, params, quant, canonical)
}

/// The messages as the chat templates expect them: an assistant turn's earlier tool calls carry their
/// `arguments` as an object (OpenAI sends a JSON string; templates iterate or `tojson` it).
fn messages_json(messages: &[ChatMessage]) -> Result<String, AdapterError> {
    let mut value = serde_json::to_value(messages).map_err(|e| AdapterError::Parse(e.to_string()))?;
    for message in value.as_array_mut().into_iter().flatten() {
        for call in message.get_mut("tool_calls").and_then(Value::as_array_mut).into_iter().flatten() {
            if let Some(function) = call.get_mut("function") {
                if let Some(Value::String(args)) = function.get("arguments") {
                    if let Ok(parsed @ Value::Object(_)) = serde_json::from_str::<Value>(args) {
                        function["arguments"] = parsed;
                    }
                }
            }
        }
    }
    Ok(value.to_string())
}

/// The engine's generation parameters: the allowed context, then only what the request sets (the engine
/// falls back to the model's `generation_config.json` for sampling, and to the rest of the context, at
/// most 8192, for length).
fn params_json(request: &InferenceRequest, max_context: u64) -> String {
    let mut params = serde_json::Map::new();
    params.insert("max_context".into(), max_context.into());
    if let Some(n) = request.max_tokens {
        params.insert("max_tokens".into(), n.into());
    }
    if let Some(t) = request.temperature {
        params.insert("temperature".into(), t.into());
    }
    if let Some(seed) = request.seed {
        // The engine takes the seed as a signed 64-bit JSON number; keep the bit pattern.
        params.insert("seed".into(), (seed as i64).into());
    }
    if !request.tools.is_empty() {
        params.insert("tools".into(), Value::Array(request.tools.clone()));
    }
    if let Some(think) = request.think {
        params.insert("enable_thinking".into(), think.into());
    }
    Value::Object(params).to_string()
}

impl EngineAdapter for MlxSwiftAdapter {
    fn engine_name(&self) -> &'static str {
        "mlx-swift"
    }

    fn detect_models(&self) -> Result<Vec<DetectedModel>, AdapterError> {
        Ok(vec![self.model.clone()])
    }

    fn serve_stream(
        &self,
        request: &InferenceRequest,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<ServeOutcome, AdapterError> {
        if request.model_ref != self.model.engine_ref {
            return Err(AdapterError::Unsupported(format!("this engine serves {}", self.model.engine_ref)));
        }
        let messages = messages_json(&request.messages)?;
        let params = params_json(request, self.context_length);
        let bytes = (messages.len() + params.len()) as u64;
        if bytes > self.context_length * MAX_BYTES_PER_CONTEXT_TOKEN {
            return Err(AdapterError::Unsupported(format!(
                "the request is {bytes} bytes; this engine takes at most {}",
                self.context_length * MAX_BYTES_PER_CONTEXT_TOKEN
            )));
        }
        // The engine streams on its own thread; text comes back here over a channel so `on_delta` stays
        // on this thread, which also watches the stop signal and cancels the engine within STOP_POLL. A
        // panic in `on_delta` cancels the engine too, and is re-raised once it has finished.
        let (tx, rx) = mpsc::channel::<String>();
        let (raw, panic) = std::thread::scope(|scope| {
            let engine = &self.engine;
            let generation = scope.spawn(move || {
                engine.generate(&messages, &params, &mut |piece: &str| tx.send(piece.to_string()).is_ok())
            });
            let mut cancelled = false;
            let mut panic = None;
            loop {
                match rx.recv_timeout(STOP_POLL) {
                    Ok(piece) if !cancelled => {
                        if let Err(p) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| on_delta(&piece))) {
                            panic = Some(p);
                            cancelled = true;
                            self.engine.cancel();
                        }
                    }
                    Ok(_) => {}
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                }
                if !cancelled && request.stop.is_set() {
                    cancelled = true;
                    self.engine.cancel();
                }
            }
            let raw = generation.join().unwrap_or_else(|_| Err(EngineError::Failed("the engine thread panicked".into())));
            (raw, panic)
        });
        if let Some(p) = panic {
            std::panic::resume_unwind(p);
        }
        let raw = raw.map_err(|e| AdapterError::Provider(e.to_string()))?;
        let result: GenerateResult = serde_json::from_str(&raw).map_err(|e| AdapterError::Parse(e.to_string()))?;
        let tool_calls = result
            .tool_calls
            .into_iter()
            .map(serde_json::from_value::<ToolCall>)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| AdapterError::Parse(format!("tool call: {e}")))?;
        let ms_to_ns = |ms: f64| (ms.max(0.0) * 1e6) as u64;
        Ok(ServeOutcome {
            tokens: result.completion_tokens,
            done: result.stop_reason != "cancelled",
            engine: EngineMetrics {
                total_duration_ns: ms_to_ns(result.prompt_ms + result.generate_ms),
                load_duration_ns: 0,
                prompt_eval_count: result.prompt_tokens,
                prompt_eval_duration_ns: ms_to_ns(result.prompt_ms),
                eval_count: result.completion_tokens,
                eval_duration_ns: ms_to_ns(result.generate_ms),
            },
            tool_calls,
            hit_token_limit: result.stop_reason == "length",
        })
    }

    fn one_request_at_a_time(&self) -> bool {
        true
    }

    fn score_prompt(&self, model_ref: &str, token_ids: &[u32]) -> Result<Vec<f32>, AdapterError> {
        if model_ref != self.model.engine_ref {
            return Err(AdapterError::Unsupported(format!("this engine serves {}", self.model.engine_ref)));
        }
        if token_ids.len() as u64 > self.context_length {
            return Err(AdapterError::Unsupported(format!("{} ids is more than this model's context", token_ids.len())));
        }
        if let Some(bad) = token_ids.iter().find(|&&id| u64::from(id) >= self.vocab_size) {
            return Err(AdapterError::Unsupported(format!("token id {bad} is outside this model's vocabulary")));
        }
        self.engine.score(token_ids).map_err(|e| match e {
            EngineError::Busy(m) => AdapterError::Busy(m),
            EngineError::Failed(m) => AdapterError::Provider(m),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ohmlx-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn hub_cache_and_lm_studio_paths_name_the_model() {
        let p = Path::new("/u/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-8bit/snapshots/11de968");
        assert_eq!(model_name(p), "mlx-community/Qwen3-0.6B-8bit");
        assert_eq!(model_name(Path::new("/u/.lmstudio/models/lmstudio-community/Qwen3-0.6B-MLX-4bit")), "lmstudio-community/Qwen3-0.6B-MLX-4bit");
        assert_eq!(model_name(Path::new("/data/Qwen3-8B-4bit")), "Qwen3-8B-4bit");
    }

    #[test]
    fn foreign_or_missing_kernels_are_refused() {
        let dir = tmp("metallib");
        let lib = dir.join("mlx.metallib");
        assert!(check_metallib(&lib).unwrap_err().contains("missing"));
        std::fs::write(&lib, b"not the kernels").unwrap();
        assert!(check_metallib(&lib).unwrap_err().contains("is not the one it was built with"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    fn manifest(files: &[(&str, &str)]) -> Manifest {
        Manifest {
            files: files.iter().map(|(n, s)| ManifestFile { name: n.to_string(), size: 1, sha256: s.to_string() }).collect(),
            metallib_sha256: "kk".into(),
            model_type: String::new(),
            vocab_size: 10,
            context_length: 10,
            gpu_working_set: 0,
            mlx_swift: "0.31.4".into(),
            mlx_swift_lm: "3.31.3".into(),
        }
    }

    #[test]
    fn the_digest_follows_the_weights_rule_and_needs_every_member() {
        let m = manifest(&[("config.json", "aa"), ("model.safetensors", "bb"), ("tokenizer.json", "cc")]);
        let members = vec!["config.json".to_string(), "model.safetensors".to_string()];
        let expected = format!("sha256:{}", hex::encode(Sha256::digest(b"config.json\0aa\nmodel.safetensors\0bb\n")));
        assert_eq!(digest_from_manifest(&members, &m).unwrap(), expected);
        assert!(digest_from_manifest(&["model-00001-of-00002.safetensors".to_string()], &m).is_err());
    }

    #[test]
    fn the_config_digest_ignores_docs_and_follows_the_template() {
        let a = manifest(&[("config.json", "aa"), ("tokenizer.json", "cc")]);
        let b = manifest(&[("config.json", "aa"), ("tokenizer.json", "cc"), ("README.md", "dd")]);
        assert_eq!(config_digest(&a, "t"), config_digest(&b, "t"));
        assert_ne!(config_digest(&a, "t"), config_digest(&a, "u"));
        let c = manifest(&[("config.json", "aa"), ("tokenizer.json", "ce")]);
        assert_ne!(config_digest(&a, "t"), config_digest(&c, "t"));
    }

    #[test]
    fn only_the_members_and_support_files_are_loaded() {
        let dir = tmp("list");
        for f in ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "README.md", ".override.safetensors"] {
            std::fs::write(dir.join(f), b"x").unwrap();
        }
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("sub/x.safetensors"), b"x").unwrap();
        let members = member_names(&dir).unwrap();
        assert_eq!(load_list(&dir, &members), ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]);
        let expected: BTreeSet<String> = ["model.safetensors".to_string()].into();
        let ignored: Vec<String> = all_safetensors(&dir).unwrap().difference(&expected).cloned().collect();
        assert_eq!(ignored, [".override.safetensors", "sub/x.safetensors"]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_context_budget_fits_the_kv_cache_beside_the_weights() {
        let gib = 1u64 << 30;
        // Qwen3-0.6B: 28 layers × 2 × 8 KV heads × 128 × 2 bytes = 114,688 bytes a token.
        let qwen3 = serde_json::json!({"num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128});
        let budget = context_budget(&qwen3, gib / 2, 6 * gib).unwrap();
        assert_eq!(budget, (((6 * gib) as f64 * WORKING_SET_SHARE) as u64 - gib / 2 - MEMORY_HEADROOM) / 114_688);
        // Qwen3.5 (hybrid): only its full-attention layers keep a KV cache.
        let qwen35 = serde_json::json!({"text_config": {"num_hidden_layers": 4, "num_attention_heads": 8, "num_key_value_heads": 2,
            "head_dim": 256, "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"]}});
        assert_eq!(context_budget(&qwen35, 0, 2 * gib).unwrap(), (((2 * gib) as f64 * WORKING_SET_SHARE) as u64 - MEMORY_HEADROOM) / 2048);
        // Too big for the memory: no room at all.
        assert_eq!(context_budget(&qwen3, 8 * gib, 6 * gib), Some(0));
        assert_eq!(context_budget(&serde_json::json!({}), 0, 6 * gib), None);
        assert_eq!(context_budget(&qwen3, 0, 0), None);
    }

    #[test]
    fn the_engines_template_is_hashed() {
        let dir = tmp("template");
        std::fs::write(dir.join("tokenizer_config.json"), br#"{"chat_template":"from-config"}"#).unwrap();
        assert_eq!(effective_chat_template(&dir), "from-config");
        std::fs::write(dir.join("tokenizer_config.json"), br#"{"chat_template":[{"name":"tool_use","template":"t"},{"name":"default","template":"d"}]}"#).unwrap();
        assert_eq!(effective_chat_template(&dir), "d");
        // A list in chat_template.json isn't read by the engine: tokenizer_config's still applies.
        std::fs::write(dir.join("chat_template.json"), br#"{"chat_template":[{"name":"default","template":"j"}]}"#).unwrap();
        assert_eq!(effective_chat_template(&dir), "d");
        std::fs::write(dir.join("chat_template.json"), br#"{"chat_template":"from-json"}"#).unwrap();
        assert_eq!(effective_chat_template(&dir), "from-json");
        std::fs::write(dir.join("chat_template.jinja"), b"from-jinja").unwrap();
        assert_eq!(effective_chat_template(&dir), "from-jinja");
        // A .jinja that isn't UTF-8 makes the engine skip chat_template.json too.
        std::fs::write(dir.join("chat_template.jinja"), [0xff, 0xfe]).unwrap();
        assert_eq!(effective_chat_template(&dir), "d");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn quant_comes_from_the_config() {
        let dir = tmp("quant");
        let template = "{{ messages }}";
        std::fs::write(dir.join("config.json"), br#"{"quantization":{"bits":4,"group_size":64}}"#).unwrap();
        let (family, params, quant, canonical) = canonicalise(&dir, "mlx-community/Qwen3-0.6B-4bit", template);
        assert_eq!((family.as_str(), params.as_str(), quant.as_str()), ("qwen3", "0.6b", "4bit"));
        assert!(canonical.starts_with("qwen3/0.6b/int4/"), "{canonical}");
        std::fs::write(dir.join("config.json"), br#"{"dtype":"bfloat16"}"#).unwrap();
        assert_eq!(canonicalise(&dir, "Qwen3-0.6B", template).2, "bfloat16");
        std::fs::write(dir.join("config.json"), br#"{}"#).unwrap();
        assert_eq!(canonicalise(&dir, "Qwen3-0.6B", template).3, "");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn earlier_tool_calls_reach_the_template_as_objects() {
        let messages: Vec<ChatMessage> = serde_json::from_value(serde_json::json!([
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get", "arguments": "{\"city\":\"Paris\"}"}},
                {"id": "call_2", "type": "function", "function": {"name": "raw", "arguments": "not json"}}
            ]},
            {"role": "tool", "content": "sunny", "tool_call_id": "call_1"}
        ]))
        .unwrap();
        let v: Value = serde_json::from_str(&messages_json(&messages).unwrap()).unwrap();
        assert_eq!(v[1]["tool_calls"][0]["function"]["arguments"], serde_json::json!({"city": "Paris"}));
        assert_eq!(v[1]["tool_calls"][1]["function"]["arguments"], "not json");
        assert_eq!(v[2]["tool_call_id"], "call_1");
    }

    #[test]
    fn only_what_the_request_sets_is_passed() {
        let mut request = InferenceRequest {
            model_ref: "m".into(),
            messages: vec![],
            max_tokens: None,
            temperature: None,
            tools: vec![],
            think: None,
            seed: None,
            stop: Default::default(),
        };
        assert_eq!(params_json(&request, 4096), r#"{"max_context":4096}"#);
        request.max_tokens = Some(7);
        request.temperature = Some(0.0);
        request.seed = Some(u64::MAX);
        request.think = Some(false);
        let v: Value = serde_json::from_str(&params_json(&request, 4096)).unwrap();
        assert_eq!(v["max_tokens"], 7);
        assert_eq!(v["temperature"], 0.0);
        assert_eq!(v["seed"], -1);
        assert_eq!(v["enable_thinking"], false);
    }
}
