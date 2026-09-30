// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! `refbuild` — build a reference set for the teacher-forced log-prob identity check.
//!
//! Three steps, so the engine runs can happen on different machines (CUDA on one box, Metal on a
//! Mac, CPU anywhere) and the result is reproducible from the saved files:
//!
//! 1. `refbuild tokenize` — turn texts into token ids with the reference engine's own tokenizer
//!    (vLLM or llama.cpp `/tokenize`, no special tokens), dropping texts outside the length window.
//! 2. `refbuild score` — score every text teacher-forced on one engine configuration (vLLM, SGLang,
//!    or the OpenHydra scoring endpoint of patched llama.cpp / mistral.rs) and save the run.
//!    Run it once for the reference, once per honest configuration, once per substitute.
//! 3. `refbuild assemble` — combine the runs into a reference file: per-text honest bands and a
//!    cut-off calibrated between honest and substitute runs
//!    ([`openhydra_protocol::logprob::build_reference`]). Refuses when they don't separate.
//!
//! The output is what `serve --logprob-reference` loads. Keep it private: a text a provider has seen
//! can be answered from an offline honest run.

use std::path::{Path, PathBuf};

use clap::{Args, Subcommand};
use openhydra_agent::adapter::HttpClient;
use openhydra_agent::adapters::scoring::{score_prompt, ScoringDialect};
use openhydra_agent::http::ReqwestClient;
use openhydra_protocol::logprob::{build_reference, BuildParams, ModelIdentity, RunScores, MIN_TEXT_TOKENS};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[derive(Args)]
pub struct RefbuildArgs {
    #[command(subcommand)]
    action: RefbuildAction,
}

#[derive(Subcommand)]
enum RefbuildAction {
    /// Tokenize texts (a JSON array of strings) with the reference engine's tokenizer.
    Tokenize(TokenizeArgs),
    /// Score every tokenized text on one engine configuration and save the run.
    Score(ScoreArgs),
    /// Combine saved runs into a reference file with a calibrated cut-off.
    Assemble(AssembleArgs),
}

/// Which tokenize endpoint to call.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum TokenizerApi {
    /// vLLM `POST /tokenize` (`model`, `prompt`, `add_special_tokens`).
    Vllm,
    /// llama.cpp `POST /tokenize` (`content`, `add_special`).
    Llamacpp,
}

#[derive(Args)]
struct TokenizeArgs {
    /// Engine base URL, e.g. http://127.0.0.1:8080.
    #[arg(long)]
    engine: String,
    #[arg(long, value_enum)]
    api: TokenizerApi,
    /// Served model name (vLLM needs it).
    #[arg(long, default_value = "default")]
    model: String,
    /// JSON array of texts.
    #[arg(long)]
    texts: PathBuf,
    /// Drop texts shorter than this many tokens.
    #[arg(long, default_value_t = 32)]
    min_tokens: usize,
    /// Drop texts longer than this many tokens.
    #[arg(long, default_value_t = 512)]
    max_tokens: usize,
    #[arg(long)]
    out: PathBuf,
}

#[derive(Args)]
struct ScoreArgs {
    /// Engine base URL.
    #[arg(long)]
    engine: String,
    /// Scoring API: vllm, sglang, or openhydra (patched llama.cpp / mistral.rs).
    #[arg(long)]
    api: ScoringDialect,
    /// Served model name.
    #[arg(long, default_value = "default")]
    model: String,
    /// Token ids from `refbuild tokenize` (or a plain JSON array of id arrays).
    #[arg(long)]
    ids: PathBuf,
    /// A name for this configuration, e.g. "llama.cpp CUDA" or "llama.cpp CPU".
    #[arg(long)]
    label: String,
    #[arg(long)]
    out: PathBuf,
}

#[derive(Args)]
struct AssembleArgs {
    #[arg(long)]
    ids: PathBuf,
    /// The reference run (the claimed model on the trusted engine).
    #[arg(long)]
    reference: PathBuf,
    /// An honest run: the same weights on another engine / backend / setting (at least two).
    #[arg(long = "honest")]
    honest: Vec<PathBuf>,
    /// A substitute run: what a cheater might serve instead (cheaper quant, fine-tune).
    #[arg(long = "substitute")]
    substitutes: Vec<PathBuf>,
    /// Label for the reference, e.g. qwen3.5-9b@q4_k_m.
    #[arg(long)]
    label: String,
    /// Model identity the reference is for (family, params, quant) — how gateways match it.
    #[arg(long)]
    family: Option<String>,
    #[arg(long)]
    params: Option<String>,
    #[arg(long)]
    quant: Option<String>,
    /// Model names to match as well (or instead of an identity).
    #[arg(long = "handle")]
    handles: Vec<String>,
    /// Texts per verification round.
    #[arg(long, default_value_t = openhydra_protocol::logprob::DEFAULT_MIN_TEXTS)]
    min_texts: usize,
    /// Required ratio between the substitutes' low end and the honest high end.
    #[arg(long, default_value_t = 1.15)]
    min_gap: f64,
    /// Fraction of calibration rounds honest runs must pass (and substitutes fail).
    #[arg(long, default_value_t = 0.995)]
    coverage: f64,
    /// Build without substitute runs (the cut-off then only protects honest providers).
    #[arg(long)]
    allow_uncalibrated: bool,
    #[arg(long)]
    out: PathBuf,
    /// Where to write the calibration report (JSON).
    #[arg(long)]
    report: Option<PathBuf>,
}

pub fn run(args: RefbuildArgs) -> Result<(), String> {
    match args.action {
        RefbuildAction::Tokenize(a) => tokenize(a),
        RefbuildAction::Score(a) => score(a),
        RefbuildAction::Assemble(a) => assemble(a),
    }
}

fn read_json(path: &Path) -> Result<Value, String> {
    let raw = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&raw).map_err(|e| format!("{}: {e}", path.display()))
}

fn write_json(path: &Path, v: &Value) -> Result<(), String> {
    std::fs::write(path, serde_json::to_vec(v).map_err(|e| e.to_string())?).map_err(|e| format!("{}: {e}", path.display()))
}

/// SHA-256 over every text's length and token ids — ties runs to the exact ids they scored.
fn ids_digest(ids: &[Vec<u32>]) -> String {
    let mut h = Sha256::new();
    for text in ids {
        h.update((text.len() as u32).to_le_bytes());
        for id in text {
            h.update(id.to_le_bytes());
        }
    }
    hex::encode(h.finalize())
}

/// Token ids from a `tokenize` output (`{"ids": [...]}`) or a plain array of id arrays.
fn parse_ids(v: &Value) -> Result<Vec<Vec<u32>>, String> {
    let arr = v.get("ids").unwrap_or(v).as_array().ok_or("ids: expected an array of token-id arrays")?;
    arr.iter()
        .map(|t| {
            t.as_array()
                .ok_or("ids: expected token-id arrays")?
                .iter()
                .map(|x| x.as_u64().and_then(|n| u32::try_from(n).ok()).ok_or_else(|| "ids: bad token id".to_string()))
                .collect()
        })
        .collect()
}

fn tokenize_request(api: TokenizerApi, model: &str, text: &str) -> (&'static str, Value) {
    match api {
        TokenizerApi::Vllm => ("/tokenize", json!({"model": model, "prompt": text, "add_special_tokens": false})),
        TokenizerApi::Llamacpp => ("/tokenize", json!({"content": text, "add_special": false})),
    }
}

fn tokens_from_reply(reply: &str) -> Result<Vec<u32>, String> {
    let v: Value = serde_json::from_str(reply).map_err(|e| format!("tokenize reply: {e}"))?;
    v.get("tokens")
        .and_then(Value::as_array)
        .ok_or("tokenize reply: no tokens")?
        .iter()
        .map(|x| x.as_u64().and_then(|n| u32::try_from(n).ok()).ok_or_else(|| "tokenize reply: bad token".to_string()))
        .collect()
}

fn tokenize(a: TokenizeArgs) -> Result<(), String> {
    let texts: Vec<String> = serde_json::from_value(read_json(&a.texts)?).map_err(|e| format!("texts: {e}"))?;
    let min = a.min_tokens.max(MIN_TEXT_TOKENS);
    let max = a.max_tokens.min(openhydra_agent::prompt_score::MAX_CHALLENGE_TOKENS);
    let http = ReqwestClient::new().map_err(|e| e.to_string())?;
    let root = a.engine.trim_end_matches('/');
    let (mut ids, mut dropped) = (Vec::new(), 0usize);
    for text in &texts {
        let (path, body) = tokenize_request(a.api, &a.model, text);
        let reply = http.post_json(&format!("{root}{path}"), &body.to_string()).map_err(|e| e.to_string())?;
        let t = tokens_from_reply(&reply)?;
        if (min..=max).contains(&t.len()) {
            ids.push(t);
        } else {
            dropped += 1;
        }
    }
    let mut h = Sha256::new();
    for t in &texts {
        h.update(t.as_bytes());
        h.update([0]);
    }
    eprintln!("refbuild: tokenized {} texts, kept {} ({min}-{max} tokens), dropped {dropped}", texts.len(), ids.len());
    write_json(
        &a.out,
        &json!({"texts_sha256": hex::encode(h.finalize()), "tokenizer": {"engine": a.engine, "api": format!("{:?}", a.api)},
                "ids_sha256": ids_digest(&ids), "ids": ids}),
    )
}

fn score(a: ScoreArgs) -> Result<(), String> {
    let ids = parse_ids(&read_json(&a.ids)?)?;
    let http = ReqwestClient::new().map_err(|e| e.to_string())?;
    let root = a.engine.trim_end_matches('/').trim_end_matches("/v1");
    let started = std::time::Instant::now();
    let mut logprobs = Vec::with_capacity(ids.len());
    for (i, t) in ids.iter().enumerate() {
        let lps = score_prompt(&http, root, a.api, &a.model, t).map_err(|e| format!("text {i}: {e}"))?;
        logprobs.push(lps);
        if (i + 1) % 10 == 0 || i + 1 == ids.len() {
            eprintln!("refbuild: {} scored {}/{} ({:.1} s)", a.label, i + 1, ids.len(), started.elapsed().as_secs_f64());
        }
    }
    write_json(
        &a.out,
        &json!({"label": a.label, "engine": a.engine, "api": format!("{:?}", a.api), "model": a.model,
                "ids_sha256": ids_digest(&ids), "logprobs": logprobs}),
    )
}

/// Load a saved run and check it scored exactly these ids.
fn load_run(path: &Path, digest: &str) -> Result<RunScores, String> {
    let v = read_json(path)?;
    let got = v.get("ids_sha256").and_then(Value::as_str).unwrap_or_default();
    if got != digest {
        return Err(format!("{}: scored different token ids than --ids (sha256 {got} ≠ {digest})", path.display()));
    }
    let label = v.get("label").and_then(Value::as_str).unwrap_or("?").to_string();
    let logprobs: Vec<Vec<f32>> =
        serde_json::from_value(v.get("logprobs").cloned().unwrap_or(Value::Null)).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(RunScores { label, logprobs })
}

fn assemble(a: AssembleArgs) -> Result<(), String> {
    let ids = parse_ids(&read_json(&a.ids)?)?;
    let digest = ids_digest(&ids);
    let reference = load_run(&a.reference, &digest)?;
    let honest = a.honest.iter().map(|p| load_run(p, &digest)).collect::<Result<Vec<_>, _>>()?;
    let subs = a.substitutes.iter().map(|p| load_run(p, &digest)).collect::<Result<Vec<_>, _>>()?;
    if subs.is_empty() && !a.allow_uncalibrated {
        return Err("no --substitute runs: pass --allow-uncalibrated to build a cut-off that only protects honest providers".into());
    }
    let identity = match (&a.family, &a.params, &a.quant) {
        (Some(f), Some(p), Some(q)) => Some(ModelIdentity::new(f, p, q).ok_or("incomplete --family/--params/--quant")?),
        (None, None, None) => None,
        _ => return Err("give all of --family, --params and --quant, or none".into()),
    };
    if identity.is_none() && a.handles.is_empty() {
        return Err("give a model identity (--family/--params/--quant) or at least one --handle".into());
    }
    let mut params = BuildParams::new(a.label.clone(), identity.clone());
    params.min_texts = a.min_texts;
    params.min_gap = a.min_gap;
    params.coverage = a.coverage;
    let (set, cal) = build_reference(&ids, &reference, &honest, &subs, &params).map_err(|e| format!("{e:?}"))?;

    let mut out = json!({
        "model_key": set.model_key,
        "handles": a.handles,
        "cutoff": set.cutoff,
        "min_texts": set.min_texts,
        "texts": set.texts.iter().map(|t| json!({"id": t.id, "token_ids": t.token_ids, "logprobs": t.logprobs, "band": t.band})).collect::<Vec<_>>(),
    });
    if let Some(id) = &identity {
        out["model"] = json!({"family": id.family, "params": id.params, "quant": id.quant});
    }
    // Round-trip through the loader the gateway uses, so a file that won't load is never written.
    openhydra_agent::prompt_score::parse_reference(&out.to_string())?;
    write_json(&a.out, &out)?;

    let mut bands: Vec<f64> = set.texts.iter().map(|t| t.band).collect();
    bands.sort_by(|x, y| x.partial_cmp(y).unwrap());
    eprintln!(
        "refbuild: {} — {} texts, bands {:.4}-{:.4} nats (median {:.4})",
        set.model_key,
        set.texts.len(),
        bands[0],
        bands[bands.len() - 1],
        bands[bands.len() / 2]
    );
    for (label, agg) in &cal.honest {
        eprintln!("  honest     {label:28} x{agg:.2} (against a band without it)");
    }
    for (label, agg) in &cal.substitutes {
        eprintln!("  substitute {label:28} x{agg:.2}");
    }
    match cal.substitute_low {
        Some(low) => eprintln!(
            "  rounds of {}: honest ≤ x{:.2}, substitutes ≥ x{low:.2} ({:.1}% coverage) → cut-off x{:.2}",
            set.min_texts,
            cal.honest_high,
            a.coverage * 100.0,
            cal.cutoff
        ),
        None => eprintln!("  UNCALIBRATED (no substitutes): honest ≤ x{:.2} → cut-off x{:.2}", cal.honest_high, cal.cutoff),
    }
    if let Some(report) = &a.report {
        write_json(
            report,
            &json!({
                "label": set.model_key, "texts": set.texts.len(), "min_texts": set.min_texts,
                "ids_sha256": digest, "reference": reference.label,
                "honest": cal.honest, "substitutes": cal.substitutes,
                "honest_high": cal.honest_high, "substitute_low": cal.substitute_low,
                "cutoff": cal.cutoff, "calibrated": cal.calibrated, "coverage": a.coverage, "min_gap": a.min_gap,
                "band_min": bands[0], "band_median": bands[bands.len() / 2], "band_max": bands[bands.len() - 1],
            }),
        )?;
    }
    eprintln!("refbuild: wrote {}", a.out.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_parse_from_either_format_and_digest_is_order_sensitive() {
        let plain = json!([[1, 2, 3], [4, 5]]);
        let wrapped = json!({"ids": [[1, 2, 3], [4, 5]]});
        assert_eq!(parse_ids(&plain).unwrap(), parse_ids(&wrapped).unwrap());
        let a = vec![vec![1, 2, 3], vec![4, 5]];
        let b = vec![vec![1, 2], vec![3, 4, 5]];
        assert_ne!(ids_digest(&a), ids_digest(&b), "the boundary between texts counts");
        assert!(parse_ids(&json!([[1, -2]])).is_err());
        assert!(parse_ids(&json!({"x": 1})).is_err());
    }

    #[test]
    fn tokenize_requests_skip_special_tokens() {
        let (path, body) = tokenize_request(TokenizerApi::Vllm, "Qwen/Qwen3.5-9B", "hi");
        assert_eq!((path, body["add_special_tokens"].clone(), body["prompt"].clone()), ("/tokenize", json!(false), json!("hi")));
        let (_, body) = tokenize_request(TokenizerApi::Llamacpp, "x", "hi");
        assert_eq!((body["add_special"].clone(), body["content"].clone()), (json!(false), json!("hi")));
        assert_eq!(tokens_from_reply(r#"{"count":2,"tokens":[9707,11]}"#).unwrap(), vec![9707, 11]);
        assert!(tokens_from_reply(r#"{"tokens":["a"]}"#).is_err());
    }

    #[test]
    fn a_run_over_different_ids_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("run.json");
        write_json(&p, &json!({"label": "x", "ids_sha256": "abc", "logprobs": [[-1.0]]})).unwrap();
        assert!(load_run(&p, "def").unwrap_err().contains("different token ids"));
        assert_eq!(load_run(&p, "abc").unwrap().logprobs, vec![vec![-1.0]]);
    }
}
