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
//! 0. `refbuild generate` / `refbuild texts` — produce texts nobody can have pre-scored: sampled
//!    from an LLM, and/or random crops of local files (see `refbuild_texts`).
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
    /// Generate fresh texts from an LLM with randomized topic / language / form prompts (JSON Lines).
    Generate(crate::refbuild_texts::GenerateArgs),
    /// Mix generated texts and random crops of local files into a texts file, with a provenance
    /// manifest.
    Texts(crate::refbuild_texts::TextsArgs),
    /// Tokenize texts (a JSON array of strings) with the reference engine's tokenizer.
    Tokenize(TokenizeArgs),
    /// Score every tokenized text on one engine configuration and save the run. Progress is saved
    /// after every text (`<out>.partial`), so an interrupted run resumes where it stopped.
    Score(ScoreArgs),
    /// Combine saved runs into a reference file with a calibrated cut-off.
    Assemble(AssembleArgs),
    /// Combine the shards of one run (`score --shard i/n`, e.g. from several CPU machines) into a run.
    Merge(MergeArgs),
}

#[derive(Args)]
struct MergeArgs {
    /// The token ids the shards were scored on.
    #[arg(long)]
    ids: PathBuf,
    /// Shard files (every shard of the run, once each).
    #[arg(long = "shard", required = true)]
    shards: Vec<PathBuf>,
    #[arg(long)]
    out: PathBuf,
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
    /// Requests in flight at once — hides network latency; results are still saved in order.
    #[arg(long, default_value_t = 1)]
    concurrency: usize,
    /// Score only shard `i` of `n` (texts whose index ≡ i mod n), e.g. `0/4` — to split a slow run
    /// (CPU) across machines; combine the shards with `refbuild merge`.
    #[arg(long, value_parser = parse_shard)]
    shard: Option<(usize, usize)>,
}

fn parse_shard(s: &str) -> Result<(usize, usize), String> {
    let (i, n) = s.split_once('/').ok_or("expected i/n, e.g. 0/4")?;
    let (i, n): (usize, usize) = (i.parse().map_err(|_| "bad shard index")?, n.parse().map_err(|_| "bad shard count")?);
    if n == 0 || i >= n {
        return Err("need 0 <= i < n".into());
    }
    Ok((i, n))
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
        RefbuildAction::Generate(a) => crate::refbuild_texts::generate(a),
        RefbuildAction::Texts(a) => crate::refbuild_texts::texts(a),
        RefbuildAction::Tokenize(a) => tokenize(a),
        RefbuildAction::Score(a) => score(a),
        RefbuildAction::Assemble(a) => assemble(a),
        RefbuildAction::Merge(a) => merge(a),
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

/// Scores already saved in `<out>.partial` for this work (header line + one line per position of the
/// work list, in order).
fn resume_partial(path: &Path, header: &Value, n: usize) -> Vec<Vec<f32>> {
    let Ok(raw) = std::fs::read_to_string(path) else { return Vec::new() };
    let mut lines = raw.lines();
    if lines.next().and_then(|l| serde_json::from_str::<Value>(l).ok()).as_ref() != Some(header) {
        return Vec::new();
    }
    let mut out = Vec::new();
    for line in lines {
        let Some(v) = serde_json::from_str::<Value>(line).ok() else { break }; // a torn last line
        if v.get("i").and_then(Value::as_u64) != Some(out.len() as u64) || out.len() >= n {
            break;
        }
        match serde_json::from_value::<Vec<f32>>(v.get("logprobs").cloned().unwrap_or(Value::Null)) {
            Ok(lps) => out.push(lps),
            Err(_) => break,
        }
    }
    out
}

fn score(a: ScoreArgs) -> Result<(), String> {
    use std::collections::BTreeMap;
    use std::io::Write;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    let ids = parse_ids(&read_json(&a.ids)?)?;
    let digest = ids_digest(&ids);
    let (shard_i, shard_n) = a.shard.unwrap_or((0, 1));
    let work: Vec<usize> = (0..ids.len()).filter(|t| t % shard_n == shard_i).collect();
    let root = a.engine.trim_end_matches('/').trim_end_matches("/v1").to_string();
    let header = json!({"ids_sha256": digest, "label": a.label, "shard": [shard_i, shard_n]});
    let partial = a.out.with_extension("partial");
    let done = resume_partial(&partial, &header, work.len());
    // Rewrite the partial file from what was recovered (drops a torn tail), then append in order.
    let mut file = std::fs::File::create(&partial).map_err(|e| format!("{}: {e}", partial.display()))?;
    writeln!(file, "{header}").map_err(|e| e.to_string())?;
    for (k, lps) in done.iter().enumerate() {
        writeln!(file, "{}", json!({"i": k, "logprobs": lps})).map_err(|e| e.to_string())?;
    }
    if !done.is_empty() {
        eprintln!("refbuild: {} resuming at {}/{}", a.label, done.len(), work.len());
    }
    let started = std::time::Instant::now();
    let resumed = done.len();
    // Workers score positions in any order; the writer saves them strictly in order, so a crash
    // leaves a clean, resumable prefix.
    struct Sink {
        file: std::fs::File,
        results: Vec<Vec<f32>>,
        pending: BTreeMap<usize, Vec<f32>>,
        error: Option<String>,
    }
    let sink = Mutex::new(Sink { file, results: done, pending: BTreeMap::new(), error: None });
    let next = AtomicUsize::new(resumed);
    std::thread::scope(|s| {
        for _ in 0..a.concurrency.max(1) {
            s.spawn(|| {
                let http = match ReqwestClient::new() {
                    Ok(h) => h,
                    Err(e) => {
                        sink.lock().unwrap_or_else(|p| p.into_inner()).error.get_or_insert(e.to_string());
                        return;
                    }
                };
                loop {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&t) = work.get(k) else { break };
                    if sink.lock().unwrap_or_else(|p| p.into_inner()).error.is_some() {
                        break;
                    }
                    let res = score_prompt(&http, &root, a.api, &a.model, &ids[t]);
                    let mut g = sink.lock().unwrap_or_else(|p| p.into_inner());
                    match res {
                        Err(e) => {
                            g.error.get_or_insert(format!("text {t}: {e}"));
                            break;
                        }
                        Ok(lps) => {
                            g.pending.insert(k, lps);
                            while let Some(lps) = { let n = g.results.len(); g.pending.remove(&n) } {
                                let n = g.results.len();
                                let line = json!({"i": n, "logprobs": lps}).to_string();
                                if let Err(e) = writeln!(g.file, "{line}").and_then(|_| g.file.flush()) {
                                    g.error.get_or_insert(e.to_string());
                                }
                                g.results.push(lps);
                                if (n + 1) % 25 == 0 || n + 1 == work.len() {
                                    eprintln!("refbuild: {} scored {}/{} ({:.1} s)", a.label, n + 1, work.len(), started.elapsed().as_secs_f64());
                                }
                            }
                        }
                    }
                }
            });
        }
    });
    let sink = sink.into_inner().unwrap_or_else(|p| p.into_inner());
    if let Some(e) = sink.error {
        return Err(format!("{e} (progress saved in {}; rerun to resume)", partial.display()));
    }
    let common = json!({"label": a.label, "engine": a.engine, "api": format!("{:?}", a.api), "model": a.model, "ids_sha256": digest});
    let mut out = common;
    if shard_n == 1 {
        out["logprobs"] = json!(sink.results);
    } else {
        out["shard"] = json!([shard_i, shard_n]);
        out["indices"] = json!(work);
        out["logprobs"] = json!(sink.results);
    }
    write_json(&a.out, &out)?;
    std::fs::remove_file(&partial).ok();
    Ok(())
}

/// Combine shard files into one run, checking they cover every text exactly once.
fn merge(a: MergeArgs) -> Result<(), String> {
    let ids = parse_ids(&read_json(&a.ids)?)?;
    let digest = ids_digest(&ids);
    let mut slots: Vec<Option<Vec<f32>>> = vec![None; ids.len()];
    let mut label = None;
    for path in &a.shards {
        let v = read_json(path)?;
        if v.get("ids_sha256").and_then(Value::as_str) != Some(digest.as_str()) {
            return Err(format!("{}: scored different token ids", path.display()));
        }
        let this = v.get("label").and_then(Value::as_str).unwrap_or("?").to_string();
        if label.get_or_insert_with(|| this.clone()) != &this {
            return Err(format!("{}: label {this} differs from the other shards", path.display()));
        }
        let indices: Vec<usize> = serde_json::from_value(v.get("indices").cloned().unwrap_or(Value::Null))
            .map_err(|_| format!("{}: not a shard (no indices)", path.display()))?;
        let lps: Vec<Vec<f32>> =
            serde_json::from_value(v.get("logprobs").cloned().unwrap_or(Value::Null)).map_err(|e| format!("{}: {e}", path.display()))?;
        if indices.len() != lps.len() {
            return Err(format!("{}: {} indices but {} score lists", path.display(), indices.len(), lps.len()));
        }
        for (i, l) in indices.into_iter().zip(lps) {
            let slot = slots.get_mut(i).ok_or_else(|| format!("{}: text {i} out of range", path.display()))?;
            if slot.replace(l).is_some() {
                return Err(format!("text {i} is in more than one shard"));
            }
        }
    }
    let missing = slots.iter().filter(|s| s.is_none()).count();
    if missing > 0 {
        return Err(format!("{missing} texts are in no shard"));
    }
    let logprobs: Vec<Vec<f32>> = slots.into_iter().map(|s| s.expect("checked")).collect();
    write_json(&a.out, &json!({"label": label, "ids_sha256": digest, "merged_from": a.shards.len(), "logprobs": logprobs}))?;
    eprintln!("refbuild: merged {} shards → {}", a.shards.len(), a.out.display());
    Ok(())
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
        "early_stop": set.early_stop.iter().map(|c| json!({"after": c.after, "pass_below": c.pass_below, "fail_above": c.fail_above})).collect::<Vec<_>>(),
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
    for c in &cal.early_stop {
        eprintln!(
            "  early stop after {:2} texts: pass below {}, fail above x{:.2}",
            c.after,
            c.pass_below.map_or("—".to_string(), |p| format!("x{p:.2}")),
            c.fail_above
        );
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
                "early_stop": cal.early_stop.iter().map(|c| json!({"after": c.after, "pass_below": c.pass_below, "fail_above": c.fail_above})).collect::<Vec<_>>(),
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
    fn partial_scores_resume_only_for_the_same_ids_and_stop_at_a_torn_line() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("run.partial");
        let header = json!({"ids_sha256": "abc", "label": "x", "shard": [0, 1]});
        std::fs::write(
            &p,
            format!("{}\n{}\n{}\n{{\"i\":2,\"logpr", header, json!({"i": 0, "logprobs": [-1.0]}), json!({"i": 1, "logprobs": [-2.0]})),
        )
        .unwrap();
        assert_eq!(resume_partial(&p, &header, 5), vec![vec![-1.0], vec![-2.0]]);
        let other = json!({"ids_sha256": "abc", "label": "x", "shard": [1, 2]});
        assert!(resume_partial(&p, &other, 5).is_empty(), "different ids or shard → start over");
        assert_eq!(resume_partial(&p, &header, 1).len(), 1, "never more than the work");
        assert!(resume_partial(&dir.path().join("none"), &header, 5).is_empty());
    }

    #[test]
    fn a_run_over_different_ids_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("run.json");
        write_json(&p, &json!({"label": "x", "ids_sha256": "abc", "logprobs": [[-1.0]]})).unwrap();
        assert!(load_run(&p, "def").unwrap_err().contains("different token ids"));
        assert_eq!(load_run(&p, "abc").unwrap().logprobs, vec![vec![-1.0]]);
    }

    #[test]
    fn shards_parse_and_merge_back_into_one_run() {
        assert_eq!(parse_shard("1/4"), Ok((1, 4)));
        assert!(parse_shard("4/4").is_err() && parse_shard("x").is_err() && parse_shard("0/0").is_err());
        let dir = tempfile::tempdir().unwrap();
        let ids: Vec<Vec<u32>> = (0..5).map(|t| vec![t, t + 1, t + 2]).collect();
        let d = ids_digest(&ids);
        let ids_path = dir.path().join("ids.json");
        write_json(&ids_path, &json!({"ids": ids})).unwrap();
        let shard = |i: usize, idx: Vec<usize>| {
            let p = dir.path().join(format!("s{i}.json"));
            let lps: Vec<Vec<f32>> = idx.iter().map(|&t| vec![-(t as f32), -1.0]).collect();
            write_json(&p, &json!({"label": "cpu", "ids_sha256": d, "shard": [i, 2], "indices": idx, "logprobs": lps})).unwrap();
            p
        };
        let (a, b) = (shard(0, vec![0, 2, 4]), shard(1, vec![1, 3]));
        let out = dir.path().join("run.json");
        merge(MergeArgs { ids: ids_path.clone(), shards: vec![b.clone(), a.clone()], out: out.clone() }).unwrap();
        let run = load_run(&out, &d).unwrap();
        assert_eq!(run.logprobs.iter().map(|l| l[0]).collect::<Vec<_>>(), vec![0.0, -1.0, -2.0, -3.0, -4.0]);
        assert!(merge(MergeArgs { ids: ids_path.clone(), shards: vec![a.clone()], out: out.clone() }).unwrap_err().contains("in no shard"));
        assert!(merge(MergeArgs { ids: ids_path, shards: vec![a.clone(), a], out }).unwrap_err().contains("more than one shard"));
    }
}
