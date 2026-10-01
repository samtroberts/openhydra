// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Prompt-scoring challenge over the network (Track A2-Logprob, ATTESTED_ENGINE_TIER_PLAN_v1 §10.1).
//!
//! A consumer sends a provider pre-tokenized text; the provider has its engine score it
//! teacher-forced ([`EngineAdapter::score_prompt`]) and returns the per-position log-probs. The
//! consumer judges them against its own reference with [`openhydra_protocol::logprob`]. The provider
//! never learns the reference or the verdict.
//!
//! Wire (consumer → provider): `0x17 ‖ u16-LE model_len ‖ model_ref ‖ u32-LE n ‖ n × u32-LE ids`.
//! Reply: `OHS1 ‖ status ‖ body`, where an OK body is `u32-LE m ‖ m × f32-LE` (m = n − 1) and a
//! refusal body is a UTF-8 reason. A legacy provider answers the unknown method byte with a framed
//! serve error, which can't start with `OHS1`, so the consumer reads it as
//! [`ScoreChallengeError::Unsupported`].
//!
//! Scoring is opt-in on the provider (`provide --prompt-scoring <dialect>`) and bounded to
//! [`MAX_INFLIGHT`] at once: concurrent prompt-log-prob requests OOM-crashed vLLM (Gemma 4 E4B at
//! 16) and SGLang (Qwen3.5-9B at 4) on a 24 GB L4.

use std::sync::atomic::{AtomicUsize, Ordering};

use openhydra_protocol::logprob::{
    normalize_weights_digest, ChallengeRound, LogprobVerdict, ModelIdentity, ReferenceSet, ReferenceText, RoundError,
    DEFAULT_CUTOFF, DEFAULT_MIN_TEXTS,
};
use openhydra_protocol::weight::{cost_tier_for_handle, cost_tier_for_params_quant, CostTier};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::adapter::{AdapterError, EngineAdapter};
use openhydra_protocol::verify::VerificationOutcome;

/// libp2p proxy method byte for a prompt-scoring challenge. (0x16 is `ATTEST_CHALLENGE`.)
pub const PROMPT_SCORE_CHALLENGE: u8 = 0x17;
/// libp2p proxy method byte asking which exact model a handle is (`0x18 ‖ u16-LE len ‖ model_ref`);
/// the reply's OK body is a JSON [`ModelInfo`]. Lets a verifier pick the reference for the precise
/// quantisation before it sends any text.
pub const MODEL_INFO_REQUEST: u8 = 0x18;

/// Every reply starts with this magic (see the module docs for why a legacy reply can't).
pub const SCORE_RESP_MAGIC: &[u8; 4] = b"OHS1";
const RESP_OK: u8 = 0x00;
const RESP_REFUSED: u8 = 0x01;

/// Longest model handle a challenge may carry.
const MAX_MODEL_REF_LEN: usize = 256;
/// Most tokens one challenge may ask the engine to score. The patched llama.cpp endpoint checks at
/// startup that a prompt this long fits in device memory (and disables scoring if not), so the two
/// must agree; `refbuild tokenize` keeps texts within it.
pub const MAX_CHALLENGE_TOKENS: usize = 512;
/// Challenges the provider scores at once; the rest are refused ("busy") for the consumer to retry.
pub const MAX_INFLIGHT: usize = 2;

/// Why a challenge didn't yield scores (consumer side).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScoreChallengeError {
    /// The request never got a reply.
    Transport(String),
    /// The provider doesn't speak this protocol (a legacy build).
    Unsupported,
    /// The provider understood but declined (not opted in, busy, engine can't score, model not public).
    Refused(String),
    /// The reply was malformed, or has the wrong number of scores.
    Malformed,
}

/// Consumer side: frame a challenge (method byte included). `None` if the model handle or the
/// token count is out of bounds.
pub fn encode_challenge(model_ref: &str, token_ids: &[u32]) -> Option<Vec<u8>> {
    if model_ref.len() > MAX_MODEL_REF_LEN || token_ids.len() < 2 || token_ids.len() > MAX_CHALLENGE_TOKENS {
        return None;
    }
    let mut b = Vec::with_capacity(1 + 2 + model_ref.len() + 4 + 4 * token_ids.len());
    b.push(PROMPT_SCORE_CHALLENGE);
    b.extend_from_slice(&(model_ref.len() as u16).to_le_bytes());
    b.extend_from_slice(model_ref.as_bytes());
    b.extend_from_slice(&(token_ids.len() as u32).to_le_bytes());
    for id in token_ids {
        b.extend_from_slice(&id.to_le_bytes());
    }
    Some(b)
}

fn decode_challenge(body: &[u8]) -> Option<(String, Vec<u32>)> {
    let len = u16::from_le_bytes(body.get(..2)?.try_into().ok()?) as usize;
    if len > MAX_MODEL_REF_LEN {
        return None;
    }
    let model = String::from_utf8(body.get(2..2 + len)?.to_vec()).ok()?;
    let rest = &body[2 + len..];
    let n = u32::from_le_bytes(rest.get(..4)?.try_into().ok()?) as usize;
    if !(2..=MAX_CHALLENGE_TOKENS).contains(&n) || rest.len() != 4 + 4 * n {
        return None;
    }
    let ids = rest[4..].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    Some((model, ids))
}

/// A refusal reply — also for refusals decided before [`handle_prompt_score_inbound`] runs (the
/// worker pool shedding), so the consumer reads "refused" rather than "legacy provider".
pub fn score_refusal(reason: &str) -> Vec<u8> {
    response(RESP_REFUSED, reason.as_bytes())
}

fn response(status: u8, body: &[u8]) -> Vec<u8> {
    let mut b = Vec::with_capacity(5 + body.len());
    b.extend_from_slice(SCORE_RESP_MAGIC);
    b.push(status);
    b.extend_from_slice(body);
    b
}

/// Caps how many challenges are scored at once, provider-wide.
#[derive(Debug, Default)]
pub struct InflightGate {
    current: AtomicUsize,
}

/// Holds one slot of an [`InflightGate`] until dropped.
pub struct InflightSlot<'a>(&'a InflightGate);

impl Drop for InflightSlot<'_> {
    fn drop(&mut self) {
        self.0.current.fetch_sub(1, Ordering::AcqRel);
    }
}

impl InflightGate {
    pub fn new() -> Self {
        Self::default()
    }

    /// Take a slot, or `None` if [`MAX_INFLIGHT`] are already taken.
    pub fn try_enter(&self) -> Option<InflightSlot<'_>> {
        self.current
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| (n < MAX_INFLIGHT).then_some(n + 1))
            .ok()
            .map(|_| InflightSlot(self))
    }
}

/// What a provider declares about a model it serves: the identity a verifier matches references
/// against. Its own word — the log-prob check is what tests it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ModelInfo {
    pub family: String,
    pub params: String,
    /// The precise quantisation (`Q4_K_M`, `Q8_0`, `BF16`), or `""` when unknown.
    pub quant: String,
    /// sha256 of the weights **file** the engine serves (`sha256:<hex>`; Ollama's model blob, or the
    /// GGUF file a llama.cpp server loaded), else `""`. Matched to references made from that file.
    #[serde(default)]
    pub weight_digest: String,
}

impl ModelInfo {
    /// The declared identity, when complete.
    pub fn identity(&self) -> Option<ModelIdentity> {
        ModelIdentity::new(&self.family, &self.params, &self.quant)
    }

    /// The declared weights file, in canonical form, when it is a sha256 digest.
    pub fn weights(&self) -> Option<String> {
        normalize_weights_digest(&self.weight_digest)
    }
}

/// Consumer side: frame a model-info request (method byte included).
pub fn encode_info_request(model_ref: &str) -> Option<Vec<u8>> {
    if model_ref.len() > MAX_MODEL_REF_LEN {
        return None;
    }
    let mut b = Vec::with_capacity(3 + model_ref.len());
    b.push(MODEL_INFO_REQUEST);
    b.extend_from_slice(&(model_ref.len() as u16).to_le_bytes());
    b.extend_from_slice(model_ref.as_bytes());
    Some(b)
}

fn decode_info_request(body: &[u8]) -> Option<String> {
    let len = u16::from_le_bytes(body.get(..2)?.try_into().ok()?) as usize;
    if len > MAX_MODEL_REF_LEN || body.len() != 2 + len {
        return None;
    }
    String::from_utf8(body[2..].to_vec()).ok()
}

/// Provider side: answer a model-info request. Same gates as scoring: only when the operator opted
/// in (`enabled`) and only for publicly announced models (`scorable`); `info(model)` looks the model
/// up in what the engine detected.
pub fn handle_model_info_inbound(
    body: &[u8],
    enabled: bool,
    scorable: &dyn Fn(&str) -> bool,
    info: &dyn Fn(&str) -> Option<ModelInfo>,
) -> Vec<u8> {
    if !enabled {
        return score_refusal("prompt scoring not offered by this provider");
    }
    let Some(model) = decode_info_request(body) else {
        return score_refusal("malformed request");
    };
    if !scorable(&model) {
        return score_refusal("model not scorable here");
    }
    match info(&model).and_then(|i| serde_json::to_vec(&i).ok()) {
        Some(json) => response(RESP_OK, &json),
        None => score_refusal("model not scorable here"),
    }
}

/// Consumer side: ask a provider which exact model `model_ref` is.
pub fn request_model_info(transport: &mut Transport<'_>, model_ref: &str) -> Result<ModelInfo, ScoreChallengeError> {
    let framed = encode_info_request(model_ref).ok_or(ScoreChallengeError::Malformed)?;
    let reply = transport(&framed).map_err(ScoreChallengeError::Transport)?;
    let Some(rest) = reply.strip_prefix(SCORE_RESP_MAGIC.as_slice()) else {
        return Err(ScoreChallengeError::Unsupported);
    };
    match rest.split_first() {
        Some((&RESP_OK, body)) => serde_json::from_slice(body).map_err(|_| ScoreChallengeError::Malformed),
        Some((&RESP_REFUSED, reason)) => {
            Err(ScoreChallengeError::Refused(String::from_utf8_lossy(reason).into_owned()))
        }
        _ => Err(ScoreChallengeError::Malformed),
    }
}

/// The quantisation a model *name* suggests (`Qwen3.5-9B-Q4_K_M` → `q4_k_m`, `…-FP8` → `fp8`), if
/// any — to flag a name that disagrees with the declared identity.
pub fn quant_hint_from_name(name: &str) -> Option<String> {
    let base = name.rsplit(['/', '\\']).next().unwrap_or(name);
    let base = base.strip_suffix(".gguf").unwrap_or(base);
    base.split(['-', '.', ':'])
        .rev()
        .map(|t| t.to_lowercase())
        .find(|t| {
            let digit_after = |p: &str| t.strip_prefix(p).is_some_and(|r| r.starts_with(|c: char| c.is_ascii_digit()));
            (digit_after("q") || digit_after("iq"))
                || matches!(t.as_str(), "bf16" | "f16" | "fp16" | "fp8" | "int4" | "int8" | "awq" | "gptq")
        })
        .and_then(|t| ModelIdentity::new("x", "1b", &t).map(|i| i.quant))
}

/// Provider side: answer a challenge (`body` = the bytes after the method byte). Never panics; every
/// failure is a refusal. `engine` is `None` unless the operator opted in. `scorable(model)` must be
/// true only for models this node announces publicly, so a challenge can't probe for a private
/// model (it gets the same refusal as an unknown one). Engine errors are logged here and replied to
/// generically, so a challenge can't read engine internals.
pub fn handle_prompt_score_inbound(
    body: &[u8],
    engine: Option<&dyn EngineAdapter>,
    gate: &InflightGate,
    scorable: &dyn Fn(&str) -> bool,
) -> Vec<u8> {
    let Some(engine) = engine else {
        return score_refusal("prompt scoring not offered by this provider");
    };
    let Some((model, ids)) = decode_challenge(body) else {
        return score_refusal("malformed challenge");
    };
    if !scorable(&model) {
        return score_refusal("model not scorable here");
    }
    let Some(_slot) = gate.try_enter() else {
        return score_refusal("prompt scoring busy, retry");
    };
    match engine.score_prompt(&model, &ids) {
        Ok(lps) if lps.len() + 1 == ids.len() => {
            let mut b = Vec::with_capacity(4 + 4 * lps.len());
            b.extend_from_slice(&(lps.len() as u32).to_le_bytes());
            for lp in &lps {
                b.extend_from_slice(&lp.to_le_bytes());
            }
            response(RESP_OK, &b)
        }
        Ok(lps) => {
            tracing::warn!(model, sent = ids.len(), got = lps.len(), "engine returned the wrong number of prompt scores");
            score_refusal("engine could not score the prompt")
        }
        Err(AdapterError::Busy(e)) => {
            tracing::info!(model, reason = %e, "prompt scoring not ready yet");
            score_refusal("prompt scoring warming up, retry")
        }
        Err(e) => {
            tracing::warn!(model, error = %e, "prompt scoring failed");
            score_refusal("engine could not score the prompt")
        }
    }
}

/// A request/response round-trip to one provider (a `proxy_forward` in production).
pub type Transport<'a> = dyn FnMut(&[u8]) -> Result<Vec<u8>, String> + 'a;

/// Consumer side: send one challenge and decode the reply into `token_ids.len() - 1` log-probs.
/// Doesn't judge them — that's [`openhydra_protocol::logprob`].
pub fn request_prompt_scores(
    transport: &mut Transport<'_>,
    model_ref: &str,
    token_ids: &[u32],
) -> Result<Vec<f32>, ScoreChallengeError> {
    let framed = encode_challenge(model_ref, token_ids).ok_or(ScoreChallengeError::Malformed)?;
    let reply = transport(&framed).map_err(ScoreChallengeError::Transport)?;
    let Some(rest) = reply.strip_prefix(SCORE_RESP_MAGIC.as_slice()) else {
        return Err(ScoreChallengeError::Unsupported);
    };
    match rest.split_first() {
        Some((&RESP_OK, body)) => {
            let m = u32::from_le_bytes(body.get(..4).ok_or(ScoreChallengeError::Malformed)?.try_into().unwrap()) as usize;
            if m + 1 != token_ids.len() || body.len() != 4 + 4 * m {
                return Err(ScoreChallengeError::Malformed);
            }
            Ok(body[4..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
        }
        Some((&RESP_REFUSED, reason)) => {
            Err(ScoreChallengeError::Refused(String::from_utf8_lossy(reason).into_owned()))
        }
        _ => Err(ScoreChallengeError::Malformed),
    }
}

// ── Consumer side: reference sets + challenge rounds ──

/// A verifier's reference set and the provider model handles it applies to (a vLLM served-model
/// name, an Ollama tag, …). Loaded from a JSON file the verifier keeps private: a text a provider
/// has already seen can be scored offline by the honest model and replayed, so the set must not be
/// published and a text is never sent twice to the same provider.
#[derive(Debug, Clone)]
pub struct LoadedReference {
    pub handles: Vec<String>,
    pub set: ReferenceSet,
}

#[derive(Deserialize)]
struct CheckpointFile {
    after: usize,
    #[serde(default)]
    pass_below: Option<f64>,
    fail_above: f64,
}

#[derive(Deserialize)]
struct ReferenceModel {
    family: String,
    params: String,
    quant: String,
}

#[derive(Deserialize)]
struct ReferenceFile {
    model_key: String,
    #[serde(default)]
    model: Option<ReferenceModel>,
    #[serde(default)]
    handles: Vec<String>,
    #[serde(default)]
    cutoff: Option<f64>,
    #[serde(default)]
    min_texts: Option<usize>,
    #[serde(default)]
    early_stop: Vec<CheckpointFile>,
    /// Digest of the weights file the reference was scored from (`sha256:<hex>`).
    #[serde(default)]
    weights: Option<String>,
    texts: Vec<ReferenceTextFile>,
}

#[derive(Deserialize)]
struct ReferenceTextFile {
    id: String,
    token_ids: Vec<u32>,
    logprobs: Vec<f32>,
    band: f64,
}

/// Parse and validate a reference-set file:
/// `{"model_key", "model"?: {"family", "params", "quant"}, "weights"?: "sha256:<hex>", "handles"?: [...],
/// "cutoff"?, "min_texts"?, "texts": [{"id", "token_ids", "logprobs", "band"}]}`. It needs a `model`
/// identity (matched against the identity a provider declares), a `weights` digest (matched against
/// the weights file it declares) or at least one handle (matched against the model's name).
pub fn parse_reference(json: &str) -> Result<LoadedReference, String> {
    let f: ReferenceFile = serde_json::from_str(json).map_err(|e| format!("reference set: {e}"))?;
    let identity = match &f.model {
        Some(m) => Some(
            ModelIdentity::new(&m.family, &m.params, &m.quant)
                .ok_or_else(|| format!("reference set {}: incomplete model identity", f.model_key))?,
        ),
        None => None,
    };
    let weights = match &f.weights {
        Some(w) => Some(
            normalize_weights_digest(w).ok_or_else(|| format!("reference set {}: weights must be sha256:<64 hex>", f.model_key))?,
        ),
        None => None,
    };
    if identity.is_none() && weights.is_none() && f.handles.iter().all(|h| h.trim().is_empty()) {
        return Err("reference set: needs a model identity, a weights digest or at least one handle".into());
    }
    let set = ReferenceSet {
        model_key: f.model_key,
        identity,
        weights,
        texts: f
            .texts
            .into_iter()
            .map(|t| ReferenceText { id: t.id, token_ids: t.token_ids, logprobs: t.logprobs, band: t.band })
            .collect(),
        cutoff: f.cutoff.unwrap_or(DEFAULT_CUTOFF),
        min_texts: f.min_texts.unwrap_or(DEFAULT_MIN_TEXTS),
        early_stop: f
            .early_stop
            .into_iter()
            .map(|c| openhydra_protocol::logprob::Checkpoint { after: c.after, pass_below: c.pass_below, fail_above: c.fail_above })
            .collect(),
    };
    set.validate().map_err(|e| format!("reference set {}: {e:?}", set.model_key))?;
    if let Some(t) = set.texts.iter().find(|t| t.token_ids.len() > MAX_CHALLENGE_TOKENS) {
        return Err(format!("reference set {}: text {} exceeds {MAX_CHALLENGE_TOKENS} tokens", set.model_key, t.id));
    }
    Ok(LoadedReference { handles: f.handles, set })
}

/// The reference to check a provider's model against: the one scored from the weights file the
/// provider declares (`weights`, any digest spelling); else, among references not tied to a file,
/// the one made from the declared identity, then one listing the model's name. A reference tied to
/// a file never applies to a different or undeclared file: two honest files with the same label
/// differ as much as a substitute does. `None` when nothing applies.
pub fn select_reference<'a>(
    refs: &'a [std::sync::Arc<LoadedReference>],
    handle: &str,
    declared: Option<&ModelIdentity>,
    weights: Option<&str>,
) -> Option<&'a std::sync::Arc<LoadedReference>> {
    let weights = weights.and_then(normalize_weights_digest);
    if let Some(r) = weights.as_ref().and_then(|w| refs.iter().find(|r| r.set.weights.as_ref() == Some(w))) {
        return Some(r);
    }
    let legacy = || refs.iter().filter(|r| r.set.weights.is_none());
    declared
        .and_then(|id| legacy().find(|r| r.set.identity.as_ref() == Some(id)))
        .or_else(|| legacy().find(|r| r.handles.iter().any(|h| h == handle)))
}

/// The cost tier a serve is priced at before the authenticity gate. By default the model name's
/// conservative estimate (its size at 4-bit, [`cost_tier_for_handle`]): a provider's word never
/// raises its price. With a qualifying, fresh pass streak (`status`) against a reference whose
/// identity is known, the price follows that **verified** identity instead — its exact size and
/// quantisation — whether that is higher (an honest BF16 model) or lower (a big name on a small model).
/// `identity_of(tag)` resolves a [`reference_tag`] to the identity of the reference it names (a
/// local reference file, or registry packs seen for that label).
pub fn priced_tier(
    handle: &str,
    status: Option<&LogprobStatus>,
    identity_of: &dyn Fn([u8; 16]) -> Option<ModelIdentity>,
    now_ms: u64,
) -> CostTier {
    let by_handle = cost_tier_for_handle(handle);
    let Some(st) = status.filter(|st| st.evidence(now_ms).0) else {
        return by_handle;
    };
    identity_of(st.reference)
        .and_then(|id| cost_tier_for_params_quant(&id.params, &id.quant))
        .unwrap_or(by_handle)
}

/// Resolve a reference tag against local reference files.
pub fn identity_in(refs: &[std::sync::Arc<LoadedReference>], tag: [u8; 16]) -> Option<ModelIdentity> {
    refs.iter().find(|r| reference_tag(&r.set) == tag).and_then(|r| r.set.identity.clone())
}

/// A 16-byte tag of a reference (its label, and its weights file when it has one), kept in
/// [`LogprobStatus`] so a verdict stays tied to the reference it was earned against.
pub fn reference_tag(set: &ReferenceSet) -> [u8; 16] {
    let mut h = Sha256::new();
    h.update(set.model_key.as_bytes());
    if let Some(w) = &set.weights {
        h.update([0]);
        h.update(w.as_bytes());
    }
    h.finalize()[..16].try_into().expect("16 of 32 bytes")
}

/// Read and validate a reference-set file.
pub fn load_reference(path: &std::path::Path) -> Result<LoadedReference, String> {
    let json = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse_reference(&json).map_err(|e| format!("{}: {e}", path.display()))
}

/// How a run of challenges ended.
#[derive(Debug, Clone, PartialEq)]
pub enum RoundOutcome {
    /// The round completed.
    Verdict(LogprobVerdict),
    /// Stopped early (refusal, transport error, malformed reply). Ratios recorded so far are kept
    /// for the next attempt; the texts already sent stay burned.
    Incomplete(String),
    /// The provider doesn't speak the protocol (legacy build).
    Unsupported,
    /// Not enough unsent texts left for this provider — the verifier needs a fresh set.
    Exhausted,
}

/// Busy / rate-limited refusals tolerated in a row before a run gives up.
const MAX_BUSY_RETRIES: usize = 3;
/// Retryable refusals [`probe_scoring`] waits through: a side-scorer loading its model answers
/// "warming up, retry" for a minute or two (with 2 s pauses, ~2.5 min), and that must never count
/// as a provider dodging the check.
const MAX_PROBE_RETRIES: usize = 75;

/// Send challenges one at a time until the round closes or something stops it. `pick(n)` chooses
/// which unsent text goes next (random in production); `pause()` backs off after a busy refusal.
/// `persist(round)` runs after each text is issued — **before** it is sent, so a crash can't forget
/// a text the provider has seen — and after each reply is recorded.
pub fn run_round(
    round: &mut ChallengeRound,
    set: &ReferenceSet,
    model_ref: &str,
    transport: &mut Transport<'_>,
    pick: &mut dyn FnMut(usize) -> usize,
    pause: &mut dyn FnMut(),
    persist: &mut dyn FnMut(&ChallengeRound),
) -> RoundOutcome {
    let mut busy = 0;
    loop {
        // Out of unsent texts: the round can't go on with this set (a first-stage registry pack is
        // topped up by the caller; a local set needs fresh texts).
        let Some(text) = round.next_text(set, &mut *pick) else {
            return RoundOutcome::Exhausted;
        };
        let (id, token_ids) = (text.id.clone(), text.token_ids.clone());
        persist(round);
        let reply = request_prompt_scores(transport, model_ref, &token_ids);
        let recorded = reply.as_ref().ok().map(|scores| round.record(set, &id, scores));
        if reply.is_err() {
            round.abandon(&id);
        }
        persist(round);
        match reply {
            Ok(_) => match recorded.expect("recorded for Ok") {
                Ok(Some(verdict)) => return RoundOutcome::Verdict(verdict),
                Ok(None) => busy = 0,
                Err(RoundError::Score(e)) => return RoundOutcome::Incomplete(format!("unscorable reply: {e:?}")),
                Err(e) => return RoundOutcome::Incomplete(format!("{e:?}")),
            },
            Err(err) => {
                match err {
                    ScoreChallengeError::Unsupported => return RoundOutcome::Unsupported,
                    ScoreChallengeError::Refused(r) if r.contains("retry") && busy + 1 < MAX_BUSY_RETRIES => {
                        busy += 1;
                        pause();
                    }
                    ScoreChallengeError::Refused(r) => return RoundOutcome::Incomplete(format!("refused: {r}")),
                    ScoreChallengeError::Transport(e) => return RoundOutcome::Incomplete(format!("transport: {e}")),
                    ScoreChallengeError::Malformed => return RoundOutcome::Incomplete("malformed reply".into()),
                }
            }
        }
    }
}

/// The throwaway prompt [`probe_scoring`] sends: low token ids, valid in any vocabulary, and not a
/// reference text (nothing is learned from its scores).
const PROBE_TOKENS: [u32; 16] = [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015];

/// Send a throwaway prompt before a round, so no reference text — a registry pack's are used up
/// network-wide the moment they're issued — is spent on a provider that can't score right now
/// (scoring off, or its engine disabled scoring at startup). `None` when it scored; else how the
/// round would have ended. Retryable refusals (busy, a side-scorer warming up) are waited through
/// for up to [`MAX_PROBE_RETRIES`] pauses, so the round that follows runs on a warm scorer. It
/// reveals nothing a real challenge doesn't: challenges are recognisable by their method byte anyway.
pub fn probe_scoring(transport: &mut Transport<'_>, model_ref: &str, pause: &mut dyn FnMut()) -> Option<RoundOutcome> {
    let mut busy = 0;
    loop {
        return match request_prompt_scores(transport, model_ref, &PROBE_TOKENS) {
            Ok(scores) if scores.iter().all(|&lp| lp.is_finite() && lp <= 0.0) => None,
            Ok(_) => Some(RoundOutcome::Incomplete("unscorable reply to the probe".into())),
            Err(ScoreChallengeError::Unsupported) => Some(RoundOutcome::Unsupported),
            Err(ScoreChallengeError::Refused(r)) if r.contains("retry") && busy + 1 < MAX_PROBE_RETRIES => {
                busy += 1;
                pause();
                continue;
            }
            Err(ScoreChallengeError::Refused(r)) => Some(RoundOutcome::Incomplete(format!("refused: {r}"))),
            Err(ScoreChallengeError::Transport(e)) => Some(RoundOutcome::Incomplete(format!("transport: {e}"))),
            Err(ScoreChallengeError::Malformed) => Some(RoundOutcome::Incomplete("malformed reply".into())),
        };
    }
}

// ── Consumer side: per-provider check history ──

/// How long a verdict counts toward the authenticity tier, at least — a longer pass streak is
/// re-checked less often, and its pass stays valid for twice that interval ([`validity_ms`]).
pub const LOGPROB_VALID_MS: u64 = 24 * 60 * 60 * 1000;
/// Re-check a provider this long after a verdict while its pass streak is short.
pub const LOGPROB_REFRESH_MS: u64 = 6 * 60 * 60 * 1000;
/// Random spread of every re-check time (±25%), so a provider can't time a model swap between checks.
pub const LOGPROB_REFRESH_JITTER: f64 = 0.25;

/// How long after a verdict to re-check, by pass streak: 6 h until the streak qualifies
/// (< [`LOGPROB_MIN_PASSES`]), 12 h for a qualified streak, 24 h once it is twice that long. A fail
/// resets the streak, so a provider caught once goes back to the 6-hour cadence.
pub fn refresh_interval_ms(pass_streak: u32) -> u64 {
    if pass_streak >= 2 * LOGPROB_MIN_PASSES {
        4 * LOGPROB_REFRESH_MS
    } else if pass_streak >= LOGPROB_MIN_PASSES {
        2 * LOGPROB_REFRESH_MS
    } else {
        LOGPROB_REFRESH_MS
    }
}

/// How long a verdict counts: twice the re-check interval (so a trusted provider doesn't drop out of
/// A2 between checks), and never less than [`LOGPROB_VALID_MS`].
pub fn validity_ms(pass_streak: u32) -> u64 {
    (2 * refresh_interval_ms(pass_streak)).max(LOGPROB_VALID_MS)
}
/// Retry an interrupted round after this long.
pub const LOGPROB_RETRY_MS: u64 = 10 * 60 * 1000;
/// Back off from a legacy provider, or from one whose unsent texts ran out.
pub const LOGPROB_BACKOFF_MS: u64 = 60 * 60 * 1000;
/// Passing rounds in a row needed before the log-prob method flag counts toward A2 …
pub const LOGPROB_MIN_PASSES: u32 = 3;
/// … with the first of them at least this long before the latest, so one lucky afternoon (or one
/// window where only the challenges went to the real model) isn't enough.
pub const LOGPROB_MIN_SPAN_MS: u64 = 48 * 60 * 60 * 1000;
/// Runs in a row that end without a verdict — refused, timed out, unsupported — from a provider that
/// has answered before, after which the gateway treats it as dodging the check.
pub const LOGPROB_MAX_EVASIONS: u32 = 3;

/// One `(peer, model)`'s log-prob check history, as the gateway sees it.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LogprobStatus {
    /// The latest completed round's verdict and when it closed.
    pub verdict: Option<(LogprobVerdict, u64)>,
    /// Passing rounds in a row (reset by a fail or an evasion penalty).
    pub pass_streak: u32,
    /// When the first pass of the current streak closed.
    pub streak_since_ms: u64,
    /// Runs in a row that ended without a verdict after the provider had answered before.
    pub evasions: u32,
    /// When the latest evasion penalty was applied (0 = never).
    pub evaded_at_ms: u64,
    /// Earliest time the next run may start.
    pub next_due_ms: u64,
    /// [`reference_tag`] of the reference the verdict and streak were earned against (zero = none).
    pub reference: [u8; 16],
}

/// What the gateway must record after a run, beyond updating the status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusEffect {
    /// Nothing: a pass (which by itself earns no audit credit or reputation), or a run that ended
    /// early without being held against the provider.
    None,
    /// A failed round: a failed audit plus a `Failed` reputation outcome.
    Fail,
    /// Repeated dodging: a failed audit, but no reputation cut — it could be a flaky connection.
    Evasion,
}

impl LogprobStatus {
    /// Point the status at the reference now in use. A different reference means a different claimed
    /// model: the verdict, streak and evasion count belong to the old one, so they're cleared, and so
    /// are the round's in-progress ratios (issued texts stay issued). Returns whether it changed.
    pub fn switch_reference(&mut self, tag: [u8; 16], round: &mut ChallengeRound) -> bool {
        if self.reference == tag {
            return false;
        }
        *self = Self { next_due_ms: self.next_due_ms, reference: tag, ..Self::default() };
        round.reset_progress();
        true
    }

    /// Record a completed round.
    pub fn on_verdict(&mut self, verdict: LogprobVerdict, now_ms: u64) -> StatusEffect {
        self.on_verdict_jittered(verdict, now_ms, 0.0)
    }

    /// As [`on_verdict`](Self::on_verdict), with the re-check time moved by `jitter` (a fraction in
    /// `[-1, 1]`, scaled by [`LOGPROB_REFRESH_JITTER`]) — the caller supplies the randomness.
    pub fn on_verdict_jittered(&mut self, verdict: LogprobVerdict, now_ms: u64, jitter: f64) -> StatusEffect {
        self.verdict = Some((verdict, now_ms));
        self.evasions = 0;
        let effect = if verdict.passed() {
            if self.pass_streak == 0 {
                self.streak_since_ms = now_ms;
            }
            self.pass_streak += 1;
            StatusEffect::None
        } else {
            self.pass_streak = 0;
            self.streak_since_ms = 0;
            StatusEffect::Fail
        };
        let base = refresh_interval_ms(self.pass_streak) as f64;
        let factor = 1.0 + LOGPROB_REFRESH_JITTER * jitter.clamp(-1.0, 1.0);
        self.next_due_ms = now_ms + (base * factor) as u64;
        effect
    }

    /// Record a run that ended without a verdict. `counts_against` is true when the provider has
    /// answered before and the run stopped on its side (refusal, timeout, a broken reply, or it
    /// stopped supporting the protocol) — false for a first contact or for our own shortage of texts.
    pub fn on_no_verdict(&mut self, counts_against: bool, retry_ms: u64, now_ms: u64) -> StatusEffect {
        self.next_due_ms = now_ms + retry_ms;
        if !counts_against {
            return StatusEffect::None;
        }
        self.evasions += 1;
        if self.evasions < LOGPROB_MAX_EVASIONS {
            return StatusEffect::None;
        }
        self.evasions = 0;
        self.pass_streak = 0;
        self.streak_since_ms = 0;
        self.evaded_at_ms = now_ms;
        StatusEffect::Evasion
    }

    /// Whether the pass streak is long enough, and spread over long enough, to count.
    pub fn streak_qualifies(&self) -> bool {
        match self.verdict {
            Some((v, at)) if v.passed() => {
                self.pass_streak >= LOGPROB_MIN_PASSES && at.saturating_sub(self.streak_since_ms) >= LOGPROB_MIN_SPAN_MS
            }
            _ => false,
        }
    }

    /// `(log-prob method flag, latest outcome)` for the authenticity evidence. The flag needs a fresh
    /// pass (within [`LOGPROB_VALID_MS`]) at the end of a qualifying streak; a fresh fail forces the
    /// floor. An evasion penalty just breaks the streak.
    pub fn evidence(&self, now_ms: u64) -> (bool, Option<VerificationOutcome>) {
        match self.verdict {
            Some((v, at)) if now_ms.saturating_sub(at) < validity_ms(self.pass_streak) => {
                (v.passed() && self.streak_qualifies(), Some(v.outcome()))
            }
            _ => (false, None),
        }
    }

    /// Short status-view label, e.g. `"pass 0.94x (1/3)"`, `"pass 0.94x"`, `"fail 1.69x"`.
    pub fn label(&self) -> Option<String> {
        let (v, _) = self.verdict?;
        Some(match v {
            LogprobVerdict::Fail { aggregate } => format!("fail {aggregate:.2}x"),
            LogprobVerdict::Pass { aggregate } if self.streak_qualifies() => format!("pass {aggregate:.2}x"),
            LogprobVerdict::Pass { aggregate } if self.pass_streak == 0 => format!("pass {aggregate:.2}x (streak reset)"),
            LogprobVerdict::Pass { aggregate } => {
                format!("pass {aggregate:.2}x ({}/{LOGPROB_MIN_PASSES})", self.pass_streak.min(LOGPROB_MIN_PASSES))
            }
        })
    }

    const LEN: usize = 66;

    /// `0x03 ‖ kind(0 none, 1 pass, 2 fail) ‖ aggregate f64 ‖ at u64 ‖ pass_streak u32 ‖
    /// streak_since u64 ‖ evasions u32 ‖ evaded_at u64 ‖ next_due u64 ‖ reference [16]` (LE).
    fn to_bytes(self) -> [u8; Self::LEN] {
        let (kind, aggregate, at) = match self.verdict {
            None => (0u8, 0.0, 0),
            Some((LogprobVerdict::Pass { aggregate }, at)) => (1, aggregate, at),
            Some((LogprobVerdict::Fail { aggregate }, at)) => (2, aggregate, at),
        };
        let mut b = Vec::with_capacity(Self::LEN);
        b.push(0x03);
        b.push(kind);
        b.extend_from_slice(&aggregate.to_le_bytes());
        b.extend_from_slice(&at.to_le_bytes());
        b.extend_from_slice(&self.pass_streak.to_le_bytes());
        b.extend_from_slice(&self.streak_since_ms.to_le_bytes());
        b.extend_from_slice(&self.evasions.to_le_bytes());
        b.extend_from_slice(&self.evaded_at_ms.to_le_bytes());
        b.extend_from_slice(&self.next_due_ms.to_le_bytes());
        b.extend_from_slice(&self.reference);
        b.try_into().expect("fixed layout")
    }

    fn from_bytes(b: &[u8]) -> Option<Self> {
        if b.len() != Self::LEN || b[0] != 0x03 {
            return None;
        }
        let u64_at = |i: usize| u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        let u32_at = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap());
        let aggregate = f64::from_le_bytes(b[2..10].try_into().unwrap());
        let at = u64_at(10);
        let verdict = match b[1] {
            0 => None,
            1 if aggregate.is_finite() => Some((LogprobVerdict::Pass { aggregate }, at)),
            2 if aggregate.is_finite() => Some((LogprobVerdict::Fail { aggregate }, at)),
            _ => return None,
        };
        Some(Self {
            verdict,
            pass_streak: u32_at(18),
            streak_since_ms: u64_at(22),
            evasions: u32_at(30),
            evaded_at_ms: u64_at(34),
            next_due_ms: u64_at(42),
            reference: b[50..66].try_into().unwrap(),
        })
    }
}

/// One `(peer, model)`'s persisted state: the status followed by the challenge round.
pub fn encode_logprob_state(status: LogprobStatus, round: &ChallengeRound) -> Vec<u8> {
    let mut b = status.to_bytes().to_vec();
    b.extend_from_slice(&round.to_bytes());
    b
}

/// Inverse of [`encode_logprob_state`]; `None` for a corrupt or older-format row.
pub fn decode_logprob_state(b: &[u8]) -> Option<(LogprobStatus, ChallengeRound)> {
    let status = LogprobStatus::from_bytes(b.get(..LogprobStatus::LEN)?)?;
    let round = ChallengeRound::from_bytes(&b[LogprobStatus::LEN..]).ok()?;
    Some((status, round))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::{AdapterError, DetectedModel, InferenceRequest, ServeOutcome};

    /// Scores position i as −(id of the next token) / 10, or fails / misbehaves on request.
    struct Scorer {
        mode: &'static str,
    }

    impl EngineAdapter for Scorer {
        fn engine_name(&self) -> &'static str {
            "stub"
        }
        fn detect_models(&self) -> Result<Vec<DetectedModel>, AdapterError> {
            Ok(vec![])
        }
        fn serve_stream(&self, _: &InferenceRequest, _: &mut dyn FnMut(&str)) -> Result<ServeOutcome, AdapterError> {
            unreachable!()
        }
        fn score_prompt(&self, _model: &str, ids: &[u32]) -> Result<Vec<f32>, AdapterError> {
            match self.mode {
                "err" => Err(AdapterError::Http("secret engine detail".into())),
                "busy" => Err(AdapterError::Busy("loading a 9 GB model".into())),
                "short" => Ok(vec![-0.1]),
                _ => Ok(ids[1..].iter().map(|&i| -(i as f32) / 10.0).collect()),
            }
        }
    }

    fn roundtrip(engine: Option<&dyn EngineAdapter>, gate: &InflightGate, model: &str, ids: &[u32]) -> Result<Vec<f32>, ScoreChallengeError> {
        let mut transport = |framed: &[u8]| -> Result<Vec<u8>, String> {
            assert_eq!(framed[0], PROMPT_SCORE_CHALLENGE);
            Ok(handle_prompt_score_inbound(&framed[1..], engine, gate, &|m| m == "public"))
        };
        request_prompt_scores(&mut transport, model, ids)
    }

    #[test]
    fn a_challenge_round_trips_the_engine_scores() {
        let gate = InflightGate::new();
        let out = roundtrip(Some(&Scorer { mode: "ok" }), &gate, "public", &[1, 5, 10]).unwrap();
        assert_eq!(out, vec![-0.5, -1.0]);
    }

    #[test]
    fn refusals_are_readable_and_generic() {
        let gate = InflightGate::new();
        let r = |e: Option<&dyn EngineAdapter>, m: &str| roundtrip(e, &gate, m, &[1, 2, 3]);
        assert!(matches!(r(None, "public"), Err(ScoreChallengeError::Refused(s)) if s.contains("not offered")));
        // A private or unknown model gets the same refusal.
        assert_eq!(r(Some(&Scorer { mode: "ok" }), "private"), r(Some(&Scorer { mode: "ok" }), "nope"));
        // Engine errors don't leak through.
        match r(Some(&Scorer { mode: "err" }), "public") {
            Err(ScoreChallengeError::Refused(s)) => assert!(!s.contains("secret")),
            other => panic!("{other:?}"),
        }
        assert!(matches!(r(Some(&Scorer { mode: "short" }), "public"), Err(ScoreChallengeError::Refused(_))));
        // A scorer still starting is a retryable refusal (never a dodge), and leaks nothing either.
        match r(Some(&Scorer { mode: "busy" }), "public") {
            Err(ScoreChallengeError::Refused(s)) => assert!(s.contains("retry") && !s.contains("9 GB"), "{s}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn the_probe_spends_no_reference_text_and_reports_why_a_provider_cannot_score() {
        let gate = InflightGate::new();
        let probe = |engine: Option<&dyn EngineAdapter>| {
            let mut transport = |framed: &[u8]| -> Result<Vec<u8>, String> {
                assert_eq!(decode_challenge(&framed[1..]).unwrap().1, PROBE_TOKENS.to_vec());
                Ok(handle_prompt_score_inbound(&framed[1..], engine, &gate, &|m| m == "public"))
            };
            probe_scoring(&mut transport, "public", &mut || {})
        };
        assert_eq!(probe(Some(&Scorer { mode: "ok" })), None);
        assert!(matches!(probe(None), Some(RoundOutcome::Incomplete(r)) if r.contains("not offered")));
        assert!(matches!(probe(Some(&Scorer { mode: "err" })), Some(RoundOutcome::Incomplete(_))));
        assert!(matches!(probe(Some(&Scorer { mode: "short" })), Some(RoundOutcome::Incomplete(_))));
        let mut legacy = |_: &[u8]| -> Result<Vec<u8>, String> { Ok(b"no magic".to_vec()) };
        assert_eq!(probe_scoring(&mut legacy, "public", &mut || {}), Some(RoundOutcome::Unsupported));
        let mut down = |_: &[u8]| -> Result<Vec<u8>, String> { Err("timeout".into()) };
        assert!(matches!(probe_scoring(&mut down, "public", &mut || {}), Some(RoundOutcome::Incomplete(r)) if r.contains("timeout")));
        // Retryable refusals (busy, a scorer warming up) are waited through, up to the probe's limit.
        for (busy_replies, expect_ok) in [(MAX_PROBE_RETRIES - 1, true), (MAX_PROBE_RETRIES, false)] {
            let (mut sent, mut pauses) = (0, 0);
            let mut flaky = |framed: &[u8]| -> Result<Vec<u8>, String> {
                sent += 1;
                if sent <= busy_replies {
                    return Ok(score_refusal(if sent % 2 == 0 { "prompt scoring busy, retry" } else { "prompt scoring warming up, retry" }));
                }
                Ok(handle_prompt_score_inbound(&framed[1..], Some(&Scorer { mode: "ok" }), &gate, &|_| true))
            };
            let out = probe_scoring(&mut flaky, "public", &mut || pauses += 1);
            assert_eq!(out.is_none(), expect_ok, "{busy_replies} busy replies: {out:?}");
            assert_eq!(pauses, busy_replies.min(MAX_PROBE_RETRIES - 1));
        }
    }

    #[test]
    fn the_gate_caps_concurrent_scoring() {
        let gate = InflightGate::new();
        let a = gate.try_enter().unwrap();
        let _b = gate.try_enter().unwrap();
        assert!(gate.try_enter().is_none());
        assert!(matches!(
            roundtrip(Some(&Scorer { mode: "ok" }), &gate, "public", &[1, 2]),
            Err(ScoreChallengeError::Refused(s)) if s.contains("busy")
        ));
        drop(a);
        assert!(roundtrip(Some(&Scorer { mode: "ok" }), &gate, "public", &[1, 2]).is_ok());
        // The slot taken by that challenge was released.
        assert!(gate.try_enter().is_some());
    }

    #[test]
    fn out_of_bounds_challenges_are_not_sent() {
        assert!(encode_challenge("m", &[1]).is_none());
        assert!(encode_challenge("m", &vec![0; MAX_CHALLENGE_TOKENS + 1]).is_none());
        assert!(encode_challenge(&"x".repeat(MAX_MODEL_REF_LEN + 1), &[1, 2]).is_none());
        assert!(encode_challenge("m", &vec![0; MAX_CHALLENGE_TOKENS]).is_some());
    }

    #[test]
    fn malformed_challenge_bodies_are_refused_without_panicking() {
        let gate = InflightGate::new();
        let engine = Scorer { mode: "ok" };
        let good = encode_challenge("public", &[1, 2, 3]).unwrap();
        for cut in 0..good.len() - 1 {
            let reply = handle_prompt_score_inbound(&good[1..1 + cut], Some(&engine), &gate, &|_| true);
            assert!(reply.starts_with(SCORE_RESP_MAGIC));
            assert_eq!(reply[4], RESP_REFUSED);
        }
        let mut long = good.clone();
        long.push(0);
        assert_eq!(handle_prompt_score_inbound(&long[1..], Some(&engine), &gate, &|_| true)[4], RESP_REFUSED);
    }

    #[test]
    fn a_legacy_or_garbled_reply_is_classified() {
        let mut legacy = |_: &[u8]| -> Result<Vec<u8>, String> { Ok(vec![0x13, 0, 0, 0, 3]) };
        assert_eq!(request_prompt_scores(&mut legacy, "m", &[1, 2]), Err(ScoreChallengeError::Unsupported));
        let mut wrong_count = |_: &[u8]| -> Result<Vec<u8>, String> {
            let mut b = SCORE_RESP_MAGIC.to_vec();
            b.push(RESP_OK);
            b.extend_from_slice(&2u32.to_le_bytes());
            b.extend_from_slice(&[0; 8]);
            Ok(b)
        };
        assert_eq!(request_prompt_scores(&mut wrong_count, "m", &[1, 2]), Err(ScoreChallengeError::Malformed));
        let mut down = |_: &[u8]| -> Result<Vec<u8>, String> { Err("timeout".into()) };
        assert_eq!(request_prompt_scores(&mut down, "m", &[1, 2]), Err(ScoreChallengeError::Transport("timeout".into())));
    }

    // ── reference sets + rounds ──

    fn ref_json(n_texts: usize, min_texts: usize) -> String {
        let texts: Vec<_> = (0..n_texts)
            .map(|i| {
                let ids: Vec<u32> = (0..20).map(|k| (i * 100 + k) as u32).collect();
                let lps: Vec<f32> = ids[1..].iter().map(|&t| -(t as f32) / 10.0).collect();
                serde_json::json!({"id": format!("t{i}"), "token_ids": ids, "logprobs": lps, "band": 0.02})
            })
            .collect();
        serde_json::json!({"model_key": "m@bf16", "handles": ["public"], "min_texts": min_texts, "texts": texts})
            .to_string()
    }

    #[test]
    fn reference_files_parse_and_validate() {
        let r = parse_reference(&ref_json(4, 4)).unwrap();
        assert_eq!(r.handles, vec!["public".to_string()]);
        assert_eq!(r.set.cutoff, DEFAULT_CUTOFF);
        assert_eq!(r.set.texts.len(), 4);
        assert!(parse_reference(&ref_json(3, 4)).unwrap_err().contains("TooFewTexts"));
        assert!(parse_reference("{").is_err());
        let no_handles = ref_json(4, 4).replace(r#"["public"]"#, "[]");
        assert!(parse_reference(&no_handles).unwrap_err().contains("or at least one handle"));
    }

    /// A transport to an in-process provider whose engine scores exactly the reference (honest, the
    /// stub's formula matches `ref_json`) or shifted by `shift` nats per position.
    fn provider(shift: f32, gate: &InflightGate) -> impl FnMut(&[u8]) -> Result<Vec<u8>, String> + '_ {
        struct Shifted(f32);
        impl EngineAdapter for Shifted {
            fn engine_name(&self) -> &'static str {
                "stub"
            }
            fn detect_models(&self) -> Result<Vec<DetectedModel>, AdapterError> {
                Ok(vec![])
            }
            fn serve_stream(&self, _: &InferenceRequest, _: &mut dyn FnMut(&str)) -> Result<ServeOutcome, AdapterError> {
                unreachable!()
            }
            fn score_prompt(&self, _: &str, ids: &[u32]) -> Result<Vec<f32>, AdapterError> {
                Ok(ids[1..].iter().map(|&t| -(t as f32) / 10.0 - self.0).collect())
            }
        }
        move |framed: &[u8]| Ok(handle_prompt_score_inbound(&framed[1..], Some(&Shifted(shift)), gate, &|m| m == "public"))
    }

    #[test]
    fn an_honest_provider_passes_and_a_substitute_fails() {
        let r = parse_reference(&ref_json(10, 4)).unwrap();
        let gate = InflightGate::new();
        let mut round = ChallengeRound::new();
        let mut honest = provider(0.01, &gate);
        match run_round(&mut round, &r.set, "public", &mut honest, &mut |n| n - 1, &mut || {}, &mut |_| {}) {
            RoundOutcome::Verdict(v) => assert!(v.passed(), "{v:?}"),
            other => panic!("{other:?}"),
        }
        let mut sub = provider(0.05, &gate);
        match run_round(&mut round, &r.set, "public", &mut sub, &mut |_| 0, &mut || {}, &mut |_| {}) {
            RoundOutcome::Verdict(v) => assert!(!v.passed(), "{v:?}"),
            other => panic!("{other:?}"),
        }
        // 8 of 10 texts used: a third round plays the last two, then runs out.
        assert_eq!(run_round(&mut round, &r.set, "public", &mut honest, &mut |_| 0, &mut || {}, &mut |_| {}), RoundOutcome::Exhausted);
        assert_eq!((round.progress(), round.issued_count()), (2, 10));
    }

    #[test]
    fn each_text_is_persisted_before_it_is_sent() {
        use std::cell::RefCell;
        let r = parse_reference(&ref_json(4, 4)).unwrap();
        let gate = InflightGate::new();
        let saved = RefCell::new(Vec::<usize>::new()); // issued count at each save
        let at_send = RefCell::new(Vec::<usize>::new()); // latest saved issued count when each text goes out
        let mut honest = provider(0.0, &gate);
        let mut transport = |f: &[u8]| {
            at_send.borrow_mut().push(saved.borrow().last().copied().unwrap_or(0));
            honest(f)
        };
        let mut round = ChallengeRound::new();
        let out = run_round(&mut round, &r.set, "public", &mut transport, &mut |_| 0, &mut || {}, &mut |rd| {
            saved.borrow_mut().push(rd.issued_count())
        });
        assert!(matches!(out, RoundOutcome::Verdict(_)));
        assert_eq!(*at_send.borrow(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn a_legacy_provider_stops_the_round_as_unsupported() {
        let r = parse_reference(&ref_json(4, 4)).unwrap();
        let mut round = ChallengeRound::new();
        let mut legacy = |_: &[u8]| -> Result<Vec<u8>, String> { Ok(vec![0x13, 0, 0, 0]) };
        assert_eq!(run_round(&mut round, &r.set, "public", &mut legacy, &mut |_| 0, &mut || {}, &mut |_| {}), RoundOutcome::Unsupported);
        assert_eq!(round.in_flight(), 0);
    }

    #[test]
    fn busy_refusals_back_off_then_give_up_keeping_progress() {
        let r = parse_reference(&ref_json(12, 4)).unwrap();
        let mut round = ChallengeRound::new();
        let gate = InflightGate::new();
        let mut honest = provider(0.0, &gate);
        let mut calls = 0;
        let mut flaky = |f: &[u8]| -> Result<Vec<u8>, String> {
            calls += 1;
            if calls <= 2 { honest(f) } else { Ok(score_refusal("prompt scoring busy, retry")) }
        };
        let mut pauses = 0;
        let out = run_round(&mut round, &r.set, "public", &mut flaky, &mut |_| 0, &mut || pauses += 1, &mut |_| {});
        assert!(matches!(out, RoundOutcome::Incomplete(ref s) if s.contains("busy")), "{out:?}");
        assert_eq!(pauses, MAX_BUSY_RETRIES - 1);
        assert_eq!(round.progress(), 2);
        // The next attempt picks up where it left off and completes with 2 more texts.
        let mut honest = provider(0.0, &gate);
        assert!(matches!(
            run_round(&mut round, &r.set, "public", &mut honest, &mut |_| 0, &mut || {}, &mut |_| {}),
            RoundOutcome::Verdict(v) if v.passed()
        ));
    }

    #[test]
    fn a_hard_refusal_or_transport_error_stops_the_round() {
        let r = parse_reference(&ref_json(8, 4)).unwrap();
        let gate = InflightGate::new();
        let mut round = ChallengeRound::new();
        let mut private = provider(0.0, &gate);
        assert!(matches!(
            run_round(&mut round, &r.set, "private", &mut private, &mut |_| 0, &mut || {}, &mut |_| {}),
            RoundOutcome::Incomplete(s) if s.contains("not scorable")
        ));
        let mut down = |_: &[u8]| -> Result<Vec<u8>, String> { Err("timeout".into()) };
        assert!(matches!(
            run_round(&mut round, &r.set, "public", &mut down, &mut |_| 0, &mut || {}, &mut |_| {}),
            RoundOutcome::Incomplete(s) if s.contains("timeout")
        ));
    }

    // ── check history (LogprobStatus) ──

    const PASS: LogprobVerdict = LogprobVerdict::Pass { aggregate: 0.94 };
    const FAIL: LogprobVerdict = LogprobVerdict::Fail { aggregate: 1.69 };
    const H: u64 = 60 * 60 * 1000;

    #[test]
    fn a_pass_needs_a_streak_spread_over_the_minimum_span() {
        let mut st = LogprobStatus::default();
        assert_eq!(st.on_verdict(PASS, 0), StatusEffect::None, "a pass earns no audit or reputation");
        assert_eq!(st.label().unwrap(), "pass 0.94x (1/3)");
        st.on_verdict(PASS, 6 * H);
        st.on_verdict(PASS, 12 * H);
        assert_eq!(st.pass_streak, 3);
        assert!(!st.streak_qualifies(), "3 passes inside 12 h don't span 48 h");
        assert!(!st.evidence(12 * H + 1).0);
        st.on_verdict(PASS, 48 * H);
        assert!(st.streak_qualifies());
        assert_eq!(st.evidence(48 * H + 1), (true, Some(VerificationOutcome::Honored)));
        assert_eq!(st.label().unwrap(), "pass 0.94x");
        // Stale after 24 h without a new round.
        assert!(!st.evidence(72 * H).0);
        assert_eq!(st.next_due_ms, 48 * H + 2 * LOGPROB_REFRESH_MS, "a qualified streak is re-checked every 12 h");
    }

    #[test]
    fn a_fail_penalises_and_resets_the_streak() {
        let mut st = LogprobStatus::default();
        st.on_verdict(PASS, 0);
        st.on_verdict(PASS, 30 * H);
        assert_eq!(st.on_verdict(FAIL, 31 * H), StatusEffect::Fail);
        assert_eq!((st.pass_streak, st.streak_since_ms), (0, 0));
        assert_eq!(st.evidence(31 * H + 1), (false, Some(VerificationOutcome::Failed)));
        assert_eq!(st.label().unwrap(), "fail 1.69x");
        st.on_verdict(PASS, 32 * H);
        assert_eq!((st.pass_streak, st.streak_since_ms), (1, 32 * H), "a new streak starts from scratch");
    }

    #[test]
    fn repeated_dodging_after_answering_is_penalised_once_per_run_of_three() {
        let mut st = LogprobStatus::default();
        // First contact: stopping early is never held against a provider.
        for i in 0..5 {
            assert_eq!(st.on_no_verdict(false, LOGPROB_RETRY_MS, i), StatusEffect::None);
        }
        assert_eq!(st.evasions, 0);
        st.on_verdict(PASS, 0);
        st.on_verdict(PASS, 30 * H);
        assert_eq!(st.on_no_verdict(true, LOGPROB_RETRY_MS, 31 * H), StatusEffect::None);
        assert_eq!(st.on_no_verdict(true, LOGPROB_RETRY_MS, 32 * H), StatusEffect::None);
        assert_eq!(st.on_no_verdict(true, LOGPROB_RETRY_MS, 33 * H), StatusEffect::Evasion);
        assert_eq!((st.pass_streak, st.evasions, st.evaded_at_ms), (0, 0, 33 * H));
        assert!(!st.evidence(33 * H + 1).0, "the old pass no longer counts");
        assert_eq!(st.label().unwrap(), "pass 0.94x (streak reset)");
        assert_eq!(st.next_due_ms, 33 * H + LOGPROB_RETRY_MS);
        // A completed round clears the count.
        st.on_no_verdict(true, LOGPROB_RETRY_MS, 34 * H);
        st.on_verdict(PASS, 35 * H);
        assert_eq!(st.evasions, 0);
    }

    #[test]
    fn check_history_round_trips_and_rejects_corruption() {
        let mut round = ChallengeRound::new();
        let r = parse_reference(&ref_json(4, 4)).unwrap();
        round.next_text(&r.set, |_| 0);
        let mut st = LogprobStatus::default();
        st.on_verdict(PASS, 5);
        st.on_verdict(FAIL, 6);
        st.on_verdict(PASS, 7);
        st.on_no_verdict(true, 9, 8);
        for s in [LogprobStatus::default(), st] {
            let (back, rd) = decode_logprob_state(&encode_logprob_state(s, &round)).unwrap();
            assert_eq!(back, s);
            assert_eq!(rd.issued_count(), 1);
        }
        let good = encode_logprob_state(st, &round);
        assert!(decode_logprob_state(&good[..LogprobStatus::LEN - 1]).is_none());
        let mut bad = good.clone();
        bad[1] = 7;
        assert!(decode_logprob_state(&bad).is_none());
        let mut old = good.clone();
        old[0] = 0x02;
        assert!(decode_logprob_state(&old).is_none(), "older format is not misread");
        let mut long = good;
        long.push(0);
        assert!(decode_logprob_state(&long).is_none());
    }

    // ── identity: model info, reference selection, pricing ──

    fn info_transport<'a>(
        enabled: bool,
        info: &'a dyn Fn(&str) -> Option<ModelInfo>,
    ) -> impl FnMut(&[u8]) -> Result<Vec<u8>, String> + 'a {
        move |framed: &[u8]| {
            assert_eq!(framed[0], MODEL_INFO_REQUEST);
            Ok(handle_model_info_inbound(&framed[1..], enabled, &|m| m != "private", info))
        }
    }

    #[test]
    fn model_info_round_trips_and_is_gated_like_scoring() {
        let q4 = ModelInfo { family: "qwen3.5".into(), params: "9b".into(), quant: "Q4_K_M".into(), weight_digest: "".into() };
        let known = |m: &str| (m != "unknown").then(|| q4.clone());
        let mut t = info_transport(true, &known);
        let got = request_model_info(&mut t, "Qwen3.5-9B-Q4_K_M").unwrap();
        assert_eq!(got, q4);
        assert_eq!(got.identity().unwrap().key(), "qwen3.5/9b/q4_k_m");
        // A private model and an undetected one get the same refusal.
        let private = request_model_info(&mut t, "private");
        assert!(matches!(&private, Err(ScoreChallengeError::Refused(_))));
        assert_eq!(private, request_model_info(&mut t, "unknown"));
        // Not opted in.
        let mut off = info_transport(false, &known);
        assert!(matches!(request_model_info(&mut off, "x"), Err(ScoreChallengeError::Refused(r)) if r.contains("not offered")));
        // Legacy provider.
        let mut legacy = |_: &[u8]| -> Result<Vec<u8>, String> { Ok(vec![0x13, 0, 0, 0]) };
        assert_eq!(request_model_info(&mut legacy, "x"), Err(ScoreChallengeError::Unsupported));
        // An empty quant means no usable identity.
        assert_eq!(ModelInfo { quant: "".into(), ..q4.clone() }.identity(), None);
    }

    #[test]
    fn names_hint_at_their_quantisation() {
        assert_eq!(quant_hint_from_name("Qwen3.5-9B-Q4_K_M").as_deref(), Some("q4_k_m"));
        assert_eq!(quant_hint_from_name("models/cheat/Qwen3.5-9B-Q4_K_M.gguf").as_deref(), Some("q4_k_m"));
        assert_eq!(quant_hint_from_name("Qwen/Qwen3.5-9B-FP8").as_deref(), Some("fp8"));
        assert_eq!(quant_hint_from_name("gemma-4-E4B-it-BF16").as_deref(), Some("bf16"));
        assert_eq!(quant_hint_from_name("llama3.2:1b-instruct-q8_0").as_deref(), Some("q8_0"));
        assert_eq!(quant_hint_from_name("Qwen/Qwen3.5-9B"), None);
        assert_eq!(quant_hint_from_name("qwen3.5:9b"), None);
    }

    fn with_identity(json: &str, family: &str, params: &str, quant: &str, handles: &[&str]) -> std::sync::Arc<LoadedReference> {
        let mut v: serde_json::Value = serde_json::from_str(json).unwrap();
        v["model"] = serde_json::json!({"family": family, "params": params, "quant": quant});
        v["model_key"] = serde_json::json!(format!("{family}-{params}@{quant}"));
        v["handles"] = serde_json::json!(handles);
        std::sync::Arc::new(parse_reference(&v.to_string()).unwrap())
    }

    #[test]
    fn references_are_chosen_by_declared_identity_then_by_name() {
        let base = ref_json(4, 4);
        let q4km = with_identity(&base, "qwen3.5", "9b", "q4_k_m", &[]);
        let q40 = with_identity(&base, "qwen3.5", "9b", "Q4_0", &[]);
        let named = with_identity(&base, "qwen3.5", "9b", "bf16", &["Qwen/Qwen3.5-9B"]);
        let refs = vec![q4km.clone(), q40.clone(), named.clone()];
        let id = |q: &str| ModelIdentity::new("Qwen3.5", "9B", q);
        // A renamed Q4_0 file is checked against the Q4_0 reference, whatever its name says.
        assert!(std::sync::Arc::ptr_eq(select_reference(&refs, "Qwen3.5-9B-Q4_K_M", id("Q4_0").as_ref(), None).unwrap(), &q40));
        assert!(std::sync::Arc::ptr_eq(select_reference(&refs, "anything", id("Q4_K_M").as_ref(), None).unwrap(), &q4km));
        // No identity (legacy provider / unknown quant) → by name.
        assert!(std::sync::Arc::ptr_eq(select_reference(&refs, "Qwen/Qwen3.5-9B", None, None).unwrap(), &named));
        assert!(select_reference(&refs, "other", id("Q6_K").as_ref(), None).is_none());
        // A reference needs an identity or a handle.
        let mut bare: serde_json::Value = serde_json::from_str(&base).unwrap();
        bare["handles"] = serde_json::json!([]);
        assert!(parse_reference(&bare.to_string()).unwrap_err().contains("or at least one handle"));
    }

    #[test]
    fn a_reference_tied_to_a_weights_file_applies_only_to_that_file() {
        let base = ref_json(4, 4);
        let with_weights = |w: Option<&str>| {
            let mut v: serde_json::Value = serde_json::from_str(&base).unwrap();
            v["model"] = serde_json::json!({"family": "qwen2.5", "params": "0.5b", "quant": "q4_k_m"});
            v["model_key"] = serde_json::json!("qwen2.5-0.5b@q4_k_m");
            if let Some(w) = w {
                v["weights"] = serde_json::json!(w);
            }
            std::sync::Arc::new(parse_reference(&v.to_string()).unwrap())
        };
        let (a, b) = ("a".repeat(64), "b".repeat(64));
        let ollama = with_weights(Some(&format!("sha256-{a}")));
        let official = with_weights(Some(&format!("sha256:{b}")));
        assert_eq!(ollama.set.weights.as_deref(), Some(format!("sha256:{a}").as_str()), "stored in canonical form");
        let id = ModelIdentity::new("qwen2.5", "0.5b", "q4_k_m");
        let refs = vec![ollama.clone(), official.clone()];
        let pick = |refs: &[std::sync::Arc<LoadedReference>], w: Option<&str>| select_reference(refs, "public", id.as_ref(), w).cloned();
        assert!(std::sync::Arc::ptr_eq(&pick(&refs, Some(&a.to_uppercase())).unwrap(), &ollama));
        assert!(std::sync::Arc::ptr_eq(&pick(&refs, Some(&format!("sha256:{b}"))).unwrap(), &official));
        // Same identity and name, but another or no declared file: neither applies.
        assert!(pick(&refs, Some(&"c".repeat(64))).is_none());
        assert!(pick(&refs, None).is_none());
        // A legacy reference (no file) is the fallback, by identity as before.
        let legacy = with_weights(None);
        let refs = vec![ollama.clone(), legacy.clone()];
        assert!(std::sync::Arc::ptr_eq(&pick(&refs, None).unwrap(), &legacy));
        assert!(std::sync::Arc::ptr_eq(&pick(&refs, Some(&"c".repeat(64))).unwrap(), &legacy));
        assert!(std::sync::Arc::ptr_eq(&pick(&refs, Some(&a)).unwrap(), &ollama));
        // Different files, different tags (a streak earned on one file doesn't carry to another).
        assert_ne!(reference_tag(&ollama.set), reference_tag(&official.set));
        assert_ne!(reference_tag(&ollama.set), reference_tag(&legacy.set));
        // A malformed digest is refused at load.
        let mut bad: serde_json::Value = serde_json::from_str(&base).unwrap();
        bad["weights"] = serde_json::json!("sha256:xyz");
        assert!(parse_reference(&bad.to_string()).unwrap_err().contains("weights"));
    }

    #[test]
    fn switching_reference_clears_the_old_claim() {
        let base = ref_json(8, 4);
        let a = with_identity(&base, "qwen3.5", "9b", "q4_k_m", &[]);
        let b = with_identity(&base, "qwen3.5", "9b", "q4_0", &[]);
        let mut round = ChallengeRound::new();
        let t = round.next_text(&a.set, |_| 0).unwrap().clone();
        round.record(&a.set, &t.id, &t.logprobs).unwrap();
        let mut st = LogprobStatus::default();
        assert!(st.switch_reference(reference_tag(&a.set), &mut round));
        assert_eq!(round.progress(), 0, "first binding also starts the round clean");
        round.next_text(&a.set, |_| 0);
        st.on_verdict(PASS, 0);
        st.next_due_ms = 99;
        assert!(!st.switch_reference(reference_tag(&a.set), &mut round), "same reference: no change");
        assert_eq!(st.pass_streak, 1);
        assert!(st.switch_reference(reference_tag(&b.set), &mut round));
        assert_eq!((st.verdict, st.pass_streak, st.next_due_ms), (None, 0, 99));
        assert_eq!(round.issued_count(), 2, "texts already sent stay burned");
    }

    #[test]
    fn price_follows_the_verified_identity_only_with_a_qualifying_streak() {
        let base = ref_json(4, 4);
        let bf16 = with_identity(&base, "qwen3.5", "9b", "bf16", &[]);
        let refs = vec![bf16.clone()];
        let handle = "Qwen/Qwen3.5-9B";
        let floor = cost_tier_for_handle(handle); // 9B at 4-bit
        let verified = cost_tier_for_params_quant("9b", "bf16").unwrap();
        assert!(verified > floor);
        let mut round = ChallengeRound::new();
        let mut st = LogprobStatus::default();
        st.switch_reference(reference_tag(&bf16.set), &mut round);
        assert_eq!(priced_tier(handle, None, &|t| identity_in(&refs, t), 0), floor);
        st.on_verdict(PASS, 0);
        assert_eq!(priced_tier(handle, Some(&st), &|t| identity_in(&refs, t), 1), floor, "one pass is not enough");
        st.on_verdict(PASS, 30 * H);
        st.on_verdict(PASS, 48 * H);
        assert_eq!(priced_tier(handle, Some(&st), &|t| identity_in(&refs, t), 48 * H + 1), verified);
        // Stale, failed, or against a reference that's no longer loaded → back to the name's price.
        assert_eq!(priced_tier(handle, Some(&st), &|t| identity_in(&refs, t), 72 * H + 1), floor);
        assert_eq!(priced_tier(handle, Some(&st), &|t| identity_in(&[], t), 48 * H + 1), floor);
        st.on_verdict(FAIL, 49 * H);
        assert_eq!(priced_tier(handle, Some(&st), &|t| identity_in(&refs, t), 49 * H + 1), floor);
        // A big name on a small verified model is priced as the small one.
        let small = with_identity(&base, "qwen3.5", "0.8b", "q4_k_m", &[]);
        let mut st = LogprobStatus::default();
        st.switch_reference(reference_tag(&small.set), &mut ChallengeRound::new());
        for at in [0, 24 * H, 48 * H] {
            st.on_verdict(PASS, at);
        }
        let smalls = [small];
        assert_eq!(priced_tier("llama-70b", Some(&st), &|t| identity_in(&smalls, t), 48 * H + 1), cost_tier_for_params_quant("0.8b", "q4_k_m").unwrap());
    }

    #[test]
    fn a_longer_clean_streak_is_checked_less_often_with_random_spread() {
        assert_eq!(refresh_interval_ms(0), 6 * H);
        assert_eq!(refresh_interval_ms(2), 6 * H);
        assert_eq!(refresh_interval_ms(3), 12 * H);
        assert_eq!(refresh_interval_ms(6), 24 * H);
        assert_eq!(validity_ms(0), 24 * H);
        assert_eq!(validity_ms(6), 48 * H, "a 24-hour cadence stays valid across one late check");
        let mut st = LogprobStatus::default();
        for i in 0..6 {
            st.on_verdict(PASS, i * 24 * H);
        }
        assert_eq!(st.next_due_ms, 5 * 24 * H + 24 * H);
        let mut early = st;
        early.on_verdict_jittered(PASS, 200 * H, -1.0);
        let mut late = st;
        late.on_verdict_jittered(PASS, 200 * H, 1.0);
        assert_eq!((early.next_due_ms - 200 * H, late.next_due_ms - 200 * H), (18 * H, 30 * H));
        // Still A2 36 hours after a pass on the 24-hour cadence; a fail drops back to 6-hourly.
        assert!(st.evidence(5 * 24 * H + 36 * H).0);
        st.on_verdict(FAIL, 300 * H);
        assert_eq!(st.next_due_ms, 306 * H);
        assert!(!st.evidence(300 * H + 1).0);
    }
}
