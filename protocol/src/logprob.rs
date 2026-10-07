// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Teacher-forced log-prob identity check — the scoring half of Track A2-Logprob
//! (ATTESTED_ENGINE_TIER_PLAN_v1 §10.1, calibration §4.3.4).
//!
//! A verifier sends a provider a pre-tokenized reference text; the provider's engine scores every
//! position teacher-forced (the log-prob of each actual next token given the prefix) and returns
//! those log-probs. The verifier compares them with a reference run of the claimed model:
//!
//! * per text, the **mean absolute difference** (MAD, nats) against the reference log-probs;
//! * divided by that text's **honest band** — the largest MAD any honest engine / dtype / batching
//!   configuration produced on it — giving a per-text ratio;
//! * the **aggregate** = the mean ratio over a round of distinct texts, compared with a cut-off.
//!
//! The aggregate, not a per-text threshold, is the decision rule: on the 24-text L4 calibration
//! (Qwen3.5-9B, Gemma 4 E4B; vLLM FP16 / BF16-batched, llama.cpp BF16, SGLang, mistral.rs) honest
//! engines scored ≤ 1.10× leave-one-engine-out and every substitute ≥ 1.44× (Q8_0 the closest, a
//! fine-tune ~21×), while a per-text rule misfires on Q8_0 because batching alone moves honest
//! scores by up to a whole band. Hence [`DEFAULT_CUTOFF`] = 1.35 and [`DEFAULT_MIN_TEXTS`] = 24 —
//! the cut-off is only calibrated for rounds of that size.
//!
//! **What a pass proves, and what it does not.** A pass shows the provider could produce the
//! claimed model's distribution *for the challenged requests*, within the verifier's deadline. It
//! does not by itself show real traffic is served by that model: a provider that can tell challenges
//! from traffic can route only the challenges to the honest model. And the reference texts must be
//! held by the verifier and never re-sent to the same provider — a text a provider has seen can be
//! scored offline by the honest model and replayed ([`ChallengeRound`] enforces the no-reuse half).
//!
//! Pure and synchronous: no I/O, no engine calls, no wire format. The agent fetches the scores and
//! feeds them here; a passing round sets [`MethodFlags::logprob_commitment`](crate::trust::MethodFlags).

use std::collections::BTreeSet;

use sha2::{Digest, Sha256};

use crate::award::AuditClass;
use crate::verify::VerificationOutcome;

/// Curated, version-pinned id of this verifier (same `name@N` convention as
/// [`crate::trust::RedundantExecAuthenticity::ID`]).
pub const LOGPROB_VERIFIER_ID: &str = "teacher-forced-logprob@1";

/// Aggregate ratio at or below which a round passes. Calibrated on the 24-text L4 set: honest
/// ≤ 1.10×, substitutes ≥ 1.44×.
pub const DEFAULT_CUTOFF: f64 = 1.35;

/// Texts in one round — the cut-off is calibrated for aggregates over this many texts.
pub const DEFAULT_MIN_TEXTS: usize = 24;

/// A log-prob above this is not a log-prob (rounding in `log_softmax` can land a hair above 0).
const MAX_LOGPROB: f32 = 1e-3;

/// Shortest text worth scoring: fewer positions make the MAD too noisy to mean anything.
pub const MIN_TEXT_TOKENS: usize = 16;

/// One reference text for one model: the token ids to send, and the reference log-probs.
#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceText {
    /// Verifier-side id, unique within its [`ReferenceSet`]. Never sent to the provider.
    pub id: String,
    /// The pre-tokenized text, in the claimed model's tokenizer.
    pub token_ids: Vec<u32>,
    /// Reference log-prob of `token_ids[i + 1]` given `token_ids[..=i]` — one fewer than
    /// `token_ids` (the first token has no prefix to score it from).
    pub logprobs: Vec<f32>,
    /// Honest band for this text, in nats: the largest MAD against `logprobs` that any honest
    /// engine configuration produced. Must be positive.
    pub band: f64,
}

/// Which exact model a reference was made from: family, size and the **precise** quantisation
/// (`q4_k_m`, not the canonical id's folded `int4` — neighbouring quants need different references).
/// Matched against the identity a provider declares for the model it serves.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ModelIdentity {
    pub family: String,
    pub params: String,
    pub quant: String,
}

impl ModelIdentity {
    /// Build a normalised identity: lower-cased and trimmed, with the float spellings folded
    /// (`BF16`/`bfloat16` → `bf16`, `F16`/`float16` → `fp16`, `F32` → `fp32`). GGUF quant labels
    /// stay precise (`Q4_K_M` → `q4_k_m`). `None` if any part is empty or the size is `unknown`.
    pub fn new(family: &str, params: &str, quant: &str) -> Option<Self> {
        let family = family.trim().to_lowercase();
        let params = params.trim().to_lowercase();
        let quant = match quant.trim().to_lowercase().as_str() {
            "bf16" | "bfloat16" => "bf16".to_string(),
            "f16" | "fp16" | "float16" | "half" => "fp16".to_string(),
            "f32" | "fp32" | "float32" => "fp32".to_string(),
            q => q.to_string(),
        };
        if family.is_empty() || params.is_empty() || params == "unknown" || quant.is_empty() {
            return None;
        }
        Some(Self { family, params, quant })
    }

    /// `family/params/quant`, e.g. `qwen3.5/9b/q4_k_m`.
    pub fn key(&self) -> String {
        format!("{}/{}/{}", self.family, self.params, self.quant)
    }
}

/// A weights-file digest in its one canonical form, `sha256:<64 lower-case hex>`. Accepts that form,
/// Ollama's blob spelling `sha256-<hex>`, or bare hex; `None` for anything else.
pub fn normalize_weights_digest(digest: &str) -> Option<String> {
    let d = digest.trim();
    let hex = d.strip_prefix("sha256:").or_else(|| d.strip_prefix("sha256-")).unwrap_or(d);
    (hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit())).then(|| format!("sha256:{}", hex.to_ascii_lowercase()))
}

/// Every reference text for one claimed model, plus the decision parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceSet {
    /// A label for the reference (e.g. `qwen3.5-9b@q4_k_m`), for logs.
    pub model_key: String,
    /// The exact model the reference was made from, when known — how a provider's declared
    /// identity is matched to it (name aliases are the fallback).
    pub identity: Option<ModelIdentity>,
    /// Digest of the weights file the reference was scored from ([`normalize_weights_digest`] form).
    /// The check tells weights **files** apart, not quantisation labels: two honest `Q4_K_M` files
    /// of one model from different quantisers differ as much as a cheaper substitute does. So a set
    /// with a digest applies only to a provider declaring that same file; one without (legacy)
    /// falls back to the identity and names.
    pub weights: Option<String>,
    pub texts: Vec<ReferenceText>,
    /// Aggregate cut-off; see [`DEFAULT_CUTOFF`].
    pub cutoff: f64,
    /// Texts per round; see [`DEFAULT_MIN_TEXTS`].
    pub min_texts: usize,
    /// Where a round may stop before `min_texts` (calibrated by [`build_reference`]); empty = never.
    pub early_stop: Vec<Checkpoint>,
}

/// An early-stopping point in a round: after `after` texts, the running mean ratio decides the round
/// if it is clearly on one side. Never looser or harsher than the full round: `fail_above` is at least
/// the cut-off, `pass_below` at most the cut-off; in between, the round goes on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Checkpoint {
    pub after: usize,
    /// Pass early if the mean is below this (substitutes' low end at this point). `None` when the
    /// reference was built without substitutes: no evidence that a low mean excludes them.
    pub pass_below: Option<f64>,
    /// Fail early if the mean is above this (honest runs' high end at this point).
    pub fail_above: f64,
}

/// Why a reference set can't be used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReferenceError {
    /// A text id appears twice.
    DuplicateId(String),
    /// Two texts have the same token sequence (it would count twice in a round).
    DuplicateText(String),
    /// A text is too short, or its log-prob count isn't `token_ids.len() - 1`.
    BadShape(String),
    /// A reference log-prob is non-finite or positive.
    BadLogprob(String),
    /// A band is non-finite or not positive.
    BadBand(String),
    /// The cut-off is non-finite or not positive, `min_texts` is 0, an early-stop checkpoint is out
    /// of order, past the round size, or looser/harsher than the cut-off, or the weights digest
    /// isn't in canonical form.
    BadParams,
    /// Tied to a weights file but with no model identity. A digest match bypasses the model's name,
    /// so without an identity a pass would leave the price to the name the provider chose: a
    /// small model's file passing under a big model's name.
    WeightsWithoutIdentity,
    /// Fewer texts than one round needs.
    TooFewTexts { have: usize, need: usize },
}

impl ReferenceText {
    /// SHA-256 of the token ids (u32 little-endian). What a provider has *seen* is the token
    /// sequence, not the verifier's label for it, so the no-reuse record is keyed on this: renaming
    /// a text in a new reference file can't make it look fresh.
    pub fn digest(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        for id in &self.token_ids {
            h.update(id.to_le_bytes());
        }
        h.finalize().into()
    }
}

impl ReferenceSet {
    /// Check the set is internally consistent. Call once on load; the scorer assumes a valid set.
    pub fn validate(&self) -> Result<(), ReferenceError> {
        self.validate_texts()?;
        if self.texts.len() < self.min_texts {
            return Err(ReferenceError::TooFewTexts { have: self.texts.len(), need: self.min_texts });
        }
        Ok(())
    }

    /// As [`validate`](Self::validate) but without requiring a full round of texts — for a first-stage
    /// round pack, which holds only enough texts to reach the first early-stop checkpoint.
    pub fn validate_texts(&self) -> Result<(), ReferenceError> {
        if !(self.cutoff.is_finite() && self.cutoff > 0.0) || self.min_texts == 0 {
            return Err(ReferenceError::BadParams);
        }
        if self.weights.as_deref().is_some_and(|w| normalize_weights_digest(w).as_deref() != Some(w)) {
            return Err(ReferenceError::BadParams);
        }
        if self.weights.is_some() && self.identity.is_none() {
            return Err(ReferenceError::WeightsWithoutIdentity);
        }
        let mut prev = 0;
        for cp in &self.early_stop {
            let pass_ok = cp.pass_below.map_or(true, |p| p.is_finite() && p > 0.0 && p <= self.cutoff);
            let fail_ok = cp.fail_above.is_finite() && cp.fail_above >= self.cutoff;
            if cp.after <= prev || cp.after >= self.min_texts || !pass_ok || !fail_ok {
                return Err(ReferenceError::BadParams);
            }
            prev = cp.after;
        }
        let mut ids = BTreeSet::new();
        let mut digests = BTreeSet::new();
        for t in &self.texts {
            if !ids.insert(t.id.as_str()) {
                return Err(ReferenceError::DuplicateId(t.id.clone()));
            }
            if !digests.insert(t.digest()) {
                return Err(ReferenceError::DuplicateText(t.id.clone()));
            }
            if t.token_ids.len() < MIN_TEXT_TOKENS || t.logprobs.len() + 1 != t.token_ids.len() {
                return Err(ReferenceError::BadShape(t.id.clone()));
            }
            if t.logprobs.iter().any(|&lp| !valid_logprob(lp)) {
                return Err(ReferenceError::BadLogprob(t.id.clone()));
            }
            if !(t.band.is_finite() && t.band > 0.0) {
                return Err(ReferenceError::BadBand(t.id.clone()));
            }
        }
        Ok(())
    }

    /// The text with this id.
    pub fn text(&self, id: &str) -> Option<&ReferenceText> {
        self.texts.iter().find(|t| t.id == id)
    }
}

fn valid_logprob(lp: f32) -> bool {
    lp.is_finite() && lp <= MAX_LOGPROB
}

/// Why a provider's reply to one text can't be scored. None of these is evidence of substitution —
/// a malformed reply proves nothing either way — so they never fail a round.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScoreError {
    /// Wrong number of log-probs for the text.
    WrongLength { expected: usize, got: usize },
    /// A log-prob is NaN, infinite, or positive.
    InvalidLogprob,
}

/// MAD between the reference and observed log-probs, in nats. Rejects malformed replies rather than
/// clamping them.
pub fn text_mad(reference: &[f32], observed: &[f32]) -> Result<f64, ScoreError> {
    if observed.len() != reference.len() || reference.is_empty() {
        return Err(ScoreError::WrongLength { expected: reference.len(), got: observed.len() });
    }
    if observed.iter().any(|&lp| !valid_logprob(lp)) {
        return Err(ScoreError::InvalidLogprob);
    }
    let sum: f64 = reference.iter().zip(observed).map(|(&r, &o)| (f64::from(r) - f64::from(o)).abs()).sum();
    Ok(sum / reference.len() as f64)
}

/// A text's MAD as a multiple of its honest band.
pub fn text_ratio(text: &ReferenceText, observed: &[f32]) -> Result<f64, ScoreError> {
    Ok(text_mad(&text.logprobs, observed)? / text.band)
}

/// The decision for one completed round.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LogprobVerdict {
    /// Aggregate at or below the cut-off: consistent with the claimed model.
    Pass { aggregate: f64 },
    /// Aggregate above the cut-off: a different model or quantization.
    Fail { aggregate: f64 },
}

impl LogprobVerdict {
    pub fn aggregate(self) -> f64 {
        match self {
            Self::Pass { aggregate } | Self::Fail { aggregate } => aggregate,
        }
    }

    /// The reputation outcome: a pass is [`Honored`](VerificationOutcome::Honored), a fail is
    /// [`Failed`](VerificationOutcome::Failed) — the same weight as a redundant-exec divergence.
    pub fn outcome(self) -> VerificationOutcome {
        match self {
            Self::Pass { .. } => VerificationOutcome::Honored,
            Self::Fail { .. } => VerificationOutcome::Failed,
        }
    }

    /// The audit-history class ([`crate::award::AuditStats`]).
    pub fn audit_class(self) -> AuditClass {
        match self {
            Self::Pass { .. } => AuditClass::Pass,
            Self::Fail { .. } => AuditClass::Fail,
        }
    }

    /// Whether this round earns the log-prob method flag.
    pub fn passed(self) -> bool {
        matches!(self, Self::Pass { .. })
    }
}

/// Mean of per-text ratios against a cut-off. `None` for an empty slice.
pub fn aggregate_verdict(ratios: &[f64], cutoff: f64) -> Option<LogprobVerdict> {
    if ratios.is_empty() {
        return None;
    }
    let aggregate = ratios.iter().sum::<f64>() / ratios.len() as f64;
    Some(if aggregate <= cutoff { LogprobVerdict::Pass { aggregate } } else { LogprobVerdict::Fail { aggregate } })
}

/// Why [`ChallengeRound::record`] refused a reply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoundError {
    /// No text with this id in the reference set.
    UnknownText(String),
    /// This text was not issued by [`ChallengeRound::next_text`], or its reply was already recorded.
    NotIssued(String),
    /// The reply couldn't be scored.
    Score(ScoreError),
}

/// One `(provider, model)`'s challenge state: which texts it has been sent (ever), and the ratios of
/// the round in progress. A text is issued at most once per provider, so it can't be replayed from
/// an offline honest run. Pure; the agent persists or rebuilds it as it sees fit.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChallengeRound {
    /// [`ReferenceText::digest`] of every text ever issued to this provider.
    issued: BTreeSet<[u8; 32]>,
    /// Issued, awaiting a reply.
    pending: BTreeSet<String>,
    /// Ratios recorded in the current round.
    ratios: Vec<f64>,
}

impl ChallengeRound {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pick an unissued text and mark it issued. `pick(n)` must return an index below `n` (the
    /// caller supplies the randomness, so the order is unpredictable to the provider). `None` when
    /// the set is exhausted for this provider — the verifier needs fresh texts.
    pub fn next_text<'a>(&mut self, set: &'a ReferenceSet, pick: impl FnOnce(usize) -> usize) -> Option<&'a ReferenceText> {
        self.next_text_where(set, pick, |_| true)
    }

    /// [`next_text`](Self::next_text) among the texts `allowed` lets through as well (e.g. ones this
    /// verifier has sent no other provider either).
    pub fn next_text_where<'a>(
        &mut self,
        set: &'a ReferenceSet,
        pick: impl FnOnce(usize) -> usize,
        allowed: impl Fn(&ReferenceText) -> bool,
    ) -> Option<&'a ReferenceText> {
        let fresh: Vec<&ReferenceText> =
            set.texts.iter().filter(|t| !self.issued.contains(&t.digest()) && allowed(t)).collect();
        if fresh.is_empty() {
            return None;
        }
        let text = fresh[pick(fresh.len()).min(fresh.len() - 1)];
        self.issued.insert(text.digest());
        self.pending.insert(text.id.clone());
        Some(text)
    }

    /// A text from `set` issued and not yet answered or given up on: sent again before any new one,
    /// so a provider can't skip a text by stalling on it.
    pub fn pending_text<'a>(&self, set: &'a ReferenceSet) -> Option<&'a ReferenceText> {
        set.texts.iter().find(|t| self.pending.contains(&t.id))
    }

    /// The digests of every text ever issued to this provider.
    pub fn issued_digests(&self) -> impl Iterator<Item = &[u8; 32]> {
        self.issued.iter()
    }

    /// Give up on an issued text (timeout, refusal, transport error). It stays issued — the provider
    /// has seen it — but no longer counts as pending.
    pub fn abandon(&mut self, id: &str) {
        self.pending.remove(id);
    }

    /// Score one reply. When the round reaches `set.min_texts` ratios it closes: the verdict is
    /// returned and the next round starts empty.
    pub fn record(&mut self, set: &ReferenceSet, id: &str, observed: &[f32]) -> Result<Option<LogprobVerdict>, RoundError> {
        let text = set.text(id).ok_or_else(|| RoundError::UnknownText(id.to_string()))?;
        if !self.pending.remove(id) {
            return Err(RoundError::NotIssued(id.to_string()));
        }
        let ratio = text_ratio(text, observed).map_err(RoundError::Score)?;
        self.ratios.push(ratio);
        let n = self.ratios.len();
        let verdict = if n >= set.min_texts {
            aggregate_verdict(&self.ratios, set.cutoff)
        } else if let Some(cp) = set.early_stop.iter().find(|c| c.after == n) {
            let aggregate = self.ratios.iter().sum::<f64>() / n as f64;
            if aggregate > cp.fail_above {
                Some(LogprobVerdict::Fail { aggregate })
            } else if cp.pass_below.is_some_and(|p| aggregate < p) {
                Some(LogprobVerdict::Pass { aggregate })
            } else {
                None
            }
        } else {
            None
        };
        if verdict.is_some() {
            self.ratios.clear();
        }
        Ok(verdict)
    }

    /// Drop the ratios of the round in progress (e.g. the provider now declares a different model,
    /// so they were scored against another reference). Issued texts stay issued.
    pub fn reset_progress(&mut self) {
        self.ratios.clear();
    }

    /// Ratios recorded so far in the current round.
    pub fn progress(&self) -> usize {
        self.ratios.len()
    }

    /// Texts issued and not yet answered or abandoned.
    pub fn in_flight(&self) -> usize {
        self.pending.len()
    }

    /// Whether another full round can still be drawn from `set` for this provider.
    pub fn can_complete_round(&self, set: &ReferenceSet) -> bool {
        let fresh = set.texts.iter().filter(|t| !self.issued.contains(&t.digest())).count();
        fresh + self.progress() + self.in_flight() >= set.min_texts
    }

    /// Texts ever issued to this provider.
    pub fn issued_count(&self) -> usize {
        self.issued.len()
    }

    /// Serialize for persistence: `0x01 ‖ u32 n ‖ n × digest(32) ‖ u32 m ‖ m × f64` (all LE). Texts
    /// still pending are not written — after a restart they count as abandoned (still issued).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(9 + 32 * self.issued.len() + 8 * self.ratios.len());
        b.push(ROUND_FORMAT_V1);
        b.extend_from_slice(&(self.issued.len() as u32).to_le_bytes());
        for d in &self.issued {
            b.extend_from_slice(d);
        }
        b.extend_from_slice(&(self.ratios.len() as u32).to_le_bytes());
        for r in &self.ratios {
            b.extend_from_slice(&r.to_le_bytes());
        }
        b
    }

    /// Inverse of [`to_bytes`](Self::to_bytes). Rejects truncated, over-long, or non-finite input.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        let mut rest = data;
        let mut take = |n: usize| -> Result<&[u8], String> {
            if rest.len() < n {
                return Err("challenge round: truncated".into());
            }
            let (head, tail) = rest.split_at(n);
            rest = tail;
            Ok(head)
        };
        if take(1)?[0] != ROUND_FORMAT_V1 {
            return Err("challenge round: unknown format".into());
        }
        let n = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
        let mut issued = BTreeSet::new();
        for _ in 0..n {
            issued.insert(<[u8; 32]>::try_from(take(32)?).unwrap());
        }
        let m = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
        let mut ratios = Vec::with_capacity(m.min(4096));
        for _ in 0..m {
            let r = f64::from_le_bytes(take(8)?.try_into().unwrap());
            if !r.is_finite() || r < 0.0 {
                return Err("challenge round: bad ratio".into());
            }
            ratios.push(r);
        }
        if !rest.is_empty() {
            return Err("challenge round: trailing bytes".into());
        }
        Ok(Self { issued, pending: BTreeSet::new(), ratios })
    }
}

const ROUND_FORMAT_V1: u8 = 0x01;

// ── Building a reference set from engine runs ──
//
// A reference is built offline by the verifier: the texts are scored teacher-forced by a reference
// run (the claimed model on a trusted engine), by several **honest** runs (the same weights on other
// engines / backends / batch settings), and optionally by **substitute** runs (cheaper quants,
// fine-tunes — what a cheater might serve). Each text's honest band is the largest honest MAD; the
// cut-off is calibrated on random rounds of `min_texts` texts, between how high honest runs score
// (each against a band built without its group) and how low substitutes score.

/// One engine configuration's teacher-forced log-probs for every text of a build, in text order.
#[derive(Debug, Clone, PartialEq)]
pub struct RunScores {
    pub label: String,
    /// Honest runs of one kind of setup (e.g. two CPU builds of the same engine) share a group and
    /// are left out together in calibration, so near-identical runs can't vouch for each other and
    /// shrink the tolerance for a kind of setup the build never measured. `None`: its own group.
    pub group: Option<String>,
    /// `logprobs[t]` has one entry per token of text `t` after the first.
    pub logprobs: Vec<Vec<f32>>,
}

/// Knobs for [`build_reference`].
#[derive(Debug, Clone, PartialEq)]
pub struct BuildParams {
    /// Label for the reference (e.g. `qwen3.5-9b@q4_k_m`); text ids are derived from it.
    pub label: String,
    pub identity: Option<ModelIdentity>,
    /// Digest of the weights file the reference run scored ([`ReferenceSet::weights`]).
    pub weights: Option<String>,
    /// Texts per verification round (the cut-off is calibrated for this round size).
    pub min_texts: usize,
    /// Smallest band allowed, in nats, so an honest run that happens to be bit-identical to the
    /// reference can't shrink a text's band to zero.
    pub band_floor: f64,
    /// Also floor each band at this fraction of the median band.
    pub band_floor_of_median: f64,
    /// Random rounds drawn for calibration (deterministic from `seed`).
    pub rounds: usize,
    pub seed: u64,
    /// Honest runs must stay below the cut-off in this fraction of calibration rounds, substitutes
    /// above it in the same fraction (e.g. 0.995).
    pub coverage: f64,
    /// The substitutes' low end must be at least this multiple of the honest high end.
    pub min_gap: f64,
    /// Early-stop checkpoints to calibrate (text counts below `min_texts`).
    pub checkpoints: Vec<usize>,
    /// Coverage for the early-stop thresholds — stricter than `coverage`, since a round may stop at
    /// any checkpoint (e.g. 0.999).
    pub early_coverage: f64,
}

impl BuildParams {
    pub fn new(label: impl Into<String>, identity: Option<ModelIdentity>) -> Self {
        Self {
            label: label.into(),
            identity,
            weights: None,
            min_texts: DEFAULT_MIN_TEXTS,
            band_floor: 1e-3,
            band_floor_of_median: 0.25,
            rounds: 2000,
            seed: 0x00DD_BA11,
            coverage: 0.995,
            min_gap: 1.15,
            checkpoints: vec![8, 12, 16],
            early_coverage: 0.999,
        }
    }
}

/// How the cut-off was chosen, for the build report.
#[derive(Debug, Clone, PartialEq)]
pub struct Calibration {
    /// Each honest run's aggregate over all texts, against a band built without its group.
    pub honest: Vec<(String, f64)>,
    /// Each substitute run's aggregate over all texts.
    pub substitutes: Vec<(String, f64)>,
    /// The `coverage` quantile of honest round aggregates (the cut-off uses it floored at 1.0).
    pub honest_high: f64,
    /// The `1 − coverage` quantile of substitute round aggregates (lowest across substitutes).
    pub substitute_low: Option<f64>,
    pub cutoff: f64,
    /// False when no substitutes were given: the cut-off then only protects honest providers and
    /// says nothing about what it would catch.
    pub calibrated: bool,
    /// The calibrated early-stop checkpoints (as stored on the set).
    pub early_stop: Vec<Checkpoint>,
}

/// Why a reference couldn't be built.
#[derive(Debug, Clone, PartialEq)]
pub enum BuildError {
    /// Fewer texts than one round.
    TooFewTexts { have: usize, need: usize },
    /// Leave-one-group-out calibration needs honest runs from at least two groups (an ungrouped
    /// run is its own group). Holds the number of groups given.
    TooFewHonestGroups(usize),
    /// A run's scores don't line up with the texts (wrong count or length), or hold invalid values.
    BadRun { run: String, text: usize },
    /// Honest and substitute scores overlap: no cut-off separates them with the required gap.
    NoSeparation { honest_high: f64, substitute_low: f64 },
    /// The assembled set failed validation.
    Invalid(ReferenceError),
}

/// Deterministic splitmix64 — enough to draw calibration rounds reproducibly without a dependency.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// `k` distinct indices below `n` (partial Fisher–Yates).
    fn sample(&mut self, n: usize, k: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..n).collect();
        for i in 0..k {
            let j = i + (self.next() % (n - i) as u64) as usize;
            idx.swap(i, j);
        }
        idx.truncate(k);
        idx
    }
}

fn quantile(values: &mut [f64], q: f64) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let pos = ((values.len() - 1) as f64 * q.clamp(0.0, 1.0)).round() as usize;
    values[pos]
}

/// Per-text bands from the honest MADs (`mads[run][text]`), each floored.
fn bands(mads: &[&Vec<f64>], p: &BuildParams) -> Vec<f64> {
    let n = mads[0].len();
    let raw: Vec<f64> = (0..n).map(|t| mads.iter().map(|m| m[t]).fold(0.0, f64::max)).collect();
    let mut sorted = raw.clone();
    let median = quantile(&mut sorted, 0.5);
    let floor = p.band_floor.max(p.band_floor_of_median * median);
    raw.into_iter().map(|b| b.max(floor)).collect()
}

/// Build a [`ReferenceSet`] from engine runs over the same pre-tokenized texts, and calibrate its
/// cut-off. Refuses (`NoSeparation`) when substitutes don't clear the honest runs by `min_gap`.
pub fn build_reference(
    token_ids: &[Vec<u32>],
    reference: &RunScores,
    honest: &[RunScores],
    substitutes: &[RunScores],
    p: &BuildParams,
) -> Result<(ReferenceSet, Calibration), BuildError> {
    let n = token_ids.len();
    if n < p.min_texts || n == 0 {
        return Err(BuildError::TooFewTexts { have: n, need: p.min_texts.max(1) });
    }
    let same_group = |k: usize, j: usize| k == j || matches!((&honest[k].group, &honest[j].group), (Some(a), Some(b)) if a == b);
    let groups = (0..honest.len()).filter(|&k| !(0..k).any(|j| same_group(k, j))).count();
    if groups < 2 {
        return Err(BuildError::TooFewHonestGroups(groups));
    }
    let check = |run: &RunScores| -> Result<(), BuildError> {
        if run.logprobs.len() != n {
            return Err(BuildError::BadRun { run: run.label.clone(), text: run.logprobs.len().min(n) });
        }
        for (t, (lps, ids)) in run.logprobs.iter().zip(token_ids).enumerate() {
            if lps.len() + 1 != ids.len() || lps.iter().any(|&lp| !valid_logprob(lp)) {
                return Err(BuildError::BadRun { run: run.label.clone(), text: t });
            }
        }
        Ok(())
    };
    for run in std::iter::once(reference).chain(honest).chain(substitutes) {
        check(run)?;
    }
    let mads = |run: &RunScores| -> Vec<f64> {
        (0..n).map(|t| text_mad(&reference.logprobs[t], &run.logprobs[t]).expect("checked")).collect()
    };
    let honest_mads: Vec<Vec<f64>> = honest.iter().map(mads).collect();
    let sub_mads: Vec<Vec<f64>> = substitutes.iter().map(mads).collect();
    let band = bands(&honest_mads.iter().collect::<Vec<_>>(), p);
    // Each honest run against a band built from the other groups (neither it nor a run of the same
    // kind may vouch for it): the stand-in for a kind of setup the build never measured.
    let loo_bands: Vec<Vec<f64>> = (0..honest.len())
        .map(|k| bands(&honest_mads.iter().enumerate().filter(|(j, _)| !same_group(k, *j)).map(|(_, m)| m).collect::<Vec<_>>(), p))
        .collect();
    let mean_ratio = |m: &[f64], b: &[f64], idx: &[usize]| idx.iter().map(|&t| m[t] / b[t]).sum::<f64>() / idx.len() as f64;
    let all: Vec<usize> = (0..n).collect();
    let honest_full: Vec<(String, f64)> =
        honest.iter().zip(&honest_mads).zip(&loo_bands).map(|((r, m), b)| (r.label.clone(), mean_ratio(m, b, &all))).collect();
    let sub_full: Vec<(String, f64)> =
        substitutes.iter().zip(&sub_mads).map(|(r, m)| (r.label.clone(), mean_ratio(m, &band, &all))).collect();

    let mut rng = SplitMix(p.seed);
    let rounds = if n == p.min_texts { 1 } else { p.rounds.max(1) };
    let mut honest_rounds = Vec::with_capacity(rounds * honest.len());
    let mut sub_rounds: Vec<Vec<f64>> = vec![Vec::with_capacity(rounds); substitutes.len()];
    for _ in 0..rounds {
        let idx = rng.sample(n, p.min_texts);
        for (m, b) in honest_mads.iter().zip(&loo_bands) {
            honest_rounds.push(mean_ratio(m, b, &idx));
        }
        for (acc, m) in sub_rounds.iter_mut().zip(&sub_mads) {
            acc.push(mean_ratio(m, &band, &idx));
        }
    }
    let honest_high = quantile(&mut honest_rounds, p.coverage);
    let substitute_low = sub_rounds
        .iter_mut()
        .map(|r| quantile(r, 1.0 - p.coverage))
        .min_by(|a, b| a.partial_cmp(b).expect("finite"));
    // Honest tolerance is at least one band: runs that agree unusually closely (scores well below
    // 1×) must not pull the cut-off down so far that an honest backend missing from the build fails.
    let honest_edge = honest_high.max(1.0);
    let (cutoff, calibrated) = match substitute_low {
        Some(low) if low < honest_edge * p.min_gap => {
            return Err(BuildError::NoSeparation { honest_high, substitute_low: low });
        }
        // Geometric midpoint: equal headroom (as a ratio) on both sides.
        Some(low) => ((honest_edge * low).sqrt(), true),
        None => (honest_edge * p.min_gap, false),
    };
    // Early stopping: at each checkpoint m, the honest runs' high end and the substitutes' low end
    // over random partial rounds of m texts, clamped so a checkpoint is never looser or harsher than
    // the full round.
    let mut early_stop = Vec::new();
    for &m in p.checkpoints.iter().filter(|&&m| m > 0 && m < p.min_texts && m <= n) {
        let mut hs = Vec::with_capacity(rounds * honest.len());
        let mut ss: Vec<Vec<f64>> = vec![Vec::with_capacity(rounds); substitutes.len()];
        for _ in 0..p.rounds.max(1) {
            let idx = rng.sample(n, m);
            for (mm, b) in honest_mads.iter().zip(&loo_bands) {
                hs.push(mean_ratio(mm, b, &idx));
            }
            for (acc, mm) in ss.iter_mut().zip(&sub_mads) {
                acc.push(mean_ratio(mm, &band, &idx));
            }
        }
        let fail_above = quantile(&mut hs, p.early_coverage).max(cutoff);
        let pass_below = ss
            .iter_mut()
            .map(|v| quantile(v, 1.0 - p.early_coverage))
            .min_by(|a, b| a.partial_cmp(b).expect("finite"))
            .map(|low| low.min(cutoff));
        early_stop.push(Checkpoint { after: m, pass_below, fail_above });
    }
    let set = ReferenceSet {
        model_key: p.label.clone(),
        identity: p.identity.clone(),
        weights: p.weights.clone(),
        texts: (0..n)
            .map(|t| ReferenceText {
                id: format!("{}#{t:05}", p.label),
                token_ids: token_ids[t].clone(),
                logprobs: reference.logprobs[t].clone(),
                band: band[t],
            })
            .collect(),
        cutoff,
        min_texts: p.min_texts,
        early_stop: early_stop.clone(),
    };
    set.validate().map_err(BuildError::Invalid)?;
    Ok((set, Calibration { honest: honest_full, substitutes: sub_full, honest_high, substitute_low, cutoff, calibrated, early_stop }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(id: &str, n: usize, band: f64) -> ReferenceText {
        ReferenceText {
            id: id.to_string(),
            token_ids: (0..n as u32).collect(),
            logprobs: (0..n - 1).map(|i| -((i % 7) as f32) * 0.5 - 0.01).collect(),
            band,
        }
    }

    fn set(n_texts: usize, min_texts: usize) -> ReferenceSet {
        ReferenceSet {
            model_key: "qwen3.5-9b@bf16".into(),
            identity: ModelIdentity::new("qwen3.5", "9b", "bf16"),
            weights: None,
            texts: (0..n_texts)
                .map(|i| {
                    let mut t = text(&format!("t{i}"), 32, 0.02);
                    t.token_ids.iter_mut().for_each(|id| *id += 1000 * i as u32);
                    t
                })
                .collect(),
            cutoff: DEFAULT_CUTOFF,
            min_texts,
            early_stop: vec![],
        }
    }

    /// Observed log-probs shifted from the reference by exactly `mad` nats at every position.
    fn shifted(t: &ReferenceText, mad: f64) -> Vec<f32> {
        t.logprobs.iter().map(|&lp| lp - mad as f32).collect()
    }

    // ── validation ──

    #[test]
    fn a_well_formed_set_validates() {
        assert_eq!(set(24, 24).validate(), Ok(()));
    }

    #[test]
    fn validation_rejects_each_defect() {
        let mut s = set(24, 24);
        s.texts[3].id = "t0".into();
        assert_eq!(s.validate(), Err(ReferenceError::DuplicateId("t0".into())));

        let mut s = set(24, 24);
        s.texts[6].token_ids = s.texts[7].token_ids.clone();
        assert_eq!(s.validate(), Err(ReferenceError::DuplicateText("t7".into())));

        let mut s = set(24, 24);
        s.texts[1].logprobs.pop();
        assert_eq!(s.validate(), Err(ReferenceError::BadShape("t1".into())));

        let mut s = set(24, 24);
        s.texts[2] = text("t2", MIN_TEXT_TOKENS - 1, 0.02);
        assert_eq!(s.validate(), Err(ReferenceError::BadShape("t2".into())));

        let mut s = set(24, 24);
        s.texts[4].logprobs[0] = f32::NAN;
        assert_eq!(s.validate(), Err(ReferenceError::BadLogprob("t4".into())));
        s.texts[4].logprobs[0] = 0.5;
        assert_eq!(s.validate(), Err(ReferenceError::BadLogprob("t4".into())));

        let mut s = set(24, 24);
        s.texts[5].band = 0.0;
        assert_eq!(s.validate(), Err(ReferenceError::BadBand("t5".into())));
        s.texts[5].band = f64::INFINITY;
        assert_eq!(s.validate(), Err(ReferenceError::BadBand("t5".into())));

        let mut s = set(24, 24);
        s.cutoff = f64::NAN;
        assert_eq!(s.validate(), Err(ReferenceError::BadParams));
        let mut s = set(24, 24);
        s.min_texts = 0;
        assert_eq!(s.validate(), Err(ReferenceError::BadParams));

        assert_eq!(set(10, 24).validate(), Err(ReferenceError::TooFewTexts { have: 10, need: 24 }));
    }

    // ── per-text scoring ──

    #[test]
    fn mad_is_the_mean_absolute_difference() {
        assert_eq!(text_mad(&[-1.0, -2.0], &[-1.0, -2.0]), Ok(0.0));
        let m = text_mad(&[-1.0, -2.0, -3.0, -4.0], &[-1.5, -1.5, -3.0, -5.0]).unwrap();
        assert!((m - 0.5).abs() < 1e-9);
    }

    #[test]
    fn a_log_prob_a_hair_above_zero_is_accepted() {
        // log_softmax of the dominant logit can round to a tiny positive value.
        assert!(text_mad(&[-0.0001, -1.0], &[1e-6, -1.0]).is_ok());
    }

    #[test]
    fn malformed_replies_are_refused_not_clamped() {
        assert_eq!(text_mad(&[-1.0, -2.0], &[-1.0]), Err(ScoreError::WrongLength { expected: 2, got: 1 }));
        assert_eq!(text_mad(&[], &[]), Err(ScoreError::WrongLength { expected: 0, got: 0 }));
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.5] {
            assert_eq!(text_mad(&[-1.0, -2.0], &[-1.0, bad]), Err(ScoreError::InvalidLogprob));
        }
    }

    #[test]
    fn ratio_divides_by_the_band() {
        let t = text("a", 32, 0.02);
        let r = text_ratio(&t, &shifted(&t, 0.03)).unwrap();
        assert!((r - 1.5).abs() < 1e-3, "{r}");
    }

    // ── aggregate rule ──

    #[test]
    fn aggregate_is_the_mean_ratio_against_the_cutoff() {
        assert_eq!(aggregate_verdict(&[], DEFAULT_CUTOFF), None);
        assert_eq!(aggregate_verdict(&[1.35], DEFAULT_CUTOFF), Some(LogprobVerdict::Pass { aggregate: 1.35 }));
        let v = aggregate_verdict(&[1.0, 1.8], DEFAULT_CUTOFF).unwrap();
        assert_eq!(v, LogprobVerdict::Fail { aggregate: 1.4 });
        assert_eq!(v.outcome(), VerificationOutcome::Failed);
        assert_eq!(v.audit_class(), AuditClass::Fail);
        assert!(!v.passed());
        let p = aggregate_verdict(&[0.2, 1.1], DEFAULT_CUTOFF).unwrap();
        assert!(p.passed());
        assert_eq!(p.outcome(), VerificationOutcome::Honored);
        assert_eq!(p.audit_class(), AuditClass::Pass);
    }

    #[test]
    fn one_noisy_text_does_not_fail_an_honest_round() {
        // The reason for the aggregate rule: batching can put one honest text at ~2x its band.
        let mut ratios = vec![0.9; 23];
        ratios.push(2.0);
        assert!(aggregate_verdict(&ratios, DEFAULT_CUTOFF).unwrap().passed());
    }

    #[test]
    fn a_close_substitute_fails_even_when_some_texts_look_honest() {
        // Q8_0-like: most texts a little above the band, a few inside it; mean ≈ 1.45.
        let mut ratios = vec![1.6; 18];
        ratios.extend([1.0; 6]);
        let v = aggregate_verdict(&ratios, DEFAULT_CUTOFF).unwrap();
        assert!(!v.passed(), "{v:?}");
    }

    // ── rounds ──

    #[test]
    fn a_round_closes_after_min_texts_and_resets() {
        let s = set(8, 4);
        let mut r = ChallengeRound::new();
        for i in 0..4 {
            let t = r.next_text(&s, |_| 0).unwrap().clone();
            let got = r.record(&s, &t.id, &shifted(&t, 0.01)).unwrap();
            if i < 3 {
                assert_eq!(got, None);
                assert_eq!(r.progress(), i + 1);
            } else {
                assert!(got.unwrap().passed());
                assert_eq!(r.progress(), 0);
            }
        }
        // Second round, a substitute.
        for i in 0..4 {
            let t = r.next_text(&s, |n| n - 1).unwrap().clone();
            let got = r.record(&s, &t.id, &shifted(&t, 0.05)).unwrap();
            if i == 3 {
                let v = got.unwrap();
                assert!(!v.passed());
                assert!((v.aggregate() - 2.5).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn a_text_is_never_issued_twice_to_the_same_provider() {
        let s = set(5, 2);
        let mut r = ChallengeRound::new();
        let mut seen = BTreeSet::new();
        while let Some(t) = r.next_text(&s, |n| n / 2) {
            assert!(seen.insert(t.id.clone()), "reissued {}", t.id);
        }
        assert_eq!(seen.len(), 5);
        assert!(r.next_text(&s, |_| 0).is_none());
    }

    #[test]
    fn replies_must_match_an_issued_pending_text() {
        let s = set(4, 2);
        let mut r = ChallengeRound::new();
        let t = s.texts[0].clone();
        // Never issued.
        assert_eq!(r.record(&s, &t.id, &t.logprobs), Err(RoundError::NotIssued(t.id.clone())));
        assert_eq!(r.record(&s, "nope", &[]), Err(RoundError::UnknownText("nope".into())));
        let t = r.next_text(&s, |_| 0).unwrap().clone();
        assert_eq!(r.record(&s, &t.id, &t.logprobs), Ok(None));
        // A second reply to the same text is refused.
        assert_eq!(r.record(&s, &t.id, &t.logprobs), Err(RoundError::NotIssued(t.id.clone())));
    }

    #[test]
    fn a_malformed_reply_consumes_the_text_but_adds_no_ratio() {
        let s = set(4, 2);
        let mut r = ChallengeRound::new();
        let t = r.next_text(&s, |_| 0).unwrap().clone();
        assert_eq!(r.in_flight(), 1);
        assert_eq!(r.record(&s, &t.id, &[-1.0]), Err(RoundError::Score(ScoreError::WrongLength { expected: 31, got: 1 })));
        assert_eq!(r.in_flight(), 0);
        assert_eq!(r.progress(), 0);
    }

    #[test]
    fn abandoned_texts_stay_burned() {
        let s = set(3, 2);
        let mut r = ChallengeRound::new();
        let t = r.next_text(&s, |_| 0).unwrap().id.clone();
        r.abandon(&t);
        assert_eq!(r.in_flight(), 0);
        assert!(r.can_complete_round(&s)); // 2 fresh left
        let _ = r.next_text(&s, |_| 0).unwrap().id.clone();
        r.abandon(&s.texts[1].id);
        // 1 fresh left, nothing in progress: a full round (2) can no longer be drawn.
        assert!(!r.can_complete_round(&s));
    }

    #[test]
    fn a_round_survives_a_round_trip_through_bytes() {
        let s = set(8, 4);
        let mut r = ChallengeRound::new();
        let a = r.next_text(&s, |_| 0).unwrap().clone();
        r.record(&s, &a.id, &shifted(&a, 0.01)).unwrap();
        let b = r.next_text(&s, |_| 0).unwrap().clone(); // still pending when saved
        let restored = ChallengeRound::from_bytes(&r.to_bytes()).unwrap();
        assert_eq!(restored.issued_count(), 2);
        assert_eq!(restored.progress(), 1);
        assert_eq!(restored.in_flight(), 0, "pending texts come back as abandoned");
        // Neither text is issued again after the restore.
        let mut restored = restored;
        while let Some(t) = restored.next_text(&s, |_| 0) {
            assert!(t.id != a.id && t.id != b.id);
        }
        assert_eq!(restored.issued_count(), 8);
    }

    #[test]
    fn corrupt_round_bytes_are_rejected() {
        let mut r = ChallengeRound::new();
        let s = set(4, 4);
        let t = r.next_text(&s, |_| 0).unwrap().clone();
        r.record(&s, &t.id, &t.logprobs).unwrap();
        let good = r.to_bytes();
        for cut in 0..good.len() {
            assert!(ChallengeRound::from_bytes(&good[..cut]).is_err(), "cut {cut}");
        }
        let mut long = good.clone();
        long.push(0);
        assert!(ChallengeRound::from_bytes(&long).is_err());
        let mut bad = good.clone();
        bad[0] = 9;
        assert!(ChallengeRound::from_bytes(&bad).is_err());
        let mut nan = good;
        let at = nan.len() - 8;
        nan[at..].copy_from_slice(&f64::NAN.to_le_bytes());
        assert!(ChallengeRound::from_bytes(&nan).is_err());
    }

    #[test]
    fn a_renamed_text_is_still_recognised_as_issued() {
        let s = set(4, 1);
        let mut r = ChallengeRound::new();
        let t = r.next_text(&s, |_| 0).unwrap().clone();
        let mut renamed = s.clone();
        renamed.texts.iter_mut().find(|x| x.id == t.id).unwrap().id = "fresh-looking".into();
        let mut seen = vec![];
        while let Some(x) = r.next_text(&renamed, |_| 0) {
            seen.push(x.id.clone());
        }
        assert!(!seen.contains(&"fresh-looking".to_string()));
        assert_eq!(seen.len(), 3);
    }

    #[test]
    fn identities_normalise_but_keep_gguf_quants_precise() {
        let a = ModelIdentity::new(" Qwen3.5 ", "9B", "BF16").unwrap();
        assert_eq!(a, ModelIdentity::new("qwen3.5", "9b", "bfloat16").unwrap());
        assert_eq!(a.key(), "qwen3.5/9b/bf16");
        assert_eq!(ModelIdentity::new("qwen3.5", "9b", "F16").unwrap().quant, "fp16");
        let k = ModelIdentity::new("qwen3.5", "9b", "Q4_K_M").unwrap();
        assert_eq!(k.quant, "q4_k_m");
        assert_ne!(k, ModelIdentity::new("qwen3.5", "9b", "Q4_0").unwrap());
        assert_eq!(ModelIdentity::new("qwen3.5", "unknown", "bf16"), None);
        assert_eq!(ModelIdentity::new("qwen3.5", "9b", ""), None);
    }

    #[test]
    fn resetting_progress_keeps_the_issued_texts() {
        let s = set(6, 3);
        let mut r = ChallengeRound::new();
        let t = r.next_text(&s, |_| 0).unwrap().clone();
        r.record(&s, &t.id, &t.logprobs).unwrap();
        assert_eq!(r.progress(), 1);
        r.reset_progress();
        assert_eq!((r.progress(), r.issued_count()), (0, 1));
    }

    #[test]
    fn weights_digests_have_one_canonical_form() {
        let hex = "C5396E06AF294BD101B30DCE59131A76D2B773E76950ACC870EDA801D3AB0515";
        let canon = format!("sha256:{}", hex.to_lowercase());
        for spelling in [format!("sha256:{hex}"), format!("sha256-{}", hex.to_lowercase()), hex.to_string(), format!(" {canon} ")] {
            assert_eq!(normalize_weights_digest(&spelling).as_deref(), Some(canon.as_str()), "{spelling}");
        }
        for bad in ["", "sha256:", "sha256:abc", "md5:c5396e06af294bd101b30dce59131a76d2b773e76950acc870eda801d3ab0515", &"g".repeat(64)] {
            assert_eq!(normalize_weights_digest(bad), None, "{bad}");
        }
        let mut s = set(3, 2);
        s.weights = Some(canon);
        assert_eq!(s.validate(), Ok(()));
        s.weights = Some(format!("sha256-{}", hex.to_lowercase()));
        assert_eq!(s.validate(), Err(ReferenceError::BadParams), "stored digests must be canonical");
        // A file-tied reference prices a pass by its identity, so it must have one.
        s.weights = normalize_weights_digest(hex);
        s.identity = None;
        assert_eq!(s.validate(), Err(ReferenceError::WeightsWithoutIdentity));
        s.weights = None;
        assert_eq!(s.validate(), Ok(()), "a name-only reference needs no identity");
    }

    #[test]
    fn pick_out_of_range_is_clamped() {
        let s = set(3, 2);
        let mut r = ChallengeRound::new();
        assert!(r.next_text(&s, |_| 99).is_some());
    }

    // ── build_reference ──

    /// `n` texts of 32 tokens with a reference, and runs offset from it by `mad` nats (with a little
    /// per-text variation so bands differ).
    fn build_fixture(n: usize) -> (Vec<Vec<u32>>, RunScores) {
        let ids: Vec<Vec<u32>> = (0..n).map(|t| (0..32).map(|k| (t * 100 + k) as u32).collect()).collect();
        let lps = ids.iter().map(|x| x[1..].iter().map(|&v| -((v % 9) as f32) * 0.3 - 0.05).collect()).collect();
        (ids, RunScores { label: "ref".into(), group: None, logprobs: lps })
    }

    fn offset_run(reference: &RunScores, label: &str, mad: f64, jitter: f64) -> RunScores {
        RunScores {
            label: label.into(),
            group: None,
            logprobs: reference
                .logprobs
                .iter()
                .enumerate()
                .map(|(t, x)| {
                    let d = mad * (1.0 + jitter * ((t % 5) as f64 - 2.0) / 2.0);
                    x.iter().map(|&v| v - d as f32).collect()
                })
                .collect(),
        }
    }

    #[test]
    fn a_reference_is_built_and_calibrated_between_honest_and_substitute_runs() {
        let (ids, reference) = build_fixture(60);
        let honest = vec![offset_run(&reference, "cpu", 0.020, 0.3), offset_run(&reference, "other-engine", 0.015, 0.3)];
        let subs = vec![offset_run(&reference, "q4_0", 0.080, 0.3)];
        let mut p = BuildParams::new("qwen3.5-9b@q4_k_m", ModelIdentity::new("qwen3.5", "9b", "q4_k_m"));
        p.rounds = 300;
        let (set, cal) = build_reference(&ids, &reference, &honest, &subs, &p).unwrap();
        assert_eq!(set.texts.len(), 60);
        assert_eq!(set.identity, p.identity);
        assert!(cal.calibrated);
        assert!(cal.honest_high.max(1.0) < cal.cutoff && cal.cutoff < cal.substitute_low.unwrap(), "{cal:?}");
        // The cut-off works on the set it was built for: a fresh honest-like run passes a round, the
        // substitute fails one.
        let mut round = ChallengeRound::new();
        let honestish = offset_run(&reference, "x", 0.017, 0.3);
        let verdict = (0..24)
            .map(|_| {
                let t = round.next_text(&set, |_| 0).unwrap().clone();
                let i: usize = t.id.rsplit('#').next().unwrap().parse().unwrap();
                round.record(&set, &t.id, &honestish.logprobs[i]).unwrap()
            })
            .last()
            .unwrap()
            .unwrap();
        assert!(verdict.passed(), "{verdict:?}");
        // Deterministic.
        let (_, again) = build_reference(&ids, &reference, &honest, &subs, &p).unwrap();
        assert_eq!(again, cal);
    }

    #[test]
    fn runs_of_one_group_are_left_out_together_so_they_cannot_vouch_for_each_other() {
        // Two near-identical CPU builds and two GPU engines. Left out one run at a time, each CPU run
        // is covered by its twin and the honest tolerance collapses to ~1x; left out as a group, it is
        // scored against the GPU-only band — the honest stand-in for a kind of setup never measured.
        let (ids, reference) = build_fixture(60);
        let cpu = |label: &str, mad| RunScores { group: Some("cpu".into()), ..offset_run(&reference, label, mad, 0.3) };
        let gpu = vec![offset_run(&reference, "gpu-a", 0.010, 0.3), offset_run(&reference, "gpu-b", 0.012, 0.3)];
        let grouped: Vec<RunScores> = [cpu("cpu-native", 0.030), cpu("cpu-fma", 0.031)].into_iter().chain(gpu.clone()).collect();
        let ungrouped: Vec<RunScores> = grouped.iter().map(|r| RunScores { group: None, ..r.clone() }).collect();
        let subs = vec![offset_run(&reference, "q4_0", 0.200, 0.3)];
        let mut p = BuildParams::new("x", None);
        p.rounds = 300;
        let (set_u, cal_u) = build_reference(&ids, &reference, &ungrouped, &subs, &p).unwrap();
        let (set_g, cal_g) = build_reference(&ids, &reference, &grouped, &subs, &p).unwrap();
        // The stored bands don't change; only the calibration does.
        assert_eq!(set_u.texts, set_g.texts);
        assert!(cal_u.honest[0].1 < 1.0, "{cal_u:?}");
        assert!(cal_g.honest[0].1 > 2.0 && cal_g.honest[1].1 > 2.0, "{cal_g:?}");
        assert_eq!(cal_u.honest[2..], cal_g.honest[2..], "ungrouped runs score as before");
        assert!(cal_g.honest_high > 2.0 && cal_u.honest_high < 1.1, "{cal_u:?} {cal_g:?}");
        assert!(cal_g.cutoff > cal_u.cutoff * 1.3, "{} vs {}", cal_g.cutoff, cal_u.cutoff);
    }

    #[test]
    fn overlapping_substitutes_are_refused() {
        let (ids, reference) = build_fixture(30);
        let honest = vec![offset_run(&reference, "a", 0.020, 0.2), offset_run(&reference, "b", 0.020, 0.2)];
        let close = vec![offset_run(&reference, "q8_0", 0.021, 0.2)];
        let p = BuildParams::new("x", None);
        assert!(matches!(build_reference(&ids, &reference, &honest, &close, &p), Err(BuildError::NoSeparation { .. })));
    }

    #[test]
    fn without_substitutes_the_cutoff_is_marked_uncalibrated() {
        let (ids, reference) = build_fixture(24);
        let honest = vec![offset_run(&reference, "a", 0.02, 0.2), offset_run(&reference, "b", 0.01, 0.2)];
        let (_, cal) = build_reference(&ids, &reference, &honest, &[], &BuildParams::new("x", None)).unwrap();
        assert!(!cal.calibrated);
        assert_eq!(cal.substitute_low, None);
        assert!((cal.cutoff - cal.honest_high.max(1.0) * 1.15).abs() < 1e-12);
    }

    #[test]
    fn a_bit_identical_honest_run_cannot_zero_a_band() {
        let (ids, reference) = build_fixture(24);
        let same = RunScores { label: "same-engine".into(), group: None, logprobs: reference.logprobs.clone() };
        let honest = vec![same.clone(), same];
        let (set, cal) = build_reference(&ids, &reference, &honest, &[], &BuildParams::new("x", None)).unwrap();
        assert!(set.texts.iter().all(|t| t.band >= 1e-3));
        assert_eq!(cal.honest_high, 0.0);
        assert!((cal.cutoff - 1.15).abs() < 1e-12, "tolerance of at least one band");
    }

    #[test]
    fn malformed_inputs_are_rejected() {
        let (ids, reference) = build_fixture(24);
        let good = offset_run(&reference, "a", 0.02, 0.0);
        let p = BuildParams::new("x", None);
        assert_eq!(build_reference(&ids[..10], &reference, &[good.clone(), good.clone()], &[], &p).unwrap_err(), BuildError::TooFewTexts { have: 10, need: 24 });
        assert_eq!(build_reference(&ids, &reference, std::slice::from_ref(&good), &[], &p).unwrap_err(), BuildError::TooFewHonestGroups(1));
        // Two runs of one group are still one group: nothing left to calibrate them against.
        let grouped = |label: &str| RunScores { label: label.into(), group: Some("cpu".into()), ..good.clone() };
        assert_eq!(
            build_reference(&ids, &reference, &[grouped("cpu-a"), grouped("cpu-b")], &[], &p).unwrap_err(),
            BuildError::TooFewHonestGroups(1)
        );
        let mut short = good.clone();
        short.logprobs[3].pop();
        assert_eq!(
            build_reference(&ids, &reference, &[good.clone(), short], &[], &p).unwrap_err(),
            BuildError::BadRun { run: "a".into(), text: 3 }
        );
        let mut nan = good.clone();
        nan.label = "nan".into();
        nan.logprobs[5][0] = f32::NAN;
        assert_eq!(
            build_reference(&ids, &reference, &[good, nan], &[], &p).unwrap_err(),
            BuildError::BadRun { run: "nan".into(), text: 5 }
        );
    }

    // ── early stopping ──

    #[test]
    fn checkpoints_must_be_ordered_inside_the_round_and_no_looser_than_the_cutoff() {
        let cp = |after, pass_below, fail_above| Checkpoint { after, pass_below, fail_above };
        let mut s = set(24, 24);
        s.early_stop = vec![cp(8, Some(1.2), 1.5), cp(12, None, 1.4)];
        assert_eq!(s.validate(), Ok(()));
        for bad in [
            vec![cp(12, None, 2.0), cp(8, None, 2.0)], // out of order
            vec![cp(24, None, 2.0)],                   // not before the end of the round
            vec![cp(8, Some(1.5), 2.0)],               // pass_below above the cut-off (looser)
            vec![cp(8, None, 1.2)],                    // fail_above below the cut-off (harsher)
            vec![cp(0, None, 2.0)],
        ] {
            s.early_stop = bad;
            assert_eq!(s.validate(), Err(ReferenceError::BadParams));
        }
        // A first-stage pack may hold fewer texts than a full round.
        let mut small = set(10, 24);
        small.early_stop = vec![cp(8, Some(1.2), 1.5)];
        assert!(small.validate().is_err());
        assert_eq!(small.validate_texts(), Ok(()));
    }

    #[test]
    fn a_round_stops_early_only_when_the_mean_is_clearly_on_one_side() {
        let mut s = set(40, 24);
        s.early_stop = vec![Checkpoint { after: 8, pass_below: Some(1.0), fail_above: 2.0 }];
        let play = |mad: f64| {
            let mut r = ChallengeRound::new();
            for i in 0..24 {
                let t = r.next_text(&s, |_| 0).unwrap().clone();
                if let Some(v) = r.record(&s, &t.id, &shifted(&t, mad)).unwrap() {
                    return (i + 1, v.passed());
                }
            }
            unreachable!()
        };
        assert_eq!(play(0.01), (8, true), "0.5x: honest-looking, passes at the checkpoint");
        assert_eq!(play(0.08), (8, false), "4x: fails at the checkpoint");
        assert_eq!(play(0.024), (24, true), "1.2x: undecided at 8, full round passes (cut-off 1.35)");
        assert_eq!(play(0.03), (24, false), "1.5x: undecided at 8, full round fails");
    }

    #[test]
    fn the_builder_calibrates_checkpoints_that_stop_clear_cases_early() {
        let (ids, reference) = build_fixture(80);
        let honest = vec![offset_run(&reference, "cpu", 0.020, 0.3), offset_run(&reference, "other", 0.015, 0.3)];
        let subs = vec![offset_run(&reference, "q4_0", 0.080, 0.3)];
        let mut p = BuildParams::new("x", None);
        p.rounds = 400;
        let (set, cal) = build_reference(&ids, &reference, &honest, &subs, &p).unwrap();
        assert_eq!(set.early_stop.iter().map(|c| c.after).collect::<Vec<_>>(), vec![8, 12, 16]);
        assert_eq!(cal.early_stop, set.early_stop);
        for cp in &set.early_stop {
            assert!(cp.fail_above >= set.cutoff && cp.pass_below.unwrap() <= set.cutoff, "{cp:?}");
        }
        // Without substitutes there is no early pass.
        let (plain, _) = build_reference(&ids, &reference, &honest, &[], &p).unwrap();
        assert!(plain.early_stop.iter().all(|c| c.pass_below.is_none()));
        // A clear substitute is decided at the first checkpoint.
        let mut r = ChallengeRound::new();
        let mut decided = None;
        for i in 0..24 {
            let t = r.next_text(&set, |_| 0).unwrap().clone();
            let k: usize = t.id.rsplit('#').next().unwrap().parse().unwrap();
            if let Some(v) = r.record(&set, &t.id, &subs[0].logprobs[k]).unwrap() {
                decided = Some((i + 1, v.passed()));
                break;
            }
        }
        assert_eq!(decided, Some((8, false)));
    }
}
