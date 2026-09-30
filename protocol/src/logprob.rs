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

/// Every reference text for one claimed model, plus the decision parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceSet {
    /// The canonical `model_id@quant` the reference was made from (the handle the provider claims).
    pub model_key: String,
    pub texts: Vec<ReferenceText>,
    /// Aggregate cut-off; see [`DEFAULT_CUTOFF`].
    pub cutoff: f64,
    /// Texts per round; see [`DEFAULT_MIN_TEXTS`].
    pub min_texts: usize,
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
    /// The cut-off is non-finite or not positive, or `min_texts` is 0.
    BadParams,
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
        if !(self.cutoff.is_finite() && self.cutoff > 0.0) || self.min_texts == 0 {
            return Err(ReferenceError::BadParams);
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
        if self.texts.len() < self.min_texts {
            return Err(ReferenceError::TooFewTexts { have: self.texts.len(), need: self.min_texts });
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
        let fresh: Vec<&ReferenceText> = set.texts.iter().filter(|t| !self.issued.contains(&t.digest())).collect();
        if fresh.is_empty() {
            return None;
        }
        let text = fresh[pick(fresh.len()).min(fresh.len() - 1)];
        self.issued.insert(text.digest());
        self.pending.insert(text.id.clone());
        Some(text)
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
        if self.ratios.len() < set.min_texts {
            return Ok(None);
        }
        let verdict = aggregate_verdict(&self.ratios, set.cutoff);
        self.ratios.clear();
        Ok(verdict)
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
            texts: (0..n_texts)
                .map(|i| {
                    let mut t = text(&format!("t{i}"), 32, 0.02);
                    t.token_ids.iter_mut().for_each(|id| *id += 1000 * i as u32);
                    t
                })
                .collect(),
            cutoff: DEFAULT_CUTOFF,
            min_texts,
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
    fn pick_out_of_range_is_clamped() {
        let s = set(3, 2);
        let mut r = ChallengeRound::new();
        assert!(r.next_text(&s, |_| 99).is_some());
    }
}
