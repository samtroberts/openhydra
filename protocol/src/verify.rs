// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Verification policy (protocol.md §7) — M2.2 scaffold.
//!
//! The trust layer rests on three mechanisms; this module holds the *pure* policy
//! logic & math for them, with no DHT/router/network wiring (that lands later, on
//! top, exactly like the M1.3 router scaffold):
//!
//! 1. **Proof-of-inference (sampling).** A TOPLOC-style hash of the model's
//!    activations lets a verifier confirm a result is consistent with the claimed
//!    model, checked on a *sampled* fraction of requests. Ported byte-identically from
//!    `verification/toploc.py` — see [`activation_hash`] / [`verify_activation_hash`].
//! 2. **Redundant execution.** A sampled fraction of requests are run on ≥2
//!    providers and compared. The comparison logic is deferred — see [`agrees`].
//! 3. **Reputation.** Providers accrue a reputation score from verification
//!    outcomes; repeat failures downrank them out of routing. This is the piece
//!    built now: [`ReputationTracker`].
//!
//! Sample rates are tuned by reputation — trusted, long-lived providers are checked
//! rarely; new or suspect ones, often (see [`sample_rate_for_reputation`]). That
//! balances verification cost against coverage.
//!
//! **Clock injection.** Like the M1.3 scaffold injected its I/O, every time-dependent
//! method takes an explicit `now_ms` (Unix milliseconds) rather than reading the wall
//! clock, so decay is deterministic and unit-testable. Live callers pass
//! `SystemTime::now()`; tests pass a synthetic clock.

/// The neutral reputation baseline. A fresh or fully-decayed provider sits here — the
/// same 50.0 the router (`router::PeerScoreInput`) already uses for an unknown peer, so
/// a decayed score converges to exactly the router's "unknown" default.
pub const NEUTRAL_REPUTATION: f64 = 50.0;

/// Reputation is clamped to `[0, 100]` to match the router's `reputation` scale.
pub const MIN_REPUTATION: f64 = 0.0;
pub const MAX_REPUTATION: f64 = 100.0;

/// Additive increase per honored receipt / passed verification (AIMD: slow to earn).
const REWARD_HONORED: f64 = 3.0;
/// Multiplicative survivor fraction when a provider rejects a *valid* receipt
/// (AIMD: fast to lose). 0.6 ⇒ a 40% cut.
const PENALTY_REJECTED: f64 = 0.6;
/// Multiplicative survivor fraction when a provider's output fails verification —
/// a wrong answer is worse than refusing to sign, so it stings harder than a rejection.
const PENALTY_FAILED: f64 = 0.4;

/// Default reputation half-life: the time over which a score drifts halfway back to
/// [`NEUTRAL_REPUTATION`] in the absence of new outcomes. Chosen shorter than the
/// credit-ledger half-life (§6, ≈ weeks) so trust tracks *recent* behaviour — here ~7
/// days. Tunable per-tracker via [`ReputationTracker::with_half_life`].
pub const DEFAULT_HALF_LIFE_MS: u64 = 7 * 24 * 60 * 60 * 1000;

/// An outcome of a verification / receipt interaction with a provider, fed into its
/// [`ReputationTracker`]. The mapping to a score change is AIMD (see the constants).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationOutcome {
    /// The provider served and co-signed a valid receipt, or passed a sampled
    /// proof-of-inference / redundant-execution check. Reputation rises (additive).
    Honored,
    /// The provider rejected or refused to co-sign a *valid* receipt. Reputation
    /// falls multiplicatively.
    Rejected,
    /// The provider's output failed verification — a proof-of-inference hash mismatch
    /// or a redundant-execution disagreement. Reputation falls hardest.
    Failed,
}

impl VerificationOutcome {
    /// Apply this outcome to a (already decayed-to-now) score, returning the clamped
    /// result. Additive for the positive case, multiplicative for the negative ones.
    fn apply(self, score: f64) -> f64 {
        let next = match self {
            VerificationOutcome::Honored => score + REWARD_HONORED,
            VerificationOutcome::Rejected => score * PENALTY_REJECTED,
            VerificationOutcome::Failed => score * PENALTY_FAILED,
        };
        next.clamp(MIN_REPUTATION, MAX_REPUTATION)
    }
}

/// Tracks one provider's reputation as a `[0, 100]` score that (a) moves on
/// verification outcomes (AIMD) and (b) decays toward [`NEUTRAL_REPUTATION`] with a
/// configurable half-life, so a node cannot coast on old good behaviour — nor is it
/// damned forever by old bad behaviour.
///
/// Pure and self-contained: no I/O, no clock reads. The router will later query
/// [`score_at`](Self::score_at) to feed `PeerScoreInput.reputation`; that wiring is M2.2
/// follow-up, not part of this scaffold.
#[derive(Debug, Clone)]
pub struct ReputationTracker {
    /// The score as of `last_update_ms`, before any further decay.
    score: f64,
    /// Unix-ms timestamp the `score` field is current as of.
    last_update_ms: u64,
    /// Half-life (ms) of the exponential decay toward [`NEUTRAL_REPUTATION`].
    half_life_ms: u64,
}

impl ReputationTracker {
    /// A fresh tracker at the neutral baseline, as of `now_ms`, with the default
    /// half-life.
    pub fn new(now_ms: u64) -> Self {
        Self::with_half_life(now_ms, DEFAULT_HALF_LIFE_MS)
    }

    /// As [`new`](Self::new) but with an explicit decay half-life. A `half_life_ms` of 0
    /// is treated as "no decay" (the score holds until the next outcome) to avoid a
    /// divide-by-zero.
    pub fn with_half_life(now_ms: u64, half_life_ms: u64) -> Self {
        Self {
            score: NEUTRAL_REPUTATION,
            last_update_ms: now_ms,
            half_life_ms,
        }
    }

    /// The decayed score *as of* `now_ms`, without mutating the tracker. This is the
    /// read the router uses at ranking time. Decay pulls the stored score toward
    /// [`NEUTRAL_REPUTATION`] by `0.5^(elapsed / half_life)`.
    ///
    /// A `now_ms` earlier than the last update (clock skew) is treated as no elapsed
    /// time — decay never runs backward.
    pub fn score_at(&self, now_ms: u64) -> f64 {
        if self.half_life_ms == 0 {
            return self.score;
        }
        let elapsed = now_ms.saturating_sub(self.last_update_ms) as f64;
        if elapsed <= 0.0 {
            return self.score;
        }
        let factor = 0.5_f64.powf(elapsed / self.half_life_ms as f64);
        NEUTRAL_REPUTATION + (self.score - NEUTRAL_REPUTATION) * factor
    }

    /// Record a verification outcome at `now_ms`: first decay the stored score forward
    /// to now, then apply the outcome. Returns the new score.
    pub fn record(&mut self, outcome: VerificationOutcome, now_ms: u64) -> f64 {
        // Roll the stored score forward to now so the AIMD step acts on the *current*
        // (decayed) value, not a stale one.
        self.score = self.score_at(now_ms);
        self.last_update_ms = self.last_update_ms.max(now_ms);
        self.score = outcome.apply(self.score);
        self.score
    }

    /// The raw stored score (as of `last_update_ms`, undecayed). Prefer
    /// [`score_at`](Self::score_at) for any time-aware decision.
    pub fn raw_score(&self) -> f64 {
        self.score
    }

    /// The timestamp the raw score is current as of.
    pub fn last_update_ms(&self) -> u64 {
        self.last_update_ms
    }

    /// Serialize the snapshot for the persistent store (M2.3): `score:f64[8] ‖
    /// last_update_ms:u64[8] ‖ half_life_ms:u64[8]`, all little-endian = 24 bytes. The
    /// f64 is stored by its exact bit pattern so a rehydrated tracker decays identically.
    pub fn to_bytes(&self) -> [u8; 24] {
        let mut b = [0u8; 24];
        b[0..8].copy_from_slice(&self.score.to_le_bytes());
        b[8..16].copy_from_slice(&self.last_update_ms.to_le_bytes());
        b[16..24].copy_from_slice(&self.half_life_ms.to_le_bytes());
        b
    }

    /// Reconstruct a tracker from [`to_bytes`](Self::to_bytes). Returns `Err` if the
    /// blob is not exactly 24 bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() != 24 {
            return Err(format!("reputation snapshot must be 24 bytes, got {}", data.len()));
        }
        let mut f = [0u8; 8];
        f.copy_from_slice(&data[0..8]);
        let mut a = [0u8; 8];
        a.copy_from_slice(&data[8..16]);
        let mut h = [0u8; 8];
        h.copy_from_slice(&data[16..24]);
        Ok(Self {
            score: f64::from_le_bytes(f),
            last_update_ms: u64::from_le_bytes(a),
            half_life_ms: u64::from_le_bytes(h),
        })
    }
}

/// Reputation → proof-of-inference sample rate (protocol.md §7): trusted providers are
/// checked rarely, suspect ones often. Linearly interpolates between `max_rate` (at the
/// minimum reputation) and `min_rate` (at the maximum), clamped to `[min_rate,
/// max_rate]`. Pure policy math — the actual sampling decision (RNG draw) lives with the
/// caller so this stays deterministic and testable.
///
/// `base_rate` semantics: callers pass the rate band they want; e.g. `min_rate = 0.01`
/// (1% for fully-trusted) and `max_rate = 0.5` (50% for a brand-new/suspect provider).
pub fn sample_rate_for_reputation(reputation: f64, min_rate: f64, max_rate: f64) -> f64 {
    let rep = reputation.clamp(MIN_REPUTATION, MAX_REPUTATION);
    let trust = (rep - MIN_REPUTATION) / (MAX_REPUTATION - MIN_REPUTATION); // 0..=1
    let rate = max_rate + (min_rate - max_rate) * trust;
    rate.clamp(min_rate.min(max_rate), min_rate.max(max_rate))
}

// ── Verification primitives (TOPLOC proof-of-inference) ──

/// TOPLOC locality-sensitive activation hash (protocol.md §7) — a **byte-identical**
/// Rust port of `verification/toploc.py::activation_hash`, so a digest computed by a
/// Python peer verifies against one computed here, and vice versa, during the
/// Python→Rust transition. (TOPLOC applies to the *sharded* path, where activations
/// exist; the whole-model path returns text, not activations — see plan §7.)
///
/// Each value is quantized, then the packed bytes are SHA-256'd:
/// * a **token id** (`|v| > 1.5`) → its rounded value as a 4-byte little-endian `i32`;
/// * a **hidden state** (`|v| <= 1.5`) → an 8-bit bucket `round((v + 1.0) * 127.5)`
///   clamped to `[0, 255]`, one byte.
///
/// An empty activation hashes the literal `b"empty"`.
///
/// **Rounding is half-to-even** (`round_ties_even`) to match Python's `round()`. A naive
/// half-away-from-zero round would diverge on exact-half bucket boundaries and cause
/// false verification failures — exactly the kind of silent mismatch this hash exists to
/// catch, so the parity is pinned by golden vectors generated from the Python reference.
pub fn activation_hash(activation: &[f64]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    if activation.is_empty() {
        hasher.update(b"empty");
    } else {
        let mut packed: Vec<u8> = Vec::with_capacity(activation.len());
        for &v in activation {
            if v.abs() > 1.5 {
                // Token id → 4-byte LE i32 of the (banker's-)rounded value.
                let id = v.round_ties_even() as i32;
                packed.extend_from_slice(&id.to_le_bytes());
            } else {
                // Hidden state → one 8-bit bucket in [0, 255].
                let bucket = (((v + 1.0) * 127.5).round_ties_even()).clamp(0.0, 255.0) as u8;
                packed.push(bucket);
            }
        }
        hasher.update(&packed);
    }
    let digest = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

/// Verify an activation vector against an expected 32-byte TOPLOC digest. Mirrors
/// `verification/toploc.py::verify_hash`: an empty/wrong-length `expected` fails closed.
pub fn verify_activation_hash(activation: &[f64], expected: &[u8]) -> bool {
    expected.len() == 32 && activation_hash(activation) == expected
}

/// **Deferred (M2.2 follow-up).** Whether two providers' results for the same request
/// agree under the redundant-execution audit. Will compare activation hashes / sampled
/// logits within tolerance. Not yet implemented (the redundant-exec comparison still
/// lives in Python `verification/redundant.py`).
#[allow(dead_code)]
pub fn agrees(_a: &[u8], _b: &[u8]) -> bool {
    false // scaffold: real comparison logic lands when verification moves into Rust
}

#[cfg(test)]
mod tests {
    use super::*;

    const HOUR_MS: u64 = 60 * 60 * 1000;
    const DAY_MS: u64 = 24 * HOUR_MS;

    #[test]
    fn starts_neutral() {
        let t = ReputationTracker::new(1_000);
        assert_eq!(t.raw_score(), NEUTRAL_REPUTATION);
        assert_eq!(t.score_at(1_000), NEUTRAL_REPUTATION);
    }

    #[test]
    fn tracker_bytes_roundtrip_preserves_decay() {
        // A snapshot survives serialization with its exact state, so a rehydrated
        // tracker decays to the same value as the original.
        let mut t = ReputationTracker::with_half_life(1_000, DAY_MS);
        t.record(VerificationOutcome::Honored, 1_000);
        t.record(VerificationOutcome::Honored, 1_000);
        let blob = t.to_bytes();
        assert_eq!(blob.len(), 24);
        let back = ReputationTracker::from_bytes(&blob).unwrap();
        assert_eq!(back.raw_score().to_bits(), t.raw_score().to_bits()); // exact bits
        assert_eq!(back.last_update_ms(), t.last_update_ms());
        // Same decayed value a day later → the half-life state round-tripped too.
        assert_eq!(back.score_at(1_000 + DAY_MS), t.score_at(1_000 + DAY_MS));
    }

    #[test]
    fn tracker_from_bytes_rejects_wrong_length() {
        assert!(ReputationTracker::from_bytes(&[0u8; 23]).is_err());
        assert!(ReputationTracker::from_bytes(&[0u8; 25]).is_err());
    }

    #[test]
    fn honored_raises_reputation() {
        let mut t = ReputationTracker::new(0);
        let after = t.record(VerificationOutcome::Honored, 0);
        assert!(after > NEUTRAL_REPUTATION, "honored receipt should raise score");
        assert_eq!(after, NEUTRAL_REPUTATION + REWARD_HONORED);
    }

    #[test]
    fn rejecting_a_valid_receipt_drops_reputation() {
        // The headline M2.2 case: a provider that rejects a valid receipt loses trust.
        let mut t = ReputationTracker::new(0);
        let before = t.raw_score();
        let after = t.record(VerificationOutcome::Rejected, 0);
        assert!(after < before, "rejecting a valid receipt must drop reputation");
        assert_eq!(after, NEUTRAL_REPUTATION * PENALTY_REJECTED);
    }

    #[test]
    fn failed_verification_stings_harder_than_rejection() {
        let mut rejected = ReputationTracker::new(0);
        let mut failed = ReputationTracker::new(0);
        let r = rejected.record(VerificationOutcome::Rejected, 0);
        let f = failed.record(VerificationOutcome::Failed, 0);
        assert!(f < r, "a wrong answer should cost more than a refusal to sign");
    }

    #[test]
    fn repeat_failures_compound_downward() {
        let mut t = ReputationTracker::with_half_life(0, 0); // decay off: isolate AIMD
        let mut last = t.raw_score();
        for _ in 0..5 {
            let next = t.record(VerificationOutcome::Failed, 0);
            assert!(next < last, "each failure should push reputation lower");
            last = next;
        }
        assert!(last < 5.0, "five failures should drive a node near the floor");
    }

    #[test]
    fn score_is_clamped_to_range() {
        let mut t = ReputationTracker::with_half_life(0, 0);
        for _ in 0..100 {
            t.record(VerificationOutcome::Honored, 0);
        }
        assert!(t.raw_score() <= MAX_REPUTATION);
        assert_eq!(t.raw_score(), MAX_REPUTATION, "honored events saturate at 100");
        for _ in 0..100 {
            t.record(VerificationOutcome::Failed, 0);
        }
        assert!(t.raw_score() >= MIN_REPUTATION);
    }

    // ── decay ──

    #[test]
    fn high_score_decays_toward_neutral_over_time() {
        // Earn a high score, then let a half-life elapse with no further activity: the
        // node cannot coast — its score must fall halfway back to neutral.
        let mut t = ReputationTracker::with_half_life(0, DAY_MS);
        for _ in 0..20 {
            t.record(VerificationOutcome::Honored, 0); // saturate toward 100
        }
        let peak = t.raw_score();
        assert!(peak > 90.0);

        let after_one_half_life = t.score_at(DAY_MS);
        let expected = NEUTRAL_REPUTATION + (peak - NEUTRAL_REPUTATION) * 0.5;
        assert!(
            (after_one_half_life - expected).abs() < 1e-9,
            "one half-life should move the score halfway to neutral: got {after_one_half_life}, want {expected}"
        );

        // Far in the future it converges to neutral (can't coast indefinitely).
        let far = t.score_at(DAY_MS * 40);
        assert!((far - NEUTRAL_REPUTATION).abs() < 0.5, "score converges to neutral: {far}");
        assert!(far < peak);
    }

    #[test]
    fn low_score_recovers_toward_neutral_over_time() {
        // Decay is symmetric: a past offender is not damned forever — it drifts back up.
        let mut t = ReputationTracker::with_half_life(0, DAY_MS);
        let sunk = t.record(VerificationOutcome::Failed, 0);
        assert!(sunk < NEUTRAL_REPUTATION);
        let recovered = t.score_at(DAY_MS);
        assert!(recovered > sunk, "score should recover toward neutral with time");
        assert!(recovered < NEUTRAL_REPUTATION, "but not overshoot neutral");
        let expected = NEUTRAL_REPUTATION + (sunk - NEUTRAL_REPUTATION) * 0.5;
        assert!((recovered - expected).abs() < 1e-9);
    }

    #[test]
    fn record_acts_on_the_decayed_value_not_a_stale_one() {
        // A high score that has decayed for a half-life, then takes a hit, should be
        // penalised from the decayed value — not the stale peak.
        let mut t = ReputationTracker::with_half_life(0, DAY_MS);
        for _ in 0..20 {
            t.record(VerificationOutcome::Honored, 0);
        }
        let decayed = t.score_at(DAY_MS);
        let after_hit = t.record(VerificationOutcome::Rejected, DAY_MS);
        assert!((after_hit - decayed * PENALTY_REJECTED).abs() < 1e-9);
    }

    #[test]
    fn clock_skew_never_decays_backward() {
        let mut t = ReputationTracker::with_half_life(100, DAY_MS);
        t.record(VerificationOutcome::Honored, 100);
        let s = t.raw_score();
        // Query with an earlier timestamp than the last update: no negative-time growth.
        assert_eq!(t.score_at(50), s);
    }

    #[test]
    fn zero_half_life_means_no_decay() {
        let mut t = ReputationTracker::with_half_life(0, 0);
        t.record(VerificationOutcome::Honored, 0);
        let s = t.raw_score();
        assert_eq!(t.score_at(DAY_MS * 1000), s, "half_life 0 ⇒ score holds");
    }

    // ── sample-rate policy ──

    #[test]
    fn sample_rate_scales_inversely_with_reputation() {
        let (lo, hi) = (0.01, 0.5);
        let trusted = sample_rate_for_reputation(100.0, lo, hi);
        let neutral = sample_rate_for_reputation(50.0, lo, hi);
        let suspect = sample_rate_for_reputation(0.0, lo, hi);
        assert!((trusted - lo).abs() < 1e-9, "fully trusted ⇒ min rate");
        assert!((suspect - hi).abs() < 1e-9, "fully suspect ⇒ max rate");
        assert!(trusted < neutral && neutral < suspect, "monotonic: more trust ⇒ less sampling");
        assert!((neutral - (lo + hi) / 2.0).abs() < 1e-9, "neutral ⇒ midpoint rate");
    }

    #[test]
    fn sample_rate_clamps_out_of_range_reputation() {
        let (lo, hi) = (0.01, 0.5);
        assert!((sample_rate_for_reputation(999.0, lo, hi) - lo).abs() < 1e-9);
        assert!((sample_rate_for_reputation(-50.0, lo, hi) - hi).abs() < 1e-9);
    }

    // ── TOPLOC activation hash ──
    //
    // Golden digests generated from the Python reference (`verification/toploc.py`) — a
    // mismatch means the Rust port diverged from what live Python peers compute, which
    // would cause false verification failures. These pin the exact byte layout +
    // banker's rounding. Regenerate via:
    //   python3 -c "from verification.toploc import activation_hash as H; print(H(<vec>).hex())"

    fn hx(s: &str) -> [u8; 32] {
        let mut out = [0u8; 32];
        for i in 0..32 {
            out[i] = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap();
        }
        out
    }

    #[test]
    fn toploc_matches_python_golden_vectors() {
        let cases: &[(&[f64], &str)] = &[
            (&[], "2e1cfa82b035c26cbbbdae632cea070514eb8b773f616aaeaf668e2f0be8f10d"),
            (&[1.0, 2.0, 3.0, 4.0], "b53fee71a9a7f14107ee19b68ddbf198e102c0881286d1be327928ac97467c9e"),
            (&[0.5, -0.3, 1.0], "c95cbccbc375d32d5fabcde2e5015c6735295a2f933251ac8eceb20dffa0dda4"),
            (&[263.0, 2217.0, 7826.0, -1.5, 0.0], "a06c1b5c64fa2ac6d5a33c5b56243dea31b4612830d725d25956db72a6253cfb"),
            (&[-1.0, 0.0, 1.0], "5240672d7b51756b829ad0ef8d9468b7a078afa2f410484fd3892dab47becb72"),
            // 1.5/-1.5 are hidden states (|v| > 1.5 is the token threshold); 1.6/-1.6 are token ids.
            (&[1.5, -1.5, 1.6, -1.6], "93399f682d2dfb1d2f8741b3cfee2e44e9309ded0ebefbfce39f7130684d7dae"),
            // (0+1)*127.5 = 127.5 → banker's-rounds to 128 (a naive round would also give 128 here,
            // but the golden still anchors the tie behaviour against Python's round()).
            (&[0.0], "76be8b528d0075f7aae98d6fa57a6d3c83ae480a8469e668d7b0af968995ac71"),
            (&[-263.0, -1.6], "e5df279826e7f401dd7437e98b33aa5cafe061b213d22b6b8f96daed7e9b67ff"),
        ];
        for (vec, hex) in cases {
            assert_eq!(activation_hash(vec), hx(hex), "TOPLOC mismatch for {vec:?}");
        }
    }

    #[test]
    fn toploc_is_deterministic_and_tamper_sensitive() {
        let a = [263.0, 2217.0, 7826.0, -0.25];
        assert_eq!(activation_hash(&a), activation_hash(&a)); // deterministic
        let tampered = [263.0, 2217.0, 7826.0, 999.0]; // one value changed
        assert_ne!(activation_hash(&a), activation_hash(&tampered));
        // A token-id vs hidden-state reinterpretation of the same magnitude differs too.
        assert_ne!(activation_hash(&[2.0]), activation_hash(&[1.0]));
    }

    #[test]
    fn verify_activation_hash_fails_closed() {
        let a = [1.0, 2.0, 3.0];
        let good = activation_hash(&a);
        assert!(verify_activation_hash(&a, &good));
        assert!(!verify_activation_hash(&[1.0, 2.0, 4.0], &good)); // tampered
        assert!(!verify_activation_hash(&a, b"")); // empty digest fails closed
        assert!(!verify_activation_hash(&a, &good[..31])); // wrong length fails closed
    }
}
