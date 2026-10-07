// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Tier-award policy — the verification gate on cost-weighted credit
//! (CREDIT_TIERING_PLAN_v2, Phase 4 · Decision #13).
//!
//! Phase 1 ([`crate::weight`]) says what a model is *worth* to serve (its [`CostTier`]). But
//! paying that worth to a provider the moment it *claims* a big model is exactly the
//! "claim `qwen-30b`, serve `tinyllama`, bank 30×" fraud. So a `(peer, model)` earns at the tier
//! it has **proven**, not the tier it **claims**:
//!
//! ```text
//! awarded_tier = min(claimed_tier, floor_tier + ⌊n / K⌋)   — only while r ≥ R_min and a ≥ A_min
//! ```
//!
//! * `claimed_tier` — the model's true cost-tier (from the signed weight registry, never the
//!   provider's word). The award never exceeds it.
//! * `floor_tier` — the cold-start floor (nano): a brand-new `(peer, model)` earns at nano
//!   regardless of the claim.
//! * `n` — recent **passing `temperature=0` audits** for this `(peer, model)`; every `K` of them
//!   lifts the cap one tier.
//! * `r` — the `(peer, model)` reputation ([`crate::verify`], per-model since Phase 4a); `a` — the
//!   recency-weighted audit pass-rate. Ascension is **gated**: below `R_min` / `A_min` the cap
//!   holds at the floor no matter how many audits accrue.
//!
//! **Escrow** ([`escrow_split`]): the difference between the claimed worth and the awarded worth is
//! *held*, not paid — released as the cap climbs, clawed back on a clear fail — so withdrawable
//! credit only ever accrues at the low *awarded* tier. **Graduated failure** ([`GradedFault`]):
//! audits classify pass / fail / **inconclusive**; only a *clear* temp=0 divergence penalizes
//! (inconclusive = non-determinism / network / wrong-auditor → never punished, retry only). Net:
//! claim-big/serve-small has **negative expected value**.
//!
//! **Purity.** No I/O, no clock — reputation, pass-rate, and audit count in; a tier + escrow split
//! out. The live wiring (feeding audits/reputation in, holding escrow in the ledger) lands on top
//! in Phase 3/4. Not yet consumed by the ledger.
//!
//! The provider-standing *faster-ramp* (Decision #4 — shrinking `K` for a peer with strong
//! standing on other models) is deliberately **deferred** with the Phase 4a Option-A migration:
//! it needs the cross-model standing aggregate, which lands with this curve's live wiring. Until
//! then `K` is the constant [`AUDITS_PER_STEP`].

use crate::weight::CostTier;

/// Reputation floor to ascend above the cold-start tier (Decision #13, `R_min ≈ 60`). Below this,
/// the awarded cap holds at the floor no matter how many audits pass — a provider must be *trusted*
/// on this model, not merely audited, to climb.
pub const R_MIN: f64 = 60.0;

/// Recency-weighted audit pass-rate floor to ascend (`A_min ≈ 0.9`). A provider that fails audits
/// even occasionally does not climb — a high count of passes can't paper over recent divergence.
pub const A_MIN: f64 = 0.9;

/// Passing audits required to lift the awarded cap by one tier (`K ≈ 3`). Constant until the
/// provider-standing faster-ramp (Decision #4) lands with the live wiring.
pub const AUDITS_PER_STEP: u32 = 3;

/// The cold-start floor tier: a brand-new `(peer, model)` earns here regardless of the claim.
pub const FLOOR_TIER: CostTier = CostTier::Nano;

/// The highest cost-tier a `(peer, model)` may currently **earn at**, given what it has proven.
/// Never exceeds `claimed` (the model's true cost-tier) nor climbs above the floor unless the
/// reputation *and* audit-pass-rate gates are both met (Decision #13). Pure.
///
/// * `claimed` — the model's true [`CostTier`] (from the weight registry).
/// * `reputation` — this `(peer, model)`'s score in `[0, 100]`.
/// * `audit_pass_rate` — recency-weighted pass-rate in `[0, 1]`.
/// * `passing_audits` — count of recent passing temp=0 audits (`n`).
/// * `audits_per_step` — `K` (use [`AUDITS_PER_STEP`]); `0` is treated as `1` to avoid div-by-zero.
pub fn max_awardable_tier(
    claimed: CostTier,
    reputation: f64,
    audit_pass_rate: f64,
    passing_audits: u32,
    audits_per_step: u32,
) -> CostTier {
    let floor_idx = FLOOR_TIER.index() as u32;
    let claimed_idx = claimed.index() as u32;
    // Gated ascension: audits only buy tiers while the provider is trusted AND consistently passing.
    let steps = if reputation >= R_MIN && audit_pass_rate >= A_MIN {
        passing_audits / audits_per_step.max(1)
    } else {
        0
    };
    // Never above the model's true tier; never below the floor.
    let awarded_idx = (floor_idx + steps).min(claimed_idx).max(floor_idx);
    CostTier::from_index(awarded_idx as u8)
}

/// How many additional passing audits (at `audits_per_step`) this `(peer, model)` needs before its
/// awarded cap would reach `claimed`, or `0` if it is already there (or the gates aren't met, in
/// which case audits alone won't help — the caller should surface the reputation/pass-rate gap).
/// Advisory, for disclosure to the provider ("N more clean audits to earn full tier").
pub fn audits_to_full_tier(
    claimed: CostTier,
    reputation: f64,
    audit_pass_rate: f64,
    passing_audits: u32,
    audits_per_step: u32,
) -> u32 {
    if reputation < R_MIN || audit_pass_rate < A_MIN {
        // Gates unmet: the cap is pinned at the floor regardless of audits.
        return if claimed.index() > FLOOR_TIER.index() { u32::MAX } else { 0 };
    }
    let k = audits_per_step.max(1);
    let steps_needed = (claimed.index() as u32).saturating_sub(FLOOR_TIER.index() as u32);
    let audits_needed = steps_needed.saturating_mul(k);
    audits_needed.saturating_sub(passing_audits).min(audits_needed)
}

/// Split the credit a receipt would earn at its *claimed* tier into the part that is **withdrawable
/// now** (earned at the low *awarded* tier) and the part held in **escrow** (the claimed − awarded
/// delta), released as the cap climbs and clawed back on a clear fail (Decision #13).
///
/// `claimed_weight` / `awarded_weight` are the [`CostTier::weight`] multipliers; `weighted_tokens`
/// is the input/output-weighted token count ([`crate::weight::weighted_token_units`]). Because
/// `awarded ≤ claimed`, escrow is never negative. Pure arithmetic; whole credits, rounded down.
pub fn escrow_split(claimed_weight: f64, awarded_weight: f64, weighted_tokens: u64) -> EscrowSplit {
    let awarded_w = awarded_weight.min(claimed_weight).max(0.0);
    let withdrawable = (awarded_w * weighted_tokens as f64) as u64;
    let full = (claimed_weight.max(0.0) * weighted_tokens as f64) as u64;
    EscrowSplit { withdrawable, escrowed: full.saturating_sub(withdrawable) }
}

/// The result of [`escrow_split`]: what a receipt pays out now vs. what is held pending proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EscrowSplit {
    /// Credit earned at the *awarded* tier — withdrawable immediately.
    pub withdrawable: u64,
    /// Credit held back (claimed − awarded delta) — released as the cap climbs, forfeited on a
    /// clear fail.
    pub escrowed: u64,
}

/// The classification of one redundant-execution audit for the graduated-failure ladder. Mirrors
/// [`crate::verify::RedundantVerdict`] resolved *for one provider*: a clear temp=0 divergence is a
/// `Fail`; agreement is a `Pass`; anything ambiguous (tie, non-determinism, network drop, wrong
/// auditor) is `Inconclusive` and **never** penalized — only retried.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditClass {
    Pass,
    Fail,
    Inconclusive,
}

/// The escalating penalty a *clear* audit failure triggers for a `(peer, model)` (Decision #13 /
/// #6). Layered on top of the reputation AIMD: the ladder governs the **tier cap + escrow**, which
/// the plain reputation slash does not. Produced by [`GradedFault::on_audit`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultAction {
    /// No clear fault (a pass or an inconclusive audit): nothing to enforce.
    None,
    /// 1st clear fail → drop the awarded tier by ≥1, apply a serve cooldown, claw back escrow.
    DropAndClawback,
    /// A repeat clear fail → multiplicative reputation slash (AIMD ×0.4) → the cap collapses to the
    /// floor.
    Slash,
    /// Sustained clear fails → evict the `(peer, model)` and forfeit remaining escrow.
    Evict,
}

/// The graduated-failure state for one `(peer, model)`: counts *consecutive clear* audit failures
/// so the enforcement escalates drop → slash → eviction, and resets on a pass. Inconclusive audits
/// leave it unchanged (they are not evidence of fraud). Pure/clock-free — the live layer maps
/// [`FaultAction`] to the actual cooldown/slash/escrow effects.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GradedFault {
    /// Consecutive clear failures since the last pass.
    consecutive_fails: u32,
}

/// Consecutive clear fails at which enforcement escalates from [`FaultAction::Slash`] to
/// [`FaultAction::Evict`]. The 1st fail drops+claws back, the 2nd slashes, the 3rd+ evicts.
pub const EVICT_AFTER_FAILS: u32 = 3;

impl GradedFault {
    /// A fresh ladder (no failures yet).
    pub fn new() -> Self {
        Self::default()
    }

    /// Consecutive clear failures recorded since the last pass.
    pub fn consecutive_fails(&self) -> u32 {
        self.consecutive_fails
    }

    /// Fold one audit outcome into the ladder and return the enforcement action.
    ///
    /// * [`AuditClass::Pass`] → reset the streak, [`FaultAction::None`].
    /// * [`AuditClass::Inconclusive`] → **unchanged**, [`FaultAction::None`] (never penalize
    ///   ambiguous evidence).
    /// * [`AuditClass::Fail`] → increment; 1st → [`DropAndClawback`](FaultAction::DropAndClawback),
    ///   2nd → [`Slash`](FaultAction::Slash), `≥ EVICT_AFTER_FAILS` → [`Evict`](FaultAction::Evict).
    pub fn on_audit(&mut self, class: AuditClass) -> FaultAction {
        match class {
            AuditClass::Pass => {
                self.consecutive_fails = 0;
                FaultAction::None
            }
            AuditClass::Inconclusive => FaultAction::None,
            AuditClass::Fail => {
                self.consecutive_fails = self.consecutive_fails.saturating_add(1);
                match self.consecutive_fails {
                    1 => FaultAction::DropAndClawback,
                    2 => FaultAction::Slash,
                    _ => FaultAction::Evict,
                }
            }
        }
    }
}

/// Recency-weighted audit history for one `(peer, model)` — the `n` and `a` inputs to
/// [`max_awardable_tier`] (Decision #13). Two accumulators that **decay toward zero** with the
/// reputation half-life ([`crate::verify::DEFAULT_HALF_LIFE_MS`], ~7d, so recent audits dominate):
///
/// * `passes` — a decaying count of clear **passing** temp=0 audits (`+1` per pass). Its floored
///   value is `n`, the audits the ascension curve counts. Because it decays, a provider that stops
///   being audited slowly loses climbed tiers ("no coasting") — the same discipline reputation has.
/// * `total` — a decaying count of clear audits (pass **or** fail; `+1` each). `passes / total` is
///   the recency-weighted pass-rate `a`.
///
/// **Inconclusive audits touch neither** (non-determinism / network / wrong-auditor is not
/// evidence, per Decision #6). Pure and clock-injected (every method takes `now_ms`), exactly like
/// [`crate::verify::ReputationTracker`]; the live layer persists/rehydrates it per `(peer, model)`.
#[derive(Debug, Clone)]
pub struct AuditStats {
    /// Decaying count of clear passing audits (`n`, before flooring).
    passes: f64,
    /// Decaying count of clear audits (pass + fail).
    total: f64,
    /// Unix-ms the accumulators are current as of.
    last_update_ms: u64,
    /// Half-life (ms) of the decay toward zero. 0 = no decay.
    half_life_ms: u64,
    /// Unix-ms of the most recent failed audit (0 = none): a fresh failure caps the tier outright,
    /// whatever the decayed counts say.
    last_failed_ms: u64,
}

impl AuditStats {
    /// A fresh, empty history as of `now_ms`, decaying with the reputation half-life.
    pub fn new(now_ms: u64) -> Self {
        Self::with_half_life(now_ms, crate::verify::DEFAULT_HALF_LIFE_MS)
    }

    /// As [`new`](Self::new) but with an explicit decay half-life (`0` = no decay).
    pub fn with_half_life(now_ms: u64, half_life_ms: u64) -> Self {
        Self { passes: 0.0, total: 0.0, last_update_ms: now_ms, half_life_ms, last_failed_ms: 0 }
    }

    /// Decay multiplier `0.5^(elapsed / half_life)` for reading as of `now_ms` without mutating.
    /// Clock skew (earlier `now_ms`) yields no decay.
    fn decay_factor(&self, now_ms: u64) -> f64 {
        if self.half_life_ms == 0 {
            return 1.0;
        }
        let elapsed = now_ms.saturating_sub(self.last_update_ms) as f64;
        if elapsed <= 0.0 {
            return 1.0;
        }
        0.5_f64.powf(elapsed / self.half_life_ms as f64)
    }

    /// Roll both accumulators forward to `now_ms` in place (so a fresh record acts on decayed
    /// values, not stale ones).
    fn decay_to(&mut self, now_ms: u64) {
        let f = self.decay_factor(now_ms);
        self.last_update_ms = self.last_update_ms.max(now_ms);
        if f < 1.0 {
            self.passes *= f;
            self.total *= f;
        }
    }

    /// Fold one audit outcome into the history at `now_ms`. A [`Pass`](AuditClass::Pass) counts
    /// toward both `passes` and `total`; a [`Fail`](AuditClass::Fail) toward `total` only; an
    /// [`Inconclusive`](AuditClass::Inconclusive) is ignored (not evidence).
    pub fn record(&mut self, class: AuditClass, now_ms: u64) {
        if class == AuditClass::Inconclusive {
            return;
        }
        self.decay_to(now_ms);
        self.total += 1.0;
        if class == AuditClass::Pass {
            self.passes += 1.0;
        } else {
            self.last_failed_ms = self.last_failed_ms.max(now_ms);
        }
    }

    /// When the most recent failed audit was, if any.
    pub fn last_failed_ms(&self) -> Option<u64> {
        (self.last_failed_ms > 0).then_some(self.last_failed_ms)
    }

    /// Whether an audit failed within `window_ms` of `now_ms`.
    pub fn failed_within(&self, now_ms: u64, window_ms: u64) -> bool {
        self.last_failed_ms().is_some_and(|at| now_ms.saturating_sub(at) <= window_ms)
    }

    /// `n` — recent passing temp=0 audits (the decayed `passes`, floored), as of `now_ms`.
    pub fn recent_passes(&self, now_ms: u64) -> u32 {
        (self.passes * self.decay_factor(now_ms)).floor().max(0.0) as u32
    }

    /// `a` — recency-weighted pass-rate in `[0, 1]` as of `now_ms`. With no clear audits yet,
    /// returns `1.0` (nothing has failed) — harmless, since ascension is separately gated on
    /// `n = 0` at cold start.
    pub fn pass_rate(&self, now_ms: u64) -> f64 {
        let f = self.decay_factor(now_ms);
        let total = self.total * f;
        if total < 1e-9 {
            return 1.0;
        }
        ((self.passes * f) / total).clamp(0.0, 1.0)
    }

    /// The tier this `(peer, model)` may currently earn at, from its own audit history plus the
    /// caller-supplied per-`(peer, model)` `reputation` — a convenience wrapper over
    /// [`max_awardable_tier`] so the ledger doesn't recompute `n`/`a`.
    pub fn awardable_tier(&self, claimed: CostTier, reputation: f64, now_ms: u64, audits_per_step: u32) -> CostTier {
        max_awardable_tier(
            claimed,
            reputation,
            self.pass_rate(now_ms),
            self.recent_passes(now_ms),
            audits_per_step,
        )
    }

    /// Serialize the snapshot for the durable store: `passes:f64[8] ‖ total:f64[8] ‖
    /// last_update_ms:u64[8] ‖ half_life_ms:u64[8] ‖ last_failed_ms:u64[8]` little-endian = 40
    /// bytes. f64s stored by exact bit pattern so a rehydrated tracker decays identically.
    pub fn to_bytes(&self) -> [u8; 40] {
        let mut b = [0u8; 40];
        b[0..8].copy_from_slice(&self.passes.to_le_bytes());
        b[8..16].copy_from_slice(&self.total.to_le_bytes());
        b[16..24].copy_from_slice(&self.last_update_ms.to_le_bytes());
        b[24..32].copy_from_slice(&self.half_life_ms.to_le_bytes());
        b[32..40].copy_from_slice(&self.last_failed_ms.to_le_bytes());
        b
    }

    /// Reconstruct from [`to_bytes`](Self::to_bytes), or from the 32-byte snapshot written before
    /// failures were timed (no failure time). `Err` for any other length.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() != 32 && data.len() != 40 {
            return Err(format!("audit stats snapshot must be 32 or 40 bytes, got {}", data.len()));
        }
        let f8 = |r: std::ops::Range<usize>| f64::from_le_bytes(data[r].try_into().unwrap());
        let u8_ = |r: std::ops::Range<usize>| u64::from_le_bytes(data[r].try_into().unwrap());
        let (passes, total) = (f8(0..8), f8(8..16));
        // Reject a corrupt (locally-tampered) snapshot with a non-finite or negative accumulator
        // rather than propagate NaN/inf into the tier math (review LOW). Only ever read from the
        // local store, so this is defence-in-depth, not an untrusted-input surface.
        if !passes.is_finite() || !total.is_finite() || passes < 0.0 || total < 0.0 {
            return Err("audit stats snapshot has a non-finite/negative accumulator".into());
        }
        Ok(Self {
            passes,
            total,
            last_update_ms: u8_(16..24),
            half_life_ms: u8_(24..32),
            last_failed_ms: if data.len() == 40 { u8_(32..40) } else { 0 },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight::CostTier::*;

    // ── max_awardable_tier ──

    #[test]
    fn cold_start_awards_nano_regardless_of_claim() {
        // A brand-new (peer, model): n=0, neutral reputation → nano, even claiming mega.
        assert_eq!(max_awardable_tier(Mega, 50.0, 1.0, 0, AUDITS_PER_STEP), Nano);
        assert_eq!(max_awardable_tier(Frontier, 50.0, 1.0, 0, AUDITS_PER_STEP), Nano);
    }

    #[test]
    fn ascends_one_tier_per_k_passing_audits_when_gates_met() {
        let climb = |n| max_awardable_tier(Mega, 80.0, 0.95, n, 3);
        assert_eq!(climb(0), Nano);
        assert_eq!(climb(2), Nano, "below K → no step yet");
        assert_eq!(climb(3), Small, "K audits → +1 tier");
        assert_eq!(climb(6), Medium);
        assert_eq!(climb(9), Large);
        assert_eq!(climb(12), Frontier);
        assert_eq!(climb(15), Mega);
        assert_eq!(climb(18), Mega, "never above the claimed/true tier");
    }

    #[test]
    fn never_exceeds_the_models_true_tier() {
        // A provider of a nano model is always awarded nano, no matter how trusted / audited.
        assert_eq!(max_awardable_tier(Nano, 100.0, 1.0, 100, 3), Nano);
        // A small model caps at small even with a huge audit count.
        assert_eq!(max_awardable_tier(Small, 100.0, 1.0, 100, 3), Small);
    }

    #[test]
    fn low_reputation_pins_the_cap_at_the_floor() {
        // Plenty of passing audits, but reputation below R_MIN → no ascension.
        assert_eq!(max_awardable_tier(Mega, R_MIN - 1.0, 1.0, 99, 3), Nano);
        // Exactly at R_MIN with a full pass-rate → may climb.
        assert_eq!(max_awardable_tier(Mega, R_MIN, A_MIN, 3, 3), Small);
    }

    #[test]
    fn low_pass_rate_pins_the_cap_at_the_floor() {
        // High reputation, many audits, but a pass-rate below A_MIN → no ascension.
        assert_eq!(max_awardable_tier(Mega, 90.0, A_MIN - 0.01, 99, 3), Nano);
    }

    #[test]
    fn claim_big_serve_small_cannot_shortcut() {
        // The headline fraud: claim a mega model with farmed reputation but no audits on THIS
        // model → still nano. Audits on this (peer, model) are the only lever, and they're gated.
        assert_eq!(max_awardable_tier(Mega, 100.0, 1.0, 0, 3), Nano);
    }

    // ── audits_to_full_tier (disclosure) ──

    #[test]
    fn audits_to_full_tier_counts_down() {
        // Mega = index 5, floor nano = 0 → 5 steps × K(3) = 15 audits for full tier.
        assert_eq!(audits_to_full_tier(Mega, 80.0, 0.95, 0, 3), 15);
        assert_eq!(audits_to_full_tier(Mega, 80.0, 0.95, 6, 3), 9);
        assert_eq!(audits_to_full_tier(Mega, 80.0, 0.95, 15, 3), 0, "already at full");
        assert_eq!(audits_to_full_tier(Nano, 50.0, 0.0, 0, 3), 0, "nano needs none");
    }

    #[test]
    fn audits_to_full_tier_is_unreachable_while_gates_unmet() {
        // Gates unmet on a > floor model → audits alone won't unlock it (surface as unreachable).
        assert_eq!(audits_to_full_tier(Mega, 40.0, 1.0, 0, 3), u32::MAX);
    }

    // ── escrow_split ──

    #[test]
    fn escrow_holds_the_claimed_minus_awarded_delta() {
        // Claimed mega (150×), awarded nano (1×), 100 weighted tokens.
        let s = escrow_split(Mega.weight(), Nano.weight(), 100);
        assert_eq!(s.withdrawable, 100, "earns at the awarded (nano) tier only");
        assert_eq!(s.escrowed, 100 * 150 - 100, "the rest is held pending proof");
    }

    #[test]
    fn no_escrow_when_awarded_equals_claimed() {
        // Fully proven: awarded == claimed → nothing held.
        let s = escrow_split(Large.weight(), Large.weight(), 200);
        assert_eq!(s.escrowed, 0);
        assert_eq!(s.withdrawable, (Large.weight() * 200.0) as u64);
    }

    #[test]
    fn escrow_never_negative_even_if_awarded_exceeds_claimed() {
        // Defensive: an awarded weight above claimed is clamped, escrow floors at 0.
        let s = escrow_split(Small.weight(), Mega.weight(), 50);
        assert_eq!(s.escrowed, 0);
        assert_eq!(s.withdrawable, (Small.weight() * 50.0) as u64, "clamped to the claimed weight");
    }

    // ── graduated failure ladder ──

    #[test]
    fn ladder_escalates_drop_slash_evict_on_consecutive_fails() {
        let mut g = GradedFault::new();
        assert_eq!(g.on_audit(AuditClass::Fail), FaultAction::DropAndClawback);
        assert_eq!(g.on_audit(AuditClass::Fail), FaultAction::Slash);
        assert_eq!(g.on_audit(AuditClass::Fail), FaultAction::Evict);
        assert_eq!(g.on_audit(AuditClass::Fail), FaultAction::Evict, "stays evicted-level on further fails");
        assert_eq!(g.consecutive_fails(), 4);
    }

    #[test]
    fn a_pass_resets_the_failure_streak() {
        let mut g = GradedFault::new();
        g.on_audit(AuditClass::Fail); // DropAndClawback
        assert_eq!(g.on_audit(AuditClass::Pass), FaultAction::None);
        assert_eq!(g.consecutive_fails(), 0);
        // Next fail starts the ladder over at the mildest rung.
        assert_eq!(g.on_audit(AuditClass::Fail), FaultAction::DropAndClawback);
    }

    #[test]
    fn inconclusive_audits_never_penalize() {
        // The false-negative guard: non-determinism / network / wrong-auditor must not punish.
        let mut g = GradedFault::new();
        assert_eq!(g.on_audit(AuditClass::Inconclusive), FaultAction::None);
        assert_eq!(g.on_audit(AuditClass::Inconclusive), FaultAction::None);
        assert_eq!(g.consecutive_fails(), 0, "inconclusive leaves the streak untouched");
        // A fail after inconclusives is still only the 1st clear fail.
        assert_eq!(g.on_audit(AuditClass::Fail), FaultAction::DropAndClawback);
    }

    #[test]
    fn inconclusive_does_not_break_a_fail_streak() {
        // An inconclusive audit between two clear fails must not reset the escalation — otherwise a
        // fraudster could inject noise to stay at the mildest rung forever.
        let mut g = GradedFault::new();
        assert_eq!(g.on_audit(AuditClass::Fail), FaultAction::DropAndClawback);
        assert_eq!(g.on_audit(AuditClass::Inconclusive), FaultAction::None);
        assert_eq!(g.on_audit(AuditClass::Fail), FaultAction::Slash, "streak preserved across inconclusive");
    }

    // ── AuditStats (4-state: the n / a inputs to the curve) ──

    const DAY_MS: u64 = 24 * 60 * 60 * 1000;

    #[test]
    fn fresh_stats_have_no_passes_and_neutral_rate() {
        let s = AuditStats::new(0);
        assert_eq!(s.recent_passes(0), 0);
        assert_eq!(s.pass_rate(0), 1.0, "no clear audits → nothing has failed");
    }

    #[test]
    fn passes_and_rate_track_recorded_outcomes() {
        let mut s = AuditStats::with_half_life(0, 0); // decay off: isolate the counting
        s.record(AuditClass::Pass, 0);
        s.record(AuditClass::Pass, 0);
        s.record(AuditClass::Pass, 0);
        s.record(AuditClass::Fail, 0);
        assert_eq!(s.recent_passes(0), 3, "n = clear passes");
        assert!((s.pass_rate(0) - 0.75).abs() < 1e-9, "a = 3/4");
    }

    #[test]
    fn inconclusive_audits_change_neither_n_nor_a() {
        let mut s = AuditStats::with_half_life(0, 0);
        s.record(AuditClass::Pass, 0);
        s.record(AuditClass::Inconclusive, 0);
        s.record(AuditClass::Inconclusive, 0);
        assert_eq!(s.recent_passes(0), 1);
        assert_eq!(s.pass_rate(0), 1.0, "inconclusive is not a fail — rate stays 1.0");
    }

    #[test]
    fn passes_decay_so_a_provider_cannot_coast() {
        // Earn several passes, then go a long time with no audits: n decays back toward 0, so a
        // climbed tier is not held forever without fresh proof.
        let mut s = AuditStats::with_half_life(0, 7 * DAY_MS);
        for _ in 0..8 {
            s.record(AuditClass::Pass, 0);
        }
        assert_eq!(s.recent_passes(0), 8);
        assert_eq!(s.recent_passes(7 * DAY_MS), 4, "one half-life halves the recent-pass count");
        assert_eq!(s.recent_passes(7 * DAY_MS * 10), 0, "far future → coasting is impossible");
    }

    #[test]
    fn record_acts_on_decayed_values_not_stale_ones() {
        let mut s = AuditStats::with_half_life(0, 7 * DAY_MS);
        for _ in 0..4 {
            s.record(AuditClass::Pass, 0);
        }
        // A half-life later, passes has decayed to ~2; a new pass adds 1 → ~3, not 5.
        s.record(AuditClass::Pass, 7 * DAY_MS);
        assert_eq!(s.recent_passes(7 * DAY_MS), 3);
    }

    #[test]
    fn clock_skew_never_grows_the_history() {
        let mut s = AuditStats::with_half_life(100, 7 * DAY_MS);
        s.record(AuditClass::Pass, 100);
        assert_eq!(s.recent_passes(50), s.recent_passes(100), "a past read must not un-decay");
    }

    #[test]
    fn awardable_tier_wires_history_into_the_curve() {
        // The convenience wrapper reproduces max_awardable_tier from the tracked n / a.
        let mut s = AuditStats::with_half_life(0, 0);
        // 3 clean passes at high reputation → +1 tier from nano = small.
        for _ in 0..3 {
            s.record(AuditClass::Pass, 0);
        }
        assert_eq!(s.awardable_tier(Mega, 80.0, 0, AUDITS_PER_STEP), Small);
        // Same passes but a fail drags the pass-rate below A_MIN → pinned to the floor.
        s.record(AuditClass::Fail, 0); // rate now 3/4 < 0.9
        assert_eq!(s.awardable_tier(Mega, 80.0, 0, AUDITS_PER_STEP), Nano);
    }

    #[test]
    fn audit_stats_bytes_roundtrip_preserves_decay() {
        let mut s = AuditStats::with_half_life(1_000, 7 * DAY_MS);
        s.record(AuditClass::Pass, 1_000);
        s.record(AuditClass::Pass, 1_000);
        s.record(AuditClass::Fail, 1_000);
        let back = AuditStats::from_bytes(&s.to_bytes()).unwrap();
        assert_eq!(back.recent_passes(1_000), s.recent_passes(1_000));
        assert!((back.pass_rate(1_000) - s.pass_rate(1_000)).abs() < 1e-12);
        // And identical decay a week later → the half-life state round-tripped.
        assert_eq!(back.recent_passes(1_000 + 7 * DAY_MS), s.recent_passes(1_000 + 7 * DAY_MS));
        assert_eq!(back.last_failed_ms(), Some(1_000));
        // A snapshot from before failures were timed still reads, with none.
        let old = AuditStats::from_bytes(&s.to_bytes()[..32]).unwrap();
        assert_eq!(old.recent_passes(1_000), s.recent_passes(1_000));
        assert_eq!(old.last_failed_ms(), None);
    }

    #[test]
    fn a_failed_audit_is_remembered_for_its_window() {
        let mut s = AuditStats::with_half_life(0, 7 * DAY_MS);
        s.record(AuditClass::Pass, 10);
        assert!(!s.failed_within(10, DAY_MS));
        s.record(AuditClass::Fail, 100);
        s.record(AuditClass::Pass, 200);
        assert!(s.failed_within(200, DAY_MS), "a later pass doesn't erase it");
        assert!(s.failed_within(100 + DAY_MS, DAY_MS));
        assert!(!s.failed_within(101 + DAY_MS, DAY_MS));
        s.record(AuditClass::Inconclusive, 5_000);
        assert_eq!(s.last_failed_ms(), Some(100));
    }

    #[test]
    fn audit_stats_from_bytes_rejects_wrong_length() {
        assert!(AuditStats::from_bytes(&[0u8; 31]).is_err());
        assert!(AuditStats::from_bytes(&[0u8; 33]).is_err());
        assert!(AuditStats::from_bytes(&[0u8; 41]).is_err());
    }
}
