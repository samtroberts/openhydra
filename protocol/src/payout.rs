// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Payout-layer earn-gate (UNIFIED_TRUST_STACK / PAYMENTS_PLAN_v1 2026-09-17).
//!
//! Where cost-weighted credit becomes **redeemable**, the achieved authenticity tier gates what a
//! `(peer, model)` actually earns — the concrete home for credit-tiering review finding **H1** (per
//! the H2 remediation: the give/take throttle stays raw-token, the cost-weighted credit *and its
//! gate* live here at payout). The gate itself is universal:
//!
//! ```text
//! awarded_tier = max_cost_tier_for_authenticity(achieved, claimed)   // trust.rs
//! ```
//!
//! …but the **enforcement shape is rail-specific**, because the two rails settle at different
//! granularity (PAYMENTS_PLAN_v1 2026-09-17 refinement):
//!
//! * **Crypto (x402 micropayments)** — meter every token at the *awarded* weight ([`gated_accrual`]).
//!   The running receipt only ever accrues at the **proven** rate, so the provider is never
//!   over-credited: there is **no escrow and nothing to claw back**. As a `(peer, model)` proves a
//!   higher tier, later tokens accrue higher. This is why micropayments make an authenticity
//!   hold-back unnecessary on crypto.
//! * **Fiat (coarse batched payout)** — it cannot meter-and-gate per token, so it pays the *awarded*
//!   portion and **escrows** the claimed-vs-awarded delta ([`escrowed_split`]), releasing it as the
//!   tier is proven and **clawing it back** on a proven substitution ([`escrow_release`], driven by
//!   [`crate::award::GradedFault`]). Escrow is the fiat stand-in for the crypto rate-cap — not a
//!   universal mechanism.
//!
//! **Both rails take a 4% protocol fee** ([`PROTOCOL_FEE_BPS`]), applied to whatever actually pays
//! out to the provider (crypto: each accrual; fiat: the redeemable now, and the escrow when later
//! released). One rate, both rails, so pricing is rail-neutral.
//!
//! **Purity.** No I/O, no clock, no rail SDK — tiers + weighted token units in, split amounts out.
//! Nothing here is wired to a live ledger or a settlement contract; the crypto facilitator / fiat
//! payout ledger consume these amounts when that layer lands. Whole credits, integer arithmetic.

use crate::award::{escrow_split, FaultAction};
use crate::trust::{max_cost_tier_for_authenticity, AuthenticityTier};
use crate::weight::CostTier;

/// The protocol fee in **basis points** (hundredths of a percent): `400 = 4.00%`. One rate on both
/// rails (PAYMENTS_PLAN_v1 2026-09-17). On crypto this is the on-chain `0.96 / 0.04` split; on fiat
/// it is the payout policy's 4% line.
pub const PROTOCOL_FEE_BPS: u64 = 400;

/// Basis-points denominator (`100% = 10_000 bps`).
const BPS_DENOMINATOR: u64 = 10_000;

/// A gross amount split into the provider's net and the protocol fee. `net_to_provider + protocol_fee
/// == gross` exactly (the fee is floored, so any rounding dust favours the provider).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeeSplit {
    /// What the provider actually receives (`gross - protocol_fee`).
    pub net_to_provider: u64,
    /// The protocol's cut (`floor(gross × PROTOCOL_FEE_BPS / 10_000)`).
    pub protocol_fee: u64,
}

/// Split a gross payout amount into the provider's net and the 4% protocol fee. Integer math; the
/// fee is floored so `net + fee == gross` and the rounding dust (at most a few thousandths of a
/// credit) goes to the provider, never the platform.
pub fn protocol_fee(gross: u64) -> FeeSplit {
    // `gross ≤ u64::MAX`, and `gross × 400` can overflow u64 only above ~4.6e16 credits — far beyond
    // any real receipt — but use u128 for the intermediate product so it is impossible.
    let fee = ((gross as u128 * PROTOCOL_FEE_BPS as u128) / BPS_DENOMINATOR as u128) as u64;
    FeeSplit { net_to_provider: gross.saturating_sub(fee), protocol_fee: fee }
}

/// The gross credit a receipt is worth at cost-tier `weight`, for `weighted_units` input/output-
/// weighted token units (`prompt·w_in + completion·w_out`, [`crate::weight::weighted_token_units`]).
/// Matches [`crate::award::escrow_split`]'s internal `weight × weighted_tokens`, so the crypto and
/// fiat paths agree on what a tier is worth. Whole credits, floored.
fn gross_at_tier(tier: CostTier, weighted_units: u64) -> u64 {
    (tier.weight() * weighted_units as f64) as u64
}

/// **Crypto rail** — the authenticity-gated amount that accrues for `weighted_units` metered this
/// step, net of the protocol fee, **in credit units**. The rate is capped at the *awarded* tier
/// ([`max_cost_tier_for_authenticity`]), so the running micropayment receipt only ever accrues at
/// the proven rate — **no escrow, no clawback**. Returns the provider net + the protocol fee (the
/// on-chain `0.96 / 0.04` split for this accrual). For on-chain settlement use
/// [`gated_accrual_asset`], which denominates in the settlement asset.
pub fn gated_accrual(claimed: CostTier, achieved: AuthenticityTier, weighted_units: u64) -> FeeSplit {
    let awarded = max_cost_tier_for_authenticity(achieved, claimed);
    protocol_fee(gross_at_tier(awarded, weighted_units))
}

/// The on-chain settlement asset a crypto-rail voucher/receipt is denominated in. The `cumulative_amount`
/// on a receipt is in **this asset's smallest unit** (micro-units), converted from credit units via
/// the reference-price peg. Bound into the signed receipt preimage (`RECEIPT_FORMAT_V5`, tag 4) so a
/// consumer and provider that disagree on the asset can't silently co-sign a claim — the mismatch
/// breaks the signature / is rejected before co-sign. Discriminants are stable + never `0` (a `0` byte
/// is reserved-invalid), mirroring [`crate::crypto_agility::SigAlg`], so the enum can grow (USDT, …)
/// without colliding with a legacy absent field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SettlementAsset {
    /// USD Coin (6-decimal micro-units). The only asset settled today.
    Usdc = 1,
}

impl SettlementAsset {
    /// Stable wire discriminant (never `0`).
    pub fn to_u8(self) -> u8 {
        self as u8
    }
    /// Parse a wire discriminant; an unknown value is rejected rather than defaulted.
    pub fn from_u8(v: u8) -> Result<Self, String> {
        match v {
            1 => Ok(Self::Usdc),
            other => Err(format!("unknown settlement asset discriminant: {other}")),
        }
    }
}

/// Convert an amount in **credit units** to the settlement asset's smallest unit (e.g. USDC
/// micro-units at 6 decimals) using `micros_per_credit` — the reference-price peg
/// (CREDIT_TIERING $/credit). Saturating `u128` product, so a pathological value pins at `u64::MAX`
/// rather than wrapping.
pub fn credits_to_asset_micros(credit_units: u64, micros_per_credit: u64) -> u64 {
    ((credit_units as u128).saturating_mul(micros_per_credit as u128)).min(u64::MAX as u128) as u64
}

/// **Crypto rail, on-chain-denominated** — like [`gated_accrual`], but the gross is converted to the
/// **settlement asset's smallest unit** (USDC micro-units) via `micros_per_credit` **before** the 4%
/// split, so the co-signed voucher/receipt carries the on-chain-settleable amount directly. The peg
/// is applied at **sign time**, which is why the Solana program needs no on-chain oracle
/// (`SOLANA_SETTLEMENT_PROGRAM_SPEC` §2). Same authenticity rate-cap as [`gated_accrual`].
pub fn gated_accrual_asset(
    claimed: CostTier,
    achieved: AuthenticityTier,
    weighted_units: u64,
    micros_per_credit: u64,
) -> FeeSplit {
    let awarded = max_cost_tier_for_authenticity(achieved, claimed);
    let gross_credits = gross_at_tier(awarded, weighted_units);
    protocol_fee(credits_to_asset_micros(gross_credits, micros_per_credit))
}

/// **Fiat rail** — the authenticity-gated split of a receipt worth `weighted_units` at its *claimed*
/// tier into: what is **redeemable now** (the awarded-tier portion, net of the protocol fee), the
/// protocol fee on that redeemable, and what is **escrowed** (the claimed-vs-awarded delta, held
/// pending proof — released or clawed back later by [`escrow_release`]). Because coarse fiat payout
/// can't meter per token, the escrow is the stand-in for the crypto rate-cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PayoutSplit {
    /// Paid to the provider now — the awarded-tier credit, net of the protocol fee.
    pub redeemable_net: u64,
    /// The protocol fee taken from the redeemable-now portion.
    pub protocol_fee: u64,
    /// Held in escrow (the claimed − awarded delta, gross) pending proof; not yet fee'd (the fee on
    /// it is taken by [`escrow_release`] if and when it pays out).
    pub escrowed: u64,
}

/// Compute the fiat [`PayoutSplit`]: award-gate the tier, take the fee on the redeemable-now
/// portion, and hold the claimed-vs-awarded delta in escrow.
pub fn escrowed_split(claimed: CostTier, achieved: AuthenticityTier, weighted_units: u64) -> PayoutSplit {
    let awarded = max_cost_tier_for_authenticity(achieved, claimed);
    // `escrow_split` splits gross-at-claimed into withdrawable (gross-at-awarded) + escrowed (delta).
    let split = escrow_split(claimed.weight(), awarded.weight(), weighted_units);
    let fee = protocol_fee(split.withdrawable);
    PayoutSplit {
        redeemable_net: fee.net_to_provider,
        protocol_fee: fee.protocol_fee,
        escrowed: split.escrowed,
    }
}

/// The disposition of a held escrow amount once an audit resolves it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EscrowRelease {
    /// Paid to the provider on release (the escrow, net of the protocol fee) — `0` on a clawback.
    pub released_net: u64,
    /// The protocol fee taken on release — `0` on a clawback.
    pub protocol_fee: u64,
    /// Returned to the consumer on a clawback — `0` on a clean release.
    pub clawed_back: u64,
}

/// Resolve a held `escrowed` amount by the [`FaultAction`] an audit produced
/// ([`crate::award::GradedFault::on_audit`]). A clean audit ([`FaultAction::None`], e.g. a passing
/// verdict that lifts the proven tier) **releases** the escrow to the provider, net of the 4% fee;
/// any fault ([`DropAndClawback`](FaultAction::DropAndClawback) / [`Slash`](FaultAction::Slash) /
/// [`Evict`](FaultAction::Evict) — a proven substitution) **claws it back** to the consumer, no fee.
/// Pure; the fault-escalation ladder itself lives in [`crate::award::GradedFault`].
pub fn escrow_release(escrowed: u64, action: FaultAction) -> EscrowRelease {
    match action {
        FaultAction::None => {
            let fee = protocol_fee(escrowed);
            EscrowRelease {
                released_net: fee.net_to_provider,
                protocol_fee: fee.protocol_fee,
                clawed_back: 0,
            }
        }
        FaultAction::DropAndClawback | FaultAction::Slash | FaultAction::Evict => {
            EscrowRelease { released_net: 0, protocol_fee: 0, clawed_back: escrowed }
        }
    }
}

/// Why a provider rejected a co-signed cumulative amount ([`check_cumulative`], review F2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CumulativeError {
    /// The cumulative went **backward** (below the last one this provider co-signed for the pair) —
    /// a replay or a stale/forged lower amount.
    NonMonotonic,
    /// The cumulative rose by **less than the floor** owed for the tokens served this receipt — a
    /// consumer under-signing the payment (e.g. `Some(1)` for a real serve).
    BelowFloor,
}

/// Provider-side validation of a co-signed `cumulative_amount` before the provider counter-signs
/// (review F2): the provider accepts the consumer-authored figure, so it must at least (a) refuse a
/// **non-monotonic** cumulative (below the last it co-signed for this pair), and (b) when it knows
/// the settlement peg, refuse one whose **delta is below the floor** (`weighted_units` at the floor
/// tier — [`CostTier::Nano`], weight `1`) for the tokens served. The floor cannot be forged down
/// because the provider computes it from its **own** validated token count + peg; it does not need
/// the consumer's private authenticity tier (which only ever *raises* the amount above the floor).
///
/// `last` = the last cumulative co-signed for this `(consumer, model)` (0 if none). `cumulative` =
/// the receipt's amount (`None` = a legacy/no-crypto receipt → always Ok, nothing to settle).
/// `weighted_units` = [`crate::weight::receipt_weighted_units`] for this receipt. `peg` = the
/// provider's configured USDC-per-credit peg, or `None` to skip the floor check (monotonic only).
pub fn check_cumulative(
    last: u64,
    cumulative: Option<u64>,
    weighted_units: u64,
    peg: Option<u64>,
) -> Result<(), CumulativeError> {
    let Some(cumulative) = cumulative else {
        return Ok(());
    };
    if cumulative < last {
        return Err(CumulativeError::NonMonotonic);
    }
    if let Some(peg) = peg {
        // Floor tier (Nano) weight is 1, so the floor gross == weighted_units converted to the asset.
        let floor = credits_to_asset_micros(weighted_units, peg);
        if cumulative.saturating_sub(last) < floor {
            return Err(CumulativeError::BelowFloor);
        }
    }
    Ok(())
}

/// A monotonic per-`(consumer, provider, model)` accrual of authenticity-gated credit — the
/// off-chain running **voucher** the crypto rail settles against (PAYMENTS_PLAN_v1 crypto sequence
/// step 3). Each served step adds its [`gated_accrual`] (provider net + the 4% protocol fee); the
/// totals only ever rise, so the latest voucher is the cumulative amount a provider could claim.
///
/// **No real value moves here** — this is the off-chain accrual (the metering half). The on-chain
/// claim that pays it out (and its legal gates: non-custodial opinion, contract audit, geofencing)
/// is a later slice. 16-byte little-endian wire form (`net ‖ fee`) so it persists beside the other
/// per-`(peer, model)` state ([`crate::award::AuditStats`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GatedVoucher {
    /// Cumulative net owed to the provider (the `0.96` share).
    pub cumulative_net: u64,
    /// Cumulative protocol fee accrued (the `0.04` share).
    pub cumulative_fee: u64,
}

impl GatedVoucher {
    /// Add one served step's gated split. Monotonic — both totals only ever rise (saturating, so a
    /// pathological overflow pins at `u64::MAX` rather than wrapping a voucher backwards).
    pub fn accrue(&mut self, split: FeeSplit) {
        self.cumulative_net = self.cumulative_net.saturating_add(split.net_to_provider);
        self.cumulative_fee = self.cumulative_fee.saturating_add(split.protocol_fee);
    }

    /// The cumulative gross authorized (`net + fee`) — what the on-chain contract would split.
    pub fn cumulative_gross(&self) -> u64 {
        self.cumulative_net.saturating_add(self.cumulative_fee)
    }

    /// Catch up to a co-signed cumulative `gross` this voucher fell behind (a co-sign reply never
    /// received, a voucher lost): re-split it with [`protocol_fee`]. Never lowers the voucher.
    pub fn raise_to_gross(&mut self, gross: u64) {
        if gross > self.cumulative_gross() {
            let split = protocol_fee(gross);
            self.cumulative_net = split.net_to_provider;
            self.cumulative_fee = split.protocol_fee;
        }
    }

    /// Serialize to the 16-byte snapshot (`cumulative_net` LE ‖ `cumulative_fee` LE).
    pub fn to_bytes(&self) -> [u8; 16] {
        let mut b = [0u8; 16];
        b[0..8].copy_from_slice(&self.cumulative_net.to_le_bytes());
        b[8..16].copy_from_slice(&self.cumulative_fee.to_le_bytes());
        b
    }

    /// Parse a 16-byte snapshot, or `Err` if the blob is the wrong length (a corrupt row must fail
    /// closed, never silently zero a voucher).
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() != 16 {
            return Err(format!("GatedVoucher: expected 16 bytes, got {}", data.len()));
        }
        let cumulative_net = u64::from_le_bytes(data[0..8].try_into().unwrap());
        let cumulative_fee = u64::from_le_bytes(data[8..16].try_into().unwrap());
        Ok(Self { cumulative_net, cumulative_fee })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trust::AuthenticityTier::*;
    use crate::weight::CostTier::*;

    // ── protocol_fee: exactly 4%, conserving, provider-favouring rounding ──

    #[test]
    fn protocol_fee_is_four_percent_and_conserves_the_gross() {
        let s = protocol_fee(1_000);
        assert_eq!(s.protocol_fee, 40, "4% of 1000");
        assert_eq!(s.net_to_provider, 960);
        assert_eq!(s.net_to_provider + s.protocol_fee, 1_000, "no credit is created or lost");
    }

    #[test]
    fn protocol_fee_floors_the_fee_so_dust_favours_the_provider() {
        // 4% of 10 = 0.4 → fee floors to 0, provider keeps all 10.
        let s = protocol_fee(10);
        assert_eq!((s.net_to_provider, s.protocol_fee), (10, 0));
        assert_eq!(protocol_fee(0), FeeSplit { net_to_provider: 0, protocol_fee: 0 });
        // Conservation holds for a spread of values.
        for gross in [1u64, 7, 25, 99, 12_345, 1_000_000] {
            let s = protocol_fee(gross);
            assert_eq!(s.net_to_provider + s.protocol_fee, gross);
            assert!(s.protocol_fee <= gross / 25 + 1); // ~4% upper bound
        }
    }

    // ── crypto: gated_accrual is a rate cap, no escrow ──

    #[test]
    fn crypto_accrues_at_the_awarded_rate_not_the_claimed_rate() {
        // A0 (unverified) claiming Mega accrues only at the floor (Nano) rate — the rate cap.
        let units = 100;
        let a0 = gated_accrual(Mega, SelfReported, units);
        let expected_gross = (Nano.weight() * units as f64) as u64; // floor tier
        assert_eq!(a0.net_to_provider + a0.protocol_fee, expected_gross);
        // A2 (hard method proven) accrues at the full claimed rate.
        let a2 = gated_accrual(Mega, EnergyFingerprintPossession, units);
        let full_gross = (Mega.weight() * units as f64) as u64;
        assert_eq!(a2.net_to_provider + a2.protocol_fee, full_gross);
        // The gated accrual is strictly less at A0 than at A2 — the cap bites.
        assert!(a0.net_to_provider < a2.net_to_provider);
    }

    #[test]
    fn crypto_accrual_takes_the_four_percent_fee() {
        // A1 caps Mega at Medium (12×): gross = 12 × 100 = 1200, fee = 48, net = 1152.
        let a1 = gated_accrual(Mega, ReputationRedundant, 100);
        assert_eq!(a1.protocol_fee, 48);
        assert_eq!(a1.net_to_provider, 1_152);
    }

    #[test]
    fn asset_denominated_accrual_applies_the_peg_before_the_split() {
        // A1 caps Mega→Medium: gross_credits = 1200. Peg = 5 µUSDC/credit → gross_asset = 6000,
        // fee = 240, net = 5760. Peg applied before the 4% split (spec §2 — no on-chain oracle).
        let peg = 5;
        let a = gated_accrual_asset(Mega, ReputationRedundant, 100, peg);
        assert_eq!(a.net_to_provider + a.protocol_fee, 6_000, "gross converted to the asset");
        assert_eq!(a.protocol_fee, 240);
        assert_eq!(a.net_to_provider, 5_760);
        // Equivalent to converting the credit-unit gross then taking the fee.
        let credit_gross =
            gated_accrual(Mega, ReputationRedundant, 100).net_to_provider + gated_accrual(Mega, ReputationRedundant, 100).protocol_fee;
        assert_eq!(a.net_to_provider + a.protocol_fee, credits_to_asset_micros(credit_gross, peg));
    }

    #[test]
    fn credits_to_asset_micros_scales_and_saturates() {
        assert_eq!(credits_to_asset_micros(1_000, 6), 6_000);
        assert_eq!(credits_to_asset_micros(0, 6), 0);
        assert_eq!(credits_to_asset_micros(u64::MAX, u64::MAX), u64::MAX, "saturates, never wraps");
    }

    #[test]
    fn settlement_asset_discriminant_roundtrips_and_rejects_unknown() {
        assert_eq!(SettlementAsset::Usdc.to_u8(), 1, "stable wire discriminant");
        assert_eq!(SettlementAsset::from_u8(1), Ok(SettlementAsset::Usdc));
        assert!(SettlementAsset::from_u8(0).is_err(), "0 is reserved-invalid (distinct from an absent field)");
        assert!(SettlementAsset::from_u8(2).is_err(), "an unknown asset is rejected, never defaulted");
    }

    // ── fiat: escrowed_split holds the delta, fees the redeemable ──

    #[test]
    fn fiat_holds_the_claimed_minus_awarded_delta_in_escrow() {
        // A0 claiming Mega (150×) over 100 units: awarded = Nano (1×).
        // withdrawable = 1×100 = 100 (fee 4 → net 96); escrowed = (150-1)×100 = 14_900.
        let s = escrowed_split(Mega, SelfReported, 100);
        assert_eq!(s.redeemable_net, 96);
        assert_eq!(s.protocol_fee, 4);
        assert_eq!(s.escrowed, 14_900, "the unproven delta is held, not paid");
    }

    #[test]
    fn fiat_fully_proven_holds_no_escrow() {
        // A2 proven: awarded == claimed, so nothing is escrowed and the whole amount is redeemable.
        let s = escrowed_split(Mega, EnergyFingerprintPossession, 100);
        assert_eq!(s.escrowed, 0);
        let gross = (Mega.weight() * 100.0) as u64; // 15_000
        assert_eq!(s.redeemable_net + s.protocol_fee, gross);
        assert_eq!(s.protocol_fee, gross / 25); // exactly 4% (15_000 divisible)
    }

    #[test]
    fn crypto_redeemable_equals_fiat_redeemable_for_the_same_evidence() {
        // The rails agree on what pays out NOW; fiat merely *also* holds the rest in escrow.
        for achieved in [SelfReported, ReputationRedundant, EnergyFingerprintPossession] {
            let crypto = gated_accrual(Frontier, achieved, 100);
            let fiat = escrowed_split(Frontier, achieved, 100);
            assert_eq!(crypto.net_to_provider, fiat.redeemable_net);
            assert_eq!(crypto.protocol_fee, fiat.protocol_fee);
        }
    }

    // ── escrow_release: clean pass releases (net of fee), a fault claws back ──

    #[test]
    fn escrow_releases_to_provider_net_of_fee_on_a_clean_audit() {
        let r = escrow_release(10_000, FaultAction::None);
        assert_eq!(r.released_net, 9_600);
        assert_eq!(r.protocol_fee, 400);
        assert_eq!(r.clawed_back, 0);
    }

    #[test]
    fn escrow_is_clawed_back_in_full_on_any_fault() {
        for action in [FaultAction::DropAndClawback, FaultAction::Slash, FaultAction::Evict] {
            let r = escrow_release(10_000, action);
            assert_eq!(r.clawed_back, 10_000, "a proven substitution returns the held delta");
            assert_eq!((r.released_net, r.protocol_fee), (0, 0), "nothing pays out on a fault");
        }
    }

    // ── GatedVoucher: monotonic accrual + round-trip ──

    #[test]
    fn voucher_accrues_monotonically() {
        let mut v = GatedVoucher::default();
        v.accrue(gated_accrual(Mega, ReputationRedundant, 100)); // A1 caps Mega→Medium: net 1152 fee 48
        assert_eq!((v.cumulative_net, v.cumulative_fee), (1_152, 48));
        assert_eq!(v.cumulative_gross(), 1_200);
        // A second, higher-tier step only ever adds — the voucher never moves backward.
        v.accrue(gated_accrual(Mega, EnergyFingerprintPossession, 100)); // full 15000: net 14400 fee 600
        assert_eq!((v.cumulative_net, v.cumulative_fee), (1_152 + 14_400, 48 + 600));
        assert!(v.cumulative_gross() >= 1_200, "cumulative gross only rises");
    }

    #[test]
    fn check_cumulative_enforces_monotonicity_and_the_floor() {
        // No cumulative (legacy receipt) is always fine.
        assert_eq!(check_cumulative(100, None, 50, Some(5)), Ok(()));
        // Backward → NonMonotonic (replay / stale / forged-lower).
        assert_eq!(check_cumulative(100, Some(99), 50, Some(5)), Err(CumulativeError::NonMonotonic));
        assert_eq!(check_cumulative(100, Some(100), 0, None), Ok(()), "equal is monotonic-ok");
        // Floor: units=50, peg=5 → floor = 250. A delta below 250 for a real serve is under-signing.
        assert_eq!(check_cumulative(100, Some(100 + 1), 50, Some(5)), Err(CumulativeError::BelowFloor));
        assert_eq!(check_cumulative(100, Some(100 + 249), 50, Some(5)), Err(CumulativeError::BelowFloor));
        assert_eq!(check_cumulative(100, Some(100 + 250), 50, Some(5)), Ok(()), "delta == floor is ok");
        assert_eq!(check_cumulative(100, Some(100 + 999), 50, Some(5)), Ok(()), "above floor is ok");
        // Without a peg, only monotonicity is enforced (no floor) — a tiny delta passes.
        assert_eq!(check_cumulative(100, Some(101), 50, None), Ok(()));
    }

    #[test]
    fn a_voucher_catches_up_but_never_goes_back() {
        let mut v = GatedVoucher { cumulative_net: 960, cumulative_fee: 40 };
        v.raise_to_gross(500);
        assert_eq!(v.cumulative_gross(), 1000, "never lowered");
        v.raise_to_gross(2500);
        assert_eq!(v.cumulative_gross(), 2500);
        assert_eq!(v.cumulative_fee, 100);
    }

    #[test]
    fn voucher_round_trips_through_bytes() {
        let v = GatedVoucher { cumulative_net: 123_456, cumulative_fee: 5_144 };
        assert_eq!(GatedVoucher::from_bytes(&v.to_bytes()).unwrap(), v);
        assert_eq!(v.to_bytes().len(), 16);
        // A wrong-length blob fails closed rather than zeroing the voucher.
        assert!(GatedVoucher::from_bytes(&[0u8; 15]).is_err());
        assert!(GatedVoucher::from_bytes(&[]).is_err());
    }
}
