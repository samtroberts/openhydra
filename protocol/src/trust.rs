// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Unified trust stack — the typed assurance model (UNIFIED_TRUST_STACK_v1 §5).
//!
//! OpenHydra, darkbloom, and AntSeed are usually mistaken for competitors; they mostly aren't.
//! Each owns a **different axis of trust**, and no system has all three:
//!
//! | Axis | Question it answers | Mechanism |
//! |---|---|---|
//! | **Authenticity** | *Did the provider run the model it claims — not a cheaper one?* | redundant-exec + energy-keystone + fingerprint/possession + reputation |
//! | **Privacy** | *Can the provider's adversarial owner read my prompt?* | attested agent / in-process engine (SIP / GPU-CC) |
//! | **Settlement** | *Do I actually get paid, and are cheats punished?* | non-custodial USDC voucher + staking/slashing |
//!
//! This module is the **pure typed spine** the rest of the stack plugs into: the three per-axis
//! tier ladders ([`AuthenticityTier`] / [`PrivacyTier`] / [`SettlementTier`]), the achieved triple
//! ([`AssuranceTier`]), the buyer's [`AssurancePolicy`] with its single routing/serve predicate
//! [`AssurancePolicy::admits`], and the evidence→tier resolution ([`resolve_assurance`],
//! [`authenticity_tier_from`]). It also states, as pure arithmetic, how an achieved authenticity
//! tier **gates the awardable [`CostTier`]** ([`max_cost_tier_for_authenticity`]) — the concrete
//! home for credit-tiering review finding **H1** (the earn-gate can't verify model identity without
//! Proof-of-Inference).
//!
//! ## Two design invariants (from the strategy doc + the PoI addendum)
//! * **Tier by verification METHOD, not device vendor.** Assurance is a property of *what is
//!   proven*, not the logo. Device capability sets the *ceiling* (a consumer Mac can reach energy /
//!   possession but not a GPU-CC enclave); the tier *label* is the mechanism. There is no
//!   "SIP-tier for Macs / TEE-tier for Nvidia".
//! * **Provider-disclosed, verifier-checked.** A tier is never the provider's word — it is what an
//!   independent verifier confirms. In this increment Privacy and Settlement are carried as
//!   *disclosed* levels (their verifiers land later); Authenticity is already resolved from the
//!   existing reputation + redundant-exec audit evidence.
//!
//! **Purity.** No I/O, no clock, no wire format — evidence in, tiers out. Nothing here is wired into
//! routing, the serve gate, or the ledger yet; [`max_cost_tier_for_authenticity`] is the documented
//! payout-layer gate and is **not** consumed by any live credit path (the give/take throttle stays
//! raw-token; the cost-weighted credit gate lives at payout — the H2 remediation). The wiring lands
//! in later increments (verifier plugins + the routing/serve gate).

use crate::award::{A_MIN, FLOOR_TIER, R_MIN};
use crate::verify::VerificationOutcome;
use crate::weight::CostTier;

/// **Authenticity** — *did the provider run the model it claims?* The axis OpenHydra owns. Ordered
/// by the strength of the *method* that backs the claim (`A0 < A1 < A2 < A3`), so a higher variant
/// is a strictly stronger authenticity guarantee (this ordering is what [`AssurancePolicy::admits`]
/// compares against).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AuthenticityTier {
    /// **Self-reported.** The provider's word only — no verification. PoI Tier 0 (free tier, lowest
    /// pay/priority). The conservative floor: an unproven or freshly-caught provider lands here.
    SelfReported,
    /// **Reputation + redundant-exec agreement.** This `(peer, model)` clears the reputation and
    /// recency-weighted audit pass-rate gates ([`R_MIN`] / [`A_MIN`]) on real passing audits — its
    /// outputs agree with independent re-execution. PoI Tier 1 (standard paid). Agreement among
    /// cheap models can be colluded, so this alone does not unlock the expensive cost-tiers.
    ReputationRedundant,
    /// **Energy / fingerprint / possession.** A1 *plus* at least one hard Proof-of-Inference method
    /// (weight-possession digest, logprob commitment, or the energy keystone). The conservation-law
    /// / cryptographic axes that make a coherent forgery combinatorially harder — this is what
    /// unlocks the frontier cost-tiers.
    EnergyFingerprintPossession,
    /// **Hardware-attested compute.** A TEE / GPU-CC (H100-CC / TDX / SEV-SNP) attests a measured
    /// binary ran in an enclave; combined with the weight hash it is the strongest model-authenticity
    /// anchor. PoI Tier 2 (premium). Stands on the attestation itself, not on accrued reputation.
    HardwareAttested,
}

impl AuthenticityTier {
    /// Every tier, ascending. The canonical order for indexing / comparison.
    pub const ALL: [AuthenticityTier; 4] = [
        AuthenticityTier::SelfReported,
        AuthenticityTier::ReputationRedundant,
        AuthenticityTier::EnergyFingerprintPossession,
        AuthenticityTier::HardwareAttested,
    ];

    /// This tier's position in [`ALL`](Self::ALL) (`SelfReported = 0 … HardwareAttested = 3`).
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&t| t == self).expect("every AuthenticityTier is in ALL") as u8
    }

    /// The tier at `index`, saturating at [`HardwareAttested`](Self::HardwareAttested) for an
    /// out-of-range index (so index arithmetic can be clamped without a bounds panic).
    pub fn from_index(index: u8) -> AuthenticityTier {
        *Self::ALL.get(index as usize).unwrap_or(&AuthenticityTier::HardwareAttested)
    }

    /// A stable short label for display / telemetry / the disclosed assurance badge.
    pub fn label(self) -> &'static str {
        match self {
            AuthenticityTier::SelfReported => "A0-self-reported",
            AuthenticityTier::ReputationRedundant => "A1-reputation-redundant",
            AuthenticityTier::EnergyFingerprintPossession => "A2-energy-fingerprint-possession",
            AuthenticityTier::HardwareAttested => "A3-hardware-attested",
        }
    }
}

/// **Privacy** — *can the provider's adversarial owner read my prompt?* darkbloom's axis. OpenHydra's
/// BYO-external engine (plaintext crosses an IPC boundary the owner controls) is inherently `P0`;
/// the higher tiers require the engine inside an attested boundary and slot in as an optional tier,
/// never the default. Disclosed in this increment; its verifier lands with the privacy tiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PrivacyTier {
    /// **None.** BYO-external engine — the provider's owner can read the prompt. The default.
    None,
    /// **Attested agent.** A genuine unmodified OpenHydra agent binary, bound to the live connection
    /// (a platform-capability-as-identity signal, à la darkbloom's APNs trick). Raises the bar on
    /// the *agent*, but the BYO engine still sees plaintext.
    AttestedAgent,
    /// **Attested in-process engine.** A first-party in-process engine (Candle/Burn) inside a
    /// hardened, attested boundary (SIP + Hardened-Runtime, GPU-CC / TDX) — the owner cannot read
    /// prompt or activations. The darkbloom guarantee, offered as a managed-engine tier.
    AttestedInProcessEngine,
}

impl PrivacyTier {
    /// Every tier, ascending.
    pub const ALL: [PrivacyTier; 3] =
        [PrivacyTier::None, PrivacyTier::AttestedAgent, PrivacyTier::AttestedInProcessEngine];

    /// This tier's position in [`ALL`](Self::ALL) (`None = 0 … AttestedInProcessEngine = 2`).
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&t| t == self).expect("every PrivacyTier is in ALL") as u8
    }

    /// The tier at `index`, saturating at the top for an out-of-range index.
    pub fn from_index(index: u8) -> PrivacyTier {
        *Self::ALL.get(index as usize).unwrap_or(&PrivacyTier::AttestedInProcessEngine)
    }

    /// A stable short label for display / telemetry.
    pub fn label(self) -> &'static str {
        match self {
            PrivacyTier::None => "P0-none",
            PrivacyTier::AttestedAgent => "P1-attested-agent",
            PrivacyTier::AttestedInProcessEngine => "P2-attested-in-process-engine",
        }
    }
}

/// **Settlement** — *do I get paid, and are cheats punished?* AntSeed's axis. OpenHydra keeps custody
/// with the user (non-custodial USDC on Base), and — uniquely — can slash on *model substitution*
/// because its authenticity layer makes it detectable. Disclosed in this increment; its wiring lands
/// at the payout layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SettlementTier {
    /// **Credits only.** The give/take reputation-gated credit ledger; no external value moves.
    CreditsOnly,
    /// **Non-custodial voucher.** A signed non-custodial USDC voucher (x402 / deferred, batched net
    /// redemption) — value settles without an intermediary holding it.
    NonCustodialVoucher,
    /// **Staked + slashable.** The provider stakes; a proven model substitution (authenticity-axis
    /// detection) triggers clawback/slash — the fault AntSeed's non-delivery-only slashing can't
    /// punish.
    StakedSlashable,
}

impl SettlementTier {
    /// Every tier, ascending.
    pub const ALL: [SettlementTier; 3] = [
        SettlementTier::CreditsOnly,
        SettlementTier::NonCustodialVoucher,
        SettlementTier::StakedSlashable,
    ];

    /// This tier's position in [`ALL`](Self::ALL) (`CreditsOnly = 0 … StakedSlashable = 2`).
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&t| t == self).expect("every SettlementTier is in ALL") as u8
    }

    /// The tier at `index`, saturating at the top for an out-of-range index.
    pub fn from_index(index: u8) -> SettlementTier {
        *Self::ALL.get(index as usize).unwrap_or(&SettlementTier::StakedSlashable)
    }

    /// A stable short label for display / telemetry.
    pub fn label(self) -> &'static str {
        match self {
            SettlementTier::CreditsOnly => "S0-credits-only",
            SettlementTier::NonCustodialVoucher => "S1-non-custodial-voucher",
            SettlementTier::StakedSlashable => "S2-staked-slashable",
        }
    }
}

/// The assurance a serve *achieved* — one independent level per axis. This is what a provider
/// discloses and a verifier confirms, and what [`AssurancePolicy::admits`] is checked against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AssuranceTier {
    pub authenticity: AuthenticityTier,
    pub privacy: PrivacyTier,
    pub settlement: SettlementTier,
}

impl AssuranceTier {
    /// The floor on every axis (`A0 / P0 / S0`) — an unverified BYO serve with credits-only
    /// settlement. What a serve is treated as before any evidence lifts it.
    pub const BASELINE: AssuranceTier = AssuranceTier {
        authenticity: AuthenticityTier::SelfReported,
        privacy: PrivacyTier::None,
        settlement: SettlementTier::CreditsOnly,
    };
}

/// How a buyer's policy behaves when a candidate does **not** meet the required floors. The predicate
/// [`AssurancePolicy::admits`] is the same in both modes; the mode tells the *caller* (the routing
/// filter / serve gate) what to do with a `false`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PolicyMode {
    /// Refuse the candidate outright (a hard 502-equivalent) — AntSeed's `--require-verifier`. The
    /// buyer would rather not be served than be served below the bar.
    HardGate,
    /// Route anyway, but surface the achieved tier (and log the shortfall). Buyer-sovereign default:
    /// verify-but-route.
    #[default]
    VerifyButRoute,
}

/// A buyer's required assurance — a per-axis floor plus a [`PolicyMode`]. The consumer sets it; the
/// routing filter and serve gate consult [`admits`](Self::admits).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AssurancePolicy {
    pub min_authenticity: AuthenticityTier,
    pub min_privacy: PrivacyTier,
    pub min_settlement: SettlementTier,
    pub mode: PolicyMode,
}

impl AssurancePolicy {
    /// The open policy: floors on the ground (`A0 / P0 / S0`) in verify-but-route mode — it
    /// [`admits`](Self::admits) every candidate. The default when a buyer expresses no requirement.
    pub const OPEN: AssurancePolicy = AssurancePolicy {
        min_authenticity: AuthenticityTier::SelfReported,
        min_privacy: PrivacyTier::None,
        min_settlement: SettlementTier::CreditsOnly,
        mode: PolicyMode::VerifyButRoute,
    };

    /// Whether an `achieved` triple clears the policy's floors on **every** axis. This is the single
    /// predicate the routing filter and the serve gate share; the [`mode`](Self::mode) governs the
    /// caller's response to a `false`, not this check.
    pub fn admits(&self, achieved: &AssuranceTier) -> bool {
        achieved.authenticity >= self.min_authenticity
            && achieved.privacy >= self.min_privacy
            && achieved.settlement >= self.min_settlement
    }
}

impl Default for AssurancePolicy {
    fn default() -> Self {
        Self::OPEN
    }
}

/// Which Proof-of-Inference methods a serve satisfied. A plain set of independent booleans (no wire
/// format, no external dep) — the caller sets the ones its verifier confirmed. `Default` = none set,
/// which resolves to the [`SelfReported`](AuthenticityTier::SelfReported) floor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MethodFlags {
    /// Weight-possession proof — e.g. the Ollama `sha256-…` blob digest matches the registered
    /// artifact for the canonical id (an A2-qualifying hard method).
    pub possession: bool,
    /// Top-k logprob commitment consistent with the claimed model's distribution — a statistical
    /// fingerprint (A2-qualifying).
    pub logprob_commitment: bool,
    /// The serve ran in deterministic mode (temp=0, fixed seed, pinned quant). A *precondition* that
    /// makes the fingerprint / redundant-exec checks meaningful — tracked, but not A2-qualifying on
    /// its own.
    pub deterministic_serve: bool,
    /// The energy keystone — measured energy/token matches the per-(model, hw) profile (a
    /// conservation-law axis; A2-qualifying).
    pub energy_keystone: bool,
    /// Hardware attestation (TEE / GPU-CC) chained to a manufacturer key — the A3 anchor.
    pub hardware_attested: bool,
}

impl MethodFlags {
    /// Whether at least one **hard** Proof-of-Inference method (possession / logprob / energy) is
    /// present — the A2 unlock. Deterministic-serve alone does not count (it is a precondition, not
    /// a proof).
    pub fn has_hard_method(self) -> bool {
        self.possession || self.logprob_commitment || self.energy_keystone
    }
}

/// The accumulated authenticity evidence for one `(peer, model)`, as gathered from the existing
/// reputation + audit machinery ([`crate::verify::ReputationTracker`], [`crate::award::AuditStats`])
/// and the per-serve method flags.
#[derive(Debug, Clone, Copy)]
pub struct AuthenticityEvidence {
    /// This `(peer, model)`'s reputation in `[0, 100]` ([`crate::verify::ReputationTracker::score_at`]).
    pub reputation: f64,
    /// Recency-weighted audit pass-rate in `[0, 1]` ([`crate::award::AuditStats::pass_rate`]).
    pub audit_pass_rate: f64,
    /// Count of recent passing temp=0 audits ([`crate::award::AuditStats::recent_passes`]).
    pub passing_audits: u32,
    /// Which PoI methods this serve satisfied.
    pub methods: MethodFlags,
    /// The most recent redundant-exec outcome for this `(peer, model)`, if any (derived from a
    /// [`crate::verify::RedundantVerdict`] via `outcome_for`). A live
    /// [`Failed`](crate::verify::VerificationOutcome::Failed) — a clear temp=0 divergence — forces
    /// the authenticity floor regardless of historical stats: a provider caught substituting *now*
    /// is unverified *now*.
    pub latest_outcome: Option<VerificationOutcome>,
}

/// Resolve accumulated evidence to an [`AuthenticityTier`]. Pure; the shared mapping both a verifier
/// and the credit gate use, so they never disagree on what a tier means.
///
/// * A live `Failed` outcome → [`SelfReported`](AuthenticityTier::SelfReported) (immediate override).
/// * Hardware attestation → [`HardwareAttested`](AuthenticityTier::HardwareAttested) (stands on the
///   attestation; independent of accrued reputation).
/// * Otherwise, when the reputation *and* pass-rate gates ([`R_MIN`] / [`A_MIN`]) are met on real
///   passing audits: [`EnergyFingerprintPossession`](AuthenticityTier::EnergyFingerprintPossession)
///   if a hard PoI method is present, else
///   [`ReputationRedundant`](AuthenticityTier::ReputationRedundant).
/// * Failing those gates → the floor.
pub fn authenticity_tier_from(
    reputation: f64,
    audit_pass_rate: f64,
    passing_audits: u32,
    methods: MethodFlags,
) -> AuthenticityTier {
    // The strongest anchor stands alone.
    if methods.hardware_attested {
        return AuthenticityTier::HardwareAttested;
    }
    // Reputation + redundant-exec agreement, both gated like the credit-tier ascension.
    let redundant_ok = reputation >= R_MIN && audit_pass_rate >= A_MIN && passing_audits > 0;
    if !redundant_ok {
        return AuthenticityTier::SelfReported;
    }
    if methods.has_hard_method() {
        AuthenticityTier::EnergyFingerprintPossession
    } else {
        AuthenticityTier::ReputationRedundant
    }
}

/// Compose authenticity evidence + the *disclosed* privacy / settlement levels into the achieved
/// [`AssuranceTier`]. The single resolution entry point. A live
/// [`Failed`](VerificationOutcome::Failed) outcome pins authenticity to the floor.
///
/// In this increment Privacy and Settlement are passed through as disclosed (their verifiers land
/// later); only Authenticity is resolved from evidence here.
pub fn resolve_assurance(
    auth: &AuthenticityEvidence,
    disclosed_privacy: PrivacyTier,
    disclosed_settlement: SettlementTier,
) -> AssuranceTier {
    let authenticity = if matches!(auth.latest_outcome, Some(VerificationOutcome::Failed)) {
        AuthenticityTier::SelfReported
    } else {
        authenticity_tier_from(auth.reputation, auth.audit_pass_rate, auth.passing_audits, auth.methods)
    };
    AssuranceTier { authenticity, privacy: disclosed_privacy, settlement: disclosed_settlement }
}

/// The highest [`CostTier`] a `(peer, model)` may **earn at** given the authenticity tier it has
/// reached — the assurance-level gate that resolves credit-tiering review finding **H1** (per the
/// PoI addendum: *the assurance tier gates the credit-tier*). Layered *above* the fine-grained
/// per-audit ascension in [`crate::award::max_awardable_tier`]: authenticity caps the ceiling, the
/// award curve steps up to it.
///
/// * `A0` (self-reported) → [`FLOOR_TIER`]: unverified earns only at the floor, whatever the claim.
/// * `A1` (reputation + agreement) → capped at [`Medium`](CostTier::Medium): agreement among cheap
///   models can be colluded, so it does not by itself unlock the expensive tiers.
/// * `A2` / `A3` (a hard PoI method / hardware attestation) → no assurance cap; `claimed` governs.
///
/// Monotonic in the authenticity tier; never exceeds `claimed`. Pure. **Documented, not wired** —
/// no live credit path calls this yet; the gate belongs at the payout layer (H2 remediation).
pub fn max_cost_tier_for_authenticity(authenticity: AuthenticityTier, claimed: CostTier) -> CostTier {
    let cap = match authenticity {
        AuthenticityTier::SelfReported => FLOOR_TIER,
        AuthenticityTier::ReputationRedundant => CostTier::Medium,
        AuthenticityTier::EnergyFingerprintPossession | AuthenticityTier::HardwareAttested => {
            CostTier::Mega
        }
    };
    claimed.min(cap)
}

// ── Verifier plugin seam (UNIFIED_TRUST_STACK Increment 3, AntSeed-borrowed) ──
//
// The detection side of the stack. A `Verifier` turns evidence into an authenticity verdict; the
// enforcement side (the credit-tiering `award` primitives, the routing `AssurancePolicy`) consumes
// the tier. Verifiers are **curated and version-pinned** — an id embeds an explicit version
// (`name@N`), never a floating "latest", so a verifier's behaviour is pinned to its id. Today there
// is exactly one (redundant-exec authenticity); TEE-attestation / possession-with-anchor verifiers
// plug into the same seam as they become buildable.

/// A verdict from one [`Verifier`]: the authenticity tier it establishes, plus whether that verdict
/// is **determinate** (settled from sufficient evidence) and so may be cached, versus transient /
/// inconclusive (insufficient evidence, a probe error) which must **never** be cached — a transient
/// "unknown" must not be frozen into a lasting low tier (AntSeed's "verdict cache that refuses to
/// cache transient failures").
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifierVerdict {
    /// The pinned id of the verifier that produced this verdict (`name@N`).
    pub verifier_id: String,
    /// The authenticity tier the verifier established.
    pub authenticity: AuthenticityTier,
    /// True when the verdict is settled and may be cached; false for a transient / inconclusive
    /// result (see [`cacheable`](Self::cacheable)).
    pub determinate: bool,
}

impl VerifierVerdict {
    /// Whether this verdict may be cached. A transient / inconclusive verdict returns `false` — the
    /// cache must re-derive it next time rather than freeze an "unknown" into a lasting tier.
    pub fn cacheable(&self) -> bool {
        self.determinate
    }
}

/// A detection plugin: turns accumulated [`AuthenticityEvidence`] into a [`VerifierVerdict`]. Pure
/// (no I/O in this trait) and object-safe, so a [`VerifierRegistry`] can hold a curated set behind
/// `dyn`. The buyer's [`AssurancePolicy`] decides what to *do* with the resulting tier; the verifier
/// only *establishes* it.
pub trait Verifier {
    /// This verifier's curated, version-pinned id (`name@N`). Stable for the life of the version.
    fn id(&self) -> &str;
    /// Establish an authenticity verdict from the evidence.
    fn verify(&self, evidence: &AuthenticityEvidence) -> VerifierVerdict;
}

/// The redundant-execution + reputation authenticity verifier — the one detection mechanism built
/// today (it reads the audit history the consumer already accrues). Establishes `A0`/`A1`; a hard
/// PoI method in the evidence lifts it to `A2`, and a hardware-attested flag to `A3`, via the shared
/// [`authenticity_tier_from`]. A live [`Failed`](VerificationOutcome::Failed) outcome pins the floor.
#[derive(Debug, Clone, Copy, Default)]
pub struct RedundantExecAuthenticity;

impl RedundantExecAuthenticity {
    /// The curated, version-pinned id.
    pub const ID: &'static str = "redundant-exec-authenticity@1";
}

impl Verifier for RedundantExecAuthenticity {
    fn id(&self) -> &str {
        Self::ID
    }

    fn verify(&self, evidence: &AuthenticityEvidence) -> VerifierVerdict {
        let authenticity = if matches!(evidence.latest_outcome, Some(VerificationOutcome::Failed)) {
            AuthenticityTier::SelfReported
        } else {
            authenticity_tier_from(
                evidence.reputation,
                evidence.audit_pass_rate,
                evidence.passing_audits,
                evidence.methods,
            )
        };
        // Determinate only with real evidence: at least one passing audit, or a decisive latest
        // outcome. With NO evidence the `A0` is a transient "unknown" (an untried `(peer, model)`),
        // not a proven floor — it must not be cached (it will lift as soon as audits accrue).
        let determinate = evidence.passing_audits > 0 || evidence.latest_outcome.is_some();
        VerifierVerdict { verifier_id: Self::ID.to_string(), authenticity, determinate }
    }
}

/// A curated, version-pinned set of [`Verifier`]s. Lookup is by exact pinned id — there is no
/// floating "latest", so which verifier runs is always explicit and reproducible.
pub struct VerifierRegistry {
    verifiers: Vec<Box<dyn Verifier>>,
}

impl VerifierRegistry {
    /// The curated default set shipped today: just the redundant-exec authenticity verifier. New
    /// verifiers (possession-with-anchor, TEE attestation) are added here as they become buildable.
    pub fn curated() -> Self {
        Self { verifiers: vec![Box::new(RedundantExecAuthenticity)] }
    }

    /// The verifier with this exact pinned id, or `None` — an unknown / unpinned id resolves to
    /// nothing rather than silently falling back to some default.
    pub fn get(&self, id: &str) -> Option<&dyn Verifier> {
        self.verifiers.iter().find(|v| v.id() == id).map(|v| v.as_ref())
    }

    /// Every curated verifier's pinned id.
    pub fn ids(&self) -> Vec<&str> {
        self.verifiers.iter().map(|v| v.id()).collect()
    }
}

impl Default for VerifierRegistry {
    fn default() -> Self {
        Self::curated()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── tier ladders: ordering, index round-trip, saturation ──

    #[test]
    fn authenticity_tiers_are_ordered_and_index_round_trips() {
        use AuthenticityTier::*;
        assert!(SelfReported < ReputationRedundant);
        assert!(ReputationRedundant < EnergyFingerprintPossession);
        assert!(EnergyFingerprintPossession < HardwareAttested);
        for (i, t) in AuthenticityTier::ALL.iter().enumerate() {
            assert_eq!(t.index() as usize, i);
            assert_eq!(AuthenticityTier::from_index(i as u8), *t);
        }
        // Out-of-range saturates at the top.
        assert_eq!(AuthenticityTier::from_index(99), HardwareAttested);
    }

    #[test]
    fn privacy_and_settlement_tiers_index_round_trip_and_saturate() {
        for (i, t) in PrivacyTier::ALL.iter().enumerate() {
            assert_eq!(t.index() as usize, i);
            assert_eq!(PrivacyTier::from_index(i as u8), *t);
        }
        assert_eq!(PrivacyTier::from_index(99), PrivacyTier::AttestedInProcessEngine);
        for (i, t) in SettlementTier::ALL.iter().enumerate() {
            assert_eq!(t.index() as usize, i);
            assert_eq!(SettlementTier::from_index(i as u8), *t);
        }
        assert_eq!(SettlementTier::from_index(99), SettlementTier::StakedSlashable);
    }

    #[test]
    fn labels_are_stable_and_distinct() {
        let mut labels: Vec<&str> = AuthenticityTier::ALL.iter().map(|t| t.label()).collect();
        labels.extend(PrivacyTier::ALL.iter().map(|t| t.label()));
        labels.extend(SettlementTier::ALL.iter().map(|t| t.label()));
        let mut deduped = labels.clone();
        deduped.sort_unstable();
        deduped.dedup();
        assert_eq!(labels.len(), deduped.len(), "every tier label is distinct");
    }

    // ── AssurancePolicy::admits — the routing/serve predicate ──

    fn achieved(a: AuthenticityTier, p: PrivacyTier, s: SettlementTier) -> AssuranceTier {
        AssuranceTier { authenticity: a, privacy: p, settlement: s }
    }

    #[test]
    fn open_policy_admits_the_baseline() {
        assert!(AssurancePolicy::OPEN.admits(&AssuranceTier::BASELINE));
        assert_eq!(AssurancePolicy::default(), AssurancePolicy::OPEN);
    }

    #[test]
    fn admits_requires_every_axis_to_clear_its_floor() {
        use AuthenticityTier::*;
        use PrivacyTier::*;
        use SettlementTier::*;
        let policy = AssurancePolicy {
            min_authenticity: EnergyFingerprintPossession,
            min_privacy: AttestedAgent,
            min_settlement: NonCustodialVoucher,
            mode: PolicyMode::HardGate,
        };
        // Exactly meeting every floor is admitted; a stronger tier on any axis still admitted.
        assert!(policy.admits(&achieved(EnergyFingerprintPossession, AttestedAgent, NonCustodialVoucher)));
        assert!(policy.admits(&achieved(HardwareAttested, AttestedInProcessEngine, StakedSlashable)));
        // A shortfall on any single axis fails the whole predicate.
        assert!(!policy.admits(&achieved(ReputationRedundant, AttestedAgent, NonCustodialVoucher)));
        assert!(!policy.admits(&achieved(EnergyFingerprintPossession, None, NonCustodialVoucher)));
        assert!(!policy.admits(&achieved(EnergyFingerprintPossession, AttestedAgent, CreditsOnly)));
    }

    #[test]
    fn admits_is_mode_independent() {
        // The predicate is identical in both modes; mode only tells the caller what to do with false.
        let hard =
            AssurancePolicy { mode: PolicyMode::HardGate, ..AssurancePolicy::OPEN };
        let soft =
            AssurancePolicy { mode: PolicyMode::VerifyButRoute, ..AssurancePolicy::OPEN };
        let a = achieved(
            AuthenticityTier::ReputationRedundant,
            PrivacyTier::None,
            SettlementTier::CreditsOnly,
        );
        assert_eq!(hard.admits(&a), soft.admits(&a));
    }

    // ── authenticity_tier_from — every boundary ──

    fn flags_none() -> MethodFlags {
        MethodFlags::default()
    }
    fn flags_possession() -> MethodFlags {
        MethodFlags { possession: true, ..MethodFlags::default() }
    }

    #[test]
    fn self_reported_when_gates_unmet() {
        // No audits at all → floor.
        assert_eq!(
            authenticity_tier_from(50.0, 1.0, 0, flags_none()),
            AuthenticityTier::SelfReported
        );
        // Reputation below R_MIN → floor, even with a perfect pass-rate and many audits.
        assert_eq!(
            authenticity_tier_from(R_MIN - 0.01, 1.0, 99, flags_possession()),
            AuthenticityTier::SelfReported
        );
        // Pass-rate below A_MIN → floor.
        assert_eq!(
            authenticity_tier_from(90.0, A_MIN - 0.01, 99, flags_possession()),
            AuthenticityTier::SelfReported
        );
    }

    #[test]
    fn a1_at_the_exact_gate_edges_without_a_hard_method() {
        // Exactly at both gates with a real passing audit and no hard method → A1.
        assert_eq!(
            authenticity_tier_from(R_MIN, A_MIN, 1, flags_none()),
            AuthenticityTier::ReputationRedundant
        );
    }

    #[test]
    fn a2_when_a_hard_method_accompanies_a_clearing_record() {
        assert_eq!(
            authenticity_tier_from(R_MIN, A_MIN, 1, flags_possession()),
            AuthenticityTier::EnergyFingerprintPossession
        );
        // Logprob or energy also unlock A2.
        let logprob = MethodFlags { logprob_commitment: true, ..MethodFlags::default() };
        assert_eq!(
            authenticity_tier_from(80.0, 0.95, 5, logprob),
            AuthenticityTier::EnergyFingerprintPossession
        );
    }

    #[test]
    fn deterministic_serve_alone_does_not_reach_a2() {
        // Deterministic mode is a precondition, not a proof — it stays A1 without a hard method.
        let det = MethodFlags { deterministic_serve: true, ..MethodFlags::default() };
        assert!(!det.has_hard_method());
        assert_eq!(
            authenticity_tier_from(80.0, 0.95, 5, det),
            AuthenticityTier::ReputationRedundant
        );
    }

    #[test]
    fn hardware_attestation_stands_alone() {
        // Even with zero reputation / no audits, a hardware attestation is A3.
        let hw = MethodFlags { hardware_attested: true, ..MethodFlags::default() };
        assert_eq!(authenticity_tier_from(0.0, 0.0, 0, hw), AuthenticityTier::HardwareAttested);
    }

    // ── resolve_assurance — composition + the live-fail override ──

    fn evidence(rep: f64, rate: f64, n: u32, m: MethodFlags, o: Option<VerificationOutcome>) -> AuthenticityEvidence {
        AuthenticityEvidence {
            reputation: rep,
            audit_pass_rate: rate,
            passing_audits: n,
            methods: m,
            latest_outcome: o,
        }
    }

    #[test]
    fn resolve_passes_through_disclosed_privacy_and_settlement() {
        let ev = evidence(R_MIN, A_MIN, 1, flags_possession(), None);
        let got = resolve_assurance(&ev, PrivacyTier::AttestedAgent, SettlementTier::NonCustodialVoucher);
        assert_eq!(got.authenticity, AuthenticityTier::EnergyFingerprintPossession);
        assert_eq!(got.privacy, PrivacyTier::AttestedAgent);
        assert_eq!(got.settlement, SettlementTier::NonCustodialVoucher);
    }

    #[test]
    fn a_live_failed_outcome_pins_authenticity_to_the_floor() {
        // Strong history + a hard method, but caught diverging on the latest audit → A0 now.
        let ev = evidence(95.0, 0.99, 50, flags_possession(), Some(VerificationOutcome::Failed));
        let got = resolve_assurance(&ev, PrivacyTier::None, SettlementTier::CreditsOnly);
        assert_eq!(got.authenticity, AuthenticityTier::SelfReported);
    }

    #[test]
    fn a_honored_or_rejected_outcome_does_not_override_the_evidence_mapping() {
        // Only a clear Failed overrides; Honored leaves the normal mapping intact.
        let ev = evidence(R_MIN, A_MIN, 1, flags_none(), Some(VerificationOutcome::Honored));
        let got = resolve_assurance(&ev, PrivacyTier::None, SettlementTier::CreditsOnly);
        assert_eq!(got.authenticity, AuthenticityTier::ReputationRedundant);
    }

    // ── max_cost_tier_for_authenticity — the H1 gate ──

    #[test]
    fn a0_earns_only_at_the_floor_regardless_of_claim() {
        for claimed in CostTier::ALL {
            assert_eq!(
                max_cost_tier_for_authenticity(AuthenticityTier::SelfReported, claimed),
                FLOOR_TIER
            );
        }
    }

    #[test]
    fn a1_caps_at_medium() {
        assert_eq!(
            max_cost_tier_for_authenticity(AuthenticityTier::ReputationRedundant, CostTier::Mega),
            CostTier::Medium
        );
        // Never exceeds the claim: a small model at A1 stays small.
        assert_eq!(
            max_cost_tier_for_authenticity(AuthenticityTier::ReputationRedundant, CostTier::Small),
            CostTier::Small
        );
    }

    #[test]
    fn a2_and_a3_lift_the_assurance_cap_to_the_claim() {
        for a in [AuthenticityTier::EnergyFingerprintPossession, AuthenticityTier::HardwareAttested] {
            for claimed in CostTier::ALL {
                assert_eq!(max_cost_tier_for_authenticity(a, claimed), claimed);
            }
        }
    }

    #[test]
    fn the_gate_is_monotonic_in_authenticity() {
        // For any claim, a higher authenticity tier never lowers the earnable cost-tier.
        for claimed in CostTier::ALL {
            let caps: Vec<CostTier> = AuthenticityTier::ALL
                .iter()
                .map(|&a| max_cost_tier_for_authenticity(a, claimed))
                .collect();
            for w in caps.windows(2) {
                assert!(w[1] >= w[0], "cap must not decrease as authenticity rises");
            }
        }
    }

    // ── verifier plugin seam ──

    #[test]
    fn redundant_exec_verifier_matches_the_shared_resolution() {
        // The verifier and the routing resolver share authenticity_tier_from, so they agree.
        let v = RedundantExecAuthenticity;
        let ev = evidence(R_MIN, A_MIN, 1, flags_none(), None);
        let verdict = v.verify(&ev);
        assert_eq!(verdict.verifier_id, "redundant-exec-authenticity@1");
        assert_eq!(verdict.authenticity, AuthenticityTier::ReputationRedundant);
        assert_eq!(
            verdict.authenticity,
            authenticity_tier_from(ev.reputation, ev.audit_pass_rate, ev.passing_audits, ev.methods)
        );
    }

    #[test]
    fn a_no_evidence_verdict_is_a0_and_not_cacheable() {
        // A fresh (peer,model): A0, but a *transient* unknown — it must not be cached (it lifts as
        // soon as audits accrue).
        let verdict = RedundantExecAuthenticity.verify(&evidence(50.0, 1.0, 0, flags_none(), None));
        assert_eq!(verdict.authenticity, AuthenticityTier::SelfReported);
        assert!(!verdict.cacheable(), "a no-evidence A0 is transient, never cached");
    }

    #[test]
    fn a_verdict_with_real_evidence_is_determinate_and_cacheable() {
        let with_audits = RedundantExecAuthenticity.verify(&evidence(R_MIN, A_MIN, 3, flags_none(), None));
        assert!(with_audits.cacheable(), "passing audits make the verdict determinate");
        // A live Failed is also determinate (a settled divergence), pinned to the floor.
        let failed = RedundantExecAuthenticity
            .verify(&evidence(95.0, 0.99, 50, flags_none(), Some(VerificationOutcome::Failed)));
        assert_eq!(failed.authenticity, AuthenticityTier::SelfReported);
        assert!(failed.cacheable());
    }

    #[test]
    fn registry_is_curated_and_version_pinned() {
        let reg = VerifierRegistry::curated();
        assert_eq!(reg.ids(), vec!["redundant-exec-authenticity@1"]);
        // Ids embed an explicit version; there is no floating "latest".
        assert!(reg.ids().iter().all(|id| id.contains('@') && !id.contains("latest")));
        assert!(reg.get("redundant-exec-authenticity@1").is_some());
        // An unknown / unpinned id resolves to nothing (no silent fallback).
        assert!(reg.get("redundant-exec-authenticity").is_none());
        assert!(reg.get("nonexistent@1").is_none());
    }
}
