// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! On-chain settlement claims for the crypto rail (PAYMENTS_PLAN_v1, crypto sequence step 3), on
//! Solana (native Ed25519) and Base (secp256k1 + EIP-712).
//!
//! The co-signed voucher receipt ([`crate::receipts::CoSignedReceipt`] with a `cumulative_amount`)
//! is the off-chain record; what a chain settles is a separate **claim** the consumer signs for that
//! chain's program ([`SettlementClaim`], format v6): *pay provider P up to `cumulative_amount` for
//! model M out of consumer C's escrow for P, until `expires_at`*, bound to one deployment (its
//! [`ChainDomain`]). The program pays only the provider's own payout account (Solana: its
//! [`PayoutBinding`]; Base: the channel's payout), so the claim names no payout: the consumer needn't
//! know it, and naming another one would gain nothing.
//!
//! - **Solana**: the consumer's OpenHydra Ed25519 identity signs [`SettlementClaim::solana_preimage`]
//!   (an account address *is* an Ed25519 key, verified natively by the program). The domain is the
//!   program id and a cluster tag (its genesis hash), so a claim made for devnet or for another
//!   deployment doesn't settle here.
//! - **Base**: the consumer's EVM wallet (the escrow owner) signs the EIP-712
//!   [`SettlementClaim::eip712_digest`] ([`sign_evm_claim`]); the domain is the chain id and contract.
//!
//! A provider checks a claim against the receipt it co-signs ([`verify_claim`]) and keeps the latest
//! per pair to submit. Pure: no I/O, no chain SDK. Nothing here moves value — the programs
//! (`onchain/`) and their legal gates (non-custodial opinion, audit, geofencing) do.

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use k256::ecdsa::{
    RecoveryId, Signature as EcdsaSignature, SigningKey as EcdsaSigningKey, VerifyingKey as EcdsaVerifyingKey,
};
use sha3::{Digest, Keccak256};

use crate::receipts::{verify_receipt, CoSignedReceipt};

/// keccak-256 (Ethereum's hash) of `data`.
fn keccak256(data: &[u8]) -> [u8; 32] {
    let mut h = Keccak256::new();
    h.update(data);
    h.finalize().into()
}

/// Domain separator for a Solana [`PayoutBinding`] signature (v2: bound to a deployment).
pub const PAYOUT_BINDING_DOMAIN: &[u8] = b"openhydra/solana-payout/v2";

/// Domain separator for a Solana [`SettlementClaim`] preimage (v2: claim format v6).
pub const SETTLEMENT_CLAIM_DOMAIN: &[u8] = b"openhydra/solana-settle/v2";

/// How long a claim a consumer signs stays settleable. A newer claim (higher cumulative) supersedes
/// it, so this only matters once a consumer stops using a provider: the provider then has this long to
/// submit the last one. Longer than any program's refund delay, so a claim outlives an exit window.
pub const CLAIM_LIFETIME_SECS: u64 = 30 * 24 * 60 * 60;

/// The least validity left that a provider accepts on a claim, so it has time to submit it.
pub const MIN_CLAIM_LIFETIME_SECS: u64 = 7 * 24 * 60 * 60;

/// Why a settlement primitive was rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SettlementError {
    /// The payout binding's signature does not verify against the identity key.
    BadBindingSignature,
    /// The receipt carries no `cumulative_amount` — it is not a crypto-rail voucher, so there is
    /// nothing to settle.
    NotAVoucher,
    /// The receipt's own signatures do not verify (tampered or corrupt).
    BadReceipt,
    /// The claim's signature doesn't verify (Solana: by the consumer; Base: recovers no signer).
    BadClaimSignature,
    /// The claim expires too soon to be submitted (see [`MIN_CLAIM_LIFETIME_SECS`]).
    ClaimExpiresTooSoon,
}

/// A settlement chain OpenHydra can pay out on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Chain {
    /// Solana — native Ed25519 accounts + on-chain Ed25519 verification.
    Solana,
    /// Base (EVM L2) — the consumer's EVM wallet signs an EIP-712 claim (`ecrecover`).
    Base,
}

impl Chain {
    /// A stable lower-case label for display / telemetry / config.
    pub fn label(self) -> &'static str {
        match self {
            Chain::Solana => "solana",
            Chain::Base => "base",
        }
    }

    /// Whether settling needs a key besides the OpenHydra Ed25519 identity (Base: an EVM wallet).
    pub fn requires_dual_sign(self) -> bool {
        matches!(self, Chain::Base)
    }
}

/// One Solana deployment: the program and the cluster it runs on (the cluster's genesis hash; the
/// program keeps the same tag in its config, set at initialize by its upgrade authority).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolanaDomain {
    pub program_id: [u8; 32],
    pub cluster: [u8; 32],
}

/// EIP-712 domain parameters for the Base settlement contract (its chain + address).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Eip712Domain {
    pub chain_id: u64,
    pub verifying_contract: [u8; 20],
}

/// The deployment a claim is for: what binds a signature to one chain, program and cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainDomain {
    Solana(SolanaDomain),
    Base(Eip712Domain),
}

impl ChainDomain {
    pub fn chain(&self) -> Chain {
        match self {
            ChainDomain::Solana(_) => Chain::Solana,
            ChainDomain::Base(_) => Chain::Base,
        }
    }

    /// A short fingerprint of the deployment (hex, 16 bytes of SHA-256 over the chain and its domain):
    /// how a consumer says up front which deployment it will settle on, so a provider can refuse a
    /// serve it couldn't be paid for.
    pub fn fingerprint(&self) -> String {
        use sha2::{Digest as _, Sha256};
        let mut h = Sha256::new();
        match self {
            ChainDomain::Solana(d) => {
                h.update(b"solana");
                h.update(d.program_id);
                h.update(d.cluster);
            }
            ChainDomain::Base(d) => {
                h.update(b"base");
                h.update(d.chain_id.to_le_bytes());
                h.update(d.verifying_contract);
            }
        }
        hex::encode(&h.finalize()[..16])
    }

    /// What the consumer signs for `claim` on this deployment: the Solana preimage (signed with
    /// Ed25519), or the 32-byte EIP-712 digest (signed with secp256k1).
    pub fn claim_message(&self, claim: &SettlementClaim) -> Vec<u8> {
        match self {
            ChainDomain::Solana(d) => claim.solana_preimage(d),
            ChainDomain::Base(d) => claim.eip712_digest(d).to_vec(),
        }
    }
}

// ── Solana payout binding ──────────────────────────────────────────────────

/// A provider's authorization of the Solana account that receives its payouts on one deployment. The
/// provider's OpenHydra Ed25519 identity (which *earns* the voucher) signs the chosen
/// `payout_address`, so the program can prove the same entity owns both. `payout_address ==
/// identity.as_bytes()` pays the identity account itself. `ts_unix_ms` orders rotations — the program
/// keeps the latest, so an old binding can't be replayed to redirect funds backward.
#[derive(Debug, Clone)]
pub struct PayoutBinding {
    /// The provider's OpenHydra identity — the voucher payee.
    pub identity: VerifyingKey,
    /// The Solana account to receive payout (a 32-byte Ed25519 public key).
    pub payout_address: [u8; 32],
    /// When this binding was made (unix ms) — later supersedes earlier for the same identity.
    pub ts_unix_ms: u64,
    /// The identity's Ed25519 signature over [`binding_message`].
    pub signature: Signature,
}

/// The bytes an identity signs to authorize a payout address on `domain`:
/// `domain_sep ‖ program_id ‖ cluster ‖ payout_address ‖ ts (LE)`.
pub fn binding_message(domain: &SolanaDomain, payout_address: &[u8; 32], ts_unix_ms: u64) -> Vec<u8> {
    let mut b = Vec::with_capacity(PAYOUT_BINDING_DOMAIN.len() + 32 * 3 + 8);
    b.extend_from_slice(PAYOUT_BINDING_DOMAIN);
    b.extend_from_slice(&domain.program_id);
    b.extend_from_slice(&domain.cluster);
    b.extend_from_slice(payout_address);
    b.extend_from_slice(&ts_unix_ms.to_le_bytes());
    b
}

/// Sign a [`PayoutBinding`] for `domain`. Pass the identity's own public-key bytes as
/// `payout_address` to be paid at the identity account.
pub fn sign_payout_binding(
    identity_key: &SigningKey,
    domain: &SolanaDomain,
    payout_address: [u8; 32],
    ts_unix_ms: u64,
) -> PayoutBinding {
    let signature = identity_key.sign(&binding_message(domain, &payout_address, ts_unix_ms));
    PayoutBinding { identity: identity_key.verifying_key(), payout_address, ts_unix_ms, signature }
}

/// Whether a payout binding's signature verifies for `domain` — the named identity authorized this
/// payout address, at this time, on this deployment.
pub fn verify_payout_binding(binding: &PayoutBinding, domain: &SolanaDomain) -> bool {
    binding
        .identity
        .verify(&binding_message(domain, &binding.payout_address, binding.ts_unix_ms), &binding.signature)
        .is_ok()
}

impl PayoutBinding {
    /// Whether this binding pays the identity's own account (no redirection).
    pub fn is_self_directed(&self) -> bool {
        self.payout_address == self.identity.to_bytes()
    }
}

// ── The claim ──────────────────────────────────────────────────────────────

/// A settlement claim (format v6): pay `provider` up to `cumulative_amount` (gross: provider net +
/// protocol fee, in USDC micro-units) for `model_id` from `consumer`'s escrow, until `expires_at`
/// (unix seconds, the chains' clock). The program releases `cumulative_amount − already_settled`, so
/// the monotonic voucher makes a replay a no-op.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SettlementClaim {
    /// The provider (voucher payee): its OpenHydra Ed25519 identity.
    pub provider: [u8; 32],
    /// The consumer (voucher payer): its OpenHydra Ed25519 identity.
    pub consumer: [u8; 32],
    /// The raw engine handle served (the pair's model bucket).
    pub model_id: String,
    /// The monotonic cumulative amount authorized.
    pub cumulative_amount: u64,
    /// Unix seconds after which the claim no longer settles.
    pub expires_at: u64,
}

/// The claim a co-signed voucher receipt stands for, valid until `expires_at`. Fails closed: the
/// receipt must carry a `cumulative_amount` and its co-signatures must verify.
pub fn claim_from_receipt(receipt: &CoSignedReceipt, expires_at: u64) -> Result<SettlementClaim, SettlementError> {
    let cumulative_amount = receipt.payload.cumulative_amount.ok_or(SettlementError::NotAVoucher)?;
    verify_receipt(receipt).map_err(|_| SettlementError::BadReceipt)?;
    Ok(SettlementClaim {
        provider: receipt.payload.provider.to_bytes(),
        consumer: receipt.payload.consumer.to_bytes(),
        model_id: receipt.payload.model_id.clone(),
        cumulative_amount,
        expires_at,
    })
}

impl SettlementClaim {
    /// The Solana claim preimage the consumer's Ed25519 identity signs and the program verifies:
    /// `domain_sep ‖ program_id ‖ cluster ‖ provider ‖ consumer ‖ cumulative (u64 LE) ‖ expires_at
    /// (u64 LE) ‖ model_len (u32 LE) ‖ model`.
    pub fn solana_preimage(&self, domain: &SolanaDomain) -> Vec<u8> {
        let model = self.model_id.as_bytes();
        let mut b = Vec::with_capacity(SETTLEMENT_CLAIM_DOMAIN.len() + 32 * 4 + 8 + 8 + 4 + model.len());
        b.extend_from_slice(SETTLEMENT_CLAIM_DOMAIN);
        b.extend_from_slice(&domain.program_id);
        b.extend_from_slice(&domain.cluster);
        b.extend_from_slice(&self.provider);
        b.extend_from_slice(&self.consumer);
        b.extend_from_slice(&self.cumulative_amount.to_le_bytes());
        b.extend_from_slice(&self.expires_at.to_le_bytes());
        b.extend_from_slice(&(model.len() as u32).to_le_bytes());
        b.extend_from_slice(model);
        b
    }

    /// The EIP-712 digest the consumer's EVM wallet signs and the Base contract `ecrecover`s:
    /// `keccak256(0x19 0x01 ‖ domainSeparator ‖ hashStruct(claim))`.
    pub fn eip712_digest(&self, domain: &Eip712Domain) -> [u8; 32] {
        let mut sd = Vec::with_capacity(32 * 6);
        sd.extend_from_slice(&keccak256(EIP712_CLAIM_TYPE));
        sd.extend_from_slice(&self.provider);
        sd.extend_from_slice(&self.consumer);
        sd.extend_from_slice(&u256_word(self.cumulative_amount));
        sd.extend_from_slice(&u256_word(self.expires_at));
        sd.extend_from_slice(&keccak256(self.model_id.as_bytes()));
        let hash_struct = keccak256(&sd);

        let mut pre = Vec::with_capacity(2 + 32 + 32);
        pre.extend_from_slice(&[0x19, 0x01]);
        pre.extend_from_slice(&domain.separator());
        pre.extend_from_slice(&hash_struct);
        keccak256(&pre)
    }
}

/// Who signed a verified claim: on Solana the consumer's identity (by construction); on Base the
/// recovered EVM wallet, which the contract requires to own the escrow drawn from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaimSigner {
    Solana,
    Evm([u8; 20]),
}

/// Check a consumer's signature `sig` over `claim` on `domain`, and that the claim leaves at least
/// [`MIN_CLAIM_LIFETIME_SECS`] after `now_secs` to submit it. Solana: a 64-byte Ed25519 signature by
/// `claim.consumer`. Base: a 65-byte `r‖s‖v` signature; the signer is recovered (the contract checks
/// it owns the channel — off-chain, a provider can only record it).
pub fn verify_claim(
    domain: &ChainDomain,
    claim: &SettlementClaim,
    sig: &[u8],
    now_secs: u64,
) -> Result<ClaimSigner, SettlementError> {
    if claim.expires_at < now_secs.saturating_add(MIN_CLAIM_LIFETIME_SECS) {
        return Err(SettlementError::ClaimExpiresTooSoon);
    }
    match domain {
        ChainDomain::Solana(d) => {
            let consumer = VerifyingKey::from_bytes(&claim.consumer).map_err(|_| SettlementError::BadClaimSignature)?;
            let sig: [u8; 64] = sig.try_into().map_err(|_| SettlementError::BadClaimSignature)?;
            consumer
                .verify(&claim.solana_preimage(d), &Signature::from_bytes(&sig))
                .map(|_| ClaimSigner::Solana)
                .map_err(|_| SettlementError::BadClaimSignature)
        }
        ChainDomain::Base(d) => {
            let sig: [u8; 65] = sig.try_into().map_err(|_| SettlementError::BadClaimSignature)?;
            recover_evm_signer(&claim.eip712_digest(d), &sig)
                .map(ClaimSigner::Evm)
                .ok_or(SettlementError::BadClaimSignature)
        }
    }
}

// ── Base / EVM ─────────────────────────────────────────────────────────────

/// Domain separator for an [`EvmPayoutBinding`] signature.
pub const EVM_PAYOUT_BINDING_DOMAIN: &[u8] = b"openhydra/base-payout/v1";

/// The 20-byte EVM address for a secp256k1 verifying key: `keccak256(uncompressed_pubkey[1..])[12..]`.
pub fn evm_address(pubkey: &EcdsaVerifyingKey) -> [u8; 20] {
    let point = pubkey.to_encoded_point(false); // uncompressed: 0x04 ‖ x(32) ‖ y(32)
    let hash = keccak256(&point.as_bytes()[1..]);
    let mut addr = [0u8; 20];
    addr.copy_from_slice(&hash[12..]);
    addr
}

/// A provider's attestation of the **EVM** address that receives its Base payouts, signed by its
/// OpenHydra Ed25519 identity — what a consumer opens a channel to (the contract pays the channel's
/// payout, which only that address can move). `ts_unix_ms` orders rotations.
#[derive(Debug, Clone)]
pub struct EvmPayoutBinding {
    pub identity: VerifyingKey,
    pub evm_address: [u8; 20],
    pub ts_unix_ms: u64,
    pub signature: Signature,
}

/// The bytes an identity signs to authorize an EVM payout address: domain ‖ evm_address ‖ ts (LE).
pub fn evm_binding_message(evm_address: &[u8; 20], ts_unix_ms: u64) -> Vec<u8> {
    let mut b = Vec::with_capacity(EVM_PAYOUT_BINDING_DOMAIN.len() + 20 + 8);
    b.extend_from_slice(EVM_PAYOUT_BINDING_DOMAIN);
    b.extend_from_slice(evm_address);
    b.extend_from_slice(&ts_unix_ms.to_le_bytes());
    b
}

/// Sign an [`EvmPayoutBinding`].
pub fn sign_evm_payout_binding(identity_key: &SigningKey, evm_address: [u8; 20], ts_unix_ms: u64) -> EvmPayoutBinding {
    let signature = identity_key.sign(&evm_binding_message(&evm_address, ts_unix_ms));
    EvmPayoutBinding { identity: identity_key.verifying_key(), evm_address, ts_unix_ms, signature }
}

/// Whether an EVM payout binding's Ed25519 signature verifies.
pub fn verify_evm_payout_binding(binding: &EvmPayoutBinding) -> bool {
    binding
        .identity
        .verify(&evm_binding_message(&binding.evm_address, binding.ts_unix_ms), &binding.signature)
        .is_ok()
}

/// The EIP-712 domain type string (`name` + `version` are fixed for the protocol).
const EIP712_DOMAIN_TYPE: &[u8] =
    b"EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)";
const EIP712_DOMAIN_NAME: &[u8] = b"OpenHydra Settlement";
/// "2": claim format v6 (an expiry; no payout address).
const EIP712_DOMAIN_VERSION: &[u8] = b"2";
/// The claim's EIP-712 type string — the contract MUST hash the identical string, so any change is a
/// breaking, coordinated version bump.
const EIP712_CLAIM_TYPE: &[u8] =
    b"SettlementClaim(bytes32 provider,bytes32 consumer,uint256 cumulativeAmount,uint256 expiresAt,string modelId)";

/// A `u64` as a 32-byte big-endian EVM word.
fn u256_word(x: u64) -> [u8; 32] {
    let mut w = [0u8; 32];
    w[24..].copy_from_slice(&x.to_be_bytes());
    w
}

/// A 20-byte EVM address left-padded to a 32-byte word.
fn address_word(addr: &[u8; 20]) -> [u8; 32] {
    let mut w = [0u8; 32];
    w[12..].copy_from_slice(addr);
    w
}

impl Eip712Domain {
    /// The EIP-712 `domainSeparator`.
    fn separator(&self) -> [u8; 32] {
        let mut enc = Vec::with_capacity(32 * 5);
        enc.extend_from_slice(&keccak256(EIP712_DOMAIN_TYPE));
        enc.extend_from_slice(&keccak256(EIP712_DOMAIN_NAME));
        enc.extend_from_slice(&keccak256(EIP712_DOMAIN_VERSION));
        enc.extend_from_slice(&u256_word(self.chain_id));
        enc.extend_from_slice(&address_word(&self.verifying_contract));
        keccak256(&enc)
    }
}

/// Sign a claim with the consumer's EVM wallet: a 65-byte `r‖s‖v` signature (`v ∈ {27,28}`, low `s`)
/// over its EIP-712 digest — what the contract's `ECDSA.recover` accepts.
pub fn sign_evm_claim(wallet: &EcdsaSigningKey, domain: &Eip712Domain, claim: &SettlementClaim) -> [u8; 65] {
    let digest = claim.eip712_digest(domain);
    let (sig, rec) = wallet.sign_prehash_recoverable(&digest).expect("a 32-byte prehash always signs");
    // k256 signs with low `s` already; normalize anyway (flipping the recovery parity with it).
    let (sig, rec) = match sig.normalize_s() {
        Some(low) => (low, RecoveryId::from_byte(rec.to_byte() ^ 1).expect("0 or 1")),
        None => (sig, rec),
    };
    let mut out = [0u8; 65];
    out[..64].copy_from_slice(&sig.to_bytes());
    out[64] = rec.to_byte() + 27;
    out
}

/// Recover the 20-byte EVM signer address from a 65-byte secp256k1 signature (`r‖s‖v`, `v ∈ {27,28}`)
/// over an EIP-712 `digest` — the on-chain `ecrecover`. `None` on any malformed input.
pub fn recover_evm_signer(digest: &[u8; 32], sig_65: &[u8; 65]) -> Option<[u8; 20]> {
    let signature = EcdsaSignature::from_slice(&sig_65[..64]).ok()?;
    // Reject high-`s` signatures (EIP-2 low-`s` requirement, review F5): `normalize_s` returns
    // `Some` only when `s` was in the upper half-order (non-canonical). A high-`s` and its low-`s`
    // twin recover the same address, so this is not a forgery guard — it matches an on-chain
    // contract (OpenZeppelin `ECDSA.recover`) that rejects the malleable encoding, so the off-chain
    // mirror and the contract agree on exactly which signatures are valid.
    if signature.normalize_s().is_some() {
        return None;
    }
    // Exactly Ethereum's `27/28`: the contract's `ecrecover` rejects every other `v` (the raw `0/1`
    // convention included), so the off-chain check must not accept a signature the chain refuses.
    let recovery_id = match sig_65[64] {
        27 => RecoveryId::from_byte(0)?,
        28 => RecoveryId::from_byte(1)?,
        _ => return None,
    };
    let vk = EcdsaVerifyingKey::recover_from_prehash(digest, &signature, recovery_id).ok()?;
    Some(evm_address(&vk))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::receipts::{build_receipt, payload_from_bytes, ReceiptPayload};

    const NOW: u64 = 1_760_000_000;

    fn keys() -> (SigningKey, SigningKey) {
        (SigningKey::from_bytes(&[7u8; 32]), SigningKey::from_bytes(&[9u8; 32]))
    }

    fn domain() -> SolanaDomain {
        SolanaDomain { program_id: [0x44; 32], cluster: [0x55; 32] }
    }

    fn voucher_receipt(consumer: &SigningKey, provider: &SigningKey, cumulative: Option<u64>) -> CoSignedReceipt {
        let mut payload: ReceiptPayload = payload_from_bytes(
            provider.verifying_key().as_bytes(),
            consumer.verifying_key().as_bytes(),
            "qwen2.5:7b",
            512,
            &[4u8; 16],
            1_700_000_000_000,
            None,
            "",
            cumulative,
            None,
            None,
        )
        .unwrap();
        payload.cumulative_amount = cumulative;
        build_receipt(payload, consumer, provider)
    }

    fn a_claim() -> SettlementClaim {
        let (consumer, provider) = keys();
        claim_from_receipt(&voucher_receipt(&consumer, &provider, Some(9_600)), NOW + CLAIM_LIFETIME_SECS).unwrap()
    }

    // ── PayoutBinding ──

    #[test]
    fn a_binding_verifies_only_as_signed_and_only_on_its_deployment() {
        let (_, provider) = keys();
        let binding = sign_payout_binding(&provider, &domain(), [1u8; 32], 1_000);
        assert!(verify_payout_binding(&binding, &domain()));
        let mut evil = binding.clone();
        evil.payout_address = [2u8; 32];
        assert!(!verify_payout_binding(&evil, &domain()), "redirecting the address breaks the signature");
        let mut evil_ts = binding.clone();
        evil_ts.ts_unix_ms = 2_000;
        assert!(!verify_payout_binding(&evil_ts, &domain()));
        // Another program, or the same program on another cluster: not this binding.
        assert!(!verify_payout_binding(&binding, &SolanaDomain { program_id: [0x45; 32], ..domain() }));
        assert!(!verify_payout_binding(&binding, &SolanaDomain { cluster: [0x56; 32], ..domain() }));
    }

    #[test]
    fn a_self_directed_binding_pays_the_identity_account() {
        let (_, provider) = keys();
        let binding = sign_payout_binding(&provider, &domain(), provider.verifying_key().to_bytes(), 0);
        assert!(binding.is_self_directed());
        assert!(!sign_payout_binding(&provider, &domain(), [3u8; 32], 0).is_self_directed());
    }

    // ── The claim ──

    #[test]
    fn a_voucher_receipt_stands_for_a_claim() {
        let (consumer, provider) = keys();
        let claim = a_claim();
        assert_eq!(claim.provider, provider.verifying_key().to_bytes());
        assert_eq!(claim.consumer, consumer.verifying_key().to_bytes());
        assert_eq!((claim.cumulative_amount, claim.model_id.as_str()), (9_600, "qwen2.5:7b"));
        assert_eq!(claim.expires_at, NOW + CLAIM_LIFETIME_SECS);
        // No cumulative: nothing to settle. Tampered: refused.
        let plain = voucher_receipt(&consumer, &provider, None);
        assert_eq!(claim_from_receipt(&plain, NOW), Err(SettlementError::NotAVoucher));
        let mut tampered = voucher_receipt(&consumer, &provider, Some(9_600));
        tampered.payload.cumulative_amount = Some(999_999);
        assert_eq!(claim_from_receipt(&tampered, NOW), Err(SettlementError::BadReceipt));
    }

    #[test]
    fn a_solana_claim_verifies_by_its_consumer_on_its_deployment_with_time_left() {
        let (consumer, _) = keys();
        let claim = a_claim();
        let d = ChainDomain::Solana(domain());
        let sig = consumer.sign(&d.claim_message(&claim)).to_bytes();
        assert_eq!(verify_claim(&d, &claim, &sig, NOW), Ok(ClaimSigner::Solana));
        // Signed by anyone else, for another deployment, or tampered: refused.
        let other = SigningKey::from_bytes(&[13u8; 32]).sign(&d.claim_message(&claim)).to_bytes();
        assert_eq!(verify_claim(&d, &claim, &other, NOW), Err(SettlementError::BadClaimSignature));
        let devnet = ChainDomain::Solana(SolanaDomain { cluster: [0x66; 32], ..domain() });
        assert_eq!(verify_claim(&devnet, &claim, &sig, NOW), Err(SettlementError::BadClaimSignature));
        let more = SettlementClaim { cumulative_amount: 99_999, ..claim.clone() };
        assert_eq!(verify_claim(&d, &more, &sig, NOW), Err(SettlementError::BadClaimSignature));
        // Too little validity left to submit it.
        assert_eq!(
            verify_claim(&d, &claim, &sig, claim.expires_at - MIN_CLAIM_LIFETIME_SECS + 1),
            Err(SettlementError::ClaimExpiresTooSoon)
        );
    }

    #[test]
    fn a_base_claim_recovers_the_consumer_wallet() {
        let claim = a_claim();
        let domain = Eip712Domain { chain_id: 8453, verifying_contract: [1u8; 20] };
        let wallet = EcdsaSigningKey::from_bytes(&[0x11u8; 32].into()).unwrap();
        let sig = sign_evm_claim(&wallet, &domain, &claim);
        assert!(sig[64] == 27 || sig[64] == 28);
        let d = ChainDomain::Base(domain);
        assert_eq!(verify_claim(&d, &claim, &sig, NOW), Ok(ClaimSigner::Evm(evm_address(wallet.verifying_key()))));
        // On another chain the same signature recovers someone else (who owns no escrow there).
        let other = ChainDomain::Base(Eip712Domain { chain_id: 1, ..domain });
        assert_ne!(verify_claim(&other, &claim, &sig, NOW), Ok(ClaimSigner::Evm(evm_address(wallet.verifying_key()))));
        assert_eq!(verify_claim(&d, &claim, &sig[..64], NOW), Err(SettlementError::BadClaimSignature));
    }

    #[test]
    fn the_eip712_digest_is_field_and_domain_sensitive() {
        let claim = a_claim();
        let domain = Eip712Domain { chain_id: 8453, verifying_contract: [1u8; 20] };
        let d1 = claim.eip712_digest(&domain);
        assert_eq!(d1, claim.eip712_digest(&domain), "deterministic");
        assert_ne!(d1, claim.eip712_digest(&Eip712Domain { chain_id: 1, ..domain }));
        assert_ne!(d1, SettlementClaim { cumulative_amount: 9_601, ..claim.clone() }.eip712_digest(&domain));
        assert_ne!(d1, SettlementClaim { expires_at: claim.expires_at + 1, ..claim.clone() }.eip712_digest(&domain));
    }

    /// CROSS-LANGUAGE GOLDEN (parity with the Anchor program's `settle_claim_preimage`, whose unit test
    /// pins the identical bytes): claim format v6. Regenerate ONLY with a coordinated wire bump.
    #[test]
    fn solana_preimage_cross_language_golden() {
        let claim = SettlementClaim {
            provider: [0x11u8; 32],
            consumer: [0x22u8; 32],
            model_id: "qwen2.5:7b".to_string(),
            cumulative_amount: 100_000,
            expires_at: 1_800_000_000,
        };
        let pre = claim.solana_preimage(&SolanaDomain { program_id: [0x44; 32], cluster: [0x55; 32] });
        let hex = pre.iter().map(|b| format!("{b:02x}")).collect::<String>();
        assert_eq!(
            hex,
            "6f70656e68796472612f736f6c616e612d736574746c652f7632\
             4444444444444444444444444444444444444444444444444444444444444444\
             5555555555555555555555555555555555555555555555555555555555555555\
             1111111111111111111111111111111111111111111111111111111111111111\
             2222222222222222222222222222222222222222222222222222222222222222\
             a086010000000000\
             00d2496b00000000\
             0a000000\
             7177656e322e353a3762",
            "Solana claim preimage drift — on-chain/off-chain parity broken (or a deliberate bump)"
        );
    }

    /// CROSS-LANGUAGE GOLDEN (parity with the Solidity `OpenHydraSettlement.digestOf`; the Foundry tests
    /// `test_eip712_digest_matches_rust_golden` / `test_contract_digestOf_matches_rust_golden` pin the
    /// identical bytes): claim format v6, EIP-712 domain version "2".
    #[test]
    fn eip712_digest_cross_language_golden() {
        let claim = SettlementClaim {
            provider: [0x11u8; 32],
            consumer: [0x22u8; 32],
            model_id: "qwen2.5:7b".to_string(),
            cumulative_amount: 100_000,
            expires_at: 1_800_000_000,
        };
        let domain = Eip712Domain { chain_id: 8453, verifying_contract: [0xCCu8; 20] };
        let hex = claim.eip712_digest(&domain).iter().map(|b| format!("{b:02x}")).collect::<String>();
        assert_eq!(hex, "4e46f39f79c49fdc540cc2be0d73aea9384df29e2fc7b7fb2ccd630e1f2b4fa6", "EIP-712 digest drift — off-chain/on-chain parity broken (or a deliberate bump)");
    }

    #[test]
    fn secp256k1_recover_accepts_only_canonical_signatures() {
        let claim = a_claim();
        let domain = Eip712Domain { chain_id: 8453, verifying_contract: [1u8; 20] };
        let digest = claim.eip712_digest(&domain);
        let wallet = EcdsaSigningKey::from_bytes(&[0x11u8; 32].into()).unwrap();
        let expected = evm_address(wallet.verifying_key());
        let sig65 = sign_evm_claim(&wallet, &domain, &claim);
        assert_eq!(recover_evm_signer(&digest, &sig65), Some(expected));
        // Only 27/28, as `ecrecover` on-chain: the raw 0/1 form and anything else are refused.
        for v in [0u8, 1, 29, 255] {
            let mut other = sig65;
            other[64] = v;
            assert_eq!(recover_evm_signer(&digest, &other), None, "v = {v}");
        }
        let mut bad = sig65;
        bad[0] ^= 0xFF;
        assert_ne!(recover_evm_signer(&digest, &bad), Some(expected));
        // High-`s` (EIP-2 non-canonical) is rejected (review F5): the malleable twin `n − s`.
        let sig = EcdsaSignature::from_slice(&sig65[..64]).unwrap();
        let high = EcdsaSignature::from_scalars((*sig.r()).to_bytes(), (-*sig.s()).to_bytes()).unwrap();
        let mut high65 = [0u8; 65];
        high65[..64].copy_from_slice(&high.to_bytes());
        high65[64] = 27;
        assert_eq!(recover_evm_signer(&digest, &high65), None, "high-s must be rejected");
    }

    #[test]
    fn an_evm_binding_verifies_and_a_tampered_one_does_not() {
        let (_, provider) = keys();
        let binding = sign_evm_payout_binding(&provider, [0xAB; 20], 1_000);
        assert!(verify_evm_payout_binding(&binding));
        let mut evil = binding.clone();
        evil.evm_address = [0xCD; 20];
        assert!(!verify_evm_payout_binding(&evil), "redirecting the EVM address breaks the signature");
    }

    #[test]
    fn a_fingerprint_tells_deployments_apart() {
        let sol = ChainDomain::Solana(domain());
        assert_eq!(sol.fingerprint(), ChainDomain::Solana(domain()).fingerprint());
        assert_ne!(sol.fingerprint(), ChainDomain::Solana(SolanaDomain { cluster: [0x66; 32], ..domain() }).fingerprint());
        let base = ChainDomain::Base(Eip712Domain { chain_id: 8453, verifying_contract: [1; 20] });
        assert_ne!(base.fingerprint(), ChainDomain::Base(Eip712Domain { chain_id: 84532, verifying_contract: [1; 20] }).fingerprint());
        assert_eq!(sol.fingerprint().len(), 32);
    }

    #[test]
    fn chains_say_whether_they_need_a_second_key() {
        assert!(!Chain::Solana.requires_dual_sign());
        assert!(Chain::Base.requires_dual_sign());
        assert_eq!(ChainDomain::Solana(domain()).chain().label(), "solana");
    }
}
