// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Solana settlement primitives for the crypto rail (PAYMENTS_PLAN_v1, crypto sequence step 3).
//!
//! **Chain = Solana (decided 2026-09-18).** Solana settles this rail with **native Ed25519** — an
//! account address *is* a 32-byte Ed25519 public key, and the runtime verifies Ed25519 signatures
//! natively (the Ed25519 program). So the provider's existing OpenHydra identity is already a valid
//! Solana payout account, and the co-signed voucher receipt ([`crate::receipts::CoSignedReceipt`]
//! with a `cumulative_amount`) is already the Ed25519-signed settlement claim the on-chain program
//! verifies. There is **no secp256k1, no EVM `ecrecover`, and no cross-curve dual-sign** — the EVM
//! path's whole reason for a second signature does not exist here. Solana's ~sub-thousandth-cent per
//! tx is also what makes the micropayment model (no authenticity escrow — the rate cap suffices)
//! actually feasible.
//!
//! This module adds the two contract-agnostic pieces that sit above the receipt:
//! 1. [`PayoutBinding`] — the identity authorizes **which Solana account receives** its payouts
//!    (its own identity by default, or a distinct address), signed by the identity so the on-chain
//!    program can prove same-entity ownership. Same-curve Ed25519; no new key material.
//! 2. [`SettlementClaim`] — the validated, canonical claim extracted from a co-signed voucher
//!    receipt + a verified binding: *pay `cumulative_amount` for this `(provider, consumer, model)`
//!    to `payout_address`*. The on-chain program releases the delta since the last settlement.
//!
//! **Purity + status.** No I/O, no chain SDK, no new dependency (ed25519 only). Nothing here moves
//! value or talks to a chain — that is the on-chain program + facilitator, a later, **legally-gated**
//! slice (non-custodial opinion, program audit, MiCA/FIU-IND geofencing). The exact on-chain
//! instruction layout is the program's to define; [`SettlementClaim::message`] is a deterministic
//! canonical statement of the claim, not a pinned program ABI.

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use k256::ecdsa::{RecoveryId, Signature as EcdsaSignature, VerifyingKey as EcdsaVerifyingKey};
use sha3::{Digest, Keccak256};

use crate::receipts::{verify_receipt, CoSignedReceipt};

/// keccak-256 (Ethereum's hash) of `data`.
fn keccak256(data: &[u8]) -> [u8; 32] {
    let mut h = Keccak256::new();
    h.update(data);
    h.finalize().into()
}

/// Domain separator for a [`PayoutBinding`] signature (prevents cross-protocol signature reuse).
pub const PAYOUT_BINDING_DOMAIN: &[u8] = b"openhydra/solana-payout/v1";

/// Domain separator for a [`SettlementClaim`] canonical message.
pub const SETTLEMENT_CLAIM_DOMAIN: &[u8] = b"openhydra/solana-settle/v1";

/// Why a settlement primitive was rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SettlementError {
    /// The payout binding's signature does not verify against the identity key.
    BadBindingSignature,
    /// The binding's identity is not the receipt's provider (the payee) — a binding for a different
    /// provider cannot redirect this provider's voucher.
    BindingProviderMismatch,
    /// The receipt carries no `cumulative_amount` — it is not a crypto-rail voucher, so there is
    /// nothing to settle.
    NotAVoucher,
    /// The receipt's own signatures do not verify (tampered or corrupt).
    BadReceipt,
    /// This chain's settlement backend is not built yet — fail closed rather than fabricate a proof.
    BackendNotImplemented,
    /// The claim's [`PayoutTarget`] is for a different chain than the backend (e.g. a Solana payout
    /// handed to the Base backend).
    ChainMismatch,
}

/// Where a settlement pays the provider — the one chain-specific field on an otherwise chain-agnostic
/// [`SettlementClaim`]. A Solana account is a 32-byte Ed25519 public key; an EVM account is a 20-byte
/// address (`keccak256(secp256k1_pubkey)[12..]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PayoutTarget {
    /// A Solana account (32-byte Ed25519 public key).
    Solana([u8; 32]),
    /// An EVM account (20-byte address).
    Evm([u8; 20]),
}

/// A provider's authorization of the Solana account that receives its payouts. The provider's
/// OpenHydra Ed25519 identity (which *earns* the voucher) signs the chosen `payout_address` (a
/// 32-byte Solana account = an Ed25519 public key), so the on-chain program can prove the same entity
/// owns both. `payout_address == identity.as_bytes()` is a self-directed default (pay the identity
/// account itself). `ts_unix_ms` orders rotations — the on-chain program uses the latest binding, so
/// an old one can't be replayed to redirect funds backward.
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

/// The bytes an identity signs to authorize a payout address: domain ‖ payout_address ‖ ts (LE).
pub fn binding_message(payout_address: &[u8; 32], ts_unix_ms: u64) -> Vec<u8> {
    let mut b = Vec::with_capacity(PAYOUT_BINDING_DOMAIN.len() + 32 + 8);
    b.extend_from_slice(PAYOUT_BINDING_DOMAIN);
    b.extend_from_slice(payout_address);
    b.extend_from_slice(&ts_unix_ms.to_le_bytes());
    b
}

/// Sign a [`PayoutBinding`]: the identity authorizes `payout_address` to receive its payouts. Pass
/// the identity's own public-key bytes as `payout_address` for the self-directed default.
pub fn sign_payout_binding(
    identity_key: &SigningKey,
    payout_address: [u8; 32],
    ts_unix_ms: u64,
) -> PayoutBinding {
    let signature = identity_key.sign(&binding_message(&payout_address, ts_unix_ms));
    PayoutBinding {
        identity: identity_key.verifying_key(),
        payout_address,
        ts_unix_ms,
        signature,
    }
}

/// Whether a payout binding's signature verifies — i.e. the named identity really authorized this
/// payout address at this time. A binding that fails this must never be used to redirect a payout.
pub fn verify_payout_binding(binding: &PayoutBinding) -> bool {
    binding
        .identity
        .verify(&binding_message(&binding.payout_address, binding.ts_unix_ms), &binding.signature)
        .is_ok()
}

impl PayoutBinding {
    /// Whether this binding pays the identity's own account (no redirection).
    pub fn is_self_directed(&self) -> bool {
        self.payout_address == self.identity.to_bytes()
    }
}

/// The validated, canonical settlement claim extracted from a co-signed voucher receipt + a verified
/// payout binding: pay `cumulative_amount` for this `(provider, consumer, model)` to
/// `payout_address`. The on-chain program releases `cumulative_amount − already_settled` (the
/// monotonic voucher makes replays a no-op). All keys are 32-byte Ed25519 accounts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SettlementClaim {
    /// The provider (voucher payee) — a Solana account.
    pub provider: [u8; 32],
    /// The consumer (voucher payer) — a Solana account.
    pub consumer: [u8; 32],
    /// The raw engine handle served (identity of the pair's model bucket).
    pub model_id: String,
    /// The monotonic cumulative amount authorized (gross: provider net + protocol fee), in the
    /// settlement asset's smallest unit (USDC micro-units).
    pub cumulative_amount: u64,
    /// Where the provider's share is paid — chain-specific (from the chain's payout binding).
    pub payout: PayoutTarget,
}

/// Build a [`SettlementClaim`] from a co-signed voucher receipt and the provider's payout binding.
/// Fails closed: the binding must verify and name the receipt's provider, the receipt must carry a
/// `cumulative_amount`, and the receipt's own co-signatures must verify. Pure — no chain contact.
pub fn claim_from_receipt(
    receipt: &CoSignedReceipt,
    binding: &PayoutBinding,
) -> Result<SettlementClaim, SettlementError> {
    if !verify_payout_binding(binding) {
        return Err(SettlementError::BadBindingSignature);
    }
    if binding.identity.to_bytes() != receipt.payload.provider.to_bytes() {
        return Err(SettlementError::BindingProviderMismatch);
    }
    let cumulative_amount = receipt.payload.cumulative_amount.ok_or(SettlementError::NotAVoucher)?;
    verify_receipt(receipt).map_err(|_| SettlementError::BadReceipt)?;
    Ok(SettlementClaim {
        provider: receipt.payload.provider.to_bytes(),
        consumer: receipt.payload.consumer.to_bytes(),
        model_id: receipt.payload.model_id.clone(),
        cumulative_amount,
        payout: PayoutTarget::Solana(binding.payout_address),
    })
}

impl SettlementClaim {
    /// The Solana canonical claim preimage (domain ‖ provider ‖ consumer ‖ payout(32) ‖ cumulative ‖
    /// model) — the message the Solana program verifies the receipt's Ed25519 signatures against.
    /// `None` if the payout is not a Solana target. **Not** a pinned program ABI.
    pub fn solana_preimage(&self) -> Option<Vec<u8>> {
        let PayoutTarget::Solana(addr) = self.payout else {
            return None;
        };
        let model = self.model_id.as_bytes();
        let mut b = Vec::with_capacity(SETTLEMENT_CLAIM_DOMAIN.len() + 32 * 3 + 8 + 4 + model.len());
        b.extend_from_slice(SETTLEMENT_CLAIM_DOMAIN);
        b.extend_from_slice(&self.provider);
        b.extend_from_slice(&self.consumer);
        b.extend_from_slice(&addr);
        b.extend_from_slice(&self.cumulative_amount.to_le_bytes());
        b.extend_from_slice(&(model.len() as u32).to_le_bytes());
        b.extend_from_slice(model);
        Some(b)
    }
}

// ── Chain-agnostic settlement backend seam ────────────────────────────────
//
// The voucher layer above ([`crate::payout::GatedVoucher`], the v4 co-signed receipt, and
// [`SettlementClaim`]) is **chain-independent** — it names amounts and Ed25519 accounts, nothing
// chain-specific. The only place chains differ is the **on-chain proof format**, so a thin backend
// trait is the whole seam. Both are built: Solana (native Ed25519, no dual-sign) and Base (secp256k1
// + EIP-712 — see the Base/EVM section below). Base still gated by its own on-chain program audit +
// geofencing before real value; ship Solana first.

/// A settlement chain OpenHydra can pay out on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Chain {
    /// Solana — native Ed25519 accounts + on-chain Ed25519 verification. No dual-sign, no new dep.
    Solana,
    /// Base (EVM L2) — needs a secp256k1 co-signature bound to the Ed25519 identity (`ecrecover`).
    Base,
}

impl Chain {
    /// A stable lower-case label for display / telemetry.
    pub fn label(self) -> &'static str {
        match self {
            Chain::Solana => "solana",
            Chain::Base => "base",
        }
    }

    /// Whether this chain needs a **secp256k1 dual-sign** bound to the Ed25519 identity (EVM
    /// `ecrecover`) *in addition to* the native Ed25519 receipt signatures. Solana verifies Ed25519
    /// natively → `false`; Base/EVM → `true`.
    pub fn requires_dual_sign(self) -> bool {
        match self {
            Chain::Solana => false,
            Chain::Base => true,
        }
    }
}

/// A per-chain settlement backend: turns a validated, chain-agnostic [`SettlementClaim`] into the
/// canonical preimage that chain's on-chain program verifies against. The claim itself (and the
/// whole voucher layer) is shared; only this is chain-specific. Object-safe, so a node can hold a
/// `Box<dyn SettlementBackend>` chosen by config.
pub trait SettlementBackend {
    /// Which chain this backend settles on.
    fn chain(&self) -> Chain;

    /// The canonical claim preimage this chain's program verifies against (a deterministic
    /// serialization of the claim — **not** a pinned program ABI). `Err(BackendNotImplemented)` for
    /// a chain whose settlement backend is not built yet.
    fn claim_preimage(&self, claim: &SettlementClaim) -> Result<Vec<u8>, SettlementError>;
}

/// The Solana settlement backend (built). Solana verifies the receipt's existing Ed25519
/// co-signatures natively and pays the [`PayoutBinding`]'s account, so the claim's Solana preimage
/// ([`SettlementClaim::solana_preimage`]) is the message — no extra signature, no dependency.
#[derive(Debug, Clone, Copy, Default)]
pub struct SolanaBackend;

impl SettlementBackend for SolanaBackend {
    fn chain(&self) -> Chain {
        Chain::Solana
    }

    fn claim_preimage(&self, claim: &SettlementClaim) -> Result<Vec<u8>, SettlementError> {
        claim.solana_preimage().ok_or(SettlementError::ChainMismatch)
    }
}

// ── Base / EVM settlement (secp256k1 + EIP-712) ───────────────────────────
//
// Base (EVM L2) cannot verify Ed25519 natively, so it settles with **secp256k1** (`ecrecover`) over
// an **EIP-712** digest, and the Ed25519 identity **attests** its EVM payout address (the cross-curve
// binding). The voucher layer is unchanged — only the on-chain proof format differs.

/// Domain separator for an [`EvmPayoutBinding`] signature.
pub const EVM_PAYOUT_BINDING_DOMAIN: &[u8] = b"openhydra/base-payout/v1";

/// The 20-byte EVM address for a secp256k1 verifying key: `keccak256(uncompressed_pubkey[1..])[12..]`
/// (the last 20 bytes of the hash of the 64-byte `x‖y`, dropping the `0x04` prefix).
pub fn evm_address(pubkey: &EcdsaVerifyingKey) -> [u8; 20] {
    let point = pubkey.to_encoded_point(false); // uncompressed: 0x04 ‖ x(32) ‖ y(32)
    let hash = keccak256(&point.as_bytes()[1..]);
    let mut addr = [0u8; 20];
    addr.copy_from_slice(&hash[12..]);
    addr
}

/// A provider's authorization of the **EVM** account that receives its Base payouts — the cross-curve
/// analogue of [`PayoutBinding`]. The provider's OpenHydra **Ed25519** identity (which earns the
/// voucher) signs the chosen 20-byte EVM address, so the on-chain program can prove the same entity
/// owns both keys. `ts_unix_ms` orders rotations. (No secp256k1 is needed to *make* the binding — it
/// attests a given address; secp256k1 enters only when the provider later signs the settlement with
/// its EVM wallet.)
#[derive(Debug, Clone)]
pub struct EvmPayoutBinding {
    /// The provider's OpenHydra identity — the voucher payee.
    pub identity: VerifyingKey,
    /// The EVM account to receive payout (20-byte address).
    pub evm_address: [u8; 20],
    /// When this binding was made (unix ms) — later supersedes earlier for the same identity.
    pub ts_unix_ms: u64,
    /// The identity's Ed25519 signature over [`evm_binding_message`].
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

/// Sign an [`EvmPayoutBinding`]: the Ed25519 identity authorizes `evm_address` to receive its Base
/// payouts.
pub fn sign_evm_payout_binding(
    identity_key: &SigningKey,
    evm_address: [u8; 20],
    ts_unix_ms: u64,
) -> EvmPayoutBinding {
    let signature = identity_key.sign(&evm_binding_message(&evm_address, ts_unix_ms));
    EvmPayoutBinding {
        identity: identity_key.verifying_key(),
        evm_address,
        ts_unix_ms,
        signature,
    }
}

/// Whether an EVM payout binding's Ed25519 signature verifies.
pub fn verify_evm_payout_binding(binding: &EvmPayoutBinding) -> bool {
    binding
        .identity
        .verify(&evm_binding_message(&binding.evm_address, binding.ts_unix_ms), &binding.signature)
        .is_ok()
}

/// EIP-712 domain parameters for the settlement contract. `chain_id` + `verifying_contract` are
/// **deployment values** (the deployed program's chain + address), not known until deploy — carried
/// here so the digest matches exactly what the contract computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Eip712Domain {
    pub chain_id: u64,
    pub verifying_contract: [u8; 20],
}

/// The EIP-712 domain type string (`name` + `version` are fixed for the protocol).
const EIP712_DOMAIN_TYPE: &[u8] =
    b"EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)";
const EIP712_DOMAIN_NAME: &[u8] = b"OpenHydra Settlement";
const EIP712_DOMAIN_VERSION: &[u8] = b"1";
/// The settlement claim's EIP-712 type string — the on-chain contract MUST hash the identical
/// string, so this is the pinned type (any change is a breaking, coordinated version bump).
const EIP712_CLAIM_TYPE: &[u8] = b"SettlementClaim(bytes32 provider,bytes32 consumer,address payoutAddress,uint256 cumulativeAmount,string modelId)";

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

impl SettlementClaim {
    /// The EIP-712 digest an EVM `ecrecover` verifies a secp256k1 settlement signature against:
    /// `keccak256(0x19 0x01 ‖ domainSeparator ‖ hashStruct(claim))`. `None` unless the payout is an
    /// EVM target. The contract computes the identical digest; only `chain_id` + `verifying_contract`
    /// are deployment-specific.
    pub fn eip712_digest(&self, domain: &Eip712Domain) -> Option<[u8; 32]> {
        let PayoutTarget::Evm(payout) = self.payout else {
            return None;
        };
        // hashStruct(claim) = keccak256(typeHash ‖ provider ‖ consumer ‖ payout ‖ cumulative ‖ keccak(modelId))
        let mut sd = Vec::with_capacity(32 * 6);
        sd.extend_from_slice(&keccak256(EIP712_CLAIM_TYPE));
        sd.extend_from_slice(&self.provider);
        sd.extend_from_slice(&self.consumer);
        sd.extend_from_slice(&address_word(&payout));
        sd.extend_from_slice(&u256_word(self.cumulative_amount));
        sd.extend_from_slice(&keccak256(self.model_id.as_bytes()));
        let hash_struct = keccak256(&sd);

        let mut pre = Vec::with_capacity(2 + 32 + 32);
        pre.extend_from_slice(&[0x19, 0x01]);
        pre.extend_from_slice(&domain.separator());
        pre.extend_from_slice(&hash_struct);
        Some(keccak256(&pre))
    }
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

/// The Base/EVM settlement backend (built). Base settles via a secp256k1 signature over the claim's
/// [`eip712_digest`](SettlementClaim::eip712_digest); [`claim_preimage`](Self::claim_preimage)
/// returns that 32-byte digest. The on-chain contract `ecrecover`s the signature to an EVM address
/// and checks it against the provider's / consumer's [`EvmPayoutBinding`]. `chain_id` +
/// `verifying_contract` are the deployed program's; until it exists this is unwired.
#[derive(Debug, Clone, Copy)]
pub struct BaseBackend {
    pub domain: Eip712Domain,
}

impl BaseBackend {
    /// A backend for the given EVM chain id + deployed contract address.
    pub fn new(chain_id: u64, verifying_contract: [u8; 20]) -> Self {
        Self { domain: Eip712Domain { chain_id, verifying_contract } }
    }
}

impl SettlementBackend for BaseBackend {
    fn chain(&self) -> Chain {
        Chain::Base
    }

    fn claim_preimage(&self, claim: &SettlementClaim) -> Result<Vec<u8>, SettlementError> {
        claim
            .eip712_digest(&self.domain)
            .map(|d| d.to_vec())
            .ok_or(SettlementError::ChainMismatch)
    }
}

/// Build a [`SettlementClaim`] with an **EVM** payout target from a co-signed voucher receipt and the
/// provider's EVM payout binding. Same fail-closed checks as [`claim_from_receipt`] (verify the
/// binding, match the provider, require a cumulative, verify the receipt).
pub fn claim_from_receipt_evm(
    receipt: &CoSignedReceipt,
    binding: &EvmPayoutBinding,
) -> Result<SettlementClaim, SettlementError> {
    if !verify_evm_payout_binding(binding) {
        return Err(SettlementError::BadBindingSignature);
    }
    if binding.identity.to_bytes() != receipt.payload.provider.to_bytes() {
        return Err(SettlementError::BindingProviderMismatch);
    }
    let cumulative_amount = receipt.payload.cumulative_amount.ok_or(SettlementError::NotAVoucher)?;
    verify_receipt(receipt).map_err(|_| SettlementError::BadReceipt)?;
    Ok(SettlementClaim {
        provider: receipt.payload.provider.to_bytes(),
        consumer: receipt.payload.consumer.to_bytes(),
        model_id: receipt.payload.model_id.clone(),
        cumulative_amount,
        payout: PayoutTarget::Evm(binding.evm_address),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::receipts::{build_receipt, payload_from_bytes, ReceiptPayload};

    fn keys() -> (SigningKey, SigningKey) {
        (SigningKey::from_bytes(&[7u8; 32]), SigningKey::from_bytes(&[9u8; 32]))
    }

    fn voucher_receipt(
        consumer: &SigningKey,
        provider: &SigningKey,
        cumulative: Option<u64>,
    ) -> CoSignedReceipt {
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

    // ── PayoutBinding ──

    #[test]
    fn a_signed_binding_verifies_and_a_tampered_one_does_not() {
        let (_, provider) = keys();
        let addr = [1u8; 32];
        let binding = sign_payout_binding(&provider, addr, 1_000);
        assert!(verify_payout_binding(&binding));
        // Tamper with the payout address → the signature no longer matches.
        let mut evil = binding.clone();
        evil.payout_address = [2u8; 32];
        assert!(!verify_payout_binding(&evil), "redirecting the address breaks the signature");
        // Tamper with the timestamp → also fails.
        let mut evil_ts = binding.clone();
        evil_ts.ts_unix_ms = 2_000;
        assert!(!verify_payout_binding(&evil_ts));
    }

    #[test]
    fn a_self_directed_binding_pays_the_identity_account() {
        let (_, provider) = keys();
        let self_addr = provider.verifying_key().to_bytes();
        let binding = sign_payout_binding(&provider, self_addr, 0);
        assert!(verify_payout_binding(&binding));
        assert!(binding.is_self_directed());
        // A distinct address is not self-directed.
        let other = sign_payout_binding(&provider, [3u8; 32], 0);
        assert!(!other.is_self_directed());
    }

    // ── claim_from_receipt ──

    #[test]
    fn a_valid_voucher_and_binding_produce_a_claim() {
        let (consumer, provider) = keys();
        let receipt = voucher_receipt(&consumer, &provider, Some(9_600));
        let binding = sign_payout_binding(&provider, [5u8; 32], 1_000);
        let claim = claim_from_receipt(&receipt, &binding).unwrap();
        assert_eq!(claim.provider, provider.verifying_key().to_bytes());
        assert_eq!(claim.consumer, consumer.verifying_key().to_bytes());
        assert_eq!(claim.cumulative_amount, 9_600);
        assert_eq!(claim.payout, PayoutTarget::Solana([5u8; 32]));
        assert_eq!(claim.model_id, "qwen2.5:7b");
        assert!(claim.solana_preimage().is_some());
    }

    #[test]
    fn a_receipt_without_a_cumulative_is_not_a_voucher() {
        let (consumer, provider) = keys();
        let receipt = voucher_receipt(&consumer, &provider, None);
        let binding = sign_payout_binding(&provider, [5u8; 32], 1_000);
        assert_eq!(claim_from_receipt(&receipt, &binding), Err(SettlementError::NotAVoucher));
    }

    #[test]
    fn a_binding_for_a_different_provider_cannot_claim_the_voucher() {
        let (consumer, provider) = keys();
        let attacker = SigningKey::from_bytes(&[13u8; 32]);
        let receipt = voucher_receipt(&consumer, &provider, Some(9_600));
        // The attacker binds ITS OWN identity to its own payout address — a valid binding, but not
        // for THIS receipt's provider, so it cannot redirect the provider's voucher.
        let evil_binding = sign_payout_binding(&attacker, [66u8; 32], 1_000);
        assert!(verify_payout_binding(&evil_binding));
        assert_eq!(
            claim_from_receipt(&receipt, &evil_binding),
            Err(SettlementError::BindingProviderMismatch)
        );
    }

    #[test]
    fn a_forged_binding_signature_is_rejected() {
        let (consumer, provider) = keys();
        let receipt = voucher_receipt(&consumer, &provider, Some(9_600));
        let mut binding = sign_payout_binding(&provider, [5u8; 32], 1_000);
        binding.payout_address = [9u8; 32]; // signature now doesn't cover this address
        assert_eq!(claim_from_receipt(&receipt, &binding), Err(SettlementError::BadBindingSignature));
    }

    #[test]
    fn a_tampered_receipt_is_rejected() {
        let (consumer, provider) = keys();
        let mut receipt = voucher_receipt(&consumer, &provider, Some(9_600));
        let binding = sign_payout_binding(&provider, [5u8; 32], 1_000);
        // Inflate the claimed cumulative after signing → the receipt's own signatures no longer verify.
        receipt.payload.cumulative_amount = Some(999_999);
        assert_eq!(claim_from_receipt(&receipt, &binding), Err(SettlementError::BadReceipt));
    }

    // ── chain-agnostic backend seam ──

    fn a_claim() -> SettlementClaim {
        let (consumer, provider) = keys();
        let receipt = voucher_receipt(&consumer, &provider, Some(9_600));
        let binding = sign_payout_binding(&provider, [5u8; 32], 1_000);
        claim_from_receipt(&receipt, &binding).unwrap()
    }

    /// CROSS-LANGUAGE GOLDEN (parity with the on-chain Anchor program's `settle_claim_preimage`): a
    /// fixed Solana claim must reconstruct this exact preimage. The Anchor program's unit test pins the
    /// identical bytes, so the message the program's Ed25519 introspection matches is byte-for-byte the
    /// one `solana_preimage` (and the off-chain signer) produced. Regenerate ONLY with a coordinated
    /// wire bump on both sides.
    #[test]
    fn solana_preimage_cross_language_golden() {
        let claim = SettlementClaim {
            provider: [0x11u8; 32],
            consumer: [0x22u8; 32],
            model_id: "qwen2.5:7b".to_string(),
            cumulative_amount: 100_000,
            payout: PayoutTarget::Solana([0x33u8; 32]),
        };
        let pre = claim.solana_preimage().unwrap();
        let hex = pre.iter().map(|b| format!("{b:02x}")).collect::<String>();
        assert_eq!(
            hex,
            "6f70656e68796472612f736f6c616e612d736574746c652f7631111111111111111111111111111111111111111111111111111111111111111122222222222222222222222222222222222222222222222222222222222222223333333333333333333333333333333333333333333333333333333333333333a0860100000000000a0000007177656e322e353a3762",
            "Solana claim preimage drift — on-chain/off-chain parity broken (or a deliberate bump)"
        );
    }

    #[test]
    fn solana_backend_yields_the_claim_message_and_needs_no_dual_sign() {
        let claim = a_claim();
        let backend = SolanaBackend;
        assert_eq!(backend.chain(), Chain::Solana);
        assert!(!Chain::Solana.requires_dual_sign(), "Solana verifies Ed25519 natively");
        assert_eq!(backend.claim_preimage(&claim).unwrap(), claim.solana_preimage().unwrap());
    }

    #[test]
    fn a_backend_rejects_a_claim_for_the_other_chain() {
        // A Solana-target claim handed to the Base backend (and vice versa) is a ChainMismatch, not a
        // fabricated proof.
        let solana_claim = a_claim(); // PayoutTarget::Solana
        let base = BaseBackend::new(8453, [0u8; 20]);
        assert_eq!(base.claim_preimage(&solana_claim), Err(SettlementError::ChainMismatch));
        let evm_claim = an_evm_claim();
        assert_eq!(SolanaBackend.claim_preimage(&evm_claim), Err(SettlementError::ChainMismatch));
    }

    #[test]
    fn backends_are_object_safe_and_selectable_by_config() {
        // A node holds whichever backend its config chose behind one trait object.
        let backends: Vec<Box<dyn SettlementBackend>> =
            vec![Box::new(SolanaBackend), Box::new(BaseBackend::new(8453, [0u8; 20]))];
        let labels: Vec<&str> = backends.iter().map(|b| b.chain().label()).collect();
        assert_eq!(labels, vec!["solana", "base"]);
        // Each backend produces a preimage for its own chain's claim.
        assert!(backends[0].claim_preimage(&a_claim()).is_ok());
        assert!(backends[1].claim_preimage(&an_evm_claim()).is_ok());
    }

    // ── Base / EVM settlement ──

    fn an_evm_claim() -> SettlementClaim {
        let (consumer, provider) = keys();
        let receipt = voucher_receipt(&consumer, &provider, Some(9_600));
        let binding = sign_evm_payout_binding(&provider, [0xAB; 20], 1_000);
        claim_from_receipt_evm(&receipt, &binding).unwrap()
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
    fn claim_from_receipt_evm_sets_an_evm_payout_target() {
        let claim = an_evm_claim();
        assert_eq!(claim.payout, PayoutTarget::Evm([0xAB; 20]));
        assert_eq!(claim.cumulative_amount, 9_600);
        // Same fail-closed checks as the Solana path: a binding for a different provider is rejected.
        let (consumer, provider) = keys();
        let receipt = voucher_receipt(&consumer, &provider, Some(9_600));
        let attacker = SigningKey::from_bytes(&[13u8; 32]);
        let evil = sign_evm_payout_binding(&attacker, [0xEE; 20], 1_000);
        assert_eq!(
            claim_from_receipt_evm(&receipt, &evil),
            Err(SettlementError::BindingProviderMismatch)
        );
    }

    #[test]
    fn eip712_digest_is_deterministic_and_field_sensitive() {
        let claim = an_evm_claim();
        let domain = Eip712Domain { chain_id: 8453, verifying_contract: [1u8; 20] };
        let d1 = claim.eip712_digest(&domain).unwrap();
        assert_eq!(d1, claim.eip712_digest(&domain).unwrap(), "deterministic");
        assert_eq!(d1.len(), 32);
        // A different chain id / contract → a different digest (domain-separated).
        let other = Eip712Domain { chain_id: 1, verifying_contract: [1u8; 20] };
        assert_ne!(d1, claim.eip712_digest(&other).unwrap());
        // A different cumulative amount → a different digest.
        let mut claim2 = claim.clone();
        claim2.cumulative_amount = 9_601;
        assert_ne!(d1, claim2.eip712_digest(&domain).unwrap());
        // A Solana-target claim has no EIP-712 digest.
        assert!(a_claim().eip712_digest(&domain).is_none());
    }

    #[test]
    fn secp256k1_recover_round_trips_to_the_signer_address() {
        use k256::ecdsa::SigningKey as EcdsaSigningKey;
        let claim = an_evm_claim();
        let domain = Eip712Domain { chain_id: 8453, verifying_contract: [1u8; 20] };
        let digest = claim.eip712_digest(&domain).unwrap();

        // An EVM wallet signs the digest; recover_evm_signer must return that wallet's address.
        let wallet = EcdsaSigningKey::from_bytes(&[0x11u8; 32].into()).unwrap();
        let expected = evm_address(wallet.verifying_key());
        let (sig, rec) = wallet.sign_prehash_recoverable(&digest).unwrap();
        let mut sig65 = [0u8; 65];
        sig65[..64].copy_from_slice(&sig.to_bytes());
        sig65[64] = rec.to_byte() + 27; // Ethereum's v convention
        assert_eq!(recover_evm_signer(&digest, &sig65), Some(expected));
        // Only 27/28, as `ecrecover` on-chain: the raw 0/1 form and anything else are refused.
        for v in [0u8, 1, 29, 255] {
            let mut other = sig65;
            other[64] = v;
            assert_eq!(recover_evm_signer(&digest, &other), None, "v = {v}");
        }
        // A wrong signature recovers a different (or no) address, never the expected one by accident.
        let mut bad = sig65;
        bad[0] ^= 0xFF;
        assert_ne!(recover_evm_signer(&digest, &bad), Some(expected));
        // High-`s` (EIP-2 non-canonical) is rejected (review F5). Build the malleable twin `n − s`.
        let s_neg = -*sig.s();
        let high = EcdsaSignature::from_scalars((*sig.r()).to_bytes(), s_neg.to_bytes()).unwrap();
        let mut high65 = [0u8; 65];
        high65[..64].copy_from_slice(&high.to_bytes());
        high65[64] = 27;
        assert_eq!(recover_evm_signer(&digest, &high65), None, "high-s must be rejected");
    }

    /// CROSS-LANGUAGE GOLDEN (parity with the Solidity `OpenHydraSettlement.digestOf`): a fixed claim
    /// + fixed EIP-712 domain must produce this exact 32-byte digest. The Foundry test
    /// `test_eip712_digest_matches_rust_golden` pins the identical bytes, so the off-chain
    /// `recover_evm_signer` and the on-chain `ecrecover` provably agree on which digest is signed.
    /// Regenerate ONLY with a deliberate, coordinated EIP-712 version bump on both sides.
    #[test]
    fn eip712_digest_cross_language_golden() {
        let claim = SettlementClaim {
            provider: [0x11u8; 32],
            consumer: [0x22u8; 32],
            model_id: "qwen2.5:7b".to_string(),
            cumulative_amount: 100_000,
            payout: PayoutTarget::Evm([0xABu8; 20]),
        };
        let domain = Eip712Domain { chain_id: 8453, verifying_contract: [0xCCu8; 20] };
        let digest = claim.eip712_digest(&domain).unwrap();
        let hex = digest.iter().map(|b| format!("{b:02x}")).collect::<String>();
        assert_eq!(
            hex,
            "a3cd9b3cb6dbd590c004a95113e457831526cff559e6707131e7f39c71e80f36",
            "EIP-712 digest drift — off-chain/on-chain parity broken (or a deliberate version bump)"
        );
    }

    #[test]
    fn base_backend_claim_preimage_is_the_eip712_digest() {
        let claim = an_evm_claim();
        let backend = BaseBackend::new(8453, [1u8; 20]);
        assert_eq!(backend.chain(), Chain::Base);
        assert!(Chain::Base.requires_dual_sign(), "Base needs a secp256k1 ecrecover dual-sign");
        let pre = backend.claim_preimage(&claim).unwrap();
        assert_eq!(pre, claim.eip712_digest(&backend.domain).unwrap().to_vec());
    }
}
