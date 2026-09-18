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

use crate::receipts::{verify_receipt, CoSignedReceipt};

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
    /// The monotonic cumulative amount authorized (gross: provider net + protocol fee).
    pub cumulative_amount: u64,
    /// Where the provider's share is paid (from the [`PayoutBinding`]).
    pub payout_address: [u8; 32],
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
        payout_address: binding.payout_address,
    })
}

impl SettlementClaim {
    /// A deterministic canonical statement of the claim (domain ‖ provider ‖ consumer ‖ payout ‖
    /// cumulative ‖ model). A digest for a claim-level signature or an on-chain log — **not** a
    /// pinned program ABI (the on-chain instruction layout is the program's to define).
    pub fn message(&self) -> Vec<u8> {
        let model = self.model_id.as_bytes();
        let mut b = Vec::with_capacity(SETTLEMENT_CLAIM_DOMAIN.len() + 32 * 3 + 8 + 4 + model.len());
        b.extend_from_slice(SETTLEMENT_CLAIM_DOMAIN);
        b.extend_from_slice(&self.provider);
        b.extend_from_slice(&self.consumer);
        b.extend_from_slice(&self.payout_address);
        b.extend_from_slice(&self.cumulative_amount.to_le_bytes());
        b.extend_from_slice(&(model.len() as u32).to_le_bytes());
        b.extend_from_slice(model);
        b
    }
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
        assert_eq!(claim.payout_address, [5u8; 32]);
        assert_eq!(claim.model_id, "qwen2.5:7b");
        assert!(!claim.message().is_empty());
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
}
