// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Co-signed inference receipts (protocol.md §6) — M2.1.
//!
//! Credit accrues only against a **co-signed receipt**:
//!
//! ```text
//! receipt = sign_provider( sign_consumer( provider, consumer, model_id, tokens, nonce, ts ) )
//! ```
//!
//! The consumer signs that they received the tokens; the provider counter-signs
//! over the consumer's signature and submits the receipt to claim credit. Neither
//! side can unilaterally inflate: tampering with any field invalidates the
//! consumer signature, and the provider cannot forge the consumer's signature.
//! A `nonce` prevents double-counting.
//!
//! This module is the **pure crypto + validation math**: nested ed25519 signatures
//! over a canonical byte encoding, plus an in-memory [`NonceTracker`] for replay
//! rejection. The *persistent* ledger (storage, gossip replication, monotonic
//! per-peer counters) is M2.3 — nothing here touches a database.

use std::collections::HashSet;

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};

/// Domain separator for the consumer-signed payload (prevents cross-protocol
/// signature reuse). Versioned: a change is a wire-format bump.
pub const RECEIPT_DOMAIN: &[u8] = b"openhydra/receipt/v1";
/// Domain separator for the provider's co-signature over (payload ‖ consumer_sig).
pub const COSIGN_DOMAIN: &[u8] = b"openhydra/receipt-cosign/v1";

/// The content both parties sign over (the consumer directly, the provider via the
/// co-signature). Identities are ed25519 public keys.
#[derive(Debug, Clone)]
pub struct ReceiptPayload {
    /// The serving provider's public key.
    pub provider: VerifyingKey,
    /// The consuming client's public key.
    pub consumer: VerifyingKey,
    /// Canonical model id served (protocol.md §4).
    pub model_id: String,
    /// Tokens the consumer acknowledges receiving.
    pub tokens: u64,
    /// 128-bit nonce — unique per receipt; prevents double-counting / replay.
    pub nonce: [u8; 16],
    /// Receipt timestamp (unix ms).
    pub ts_unix_ms: u64,
}

impl ReceiptPayload {
    /// Canonical bytes the **consumer** signs. Deterministic: a fixed payload always
    /// produces the same bytes (and, with ed25519's RFC-8032 determinism, the same
    /// signature — see the golden test).
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(RECEIPT_DOMAIN.len() + 32 + 32 + 4 + self.model_id.len() + 8 + 16 + 8);
        b.extend_from_slice(RECEIPT_DOMAIN);
        b.extend_from_slice(self.provider.as_bytes());
        b.extend_from_slice(self.consumer.as_bytes());
        b.extend_from_slice(&(self.model_id.len() as u32).to_le_bytes());
        b.extend_from_slice(self.model_id.as_bytes());
        b.extend_from_slice(&self.tokens.to_le_bytes());
        b.extend_from_slice(&self.nonce);
        b.extend_from_slice(&self.ts_unix_ms.to_le_bytes());
        b
    }
}

/// Bytes the **provider** co-signs: the consumer-signed payload plus the consumer's
/// signature, so the provider's signature commits to *exactly* what the consumer
/// signed.
pub fn cosign_bytes(payload: &ReceiptPayload, consumer_sig: &Signature) -> Vec<u8> {
    let payload_bytes = payload.canonical_bytes();
    let mut b = Vec::with_capacity(COSIGN_DOMAIN.len() + payload_bytes.len() + 64);
    b.extend_from_slice(COSIGN_DOMAIN);
    b.extend_from_slice(&payload_bytes);
    b.extend_from_slice(&consumer_sig.to_bytes());
    b
}

/// Build a [`ReceiptPayload`] from raw byte components (32-byte public keys, 16-byte
/// nonce), validating lengths. For the FFI / node-method callers.
pub fn payload_from_bytes(
    provider_pub: &[u8],
    consumer_pub: &[u8],
    model_id: &str,
    tokens: u64,
    nonce: &[u8],
    ts_unix_ms: u64,
) -> Result<ReceiptPayload, String> {
    let provider = VerifyingKey::from_bytes(
        provider_pub
            .try_into()
            .map_err(|_| "provider public key must be 32 bytes".to_string())?,
    )
    .map_err(|e| format!("invalid provider public key: {e}"))?;
    let consumer = VerifyingKey::from_bytes(
        consumer_pub
            .try_into()
            .map_err(|_| "consumer public key must be 32 bytes".to_string())?,
    )
    .map_err(|e| format!("invalid consumer public key: {e}"))?;
    let nonce: [u8; 16] = nonce
        .try_into()
        .map_err(|_| "nonce must be 16 bytes".to_string())?;
    Ok(ReceiptPayload {
        provider,
        consumer,
        model_id: model_id.to_string(),
        tokens,
        nonce,
        ts_unix_ms,
    })
}

/// A complete, co-signed receipt.
#[derive(Debug, Clone)]
pub struct CoSignedReceipt {
    pub payload: ReceiptPayload,
    pub consumer_sig: Signature,
    pub provider_sig: Signature,
}

/// Fixed prefix of [`CoSignedReceipt::to_bytes`]: the two 32-byte keys, 16-byte nonce,
/// two u64s, two 64-byte signatures, and the u32 model-id length.
const RECEIPT_BLOB_FIXED: usize = 32 + 32 + 16 + 8 + 8 + 64 + 64 + 4;

impl CoSignedReceipt {
    /// Serialize to a self-describing byte blob for the persistent ledger (M2.3).
    ///
    /// Layout (little-endian): `provider[32] consumer[32] nonce[16] tokens:u64[8]
    /// ts_unix_ms:u64[8] consumer_sig[64] provider_sig[64] model_len:u32[4]
    /// model_id[model_len]`. This is the **full reversible record** — distinct from
    /// [`ReceiptPayload::canonical_bytes`], which is the domain-tagged *signed preimage*
    /// and carries no signatures.
    pub fn to_bytes(&self) -> Vec<u8> {
        let model = self.payload.model_id.as_bytes();
        let mut b = Vec::with_capacity(RECEIPT_BLOB_FIXED + model.len());
        b.extend_from_slice(self.payload.provider.as_bytes());
        b.extend_from_slice(self.payload.consumer.as_bytes());
        b.extend_from_slice(&self.payload.nonce);
        b.extend_from_slice(&self.payload.tokens.to_le_bytes());
        b.extend_from_slice(&self.payload.ts_unix_ms.to_le_bytes());
        b.extend_from_slice(&self.consumer_sig.to_bytes());
        b.extend_from_slice(&self.provider_sig.to_bytes());
        b.extend_from_slice(&(model.len() as u32).to_le_bytes());
        b.extend_from_slice(model);
        b
    }

    /// Reconstruct a receipt from [`to_bytes`](Self::to_bytes). Validates key bytes and
    /// the trailing length; signatures are reconstructed verbatim (call
    /// [`verify_receipt`] to re-check them). Returns `Err` on a malformed blob.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() < RECEIPT_BLOB_FIXED {
            return Err(format!(
                "receipt blob too short: {} < {RECEIPT_BLOB_FIXED}",
                data.len()
            ));
        }
        let mut key = [0u8; 32];
        key.copy_from_slice(&data[0..32]);
        let provider = VerifyingKey::from_bytes(&key).map_err(|e| format!("bad provider key: {e}"))?;
        key.copy_from_slice(&data[32..64]);
        let consumer = VerifyingKey::from_bytes(&key).map_err(|e| format!("bad consumer key: {e}"))?;
        let mut nonce = [0u8; 16];
        nonce.copy_from_slice(&data[64..80]);
        let mut u8x8 = [0u8; 8];
        u8x8.copy_from_slice(&data[80..88]);
        let tokens = u64::from_le_bytes(u8x8);
        u8x8.copy_from_slice(&data[88..96]);
        let ts_unix_ms = u64::from_le_bytes(u8x8);
        let mut sig = [0u8; 64];
        sig.copy_from_slice(&data[96..160]);
        let consumer_sig = Signature::from_bytes(&sig);
        sig.copy_from_slice(&data[160..224]);
        let provider_sig = Signature::from_bytes(&sig);
        let mut u8x4 = [0u8; 4];
        u8x4.copy_from_slice(&data[224..228]);
        let model_len = u32::from_le_bytes(u8x4) as usize;
        if data.len() != RECEIPT_BLOB_FIXED + model_len {
            return Err(format!(
                "receipt blob length mismatch: {} != {}",
                data.len(),
                RECEIPT_BLOB_FIXED + model_len
            ));
        }
        let model_id = std::str::from_utf8(&data[RECEIPT_BLOB_FIXED..RECEIPT_BLOB_FIXED + model_len])
            .map_err(|e| format!("bad model id utf8: {e}"))?
            .to_string();
        Ok(CoSignedReceipt {
            payload: ReceiptPayload { provider, consumer, model_id, tokens, nonce, ts_unix_ms },
            consumer_sig,
            provider_sig,
        })
    }
}

/// Why a receipt was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReceiptError {
    /// The consumer signature does not verify against the payload (tampered field,
    /// swapped consumer/provider key, or a corrupt signature).
    BadConsumerSig,
    /// The provider co-signature does not verify (corrupt, or signed by the wrong key).
    BadProviderSig,
    /// This nonce was already seen — a replay / double-submission.
    ReplayedNonce,
}

/// Consumer-side signature over the payload.
pub fn consumer_sign(payload: &ReceiptPayload, consumer_key: &SigningKey) -> Signature {
    consumer_key.sign(&payload.canonical_bytes())
}

/// Provider-side co-signature over (payload ‖ consumer_sig).
pub fn provider_cosign(
    payload: &ReceiptPayload,
    consumer_sig: &Signature,
    provider_key: &SigningKey,
) -> Signature {
    provider_key.sign(&cosign_bytes(payload, consumer_sig))
}

/// Build a complete co-signed receipt: the consumer signs, then the provider
/// counter-signs. (In the live flow these happen on two different machines; this
/// helper is for the originating side / tests.)
pub fn build_receipt(
    payload: ReceiptPayload,
    consumer_key: &SigningKey,
    provider_key: &SigningKey,
) -> CoSignedReceipt {
    let consumer_sig = consumer_sign(&payload, consumer_key);
    let provider_sig = provider_cosign(&payload, &consumer_sig, provider_key);
    CoSignedReceipt {
        payload,
        consumer_sig,
        provider_sig,
    }
}

/// Verify both signatures of a receipt (no replay check — see [`NonceTracker`]).
///
/// Returns `Ok(())` only when the consumer signature verifies against the payload
/// **and** the provider co-signature verifies against (payload ‖ consumer_sig),
/// each against the public key named in the payload.
pub fn verify_receipt(receipt: &CoSignedReceipt) -> Result<(), ReceiptError> {
    let payload_bytes = receipt.payload.canonical_bytes();
    receipt
        .payload
        .consumer
        .verify(&payload_bytes, &receipt.consumer_sig)
        .map_err(|_| ReceiptError::BadConsumerSig)?;
    let cosign = cosign_bytes(&receipt.payload, &receipt.consumer_sig);
    receipt
        .payload
        .provider
        .verify(&cosign, &receipt.provider_sig)
        .map_err(|_| ReceiptError::BadProviderSig)?;
    Ok(())
}

/// In-memory replay guard: a set of nonces already accepted.
///
/// Pure memory — the durable nonce store / ledger lands in M2.3. A real deployment
/// would also bound / persist this; here it is just the validation math.
#[derive(Debug, Default)]
pub struct NonceTracker {
    seen: HashSet<[u8; 16]>,
}

impl NonceTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Verify a receipt and record its nonce. Rejects a receipt whose signatures
    /// don't verify, or whose nonce was already accepted (replay). A receipt that
    /// fails signature verification does **not** consume its nonce.
    pub fn accept(&mut self, receipt: &CoSignedReceipt) -> Result<(), ReceiptError> {
        verify_receipt(receipt)?;
        if !self.seen.insert(receipt.payload.nonce) {
            return Err(ReceiptError::ReplayedNonce);
        }
        Ok(())
    }

    /// Insert a raw nonce as already-spent — used to **rehydrate** the guard from the
    /// persistent store on boot (the durable side records bare nonces, not whole
    /// receipts). Returns `true` if newly inserted, `false` if already present.
    pub fn mark_seen(&mut self, nonce: [u8; 16]) -> bool {
        self.seen.insert(nonce)
    }

    /// Whether this nonce has already been spent.
    pub fn contains(&self, nonce: &[u8; 16]) -> bool {
        self.seen.contains(nonce)
    }

    /// Number of distinct nonces accepted so far.
    pub fn len(&self) -> usize {
        self.seen.len()
    }

    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys() -> (SigningKey, SigningKey) {
        // Fixed seeds → deterministic keys (and, via RFC 8032, deterministic sigs).
        (SigningKey::from_bytes(&[7u8; 32]), SigningKey::from_bytes(&[9u8; 32]))
    }

    fn payload(consumer: &SigningKey, provider: &SigningKey, tokens: u64, nonce: [u8; 16]) -> ReceiptPayload {
        ReceiptPayload {
            provider: provider.verifying_key(),
            consumer: consumer.verifying_key(),
            model_id: "qwen3.5/2b/fp16/5632a1b48425a5ae".to_string(),
            tokens,
            nonce,
            ts_unix_ms: 1_700_000_000_000,
        }
    }

    fn corrupt(sig: &Signature) -> Signature {
        let mut b = sig.to_bytes();
        b[0] ^= 0xFF;
        Signature::from_bytes(&b)
    }

    #[test]
    fn valid_receipt_verifies() {
        let (c, p) = keys();
        let r = build_receipt(payload(&c, &p, 512, [1u8; 16]), &c, &p);
        assert_eq!(verify_receipt(&r), Ok(()));
    }

    #[test]
    fn tampered_token_count_is_rejected() {
        let (c, p) = keys();
        let mut r = build_receipt(payload(&c, &p, 512, [1u8; 16]), &c, &p);
        r.payload.tokens = 1_000_000; // inflate the claim
        assert_eq!(verify_receipt(&r), Err(ReceiptError::BadConsumerSig));
    }

    #[test]
    fn swapped_consumer_key_is_rejected() {
        let (c, p) = keys();
        let attacker = SigningKey::from_bytes(&[13u8; 32]);
        let mut r = build_receipt(payload(&c, &p, 512, [1u8; 16]), &c, &p);
        r.payload.consumer = attacker.verifying_key(); // pretend a different consumer
        assert_eq!(verify_receipt(&r), Err(ReceiptError::BadConsumerSig));
    }

    #[test]
    fn swapped_provider_key_is_rejected() {
        let (c, p) = keys();
        let attacker = SigningKey::from_bytes(&[13u8; 32]);
        let mut r = build_receipt(payload(&c, &p, 512, [1u8; 16]), &c, &p);
        // Provider is inside the consumer-signed payload, so swapping it breaks the
        // consumer signature first.
        r.payload.provider = attacker.verifying_key();
        assert!(verify_receipt(&r).is_err());
    }

    #[test]
    fn broken_consumer_signature_is_rejected() {
        let (c, p) = keys();
        let mut r = build_receipt(payload(&c, &p, 512, [1u8; 16]), &c, &p);
        r.consumer_sig = corrupt(&r.consumer_sig);
        assert_eq!(verify_receipt(&r), Err(ReceiptError::BadConsumerSig));
    }

    #[test]
    fn broken_provider_signature_is_rejected() {
        let (c, p) = keys();
        let mut r = build_receipt(payload(&c, &p, 512, [1u8; 16]), &c, &p);
        r.provider_sig = corrupt(&r.provider_sig);
        assert_eq!(verify_receipt(&r), Err(ReceiptError::BadProviderSig));
    }

    #[test]
    fn wrong_provider_cosigner_is_rejected() {
        let (c, p) = keys();
        let imposter = SigningKey::from_bytes(&[99u8; 32]);
        let pl = payload(&c, &p, 512, [1u8; 16]);
        let consumer_sig = consumer_sign(&pl, &c);
        // Co-signed by the imposter, but the payload still names the real provider.
        let provider_sig = provider_cosign(&pl, &consumer_sig, &imposter);
        let r = CoSignedReceipt { payload: pl, consumer_sig, provider_sig };
        assert_eq!(verify_receipt(&r), Err(ReceiptError::BadProviderSig));
    }

    #[test]
    fn replayed_nonce_is_rejected() {
        let (c, p) = keys();
        let r = build_receipt(payload(&c, &p, 512, [42u8; 16]), &c, &p);
        let mut tracker = NonceTracker::new();
        assert_eq!(tracker.accept(&r), Ok(()));
        assert_eq!(tracker.accept(&r), Err(ReceiptError::ReplayedNonce)); // double-submit
        assert_eq!(tracker.len(), 1);
    }

    #[test]
    fn distinct_nonces_are_both_accepted() {
        let (c, p) = keys();
        let mut tracker = NonceTracker::new();
        assert_eq!(tracker.accept(&build_receipt(payload(&c, &p, 10, [1u8; 16]), &c, &p)), Ok(()));
        assert_eq!(tracker.accept(&build_receipt(payload(&c, &p, 20, [2u8; 16]), &c, &p)), Ok(()));
        assert_eq!(tracker.len(), 2);
    }

    #[test]
    fn invalid_receipt_does_not_consume_its_nonce() {
        let (c, p) = keys();
        let mut r = build_receipt(payload(&c, &p, 512, [7u8; 16]), &c, &p);
        r.consumer_sig = corrupt(&r.consumer_sig);
        let mut tracker = NonceTracker::new();
        assert_eq!(tracker.accept(&r), Err(ReceiptError::BadConsumerSig));
        assert!(tracker.is_empty()); // a bad receipt must not burn the nonce
    }

    #[test]
    fn receipt_bytes_roundtrip() {
        // The persistent-ledger codec is reversible and signature-preserving.
        let (c, p) = keys();
        let r = build_receipt(payload(&c, &p, 512, [42u8; 16]), &c, &p);
        let blob = r.to_bytes();
        let back = CoSignedReceipt::from_bytes(&blob).unwrap();
        assert_eq!(back.payload.tokens, 512);
        assert_eq!(back.payload.nonce, [42u8; 16]);
        assert_eq!(back.payload.model_id, r.payload.model_id);
        assert_eq!(back.payload.provider.as_bytes(), r.payload.provider.as_bytes());
        assert_eq!(back.consumer_sig.to_bytes(), r.consumer_sig.to_bytes());
        assert_eq!(back.provider_sig.to_bytes(), r.provider_sig.to_bytes());
        // Signatures survive the round-trip — the decoded receipt still verifies.
        assert_eq!(verify_receipt(&back), Ok(()));
    }

    #[test]
    fn receipt_from_bytes_rejects_malformed() {
        assert!(CoSignedReceipt::from_bytes(b"too short").is_err());
        let (c, p) = keys();
        let mut blob = build_receipt(payload(&c, &p, 1, [1u8; 16]), &c, &p).to_bytes();
        blob.push(0xFF); // trailing byte → declared model_len no longer matches
        assert!(CoSignedReceipt::from_bytes(&blob).is_err());
    }

    #[test]
    fn nonce_tracker_mark_seen_and_contains() {
        let mut t = NonceTracker::new();
        assert!(!t.contains(&[5u8; 16]));
        assert!(t.mark_seen([5u8; 16])); // newly inserted
        assert!(!t.mark_seen([5u8; 16])); // already present
        assert!(t.contains(&[5u8; 16]));
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn signatures_are_deterministic_golden() {
        // ed25519 (RFC 8032) is deterministic: a fixed payload + keys → fixed sigs.
        // This pins the canonical encoding — changing canonical_bytes() flips these.
        let (c, p) = keys();
        let pl = payload(&c, &p, 512, [42u8; 16]);
        let r1 = build_receipt(pl.clone(), &c, &p);
        let r2 = build_receipt(pl, &c, &p);
        assert_eq!(r1.consumer_sig.to_bytes(), r2.consumer_sig.to_bytes());
        assert_eq!(r1.provider_sig.to_bytes(), r2.provider_sig.to_bytes());
        assert_eq!(verify_receipt(&r1), Ok(()));
    }
}
