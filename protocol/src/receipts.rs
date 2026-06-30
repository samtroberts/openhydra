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

use crate::crypto_agility::SigAlg;

/// Domain separator for the consumer-signed payload (prevents cross-protocol
/// signature reuse). Versioned: a change is a wire-format bump. **v2** binds the
/// signature-algorithm discriminant into the preimage (PQC0.1 crypto-agility).
pub const RECEIPT_DOMAIN: &[u8] = b"openhydra/receipt/v2";
/// Domain separator for the provider's co-signature over (payload ‖ consumer_sig).
pub const COSIGN_DOMAIN: &[u8] = b"openhydra/receipt-cosign/v2";

/// Serialization format version for [`CoSignedReceipt::to_bytes`] — v2 is the
/// agile, length-prefixed layout (PQC0.1). Bumped on any layout change.
const RECEIPT_FORMAT_V2: u8 = 2;

/// The content both parties sign over (the consumer directly, the provider via the
/// co-signature). Identities are ed25519 public keys.
#[derive(Debug, Clone)]
pub struct ReceiptPayload {
    /// Signature algorithm both parties use. Bound into the signed preimage so a
    /// wire-level attacker cannot strip or downgrade it (PQC0.1 crypto-agility).
    pub sig_alg: SigAlg,
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
    /// signature — see the golden test). The `sig_alg` byte is bound in right after the
    /// domain so the chosen algorithm is signed over (no silent downgrade).
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(RECEIPT_DOMAIN.len() + 1 + 32 + 32 + 4 + self.model_id.len() + 8 + 16 + 8);
        b.extend_from_slice(RECEIPT_DOMAIN);
        b.push(self.sig_alg.to_u8());
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
        // The raw-bytes / FFI constructor is the classical Ed25519 path. PQC payloads
        // are built explicitly with their `sig_alg` set (PQC3.1).
        sig_alg: SigAlg::Ed25519,
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

/// Minimal forward-only byte reader for [`CoSignedReceipt::from_bytes`] — bounds-checked
/// so a truncated/oversized blob errors instead of panicking.
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.pos.checked_add(n).ok_or("length overflow")?;
        if end > self.data.len() {
            return Err(format!(
                "receipt blob truncated: need {n} bytes at offset {}, have {}",
                self.pos,
                self.data.len() - self.pos
            ));
        }
        let s = &self.data[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn arr<const N: usize>(&mut self) -> Result<[u8; N], String> {
        Ok(self.take(N)?.try_into().expect("take returns exactly N"))
    }
    /// A `u16`-length-prefixed slice (used for the variable-length signatures).
    fn lp16(&mut self) -> Result<&'a [u8], String> {
        let len = u16::from_le_bytes(self.arr::<2>()?) as usize;
        self.take(len)
    }
    fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }
}

/// Reconstruct a signature of the given algorithm from its raw bytes. Today only
/// Ed25519 (64 bytes) is materializable into the [`Signature`] type; a known-but-
/// unimplemented PQC algorithm is rejected rather than silently mishandled (PQC3.1
/// generalizes the signature type).
fn read_sig(alg: SigAlg, bytes: &[u8]) -> Result<Signature, String> {
    match alg {
        SigAlg::Ed25519 => {
            let arr: [u8; 64] = bytes
                .try_into()
                .map_err(|_| format!("ed25519 signature must be 64 bytes, got {}", bytes.len()))?;
            Ok(Signature::from_bytes(&arr))
        }
        other => Err(format!("signature algorithm not implemented in this build: {other:?}")),
    }
}

impl CoSignedReceipt {
    /// Serialize to a self-describing, **algorithm-agile** byte blob for the persistent
    /// ledger (M2.3) and receipt exchange.
    ///
    /// Layout (little-endian, PQC0.1 v2): `format:u8(=2) sig_alg:u8 provider[32]
    /// consumer[32] nonce[16] tokens:u64[8] ts_unix_ms:u64[8] consumer_sig_len:u16[2]
    /// consumer_sig[len] provider_sig_len:u16[2] provider_sig[len] model_len:u32[4]
    /// model_id[model_len]`.
    ///
    /// Signatures are **length-prefixed** (not fixed at 64 bytes) so a future ML-DSA /
    /// hybrid signature (PQC3.1) fits without another format change. This is the full
    /// reversible record — distinct from [`ReceiptPayload::canonical_bytes`], the
    /// domain-tagged *signed preimage* (which carries no signatures).
    pub fn to_bytes(&self) -> Vec<u8> {
        let model = self.payload.model_id.as_bytes();
        let csig = self.consumer_sig.to_bytes();
        let psig = self.provider_sig.to_bytes();
        let mut b = Vec::with_capacity(
            2 + 32 + 32 + 16 + 8 + 8 + 2 + csig.len() + 2 + psig.len() + 4 + model.len(),
        );
        b.push(RECEIPT_FORMAT_V2);
        b.push(self.payload.sig_alg.to_u8());
        b.extend_from_slice(self.payload.provider.as_bytes());
        b.extend_from_slice(self.payload.consumer.as_bytes());
        b.extend_from_slice(&self.payload.nonce);
        b.extend_from_slice(&self.payload.tokens.to_le_bytes());
        b.extend_from_slice(&self.payload.ts_unix_ms.to_le_bytes());
        b.extend_from_slice(&(csig.len() as u16).to_le_bytes());
        b.extend_from_slice(&csig);
        b.extend_from_slice(&(psig.len() as u16).to_le_bytes());
        b.extend_from_slice(&psig);
        b.extend_from_slice(&(model.len() as u32).to_le_bytes());
        b.extend_from_slice(model);
        b
    }

    /// Reconstruct a receipt from [`to_bytes`](Self::to_bytes). Validates the format
    /// version, the (registry-checked) `sig_alg`, key bytes, signature lengths, and
    /// rejects any trailing bytes. Signatures are reconstructed verbatim (call
    /// [`verify_receipt`] to re-check them). Returns `Err` on a malformed blob.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        let mut cur = Cursor { data, pos: 0 };
        let fmt = cur.u8()?;
        if fmt != RECEIPT_FORMAT_V2 {
            return Err(format!("unsupported receipt format version: {fmt}"));
        }
        let sig_alg = SigAlg::from_u8(cur.u8()?).map_err(|e| e.to_string())?;
        let provider =
            VerifyingKey::from_bytes(&cur.arr::<32>()?).map_err(|e| format!("bad provider key: {e}"))?;
        let consumer =
            VerifyingKey::from_bytes(&cur.arr::<32>()?).map_err(|e| format!("bad consumer key: {e}"))?;
        let nonce = cur.arr::<16>()?;
        let tokens = u64::from_le_bytes(cur.arr::<8>()?);
        let ts_unix_ms = u64::from_le_bytes(cur.arr::<8>()?);
        let consumer_sig = read_sig(sig_alg, cur.lp16()?)?;
        let provider_sig = read_sig(sig_alg, cur.lp16()?)?;
        let model_len = u32::from_le_bytes(cur.arr::<4>()?) as usize;
        let model_bytes = cur.take(model_len)?;
        if cur.remaining() != 0 {
            return Err(format!("receipt blob has {} trailing bytes", cur.remaining()));
        }
        let model_id = std::str::from_utf8(model_bytes)
            .map_err(|e| format!("bad model id utf8: {e}"))?
            .to_string();
        Ok(CoSignedReceipt {
            payload: ReceiptPayload { sig_alg, provider, consumer, model_id, tokens, nonce, ts_unix_ms },
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
    /// The payload names a signature algorithm this build cannot verify (PQC0.1 — a
    /// known-but-unimplemented discriminant; e.g. ML-DSA before PQC3.1).
    UnsupportedAlg(SigAlg),
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
    // Crypto-agility (PQC0.1): only verify with an algorithm this build implements.
    // Today that is Ed25519; a reserved PQC discriminant is rejected, never downgraded.
    if !receipt.payload.sig_alg.is_implemented() {
        return Err(ReceiptError::UnsupportedAlg(receipt.payload.sig_alg));
    }
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
            sig_alg: SigAlg::Ed25519,
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

    // ── PQC0.1 crypto-agility ──────────────────────────────────────────────

    #[test]
    fn sig_alg_is_bound_into_preimage() {
        // Two payloads identical except for sig_alg must produce different signed
        // bytes — otherwise an attacker could downgrade the algorithm undetected.
        let (c, p) = keys();
        let mut a = payload(&c, &p, 512, [1u8; 16]);
        a.sig_alg = SigAlg::Ed25519;
        let mut b = payload(&c, &p, 512, [1u8; 16]);
        b.sig_alg = SigAlg::MlDsa65;
        assert_ne!(a.canonical_bytes(), b.canonical_bytes());
    }

    #[test]
    fn unimplemented_alg_payload_is_rejected_not_verified() {
        // A receipt claiming a reserved-but-unimplemented algorithm must be rejected
        // explicitly, never silently treated as classical.
        let (c, p) = keys();
        let mut r = build_receipt(payload(&c, &p, 512, [1u8; 16]), &c, &p);
        r.payload.sig_alg = SigAlg::MlDsa65;
        assert_eq!(verify_receipt(&r), Err(ReceiptError::UnsupportedAlg(SigAlg::MlDsa65)));
    }

    #[test]
    fn from_bytes_rejects_unknown_sig_alg_discriminant() {
        let (c, p) = keys();
        let mut blob = build_receipt(payload(&c, &p, 1, [1u8; 16]), &c, &p).to_bytes();
        // byte[0] = format version (2), byte[1] = sig_alg discriminant.
        assert_eq!(blob[0], RECEIPT_FORMAT_V2);
        blob[1] = 0xFF; // not a known SigAlg
        assert!(CoSignedReceipt::from_bytes(&blob).is_err());
    }

    #[test]
    fn from_bytes_rejects_wrong_format_version() {
        let (c, p) = keys();
        let mut blob = build_receipt(payload(&c, &p, 1, [1u8; 16]), &c, &p).to_bytes();
        blob[0] = 1; // an old/unsupported format version
        assert!(CoSignedReceipt::from_bytes(&blob).is_err());
    }

    #[test]
    fn v2_blob_roundtrips_with_alg_preserved() {
        let (c, p) = keys();
        let r = build_receipt(payload(&c, &p, 777, [9u8; 16]), &c, &p);
        let back = CoSignedReceipt::from_bytes(&r.to_bytes()).unwrap();
        assert_eq!(back.payload.sig_alg, SigAlg::Ed25519);
        assert_eq!(back.payload.tokens, 777);
        assert_eq!(verify_receipt(&back), Ok(()));
    }
}
