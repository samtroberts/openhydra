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
use crate::payout::SettlementAsset;

/// Domain separator for the consumer-signed payload (prevents cross-protocol
/// signature reuse). Versioned: a change is a wire-format bump. **v2** binds the
/// signature-algorithm discriminant into the preimage (PQC0.1 crypto-agility).
pub const RECEIPT_DOMAIN: &[u8] = b"openhydra/receipt/v2";
/// Domain separator for the provider's co-signature over (payload ‖ consumer_sig).
pub const COSIGN_DOMAIN: &[u8] = b"openhydra/receipt-cosign/v2";

/// Serialization format version for [`CoSignedReceipt::to_bytes`] — v2 is the
/// agile, length-prefixed layout (PQC0.1). Bumped on any layout change.
const RECEIPT_FORMAT_V2: u8 = 2;
/// v3 (credit-tiering Phase 2) is v2 plus a `completion_tokens: u64` field after `ts_unix_ms`.
/// A receipt carrying the prompt/completion split (`Some`) serializes as v3; a legacy receipt
/// with no split (`None`) still serializes as **v2** (byte-identical), and a v2 blob decodes to
/// `None` — so old stored receipts round-trip unchanged and new ones are self-describing.
const RECEIPT_FORMAT_V3: u8 = 3;
/// v4 (crypto-rail, PAYMENTS_PLAN_v1 step 3) is v3 plus a monotonic `cumulative_amount: u64` — the
/// running authenticity-gated voucher total this receipt attests. A receipt carrying it serializes
/// as v4 (a strict superset of the v3 block); a receipt without it stays byte-identical v2/v3, and a
/// v2/v3 blob decodes to `None`.
const RECEIPT_FORMAT_V4: u8 = 4;
/// v5 (crypto-rail denomination binding) is v4 plus the **settlement asset** + the **reference-price
/// peg** the `cumulative_amount` was converted with. A receipt carrying either serializes as v5 (a
/// strict superset of the v4 block); a receipt without them stays byte-identical v2/v3/v4, and a
/// v2/v3/v4 blob decodes them to `None`. Binding these makes a consumer/provider asset-or-peg
/// disagreement non-repudiable rather than a silently mis-denominated claim.
const RECEIPT_FORMAT_V5: u8 = 5;

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
    /// Tokens the consumer acknowledges receiving (the **total**: prompt + completion).
    pub tokens: u64,
    /// 128-bit nonce — unique per receipt; prevents double-counting / replay.
    pub nonce: [u8; 16],
    /// Receipt timestamp (unix ms).
    pub ts_unix_ms: u64,
    /// **Completion (output) tokens** of the `tokens` total, so credit can weight decode above
    /// prefill (`prompt = tokens − completion`; credit-tiering Phase 2). `None` = the split is
    /// **unknown** — a legacy receipt signed before this field existed (decoded from a v2 blob), or
    /// a path that doesn't carry it; Phase 3 weights those at a neutral blended in/out rate.
    /// `Some(c)` is **bound into the signed preimage** ([`canonical_bytes`](Self::canonical_bytes))
    /// so it can't be stripped or altered without invalidating both signatures. Prospective: real
    /// completions are ≥ 1, so `Some(0)` (a genuine empty completion) stays distinct from `None`.
    pub completion_tokens: Option<u64>,
    /// The **canonical** model id (`family/params/quant/template_hash`) both parties agree was
    /// served, so the credit ledger can look up the cost-tier weight from the *signed* receipt
    /// rather than a per-node guess (credit-tiering — receipt-carries-canonical). `model_id` above
    /// stays the raw engine handle the consumer requested (used for the B-S1 serve-commitment
    /// binding); this is the precise identity for **weighting**. `""` = unknown/legacy (a v2
    /// receipt, or a provider that advertised no canonical id) → the ledger falls back to the
    /// conservative nano tier. Bound into the signed preimage (only when non-empty, so a legacy
    /// receipt stays byte-identical) — a provider can't inflate its earned tier by forging it.
    pub canonical_model_id: String,
    /// The **cumulative** authenticity-gated amount (gross: provider net + protocol fee), in the
    /// **settlement asset's smallest unit** (USDC micro-units — converted from credit units at sign
    /// time via the reference-price peg, `SOLANA_SETTLEMENT_PROGRAM_SPEC` §2, so the on-chain program
    /// settles it directly with no oracle), that this consumer has authorized to this provider for
    /// this model across all serves in the pair — the monotonic crypto-rail voucher total
    /// ([`crate::payout::GatedVoucher`]) the on-chain claim settles against (PAYMENTS_PLAN_v1 crypto
    /// step 3). `Some` binds it into the signed preimage
    /// (tag 3) so neither party can alter the claim amount without invalidating both signatures;
    /// `None` = a receipt carrying no crypto-rail voucher (every non-crypto serve), which stays
    /// **byte-identical** to v2/v3. No real value moves at receipt time; this is the signed claim
    /// the (later, legally-gated) on-chain settlement pays out against.
    pub cumulative_amount: Option<u64>,
    /// The **settlement asset** the `cumulative_amount` is denominated in (`RECEIPT_FORMAT_V5`, tag 4).
    /// `Some` binds it into the signed preimage so a consumer and provider that disagree on the asset
    /// can't co-sign a claim whose units they read differently — the on-chain program settles the
    /// cumulative in exactly this asset. `None` = a receipt carrying no crypto-rail voucher (every
    /// non-crypto serve), byte-identical to v2/v3/v4. Set together with `cumulative_amount` +
    /// `peg_micros_per_credit` by the crypto rail.
    pub settlement_asset: Option<SettlementAsset>,
    /// The **reference-price peg** (settlement-asset micro-units per credit unit) the consumer used to
    /// convert credit units → `cumulative_amount` (`RECEIPT_FORMAT_V5`, tag 5). Bound into the signed
    /// preimage so the peg is non-repudiable: a provider whose configured peg differs rejects the
    /// receipt before co-signing (the amount would otherwise be settled at a rate the provider never
    /// agreed to). `None` for non-crypto receipts (byte-identical to v2/v3/v4).
    pub peg_micros_per_credit: Option<u64>,
}

impl ReceiptPayload {
    /// Canonical bytes the **consumer** signs. Deterministic: a fixed payload always
    /// produces the same bytes (and, with ed25519's RFC-8032 determinism, the same
    /// signature — see the golden test). The `sig_alg` byte is bound in right after the
    /// domain so the chosen algorithm is signed over (no silent downgrade).
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(RECEIPT_DOMAIN.len() + 1 + 32 + 32 + 4 + self.model_id.len() + 8 + 16 + 8 + 9);
        b.extend_from_slice(RECEIPT_DOMAIN);
        b.push(self.sig_alg.to_u8());
        b.extend_from_slice(self.provider.as_bytes());
        b.extend_from_slice(self.consumer.as_bytes());
        b.extend_from_slice(&(self.model_id.len() as u32).to_le_bytes());
        b.extend_from_slice(self.model_id.as_bytes());
        b.extend_from_slice(&self.tokens.to_le_bytes());
        b.extend_from_slice(&self.nonce);
        b.extend_from_slice(&self.ts_unix_ms.to_le_bytes());
        // Phase 2: the prompt/completion split, bound in *only when known* so it can't be stripped
        // or forged without breaking both signatures. A legacy receipt (`None`) appends nothing, so
        // its preimage is **byte-identical** to the pre-Phase-2 format — old signatures still verify
        // and the golden vectors are unchanged. A `1` tag before the value keeps `Some(c)`
        // unambiguous against the empty (`None`) tail (an attacker can't downgrade a signed
        // `Some` to `None`, nor promote a `None` to `Some`, without a signature that doesn't exist).
        if let Some(completion) = self.completion_tokens {
            b.push(1u8);
            b.extend_from_slice(&completion.to_le_bytes());
        }
        // The canonical model id, bound in only when present — same downgrade-proofing as the
        // completion split: an empty canonical appends nothing (legacy byte-identical), and a `2`
        // tag + length-prefixed id keeps it unforgeable and unambiguous against the empty tail.
        if !self.canonical_model_id.is_empty() {
            b.push(2u8);
            b.extend_from_slice(&(self.canonical_model_id.len() as u32).to_le_bytes());
            b.extend_from_slice(self.canonical_model_id.as_bytes());
        }
        // The crypto-rail cumulative voucher amount (tag 3), bound in only when present — same
        // downgrade-proofing as the two fields above: `None` appends nothing (legacy byte-identical),
        // a `3` tag + u64 keeps it unforgeable and unambiguous against the empty tail. Tags are
        // appended in a fixed order (1 completion, 2 canonical, 3 cumulative) so the preimage is
        // deterministic.
        if let Some(cumulative) = self.cumulative_amount {
            b.push(3u8);
            b.extend_from_slice(&cumulative.to_le_bytes());
        }
        // The settlement asset (tag 4) + the reference-price peg (tag 5) the cumulative was denominated
        // with (v5) — same downgrade-proofing: `None` appends nothing (legacy byte-identical), and a
        // tag + value keeps each unforgeable and unambiguous against the empty tail. Fixed order
        // (…3 cumulative, 4 asset, 5 peg) keeps the preimage deterministic. Binding these means a peer
        // that co-signs a claim implicitly agrees to its asset + peg — a mismatch can't be signed over.
        if let Some(asset) = self.settlement_asset {
            b.push(4u8);
            b.push(asset.to_u8());
        }
        if let Some(peg) = self.peg_micros_per_credit {
            b.push(5u8);
            b.extend_from_slice(&peg.to_le_bytes());
        }
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
#[allow(clippy::too_many_arguments)]
pub fn payload_from_bytes(
    provider_pub: &[u8],
    consumer_pub: &[u8],
    model_id: &str,
    tokens: u64,
    nonce: &[u8],
    ts_unix_ms: u64,
    completion_tokens: Option<u64>,
    canonical_model_id: &str,
    cumulative_amount: Option<u64>,
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
        completion_tokens,
        canonical_model_id: canonical_model_id.to_string(),
        cumulative_amount,
        // The crypto-rail denomination (v5) isn't a raw-bytes/FFI concern — the caller (consumer
        // settle path) stamps `settlement_asset` + `peg_micros_per_credit` explicitly after building,
        // and the wire decoder sets them from the framing trailers. Default to absent here.
        settlement_asset: None,
        peg_micros_per_credit: None,
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
    /// model_id[model_len]`. **v3** (credit-tiering) inserts, right after `ts_unix_ms`, an optional
    /// `completion_tokens` and the `canonical_model_id`: `has_completion:u8[1]
    /// completion_tokens:u64[8]?(only if has_completion=1) canonical_len:u32[4]
    /// canonical_model_id[canonical_len]`. A receipt serializes as v3 when it carries **either** the
    /// split or a canonical id; a receipt with neither stays byte-identical v2, so old stored blobs
    /// round-trip unchanged.
    ///
    /// Signatures are **length-prefixed** (not fixed at 64 bytes) so a future ML-DSA /
    /// hybrid signature (PQC3.1) fits without another format change. This is the full
    /// reversible record — distinct from [`ReceiptPayload::canonical_bytes`], the
    /// domain-tagged *signed preimage* (which carries no signatures).
    pub fn to_bytes(&self) -> Vec<u8> {
        let model = self.payload.model_id.as_bytes();
        let canonical = self.payload.canonical_model_id.as_bytes();
        let csig = self.consumer_sig.to_bytes();
        let psig = self.provider_sig.to_bytes();
        // Version escalation (each a strict superset): v5 iff a settlement asset/peg (crypto-rail
        // denomination binding), else v4 iff a cumulative voucher amount, else v3 iff the split or a
        // canonical id, else byte-identical legacy v2. Each higher version still carries every lower
        // block so old fields parse; only the tail differs.
        let is_v5 = self.payload.settlement_asset.is_some() || self.payload.peg_micros_per_credit.is_some();
        let is_v4 = self.payload.cumulative_amount.is_some();
        let is_v3 = self.payload.completion_tokens.is_some() || !canonical.is_empty();
        let mut b = Vec::with_capacity(
            2 + 32 + 32 + 16 + 8 + 8 + 1 + 8 + 4 + canonical.len() + 1 + 8 + 1 + 1 + 1 + 8 + 2 + csig.len() + 2 + psig.len() + 4 + model.len(),
        );
        b.push(if is_v5 {
            RECEIPT_FORMAT_V5
        } else if is_v4 {
            RECEIPT_FORMAT_V4
        } else if is_v3 {
            RECEIPT_FORMAT_V3
        } else {
            RECEIPT_FORMAT_V2
        });
        b.push(self.payload.sig_alg.to_u8());
        b.extend_from_slice(self.payload.provider.as_bytes());
        b.extend_from_slice(self.payload.consumer.as_bytes());
        b.extend_from_slice(&self.payload.nonce);
        b.extend_from_slice(&self.payload.tokens.to_le_bytes());
        b.extend_from_slice(&self.payload.ts_unix_ms.to_le_bytes());
        // The v3 field block is present in v3, v4, and v5 blobs.
        if is_v3 || is_v4 || is_v5 {
            match self.payload.completion_tokens {
                Some(completion) => {
                    b.push(1u8);
                    b.extend_from_slice(&completion.to_le_bytes());
                }
                None => b.push(0u8),
            }
            b.extend_from_slice(&(canonical.len() as u32).to_le_bytes());
            b.extend_from_slice(canonical);
        }
        if is_v5 {
            // v5 tail: cumulative + asset + peg, each presence-flagged so the crypto-rail denomination
            // fields are self-describing (v5 is chosen on asset/peg, which are logically paired with a
            // cumulative but encoded independently for robustness).
            match self.payload.cumulative_amount {
                Some(c) => {
                    b.push(1u8);
                    b.extend_from_slice(&c.to_le_bytes());
                }
                None => b.push(0u8),
            }
            match self.payload.settlement_asset {
                Some(a) => {
                    b.push(1u8);
                    b.push(a.to_u8());
                }
                None => b.push(0u8),
            }
            match self.payload.peg_micros_per_credit {
                Some(p) => {
                    b.push(1u8);
                    b.extend_from_slice(&p.to_le_bytes());
                }
                None => b.push(0u8),
            }
        } else if let Some(cumulative) = self.payload.cumulative_amount {
            // v4 tail: the cumulative voucher amount (always `Some` when `is_v4`), positional.
            b.extend_from_slice(&cumulative.to_le_bytes());
        }
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
        if fmt != RECEIPT_FORMAT_V2
            && fmt != RECEIPT_FORMAT_V3
            && fmt != RECEIPT_FORMAT_V4
            && fmt != RECEIPT_FORMAT_V5
        {
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
        // v3, v4 and v5 carry the completion split + canonical id here; v2 predates both (→ None /
        // empty, a legacy receipt).
        let (completion_tokens, canonical_model_id) =
            if fmt == RECEIPT_FORMAT_V3 || fmt == RECEIPT_FORMAT_V4 || fmt == RECEIPT_FORMAT_V5 {
                let completion = match cur.u8()? {
                    0 => None,
                    1 => Some(u64::from_le_bytes(cur.arr::<8>()?)),
                    other => return Err(format!("bad completion presence flag: {other}")),
                };
                let clen = u32::from_le_bytes(cur.arr::<4>()?) as usize;
                let cbytes = cur.take(clen)?;
                let canonical = std::str::from_utf8(cbytes)
                    .map_err(|e| format!("bad canonical id utf8: {e}"))?
                    .to_string();
                (completion, canonical)
            } else {
                (None, String::new())
            };
        // The crypto-rail tail. v4 is a bare positional cumulative `u64`. v5 is a strict superset with
        // the cumulative + settlement asset + peg, each presence-flagged (so the denomination fields
        // are self-describing).
        let (cumulative_amount, settlement_asset, peg_micros_per_credit) = if fmt == RECEIPT_FORMAT_V5 {
            let cumulative = match cur.u8()? {
                0 => None,
                1 => Some(u64::from_le_bytes(cur.arr::<8>()?)),
                other => return Err(format!("bad cumulative presence flag: {other}")),
            };
            let asset = match cur.u8()? {
                0 => None,
                1 => Some(SettlementAsset::from_u8(cur.u8()?)?),
                other => return Err(format!("bad settlement-asset presence flag: {other}")),
            };
            let peg = match cur.u8()? {
                0 => None,
                1 => Some(u64::from_le_bytes(cur.arr::<8>()?)),
                other => return Err(format!("bad peg presence flag: {other}")),
            };
            (cumulative, asset, peg)
        } else if fmt == RECEIPT_FORMAT_V4 {
            (Some(u64::from_le_bytes(cur.arr::<8>()?)), None, None)
        } else {
            (None, None, None)
        };
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
            payload: ReceiptPayload {
                sig_alg, provider, consumer, model_id, tokens, nonce, ts_unix_ms, completion_tokens,
                canonical_model_id, cumulative_amount, settlement_asset, peg_micros_per_credit,
            },
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
            completion_tokens: None,
            canonical_model_id: String::new(),
            cumulative_amount: None,
            settlement_asset: None,
            peg_micros_per_credit: None,
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
        // FROZEN backward-compat vectors (review crypto-F4): this fixed payload is a *legacy* receipt
        // (completion_tokens=None, canonical_model_id="") so its preimage is the pre-credit-tiering
        // v2 format. Pinning the exact signatures means any future change to `canonical_bytes()` that
        // would break an already-signed on-disk receipt fails THIS test loudly. Regenerate ONLY with
        // a deliberate, documented wire-format bump.
        let hex = |b: &[u8]| b.iter().map(|x| format!("{x:02x}")).collect::<String>();
        assert_eq!(
            hex(&r1.consumer_sig.to_bytes()),
            "bdfcb72a888d254dde366d5025b712aa41fdeca7ef2200ee296940f917be0aa65c5d16c556c587e5a69eb64ef51731e89d85587372a0fd84d9b0700430bf6b01",
            "legacy consumer signature changed — a credit-tiering field leaked into the v2 preimage?"
        );
        assert_eq!(
            hex(&r1.provider_sig.to_bytes()),
            "8596fd54d823cc2e39f23c485d6a68e1764fd5e3ff54c5f90544d54b0edb0ab08f45bf577725a44e9b8b9d22ac7c83606c0ea114f9b1b23291cbfc89f2f46907",
            "legacy provider co-signature changed — v2 cosign preimage drift"
        );
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

    // ── Phase 2: completion_tokens split ───────────────────────────────────

    fn split_payload(consumer: &SigningKey, provider: &SigningKey, tokens: u64, completion: u64) -> ReceiptPayload {
        let mut pl = payload(consumer, provider, tokens, [3u8; 16]);
        pl.completion_tokens = Some(completion);
        pl
    }

    #[test]
    fn a_no_split_receipt_is_byte_identical_to_the_legacy_format() {
        // Backward compat: a `None` receipt appends nothing to the preimage and serializes as v2,
        // so old signatures verify and old blobs are unchanged. (The golden-vectors test above
        // pins the exact bytes; here we assert the shape directly.)
        let (c, p) = keys();
        let r = build_receipt(payload(&c, &p, 512, [1u8; 16]), &c, &p); // completion_tokens: None
        assert_eq!(r.to_bytes()[0], RECEIPT_FORMAT_V2, "no-split receipt stays v2 on the wire");
        assert_eq!(verify_receipt(&r), Ok(()));
    }

    #[test]
    fn completion_split_binds_into_preimage_and_verifies() {
        // Two payloads identical except for completion_tokens must sign different bytes — the split
        // is committed, so it can't be altered without breaking the signature.
        let (c, p) = keys();
        let none = payload(&c, &p, 100, [3u8; 16]);
        let some = split_payload(&c, &p, 100, 80);
        assert_ne!(none.canonical_bytes(), some.canonical_bytes(), "the split must change the preimage");
        assert_eq!(some.canonical_bytes().len(), none.canonical_bytes().len() + 9, "tag byte + u64");
        // Two different splits also differ.
        assert_ne!(split_payload(&c, &p, 100, 80).canonical_bytes(), split_payload(&c, &p, 100, 20).canonical_bytes());
        // A receipt carrying the split still co-signs and verifies.
        let r = build_receipt(some, &c, &p);
        assert_eq!(verify_receipt(&r), Ok(()));
    }

    #[test]
    fn v3_blob_roundtrips_preserving_the_split() {
        let (c, p) = keys();
        let r = build_receipt(split_payload(&c, &p, 512, 300), &c, &p);
        let blob = r.to_bytes();
        assert_eq!(blob[0], RECEIPT_FORMAT_V3, "a receipt with the split serializes as v3");
        let back = CoSignedReceipt::from_bytes(&blob).unwrap();
        assert_eq!(back.payload.completion_tokens, Some(300));
        assert_eq!(back.payload.tokens, 512);
        assert_eq!(verify_receipt(&back), Ok(()), "signatures survive the v3 round-trip");
        // Some(0) — a genuine empty completion — stays distinct from None (legacy/unknown).
        let z = build_receipt(split_payload(&c, &p, 5, 0), &c, &p);
        assert_eq!(z.to_bytes()[0], RECEIPT_FORMAT_V3);
        assert_eq!(CoSignedReceipt::from_bytes(&z.to_bytes()).unwrap().payload.completion_tokens, Some(0));
    }

    // ── v4: crypto-rail cumulative voucher amount ──────────────────────────

    fn cumulative_payload(consumer: &SigningKey, provider: &SigningKey, tokens: u64, cumulative: u64) -> ReceiptPayload {
        let mut pl = payload(consumer, provider, tokens, [4u8; 16]);
        pl.cumulative_amount = Some(cumulative);
        pl
    }

    #[test]
    fn a_no_cumulative_receipt_stays_v2_v3_byte_identical() {
        // A receipt without a cumulative voucher never serializes as v4 — a plain one stays v2, a
        // split/canonical one stays v3 — so legacy blobs + the golden signatures are untouched.
        let (c, p) = keys();
        let plain = build_receipt(payload(&c, &p, 512, [1u8; 16]), &c, &p);
        assert_eq!(plain.to_bytes()[0], RECEIPT_FORMAT_V2);
        let split = build_receipt(split_payload(&c, &p, 512, 300), &c, &p);
        assert_eq!(split.to_bytes()[0], RECEIPT_FORMAT_V3);
    }

    #[test]
    fn cumulative_binds_into_preimage_and_verifies() {
        // Two payloads identical except for the cumulative amount must sign different bytes — the
        // claim amount is committed, so neither party can alter it without breaking the signature.
        let (c, p) = keys();
        let none = payload(&c, &p, 100, [4u8; 16]);
        let some = cumulative_payload(&c, &p, 100, 9_600);
        assert_ne!(none.canonical_bytes(), some.canonical_bytes(), "the cumulative must change the preimage");
        assert_eq!(some.canonical_bytes().len(), none.canonical_bytes().len() + 9, "tag byte + u64");
        // Two different cumulative totals also differ.
        assert_ne!(
            cumulative_payload(&c, &p, 100, 9_600).canonical_bytes(),
            cumulative_payload(&c, &p, 100, 19_200).canonical_bytes()
        );
        let r = build_receipt(some, &c, &p);
        assert_eq!(verify_receipt(&r), Ok(()));
    }

    #[test]
    fn v4_blob_roundtrips_preserving_cumulative_and_the_v3_fields() {
        let (c, p) = keys();
        // A v4 receipt that ALSO carries the split + canonical — v4 is a strict superset of v3.
        let mut pl = cumulative_payload(&c, &p, 512, 48_000);
        pl.completion_tokens = Some(300);
        pl.canonical_model_id = "qwen3.5/2b/fp16/5632a1b48425a5ae".to_string();
        let r = build_receipt(pl, &c, &p);
        let blob = r.to_bytes();
        assert_eq!(blob[0], RECEIPT_FORMAT_V4, "a receipt with a cumulative serializes as v4");
        let back = CoSignedReceipt::from_bytes(&blob).unwrap();
        assert_eq!(back.payload.cumulative_amount, Some(48_000));
        assert_eq!(back.payload.completion_tokens, Some(300));
        assert_eq!(back.payload.canonical_model_id, "qwen3.5/2b/fp16/5632a1b48425a5ae");
        assert_eq!(back.payload.tokens, 512);
        assert_eq!(verify_receipt(&back), Ok(()), "signatures survive the v4 round-trip");
        // A v4 receipt with NO split/canonical still round-trips (the v3 block is empty but present).
        let bare = build_receipt(cumulative_payload(&c, &p, 5, 1), &c, &p);
        assert_eq!(bare.to_bytes()[0], RECEIPT_FORMAT_V4);
        let back = CoSignedReceipt::from_bytes(&bare.to_bytes()).unwrap();
        assert_eq!(back.payload.cumulative_amount, Some(1));
        assert_eq!(back.payload.completion_tokens, None);
        assert!(back.payload.canonical_model_id.is_empty());
    }

    #[test]
    fn legacy_v2_blob_decodes_as_no_split() {
        // A stored v2 blob (no completion field) must decode to None, not error — this is what
        // lets pre-Phase-2 receipts on live nodes still load.
        let (c, p) = keys();
        let r = build_receipt(payload(&c, &p, 128, [7u8; 16]), &c, &p);
        let blob = r.to_bytes();
        assert_eq!(blob[0], RECEIPT_FORMAT_V2);
        let back = CoSignedReceipt::from_bytes(&blob).unwrap();
        assert_eq!(back.payload.completion_tokens, None, "a v2 blob decodes as unknown split");
        assert_eq!(verify_receipt(&back), Ok(()));
    }

    #[test]
    fn tampering_with_the_split_is_rejected() {
        // Downgrade / forgery defence: altering the signed completion count, or stripping it to
        // pretend the receipt is legacy, breaks the consumer signature.
        let (c, p) = keys();
        let mut r = build_receipt(split_payload(&c, &p, 100, 80), &c, &p);
        r.payload.completion_tokens = Some(20); // inflate the cheap-prefill share
        assert_eq!(verify_receipt(&r), Err(ReceiptError::BadConsumerSig));
        // Strip the split entirely (claim legacy) — also breaks the signature it was signed with.
        let mut stripped = build_receipt(split_payload(&c, &p, 100, 80), &c, &p);
        stripped.payload.completion_tokens = None;
        assert_eq!(verify_receipt(&stripped), Err(ReceiptError::BadConsumerSig));
    }

    #[test]
    fn payload_from_bytes_threads_the_split() {
        let (c, p) = keys();
        let pl = payload_from_bytes(
            p.verifying_key().as_bytes(),
            c.verifying_key().as_bytes(),
            "qwen3.5/2b/fp16/5632a1b48425a5ae",
            100,
            &[3u8; 16],
            1_700_000_000_000,
            Some(64),
            "qwen3.5/2b/fp16/5632a1b48425a5ae",
            None,
        )
        .unwrap();
        assert_eq!(pl.completion_tokens, Some(64));
        assert_eq!(pl.canonical_model_id, "qwen3.5/2b/fp16/5632a1b48425a5ae");
    }

    // ── receipt-carries-canonical ──

    #[test]
    fn canonical_id_binds_into_preimage_and_roundtrips() {
        let (c, p) = keys();
        let mut pl = payload(&c, &p, 100, [4u8; 16]);
        pl.canonical_model_id = "qwen3-coder/30b-a3b/int4/abcd0123abcd0123".to_string();
        // Bound into the signed bytes → can't be swapped without breaking the signature.
        let mut other = payload(&c, &p, 100, [4u8; 16]);
        other.canonical_model_id = "tinyllama/1b/int4/abcd0123abcd0123".to_string();
        assert_ne!(pl.canonical_bytes(), other.canonical_bytes());
        // A carrying receipt serializes v3 and round-trips the canonical id + verifies.
        let r = build_receipt(pl, &c, &p);
        assert_eq!(r.to_bytes()[0], RECEIPT_FORMAT_V3);
        let back = CoSignedReceipt::from_bytes(&r.to_bytes()).unwrap();
        assert_eq!(back.payload.canonical_model_id, "qwen3-coder/30b-a3b/int4/abcd0123abcd0123");
        assert_eq!(back.payload.completion_tokens, None, "canonical alone doesn't imply a split");
        assert_eq!(verify_receipt(&back), Ok(()));
    }

    #[test]
    fn empty_canonical_stays_legacy_v2_and_forging_it_is_rejected() {
        let (c, p) = keys();
        // No canonical, no split → byte-identical legacy v2.
        let r = build_receipt(payload(&c, &p, 50, [5u8; 16]), &c, &p);
        assert_eq!(r.to_bytes()[0], RECEIPT_FORMAT_V2);
        let back = CoSignedReceipt::from_bytes(&r.to_bytes()).unwrap();
        assert_eq!(back.payload.canonical_model_id, "");
        // A provider that forges a big-model canonical onto a signed receipt breaks the signature.
        let mut forged = build_receipt(payload(&c, &p, 50, [5u8; 16]), &c, &p);
        forged.payload.canonical_model_id = "deepseek/671b/int8/abcd0123abcd0123".to_string();
        assert_eq!(verify_receipt(&forged), Err(ReceiptError::BadConsumerSig));
    }

    #[test]
    fn both_canonical_and_split_together_roundtrip() {
        let (c, p) = keys();
        let mut pl = payload(&c, &p, 200, [6u8; 16]);
        pl.completion_tokens = Some(150);
        pl.canonical_model_id = "llama/70b/int8/abcd0123abcd0123".to_string();
        let r = build_receipt(pl, &c, &p);
        assert_eq!(r.to_bytes()[0], RECEIPT_FORMAT_V3);
        let back = CoSignedReceipt::from_bytes(&r.to_bytes()).unwrap();
        assert_eq!(back.payload.completion_tokens, Some(150));
        assert_eq!(back.payload.canonical_model_id, "llama/70b/int8/abcd0123abcd0123");
        assert_eq!(verify_receipt(&back), Ok(()));
    }

    // ── v5: crypto-rail denomination binding (settlement asset + reference-price peg) ──

    fn denomination_payload(consumer: &SigningKey, provider: &SigningKey, cumulative: u64, peg: u64) -> ReceiptPayload {
        let mut pl = cumulative_payload(consumer, provider, 100, cumulative);
        pl.settlement_asset = Some(SettlementAsset::Usdc);
        pl.peg_micros_per_credit = Some(peg);
        pl
    }

    #[test]
    fn a_no_denomination_receipt_stays_v2_v3_v4_byte_identical() {
        // A receipt without an asset/peg never serializes as v5 — plain stays v2, split/canonical
        // stays v3, a bare cumulative stays v4 — so legacy blobs + the golden signatures are untouched.
        let (c, p) = keys();
        assert_eq!(build_receipt(payload(&c, &p, 512, [1u8; 16]), &c, &p).to_bytes()[0], RECEIPT_FORMAT_V2);
        assert_eq!(build_receipt(split_payload(&c, &p, 512, 300), &c, &p).to_bytes()[0], RECEIPT_FORMAT_V3);
        assert_eq!(build_receipt(cumulative_payload(&c, &p, 512, 9_600), &c, &p).to_bytes()[0], RECEIPT_FORMAT_V4);
    }

    #[test]
    fn denomination_binds_into_preimage_and_verifies() {
        // The asset + peg are committed, so neither party can alter the claim's denomination without
        // breaking the signature.
        let (c, p) = keys();
        let bare = cumulative_payload(&c, &p, 100, 9_600); // v4: cumulative but no asset/peg
        let denom = denomination_payload(&c, &p, 9_600, 1_200);
        assert_ne!(bare.canonical_bytes(), denom.canonical_bytes(), "asset+peg must change the preimage");
        // tag 4 (1+1) + tag 5 (1+8) = 11 bytes over the v4 preimage.
        assert_eq!(denom.canonical_bytes().len(), bare.canonical_bytes().len() + 11, "asset tag+byte, peg tag+u64");
        // A different peg signs different bytes.
        assert_ne!(
            denomination_payload(&c, &p, 9_600, 1_200).canonical_bytes(),
            denomination_payload(&c, &p, 9_600, 1_300).canonical_bytes()
        );
        let r = build_receipt(denom, &c, &p);
        assert_eq!(verify_receipt(&r), Ok(()));
    }

    #[test]
    fn v5_blob_roundtrips_preserving_denomination_and_the_lower_fields() {
        let (c, p) = keys();
        // A v5 receipt that ALSO carries the split + canonical + cumulative — v5 is a strict superset.
        let mut pl = denomination_payload(&c, &p, 48_000, 1_200);
        pl.completion_tokens = Some(300);
        pl.canonical_model_id = "qwen3.5/2b/fp16/5632a1b48425a5ae".to_string();
        let r = build_receipt(pl, &c, &p);
        assert_eq!(r.to_bytes()[0], RECEIPT_FORMAT_V5, "a receipt with an asset/peg serializes as v5");
        let back = CoSignedReceipt::from_bytes(&r.to_bytes()).unwrap();
        assert_eq!(back.payload.settlement_asset, Some(SettlementAsset::Usdc));
        assert_eq!(back.payload.peg_micros_per_credit, Some(1_200));
        assert_eq!(back.payload.cumulative_amount, Some(48_000));
        assert_eq!(back.payload.completion_tokens, Some(300));
        assert_eq!(back.payload.canonical_model_id, "qwen3.5/2b/fp16/5632a1b48425a5ae");
        assert_eq!(verify_receipt(&back), Ok(()), "signatures survive the v5 round-trip");
        // A v5 receipt with ONLY the peg (no asset) still round-trips (each field is independent).
        let mut peg_only = cumulative_payload(&c, &p, 5, 1);
        peg_only.peg_micros_per_credit = Some(999);
        let r2 = build_receipt(peg_only, &c, &p);
        assert_eq!(r2.to_bytes()[0], RECEIPT_FORMAT_V5);
        let back2 = CoSignedReceipt::from_bytes(&r2.to_bytes()).unwrap();
        assert_eq!(back2.payload.peg_micros_per_credit, Some(999));
        assert_eq!(back2.payload.settlement_asset, None);
        assert_eq!(back2.payload.cumulative_amount, Some(1));
    }

    #[test]
    fn tampering_with_the_peg_or_asset_is_rejected() {
        // Downgrade / forgery defence: altering the signed peg (settle at a different rate), or
        // stripping the denomination to pretend it's a v4 receipt, breaks the consumer signature.
        let (c, p) = keys();
        let mut r = build_receipt(denomination_payload(&c, &p, 9_600, 1_200), &c, &p);
        r.payload.peg_micros_per_credit = Some(2_400); // double the settle rate
        assert_eq!(verify_receipt(&r), Err(ReceiptError::BadConsumerSig));
        let mut stripped = build_receipt(denomination_payload(&c, &p, 9_600, 1_200), &c, &p);
        stripped.payload.settlement_asset = None;
        stripped.payload.peg_micros_per_credit = None;
        assert_eq!(verify_receipt(&stripped), Err(ReceiptError::BadConsumerSig));
    }
}
