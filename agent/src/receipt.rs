// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! The co-signed receipt handshake (protocol.md §6) over the agent's swarm transport.
//!
//! At end-of-stream the consumer settles a receipt with the provider:
//! consumer signs `(provider, consumer, model, tokens, nonce, ts)` → provider verifies +
//! counter-signs → consumer verifies the co-signed receipt. The crypto is
//! [`openhydra_protocol::receipts`]; **signing is injected** as `&dyn Fn(&[u8]) -> Vec<u8>`
//! so the private key never leaves the node (live: `NetworkHandle::sign`; tests: an
//! ed25519 `SigningKey`). Nonce + timestamp are injected too, for deterministic tests.

use ed25519_dalek::{Signature, Verifier};

use openhydra_protocol::crypto_agility::SigAlg;
use openhydra_protocol::payout::SettlementAsset;
use openhydra_protocol::receipts::{
    cosign_bytes, payload_from_bytes, verify_receipt, CoSignedReceipt, ReceiptError, ReceiptPayload,
};

use crate::adapter::AdapterError;

/// libp2p proxy method byte for a receipt settlement (consumer → provider).
pub const RECEIPT_REQUEST: u8 = 0x12;

const RESP_OK: u8 = 0x00;
const RESP_ERR: u8 = 0x01;
/// The receipt's cumulative is behind the provider's latest co-signed one for the pair, which
/// follows ([`encode_cosigned`]) so the consumer can catch up.
const RESP_BEHIND: u8 = 0x02;

/// Why a provider's bind check turned a receipt down.
#[derive(Debug, Clone)]
pub enum BindRejection {
    /// Refused; the reason is sent back.
    Reject(String),
    /// The receipt's cumulative is behind the latest one the provider co-signed for this
    /// `(consumer, model)` — a co-sign reply the consumer never got, or a voucher it lost. That
    /// receipt goes back: it carries the consumer's own signature, so the consumer can adopt its
    /// cumulative and sign again on top of it.
    Behind(Box<CoSignedReceipt>),
}

impl From<String> for BindRejection {
    fn from(reason: String) -> Self {
        BindRejection::Reject(reason)
    }
}

impl From<&str> for BindRejection {
    fn from(reason: &str) -> Self {
        BindRejection::Reject(reason.to_string())
    }
}

/// Why a consumer's settlement attempt didn't produce a co-signed receipt.
#[derive(Debug)]
pub enum ReceiptFailure {
    /// The provider refused it, or the round-trip or the reply was bad.
    Failed(AdapterError),
    /// The provider holds a later co-signed receipt for the pair (see [`BindRejection::Behind`]).
    /// Not yet checked: the consumer verifies it is its own before trusting it.
    Behind(Box<CoSignedReceipt>),
}

impl From<AdapterError> for ReceiptFailure {
    fn from(e: AdapterError) -> Self {
        ReceiptFailure::Failed(e)
    }
}

// Request body after the method byte: provider_pub[32] consumer_pub[32] nonce[16]
// tokens:u64[8] ts:u64[8] consumer_sig[64] model_len:u16[2] model.
const REQ_FIXED: usize = 32 + 32 + 16 + 8 + 8 + 64 + 2;

fn to_sig(bytes: &[u8]) -> Result<Signature, AdapterError> {
    let arr: [u8; 64] = bytes
        .try_into()
        .map_err(|_| AdapterError::Parse("signature must be 64 bytes".into()))?;
    Ok(Signature::from_bytes(&arr))
}

fn receipt_err(e: ReceiptError) -> AdapterError {
    let which = match e {
        ReceiptError::BadConsumerSig => "bad_consumer_sig",
        ReceiptError::BadProviderSig => "bad_provider_sig",
        ReceiptError::ReplayedNonce => "replayed_nonce",
        ReceiptError::UnsupportedAlg(_) => "unsupported_sig_alg",
    };
    AdapterError::Http(format!("receipt rejected: {which}"))
}

/// A consumer's signature over a paying receipt's on-chain claim (claim format v6,
/// [`openhydra_protocol::settlement::SettlementClaim`]): the claim is the receipt's provider,
/// consumer, model and cumulative amount, with this expiry. Carried beside the receipt (its own
/// signature, outside the receipt's), so a provider co-signs a paying receipt only with a claim it can
/// settle on-chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimSig {
    /// Unix seconds after which the claim no longer settles.
    pub expires_at: u64,
    /// Solana: the 64-byte Ed25519 signature by the consumer; Base: the 65-byte `r‖s‖v` by its wallet.
    pub sig: Vec<u8>,
}

/// The longest claim signature carried (Base's 65 bytes).
const MAX_CLAIM_SIG: usize = 65;

/// The on-chain claim a receipt payload stands for, valid until `expires_at` (`None` without a
/// cumulative amount: not a paying receipt).
pub fn claim_for(payload: &ReceiptPayload, expires_at: u64) -> Option<openhydra_protocol::settlement::SettlementClaim> {
    Some(openhydra_protocol::settlement::SettlementClaim {
        provider: payload.provider.to_bytes(),
        consumer: payload.consumer.to_bytes(),
        model_id: payload.model_id.clone(),
        cumulative_amount: payload.cumulative_amount?,
        expires_at,
    })
}

fn encode_request(payload: &ReceiptPayload, consumer_sig: &Signature) -> Vec<u8> {
    encode_request_with_claim(payload, consumer_sig, None)
}

fn encode_request_with_claim(payload: &ReceiptPayload, consumer_sig: &Signature, claim: Option<&ClaimSig>) -> Vec<u8> {
    let model = payload.model_id.as_bytes();
    let canonical = payload.canonical_model_id.as_bytes();
    let mut b = Vec::with_capacity(REQ_FIXED + model.len() + 2 + canonical.len());
    b.extend_from_slice(payload.provider.as_bytes());
    b.extend_from_slice(payload.consumer.as_bytes());
    b.extend_from_slice(&payload.nonce);
    b.extend_from_slice(&payload.tokens.to_le_bytes());
    b.extend_from_slice(&payload.ts_unix_ms.to_le_bytes());
    b.extend_from_slice(&consumer_sig.to_bytes());
    b.extend_from_slice(&(model.len() as u16).to_le_bytes());
    b.extend_from_slice(model);
    // receipt-carries-canonical: append the canonical id (length-prefixed, possibly empty) so the
    // provider co-signs the same one the consumer weights by. A legacy request that omits this
    // trailer decodes to an empty canonical (see decode_request).
    b.extend_from_slice(&(canonical.len() as u16).to_le_bytes());
    b.extend_from_slice(canonical);
    // Carry the completion split too (review M3): the consumer signs `canonical_bytes()` including
    // completion, so the framing MUST transmit it or the provider reconstructs a different preimage
    // and every settlement fails closed once Phase 2b sets `Some`. `has_completion:u8 [+ u64]`.
    match payload.completion_tokens {
        Some(c) => {
            b.push(1u8);
            b.extend_from_slice(&c.to_le_bytes());
        }
        None => b.push(0u8),
    }
    // Crypto-rail cumulative voucher amount: the consumer signs it into `canonical_bytes()`, so the
    // framing MUST transmit it or the provider reconstructs a different preimage and settlement fails
    // closed. Appended ONLY when `Some` (`has_cumulative:u8=1 + u64`), so every non-crypto serve's
    // framing stays byte-identical to before and a provider that never sees a voucher is unaffected.
    //
    // The remaining trailers all bind into `canonical_bytes()`, so the framing MUST transmit each or
    // the provider reconstructs a different preimage and settlement fails closed. Fixed frame order
    // (…cumulative, settlement-asset [v5], peg [v5], sig_alg [F6]) with a presence-chain rule: a
    // trailer's flag byte is written when it OR any LATER trailer is present, so the positional decode
    // stays unambiguous. A plain non-crypto Ed25519 serve emits none of them and stays byte-identical
    // to the pre-v5/pre-F6 framing (absent ⇒ None / Ed25519 on decode), so a deployed peer predating
    // these trailers is unaffected.
    //
    // v5 (denomination binding): the settlement asset + reference-price peg the cumulative was
    // converted with. Binding + transmitting them lets the provider reject a peg/asset it never agreed
    // to before co-signing, instead of silently co-signing a mis-denominated claim.
    // A claim trailer follows sig_alg, so a claim makes it explicit (positional decode).
    let emit_sig_alg = payload.sig_alg != SigAlg::Ed25519 || claim.is_some();
    let emit_peg = payload.peg_micros_per_credit.is_some() || emit_sig_alg;
    let emit_asset = payload.settlement_asset.is_some() || emit_peg;
    let emit_cumulative = payload.cumulative_amount.is_some() || emit_asset;
    match payload.cumulative_amount {
        Some(v) => {
            b.push(1u8);
            b.extend_from_slice(&v.to_le_bytes());
        }
        None if emit_cumulative => b.push(0u8), // explicit `no cumulative`, so later trailers follow cleanly
        None => {}
    }
    if emit_asset {
        match payload.settlement_asset {
            Some(a) => {
                b.push(1u8);
                b.push(a.to_u8());
            }
            None => b.push(0u8),
        }
    }
    if emit_peg {
        match payload.peg_micros_per_credit {
            Some(p) => {
                b.push(1u8);
                b.extend_from_slice(&p.to_le_bytes());
            }
            None => b.push(0u8),
        }
    }
    if emit_sig_alg {
        b.push(payload.sig_alg.to_u8()); // bare discriminant
    }
    // Claim trailer (claim format v6): `expires_at:u64 ‖ sig_len:u8 ‖ sig`, last.
    if let Some(c) = claim {
        b.extend_from_slice(&c.expires_at.to_le_bytes());
        b.push(c.sig.len().min(MAX_CLAIM_SIG) as u8);
        b.extend_from_slice(&c.sig[..c.sig.len().min(MAX_CLAIM_SIG)]);
    }
    b
}

/// Read an optional `has_x:u8 [+ u64-LE]` trailer at `*pos`, advancing `*pos`. No bytes remaining →
/// `None`; flag `0` → `None`; flag `1` → the following `u64`; any other flag or a truncated value is
/// an error. Shared by the completion + cumulative receipt-request trailers.
fn read_flag_u64(data: &[u8], pos: &mut usize, what: &str) -> Result<Option<u64>, AdapterError> {
    if data.len() == *pos {
        return Ok(None);
    }
    let flag = data[*pos];
    *pos += 1;
    match flag {
        0 => Ok(None),
        1 => {
            if data.len() < *pos + 8 {
                return Err(AdapterError::Parse(format!("receipt request {what} truncated")));
            }
            let mut v = [0u8; 8];
            v.copy_from_slice(&data[*pos..*pos + 8]);
            *pos += 8;
            Ok(Some(u64::from_le_bytes(v)))
        }
        other => Err(AdapterError::Parse(format!("bad {what} flag: {other}"))),
    }
}

/// Read an optional `has_x:u8 [+ value:u8]` trailer at `*pos`, advancing `*pos`. Same shape as
/// [`read_flag_u64`] but a single-byte value — used by the v5 settlement-asset trailer.
fn read_flag_u8(data: &[u8], pos: &mut usize, what: &str) -> Result<Option<u8>, AdapterError> {
    if data.len() == *pos {
        return Ok(None);
    }
    let flag = data[*pos];
    *pos += 1;
    match flag {
        0 => Ok(None),
        1 => {
            if data.len() < *pos + 1 {
                return Err(AdapterError::Parse(format!("receipt request {what} truncated")));
            }
            let v = data[*pos];
            *pos += 1;
            Ok(Some(v))
        }
        other => Err(AdapterError::Parse(format!("bad {what} flag: {other}"))),
    }
}

fn decode_request(data: &[u8]) -> Result<(ReceiptPayload, Signature), AdapterError> {
    decode_request_with_claim(data).map(|(p, s, _)| (p, s))
}

fn decode_request_with_claim(data: &[u8]) -> Result<(ReceiptPayload, Signature, Option<ClaimSig>), AdapterError> {
    if data.len() < REQ_FIXED {
        return Err(AdapterError::Parse("receipt request truncated".into()));
    }
    let mut u8x8 = [0u8; 8];
    u8x8.copy_from_slice(&data[80..88]);
    let tokens = u64::from_le_bytes(u8x8);
    u8x8.copy_from_slice(&data[88..96]);
    let ts = u64::from_le_bytes(u8x8);
    let model_len = u16::from_le_bytes([data[160], data[161]]) as usize;
    let model_end = REQ_FIXED + model_len;
    if data.len() < model_end {
        return Err(AdapterError::Parse("receipt request length mismatch".into()));
    }
    let model = std::str::from_utf8(&data[162..model_end])
        .map_err(|e| AdapterError::Parse(format!("receipt model utf8: {e}")))?;
    // Optional trailers after the model, parsed in order: a length-prefixed canonical id, then the
    // completion split, then the crypto-rail cumulative voucher amount — each present iff bytes
    // remain, so every earlier wire shape still decodes (legacy no-trailer → empty/None; canonical-
    // only; canonical + completion; canonical + completion + cumulative). A trailing-bytes check at
    // the end keeps the framing strict.
    let mut pos = model_end;
    let canonical = if data.len() == pos {
        ""
    } else {
        if data.len() < pos + 2 {
            return Err(AdapterError::Parse("receipt request canonical length truncated".into()));
        }
        let clen = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2;
        let end = pos + clen;
        if data.len() < end {
            return Err(AdapterError::Parse("receipt request canonical truncated".into()));
        }
        let canonical = std::str::from_utf8(&data[pos..end])
            .map_err(|e| AdapterError::Parse(format!("receipt canonical utf8: {e}")))?;
        pos = end;
        canonical
    };
    // Completion trailer (M3): `has_completion:u8 [+ u64]`.
    let completion = read_flag_u64(data, &mut pos, "completion")?;
    // Cumulative-voucher trailer (crypto-rail, v4): `has_cumulative:u8 [+ u64]`.
    let cumulative = read_flag_u64(data, &mut pos, "cumulative")?;
    // Settlement-asset (v5): `has_asset:u8 [+ discriminant:u8]`.
    let settlement_asset = match read_flag_u8(data, &mut pos, "settlement asset")? {
        Some(disc) => Some(
            SettlementAsset::from_u8(disc)
                .map_err(|e| AdapterError::Parse(format!("receipt request settlement asset: {e}")))?,
        ),
        None => None,
    };
    // Reference-price peg (v5): `has_peg:u8 [+ u64]`.
    let peg = read_flag_u64(data, &mut pos, "peg")?;
    // sig_alg trailer (F6): a single discriminant byte, present iff bytes remain. Absent ⇒ Ed25519
    // (the legacy/common path, byte-identical to before). A future non-Ed25519 consumer transmits it
    // so the provider reconstructs the SAME `canonical_bytes()` preimage the consumer signed; because
    // sig_alg is bound into that preimage, a tampered discriminant just fails signature verification.
    let sig_alg = if pos == data.len() {
        SigAlg::Ed25519
    } else {
        let alg = SigAlg::from_u8(data[pos])
            .map_err(|e| AdapterError::Parse(format!("receipt request sig_alg: {e}")))?;
        pos += 1;
        alg
    };
    let claim = if pos == data.len() {
        None
    } else {
        if data.len() < pos + 9 {
            return Err(AdapterError::Parse("receipt request claim truncated".into()));
        }
        let expires_at = u64::from_le_bytes(data[pos..pos + 8].try_into().expect("8 bytes"));
        let len = data[pos + 8] as usize;
        pos += 9;
        if len > MAX_CLAIM_SIG || data.len() < pos + len {
            return Err(AdapterError::Parse("receipt request claim signature truncated".into()));
        }
        let sig = data[pos..pos + len].to_vec();
        pos += len;
        Some(ClaimSig { expires_at, sig })
    };
    if pos != data.len() {
        return Err(AdapterError::Parse("receipt request has trailing bytes".into()));
    }
    let mut payload = payload_from_bytes(
        &data[0..32], &data[32..64], model, tokens, &data[64..80], ts, completion, canonical,
        cumulative, settlement_asset, peg,
    )
    .map_err(AdapterError::Parse)?;
    // `payload_from_bytes` is the classical Ed25519 constructor; stamp the wire-declared algorithm so
    // the co-sign preimage matches the consumer's exactly (the v5 asset/peg are threaded as params).
    payload.sig_alg = sig_alg;
    let consumer_sig = to_sig(&data[96..160])?;
    Ok((payload, consumer_sig, claim))
}

fn ok_response(provider_sig: &Signature) -> Vec<u8> {
    let mut b = Vec::with_capacity(65);
    b.push(RESP_OK);
    b.extend_from_slice(&provider_sig.to_bytes());
    b
}

fn err_response(msg: &str) -> Vec<u8> {
    let mut b = Vec::with_capacity(1 + msg.len());
    b.push(RESP_ERR);
    b.extend_from_slice(msg.as_bytes());
    b
}

/// A co-signed receipt as bytes: the provider's signature, then the request framing (payload and
/// consumer signature). What a provider keeps of the latest receipt per pair, and sends back in a
/// [`RESP_BEHIND`] reply.
pub fn encode_cosigned(receipt: &CoSignedReceipt) -> Vec<u8> {
    let mut b = receipt.provider_sig.to_bytes().to_vec();
    b.extend_from_slice(&encode_request(&receipt.payload, &receipt.consumer_sig));
    b
}

/// The inverse of [`encode_cosigned`]. Decodes only; the signatures are not checked.
pub fn decode_cosigned(bytes: &[u8]) -> Result<CoSignedReceipt, AdapterError> {
    if bytes.len() < 64 {
        return Err(AdapterError::Parse("co-signed receipt truncated".into()));
    }
    let provider_sig = to_sig(&bytes[..64])?;
    let (payload, consumer_sig) = decode_request(&bytes[64..])?;
    Ok(CoSignedReceipt { payload, consumer_sig, provider_sig })
}

/// Consumer side: build the receipt, sign it, settle with the provider over `transport`,
/// and verify the returned co-signed receipt. Returns it on success.
///
/// `sign(canonical_bytes) -> 64-byte sig` is the consumer node signing with its identity
/// key (never exported). `transport(framed) -> response` is a `proxy_forward` to the
/// provider. `nonce`/`ts_unix_ms` are supplied by the caller (random/now live).
#[allow(clippy::too_many_arguments)]
pub fn request_receipt(
    sign: &dyn Fn(&[u8]) -> Vec<u8>,
    transport: &mut dyn FnMut(&[u8]) -> Result<Vec<u8>, AdapterError>,
    provider_pub: &[u8],
    consumer_pub: &[u8],
    model_id: &str,
    tokens: u64,
    nonce: [u8; 16],
    ts_unix_ms: u64,
    canonical_model_id: &str,
    cumulative_amount: Option<u64>,
    settlement_asset: Option<SettlementAsset>,
    peg_micros_per_credit: Option<u64>,
    claim_signer: Option<(u64, &dyn Fn(&openhydra_protocol::settlement::SettlementClaim) -> Vec<u8>)>,
) -> Result<CoSignedReceipt, ReceiptFailure> {
    let payload = payload_from_bytes(
        provider_pub,
        consumer_pub,
        model_id,
        tokens,
        &nonce,
        ts_unix_ms,
        None,
        canonical_model_id,
        cumulative_amount,
        // v5 denomination binding: the consumer supplies the asset + peg it converted the cumulative
        // with, so both parties co-sign the same claim units (and a provider can reject a peg it never
        // agreed to).
        settlement_asset,
        peg_micros_per_credit,
    )
    .map_err(AdapterError::Parse)?;
    let consumer_sig = to_sig(&sign(&payload.canonical_bytes()))?;

    // On a settlement chain, a paying receipt carries its on-chain claim, signed for that chain.
    let claim = claim_signer.and_then(|(expires_at, sign_claim)| {
        claim_for(&payload, expires_at).map(|c| ClaimSig { expires_at, sig: sign_claim(&c) })
    });
    let mut framed = vec![RECEIPT_REQUEST];
    framed.extend_from_slice(&encode_request_with_claim(&payload, &consumer_sig, claim.as_ref()));
    let response = transport(&framed)?;

    let provider_sig = match response.first() {
        Some(&RESP_OK) => to_sig(response.get(1..65).unwrap_or_default())?,
        Some(&RESP_ERR) => {
            return Err(AdapterError::Http(format!(
                "provider rejected receipt: {}",
                String::from_utf8_lossy(&response[1..])
            ))
            .into())
        }
        Some(&RESP_BEHIND) => return Err(ReceiptFailure::Behind(Box::new(decode_cosigned(&response[1..])?))),
        _ => return Err(AdapterError::Parse("receipt response malformed".into()).into()),
    };

    let receipt = CoSignedReceipt { payload, consumer_sig, provider_sig };
    verify_receipt(&receipt).map_err(receipt_err)?; // consumer independently verifies
    Ok(receipt)
}

/// Provider side: decode an inbound receipt request, refuse if it names a different
/// provider, counter-sign, verify the whole package, and return the response bytes (never
/// panics — a rejection is an error response, not a raise).
///
/// Returns `(response_bytes, Option<verified_receipt>)`; the receipt is `Some` only when
/// accepted (so the caller can ledger it).
pub fn handle_receipt_inbound(
    data: &[u8],
    sign: &dyn Fn(&[u8]) -> Vec<u8>,
    this_provider_pub: &[u8],
    bind_check: &dyn Fn(&ReceiptPayload, Option<&ClaimSig>) -> Result<(), BindRejection>,
) -> (Vec<u8>, Option<CoSignedReceipt>) {
    if data.first() != Some(&RECEIPT_REQUEST) {
        return (err_response("unsupported method"), None);
    }
    let (payload, consumer_sig, claim) = match decode_request_with_claim(&data[1..]) {
        Ok(x) => x,
        Err(e) => return (err_response(&e.to_string()), None),
    };
    if payload.provider.as_bytes() != this_provider_pub {
        return (err_response("receipt names a different provider"), None);
    }
    // The consumer's signature first: nothing below (a single-use serve commitment consumed, a
    // pair's latest receipt sent back) may happen for a receipt its consumer didn't sign.
    if !payload.sig_alg.is_implemented()
        || payload.consumer.verify(&payload.canonical_bytes(), &consumer_sig).is_err()
    {
        return (err_response("bad consumer signature"), None);
    }
    // B-S1: the receipt must settle against a real serve the provider recorded (right
    // tokens/model, fresh, single-use). Reject before co-signing if the binding fails.
    match bind_check(&payload, claim.as_ref()) {
        Ok(()) => {}
        Err(BindRejection::Reject(reason)) => return (err_response(&reason), None),
        Err(BindRejection::Behind(latest)) => {
            let mut b = vec![RESP_BEHIND];
            b.extend_from_slice(&encode_cosigned(&latest));
            return (b, None);
        }
    }
    let provider_sig = match to_sig(&sign(&cosign_bytes(&payload, &consumer_sig))) {
        Ok(s) => s,
        Err(e) => return (err_response(&e.to_string()), None),
    };
    let receipt = CoSignedReceipt { payload, consumer_sig, provider_sig };
    match verify_receipt(&receipt) {
        Ok(()) => (ok_response(&provider_sig), Some(receipt)),
        Err(e) => (err_response(&format!("{:?}", e)), None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    fn signer(seed: u8) -> SigningKey {
        SigningKey::from_bytes(&[seed; 32])
    }
    fn pubkey(k: &SigningKey) -> [u8; 32] {
        k.verifying_key().to_bytes()
    }
    fn sign_with(k: &SigningKey) -> impl Fn(&[u8]) -> Vec<u8> + '_ {
        move |msg: &[u8]| k.sign(msg).to_bytes().to_vec()
    }

    #[test]
    fn full_handshake_settles_and_verifies() {
        let consumer = signer(7);
        let provider = signer(9);
        let (provider_pub, consumer_pub) = (pubkey(&provider), pubkey(&consumer));

        let csign = sign_with(&consumer);
        let psign = sign_with(&provider);
        // The transport IS the provider's handler — in-process round-trip.
        let mut transport = |req: &[u8]| -> Result<Vec<u8>, AdapterError> {
            Ok(handle_receipt_inbound(req, &psign, &provider_pub, &|_: &ReceiptPayload, _: Option<&ClaimSig>| Ok::<(), BindRejection>(())).0)
        };

        let receipt = request_receipt(
            &csign,
            &mut transport,
            &provider_pub,
            &consumer_pub,
            "qwen2.5:7b",
            512,
            [42u8; 16],
            1_700_000_000_000,
            "qwen2.5/7b/q4_k_m/abcd0123abcd0123",
            Some(9_600),
            Some(SettlementAsset::Usdc),
            Some(1_200), None)
        .unwrap();
        assert_eq!(receipt.payload.tokens, 512);
        assert_eq!(receipt.payload.canonical_model_id, "qwen2.5/7b/q4_k_m/abcd0123abcd0123",
            "canonical id flows end-to-end through the request framing and is co-signed");
        assert_eq!(receipt.payload.cumulative_amount, Some(9_600),
            "the crypto-rail cumulative voucher amount flows end-to-end and is co-signed");
        assert_eq!(receipt.payload.settlement_asset, Some(SettlementAsset::Usdc),
            "the v5 settlement asset flows end-to-end and is co-signed");
        assert_eq!(receipt.payload.peg_micros_per_credit, Some(1_200),
            "the v5 reference-price peg flows end-to-end and is co-signed");
        assert_eq!(verify_receipt(&receipt), Ok(()));
    }

    /// One settle against a provider that accepts everything, returning the co-signed receipt.
    fn settle_once(consumer: &SigningKey, provider: &SigningKey, nonce: u8, cumulative: u64) -> CoSignedReceipt {
        let (csign, psign) = (sign_with(consumer), sign_with(provider));
        let ppub = pubkey(provider);
        let mut transport = |req: &[u8]| -> Result<Vec<u8>, AdapterError> {
            Ok(handle_receipt_inbound(req, &psign, &ppub, &|_: &ReceiptPayload, _: Option<&ClaimSig>| Ok::<(), BindRejection>(())).0)
        };
        request_receipt(&csign, &mut transport, &ppub, &pubkey(consumer), "m", 10, [nonce; 16], 1, "",
            Some(cumulative), Some(SettlementAsset::Usdc), Some(1), None).unwrap()
    }

    #[test]
    fn a_provider_sends_back_its_latest_receipt_to_a_consumer_behind_it() {
        let (consumer, provider) = (signer(7), signer(9));
        let latest = settle_once(&consumer, &provider, 1, 500);
        // The next receipt is behind: the provider answers with the receipt that set its high-water.
        let (csign, psign, ppub) = (sign_with(&consumer), sign_with(&provider), pubkey(&provider));
        let kept = latest.clone();
        let mut transport = |req: &[u8]| -> Result<Vec<u8>, AdapterError> {
            let bind = |_: &ReceiptPayload, _: Option<&ClaimSig>| Err(BindRejection::Behind(Box::new(kept.clone())));
            Ok(handle_receipt_inbound(req, &psign, &ppub, &bind).0)
        };
        let err = request_receipt(&csign, &mut transport, &ppub, &pubkey(&consumer), "m", 10, [2u8; 16], 2, "",
            Some(300), Some(SettlementAsset::Usdc), Some(1), None).unwrap_err();
        let ReceiptFailure::Behind(back) = err else { panic!("expected the provider's latest receipt") };
        assert_eq!(back.payload.cumulative_amount, Some(500));
        assert_eq!(back.payload.nonce, [1u8; 16]);
        assert_eq!(verify_receipt(&back), Ok(()), "it arrives intact, both signatures valid");
        assert_eq!(decode_cosigned(&encode_cosigned(&latest)).unwrap().to_bytes(), latest.to_bytes());
    }

    #[test]
    fn a_paying_receipt_carries_its_on_chain_claim_to_the_provider() {
        use openhydra_protocol::settlement::{verify_claim, ChainDomain, ClaimSigner, SolanaDomain, CLAIM_LIFETIME_SECS};
        let (consumer, provider) = (signer(7), signer(9));
        let (csign, psign, ppub) = (sign_with(&consumer), sign_with(&provider), pubkey(&provider));
        let chain = ChainDomain::Solana(SolanaDomain { program_id: [4; 32], cluster: [5; 32] });
        let now = 1_760_000_000u64;
        let seen = std::cell::RefCell::new(None);
        let mut transport = |req: &[u8]| -> Result<Vec<u8>, AdapterError> {
            let bind = |p: &ReceiptPayload, claim: Option<&ClaimSig>| {
                let c = claim.ok_or("no claim")?;
                let settlement = claim_for(p, c.expires_at).ok_or("no cumulative")?;
                *seen.borrow_mut() = Some(verify_claim(&chain, &settlement, &c.sig, now));
                Ok::<(), BindRejection>(())
            };
            Ok(handle_receipt_inbound(req, &psign, &ppub, &bind).0)
        };
        let sign_claim = |c: &openhydra_protocol::settlement::SettlementClaim| csign(&chain.claim_message(c));
        let receipt = request_receipt(&csign, &mut transport, &ppub, &pubkey(&consumer), "m", 10, [1u8; 16], 1, "",
            Some(500), Some(SettlementAsset::Usdc), Some(1), Some((now + CLAIM_LIFETIME_SECS, &sign_claim))).unwrap();
        assert_eq!(receipt.payload.cumulative_amount, Some(500));
        assert_eq!(*seen.borrow(), Some(Ok(ClaimSigner::Solana)), "the provider got a claim valid for its deployment");
    }

    #[test]
    fn the_claim_trailer_round_trips_and_is_strict() {
        let (consumer, provider) = (signer(7), signer(9));
        let payload = payload_from_bytes(&pubkey(&provider), &pubkey(&consumer), "m", 10, &[5u8; 16], 1, None, "",
            Some(100), Some(SettlementAsset::Usdc), Some(1)).unwrap();
        let sig = to_sig(&sign_with(&consumer)(&payload.canonical_bytes())).unwrap();
        let claim = ClaimSig { expires_at: 1_800_000_000, sig: vec![9u8; 65] };
        let framed = encode_request_with_claim(&payload, &sig, Some(&claim));
        let (p2, s2, c2) = decode_request_with_claim(&framed).unwrap();
        assert_eq!((p2.canonical_bytes(), s2, c2), (payload.canonical_bytes(), sig, Some(claim)));
        // The receipt's own framing (no claim) is unchanged by the trailer's existence.
        assert_eq!(decode_request_with_claim(&encode_request(&payload, &sig)).unwrap().2, None);
        // A signature length past the largest is malformed, and so is a truncated trailer.
        let mut bad = framed.clone();
        let at = bad.len() - 65 - 1;
        bad[at] = 66;
        assert!(decode_request_with_claim(&bad).is_err());
        assert!(decode_request_with_claim(&framed[..framed.len() - 1]).is_err());
    }

    #[test]
    fn a_receipt_its_consumer_did_not_sign_never_reaches_the_bind_check() {
        // Otherwise a forged receipt could burn a real serve's single-use nonce, or be handed a
        // pair's latest receipt.
        let (consumer, provider) = (signer(7), signer(9));
        let payload = payload_from_bytes(&pubkey(&provider), &pubkey(&consumer), "m", 10, &[5u8; 16], 1, None, "",
            Some(100), None, None).unwrap();
        let forged = to_sig(&sign_with(&signer(8))(&payload.canonical_bytes())).unwrap();
        let mut framed = vec![RECEIPT_REQUEST];
        framed.extend_from_slice(&encode_request(&payload, &forged));
        let reached = std::cell::Cell::new(false);
        let bind = |_: &ReceiptPayload, _: Option<&ClaimSig>| {
            reached.set(true);
            Ok::<(), BindRejection>(())
        };
        let (resp, accepted) = handle_receipt_inbound(&framed, &sign_with(&provider), &pubkey(&provider), &bind);
        assert!(accepted.is_none());
        assert_eq!(resp.first(), Some(&RESP_ERR));
        assert!(!reached.get());
    }

    #[test]
    fn request_framing_round_trips_a_completion_split() {
        // Review M3: the framing must carry `completion_tokens`, else the day Phase 2b signs a
        // `Some` split the provider would reconstruct `None`, the co-sign preimage would differ, and
        // every settlement would fail closed. Here: a Some-completion payload survives encode→decode.
        use ed25519_dalek::{Signer, SigningKey};
        use openhydra_protocol::receipts::{ReceiptPayload, cosign_bytes, verify_receipt, CoSignedReceipt};
        use openhydra_protocol::crypto_agility::SigAlg;
        let consumer = SigningKey::from_bytes(&[7u8; 32]);
        let provider = SigningKey::from_bytes(&[9u8; 32]);
        let payload = ReceiptPayload {
            sig_alg: SigAlg::Ed25519,
            provider: provider.verifying_key(),
            consumer: consumer.verifying_key(),
            model_id: "qwen2.5:7b".to_string(),
            tokens: 100,
            nonce: [3u8; 16],
            ts_unix_ms: 1_700_000_000_000,
            completion_tokens: Some(60),
            canonical_model_id: "qwen2.5/7b/q4_k_m/abcd0123abcd0123".to_string(),
            cumulative_amount: None,
            settlement_asset: None,
            peg_micros_per_credit: None,
        };
        let consumer_sig = consumer.sign(&payload.canonical_bytes());
        let framed = encode_request(&payload, &consumer_sig);
        let (decoded, decoded_sig) = decode_request(&framed).unwrap();
        assert_eq!(decoded.completion_tokens, Some(60), "the split survives the framing");
        assert_eq!(decoded.canonical_model_id, payload.canonical_model_id);
        // The provider co-signs the decoded payload; if the preimage matched, verify passes.
        let provider_sig = provider.sign(&cosign_bytes(&decoded, &decoded_sig));
        let receipt = CoSignedReceipt { payload: decoded, consumer_sig: decoded_sig, provider_sig };
        assert_eq!(verify_receipt(&receipt), Ok(()), "provider reconstructs the identical preimage");
    }

    #[test]
    fn request_framing_transmits_sig_alg_and_stays_byte_identical_for_ed25519() {
        // F6: `canonical_bytes()` binds sig_alg into the signed preimage, so the request framing must
        // transmit it — otherwise a future non-Ed25519 consumer's provider reconstructs a different
        // preimage via `payload_from_bytes` (which defaults Ed25519) and every settlement fails closed.
        use openhydra_protocol::crypto_agility::SigAlg;
        let consumer = signer(7);
        let provider = signer(9);
        let base = ReceiptPayload {
            sig_alg: SigAlg::Ed25519,
            provider: provider.verifying_key(),
            consumer: consumer.verifying_key(),
            model_id: "qwen2.5:7b".to_string(),
            tokens: 100,
            nonce: [3u8; 16],
            ts_unix_ms: 1_700_000_000_000,
            completion_tokens: None,
            canonical_model_id: String::new(),
            cumulative_amount: None,
            settlement_asset: None,
            peg_micros_per_credit: None,
        };
        let sig = provider.sign(b"placeholder"); // any 64-byte sig; the framing doesn't re-verify it here

        // Ed25519 (the only implemented alg): NO trailer emitted, and decode still recovers Ed25519.
        let ed_framed = encode_request(&base, &sig);
        let (ed_decoded, _) = decode_request(&ed_framed).unwrap();
        assert_eq!(ed_decoded.sig_alg, SigAlg::Ed25519);

        // A non-Ed25519 discriminant is carried through the framing verbatim (forward-compat plumbing
        // for PQC3.1 — we can't verify it yet, but the provider must reconstruct the SAME sig_alg the
        // consumer signed). sig_alg is the LAST trailer, so emitting it forces the preceding optional
        // trailers' presence flags: cumulative `0`, asset `0`, peg `0`, then the discriminant = +4.
        let mut pqc = base.clone();
        pqc.sig_alg = SigAlg::MlDsa65;
        let pqc_framed = encode_request(&pqc, &sig);
        assert_eq!(pqc_framed.len(), ed_framed.len() + 4,
            "cumulative/asset/peg `0` flags (positional decode stays unambiguous) + the discriminant byte");
        let (pqc_decoded, _) = decode_request(&pqc_framed).unwrap();
        assert_eq!(pqc_decoded.sig_alg, SigAlg::MlDsa65, "the wire-declared algorithm survives the round-trip");

        // The trailer must also compose with a cumulative voucher: cumulative Some + non-Ed25519.
        let mut both = base.clone();
        both.sig_alg = SigAlg::MlDsa65;
        both.cumulative_amount = Some(9_600);
        let (both_decoded, _) = decode_request(&encode_request(&both, &sig)).unwrap();
        assert_eq!(both_decoded.sig_alg, SigAlg::MlDsa65);
        assert_eq!(both_decoded.cumulative_amount, Some(9_600));
    }

    #[test]
    fn request_framing_round_trips_the_v5_denomination() {
        // v5: the settlement asset + peg bind into `canonical_bytes()`, so the framing must carry them
        // or the provider reconstructs a different preimage and settlement fails closed. A full
        // cumulative+asset+peg payload survives encode→decode and the provider re-derives the identical
        // preimage (verify passes).
        let consumer = signer(7);
        let provider = signer(9);
        let base = ReceiptPayload {
            sig_alg: SigAlg::Ed25519,
            provider: provider.verifying_key(),
            consumer: consumer.verifying_key(),
            model_id: "qwen2.5:7b".to_string(),
            tokens: 100,
            nonce: [3u8; 16],
            ts_unix_ms: 1_700_000_000_000,
            completion_tokens: None,
            canonical_model_id: String::new(),
            cumulative_amount: Some(9_600),
            settlement_asset: Some(SettlementAsset::Usdc),
            peg_micros_per_credit: Some(1_200),
        };
        let consumer_sig = consumer.sign(&base.canonical_bytes());
        let (decoded, decoded_sig) = decode_request(&encode_request(&base, &consumer_sig)).unwrap();
        assert_eq!(decoded.settlement_asset, Some(SettlementAsset::Usdc), "the asset survives the framing");
        assert_eq!(decoded.peg_micros_per_credit, Some(1_200), "the peg survives the framing");
        assert_eq!(decoded.cumulative_amount, Some(9_600));
        let provider_sig = provider.sign(&cosign_bytes(&decoded, &decoded_sig));
        let receipt = CoSignedReceipt { payload: decoded, consumer_sig: decoded_sig, provider_sig };
        assert_eq!(verify_receipt(&receipt), Ok(()), "provider reconstructs the identical v5 preimage");

        // Asset-only (no peg) also survives — each field is independent in the framing.
        let mut asset_only = base.clone();
        asset_only.peg_micros_per_credit = None;
        let s2 = consumer.sign(&asset_only.canonical_bytes());
        let (d2, _) = decode_request(&encode_request(&asset_only, &s2)).unwrap();
        assert_eq!(d2.settlement_asset, Some(SettlementAsset::Usdc));
        assert_eq!(d2.peg_micros_per_credit, None);
    }

    #[test]
    fn provider_refuses_receipt_naming_a_different_provider() {
        let consumer = signer(7);
        let provider_a = signer(9);
        let provider_b = signer(11); // the receipt names B...
        let b_pub = pubkey(&provider_b);

        let csign = sign_with(&consumer);
        let a_sign = sign_with(&provider_a);
        let a_pub = pubkey(&provider_a);
        // ...but it's delivered to A's handler.
        let mut transport =
            |req: &[u8]| -> Result<Vec<u8>, AdapterError> { Ok(handle_receipt_inbound(req, &a_sign, &a_pub, &|_: &ReceiptPayload, _: Option<&ClaimSig>| Ok::<(), BindRejection>(())).0) };

        let err = request_receipt(
            &csign,
            &mut transport,
            &b_pub,
            &pubkey(&consumer),
            "m",
            10,
            [1u8; 16],
            1,
            "",
            None,
            None,
            None, None)
        .unwrap_err();
        assert!(matches!(err, ReceiptFailure::Failed(AdapterError::Http(m)) if m.contains("different provider")));
    }

    #[test]
    fn provider_handler_ledgers_only_accepted_receipts() {
        let provider = signer(9);
        let (_, accepted) = handle_receipt_inbound(b"\x99garbage", &sign_with(&provider), &pubkey(&provider), &|_: &ReceiptPayload, _: Option<&ClaimSig>| Ok::<(), BindRejection>(()));
        assert!(accepted.is_none()); // wrong method byte → no receipt to ledger
    }

    #[test]
    fn consumer_rejects_a_provider_error_response() {
        let consumer = signer(7);
        let provider = signer(9);
        let mut transport = |_: &[u8]| Ok(err_response("nope"));
        let err = request_receipt(
            &sign_with(&consumer),
            &mut transport,
            &pubkey(&provider),
            &pubkey(&consumer),
            "m",
            1,
            [1u8; 16],
            1,
            "",
            None,
            None,
            None, None)
        .unwrap_err();
        assert!(matches!(err, ReceiptFailure::Failed(AdapterError::Http(m)) if m.contains("provider rejected")));
    }

    /// Audit F3 (receipt-decoder coverage): the receipt request decoder runs on RAW, UNTRUSTED
    /// bytes — the network layer hands any inbound `request_response` body (method byte `0x12`)
    /// straight to `handle_receipt_inbound` → `decode_request`, which slices a fixed 162-byte
    /// header and then a sender-declared `model_len` region. Those slices MUST be length-guarded
    /// so adversarial input returns `Err`, never panics. This is the receipt-path twin of
    /// `serve::tests::serve_decoders_never_panic_on_malformed_input`; it pins the guarantee against
    /// a future edit that reorders a check or adds an unchecked slice. Passing = no unwind escapes.
    #[test]
    fn receipt_decoder_never_panics_on_malformed_input() {
        // Hostile shapes tuned to the layout (REQ_FIXED = 162, then u16 model_len at [160..162],
        // then `model_len` model bytes): empty, single bytes, truncated below the fixed header,
        // exact-header with a nonzero declared model_len (length-mismatch path), a huge declared
        // model_len over a short body, invalid-utf8 model bytes, and pseudo-random blobs.
        let mut corpus: Vec<Vec<u8>> = vec![
            vec![],
            vec![0x00],
            vec![RECEIPT_REQUEST],            // method byte only, no payload
            vec![0xFF; 1],
            vec![0u8; 80],                    // truncated mid fixed-header
            vec![0u8; REQ_FIXED - 2],         // one field short of the fixed header
            vec![0u8; REQ_FIXED - 1],
            vec![7u8; REQ_FIXED],             // full header, model_len from [160..162] = 0x0707 → len mismatch
            {
                // Exact fixed header, model_len declared huge, but no model body → length mismatch.
                let mut v = vec![0u8; REQ_FIXED];
                v[160] = 0xFF;
                v[161] = 0xFF; // model_len = 65535, but data.len() == REQ_FIXED → Err, no slice
                v
            },
            {
                // Header + a small model region that is invalid UTF-8 → from_utf8 Err, not a panic.
                let mut v = vec![0u8; REQ_FIXED + 3];
                v[160] = 3; // model_len = 3
                v[161] = 0;
                let n = v.len();
                v[n - 3..].copy_from_slice(&[0xff, 0xfe, 0x80]);
                v
            },
        ];
        // Deterministic pseudo-random blobs (no rng in the test path), spanning both sides of
        // REQ_FIXED so the length checks and the variable slice are all exercised.
        for seed in 0u16..96 {
            let n = (seed as usize * 7) % (REQ_FIXED + 40);
            corpus.push((0..n).map(|i| (seed.wrapping_mul(31).wrapping_add(i as u16)) as u8).collect());
        }
        // v5/F6 trailer-targeted blobs: a well-formed fixed header + empty model + canonical(len 0),
        // then ADVERSARIAL cumulative/asset/peg/sig_alg trailer bytes, so the new `read_flag_u8`/
        // `read_flag_u64`/`from_u8`/trailing-bytes paths are exercised (not just the header length
        // checks the pseudo-random blobs mostly hit). Each must Err, never panic.
        {
            // base: REQ_FIXED bytes, model_len = 0 at [160..162], then canonical clen = 0.
            let base = || {
                let mut v = vec![0u8; REQ_FIXED];
                v[160] = 0;
                v[161] = 0;
                v.extend_from_slice(&[0x00, 0x00]); // canonical length-prefix = 0
                v
            };
            let trailers: &[&[u8]] = &[
                &[0x00, 0x01],                   // completion=0, cumulative flag=1 but no u64 (truncated)
                &[0x03],                         // bad completion presence flag
                &[0x00, 0x00, 0x01, 0x00],       // asset flag=1, disc=0 → invalid SettlementAsset
                &[0x00, 0x00, 0x02],             // bad asset presence flag
                &[0x00, 0x00, 0x00, 0x01],       // peg flag=1 but no u64 (truncated)
                &[0x00, 0x00, 0x00, 0x00, 0xFF], // all absent, then sig_alg disc 0xFF (unknown)
                &[0x00, 0x00, 0x00, 0x00, 0x01, 0x00], // sig_alg=Ed25519 then a trailing byte
                &[0x01, 0, 0, 0, 0, 0, 0, 0, 0,  // completion=1 + u64
                  0x01, 0, 0, 0, 0, 0, 0, 0, 0,  // cumulative=1 + u64
                  0x01, 0x01,                    // asset=1 + Usdc
                  0x01, 0, 0, 0, 0, 0, 0, 0, 0,  // peg=1 + u64
                  0x02],                         // sig_alg = MlDsa65 (well-formed v5+PQC frame shape)
            ];
            for t in trailers {
                let mut v = base();
                v.extend_from_slice(t);
                corpus.push(v);
            }
        }

        let provider = signer(9);
        let psign = sign_with(&provider);
        let ppub = pubkey(&provider);
        for bytes in &corpus {
            // Core decoder on the payload (post method-byte), where the slicing risk lives.
            let _ = decode_request(bytes);
            // Full inbound path incl. the `data.first()` method-byte strip — results ignored,
            // the point is "no panic". Feed both the raw bytes and a 0x12-prefixed variant.
            let _ = handle_receipt_inbound(bytes, &psign, &ppub, &|_: &ReceiptPayload, _: Option<&ClaimSig>| Ok::<(), BindRejection>(()));
            let mut framed = vec![RECEIPT_REQUEST];
            framed.extend_from_slice(bytes);
            let _ = handle_receipt_inbound(&framed, &psign, &ppub, &|_: &ReceiptPayload, _: Option<&ClaimSig>| Ok::<(), BindRejection>(()));
        }
    }
}
