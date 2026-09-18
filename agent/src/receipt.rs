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

use ed25519_dalek::Signature;

use openhydra_protocol::crypto_agility::SigAlg;
use openhydra_protocol::receipts::{
    cosign_bytes, payload_from_bytes, verify_receipt, CoSignedReceipt, ReceiptError, ReceiptPayload,
};

use crate::adapter::AdapterError;

/// libp2p proxy method byte for a receipt settlement (consumer → provider).
pub const RECEIPT_REQUEST: u8 = 0x12;

const RESP_OK: u8 = 0x00;
const RESP_ERR: u8 = 0x01;

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

fn encode_request(payload: &ReceiptPayload, consumer_sig: &Signature) -> Vec<u8> {
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
    // sig_alg (F6): `canonical_bytes()` binds the signature-algorithm discriminant into the signed
    // preimage, so the framing MUST transmit it too or a non-Ed25519 consumer's provider would
    // reconstruct a different preimage and every settlement fails closed. It is the LAST trailer,
    // emitted as a single discriminant byte and ONLY when non-Ed25519 — so today's Ed25519-only
    // traffic stays byte-identical (absent trailer ⇒ Ed25519 on decode) and an already-deployed
    // provider that predates this trailer is unaffected. When it IS emitted, the (otherwise optional)
    // cumulative trailer's flag byte is written first even if `None`, so the positional decode stays
    // unambiguous.
    let emit_sig_alg = payload.sig_alg != SigAlg::Ed25519;
    match payload.cumulative_amount {
        Some(v) => {
            b.push(1u8);
            b.extend_from_slice(&v.to_le_bytes());
        }
        None if emit_sig_alg => b.push(0u8), // explicit `no cumulative`, so the sig_alg byte follows cleanly
        None => {}
    }
    if emit_sig_alg {
        b.push(payload.sig_alg.to_u8());
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

fn decode_request(data: &[u8]) -> Result<(ReceiptPayload, Signature), AdapterError> {
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
    if pos != data.len() {
        return Err(AdapterError::Parse("receipt request has trailing bytes".into()));
    }
    let mut payload = payload_from_bytes(
        &data[0..32], &data[32..64], model, tokens, &data[64..80], ts, completion, canonical,
        cumulative,
    )
    .map_err(AdapterError::Parse)?;
    // `payload_from_bytes` is the classical constructor (defaults Ed25519); stamp the wire-declared
    // algorithm so the co-sign preimage matches the consumer's.
    payload.sig_alg = sig_alg;
    let consumer_sig = to_sig(&data[96..160])?;
    Ok((payload, consumer_sig))
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
) -> Result<CoSignedReceipt, AdapterError> {
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
    )
    .map_err(AdapterError::Parse)?;
    let consumer_sig = to_sig(&sign(&payload.canonical_bytes()))?;

    let mut framed = vec![RECEIPT_REQUEST];
    framed.extend_from_slice(&encode_request(&payload, &consumer_sig));
    let response = transport(&framed)?;

    let provider_sig = match response.first() {
        Some(&RESP_OK) => to_sig(response.get(1..65).unwrap_or_default())?,
        Some(&RESP_ERR) => {
            return Err(AdapterError::Http(format!(
                "provider rejected receipt: {}",
                String::from_utf8_lossy(&response[1..])
            )))
        }
        _ => return Err(AdapterError::Parse("receipt response malformed".into())),
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
    bind_check: &dyn Fn(&ReceiptPayload) -> Result<(), String>,
) -> (Vec<u8>, Option<CoSignedReceipt>) {
    if data.first() != Some(&RECEIPT_REQUEST) {
        return (err_response("unsupported method"), None);
    }
    let (payload, consumer_sig) = match decode_request(&data[1..]) {
        Ok(x) => x,
        Err(e) => return (err_response(&e.to_string()), None),
    };
    if payload.provider.as_bytes() != this_provider_pub {
        return (err_response("receipt names a different provider"), None);
    }
    // B-S1: the receipt must settle against a real serve the provider recorded (right
    // tokens/model, fresh, single-use). Reject before co-signing if the binding fails.
    if let Err(reason) = bind_check(&payload) {
        return (err_response(&reason), None);
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
            Ok(handle_receipt_inbound(req, &psign, &provider_pub, &|_: &ReceiptPayload| Ok::<(), String>(())).0)
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
        )
        .unwrap();
        assert_eq!(receipt.payload.tokens, 512);
        assert_eq!(receipt.payload.canonical_model_id, "qwen2.5/7b/q4_k_m/abcd0123abcd0123",
            "canonical id flows end-to-end through the request framing and is co-signed");
        assert_eq!(receipt.payload.cumulative_amount, Some(9_600),
            "the crypto-rail cumulative voucher amount flows end-to-end and is co-signed");
        assert_eq!(verify_receipt(&receipt), Ok(()));
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
        };
        let sig = provider.sign(b"placeholder"); // any 64-byte sig; the framing doesn't re-verify it here

        // Ed25519 (the only implemented alg): NO trailer emitted, and decode still recovers Ed25519.
        let ed_framed = encode_request(&base, &sig);
        let (ed_decoded, _) = decode_request(&ed_framed).unwrap();
        assert_eq!(ed_decoded.sig_alg, SigAlg::Ed25519);

        // A non-Ed25519 discriminant is carried through the framing verbatim (forward-compat plumbing
        // for PQC3.1 — we can't verify it yet, but the provider must reconstruct the SAME sig_alg the
        // consumer signed). Emitting the trailer must lengthen the frame by exactly one byte.
        let mut pqc = base.clone();
        pqc.sig_alg = SigAlg::MlDsa65;
        let pqc_framed = encode_request(&pqc, &sig);
        assert_eq!(pqc_framed.len(), ed_framed.len() + 2,
            "explicit cumulative `0` flag (so the positional decode stays unambiguous) + the discriminant byte");
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
            |req: &[u8]| -> Result<Vec<u8>, AdapterError> { Ok(handle_receipt_inbound(req, &a_sign, &a_pub, &|_: &ReceiptPayload| Ok::<(), String>(())).0) };

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
        )
        .unwrap_err();
        assert!(matches!(err, AdapterError::Http(m) if m.contains("different provider")));
    }

    #[test]
    fn provider_handler_ledgers_only_accepted_receipts() {
        let provider = signer(9);
        let (_, accepted) = handle_receipt_inbound(b"\x99garbage", &sign_with(&provider), &pubkey(&provider), &|_: &ReceiptPayload| Ok::<(), String>(()));
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
        )
        .unwrap_err();
        assert!(matches!(err, AdapterError::Http(m) if m.contains("provider rejected")));
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

        let provider = signer(9);
        let psign = sign_with(&provider);
        let ppub = pubkey(&provider);
        for bytes in &corpus {
            // Core decoder on the payload (post method-byte), where the slicing risk lives.
            let _ = decode_request(bytes);
            // Full inbound path incl. the `data.first()` method-byte strip — results ignored,
            // the point is "no panic". Feed both the raw bytes and a 0x12-prefixed variant.
            let _ = handle_receipt_inbound(bytes, &psign, &ppub, &|_: &ReceiptPayload| Ok::<(), String>(()));
            let mut framed = vec![RECEIPT_REQUEST];
            framed.extend_from_slice(bytes);
            let _ = handle_receipt_inbound(&framed, &psign, &ppub, &|_: &ReceiptPayload| Ok::<(), String>(()));
        }
    }
}
