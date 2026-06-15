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
    };
    AdapterError::Http(format!("receipt rejected: {which}"))
}

fn encode_request(payload: &ReceiptPayload, consumer_sig: &Signature) -> Vec<u8> {
    let model = payload.model_id.as_bytes();
    let mut b = Vec::with_capacity(REQ_FIXED + model.len());
    b.extend_from_slice(payload.provider.as_bytes());
    b.extend_from_slice(payload.consumer.as_bytes());
    b.extend_from_slice(&payload.nonce);
    b.extend_from_slice(&payload.tokens.to_le_bytes());
    b.extend_from_slice(&payload.ts_unix_ms.to_le_bytes());
    b.extend_from_slice(&consumer_sig.to_bytes());
    b.extend_from_slice(&(model.len() as u16).to_le_bytes());
    b.extend_from_slice(model);
    b
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
    if data.len() != REQ_FIXED + model_len {
        return Err(AdapterError::Parse("receipt request length mismatch".into()));
    }
    let model = std::str::from_utf8(&data[162..162 + model_len])
        .map_err(|e| AdapterError::Parse(format!("receipt model utf8: {e}")))?;
    let payload = payload_from_bytes(&data[0..32], &data[32..64], model, tokens, &data[64..80], ts)
        .map_err(AdapterError::Parse)?;
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
) -> Result<CoSignedReceipt, AdapterError> {
    let payload =
        payload_from_bytes(provider_pub, consumer_pub, model_id, tokens, &nonce, ts_unix_ms)
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
            Ok(handle_receipt_inbound(req, &psign, &provider_pub).0)
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
        )
        .unwrap();
        assert_eq!(receipt.payload.tokens, 512);
        assert_eq!(verify_receipt(&receipt), Ok(()));
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
            |req: &[u8]| -> Result<Vec<u8>, AdapterError> { Ok(handle_receipt_inbound(req, &a_sign, &a_pub).0) };

        let err = request_receipt(
            &csign,
            &mut transport,
            &b_pub,
            &pubkey(&consumer),
            "m",
            10,
            [1u8; 16],
            1,
        )
        .unwrap_err();
        assert!(matches!(err, AdapterError::Http(m) if m.contains("different provider")));
    }

    #[test]
    fn provider_handler_ledgers_only_accepted_receipts() {
        let provider = signer(9);
        let (_, accepted) = handle_receipt_inbound(b"\x99garbage", &sign_with(&provider), &pubkey(&provider));
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
        )
        .unwrap_err();
        assert!(matches!(err, AdapterError::Http(m) if m.contains("provider rejected")));
    }
}
