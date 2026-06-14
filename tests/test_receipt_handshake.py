# Copyright 2026 OpenHydra contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Integration test: the two-way co-signed receipt handshake across the FFI (M2.1).

Mocks an inference-stream completion and runs the full consumer→provider
cryptographic handshake through the Rust ed25519 FFI (openhydra_network):
consumer signs → provider counter-signs → provider verifies before acknowledging.
The receipt math itself is covered by the Rust unit tests; this proves the FFI
boundary and the Python lifecycle wiring (coordinator/receipts.py).
"""

from __future__ import annotations

import pytest

ohn = pytest.importorskip("openhydra_network")
from coordinator import receipts as rc  # noqa: E402

MODEL = "qwen3.5/2b/fp16/5632a1b48425a5ae"


def _identity(seed: int) -> tuple[bytes, bytes]:
    """A deterministic (signing_key, public_key) pair from a seed byte."""
    signing_key = bytes([seed]) * 32
    return signing_key, rc.public_key(signing_key)


def test_full_two_way_handshake_valid():
    consumer_sk, consumer_pub = _identity(7)
    provider_sk, provider_pub = _identity(9)

    # Stream completes: the provider served 512 tokens of MODEL to the consumer.
    payload = rc.ReceiptPayload.for_completion(provider_pub, consumer_pub, MODEL, 512)

    # 1) Consumer signs the final receipt → sends {payload, consumer_sig} to provider.
    consumer_sig = rc.consumer_sign(payload, consumer_sk)
    # 2) Provider counter-signs.
    provider_sig = rc.provider_cosign(payload, provider_sk, consumer_sig)
    # 3) Provider verifies + holds before acknowledging completion.
    ledger = rc.ReceiptLedger()
    ledger.record(payload, consumer_sig, provider_sig)
    assert len(ledger) == 1
    assert ledger.receipts[0].payload.tokens == 512
    # 4) Consumer also verifies the returned co-signed receipt — no raise.
    rc.verify(payload, consumer_sig, provider_sig)


def test_provider_cannot_inflate_token_count():
    consumer_sk, consumer_pub = _identity(7)
    provider_sk, provider_pub = _identity(9)
    nonce = rc.new_nonce()

    payload = rc.ReceiptPayload.for_completion(provider_pub, consumer_pub, MODEL, 512, nonce=nonce, ts_unix_ms=1)
    consumer_sig = rc.consumer_sign(payload, consumer_sk)

    # A greedy provider co-signs an inflated payload — but the consumer's signature
    # is over the original 512, so verification fails on the consumer signature.
    inflated = rc.ReceiptPayload.for_completion(provider_pub, consumer_pub, MODEL, 999_999, nonce=nonce, ts_unix_ms=1)
    provider_sig = rc.provider_cosign(inflated, provider_sk, consumer_sig)

    ledger = rc.ReceiptLedger()
    with pytest.raises(ValueError, match="bad_consumer_sig"):
        ledger.record(inflated, consumer_sig, provider_sig)
    assert len(ledger) == 0


def test_corrupt_provider_signature_rejected():
    consumer_sk, consumer_pub = _identity(7)
    provider_sk, provider_pub = _identity(9)
    payload = rc.ReceiptPayload.for_completion(provider_pub, consumer_pub, MODEL, 100)
    consumer_sig = rc.consumer_sign(payload, consumer_sk)
    provider_sig = rc.provider_cosign(payload, provider_sk, consumer_sig)

    corrupt = bytes([provider_sig[0] ^ 0xFF]) + provider_sig[1:]
    with pytest.raises(ValueError, match="bad_provider_sig"):
        rc.verify(payload, consumer_sig, corrupt)


def test_replay_rejected_by_ledger():
    consumer_sk, consumer_pub = _identity(7)
    provider_sk, provider_pub = _identity(9)
    payload = rc.ReceiptPayload.for_completion(provider_pub, consumer_pub, MODEL, 100)
    consumer_sig = rc.consumer_sign(payload, consumer_sk)
    provider_sig = rc.provider_cosign(payload, provider_sk, consumer_sig)

    ledger = rc.ReceiptLedger()
    ledger.record(payload, consumer_sig, provider_sig)
    with pytest.raises(ValueError, match="replayed_nonce"):
        ledger.record(payload, consumer_sig, provider_sig)  # double-submit
    assert len(ledger) == 1


def test_malformed_key_length_raises():
    with pytest.raises(ValueError):
        rc.public_key(b"too-short")


# --- receipt exchange over the (injected) proxy transport ---


def test_wire_request_round_trips():
    _, consumer_pub = _identity(7)
    _, provider_pub = _identity(9)
    payload = rc.ReceiptPayload.for_completion(provider_pub, consumer_pub, MODEL, 512)
    consumer_sig = b"\xab" * 64
    decoded, sig = rc.decode_receipt_request(rc.encode_receipt_request(payload, consumer_sig))
    assert decoded == payload
    assert sig == consumer_sig


def test_exchange_round_trip_consumer_to_provider():
    consumer_sk, consumer_pub = _identity(7)
    provider_sk, provider_pub = _identity(9)
    payload = rc.ReceiptPayload.for_completion(provider_pub, consumer_pub, MODEL, 512)

    provider_ledger = rc.ReceiptLedger()
    # The transport wires the consumer request straight into the provider handler;
    # live, this is `lambda req: node.proxy_forward(provider_peer_id, req)`.
    transport = lambda req: rc.handle_receipt_request(req, provider_sk, provider_ledger)  # noqa: E731

    provider_sig = rc.exchange_receipt(transport, payload, consumer_sk)
    assert len(provider_sig) == 64
    assert len(provider_ledger) == 1  # provider verified + committed before responding


def test_exchange_replay_rejected_over_transport():
    consumer_sk, consumer_pub = _identity(7)
    provider_sk, provider_pub = _identity(9)
    payload = rc.ReceiptPayload.for_completion(provider_pub, consumer_pub, MODEL, 512)
    provider_ledger = rc.ReceiptLedger()
    transport = lambda req: rc.handle_receipt_request(req, provider_sk, provider_ledger)  # noqa: E731

    rc.exchange_receipt(transport, payload, consumer_sk)  # first ok
    with pytest.raises(ValueError, match="replayed_nonce"):
        rc.exchange_receipt(transport, payload, consumer_sk)  # replay → provider error response
    assert len(provider_ledger) == 1


def test_consumer_detects_provider_payload_tampering():
    # A greedy provider co-signs an inflated payload and returns OK; the consumer
    # verifies the provider signature against the ORIGINAL payload it sent, so the
    # tamper is caught client-side.
    consumer_sk, consumer_pub = _identity(7)
    provider_sk, provider_pub = _identity(9)
    payload = rc.ReceiptPayload.for_completion(provider_pub, consumer_pub, MODEL, 512)

    def malicious_transport(req: bytes) -> bytes:
        recv_payload, consumer_sig = rc.decode_receipt_request(req)
        inflated = rc.ReceiptPayload(
            provider_pub=recv_payload.provider_pub,
            consumer_pub=recv_payload.consumer_pub,
            model_id=recv_payload.model_id,
            tokens=999_999,
            nonce=recv_payload.nonce,
            ts_unix_ms=recv_payload.ts_unix_ms,
        )
        provider_sig = rc.provider_cosign(inflated, provider_sk, consumer_sig)
        return rc._RECEIPT_OK + provider_sig

    with pytest.raises(ValueError, match="bad_provider_sig"):
        rc.exchange_receipt(malicious_transport, payload, consumer_sk)


# --- secure node-identity path (private keys never leave Rust) ---


def test_secure_two_node_exchange(tmp_path):
    # Two real node identities; the consumer signs and the provider co-signs each with
    # its own *internal* key (node.receipt_sign / node.receipt_cosign). Transport is
    # injected (live: consumer_node.proxy_forward) — no swarm.
    consumer_node = ohn.P2PNode(identity_key_path=str(tmp_path / "consumer.key"))
    provider_node = ohn.P2PNode(identity_key_path=str(tmp_path / "provider.key"))
    provider_pub = bytes(provider_node.public_key_bytes())

    ledger = rc.ReceiptLedger()
    transport = lambda req: rc.handle_receipt_request_secure(req, provider_node, ledger)  # noqa: E731
    receipt = rc.consumer_request_receipt(consumer_node, transport, provider_pub, MODEL, 512)

    assert receipt.payload.tokens == 512
    assert len(ledger) == 1
    # The consumer never sees either private key — it only passed pubkeys + the node.


def test_auto_fire_receipt_from_route_outcome(tmp_path):
    # Simulates the live consumer flow: resolve_and_route returns an outcome carrying
    # the serving provider's raw public key (surfaced by the Rust router); the consumer
    # auto-fires the co-signed receipt straight off that outcome at stream completion.
    consumer_node = ohn.P2PNode(identity_key_path=str(tmp_path / "consumer.key"))
    provider_node = ohn.P2PNode(identity_key_path=str(tmp_path / "provider.key"))
    provider_pub = bytes(provider_node.public_key_bytes())

    # The shape P2PNode.resolve_and_route returns (response elided), incl. provider_pub.
    outcome = {
        "model_id": MODEL,
        "peer_id": "12D3KooWProvider",
        "response": b"...streamed tokens...",
        "degraded": False,
        "provider_pub": provider_pub,
    }

    ledger = rc.ReceiptLedger()
    transport = lambda req: rc.handle_receipt_request_secure(req, provider_node, ledger)  # noqa: E731
    receipt = rc.request_receipt_for_route(consumer_node, outcome, 512, transport=transport)

    assert receipt is not None
    assert receipt.payload.tokens == 512
    assert receipt.payload.provider_pub == provider_pub
    assert len(ledger) == 1


def test_auto_fire_skips_legacy_provider_without_key(tmp_path):
    # A legacy provider that advertised no public key → empty provider_pub on the
    # outcome. The auto-fire is a soft miss (returns None), not a stream failure.
    consumer_node = ohn.P2PNode(identity_key_path=str(tmp_path / "consumer.key"))
    outcome = {"model_id": MODEL, "peer_id": "legacy", "response": b"x", "degraded": False, "provider_pub": b""}

    # transport must never be invoked when there is no key to receipt against.
    def transport(_req):
        raise AssertionError("transport should not be called for an unkeyed provider")

    assert rc.request_receipt_for_route(consumer_node, outcome, 100, transport=transport) is None


def test_secure_provider_refuses_receipt_naming_another_provider(tmp_path):
    # A receipt addressed to provider B, delivered to provider A's handler: A refuses
    # to co-sign a receipt that names a different provider's identity.
    consumer_node = ohn.P2PNode(identity_key_path=str(tmp_path / "c.key"))
    provider_a = ohn.P2PNode(identity_key_path=str(tmp_path / "a.key"))
    provider_b = ohn.P2PNode(identity_key_path=str(tmp_path / "b.key"))
    b_pub = bytes(provider_b.public_key_bytes())

    ledger = rc.ReceiptLedger()
    transport = lambda req: rc.handle_receipt_request_secure(req, provider_a, ledger)  # noqa: E731
    with pytest.raises(ValueError, match="provider key does not match"):
        rc.consumer_request_receipt(consumer_node, transport, b_pub, MODEL, 100)
    assert len(ledger) == 0
