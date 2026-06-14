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

pytest.importorskip("openhydra_network")
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
