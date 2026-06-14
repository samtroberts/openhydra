# Copyright 2026 OpenHydra contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Co-signed receipt lifecycle (protocol.md §6) — M2.1.

Wraps the Rust ed25519 receipt FFI (``openhydra_network``) into the two-way
handshake performed at inference-stream completion:

1. The **consumer** signs the final ``(provider, consumer, model_id, tokens,
   nonce, ts)`` and sends ``{payload, consumer_sig}`` to the provider.
2. The **provider** counter-signs, then verifies the full co-signed receipt
   *before acknowledging* completion, and holds it.

Keys/signatures are raw bytes (32-byte ed25519 seeds & public keys, 64-byte
signatures, 16-byte nonces). Validated receipts are held in memory here — durable
storage, gossip replication, and monotonic counters are M2.3.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import openhydra_network as _ohn


def new_nonce() -> bytes:
    """A fresh 128-bit receipt nonce."""
    return os.urandom(16)


def public_key(signing_key: bytes) -> bytes:
    """Derive the 32-byte ed25519 public key from a 32-byte signing-key seed."""
    return _ohn.ed25519_public_key(signing_key)


@dataclass(frozen=True)
class ReceiptPayload:
    """The content both parties sign over."""

    provider_pub: bytes
    consumer_pub: bytes
    model_id: str
    tokens: int
    nonce: bytes
    ts_unix_ms: int

    @staticmethod
    def for_completion(
        provider_pub: bytes,
        consumer_pub: bytes,
        model_id: str,
        tokens: int,
        *,
        nonce: bytes | None = None,
        ts_unix_ms: int | None = None,
    ) -> "ReceiptPayload":
        """Build a payload at stream completion (fresh nonce + now() by default)."""
        return ReceiptPayload(
            provider_pub=provider_pub,
            consumer_pub=consumer_pub,
            model_id=model_id,
            tokens=int(tokens),
            nonce=nonce if nonce is not None else new_nonce(),
            ts_unix_ms=ts_unix_ms if ts_unix_ms is not None else int(time.time() * 1000),
        )


def consumer_sign(payload: ReceiptPayload, consumer_signing_key: bytes) -> bytes:
    """Consumer signs the completed-stream receipt; returns the 64-byte signature."""
    return _ohn.receipt_sign_consumer(
        consumer_signing_key,
        payload.provider_pub,
        payload.consumer_pub,
        payload.model_id,
        payload.tokens,
        payload.nonce,
        payload.ts_unix_ms,
    )


def provider_cosign(payload: ReceiptPayload, provider_signing_key: bytes, consumer_sig: bytes) -> bytes:
    """Provider counter-signs (payload ‖ consumer_sig); returns the 64-byte signature."""
    return _ohn.receipt_cosign_provider(
        provider_signing_key,
        payload.provider_pub,
        payload.consumer_pub,
        payload.model_id,
        payload.tokens,
        payload.nonce,
        payload.ts_unix_ms,
        consumer_sig,
    )


def verify(payload: ReceiptPayload, consumer_sig: bytes, provider_sig: bytes) -> None:
    """Verify a full co-signed receipt. Raises ``ValueError`` on rejection."""
    _ohn.receipt_verify(
        payload.provider_pub,
        payload.consumer_pub,
        payload.model_id,
        payload.tokens,
        payload.nonce,
        payload.ts_unix_ms,
        consumer_sig,
        provider_sig,
    )


@dataclass(frozen=True)
class CoSignedReceipt:
    payload: ReceiptPayload
    consumer_sig: bytes
    provider_sig: bytes


class ReceiptLedger:
    """In-memory hold of validated co-signed receipts (persistence is M2.3)."""

    def __init__(self) -> None:
        self._receipts: list[CoSignedReceipt] = []
        self._nonces: set[bytes] = set()

    def record(self, payload: ReceiptPayload, consumer_sig: bytes, provider_sig: bytes) -> CoSignedReceipt:
        """Provider-side: verify before acknowledging, reject replays, then hold."""
        verify(payload, consumer_sig, provider_sig)  # raises ValueError on a bad signature
        if payload.nonce in self._nonces:
            raise ValueError("receipt rejected: replayed_nonce")
        self._nonces.add(payload.nonce)
        receipt = CoSignedReceipt(payload, consumer_sig, provider_sig)
        self._receipts.append(receipt)
        return receipt

    @property
    def receipts(self) -> list[CoSignedReceipt]:
        return list(self._receipts)

    def __len__(self) -> int:
        return len(self._receipts)
