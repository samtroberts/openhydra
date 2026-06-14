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
import struct
import time
from dataclasses import dataclass
from typing import Callable

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

    def add_verified(self, payload: ReceiptPayload, consumer_sig: bytes, provider_sig: bytes) -> CoSignedReceipt:
        """Hold an already-verified receipt (crypto checked by node.receipt_cosign in
        Rust); reject a replayed nonce. Used by the secure provider path."""
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


# --- Receipt exchange over libp2p (protocol.md §6 live handshake) ---
#
# The handshake is a single request/response over the proxy transport, NOT a
# streamed await: after a whole-model serve, the consumer signs a receipt and
# proxy_forward()s it to the provider; the provider verifies + co-signs + ledgers
# and returns its signature. The consumer then verifies the co-signed receipt.
#
# A 1-byte method prefix (0x07) demultiplexes it from Forward/Ping/… on the peer's
# proxy-request loop (peer/server.py). The transport is injected (a bytes->bytes
# callable), so the protocol is testable single-process and, live, is just
# `lambda req: node.proxy_forward(provider_peer_id, req)`.

RECEIPT_METHOD_PREFIX = b"\x07"  # libp2p proxy method byte for the receipt exchange
_RECEIPT_OK = b"\x00"
_RECEIPT_ERR = b"\x01"

# Request layout after the 1-byte prefix: provider_pub(32) consumer_pub(32)
# nonce(16) tokens(u64 LE) ts(u64 LE) consumer_sig(64) model_len(u16 LE) model.
_REQ_FIXED = 32 + 32 + 16 + 8 + 8 + 64 + 2


def encode_receipt_request(payload: ReceiptPayload, consumer_sig: bytes) -> bytes:
    """Consumer→provider receipt message (prefixed with RECEIPT_METHOD_PREFIX)."""
    model = payload.model_id.encode("utf-8")
    return b"".join(
        [
            RECEIPT_METHOD_PREFIX,
            payload.provider_pub,
            payload.consumer_pub,
            payload.nonce,
            struct.pack("<Q", int(payload.tokens)),
            struct.pack("<Q", int(payload.ts_unix_ms)),
            consumer_sig,
            struct.pack("<H", len(model)),
            model,
        ]
    )


def decode_receipt_request(data: bytes) -> tuple[ReceiptPayload, bytes]:
    """Parse a consumer→provider receipt message → (payload, consumer_sig)."""
    if not data or data[0:1] != RECEIPT_METHOD_PREFIX:
        raise ValueError("receipt request: missing 0x07 method prefix")
    body = data[1:]
    if len(body) < _REQ_FIXED:
        raise ValueError("receipt request: truncated")
    provider_pub = body[0:32]
    consumer_pub = body[32:64]
    nonce = body[64:80]
    tokens = struct.unpack_from("<Q", body, 80)[0]
    ts_unix_ms = struct.unpack_from("<Q", body, 88)[0]
    consumer_sig = body[96:160]
    (model_len,) = struct.unpack_from("<H", body, 160)
    model_id = body[162 : 162 + model_len].decode("utf-8")
    if len(body) != _REQ_FIXED + model_len:
        raise ValueError("receipt request: trailing bytes")
    payload = ReceiptPayload(
        provider_pub=provider_pub,
        consumer_pub=consumer_pub,
        model_id=model_id,
        tokens=tokens,
        nonce=nonce,
        ts_unix_ms=ts_unix_ms,
    )
    return payload, consumer_sig


def handle_receipt_request(message: bytes, provider_signing_key: bytes, ledger: ReceiptLedger) -> bytes:
    """Provider side: decode → co-sign → verify + ledger → response bytes.

    Never raises into the proxy loop — a rejection (bad consumer signature, replay,
    malformed message) becomes an error response so the consumer learns why. On
    success returns ``0x00 || provider_sig`` (65 bytes).
    """
    try:
        payload, consumer_sig = decode_receipt_request(message)
        provider_sig = provider_cosign(payload, provider_signing_key, consumer_sig)
        ledger.record(payload, consumer_sig, provider_sig)  # verifies + replay-guards + holds
        return _RECEIPT_OK + provider_sig
    except ValueError as exc:
        return _RECEIPT_ERR + str(exc).encode("utf-8")


def exchange_receipt(
    transport: Callable[[bytes], bytes],
    payload: ReceiptPayload,
    consumer_signing_key: bytes,
) -> bytes:
    """Consumer side: sign → send over ``transport`` → verify the returned co-signature.

    ``transport(request_bytes) -> response_bytes`` is the network round-trip — live,
    ``lambda req: node.proxy_forward(provider_peer_id, req)``. Returns the 64-byte
    provider signature; raises ``ValueError`` if the provider rejected the receipt or
    its co-signature doesn't verify (so a provider cannot alter the payload either).
    """
    consumer_sig = consumer_sign(payload, consumer_signing_key)
    response = transport(encode_receipt_request(payload, consumer_sig))
    if not response or response[0:1] != _RECEIPT_OK:
        reason = response[1:].decode("utf-8", "replace") if response else "empty response"
        raise ValueError(f"provider rejected receipt: {reason}")
    provider_sig = response[1:65]
    if len(provider_sig) != 64:
        raise ValueError("receipt response: short provider signature")
    verify(payload, consumer_sig, provider_sig)  # consumer independently verifies the co-signed receipt
    return provider_sig


# --- Secure node-identity paths (keys stay locked in the Rust daemon) ---
#
# These keep the ed25519 private key inside Rust: the node signs with its internal
# identity via node.receipt_sign / node.receipt_cosign — nothing is passed in Python.


def handle_receipt_request_secure(message: bytes, node, ledger: ReceiptLedger) -> bytes:
    """Provider side using the node's internal identity (key never leaves Rust).

    `node.receipt_cosign` verifies the consumer signature + co-signs with the node's
    own key; the ledger then replay-guards and holds. Rejections become an error
    response, never an exception into the proxy loop.
    """
    try:
        payload, consumer_sig = decode_receipt_request(message)
        provider_sig = node.receipt_cosign(
            payload.provider_pub,
            payload.consumer_pub,
            payload.model_id,
            payload.tokens,
            payload.nonce,
            payload.ts_unix_ms,
            consumer_sig,
        )
        ledger.add_verified(payload, consumer_sig, provider_sig)  # replay-guard + hold
        return _RECEIPT_OK + provider_sig
    except ValueError as exc:
        return _RECEIPT_ERR + str(exc).encode("utf-8")


def consumer_request_receipt(
    node,
    transport: Callable[[bytes], bytes],
    provider_pub: bytes,
    model_id: str,
    tokens: int,
    *,
    nonce: bytes | None = None,
    ts_unix_ms: int | None = None,
) -> CoSignedReceipt:
    """Consumer side using the node's internal identity. Signs with `node.receipt_sign`,
    sends over `transport` (live: `lambda r: node.proxy_forward(provider_peer_id, r)`),
    and verifies the co-signature. Returns the co-signed receipt; raises if the provider
    rejected it or the co-signature doesn't verify.
    """
    consumer_pub = bytes(node.public_key_bytes())
    nonce = nonce if nonce is not None else new_nonce()
    ts = ts_unix_ms if ts_unix_ms is not None else int(time.time() * 1000)
    payload = ReceiptPayload(provider_pub, consumer_pub, model_id, int(tokens), nonce, ts)
    consumer_sig = node.receipt_sign(provider_pub, model_id, int(tokens), nonce, ts)
    response = transport(encode_receipt_request(payload, consumer_sig))
    if not response or response[0:1] != _RECEIPT_OK:
        reason = response[1:].decode("utf-8", "replace") if response else "empty response"
        raise ValueError(f"provider rejected receipt: {reason}")
    provider_sig = response[1:65]
    verify(payload, consumer_sig, provider_sig)  # consumer independently verifies (no keys)
    return CoSignedReceipt(payload, consumer_sig, provider_sig)


def request_receipt_for_route(
    node,
    outcome: dict,
    tokens: int,
    *,
    transport: Callable[[bytes], bytes] | None = None,
) -> CoSignedReceipt | None:
    """Auto-fire the consumer→provider co-signed receipt right after a whole-model route.

    Wires `P2PNode.resolve_and_route`'s outcome straight into the M2.1 handshake: it
    reads the serving provider's identity (`outcome["provider_pub"]`, the raw 32-byte
    ed25519 key surfaced by the router) and the peer to reach (`outcome["peer_id"]`),
    then signs + exchanges the receipt for `tokens` of `outcome["model_id"]` using the
    node's *internal* key (never crosses the FFI).

    `transport` defaults to a live libp2p round-trip to the serving peer
    (`lambda r: node.proxy_forward(outcome["peer_id"], r)`); tests inject the provider's
    secure handler directly. Returns the co-signed receipt, or **None** when the
    provider advertised no public key (a legacy peer that cannot be receipted) — the
    caller treats a missing receipt as a soft miss, not a stream failure.
    """
    provider_pub = bytes(outcome.get("provider_pub") or b"")
    if len(provider_pub) != 32:
        return None  # legacy/unkeyed provider — nothing to co-sign against
    if transport is None:
        peer_id = outcome["peer_id"]
        transport = lambda req: bytes(node.proxy_forward(peer_id, req))  # noqa: E731
    return consumer_request_receipt(
        node, transport, provider_pub, outcome["model_id"], int(tokens)
    )
