"""Tests for peer/identity.py Ed25519 keypair-based peer identity."""
from __future__ import annotations

import json
import stat

import pytest

from peer.identity import (
    generate_keypair,
    load_or_create_identity,
    peer_id_from_public_key,
    sign_announce,
    verify_announce,
)


def test_generate_keypair():
    private_key, public_key_bytes = generate_keypair()
    assert isinstance(public_key_bytes, bytes)
    assert len(public_key_bytes) == 32


def test_peer_id_format():
    _, public_key_bytes = generate_keypair()
    peer_id = peer_id_from_public_key(public_key_bytes)
    assert len(peer_id) == 16
    assert all(c in "0123456789abcdef" for c in peer_id)


def test_load_creates_file(tmp_path):
    identity_file = tmp_path / "identity.key"
    result = load_or_create_identity(str(identity_file))
    assert identity_file.exists()
    mode = stat.S_IMODE(identity_file.stat().st_mode)
    assert mode == 0o600
    assert "private_key" in result
    assert "public_key_hex" in result
    assert "peer_id" in result


def test_load_idempotent(tmp_path):
    identity_file = tmp_path / "identity.key"
    result1 = load_or_create_identity(str(identity_file))
    result2 = load_or_create_identity(str(identity_file))
    assert result1["peer_id"] == result2["peer_id"]
    assert result1["public_key_hex"] == result2["public_key_hex"]


def test_sign_verify_roundtrip():
    private_key, public_key_bytes = generate_keypair()
    public_key_hex = public_key_bytes.hex()
    peer_id = peer_id_from_public_key(public_key_bytes)

    sig = sign_announce(private_key, peer_id, "127.0.0.1", 50051, "test-model")
    result = verify_announce(public_key_hex, peer_id, "127.0.0.1", 50051, "test-model", sig)
    assert result is True


def test_verify_wrong_signature():
    private_key, public_key_bytes = generate_keypair()
    public_key_hex = public_key_bytes.hex()
    peer_id = peer_id_from_public_key(public_key_bytes)

    # Corrupt the signature
    sig = sign_announce(private_key, peer_id, "127.0.0.1", 50051, "test-model")
    corrupted = sig[:-4] + "AAAA"
    result = verify_announce(public_key_hex, peer_id, "127.0.0.1", 50051, "test-model", corrupted)
    assert result is False


def test_verify_wrong_field():
    private_key, public_key_bytes = generate_keypair()
    public_key_hex = public_key_bytes.hex()
    peer_id = peer_id_from_public_key(public_key_bytes)

    sig = sign_announce(private_key, peer_id, "127.0.0.1", 50051, "test-model")
    # Verify with a different model_id
    result = verify_announce(public_key_hex, peer_id, "127.0.0.1", 50051, "other-model", sig)
    assert result is False
