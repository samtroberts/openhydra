from __future__ import annotations

import time
from pathlib import Path

import cbor2
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from supernode.adapter import PromptRequest
from supernode.attested_base import AttestedRuntime
from supernode.attestation_utils import canonical_request_hash

import hashlib


class _StubRuntime(AttestedRuntime):
    """Minimal concrete subclass for testing the base."""

    async def list_models(self):
        return []

    async def generate(self, request):
        yield  # pragma: no cover

    async def cancel(self, request_id):
        pass

    async def get_status(self):
        pass  # pragma: no cover

    async def health_check(self):
        return True

    async def warmup(self, model_id):
        return True


def _make_runtime(peer_id="test-peer", model_id="test-model"):
    key = Ed25519PrivateKey.generate()
    return _StubRuntime(peer_id=peer_id, private_key=key, model_id=model_id), key


def test_trust_tier_and_integration_level():
    rt, _ = _make_runtime()
    assert rt.trust_tier() == "attested"
    assert rt.integration_level() == 3


def test_register_weights(tmp_path: Path):
    (tmp_path / "model.safetensors").write_bytes(b"weights" * 100)
    (tmp_path / "config.json").write_text('{"hidden_size": 768}')
    rt, _ = _make_runtime()
    h = rt._register_weights(tmp_path)
    assert len(h) == 64
    assert rt.get_weights_hash("any") == h


def test_register_weights_no_files_raises(tmp_path: Path):
    (tmp_path / "readme.txt").write_text("not a model file")
    rt, _ = _make_runtime()
    with pytest.raises(ValueError, match="No weight/config files"):
        rt._register_weights(tmp_path)


def test_register_weights_gguf(tmp_path: Path):
    (tmp_path / "model.gguf").write_bytes(b"\x00" * 256)
    rt, _ = _make_runtime()
    h = rt._register_weights(tmp_path)
    assert len(h) == 64


def test_sign_output_roundtrip():
    rt, key = _make_runtime()
    req = PromptRequest(request_id="r1", model_id="llama-7b", prompt="hello")
    rt._weights_hash = "deadbeef" * 8
    ts = int(time.time() * 1000)
    sig = rt.sign_output(req, "llama-7b", [1, 2, 3], ts)

    assert isinstance(sig, bytes)
    assert len(sig) == 64

    request_hash = canonical_request_hash(
        model_id="llama-7b",
        prompt="hello",
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
    )
    output_hash = hashlib.sha256(
        cbor2.dumps([1, 2, 3], canonical=True)
    ).hexdigest()
    payload = {
        "v": 1,
        "peer_id": "test-peer",
        "request_id": "r1",
        "request_hash": request_hash,
        "model_id": "llama-7b",
        "weights_hash": "deadbeef" * 8,
        "output_token_hash": output_hash,
        "completion_tokens": 3,
        "timestamp_ms": ts,
    }
    payload_cbor = cbor2.dumps(payload, canonical=True)
    pub_bytes = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    assert AttestedRuntime.verify_output_signature(payload_cbor, sig, pub_bytes)


def test_sign_output_tamper_detection():
    rt, key = _make_runtime()
    req = PromptRequest(request_id="r1", model_id="m", prompt="hello")
    rt._weights_hash = "abcd" * 16
    ts = int(time.time() * 1000)
    sig = rt.sign_output(req, "m", [10, 20], ts)

    tampered_payload = cbor2.dumps({"v": 1, "peer_id": "evil"}, canonical=True)
    pub_bytes = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    assert not AttestedRuntime.verify_output_signature(tampered_payload, sig, pub_bytes)


def test_sign_output_wrong_key():
    rt, _ = _make_runtime()
    req = PromptRequest(request_id="r1", model_id="m", prompt="test")
    rt._weights_hash = "0000" * 16
    ts = int(time.time() * 1000)
    sig = rt.sign_output(req, "m", [1], ts)

    wrong_key = Ed25519PrivateKey.generate()
    wrong_pub = wrong_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    payload = cbor2.dumps({"anything": True}, canonical=True)
    assert not AttestedRuntime.verify_output_signature(payload, sig, wrong_pub)


def test_register_weights_deterministic(tmp_path: Path):
    (tmp_path / "a.safetensors").write_bytes(b"aaa")
    (tmp_path / "b.safetensors").write_bytes(b"bbb")
    (tmp_path / "config.json").write_text("{}")
    rt1, _ = _make_runtime()
    rt2, _ = _make_runtime()
    h1 = rt1._register_weights(tmp_path)
    h2 = rt2._register_weights(tmp_path)
    assert h1 == h2
