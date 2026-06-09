"""Tests for supernode.manifest — CBOR encoding, Ed25519 signing, normalization."""

import time

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from supernode.manifest import (
    SupernodeManifest,
    ModelCapability,
    HardwareInfo,
    normalize_model_id,
    supernode_record_key,
    model_provider_key,
    MANIFEST_TTL_MS,
)


@pytest.fixture
def key():
    return Ed25519PrivateKey.generate()


@pytest.fixture
def manifest():
    return SupernodeManifest(
        peer_id="test-peer",
        libp2p_peer_id="12D3KooWTest",
        backend_type="ollama",
        models=[
            ModelCapability(
                model_id="llama3.1:8b",
                model_family="llama",
                parameter_count=8000,
                quantization="Q4_0",
                context_length=131072,
            ),
            ModelCapability(
                model_id="qwen:2b",
                model_family="qwen",
                parameter_count=2500,
                quantization="Q8_0",
                context_length=32768,
                warm=True,
                estimated_tps=12.5,
            ),
        ],
        hardware=HardwareInfo(
            accelerator="cuda",
            gpu_name="NVIDIA T4",
            gpu_memory_mb=15360,
            gpu_memory_free_mb=12000,
            cpu_cores=8,
            ram_mb=32768,
        ),
        max_concurrent=4,
        max_context_length=131072,
        listen_addrs=["/ip4/1.2.3.4/tcp/4001"],
        nat_status="public",
        region="us-east",
    )


# ---------------------------------------------------------------------------
# CBOR roundtrip
# ---------------------------------------------------------------------------

class TestCBORRoundtrip:
    def test_encode_decode(self, manifest):
        data = manifest.to_cbor()
        assert isinstance(data, bytes)
        assert len(data) > 0

        restored = SupernodeManifest.from_cbor(data)
        assert restored.peer_id == "test-peer"
        assert restored.libp2p_peer_id == "12D3KooWTest"
        assert restored.backend_type == "ollama"

    def test_models_preserved(self, manifest):
        data = manifest.to_cbor()
        restored = SupernodeManifest.from_cbor(data)
        assert len(restored.models) == 2
        assert restored.models[0].model_id == "llama3.1:8b"
        assert restored.models[1].warm is True
        assert restored.models[1].estimated_tps == 12.5

    def test_hardware_preserved(self, manifest):
        data = manifest.to_cbor()
        restored = SupernodeManifest.from_cbor(data)
        assert restored.hardware.accelerator == "cuda"
        assert restored.hardware.gpu_name == "NVIDIA T4"
        assert restored.hardware.gpu_memory_mb == 15360

    def test_model_ids(self, manifest):
        assert manifest.model_ids() == ["llama3.1:8b", "qwen:2b"]


# ---------------------------------------------------------------------------
# Signing and verification
# ---------------------------------------------------------------------------

class TestSigning:
    def test_sign_sets_fields(self, manifest, key):
        manifest.sign(key)
        assert manifest.timestamp > 0
        assert len(manifest.signature) == 64
        assert len(manifest.public_key) == 32

    def test_verify_valid(self, manifest, key):
        manifest.sign(key)
        assert manifest.verify_signature() is True

    def test_verify_roundtrip(self, manifest, key):
        manifest.sign(key)
        data = manifest.to_cbor()
        restored = SupernodeManifest.from_cbor(data)
        assert restored.verify_signature() is True

    def test_verify_tampered(self, manifest, key):
        manifest.sign(key)
        manifest.peer_id = "evil-peer"
        assert manifest.verify_signature() is False

    def test_verify_wrong_key(self, manifest, key):
        manifest.sign(key)
        other_key = Ed25519PrivateKey.generate()
        manifest.public_key = other_key.public_key().public_bytes(
            __import__("cryptography.hazmat.primitives.serialization", fromlist=["Encoding"]).Encoding.Raw,
            __import__("cryptography.hazmat.primitives.serialization", fromlist=["PublicFormat"]).PublicFormat.Raw,
        )
        assert manifest.verify_signature() is False

    def test_verify_no_signature(self, manifest):
        assert manifest.verify_signature() is False

    def test_verify_empty_key(self, manifest):
        manifest.signature = b"x" * 64
        assert manifest.verify_signature() is False


# ---------------------------------------------------------------------------
# Freshness
# ---------------------------------------------------------------------------

class TestFreshness:
    def test_fresh_after_sign(self, manifest, key):
        manifest.sign(key)
        assert manifest.is_fresh() is True

    def test_stale(self, manifest, key):
        manifest.sign(key)
        manifest.timestamp = int(time.time() * 1000) - MANIFEST_TTL_MS - 1000
        assert manifest.is_fresh() is False

    def test_custom_now(self, manifest, key):
        manifest.sign(key)
        future = manifest.timestamp + MANIFEST_TTL_MS + 1
        assert manifest.is_fresh(now_ms=future) is False
        assert manifest.is_fresh(now_ms=manifest.timestamp + 1000) is True


# ---------------------------------------------------------------------------
# DHT key helpers
# ---------------------------------------------------------------------------

class TestDHTKeys:
    def test_supernode_key(self):
        assert supernode_record_key("12D3KooWABC") == "/openhydra/supernode/12D3KooWABC"

    def test_model_provider_key(self):
        assert model_provider_key("llama3.1:8b") == "/openhydra/model/llama3.1-8b/provider"

    def test_model_provider_key_normalized(self):
        k1 = model_provider_key("llama3.1:8b")
        k2 = model_provider_key("Llama3.1:8B")
        assert k1 == k2


# ---------------------------------------------------------------------------
# Model ID normalization
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_lowercase(self):
        assert normalize_model_id("Llama3.1:8B") == "llama3.1-8b"

    def test_strip(self):
        assert normalize_model_id("  llama3:8b  ") == "llama3-8b"

    def test_slashes(self):
        assert normalize_model_id("meta-llama/Llama-3.1-8B") == "llama-3.1-8b"

    def test_underscores(self):
        assert normalize_model_id("my_model_7b") == "my-model-7b"

    def test_colons(self):
        assert normalize_model_id("qwen:2b") == "qwen-2b"

    def test_meta_prefix_stripped(self):
        assert normalize_model_id("meta-llama/Llama-3.1-8B") == "llama-3.1-8b"

    def test_already_normalized(self):
        assert normalize_model_id("llama3.1-8b") == "llama3.1-8b"
