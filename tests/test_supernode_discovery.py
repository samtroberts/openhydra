"""Tests for supernode.discovery — SupernodeDiscovery with manifest cache."""

import time

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from supernode.manifest import (
    SupernodeManifest,
    ModelCapability,
    HardwareInfo,
    MANIFEST_TTL_MS,
)
from supernode.discovery import SupernodeDiscovery, CACHE_TTL_S


@pytest.fixture
def key():
    return Ed25519PrivateKey.generate()


def _make_manifest(peer_id, libp2p_id, models, key):
    m = SupernodeManifest(
        peer_id=peer_id,
        libp2p_peer_id=libp2p_id,
        backend_type="ollama",
        models=[
            ModelCapability(
                model_id=mid,
                model_family=mid.split(":")[0],
                parameter_count=7000,
                quantization="Q4_0",
                context_length=8192,
            )
            for mid in models
        ],
    )
    m.sign(key)
    return m


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegister:
    def test_valid_manifest(self, key):
        d = SupernodeDiscovery()
        m = _make_manifest("p1", "12D3Koo1", ["llama3:8b"], key)
        assert d.register_manifest(m) is True

    def test_bad_signature_rejected(self, key):
        d = SupernodeDiscovery()
        m = _make_manifest("p1", "12D3Koo1", ["llama3:8b"], key)
        m.peer_id = "tampered"
        assert d.register_manifest(m) is False

    def test_stale_manifest_rejected(self, key):
        d = SupernodeDiscovery()
        m = _make_manifest("p1", "12D3Koo1", ["llama3:8b"], key)
        m.timestamp = int(time.time() * 1000) - MANIFEST_TTL_MS - 1000
        # Re-sign so signature is valid but timestamp is old
        m.signature = b""
        m.public_key = b""
        m.sign(key)
        m.timestamp = int(time.time() * 1000) - MANIFEST_TTL_MS - 1000
        assert d.register_manifest(m) is False

    def test_unsigned_rejected(self):
        d = SupernodeDiscovery()
        m = SupernodeManifest(
            peer_id="p1", libp2p_peer_id="12D3Koo1", backend_type="ollama",
        )
        assert d.register_manifest(m) is False


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

class TestDiscover:
    def test_discover_by_model(self, key):
        d = SupernodeDiscovery()
        m1 = _make_manifest("p1", "id1", ["llama3:8b"], key)
        m2 = _make_manifest("p2", "id2", ["qwen:2b"], key)
        d.register_manifest(m1)
        d.register_manifest(m2)

        results = d.discover_supernodes("llama3:8b")
        assert len(results) == 1
        assert results[0].peer_id == "p1"

    def test_discover_normalized(self, key):
        d = SupernodeDiscovery()
        m = _make_manifest("p1", "id1", ["llama3:8b"], key)
        d.register_manifest(m)

        results = d.discover_supernodes("Llama3:8B")
        assert len(results) == 1

    def test_discover_multiple_providers(self, key):
        d = SupernodeDiscovery()
        key2 = Ed25519PrivateKey.generate()
        m1 = _make_manifest("p1", "id1", ["llama3:8b"], key)
        m2 = _make_manifest("p2", "id2", ["llama3:8b"], key2)
        d.register_manifest(m1)
        d.register_manifest(m2)

        results = d.discover_supernodes("llama3:8b")
        assert len(results) == 2
        peer_ids = {r.peer_id for r in results}
        assert peer_ids == {"p1", "p2"}

    def test_discover_unknown_model(self, key):
        d = SupernodeDiscovery()
        m = _make_manifest("p1", "id1", ["llama3:8b"], key)
        d.register_manifest(m)

        results = d.discover_supernodes("nonexistent:7b")
        assert len(results) == 0

    def test_multi_model_node(self, key):
        d = SupernodeDiscovery()
        m = _make_manifest("p1", "id1", ["llama3:8b", "qwen:2b"], key)
        d.register_manifest(m)

        assert len(d.discover_supernodes("llama3:8b")) == 1
        assert len(d.discover_supernodes("qwen:2b")) == 1


# ---------------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------------

class TestRemove:
    def test_remove_manifest(self, key):
        d = SupernodeDiscovery()
        m = _make_manifest("p1", "id1", ["llama3:8b"], key)
        d.register_manifest(m)

        d.remove_manifest("id1")
        assert len(d.discover_supernodes("llama3:8b")) == 0
        assert len(d.all_manifests()) == 0

    def test_remove_nonexistent(self):
        d = SupernodeDiscovery()
        d.remove_manifest("nope")  # should not raise


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------

class TestEnumeration:
    def test_all_manifests(self, key):
        d = SupernodeDiscovery()
        m1 = _make_manifest("p1", "id1", ["llama3:8b"], key)
        m2 = _make_manifest("p2", "id2", ["qwen:2b"], key)
        d.register_manifest(m1)
        d.register_manifest(m2)

        all_m = d.all_manifests()
        assert len(all_m) == 2

    def test_known_models(self, key):
        d = SupernodeDiscovery()
        m = _make_manifest("p1", "id1", ["llama3:8b", "qwen:2b"], key)
        d.register_manifest(m)

        models = d.known_models()
        assert models == ["llama3:8b", "qwen:2b"]


# ---------------------------------------------------------------------------
# Manifest update (re-register)
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_re_register_updates(self, key):
        d = SupernodeDiscovery()
        m1 = _make_manifest("p1", "id1", ["llama3:8b"], key)
        d.register_manifest(m1)

        m2 = _make_manifest("p1", "id1", ["llama3:8b", "qwen:2b"], key)
        d.register_manifest(m2)

        assert len(d.all_manifests()) == 1
        assert len(d.known_models()) == 2


# ---------------------------------------------------------------------------
# Prune
# ---------------------------------------------------------------------------

class TestPrune:
    def test_prune_removes_nothing_when_fresh(self, key):
        d = SupernodeDiscovery()
        m = _make_manifest("p1", "id1", ["llama3:8b"], key)
        d.register_manifest(m)
        assert d.prune_stale() == 0
        assert len(d.all_manifests()) == 1


# ---------------------------------------------------------------------------
# DHT discovery
# ---------------------------------------------------------------------------

class MockDHTNode:
    def __init__(self, providers=None, records=None):
        self._providers = providers or []
        self._records = records or {}

    def get_providers(self, key: bytes) -> list[str]:
        return self._providers

    def get_record_raw(self, key: bytes) -> bytes | None:
        return self._records.get(key)


class TestDHTDiscovery:
    def test_discover_from_dht(self, key):
        m = _make_manifest("p1", "12D3KooW1", ["llama3:8b"], key)
        manifest_key = b"/openhydra/supernode/12D3KooW1"

        dht = MockDHTNode(
            providers=["12D3KooW1"],
            records={manifest_key: m.to_cbor()},
        )
        d = SupernodeDiscovery()
        results = d.discover_from_dht("llama3:8b", dht)
        assert len(results) == 1
        assert results[0].peer_id == "p1"

    def test_dht_skips_cached(self, key):
        m = _make_manifest("p1", "12D3KooW1", ["llama3:8b"], key)
        dht = MockDHTNode(
            providers=["12D3KooW1"],
            records={b"/openhydra/supernode/12D3KooW1": m.to_cbor()},
        )
        d = SupernodeDiscovery()
        d.register_manifest(m)

        results = d.discover_from_dht("llama3:8b", dht)
        assert len(results) == 1

    def test_dht_no_providers(self, key):
        dht = MockDHTNode(providers=[])
        d = SupernodeDiscovery()
        results = d.discover_from_dht("llama3:8b", dht)
        assert len(results) == 0

    def test_dht_missing_record(self, key):
        dht = MockDHTNode(providers=["12D3KooW1"], records={})
        d = SupernodeDiscovery()
        results = d.discover_from_dht("llama3:8b", dht)
        assert len(results) == 0

    def test_dht_bad_cbor(self, key):
        dht = MockDHTNode(
            providers=["12D3KooW1"],
            records={b"/openhydra/supernode/12D3KooW1": b"not-cbor"},
        )
        d = SupernodeDiscovery()
        results = d.discover_from_dht("llama3:8b", dht)
        assert len(results) == 0
