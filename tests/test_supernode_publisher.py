"""Tests for supernode.publisher — ManifestPublisher lifecycle."""

import time

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from supernode.adapter import (
    SupernodeAdapter,
    PromptRequest,
    TokenChunk,
    ModelInfo,
    BackendStatus,
)
from supernode.discovery import SupernodeDiscovery
from supernode.manifest import HardwareInfo
from supernode.publisher import ManifestPublisher


# ---------------------------------------------------------------------------
# Mock adapter
# ---------------------------------------------------------------------------

class StubAdapter(SupernodeAdapter):
    def __init__(self, models=None):
        self._models = models or [
            ModelInfo("llama3:8b", "llama", 8000, "Q4_0", 8192),
        ]
        self._loaded = [m.model_id for m in self._models]

    async def list_models(self):
        return self._models

    async def generate(self, request):
        yield TokenChunk(token="x", finish_reason="stop")

    async def cancel(self, request_id):
        pass

    async def get_status(self):
        return BackendStatus(
            current_load=0.0,
            active_requests=0,
            max_concurrent=4,
            gpu_memory_free_mb=12000,
            models_loaded=self._loaded,
        )

    async def health_check(self):
        return True

    async def warmup(self, model_id):
        return True


class FailingListAdapter(StubAdapter):
    async def list_models(self):
        raise ConnectionError("unreachable")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def key():
    return Ed25519PrivateKey.generate()


@pytest.fixture
def discovery():
    return SupernodeDiscovery()


@pytest.fixture
def adapter():
    return StubAdapter()


@pytest.fixture
def publisher(adapter, discovery, key):
    return ManifestPublisher(
        adapter=adapter,
        discovery=discovery,
        private_key=key,
        peer_id="test-peer",
        libp2p_peer_id="12D3KooWTest",
        refresh_interval=0.1,
    )


# ---------------------------------------------------------------------------
# Tests: publish_now (synchronous, no background thread)
# ---------------------------------------------------------------------------

class TestPublishNow:
    def test_publishes_manifest(self, publisher, discovery):
        publisher.publish_now()
        assert publisher.publish_count == 1
        manifests = discovery.all_manifests()
        assert len(manifests) == 1
        assert manifests[0].peer_id == "test-peer"

    def test_manifest_has_models(self, publisher, discovery):
        publisher.publish_now()
        m = discovery.all_manifests()[0]
        assert len(m.models) == 1
        assert m.models[0].model_id == "llama3:8b"

    def test_manifest_signed(self, publisher, discovery):
        publisher.publish_now()
        m = discovery.all_manifests()[0]
        assert m.verify_signature() is True
        assert m.is_fresh() is True

    def test_manifest_warm_status(self, publisher, discovery):
        publisher.publish_now()
        m = discovery.all_manifests()[0]
        assert m.models[0].warm is True

    def test_manifest_adapter_fields(self, publisher, discovery):
        publisher.publish_now()
        m = discovery.all_manifests()[0]
        assert m.backend_type == "stub"
        assert m.trust_tier == "unverified"
        assert m.integration_level == 1

    def test_discoverable_by_model(self, publisher, discovery):
        publisher.publish_now()
        results = discovery.discover_supernodes("llama3:8b")
        assert len(results) == 1
        assert results[0].peer_id == "test-peer"

    def test_multiple_models(self, discovery, key):
        adapter = StubAdapter(models=[
            ModelInfo("llama3:8b", "llama", 8000, "Q4_0", 8192),
            ModelInfo("qwen:2b", "qwen", 2500, "Q8_0", 32768),
        ])
        pub = ManifestPublisher(
            adapter=adapter, discovery=discovery, private_key=key,
            peer_id="p1", libp2p_peer_id="id1",
        )
        pub.publish_now()

        assert len(discovery.discover_supernodes("llama3:8b")) == 1
        assert len(discovery.discover_supernodes("qwen:2b")) == 1

    def test_failing_adapter_no_crash(self, discovery, key):
        adapter = FailingListAdapter()
        pub = ManifestPublisher(
            adapter=adapter, discovery=discovery, private_key=key,
            peer_id="p1", libp2p_peer_id="id1",
        )
        pub.publish_now()
        assert pub.publish_count == 0
        assert len(discovery.all_manifests()) == 0


# ---------------------------------------------------------------------------
# Tests: start/stop lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_start_stop(self, publisher, discovery):
        publisher.start()
        assert publisher.is_running is True

        time.sleep(0.3)
        assert publisher.publish_count >= 1

        publisher.stop()
        assert publisher.is_running is False

    def test_stop_removes_manifest(self, publisher, discovery):
        publisher.publish_now()
        assert len(discovery.all_manifests()) == 1

        publisher.stop()
        assert len(discovery.all_manifests()) == 0

    def test_refresh_publishes_multiple(self, publisher, discovery):
        publisher.start()
        time.sleep(0.5)
        publisher.stop()
        assert publisher.publish_count >= 2

    def test_double_start(self, publisher):
        publisher.start()
        publisher.start()  # should be idempotent
        time.sleep(0.15)
        publisher.stop()

    def test_double_stop(self, publisher):
        publisher.start()
        time.sleep(0.15)
        publisher.stop()
        publisher.stop()  # should not raise


# ---------------------------------------------------------------------------
# Tests: model change detection
# ---------------------------------------------------------------------------

class TestModelChange:
    def test_detects_model_change(self, discovery, key):
        adapter = StubAdapter(models=[
            ModelInfo("llama3:8b", "llama", 8000, "Q4_0", 8192),
        ])
        pub = ManifestPublisher(
            adapter=adapter, discovery=discovery, private_key=key,
            peer_id="p1", libp2p_peer_id="id1",
        )
        pub.publish_now()
        assert len(discovery.known_models()) == 1

        adapter._models = [
            ModelInfo("llama3:8b", "llama", 8000, "Q4_0", 8192),
            ModelInfo("qwen:2b", "qwen", 2500, "Q8_0", 32768),
        ]
        adapter._loaded = ["llama3:8b", "qwen:2b"]
        pub.publish_now()

        assert len(discovery.known_models()) == 2
        assert pub.publish_count == 2


# ---------------------------------------------------------------------------
# Tests: DHT integration
# ---------------------------------------------------------------------------

class MockDHTNode:
    def __init__(self):
        self.put_record_calls: list[tuple[bytes, bytes]] = []
        self.start_providing_calls: list[bytes] = []
        self.stop_providing_calls: list[bytes] = []

    def put_record_raw(self, key: bytes, value: bytes):
        self.put_record_calls.append((key, value))

    def start_providing(self, key: bytes):
        self.start_providing_calls.append(key)

    def stop_providing(self, key: bytes):
        self.stop_providing_calls.append(key)


class TestDHTPublishing:
    def test_publish_puts_record_and_provides(self, discovery, key):
        dht_node = MockDHTNode()
        adapter = StubAdapter()
        pub = ManifestPublisher(
            adapter=adapter, discovery=discovery, private_key=key,
            peer_id="test-peer", libp2p_peer_id="12D3KooWTest",
            p2p_node=dht_node,
        )
        pub.publish_now()

        assert len(dht_node.put_record_calls) == 1
        rec_key = dht_node.put_record_calls[0][0]
        assert b"/openhydra/supernode/12D3KooWTest" in rec_key

        assert len(dht_node.start_providing_calls) == 1
        prov_key = dht_node.start_providing_calls[0]
        assert b"/openhydra/model/" in prov_key

    def test_stop_calls_stop_providing(self, discovery, key):
        dht_node = MockDHTNode()
        adapter = StubAdapter()
        pub = ManifestPublisher(
            adapter=adapter, discovery=discovery, private_key=key,
            peer_id="test-peer", libp2p_peer_id="12D3KooWTest",
            p2p_node=dht_node,
        )
        pub.publish_now()
        pub.stop()

        assert len(dht_node.stop_providing_calls) == 1

    def test_no_dht_without_node(self, discovery, key):
        adapter = StubAdapter()
        pub = ManifestPublisher(
            adapter=adapter, discovery=discovery, private_key=key,
            peer_id="test-peer", libp2p_peer_id="12D3KooWTest",
        )
        pub.publish_now()
        pub.stop()
        # No crash — DHT calls are skipped when p2p_node is None
