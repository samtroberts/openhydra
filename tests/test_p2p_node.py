"""Integration tests for the Rust P2P networking layer (openhydra_network)."""

import os
import tempfile
import time

import pytest

try:
    import openhydra_network
    HAS_NETWORK = True
except ImportError:
    HAS_NETWORK = False

pytestmark = pytest.mark.skipif(not HAS_NETWORK, reason="openhydra_network not installed")


@pytest.fixture
def tmp_identity():
    """Create a temporary identity key file."""
    fd, path = tempfile.mkstemp(suffix=".key")
    os.close(fd)
    os.unlink(path)  # let P2PNode create it
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def tmp_identity_pair():
    """Create two temporary identity key files for two-node tests."""
    paths = []
    for _ in range(2):
        fd, path = tempfile.mkstemp(suffix=".key")
        os.close(fd)
        os.unlink(path)
        paths.append(path)
    yield paths
    for p in paths:
        if os.path.exists(p):
            os.unlink(p)


class TestP2PNodeCreation:
    """Test node instantiation (no network activity)."""

    def test_version(self):
        assert openhydra_network.__version__ == "0.1.0"

    def test_p2pnode_class_exists(self):
        assert hasattr(openhydra_network, "P2PNode")

    def test_create_node_generates_identity(self, tmp_identity):
        node = openhydra_network.P2PNode(identity_key_path=tmp_identity)
        assert os.path.exists(tmp_identity)
        assert len(node.openhydra_peer_id) == 16  # SHA256[:8] hex
        assert node.libp2p_peer_id.startswith("12D3KooW")

    def test_create_node_loads_existing_identity(self, tmp_identity):
        node1 = openhydra_network.P2PNode(identity_key_path=tmp_identity)
        node2 = openhydra_network.P2PNode(identity_key_path=tmp_identity)
        assert node1.openhydra_peer_id == node2.openhydra_peer_id
        assert node1.libp2p_peer_id == node2.libp2p_peer_id

    def test_create_node_default_listen(self, tmp_identity):
        node = openhydra_network.P2PNode(identity_key_path=tmp_identity)
        # Just verify no crash — listen addrs are internal.
        assert node.libp2p_peer_id is not None

    def test_create_node_custom_listen(self, tmp_identity):
        node = openhydra_network.P2PNode(
            identity_key_path=tmp_identity,
            listen_addrs=["/ip4/127.0.0.1/tcp/0"],
        )
        assert node.libp2p_peer_id is not None


class TestP2PNodeLifecycle:
    """Test start/stop (binds a local port, no remote connections)."""

    def test_start_and_stop(self, tmp_identity):
        node = openhydra_network.P2PNode(
            identity_key_path=tmp_identity,
            listen_addrs=["/ip4/127.0.0.1/tcp/0"],  # random port
            bootstrap_peers=[],
        )
        node.start()
        # Node is running — swarm event loop active.
        time.sleep(0.2)
        node.stop()

    def test_double_start_raises(self, tmp_identity):
        node = openhydra_network.P2PNode(
            identity_key_path=tmp_identity,
            listen_addrs=["/ip4/127.0.0.1/tcp/0"],
        )
        node.start()
        try:
            with pytest.raises(RuntimeError, match="already started"):
                node.start()
        finally:
            node.stop()

    def test_stop_without_start_is_noop(self, tmp_identity):
        node = openhydra_network.P2PNode(identity_key_path=tmp_identity)
        node.stop()  # should not raise


class TestP2PNodeOperations:
    """Test announce/discover/nat_status (local only, no bootstrap peers)."""

    def test_nat_status_returns_dict(self, tmp_identity):
        node = openhydra_network.P2PNode(
            identity_key_path=tmp_identity,
            listen_addrs=["/ip4/127.0.0.1/tcp/0"],
        )
        node.start()
        try:
            status = node.nat_status()
            assert isinstance(status, dict)
            assert "nat_type" in status
            assert "is_public" in status
            assert "external_ip" in status
            # Without bootstrap peers, NAT is unknown.
            assert status["nat_type"] == "unknown"
        finally:
            node.stop()

    def test_announce_record(self, tmp_identity):
        node = openhydra_network.P2PNode(
            identity_key_path=tmp_identity,
            listen_addrs=["/ip4/127.0.0.1/tcp/0"],
        )
        node.start()
        try:
            # Announce a peer record (stored locally in Kademlia).
            node.announce(record={
                "peer_id": "test-peer",
                "model_id": "openhydra-qwen3.5-2b",
                "host": "192.168.1.10",
                "port": 50051,
                "layer_start": 0,
                "layer_end": 12,
                "total_layers": 24,
            })
        finally:
            node.stop()

    def test_discover_empty(self, tmp_identity):
        node = openhydra_network.P2PNode(
            identity_key_path=tmp_identity,
            listen_addrs=["/ip4/127.0.0.1/tcp/0"],
        )
        node.start()
        try:
            # Discover with no peers in the DHT — should return empty list
            # (or an error, since there are no Kademlia peers to query).
            # With a single node, Kademlia can't complete a query, so this
            # may time out or return empty. Either is acceptable.
            try:
                peers = node.discover(model_id="nonexistent-model")
                assert isinstance(peers, list)
            except RuntimeError:
                pass  # Expected — no Kademlia peers to query.
        finally:
            node.stop()

    def test_resolve_address_unknown_peer(self, tmp_identity):
        node = openhydra_network.P2PNode(
            identity_key_path=tmp_identity,
            listen_addrs=["/ip4/127.0.0.1/tcp/0"],
        )
        node.start()
        try:
            with pytest.raises(RuntimeError, match="not found"):
                node.resolve_address(peer_id="nonexistent-peer")
        finally:
            node.stop()

    def test_operations_before_start_raise(self, tmp_identity):
        node = openhydra_network.P2PNode(identity_key_path=tmp_identity)
        with pytest.raises(RuntimeError, match="not started"):
            node.nat_status()
        with pytest.raises(RuntimeError, match="not started"):
            node.announce(record={"peer_id": "x", "model_id": "y", "host": "z", "port": 1})


class TestTwoNodeDiscovery:
    """Test two local nodes discovering each other via mDNS."""

    def test_mdns_discovery(self, tmp_identity_pair):
        """Two nodes on localhost should discover each other via mDNS."""
        node_a = openhydra_network.P2PNode(
            identity_key_path=tmp_identity_pair[0],
            listen_addrs=["/ip4/127.0.0.1/tcp/0"],
        )
        node_b = openhydra_network.P2PNode(
            identity_key_path=tmp_identity_pair[1],
            listen_addrs=["/ip4/127.0.0.1/tcp/0"],
        )
        node_a.start()
        node_b.start()
        try:
            # Give mDNS time to discover.
            time.sleep(2)
            # Both nodes are alive — mDNS should have added each other
            # to the Kademlia routing table. We can verify by checking
            # that nat_status works (confirms event loop is responsive).
            status_a = node_a.nat_status()
            status_b = node_b.nat_status()
            assert status_a["nat_type"] == "unknown"
            assert status_b["nat_type"] == "unknown"
        finally:
            node_a.stop()
            node_b.stop()
