from concurrent import futures

import grpc
import pytest

from coordinator.chain import InferenceChain
from coordinator.path_finder import PeerEndpoint
from peer.crypto import cryptography_available
from peer.server import PeerService
from peer import peer_pb2_grpc


def _start_peer(
    peer_id: str,
    shard_index: int,
    broken: bool = False,
    advanced_encryption_enabled: bool = False,
    advanced_encryption_seed: str = "openhydra-tier3-dev-seed",
):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    peer_pb2_grpc.add_PeerServicer_to_server(
        PeerService(
            peer_id=peer_id,
            model_id="openhydra-toy-345m",
            shard_index=shard_index,
            total_shards=3,
            daemon_mode="polite",
            broken=broken,
            advanced_encryption_enabled=advanced_encryption_enabled,
            advanced_encryption_seed=advanced_encryption_seed,
        ),
        server,
    )
    try:
        port = server.add_insecure_port("127.0.0.1:0")
    except RuntimeError as exc:
        pytest.skip(f"grpc listener unavailable in this environment: {exc}")

    if port == 0:
        pytest.skip("grpc listener unavailable in this environment")

    server.start()
    return server, port


def test_chain_runs_across_three_peers():
    started = [
        _start_peer("peer-a", 0),
        _start_peer("peer-b", 1),
        _start_peer("peer-c", 2),
    ]
    servers = [srv for srv, _ in started]
    ports = [port for _, port in started]

    try:
        pipeline = [
            PeerEndpoint(peer_id="peer-a", host="127.0.0.1", port=ports[0]),
            PeerEndpoint(peer_id="peer-b", host="127.0.0.1", port=ports[1]),
            PeerEndpoint(peer_id="peer-c", host="127.0.0.1", port=ports[2]),
        ]
        chain = InferenceChain(pipeline, timeout_ms=2000)
        result = chain.run("OpenHydra test prompt", max_tokens=12)
        assert result.text.endswith(".")
        assert len(result.traces) == 3
        assert result.latency_ms > 0
    finally:
        for server in servers:
            server.stop(grace=0)


@pytest.mark.skipif(not cryptography_available(), reason="cryptography dependency unavailable")
def test_chain_runs_across_three_peers_with_advanced_encryption():
    seed = "integration-seed"
    started = [
        _start_peer("peer-a", 0, advanced_encryption_enabled=True, advanced_encryption_seed=seed),
        _start_peer("peer-b", 1, advanced_encryption_enabled=True, advanced_encryption_seed=seed),
        _start_peer("peer-c", 2, advanced_encryption_enabled=True, advanced_encryption_seed=seed),
    ]
    servers = [srv for srv, _ in started]
    ports = [port for _, port in started]

    try:
        pipeline = [
            PeerEndpoint(peer_id="peer-a", host="127.0.0.1", port=ports[0]),
            PeerEndpoint(peer_id="peer-b", host="127.0.0.1", port=ports[1]),
            PeerEndpoint(peer_id="peer-c", host="127.0.0.1", port=ports[2]),
        ]
        chain = InferenceChain(
            pipeline,
            timeout_ms=2000,
            advanced_encryption_enabled=True,
            advanced_encryption_seed=seed,
            advanced_encryption_level="enhanced",
        )
        result = chain.run("OpenHydra encrypted prompt", max_tokens=10)
        assert result.text.endswith(".")
        assert len(result.traces) == 3
        assert result.encryption is not None
        assert result.encryption["enabled"] is True
        assert result.encryption["layers_per_hop"] == 2
        assert result.encryption["encrypted_hops"] == 2
        assert result.encryption["onion_routing"] is True
        assert result.encryption["onion_layers"] == 3
        assert result.encryption["onion_layers_peeled"] == 3
    finally:
        for server in servers:
            server.stop(grace=0)
