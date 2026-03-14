import json
from http.server import ThreadingHTTPServer
import threading
from urllib import request

import pytest

from coordinator.path_finder import PeerEndpoint, load_peer_config, load_peers_from_dht
from dht.bootstrap import DhtBootstrapHandler
from dht.node import InMemoryDhtNode


def _post_json(url: str, payload: dict) -> None:
    req = request.Request(
        url,
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8"),
    )
    with request.urlopen(req, timeout=2.0):
        pass


def _reset_bootstrap(ttl_seconds: int = 60) -> None:
    DhtBootstrapHandler.dht = InMemoryDhtNode(ttl_seconds=ttl_seconds)
    DhtBootstrapHandler.default_ttl_seconds = ttl_seconds
    DhtBootstrapHandler.default_dsht_replicas = 2
    DhtBootstrapHandler.default_dsht_max_replicas = 32
    DhtBootstrapHandler.default_lookup_window_seconds = 1
    DhtBootstrapHandler.default_lookup_max_requests_per_window = 120
    DhtBootstrapHandler.default_geo_challenge_enabled = False
    DhtBootstrapHandler.default_geo_challenge_timeout_ms = 800
    DhtBootstrapHandler.default_geo_max_rtt_ms = 50.0
    DhtBootstrapHandler.default_geo_challenge_seed = "openhydra-geo-dev-seed"
    DhtBootstrapHandler.default_expert_min_reputation_score = 60.0
    DhtBootstrapHandler.default_expert_min_staked_balance = 0.01
    DhtBootstrapHandler.default_expert_require_stake = True
    DhtBootstrapHandler._lookup_buckets = {}
    DhtBootstrapHandler._rebalance_hints = {}


def test_load_peers_from_dht_returns_endpoints():
    _reset_bootstrap(ttl_seconds=60)

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), DhtBootstrapHandler)
    except OSError as exc:
        pytest.skip(f"socket bind unavailable: {exc}")

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base = f"http://{host}:{port}"

    try:
        _post_json(
            f"{base}/announce",
            {
                "peer_id": "peer-a",
                "model_id": "openhydra-toy-345m",
                "host": "127.0.0.1",
                "port": 50051,
                "operator_id": "op-a",
                "region": "us-east",
                "seeding_enabled": True,
                "seed_upload_limit_mbps": 12.5,
                "seed_target_upload_limit_mbps": 10.0,
                "seed_inference_active": True,
                "runtime_backend": "toy_gpu_sim",
                "runtime_target": "cuda",
                "runtime_model_id": "Qwen/Qwen3.5-0.8B",
                "quantization_mode": "int4",
                "quantization_bits": 4,
                "runtime_gpu_available": True,
                "runtime_estimated_tokens_per_sec": 280.0,
                "runtime_estimated_memory_mb": 640,
                "reputation_score": 91.0,
                "staked_balance": 0.5,
                "expert_tags": ["vision", "CODE", "vision"],
                "expert_layer_indices": [5, 1, -2, "x", 5],
                "expert_router": True,
                "peer_public_key": "b" * 64,
            },
        )

        _post_json(
            f"{base}/announce",
            {
                "peer_id": "peer-b",
                "model_id": "openhydra-toy-345m",
                "host": "127.0.0.1",
                "port": 50052,
                "operator_id": "op-b",
                "region": "eu-west",
                "seeding_enabled": False,
                "seed_upload_limit_mbps": 0.0,
                "seed_target_upload_limit_mbps": 0.0,
                "seed_inference_active": False,
            },
        )

        peers = load_peers_from_dht(
            base,
            model_id="openhydra-toy-345m",
            preferred_region="us-east",
            limit=1,
            sloppy_factor=1,
            dsht_replicas=2,
        )
        assert len(peers) == 1
        assert peers[0].peer_id == "peer-a"
        assert peers[0].operator_id == "op-a"
        assert peers[0].region == "us-east"
        assert peers[0].seeding_enabled is True
        assert peers[0].seed_upload_limit_mbps == 12.5
        assert peers[0].seed_target_upload_limit_mbps == 10.0
        assert peers[0].seed_inference_active is True
        assert peers[0].runtime_backend == "toy_gpu_sim"
        assert peers[0].runtime_target == "cuda"
        assert peers[0].runtime_model_id == "Qwen/Qwen3.5-0.8B"
        assert peers[0].quantization_mode == "int4"
        assert peers[0].quantization_bits == 4
        assert peers[0].runtime_gpu_available is True
        assert peers[0].runtime_estimated_tokens_per_sec == 280.0
        assert peers[0].runtime_estimated_memory_mb == 640
        assert peers[0].expert_admission_approved is True
        assert peers[0].expert_tags == ("vision", "code")
        assert peers[0].expert_layer_indices == (1, 5)
        assert peers[0].expert_router is True
        assert peers[0].public_key_hex == "b" * 64
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


class _FakeResponse:
    """Minimal requests.Response stand-in for monkeypatching _DHT_SESSION.get."""

    def __init__(self, payload: dict):
        self._data = payload

    def json(self) -> dict:
        return self._data

    def raise_for_status(self) -> None:
        pass


def test_load_peers_from_multiple_dht_urls_merges_and_prefers_latest(monkeypatch):
    import requests as _requests

    def fake_get(url, timeout=2.0):  # type: ignore[no-untyped-def]
        if "dht-a" in url:
            return _FakeResponse(
                {
                    "peers": [
                        {"peer_id": "peer-1", "host": "10.0.0.1", "port": 5001, "updated_unix_ms": 100},
                        {"peer_id": "peer-2", "host": "10.0.0.2", "port": 5002, "updated_unix_ms": 80},
                    ]
                }
            )
        if "dht-b" in url:
            return _FakeResponse(
                {
                    "peers": [
                        {"peer_id": "peer-1", "host": "10.0.1.1", "port": 5101, "updated_unix_ms": 250},
                        {"peer_id": "peer-3", "host": "10.0.0.3", "port": 5003, "updated_unix_ms": 60},
                    ]
                }
            )
        raise _requests.exceptions.Timeout(f"unexpected_url:{url}")

    monkeypatch.setattr("coordinator.path_finder._DHT_SESSION.get", fake_get)
    peers = load_peers_from_dht(
        model_id="openhydra-toy-345m",
        dht_urls=["http://dht-a:8468", "http://dht-b:8468"],
        timeout_s=0.5,
    )

    assert [peer.peer_id for peer in peers] == ["peer-1", "peer-2", "peer-3"]
    assert peers[0].host == "10.0.1.1"
    assert peers[0].port == 5101


def test_load_peers_from_multiple_dht_urls_allows_partial_failure(monkeypatch, caplog):
    import requests as _requests

    def fake_get(url, timeout=2.0):  # type: ignore[no-untyped-def]
        if "dht-ok" in url:
            return _FakeResponse({"peers": [{"peer_id": "peer-ok", "host": "127.0.0.1", "port": 5001, "updated_unix_ms": 20}]})
        raise _requests.exceptions.Timeout("boom")

    monkeypatch.setattr("coordinator.path_finder._DHT_SESSION.get", fake_get)
    with caplog.at_level("WARNING"):
        peers = load_peers_from_dht(
            model_id="openhydra-toy-345m",
            dht_urls=["http://dht-ok:8468", "http://dht-down:8468"],
            timeout_s=0.5,
        )

    assert [peer.peer_id for peer in peers] == ["peer-ok"]
    assert any("dht_lookup_partial_failure" in record.message for record in caplog.records)


def test_peer_endpoint_has_public_key_hex_field():
    endpoint = PeerEndpoint(peer_id="p1", host="127.0.0.1", port=9000)
    assert hasattr(endpoint, "public_key_hex")
    assert endpoint.public_key_hex == ""


def test_load_peer_config_reads_public_key_hex(tmp_path):
    path = tmp_path / "peers.json"
    path.write_text(
        json.dumps(
            [
                {
                    "peer_id": "peer-a",
                    "host": "127.0.0.1",
                    "port": 5001,
                    "public_key_hex": "c" * 64,
                }
            ]
        )
    )
    peers = load_peer_config(path)
    assert len(peers) == 1
    assert peers[0].public_key_hex == "c" * 64
