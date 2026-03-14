import json
from http.server import ThreadingHTTPServer
import threading
from urllib import request
from urllib.error import HTTPError

import pytest

from dht.bootstrap import DhtBootstrapHandler
from dht.node import InMemoryDhtNode


def _post_json(url: str, payload: dict) -> dict:
    req = request.Request(
        url,
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8"),
    )
    with request.urlopen(req, timeout=2.0) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str) -> dict:
    with request.urlopen(url, timeout=2.0) as response:
        return json.loads(response.read().decode("utf-8"))


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


def test_bootstrap_announce_and_lookup():
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
        ack = _post_json(
            f"{base}/announce",
            {
                "peer_id": "peer-a",
                "model_id": "openhydra-toy-345m",
                "host": "127.0.0.1",
                "port": 50051,
                "operator_id": "op-a",
                "load_pct": 0,
                "daemon_mode": "polite",
                "runtime_backend": "toy_gpu_sim",
                "runtime_target": "cuda",
                "runtime_model_id": "Qwen/Qwen3.5-0.8B",
                "quantization_mode": "int8",
                "quantization_bits": 8,
                "runtime_gpu_available": True,
                "runtime_estimated_tokens_per_sec": 222.5,
                "runtime_estimated_memory_mb": 860,
            },
        )
        assert ack["ok"] is True

        data = _get_json(f"{base}/lookup?model_id=openhydra-toy-345m")
        assert data["count"] == 1
        assert data["peers"][0]["peer_id"] == "peer-a"
        assert data["peers"][0]["operator_id"] == "op-a"
        assert data["peers"][0]["runtime_backend"] == "toy_gpu_sim"
        assert data["peers"][0]["runtime_target"] == "cuda"
        assert data["peers"][0]["runtime_model_id"] == "Qwen/Qwen3.5-0.8B"
        assert data["peers"][0]["quantization_mode"] == "int8"
        assert data["peers"][0]["quantization_bits"] == 8
        assert data["peers"][0]["runtime_gpu_available"] is True
        assert data["peers"][0]["runtime_estimated_tokens_per_sec"] == 222.5
        assert data["peers"][0]["runtime_estimated_memory_mb"] == 860
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_bootstrap_operator_id_defaults_to_null():
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
                "peer_id": "peer-b",
                "model_id": "openhydra-toy-345m",
                "host": "127.0.0.1",
                "port": 50052,
            },
        )
        data = _get_json(f"{base}/lookup?model_id=openhydra-toy-345m")
        assert data["count"] == 1
        assert data["peers"][0]["operator_id"] is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_bootstrap_announce_and_lookup_preserves_expert_metadata():
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
        ack = _post_json(
            f"{base}/announce",
            {
                "peer_id": "peer-expert",
                "model_id": "openhydra-toy-345m",
                "host": "127.0.0.1",
                "port": 50061,
                "reputation_score": 92.0,
                "staked_balance": 1.2,
                "expert_tags": ["vision", "CODE", "vision"],
                "expert_layer_indices": [7, 2, 7, -1, "x"],
                "expert_router": True,
            },
        )
        assert ack["ok"] is True

        data = _get_json(f"{base}/lookup?model_id=openhydra-toy-345m")
        assert data["count"] == 1
        assert data["peers"][0]["peer_id"] == "peer-expert"
        assert data["peers"][0]["expert_tags"] == ["vision", "code"]
        assert data["peers"][0]["expert_layer_indices"] == [2, 7]
        assert data["peers"][0]["expert_router"] is True
        assert data["peers"][0]["expert_admission_approved"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_bootstrap_rejects_expert_claim_for_low_reputation_or_unstaked_peer():
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
                "peer_id": "peer-lowrep",
                "model_id": "openhydra-toy-345m",
                "host": "127.0.0.1",
                "port": 50062,
                "reputation_score": 20.0,
                "staked_balance": 0.0,
                "expert_tags": ["niche-finance"],
                "expert_layer_indices": [10],
                "expert_router": True,
            },
        )

        data = _get_json(f"{base}/lookup?model_id=openhydra-toy-345m")
        assert data["count"] == 1
        peer = data["peers"][0]
        assert peer["peer_id"] == "peer-lowrep"
        assert peer["expert_admission_approved"] is False
        assert peer["expert_admission_reason"] in {"low_reputation", "unstaked_or_new"}
        assert peer["expert_tags"] == []
        assert peer["expert_layer_indices"] == []
        assert peer["expert_router"] is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_normalize_peer_record_passes_through_public_key():
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
                "peer_id": "peer-key",
                "model_id": "openhydra-toy-345m",
                "host": "127.0.0.1",
                "port": 50072,
                "peer_public_key": "a" * 64,
            },
        )
        data = _get_json(f"{base}/lookup?model_id=openhydra-toy-345m")
        assert data["count"] == 1
        assert data["peers"][0]["peer_public_key"] == "a" * 64
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_normalize_peer_record_drops_invalid_public_key():
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
                "peer_id": "peer-key-invalid",
                "model_id": "openhydra-toy-345m",
                "host": "127.0.0.1",
                "port": 50073,
                "peer_public_key": "short",
            },
        )
        data = _get_json(f"{base}/lookup?model_id=openhydra-toy-345m")
        assert data["count"] == 1
        assert data["peers"][0]["peer_public_key"] == ""
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_lookup_prefers_region_with_sloppy_fallback_and_limit():
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
                "peer_id": "peer-us-a",
                "model_id": "openhydra-toy-345m",
                "host": "127.0.0.1",
                "port": 50051,
                "operator_id": "op-a",
                "region": "us-east",
                "load_pct": 10,
                "bandwidth_mbps": 700,
            },
        )
        _post_json(
            f"{base}/announce",
            {
                "peer_id": "peer-eu",
                "model_id": "openhydra-toy-345m",
                "host": "127.0.0.1",
                "port": 50052,
                "operator_id": "op-b",
                "region": "eu-west",
                "load_pct": 5,
                "bandwidth_mbps": 500,
            },
        )
        _post_json(
            f"{base}/announce",
            {
                "peer_id": "peer-us-b",
                "model_id": "openhydra-toy-345m",
                "host": "127.0.0.1",
                "port": 50053,
                "operator_id": "op-c",
                "region": "us-east",
                "load_pct": 20,
                "bandwidth_mbps": 600,
            },
        )

        preferred = _get_json(
            f"{base}/lookup?model_id=openhydra-toy-345m&preferred_region=us-east&limit=2&sloppy_factor=1"
        )
        assert preferred["count"] == 2
        assert [item["peer_id"] for item in preferred["peers"]] == ["peer-us-a", "peer-us-b"]

        fallback = _get_json(
            f"{base}/lookup?model_id=openhydra-toy-345m&preferred_region=ap-south&limit=2&sloppy_factor=1"
        )
        assert fallback["count"] == 2
        assert fallback["peers"][0]["peer_id"] == "peer-eu"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_lookup_uses_dsht_replicas_and_dedupes_results():
    _reset_bootstrap(ttl_seconds=60)
    DhtBootstrapHandler.default_dsht_replicas = 3

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
            },
        )

        health = _get_json(f"{base}/health")
        keys = sorted(health["keys"])
        assert "model:openhydra-toy-345m" in keys
        assert "model:openhydra-toy-345m:dsht:0" in keys
        assert "model:openhydra-toy-345m:dsht:1" in keys
        assert "model:openhydra-toy-345m:dsht:2" in keys

        lookup = _get_json(f"{base}/lookup?model_id=openhydra-toy-345m&dsht_replicas=3")
        assert lookup["count"] == 1
        assert lookup["peers"][0]["peer_id"] == "peer-a"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_lookup_rate_limit_returns_429():
    _reset_bootstrap(ttl_seconds=60)
    DhtBootstrapHandler.default_lookup_window_seconds = 60
    DhtBootstrapHandler.default_lookup_max_requests_per_window = 1
    DhtBootstrapHandler.default_dsht_replicas = 2
    DhtBootstrapHandler.default_dsht_max_replicas = 8

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
                "peer_id": "peer-rate",
                "model_id": "openhydra-toy-345m",
                "host": "127.0.0.1",
                "port": 50059,
            },
        )

        first = _get_json(f"{base}/lookup?model_id=openhydra-toy-345m")
        assert first["count"] == 1

        with pytest.raises(HTTPError) as exc_info:
            request.urlopen(f"{base}/lookup?model_id=openhydra-toy-345m", timeout=2.0)
        err = exc_info.value
        assert err.code == 429
        payload = json.loads(err.read().decode("utf-8"))
        assert payload["error"] == "lookup_rate_limited"
        assert int(payload["retry_after_seconds"]) >= 1
        assert payload["rebalance"]["active"] is True
        assert int(payload["rebalance"]["recommended_dsht_replicas"]) >= 3

        health = _get_json(f"{base}/health")
        keys = health["keys"]
        assert any(key.startswith("model:openhydra-toy-345m:dsht:2") for key in keys)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
