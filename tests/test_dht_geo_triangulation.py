import json
from http.server import ThreadingHTTPServer
import threading
from urllib import request

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
    with request.urlopen(req, timeout=3.0) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str) -> dict:
    with request.urlopen(url, timeout=3.0) as response:
        return json.loads(response.read().decode("utf-8"))


def _reset_geo_bootstrap(*, seed: str, max_rtt_ms: float) -> None:
    DhtBootstrapHandler.dht = InMemoryDhtNode(ttl_seconds=60)
    DhtBootstrapHandler.default_ttl_seconds = 60
    DhtBootstrapHandler.default_dsht_replicas = 2
    DhtBootstrapHandler.default_dsht_max_replicas = 32
    DhtBootstrapHandler.default_lookup_window_seconds = 1
    DhtBootstrapHandler.default_lookup_max_requests_per_window = 120
    DhtBootstrapHandler.default_geo_challenge_enabled = True
    DhtBootstrapHandler.default_geo_challenge_timeout_ms = 1500
    DhtBootstrapHandler.default_geo_max_rtt_ms = float(max_rtt_ms)
    DhtBootstrapHandler.default_geo_challenge_seed = seed
    DhtBootstrapHandler.default_expert_min_reputation_score = 60.0
    DhtBootstrapHandler._lookup_buckets = {}
    DhtBootstrapHandler._rebalance_hints = {}


def test_dht_geo_challenge_pending_libp2p_bootstrap():
    """Geo-challenge now defers to libp2p bootstrap (gRPC removed).

    With the unified libp2p transport, the geo-challenge gRPC Ping is
    disabled.  _geo_verify_record returns verified=True with reason
    'challenge_pending_libp2p_bootstrap', so the claimed region is
    accepted until libp2p-native geo-verification is implemented.
    """
    seed = "geo-triangulation-test-seed"
    _reset_geo_bootstrap(seed=seed, max_rtt_ms=50.0)

    try:
        dht_server = ThreadingHTTPServer(("127.0.0.1", 0), DhtBootstrapHandler)
    except OSError as exc:
        pytest.skip(f"socket bind unavailable: {exc}")

    dht_thread = threading.Thread(target=dht_server.serve_forever, daemon=True)
    dht_thread.start()
    host, port = dht_server.server_address
    base = f"http://{host}:{port}"

    try:
        ack = _post_json(
            f"{base}/announce",
            {
                "peer_id": "peer-geo-pending",
                "model_id": "openhydra-toy-345m",
                "host": "127.0.0.1",
                "port": 50051,
                "region": "us-east",
                "operator_id": "op-geo",
                "load_pct": 1.0,
            },
        )
        assert ack["ok"] is True

        lookup = _get_json(f"{base}/lookup?model_id=openhydra-toy-345m")
        assert lookup["count"] == 1
        peer = lookup["peers"][0]
        assert peer["peer_id"] == "peer-geo-pending"
        assert peer["region"] == "us-east"
        assert peer["geo_verified"] is True
        assert peer["geo_challenge_reason"] == "challenge_pending_libp2p_bootstrap"
    finally:
        dht_server.shutdown()
        dht_server.server_close()
        dht_thread.join(timeout=2.0)
