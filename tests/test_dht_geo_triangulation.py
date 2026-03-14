import json
from concurrent import futures
from http.server import ThreadingHTTPServer
import threading
import time
from urllib import request

import grpc
import pytest

from dht.bootstrap import DhtBootstrapHandler
from dht.node import InMemoryDhtNode
from peer.server import PeerService
from peer import peer_pb2_grpc


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


class _DelayedPingPeerService(PeerService):
    def __init__(self, *, ping_delay_s: float, **kwargs):
        super().__init__(**kwargs)
        self._ping_delay_s = max(0.0, float(ping_delay_s))

    def Ping(self, request, context):
        if self._ping_delay_s > 0.0:
            time.sleep(self._ping_delay_s)
        return super().Ping(request, context)


def _start_delayed_peer(seed: str, delay_s: float) -> tuple[grpc.Server, int]:
    service = _DelayedPingPeerService(
        ping_delay_s=delay_s,
        peer_id="peer-geo-delayed",
        model_id="openhydra-toy-345m",
        shard_index=0,
        total_shards=1,
        daemon_mode="polite",
        broken=False,
        runtime_backend="toy_cpu",
        runtime_target="cpu",
        quantization_mode="fp32",
        geo_challenge_seed=seed,
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    peer_pb2_grpc.add_PeerServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    if port == 0:
        raise RuntimeError("grpc_listener_unavailable")
    server.start()
    return server, port


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
    DhtBootstrapHandler.default_expert_min_staked_balance = 0.01
    DhtBootstrapHandler.default_expert_require_stake = True
    DhtBootstrapHandler._lookup_buckets = {}
    DhtBootstrapHandler._rebalance_hints = {}


def test_dht_geo_triangulation_downgrades_region_on_latency_violation():
    seed = "geo-triangulation-test-seed"
    _reset_geo_bootstrap(seed=seed, max_rtt_ms=50.0)

    try:
        peer_server, peer_port = _start_delayed_peer(seed=seed, delay_s=0.08)
    except Exception as exc:
        pytest.skip(f"unable to start delayed peer server: {exc}")

    try:
        dht_server = ThreadingHTTPServer(("127.0.0.1", 0), DhtBootstrapHandler)
    except OSError as exc:
        peer_server.stop(grace=0)
        pytest.skip(f"socket bind unavailable: {exc}")

    dht_thread = threading.Thread(target=dht_server.serve_forever, daemon=True)
    dht_thread.start()
    host, port = dht_server.server_address
    base = f"http://{host}:{port}"

    try:
        ack = _post_json(
            f"{base}/announce",
            {
                "peer_id": "peer-geo-delayed",
                "model_id": "openhydra-toy-345m",
                "host": "127.0.0.1",
                "port": int(peer_port),
                "region": "us-east",
                "operator_id": "op-geo",
                "load_pct": 1.0,
            },
        )
        assert ack["ok"] is True

        lookup = _get_json(f"{base}/lookup?model_id=openhydra-toy-345m")
        assert lookup["count"] == 1
        peer = lookup["peers"][0]
        assert peer["peer_id"] == "peer-geo-delayed"
        # Region is not trusted when crypto challenge RTT exceeds policy.
        assert peer["region"] is None
        assert peer["region_claimed"] == "us-east"
        assert peer["geo_verified"] is False
        assert peer["geo_challenge_reason"] == "latency_violation"
        assert float(peer["geo_penalty_score"]) >= 1.0
        assert float(peer["geo_challenge_rtt_ms"]) > 50.0
    finally:
        dht_server.shutdown()
        dht_server.server_close()
        dht_thread.join(timeout=2.0)
        peer_server.stop(grace=0)
