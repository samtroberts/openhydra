import json
from http.server import ThreadingHTTPServer
import threading
from urllib import request

from coordinator.chain import ChainResult, StageTrace
from coordinator.ledger_bridge import OpenHydraLedgerBridge
from coordinator.engine import CoordinatorEngine, EngineConfig
from coordinator.path_finder import PeerEndpoint, PeerHealth
from dht.bootstrap import DhtBootstrapHandler
from dht.node import InMemoryDhtNode
import pytest


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


def _wire_single_peer(monkeypatch, engine: CoordinatorEngine, peer_id: str = "peer-a") -> PeerEndpoint:
    peer = PeerEndpoint(peer_id=peer_id, host="127.0.0.1", port=1, model_id=engine.config.default_model)
    health = [
        PeerHealth(peer=peer, healthy=True, latency_ms=5.0, load_pct=0.0, daemon_mode="polite"),
    ]
    monkeypatch.setattr(
        engine,
        "_discover_for_model",
        lambda requested_model, allow_degradation: (
            health,
            [peer],
            type(
                "Decision",
                (),
                {
                    "requested_model": requested_model,
                    "served_model": engine.config.default_model,
                    "degraded": False,
                    "available": True,
                    "reason": "ok",
                    "detail": "ok",
                },
            )(),
            {engine.config.default_model: 1},
        ),
    )
    monkeypatch.setattr(engine, "_select_pipeline", lambda candidates, pipeline_width=None: [candidates[0]])
    return peer


def test_hydra_ledger_bridge_enforces_bme_supply_cap():
    bridge = OpenHydraLedgerBridge(mock_mode=True, supply_cap=10.0)
    bridge.seed_account("payer", balance=9.0, staked_balance=0.0)
    bridge.mint_provider_rewards("peer-a", 1.0)
    with pytest.raises(RuntimeError, match="hydra_bridge_supply_cap_exceeded"):
        bridge.mint_provider_rewards("peer-b", 0.1)

    status = bridge.summary()
    assert status["supply_cap"] == 10.0
    assert status["total_supply"] == 10.0


def test_slash_stake_concurrent_updates_totals_without_double_count():
    bridge = OpenHydraLedgerBridge(mock_mode=True)
    bridge.seed_account("peer-race", balance=0.0, staked_balance=4.0)

    barrier = threading.Barrier(3)
    results: list[dict[str, float | str]] = []
    result_lock = threading.Lock()

    def _worker(amount: float) -> None:
        barrier.wait(timeout=2.0)
        payload = bridge.slash_stake("peer-race", amount)
        with result_lock:
            results.append(payload)

    threads = [
        threading.Thread(target=_worker, args=(1.5,), daemon=True),
        threading.Thread(target=_worker, args=(1.0,), daemon=True),
    ]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=2.0)
    for thread in threads:
        thread.join(timeout=2.0)

    assert len(results) == 2
    total_slashed = sum(float(item["slashed"]) for item in results)
    assert total_slashed == pytest.approx(2.5, abs=1e-9)

    summary = bridge.summary()
    assert float(summary["total_burned"]) == pytest.approx(2.5, abs=1e-9)
    assert float(summary["total_slashed"]) == pytest.approx(2.5, abs=1e-9)
    assert bridge.verify_staked_balance("peer-race") == pytest.approx(1.5, abs=1e-9)


def test_dht_admission_allows_unstaked_peer():
    DhtBootstrapHandler.dht = InMemoryDhtNode(ttl_seconds=60)
    DhtBootstrapHandler.default_ttl_seconds = 60
    DhtBootstrapHandler.default_geo_challenge_enabled = False
    DhtBootstrapHandler.default_expert_min_reputation_score = 60.0
    DhtBootstrapHandler.default_expert_min_staked_balance = 0.01
    DhtBootstrapHandler.default_expert_require_stake = True
    DhtBootstrapHandler._lookup_buckets = {}
    DhtBootstrapHandler._rebalance_hints = {}

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
                "peer_id": "unstaked-peer",
                "model_id": "openhydra-toy-345m",
                "host": "127.0.0.1",
                "port": 50051,
            },
        )
        assert ack["ok"] is True

        lookup = _get_json(f"{base}/lookup?model_id=openhydra-toy-345m")
        assert lookup["count"] == 1
        assert lookup["peers"][0]["peer_id"] == "unstaked-peer"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_staked_peer_priority_and_bme_settlement(tmp_path, monkeypatch):
    engine = CoordinatorEngine(
        EngineConfig(
            peers_config_path="/tmp/unused.json",
            ledger_path=str(tmp_path / "credits.json"),
            hydra_token_ledger_path=str(tmp_path / "hydra_tokens.json"),
            health_store_path=str(tmp_path / "health.json"),
            hydra_reward_per_1k_tokens=0.0,
            barter_decay_per_day=0.0,
            tier=2,
            required_replicas=1,
        )
    )

    peer_a = PeerEndpoint(peer_id="peer-a", host="127.0.0.1", port=1, model_id=engine.config.default_model, bandwidth_mbps=100)
    peer_b = PeerEndpoint(peer_id="peer-b", host="127.0.0.1", port=2, model_id=engine.config.default_model, bandwidth_mbps=100)
    health = [
        PeerHealth(peer=peer_a, healthy=True, latency_ms=12.0, load_pct=20.0, daemon_mode="polite"),
        PeerHealth(peer=peer_b, healthy=True, latency_ms=12.0, load_pct=20.0, daemon_mode="polite"),
    ]
    monkeypatch.setattr(engine, "_scan_network", lambda model_ids=None: (health, {engine.config.default_model: 2}))

    # Optional stake boosts ranking priority without gating admission.
    engine.hydra.mint_for_inference("peer-b", tokens_served=1000, reward_per_1k_tokens=1.0)
    engine.hydra.stake("peer-b", 0.5)

    _, candidates, _, _ = engine._discover_for_model(
        requested_model=engine.config.default_model,
        allow_degradation=True,
    )
    assert [peer.peer_id for peer in candidates] == ["peer-b", "peer-a"]

    _wire_single_peer(monkeypatch, engine, peer_id="peer-a")

    call_count = {"value": 0}

    def fake_run_chain(prompt, candidates, pipeline, max_tokens, request_id=None, initial_activation=None, **kwargs):
        call_count["value"] += 1
        return ChainResult(
            request_id=request_id or "rid-1",
            text=f"step-{call_count['value']}",
            activation=[float(call_count["value"])],
            traces=[StageTrace(peer_id="peer-a", latency_ms=1.0, stage_index=0)],
            latency_ms=5.0,
        )

    monkeypatch.setattr(engine, "_run_chain", fake_run_chain)
    monkeypatch.setattr(
        "coordinator.engine.ModelShard.decode_tokens",
        lambda activation, max_tokens, tokenizer_model_id=None: [f"tok{int(activation[0])}"],
    )

    # Seed user balances: HYDRA state-channel ledger and bridge settlement ledger.
    engine.hydra.mint_for_inference("payer", tokens_served=2000, reward_per_1k_tokens=1.0)
    engine.ledger_bridge.seed_account("payer", balance=5.0, staked_balance=0.0)

    engine.hydra_open_channel(channel_id="ch-bme", payer="payer", payee="peer-a", deposit=1.8)

    stream_payload = engine.infer_stream(prompt="hello", max_tokens=2, grounding=False)
    chunks = list(stream_payload["stream"])
    assert chunks == ["Tok1", " tok2", "."]

    engine.hydra_charge_channel("ch-bme", 1.5, provider_peer_id="peer-a")
    close_payload = engine.hydra_close_channel("ch-bme")

    settlement = close_payload["hydra_bridge_settlement"]
    assert settlement["errors"] == []
    assert settlement["burn_receipt"]["burned"] == 1.5
    assert settlement["mint_receipts"][0]["payee_pubkey"] == "peer-a"
    assert settlement["mint_receipts"][0]["minted"] == 1.5

    payer_bridge = engine.ledger_bridge.account_snapshot("payer")
    payee_bridge = engine.ledger_bridge.account_snapshot("peer-a")
    assert payer_bridge["balance"] == 3.5
    assert payee_bridge["balance"] == 1.5


def test_hydra_governance_params_and_mock_vote(tmp_path):
    engine = CoordinatorEngine(
        EngineConfig(
            peers_config_path="/tmp/unused.json",
            ledger_path=str(tmp_path / "credits.json"),
            hydra_token_ledger_path=str(tmp_path / "hydra_tokens.json"),
            health_store_path=str(tmp_path / "health.json"),
            barter_decay_per_day=0.0,
        )
    )

    params = engine.hydra_governance_params()["hydra_governance"]["params"]
    assert params["supply_cap"] == 69_000_000.0
    assert params["daily_mint_rate"] > 0.0
    assert params["min_slash_penalty"] >= 0.0

    vote = engine.hydra_governance_vote(
        pubkey="voter-a",
        proposal_id="prop-1",
        vote="yes",
    )["hydra_governance_vote"]
    assert vote["accepted"] is True
    assert vote["proposal_id"] == "prop-1"
    assert vote["vote"] == "yes"
