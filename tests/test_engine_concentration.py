from coordinator.engine import CoordinatorEngine, EngineConfig
from coordinator.path_finder import PeerEndpoint, PeerHealth


def test_network_status_emits_operator_concentration_alert(tmp_path, monkeypatch):
    engine = CoordinatorEngine(
        EngineConfig(
            peers_config_path="/tmp/unused.json",
            ledger_path=str(tmp_path / "credits.json"),
            health_store_path=str(tmp_path / "health.json"),
            barter_decay_per_day=0.0,
            operator_cap_fraction=0.33,
            required_replicas=1,
        )
    )

    health = [
        PeerHealth(peer=PeerEndpoint(peer_id="a1", host="127.0.0.1", port=1, operator_id="op-a"), healthy=True, latency_ms=10.0, load_pct=10.0, daemon_mode="polite"),
        PeerHealth(peer=PeerEndpoint(peer_id="a2", host="127.0.0.1", port=2, operator_id="op-a"), healthy=True, latency_ms=11.0, load_pct=10.0, daemon_mode="polite"),
        PeerHealth(peer=PeerEndpoint(peer_id="a3", host="127.0.0.1", port=3, operator_id="op-a"), healthy=True, latency_ms=12.0, load_pct=10.0, daemon_mode="polite"),
    ]

    monkeypatch.setattr(engine, "_scan_network", lambda model_ids=None: (health, {engine.config.default_model: 3}))

    status = engine.network_status()

    assert "operator_concentration" in status["alerts"]
    model_conc = status["concentration"][engine.config.default_model]
    assert model_conc["max_operator"] == "op-a"
    assert model_conc["max_share"] == 1.0
