from coordinator.engine import CoordinatorEngine, EngineConfig
from coordinator.path_finder import PeerEndpoint, PeerHealth


def test_network_status_includes_seeding_metrics(tmp_path, monkeypatch):
    engine = CoordinatorEngine(
        EngineConfig(
            peers_config_path="/tmp/unused.json",
            ledger_path=str(tmp_path / "credits.json"),
            health_store_path=str(tmp_path / "health.json"),
            barter_decay_per_day=0.0,
            required_replicas=1,
        )
    )

    peer_a = PeerEndpoint(
        peer_id="a",
        host="127.0.0.1",
        port=1,
        model_id=engine.config.default_model,
        seeding_enabled=True,
        seed_upload_limit_mbps=10.0,
        seed_target_upload_limit_mbps=10.0,
        seed_inference_active=True,
    )
    peer_b = PeerEndpoint(
        peer_id="b",
        host="127.0.0.1",
        port=2,
        model_id=engine.config.default_model,
        seeding_enabled=True,
        seed_upload_limit_mbps=20.0,
        seed_target_upload_limit_mbps=100.0,
        seed_inference_active=False,
    )

    health = [
        PeerHealth(peer=peer_a, healthy=True, latency_ms=10.0, load_pct=0.0, daemon_mode="polite"),
        PeerHealth(peer=peer_b, healthy=True, latency_ms=11.0, load_pct=0.0, daemon_mode="polite"),
    ]

    monkeypatch.setattr(engine, "_scan_network", lambda model_ids=None: (health, {engine.config.default_model: 2}))

    status = engine.network_status()
    metrics = status["seeding"][engine.config.default_model]

    assert metrics["seeding_enabled_peers"] == 2
    assert metrics["seed_inference_active_peers"] == 1
    assert metrics["total_seed_upload_limit_mbps"] == 30.0
    assert metrics["avg_seed_upload_limit_mbps"] == 15.0
