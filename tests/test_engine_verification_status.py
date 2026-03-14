from coordinator.engine import CoordinatorEngine, EngineConfig
from coordinator.path_finder import PeerEndpoint, PeerHealth


def test_network_status_includes_verification_feedback_by_model(tmp_path, monkeypatch):
    engine = CoordinatorEngine(
        EngineConfig(
            peers_config_path="/tmp/unused.json",
            ledger_path=str(tmp_path / "credits.json"),
            health_store_path=str(tmp_path / "health.json"),
            barter_decay_per_day=0.0,
            required_replicas=1,
        )
    )

    peer_a = PeerEndpoint(peer_id="a", host="127.0.0.1", port=1, model_id=engine.config.default_model)
    peer_b = PeerEndpoint(peer_id="b", host="127.0.0.1", port=2, model_id=engine.config.default_model)
    health = [
        PeerHealth(peer=peer_a, healthy=True, latency_ms=10.0, load_pct=0.0, daemon_mode="polite"),
        PeerHealth(peer=peer_b, healthy=True, latency_ms=11.0, load_pct=0.0, daemon_mode="polite"),
    ]

    engine.health.record_verification("a", success=True)
    engine.health.record_verification("a", success=False)
    engine.health.record_verification("b", success=True)

    monkeypatch.setattr(engine, "_scan_network", lambda model_ids=None: (health, {engine.config.default_model: 2}))

    status = engine.network_status()
    metrics = status["verification_feedback"][engine.config.default_model]

    assert metrics["verified_peers"] == 2
    assert metrics["peers_with_failed_verifications"] == 1
    assert metrics["total_verifications_ok"] == 2
    assert metrics["total_verifications_failed"] == 1
    assert metrics["verification_events"] == 3
    assert metrics["verification_success_rate"] == 0.666667


def test_network_status_emits_verification_degraded_alert(tmp_path, monkeypatch):
    engine = CoordinatorEngine(
        EngineConfig(
            peers_config_path="/tmp/unused.json",
            ledger_path=str(tmp_path / "credits.json"),
            health_store_path=str(tmp_path / "health.json"),
            barter_decay_per_day=0.0,
            required_replicas=1,
            verification_alert_min_events=3,
            verification_alert_min_success_rate=0.8,
        )
    )

    peer_a = PeerEndpoint(peer_id="a", host="127.0.0.1", port=1, model_id=engine.config.default_model)
    peer_b = PeerEndpoint(peer_id="b", host="127.0.0.1", port=2, model_id=engine.config.default_model)
    health = [
        PeerHealth(peer=peer_a, healthy=True, latency_ms=10.0, load_pct=0.0, daemon_mode="polite"),
        PeerHealth(peer=peer_b, healthy=True, latency_ms=11.0, load_pct=0.0, daemon_mode="polite"),
    ]

    engine.health.record_verification("a", success=False)
    engine.health.record_verification("a", success=False)
    engine.health.record_verification("b", success=True)

    monkeypatch.setattr(engine, "_scan_network", lambda model_ids=None: (health, {engine.config.default_model: 2}))

    status = engine.network_status()

    assert "verification_degraded" in status["alerts"]
    metrics = status["verification_alerts"][engine.config.default_model]
    assert metrics["verification_events"] == 3
    assert metrics["verification_success_rate"] == 0.333333
