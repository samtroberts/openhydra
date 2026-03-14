from coordinator.engine import CoordinatorEngine, EngineConfig
from coordinator.path_finder import PeerEndpoint, PeerHealth


def test_network_status_handles_no_peers(tmp_path, monkeypatch):
    engine = CoordinatorEngine(
        EngineConfig(
            peers_config_path="/tmp/unused.json",
            ledger_path=str(tmp_path / "credits.json"),
            health_store_path=str(tmp_path / "health.json"),
            barter_decay_per_day=0.0,
            required_replicas=3,
        )
    )

    monkeypatch.setattr(
        engine,
        "_scan_network",
        lambda model_ids=None: (_ for _ in ()).throw(RuntimeError("No healthy peers discovered")),
    )

    status = engine.network_status()
    assert status["healthy_peers"] == 0
    assert status["replication"][0]["under_replicated"] is True
    assert "concentration" in status
    assert "runtime_profiles" in status
    assert status["runtime_profiles"][engine.config.default_model]["total_peers"] == 0
    assert "verification_feedback" in status
    assert status["verification_feedback"][engine.config.default_model]["verification_events"] == 0
    assert status["verification_alerts"] == {}
    assert "no_healthy_peers" in status["alerts"]


def test_network_status_aggregates_runtime_profiles(tmp_path, monkeypatch):
    engine = CoordinatorEngine(
        EngineConfig(
            peers_config_path="/tmp/unused.json",
            ledger_path=str(tmp_path / "credits.json"),
            health_store_path=str(tmp_path / "health.json"),
            barter_decay_per_day=0.0,
            required_replicas=1,
        )
    )
    health = [
        PeerHealth(
            peer=PeerEndpoint(
                peer_id="peer-a",
                host="127.0.0.1",
                port=5001,
                model_id=engine.config.default_model,
                runtime_backend="toy_gpu_sim",
                runtime_target="cuda",
                quantization_mode="int8",
                quantization_bits=8,
                runtime_gpu_available=True,
                runtime_estimated_tokens_per_sec=240.0,
                runtime_estimated_memory_mb=900,
                expert_tags=("vision", "code"),
                expert_layer_indices=(1, 2),
                expert_router=True,
            ),
            healthy=True,
            latency_ms=9.0,
            load_pct=12.0,
            daemon_mode="polite",
        ),
        PeerHealth(
            peer=PeerEndpoint(
                peer_id="peer-b",
                host="127.0.0.1",
                port=5002,
                model_id=engine.config.default_model,
                runtime_backend="toy_cpu",
                runtime_target="cpu",
                quantization_mode="fp32",
                quantization_bits=0,
                runtime_gpu_available=False,
                runtime_estimated_tokens_per_sec=90.0,
                runtime_estimated_memory_mb=1400,
                expert_tags=("code",),
                expert_layer_indices=(2, 4),
                expert_router=False,
            ),
            healthy=True,
            latency_ms=11.0,
            load_pct=14.0,
            daemon_mode="polite",
        ),
    ]

    monkeypatch.setattr(engine, "_scan_network", lambda model_ids=None: (health, {engine.config.default_model: 2}))
    status = engine.network_status()
    runtime = status["runtime_profiles"][engine.config.default_model]
    assert runtime["total_peers"] == 2
    assert runtime["backends"] == {"toy_gpu_sim": 1, "toy_cpu": 1}
    assert runtime["quantization_modes"] == {"int8": 1, "fp32": 1}
    assert runtime["gpu_available_peers"] == 1
    assert runtime["avg_estimated_tokens_per_sec"] == 165.0
    assert runtime["avg_estimated_memory_mb"] == 1150.0
    experts = status["expert_profiles"][engine.config.default_model]
    assert experts["total_peers"] == 2
    assert experts["expert_peers"] == 2
    assert experts["router_capable_peers"] == 1
    assert experts["tags"] == {"vision": 1, "code": 2}
    assert experts["layer_coverage"] == {"1": 1, "2": 2, "4": 1}
