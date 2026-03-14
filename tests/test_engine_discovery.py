import json

import pytest

from coordinator.chain import ChainResult, StageTrace
from coordinator.engine import CoordinatorEngine, EngineConfig
from coordinator.path_finder import PeerEndpoint, PeerHealth


def test_engine_load_candidate_peers_from_dht(monkeypatch, tmp_path):
    engine = CoordinatorEngine(
        EngineConfig(
            peers_config_path=None,
            dht_url="http://127.0.0.1:8468",
            ledger_path=str(tmp_path / "credits.json"),
            health_store_path=str(tmp_path / "health.json"),
            barter_decay_per_day=0.0,
        )
    )

    monkeypatch.setattr(
        "coordinator.engine.load_peers_from_dht",
        lambda dht_url, model_id, timeout_s, preferred_region=None, limit=None, sloppy_factor=None, dsht_replicas=None: [
            PeerEndpoint(
                peer_id="p1",
                host="127.0.0.1",
                port=5001,
                seeding_enabled=True,
                seed_upload_limit_mbps=25.0,
                seed_target_upload_limit_mbps=30.0,
                seed_inference_active=True,
            ),
            PeerEndpoint(
                peer_id="p1",
                host="127.0.0.1",
                port=5001,
                seeding_enabled=True,
                seed_upload_limit_mbps=25.0,
                seed_target_upload_limit_mbps=30.0,
                seed_inference_active=True,
            ),
            PeerEndpoint(peer_id="p2", host="127.0.0.1", port=5002),
        ],
    )

    peers = engine._load_candidate_peers()
    assert [p.peer_id for p in peers] == ["p1", "p2"]
    assert peers[0].seeding_enabled is True
    assert peers[0].seed_upload_limit_mbps == 25.0
    assert peers[0].seed_target_upload_limit_mbps == 30.0
    assert peers[0].seed_inference_active is True


def test_engine_prefers_dht_duplicate_over_peer_config(monkeypatch, tmp_path):
    peer_config_path = tmp_path / "peers.json"
    peer_config_path.write_text(
        json.dumps(
            [
                {
                    "peer_id": "p1",
                    "host": "127.0.0.1",
                    "port": 5001,
                    "bandwidth_mbps": 100.0,
                    "seeding_enabled": False,
                    "seed_upload_limit_mbps": 0.0,
                }
            ]
        )
    )

    engine = CoordinatorEngine(
        EngineConfig(
            peers_config_path=str(peer_config_path),
            dht_url="http://127.0.0.1:8468",
            ledger_path=str(tmp_path / "credits.json"),
            health_store_path=str(tmp_path / "health.json"),
            barter_decay_per_day=0.0,
        )
    )

    monkeypatch.setattr(
        "coordinator.engine.load_peers_from_dht",
        lambda dht_url, model_id, timeout_s, preferred_region=None, limit=None, sloppy_factor=None, dsht_replicas=None: [
            PeerEndpoint(
                peer_id="p1",
                host="127.0.0.1",
                port=5001,
                bandwidth_mbps=250.0,
                seeding_enabled=True,
                seed_upload_limit_mbps=40.0,
                seed_target_upload_limit_mbps=50.0,
                seed_inference_active=True,
            )
        ],
    )

    peers = engine._load_candidate_peers()
    assert len(peers) == 1
    assert peers[0].peer_id == "p1"
    assert peers[0].bandwidth_mbps == 250.0
    assert peers[0].seeding_enabled is True
    assert peers[0].seed_upload_limit_mbps == 40.0
    assert peers[0].seed_target_upload_limit_mbps == 50.0
    assert peers[0].seed_inference_active is True


def test_engine_passes_dht_advanced_lookup_options(monkeypatch, tmp_path):
    engine = CoordinatorEngine(
        EngineConfig(
            peers_config_path=None,
            dht_url="http://127.0.0.1:8468",
            dht_lookup_limit=4,
            dht_lookup_sloppy_factor=2,
            dht_lookup_dsht_replicas=5,
            dht_preferred_region="us-east",
            ledger_path=str(tmp_path / "credits.json"),
            health_store_path=str(tmp_path / "health.json"),
            barter_decay_per_day=0.0,
        )
    )

    captured: list[dict[str, object]] = []

    def fake_load(dht_url, model_id, timeout_s, preferred_region=None, limit=None, sloppy_factor=None, dsht_replicas=None):
        captured.append(
            {
                "dht_url": dht_url,
                "model_id": model_id,
                "timeout_s": timeout_s,
                "preferred_region": preferred_region,
                "limit": limit,
                "sloppy_factor": sloppy_factor,
                "dsht_replicas": dsht_replicas,
            }
        )
        return [PeerEndpoint(peer_id=f"peer-{model_id}", host="127.0.0.1", port=5001, region="us-east")]

    monkeypatch.setattr("coordinator.engine.load_peers_from_dht", fake_load)

    peers = engine._load_candidate_peers()
    assert peers
    assert captured
    assert captured[0]["dht_url"] == "http://127.0.0.1:8468"
    assert captured[0]["preferred_region"] == "us-east"
    assert captured[0]["limit"] == 4
    assert captured[0]["sloppy_factor"] == 2
    assert captured[0]["dsht_replicas"] == 5


def test_engine_uses_cached_dht_peers_on_lookup_timeout(monkeypatch, tmp_path):
    engine = CoordinatorEngine(
        EngineConfig(
            peers_config_path=None,
            dht_url="http://127.0.0.1:8468",
            dht_lookup_cache_ttl_s=300.0,
            ledger_path=str(tmp_path / "credits.json"),
            health_store_path=str(tmp_path / "health.json"),
            barter_decay_per_day=0.0,
        )
    )

    calls = {"count": 0}

    def fake_load(dht_url, model_id, timeout_s, preferred_region=None, limit=None, sloppy_factor=None, dsht_replicas=None):
        if calls["count"] == 0:
            calls["count"] += 1
            return [PeerEndpoint(peer_id="cached-peer", host="127.0.0.1", port=5001)]
        raise TimeoutError("timed out")

    monkeypatch.setattr("coordinator.engine.load_peers_from_dht", fake_load)

    first = engine._load_candidate_peers()
    assert [peer.peer_id for peer in first] == ["cached-peer"]

    second = engine._load_candidate_peers()
    assert [peer.peer_id for peer in second] == ["cached-peer"]


def test_engine_uses_multi_dht_urls_when_configured(monkeypatch, tmp_path):
    engine = CoordinatorEngine(
        EngineConfig(
            peers_config_path=None,
            dht_urls=["http://127.0.0.1:8468", "http://127.0.0.2:8468"],
            ledger_path=str(tmp_path / "credits.json"),
            health_store_path=str(tmp_path / "health.json"),
            barter_decay_per_day=0.0,
        )
    )

    captured: list[dict[str, object]] = []

    def fake_load(*args, **kwargs):
        captured.append({"args": list(args), "kwargs": dict(kwargs)})
        return [PeerEndpoint(peer_id="peer-multi", host="127.0.0.1", port=5001)]

    monkeypatch.setattr("coordinator.engine.load_peers_from_dht", fake_load)

    peers = engine._load_candidate_peers(model_ids=[engine.config.default_model])
    assert [peer.peer_id for peer in peers] == ["peer-multi"]
    assert captured
    assert captured[0]["kwargs"]["dht_urls"] == ["http://127.0.0.1:8468", "http://127.0.0.2:8468"]


def test_engine_routes_dynamic_model_id_from_healthy_peers(monkeypatch, tmp_path):
    dynamic_model = "HuggingFaceTB/SmolLM2-135M-Instruct"
    engine = CoordinatorEngine(
        EngineConfig(
            peers_config_path="/tmp/unused.json",
            ledger_path=str(tmp_path / "credits.json"),
            health_store_path=str(tmp_path / "health.json"),
            allow_dynamic_model_ids=True,
            grounding_use_network=False,
            audit_rate=0.0,
            redundant_exec_rate=0.0,
            barter_decay_per_day=0.0,
        )
    )

    peer = PeerEndpoint(peer_id="dyn-peer", host="127.0.0.1", port=5001, model_id=dynamic_model)
    health = [PeerHealth(peer=peer, healthy=True, latency_ms=10.0, load_pct=0.0, daemon_mode="polite")]

    monkeypatch.setattr(engine, "_scan_network", lambda model_ids=None: (health, {dynamic_model: 1}))
    monkeypatch.setattr(engine, "_select_pipeline", lambda candidates, pipeline_width=None: [peer])
    monkeypatch.setattr(
        engine,
        "_run_chain",
        lambda prompt, candidates, pipeline, max_tokens, request_id=None, deadline=None, **_: ChainResult(
            request_id=request_id or "r1",
            text="dynamic ok",
            activation=[0.1],
            traces=[StageTrace(peer_id=peer.peer_id, latency_ms=1.0, stage_index=0)],
            latency_ms=5.0,
        ),
    )

    payload = engine.infer(prompt="hello", model_id=dynamic_model, grounding=False, allow_degradation=True)
    assert payload["model"]["requested"] == dynamic_model
    assert payload["model"]["served"] == dynamic_model
    assert payload["model"]["degraded"] is False


def test_engine_rejects_dynamic_model_id_when_disabled(monkeypatch, tmp_path):
    dynamic_model = "EleutherAI/gpt-neo-125m"
    engine = CoordinatorEngine(
        EngineConfig(
            peers_config_path="/tmp/unused.json",
            ledger_path=str(tmp_path / "credits.json"),
            health_store_path=str(tmp_path / "health.json"),
            allow_dynamic_model_ids=False,
            grounding_use_network=False,
            audit_rate=0.0,
            redundant_exec_rate=0.0,
            barter_decay_per_day=0.0,
        )
    )

    monkeypatch.setattr(engine, "_scan_network", lambda model_ids=None: ([], {}))

    with pytest.raises(RuntimeError, match="unknown_model:"):
        engine.infer(prompt="hello", model_id=dynamic_model, grounding=False, allow_degradation=True)


def test_engine_resolves_catalog_hf_model_id(tmp_path):
    catalog_path = tmp_path / "models.catalog.json"
    catalog_path.write_text(
        json.dumps(
            [
                {
                    "model_id": "openhydra-qwen3.5-0.8b",
                    "required_peers": 1,
                    "hf_model_id": "Qwen/Qwen3.5-0.8B",
                }
            ],
            indent=2,
        )
    )
    engine = CoordinatorEngine(
        EngineConfig(
            peers_config_path="/tmp/unused.json",
            model_catalog_path=str(catalog_path),
            default_model="openhydra-qwen3.5-0.8b",
            ledger_path=str(tmp_path / "credits.json"),
            health_store_path=str(tmp_path / "health.json"),
            barter_decay_per_day=0.0,
        )
    )
    assert engine._resolve_runtime_model_id("openhydra-qwen3.5-0.8b") == "Qwen/Qwen3.5-0.8B"
    assert engine._resolve_runtime_model_id("EleutherAI/gpt-neo-125m") == "EleutherAI/gpt-neo-125m"


def test_engine_prefers_pipeline_runtime_model_id(tmp_path):
    engine = CoordinatorEngine(
        EngineConfig(
            peers_config_path="/tmp/unused.json",
            ledger_path=str(tmp_path / "credits.json"),
            health_store_path=str(tmp_path / "health.json"),
            barter_decay_per_day=0.0,
            pytorch_generation_model_id="Qwen/Qwen3.5-0.8B",
        )
    )
    pipeline = [
        PeerEndpoint(
            peer_id="peer-a",
            host="127.0.0.1",
            port=5001,
            model_id="openhydra-qwen3.5-0.8b",
            runtime_backend="pytorch_cpu",
            runtime_model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        )
    ]
    resolved = engine._resolve_pipeline_runtime_model_id(pipeline, "openhydra-qwen3.5-0.8b")
    assert resolved == "HuggingFaceTB/SmolLM2-135M-Instruct"


def test_list_models_includes_catalog_hf_model_id(tmp_path):
    catalog_path = tmp_path / "models.catalog.json"
    catalog_path.write_text(
        json.dumps(
            [
                {
                    "model_id": "openhydra-qwen3.5-0.8b",
                    "required_peers": 1,
                    "hf_model_id": "Qwen/Qwen3.5-0.8B",
                }
            ],
            indent=2,
        )
    )
    engine = CoordinatorEngine(
        EngineConfig(
            peers_config_path="/tmp/unused.json",
            model_catalog_path=str(catalog_path),
            default_model="openhydra-qwen3.5-0.8b",
            ledger_path=str(tmp_path / "credits.json"),
            health_store_path=str(tmp_path / "health.json"),
            barter_decay_per_day=0.0,
        )
    )
    payload = engine.list_models()
    assert payload["object"] == "list"
    assert payload["data"][0]["id"] == "openhydra-qwen3.5-0.8b"
    assert payload["data"][0]["hf_model_id"] == "Qwen/Qwen3.5-0.8B"


def test_discover_finds_peer_by_hf_id_when_catalog_alias_requested(tmp_path, monkeypatch):
    """
    Peers that announce model_id='Qwen/Qwen3.5-0.8B' (HF Hub ID) must be
    discovered when the user requests the catalog alias 'openhydra-qwen3.5-0.8b'.

    This covers the production pattern: peer started with --model-id Qwen/Qwen3.5-0.8B
    while the API request uses the stable catalog alias.
    """
    catalog_path = tmp_path / "models.catalog.json"
    catalog_path.write_text(
        json.dumps(
            [
                {
                    "model_id": "openhydra-qwen3.5-0.8b",
                    "required_peers": 1,
                    "hf_model_id": "Qwen/Qwen3.5-0.8B",
                }
            ],
            indent=2,
        )
    )
    engine = CoordinatorEngine(
        EngineConfig(
            peers_config_path="/tmp/unused.json",
            model_catalog_path=str(catalog_path),
            default_model="openhydra-qwen3.5-0.8b",
            ledger_path=str(tmp_path / "credits.json"),
            health_store_path=str(tmp_path / "health.json"),
            barter_decay_per_day=0.0,
            audit_rate=0.0,
            redundant_exec_rate=0.0,
            grounding_use_network=False,
        )
    )

    # Peer announces itself using the HF Hub ID (not the catalog alias)
    hf_peer = PeerEndpoint(
        peer_id="qwen-peer",
        host="127.0.0.1",
        port=50051,
        model_id="Qwen/Qwen3.5-0.8B",  # HF ID, not the alias
        runtime_backend="mlx",
        runtime_model_id="Qwen/Qwen3.5-0.8B",
    )
    health = [PeerHealth(peer=hf_peer, healthy=True, latency_ms=5.0, load_pct=0.0, daemon_mode="polite")]

    # _scan_network returns the peer under its HF model ID key
    monkeypatch.setattr(
        engine, "_scan_network",
        lambda model_ids=None: (health, {"Qwen/Qwen3.5-0.8B": 1}),
    )
    monkeypatch.setattr(engine, "_select_pipeline", lambda candidates, pipeline_width=None: [hf_peer])
    monkeypatch.setattr(
        engine,
        "_run_chain",
        lambda prompt, candidates, pipeline, max_tokens, request_id=None, deadline=None, **_: ChainResult(
            request_id=request_id or "r-alias",
            text="hello from qwen",
            activation=[0.1],
            traces=[StageTrace(peer_id=hf_peer.peer_id, latency_ms=2.0, stage_index=0)],
            latency_ms=10.0,
        ),
    )

    # Requesting by catalog alias must succeed and route to the HF-ID-announcing peer
    payload = engine.infer(
        prompt="hello",
        model_id="openhydra-qwen3.5-0.8b",
        grounding=False,
        allow_degradation=True,
    )
    assert payload["model"]["requested"] == "openhydra-qwen3.5-0.8b"
    assert payload["response"] == "hello from qwen"
