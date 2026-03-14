import json

import pytest

from coordinator.chain import ChainResult, StageTrace
from coordinator.engine import CoordinatorEngine, EngineConfig
from coordinator.path_finder import PeerEndpoint, PeerHealth


def _engine_with_catalog(tmp_path):
    catalog = [
        {"model_id": "openhydra-70b", "required_peers": 8},
        {"model_id": "openhydra-8b", "required_peers": 3},
        {"model_id": "openhydra-toy-345m", "required_peers": 1},
    ]
    catalog_path = tmp_path / "models.catalog.json"
    catalog_path.write_text(json.dumps(catalog))

    return CoordinatorEngine(
        EngineConfig(
            peers_config_path="/tmp/unused.json",
            model_catalog_path=str(catalog_path),
            ledger_path=str(tmp_path / "credits.json"),
            health_store_path=str(tmp_path / "health.json"),
            barter_decay_per_day=0.0,
            audit_rate=0.0,
        )
    )


def test_engine_degrades_to_available_model(tmp_path, monkeypatch):
    engine = _engine_with_catalog(tmp_path)

    peer = PeerEndpoint(peer_id="peer-a", host="127.0.0.1", port=1, model_id="openhydra-toy-345m")
    health = [PeerHealth(peer=peer, healthy=True, latency_ms=10.0, load_pct=0.0, daemon_mode="polite")]

    monkeypatch.setattr(
        engine,
        "_scan_network",
        lambda model_ids=None: (health, {"openhydra-toy-345m": 1}),
    )
    monkeypatch.setattr(engine, "_select_pipeline", lambda candidates, pipeline_width=None: [peer])
    monkeypatch.setattr(
        engine,
        "_run_chain",
        lambda prompt, candidates, pipeline, max_tokens, request_id=None, deadline=None, **_: ChainResult(
            request_id=request_id or "r1",
            text="fallback result",
            activation=[0.1],
            traces=[StageTrace(peer_id="peer-a", latency_ms=1.0, stage_index=0)],
            latency_ms=5.0,
        ),
    )

    payload = engine.infer(
        prompt="hello",
        model_id="openhydra-70b",
        allow_degradation=True,
        grounding=False,
    )

    assert payload["model"]["requested"] == "openhydra-70b"
    assert payload["model"]["served"] == "openhydra-toy-345m"
    assert payload["model"]["degraded"] is True
    assert payload["model"]["available"] is True


def test_engine_fails_when_degradation_disabled(tmp_path, monkeypatch):
    engine = _engine_with_catalog(tmp_path)

    monkeypatch.setattr(
        engine,
        "_scan_network",
        lambda model_ids=None: ([], {}),
    )

    with pytest.raises(RuntimeError, match="no_viable_model:insufficient_peers"):
        engine.infer(
            prompt="hello",
            model_id="openhydra-70b",
            allow_degradation=False,
            grounding=False,
        )


def test_engine_degrades_when_verification_qos_breached(tmp_path, monkeypatch):
    catalog = [
        {"model_id": "openhydra-8b", "required_peers": 1},
        {"model_id": "openhydra-toy-345m", "required_peers": 1},
    ]
    catalog_path = tmp_path / "models.catalog.qos.json"
    catalog_path.write_text(json.dumps(catalog))

    engine = CoordinatorEngine(
        EngineConfig(
            peers_config_path="/tmp/unused.json",
            model_catalog_path=str(catalog_path),
            ledger_path=str(tmp_path / "credits.json"),
            health_store_path=str(tmp_path / "health.json"),
            barter_decay_per_day=0.0,
            audit_rate=0.0,
            verification_qos_min_events=2,
            verification_qos_min_success_rate=0.8,
        )
    )

    peer_big = PeerEndpoint(peer_id="peer-big", host="127.0.0.1", port=1, model_id="openhydra-8b")
    peer_fallback = PeerEndpoint(peer_id="peer-fallback", host="127.0.0.1", port=2, model_id="openhydra-toy-345m")
    health = [
        PeerHealth(peer=peer_big, healthy=True, latency_ms=10.0, load_pct=0.0, daemon_mode="polite"),
        PeerHealth(peer=peer_fallback, healthy=True, latency_ms=11.0, load_pct=0.0, daemon_mode="polite"),
    ]

    engine.health.record_verification("peer-big", success=False)
    engine.health.record_verification("peer-big", success=False)

    monkeypatch.setattr(
        engine,
        "_scan_network",
        lambda model_ids=None: (health, {"openhydra-8b": 1, "openhydra-toy-345m": 1}),
    )
    monkeypatch.setattr(engine, "_select_pipeline", lambda candidates, pipeline_width=None: [candidates[0]])
    monkeypatch.setattr(
        engine,
        "_run_chain",
        lambda prompt, candidates, pipeline, max_tokens, request_id=None, deadline=None, **_: ChainResult(
            request_id=request_id or "r1",
            text=f"served:{pipeline[0].peer_id}",
            activation=[0.1],
            traces=[StageTrace(peer_id=pipeline[0].peer_id, latency_ms=1.0, stage_index=0)],
            latency_ms=5.0,
        ),
    )

    payload = engine.infer(
        prompt="hello",
        model_id="openhydra-8b",
        allow_degradation=True,
        grounding=False,
    )

    assert payload["model"]["served"] == "openhydra-toy-345m"
    assert payload["model"]["degraded"] is True
    assert payload["model"]["available"] is True
    assert payload["model"]["reason"] == "verification_qos"
    assert "verification_qos_floor_breached" in payload["model"]["detail"]
    assert payload["model"]["verification_qos"]["requested_model_blocked"] is True


def test_engine_raises_when_qos_breached_and_degradation_disabled(tmp_path, monkeypatch):
    catalog = [
        {"model_id": "openhydra-8b", "required_peers": 1},
        {"model_id": "openhydra-toy-345m", "required_peers": 1},
    ]
    catalog_path = tmp_path / "models.catalog.qos.fail.json"
    catalog_path.write_text(json.dumps(catalog))

    engine = CoordinatorEngine(
        EngineConfig(
            peers_config_path="/tmp/unused.json",
            model_catalog_path=str(catalog_path),
            ledger_path=str(tmp_path / "credits.json"),
            health_store_path=str(tmp_path / "health.json"),
            barter_decay_per_day=0.0,
            audit_rate=0.0,
            verification_qos_min_events=1,
            verification_qos_min_success_rate=0.95,
        )
    )

    peer_big = PeerEndpoint(peer_id="peer-big", host="127.0.0.1", port=1, model_id="openhydra-8b")
    health = [PeerHealth(peer=peer_big, healthy=True, latency_ms=10.0, load_pct=0.0, daemon_mode="polite")]
    engine.health.record_verification("peer-big", success=False)

    monkeypatch.setattr(
        engine,
        "_scan_network",
        lambda model_ids=None: (health, {"openhydra-8b": 1}),
    )

    with pytest.raises(RuntimeError, match="no_viable_model:insufficient_peers"):
        engine.infer(
            prompt="hello",
            model_id="openhydra-8b",
            allow_degradation=False,
            grounding=False,
        )
