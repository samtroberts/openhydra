"""Tests for elastic output token cap (The Great Pruning).

The engine enforces a 2048-token floor (always) and an 8192-token ceiling
that is only available when the model's effective redundancy >= 3.0.
Requests exceeding the elastic ceiling must raise ValueError (not silently clamp).
"""

import pytest

from coordinator.chain import ChainResult, StageTrace
from coordinator.engine import CoordinatorEngine, EngineConfig
from coordinator.path_finder import PeerEndpoint, PeerHealth


def _make_engine(tmp_path):
    return CoordinatorEngine(
        EngineConfig(
            peers_config_path="/tmp/unused.json",
            ledger_path=str(tmp_path / "credits.json"),
            health_store_path=str(tmp_path / "health.json"),
            audit_rate=0.0,
            barter_decay_per_day=0.0,
        )
    )


def _wire_peer_with_counts(monkeypatch, engine, model_id, peer_count, required_peers=1):
    """Wire a single peer but report `peer_count` available peers in counts."""
    peer = PeerEndpoint(peer_id="peer-a", host="127.0.0.1", port=1)
    health = [PeerHealth(peer=peer, healthy=True, latency_ms=10.0, load_pct=0.0, daemon_mode="polite")]

    monkeypatch.setattr(
        engine,
        "_discover_for_model",
        lambda requested_model, allow_degradation: (
            health,
            [peer],
            type("Decision", (), {
                "requested_model": requested_model,
                "served_model": model_id,
                "degraded": False,
                "available": True,
                "reason": "ok",
                "detail": "ok",
            })(),
            {model_id: peer_count},
        ),
    )
    monkeypatch.setattr(engine, "_select_pipeline", lambda candidates, pipeline_width=None: [peer])
    monkeypatch.setattr(engine, "_required_replicas", lambda mid: required_peers)

    def fake_run_chain(prompt, candidates, pipeline, max_tokens, request_id=None, deadline=None, **kwargs):
        return ChainResult(
            request_id=request_id or "r1",
            text="ok",
            activation=[0.1],
            traces=[StageTrace(peer_id="peer-a", latency_ms=1.0, stage_index=0)],
            latency_ms=5.0,
        )

    monkeypatch.setattr(engine, "_run_chain", fake_run_chain)


def test_elastic_cap_rejects_over_floor_when_low_redundancy(tmp_path, monkeypatch):
    """max_tokens > 2048 raises ValueError when redundancy < 3.0."""
    engine = _make_engine(tmp_path)
    # model requires 1 peer, only 2 available → redundancy = 2.0 < 3.0
    _wire_peer_with_counts(monkeypatch, engine, "openhydra-qwen3.5-0.8b", peer_count=2)

    with pytest.raises(ValueError, match="Maximum allowed output is 2048 tokens"):
        engine.infer(prompt="hello", max_tokens=2049, grounding=False)


def test_elastic_cap_allows_ceiling_when_high_redundancy(tmp_path, monkeypatch):
    """max_tokens up to 8192 succeeds when redundancy >= 3.0."""
    engine = _make_engine(tmp_path)
    # model requires 1 peer, 3 available → redundancy = 3.0 → ceiling = 8192
    _wire_peer_with_counts(monkeypatch, engine, "openhydra-qwen3.5-0.8b", peer_count=3)

    # Should NOT raise — 4096 is under the 8192 ceiling
    result = engine.infer(prompt="hello", max_tokens=4096, grounding=False)
    assert result["response"] == "ok"


def test_elastic_cap_rejects_over_ceiling_even_with_high_redundancy(tmp_path, monkeypatch):
    """max_tokens > 8192 raises ValueError even with high redundancy."""
    engine = _make_engine(tmp_path)
    # model requires 1 peer, 10 available → redundancy = 10.0 → ceiling = 8192
    _wire_peer_with_counts(monkeypatch, engine, "openhydra-qwen3.5-0.8b", peer_count=10)

    with pytest.raises(ValueError, match="Maximum allowed output is 8192 tokens"):
        engine.infer(prompt="hello", max_tokens=8193, grounding=False)
