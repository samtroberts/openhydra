import pytest

from coordinator.chain import ChainResult, StageTrace
from coordinator.engine import CoordinatorEngine, EngineConfig
from coordinator.path_finder import PeerEndpoint, PeerHealth
from coordinator.speculative import SpeculativeSelection


def _engine(tmp_path):
    return CoordinatorEngine(
        EngineConfig(
            peers_config_path="/tmp/unused.json",
            ledger_path=str(tmp_path / "credits.json"),
            health_store_path=str(tmp_path / "health.json"),
            audit_rate=0.0,
            barter_decay_per_day=0.0,
        )
    )


def test_messages_to_prompt_formats_roles(tmp_path):
    engine = _engine(tmp_path)
    prompt = engine._messages_to_prompt(
        [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
    )
    assert prompt == "system: rules\nuser: hello\nassistant: hi"


def test_messages_to_model_prompt_uses_chat_template_when_available(tmp_path, monkeypatch):
    engine = _engine(tmp_path)

    class _Tokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            assert tokenize is False
            assert add_generation_prompt is True
            return f"TEMPLATE::{messages[0]['role']}::{messages[0]['content']}"

    monkeypatch.setattr(engine, "_load_generation_tokenizer", lambda model_id: _Tokenizer())

    prompt = engine._messages_to_model_prompt(
        [{"role": "user", "content": "hello hydra"}],
        model_id="Qwen/Qwen3.5-0.8B",
    )
    assert prompt == "TEMPLATE::user::hello hydra"


def test_messages_to_model_prompt_passes_enable_thinking_false_by_default(tmp_path, monkeypatch):
    """2026-04-24 free-win fix: ``EngineConfig.chat_template_default_kwargs``
    defaults to ``{"enable_thinking": False}`` and is forwarded to
    ``apply_chat_template``. On Qwen3.5 this drops the ``<think>...</think>``
    preamble (~30-40% of the user-visible token budget).
    """
    engine = _engine(tmp_path)

    received: dict = {}

    class _ThinkingAwareTokenizer:
        # Accepts the kwarg — like Qwen3.5's tokenizer.
        def apply_chat_template(
            self, messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=True,
        ):
            received["enable_thinking"] = enable_thinking
            return f"PROMPT(think={enable_thinking})::{messages[0]['content']}"

    monkeypatch.setattr(
        engine, "_load_generation_tokenizer",
        lambda model_id: _ThinkingAwareTokenizer(),
    )
    prompt = engine._messages_to_model_prompt(
        [{"role": "user", "content": "haiku"}],
        model_id="Qwen/Qwen3.5-2B",
    )
    assert received.get("enable_thinking") is False
    assert prompt == "PROMPT(think=False)::haiku"


def test_messages_to_model_prompt_falls_back_when_kwarg_unsupported(tmp_path, monkeypatch):
    """A tokenizer that doesn't accept ``enable_thinking`` (older HF
    transformers, or non-Qwen models) raises ``TypeError`` — the
    retry path strips the extra kwargs and re-applies the template.
    """
    engine = _engine(tmp_path)
    call_log: list[dict] = []

    class _StrictTokenizer:
        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=True,
        ):
            call_log.append({"add_generation_prompt": add_generation_prompt})
            return f"BASIC::{messages[0]['content']}"

    monkeypatch.setattr(
        engine, "_load_generation_tokenizer",
        lambda model_id: _StrictTokenizer(),
    )
    prompt = engine._messages_to_model_prompt(
        [{"role": "user", "content": "haiku"}],
        model_id="some/other-model",
    )
    # Two attempts: one with enable_thinking (TypeError), one without.
    # Final assertion: the strict tokeniser was called once successfully
    # and the prompt is the basic template.
    assert len(call_log) == 1
    assert call_log[0]["add_generation_prompt"] is True
    assert prompt == "BASIC::haiku"


def test_infer_chat_stream_normalizes_openai_decode_aliases(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    seen: dict[str, object] = {}

    monkeypatch.setattr(engine, "_messages_to_model_prompt", lambda messages, model_id: "prompt")

    def _fake_infer_stream(**kwargs):
        seen.update(kwargs)
        return {"request_id": "r1", "stream": iter(())}

    monkeypatch.setattr(engine, "infer_stream", _fake_infer_stream)

    engine.infer_chat_stream(
        messages=[{"role": "user", "content": "hello"}],
        model_id="openhydra-toy-345m",
        do_sample=False,
        temperature=0.2,
        top_p=0.85,
        top_k=9,
        seed=7,
    )

    assert seen["decode_do_sample"] is False
    assert seen["decode_temperature"] == pytest.approx(0.2)
    assert seen["decode_top_p"] == pytest.approx(0.85)
    assert seen["decode_top_k"] == 9
    assert seen["decode_seed"] == 7
    assert "do_sample" not in seen
    assert "temperature" not in seen
    assert "top_p" not in seen
    assert "top_k" not in seen
    assert "seed" not in seen


def test_extract_prompt_expert_layer_indices_parses_hints(tmp_path):
    engine = _engine(tmp_path)
    layers = engine._extract_prompt_expert_layer_indices("draft layer:8 and expert-layer:12 then layer:8 again")
    assert layers == [8, 12]


def test_priority_requires_credits_and_earns_for_serving(tmp_path, monkeypatch):
    engine = _engine(tmp_path)

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
                "served_model": "openhydra-toy-345m",
                "degraded": False,
                "available": True,
                "reason": "ok",
                "detail": "ok",
            })(),
            {"openhydra-toy-345m": 1},
        ),
    )
    monkeypatch.setattr(engine, "_select_pipeline", lambda candidates, pipeline_width=None: [peer])

    def fake_run_chain(prompt, candidates, pipeline, max_tokens, request_id=None, deadline=None):
        return ChainResult(
            request_id=request_id or "r1",
            text="Hydra output.",
            activation=[0.1],
            traces=[StageTrace(peer_id="peer-a", latency_ms=1.0, stage_index=0)],
            latency_ms=5.0,
        )

    monkeypatch.setattr(engine, "_run_chain", fake_run_chain)

    with pytest.raises(RuntimeError, match="insufficient_priority_credits"):
        engine.infer(prompt="hello", priority=True, client_id="user-1")

    engine.ledger.earn("user-1", tokens_served=1000)
    payload = engine.infer(prompt="hello", priority=True, client_id="user-1", grounding=False)

    assert payload["response"] == "Hydra output."
    assert payload["replication"]["under_replicated"] is True
    assert engine.ledger.balance("user-1") == pytest.approx(0.0, abs=1e-6)
    assert engine.ledger.balance("peer-a") > 0.0


def _wire_single_peer(monkeypatch, engine):
    peer = PeerEndpoint(peer_id="peer-a", host="127.0.0.1", port=1)
    peer_b = PeerEndpoint(peer_id="peer-b", host="127.0.0.1", port=2)
    health = [
        PeerHealth(peer=peer, healthy=True, latency_ms=10.0, load_pct=0.0, daemon_mode="polite"),
        PeerHealth(peer=peer_b, healthy=True, latency_ms=10.0, load_pct=0.0, daemon_mode="polite"),
    ]

    monkeypatch.setattr(
        engine,
        "_discover_for_model",
        lambda requested_model, allow_degradation: (
            health,
            [peer, peer_b],
            type("Decision", (), {
                "requested_model": requested_model,
                "served_model": "openhydra-toy-345m",
                "degraded": False,
                "available": True,
                "reason": "ok",
                "detail": "ok",
            })(),
            {"openhydra-toy-345m": 2},
        ),
    )
    monkeypatch.setattr(engine, "_select_pipeline", lambda candidates, pipeline_width=None: [peer])


def test_infer_moe_geo_reorders_pipeline_for_requested_experts(tmp_path, monkeypatch):
    engine = CoordinatorEngine(
        EngineConfig(
            peers_config_path="/tmp/unused.json",
            ledger_path=str(tmp_path / "credits.json"),
            health_store_path=str(tmp_path / "health.json"),
            audit_rate=0.0,
            redundant_exec_rate=0.0,
            moe_geo_enabled=True,
            moe_geo_min_tag_matches=1,
            dht_preferred_region="us-east",
            barter_decay_per_day=0.0,
        )
    )

    peer_a = PeerEndpoint(
        peer_id="peer-a",
        host="127.0.0.1",
        port=5001,
        region="us-east",
        bandwidth_mbps=200.0,
        expert_tags=("vision",),
        expert_layer_indices=(0, 1),
        expert_router=False,
    )
    peer_b = PeerEndpoint(
        peer_id="peer-b",
        host="127.0.0.1",
        port=5002,
        region="us-east",
        bandwidth_mbps=120.0,
        expert_tags=("code",),
        expert_layer_indices=(4, 5),
        expert_router=True,
    )
    peer_c = PeerEndpoint(
        peer_id="peer-c",
        host="127.0.0.1",
        port=5003,
        region="eu-west",
        bandwidth_mbps=300.0,
        expert_tags=("code", "math"),
        expert_layer_indices=(6,),
        expert_router=False,
    )
    health = [
        PeerHealth(peer=peer_a, healthy=True, latency_ms=9.0, load_pct=10.0, daemon_mode="polite"),
        PeerHealth(peer=peer_b, healthy=True, latency_ms=10.0, load_pct=11.0, daemon_mode="polite"),
        PeerHealth(peer=peer_c, healthy=True, latency_ms=12.0, load_pct=12.0, daemon_mode="polite"),
    ]

    monkeypatch.setattr(
        engine,
        "_discover_for_model",
        lambda requested_model, allow_degradation: (
            health,
            [peer_a, peer_b, peer_c],
            type("Decision", (), {
                "requested_model": requested_model,
                "served_model": "openhydra-toy-345m",
                "degraded": False,
                "available": True,
                "reason": "ok",
                "detail": "ok",
            })(),
            {"openhydra-toy-345m": 3},
        ),
    )
    monkeypatch.setattr(engine, "_select_pipeline", lambda candidates, pipeline_width=None: [peer_a, peer_c])
    monkeypatch.setattr(
        engine,
        "_apply_bandwidth_asymmetry",
        lambda pipeline, ranked_candidates, prompt_tokens_est, session_id=None, model_id=None: (
            pipeline,
            {
                "prefill_required": False,
                "prefill_peer_id": pipeline[0].peer_id if pipeline else None,
            },
        ),
    )

    seen: dict[str, list[str]] = {}

    def fake_run_chain(prompt, candidates, pipeline, max_tokens, request_id=None, deadline=None):
        seen["pipeline"] = [item.peer_id for item in pipeline]
        return ChainResult(
            request_id=request_id or "moe-r1",
            text="Hydra output.",
            activation=[0.1],
            traces=[StageTrace(peer_id=pipeline[0].peer_id, latency_ms=1.0, stage_index=0)],
            latency_ms=5.0,
        )

    monkeypatch.setattr(engine, "_run_chain", fake_run_chain)

    payload = engine.infer(
        prompt="hello",
        grounding=False,
        pipeline_width=2,
        expert_tags=["CODE"],
    )

    assert seen["pipeline"] == ["peer-b", "peer-c"]
    assert payload["moe_geo"]["enabled"] is True
    assert payload["moe_geo"]["requested_experts"] == ["code"]
    assert payload["moe_geo"]["matched_peer_ids"] == ["peer-b", "peer-c"]
    assert payload["moe_geo"]["matched_tags"] == ["code"]
    assert payload["moe_geo"]["router_peer_ids"] == ["peer-b"]
    assert payload["moe_geo"]["applied"] is True
    assert payload["pipeline"][0]["peer_id"] == "peer-b"


def test_infer_moe_geo_reorders_pipeline_for_requested_layers(tmp_path, monkeypatch):
    engine = CoordinatorEngine(
        EngineConfig(
            peers_config_path="/tmp/unused.json",
            ledger_path=str(tmp_path / "credits.json"),
            health_store_path=str(tmp_path / "health.json"),
            audit_rate=0.0,
            redundant_exec_rate=0.0,
            moe_geo_enabled=True,
            moe_geo_min_layer_matches=1,
            barter_decay_per_day=0.0,
        )
    )

    peer_a = PeerEndpoint(
        peer_id="peer-a",
        host="127.0.0.1",
        port=5001,
        bandwidth_mbps=250.0,
        expert_tags=("vision",),
        expert_layer_indices=(1, 2),
    )
    peer_b = PeerEndpoint(
        peer_id="peer-b",
        host="127.0.0.1",
        port=5002,
        bandwidth_mbps=120.0,
        expert_tags=("audio",),
        expert_layer_indices=(8,),
    )
    peer_c = PeerEndpoint(
        peer_id="peer-c",
        host="127.0.0.1",
        port=5003,
        bandwidth_mbps=200.0,
        expert_tags=("code",),
        expert_layer_indices=(6,),
    )
    health = [
        PeerHealth(peer=peer_a, healthy=True, latency_ms=8.0, load_pct=8.0, daemon_mode="polite"),
        PeerHealth(peer=peer_b, healthy=True, latency_ms=9.0, load_pct=9.0, daemon_mode="polite"),
        PeerHealth(peer=peer_c, healthy=True, latency_ms=10.0, load_pct=10.0, daemon_mode="polite"),
    ]

    monkeypatch.setattr(
        engine,
        "_discover_for_model",
        lambda requested_model, allow_degradation: (
            health,
            [peer_a, peer_b, peer_c],
            type("Decision", (), {
                "requested_model": requested_model,
                "served_model": "openhydra-toy-345m",
                "degraded": False,
                "available": True,
                "reason": "ok",
                "detail": "ok",
            })(),
            {"openhydra-toy-345m": 3},
        ),
    )
    monkeypatch.setattr(engine, "_select_pipeline", lambda candidates, pipeline_width=None: [peer_a, peer_c])
    monkeypatch.setattr(
        engine,
        "_apply_bandwidth_asymmetry",
        lambda pipeline, ranked_candidates, prompt_tokens_est, session_id=None, model_id=None: (
            pipeline,
            {
                "prefill_required": False,
                "prefill_peer_id": pipeline[0].peer_id if pipeline else None,
            },
        ),
    )

    seen: dict[str, list[str]] = {}

    def fake_run_chain(prompt, candidates, pipeline, max_tokens, request_id=None, deadline=None):
        seen["pipeline"] = [item.peer_id for item in pipeline]
        return ChainResult(
            request_id=request_id or "moe-layer-r1",
            text="Hydra output.",
            activation=[0.1],
            traces=[StageTrace(peer_id=pipeline[0].peer_id, latency_ms=1.0, stage_index=0)],
            latency_ms=5.0,
        )

    monkeypatch.setattr(engine, "_run_chain", fake_run_chain)

    payload = engine.infer(
        prompt="hello",
        grounding=False,
        pipeline_width=2,
        expert_layer_indices=[8],
    )

    assert seen["pipeline"] == ["peer-b", "peer-a"]
    assert payload["moe_geo"]["enabled"] is True
    assert payload["moe_geo"]["requested_experts"] == []
    assert payload["moe_geo"]["requested_layer_indices"] == [8]
    assert payload["moe_geo"]["matched_peer_ids"] == ["peer-b"]
    assert payload["moe_geo"]["matched_layer_indices"] == [8]
    assert payload["moe_geo"]["applied"] is True
    assert payload["pipeline"][0]["peer_id"] == "peer-b"


def test_tier2_verification_uses_redundant_exec_rate(tmp_path, monkeypatch):
    engine = CoordinatorEngine(
        EngineConfig(
            peers_config_path="/tmp/unused.json",
            ledger_path=str(tmp_path / "credits.json"),
            health_store_path=str(tmp_path / "health.json"),
            audit_rate=1.0,
            redundant_exec_rate=0.0,
            tier=2,
            barter_decay_per_day=0.0,
        )
    )
    _wire_single_peer(monkeypatch, engine)

    calls = {"count": 0}

    def fake_run_chain(prompt, candidates, pipeline, max_tokens, request_id=None, deadline=None):
        calls["count"] += 1
        return ChainResult(
            request_id=request_id or "r1",
            text="Hydra output.",
            activation=[0.1],
            traces=[StageTrace(peer_id="peer-a", latency_ms=1.0, stage_index=0)],
            latency_ms=5.0,
        )

    monkeypatch.setattr(engine, "_run_chain", fake_run_chain)

    payload = engine.infer(prompt="hello", grounding=False)
    assert payload["verification"]["audited"] is False
    assert payload["verification"]["mode"] == "redundant_execution"
    assert calls["count"] == 1


def test_tier1_verification_uses_audit_rate(tmp_path, monkeypatch):
    engine = CoordinatorEngine(
        EngineConfig(
            peers_config_path="/tmp/unused.json",
            ledger_path=str(tmp_path / "credits.json"),
            health_store_path=str(tmp_path / "health.json"),
            audit_rate=1.0,
            redundant_exec_rate=0.0,
            tier=1,
            barter_decay_per_day=0.0,
        )
    )
    _wire_single_peer(monkeypatch, engine)

    calls = {"count": 0}

    def fake_run_chain(prompt, candidates, pipeline, max_tokens, request_id=None, deadline=None):
        calls["count"] += 1
        return ChainResult(
            request_id=request_id or "r1",
            text="Hydra output.",
            activation=[0.1],
            traces=[StageTrace(peer_id="peer-a", latency_ms=1.0, stage_index=0)],
            latency_ms=5.0,
        )

    monkeypatch.setattr(engine, "_run_chain", fake_run_chain)

    payload = engine.infer(prompt="hello", grounding=False)
    assert payload["verification"]["audited"] is True
    assert payload["verification"]["mode"] == "mystery_shopper"
    assert calls["count"] == 2


def test_verification_feedback_updates_health_for_winner_and_loser(tmp_path, monkeypatch):
    engine = CoordinatorEngine(
        EngineConfig(
            peers_config_path="/tmp/unused.json",
            ledger_path=str(tmp_path / "credits.json"),
            health_store_path=str(tmp_path / "health.json"),
            tier=2,
            redundant_exec_rate=1.0,
            audit_rate=0.0,
            barter_decay_per_day=0.0,
        )
    )

    peer_a = PeerEndpoint(peer_id="peer-a", host="127.0.0.1", port=1)
    peer_b = PeerEndpoint(peer_id="peer-b", host="127.0.0.1", port=2)
    health = [
        PeerHealth(peer=peer_a, healthy=True, latency_ms=10.0, load_pct=0.0, daemon_mode="polite"),
        PeerHealth(peer=peer_b, healthy=True, latency_ms=11.0, load_pct=0.0, daemon_mode="polite"),
    ]

    monkeypatch.setattr(
        engine,
        "_discover_for_model",
        lambda requested_model, allow_degradation: (
            health,
            [peer_a, peer_b],
            type("Decision", (), {
                "requested_model": requested_model,
                "served_model": "openhydra-toy-345m",
                "degraded": False,
                "available": True,
                "reason": "ok",
                "detail": "ok",
            })(),
            {"openhydra-toy-345m": 2},
        ),
    )
    monkeypatch.setattr(engine, "_select_pipeline", lambda candidates, pipeline_width=None: [candidates[0]])

    def fake_run_chain(prompt, candidates, pipeline, max_tokens, request_id=None, deadline=None):
        chosen = pipeline[0].peer_id
        text = "alpha" if chosen == "peer-a" else "beta"
        return ChainResult(
            request_id=request_id or "r1",
            text=text,
            activation=[0.1],
            traces=[StageTrace(peer_id=chosen, latency_ms=1.0, stage_index=0)],
            latency_ms=5.0,
        )

    monkeypatch.setattr(engine, "_run_chain", fake_run_chain)

    payload = engine.infer(prompt="hello", grounding=False, pipeline_width=1)

    assert payload["verification"]["audited"] is True
    assert payload["verification"]["winner"] == "secondary"
    assert payload["verification_feedback"]["rewarded_peers"] == ["peer-b"]
    assert payload["verification_feedback"]["penalized_peers"] == ["peer-a"]

    snapshot = engine.health.snapshot()
    assert snapshot["peer-a"]["verifications_failed"] >= 1
    assert snapshot["peer-b"]["verifications_ok"] >= 1


def test_auditor_spotcheck_penalizes_only_divergent_tertiary(tmp_path, monkeypatch):
    engine = CoordinatorEngine(
        EngineConfig(
            peers_config_path="/tmp/unused.json",
            ledger_path=str(tmp_path / "credits.json"),
            health_store_path=str(tmp_path / "health.json"),
            tier=2,
            redundant_exec_rate=1.0,
            auditor_rate=1.0,
            audit_rate=0.0,
            barter_decay_per_day=0.0,
        )
    )

    peer_a = PeerEndpoint(peer_id="peer-a", host="127.0.0.1", port=1)
    peer_b = PeerEndpoint(peer_id="peer-b", host="127.0.0.1", port=2)
    peer_c = PeerEndpoint(peer_id="peer-c", host="127.0.0.1", port=3)
    health = [
        PeerHealth(peer=peer_a, healthy=True, latency_ms=10.0, load_pct=0.0, daemon_mode="polite"),
        PeerHealth(peer=peer_b, healthy=True, latency_ms=11.0, load_pct=0.0, daemon_mode="polite"),
        PeerHealth(peer=peer_c, healthy=True, latency_ms=12.0, load_pct=0.0, daemon_mode="polite"),
    ]

    monkeypatch.setattr(
        engine,
        "_discover_for_model",
        lambda requested_model, allow_degradation: (
            health,
            [peer_a, peer_b, peer_c],
            type("Decision", (), {
                "requested_model": requested_model,
                "served_model": "openhydra-toy-345m",
                "degraded": False,
                "available": True,
                "reason": "ok",
                "detail": "ok",
            })(),
            {"openhydra-toy-345m": 3},
        ),
    )
    monkeypatch.setattr(engine, "_select_pipeline", lambda candidates, pipeline_width=None: [candidates[0]])

    def fake_run_chain(prompt, candidates, pipeline, max_tokens, request_id=None, deadline=None):
        chosen = pipeline[0].peer_id
        text = {"peer-a": "alpha", "peer-b": "alpha", "peer-c": "gamma"}[chosen]
        return ChainResult(
            request_id=request_id or "r1",
            text=text,
            activation=[0.1],
            traces=[StageTrace(peer_id=chosen, latency_ms=1.0, stage_index=0)],
            latency_ms=5.0,
        )

    monkeypatch.setattr(engine, "_run_chain", fake_run_chain)
    payload = engine.infer(prompt="hello", grounding=False, pipeline_width=1)

    assert payload["verification"]["audited"] is True
    assert payload["verification"]["auditor_triggered"] is True
    assert payload["verification"]["winner"] == "primary"
    assert payload["verification"]["match"] is False
    assert payload["verification_feedback"]["rewarded_peers"] == ["peer-a", "peer-b"]
    assert payload["verification_feedback"]["penalized_peers"] == ["peer-c"]

    snapshot = engine.health.snapshot()
    assert snapshot["peer-a"]["verifications_ok"] >= 1
    assert snapshot["peer-b"]["verifications_ok"] >= 1
    assert snapshot["peer-c"]["verifications_failed"] >= 1


def test_infer_stream_executes_decode_path_incrementally(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    _wire_single_peer(monkeypatch, engine)

    calls: list[tuple[str, list[float]]] = []

    def fake_run_chain(prompt, candidates, pipeline, max_tokens, request_id=None, initial_activation=None, deadline=None):
        seed = list(initial_activation or [])
        calls.append((prompt, seed))
        step = len(calls)
        return ChainResult(
            request_id=request_id or "stream-rid",
            text=f"unused-{step}",
            activation=[float(step)],
            traces=[StageTrace(peer_id="peer-a", latency_ms=1.0, stage_index=0)],
            latency_ms=5.0,
        )

    monkeypatch.setattr(engine, "_run_chain", fake_run_chain)
    monkeypatch.setattr(
        "coordinator.engine.ModelShard.decode_tokens",
        lambda activation, max_tokens, tokenizer_model_id=None: [f"tok{int(activation[0])}"],
    )

    payload = engine.infer_stream(prompt="hello", max_tokens=3, grounding=False)
    chunks = list(payload["stream"])

    assert payload["streaming"]["execution_path"] is True
    assert chunks == ["Tok1", " tok2", " tok3", "."]
    assert calls == [("hello", []), ("", [1.0]), ("", [2.0])]
    assert payload["streaming"]["kv_data_plane"]["seeded_rounds"] == 2
    assert payload["streaming"]["pipeline_parallel"] == {
        "enabled": False,
        "workers": 1,
        "prefetch_submitted": 0,
        "prefetch_hits": 0,
        "prefetch_misses": 0,
        "prefetch_failures": 0,
        "prefetch_waits": 0,
    }


def test_infer_stream_pipeline_parallel_prefetch_hits(tmp_path, monkeypatch):
    engine = CoordinatorEngine(
        EngineConfig(
            peers_config_path="/tmp/unused.json",
            ledger_path=str(tmp_path / "credits.json"),
            health_store_path=str(tmp_path / "health.json"),
            audit_rate=0.0,
            redundant_exec_rate=0.0,
            pipeline_parallel_enabled=True,
            pipeline_parallel_workers=1,
            barter_decay_per_day=0.0,
        )
    )
    _wire_single_peer(monkeypatch, engine)

    calls: list[tuple[str, list[float]]] = []

    def fake_run_chain(prompt, candidates, pipeline, max_tokens, request_id=None, initial_activation=None, deadline=None):
        seed = list(initial_activation or [])
        calls.append((prompt, seed))
        step = len(calls)
        return ChainResult(
            request_id=request_id or "stream-rid",
            text=f"unused-{step}",
            activation=[float(step)],
            traces=[StageTrace(peer_id="peer-a", latency_ms=1.0, stage_index=0)],
            latency_ms=5.0,
        )

    monkeypatch.setattr(engine, "_run_chain", fake_run_chain)
    monkeypatch.setattr(
        "coordinator.engine.ModelShard.decode_tokens",
        lambda activation, max_tokens, tokenizer_model_id=None: [f"tok{int(activation[0])}"],
    )

    payload = engine.infer_stream(prompt="hello", max_tokens=3, grounding=False)
    chunks = list(payload["stream"])

    assert chunks == ["Tok1", " tok2", " tok3", "."]
    assert calls == [("hello", []), ("", [1.0]), ("", [2.0])]
    assert payload["streaming"]["pipeline_parallel"] == {
        "enabled": True,
        "workers": 1,
        "prefetch_submitted": 2,
        "prefetch_hits": 2,
        "prefetch_misses": 0,
        "prefetch_failures": 0,
        "prefetch_waits": 2,
    }


def test_infer_stream_speculative_batches_and_accepts_mismatches(tmp_path, monkeypatch):
    engine = CoordinatorEngine(
        EngineConfig(
            peers_config_path="/tmp/unused.json",
            ledger_path=str(tmp_path / "credits.json"),
            health_store_path=str(tmp_path / "health.json"),
            audit_rate=0.0,
            redundant_exec_rate=0.0,
            speculative_enabled=True,
            speculative_draft_tokens=3,
            barter_decay_per_day=0.0,
        )
    )
    _wire_single_peer(monkeypatch, engine)

    run_calls: list[tuple[str, int, list[float]]] = []
    draft_calls: list[tuple[str, int]] = []

    def fake_run_chain(prompt, candidates, pipeline, max_tokens, request_id=None, initial_activation=None, deadline=None):
        seed = list(initial_activation or [])
        run_calls.append((prompt, max_tokens, seed))
        step = len(run_calls)
        return ChainResult(
            request_id=request_id or "stream-rid",
            text=f"unused-{step}",
            activation=[float(step)],
            traces=[StageTrace(peer_id="peer-a", latency_ms=1.0, stage_index=0)],
            latency_ms=5.0,
        )

    monkeypatch.setattr(engine, "_run_chain", fake_run_chain)
    monkeypatch.setattr(
        "coordinator.engine.ModelShard.decode_tokens",
        lambda activation, max_tokens, tokenizer_model_id=None: (
            ["aa", "bb", "cc"] if int(activation[0]) == 1 else ["cc", "dd"]
        ),
    )

    def fake_propose(prompt, max_tokens):
        draft_calls.append((prompt, max_tokens))
        return ["aa", "xx", "yy"] if len(draft_calls) == 1 else ["cc", "dd"]

    monkeypatch.setattr(engine.draft_model, "propose", fake_propose)

    payload = engine.infer_stream(prompt="hello", max_tokens=4, grounding=False)
    chunks = list(payload["stream"])

    assert payload["streaming"]["mode"] == "speculative_decode"
    assert payload["streaming"]["speculative_enabled"] is True
    assert payload["streaming"]["speculative_draft_tokens"] == 3
    assert run_calls == [("hello", 3, []), ("", 2, [1.0])]
    assert draft_calls == [("hello", 3), ("hello aa bb", 2)]
    assert chunks == ["Aa", " bb", " cc", " dd", "."]
    assert payload["streaming"]["kv_data_plane"]["seeded_rounds"] == 1


def test_infer_stream_speculative_adaptive_batching(tmp_path, monkeypatch):
    engine = CoordinatorEngine(
        EngineConfig(
            peers_config_path="/tmp/unused.json",
            ledger_path=str(tmp_path / "credits.json"),
            health_store_path=str(tmp_path / "health.json"),
            audit_rate=0.0,
            redundant_exec_rate=0.0,
            speculative_enabled=True,
            speculative_draft_tokens=4,
            speculative_adaptive_enabled=True,
            speculative_min_draft_tokens=2,
            speculative_max_draft_tokens=4,
            speculative_acceptance_low_watermark=0.55,
            speculative_acceptance_high_watermark=0.80,
            barter_decay_per_day=0.0,
        )
    )
    _wire_single_peer(monkeypatch, engine)

    run_batch_sizes: list[int] = []

    def fake_run_chain(prompt, candidates, pipeline, max_tokens, request_id=None, initial_activation=None, deadline=None):
        run_batch_sizes.append(max_tokens)
        return ChainResult(
            request_id=request_id or "stream-rid",
            text="unused",
            activation=[0.1],
            traces=[StageTrace(peer_id="peer-a", latency_ms=1.0, stage_index=0)],
            latency_ms=5.0,
        )

    monkeypatch.setattr(engine, "_run_chain", fake_run_chain)
    monkeypatch.setattr(
        "coordinator.engine.ModelShard.decode_tokens",
        lambda activation, max_tokens, tokenizer_model_id=None: [f"tok{i}" for i in range(max_tokens)],
    )
    monkeypatch.setattr(
        engine.draft_model,
        "propose",
        lambda prompt, max_tokens: [f"draft{i}" for i in range(max_tokens)],
    )

    selections = [
        SpeculativeSelection(accepted_tokens=["tok0"], matched_prefix=0, mismatch=True),
        SpeculativeSelection(accepted_tokens=["tok0"], matched_prefix=0, mismatch=True),
        SpeculativeSelection(accepted_tokens=["tok0", "tok1"], matched_prefix=2, mismatch=False),
    ]

    monkeypatch.setattr(
        "coordinator.engine.select_verified_tokens",
        lambda verified_tokens, draft_tokens: selections.pop(0),
    )

    payload = engine.infer_stream(prompt="hello", max_tokens=4, grounding=False)
    chunks = list(payload["stream"])

    assert run_batch_sizes == [4, 3, 2]
    assert chunks == ["Tok0", " tok0", " tok0", " tok1", "."]
    stats = payload["streaming"]["speculative"]
    assert stats["rounds"] == 3
    assert stats["mismatch_rounds"] == 2
    assert stats["accepted_tokens"] == 4
    assert stats["verified_tokens"] == 9
    assert stats["acceptance_rate"] == pytest.approx(4 / 9, abs=1e-6)
    assert stats["current_draft_tokens"] == 3
    assert payload["streaming"]["kv_data_plane"]["seeded_rounds"] == 2


def test_run_chain_passes_tensor_autoencoder_config(tmp_path, monkeypatch):
    engine = CoordinatorEngine(
        EngineConfig(
            peers_config_path="/tmp/unused.json",
            ledger_path=str(tmp_path / "credits.json"),
            health_store_path=str(tmp_path / "health.json"),
            audit_rate=0.0,
            redundant_exec_rate=0.0,
            tensor_autoencoder_enabled=True,
            tensor_autoencoder_latent_dim=7,
            advanced_encryption_enabled=True,
            advanced_encryption_seed="enc-seed",
            advanced_encryption_level="enhanced",
            barter_decay_per_day=0.0,
        )
    )

    peer = PeerEndpoint(peer_id="peer-a", host="127.0.0.1", port=1)
    captured: dict[str, object] = {}

    class _DummyChain:
        def __init__(
            self,
            pipeline,
            timeout_ms,
            transport_config,
            tensor_autoencoder_enabled,
            tensor_autoencoder_latent_dim,
            advanced_encryption_enabled,
            advanced_encryption_seed,
            advanced_encryption_level,
            activation_quantization_enabled=False,
            **kwargs,
        ):
            captured["pipeline"] = pipeline
            captured["timeout_ms"] = timeout_ms
            captured["tensor_autoencoder_enabled"] = tensor_autoencoder_enabled
            captured["tensor_autoencoder_latent_dim"] = tensor_autoencoder_latent_dim
            captured["advanced_encryption_enabled"] = advanced_encryption_enabled
            captured["advanced_encryption_seed"] = advanced_encryption_seed
            captured["advanced_encryption_level"] = advanced_encryption_level

        def run(self, prompt, max_tokens, request_id=None, failover_pool=None, max_failovers_per_stage=0, deadline=None, **kwargs):
            return ChainResult(
                request_id=request_id or "r1",
                text="ok",
                activation=[0.1],
                traces=[StageTrace(peer_id="peer-a", latency_ms=1.0, stage_index=0)],
                latency_ms=5.0,
                compression={"enabled": True, "method": "tensor_autoencoder_mean_pool"},
            )

    monkeypatch.setattr("coordinator.engine.InferenceChain", _DummyChain)

    result = engine._run_chain("hello", [peer], [peer], max_tokens=3)

    assert captured["tensor_autoencoder_enabled"] is True
    assert captured["tensor_autoencoder_latent_dim"] == 7
    assert captured["advanced_encryption_enabled"] is True
    assert captured["advanced_encryption_seed"] == "enc-seed"
    assert captured["advanced_encryption_level"] == "enhanced"
    assert result.compression["enabled"] is True


def test_infer_payload_includes_compression_metrics(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    _wire_single_peer(monkeypatch, engine)

    def fake_run_chain(prompt, candidates, pipeline, max_tokens, request_id=None, deadline=None):
        return ChainResult(
            request_id=request_id or "r1",
            text="Hydra output.",
            activation=[0.1],
            traces=[StageTrace(peer_id="peer-a", latency_ms=1.0, stage_index=0)],
            latency_ms=5.0,
            compression={"enabled": True, "hops_compressed": 1},
            encryption={"enabled": True, "encrypted_hops": 0},
        )

    monkeypatch.setattr(engine, "_run_chain", fake_run_chain)

    payload = engine.infer(prompt="hello", grounding=False)
    assert payload["compression"]["enabled"] is True
    assert payload["compression"]["hops_compressed"] == 1
    assert payload["encryption"]["enabled"] is True


def test_infer_stream_payload_includes_encryption_config(tmp_path, monkeypatch):
    engine = CoordinatorEngine(
        EngineConfig(
            peers_config_path="/tmp/unused.json",
            ledger_path=str(tmp_path / "credits.json"),
            health_store_path=str(tmp_path / "health.json"),
            audit_rate=0.0,
            redundant_exec_rate=0.0,
            advanced_encryption_enabled=True,
            advanced_encryption_level="maximum",
            barter_decay_per_day=0.0,
        )
    )
    _wire_single_peer(monkeypatch, engine)

    def fake_run_chain(prompt, candidates, pipeline, max_tokens, request_id=None, deadline=None):
        return ChainResult(
            request_id=request_id or "r1",
            text="Hydra output.",
            activation=[0.1],
            traces=[StageTrace(peer_id="peer-a", latency_ms=1.0, stage_index=0)],
            latency_ms=5.0,
        )

    monkeypatch.setattr(engine, "_run_chain", fake_run_chain)

    payload = engine.infer_stream(prompt="hello", max_tokens=1, grounding=False)
    assert payload["encryption"]["enabled"] is True
    assert payload["encryption"]["level"] == "maximum"
    assert payload["encryption"]["layers_per_hop"] == 3


def test_infer_mints_hydra_rewards_and_account_balance_includes_hydra(tmp_path, monkeypatch):
    engine = CoordinatorEngine(
        EngineConfig(
            peers_config_path="/tmp/unused.json",
            ledger_path=str(tmp_path / "credits.json"),
            hydra_token_ledger_path=str(tmp_path / "hydra_tokens.json"),
            health_store_path=str(tmp_path / "health.json"),
            audit_rate=0.0,
            redundant_exec_rate=0.0,
            hydra_reward_per_1k_tokens=2.0,
            barter_decay_per_day=0.0,
        )
    )
    _wire_single_peer(monkeypatch, engine)

    def fake_run_chain(prompt, candidates, pipeline, max_tokens, request_id=None, deadline=None):
        return ChainResult(
            request_id=request_id or "r1",
            text="Hydra output.",
            activation=[0.1],
            traces=[StageTrace(peer_id="peer-a", latency_ms=1.0, stage_index=0)],
            latency_ms=5.0,
        )

    monkeypatch.setattr(engine, "_run_chain", fake_run_chain)
    engine.infer(prompt="hello", max_tokens=5, grounding=False)

    account = engine.account_balance("peer-a")
    assert "hydra" in account
    assert account["hydra"]["balance"] == 0.01

    status = engine.network_status()
    assert "hydra_economy" in status
    assert status["hydra_economy"]["total_minted"] >= 0.01


def test_verification_penalty_slashes_hydra_stake_when_enabled(tmp_path, monkeypatch):
    engine = CoordinatorEngine(
        EngineConfig(
            peers_config_path="/tmp/unused.json",
            ledger_path=str(tmp_path / "credits.json"),
            hydra_token_ledger_path=str(tmp_path / "hydra_tokens.json"),
            health_store_path=str(tmp_path / "health.json"),
            tier=2,
            redundant_exec_rate=1.0,
            audit_rate=0.0,
            hydra_reward_per_1k_tokens=0.0,
            hydra_slash_per_failed_verification=0.5,
            barter_decay_per_day=0.0,
        )
    )

    peer_a = PeerEndpoint(peer_id="peer-a", host="127.0.0.1", port=1)
    peer_b = PeerEndpoint(peer_id="peer-b", host="127.0.0.1", port=2)
    health = [
        PeerHealth(peer=peer_a, healthy=True, latency_ms=10.0, load_pct=0.0, daemon_mode="polite"),
        PeerHealth(peer=peer_b, healthy=True, latency_ms=11.0, load_pct=0.0, daemon_mode="polite"),
    ]

    monkeypatch.setattr(
        engine,
        "_discover_for_model",
        lambda requested_model, allow_degradation: (
            health,
            [peer_a, peer_b],
            type("Decision", (), {
                "requested_model": requested_model,
                "served_model": "openhydra-toy-345m",
                "degraded": False,
                "available": True,
                "reason": "ok",
                "detail": "ok",
            })(),
            {"openhydra-toy-345m": 2},
        ),
    )
    monkeypatch.setattr(engine, "_select_pipeline", lambda candidates, pipeline_width=None: [candidates[0]])

    def fake_run_chain(prompt, candidates, pipeline, max_tokens, request_id=None, deadline=None):
        chosen = pipeline[0].peer_id
        text = "alpha" if chosen == "peer-a" else "beta"
        return ChainResult(
            request_id=request_id or "r1",
            text=text,
            activation=[0.1],
            traces=[StageTrace(peer_id=chosen, latency_ms=1.0, stage_index=0)],
            latency_ms=5.0,
        )

    monkeypatch.setattr(engine, "_run_chain", fake_run_chain)
    engine.hydra.mint_for_inference("peer-a", tokens_served=1000, reward_per_1k_tokens=1.0)
    engine.hydra.stake("peer-a", 0.5)

    payload = engine.infer(prompt="hello", grounding=False, pipeline_width=1)
    assert payload["verification"]["audited"] is True
    assert payload["verification"]["winner"] == "secondary"

    slashed = engine.hydra.account_snapshot("peer-a")
    assert slashed["stake"] == 0.0
    assert slashed["slashed_total"] == 0.5


def test_hydra_channel_policy_reflects_engine_config(tmp_path):
    engine = CoordinatorEngine(
        EngineConfig(
            peers_config_path="/tmp/unused.json",
            ledger_path=str(tmp_path / "credits.json"),
            hydra_token_ledger_path=str(tmp_path / "hydra_tokens.json"),
            hydra_channel_default_ttl_seconds=123,
            hydra_channel_max_open_per_payer=4,
            hydra_channel_min_deposit=0.25,
            health_store_path=str(tmp_path / "health.json"),
            audit_rate=0.0,
            redundant_exec_rate=0.0,
            barter_decay_per_day=0.0,
        )
    )

    summary = engine.hydra_status()["hydra"]
    assert summary["channel_policy"] == {
        "default_ttl_seconds": 123,
        "max_open_per_payer": 4,
        "min_deposit": 0.25,
    }
