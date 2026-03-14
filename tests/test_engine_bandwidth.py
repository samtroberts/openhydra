from types import SimpleNamespace

from coordinator.chain import ChainResult, StageTrace
from coordinator.engine import CoordinatorEngine, EngineConfig
from coordinator.path_finder import PeerEndpoint, PeerHealth


def _engine(tmp_path):
    return CoordinatorEngine(
        EngineConfig(
            peers_config_path="/tmp/unused.json",
            ledger_path=str(tmp_path / "credits.json"),
            health_store_path=str(tmp_path / "health.json"),
            barter_decay_per_day=0.0,
            audit_rate=0.0,
            redundant_exec_rate=0.0,
            prefill_token_threshold=5,
        )
    )


def test_infer_prefers_prefill_capable_peer_for_long_prompt(tmp_path, monkeypatch):
    engine = _engine(tmp_path)

    decode = PeerEndpoint(peer_id="decode", host="127.0.0.1", port=1, model_id=engine.config.default_model, bandwidth_mbps=10)
    prefill = PeerEndpoint(peer_id="prefill", host="127.0.0.1", port=2, model_id=engine.config.default_model, bandwidth_mbps=800)
    balanced = PeerEndpoint(peer_id="balanced", host="127.0.0.1", port=3, model_id=engine.config.default_model, bandwidth_mbps=120)

    health = [
        PeerHealth(peer=decode, healthy=True, latency_ms=10.0, load_pct=0.0, daemon_mode="polite"),
        PeerHealth(peer=prefill, healthy=True, latency_ms=10.0, load_pct=0.0, daemon_mode="polite"),
        PeerHealth(peer=balanced, healthy=True, latency_ms=10.0, load_pct=0.0, daemon_mode="polite"),
    ]

    monkeypatch.setattr(
        engine,
        "_discover_for_model",
        lambda requested_model, allow_degradation: (
            health,
            [decode, prefill, balanced],
            SimpleNamespace(
                requested_model=requested_model,
                served_model=engine.config.default_model,
                degraded=False,
                available=True,
                reason="ok",
                detail="ok",
            ),
            {engine.config.default_model: 3},
        ),
    )

    captured = {}

    def fake_run_chain(prompt, candidates, pipeline, max_tokens, request_id=None, **kwargs):
        captured["pipeline"] = [peer.peer_id for peer in pipeline]
        return ChainResult(
            request_id=request_id or "r1",
            text="result",
            activation=[0.1],
            traces=[StageTrace(peer_id=pipeline[0].peer_id, latency_ms=1.0, stage_index=0)],
            latency_ms=3.0,
        )

    monkeypatch.setattr(engine, "_run_chain", fake_run_chain)

    payload = engine.infer(
        prompt="this prompt is intentionally long for prefill",
        grounding=False,
    )

    assert captured["pipeline"][0] == "prefill"
    assert payload["bandwidth_policy"]["prefill_required"] is True
    assert payload["bandwidth_policy"]["prefill_peer_id"] == "prefill"


def test_kv_affinity_sticks_prefill_and_flags_cold_restart(tmp_path, monkeypatch):
    engine = _engine(tmp_path)

    decode = PeerEndpoint(peer_id="decode", host="127.0.0.1", port=1, model_id=engine.config.default_model, bandwidth_mbps=10)
    prefill_a = PeerEndpoint(peer_id="prefill-a", host="127.0.0.1", port=2, model_id=engine.config.default_model, bandwidth_mbps=900)
    prefill_b = PeerEndpoint(peer_id="prefill-b", host="127.0.0.1", port=3, model_id=engine.config.default_model, bandwidth_mbps=800)

    def _health(peers):
        return [PeerHealth(peer=peer, healthy=True, latency_ms=10.0, load_pct=0.0, daemon_mode="polite") for peer in peers]

    calls = {"count": 0}

    def fake_discover(requested_model, allow_degradation):
        calls["count"] += 1
        if calls["count"] == 1:
            peers = [prefill_a, decode, prefill_b]
        else:
            peers = [decode, prefill_b]
        return (
            _health(peers),
            peers,
            SimpleNamespace(
                requested_model=requested_model,
                served_model=engine.config.default_model,
                degraded=False,
                available=True,
                reason="ok",
                detail="ok",
            ),
            {engine.config.default_model: len(peers)},
        )

    monkeypatch.setattr(engine, "_discover_for_model", fake_discover)
    monkeypatch.setattr(engine, "_select_pipeline", lambda candidates, pipeline_width=None: candidates[:2])

    used_prefill: list[str] = []

    def fake_run_chain(prompt, candidates, pipeline, max_tokens, request_id=None, **kwargs):
        used_prefill.append(pipeline[0].peer_id)
        return ChainResult(
            request_id=request_id or "r1",
            text="result",
            activation=[0.1],
            traces=[StageTrace(peer_id=pipeline[0].peer_id, latency_ms=1.0, stage_index=0)],
            latency_ms=3.0,
        )

    monkeypatch.setattr(engine, "_run_chain", fake_run_chain)

    payload_1 = engine.infer(
        prompt="this prompt is intentionally long for prefill",
        grounding=False,
        session_id="session-1",
    )
    payload_2 = engine.infer(
        prompt="this prompt is intentionally long for prefill",
        grounding=False,
        session_id="session-1",
    )
    payload_3 = engine.infer(
        prompt="this prompt is intentionally long for prefill",
        grounding=False,
        session_id="session-1",
    )

    assert used_prefill == ["prefill-a", "prefill-b", "prefill-b"]

    p1 = payload_1["bandwidth_policy"]
    assert p1["kv_affinity_requested"] is True
    assert p1["kv_affinity_hit"] is False
    assert p1["kv_cold_restart"] is False
    assert p1["kv_affinity_updated"] is True

    p2 = payload_2["bandwidth_policy"]
    assert p2["kv_previous_prefill_peer_id"] == "prefill-a"
    assert p2["kv_affinity_hit"] is False
    assert p2["kv_cold_restart"] is True
    assert p2["prefill_peer_id"] == "prefill-b"

    p3 = payload_3["bandwidth_policy"]
    assert p3["kv_previous_prefill_peer_id"] == "prefill-b"
    assert p3["kv_affinity_hit"] is True
    assert p3["kv_cold_restart"] is False
    assert p3["prefill_peer_id"] == "prefill-b"


def test_infer_stream_reuses_kv_data_plane_cache_on_affinity_hit(tmp_path, monkeypatch):
    engine = _engine(tmp_path)

    decode = PeerEndpoint(peer_id="decode", host="127.0.0.1", port=1, model_id=engine.config.default_model, bandwidth_mbps=10)
    prefill = PeerEndpoint(peer_id="prefill", host="127.0.0.1", port=2, model_id=engine.config.default_model, bandwidth_mbps=900)

    peers = [prefill, decode]
    health = [PeerHealth(peer=peer, healthy=True, latency_ms=10.0, load_pct=0.0, daemon_mode="polite") for peer in peers]

    monkeypatch.setattr(
        engine,
        "_discover_for_model",
        lambda requested_model, allow_degradation: (
            health,
            peers,
            SimpleNamespace(
                requested_model=requested_model,
                served_model=engine.config.default_model,
                degraded=False,
                available=True,
                reason="ok",
                detail="ok",
            ),
            {engine.config.default_model: 2},
        ),
    )
    monkeypatch.setattr(engine, "_select_pipeline", lambda candidates, pipeline_width=None: candidates[:2])

    calls: list[dict[str, object]] = []

    peer_cache_state: list[float] | None = None

    def fake_run_chain(prompt, candidates, pipeline, max_tokens, request_id=None, initial_activation=None, **kwargs):
        nonlocal peer_cache_state
        seed = list(initial_activation or [])
        use_cached = bool(kwargs.get("kv_use_cached_activation", False))
        if use_cached:
            assert peer_cache_state is not None
            base = list(peer_cache_state)
        else:
            base = list(seed)
        calls.append({"prompt": prompt, "seed": seed, "kv_use_cached": use_cached})
        activation = [base[0] + 1.0] if base else [1.0]
        if kwargs.get("kv_store_activation"):
            peer_cache_state = list(activation)
        return ChainResult(
            request_id=request_id or "r1",
            text="result",
            activation=activation,
            traces=[StageTrace(peer_id=pipeline[0].peer_id, latency_ms=1.0, stage_index=0)],
            latency_ms=3.0,
            kv={
                "cache_requested": use_cached,
                "cache_hit": use_cached,
            },
        )

    monkeypatch.setattr(engine, "_run_chain", fake_run_chain)
    monkeypatch.setattr(
        "coordinator.engine.ModelShard.decode_tokens",
        lambda activation, max_tokens, tokenizer_model_id=None: [f"tok{int(activation[0])}"],
    )
    long_prompt = "this prompt is intentionally long for prefill"

    payload_1 = engine.infer_stream(
        prompt=long_prompt,
        max_tokens=1,
        grounding=False,
        session_id="stream-session",
    )
    chunks_1 = list(payload_1["stream"])
    kv_1 = payload_1["streaming"]["kv_data_plane"]

    payload_2 = engine.infer_stream(
        prompt=long_prompt,
        max_tokens=1,
        grounding=False,
        session_id="stream-session",
    )
    chunks_2 = list(payload_2["stream"])
    kv_2 = payload_2["streaming"]["kv_data_plane"]

    assert chunks_1 == ["Tok1", "."]
    assert kv_1["cache_available"] is False
    assert kv_1["external_cache_seeded"] is False
    assert kv_1["cross_peer_relay"] is False
    assert kv_1["cache_used"] is False
    assert kv_1["cache_updated"] is True
    assert kv_1["seeded_rounds"] == 0

    assert chunks_2 == ["Tok2", "."]
    assert kv_2["cache_available"] is True
    assert kv_2["external_cache_seeded"] is False
    assert kv_2["cross_peer_relay"] is False
    assert kv_2["cache_source_peer_id"] == "prefill"
    assert kv_2["cache_target_peer_id"] == "prefill"
    assert kv_2["cache_used"] is True
    assert kv_2["cache_updated"] is True
    assert kv_2["seeded_rounds"] == 0
    assert kv_2["peer_native_cache_enabled"] is True
    assert kv_2["peer_cache_requested"] is True
    assert kv_2["peer_cache_hits"] == 1
    assert kv_2["peer_cache_misses"] == 0

    assert calls[0] == {"prompt": long_prompt, "seed": [], "kv_use_cached": False}
    assert calls[1] == {"prompt": "", "seed": [], "kv_use_cached": True}


def test_infer_stream_relays_kv_cache_across_cold_restart(tmp_path, monkeypatch):
    engine = _engine(tmp_path)

    decode = PeerEndpoint(peer_id="decode", host="127.0.0.1", port=1, model_id=engine.config.default_model, bandwidth_mbps=10)
    prefill_a = PeerEndpoint(peer_id="prefill-a", host="127.0.0.1", port=2, model_id=engine.config.default_model, bandwidth_mbps=900)
    prefill_b = PeerEndpoint(peer_id="prefill-b", host="127.0.0.1", port=3, model_id=engine.config.default_model, bandwidth_mbps=800)

    calls = {"count": 0}

    def _health(peers):
        return [PeerHealth(peer=peer, healthy=True, latency_ms=10.0, load_pct=0.0, daemon_mode="polite") for peer in peers]

    def fake_discover(requested_model, allow_degradation):
        calls["count"] += 1
        if calls["count"] == 1:
            peers = [prefill_a, decode, prefill_b]
        else:
            peers = [decode, prefill_b]
        return (
            _health(peers),
            peers,
            SimpleNamespace(
                requested_model=requested_model,
                served_model=engine.config.default_model,
                degraded=False,
                available=True,
                reason="ok",
                detail="ok",
            ),
            {engine.config.default_model: len(peers)},
        )

    monkeypatch.setattr(engine, "_discover_for_model", fake_discover)
    monkeypatch.setattr(engine, "_select_pipeline", lambda candidates, pipeline_width=None: candidates[:2])

    run_calls: list[dict[str, object]] = []

    def fake_run_chain(prompt, candidates, pipeline, max_tokens, request_id=None, initial_activation=None, **kwargs):
        seed = list(initial_activation or [])
        run_calls.append(
            {
                "prompt": prompt,
                "seed": seed,
                "prefill": pipeline[0].peer_id,
                "kv_use_cached": bool(kwargs.get("kv_use_cached_activation", False)),
            }
        )
        activation = [seed[0] + 1.0] if seed else [1.0]
        return ChainResult(
            request_id=request_id or "r1",
            text="result",
            activation=activation,
            traces=[StageTrace(peer_id=pipeline[0].peer_id, latency_ms=1.0, stage_index=0)],
            latency_ms=3.0,
        )

    monkeypatch.setattr(engine, "_run_chain", fake_run_chain)
    monkeypatch.setattr(
        "coordinator.engine.ModelShard.decode_tokens",
        lambda activation, max_tokens, tokenizer_model_id=None: [f"tok{int(activation[0])}"],
    )
    long_prompt = "this prompt is intentionally long for prefill"

    payload_1 = engine.infer_stream(
        prompt=long_prompt,
        max_tokens=1,
        grounding=False,
        session_id="relay-session",
    )
    _ = list(payload_1["stream"])

    payload_2 = engine.infer_stream(
        prompt=long_prompt,
        max_tokens=1,
        grounding=False,
        session_id="relay-session",
    )
    chunks_2 = list(payload_2["stream"])
    kv_2 = payload_2["streaming"]["kv_data_plane"]

    assert payload_2["bandwidth_policy"]["kv_cold_restart"] is True
    assert chunks_2 == ["Tok2", "."]
    assert kv_2["external_cache_seeded"] is True
    assert kv_2["cross_peer_relay"] is True
    assert kv_2["cache_source_peer_id"] == "prefill-a"
    assert kv_2["cache_target_peer_id"] == "prefill-b"
    assert kv_2["cache_used"] is True
    assert kv_2["seeded_rounds"] == 1

    assert run_calls[0] == {"prompt": long_prompt, "seed": [], "prefill": "prefill-a", "kv_use_cached": False}
    assert run_calls[1] == {"prompt": "", "seed": [1.0], "prefill": "prefill-b", "kv_use_cached": False}


def test_network_status_includes_bandwidth_roles(tmp_path, monkeypatch):
    engine = _engine(tmp_path)

    peers = [
        PeerHealth(peer=PeerEndpoint(peer_id="p1", host="127.0.0.1", port=1, model_id=engine.config.default_model, bandwidth_mbps=700), healthy=True, latency_ms=10.0, load_pct=0.0, daemon_mode="polite"),
        PeerHealth(peer=PeerEndpoint(peer_id="p2", host="127.0.0.1", port=2, model_id=engine.config.default_model, bandwidth_mbps=150), healthy=True, latency_ms=11.0, load_pct=0.0, daemon_mode="polite"),
        PeerHealth(peer=PeerEndpoint(peer_id="p3", host="127.0.0.1", port=3, model_id=engine.config.default_model, bandwidth_mbps=20), healthy=True, latency_ms=12.0, load_pct=0.0, daemon_mode="polite"),
    ]

    monkeypatch.setattr(engine, "_scan_network", lambda model_ids=None: (peers, {engine.config.default_model: 3}))

    status = engine.network_status()
    assert status["bandwidth_roles"][engine.config.default_model] == {
        "prefill_capable": 1,
        "balanced": 1,
        "decode_only": 1,
    }
