"""Tests for peer/mlx_runtime.py — MLX inference backend (Phase 1).

All tests that require MLX are guarded with ``pytest.importorskip("mlx.core")``
so the suite remains green on machines without mlx installed (e.g. Linux CI).

Structure
---------
Group A  — module-level checks (no MLX needed)
Group B  — interface / smoke tests (skip if mlx absent)
Group C  — DLPack bridge tests (skip if mlx absent)
Group D  — encode_prompt tests (skip if mlx absent)
Group E  — forward() / sampler tests (skip if mlx absent)
Group F  — runtime_profile tests (skip if mlx absent)
Group G  — Phase 3 stubs raise NotImplementedError
Group H  — ModelShard integration: backend="mlx" routes correctly
Group I  — Real text generation quality
Group J  — _MlxWatchdog timeout / busy-guard tests
"""
from __future__ import annotations

import time
import types
import logging
import threading
import pytest

# ── helpers ───────────────────────────────────────────────────────────────────

def _mlx_available() -> bool:
    try:
        import mlx.core  # noqa: F401
        import mlx_lm    # noqa: F401
        return True
    except ImportError:
        return False


MLX_MARK = pytest.mark.skipif(
    not _mlx_available(),
    reason="mlx / mlx-lm not installed",
)


def _make_minimal_config(**kwargs):
    """Return a ToyShardConfig with overrides."""
    from peer.model_shard import ToyShardConfig
    defaults = dict(
        runtime_model_id="Qwen/Qwen3.5-0.8B",
        runtime_warmup_on_start=False,
    )
    defaults.update(kwargs)
    return ToyShardConfig(**defaults)


# ═══════════════════════════════════════════════════════════════════════════════
# Group A — module-level checks (no MLX import needed)
# ═══════════════════════════════════════════════════════════════════════════════

def test_mlx_runtime_module_importable():
    """peer.mlx_runtime must be importable without triggering mlx import."""
    import importlib
    spec = importlib.util.find_spec("peer.mlx_runtime")
    assert spec is not None, "peer.mlx_runtime module not found"


def test_mlxruntime_class_exists():
    """MLXRuntime must be exported from peer.mlx_runtime.__all__."""
    import peer.mlx_runtime as m
    assert "MLXRuntime" in m.__all__


def test_mlxruntime_class_is_class():
    """MLXRuntime must be a class (not a function or constant)."""
    from peer.mlx_runtime import MLXRuntime
    import inspect
    assert inspect.isclass(MLXRuntime)


def test_mlxruntime_has_required_methods():
    """MLXRuntime must declare the same public interface as PyTorchRuntime."""
    from peer.mlx_runtime import MLXRuntime
    required = [
        "_warmup",
        "encode_prompt",
        "forward",
        "runtime_profile",
        "_torch_to_mx",
        "_mx_to_torch",
        "_activation_to_hidden",
        "_hidden_to_payload",
    ]
    for name in required:
        assert callable(getattr(MLXRuntime, name, None)), \
            f"MLXRuntime.{name} is missing or not callable"


# ═══════════════════════════════════════════════════════════════════════════════
# Group B — interface / smoke tests (requires mlx)
# ═══════════════════════════════════════════════════════════════════════════════

@MLX_MARK
def test_mlxruntime_instantiates():
    """MLXRuntime.__init__ must complete without error on default model."""
    from peer.mlx_runtime import MLXRuntime
    rt = MLXRuntime(_make_minimal_config())
    assert rt is not None


@MLX_MARK
def test_mlxruntime_model_name_stored():
    """model_name must reflect the configured model ID."""
    from peer.mlx_runtime import MLXRuntime
    cfg = _make_minimal_config(runtime_model_id="Qwen/Qwen3.5-0.8B")
    rt = MLXRuntime(cfg)
    assert rt.model_name == "Qwen/Qwen3.5-0.8B"


@MLX_MARK
def test_mlxruntime_warmup_runs_without_error(caplog):
    """_warmup() must run a 2-token dummy pass without raising."""
    from peer.mlx_runtime import MLXRuntime
    rt = MLXRuntime(_make_minimal_config())
    with caplog.at_level(logging.INFO):
        rt._warmup()   # must not raise
    assert any("warmup" in r.message.lower() for r in caplog.records)


@MLX_MARK
def test_mlxruntime_warmup_on_start(caplog):
    """runtime_warmup_on_start=True must trigger _warmup during __init__."""
    from peer.mlx_runtime import MLXRuntime
    with caplog.at_level(logging.INFO):
        MLXRuntime(_make_minimal_config(runtime_warmup_on_start=True))
    assert any("warmup" in r.message.lower() for r in caplog.records)


# ═══════════════════════════════════════════════════════════════════════════════
# Group C — DLPack bridge tests
# ═══════════════════════════════════════════════════════════════════════════════

@MLX_MARK
def test_torch_to_mx_basic_roundtrip():
    """_torch_to_mx must convert a float32 CPU tensor without error."""
    import torch
    from peer.mlx_runtime import MLXRuntime
    rt = MLXRuntime(_make_minimal_config())
    t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    mx_arr = rt._torch_to_mx(t)
    assert mx_arr is not None
    assert list(mx_arr.tolist()) == [1.0, 2.0, 3.0]


@MLX_MARK
def test_mx_to_torch_basic_roundtrip():
    """_mx_to_torch must convert an MLX array back to a PyTorch tensor."""
    import torch
    import mlx.core as mx
    from peer.mlx_runtime import MLXRuntime
    rt = MLXRuntime(_make_minimal_config())
    arr = mx.array([4.0, 5.0, 6.0])
    mx.eval(arr)
    t = rt._mx_to_torch(arr)
    assert isinstance(t, torch.Tensor)
    assert t.tolist() == [4.0, 5.0, 6.0]


@MLX_MARK
def test_dlpack_roundtrip_zero_copy():
    """Full round-trip torch → mlx → torch must preserve values."""
    import torch
    from peer.mlx_runtime import MLXRuntime
    rt = MLXRuntime(_make_minimal_config())
    original = torch.arange(10, dtype=torch.float32)
    mx_arr = rt._torch_to_mx(original)
    recovered = rt._mx_to_torch(mx_arr)
    assert torch.allclose(original, recovered.float())


@MLX_MARK
def test_torch_to_mx_2d_tensor():
    """_torch_to_mx must handle 2-D tensors (e.g. tokenizer output)."""
    import torch
    from peer.mlx_runtime import MLXRuntime
    rt = MLXRuntime(_make_minimal_config())
    t = torch.zeros(2, 8, dtype=torch.float32)
    mx_arr = rt._torch_to_mx(t)
    assert mx_arr.shape == (2, 8)


# ═══════════════════════════════════════════════════════════════════════════════
# Group D — encode_prompt tests
# ═══════════════════════════════════════════════════════════════════════════════

@MLX_MARK
def test_encode_prompt_returns_list():
    """encode_prompt must return a non-empty list of floats."""
    from peer.mlx_runtime import MLXRuntime
    rt = MLXRuntime(_make_minimal_config())
    result = rt.encode_prompt("hello world", max_tokens=16)
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(x, float) for x in result)


@MLX_MARK
def test_encode_prompt_empty_string_returns_list():
    """encode_prompt('', ...) must return a list (possibly with BOS only)."""
    from peer.mlx_runtime import MLXRuntime
    rt = MLXRuntime(_make_minimal_config())
    result = rt.encode_prompt("", max_tokens=4)
    assert isinstance(result, list)


@MLX_MARK
def test_encode_prompt_longer_prompt_more_tokens():
    """Longer prompts should produce more token IDs."""
    from peer.mlx_runtime import MLXRuntime
    rt = MLXRuntime(_make_minimal_config())
    short = rt.encode_prompt("hi", max_tokens=32)
    long_ = rt.encode_prompt(
        "Explain the full history of machine learning from Rosenblatt to transformers.",
        max_tokens=32,
    )
    assert len(long_) >= len(short)


# ═══════════════════════════════════════════════════════════════════════════════
# Group E — forward() / sampler tests
# ═══════════════════════════════════════════════════════════════════════════════

@MLX_MARK
def test_forward_single_stage_returns_token_list():
    """forward() with total_stages=1 must return a non-empty list of floats."""
    from peer.mlx_runtime import MLXRuntime
    rt = MLXRuntime(_make_minimal_config())
    result = rt.forward("hi", activation=[], max_tokens=4, total_stages=1)
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(x, float) for x in result)


@MLX_MARK
def test_forward_multi_stage_raises_not_implemented():
    """forward() with total_stages > 1 must raise NotImplementedError (Phase 3)."""
    from peer.mlx_runtime import MLXRuntime
    rt = MLXRuntime(_make_minimal_config())
    with pytest.raises(NotImplementedError, match="Phase 3"):
        rt.forward("hi", activation=[], max_tokens=4, total_stages=2)


@MLX_MARK
def test_forward_greedy_decoding():
    """decode_do_sample=False must produce a deterministic (greedy) output."""
    from peer.mlx_runtime import MLXRuntime
    rt = MLXRuntime(_make_minimal_config())
    r1 = rt.forward("The capital of France is", activation=[], max_tokens=4,
                    decode_do_sample=False)
    r2 = rt.forward("The capital of France is", activation=[], max_tokens=4,
                    decode_do_sample=False)
    # Greedy decoding is deterministic.
    assert r1 == r2


@MLX_MARK
def test_forward_max_tokens_respected():
    """forward() must return at most max_tokens tokens."""
    from peer.mlx_runtime import MLXRuntime
    rt = MLXRuntime(_make_minimal_config())
    result = rt.forward("hello", activation=[], max_tokens=8)
    assert len(result) <= 8


@MLX_MARK
def test_forward_with_temperature():
    """decode_temperature kwarg must be accepted without error."""
    from peer.mlx_runtime import MLXRuntime
    rt = MLXRuntime(_make_minimal_config())
    result = rt.forward("hello", activation=[], max_tokens=4,
                        decode_do_sample=True, decode_temperature=0.7)
    assert isinstance(result, list)


# ═══════════════════════════════════════════════════════════════════════════════
# Group F — runtime_profile tests
# ═══════════════════════════════════════════════════════════════════════════════

@MLX_MARK
def test_runtime_profile_returns_dict():
    """runtime_profile() must return a dict."""
    from peer.mlx_runtime import MLXRuntime
    rt = MLXRuntime(_make_minimal_config())
    profile = rt.runtime_profile()
    assert isinstance(profile, dict)


@MLX_MARK
def test_runtime_profile_backend_is_mlx():
    """runtime_profile()['backend'] must be 'mlx'."""
    from peer.mlx_runtime import MLXRuntime
    rt = MLXRuntime(_make_minimal_config())
    assert rt.runtime_profile()["backend"] == "mlx"


@MLX_MARK
def test_runtime_profile_target_is_metal():
    """runtime_profile()['target'] must be 'metal'."""
    from peer.mlx_runtime import MLXRuntime
    rt = MLXRuntime(_make_minimal_config())
    assert rt.runtime_profile()["target"] == "metal"


@MLX_MARK
def test_runtime_profile_required_keys():
    """runtime_profile() must include all coordinator-expected keys."""
    from peer.mlx_runtime import MLXRuntime
    rt = MLXRuntime(_make_minimal_config())
    profile = rt.runtime_profile()
    required_keys = {
        "backend", "target", "quantization_mode", "quantization_bits",
        "gpu_available", "param_count", "estimated_memory_mb",
        "estimated_tokens_per_sec", "layer_start", "layer_end",
        "total_layers", "runtime_model_id",
    }
    missing = required_keys - set(profile.keys())
    assert not missing, f"runtime_profile() is missing keys: {missing}"


@MLX_MARK
def test_runtime_profile_param_count_positive():
    """param_count must be a positive integer (model was loaded)."""
    from peer.mlx_runtime import MLXRuntime
    rt = MLXRuntime(_make_minimal_config())
    assert rt.runtime_profile()["param_count"] > 0


@MLX_MARK
def test_runtime_profile_is_copy():
    """runtime_profile() must return a fresh copy each time."""
    from peer.mlx_runtime import MLXRuntime
    rt = MLXRuntime(_make_minimal_config())
    p1 = rt.runtime_profile()
    p1["backend"] = "mutated"
    p2 = rt.runtime_profile()
    assert p2["backend"] == "mlx", "runtime_profile must return a copy, not a reference"


# ═══════════════════════════════════════════════════════════════════════════════
# Group G — Phase 3 stubs
# ═══════════════════════════════════════════════════════════════════════════════

@MLX_MARK
def test_activation_to_hidden_raises():
    """_activation_to_hidden() must raise NotImplementedError (Phase 3 stub)."""
    from peer.mlx_runtime import MLXRuntime
    rt = MLXRuntime(_make_minimal_config())
    with pytest.raises(NotImplementedError, match="Phase 3"):
        rt._activation_to_hidden([1.0, 2.0, 3.0])


@MLX_MARK
def test_hidden_to_payload_raises():
    """_hidden_to_payload() must raise NotImplementedError (Phase 3 stub)."""
    from peer.mlx_runtime import MLXRuntime
    rt = MLXRuntime(_make_minimal_config())
    with pytest.raises(NotImplementedError, match="Phase 3"):
        rt._hidden_to_payload(None)


# ═══════════════════════════════════════════════════════════════════════════════
# Group H — ModelShard integration
# ═══════════════════════════════════════════════════════════════════════════════

def test_modelshard_mlx_choice_routes_to_mlxruntime():
    """ModelShard(runtime_backend='mlx') must instantiate MLXRuntime.

    If mlx is not installed this test is skipped.  If it IS installed
    we verify that the correct runtime class is used.
    """
    if not _mlx_available():
        pytest.skip("mlx / mlx-lm not installed")
    from peer.model_shard import ModelShard, ToyShardConfig
    from peer.mlx_runtime import MLXRuntime
    shard = ModelShard(ToyShardConfig(runtime_backend="mlx"))
    assert isinstance(shard._runtime, MLXRuntime)


def test_modelshard_mlx_profile_backend_key():
    """ModelShard with mlx backend must report backend='mlx' in profile."""
    if not _mlx_available():
        pytest.skip("mlx / mlx-lm not installed")
    from peer.model_shard import ModelShard, ToyShardConfig
    shard = ModelShard(ToyShardConfig(runtime_backend="mlx"))
    profile = shard.runtime_profile()
    assert profile["backend"] == "mlx"


def test_modelshard_toy_auto_unaffected_by_mlx_module():
    """Adding MLX branch must not break the toy_auto path."""
    from peer.model_shard import ModelShard, ToyShardConfig
    shard = ModelShard(ToyShardConfig(runtime_backend="toy_auto"))
    result = shard.forward("hello", [], 4)
    assert result  # non-empty activation


# ═══════════════════════════════════════════════════════════════════════════════
# Group I — Real text generation quality ("Five facts about Bangalore")
# ═══════════════════════════════════════════════════════════════════════════════

@MLX_MARK
def test_mlx_bangalore_generation():
    """MLX must generate coherent text about Bangalore."""
    from peer.mlx_runtime import MLXRuntime
    rt = MLXRuntime(_make_minimal_config())
    tokens = rt.forward(
        "Five facts about Bangalore",
        activation=[],
        max_tokens=128,
        decode_do_sample=False,
    )
    # Decode tokens back to text for validation
    text = rt._tokenizer.decode([int(t) for t in tokens])
    assert len(text) > 20, f"Expected coherent response, got: {text!r}"
    assert any(c.isalpha() for c in text)
    print(f"\n{'=' * 60}")
    print(f"  MLX Zero-Copy Bridge — Bangalore Generation")
    print(f"{'=' * 60}")
    print(f"  Tokens: {len(tokens)}")
    print(f"  Text:\n{text}")
    print(f"{'=' * 60}")


# ═══════════════════════════════════════════════════════════════════════════════
# Group J — _MlxWatchdog timeout / busy-guard tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_watchdog_class_exists():
    """_MlxWatchdog must be importable from peer.mlx_runtime."""
    from peer.mlx_runtime import _MlxWatchdog
    import inspect
    assert inspect.isclass(_MlxWatchdog)


def test_watchdog_normal_completion():
    """Watchdog must return the result of a fast function."""
    from peer.mlx_runtime import _MlxWatchdog
    wd = _MlxWatchdog(default_timeout_s=5.0)
    result = wd.run(lambda: 42)
    assert result == 42
    assert not wd.is_busy


def test_watchdog_timeout_raises():
    """Watchdog must raise TimeoutError when function exceeds deadline."""
    from peer.mlx_runtime import _MlxWatchdog
    wd = _MlxWatchdog(default_timeout_s=0.2)

    def slow_fn():
        time.sleep(5.0)
        return "should not reach"

    with pytest.raises(TimeoutError, match="mlx_watchdog_timeout"):
        wd.run(slow_fn)
    # After timeout, watchdog remains busy (zombie in-flight).
    assert wd.is_busy


def test_watchdog_busy_guard_rejects_concurrent():
    """Watchdog must reject a second submission while a zombie is in-flight."""
    from peer.mlx_runtime import _MlxWatchdog
    wd = _MlxWatchdog(default_timeout_s=0.2)

    # Create a zombie by timing out.
    def slow_fn():
        time.sleep(5.0)

    with pytest.raises(TimeoutError):
        wd.run(slow_fn)
    assert wd.is_busy

    # Second call must fail immediately with RuntimeError, not TimeoutError.
    with pytest.raises(RuntimeError, match="mlx_watchdog_busy"):
        wd.run(lambda: 1)


def test_watchdog_clears_busy_on_success():
    """Watchdog must clear busy flag after successful completion."""
    from peer.mlx_runtime import _MlxWatchdog
    wd = _MlxWatchdog(default_timeout_s=5.0)
    assert not wd.is_busy
    wd.run(lambda: "ok")
    assert not wd.is_busy
    # Should be able to run again immediately.
    assert wd.run(lambda: 99) == 99


def test_watchdog_clears_busy_on_exception():
    """Watchdog must clear busy flag when function raises a non-timeout error."""
    from peer.mlx_runtime import _MlxWatchdog
    wd = _MlxWatchdog(default_timeout_s=5.0)

    with pytest.raises(ValueError, match="boom"):
        wd.run(lambda: (_ for _ in ()).throw(ValueError("boom")))

    # GPU is free after a normal exception (not a zombie).
    assert not wd.is_busy
    # Should be usable again.
    assert wd.run(lambda: "recovered") == "recovered"


def test_watchdog_custom_timeout_override():
    """Per-call timeout_s must override the default."""
    from peer.mlx_runtime import _MlxWatchdog
    wd = _MlxWatchdog(default_timeout_s=60.0)  # very long default

    def slow_fn():
        time.sleep(5.0)

    with pytest.raises(TimeoutError, match="0.2s exceeded"):
        wd.run(slow_fn, timeout_s=0.2)


def test_watchdog_config_flows_from_toyshard():
    """runtime_mlx_eval_timeout_s must propagate from ToyShardConfig."""
    from peer.model_shard import ToyShardConfig
    cfg = ToyShardConfig(runtime_mlx_eval_timeout_s=15.0)
    assert cfg.runtime_mlx_eval_timeout_s == 15.0

    # Default raised to 120.0 to support 8 GB machines under memory pressure.
    cfg_default = ToyShardConfig()
    assert cfg_default.runtime_mlx_eval_timeout_s == 120.0


# ═══════════════════════════════════════════════════════════════════════════════
# Group K — KV Prefix Caching (RadixKVCache + MLXRuntime)
# ═══════════════════════════════════════════════════════════════════════════════

def test_radix_cache_config_disabled_by_default():
    """RadixKVCache must NOT be created when config flag is off (default)."""
    if not _mlx_available():
        pytest.skip("mlx / mlx-lm not installed")
    from peer.mlx_runtime import MLXRuntime
    rt = MLXRuntime(_make_minimal_config())
    assert rt._radix_cache is None


@MLX_MARK
def test_radix_cache_created_when_enabled():
    """RadixKVCache must be created when runtime_kv_radix_cache_enabled=True."""
    from peer.mlx_runtime import MLXRuntime
    cfg = _make_minimal_config(
        runtime_kv_radix_cache_enabled=True,
        runtime_kv_radix_cache_max_entries=64,
        runtime_kv_radix_cache_min_prefix_len=4,
    )
    rt = MLXRuntime(cfg)
    assert rt._radix_cache is not None
    stats = rt._radix_cache.stats()
    assert stats["radix_max_entries"] == 64
    assert stats["radix_entries"] == 0


@MLX_MARK
def test_radix_cache_prompt_cache_supported():
    """MLXRuntime must detect prompt_cache support in mlx_lm."""
    from peer.mlx_runtime import MLXRuntime
    cfg = _make_minimal_config(runtime_kv_radix_cache_enabled=True)
    rt = MLXRuntime(cfg)
    # If radix_cache was created, prompt_cache must be supported.
    if rt._radix_cache is not None:
        assert rt._prompt_cache_supported is True


@MLX_MARK
def test_radix_cache_hit_on_repeated_prefix():
    """Second call with same prompt must produce a cache entry."""
    from peer.mlx_runtime import MLXRuntime
    cfg = _make_minimal_config(
        runtime_kv_radix_cache_enabled=True,
        runtime_kv_radix_cache_min_prefix_len=2,
    )
    rt = MLXRuntime(cfg)
    if rt._radix_cache is None:
        pytest.skip("RadixKVCache not available")

    # First call: populates cache.
    rt.forward("The capital of France is", activation=[], max_tokens=4,
               decode_do_sample=False)
    assert rt._radix_cache.stats()["radix_entries"] >= 1

    # Second call with same prompt: should have a cache entry.
    cached_kv, prefix_len = rt._radix_cache.lookup(
        tuple(rt._tokenizer.encode("The capital of France is"))
    )
    assert prefix_len > 0
    assert cached_kv is not None


@MLX_MARK
def test_radix_cache_miss_on_novel_prompt():
    """A completely new prompt must miss the cache."""
    from peer.mlx_runtime import MLXRuntime
    cfg = _make_minimal_config(
        runtime_kv_radix_cache_enabled=True,
        runtime_kv_radix_cache_min_prefix_len=2,
    )
    rt = MLXRuntime(cfg)
    if rt._radix_cache is None:
        pytest.skip("RadixKVCache not available")

    # Populate with one prompt.
    rt.forward("The capital of France is", activation=[], max_tokens=4,
               decode_do_sample=False)

    # Different prompt must miss.
    cached_kv, prefix_len = rt._radix_cache.lookup(
        tuple(rt._tokenizer.encode("Explain quantum computing"))
    )
    assert prefix_len == 0
    assert cached_kv is None


@MLX_MARK
def test_radix_cache_lru_eviction():
    """Cache must evict oldest entry when max_entries is exceeded."""
    from peer.mlx_runtime import MLXRuntime
    cfg = _make_minimal_config(
        runtime_kv_radix_cache_enabled=True,
        runtime_kv_radix_cache_max_entries=2,
        runtime_kv_radix_cache_min_prefix_len=2,
    )
    rt = MLXRuntime(cfg)
    if rt._radix_cache is None:
        pytest.skip("RadixKVCache not available")

    # Fill cache with 2 prompts.
    rt.forward("The history of Rome is long", activation=[], max_tokens=4,
               decode_do_sample=False)
    rt.forward("The future of AI is bright", activation=[], max_tokens=4,
               decode_do_sample=False)
    assert rt._radix_cache.stats()["radix_entries"] == 2

    # Third prompt should evict the oldest (Rome).
    rt.forward("Python is a programming language", activation=[], max_tokens=4,
               decode_do_sample=False)
    # At most 2 entries remain.
    assert rt._radix_cache.stats()["radix_entries"] <= 2


@MLX_MARK
def test_radix_cache_forward_still_works_with_cache():
    """forward() must produce valid output when cache is enabled."""
    from peer.mlx_runtime import MLXRuntime
    cfg = _make_minimal_config(
        runtime_kv_radix_cache_enabled=True,
        runtime_kv_radix_cache_min_prefix_len=2,
    )
    rt = MLXRuntime(cfg)
    result = rt.forward("Hello world", activation=[], max_tokens=4,
                        decode_do_sample=False)
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(x, float) for x in result)


@MLX_MARK
def test_radix_cache_batch_bypass():
    """forward_batch() must NOT use or corrupt the RadixKVCache."""
    from peer.mlx_runtime import MLXRuntime
    cfg = _make_minimal_config(
        runtime_kv_radix_cache_enabled=True,
        runtime_kv_radix_cache_min_prefix_len=2,
    )
    rt = MLXRuntime(cfg)
    if rt._radix_cache is None:
        pytest.skip("RadixKVCache not available")

    # Batch with 2 items — cache should remain empty after batch.
    class _FakeItem:
        def __init__(self, prompt, max_tokens=4, total_stages=1):
            self.prompt = prompt
            self.max_tokens = max_tokens
            self.total_stages = total_stages

    results = rt.forward_batch([
        _FakeItem("Hello world"),
        _FakeItem("Goodbye world"),
    ])
    assert len(results) == 2
    # Batch path does NOT interact with RadixKVCache.
    assert rt._radix_cache.stats()["radix_entries"] == 0
