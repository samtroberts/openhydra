"""Phase 0 tests — warmup, timeout_ms default, serve() signature."""
from __future__ import annotations

import inspect
import logging
import types as _types

import pytest

from coordinator.engine import EngineConfig
from peer.model_shard import ModelShard, PyTorchRuntime, ToyShardConfig


# ─── ToyShardConfig — warmup field ──────────────────────────────────────────

def test_toyshardconfig_warmup_defaults_false():
    """runtime_warmup_on_start should be False out of the box."""
    config = ToyShardConfig()
    assert config.runtime_warmup_on_start is False


def test_toyshardconfig_warmup_can_be_enabled():
    """runtime_warmup_on_start=True should be stored correctly."""
    config = ToyShardConfig(runtime_warmup_on_start=True)
    assert config.runtime_warmup_on_start is True


def test_toyshardconfig_warmup_preserved_through_default_fields():
    """Ensure warmup field doesn't collide with other config fields."""
    config = ToyShardConfig(
        model_id="test-model",
        shard_index=1,
        total_shards=4,
        runtime_warmup_on_start=True,
        runtime_kv_cache_max_entries=512,
    )
    assert config.runtime_warmup_on_start is True
    assert config.shard_index == 1
    assert config.total_shards == 4
    assert config.runtime_kv_cache_max_entries == 512


# ─── PyTorchRuntime — _warmup() method exists ───────────────────────────────

def test_pytorch_runtime_has_warmup_method():
    """_warmup() must exist on PyTorchRuntime (not just as a config attribute)."""
    assert callable(getattr(PyTorchRuntime, "_warmup", None))


# ─── PyTorchRuntime — _warmup() error handling ──────────────────────────────

class _NoGradCtx:
    """Minimal no_grad() context manager stand-in."""
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _make_minimal_pytorch_instance(*, tensor_side_effect=None) -> PyTorchRuntime:
    """Build a bare PyTorchRuntime with enough state to call _warmup()."""

    def _fail(*a, **kw):
        if tensor_side_effect is not None:
            raise tensor_side_effect
        raise RuntimeError("forced_test_failure")

    fake_torch = _types.SimpleNamespace(
        no_grad=_NoGradCtx,   # called as no_grad() → returns ctx-manager
        tensor=_fail,
        long=0,
    )

    instance = object.__new__(PyTorchRuntime)
    object.__setattr__(instance, "_torch",     fake_torch)
    object.__setattr__(instance, "_device",    "cpu")
    object.__setattr__(instance, "model_name", "test-model")
    object.__setattr__(instance, "_tokenizer",
                       _types.SimpleNamespace(eos_token_id=1))
    object.__setattr__(instance, "_lm_head",   None)
    return instance


def test_pytorch_runtime_warmup_catches_forward_error(caplog):
    """_warmup() must catch failures and log a warning rather than raising."""
    instance = _make_minimal_pytorch_instance(
        tensor_side_effect=RuntimeError("no_gpu_available")
    )
    with caplog.at_level(logging.WARNING):
        instance._warmup()   # must NOT raise

    assert any("runtime_warmup_failed" in r.message for r in caplog.records)


def test_pytorch_runtime_warmup_catches_attribute_error(caplog):
    """_warmup() must survive AttributeError on missing instance state."""
    instance = object.__new__(PyTorchRuntime)
    # Deliberately omit all attributes so any attribute access raises.
    with caplog.at_level(logging.WARNING):
        instance._warmup()   # must NOT raise

    assert any("runtime_warmup_failed" in r.message for r in caplog.records)


# ─── ToyRuntime — warmup flag is inert ──────────────────────────────────────

def test_toy_runtime_unaffected_by_warmup_flag():
    """warmup_on_start=True on a toy_auto shard must not break forward()."""
    shard = ModelShard(ToyShardConfig(
        model_id="openhydra-toy-345m",
        runtime_backend="toy_auto",
        runtime_warmup_on_start=True,
    ))
    result = shard.forward("hello hydra", [], 4)
    assert result, "forward() must still return a non-empty activation"


def test_toy_runtime_forward_identical_with_and_without_warmup_flag():
    """ToyRuntime output must be deterministic regardless of warmup flag."""
    shard_no_warmup  = ModelShard(ToyShardConfig(runtime_warmup_on_start=False))
    shard_with_warmup = ModelShard(ToyShardConfig(runtime_warmup_on_start=True))
    r1 = shard_no_warmup.forward("hello", [], 8)
    r2 = shard_with_warmup.forward("hello", [], 8)
    assert r1 == r2, "warmup flag must not affect ToyRuntime output"


# ─── EngineConfig — timeout_ms default ──────────────────────────────────────

def test_engine_config_timeout_ms_default_is_5000():
    """timeout_ms default must be raised from 500 → 5000 (Phase 0)."""
    config = EngineConfig()
    assert config.timeout_ms == 5000, (
        f"Expected timeout_ms=5000 but got {config.timeout_ms}. "
        "Phase 0 requires the default to be raised from 500 to 5000 ms."
    )


def test_engine_config_timeout_ms_is_overridable():
    """Explicit timeout_ms values must still be respected."""
    assert EngineConfig(timeout_ms=500).timeout_ms   == 500
    assert EngineConfig(timeout_ms=60_000).timeout_ms == 60_000
    assert EngineConfig(timeout_ms=1).timeout_ms      == 1


# ─── serve() / PeerService — warmup_on_start parameter ──────────────────────

def test_serve_accepts_warmup_on_start_param():
    """serve() must have a warmup_on_start keyword argument defaulting to False."""
    from peer.server import serve
    sig = inspect.signature(serve)
    assert "warmup_on_start" in sig.parameters, \
        "serve() is missing the warmup_on_start parameter"
    assert sig.parameters["warmup_on_start"].default is False


def test_peer_service_accepts_warmup_on_start_param():
    """PeerService.__init__ must accept warmup_on_start, defaulting to False."""
    from peer.server import PeerService
    sig = inspect.signature(PeerService.__init__)
    assert "warmup_on_start" in sig.parameters, \
        "PeerService.__init__ is missing the warmup_on_start parameter"
    assert sig.parameters["warmup_on_start"].default is False
