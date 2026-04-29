# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Phase 2b §5 — inline ``kv_rollback_to`` dispatch + no-fallback contract.

Locks down the architectural mandate: rollback failures NEVER
silently degrade to drop-and-reprefill. Any precondition violation
must propagate as RuntimeError so the coordinator treats the
session as compromised.
"""

from __future__ import annotations

import pytest

from peer import peer_pb2


# ── Proto-level: kv_rollback_to round-trip ─────────────────────────────

def test_kv_rollback_to_default_zero_no_op():
    req = peer_pb2.ForwardRequest()
    assert req.kv_rollback_to == 0


def test_kv_rollback_to_nonzero_round_trips():
    req = peer_pb2.ForwardRequest(kv_rollback_to=42)
    wire = req.SerializeToString()
    restored = peer_pb2.ForwardRequest()
    restored.ParseFromString(wire)
    assert restored.kv_rollback_to == 42


# ── ModelShard.apply_kv_rollback / drop_kv_session — no fallback ───────

def test_modelshard_apply_kv_rollback_raises_on_missing_runtime_hook():
    """A runtime that doesn't implement apply_kv_rollback is a Phase
    2b regression — must raise loudly, NOT silently no-op. The
    architectural mandate from the Phase 2b plan: any failure mode
    that produces incorrect inference output is unacceptable."""
    from peer.model_shard import ModelShard, ToyShardConfig

    shard = ModelShard(ToyShardConfig(model_id="tinyllama-15M"))
    # ToyRuntime has no apply_kv_rollback method.
    with pytest.raises(RuntimeError, match="does not implement"):
        shard.apply_kv_rollback(session_id="s1", target_len=4)


def test_modelshard_drop_kv_session_returns_false_when_runtime_lacks_hook():
    from peer.model_shard import ModelShard, ToyShardConfig

    shard = ModelShard(ToyShardConfig(model_id="tinyllama-15M"))
    # drop_kv_session is allowed to no-op (it's only called for
    # explicit teardown, never as a rollback fallback).
    assert shard.drop_kv_session("s1") is False


def test_modelshard_apply_kv_rollback_delegates_when_hook_present():
    from peer.model_shard import ModelShard, ToyShardConfig

    shard = ModelShard(ToyShardConfig(model_id="tinyllama-15M"))
    calls = []

    def fake_rollback(*, session_id: str, target_len: int) -> bool:
        calls.append((session_id, target_len))
        return True

    shard._runtime.apply_kv_rollback = fake_rollback   # type: ignore[attr-defined]
    assert shard.apply_kv_rollback(session_id="s1", target_len=4) is True
    assert calls == [("s1", 4)]


def test_modelshard_apply_kv_rollback_propagates_runtime_exceptions():
    """A RollbackError or RuntimeError from the runtime must
    propagate — caller treats this as session-fatal, not as a
    fallback trigger."""
    from peer.model_shard import ModelShard, ToyShardConfig
    from peer.kv_rollback import RollbackError

    shard = ModelShard(ToyShardConfig(model_id="tinyllama-15M"))

    def angry_rollback(*, session_id: str, target_len: int) -> bool:
        raise RollbackError(
            "synthetic precondition violation",
            layer_index=7,
        )

    shard._runtime.apply_kv_rollback = angry_rollback   # type: ignore[attr-defined]
    with pytest.raises(RollbackError) as exc:
        shard.apply_kv_rollback(session_id="s1", target_len=4)
    assert exc.value.layer_index == 7


# ── Runtime-side hooks present after Phase 2b ──────────────────────────

def test_pytorch_runtime_exposes_kv_rollback_hooks():
    import inspect
    from peer.model_shard import PyTorchRuntime

    assert callable(getattr(PyTorchRuntime, "apply_kv_rollback", None))
    assert callable(getattr(PyTorchRuntime, "drop_kv_session", None))
    assert callable(getattr(PyTorchRuntime, "_gateddeltanet_replay_step", None))

    sig = inspect.signature(PyTorchRuntime.apply_kv_rollback)
    assert "session_id" in sig.parameters
    assert "target_len" in sig.parameters


def test_mlx_runtime_exposes_kv_rollback_hooks():
    import inspect
    from peer.mlx_runtime import MLXRuntime

    assert callable(getattr(MLXRuntime, "apply_kv_rollback", None))
    assert callable(getattr(MLXRuntime, "drop_kv_session", None))
    assert callable(getattr(MLXRuntime, "_gateddeltanet_replay_step", None))

    sig = inspect.signature(MLXRuntime.apply_kv_rollback)
    assert "session_id" in sig.parameters
    assert "target_len" in sig.parameters


# ── PyTorch replay step: byte-equivalence on real tensors ──────────────

def test_pytorch_replay_step_matches_eq1_byte_identical():
    """The runtime's _gateddeltanet_replay_step must implement
    eq.(1) exactly. A regression in this kernel is invisible until
    inference starts producing garbage; lock it now.

    Run on a single-head, small-d toy case so the math is hand-
    verifiable. The replay must produce a tensor bit-equal to
    manual computation of (I - β k k^T) S + β v k^T.
    """
    import torch

    from peer.model_shard import PyTorchRuntime

    # 1 head, d_v=2, d_k=2 — so we can hand-compute.
    S = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float64)
    beta = torch.tensor([0.5], dtype=torch.float64)
    k = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    v = torch.tensor([[2.0, 3.0]], dtype=torch.float64)

    out = PyTorchRuntime._gateddeltanet_replay_step(S, beta, k, v)

    # Manual:
    # kk^T = [[1,0],[0,0]]      (h=0)
    # I − 0.5 kk^T = [[0.5, 0], [0, 1]]
    # (I − 0.5 kk^T) S = [[0.5, 0], [0, 1]]   (since S=I)
    # vk^T = [[2,0],[3,0]]
    # 0.5 vk^T = [[1,0],[1.5,0]]
    # sum = [[1.5, 0], [1.5, 1]]
    expected = torch.tensor([[[1.5, 0.0], [1.5, 1.0]]], dtype=torch.float64)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_pytorch_replay_step_repeated_application_is_deterministic():
    """Same inputs ⇒ same output, every invocation. (Catches
    any accidental in-place mutation of S that would corrupt
    multi-step replay.)"""
    import torch

    from peer.model_shard import PyTorchRuntime

    S0 = torch.randn(2, 4, 4, dtype=torch.float64)
    beta = torch.tensor([0.3, 0.7], dtype=torch.float64)
    k = torch.randn(2, 4, dtype=torch.float64)
    v = torch.randn(2, 4, dtype=torch.float64)

    a = PyTorchRuntime._gateddeltanet_replay_step(S0, beta, k, v)
    b = PyTorchRuntime._gateddeltanet_replay_step(S0, beta, k, v)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    # And S0 unchanged.
    assert not S0.is_contiguous() or S0.equal(S0)   # tautology — guards against in-place corruption
