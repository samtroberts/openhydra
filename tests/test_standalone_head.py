# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Tests for the Phase 6 standalone head loader (true Petals topology).

Covers:
    * StandaloneHead surface + duck-typing as a HeadSampler runtime
    * Backend dispatch (MLX vs PyTorch paths)
    * Decode-kwargs forwarding to the sampler
    * Tied vs untied lm_head dispatch
    * apply_final_head error handling (hidden-size mismatch, missing lm_head)
    * Backend resolution (_resolve_backend: auto/mlx/pytorch)
    * CLI guardrails on --no-local-peer
    * HeadSampler integration — borrows transparently from StandaloneHead

The live load of an actual MLX or PyTorch model is covered by the
pre-commit sanity checks we run on Mac + Lightning; here we stick to
mocks so the tests stay fast and dependency-minimal.
"""

from __future__ import annotations

import inspect
import types
from unittest.mock import MagicMock

import pytest


# ── _resolve_backend ─────────────────────────────────────────────────────

def test_resolve_backend_auto_prefers_mlx_when_available(monkeypatch):
    from coordinator import standalone_head as sh

    monkeypatch.setattr(sh, "_mlx_available", lambda: True)
    monkeypatch.setattr(sh, "_pytorch_available", lambda: True)
    assert sh._resolve_backend("auto") == "mlx"


def test_resolve_backend_auto_falls_back_to_pytorch(monkeypatch):
    from coordinator import standalone_head as sh

    monkeypatch.setattr(sh, "_mlx_available", lambda: False)
    monkeypatch.setattr(sh, "_pytorch_available", lambda: True)
    assert sh._resolve_backend("auto") == "pytorch"


def test_resolve_backend_auto_raises_when_neither_available(monkeypatch):
    from coordinator import standalone_head as sh

    monkeypatch.setattr(sh, "_mlx_available", lambda: False)
    monkeypatch.setattr(sh, "_pytorch_available", lambda: False)
    with pytest.raises(RuntimeError, match="neither mlx_lm nor"):
        sh._resolve_backend("auto")


def test_resolve_backend_mlx_raises_when_mlx_missing(monkeypatch):
    from coordinator import standalone_head as sh

    monkeypatch.setattr(sh, "_mlx_available", lambda: False)
    with pytest.raises(RuntimeError, match="mlx_lm is not installed"):
        sh._resolve_backend("mlx")


def test_resolve_backend_pytorch_raises_when_pytorch_missing(monkeypatch):
    from coordinator import standalone_head as sh

    monkeypatch.setattr(sh, "_pytorch_available", lambda: False)
    with pytest.raises(RuntimeError, match="torch\\+transformers are not"):
        sh._resolve_backend("pytorch")


def test_resolve_backend_unknown_raises():
    from coordinator import standalone_head as sh

    with pytest.raises(RuntimeError, match="unknown backend"):
        sh._resolve_backend("bogus")


# ── StandaloneHead surface ──────────────────────────────────────────────

def _build_mock_head(*, backend: str = "mlx", tied: bool = True) -> "StandaloneHead":
    """Helper to build a StandaloneHead with mocked module references.

    Mocks stand in for the real MLX/PyTorch weight modules so the tests
    exercise dispatch logic without needing the heavy runtime deps.
    """
    from coordinator.standalone_head import StandaloneHead

    return StandaloneHead(
        backend=backend,
        hf_model_id="mock/model",
        norm_module=MagicMock(side_effect=lambda h: h),
        embed_tokens_module=MagicMock(),
        lm_head_module=None if tied else MagicMock(return_value="LOGITS"),
        tie_word_embeddings=tied,
        hidden_size=4,
        vocab_size=10,
    )


def test_standalone_head_quacks_like_a_runtime():
    """HeadSampler's borrow protocol checks apply_final_head +
    _has_final_head + _is_last_shard; StandaloneHead must satisfy
    all three."""
    head = _build_mock_head()
    assert head._has_final_head is True
    assert head._is_last_shard is True
    assert callable(head.apply_final_head)
    assert head.backend == "mlx"


def test_standalone_head_apply_final_head_signature_matches_mlx_runtime():
    """apply_final_head signature must match MLXRuntime's verbatim —
    HeadSampler.sample passes decode kwargs by name."""
    from coordinator.standalone_head import StandaloneHead
    from peer.mlx_runtime import MLXRuntime

    expected = {
        name for name in inspect.signature(MLXRuntime.apply_final_head).parameters
        if name != "self"
    }
    actual = {
        name for name in inspect.signature(StandaloneHead.apply_final_head).parameters
        if name != "self"
    }
    assert actual == expected, (
        f"Signature drift.\n  expected: {sorted(expected)}\n"
        f"  got:      {sorted(actual)}"
    )


def test_standalone_head_apply_final_head_signature_matches_pytorch_runtime():
    """Also verify parity with PyTorchRuntime.apply_final_head."""
    from coordinator.standalone_head import StandaloneHead
    from peer.model_shard import PyTorchRuntime

    expected = {
        name for name in inspect.signature(PyTorchRuntime.apply_final_head).parameters
        if name != "self"
    }
    actual = {
        name for name in inspect.signature(StandaloneHead.apply_final_head).parameters
        if name != "self"
    }
    assert actual == expected


def test_standalone_head_unknown_backend_raises():
    """A StandaloneHead constructed with an unknown backend must fail
    loudly on apply_final_head — never silently return junk."""
    head = _build_mock_head(backend="klingon")
    with pytest.raises(RuntimeError, match="unknown backend"):
        head.apply_final_head([1.0, 4.0, 0, 0, 0, 0])


# ── MLX backend dispatch ────────────────────────────────────────────────

def test_apply_final_head_dispatches_to_backend_mlx():
    """backend='mlx' must route to ``_apply_mlx`` (not ``_apply_pytorch``).

    We patch the backend-specific methods rather than exercising real
    MLX matmuls — the live end-to-end path is covered by the Mac
    sanity check; here we're just verifying the dispatch table.
    """
    head = _build_mock_head(backend="mlx")
    head._apply_mlx = MagicMock(return_value=11)
    head._apply_pytorch = MagicMock(return_value=22)
    tok = head.apply_final_head([1.0, 4.0, 0, 0, 0, 0])
    assert tok == 11
    head._apply_mlx.assert_called_once()
    head._apply_pytorch.assert_not_called()


def test_apply_final_head_dispatches_to_backend_pytorch():
    """backend='pytorch' must route to ``_apply_pytorch``."""
    head = _build_mock_head(backend="pytorch")
    head._apply_mlx = MagicMock(return_value=11)
    head._apply_pytorch = MagicMock(return_value=22)
    tok = head.apply_final_head([1.0, 4.0, 0, 0, 0, 0])
    assert tok == 22
    head._apply_pytorch.assert_called_once()
    head._apply_mlx.assert_not_called()


def test_apply_final_head_threads_decode_kwargs_to_backend():
    """Every decode_* kwarg must be passed through verbatim to the
    backend-specific implementation."""
    head = _build_mock_head(backend="pytorch")
    head._apply_pytorch = MagicMock(return_value=5)
    head.apply_final_head(
        [1.0, 4.0, 0, 0, 0, 0],
        packed_bytes=b"\x00\x01",
        decode_do_sample=True,
        decode_temperature=0.7,
        decode_top_p=0.9,
        decode_top_k=40,
        decode_seed=123,
    )
    _, kwargs = head._apply_pytorch.call_args
    assert kwargs["packed_bytes"] == b"\x00\x01"
    assert kwargs["decode_do_sample"] is True
    assert kwargs["decode_temperature"] == pytest.approx(0.7)
    assert kwargs["decode_top_p"] == pytest.approx(0.9)
    assert kwargs["decode_top_k"] == 40
    assert kwargs["decode_seed"] == 123


def test_activation_to_hidden_rejects_hidden_size_mismatch_pytorch():
    """PyTorch path validates hidden_size. MLX path has the same check
    — both tested via the PyTorch branch because it doesn't require
    the MLX runtime to be loaded."""
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch not available")
    head = _build_mock_head(backend="pytorch")
    head._hidden_size = 2048
    bad_payload = [1.0, 999.0] + [0.0] * 999
    with pytest.raises(RuntimeError, match="hidden_size mismatch"):
        head._pytorch_activation_to_hidden(bad_payload)


# ── PyTorch backend dispatch ────────────────────────────────────────────

@pytest.fixture
def _torch():
    """Skip PyTorch-specific tests if torch isn't installed — keeps the
    suite runnable on minimal MLX-only envs."""
    try:
        import torch
        return torch
    except ImportError:
        pytest.skip("torch not available — PyTorch backend tests skipped")


def test_pytorch_backend_dispatches_tied_embeddings_path(_torch):
    """For tied models, _apply_pytorch uses matmul with embed.weight.T."""
    from coordinator.standalone_head import StandaloneHead

    # Real-ish tiny tensors so torch.matmul works.
    hidden_size = 4
    vocab_size = 8
    embed_weight = _torch.randn(vocab_size, hidden_size, dtype=_torch.float32)
    embed_module = MagicMock()
    embed_module.weight = embed_weight
    norm_module = MagicMock(side_effect=lambda h: h)

    head = StandaloneHead(
        backend="pytorch",
        hf_model_id="mock",
        norm_module=norm_module,
        embed_tokens_module=embed_module,
        lm_head_module=None,
        tie_word_embeddings=True,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        torch_device="cpu",
        torch_dtype=_torch.float32,
    )
    # Feed a well-formed list payload: [seq_len=1, hidden_size=4, v0..v3]
    payload = [1.0, float(hidden_size), 1.0, 0.0, 0.0, 0.0]
    tok = head.apply_final_head(payload, decode_do_sample=False)
    assert isinstance(tok, int)
    assert 0 <= tok < vocab_size
    norm_module.assert_called_once()


def test_pytorch_backend_dispatches_untied_lm_head_path(_torch):
    """For untied models, _apply_pytorch calls lm_head(normed)."""
    from coordinator.standalone_head import StandaloneHead

    hidden_size = 4
    vocab_size = 8

    def _lm_head_fn(normed):
        # Simulate a linear layer: return fixed logits.
        return _torch.zeros(1, 1, vocab_size)

    lm_head = MagicMock(side_effect=_lm_head_fn)
    head = StandaloneHead(
        backend="pytorch",
        hf_model_id="mock",
        norm_module=MagicMock(side_effect=lambda h: h),
        embed_tokens_module=MagicMock(),
        lm_head_module=lm_head,
        tie_word_embeddings=False,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        torch_device="cpu",
        torch_dtype=_torch.float32,
    )
    payload = [1.0, float(hidden_size), 0.0, 0.0, 0.0, 0.0]
    tok = head.apply_final_head(payload, decode_do_sample=False)
    assert isinstance(tok, int)
    lm_head.assert_called_once()


def test_pytorch_backend_greedy_vs_sample_split(_torch):
    """do_sample=False → argmax path; do_sample=True + temperature →
    multinomial path with seed determinism."""
    from coordinator.standalone_head import StandaloneHead

    # Build a head where logits strongly prefer token 3.
    hidden_size = 4
    vocab_size = 8
    one_hot_weight = _torch.zeros(vocab_size, hidden_size, dtype=_torch.float32)
    one_hot_weight[3, 0] = 100.0  # Token 3 gets a huge logit for any non-zero h[0].
    embed_module = MagicMock()
    embed_module.weight = one_hot_weight

    head = StandaloneHead(
        backend="pytorch",
        hf_model_id="mock",
        norm_module=MagicMock(side_effect=lambda h: h),
        embed_tokens_module=embed_module,
        lm_head_module=None,
        tie_word_embeddings=True,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        torch_device="cpu",
        torch_dtype=_torch.float32,
    )
    payload = [1.0, float(hidden_size), 1.0, 0.0, 0.0, 0.0]

    # Greedy: always picks 3.
    assert head.apply_final_head(payload, decode_do_sample=False) == 3

    # Sampled (high temperature would be random, but 100-logit hot one
    # still dominates): expect 3 even with sampling.
    assert head.apply_final_head(
        payload, decode_do_sample=True, decode_temperature=0.7, decode_seed=42,
    ) == 3


def test_pytorch_activation_to_hidden_rejects_hidden_size_mismatch(_torch):
    from coordinator.standalone_head import StandaloneHead

    head = StandaloneHead(
        backend="pytorch", hf_model_id="mock",
        norm_module=MagicMock(), embed_tokens_module=MagicMock(),
        lm_head_module=None, tie_word_embeddings=True,
        hidden_size=2048, vocab_size=10,
        torch_device="cpu", torch_dtype=_torch.float32,
    )
    bad_payload = [1.0, 1024.0] + [0.0] * 1024
    with pytest.raises(RuntimeError, match="hidden_size mismatch"):
        head._pytorch_activation_to_hidden(bad_payload)


# ── HeadSampler integration ─────────────────────────────────────────────

def test_head_sampler_borrows_from_standalone_head():
    """HeadSampler.sample must borrow transparently from a StandaloneHead
    regardless of backend."""
    from coordinator.head_sampler import (
        HeadSampler, DecodeConfig, register_head_source, get_head_sampler,
        clear_head_source,
    )

    clear_head_source()
    head = _build_mock_head()
    head.apply_final_head = MagicMock(return_value=7)

    register_head_source("coordinator-standalone-head", head)
    sampler = get_head_sampler()
    assert sampler is not None
    assert sampler.peer_id == "coordinator-standalone-head"

    tok = sampler.sample(
        hidden_state=[1.0, 4.0, 0, 0, 0, 0],
        decode=DecodeConfig(do_sample=True, temperature=0.5, top_p=0.9, seed=42),
        packed_bytes=None,
    )
    assert tok == 7
    _, kwargs = head.apply_final_head.call_args
    assert kwargs["decode_temperature"] == pytest.approx(0.5)
    assert kwargs["decode_top_p"] == pytest.approx(0.9)
    assert kwargs["decode_seed"] == 42
    clear_head_source()


# ── CLI guardrails ──────────────────────────────────────────────────────

def test_no_local_peer_flag_present_in_node_argparse():
    import subprocess, sys
    out = subprocess.run(
        [sys.executable, "-m", "coordinator.node", "--help"],
        capture_output=True, text=True, timeout=15,
    )
    assert "--no-local-peer" in out.stdout
    assert "--standalone-head-backend" in out.stdout
    assert "--standalone-head-device" in out.stdout


def test_no_local_peer_without_sample_on_coordinator_errors():
    """Guardrail #1 — the flags must come together."""
    import subprocess, sys
    out = subprocess.run(
        [sys.executable, "-m", "coordinator.node", "--no-local-peer"],
        capture_output=True, text=True, timeout=15,
    )
    assert out.returncode == 2
    assert "requires --sample-on-coordinator" in out.stderr


def test_no_local_peer_without_hf_runtime_model_id_errors():
    """Guardrail #2 — must point at a real HF repo id (slash required).
    Error message should mention both PyTorch and MLX options."""
    import subprocess, sys
    out = subprocess.run(
        [sys.executable, "-m", "coordinator.node",
         "--no-local-peer", "--sample-on-coordinator"],
        capture_output=True, text=True, timeout=15,
    )
    assert out.returncode == 2
    assert "HF repo id" in out.stderr
    # Ensures the error mentions BOTH backends so a Linux user gets
    # pointed at the right choice.
    assert "Qwen/Qwen3.5-2B" in out.stderr or "PyTorch" in out.stderr
    assert "mlx-community" in out.stderr or "MLX" in out.stderr
