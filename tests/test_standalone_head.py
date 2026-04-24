# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Tests for the Phase 6 standalone head loader (true Petals topology).

These tests use mock MLX modules — actually loading
``mlx-community/Qwen3.5-2B-MLX-8bit`` would take ~5 s on Apple Silicon and
download ~1 GB on a fresh runner; that's covered by the 2026-04-24
in-process sanity check we ran before committing.

Coverage:
    * StandaloneHead duck-types as a HeadSampler runtime
      (``_has_final_head``, ``_is_last_shard``, ``apply_final_head``)
    * apply_final_head decode-kwargs are forwarded to the sampler
    * tied vs untied lm_head dispatch picks the right matmul path
    * apply_final_head raises on impossible shapes (hidden-size mismatch)
    * HeadSampler can borrow from a StandaloneHead just like from a
      MLXRuntime / PyTorchRuntime peer
"""

from __future__ import annotations

import inspect
import types
from unittest.mock import MagicMock

import pytest


# ── StandaloneHead surface ───────────────────────────────────────────────

def test_standalone_head_quacks_like_a_runtime():
    """HeadSampler's borrow protocol checks `apply_final_head`,
    `_has_final_head`, and `_is_last_shard`. StandaloneHead must satisfy
    all three so server.py::_maybe_register_head_source-style code can
    treat it identically to an MLXRuntime peer."""
    from coordinator.standalone_head import StandaloneHead

    head = StandaloneHead(
        hf_model_id="x",
        norm_module=MagicMock(),
        embed_tokens_module=MagicMock(),
        lm_head_module=None,
        tie_word_embeddings=True,
        hidden_size=2048,
        vocab_size=248320,
    )
    assert head._has_final_head is True
    assert head._is_last_shard is True
    assert callable(head.apply_final_head)


def test_standalone_head_apply_final_head_signature_matches_mlx_runtime():
    """apply_final_head MUST have the same kwargs MLXRuntime exposes —
    HeadSampler.sample passes them positionally and any drift breaks
    the borrow protocol silently."""
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
        f"StandaloneHead drifted from MLXRuntime contract.\n"
        f"  expected: {sorted(expected)}\n  got:      {sorted(actual)}"
    )


def test_standalone_head_dispatches_tied_embeddings_path():
    """For tied-embedding models (Qwen3.5), apply_final_head must call
    ``embed_tokens.as_linear(h)`` rather than ``lm_head(h)``."""
    from coordinator.standalone_head import StandaloneHead

    norm = MagicMock(side_effect=lambda h: h)  # passthrough
    embed = MagicMock()
    embed.as_linear = MagicMock()

    head = StandaloneHead(
        hf_model_id="x",
        norm_module=norm,
        embed_tokens_module=embed,
        lm_head_module=None,
        tie_word_embeddings=True,
        hidden_size=4,
        vocab_size=10,
    )
    # Patch _activation_to_hidden + _sample_from_logits so we can run
    # without MLX present in CI. The only thing we care about is which
    # head matmul is chosen.
    head._activation_to_hidden = MagicMock(return_value="HIDDEN_TENSOR")
    head._sample_from_logits = MagicMock(return_value=42)
    embed.as_linear.return_value = "LOGITS"

    # mlx.core.eval is called during apply_final_head; stub it out.
    import sys
    if "mlx" not in sys.modules:
        _fake_mx = types.ModuleType("mlx")
        _fake_core = types.ModuleType("mlx.core")
        _fake_core.eval = lambda x: None
        _fake_mx.core = _fake_core
        sys.modules["mlx"] = _fake_mx
        sys.modules["mlx.core"] = _fake_core

    tok = head.apply_final_head([1.0, 4.0, 0, 0, 0, 0])
    assert tok == 42
    # Tied path used.
    embed.as_linear.assert_called_once()
    norm.assert_called_once()


def test_standalone_head_dispatches_untied_lm_head_path():
    """For non-tied models, apply_final_head must call lm_head(h)."""
    from coordinator.standalone_head import StandaloneHead

    norm = MagicMock(side_effect=lambda h: h)
    embed = MagicMock()
    lm_head = MagicMock(return_value="LOGITS")

    head = StandaloneHead(
        hf_model_id="x",
        norm_module=norm,
        embed_tokens_module=embed,
        lm_head_module=lm_head,
        tie_word_embeddings=False,
        hidden_size=4,
        vocab_size=10,
    )
    head._activation_to_hidden = MagicMock(return_value="HIDDEN_TENSOR")
    head._sample_from_logits = MagicMock(return_value=99)

    import sys
    if "mlx" not in sys.modules:
        _fake_mx = types.ModuleType("mlx")
        _fake_core = types.ModuleType("mlx.core")
        _fake_core.eval = lambda x: None
        _fake_mx.core = _fake_core
        sys.modules["mlx"] = _fake_mx
        sys.modules["mlx.core"] = _fake_core

    tok = head.apply_final_head([1.0, 4.0, 0, 0, 0, 0])
    assert tok == 99
    lm_head.assert_called_once()
    embed.as_linear.assert_not_called()  # tied path NOT taken


def test_standalone_head_untied_without_lm_head_raises():
    """Construction-time consistency: untied model + no lm_head module
    is a misconfiguration; apply_final_head should fail loudly."""
    from coordinator.standalone_head import StandaloneHead

    head = StandaloneHead(
        hf_model_id="x",
        norm_module=MagicMock(side_effect=lambda h: h),
        embed_tokens_module=MagicMock(),
        lm_head_module=None,  # MISSING
        tie_word_embeddings=False,  # untied
        hidden_size=4,
        vocab_size=10,
    )
    head._activation_to_hidden = MagicMock(return_value="HIDDEN")

    import sys
    if "mlx" not in sys.modules:
        _fake_mx = types.ModuleType("mlx")
        _fake_core = types.ModuleType("mlx.core")
        _fake_core.eval = lambda x: None
        _fake_mx.core = _fake_core
        sys.modules["mlx"] = _fake_mx
        sys.modules["mlx.core"] = _fake_core

    with pytest.raises(RuntimeError, match="untied .* no lm_head"):
        head.apply_final_head([1.0, 4.0, 0, 0, 0, 0])


def test_standalone_head_validates_hidden_size_mismatch():
    """activation_to_hidden must reject payloads whose declared hidden
    size doesn't match the head's expected hidden size."""
    from coordinator.standalone_head import StandaloneHead

    head = StandaloneHead(
        hf_model_id="x",
        norm_module=MagicMock(),
        embed_tokens_module=MagicMock(),
        lm_head_module=None,
        tie_word_embeddings=True,
        hidden_size=2048,
        vocab_size=10,
    )
    # Wrong hidden size in payload.
    bad_payload = [1.0, 1024.0] + [0.0] * 1024
    with pytest.raises(RuntimeError, match="hidden_size mismatch"):
        head._activation_to_hidden(bad_payload)


# ── HeadSampler integration ──────────────────────────────────────────────

def test_head_sampler_borrows_from_standalone_head():
    """HeadSampler.sample must work transparently when the registered
    runtime is a StandaloneHead rather than a peer-runtime."""
    from coordinator.head_sampler import (
        HeadSampler, DecodeConfig, register_head_source, get_head_sampler,
        clear_head_source,
    )
    from coordinator.standalone_head import StandaloneHead

    clear_head_source()

    head = StandaloneHead(
        hf_model_id="x",
        norm_module=MagicMock(),
        embed_tokens_module=MagicMock(),
        lm_head_module=None,
        tie_word_embeddings=True,
        hidden_size=4,
        vocab_size=10,
    )
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
    head.apply_final_head.assert_called_once()
    _, kwargs = head.apply_final_head.call_args
    assert kwargs["decode_temperature"] == pytest.approx(0.5)
    assert kwargs["decode_top_p"] == pytest.approx(0.9)
    assert kwargs["decode_seed"] == 42

    clear_head_source()


# ── CLI guardrails (smoke; deeper E2E tested via live launch) ────────────

def test_no_local_peer_flag_present_in_node_argparse():
    """The CLI flag must be registered or coordinator/node.py won't
    parse the new mode."""
    import subprocess, sys
    out = subprocess.run(
        [sys.executable, "-m", "coordinator.node", "--help"],
        capture_output=True, text=True, timeout=15,
    )
    assert "--no-local-peer" in out.stdout
    assert "Phase 6" in out.stdout or "true Petals" in out.stdout


def test_no_local_peer_without_sample_on_coordinator_errors():
    """Guardrail #1 — the flags must come together."""
    import subprocess, sys
    out = subprocess.run(
        [sys.executable, "-m", "coordinator.node", "--no-local-peer"],
        capture_output=True, text=True, timeout=15,
    )
    # argparse errors go to stderr with exit code 2.
    assert out.returncode == 2
    assert "requires --sample-on-coordinator" in out.stderr


def test_no_local_peer_without_hf_runtime_model_id_errors():
    """Guardrail #2 — must point at a real HF repo id (slash required)."""
    import subprocess, sys
    out = subprocess.run(
        [sys.executable, "-m", "coordinator.node",
         "--no-local-peer", "--sample-on-coordinator"],
        capture_output=True, text=True, timeout=15,
    )
    assert out.returncode == 2
    assert "HF repo id" in out.stderr
    assert "user/repo" in out.stderr or "slash" in out.stderr
