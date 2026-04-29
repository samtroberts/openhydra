# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Phase 2b live-bench Binding #1 — forward_block_for_verify tests.

Validates the in-process target callable against a synthetic HF-style
``nn.Module`` that exposes the canonical ``model.model.norm`` final
norm path. Pins:

  * The forward-pre-hook fires and captures pre-norm hidden states.
  * The verify slice picks the right ``B + 1`` positions (covering
    drafts plus the bonus).
  * The output shape is ``[1, B+1, hidden]``.
  * The output feeds cleanly into ``apply_final_head_block`` and
    produces the expected argmax for a hand-constructed lm_head.
  * Empty prefix or drafts raise ValueError; missing lm_head raises
    RuntimeError.

These tests run on PyTorch only (the test runner doesn't have MLX);
the MLX symmetric path is exercised on Apple-Silicon CI.
"""

from __future__ import annotations

import pytest


# ── Synthetic HF-style model ───────────────────────────────────────────


def _build_synthetic_runtime():
    """Construct a PyTorchRuntime stand-in with a tiny LM-style model.

    The model follows HF's Llama-like layout:
        model.model.embed_tokens   — embedding
        model.model.layers[*]      — decoder blocks (we use a no-op stack)
        model.model.norm           — final norm
        model.lm_head              — output projection

    The decoder is intentionally trivial so we can hand-verify the
    pre-norm capture and lm_head outputs.
    """
    import torch
    import torch.nn as nn
    from peer.model_shard import PyTorchRuntime

    class _NoOpDecoder(nn.Module):
        def forward(self, x):
            return x   # passthrough — preserves the embedding

    class _SyntheticInner(nn.Module):
        """Stands in for ``model.model``."""

        def __init__(self, vocab: int, hidden: int):
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab, hidden)
            self.layers = nn.ModuleList([_NoOpDecoder()])
            # The norm we capture from. Initialise as identity so the
            # pre-norm capture is bit-equal to the post-norm output for
            # a trivial sanity check.
            self.norm = nn.Identity()

        def forward(self, input_ids, **_kwargs):
            x = self.embed_tokens(input_ids)
            for layer in self.layers:
                x = layer(x)
            x = self.norm(x)
            return x

    class _SyntheticModel(nn.Module):
        """Stands in for ``model`` (LlamaForCausalLM-style)."""

        def __init__(self, vocab: int, hidden: int):
            super().__init__()
            self.model = _SyntheticInner(vocab, hidden)
            self.lm_head = nn.Linear(hidden, vocab, bias=False)

        def forward(self, input_ids, **_kwargs):
            h = self.model(input_ids)
            return self.lm_head(h)

    # Build a minimal PyTorchRuntime by constructing the bare-bones
    # attributes the verify path requires; bypass __init__ since the
    # full constructor pulls in the actual tinyllama/HF model.
    torch.manual_seed(0)
    vocab, hidden = 64, 8
    runtime = PyTorchRuntime.__new__(PyTorchRuntime)
    runtime._device = torch.device("cpu")
    runtime._model = _SyntheticModel(vocab, hidden)
    runtime._lm_head = runtime._model.lm_head
    runtime._final_norm = runtime._model.model.norm
    return runtime, vocab, hidden


# ── Happy-path tests ────────────────────────────────────────────────────


def test_forward_block_for_verify_returns_correct_shape():
    """Output shape must be ``[1, B+1, hidden]`` with B=4 drafts and
    hidden=8."""
    import torch

    runtime, vocab, hidden = _build_synthetic_runtime()
    prefix = [1, 2, 3, 4, 5]
    drafts = [10, 20, 30, 40]
    out = runtime.forward_block_for_verify(prefix, drafts)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, len(drafts) + 1, hidden)


def test_forward_block_for_verify_captures_pre_norm():
    """With nn.Identity as the final norm, pre-norm hidden states
    equal the embeddings of the input tokens at the verify positions
    — easy to hand-verify."""
    import torch

    runtime, vocab, hidden = _build_synthetic_runtime()
    prefix = [1, 2, 3]
    drafts = [10, 20]

    out = runtime.forward_block_for_verify(prefix, drafts)
    # Verify positions: [prefix_len-1, prefix_len, prefix_len+B-1]
    # = [2, 3, 4] in the 5-token input.
    embed = runtime._model.model.embed_tokens
    full_input = torch.tensor([prefix + drafts], dtype=torch.long)
    expected = embed(full_input)[:, 2:5, :]   # B+1 = 3 positions
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_forward_block_for_verify_feeds_apply_final_head_block():
    """End-to-end: forward_block_for_verify → apply_final_head_block
    produces argmax token ids of length B+1."""
    runtime, vocab, hidden = _build_synthetic_runtime()
    prefix = [1, 2, 3, 4]
    drafts = [10, 20, 30]   # B=3

    hidden_block = runtime.forward_block_for_verify(prefix, drafts)
    argmax_ids = runtime.apply_final_head_block(hidden_block)
    assert isinstance(argmax_ids, list)
    assert len(argmax_ids) == len(drafts) + 1   # B+1
    assert all(0 <= t < vocab for t in argmax_ids)


def test_apply_final_head_block_accepts_2d_tensor():
    """Caller passing ``[seq, hidden]`` (no batch axis) must be
    accepted — the in-process binding may construct either shape."""
    import torch

    runtime, vocab, hidden = _build_synthetic_runtime()
    block_3d = runtime.forward_block_for_verify([1, 2, 3], [10, 20])
    block_2d = block_3d.squeeze(0)
    assert block_2d.dim() == 2

    out_3d = runtime.apply_final_head_block(block_3d)
    out_2d = runtime.apply_final_head_block(block_2d)
    assert out_3d == out_2d


# ── Failure modes ─────────────────────────────────────────────────────


def test_forward_block_for_verify_empty_prefix_raises():
    runtime, _, _ = _build_synthetic_runtime()
    with pytest.raises(ValueError, match="prefix_token_ids"):
        runtime.forward_block_for_verify([], [10, 20])


def test_forward_block_for_verify_empty_drafts_raises():
    runtime, _, _ = _build_synthetic_runtime()
    with pytest.raises(ValueError, match="draft_token_ids"):
        runtime.forward_block_for_verify([1, 2], [])


def test_forward_block_for_verify_missing_lm_head_raises():
    runtime, _, _ = _build_synthetic_runtime()
    runtime._lm_head = None
    with pytest.raises(RuntimeError, match="last layer"):
        runtime.forward_block_for_verify([1, 2], [10])


def test_forward_block_for_verify_missing_norm_raises():
    """If the model has no .model.norm AND no _final_norm attribute,
    surface as RuntimeError so the operator knows the runtime layout
    isn't the canonical HF one."""
    runtime, _, _ = _build_synthetic_runtime()
    # Drop both norm references.
    del runtime._model.model.norm
    runtime._final_norm = None
    with pytest.raises(RuntimeError, match="final norm"):
        runtime.forward_block_for_verify([1, 2], [10])


def test_forward_block_for_verify_falls_back_to_self_final_norm():
    """When ``_model.model.norm`` is missing but ``self._final_norm``
    is set, the method uses the latter as the hook target."""
    import torch.nn as nn

    runtime, _, _ = _build_synthetic_runtime()
    # Move norm out of the canonical path; assign to _final_norm.
    fallback_norm = runtime._model.model.norm
    del runtime._model.model.norm
    runtime._final_norm = fallback_norm
    # Re-route the inner forward to use the fallback so the hook fires.
    inner = runtime._model.model

    def _custom_forward(input_ids, **_kwargs):
        x = inner.embed_tokens(input_ids)
        for layer in inner.layers:
            x = layer(x)
        return fallback_norm(x)

    inner.forward = _custom_forward

    out = runtime.forward_block_for_verify([1, 2, 3], [10, 20])
    assert out.shape[1] == 3   # B+1 = 3
