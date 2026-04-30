# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Phase 2b live-bench Binding #3 — sharded block-verify tests.

Exercises ``forward_block_layer_slice`` against a synthetic
multi-shard PyTorchRuntime. Pins:

  * First-stage path embeds [prefix + drafts] then runs the slice.
  * Intermediate path takes incoming hidden states and runs the slice.
  * Last-stage path slices last B+1 positions before returning.
  * The chain (first → intermediate → last) produces hidden states
    bit-equal to a single full-model forward over the same input.
  * Empty drafts / missing inputs raise ValueError.
"""

from __future__ import annotations

import pytest


# ── Synthetic multi-shard model ───────────────────────────────────────


def _build_synthetic_shard(layer_indices: list[int], total_layers: int):
    """Construct a PyTorchRuntime stand-in with a tiny 4-layer model
    and the given layer indices selected.

    The decoder layers are all set to nn.Identity so we can hand-verify
    the chain's pass-through behaviour. Real Qwen3.5 layers do non-
    trivial work; the byte-equivalence test below uses linear layers
    with deterministic weights so the chain is non-trivial yet
    verifiable.
    """
    import torch
    import torch.nn as nn
    from peer.model_shard import PyTorchRuntime

    class _Block(nn.Module):
        """Mimics a transformer block's signature (returns (hidden,))
        and applies a deterministic linear transform."""
        def __init__(self, hidden: int, layer_idx: int):
            super().__init__()
            self.layer_idx = layer_idx
            self.proj = nn.Linear(hidden, hidden, bias=False)
            # Deterministic init keyed on layer_idx.
            with torch.no_grad():
                eye = torch.eye(hidden) * (1.0 + 0.01 * layer_idx)
                self.proj.weight.copy_(eye)

        def forward(self, x, *, past_key_values=None, use_cache=False,
                    attention_mask=None, cache_position=None,
                    position_ids=None, position_embeddings=None):
            out = self.proj(x)
            return (out,)

    torch.manual_seed(0)
    vocab, hidden = 32, 8
    runtime = PyTorchRuntime.__new__(PyTorchRuntime)
    runtime._device = torch.device("cpu")
    runtime._dtype = torch.float32
    runtime._embed_tokens = nn.Embedding(vocab, hidden)
    runtime._blocks = [_Block(hidden, i) for i in range(total_layers)]
    runtime.layer_indices = list(layer_indices)
    runtime._selected_layers = [runtime._blocks[i] for i in layer_indices]
    runtime.total_layers = total_layers
    runtime._decoder_family = "llama"   # use the standard branch in _run_layers
    runtime._rotary_emb = None          # no rotary for the synthetic test
    return runtime, vocab, hidden


# ── Per-stage shape tests ──────────────────────────────────────────────


def test_first_stage_embeds_then_runs_slice():
    """is_first_stage=True takes prefix + drafts as token IDs, embeds
    them, and runs the slice. Output shape is [1, prefix_len + B, hidden]."""
    runtime, _, hidden = _build_synthetic_shard([0, 1], total_layers=4)
    out = runtime.forward_block_layer_slice(
        is_first_stage=True, is_last_stage=False,
        prefix_token_ids=[1, 2, 3, 4],
        draft_token_ids=[10, 20, 30],
    )
    assert out.shape == (1, 4 + 3, hidden)


def test_intermediate_stage_takes_incoming_hidden():
    import torch
    runtime, _, hidden = _build_synthetic_shard([2], total_layers=4)
    incoming = torch.randn(1, 7, hidden, dtype=torch.float32)
    out = runtime.forward_block_layer_slice(
        is_first_stage=False, is_last_stage=False,
        draft_token_ids=[10, 20, 30],
        incoming_hidden=incoming,
    )
    assert out.shape == (1, 7, hidden)


def test_last_stage_slices_last_b_plus_one():
    """Terminal stage returns the last B+1 positions."""
    import torch
    runtime, _, hidden = _build_synthetic_shard([3], total_layers=4)
    incoming = torch.randn(1, 10, hidden, dtype=torch.float32)
    out = runtime.forward_block_layer_slice(
        is_first_stage=False, is_last_stage=True,
        draft_token_ids=[10, 20, 30],   # B=3
        incoming_hidden=incoming,
    )
    assert out.shape == (1, 4, hidden)   # B+1 = 4


def test_first_and_last_stage_combined():
    """Single-shard deployment with all layers on one peer — the
    method handles is_first_stage=True AND is_last_stage=True."""
    runtime, _, hidden = _build_synthetic_shard(
        [0, 1, 2, 3], total_layers=4,
    )
    out = runtime.forward_block_layer_slice(
        is_first_stage=True, is_last_stage=True,
        prefix_token_ids=[1, 2, 3, 4],
        draft_token_ids=[10, 20, 30],
    )
    assert out.shape == (1, 4, hidden)   # B+1 = 4 (sliced)


# ── Headline: chain reconstruction matches single-pass forward ────────


def test_sharded_chain_reproduces_single_pass_forward():
    """Compose stage-0 → stage-1 → stage-2 (3 shards over 4 layers)
    and assert the final hidden states bit-equal a full-model run
    over the same input.

    With deterministic Linear blocks and rotary disabled, this
    pins that the layer-slice routing in forward_block_layer_slice
    correctly preserves the per-layer math the unsharded path does.
    """
    import torch

    # 4-layer model. Three shards: [0], [1, 2], [3].
    runtime_full, _, hidden = _build_synthetic_shard(
        [0, 1, 2, 3], total_layers=4,
    )
    runtime_s0, _, _ = _build_synthetic_shard([0], total_layers=4)
    runtime_s1, _, _ = _build_synthetic_shard([1, 2], total_layers=4)
    runtime_s2, _, _ = _build_synthetic_shard([3], total_layers=4)

    # Sync embeddings + block weights across the four runtimes so
    # they parameterise the same model.
    with torch.no_grad():
        for rt in (runtime_s0, runtime_s1, runtime_s2):
            rt._embed_tokens.weight.copy_(runtime_full._embed_tokens.weight)
        # Block weights — the synthetic init is deterministic on layer_idx
        # so no-op assert that they match.
        for i in range(4):
            assert torch.equal(
                runtime_full._blocks[i].proj.weight,
                {0: runtime_s0, 1: runtime_s1, 2: runtime_s1, 3: runtime_s2}[i]
                ._blocks[i].proj.weight,
            )

    prefix = [1, 2, 3, 4]
    drafts = [10, 20, 30]

    # Single full-pass forward (gold).
    full_out = runtime_full.forward_block_layer_slice(
        is_first_stage=True, is_last_stage=True,
        prefix_token_ids=prefix, draft_token_ids=drafts,
    )

    # Sharded chain.
    s0_out = runtime_s0.forward_block_layer_slice(
        is_first_stage=True, is_last_stage=False,
        prefix_token_ids=prefix, draft_token_ids=drafts,
    )
    s1_out = runtime_s1.forward_block_layer_slice(
        is_first_stage=False, is_last_stage=False,
        draft_token_ids=drafts, incoming_hidden=s0_out,
    )
    s2_out = runtime_s2.forward_block_layer_slice(
        is_first_stage=False, is_last_stage=True,
        draft_token_ids=drafts, incoming_hidden=s1_out,
    )

    # Bit-equivalence between the chain and the single-pass forward.
    torch.testing.assert_close(s2_out, full_out, rtol=0, atol=0)


def test_sharded_chain_two_shards():
    """Same byte-equivalence claim, simpler topology: 2 shards
    [0, 1] and [2, 3]. This is the actual GPU1+GPU2 cross-ISP layout."""
    import torch

    runtime_full, _, _ = _build_synthetic_shard([0, 1, 2, 3], total_layers=4)
    runtime_s0, _, _ = _build_synthetic_shard([0, 1], total_layers=4)
    runtime_s1, _, _ = _build_synthetic_shard([2, 3], total_layers=4)

    with torch.no_grad():
        runtime_s0._embed_tokens.weight.copy_(runtime_full._embed_tokens.weight)
        runtime_s1._embed_tokens.weight.copy_(runtime_full._embed_tokens.weight)

    prefix = [5, 6, 7, 8, 9]
    drafts = [11, 13, 17, 19]   # all < vocab=32

    full_out = runtime_full.forward_block_layer_slice(
        is_first_stage=True, is_last_stage=True,
        prefix_token_ids=prefix, draft_token_ids=drafts,
    )
    s0_out = runtime_s0.forward_block_layer_slice(
        is_first_stage=True, is_last_stage=False,
        prefix_token_ids=prefix, draft_token_ids=drafts,
    )
    s1_out = runtime_s1.forward_block_layer_slice(
        is_first_stage=False, is_last_stage=True,
        draft_token_ids=drafts, incoming_hidden=s0_out,
    )

    torch.testing.assert_close(s1_out, full_out, rtol=0, atol=0)


# ── Failure modes ─────────────────────────────────────────────────────


def test_first_stage_empty_prefix_raises():
    runtime, _, _ = _build_synthetic_shard([0], total_layers=4)
    with pytest.raises(ValueError, match="prefix_token_ids"):
        runtime.forward_block_layer_slice(
            is_first_stage=True, is_last_stage=False,
            prefix_token_ids=[], draft_token_ids=[1],
        )


def test_empty_drafts_raises():
    runtime, _, _ = _build_synthetic_shard([0], total_layers=4)
    with pytest.raises(ValueError, match="draft_token_ids"):
        runtime.forward_block_layer_slice(
            is_first_stage=True, is_last_stage=False,
            prefix_token_ids=[1, 2], draft_token_ids=[],
        )


def test_intermediate_stage_missing_incoming_raises():
    runtime, _, _ = _build_synthetic_shard([1], total_layers=4)
    with pytest.raises(ValueError, match="incoming_hidden"):
        runtime.forward_block_layer_slice(
            is_first_stage=False, is_last_stage=False,
            draft_token_ids=[10],
            incoming_hidden=None,
        )


def test_2d_incoming_hidden_promoted_to_3d():
    """A caller passing [seq, hidden] (no batch axis) must be
    accepted and promoted to [1, seq, hidden]."""
    import torch
    runtime, _, hidden = _build_synthetic_shard([1], total_layers=4)
    incoming_2d = torch.randn(7, hidden, dtype=torch.float32)
    out = runtime.forward_block_layer_slice(
        is_first_stage=False, is_last_stage=False,
        draft_token_ids=[10, 20, 30],
        incoming_hidden=incoming_2d,
    )
    assert out.shape == (1, 7, hidden)


# ── Output feeds apply_final_head_block ──────────────────────────────


def test_terminal_output_feeds_apply_final_head_block():
    """The [1, B+1, hidden] tensor returned by the terminal stage
    must feed cleanly into apply_final_head_block — that's the
    contract the coord side relies on."""
    import torch
    import torch.nn as nn

    runtime, vocab, hidden = _build_synthetic_shard(
        [0, 1, 2, 3], total_layers=4,
    )
    # Wire up final norm + lm_head so apply_final_head_block can run.
    runtime._final_norm = nn.Identity()
    runtime._lm_head = nn.Linear(hidden, vocab, bias=False)

    block = runtime.forward_block_layer_slice(
        is_first_stage=True, is_last_stage=True,
        prefix_token_ids=[1, 2, 3], draft_token_ids=[10, 20],
    )
    argmax_ids = runtime.apply_final_head_block(block)
    assert isinstance(argmax_ids, list)
    assert len(argmax_ids) == 3   # B+1 = 3
    assert all(0 <= t < vocab for t in argmax_ids)
