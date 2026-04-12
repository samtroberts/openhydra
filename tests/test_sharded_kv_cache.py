# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Regression tests for Phase 2B — offloaded-layer Identity replacement.

These tests cover the helpers in ``peer/model_shard.py`` that drop meta-device
placeholder modules out of the decoder ``ModuleList`` so the KV-cache code
paths (streaming ``infer_chat_stream``, any transformers op that iterates the
full layer list) no longer dispatch aten ops on meta tensors.

Root cause fixed here:
    ``self._blocks[idx] = None`` in the legacy cleanup path only cleared our
    local tuple reference; the real ``nn.ModuleList`` still held the meta
    placeholders that accelerate mapped to ``"disk"``. When any code path
    iterated the full list (DynamicCache internals, ``_update_causal_mask``,
    ``model.parameters()``) PyTorch's dispatcher raised
    ``GET was unable to find an engine to execute this computation`` because
    the meta device has no kernels.

The fix: after accelerate loads the shard, replace every out-of-shard slot
with ``nn.Identity()``. These tests use plain torch modules (no accelerate,
no HuggingFace) so they run fast and don't need the Lightning GPUs.
"""

from __future__ import annotations

import torch
from torch import nn

from peer.model_shard import (
    _find_decoder_layer_list,
    _replace_offloaded_layers_with_identity,
)


class _FakeDecoderLayer(nn.Module):
    """Stand-in for a Qwen/LLaMA decoder layer with the exact attributes
    OpenHydra's forward path touches."""

    def __init__(self, layer_idx: int, hidden_size: int = 8):
        super().__init__()
        self.layer_idx = layer_idx
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, **kwargs):
        return self.linear(hidden_states)


class _FakeTextModel(nn.Module):
    """Stand-in for a ``model.model`` wrapper with a ``.layers`` ModuleList
    (matches the Qwen3_5ForCausalLM / LlamaModel shape)."""

    def __init__(self, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            _FakeDecoderLayer(layer_idx=i) for i in range(num_layers)
        )
        self.embed_tokens = nn.Embedding(100, 8)
        self.norm = nn.LayerNorm(8)


class _FakeOuterModel(nn.Module):
    """Stand-in for a ``ForCausalLM`` outer wrapper with a ``.model``
    attribute — the same shape ``_find_decoder_layer_list`` probes."""

    def __init__(self, num_layers: int):
        super().__init__()
        self.model = _FakeTextModel(num_layers)
        self.lm_head = nn.Linear(8, 100)


def _move_layer_to_meta(layer: nn.Module) -> None:
    """Force every parameter of a module onto the ``meta`` device.

    This is what accelerate does when a device_map entry is ``"disk"`` —
    the submodule stays attached to the parent but all its tensors are
    materialized on ``meta``. Any real compute op on them raises
    ``GET was unable to find an engine to execute this computation``.
    """
    layer.to_empty(device="meta")


def _count_meta_params(model: nn.Module) -> int:
    return sum(1 for p in model.parameters() if p.device.type == "meta")


# ──────────────────────────────────────────────────────────────────────────
# _find_decoder_layer_list
# ──────────────────────────────────────────────────────────────────────────


def test_find_decoder_layer_list_standard_wrapper():
    """Standard ``model.model.layers`` path is discoverable."""
    m = _FakeOuterModel(num_layers=4)
    found = _find_decoder_layer_list(m)
    assert found is m.model.layers
    assert len(found) == 4


def test_find_decoder_layer_list_no_match_returns_none():
    """Bare module with no ``.model.layers`` returns None, not crash."""

    class _Bare(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

    assert _find_decoder_layer_list(_Bare()) is None


# ──────────────────────────────────────────────────────────────────────────
# _replace_offloaded_layers_with_identity
# ──────────────────────────────────────────────────────────────────────────


def test_identity_swap_removes_meta_tensors_from_module_list():
    """Core regression: after Identity replacement no meta-device params
    remain on the model — the previous pain point for KV-cache-aware
    code paths that iterated ``model.parameters()``.
    """
    m = _FakeOuterModel(num_layers=8)

    # Simulate accelerate's disk offload: layers 4-7 live on meta,
    # layers 0-3 are real (the shard we actually run).
    for idx in range(4, 8):
        _move_layer_to_meta(m.model.layers[idx])

    assert _count_meta_params(m) > 0, "precondition: meta params present"

    replaced = _replace_offloaded_layers_with_identity(m, tuple(range(4)))

    assert replaced == 4
    assert _count_meta_params(m) == 0, (
        "After Identity swap, no meta-device params should remain — "
        "this is what unblocks the streaming KV-cache path."
    )
    # Kept layers still work as real decoder layers.
    for idx in range(4):
        assert isinstance(m.model.layers[idx], _FakeDecoderLayer)
    # Dropped layers are now Identity modules (parameterless).
    for idx in range(4, 8):
        assert isinstance(m.model.layers[idx], nn.Identity)


def test_identity_swap_preserves_indexing():
    """Block indexing (``layers[layer_idx]``) must stay valid — the
    Identity swap mustn't reorder the list. The block's ``layer_idx``
    attribute (set at model construction) is what Qwen3.5 uses to key
    into DynamicCache, so position-preserving replacement is required.
    """
    m = _FakeOuterModel(num_layers=6)
    _replace_offloaded_layers_with_identity(m, (0, 2, 5))

    assert isinstance(m.model.layers[0], _FakeDecoderLayer)
    assert m.model.layers[0].layer_idx == 0
    assert isinstance(m.model.layers[1], nn.Identity)
    assert isinstance(m.model.layers[2], _FakeDecoderLayer)
    assert m.model.layers[2].layer_idx == 2
    assert isinstance(m.model.layers[3], nn.Identity)
    assert isinstance(m.model.layers[4], nn.Identity)
    assert isinstance(m.model.layers[5], _FakeDecoderLayer)
    assert m.model.layers[5].layer_idx == 5


def test_identity_swap_is_idempotent():
    """Calling the helper twice with the same kept set must be a no-op
    the second time — prevents double-replacement surprises if the
    model is reused or reloaded within the same process."""
    m = _FakeOuterModel(num_layers=4)
    first = _replace_offloaded_layers_with_identity(m, (0, 1))
    second = _replace_offloaded_layers_with_identity(m, (0, 1))
    assert first == 2
    assert second == 0  # Already Identity — nothing to do.


def test_identity_swap_no_op_when_all_kept():
    """Single-peer full-model loads (``len(layer_indices) == total``) must
    be completely untouched by the swap — the caller in
    ``PyTorchRuntime.__init__`` guards this with
    ``if len(self.layer_indices) < self.total_layers``, but the helper
    itself should also be safe to call as a no-op."""
    m = _FakeOuterModel(num_layers=4)
    replaced = _replace_offloaded_layers_with_identity(m, (0, 1, 2, 3))
    assert replaced == 0
    for idx in range(4):
        assert isinstance(m.model.layers[idx], _FakeDecoderLayer)


def test_identity_swap_handles_unknown_model_shape():
    """Helper on a model with no discoverable layer list returns 0
    without raising — matches ``_find_decoder_layer_list`` behavior."""

    class _Bare(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

    assert _replace_offloaded_layers_with_identity(_Bare(), (0, 1)) == 0


def test_identity_swap_allows_parameters_iteration_without_crash():
    """End-to-end regression: ``list(model.parameters())`` is the same
    iteration pattern that DynamicCache / accelerate hooks eventually
    touch. After the swap it must complete on CPU without the
    dispatcher error the real Lightning peers hit."""
    m = _FakeOuterModel(num_layers=8)
    for idx in range(4, 8):
        _move_layer_to_meta(m.model.layers[idx])
    _replace_offloaded_layers_with_identity(m, tuple(range(4)))

    # Touching every parameter via a trivial op must now succeed —
    # with meta tensors present, ``p + 0`` would raise.
    for p in m.parameters():
        _ = p + 0
