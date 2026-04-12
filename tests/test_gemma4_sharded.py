# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Regression tests for Phase 4 — Gemma 4 sharded adapter.

These tests exercise the Python-side plumbing only (no real Gemma 4 weights
required):

- ``_DecoderArchitecture`` has the new ``layer_types`` / ``per_layer_*`` /
  ``text_model`` fields and defaults to empty so non-Gemma-4 callers stay
  untouched.
- ``ForwardRequest.prompt_token_ids`` round-trips through the regenerated
  protobuf and survives a call to ``_request_stage``.
- ``chain.run()`` auto-derives ``prompt_token_ids`` from an
  ``initial_activation`` of integer-valued floats (matches what the Phase 1
  non-streaming decode loop actually passes).
- ``PyTorchRuntime._compute_gemma4_per_layer_inputs`` is a no-op for
  non-gemma4 families and returns None cleanly without a model reference.

The full end-to-end forward is covered by the live benchmark; these tests
exist to lock in the wire/code contract so future refactors can't silently
break Gemma 4 sharded inference.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from peer import peer_pb2
from peer.model_shard import _DecoderArchitecture


# ──────────────────────────────────────────────────────────────────────────
# _DecoderArchitecture — Gemma 4 extras default to empty / None
# ──────────────────────────────────────────────────────────────────────────


def test_decoder_architecture_gemma4_fields_default_empty():
    """Non-Gemma-4 families should not carry any per-layer-input state —
    the extras default to empty so the existing qwen_llama / llama code
    paths are completely unchanged."""
    arch = _DecoderArchitecture(
        family="qwen_llama",
        layers=(),
        embed_tokens=object(),
    )
    assert arch.layer_types == ()
    assert arch.per_layer_embed is None
    assert arch.per_layer_proj is None
    assert arch.per_layer_norm is None
    assert arch.hidden_size_per_layer == 0
    assert arch.text_model is None


def test_decoder_architecture_gemma4_fields_populated():
    """Gemma 4 detection populates all the extras in one shot."""
    sentinel_text_model = object()
    sentinel_embed = object()
    sentinel_proj = object()
    sentinel_norm = object()
    arch = _DecoderArchitecture(
        family="gemma4",
        layers=(),
        embed_tokens=object(),
        layer_types=("full_attention", "sliding_attention", "full_attention"),
        per_layer_embed=sentinel_embed,
        per_layer_proj=sentinel_proj,
        per_layer_norm=sentinel_norm,
        hidden_size_per_layer=256,
        text_model=sentinel_text_model,
    )
    assert arch.family == "gemma4"
    assert arch.layer_types == ("full_attention", "sliding_attention", "full_attention")
    assert arch.per_layer_embed is sentinel_embed
    assert arch.per_layer_proj is sentinel_proj
    assert arch.per_layer_norm is sentinel_norm
    assert arch.hidden_size_per_layer == 256
    assert arch.text_model is sentinel_text_model


# ──────────────────────────────────────────────────────────────────────────
# ForwardRequest.prompt_token_ids — wire-format contract
# ──────────────────────────────────────────────────────────────────────────


def test_forward_request_prompt_token_ids_roundtrip():
    """The new proto field serialises and deserialises cleanly and is
    empty by default so old peers / non-Gemma-4 families are untouched."""
    req = peer_pb2.ForwardRequest()
    assert list(req.prompt_token_ids) == []  # default: empty

    req = peer_pb2.ForwardRequest(prompt_token_ids=[1, 2, 3, 248046])
    wire = req.SerializeToString()
    roundtrip = peer_pb2.ForwardRequest()
    roundtrip.ParseFromString(wire)
    assert list(roundtrip.prompt_token_ids) == [1, 2, 3, 248046]


def test_forward_request_prompt_token_ids_field_number_stable():
    """Lock in the field number so future proto edits can't silently
    collide with other fields — breaking this test is a loud signal
    that an old peer/coordinator pair won't interop."""
    field = peer_pb2.ForwardRequest.DESCRIPTOR.fields_by_name["prompt_token_ids"]
    assert field.number == 42
    # Protobuf int64 enum value is 3 on the runtime side.
    assert field.type == field.TYPE_INT64
    assert field.label == field.LABEL_REPEATED


# ──────────────────────────────────────────────────────────────────────────
# chain.run() auto-derives prompt_token_ids from initial_activation
# ──────────────────────────────────────────────────────────────────────────


def test_chain_run_auto_derives_prompt_token_ids_from_initial_activation(monkeypatch):
    """The Phase 1 non-streaming loop passes ``initial_activation`` as a
    list of token-IDs-as-floats. ``chain.run()`` must sniff that and
    auto-populate the Gemma 4 sidecar so the caller doesn't need to know
    about the distinction."""
    from coordinator.chain import InferenceChain, ChainResult

    captured: dict[str, object] = {}

    def _fake_request_stage(self, **kw):
        captured.setdefault("stages", []).append(dict(kw))
        # Return hidden-state activation for intermediate stages, token IDs for last
        stage_index = int(kw["stage_index"])
        total = int(kw["total_stages"])
        if stage_index == total - 1:
            activation = [271.0]  # fake sampled token id
        else:
            activation = [0.1, 0.2, 0.3]  # fake hidden state
        return SimpleNamespace(
            activation=activation,
            latency_ms=1.0,
            latent_dim=0,
            activation_hash=b"",
        )

    monkeypatch.setattr(InferenceChain, "_request_stage", _fake_request_stage, raising=True)

    peer1 = SimpleNamespace(
        peer_id="p1", host="h", port=1, runtime_backend="pytorch_auto",
        public_key_hex="", privacy_noise_variance=0.0,
        layer_start=0, layer_end=16, total_layers=32,
    )
    peer2 = SimpleNamespace(
        peer_id="p2", host="h", port=2, runtime_backend="pytorch_auto",
        public_key_hex="", privacy_noise_variance=0.0,
        layer_start=16, layer_end=32, total_layers=32,
    )
    chain = InferenceChain(pipeline=[peer1, peer2], timeout_ms=1000)
    chain.run(
        prompt="",
        max_tokens=1,
        initial_activation=[float(t) for t in (1, 2, 3, 4, 5)],
        request_id="test-auto-derive",
    )

    # Every stage should have been called with prompt_token_ids == [1,2,3,4,5]
    stages = captured["stages"]
    assert len(stages) == 2
    for s in stages:
        assert list(s.get("prompt_token_ids", [])) == [1, 2, 3, 4, 5]


def test_chain_run_explicit_prompt_token_ids_wins(monkeypatch):
    """If the caller passes ``prompt_token_ids`` explicitly it overrides
    the auto-derived value from ``initial_activation``."""
    from coordinator.chain import InferenceChain

    captured: dict[str, object] = {}

    def _fake_request_stage(self, **kw):
        captured.setdefault("stages", []).append(dict(kw))
        return SimpleNamespace(
            activation=[99.0],
            latency_ms=1.0,
            latent_dim=0,
            activation_hash=b"",
        )

    monkeypatch.setattr(InferenceChain, "_request_stage", _fake_request_stage, raising=True)

    peer = SimpleNamespace(
        peer_id="solo", host="h", port=1, runtime_backend="pytorch_auto",
        public_key_hex="", privacy_noise_variance=0.0,
        layer_start=0, layer_end=32, total_layers=32,
    )
    chain = InferenceChain(pipeline=[peer], timeout_ms=1000)
    chain.run(
        prompt="",
        max_tokens=1,
        initial_activation=[10.0, 20.0],
        prompt_token_ids=[100, 200, 300],
        request_id="test-explicit",
    )

    assert list(captured["stages"][0].get("prompt_token_ids", [])) == [100, 200, 300]


def test_chain_run_skips_auto_derive_for_hidden_state(monkeypatch):
    """When ``initial_activation`` is a hidden-state tensor (floats that
    are NOT integer-valued), ``chain.run()`` must NOT try to interpret
    them as token IDs — the sidecar ends up empty and downstream peers
    fall back to the non-Gemma-4 path."""
    from coordinator.chain import InferenceChain

    captured: dict[str, object] = {}

    def _fake_request_stage(self, **kw):
        captured.setdefault("stages", []).append(dict(kw))
        return SimpleNamespace(
            activation=[0.5],
            latency_ms=1.0,
            latent_dim=0,
            activation_hash=b"",
        )

    monkeypatch.setattr(InferenceChain, "_request_stage", _fake_request_stage, raising=True)

    peer = SimpleNamespace(
        peer_id="solo", host="h", port=1, runtime_backend="pytorch_auto",
        public_key_hex="", privacy_noise_variance=0.0,
        layer_start=0, layer_end=32, total_layers=32,
    )
    chain = InferenceChain(pipeline=[peer], timeout_ms=1000)
    chain.run(
        prompt="",
        max_tokens=1,
        initial_activation=[0.123, -0.456, 0.789],  # NOT token IDs
        request_id="test-hidden",
    )

    assert list(captured["stages"][0].get("prompt_token_ids", [])) == []


# ──────────────────────────────────────────────────────────────────────────
# _compute_gemma4_per_layer_inputs — safe defaults
# ──────────────────────────────────────────────────────────────────────────


def test_compute_gemma4_per_layer_inputs_noop_for_non_gemma4():
    """On a runtime whose ``_decoder_family != "gemma4"``, the helper is
    a no-op — returns None regardless of what token ids are passed in."""
    from peer.model_shard import PyTorchRuntime

    # Bypass __init__ and set only the fields the method reads.
    rt = PyTorchRuntime.__new__(PyTorchRuntime)
    rt._decoder_family = "qwen_llama"
    rt._gemma4_text_model = None
    rt._per_layer_embed = None
    rt._per_layer_proj = None

    assert rt._compute_gemma4_per_layer_inputs([1, 2, 3]) is None
    assert rt._compute_gemma4_per_layer_inputs(None) is None


def test_compute_gemma4_per_layer_inputs_returns_none_without_text_model():
    """Even when the family is set to ``gemma4``, if the text model
    reference wasn't captured (e.g. an older checkpoint shape) the helper
    must return None without raising."""
    from peer.model_shard import PyTorchRuntime

    rt = PyTorchRuntime.__new__(PyTorchRuntime)
    rt._decoder_family = "gemma4"
    rt._gemma4_text_model = None
    rt._per_layer_embed = object()
    rt._per_layer_proj = object()

    assert rt._compute_gemma4_per_layer_inputs([1, 2, 3]) is None


def test_compute_gemma4_per_layer_inputs_empty_token_ids():
    """Empty token IDs → None (and no exception)."""
    from peer.model_shard import PyTorchRuntime

    rt = PyTorchRuntime.__new__(PyTorchRuntime)
    rt._decoder_family = "gemma4"
    rt._gemma4_text_model = object()  # would otherwise be truthy
    rt._per_layer_embed = object()
    rt._per_layer_proj = object()

    assert rt._compute_gemma4_per_layer_inputs([]) is None
    assert rt._compute_gemma4_per_layer_inputs(None) is None
