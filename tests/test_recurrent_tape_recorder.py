# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Phase 2b — recurrent tape recorder tests.

Validates the recorder against a synthetic ``nn.Module`` that
mimics the Qwen3NextGatedDeltaNet attribute layout. These tests
PIN the contract the recorder requires from upstream; a transformers
upgrade that renames sub-modules will fail here with an actionable
``LayerLayoutError`` before any inference happens.

The recorder code under test is also exercised end-to-end against
the rollback strategy: snapshot → record innovations → replay →
assert state matches autoregressive baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from peer.kv_rollback import TapeReplayRecurrentRollback
from peer.recurrent_tape_recorder import (
    LayerLayoutError,
    RecurrentTapeRecorder,
    probe_layer_kind,
    probe_qwen3_next_layer,
    replay_step_qwen3_next,
)


# ── Synthetic Qwen3NextGatedDeltaNet stand-in ──────────────────────────


class _FakeProj:
    """Stand-in for an ``nn.Linear`` projection. We only check
    presence-of-attribute, not behaviour."""


@dataclass
class _FakeQwen3NextCache:
    """Mimics Qwen3NextDynamicCache enough for the recorder to
    address per-layer state."""

    recurrent_states: list = field(default_factory=list)
    conv_states: list = field(default_factory=list)


class Qwen3NextGatedDeltaNet:
    """Synthetic class with the canonical attribute layout. The
    name MUST match the constant in
    ``peer/recurrent_tape_recorder.py::_QWEN3_NEXT_GATEDDELTANET_CLASSES``
    so the probe accepts it. Any rename in this test class is a
    deliberate signal to the recorder's class-name allowlist."""

    def __init__(self, layer_idx: int):
        self.in_proj_qkvz = _FakeProj()
        self.in_proj_ba = _FakeProj()
        self.layer_idx = layer_idx

    def __call__(self, hidden_states, *, cache_params):
        """Synthetic forward: writes a deterministic state update
        to cache so the recorder's replay can be validated.

        For a single-position innovation, the new recurrent state
        is ``old_state + hidden_states.sum()`` and the new conv
        state is the input itself. Trivial and deterministic so we
        can hand-verify replay.
        """
        old_rec = cache_params.recurrent_states[self.layer_idx]
        cache_params.recurrent_states[self.layer_idx] = (
            old_rec + float(_sum(hidden_states))
        )
        cache_params.conv_states[self.layer_idx] = hidden_states


def _sum(hidden_slice):
    """Backend-neutral sum() for the synthetic forward. Real
    runtime uses torch.sum / mx.sum; here we just walk the
    nested list / tensor."""
    if isinstance(hidden_slice, (int, float)):
        return hidden_slice
    if hasattr(hidden_slice, "sum"):
        return float(hidden_slice.sum())
    if hasattr(hidden_slice, "__iter__"):
        return sum(_sum(x) for x in hidden_slice)
    return float(hidden_slice)


# ── Layer probing ──────────────────────────────────────────────────────


def test_probe_layer_kind_qwen3_next():
    layer = Qwen3NextGatedDeltaNet(layer_idx=0)
    assert probe_layer_kind(layer) == "qwen3_next"


def test_probe_layer_kind_unknown():
    class SomeOtherLayer:
        pass
    assert probe_layer_kind(SomeOtherLayer()) == "unknown"


def test_probe_qwen3_next_accepts_canonical_layout():
    layer = Qwen3NextGatedDeltaNet(layer_idx=3)
    # Must not raise.
    probe_qwen3_next_layer(layer)


def test_probe_qwen3_next_rejects_unknown_class():
    class Wrong:
        in_proj_qkvz = _FakeProj()
        in_proj_ba = _FakeProj()
        layer_idx = 0
    with pytest.raises(LayerLayoutError) as exc:
        probe_qwen3_next_layer(Wrong())
    assert exc.value.reason == "unsupported_class"


def test_probe_qwen3_next_rejects_missing_projection():
    """If transformers ever renames ``in_proj_qkvz`` we want to
    fail at startup, not produce silently-wrong rollbacks at
    inference time."""
    layer = Qwen3NextGatedDeltaNet(layer_idx=0)
    del layer.in_proj_qkvz
    with pytest.raises(LayerLayoutError) as exc:
        probe_qwen3_next_layer(layer)
    assert exc.value.reason == "missing_projections"
    assert "in_proj_qkvz" in str(exc.value)


def test_probe_qwen3_next_rejects_missing_layer_idx():
    layer = Qwen3NextGatedDeltaNet(layer_idx=0)
    del layer.layer_idx
    with pytest.raises(LayerLayoutError) as exc:
        probe_qwen3_next_layer(layer)
    assert exc.value.reason == "missing_layer_idx"


def test_probe_qwen3_next_rejects_missing_in_proj_ba():
    layer = Qwen3NextGatedDeltaNet(layer_idx=0)
    del layer.in_proj_ba
    with pytest.raises(LayerLayoutError) as exc:
        probe_qwen3_next_layer(layer)
    assert exc.value.reason == "missing_projections"


# ── Recorder lifecycle ────────────────────────────────────────────────


def test_recorder_snapshot_then_append_then_to_kv_state():
    """Happy path — verify-time recording matches the kv_state
    contract that TapeReplayRecurrentRollback expects."""
    cache = _FakeQwen3NextCache(
        recurrent_states=[10.0, 20.0, 30.0],
        conv_states=[1.0, 2.0, 3.0],
    )
    layer = Qwen3NextGatedDeltaNet(layer_idx=1)
    recorder = RecurrentTapeRecorder(
        session_id="s1", layer_idx=1, block_start_pos=100,
    )
    recorder.snapshot_pre_state(layer, cache)
    recorder.append_innovation(5.0)
    recorder.append_innovation(7.0)

    replay = replay_step_qwen3_next(
        layer=layer, cache=cache, layer_idx=1,
    )
    kv = recorder.to_kv_state(replay_step=replay)

    assert kv["block_start_pos"] == 100
    assert kv["pre_block_state"] == {"recurrent": 20.0, "conv": 2.0}
    assert kv["innovations"] == [5.0, 7.0]
    assert callable(kv["replay_step"])


def test_recorder_snapshot_is_idempotent_within_block():
    """A second snapshot within the same block must NOT overwrite
    the first — would corrupt the rollback baseline."""
    cache = _FakeQwen3NextCache(
        recurrent_states=[10.0], conv_states=[1.0],
    )
    layer = Qwen3NextGatedDeltaNet(layer_idx=0)
    recorder = RecurrentTapeRecorder(
        session_id="s", layer_idx=0, block_start_pos=0,
    )
    recorder.snapshot_pre_state(layer, cache)
    # Cache mutates as the verify forward proceeds.
    cache.recurrent_states[0] = 999.0
    recorder.snapshot_pre_state(layer, cache)   # second call no-ops
    assert recorder.pre_block_state == {"recurrent": 10.0, "conv": 1.0}


def test_recorder_to_kv_state_without_snapshot_raises():
    """A call to to_kv_state before snapshot_pre_state means the
    verify forward never entered this layer — surface as
    LayerLayoutError(reason='missing_cache') rather than producing
    a None pre_block_state that fails opaquely later."""
    recorder = RecurrentTapeRecorder(
        session_id="s", layer_idx=2, block_start_pos=0,
    )
    with pytest.raises(LayerLayoutError) as exc:
        recorder.to_kv_state(replay_step=lambda S, i: S)
    assert exc.value.reason == "missing_cache"


def test_recorder_resolves_cache_shape_mismatch():
    """A cache without recurrent_states / conv_states must surface
    as cache_shape_mismatch — common cause is passing the wrong
    cache type into the verify forward."""
    @dataclass
    class _BadCache:
        nope: int = 0
    layer = Qwen3NextGatedDeltaNet(layer_idx=0)
    recorder = RecurrentTapeRecorder(
        session_id="s", layer_idx=0, block_start_pos=0,
    )
    with pytest.raises(LayerLayoutError) as exc:
        recorder.snapshot_pre_state(layer, _BadCache())
    assert exc.value.reason == "cache_shape_mismatch"


# ── End-to-end: recorder + replay reproduce verify-forward state ───────


def test_recorder_replay_matches_autoregressive_baseline():
    """The headline integration test for layer instrumentation:

    1. Set up a synthetic layer + cache.
    2. Run the verify forward over 5 innovations, recording them.
    3. Compute the autoregressive baseline state by running the
       layer over each innovation in order from the same starting
       cache.
    4. Run the recorder's tape through the rollback strategy at
       every K from 0 to 5.
    5. Assert: rolled-back state at K equals AR baseline at K.

    Bit-identical replay because the synthetic layer is
    deterministic and we feed the SAME innovations through both
    paths.
    """
    # ── Setup ──
    cache = _FakeQwen3NextCache(
        recurrent_states=[100.0],
        conv_states=[0.0],
    )
    layer = Qwen3NextGatedDeltaNet(layer_idx=0)
    block_start_pos = 200
    innovations = [1.0, 2.0, 3.0, 4.0, 5.0]

    # ── Verify forward: record while running ──
    recorder = RecurrentTapeRecorder(
        session_id="s", layer_idx=0,
        block_start_pos=block_start_pos,
    )
    recorder.snapshot_pre_state(layer, cache)
    for innov in innovations:
        recorder.append_innovation(innov)
        layer(innov, cache_params=cache)

    # ── AR baseline: run from the same starting cache ──
    def ar_baseline(K: int) -> tuple[float, float]:
        ar_cache = _FakeQwen3NextCache(
            recurrent_states=[100.0],
            conv_states=[0.0],
        )
        for i in range(K):
            layer(innovations[i], cache_params=ar_cache)
        return ar_cache.recurrent_states[0], ar_cache.conv_states[0]

    # ── Rollback path: at each K, replay first K innovations ──
    replay = replay_step_qwen3_next(
        layer=layer, cache=cache, layer_idx=0,
    )
    kv = recorder.to_kv_state(replay_step=replay)

    for K in range(0, len(innovations) + 1):
        rolled = TapeReplayRecurrentRollback().rollback_to(
            kv, target_len=block_start_pos + K,
        )
        baseline_rec, baseline_conv = ar_baseline(K)
        assert rolled["live_state"]["recurrent"] == baseline_rec, (
            f"K={K}: recurrent mismatch — "
            f"rolled={rolled['live_state']['recurrent']} "
            f"baseline={baseline_rec}"
        )
        assert rolled["live_state"]["conv"] == baseline_conv, (
            f"K={K}: conv mismatch — "
            f"rolled={rolled['live_state']['conv']} "
            f"baseline={baseline_conv}"
        )


def test_recorder_innovation_detach_clone():
    """Innovations are .detach().clone()'d on append so the recorder
    doesn't keep autograd references or share storage with the
    layer's internal buffers. We model this by checking that
    mutating the original input after append doesn't change the
    recorded innovation."""
    class _MutableTensor:
        def __init__(self, value):
            self.value = value
        def detach(self):
            return _MutableTensor(self.value)
        def clone(self):
            return _MutableTensor(self.value)

    cache = _FakeQwen3NextCache(
        recurrent_states=[0.0], conv_states=[0.0],
    )
    layer = Qwen3NextGatedDeltaNet(layer_idx=0)
    recorder = RecurrentTapeRecorder(
        session_id="s", layer_idx=0, block_start_pos=0,
    )
    recorder.snapshot_pre_state(layer, cache)

    innov = _MutableTensor(value=42)
    recorder.append_innovation(innov)
    innov.value = 9999   # mutate after append
    assert recorder.innovations[0].value == 42   # clone protected
