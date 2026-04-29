# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Phase 2b — KV rollback strategies (per-layer-type).

The riskiest engineering surface in Phase 2b: getting rollback wrong
silently produces incoherent inference output. Tests pin every code
path including the failure modes (replay-without-tape, target_len
beyond tape length, unsupported kv shapes).

Backend-neutral: uses numpy arrays so the tests don't pull torch or
mlx. The strategies are duck-typed so the same code paths exercise
the production tensor shapes.
"""

from __future__ import annotations

import numpy as np
import pytest

from peer.kv_rollback import (
    KVKind,
    LayerRollbackEntry,
    RollbackError,
    SessionRollbackPlan,
    TapeReplayRecurrentRollback,
    TruncateAttentionRollback,
    classify_layer_kind,
    rollback_session_kv,
)


# ── classify_layer_kind ────────────────────────────────────────────────

def test_classify_explicit_kv_kind_attribute_wins():
    """A layer can stamp ``kv_kind`` for forward compatibility — the
    classifier honours that and skips the heuristic."""
    class _CustomLayer:
        kv_kind = KVKind.RECURRENT
    assert classify_layer_kind(_CustomLayer()) == KVKind.RECURRENT


def test_classify_attention_class_name():
    class Qwen3Attention:  # canonical HF/MLX naming
        pass
    assert classify_layer_kind(Qwen3Attention()) == KVKind.ATTENTION


def test_classify_gateddeltanet_class_name():
    """Qwen3.5 hybrid-Mamba layer recognised by name."""
    class Qwen3GatedDeltaNet:
        pass
    assert classify_layer_kind(Qwen3GatedDeltaNet()) == KVKind.RECURRENT


def test_classify_mamba_class_name():
    class MambaLayer:
        pass
    assert classify_layer_kind(MambaLayer()) == KVKind.RECURRENT


def test_classify_unknown_falls_back_to_attention():
    """Unknown layers default to ATTENTION (the safer-but-slower
    default — truncate is O(1) and never wrong; tape-replay would
    panic on a missing tape)."""
    class Unknown:
        pass
    assert classify_layer_kind(Unknown()) == KVKind.ATTENTION


def test_classify_explicit_none_kind_passes_through():
    """Layers that hold no KV (norms, lm_head) can stamp NONE so
    the dispatcher skips them entirely."""
    class FinalNorm:
        kv_kind = KVKind.NONE
    assert classify_layer_kind(FinalNorm()) == KVKind.NONE


# ── TruncateAttentionRollback — sliceable shapes ────────────────────────

def test_truncate_attention_slices_dict_keys_values():
    """HF/MLX convention: cache is {"keys": K, "values": V} with
    sequence axis at -2 (shape [batch, heads, seq, head_dim])."""
    K = np.arange(2 * 4 * 16 * 8).reshape(2, 4, 16, 8).astype(np.float32)
    V = np.arange(2 * 4 * 16 * 8).reshape(2, 4, 16, 8).astype(np.float32)
    cache = {"keys": K, "values": V}

    out = TruncateAttentionRollback().rollback_to(cache, target_len=10)
    assert out["keys"].shape == (2, 4, 10, 8)
    assert out["values"].shape == (2, 4, 10, 8)
    # Values past target_len must be dropped.
    np.testing.assert_array_equal(out["keys"], K[:, :, :10, :])


def test_truncate_attention_slices_tuple_kv():
    """Some runtimes return (K, V) tuples — supported."""
    K = np.zeros((1, 4, 16, 8), dtype=np.float32)
    V = np.ones((1, 4, 16, 8), dtype=np.float32)
    out = TruncateAttentionRollback().rollback_to((K, V), target_len=8)
    assert isinstance(out, tuple) and len(out) == 2
    assert out[0].shape == (1, 4, 8, 8)
    assert out[1].shape == (1, 4, 8, 8)


def test_truncate_attention_handles_short_aliases():
    """Some caches use {"k": K, "v": V} — also supported."""
    K = np.arange(16 * 8).reshape(16, 8).astype(np.float32)
    V = np.arange(16 * 8).reshape(16, 8).astype(np.float32)
    out = TruncateAttentionRollback(seq_axis=0).rollback_to(
        {"k": K, "v": V}, target_len=4,
    )
    assert out["k"].shape == (4, 8)
    assert out["v"].shape == (4, 8)


def test_truncate_attention_to_zero_clears_cache():
    """target_len=0 = drop everything (next forward starts fresh)."""
    K = np.zeros((1, 4, 16, 8), dtype=np.float32)
    out = TruncateAttentionRollback().rollback_to({"keys": K, "values": K},
                                                   target_len=0)
    assert out["keys"].shape == (1, 4, 0, 8)


def test_truncate_attention_negative_target_len_raises():
    K = np.zeros((1, 4, 16, 8), dtype=np.float32)
    with pytest.raises(RollbackError, match="target_len"):
        TruncateAttentionRollback().rollback_to({"keys": K, "values": K},
                                                 target_len=-1)


def test_truncate_attention_invalid_axis_raises():
    K = np.zeros((1, 4, 16, 8), dtype=np.float32)
    with pytest.raises(RollbackError, match="seq_axis"):
        TruncateAttentionRollback(seq_axis=10).rollback_to(
            {"keys": K, "values": K}, target_len=4,
        )


# ── TapeReplayRecurrentRollback ─────────────────────────────────────────

def test_tape_replay_returns_pre_state_at_target():
    """The accepted-prefix pre-state at index target_len is the
    state we want after rollback."""
    pre_states = [f"pre-{i}" for i in range(17)]   # 16 drafts + bonus
    tape = [f"innov-{i}" for i in range(16)]
    kv = {"state": "post-final", "tape": tape, "pre_states": pre_states}

    out = TapeReplayRecurrentRollback().rollback_to(kv, target_len=4)
    assert out["state"] == "pre-4"           # the snapshot we want
    assert out["tape"] == tape[:4]           # tape truncated
    assert out["pre_states"] == pre_states[:5]   # +1 to allow next step


def test_tape_replay_to_zero_returns_pre_prefix_state():
    """target_len=0 = no drafts accepted = recurrent state at the
    prefix-end position (before drafting started)."""
    pre_states = ["pre-prefix"] + [f"pre-{i}" for i in range(16)]
    out = TapeReplayRecurrentRollback().rollback_to(
        {"state": "post-all", "tape": ["t0"] * 16, "pre_states": pre_states},
        target_len=0,
    )
    assert out["state"] == "pre-prefix"
    assert out["tape"] == []


def test_tape_replay_full_acceptance_keeps_state():
    """When all 16 drafts are accepted, target_len equals the tape
    length — the rollback is effectively a no-op (just keep current
    state)."""
    pre_states = [f"pre-{i}" for i in range(16)]   # 16 entries
    tape = [f"innov-{i}" for i in range(16)]
    kv = {"state": "post-final", "tape": tape, "pre_states": pre_states}

    # target_len == len(pre_states) means "accept everything in tape"
    out = TapeReplayRecurrentRollback().rollback_to(kv, target_len=16)
    assert out["state"] == "post-final"


def test_tape_replay_target_len_beyond_tape_raises():
    pre_states = [f"pre-{i}" for i in range(8)]
    kv = {"state": "x", "tape": ["t"] * 8, "pre_states": pre_states}
    with pytest.raises(RollbackError, match="exceeds tape length"):
        TapeReplayRecurrentRollback().rollback_to(kv, target_len=12)


def test_tape_replay_requires_dict_kv():
    with pytest.raises(RollbackError, match="dict-shaped"):
        TapeReplayRecurrentRollback().rollback_to(
            np.zeros((1, 4, 16, 8)), target_len=4,
        )


def test_tape_replay_requires_pre_states_list():
    with pytest.raises(RollbackError, match="pre_states"):
        TapeReplayRecurrentRollback().rollback_to(
            {"state": "x", "tape": []}, target_len=0,
        )


# ── rollback_session_kv dispatcher ──────────────────────────────────────

def test_dispatcher_routes_per_layer_type():
    """Hybrid Qwen3.5-style shard: layers 0,2 attention; layer 1
    GatedDeltaNet. Each gets its own strategy."""
    plan = SessionRollbackPlan(session_id="s1")
    plan.add(0, KVKind.ATTENTION, TruncateAttentionRollback())
    plan.add(1, KVKind.RECURRENT, TapeReplayRecurrentRollback())
    plan.add(2, KVKind.ATTENTION, TruncateAttentionRollback())

    K_attn = np.arange(16 * 8).reshape(16, 8).astype(np.float32)
    # 17 entries (16 drafts + 1 bonus position) = same indexing as
    # the standalone tape-replay test: pre[i] is the snapshot before
    # step i, so pre[target_len] is the rollback target.
    pre = [f"pre-{i}" for i in range(17)]
    layer_kv = {
        0: {"keys": K_attn, "values": K_attn},
        1: {"state": "post", "tape": ["t"] * 16, "pre_states": pre},
        2: {"keys": K_attn, "values": K_attn},
    }

    # Use seq_axis=0 since our attention K is 2D in this test.
    plan.entries = [
        LayerRollbackEntry(0, KVKind.ATTENTION, TruncateAttentionRollback(seq_axis=0)),
        LayerRollbackEntry(1, KVKind.RECURRENT, TapeReplayRecurrentRollback()),
        LayerRollbackEntry(2, KVKind.ATTENTION, TruncateAttentionRollback(seq_axis=0)),
    ]

    out = rollback_session_kv(plan, layer_kv, target_len=4)
    # Attention layers truncated.
    assert out[0]["keys"].shape == (4, 8)
    assert out[2]["keys"].shape == (4, 8)
    # Recurrent layer rolled back via tape replay.
    assert out[1]["state"] == "pre-4"
    assert out[1]["tape"] == ["t"] * 4


def test_dispatcher_skips_layers_not_yet_warmed_up():
    """First ForwardRequest after session creation may not have
    populated all layer caches yet. Skip absent layers — don't crash."""
    plan = SessionRollbackPlan(session_id="s1")
    plan.add(0, KVKind.ATTENTION, TruncateAttentionRollback(seq_axis=0))
    plan.add(1, KVKind.ATTENTION, TruncateAttentionRollback(seq_axis=0))

    K = np.zeros((16, 8))
    layer_kv = {0: {"keys": K, "values": K}}   # layer 1 not in cache yet
    out = rollback_session_kv(plan, layer_kv, target_len=4)
    assert 0 in out
    assert out[0]["keys"].shape == (4, 8)
    assert 1 not in out


def test_dispatcher_passes_through_layers_not_in_plan():
    """Defence in depth: layer present in cache but absent from plan
    is passed through unchanged (signals a bug, but doesn't drop data)."""
    plan = SessionRollbackPlan(session_id="s1")
    plan.add(0, KVKind.ATTENTION, TruncateAttentionRollback(seq_axis=0))
    K = np.ones((16, 8))
    layer_kv = {
        0: {"keys": K, "values": K},
        99: {"keys": K, "values": K},   # not in plan
    }
    out = rollback_session_kv(plan, layer_kv, target_len=4)
    assert 99 in out
    np.testing.assert_array_equal(out[99]["keys"], K)   # unchanged


def test_dispatcher_skips_kv_kind_none_layers():
    plan = SessionRollbackPlan(session_id="s1")
    plan.add(0, KVKind.NONE, TruncateAttentionRollback(seq_axis=0))
    layer_kv = {0: "no-kv-here"}
    out = rollback_session_kv(plan, layer_kv, target_len=4)
    assert out[0] == "no-kv-here"   # passed through unchanged


def test_dispatcher_negative_target_len_raises():
    plan = SessionRollbackPlan(session_id="s1")
    with pytest.raises(RollbackError, match="target_len"):
        rollback_session_kv(plan, {}, target_len=-5)


def test_dispatcher_propagates_layer_failure_with_index():
    """A single-layer failure annotates the exception with the
    failing layer index for diagnostics."""
    plan = SessionRollbackPlan(session_id="s1")
    plan.add(7, KVKind.RECURRENT, TapeReplayRecurrentRollback())
    layer_kv = {7: {"state": "x", "tape": [], "pre_states": []}}   # empty pre_states
    with pytest.raises(RollbackError) as exc:
        rollback_session_kv(plan, layer_kv, target_len=4)
    assert exc.value.layer_index == 7


def test_dispatcher_target_len_zero_clears_all_attention_caches():
    """target_len=0 = drop everything. Every attention layer gets
    a length-0 tensor; recurrent layers get the prefix-end state."""
    plan = SessionRollbackPlan(session_id="s1")
    plan.add(0, KVKind.ATTENTION, TruncateAttentionRollback(seq_axis=0))
    K = np.ones((16, 8))
    out = rollback_session_kv(plan, {0: {"keys": K, "values": K}}, target_len=0)
    assert out[0]["keys"].shape == (0, 8)
