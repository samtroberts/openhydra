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


# ── TapeReplayRecurrentRollback (Strategy B: snapshot-once + replay) ────


def _scalar_replay_step(S, beta, k, v):
    """Test-only replay step that works on plain scalars/lists.

    For unit tests we model S as a single float and (β, k, v) as
    floats; eq. (1) collapses to ``S' = (1 - β·k·k)·S + β·v·k``.
    The dispatcher uses this as the ``replay_step`` callable so we
    can validate the math without standing up a tensor library.
    """
    return (1.0 - float(beta) * float(k) * float(k)) * float(S) + float(beta) * float(v) * float(k)


def _make_tape(pre_state, innovations, block_start_pos=0, replay=_scalar_replay_step):
    return {
        "live_state": pre_state,
        "pre_block_state": pre_state,
        "innovations": list(innovations),
        "block_start_pos": int(block_start_pos),
        "replay_step": replay,
    }


def test_tape_replay_zero_accepted_returns_pre_state():
    """target_len == block_start_pos = 0 drafts accepted → restore
    the pre-block snapshot exactly."""
    innovations = [(0.5, 1.0, 2.0), (0.3, 0.5, 1.5)]
    out = TapeReplayRecurrentRollback().rollback_to(
        _make_tape(pre_state=10.0, innovations=innovations, block_start_pos=100),
        target_len=100,
    )
    assert out["live_state"] == 10.0


def test_tape_replay_full_acceptance_replays_all_innovations():
    """All B drafts accepted → replay all B innovations from S_pre."""
    innovations = [(0.5, 1.0, 2.0), (0.3, 0.5, 1.5)]
    out = TapeReplayRecurrentRollback().rollback_to(
        _make_tape(pre_state=10.0, innovations=innovations, block_start_pos=100),
        target_len=102,   # 100 + 2 drafts accepted
    )
    # Manual replay:
    #  S_0 = 10.0
    #  S_1 = (1 - 0.5*1*1)*10.0 + 0.5*2*1     = 5.0 + 1.0 = 6.0
    #  S_2 = (1 - 0.3*0.5*0.5)*6.0 + 0.3*1.5*0.5 = (1 - 0.075)*6.0 + 0.225
    #      = 0.925*6.0 + 0.225 = 5.55 + 0.225 = 5.775
    assert abs(out["live_state"] - 5.775) < 1e-9


def test_tape_replay_partial_acceptance_replays_only_accepted():
    """Mid-block divergence: replay first K innovations, drop rest."""
    innovations = [(0.5, 1.0, 2.0), (0.3, 0.5, 1.5), (0.9, 0.1, 9.9)]
    out = TapeReplayRecurrentRollback().rollback_to(
        _make_tape(pre_state=10.0, innovations=innovations, block_start_pos=100),
        target_len=101,   # 1 draft accepted, last 2 rolled back
    )
    # S_1 = (1 - 0.5*1*1)*10.0 + 0.5*2*1 = 5.0 + 1.0 = 6.0
    assert abs(out["live_state"] - 6.0) < 1e-9


def test_tape_replay_preserves_tape_for_further_rollback():
    """The pre_block_state and innovations stay in the returned KV
    dict so a subsequent rollback (e.g. tree speculation in Phase 2c)
    can replay from the same baseline."""
    innovations = [(0.5, 1.0, 2.0), (0.3, 0.5, 1.5)]
    kv = _make_tape(pre_state=10.0, innovations=innovations, block_start_pos=100)
    out = TapeReplayRecurrentRollback().rollback_to(kv, target_len=101)
    assert out["pre_block_state"] == 10.0
    assert out["innovations"] == innovations
    assert out["block_start_pos"] == 100


def test_tape_replay_target_predates_block_start_raises():
    """target_len < block_start_pos = the rollback wants a state
    from BEFORE this block was even entered. Caller bug; fail loud."""
    innovations = [(0.5, 1.0, 2.0)]
    with pytest.raises(RollbackError, match="predates block_start_pos"):
        TapeReplayRecurrentRollback().rollback_to(
            _make_tape(pre_state=10.0, innovations=innovations, block_start_pos=100),
            target_len=99,
        )


def test_tape_replay_target_beyond_block_raises():
    """target_len > block_start + len(innovations) = nothing in the
    tape. Caller bug; fail loud."""
    innovations = [(0.5, 1.0, 2.0)]
    with pytest.raises(RollbackError, match="tape only has"):
        TapeReplayRecurrentRollback().rollback_to(
            _make_tape(pre_state=10.0, innovations=innovations, block_start_pos=100),
            target_len=110,
        )


def test_tape_replay_requires_dict_kv():
    with pytest.raises(RollbackError, match="dict-shaped"):
        TapeReplayRecurrentRollback().rollback_to(
            np.zeros((1, 4, 16, 8)), target_len=4,
        )


def test_tape_replay_requires_all_required_keys():
    """Each required field must be present; caller bug otherwise."""
    full = _make_tape(pre_state=1.0, innovations=[(0.5, 1.0, 2.0)],
                      block_start_pos=0)
    for missing in ("pre_block_state", "innovations", "block_start_pos", "replay_step"):
        partial = {k: v for k, v in full.items() if k != missing}
        with pytest.raises(RollbackError, match=missing):
            TapeReplayRecurrentRollback().rollback_to(partial, target_len=0)


def test_tape_replay_rejects_non_callable_replay_step():
    kv = _make_tape(pre_state=1.0, innovations=[(0.5, 1.0, 2.0)],
                    block_start_pos=0, replay="not callable")
    with pytest.raises(RollbackError, match="callable"):
        TapeReplayRecurrentRollback().rollback_to(kv, target_len=0)


# ── Byte-equivalence: replay == autoregressive forward ─────────────────


def test_hybrid_rollback_byte_equivalence_attention():
    """Attention in-place truncation must produce a state byte-identical
    to a hypothetical AR forward over the accepted prefix only.

    Because attention K/V is purely concatenative (eq. trivial: K_t is
    just [k_0, ..., k_{t-1}] stacked), truncation IS the same operation
    as AR forward over the prefix — both produce a [K, ..., :] slice of
    the same elements. We verify by constructing K, V from per-position
    keys/values and asserting the truncated tensor equals the
    AR-built tensor.
    """
    # AR forward emits k, v per position.
    per_pos_keys = [np.full((4, 8), float(i), dtype=np.float32) for i in range(16)]
    per_pos_vals = [np.full((4, 8), float(i + 100), dtype=np.float32) for i in range(16)]

    # Verify-time cache after full block: stack all 16 positions.
    K_verify = np.stack(per_pos_keys, axis=1)   # [4, 16, 8]
    V_verify = np.stack(per_pos_vals, axis=1)
    cache_verify = {"keys": K_verify, "values": V_verify}

    # AR forward over accepted prefix (K=5).
    K_ar = np.stack(per_pos_keys[:5], axis=1)
    V_ar = np.stack(per_pos_vals[:5], axis=1)

    # Truncate verify-time cache to K=5.
    cache_rolled = TruncateAttentionRollback(seq_axis=1).rollback_to(
        cache_verify, target_len=5,
    )

    # Bit-identical.
    np.testing.assert_array_equal(cache_rolled["keys"], K_ar)
    np.testing.assert_array_equal(cache_rolled["values"], V_ar)


def test_hybrid_rollback_byte_equivalence_recurrent():
    """Tape-replay must produce a state bit-identical to autoregressive
    forward over the accepted prefix.

    Construction: pick a synthetic (S_pre, innovations) tape; compute
    AR-baseline S_K by running the replay_step for K steps from S_pre;
    compute rollback S_K via TapeReplayRecurrentRollback.rollback_to;
    assert they match exactly.

    This is the headline lossless guarantee for the recurrent path.
    """
    S_pre = 7.0
    innovations = [
        (0.5, 1.0, 2.0),
        (0.3, 0.5, 1.5),
        (0.7, 0.2, 4.0),
        (0.1, 0.9, 0.5),
        (0.4, 0.3, 3.3),
    ]
    block_start = 200

    # AR baseline: walk replay_step K times manually.
    def ar_baseline(K: int) -> float:
        S = S_pre
        for i in range(K):
            beta, k, v = innovations[i]
            S = _scalar_replay_step(S, beta, k, v)
        return S

    # Rollback via the strategy.
    def rolled(K: int) -> float:
        kv = _make_tape(pre_state=S_pre, innovations=innovations,
                        block_start_pos=block_start)
        out = TapeReplayRecurrentRollback().rollback_to(
            kv, target_len=block_start + K,
        )
        return out["live_state"]

    # Every accepted-prefix length must match the AR baseline exactly.
    for K in range(0, len(innovations) + 1):
        ar_S = ar_baseline(K)
        rolled_S = rolled(K)
        assert ar_S == rolled_S, (
            f"byte-equivalence broken at K={K}: ar={ar_S} rolled={rolled_S}"
        )


def test_hybrid_rollback_byte_equivalence_repeated_rollbacks():
    """Repeated rollbacks within the same block (Phase 2c tree
    speculation reuses this entry) must each produce the AR-equivalent
    state because pre_block_state and innovations are preserved
    untouched across calls.
    """
    S_pre = 7.0
    innovations = [(0.5, 1.0, 2.0), (0.3, 0.5, 1.5), (0.7, 0.2, 4.0)]
    kv = _make_tape(pre_state=S_pre, innovations=innovations, block_start_pos=0)

    out_3 = TapeReplayRecurrentRollback().rollback_to(kv, target_len=3)
    out_2 = TapeReplayRecurrentRollback().rollback_to(out_3, target_len=2)
    out_1 = TapeReplayRecurrentRollback().rollback_to(out_2, target_len=1)
    out_0 = TapeReplayRecurrentRollback().rollback_to(out_1, target_len=0)

    # Each step must equal the corresponding AR-baseline.
    S = S_pre
    expected = [S]
    for beta, k, v in innovations:
        S = _scalar_replay_step(S, beta, k, v)
        expected.append(S)

    assert out_0["live_state"] == expected[0]
    assert out_1["live_state"] == expected[1]
    assert out_2["live_state"] == expected[2]
    assert out_3["live_state"] == expected[3]


# ── rollback_session_kv dispatcher ──────────────────────────────────────

def test_dispatcher_routes_per_layer_type():
    """Hybrid Qwen3.5-style shard: layers 0,2 attention; layer 1
    GatedDeltaNet. Each gets its own strategy."""
    K_attn = np.arange(16 * 8).reshape(16, 8).astype(np.float32)
    pre_state = 7.0
    innovations = [(0.5, 1.0, 2.0)] * 16   # 16 drafts

    # Build the dispatch plan with the appropriate seq_axis for our
    # 2-D attention test tensor.
    plan = SessionRollbackPlan(session_id="s1")
    plan.entries = [
        LayerRollbackEntry(0, KVKind.ATTENTION, TruncateAttentionRollback(seq_axis=0)),
        LayerRollbackEntry(1, KVKind.RECURRENT, TapeReplayRecurrentRollback()),
        LayerRollbackEntry(2, KVKind.ATTENTION, TruncateAttentionRollback(seq_axis=0)),
    ]

    layer_kv = {
        0: {"keys": K_attn, "values": K_attn},
        1: _make_tape(
            pre_state=pre_state, innovations=innovations,
            block_start_pos=0,
        ),
        2: {"keys": K_attn, "values": K_attn},
    }

    out = rollback_session_kv(plan, layer_kv, target_len=4)
    # Attention layers truncated to 4 entries.
    assert out[0]["keys"].shape == (4, 8)
    assert out[2]["keys"].shape == (4, 8)
    # Recurrent layer rolled back via tape replay (4 innovations).
    expected_S = pre_state
    for i in range(4):
        beta, k, v = innovations[i]
        expected_S = _scalar_replay_step(expected_S, beta, k, v)
    assert out[1]["live_state"] == expected_S


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
