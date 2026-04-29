# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Phase 2b — block-verify algorithm tests.

Pins the lossless guarantee: ``select_accepted_prefix`` must produce
exactly the byte-equivalent output an autoregressive greedy decoder
would, for every input pattern from full acceptance to full
rejection.

Plus a minimal HeadSampler.verify_block integration test using a
fake runtime that exposes ``apply_final_head_block``, so the
orchestrator path is covered without pulling MLX/PyTorch.
"""

from __future__ import annotations

import pytest

from coordinator.head_sampler import (
    DecodeConfig,
    HeadSampler,
    select_accepted_prefix,
)


# ── select_accepted_prefix — pure algorithm ─────────────────────────────

def test_full_acceptance_returns_n_drafts_plus_bonus():
    """Every draft matches the target argmax → accept all, emit
    block_size + 1 tokens. Maximum speedup case."""
    drafts = [10, 20, 30, 40]
    argmax = [10, 20, 30, 40, 99]    # +1 for bonus at position 4
    accepted, bonus = select_accepted_prefix(
        argmax_per_position=argmax, draft_token_ids=drafts,
    )
    assert accepted == 4
    assert bonus == 99


def test_partial_acceptance_finds_first_divergence():
    """Drafts match through position 2, diverge at position 3.
    accepted_len == 3 (positions 0,1,2 accepted); bonus is the
    target's argmax at position 3."""
    drafts = [10, 20, 30, 99, 88]    # draft[3]=99, draft[4]=88 — wrong
    argmax = [10, 20, 30, 40, 50, 60]   # target says 40 at pos 3
    accepted, bonus = select_accepted_prefix(
        argmax_per_position=argmax, draft_token_ids=drafts,
    )
    assert accepted == 3
    assert bonus == 40   # target's argmax at the divergence position


def test_first_position_rejection_emits_one_token():
    """Drafter wrong from the start → accepted_len=0, only the
    bonus token is emitted (the target's argmax at pos 0).
    Equivalent to single-token autoregressive decoding."""
    drafts = [99, 99, 99, 99]
    argmax = [10, 20, 30, 40, 50]
    accepted, bonus = select_accepted_prefix(
        argmax_per_position=argmax, draft_token_ids=drafts,
    )
    assert accepted == 0
    assert bonus == 10
    # Total emitted = accepted_len + 1 = 1 token. Matches non-spec.


def test_byte_equivalence_to_autoregressive_under_full_acceptance():
    """Lossless guarantee: when every draft matches argmax, the
    emitted prefix is exactly what greedy autoregressive decoding
    would produce, token-by-token. This is the core DFlash claim."""
    # Simulated target argmax for a fixed prompt — what AR would
    # produce step-by-step.
    ar_output = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010,
                 1111, 1212, 1313, 1414, 1515, 1616, 1717]   # 17 tokens

    # Drafter happens to predict perfectly — an artificial best case.
    drafts = ar_output[:16]   # block_size=16 drafts

    # Target argmax for the verify pass: same as AR's output (because
    # at each position the target sees the same prefix the AR decoder
    # saw — the 16 accepted drafts plus the bonus at the end).
    argmax = ar_output   # 17 entries: 16 verify + 1 bonus

    accepted, bonus = select_accepted_prefix(
        argmax_per_position=argmax, draft_token_ids=drafts,
    )
    assert accepted == 16
    assert bonus == 1717
    # The full emitted stream: drafts[:accepted] + [bonus] ==
    # drafts[:16] + [1717] == 17 tokens identical to AR.
    emitted = drafts[:accepted] + [bonus]
    assert emitted == ar_output


def test_block_size_one_works():
    """Edge: a single-token draft block. accepted ∈ {0,1};
    bonus is always the target's argmax at the right position."""
    # Match
    a, b = select_accepted_prefix(
        argmax_per_position=[42, 99],
        draft_token_ids=[42],
    )
    assert (a, b) == (1, 99)
    # Mismatch
    a, b = select_accepted_prefix(
        argmax_per_position=[42, 99],
        draft_token_ids=[5],
    )
    assert (a, b) == (0, 42)


def test_block_size_thirty_two_works():
    """Phase 2b's hard upper bound. Should produce no off-by-one
    issues at the boundary."""
    drafts = list(range(100, 132))             # 32 drafts
    argmax = list(range(100, 132)) + [9999]    # 33 argmax (32 verify + bonus)
    a, b = select_accepted_prefix(
        argmax_per_position=argmax, draft_token_ids=drafts,
    )
    assert a == 32
    assert b == 9999


def test_empty_drafts_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        select_accepted_prefix(argmax_per_position=[1], draft_token_ids=[])


def test_argmax_length_must_be_drafts_plus_one():
    """Defence in depth: the +1 bonus position is required. Caller
    bug (forgot to feed the bonus position) = explicit error, not
    a wrong-output silent miss."""
    with pytest.raises(ValueError, match="length 5"):
        select_accepted_prefix(
            argmax_per_position=[1, 2, 3],   # only 3 entries
            draft_token_ids=[1, 2, 3, 4],    # 4 drafts → need 5 argmax
        )


def test_acceptance_stops_at_first_divergence_not_just_overall_mismatch():
    """Walk-and-stop semantics: a later match doesn't 'rescue' an
    earlier divergence. accept = longest prefix only."""
    drafts = [10, 99, 30, 40]    # diverges at position 1
    argmax = [10, 20, 30, 40, 50]
    a, b = select_accepted_prefix(
        argmax_per_position=argmax, draft_token_ids=drafts,
    )
    assert a == 1     # NOT 3 — position 1 stops the walk
    assert b == 20


# ── HeadSampler.verify_block integration with a fake runtime ────────────


class _FakeBlockRuntime:
    """Stand-in for an MLX/PyTorch runtime that exposes
    ``apply_final_head_block``. Returns a pre-canned argmax sequence
    so we can exercise HeadSampler.verify_block without backends."""

    def __init__(self, argmax_per_position: list[int]):
        self._canned = list(argmax_per_position)

    def apply_final_head_block(
        self, hidden_states_block, *, packed_bytes=None, **decode_kwargs,
    ):
        return list(self._canned)


def test_head_sampler_verify_block_orchestrates_runtime_and_algorithm():
    """End-to-end: runtime returns argmax, HeadSampler walks the
    prefix, returns (accepted_len, bonus_token)."""
    runtime = _FakeBlockRuntime(argmax_per_position=[10, 20, 30, 40, 99])
    sampler = HeadSampler(runtime=runtime, peer_id="test-peer")

    accepted, bonus = sampler.verify_block(
        hidden_states_block=object(),         # opaque to fake
        draft_token_ids=[10, 20, 30, 40],
        decode=DecodeConfig(),
    )
    assert accepted == 4
    assert bonus == 99


def test_head_sampler_verify_block_rejects_runtime_without_block_method():
    """Runtimes that haven't been upgraded to the block-decode path
    must produce a clear, actionable error pointing at Commit 7."""
    class _OldRuntime:
        def apply_final_head(self, *args, **kwargs):
            return 42   # has the single-token method only

    sampler = HeadSampler(runtime=_OldRuntime(), peer_id="old-peer")
    with pytest.raises(RuntimeError, match="Commit 7"):
        sampler.verify_block(
            hidden_states_block=object(),
            draft_token_ids=[1, 2, 3],
            decode=DecodeConfig(),
        )


def test_head_sampler_verify_block_rejects_empty_drafts():
    runtime = _FakeBlockRuntime(argmax_per_position=[42])
    sampler = HeadSampler(runtime=runtime, peer_id="test-peer")
    with pytest.raises(ValueError, match="non-empty"):
        sampler.verify_block(
            hidden_states_block=object(),
            draft_token_ids=[],
            decode=DecodeConfig(),
        )
