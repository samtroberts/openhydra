# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Phase 2b — DFlash Topology A driver tests.

Exercises the orchestration loop end-to-end against deterministic
mocks: a synthetic drafter, a fake ring transport that returns
pre-canned argmax sequences, and a verifier that runs the real
``select_accepted_prefix`` algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from coordinator.dflash_driver import (
    BlockVerifyResult,
    DFlashTopologyADriver,
)
from coordinator.head_sampler import select_accepted_prefix


# ── Test scaffolding ───────────────────────────────────────────────────


class _ScriptedDrafter:
    def __init__(self, blocks: list[list[int]]):
        self._blocks = list(blocks)
        self._i = 0

    def draft(self, prefix_token_ids: list[int]) -> list[int]:
        if self._i >= len(self._blocks):
            return [0] * len(self._blocks[0])
        out = list(self._blocks[self._i])
        self._i += 1
        return out


@dataclass
class _RecordedVerify:
    prefix_len: int
    draft_token_ids: list[int]
    kv_rollback_to: int


class _ScriptedTransport:
    def __init__(self, argmax_sequences: list[list[int]]):
        self._argmax = list(argmax_sequences)
        self._i = 0
        self.calls: list[_RecordedVerify] = []

    def verify(
        self, *, prefix_token_ids, draft_token_ids,
        kv_rollback_to, request_id, kv_session_id,
    ) -> Any:
        self.calls.append(_RecordedVerify(
            prefix_len=len(prefix_token_ids),
            draft_token_ids=list(draft_token_ids),
            kv_rollback_to=int(kv_rollback_to),
        ))
        argmax = self._argmax[self._i]
        self._i += 1
        return argmax


def _make_verifier():
    def _verify(hidden_states_block, draft_token_ids):
        return select_accepted_prefix(
            argmax_per_position=list(hidden_states_block),
            draft_token_ids=list(draft_token_ids),
        )
    return _verify


# ── Full acceptance ────────────────────────────────────────────────────

def test_full_acceptance_emits_seventeen_per_block():
    block = list(range(100, 116))
    argmax = list(range(100, 116)) + [9999]
    drafter = _ScriptedDrafter([block, [0] * 16])
    transport = _ScriptedTransport([argmax, [0] * 17])
    emitted: list[int] = []

    driver = DFlashTopologyADriver(
        drafter=drafter, verifier=_make_verifier(),
        transport=transport, emit=emitted.append,
        block_size=16, max_tokens=17,
    )
    stats = driver.run([1, 2, 3])
    assert emitted == block + [9999]
    assert stats.blocks == 1
    assert stats.tokens_emitted == 17
    assert stats.drafts_total == 16
    assert stats.drafts_accepted == 16


def test_first_position_rejection_emits_only_bonus():
    drafts_a = [99] * 16
    argmax_a = [10] + [11] * 15 + [12]
    drafts_b = [99] * 16
    argmax_b = [20] + [21] * 15 + [22]
    drafter = _ScriptedDrafter([drafts_a, drafts_b])
    transport = _ScriptedTransport([argmax_a, argmax_b])
    emitted: list[int] = []

    driver = DFlashTopologyADriver(
        drafter=drafter, verifier=_make_verifier(),
        transport=transport, emit=emitted.append,
        block_size=16, max_tokens=2,
    )
    stats = driver.run([1])
    assert emitted == [10, 20]
    assert stats.drafts_accepted == 0
    assert stats.acceptance_rate == 0.0


def test_mid_block_divergence_emits_accepted_plus_bonus():
    drafts = list(range(50, 58)) + [999] * 8
    argmax = list(range(50, 58)) + [88] + [99] * 8
    drafter = _ScriptedDrafter([drafts, [0] * 16])
    transport = _ScriptedTransport([argmax, [0] * 17])
    emitted: list[int] = []

    driver = DFlashTopologyADriver(
        drafter=drafter, verifier=_make_verifier(),
        transport=transport, emit=emitted.append,
        block_size=16, max_tokens=9,
    )
    stats = driver.run([])
    assert emitted == list(range(50, 58)) + [88]
    assert stats.tokens_emitted == 9
    assert stats.drafts_accepted == 8


def test_stop_token_in_emit_terminates():
    drafts = [10, 20, 999, 30, 40] + [0] * 11
    argmax = [10, 20, 999, 30, 40, 50] + [99] * 11
    drafter = _ScriptedDrafter([drafts])
    transport = _ScriptedTransport([argmax])
    emitted: list[int] = []

    driver = DFlashTopologyADriver(
        drafter=drafter, verifier=_make_verifier(),
        transport=transport, emit=emitted.append,
        block_size=16, max_tokens=100,
        stop_token_ids=frozenset({999}),
    )
    stats = driver.run([])
    assert emitted == [10, 20, 999]
    assert stats.blocks == 1


def test_max_tokens_truncates_final_block():
    drafts = list(range(100, 116))
    argmax = list(range(100, 116)) + [9999]
    drafter = _ScriptedDrafter([drafts])
    transport = _ScriptedTransport([argmax])
    emitted: list[int] = []

    driver = DFlashTopologyADriver(
        drafter=drafter, verifier=_make_verifier(),
        transport=transport, emit=emitted.append,
        block_size=16, max_tokens=5,
    )
    stats = driver.run([])
    assert emitted == [100, 101, 102, 103, 104]
    assert stats.tokens_emitted == 5


# ── kv_rollback_to advances correctly ──────────────────────────────────

def test_kv_rollback_to_advances_across_blocks():
    """The peer-side rollback target must equal the prefix length at
    the START of each verify block. The driver bookkeeping for this
    is critical — get it wrong and peers truncate their KV cache to
    the wrong position, producing incoherent output."""
    drafts1 = [10, 20, 30, 40] + [0] * 12
    argmax1 = [10, 20, 30, 99] + [88] * 13     # accept 3, bonus=99
    drafts2 = [50, 60] + [0] * 14
    argmax2 = [50, 77] + [66] * 15             # accept 1, bonus=77
    drafter = _ScriptedDrafter([drafts1, drafts2])
    transport = _ScriptedTransport([argmax1, argmax2])

    driver = DFlashTopologyADriver(
        drafter=drafter, verifier=_make_verifier(),
        transport=transport, emit=lambda t: None,
        block_size=16,
        # Stop after exactly the two scripted blocks' emits.
        max_tokens=6,
    )
    stats = driver.run(prompt_token_ids=[1, 2, 3])

    # Block 1: rollback target = 3 (prompt length).
    assert transport.calls[0].kv_rollback_to == 3
    # Block 1 emits 4 tokens (3 accepted + 1 bonus). Prefix → 7.
    # Block 2: rollback target = 7.
    assert transport.calls[1].kv_rollback_to == 7
    assert stats.tokens_emitted == 6
    assert stats.blocks == 2


# ── Drafter contract enforcement ────────────────────────────────────────

def test_wrong_block_size_from_drafter_raises():
    drafter = _ScriptedDrafter([[1, 2, 3]])
    transport = _ScriptedTransport([[1, 2, 3, 99]])

    driver = DFlashTopologyADriver(
        drafter=drafter, verifier=_make_verifier(),
        transport=transport, emit=lambda t: None,
        block_size=16, max_tokens=5,
    )
    with pytest.raises(ValueError, match="expected block_size=16"):
        driver.run([])


# ── Stats correctness ──────────────────────────────────────────────────

def test_driver_stats_track_acceptance_correctly():
    drafts = [10, 20, 30, 99, 88] + [0] * 11
    argmax = [10, 20, 30, 40, 99] + [77] * 12
    drafter = _ScriptedDrafter([drafts])
    transport = _ScriptedTransport([argmax])

    driver = DFlashTopologyADriver(
        drafter=drafter, verifier=_make_verifier(),
        transport=transport, emit=lambda t: None,
        block_size=16, max_tokens=4,
    )
    stats = driver.run([])
    assert stats.blocks == 1
    assert stats.drafts_total == 16
    assert stats.drafts_accepted == 3
    assert stats.tokens_emitted == 4
    assert abs(stats.acceptance_rate - 3 / 16) < 1e-9
    assert stats.avg_block_size_emitted == 4.0


# ── BlockVerifyResult helpers ──────────────────────────────────────────

def test_block_verify_result_emitted_property():
    r = BlockVerifyResult(
        block_index=0, draft_token_ids=[1, 2, 3, 4, 5],
        accepted_len=3, bonus_token=99, new_prefix_len=4,
    )
    assert r.emitted == [1, 2, 3, 99]
    assert abs(r.acceptance_rate - 3 / 5) < 1e-9


def test_block_verify_result_zero_acceptance():
    r = BlockVerifyResult(
        block_index=0, draft_token_ids=[1, 2, 3],
        accepted_len=0, bonus_token=99, new_prefix_len=1,
    )
    assert r.emitted == [99]
    assert r.acceptance_rate == 0.0
