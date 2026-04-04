# Copyright 2026 OpenHydra contributors — Apache 2.0

"""TDD tests for SwarmSpeculativeDecoder (DSD P0-A).

Run:  pytest tests/test_speculative_swarm.py -v
"""

from __future__ import annotations

import pytest


def _make_decoder(draft_fn=None, draft_tokens=4):
    from coordinator.speculative_swarm import SwarmSpeculativeDecoder, SwarmSpecConfig
    if draft_fn is None:
        # Default: echo context[-1]+1, +2, +3, ...
        def draft_fn(ctx, k):
            base = ctx[-1] if ctx else 0
            return [base + i + 1 for i in range(k)]
    config = SwarmSpecConfig(draft_tokens=draft_tokens)
    return SwarmSpeculativeDecoder(config=config, draft_fn=draft_fn)


class TestPropose:
    def test_propose_returns_k_tokens(self):
        decoder = _make_decoder()
        draft = decoder.propose([10, 20, 30], k=4)
        assert len(draft) == 4
        assert draft == [31, 32, 33, 34]

    def test_propose_delegates_to_draft_fn(self):
        calls = []
        def spy(ctx, k):
            calls.append((list(ctx), k))
            return [99] * k
        decoder = _make_decoder(draft_fn=spy)
        decoder.propose([1, 2], k=3)
        assert calls == [([1, 2], 3)]


class TestAcceptReject:
    def test_full_match(self):
        decoder = _make_decoder()
        accepted, all_matched = decoder.accept_reject(
            draft_ids=[10, 20, 30],
            verified_ids=[10, 20, 30],
        )
        assert accepted == [10, 20, 30]
        assert all_matched is True

    def test_partial_match(self):
        decoder = _make_decoder()
        accepted, all_matched = decoder.accept_reject(
            draft_ids=[10, 20, 30],
            verified_ids=[10, 20, 99],  # mismatch at index 2
        )
        # Accept draft prefix [10, 20] + first verified mismatch [99]
        assert accepted == [10, 20, 99]
        assert all_matched is False

    def test_no_match(self):
        decoder = _make_decoder()
        accepted, all_matched = decoder.accept_reject(
            draft_ids=[10, 20, 30],
            verified_ids=[99, 88, 77],  # mismatch at index 0
        )
        assert accepted == [99]  # Just the first verified token
        assert all_matched is False

    def test_empty_draft(self):
        decoder = _make_decoder()
        accepted, all_matched = decoder.accept_reject(
            draft_ids=[],
            verified_ids=[42],
        )
        assert accepted == [42]


class TestAdaptiveK:
    def test_k_increases_on_high_acceptance(self):
        decoder = _make_decoder(draft_tokens=4)
        # Simulate 10 rounds of full acceptance
        for _ in range(10):
            decoder.accept_reject([1, 2, 3, 4], [1, 2, 3, 4])
        assert decoder.current_k > 4

    def test_k_decreases_on_low_acceptance(self):
        decoder = _make_decoder(draft_tokens=6)
        # Simulate 10 rounds of zero acceptance
        for _ in range(10):
            decoder.accept_reject([1, 2, 3, 4, 5, 6], [99, 88, 77, 66, 55, 44])
        assert decoder.current_k < 6

    def test_k_bounded(self):
        from coordinator.speculative_swarm import SwarmSpecConfig
        config = SwarmSpecConfig(min_draft_tokens=2, max_draft_tokens=8, draft_tokens=4)
        from coordinator.speculative_swarm import SwarmSpeculativeDecoder
        decoder = SwarmSpeculativeDecoder(config=config, draft_fn=lambda c, k: [0]*k)
        # Force many rejections
        for _ in range(50):
            decoder.accept_reject([1]*8, [99]*8)
        assert decoder.current_k >= 2
        # Force many acceptances
        for _ in range(50):
            decoder.accept_reject([1]*2, [1]*2)
        assert decoder.current_k <= 8


class TestStats:
    def test_stats_tracked(self):
        decoder = _make_decoder()
        decoder.accept_reject([10, 20], [10, 99])
        stats = decoder.stats
        assert stats.rounds == 1
        assert stats.draft_tokens == 2
        assert stats.accepted_tokens == 2  # 10 + 99
        assert stats.acceptance_rate is not None
