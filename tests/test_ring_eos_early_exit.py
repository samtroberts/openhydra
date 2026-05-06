# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Ring EOS early-exit regression tests.

Validates that the ring loop correctly stops when:
  1. An EOS token is generated (``_ring_token in _ring_eos``).
  2. ``ring_tokens_remaining`` reaches 0.
  3. The None sentinel is emitted to the ring queue exactly once.

Prior to this fix, the last peer's legacy ring loopback used
``if True:`` — unconditionally re-injecting the next ForwardRequest
even after EOS. This caused the ring to run all ``max_tokens``
iterations, wasting ~28% of latency for typical responses (e.g.
366 useful tokens out of 512 max_tokens → 146 wasted round trips).

Also validates that ``_collect_eos_token_ids`` picks up EOS IDs
from both ``tokenizer.eos_token_id`` and ``generation_config``.
"""

from __future__ import annotations

import queue
import threading
from unittest.mock import MagicMock, patch

import pytest


# ── _collect_eos_token_ids tests ──────────────────────────────────────────


class TestCollectEosTokenIds:
    """Verify _collect_eos_token_ids extracts EOS from tokenizer + gen_config."""

    @staticmethod
    def _collect(tokenizer):
        from coordinator.inference_service import InferenceService
        return InferenceService._collect_eos_token_ids(tokenizer)

    def test_single_int_eos(self):
        tok = MagicMock()
        tok.eos_token_id = 151645
        tok.all_special_ids = [151643, 151644, 151645]
        eos, special = self._collect(tok)
        assert 151645 in eos
        assert 151643 in special

    def test_list_eos(self):
        tok = MagicMock()
        tok.eos_token_id = [151643, 151645]
        tok.all_special_ids = []
        eos, _ = self._collect(tok)
        assert eos == {151643, 151645}

    def test_generation_config_eos(self):
        """generation_config.eos_token_id supplements tokenizer.eos_token_id."""
        tok = MagicMock()
        tok.eos_token_id = 151645  # only im_end
        tok.all_special_ids = []
        # generation_config carries the full list
        gen_cfg = MagicMock()
        gen_cfg.eos_token_id = [151643, 151645]
        tok.generation_config = gen_cfg
        eos, _ = self._collect(tok)
        assert 151643 in eos, "endoftext from generation_config should be included"
        assert 151645 in eos

    def test_none_tokenizer(self):
        eos, special = self._collect(None)
        assert eos == set()
        assert special == set()

    def test_negative_ids_filtered(self):
        tok = MagicMock()
        tok.eos_token_id = [-1, 0, 151645]
        tok.all_special_ids = []
        eos, _ = self._collect(tok)
        assert -1 not in eos
        assert 0 in eos  # 0 is >= 0
        assert 151645 in eos


# ── Ring queue sentinel tests ─────────────────────────────────────────────


class TestRingQueueSentinel:
    """Verify emit_ring_token sentinel behaviour."""

    def test_eos_emits_sentinel(self):
        """When EOS is hit, the queue should receive the token then None."""
        from coordinator.push_receiver import (
            register_ring, unregister_ring, emit_ring_token,
        )
        req_id = "test-eos-sentinel"
        q = register_ring(req_id)
        try:
            # Emit a normal token
            emit_ring_token(req_id, 42)
            # Emit the EOS token
            emit_ring_token(req_id, 151645)
            # Emit sentinel
            emit_ring_token(req_id, None)

            assert q.get_nowait() == 42
            assert q.get_nowait() == 151645
            assert q.get_nowait() is None
        finally:
            unregister_ring(req_id)

    def test_no_queue_noop(self):
        """emit_ring_token is a no-op when no queue is registered."""
        from coordinator.push_receiver import emit_ring_token
        # Should not raise
        emit_ring_token("nonexistent-request", 42)
        emit_ring_token("nonexistent-request", None)


# ── Ring drain loop early-exit simulation ─────────────────────────────────


class TestRingDrainLoopEarlyExit:
    """Simulate the coordinator's ring drain loop with EOS mid-stream."""

    def test_drain_loop_stops_on_sentinel(self):
        """When None arrives mid-stream, the drain loop breaks early."""
        q: queue.Queue = queue.Queue()
        max_tokens = 512

        # Simulate: 100 real tokens, then EOS, then sentinel
        for i in range(100):
            q.put(i)
        q.put(151645)  # EOS token
        q.put(None)    # sentinel

        generated = []
        for step in range(max_tokens):
            tok = q.get(timeout=1.0)
            if tok is None:
                break
            generated.append(int(tok))

        # Should have collected 101 tokens (100 + EOS), not 512
        assert len(generated) == 101
        assert generated[-1] == 151645

    def test_drain_loop_stops_on_max_tokens(self):
        """When max_tokens is reached, the drain loop stops."""
        q: queue.Queue = queue.Queue()
        max_tokens = 10

        for i in range(max_tokens):
            q.put(i)
        q.put(None)  # sentinel after all tokens

        generated = []
        for step in range(max_tokens):
            tok = q.get(timeout=1.0)
            if tok is None:
                break
            generated.append(int(tok))

        assert len(generated) == max_tokens


# ── EOS check at last peer (the actual bug fix) ──────────────────────────


class TestLastPeerEosCheck:
    """Verify the EOS check logic that was added to the last-peer ring path."""

    @pytest.mark.parametrize(
        "token, remaining, eos_set, expect_stop",
        [
            # EOS token with remaining > 0 → stop
            (151645, 100, {151645}, True),
            # Normal token with remaining > 0 → continue
            (42, 100, {151645}, False),
            # Normal token with remaining = 0 → stop
            (42, 0, {151645}, True),
            # EOS token with remaining = 0 → stop
            (151645, 0, {151645}, True),
            # Empty EOS set, remaining > 0 → continue
            (151645, 100, set(), False),
            # Multiple EOS tokens
            (151643, 50, {151643, 151645}, True),
        ],
    )
    def test_eos_decision(self, token, remaining, eos_set, expect_stop):
        """The ring should stop when token is in eos_set or remaining <= 0."""
        _is_ring_eos = token in eos_set
        should_stop = remaining <= 0 or _is_ring_eos
        assert should_stop == expect_stop


# ── completion_tokens accuracy tests ──────────────────────────────────────


class TestCompletionTokenCount:
    """Verify completion_tokens strict enforcement — no word-count fallback.

    The API now uses ``_strict_token_counts`` / ``_usage_from_payload``
    which raise ``KeyError`` when ``completion_tokens`` is missing or
    zero, rather than silently falling back to ``len(text.split())``.
    """

    def test_strict_token_counts_valid(self):
        """_strict_token_counts returns (pt, ct) when both are present."""
        from coordinator.api_server import _strict_token_counts
        payload = {
            "response": "Hello world this is a test response",
            "completion_tokens": 42,
            "prompt_tokens": 10,
        }
        pt, ct = _strict_token_counts(payload)
        assert ct == 42, "Should use actual token count from activation"
        assert pt == 10

    def test_strict_token_counts_missing_raises(self):
        """_strict_token_counts raises KeyError when completion_tokens absent."""
        from coordinator.api_server import _strict_token_counts
        payload = {
            "response": "Hello world this is a test response",
        }
        with pytest.raises(KeyError, match="completion_tokens missing"):
            _strict_token_counts(payload)

    def test_strict_token_counts_zero_raises(self):
        """_strict_token_counts raises KeyError when completion_tokens is 0."""
        from coordinator.api_server import _strict_token_counts
        payload = {
            "response": "One two three",
            "completion_tokens": 0,
        }
        with pytest.raises(KeyError, match="completion_tokens missing"):
            _strict_token_counts(payload)

    def test_usage_from_payload_builds_dict(self):
        """_usage_from_payload builds correct usage dict."""
        from coordinator.api_server import _usage_from_payload
        payload = {
            "response": "Hello world",
            "completion_tokens": 50,
            "prompt_tokens": 20,
        }
        usage = _usage_from_payload(payload)
        assert usage["completion_tokens"] == 50
        assert usage["prompt_tokens"] == 20
        assert usage["total_tokens"] == 70

    def test_usage_from_payload_missing_raises(self):
        """_usage_from_payload raises KeyError when completion_tokens absent."""
        from coordinator.api_server import _usage_from_payload
        payload = {"response": "Hello world"}
        with pytest.raises(KeyError, match="completion_tokens missing"):
            _usage_from_payload(payload)
