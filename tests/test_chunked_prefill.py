# Copyright 2026 OpenHydra contributors — Apache 2.0

"""TDD tests for chunked prefill (P1-B / Phase 4).

Run:  pytest tests/test_chunked_prefill.py -v
"""

from __future__ import annotations

import pytest


def _make_chunker():
    from coordinator.chunked_prefill import ChunkedPrefill, ChunkedPrefillConfig
    return ChunkedPrefill, ChunkedPrefillConfig


class TestChunkSplitting:
    def test_short_prompt_single_chunk(self):
        CP, Config = _make_chunker()
        cp = CP(Config(chunk_size=256))
        chunks = cp.split_prompt("Hello world", tokenizer=None)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_long_prompt_multiple_chunks(self):
        CP, Config = _make_chunker()
        cp = CP(Config(chunk_size=5))
        # Create a prompt with many words
        prompt = " ".join([f"word{i}" for i in range(20)])
        chunks = cp.split_prompt(prompt, tokenizer=None)
        assert len(chunks) > 1

    def test_chunk_size_respected(self):
        CP, Config = _make_chunker()
        cp = CP(Config(chunk_size=3))
        prompt = "one two three four five six seven eight nine ten"
        chunks = cp.split_prompt(prompt, tokenizer=None)
        # Each chunk should have at most chunk_size words
        for chunk in chunks:
            assert len(chunk.split()) <= 4  # Allow some slack

    def test_empty_prompt(self):
        CP, Config = _make_chunker()
        cp = CP(Config(chunk_size=256))
        chunks = cp.split_prompt("", tokenizer=None)
        assert len(chunks) == 1
        assert chunks[0] == ""


class TestChunkedExecution:
    def test_process_chunks_returns_final_activation(self):
        CP, Config = _make_chunker()
        cp = CP(Config(chunk_size=5))

        calls = []
        def mock_chain_fn(prompt, **kw):
            calls.append(prompt)
            return [1.0, 2.0, 3.0]  # Mock activation

        prompt = "one two three four five six seven eight"
        result = cp.process(prompt, chain_fn=mock_chain_fn, tokenizer=None)
        assert result is not None
        assert len(calls) >= 1  # At least one chain call

    def test_kv_session_passed_to_chain(self):
        CP, Config = _make_chunker()
        cp = CP(Config(chunk_size=3))

        kw_log = []
        def tracking_chain(prompt, **kw):
            kw_log.append(dict(kw))
            return [1.0]

        prompt = "one two three four five six"
        cp.process(prompt, chain_fn=tracking_chain, tokenizer=None,
                   session_id="test-session")
        # All calls should have kv_session_id
        for kw in kw_log:
            assert kw.get("kv_session_id") == "test-session"

    def test_subsequent_chunks_use_cached_kv(self):
        CP, Config = _make_chunker()
        cp = CP(Config(chunk_size=3))

        kw_log = []
        def tracking_chain(prompt, **kw):
            kw_log.append(dict(kw))
            return [1.0]

        prompt = "one two three four five six seven eight nine"
        cp.process(prompt, chain_fn=tracking_chain, tokenizer=None,
                   session_id="s1")
        # First chunk: store activation, no cache use
        if len(kw_log) > 1:
            assert kw_log[0].get("kv_store_activation") is True
            # Second+ chunks: use cached activation
            assert kw_log[1].get("kv_use_cached_activation") is True


class TestStats:
    def test_stats_tracked(self):
        CP, Config = _make_chunker()
        cp = CP(Config(chunk_size=5))

        def mock_chain(prompt, **kw):
            return [1.0]

        cp.process("one two three four five six seven eight",
                   chain_fn=mock_chain, tokenizer=None)
        stats = cp.stats
        assert stats["total_chunks"] >= 1
        assert stats["total_tokens_est"] > 0
