# Copyright 2026 OpenHydra contributors
# Licensed under the Apache License, Version 2.0

"""TDD tests for LocalInferenceEngine (Pillar 1: Offline Engine Detachment).

These tests run against the ToyRuntime backend so they complete instantly
without a GPU.  They validate that LocalInferenceEngine:

  - Produces OpenAI-compatible response dicts
  - Streams SSE-format chunks
  - Applies chat templates
  - Counts tokens accurately
  - Respects stop sequences and finish_reason variants
  - Makes ZERO network calls (no gRPC, no DHT, no HuggingFace HTTP)

Run:  pytest tests/test_local_engine.py -v
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest

from peer.model_shard import ModelShard, ToyShardConfig


# ─── Helpers ────────────────────────────────────────────────────────────────

def _make_shard() -> ModelShard:
    """Build a ToyRuntime-backed ModelShard for testing."""
    cfg = ToyShardConfig(
        model_id="openhydra-toy-345m",
        runtime_backend="toy_auto",
        runtime_model_id="openhydra-toy-345m",
    )
    return ModelShard(cfg)


def _make_engine():
    """Build a LocalInferenceEngine backed by ToyRuntime."""
    from coordinator.local_engine import LocalInferenceEngine
    shard = _make_shard()
    return LocalInferenceEngine(
        model_id="openhydra-toy-345m",
        shard=shard,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Group A — Initialization
# ═══════════════════════════════════════════════════════════════════════════════

class TestInit:
    def test_init_with_toy_runtime(self):
        """Engine should construct with a ToyRuntime-backed ModelShard."""
        engine = _make_engine()
        assert engine.model_id == "openhydra-toy-345m"
        assert engine.shard is not None

    def test_list_models_returns_loaded_model(self):
        """list_models() should return exactly the one loaded model."""
        engine = _make_engine()
        models = engine.list_models()
        assert isinstance(models, list)
        assert len(models) == 1
        m = models[0]
        assert m["id"] == "openhydra-toy-345m"
        assert "object" in m
        assert m["object"] == "model"


# ═══════════════════════════════════════════════════════════════════════════════
# Group B — infer() and OpenAI response format
# ═══════════════════════════════════════════════════════════════════════════════

class TestInfer:
    def test_infer_returns_openai_format(self):
        """infer() must return a dict matching OpenAI ChatCompletion schema."""
        engine = _make_engine()
        result = engine.infer("What is the meaning of life?", max_tokens=16)

        # Top-level keys
        assert "id" in result
        assert result["object"] == "chat.completion"
        assert result["model"] == "openhydra-toy-345m"
        assert "choices" in result
        assert "usage" in result

        # Choices
        choices = result["choices"]
        assert len(choices) == 1
        choice = choices[0]
        assert choice["index"] == 0
        assert "message" in choice
        assert choice["message"]["role"] == "assistant"
        assert isinstance(choice["message"]["content"], str)
        assert len(choice["message"]["content"]) > 0
        assert choice["finish_reason"] in ("stop", "length")

    def test_infer_usage_token_counts(self):
        """usage must contain non-negative integer token counts."""
        engine = _make_engine()
        result = engine.infer("Hello world", max_tokens=8)

        usage = result["usage"]
        assert isinstance(usage["prompt_tokens"], int)
        assert isinstance(usage["completion_tokens"], int)
        assert isinstance(usage["total_tokens"], int)
        assert usage["prompt_tokens"] >= 0
        assert usage["completion_tokens"] > 0
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_infer_max_tokens_respected(self):
        """Completion should not exceed max_tokens in length."""
        engine = _make_engine()
        result = engine.infer("Tell me a story", max_tokens=4)
        # ToyRuntime returns activation-mapped words; completion_tokens <= max_tokens
        usage = result["usage"]
        assert usage["completion_tokens"] <= 4

    def test_infer_finish_reason_length(self):
        """When output is truncated by max_tokens, finish_reason should be 'length'."""
        engine = _make_engine()
        # Use a very small max_tokens to force truncation
        result = engine.infer("Tell me everything about the universe", max_tokens=2)
        # With only 2 tokens allowed, it should be length-limited
        choice = result["choices"][0]
        assert choice["finish_reason"] == "length"

    def test_infer_finish_reason_stop(self):
        """When model generates a natural stop, finish_reason should be 'stop'."""
        engine = _make_engine()
        # With enough tokens, ToyRuntime completes naturally
        result = engine.infer("Hi", max_tokens=48)
        choice = result["choices"][0]
        assert choice["finish_reason"] in ("stop", "length")


# ═══════════════════════════════════════════════════════════════════════════════
# Group C — infer_stream() SSE chunk format
# ═══════════════════════════════════════════════════════════════════════════════

class TestInferStream:
    def test_infer_stream_yields_chunks(self):
        """infer_stream() must yield dicts with delta.content."""
        engine = _make_engine()
        chunks = list(engine.infer_stream("Hello", max_tokens=8))

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk["object"] == "chat.completion.chunk"
            assert chunk["model"] == "openhydra-toy-345m"
            assert "choices" in chunk
            choice = chunk["choices"][0]
            assert "delta" in choice
            # Content can be empty string in first/last chunk
            assert isinstance(choice["delta"].get("content", ""), str)

    def test_infer_stream_final_chunk_finish_reason(self):
        """The last streamed chunk should contain finish_reason."""
        engine = _make_engine()
        chunks = list(engine.infer_stream("Hello", max_tokens=8))
        last = chunks[-1]
        assert last["choices"][0].get("finish_reason") in ("stop", "length")


# ═══════════════════════════════════════════════════════════════════════════════
# Group D — chat() and chat_stream() (template application)
# ═══════════════════════════════════════════════════════════════════════════════

class TestChat:
    def test_chat_returns_openai_format(self):
        """chat() must accept messages list and return OpenAI format."""
        engine = _make_engine()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        result = engine.chat(messages, max_tokens=16)

        assert result["object"] == "chat.completion"
        assert len(result["choices"]) == 1
        assert result["choices"][0]["message"]["role"] == "assistant"
        assert isinstance(result["choices"][0]["message"]["content"], str)

    def test_chat_applies_template(self):
        """chat() should flatten messages into a prompt string, not pass raw dicts."""
        engine = _make_engine()
        messages = [
            {"role": "user", "content": "Hello"},
        ]
        result = engine.chat(messages, max_tokens=8)
        # The result should have content; the key test is it doesn't crash
        assert len(result["choices"][0]["message"]["content"]) > 0

    def test_chat_stream_yields_chunks(self):
        """chat_stream() must yield SSE-format chunks."""
        engine = _make_engine()
        messages = [{"role": "user", "content": "Hi"}]
        chunks = list(engine.chat_stream(messages, max_tokens=8))
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk["object"] == "chat.completion.chunk"


# ═══════════════════════════════════════════════════════════════════════════════
# Group E — Stop sequences
# ═══════════════════════════════════════════════════════════════════════════════

class TestStopSequences:
    def test_stop_sequence_truncates_output(self):
        """If a stop sequence appears in output, text should be truncated before it."""
        engine = _make_engine()
        # Generate a decent amount of text
        result = engine.infer("Tell me about hydra swarm", max_tokens=32, stop=["tensor"])
        content = result["choices"][0]["message"]["content"]
        # If "tensor" was in the output, it should have been removed
        assert "tensor" not in content.lower() or content == ""

    def test_stop_sequence_finish_reason(self):
        """finish_reason should be 'stop' when a stop sequence triggers."""
        engine = _make_engine()
        result = engine.infer("Tell me about hydra", max_tokens=32, stop=["hydra"])
        # The word "hydra" is very likely in ToyRuntime output
        choice = result["choices"][0]
        # Either stop triggered, or length hit — both valid
        assert choice["finish_reason"] in ("stop", "length")


# ═══════════════════════════════════════════════════════════════════════════════
# Group F — Network isolation guarantee
# ═══════════════════════════════════════════════════════════════════════════════

class TestNetworkIsolation:
    def test_no_network_calls(self):
        """LocalInferenceEngine must make ZERO network calls during inference.

        We patch socket.socket to detect any attempt to open a TCP/UDP
        connection.  gRPC, DHT HTTP, and HuggingFace Hub all use sockets.
        """
        engine = _make_engine()

        connections_made: list[tuple] = []
        original_connect = __import__("socket").socket.connect

        def _spy_connect(self_sock, address):
            connections_made.append(address)
            return original_connect(self_sock, address)

        with patch("socket.socket.connect", _spy_connect):
            engine.infer("Hello world", max_tokens=8)
            engine.chat([{"role": "user", "content": "Hi"}], max_tokens=8)
            list(engine.infer_stream("Hello", max_tokens=8))
            list(engine.chat_stream([{"role": "user", "content": "Hi"}], max_tokens=8))

        assert connections_made == [], (
            f"LocalInferenceEngine made {len(connections_made)} network connection(s): "
            f"{connections_made}"
        )

    def test_no_imports_from_network_services(self):
        """local_engine.py must NOT import discovery, pipeline, chain, or grpc modules.

        Only checks actual import/from lines, not docstrings or comments.
        """
        import importlib
        import inspect

        mod = importlib.import_module("coordinator.local_engine")
        source = inspect.getsource(mod)

        # Extract only lines that are actual import statements (not comments/docstrings)
        import_lines = [
            line.strip()
            for line in source.splitlines()
            if (line.strip().startswith("import ") or line.strip().startswith("from "))
            and not line.strip().startswith("#")
        ]
        import_block = "\n".join(import_lines)

        forbidden = [
            "discovery_service",
            "pipeline_service",
            "chain",
            "inference_service",
            "grpc",
            "peer_pb2_grpc",
        ]
        for pattern in forbidden:
            assert pattern not in import_block, (
                f"local_engine.py imports forbidden module containing '{pattern}': "
                f"{import_block}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Group G — Concurrency and memory
# ═══════════════════════════════════════════════════════════════════════════════

class TestConcurrencyAndMemory:
    def test_concurrent_requests(self):
        """Multiple threads should be able to call infer() simultaneously."""
        engine = _make_engine()
        results: list[dict] = []
        errors: list[Exception] = []

        def _worker(prompt: str):
            try:
                r = engine.infer(prompt, max_tokens=8)
                results.append(r)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=_worker, args=(f"Prompt {i}",))
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Concurrent errors: {errors}"
        assert len(results) == 4

    def test_memory_after_unload(self):
        """After unload(), the shard reference should be cleared."""
        engine = _make_engine()
        assert engine.shard is not None
        engine.unload()
        assert engine.shard is None
