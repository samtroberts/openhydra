"""Tests for supernode.prompt_protocol — wire format CBOR roundtrip."""

import pytest

from supernode.prompt_protocol import (
    METHOD_PROMPT_REQUEST,
    METHOD_PROMPT_CANCEL,
    METHOD_MANIFEST_REQUEST,
    METHOD_MANIFEST_RESPONSE,
    METHOD_LOAD_PROBE,
    WirePromptRequest,
    PromptChunk,
    UsageStats,
)


class TestMethodPrefixes:
    def test_values_distinct(self):
        vals = [
            METHOD_PROMPT_REQUEST,
            METHOD_PROMPT_CANCEL,
            METHOD_MANIFEST_REQUEST,
            METHOD_MANIFEST_RESPONSE,
            METHOD_LOAD_PROBE,
        ]
        assert len(set(vals)) == 5

    def test_range(self):
        assert METHOD_PROMPT_REQUEST == 0x10
        assert METHOD_LOAD_PROBE == 0x14


class TestWirePromptRequest:
    def test_roundtrip_minimal(self):
        req = WirePromptRequest(request_id="r1", model_id="llama3:8b")
        data = req.to_cbor()
        decoded = WirePromptRequest.from_cbor(data)
        assert decoded.request_id == "r1"
        assert decoded.model_id == "llama3:8b"
        assert decoded.max_tokens == 512
        assert decoded.temperature == 0.7
        assert decoded.stream is True

    def test_roundtrip_full(self):
        req = WirePromptRequest(
            request_id="r2",
            model_id="qwen:2b",
            prompt="hello",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1024,
            temperature=0.5,
            top_p=0.95,
            top_k=50,
            stop=["END"],
            stream=False,
            session_id="s1",
            origin_peer_id="peer-A",
            hops=2,
            min_trust_level=3,
            system_prompt="be helpful",
            response_format="json",
        )
        decoded = WirePromptRequest.from_cbor(req.to_cbor())
        assert decoded.prompt == "hello"
        assert decoded.messages == [{"role": "user", "content": "hi"}]
        assert decoded.max_tokens == 1024
        assert decoded.stop == ["END"]
        assert decoded.session_id == "s1"
        assert decoded.origin_peer_id == "peer-A"
        assert decoded.hops == 2
        assert decoded.min_trust_level == 3
        assert decoded.system_prompt == "be helpful"
        assert decoded.response_format == "json"

    def test_deterministic_encoding(self):
        req = WirePromptRequest(request_id="r1", model_id="m1")
        assert req.to_cbor() == req.to_cbor()

    def test_none_fields_preserved(self):
        req = WirePromptRequest(request_id="r1", model_id="m1", prompt=None, messages=None)
        decoded = WirePromptRequest.from_cbor(req.to_cbor())
        assert decoded.prompt is None
        assert decoded.messages is None


class TestPromptChunk:
    def test_token_chunk_roundtrip(self):
        chunk = PromptChunk(request_id="r1", chunk_type="token", token="Hello", token_id=42)
        decoded = PromptChunk.from_cbor(chunk.to_cbor())
        assert decoded.request_id == "r1"
        assert decoded.chunk_type == "token"
        assert decoded.token == "Hello"
        assert decoded.token_id == 42

    def test_done_chunk_with_usage(self):
        usage = UsageStats(
            prompt_tokens=10,
            completion_tokens=50,
            total_tokens=60,
            tokens_per_second=12.5,
            time_to_first_token_ms=80,
        )
        chunk = PromptChunk(
            request_id="r1",
            chunk_type="done",
            finish_reason="stop",
            usage=usage,
        )
        decoded = PromptChunk.from_cbor(chunk.to_cbor())
        assert decoded.chunk_type == "done"
        assert decoded.finish_reason == "stop"
        assert decoded.usage is not None
        assert decoded.usage.prompt_tokens == 10
        assert decoded.usage.completion_tokens == 50
        assert decoded.usage.tokens_per_second == 12.5

    def test_error_chunk(self):
        chunk = PromptChunk(
            request_id="r1",
            chunk_type="error",
            error="model not found",
            retryable=True,
        )
        decoded = PromptChunk.from_cbor(chunk.to_cbor())
        assert decoded.chunk_type == "error"
        assert decoded.error == "model not found"
        assert decoded.retryable is True

    def test_status_chunk(self):
        chunk = PromptChunk(
            request_id="r1",
            chunk_type="status",
            status="queued",
            estimated_wait_s=2.5,
        )
        decoded = PromptChunk.from_cbor(chunk.to_cbor())
        assert decoded.status == "queued"
        assert decoded.estimated_wait_s == 2.5

    def test_no_usage(self):
        chunk = PromptChunk(request_id="r1", chunk_type="token", token="x")
        decoded = PromptChunk.from_cbor(chunk.to_cbor())
        assert decoded.usage is None

    def test_deterministic(self):
        chunk = PromptChunk(request_id="r1", chunk_type="token", token="x")
        assert chunk.to_cbor() == chunk.to_cbor()


class TestUsageStats:
    def test_defaults(self):
        u = UsageStats()
        assert u.prompt_tokens == 0
        assert u.checkpoint_hashes == []
        assert u.output_signature == b""

    def test_attestation_fields(self):
        u = UsageStats(
            model_weights_hash="abc123",
            output_signature=b"\x01\x02",
            checkpoint_hashes=["h1", "h2"],
        )
        assert u.model_weights_hash == "abc123"
        assert u.output_signature == b"\x01\x02"
        assert len(u.checkpoint_hashes) == 2
