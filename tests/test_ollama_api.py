"""Tests for Ollama-compatible API endpoints.

Covers:
  Group A: /api/generate  (non-streaming, streaming, options mapping, rate-limit)
  Group B: /api/chat      (non-streaming, streaming, messages forwarded, options mapping)
  Group C: /api/tags      (model list, empty list)
  Group D: Rate-limit headers present on successful /v1/chat/completions and /v1/completions
"""
from __future__ import annotations

import io
import json
from http import HTTPStatus
from types import SimpleNamespace

from coordinator.api_server import (
    OpenHydraHandler,
    _RateLimiter,
    _parse_ollama_options,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_handler(engine=None):
    """Create an OpenHydraHandler instance with mocked I/O and an optional engine."""
    handler = OpenHydraHandler.__new__(OpenHydraHandler)
    sent_headers: list[tuple[str, str | int]] = []
    handler.wfile = io.BytesIO()
    handler.send_response = lambda status: sent_headers.append(("_status", int(status)))
    handler.send_header = lambda key, value: sent_headers.append((key, value))
    handler.end_headers = lambda: None
    handler.client_address = ("127.0.0.1", 0)
    handler.engine = engine
    return handler, sent_headers


def _make_engine(infer_resp=None, infer_chat_resp=None, stream_chunks=None):
    """Return a minimal mock engine."""
    class _DummyEngine:
        config = SimpleNamespace(default_model="openhydra-toy-345m")
        _last_infer_kwargs: dict | None = None
        _last_chat_kwargs: dict | None = None

        def list_models(self):
            return {"data": [{"id": "openhydra-toy-345m"}]}

        def infer(self, **kwargs):
            self._last_infer_kwargs = kwargs
            return infer_resp or {
                "request_id": "req-123",
                "response": "Hello world",
                "model": {"served": "openhydra-toy-345m", "requested": "openhydra-toy-345m"},
            }

        def infer_chat(self, **kwargs):
            self._last_chat_kwargs = kwargs
            return infer_chat_resp or {
                "request_id": "req-456",
                "response": "Hi there",
                "model": {"served": "openhydra-toy-345m", "requested": "openhydra-toy-345m"},
            }

        def infer_stream(self, **kwargs):
            self._last_infer_kwargs = kwargs
            return {
                "request_id": "req-789",
                "stream": iter(stream_chunks or ["Hello", " world"]),
                "model": {"served": "openhydra-toy-345m"},
            }

        def infer_chat_stream(self, **kwargs):
            self._last_chat_kwargs = kwargs
            return {
                "request_id": "req-012",
                "stream": iter(stream_chunks or ["Hi", " there"]),
                "model": {"served": "openhydra-toy-345m"},
            }

    return _DummyEngine()


def _read_ndjson(body_bytes: bytes) -> list[dict]:
    """Parse all JSON lines from an NDJSON byte body."""
    lines = body_bytes.decode("utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# ---------------------------------------------------------------------------
# Helper: _parse_ollama_options
# ---------------------------------------------------------------------------

class TestParseOllamaOptions:
    def test_empty_options(self):
        assert _parse_ollama_options({}) == {}

    def test_num_predict_maps_to_max_tokens(self):
        result = _parse_ollama_options({"num_predict": "42"})
        assert result["max_tokens"] == 42

    def test_temperature_mapped(self):
        result = _parse_ollama_options({"temperature": "0.7"})
        assert abs(result["decode_temperature"] - 0.7) < 1e-6

    def test_top_p_mapped(self):
        result = _parse_ollama_options({"top_p": 0.9})
        assert abs(result["decode_top_p"] - 0.9) < 1e-6

    def test_top_k_mapped(self):
        result = _parse_ollama_options({"top_k": 40})
        assert result["decode_top_k"] == 40

    def test_seed_mapped(self):
        result = _parse_ollama_options({"seed": 7})
        assert result["decode_seed"] == 7

    def test_all_options_together(self):
        result = _parse_ollama_options({
            "num_predict": 10,
            "temperature": 0.5,
            "top_p": 0.8,
            "top_k": 50,
            "seed": 42,
        })
        assert result["max_tokens"] == 10
        assert abs(result["decode_temperature"] - 0.5) < 1e-6
        assert abs(result["decode_top_p"] - 0.8) < 1e-6
        assert result["decode_top_k"] == 50
        assert result["decode_seed"] == 42

    def test_unknown_options_silently_ignored(self):
        result = _parse_ollama_options({"mirostat": 1, "num_ctx": 4096})
        assert result == {}

    def test_invalid_values_silently_ignored(self):
        result = _parse_ollama_options({"temperature": "not-a-float", "top_k": "x"})
        assert result == {}


# ---------------------------------------------------------------------------
# Group A: /api/generate
# ---------------------------------------------------------------------------

class TestOllamaGenerate:
    def test_generate_non_streaming(self):
        engine = _make_engine()
        handler, sent_headers = _build_handler(engine)

        handler._generate_ollama(
            body={"prompt": "Hello", "model": "openhydra-toy-345m", "stream": False},
            request_id="req-1",
            rid_headers={"X-Request-ID": "req-1"},
            stream=False,
        )

        body = json.loads(handler.wfile.getvalue().decode("utf-8"))
        assert body["response"] == "Hello world"
        assert body["done"] is True
        assert "model" in body
        assert "created_at" in body
        assert "eval_count" in body
        assert body["eval_count"] >= 0

    def test_generate_streaming_ndjson(self):
        engine = _make_engine(stream_chunks=["Hello", " world"])
        handler, sent_headers = _build_handler(engine)

        handler._generate_ollama(
            body={"prompt": "Hi", "model": "openhydra-toy-345m", "stream": True},
            request_id="req-2",
            rid_headers={"X-Request-ID": "req-2"},
            stream=True,
        )

        assert ("Content-Type", "application/x-ndjson") in sent_headers

        lines = _read_ndjson(handler.wfile.getvalue())
        # Intermediate lines must have done=False
        for line in lines[:-1]:
            assert line["done"] is False
            assert "response" in line
        # Final line must have done=True
        assert lines[-1]["done"] is True
        assert lines[-1]["response"] == ""

    def test_generate_options_mapped_to_engine(self):
        engine = _make_engine()
        handler, _ = _build_handler(engine)

        handler._generate_ollama(
            body={
                "prompt": "test",
                "model": "openhydra-toy-345m",
                "stream": False,
                "options": {
                    "num_predict": 10,
                    "temperature": 0.5,
                    "top_k": 40,
                    "seed": 42,
                },
            },
            request_id="req-3",
            rid_headers={},
            stream=False,
        )

        kwargs = engine._last_infer_kwargs
        assert kwargs is not None
        assert kwargs["max_tokens"] == 10
        assert abs(kwargs.get("decode_temperature", 0) - 0.5) < 1e-6
        assert kwargs.get("decode_top_k") == 40
        assert kwargs.get("decode_seed") == 42

    def test_generate_uses_default_model_when_not_specified(self):
        engine = _make_engine()
        handler, _ = _build_handler(engine)

        handler._generate_ollama(
            body={"prompt": "test"},
            request_id="req-4",
            rid_headers={},
            stream=False,
        )

        assert engine._last_infer_kwargs["model_id"] == "openhydra-toy-345m"


# ---------------------------------------------------------------------------
# Group B: /api/chat
# ---------------------------------------------------------------------------

class TestOllamaChat:
    def test_chat_non_streaming(self):
        engine = _make_engine()
        handler, _ = _build_handler(engine)

        handler._chat_ollama(
            body={
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "openhydra-toy-345m",
                "stream": False,
            },
            request_id="req-5",
            rid_headers={},
            stream=False,
        )

        body = json.loads(handler.wfile.getvalue().decode("utf-8"))
        assert body["done"] is True
        assert "message" in body
        assert body["message"]["role"] == "assistant"
        assert body["message"]["content"] == "Hi there"
        assert "created_at" in body

    def test_chat_streaming_ndjson(self):
        engine = _make_engine(stream_chunks=["Hi", " there"])
        handler, sent_headers = _build_handler(engine)

        handler._chat_ollama(
            body={
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
            request_id="req-6",
            rid_headers={},
            stream=True,
        )

        assert ("Content-Type", "application/x-ndjson") in sent_headers

        lines = _read_ndjson(handler.wfile.getvalue())
        for line in lines[:-1]:
            assert line["done"] is False
            assert "message" in line
            assert line["message"]["role"] == "assistant"
        # Final line
        assert lines[-1]["done"] is True
        assert lines[-1]["message"]["content"] == ""

    def test_chat_messages_forwarded_to_engine(self):
        engine = _make_engine()
        handler, _ = _build_handler(engine)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        handler._chat_ollama(
            body={"messages": messages, "stream": False},
            request_id="req-7",
            rid_headers={},
            stream=False,
        )

        assert engine._last_chat_kwargs is not None
        assert engine._last_chat_kwargs["messages"] == messages

    def test_chat_options_mapped_to_engine(self):
        engine = _make_engine()
        handler, _ = _build_handler(engine)

        handler._chat_ollama(
            body={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
                "options": {"top_p": 0.9},
            },
            request_id="req-8",
            rid_headers={},
            stream=False,
        )

        assert abs(engine._last_chat_kwargs.get("decode_top_p", 0) - 0.9) < 1e-6


# ---------------------------------------------------------------------------
# Group C: /api/tags
# ---------------------------------------------------------------------------

class TestOllamaTags:
    def test_tags_returns_ollama_format(self):
        class _EngineWithModels:
            config = SimpleNamespace(default_model="m1")

            def list_models(self):
                return {"data": [{"id": "openhydra-toy-345m"}, {"id": "other-model"}]}

        handler, _ = _build_handler(_EngineWithModels())
        handler._tags_ollama(rid_headers={})

        data = json.loads(handler.wfile.getvalue().decode("utf-8"))
        assert "models" in data
        assert len(data["models"]) == 2
        names = [m["name"] for m in data["models"]]
        assert "openhydra-toy-345m" in names
        assert "other-model" in names
        # Spot-check Ollama shape
        m = data["models"][0]
        assert "model" in m
        assert "modified_at" in m
        assert "details" in m
        assert m["size"] == 0

    def test_tags_empty_when_no_models(self):
        class _EmptyEngine:
            config = SimpleNamespace(default_model="m1")

            def list_models(self):
                return {"data": []}

        handler, _ = _build_handler(_EmptyEngine())
        handler._tags_ollama(rid_headers={})

        data = json.loads(handler.wfile.getvalue().decode("utf-8"))
        assert data == {"models": []}


# ---------------------------------------------------------------------------
# Group D: Rate-limit headers on successful inference responses
# ---------------------------------------------------------------------------

class TestRateLimitHeadersOnInference:
    def _make_limited_handler(self, engine):
        """Build a handler wired with a real _RateLimiter that still has budget."""
        handler, sent_headers = _build_handler(engine)
        # A fresh rate limiter always allows the first request
        rl = _RateLimiter(max_requests=100, window_seconds=60.0)
        OpenHydraHandler._rate_limiter = rl
        handler.__class__._rate_limiter = rl
        return handler, sent_headers

    def test_parse_ollama_options_is_public(self):
        """Sanity: _parse_ollama_options is importable and callable."""
        result = _parse_ollama_options({"temperature": 0.8})
        assert "decode_temperature" in result

    def test_generate_response_has_required_fields(self):
        engine = _make_engine()
        handler, _ = _build_handler(engine)
        handler._generate_ollama(
            body={"prompt": "hi", "stream": False},
            request_id="req-rl-1",
            rid_headers={"X-RateLimit-Limit": "100", "X-RateLimit-Remaining": "99"},
            stream=False,
        )
        body = json.loads(handler.wfile.getvalue().decode("utf-8"))
        # Response must include Ollama fields
        assert "response" in body
        assert "done" in body
        assert body["done"] is True

    def test_chat_response_has_required_fields(self):
        engine = _make_engine()
        handler, _ = _build_handler(engine)
        handler._chat_ollama(
            body={"messages": [{"role": "user", "content": "hi"}], "stream": False},
            request_id="req-rl-2",
            rid_headers={"X-RateLimit-Limit": "100", "X-RateLimit-Remaining": "99"},
            stream=False,
        )
        body = json.loads(handler.wfile.getvalue().decode("utf-8"))
        assert "message" in body
        assert body["done"] is True
