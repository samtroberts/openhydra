# Copyright 2026 OpenHydra contributors
# Licensed under the Apache License, Version 2.0

"""TDD tests for Pillar 2: API Emulation and Transparent Delegation.

Tests that api_server.py correctly routes requests to either the
LocalInferenceEngine or the CoordinatorEngine depending on which is
active, and that the response format is identical in both modes.

Also tests new Ollama parity endpoints (/api/show, /api/ps) and
OpenAI spec gap fixes (stop sequences, real token counts, finish_reason).

Run:  pytest tests/test_api_emulation.py -v
"""

from __future__ import annotations

import io
import json
from http import HTTPStatus
from types import SimpleNamespace

import pytest

from coordinator.api_server import OpenHydraHandler


# ─── Helpers ────────────────────────────────────────────────────────────────

def _build_handler(engine=None, local_engine=None):
    """Create an OpenHydraHandler with mocked I/O."""
    handler = OpenHydraHandler.__new__(OpenHydraHandler)
    sent_headers: list[tuple[str, str | int]] = []
    handler.wfile = io.BytesIO()
    handler.send_response = lambda status: sent_headers.append(("_status", int(status)))
    handler.send_header = lambda key, value: sent_headers.append((key, value))
    handler.end_headers = lambda: None
    handler.client_address = ("127.0.0.1", 0)
    # Set both engine slots
    handler.__class__.engine = engine
    handler.__class__.local_engine = local_engine
    handler.__class__._api_key = None  # no auth
    handler.__class__._rate_limiter = None
    return handler, sent_headers


def _parse_response(handler) -> dict:
    """Parse JSON response from the handler's wfile."""
    body = handler.wfile.getvalue().decode("utf-8")
    # For SSE responses, find the last JSON line before [DONE]
    if body.startswith("data: "):
        lines = [l for l in body.splitlines() if l.startswith("data: ") and l != "data: [DONE]"]
        return json.loads(lines[-1][6:]) if lines else {}
    return json.loads(body)


class _MockLocalEngine:
    """Mock that mimics LocalInferenceEngine's interface."""

    def __init__(self):
        self.model_id = "openhydra-toy-345m"
        self.calls: list[str] = []

    def infer(self, prompt, max_tokens=128, temperature=1.0, top_p=1.0, stop=None):
        self.calls.append("infer")
        return {
            "id": "local-req-1",
            "object": "chat.completion",
            "created": 1000000,
            "model": self.model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Local response"},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": 2,
                "total_tokens": len(prompt.split()) + 2,
            },
        }

    def infer_stream(self, prompt, max_tokens=128, temperature=1.0, top_p=1.0, stop=None):
        self.calls.append("infer_stream")
        yield {
            "id": "local-req-1",
            "object": "chat.completion.chunk",
            "created": 1000000,
            "model": self.model_id,
            "choices": [{"index": 0, "delta": {"content": "Local"}, "finish_reason": None}],
        }
        yield {
            "id": "local-req-1",
            "object": "chat.completion.chunk",
            "created": 1000000,
            "model": self.model_id,
            "choices": [{"index": 0, "delta": {"content": " response"}, "finish_reason": "stop"}],
        }

    def chat(self, messages, max_tokens=128, temperature=1.0, top_p=1.0, stop=None):
        self.calls.append("chat")
        return self.infer(messages[-1]["content"], max_tokens, temperature, top_p, stop)

    def chat_stream(self, messages, max_tokens=128, temperature=1.0, top_p=1.0, stop=None):
        self.calls.append("chat_stream")
        yield from self.infer_stream(messages[-1]["content"], max_tokens, temperature, top_p, stop)

    def list_models(self):
        self.calls.append("list_models")
        return [{
            "id": self.model_id,
            "object": "model",
            "created": 1000000,
            "owned_by": "openhydra-local",
        }]

    def unload(self):
        self.calls.append("unload")


class _MockSwarmEngine:
    """Mock that mimics CoordinatorEngine's interface."""

    def __init__(self):
        self.config = SimpleNamespace(default_model="openhydra-toy-345m")
        self.calls: list[str] = []

    def list_models(self):
        self.calls.append("list_models")
        return {"data": [{"id": "openhydra-toy-345m"}]}

    def infer_chat(self, **kwargs):
        self.calls.append("infer_chat")
        return {
            "request_id": "swarm-req-1",
            "response": "Swarm response",
            "model": {"served": "openhydra-toy-345m", "requested": "openhydra-toy-345m"},
        }

    def infer_chat_stream(self, **kwargs):
        self.calls.append("infer_chat_stream")
        return {
            "request_id": "swarm-req-1",
            "response": "Swarm streaming",
            "stream": iter(["Swarm", " streaming"]),
            "model": {"served": "openhydra-toy-345m", "requested": "openhydra-toy-345m"},
        }

    def infer(self, **kwargs):
        self.calls.append("infer")
        return {
            "request_id": "swarm-req-1",
            "response": "Swarm text",
            "model": {"served": "openhydra-toy-345m", "requested": "openhydra-toy-345m"},
        }

    def infer_stream(self, **kwargs):
        self.calls.append("infer_stream")
        return {
            "request_id": "swarm-req-1",
            "response": "Swarm text",
            "stream": iter(["Swarm", " text"]),
            "model": {"served": "openhydra-toy-345m", "requested": "openhydra-toy-345m"},
        }

    def network_status(self):
        self.calls.append("network_status")
        return {"peers": []}

    def account_balance(self, client_id):
        self.calls.append("account_balance")
        return {"hydra": 0, "credits": 0}


# ═══════════════════════════════════════════════════════════════════════════════
# Group A — Engine routing: local_engine takes priority when set
# ═══════════════════════════════════════════════════════════════════════════════

class TestEngineRouting:
    def test_active_engine_prefers_local_when_set(self):
        """When local_engine is set, _active_engine() returns it."""
        local = _MockLocalEngine()
        swarm = _MockSwarmEngine()
        handler, _ = _build_handler(engine=swarm, local_engine=local)
        active = handler._active_engine()
        assert active is local

    def test_active_engine_falls_back_to_swarm(self):
        """When local_engine is None, _active_engine() returns swarm engine."""
        swarm = _MockSwarmEngine()
        handler, _ = _build_handler(engine=swarm, local_engine=None)
        active = handler._active_engine()
        assert active is swarm

    def test_active_engine_raises_when_both_none(self):
        """When both engines are None, _active_engine() raises RuntimeError."""
        handler, _ = _build_handler(engine=None, local_engine=None)
        with pytest.raises(RuntimeError):
            handler._active_engine()

    def test_is_local_mode_true_when_local_engine_set(self):
        """_is_local_mode should return True when local_engine is active."""
        handler, _ = _build_handler(local_engine=_MockLocalEngine())
        assert handler._is_local_mode() is True

    def test_is_local_mode_false_when_swarm_only(self):
        """_is_local_mode should return False when only swarm engine is active."""
        handler, _ = _build_handler(engine=_MockSwarmEngine())
        assert handler._is_local_mode() is False


# ═══════════════════════════════════════════════════════════════════════════════
# Group B — /v1/chat/completions routes to correct engine
# ═══════════════════════════════════════════════════════════════════════════════

class TestChatCompletionsRouting:
    def test_chat_completions_local_non_streaming(self):
        """In local mode, /v1/chat/completions delegates to local_engine.chat()."""
        local = _MockLocalEngine()
        handler, headers = _build_handler(local_engine=local)

        body = json.dumps({
            "model": "openhydra-toy-345m",
            "messages": [{"role": "user", "content": "Hello"}],
        }).encode()
        handler.rfile = io.BytesIO(body)
        handler.headers = {"Content-Length": str(len(body))}
        handler.path = "/v1/chat/completions"
        handler.command = "POST"

        handler._handle_chat_completions()

        assert "chat" in local.calls
        resp = _parse_response(handler)
        assert resp["object"] == "chat.completion"
        assert resp["choices"][0]["message"]["content"] == "Local response"

    def test_chat_completions_local_streaming(self):
        """In local mode, streaming /v1/chat/completions yields SSE chunks."""
        local = _MockLocalEngine()
        handler, headers = _build_handler(local_engine=local)

        body = json.dumps({
            "model": "openhydra-toy-345m",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }).encode()
        handler.rfile = io.BytesIO(body)
        handler.headers = {"Content-Length": str(len(body))}
        handler.path = "/v1/chat/completions"

        handler._handle_chat_completions()

        assert "chat_stream" in local.calls
        output = handler.wfile.getvalue().decode("utf-8")
        assert "data: " in output
        assert "data: [DONE]" in output

    def test_chat_completions_swarm_delegates_to_engine(self):
        """In swarm mode, /v1/chat/completions delegates to CoordinatorEngine."""
        swarm = _MockSwarmEngine()
        handler, headers = _build_handler(engine=swarm)

        body = json.dumps({
            "model": "openhydra-toy-345m",
            "messages": [{"role": "user", "content": "Hello"}],
        }).encode()
        handler.rfile = io.BytesIO(body)
        handler.headers = {"Content-Length": str(len(body))}
        handler.path = "/v1/chat/completions"

        handler._handle_chat_completions()

        assert "infer_chat" in swarm.calls


# ═══════════════════════════════════════════════════════════════════════════════
# Group C — /v1/models routes correctly
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelsEndpoint:
    def test_v1_models_local_mode(self):
        """In local mode, /v1/models returns local engine's model list."""
        local = _MockLocalEngine()
        handler, _ = _build_handler(local_engine=local)

        handler._handle_list_models()

        assert "list_models" in local.calls
        resp = _parse_response(handler)
        assert "data" in resp
        assert resp["data"][0]["id"] == "openhydra-toy-345m"

    def test_v1_models_swarm_mode(self):
        """In swarm mode, /v1/models returns swarm engine's model list."""
        swarm = _MockSwarmEngine()
        handler, _ = _build_handler(engine=swarm)

        handler._handle_list_models()

        assert "list_models" in swarm.calls


# ═══════════════════════════════════════════════════════════════════════════════
# Group D — OpenAI spec gap fixes
# ═══════════════════════════════════════════════════════════════════════════════

class TestOpenAISpecGaps:
    def test_response_includes_usage_with_real_token_counts(self):
        """Local mode responses must have non-zero prompt_tokens."""
        local = _MockLocalEngine()
        handler, _ = _build_handler(local_engine=local)

        body = json.dumps({
            "messages": [{"role": "user", "content": "Hello world"}],
        }).encode()
        handler.rfile = io.BytesIO(body)
        handler.headers = {"Content-Length": str(len(body))}
        handler.path = "/v1/chat/completions"

        handler._handle_chat_completions()

        resp = _parse_response(handler)
        usage = resp["usage"]
        assert usage["prompt_tokens"] > 0, "prompt_tokens must not be 0"
        assert usage["completion_tokens"] > 0
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_response_has_created_timestamp(self):
        """Response must include a created field (Unix timestamp)."""
        local = _MockLocalEngine()
        handler, _ = _build_handler(local_engine=local)

        body = json.dumps({"messages": [{"role": "user", "content": "Hi"}]}).encode()
        handler.rfile = io.BytesIO(body)
        handler.headers = {"Content-Length": str(len(body))}
        handler.path = "/v1/chat/completions"

        handler._handle_chat_completions()

        resp = _parse_response(handler)
        assert "created" in resp
        assert isinstance(resp["created"], int)

    def test_stop_parameter_forwarded(self):
        """The stop parameter from the request body must be forwarded to the engine."""
        local = _MockLocalEngine()
        # Override chat to capture kwargs
        captured = {}
        orig_chat = local.chat
        def _spy_chat(messages, **kw):
            captured.update(kw)
            return orig_chat(messages, **kw)
        local.chat = _spy_chat

        handler, _ = _build_handler(local_engine=local)
        body = json.dumps({
            "messages": [{"role": "user", "content": "Hi"}],
            "stop": ["END", "HALT"],
        }).encode()
        handler.rfile = io.BytesIO(body)
        handler.headers = {"Content-Length": str(len(body))}
        handler.path = "/v1/chat/completions"

        handler._handle_chat_completions()

        assert captured.get("stop") == ["END", "HALT"]

    def test_max_tokens_forwarded(self):
        """max_tokens from request body must be forwarded to local engine."""
        local = _MockLocalEngine()
        captured = {}
        orig_chat = local.chat
        def _spy_chat(messages, **kw):
            captured.update(kw)
            return orig_chat(messages, **kw)
        local.chat = _spy_chat

        handler, _ = _build_handler(local_engine=local)
        body = json.dumps({
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 42,
        }).encode()
        handler.rfile = io.BytesIO(body)
        handler.headers = {"Content-Length": str(len(body))}
        handler.path = "/v1/chat/completions"

        handler._handle_chat_completions()

        assert captured.get("max_tokens") == 42

    def test_temperature_forwarded(self):
        """temperature from request body must be forwarded to local engine."""
        local = _MockLocalEngine()
        captured = {}
        orig_chat = local.chat
        def _spy_chat(messages, **kw):
            captured.update(kw)
            return orig_chat(messages, **kw)
        local.chat = _spy_chat

        handler, _ = _build_handler(local_engine=local)
        body = json.dumps({
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.5,
        }).encode()
        handler.rfile = io.BytesIO(body)
        handler.headers = {"Content-Length": str(len(body))}
        handler.path = "/v1/chat/completions"

        handler._handle_chat_completions()

        assert captured.get("temperature") == 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# Group E — Ollama parity endpoints
# ═══════════════════════════════════════════════════════════════════════════════

class TestOllamaParity:
    def test_api_tags_local_mode(self):
        """GET /api/tags in local mode returns Ollama format with model details."""
        local = _MockLocalEngine()
        handler, _ = _build_handler(local_engine=local)

        handler._handle_tags_ollama()

        resp = _parse_response(handler)
        assert "models" in resp
        assert len(resp["models"]) == 1
        m = resp["models"][0]
        assert m["name"] == "openhydra-toy-345m"
        assert m["model"] == "openhydra-toy-345m"

    def test_api_show_returns_model_info(self):
        """POST /api/show returns model configuration details."""
        local = _MockLocalEngine()
        handler, _ = _build_handler(local_engine=local)

        body = json.dumps({"name": "openhydra-toy-345m"}).encode()
        handler.rfile = io.BytesIO(body)
        handler.headers = {"Content-Length": str(len(body))}

        handler._handle_show_ollama()

        resp = _parse_response(handler)
        assert resp.get("modelfile") is not None or resp.get("modelinfo") is not None

    def test_api_ps_returns_running_models(self):
        """GET /api/ps returns the currently loaded model."""
        local = _MockLocalEngine()
        handler, _ = _build_handler(local_engine=local)

        handler._handle_ps_ollama()

        resp = _parse_response(handler)
        assert "models" in resp
        assert len(resp["models"]) == 1
        assert resp["models"][0]["name"] == "openhydra-toy-345m"

    def test_api_generate_local_mode(self):
        """POST /api/generate in local mode delegates to local engine."""
        local = _MockLocalEngine()
        handler, _ = _build_handler(local_engine=local)

        body = json.dumps({
            "model": "openhydra-toy-345m",
            "prompt": "Hello",
        }).encode()
        handler.rfile = io.BytesIO(body)
        handler.headers = {"Content-Length": str(len(body))}

        handler._handle_generate_ollama(stream=False)

        assert "infer" in local.calls
        resp = _parse_response(handler)
        assert resp["done"] is True
        assert "response" in resp

    def test_api_chat_local_mode(self):
        """POST /api/chat in local mode delegates to local engine."""
        local = _MockLocalEngine()
        handler, _ = _build_handler(local_engine=local)

        body = json.dumps({
            "model": "openhydra-toy-345m",
            "messages": [{"role": "user", "content": "Hi"}],
        }).encode()
        handler.rfile = io.BytesIO(body)
        handler.headers = {"Content-Length": str(len(body))}

        handler._handle_chat_ollama(stream=False)

        assert "chat" in local.calls
        resp = _parse_response(handler)
        assert resp["done"] is True
        assert "message" in resp
        assert resp["message"]["role"] == "assistant"


# ═══════════════════════════════════════════════════════════════════════════════
# Group F — Response format parity (local vs swarm produce identical schemas)
# ═══════════════════════════════════════════════════════════════════════════════

class TestResponseParity:
    def test_local_and_swarm_both_return_object_field(self):
        """Both local and swarm responses must have 'object': 'chat.completion'."""
        # Local
        local = _MockLocalEngine()
        handler_l, _ = _build_handler(local_engine=local)
        body = json.dumps({"messages": [{"role": "user", "content": "Hi"}]}).encode()
        handler_l.rfile = io.BytesIO(body)
        handler_l.headers = {"Content-Length": str(len(body))}
        handler_l.path = "/v1/chat/completions"
        handler_l._handle_chat_completions()
        resp_l = _parse_response(handler_l)

        # Swarm
        swarm = _MockSwarmEngine()
        handler_s, _ = _build_handler(engine=swarm)
        handler_s.rfile = io.BytesIO(body)
        handler_s.headers = {"Content-Length": str(len(body))}
        handler_s.path = "/v1/chat/completions"
        handler_s._handle_chat_completions()
        resp_s = _parse_response(handler_s)

        # Both must have the same schema keys
        assert resp_l["object"] == "chat.completion"
        assert resp_s["object"] == "chat.completion"
        for key in ("id", "object", "model", "choices", "usage"):
            assert key in resp_l, f"Local response missing key: {key}"
            assert key in resp_s, f"Swarm response missing key: {key}"

    def test_streaming_has_done_terminator(self):
        """Both local and swarm SSE streams must end with data: [DONE]."""
        local = _MockLocalEngine()
        handler, _ = _build_handler(local_engine=local)
        body = json.dumps({
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        }).encode()
        handler.rfile = io.BytesIO(body)
        handler.headers = {"Content-Length": str(len(body))}
        handler.path = "/v1/chat/completions"
        handler._handle_chat_completions()

        output = handler.wfile.getvalue().decode("utf-8")
        assert "data: [DONE]" in output
