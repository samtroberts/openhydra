"""Tests for supernode.ollama_adapter — OllamaAdapter against a mock HTTP server."""

import asyncio
import json

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestServer

from supernode.adapter import (
    PromptRequest,
    TokenChunk,
    ModelInfo,
    BackendStatus,
    BackendError,
)
from supernode.ollama_adapter import (
    OllamaAdapter,
    _parse_param_count,
    _get_context_length,
    _estimate_free_memory,
)


# ---------------------------------------------------------------------------
# Mock Ollama server
# ---------------------------------------------------------------------------

def _build_mock_app():
    app = web.Application()

    async def root(request):
        return web.Response(text="Ollama is running")

    async def tags(request):
        return web.json_response({
            "models": [
                {
                    "name": "llama3.1:8b-q4_0",
                    "details": {
                        "family": "llama",
                        "parameter_size": "8B",
                        "quantization_level": "Q4_0",
                    },
                },
                {
                    "name": "qwen:2b",
                    "details": {
                        "family": "qwen",
                        "parameter_size": "2.5B",
                        "quantization_level": "Q8_0",
                    },
                },
            ]
        })

    async def generate(request):
        body = await request.json()
        model = body.get("model", "")

        if model == "nonexistent":
            return web.Response(status=404, text="model not found")

        tokens = ["Hello", " world", "!"]
        resp = web.StreamResponse()
        resp.content_type = "application/x-ndjson"
        await resp.prepare(request)

        for tok in tokens:
            chunk = json.dumps({"response": tok, "done": False}) + "\n"
            await resp.write(chunk.encode())

        final = json.dumps({
            "response": "",
            "done": True,
            "done_reason": "stop",
            "eval_count": len(tokens),
        }) + "\n"
        await resp.write(final.encode())
        return resp

    async def chat(request):
        body = await request.json()
        model = body.get("model", "")

        if model == "nonexistent":
            return web.Response(status=404, text="model not found")

        tokens = ["Hi", " there"]
        resp = web.StreamResponse()
        resp.content_type = "application/x-ndjson"
        await resp.prepare(request)

        for tok in tokens:
            chunk = json.dumps({
                "message": {"role": "assistant", "content": tok},
                "done": False,
            }) + "\n"
            await resp.write(chunk.encode())

        final = json.dumps({
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "stop",
        }) + "\n"
        await resp.write(final.encode())
        return resp

    async def ps(request):
        return web.json_response({
            "models": [
                {
                    "name": "llama3.1:8b-q4_0",
                    "size_vram": 4 * 1024 * 1024 * 1024,
                },
            ]
        })

    async def generate_warmup(request):
        body = await request.json()
        if body.get("prompt") == "" and "keep_alive" in body:
            resp = web.StreamResponse()
            resp.content_type = "application/x-ndjson"
            await resp.prepare(request)
            final = json.dumps({"response": "", "done": True}) + "\n"
            await resp.write(final.encode())
            return resp
        return await generate(request)

    app.router.add_get("/", root)
    app.router.add_get("/api/tags", tags)
    app.router.add_post("/api/generate", generate_warmup)
    app.router.add_post("/api/chat", chat)
    app.router.add_get("/api/ps", ps)
    return app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def mock_server():
    app = _build_mock_app()
    server = TestServer(app)
    await server.start_server()
    yield server
    await server.close()


@pytest_asyncio.fixture
async def adapter(mock_server):
    a = OllamaAdapter(base_url=str(mock_server.make_url("")))
    yield a
    await a.close()


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestParseParamCount:
    def test_integer(self):
        assert _parse_param_count("8B") == 8000

    def test_decimal(self):
        assert _parse_param_count("2.5B") == 2500

    def test_large(self):
        assert _parse_param_count("70B") == 70000

    def test_zero(self):
        assert _parse_param_count("0B") == 0

    def test_no_suffix(self):
        assert _parse_param_count("123") == 0

    def test_whitespace(self):
        assert _parse_param_count("  8B  ") == 8000

    def test_invalid(self):
        assert _parse_param_count("xyzB") == 0


class TestGetContextLength:
    def test_llama31(self):
        assert _get_context_length("llama3.1:8b", {}) == 131072

    def test_llama3(self):
        assert _get_context_length("llama3:8b", {}) == 8192

    def test_qwen(self):
        assert _get_context_length("qwen:2b", {}) == 32768

    def test_unknown(self):
        assert _get_context_length("some-random-model", {}) == 4096

    def test_case_insensitive(self):
        assert _get_context_length("Mistral:7b", {}) == 32768

    def test_deepseek(self):
        assert _get_context_length("deepseek-coder:7b", {}) == 65536


class TestEstimateFreeMemory:
    def test_zero_used(self):
        assert _estimate_free_memory(0) == 16000

    def test_some_used(self):
        assert _estimate_free_memory(4 * 1024 * 1024 * 1024) == 16000 - 4096

    def test_all_used(self):
        assert _estimate_free_memory(20 * 1024 * 1024 * 1024) == 0


# ---------------------------------------------------------------------------
# Adapter tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestListModels:
    async def test_returns_models(self, adapter):
        models = await adapter.list_models()
        assert len(models) == 2
        assert all(isinstance(m, ModelInfo) for m in models)

    async def test_model_fields(self, adapter):
        models = await adapter.list_models()
        llama = next(m for m in models if "llama" in m.model_id)
        assert llama.model_family == "llama"
        assert llama.parameter_count == 8000
        assert llama.quantization == "Q4_0"
        assert llama.context_length == 131072

    async def test_qwen_model(self, adapter):
        models = await adapter.list_models()
        qwen = next(m for m in models if "qwen" in m.model_id)
        assert qwen.parameter_count == 2500
        assert qwen.context_length == 32768


@pytest.mark.asyncio
class TestGenerateCompletion:
    async def test_streaming_tokens(self, adapter):
        req = PromptRequest(
            request_id="t1",
            model_id="llama3.1:8b-q4_0",
            prompt="Say hello",
        )
        tokens = []
        async for chunk in adapter.generate(req):
            tokens.append(chunk)

        assert len(tokens) == 4  # 3 content + 1 final
        assert tokens[0].token == "Hello"
        assert tokens[0].finish_reason is None
        assert tokens[-1].finish_reason == "stop"

    async def test_full_text(self, adapter):
        req = PromptRequest(
            request_id="t2",
            model_id="llama3.1:8b-q4_0",
            prompt="Say hello",
        )
        text = ""
        async for chunk in adapter.generate(req):
            text += chunk.token
        assert text == "Hello world!"


@pytest.mark.asyncio
class TestGenerateChat:
    async def test_chat_streaming(self, adapter):
        req = PromptRequest(
            request_id="t3",
            model_id="llama3.1:8b-q4_0",
            messages=[{"role": "user", "content": "Hi"}],
        )
        tokens = []
        async for chunk in adapter.generate(req):
            tokens.append(chunk)

        assert len(tokens) == 3  # 2 content + 1 final
        assert tokens[0].token == "Hi"
        assert tokens[1].token == " there"
        assert tokens[-1].finish_reason == "stop"

    async def test_chat_with_system_prompt(self, adapter):
        req = PromptRequest(
            request_id="t4",
            model_id="llama3.1:8b-q4_0",
            messages=[{"role": "user", "content": "Hi"}],
            system_prompt="Be brief.",
        )
        tokens = []
        async for chunk in adapter.generate(req):
            tokens.append(chunk)
        assert len(tokens) == 3


@pytest.mark.asyncio
class TestGenerateError:
    async def test_model_not_found(self, adapter):
        req = PromptRequest(
            request_id="t5",
            model_id="nonexistent",
            prompt="hello",
        )
        with pytest.raises(BackendError, match="404"):
            async for _ in adapter.generate(req):
                pass


@pytest.mark.asyncio
class TestCancel:
    async def test_cancel_yields_cancelled(self, adapter):
        req = PromptRequest(
            request_id="t6",
            model_id="llama3.1:8b-q4_0",
            prompt="long text",
        )
        tokens = []
        async for chunk in adapter.generate(req):
            tokens.append(chunk)
            if len(tokens) == 1:
                await adapter.cancel("t6")

        last = tokens[-1]
        assert last.finish_reason == "cancelled"

    async def test_request_cleaned_up_after_cancel(self, adapter):
        req = PromptRequest(
            request_id="t7",
            model_id="llama3.1:8b-q4_0",
            prompt="text",
        )
        gen = adapter.generate(req)
        async for chunk in gen:
            await adapter.cancel("t7")
            break
        await gen.aclose()

        assert "t7" not in adapter._active_requests


@pytest.mark.asyncio
class TestGetStatus:
    async def test_status_fields(self, adapter):
        status = await adapter.get_status()
        assert isinstance(status, BackendStatus)
        assert status.max_concurrent == 4
        assert "llama3.1:8b-q4_0" in status.models_loaded
        assert status.gpu_memory_free_mb == 16000 - 4096


@pytest.mark.asyncio
class TestHealthCheck:
    async def test_healthy(self, adapter):
        assert await adapter.health_check() is True

    async def test_unhealthy(self):
        a = OllamaAdapter(base_url="http://127.0.0.1:1")
        result = await a.health_check()
        assert result is False
        await a.close()


@pytest.mark.asyncio
class TestWarmup:
    async def test_warmup_success(self, adapter):
        result = await adapter.warmup("llama3.1:8b-q4_0")
        assert result is True

    async def test_warmup_unreachable(self):
        a = OllamaAdapter(base_url="http://127.0.0.1:1")
        result = await a.warmup("llama3:8b")
        assert result is False
        await a.close()


@pytest.mark.asyncio
class TestAdapterIdentity:
    async def test_backend_type(self, adapter):
        assert adapter.backend_type() == "ollama"

    async def test_trust_tier(self, adapter):
        assert adapter.trust_tier() == "unverified"

    async def test_integration_level(self, adapter):
        assert adapter.integration_level() == 1

    async def test_no_attestation(self, adapter):
        assert adapter.get_weights_hash("any") is None
        assert adapter.sign_output(
            PromptRequest(request_id="x", model_id="m"),
            "m", [1], 0,
        ) is None
