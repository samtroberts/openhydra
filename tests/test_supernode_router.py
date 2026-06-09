"""Tests for supernode.router — SupernodeRouter with mock adapters."""

import asyncio
import time

import pytest
import pytest_asyncio

from supernode.adapter import (
    SupernodeAdapter,
    PromptRequest,
    TokenChunk,
    ModelInfo,
    BackendStatus,
    BackendError,
)
from supernode.router import SupernodeRouter


# ---------------------------------------------------------------------------
# Mock adapter
# ---------------------------------------------------------------------------

class MockAdapter(SupernodeAdapter):
    def __init__(self, models=None, tokens=None, healthy=True):
        self._models = models or [
            ModelInfo(
                model_id="test-model:7b",
                model_family="test",
                parameter_count=7000,
                quantization="Q4_0",
                context_length=8192,
            ),
        ]
        self._tokens = tokens or ["Hello", " world", "!"]
        self._healthy = healthy
        self._last_request: PromptRequest | None = None
        self._cancel_called: list[str] = []

    async def list_models(self) -> list[ModelInfo]:
        return self._models

    async def generate(self, request: PromptRequest):
        self._last_request = request
        for tok in self._tokens:
            yield TokenChunk(token=tok)
        yield TokenChunk(token="", finish_reason="stop")

    async def cancel(self, request_id: str) -> None:
        self._cancel_called.append(request_id)

    async def get_status(self) -> BackendStatus:
        return BackendStatus(
            current_load=0.25,
            active_requests=1,
            max_concurrent=4,
            gpu_memory_free_mb=12000,
            models_loaded=[m.model_id for m in self._models],
        )

    async def health_check(self) -> bool:
        return self._healthy

    async def warmup(self, model_id: str) -> bool:
        return True


class FailingAdapter(SupernodeAdapter):
    async def list_models(self):
        raise ConnectionError("unreachable")

    async def generate(self, request):
        raise BackendError("generation failed")
        yield  # make it a generator

    async def cancel(self, request_id):
        pass

    async def get_status(self):
        raise ConnectionError("unreachable")

    async def health_check(self):
        return False

    async def warmup(self, model_id):
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def loop():
    l = asyncio.new_event_loop()
    yield l
    l.close()


@pytest.fixture
def mock_adapter():
    return MockAdapter()


@pytest.fixture
def router(loop, mock_adapter):
    r = SupernodeRouter(loop=loop)
    r.register_adapter("test-ollama", mock_adapter)
    return r


# ---------------------------------------------------------------------------
# Tests: list_models_openai
# ---------------------------------------------------------------------------

class TestListModels:
    def test_returns_openai_format(self, router):
        result = router.list_models_openai()
        assert result["object"] == "list"
        assert len(result["data"]) == 1
        model = result["data"][0]
        assert model["id"] == "test-model:7b"
        assert model["object"] == "model"
        assert model["owned_by"] == "openhydra"

    def test_openhydra_metadata(self, router):
        result = router.list_models_openai()
        meta = result["data"][0]["openhydra"]
        assert meta["family"] == "test"
        assert meta["parameter_count"] == 7000
        assert meta["quantization"] == "Q4_0"
        assert meta["context_length"] == 8192

    def test_multiple_adapters(self, loop):
        r = SupernodeRouter(loop=loop)
        r.register_adapter("a", MockAdapter(models=[
            ModelInfo("m1", "f1", 1000, "Q4", 4096),
        ]))
        r.register_adapter("b", MockAdapter(models=[
            ModelInfo("m2", "f2", 2000, "Q8", 8192),
        ]))
        result = r.list_models_openai()
        assert len(result["data"]) == 2
        ids = {m["id"] for m in result["data"]}
        assert ids == {"m1", "m2"}

    def test_failing_adapter_skipped(self, loop):
        r = SupernodeRouter(loop=loop)
        r.register_adapter("good", MockAdapter())
        r.register_adapter("bad", FailingAdapter())
        result = r.list_models_openai()
        assert len(result["data"]) == 1


# ---------------------------------------------------------------------------
# Tests: list_supernodes
# ---------------------------------------------------------------------------

class TestListSupernodes:
    def test_healthy_supernode(self, router):
        result = router.list_supernodes()
        assert len(result) == 1
        sn = result[0]
        assert sn["name"] == "test-ollama"
        assert sn["backend"] == "mock"
        assert sn["trust_tier"] == "unverified"
        assert sn["healthy"] is True
        assert sn["status"]["active_requests"] == 1
        assert "test-model:7b" in sn["models"]

    def test_unhealthy_supernode(self, loop):
        r = SupernodeRouter(loop=loop)
        r.register_adapter("down", MockAdapter(healthy=False))
        result = r.list_supernodes()
        assert result[0]["healthy"] is False
        assert result[0]["status"] is None

    def test_failing_supernode(self, loop):
        r = SupernodeRouter(loop=loop)
        r.register_adapter("broken", FailingAdapter())
        result = r.list_supernodes()
        assert result[0]["healthy"] is False


# ---------------------------------------------------------------------------
# Tests: chat_completion (non-streaming)
# ---------------------------------------------------------------------------

class TestChatCompletion:
    def test_basic_response(self, router):
        body = {
            "model": "test-model:7b",
            "messages": [{"role": "user", "content": "hello"}],
        }
        result = router.chat_completion(body, request_id="req-1")
        assert result["id"] == "req-1"
        assert result["object"] == "chat.completion"
        assert result["model"] == "test-model:7b"
        content = result["choices"][0]["message"]["content"]
        assert content == "Hello world!"
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_usage_counts(self, router):
        body = {"model": "test-model:7b", "messages": [{"role": "user", "content": "hi"}]}
        result = router.chat_completion(body)
        assert result["usage"]["completion_tokens"] == 4  # 3 text + 1 final

    def test_request_params_passed(self, router, mock_adapter):
        body = {
            "model": "test-model:7b",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 100,
            "temperature": 0.5,
            "stop": [".", "\n"],
        }
        router.chat_completion(body)
        req = mock_adapter._last_request
        assert req.max_tokens == 100
        assert req.temperature == 0.5
        assert req.stop == [".", "\n"]

    def test_auto_request_id(self, router):
        body = {"model": "test-model:7b", "messages": [{"role": "user", "content": "hi"}]}
        result = router.chat_completion(body)
        assert result["id"]  # should be a UUID

    def test_no_adapter_raises(self, loop):
        r = SupernodeRouter(loop=loop)
        body = {"model": "test-model:7b", "messages": [{"role": "user", "content": "hi"}]}
        with pytest.raises(BackendError, match="No adapter"):
            r.chat_completion(body)


# ---------------------------------------------------------------------------
# Tests: chat_completion_stream
# ---------------------------------------------------------------------------

class TestChatCompletionStream:
    def test_yields_tokens(self, router):
        body = {"model": "test-model:7b", "messages": [{"role": "user", "content": "hi"}]}
        tokens = list(router.chat_completion_stream(body))
        assert tokens == ["Hello", " world", "!"]

    def test_empty_tokens_filtered(self, loop):
        adapter = MockAdapter(tokens=["a", "", "b"])
        r = SupernodeRouter(loop=loop)
        r.register_adapter("x", adapter)
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        tokens = list(r.chat_completion_stream(body))
        assert tokens == ["a", "b"]


# ---------------------------------------------------------------------------
# Tests: text_completion
# ---------------------------------------------------------------------------

class TestTextCompletion:
    def test_basic_response(self, router):
        body = {"model": "test-model:7b", "prompt": "Once upon"}
        result = router.text_completion(body, request_id="req-2")
        assert result["object"] == "text_completion"
        assert result["choices"][0]["text"] == "Hello world!"
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_prompt_passed(self, router, mock_adapter):
        body = {"model": "test-model:7b", "prompt": "Once upon a time"}
        router.text_completion(body)
        assert mock_adapter._last_request.prompt == "Once upon a time"
        assert mock_adapter._last_request.messages is None


class TestTextCompletionStream:
    def test_yields_tokens(self, router):
        body = {"model": "test-model:7b", "prompt": "Once upon"}
        tokens = list(router.text_completion_stream(body))
        assert tokens == ["Hello", " world", "!"]


# ---------------------------------------------------------------------------
# Tests: body parsing edge cases
# ---------------------------------------------------------------------------

class TestBodyParsing:
    def test_stop_string_to_list(self, router, mock_adapter):
        body = {"model": "m", "messages": [{"role": "user", "content": "x"}], "stop": "."}
        router.chat_completion(body)
        assert mock_adapter._last_request.stop == ["."]

    def test_response_format_dict(self, router, mock_adapter):
        body = {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "response_format": {"type": "json_object"},
        }
        router.chat_completion(body)
        assert mock_adapter._last_request.response_format == "json_object"

    def test_response_format_string(self, router, mock_adapter):
        body = {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "response_format": "json",
        }
        router.chat_completion(body)
        assert mock_adapter._last_request.response_format == "json"

    def test_defaults(self, router, mock_adapter):
        body = {"model": "m", "messages": [{"role": "user", "content": "x"}]}
        router.chat_completion(body)
        req = mock_adapter._last_request
        assert req.temperature == 0.7
        assert req.top_p == 0.9
        assert req.max_tokens == 512


# ---------------------------------------------------------------------------
# Tests: adapter selection
# ---------------------------------------------------------------------------

class TestAdapterSelection:
    def test_single_adapter_always_selected(self, router, mock_adapter):
        body = {"model": "nonexistent", "messages": [{"role": "user", "content": "hi"}]}
        result = router.chat_completion(body)
        assert mock_adapter._last_request is not None

    def test_model_cache_routing(self, loop):
        a1 = MockAdapter(
            models=[ModelInfo("m1", "f", 1000, "Q4", 4096)],
            tokens=["from-a1"],
        )
        a2 = MockAdapter(
            models=[ModelInfo("m2", "f", 1000, "Q4", 4096)],
            tokens=["from-a2"],
        )
        r = SupernodeRouter(loop=loop)
        r.register_adapter("a1", a1)
        r.register_adapter("a2", a2)
        r.list_models_openai()  # populate cache

        body = {"model": "m2", "messages": [{"role": "user", "content": "hi"}]}
        result = r.chat_completion(body)
        assert a2._last_request is not None
        assert a1._last_request is None
