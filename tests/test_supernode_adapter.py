"""Tests for supernode.adapter — ABC contract and dataclass invariants."""

import pytest

from supernode.adapter import (
    SupernodeAdapter,
    PromptRequest,
    TokenChunk,
    ModelInfo,
    BackendStatus,
    BackendError,
)


# --- Dataclass construction ---


class TestPromptRequest:
    def test_minimal(self):
        r = PromptRequest(request_id="r1", model_id="llama3:8b")
        assert r.request_id == "r1"
        assert r.model_id == "llama3:8b"
        assert r.prompt is None
        assert r.messages is None
        assert r.max_tokens == 512
        assert r.temperature == 0.7
        assert r.stream is True
        assert r.response_format == "text"

    def test_chat_mode(self):
        msgs = [{"role": "user", "content": "hi"}]
        r = PromptRequest(
            request_id="r2",
            model_id="qwen:2b",
            messages=msgs,
            system_prompt="Be helpful.",
        )
        assert r.messages == msgs
        assert r.system_prompt == "Be helpful."

    def test_completion_mode(self):
        r = PromptRequest(
            request_id="r3",
            model_id="phi:3b",
            prompt="Once upon a time",
            max_tokens=100,
            temperature=0.0,
            stop=[".", "\n"],
        )
        assert r.prompt == "Once upon a time"
        assert r.stop == [".", "\n"]
        assert r.temperature == 0.0

    def test_json_format(self):
        r = PromptRequest(
            request_id="r4",
            model_id="m",
            prompt="x",
            response_format="json",
        )
        assert r.response_format == "json"


class TestTokenChunk:
    def test_streaming_chunk(self):
        c = TokenChunk(token="hello")
        assert c.token == "hello"
        assert c.token_id is None
        assert c.finish_reason is None

    def test_final_chunk(self):
        c = TokenChunk(token=".", token_id=42, finish_reason="stop")
        assert c.finish_reason == "stop"
        assert c.token_id == 42

    def test_length_finish(self):
        c = TokenChunk(token="", finish_reason="length")
        assert c.finish_reason == "length"


class TestModelInfo:
    def test_construction(self):
        m = ModelInfo(
            model_id="llama3.1:8b-q4_0",
            model_family="llama",
            parameter_count=8000,
            quantization="Q4_0",
            context_length=131072,
        )
        assert m.model_id == "llama3.1:8b-q4_0"
        assert m.parameter_count == 8000
        assert m.supports_streaming is True
        assert m.supports_system_prompt is True

    def test_no_streaming(self):
        m = ModelInfo(
            model_id="m",
            model_family="f",
            parameter_count=0,
            quantization="none",
            context_length=2048,
            supports_streaming=False,
        )
        assert m.supports_streaming is False


class TestBackendStatus:
    def test_defaults(self):
        s = BackendStatus(
            current_load=0.5,
            active_requests=2,
            max_concurrent=4,
            gpu_memory_free_mb=8000,
        )
        assert s.models_loaded == []
        assert s.current_load == 0.5

    def test_with_models(self):
        s = BackendStatus(
            current_load=0.0,
            active_requests=0,
            max_concurrent=4,
            gpu_memory_free_mb=16000,
            models_loaded=["llama3:8b", "qwen:2b"],
        )
        assert len(s.models_loaded) == 2


# --- ABC contract ---


class TestABCContract:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            SupernodeAdapter()

    def test_abstract_methods_required(self):
        # A subclass missing any abstract method should fail
        class Incomplete(SupernodeAdapter):
            async def list_models(self):
                return []

        with pytest.raises(TypeError):
            Incomplete()

    def test_defaults_for_l1(self):
        class Stub(SupernodeAdapter):
            async def list_models(self):
                return []

            async def generate(self, request):
                yield TokenChunk(token="x", finish_reason="stop")

            async def cancel(self, request_id):
                pass

            async def get_status(self):
                return BackendStatus(0.0, 0, 1, 0)

            async def health_check(self):
                return True

            async def warmup(self, model_id):
                return True

        s = Stub()
        assert s.trust_tier() == "unverified"
        assert s.integration_level() == 1
        assert s.get_weights_hash("any") is None
        assert s.sign_output(
            PromptRequest(request_id="x", model_id="m"),
            "m", [1, 2, 3], 0,
        ) is None

    def test_backend_type_derivation(self):
        class FooAdapter(SupernodeAdapter):
            async def list_models(self):
                return []

            async def generate(self, request):
                yield TokenChunk(token="", finish_reason="stop")

            async def cancel(self, request_id):
                pass

            async def get_status(self):
                return BackendStatus(0.0, 0, 1, 0)

            async def health_check(self):
                return True

            async def warmup(self, model_id):
                return True

        assert FooAdapter().backend_type() == "foo"

    def test_backend_type_runtime_suffix(self):
        class BarRuntime(SupernodeAdapter):
            async def list_models(self):
                return []

            async def generate(self, request):
                yield TokenChunk(token="", finish_reason="stop")

            async def cancel(self, request_id):
                pass

            async def get_status(self):
                return BackendStatus(0.0, 0, 1, 0)

            async def health_check(self):
                return True

            async def warmup(self, model_id):
                return True

        assert BarRuntime().backend_type() == "bar"


class TestBackendError:
    def test_is_exception(self):
        assert issubclass(BackendError, Exception)

    def test_message(self):
        e = BackendError("something went wrong")
        assert str(e) == "something went wrong"
