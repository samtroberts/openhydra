from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from supernode.adapter import PromptRequest
from supernode.native_adapter import NativeAdapter


def _make_adapter():
    key = Ed25519PrivateKey.generate()
    mock_engine = MagicMock()
    return NativeAdapter(
        peer_id="test-peer",
        private_key=key,
        model_id="test-native",
        engine=mock_engine,
    ), mock_engine, key


@pytest.mark.asyncio
async def test_trust_tier():
    adapter, _, _ = _make_adapter()
    assert adapter.trust_tier() == "attested"
    assert adapter.integration_level() == 3


@pytest.mark.asyncio
async def test_health_check():
    adapter, _, _ = _make_adapter()
    assert await adapter.health_check()


@pytest.mark.asyncio
async def test_warmup_always_true():
    adapter, _, _ = _make_adapter()
    assert await adapter.warmup("any-model")


@pytest.mark.asyncio
async def test_list_models():
    adapter, _, _ = _make_adapter()
    models = await adapter.list_models()
    assert len(models) == 1
    assert models[0].model_family == "native"


@pytest.mark.asyncio
async def test_generate_streaming():
    adapter, engine, _ = _make_adapter()

    def fake_stream():
        yield "Hello"
        yield " world"

    engine.infer_stream.return_value = {"stream": fake_stream()}

    req = PromptRequest(request_id="r1", model_id="test-native", prompt="hi")
    tokens = []
    async for chunk in adapter.generate(req):
        tokens.append(chunk.token)

    assert tokens == ["Hello", " world", ""]
    engine.infer_stream.assert_called_once()
    call_kwargs = engine.infer_stream.call_args[1]
    assert call_kwargs["prompt"] == "hi"


@pytest.mark.asyncio
async def test_generate_non_streaming():
    adapter, engine, _ = _make_adapter()
    engine.infer_stream.return_value = {"response": "full text"}

    req = PromptRequest(request_id="r1", model_id="test-native", prompt="hi")
    tokens = []
    async for chunk in adapter.generate(req):
        tokens.append(chunk.token)

    assert tokens == ["full text"]


@pytest.mark.asyncio
async def test_generate_with_messages():
    adapter, engine, _ = _make_adapter()

    engine.infer_stream.return_value = {"stream": iter(["ok"])}

    req = PromptRequest(
        request_id="r1",
        model_id="test-native",
        messages=[{"role": "user", "content": "hello"}],
    )
    tokens = []
    async for chunk in adapter.generate(req):
        tokens.append(chunk.token)

    assert "ok" in tokens
    prompt_arg = engine.infer_stream.call_args[1]["prompt"]
    assert "user: hello" in prompt_arg


@pytest.mark.asyncio
async def test_sign_output():
    adapter, _, key = _make_adapter()
    adapter._weights_hash = "beef" * 16
    req = PromptRequest(request_id="r1", model_id="test-native", prompt="hello")
    sig = adapter.sign_output(req, "test-native", [1, 2], int(time.time() * 1000))
    assert len(sig) == 64
