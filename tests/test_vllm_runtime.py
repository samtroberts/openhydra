from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from supernode.adapter import PromptRequest
from supernode.vllm_runtime import VLLMRuntime


def _make_runtime():
    key = Ed25519PrivateKey.generate()
    return VLLMRuntime(
        peer_id="test-peer",
        private_key=key,
        model_id="test-vllm",
        model_path="/tmp/fake-model",
    ), key


@pytest.mark.asyncio
async def test_trust_tier():
    rt, _ = _make_runtime()
    assert rt.trust_tier() == "attested"
    assert rt.integration_level() == 3


@pytest.mark.asyncio
async def test_health_check_before_warmup():
    rt, _ = _make_runtime()
    assert not await rt.health_check()


@pytest.mark.asyncio
async def test_list_models():
    rt, _ = _make_runtime()
    models = await rt.list_models()
    assert len(models) == 1
    assert models[0].model_family == "vllm"


@pytest.mark.asyncio
async def test_generate_emits_deltas():
    rt, _ = _make_runtime()

    async def mock_generate(prompt, params, request_id=None):
        cumulative = ""
        for word in ["Hello", " world", "!"]:
            cumulative += word
            output = SimpleNamespace(text=cumulative, finish_reason=None)
            yield SimpleNamespace(outputs=[output])
        output = SimpleNamespace(text=cumulative, finish_reason="stop")
        yield SimpleNamespace(outputs=[output])

    mock_engine = MagicMock()
    mock_engine.generate = mock_generate
    mock_engine.abort = AsyncMock()
    rt._engine = mock_engine

    mock_sp = MagicMock()
    req = PromptRequest(request_id="r1", model_id="test-vllm", prompt="hi")
    tokens = []
    with patch("supernode.vllm_runtime.SamplingParams", mock_sp):
        async for chunk in rt.generate(req):
            tokens.append(chunk.token)

    assert "Hello" in tokens
    assert " world" in tokens
    assert "!" in tokens
    assert tokens[-1] == ""  # finish sentinel with finish_reason="stop"


@pytest.mark.asyncio
async def test_generate_not_loaded_raises():
    rt, _ = _make_runtime()
    req = PromptRequest(request_id="r1", model_id="test-vllm", prompt="hi")
    with pytest.raises(RuntimeError, match="not loaded"):
        async for _ in rt.generate(req):
            pass


@pytest.mark.asyncio
async def test_cancel_delegates_to_engine():
    rt, _ = _make_runtime()
    mock_engine = MagicMock()
    mock_engine.abort = AsyncMock()
    rt._engine = mock_engine
    await rt.cancel("req-42")
    mock_engine.abort.assert_awaited_once_with("req-42")


@pytest.mark.asyncio
async def test_sign_output():
    rt, _ = _make_runtime()
    rt._weights_hash = "face" * 16
    req = PromptRequest(request_id="r1", model_id="test-vllm", prompt="hello")
    sig = rt.sign_output(req, "test-vllm", [5, 6, 7], int(time.time() * 1000))
    assert len(sig) == 64
