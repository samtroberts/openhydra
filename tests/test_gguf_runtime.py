from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from supernode.adapter import PromptRequest, TokenChunk
from supernode.gguf_runtime import GGUFRuntime


def _make_runtime(tmp_path: Path):
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"\x00" * 256)
    key = Ed25519PrivateKey.generate()
    return GGUFRuntime(
        peer_id="test-peer",
        private_key=key,
        model_id="test-gguf",
        model_path=model_file,
    ), key


@pytest.mark.asyncio
async def test_health_check_before_warmup(tmp_path: Path):
    rt, _ = _make_runtime(tmp_path)
    assert not await rt.health_check()


@pytest.mark.asyncio
async def test_trust_tier(tmp_path: Path):
    rt, _ = _make_runtime(tmp_path)
    assert rt.trust_tier() == "attested"
    assert rt.integration_level() == 3


@pytest.mark.asyncio
async def test_list_models(tmp_path: Path):
    rt, _ = _make_runtime(tmp_path)
    models = await rt.list_models()
    assert len(models) == 1
    assert models[0].model_id == "test-gguf"
    assert models[0].model_family == "gguf"


@pytest.mark.asyncio
async def test_generate_with_mock_llama(tmp_path: Path):
    rt, key = _make_runtime(tmp_path)

    mock_llm = MagicMock()
    mock_llm.create_completion.return_value = iter([
        {"choices": [{"text": "Hello", "finish_reason": None}]},
        {"choices": [{"text": " world", "finish_reason": None}]},
        {"choices": [{"text": "", "finish_reason": "stop"}]},
    ])

    rt._llm = mock_llm

    req = PromptRequest(request_id="r1", model_id="test-gguf", prompt="hi")
    tokens = []
    async for chunk in rt.generate(req):
        tokens.append(chunk.token)

    assert tokens == ["Hello", " world", ""]
    mock_llm.create_completion.assert_called_once()
    call_kwargs = mock_llm.create_completion.call_args
    assert call_kwargs[0][0] == "hi"
    assert call_kwargs[1]["stream"] is True


@pytest.mark.asyncio
async def test_generate_with_messages(tmp_path: Path):
    rt, _ = _make_runtime(tmp_path)

    mock_llm = MagicMock()
    mock_llm.create_completion.return_value = iter([
        {"choices": [{"text": "response", "finish_reason": "stop"}]},
    ])
    rt._llm = mock_llm

    req = PromptRequest(
        request_id="r1",
        model_id="test-gguf",
        messages=[
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "hi"},
        ],
    )
    tokens = []
    async for chunk in rt.generate(req):
        tokens.append(chunk.token)

    assert tokens == ["response"]
    prompt_arg = mock_llm.create_completion.call_args[0][0]
    assert "system: Be helpful" in prompt_arg
    assert "user: hi" in prompt_arg


@pytest.mark.asyncio
async def test_generate_not_loaded_raises(tmp_path: Path):
    rt, _ = _make_runtime(tmp_path)
    req = PromptRequest(request_id="r1", model_id="test-gguf", prompt="hi")
    with pytest.raises(RuntimeError, match="not loaded"):
        async for _ in rt.generate(req):
            pass


@pytest.mark.asyncio
async def test_sign_output_after_generate(tmp_path: Path):
    rt, key = _make_runtime(tmp_path)
    rt._weights_hash = "abcd" * 16

    req = PromptRequest(request_id="r1", model_id="test-gguf", prompt="hello")
    ts = int(time.time() * 1000)
    sig = rt.sign_output(req, "test-gguf", [1, 2, 3], ts)
    assert isinstance(sig, bytes)
    assert len(sig) == 64


@pytest.mark.asyncio
async def test_get_status(tmp_path: Path):
    rt, _ = _make_runtime(tmp_path)
    status = await rt.get_status()
    assert status.active_requests == 0
    assert status.models_loaded == []
