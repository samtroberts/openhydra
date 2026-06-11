from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from supernode.adapter import PromptRequest
from supernode.mlx_attested import MLXAttestedRuntime


def _make_runtime(tmp_path: Path):
    (tmp_path / "model.safetensors").write_bytes(b"\x00" * 256)
    (tmp_path / "config.json").write_text("{}")
    key = Ed25519PrivateKey.generate()
    return MLXAttestedRuntime(
        peer_id="test-peer",
        private_key=key,
        model_id="test-mlx",
        model_path=tmp_path,
    ), key


@pytest.mark.asyncio
async def test_trust_tier(tmp_path: Path):
    rt, _ = _make_runtime(tmp_path)
    assert rt.trust_tier() == "attested"
    assert rt.integration_level() == 3


@pytest.mark.asyncio
async def test_health_check_before_warmup(tmp_path: Path):
    rt, _ = _make_runtime(tmp_path)
    assert not await rt.health_check()


@pytest.mark.asyncio
async def test_list_models(tmp_path: Path):
    rt, _ = _make_runtime(tmp_path)
    models = await rt.list_models()
    assert len(models) == 1
    assert models[0].model_family == "mlx"


@pytest.mark.asyncio
async def test_generate_with_mock(tmp_path: Path):
    rt, _ = _make_runtime(tmp_path)
    rt._model = MagicMock()
    rt._tokenizer = MagicMock()

    with patch("supernode.mlx_attested.mlx_lm") as mock_mlx:
        mock_mlx.stream_generate.return_value = iter(["Hello", " world"])

        req = PromptRequest(request_id="r1", model_id="test-mlx", prompt="hi")
        tokens = []
        async for chunk in rt.generate(req):
            tokens.append(chunk.token)

        assert "Hello" in tokens
        assert " world" in tokens
        assert tokens[-1] == ""  # finish sentinel


@pytest.mark.asyncio
async def test_generate_not_loaded_raises(tmp_path: Path):
    rt, _ = _make_runtime(tmp_path)
    req = PromptRequest(request_id="r1", model_id="test-mlx", prompt="hi")
    with pytest.raises(RuntimeError, match="not loaded"):
        async for _ in rt.generate(req):
            pass


@pytest.mark.asyncio
async def test_sign_output(tmp_path: Path):
    rt, key = _make_runtime(tmp_path)
    rt._weights_hash = "1234" * 16
    req = PromptRequest(request_id="r1", model_id="test-mlx", prompt="hello")
    sig = rt.sign_output(req, "test-mlx", [10, 20], int(time.time() * 1000))
    assert len(sig) == 64
