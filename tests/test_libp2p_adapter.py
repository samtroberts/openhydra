"""Tests for supernode.libp2p_adapter — LibP2PAdapter."""

import asyncio
import threading
import time

import pytest

from supernode.adapter import BackendError, PromptRequest
from supernode.libp2p_adapter import LibP2PAdapter
from supernode.manifest import SupernodeManifest, ModelCapability
from supernode.prompt_protocol import (
    METHOD_PROMPT_REQUEST,
    PromptChunk as WireChunk,
)


def _make_manifest(**kwargs):
    defaults = dict(
        peer_id="test-peer",
        libp2p_peer_id="12D3KooWTest",
        backend_type="ollama",
        version="0.1.0",
        models=[
            ModelCapability(
                model_id="test:1b",
                model_family="test",
                parameter_count=1_000_000_000,
                quantization="Q4_0",
                context_length=4096,
            ),
        ],
    )
    defaults.update(kwargs)
    return SupernodeManifest(**defaults)


class MockP2PNode:
    def __init__(self, chunks=None, error_on_open=False):
        self._opened_streams: list[tuple[str, bytes]] = []
        self._closed_streams: list[str] = []
        self._chunks = chunks or []
        self._chunk_index = 0
        self._error_on_open = error_on_open
        self._connected = True
        self._stream_counter = 0

    def open_prompt_stream(self, peer_id: str, data: bytes) -> str:
        if self._error_on_open:
            raise RuntimeError("connection failed")
        self._stream_counter += 1
        sid = f"pstream-client-{self._stream_counter}"
        self._opened_streams.append((sid, data))
        return sid

    def poll_prompt_chunk(self, stream_id: str, timeout_ms: int = 500):
        if self._chunk_index < len(self._chunks):
            chunk = self._chunks[self._chunk_index]
            self._chunk_index += 1
            return chunk
        return b""  # empty = stream closed

    def close_prompt_stream(self, stream_id: str):
        self._closed_streams.append(stream_id)

    def is_peer_connected(self, peer_id: str) -> bool:
        return self._connected


def _make_token_chunk(token: str, request_id: str = "r1") -> bytes:
    chunk = WireChunk(request_id=request_id, chunk_type="token", token=token)
    return bytes([METHOD_PROMPT_REQUEST]) + chunk.to_cbor()


def _make_done_chunk(request_id: str = "r1", finish_reason: str = "stop") -> bytes:
    chunk = WireChunk(
        request_id=request_id,
        chunk_type="done",
        finish_reason=finish_reason,
    )
    return bytes([METHOD_PROMPT_REQUEST]) + chunk.to_cbor()


def _make_error_chunk(error: str, request_id: str = "r1") -> bytes:
    chunk = WireChunk(
        request_id=request_id, chunk_type="error", error=error,
    )
    return bytes([METHOD_PROMPT_REQUEST]) + chunk.to_cbor()


class TestLibP2PAdapter:
    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    @pytest.fixture(autouse=True)
    def _loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        yield loop
        loop.close()

    def test_generate_tokens(self, _loop):
        chunks = [
            _make_token_chunk("Hello"),
            _make_token_chunk(" world"),
            _make_done_chunk(),
        ]
        node = MockP2PNode(chunks=chunks)
        manifest = _make_manifest()
        adapter = LibP2PAdapter(node, "12D3KooWTest", manifest)

        tokens = []

        async def run():
            req = PromptRequest(request_id="r1", model_id="test:1b", prompt="hi")
            async for chunk in adapter.generate(req):
                tokens.append(chunk.token)

        _loop.run_until_complete(run())
        assert tokens == ["Hello", " world", ""]
        assert len(node._closed_streams) == 1

    def test_generate_error(self, _loop):
        chunks = [_make_error_chunk("GPU OOM")]
        node = MockP2PNode(chunks=chunks)
        adapter = LibP2PAdapter(node, "12D3KooWTest", _make_manifest())

        async def run():
            req = PromptRequest(request_id="r1", model_id="test:1b", prompt="hi")
            async for _ in adapter.generate(req):
                pass

        with pytest.raises(BackendError, match="GPU OOM"):
            _loop.run_until_complete(run())

    def test_list_models(self, _loop):
        adapter = LibP2PAdapter(MockP2PNode(), "12D3KooWTest", _make_manifest())
        models = _loop.run_until_complete(adapter.list_models())
        assert len(models) == 1
        assert models[0].model_id == "test:1b"

    def test_health_check(self, _loop):
        node = MockP2PNode()
        manifest = _make_manifest()
        manifest.timestamp = int(time.time() * 1000)
        adapter = LibP2PAdapter(node, "12D3KooWTest", manifest)
        assert _loop.run_until_complete(adapter.health_check()) is True

        node._connected = False
        assert _loop.run_until_complete(adapter.health_check()) is False

    def test_backend_type(self):
        adapter = LibP2PAdapter(MockP2PNode(), "12D3KooWTest", _make_manifest())
        assert adapter.backend_type() == "libp2p-ollama"

    def test_open_failure(self, _loop):
        node = MockP2PNode(error_on_open=True)
        adapter = LibP2PAdapter(node, "12D3KooWTest", _make_manifest())

        async def run():
            req = PromptRequest(request_id="r1", model_id="test:1b", prompt="hi")
            async for _ in adapter.generate(req):
                pass

        with pytest.raises(RuntimeError, match="connection failed"):
            _loop.run_until_complete(run())
