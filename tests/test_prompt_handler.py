"""Tests for supernode.prompt_handler — PromptHandlerLoop dispatch."""

import threading
import time

import pytest

from supernode.adapter import (
    SupernodeAdapter,
    PromptRequest,
    TokenChunk,
    ModelInfo,
    BackendStatus,
    BackendError,
)
from supernode.prompt_protocol import (
    METHOD_PROMPT_REQUEST,
    METHOD_PROMPT_CANCEL,
    WirePromptRequest,
    PromptChunk as WireChunk,
)
from supernode.prompt_handler import PromptHandlerLoop, _wire_to_adapter_request


class StubAdapter(SupernodeAdapter):
    def __init__(self, tokens=None, delay=0.0, fail_before_token=False):
        self._tokens = tokens or ["Hello", " world"]
        self._delay = delay
        self._fail_before_token = fail_before_token
        self._generate_count = 0

    async def list_models(self):
        return [ModelInfo("test:1b", "test", 1000, "Q4_0", 4096)]

    async def generate(self, request):
        self._generate_count += 1
        if self._fail_before_token:
            raise BackendError("backend down")
        for tok in self._tokens:
            if self._delay:
                import asyncio
                await asyncio.sleep(self._delay)
            yield TokenChunk(token=tok)
        yield TokenChunk(token="", finish_reason="stop")

    async def cancel(self, request_id):
        pass

    async def get_status(self):
        return BackendStatus(0.0, 0, 4, 8000, ["test:1b"])

    async def health_check(self):
        return True

    async def warmup(self, model_id):
        return True


class MockP2PNode:
    def __init__(self):
        self._prompt_inbox: list[tuple[str, bytes]] = []
        self._responses: dict[str, bytes] = {}
        self._stream_chunks: dict[str, list[bytes]] = {}
        self._stream_closed: set[str] = set()
        self._lock = threading.Lock()
        self._has_request = threading.Event()

    def enqueue_prompt(self, req_id: str, data: bytes) -> None:
        with self._lock:
            self._prompt_inbox.append((req_id, data))
            self._has_request.set()

    def poll_prompt_request(self, timeout_ms=500):
        self._has_request.wait(timeout=timeout_ms / 1000)
        with self._lock:
            if self._prompt_inbox:
                item = self._prompt_inbox.pop(0)
                if not self._prompt_inbox:
                    self._has_request.clear()
                return item
            self._has_request.clear()
            return None

    def respond_prompt(self, request_id: str, data: bytes) -> None:
        with self._lock:
            self._responses[request_id] = data

    def send_prompt_chunk(self, stream_id: str, data: bytes) -> None:
        with self._lock:
            if stream_id not in self._stream_chunks:
                self._stream_chunks[stream_id] = []
            self._stream_chunks[stream_id].append(data)

    def close_prompt_stream(self, stream_id: str) -> None:
        with self._lock:
            self._stream_closed.add(stream_id)

    def get_response(self, req_id: str, timeout: float = 5.0) -> bytes | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if req_id in self._responses:
                    return self._responses.pop(req_id)
            time.sleep(0.05)
        return None

    def get_stream_chunks(self, stream_id: str, timeout: float = 5.0) -> list[bytes]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if stream_id in self._stream_closed:
                    return self._stream_chunks.get(stream_id, [])
            time.sleep(0.05)
        with self._lock:
            return self._stream_chunks.get(stream_id, [])


class TestWireToAdapterRequest:
    def test_conversion(self):
        wire = WirePromptRequest(
            request_id="r1",
            model_id="llama3:8b",
            prompt="hello",
            max_tokens=100,
            temperature=0.5,
        )
        req = _wire_to_adapter_request(wire)
        assert isinstance(req, PromptRequest)
        assert req.request_id == "r1"
        assert req.model_id == "llama3:8b"
        assert req.prompt == "hello"
        assert req.max_tokens == 100
        assert req.temperature == 0.5


class TestPromptHandlerLoop:
    @pytest.fixture
    def node(self):
        return MockP2PNode()

    @pytest.fixture
    def adapter(self):
        return StubAdapter()

    @pytest.fixture
    def handler(self, node, adapter):
        stop = threading.Event()
        h = PromptHandlerLoop(p2p_node=node, adapter=adapter, stop_event=stop)
        h.start()
        yield h
        stop.set()
        h.stop()

    def _send_prompt(self, node, req_id="r1", model_id="test:1b"):
        wire = WirePromptRequest(request_id=req_id, model_id=model_id, prompt="hello")
        payload = bytes([METHOD_PROMPT_REQUEST]) + wire.to_cbor()
        node.enqueue_prompt(req_id, payload)

    def test_basic_inference(self, node, handler):
        self._send_prompt(node)
        resp_data = node.get_response("r1")
        assert resp_data is not None
        assert resp_data[0] == METHOD_PROMPT_REQUEST
        chunk = WireChunk.from_cbor(resp_data[1:])
        assert chunk.chunk_type == "done"
        assert chunk.finish_reason == "stop"
        assert chunk.usage is not None
        assert chunk.usage.completion_tokens == 2

    def test_tps_and_ttft(self, node, handler):
        self._send_prompt(node)
        resp_data = node.get_response("r1")
        chunk = WireChunk.from_cbor(resp_data[1:])
        assert chunk.usage.tokens_per_second > 0
        assert chunk.usage.time_to_first_token_ms >= 0

    def test_backend_error_retryable(self, node):
        stop = threading.Event()
        adapter = StubAdapter(fail_before_token=True)
        handler = PromptHandlerLoop(p2p_node=node, adapter=adapter, stop_event=stop)
        handler.start()
        try:
            self._send_prompt(node, req_id="r2")
            resp_data = node.get_response("r2")
            assert resp_data is not None
            chunk = WireChunk.from_cbor(resp_data[1:])
            assert chunk.chunk_type == "error"
            assert chunk.retryable is True
            assert "backend down" in chunk.error
        finally:
            stop.set()
            handler.stop()

    def test_empty_request(self, node, handler):
        node.enqueue_prompt("r3", b"")
        resp_data = node.get_response("r3")
        assert resp_data is not None
        chunk = WireChunk.from_cbor(resp_data[1:])
        assert chunk.chunk_type == "error"
        assert "empty_request" in chunk.error

    def test_unknown_method(self, node, handler):
        node.enqueue_prompt("r4", bytes([0xFF, 0x00]))
        resp_data = node.get_response("r4")
        assert resp_data is not None
        chunk = WireChunk.from_cbor(resp_data[1:])
        assert chunk.chunk_type == "error"
        assert "unknown_method" in chunk.error

    def test_cancel(self, node):
        stop = threading.Event()
        adapter = StubAdapter(tokens=["t"] * 100, delay=0.1)
        handler = PromptHandlerLoop(p2p_node=node, adapter=adapter, stop_event=stop)
        handler.start()
        try:
            wire = WirePromptRequest(request_id="cancel-1", model_id="test:1b", prompt="long")
            payload = bytes([METHOD_PROMPT_REQUEST]) + wire.to_cbor()
            node.enqueue_prompt("r5", payload)
            time.sleep(0.2)
            cancel_payload = bytes([METHOD_PROMPT_CANCEL]) + wire.to_cbor()
            node.enqueue_prompt("r6", cancel_payload)
            cancel_resp = node.get_response("r6")
            assert cancel_resp is not None
            assert cancel_resp[0] == METHOD_PROMPT_CANCEL
            resp_data = node.get_response("r5", timeout=3.0)
            assert resp_data is not None
            chunk = WireChunk.from_cbor(resp_data[1:])
            assert chunk.finish_reason == "cancelled"
        finally:
            stop.set()
            handler.stop()

    def test_multiple_concurrent(self, node, handler):
        self._send_prompt(node, req_id="c1")
        self._send_prompt(node, req_id="c2")
        r1 = node.get_response("c1")
        r2 = node.get_response("c2")
        assert r1 is not None
        assert r2 is not None
        c1 = WireChunk.from_cbor(r1[1:])
        c2 = WireChunk.from_cbor(r2[1:])
        assert c1.chunk_type == "done"
        assert c2.chunk_type == "done"

    def test_request_response_includes_text(self, node, handler):
        """Request-response done chunk includes accumulated text."""
        self._send_prompt(node)
        resp_data = node.get_response("r1")
        assert resp_data is not None
        chunk = WireChunk.from_cbor(resp_data[1:])
        assert chunk.chunk_type == "done"
        assert chunk.token == "Hello world"


class TestPromptHandlerStreamPath:
    """Tests for the pstream- prefix stream path."""

    @pytest.fixture
    def node(self):
        return MockP2PNode()

    @pytest.fixture
    def adapter(self):
        return StubAdapter()

    @pytest.fixture
    def handler(self, node, adapter):
        stop = threading.Event()
        h = PromptHandlerLoop(p2p_node=node, adapter=adapter, stop_event=stop)
        h.start()
        yield h
        stop.set()
        h.stop()

    def _send_stream_prompt(self, node, req_id="pstream-1", model_id="test:1b"):
        wire = WirePromptRequest(request_id=req_id, model_id=model_id, prompt="hello")
        payload = bytes([METHOD_PROMPT_REQUEST]) + wire.to_cbor()
        node.enqueue_prompt(req_id, payload)

    def test_stream_sends_individual_tokens(self, node, handler):
        self._send_stream_prompt(node)
        chunks = node.get_stream_chunks("pstream-1")
        assert len(chunks) >= 3  # 2 tokens + 1 done
        token_chunks = []
        done_chunk = None
        for raw in chunks:
            c = WireChunk.from_cbor(raw[1:])
            if c.chunk_type == "token":
                token_chunks.append(c)
            elif c.chunk_type == "done":
                done_chunk = c
        assert len(token_chunks) == 2
        assert token_chunks[0].token == "Hello"
        assert token_chunks[1].token == " world"
        assert done_chunk is not None
        assert done_chunk.finish_reason == "stop"
        assert done_chunk.usage.completion_tokens == 2

    def test_stream_closes(self, node, handler):
        self._send_stream_prompt(node)
        node.get_stream_chunks("pstream-1")
        with node._lock:
            assert "pstream-1" in node._stream_closed

    def test_stream_error(self, node):
        stop = threading.Event()
        adapter = StubAdapter(fail_before_token=True)
        handler = PromptHandlerLoop(p2p_node=node, adapter=adapter, stop_event=stop)
        handler.start()
        try:
            self._send_stream_prompt(node, req_id="pstream-err")
            chunks = node.get_stream_chunks("pstream-err")
            assert len(chunks) >= 1
            last = WireChunk.from_cbor(chunks[-1][1:])
            assert last.chunk_type == "error"
            assert "backend down" in last.error
        finally:
            stop.set()
            handler.stop()
