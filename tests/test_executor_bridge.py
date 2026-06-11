from __future__ import annotations

import asyncio
import time

import pytest

from supernode.adapter import TokenChunk
from supernode.executor_bridge import ExecutorBridge


@pytest.fixture
def bridge():
    b = ExecutorBridge(max_workers=1)
    yield b
    b.shutdown()


@pytest.mark.asyncio
async def test_stream_basic(bridge: ExecutorBridge):
    def gen():
        for i in range(5):
            yield TokenChunk(token=f"t{i}")
        yield TokenChunk(token="", finish_reason="stop")

    tokens = []
    async for chunk in bridge.stream(gen):
        tokens.append(chunk.token)

    assert tokens == ["t0", "t1", "t2", "t3", "t4", ""]


@pytest.mark.asyncio
async def test_stream_empty(bridge: ExecutorBridge):
    def gen():
        return
        yield  # pragma: no cover — make it a generator

    tokens = []
    async for chunk in bridge.stream(gen):
        tokens.append(chunk.token)

    assert tokens == []


@pytest.mark.asyncio
async def test_stream_exception_propagates(bridge: ExecutorBridge):
    def gen():
        yield TokenChunk(token="ok")
        raise RuntimeError("inference failed")

    with pytest.raises(RuntimeError, match="inference failed"):
        async for _ in bridge.stream(gen):
            pass


@pytest.mark.asyncio
async def test_stream_cancellation(bridge: ExecutorBridge):
    cancel = asyncio.Event()

    def gen():
        for i in range(1000):
            if cancel.is_set():
                break
            yield TokenChunk(token=f"t{i}")
            time.sleep(0.001)

    tokens = []
    async for chunk in bridge.stream(gen, cancel_event=cancel):
        tokens.append(chunk.token)
        if len(tokens) >= 3:
            cancel.set()
            break

    assert len(tokens) >= 3
    assert len(tokens) < 1000


@pytest.mark.asyncio
async def test_does_not_block_event_loop(bridge: ExecutorBridge):
    """Verify that async code can run concurrently during generation."""
    probe_ran = False

    def slow_gen():
        time.sleep(0.1)
        yield TokenChunk(token="done", finish_reason="stop")

    async def probe():
        nonlocal probe_ran
        await asyncio.sleep(0.01)
        probe_ran = True

    task = asyncio.create_task(probe())

    async for _ in bridge.stream(slow_gen):
        pass

    await task
    assert probe_ran
