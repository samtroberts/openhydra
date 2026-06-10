from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Callable, Iterator

from .adapter import TokenChunk

logger = logging.getLogger(__name__)

_SENTINEL = object()


class ExecutorBridge:
    """Bridges sync generation to async iteration without blocking the event loop.

    Worker thread runs sync_gen_factory() and pushes TokenChunks to an
    asyncio.Queue via loop.call_soon_threadsafe. Async side awaits + yields.
    """

    def __init__(self, max_workers: int = 1):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    async def stream(
        self,
        sync_gen_factory: Callable[[], Iterator[TokenChunk]],
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncIterator[TokenChunk]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=64)

        def _worker():
            try:
                gen = sync_gen_factory()
                for chunk in gen:
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
                    if cancel_event is not None and cancel_event.is_set():
                        break
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

        future = loop.run_in_executor(self._executor, _worker)

        try:
            while True:
                item = await queue.get()
                if item is _SENTINEL:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            if cancel_event is not None:
                cancel_event.set()
            await future

    def shutdown(self):
        self._executor.shutdown(wait=False)
