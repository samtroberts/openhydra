# Copyright 2026 OpenHydra contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import Any


@dataclass
class DhtRecord:
    value: dict[str, Any]
    expires_at: float


@dataclass
class InMemoryDhtNode:
    """Tier 2 list-appending DHT semantics with per-key TTL pruning."""

    ttl_seconds: int = 300
    _store: dict[str, list[DhtRecord]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _pruner_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _pruner_stop_event: threading.Event | None = field(default=None, init=False, repr=False)

    def _prune(self) -> None:
        now = time.time()
        for key in list(self._store.keys()):
            records = [item for item in self._store[key] if item.expires_at >= now]
            if records:
                self._store[key] = records
            else:
                del self._store[key]

    def put(
        self,
        key: str,
        value: dict[str, Any],
        *,
        unique_field: str | None = "peer_id",
        ttl_seconds: int | None = None,
    ) -> None:
        now = time.time()
        expiry = now + float(ttl_seconds or self.ttl_seconds)
        record = DhtRecord(value=dict(value), expires_at=expiry)

        with self._lock:
            records = self._store.setdefault(key, [])
            records[:] = [item for item in records if item.expires_at >= now]
            if unique_field:
                incoming_id = value.get(unique_field)
                records[:] = [r for r in records if r.value.get(unique_field) != incoming_id]
            records.append(record)

    def get(self, key: str) -> list[dict[str, Any]]:
        now = time.time()
        with self._lock:
            records = [item for item in self._store.get(key, []) if item.expires_at >= now]
            if records:
                self._store[key] = records
            else:
                self._store.pop(key, None)
            return [dict(item.value) for item in records]

    def keys(self) -> list[str]:
        with self._lock:
            return sorted(self._store.keys())

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {key: len(records) for key, records in self._store.items()}

    def start_background_pruner(self, interval_s: float = 30.0) -> None:
        with self._lock:
            if self._pruner_thread is not None and self._pruner_thread.is_alive():
                return
            stop_event = threading.Event()
            interval = max(0.5, float(interval_s))

            def _loop() -> None:
                while not stop_event.wait(interval):
                    with self._lock:
                        self._prune()

            thread = threading.Thread(
                target=_loop,
                name="openhydra-dht-pruner",
                daemon=True,
            )
            self._pruner_stop_event = stop_event
            self._pruner_thread = thread
            thread.start()

    def stop_background_pruner(self) -> None:
        thread: threading.Thread | None
        stop_event: threading.Event | None
        with self._lock:
            thread = self._pruner_thread
            stop_event = self._pruner_stop_event
            self._pruner_thread = None
            self._pruner_stop_event = None

        if stop_event is not None:
            stop_event.set()
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
