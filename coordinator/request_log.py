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

"""Sliding-window demand tracker for the auto-scaler.

Records inference requests by model quality tier (basic / standard / advanced /
frontier) and computes demand weights for use in promotion decisions.

Thread-safe via an internal RLock.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

# ── Quality tier → model-size markers ─────────────────────────────────────────
# These substrings are matched against model_id.lower() to infer tier.
_TIER_MARKERS: dict[str, tuple[str, ...]] = {
    "basic":    ("0.6b", "0.8b", "1b", "1.7b", "2b", "smol", "tiny"),
    "standard": ("3b", "4b", "7b", "8b", "lite"),
    "advanced": ("9b", "12b", "13b", "14b", "32b"),
    "frontier": ("27b", "30b", "34b", "70b", "72b", "405b"),
}

# Pre-sorted longest-first to avoid substring false-positives:
# e.g. "0.8b" (basic) must be checked before "8b" (standard),
# and "27b" (frontier) before "2b" (basic).
_TIER_MARKERS_FLAT: list[tuple[str, str]] = sorted(
    [(marker, tier) for tier, markers in _TIER_MARKERS.items() for marker in markers],
    key=lambda x: len(x[0]),
    reverse=True,
)

_ALL_TIERS = ["basic", "standard", "advanced", "frontier"]


def quality_tier_for_model_id(model_id: str) -> str:
    """Infer quality tier from *model_id* string.  Returns 'standard' on no match."""
    m = model_id.lower()
    for marker, tier in _TIER_MARKERS_FLAT:
        if marker in m:
            return tier
    return "standard"


class RequestLog:
    """Sliding-window request demand tracker.

    Records inference requests by quality tier and computes demand weights
    for auto-scaling promotion decisions.

    Usage::

        log = RequestLog(window_seconds=3600)
        log.record("openhydra-qwen3.5-0.8b")
        weight = log.demand_weight("basic")  # fraction in [0, 1]

    When the window is empty ``demand_weight`` returns ``0.5`` (neutral default)
    so the auto-scaler does not block promotions purely due to lack of data.
    """

    def __init__(self, window_seconds: float = 3600.0):
        self._window = max(0.0, float(window_seconds))
        self._events: deque[tuple[float, str]] = deque()  # (monotonic_ts, tier)
        self._lock = threading.RLock()

    # ── Recording ─────────────────────────────────────────────────────────────

    def record(self, model_id: str) -> None:
        """Record an inference request for *model_id*."""
        tier = quality_tier_for_model_id(model_id)
        now = time.monotonic()
        with self._lock:
            self._events.append((now, tier))
            self._prune(now)

    def record_tier(self, tier: str) -> None:
        """Record a request directly by quality tier (for testing)."""
        now = time.monotonic()
        with self._lock:
            self._events.append((now, tier))
            self._prune(now)

    # ── Querying ──────────────────────────────────────────────────────────────

    def demand_weight(self, tier: str) -> float:
        """Fraction of recent requests for *tier*.

        Returns ``0.5`` (neutral) if the window is empty.
        """
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            total = len(self._events)
            if total == 0:
                return 0.5
            matching = sum(1 for _, t in self._events if t == tier)
            return matching / total

    def snapshot(self) -> dict[str, float]:
        """Return a ``{tier: demand_weight}`` snapshot for all tiers."""
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            total = len(self._events)
            if total == 0:
                return {t: 0.5 for t in _ALL_TIERS}
            counts: dict[str, int] = {t: 0 for t in _ALL_TIERS}
            for _, tier in self._events:
                if tier in counts:
                    counts[tier] += 1
            return {t: counts[t] / total for t in _ALL_TIERS}

    def __len__(self) -> int:
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            return len(self._events)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _prune(self, now: float) -> None:
        """Drop events older than the window.  Must hold ``_lock``."""
        cutoff = now - self._window
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()
