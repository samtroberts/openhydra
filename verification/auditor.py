from __future__ import annotations

import random
import threading


def _clamp_rate(sample_rate: float) -> float:
    return max(0.0, min(1.0, float(sample_rate)))


class AuditSampler:
    """Thread-safe Bernoulli sampler for verification/auditor decisions."""

    def __init__(self, sample_rate: float, seed: int | None = None):
        self.sample_rate = _clamp_rate(sample_rate)
        self._rng = random.Random(seed)
        self._lock = threading.Lock()

    def should_sample(self) -> bool:
        with self._lock:
            return self._rng.random() < self.sample_rate


def should_audit(sample_rate: float, seed: int | None = None) -> bool:
    return AuditSampler(sample_rate=sample_rate, seed=seed).should_sample()
