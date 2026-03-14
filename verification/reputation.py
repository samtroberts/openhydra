from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Reputation:
    verification_success: float = 1.0
    uptime: float = 1.0
    latency_consistency: float = 1.0
    stake_factor: float = 0.0

    def score(self) -> float:
        value = (
            40.0 * self.verification_success
            + 25.0 * self.uptime
            + 20.0 * self.latency_consistency
            + 15.0 * self.stake_factor
        )
        return max(0.0, min(100.0, value))
