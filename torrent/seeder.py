from __future__ import annotations

from dataclasses import asdict, dataclass
import time


@dataclass(frozen=True)
class ArbitrationConfig:
    base_upload_mbps: float = 100.0
    inference_seed_fraction: float = 0.10
    min_seed_upload_mbps: float = 1.0
    smoothing_alpha: float = 0.35


@dataclass
class SeederState:
    config: ArbitrationConfig
    inference_active: bool = False
    inference_observed_mbps: float = 0.0
    seed_upload_limit_mbps: float = 0.0
    updated_unix_ms: int = 0

    def effective_upload_limit(self) -> float:
        return max(self.config.min_seed_upload_mbps, self.seed_upload_limit_mbps)


class BandwidthArbitrator:
    """Implements inference-first upload policy for genesis seeding.

    Policy:
    - Inference active: seeding throttled to ~10% of upload budget.
    - Inference idle: seeding restored to full upload budget.
    - Smoothed transitions avoid abrupt oscillations.
    """

    def __init__(self, config: ArbitrationConfig | None = None):
        self.config = config or ArbitrationConfig()
        self.state = SeederState(
            config=self.config,
            seed_upload_limit_mbps=self.config.base_upload_mbps,
            updated_unix_ms=int(time.time() * 1000),
        )

    @staticmethod
    def _ema(current: float, target: float, alpha: float) -> float:
        return (alpha * target) + ((1.0 - alpha) * current)

    def target_seed_limit(self, inference_active: bool) -> float:
        if inference_active:
            return max(
                self.config.min_seed_upload_mbps,
                self.config.base_upload_mbps * self.config.inference_seed_fraction,
            )
        return self.config.base_upload_mbps

    def update(self, *, inference_active: bool, inference_observed_mbps: float | None = None) -> SeederState:
        target = self.target_seed_limit(inference_active)
        next_limit = self._ema(self.state.seed_upload_limit_mbps, target, self.config.smoothing_alpha)

        self.state.inference_active = bool(inference_active)
        if inference_observed_mbps is not None:
            self.state.inference_observed_mbps = max(0.0, float(inference_observed_mbps))
        self.state.seed_upload_limit_mbps = max(self.config.min_seed_upload_mbps, next_limit)
        self.state.updated_unix_ms = int(time.time() * 1000)
        return self.state

    def snapshot(self) -> dict:
        payload = asdict(self.state)
        payload["effective_seed_upload_limit_mbps"] = round(self.state.effective_upload_limit(), 6)
        payload["target_seed_upload_limit_mbps"] = round(
            self.target_seed_limit(self.state.inference_active),
            6,
        )
        return payload
