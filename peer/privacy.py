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

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NoiseStats:
    applied_payloads: int
    configured_variance: float
    observed_variance_ema: float
    last_observed_variance: float
    last_observed_std: float
    last_audit_tag: str


class PyTorchDifferentialPrivacyNoise:
    """Gaussian activation obfuscation for honest-but-curious peers."""

    def __init__(
        self,
        *,
        torch_module: Any,
        variance: float,
        seed: int = 0,
    ):
        self._torch = torch_module
        self.variance = max(0.0, float(variance))
        self._std = math.sqrt(self.variance)
        self._generator = self._torch.Generator(device="cpu")
        self._generator.manual_seed(int(seed))
        self._applied_payloads = 0
        self._observed_variance_ema = 0.0
        self._last_observed_variance = 0.0
        self._last_observed_std = 0.0
        self._last_audit_tag = ""

    @property
    def enabled(self) -> bool:
        return self.variance > 0.0

    def stats(self) -> NoiseStats:
        return NoiseStats(
            applied_payloads=int(self._applied_payloads),
            configured_variance=float(self.variance),
            observed_variance_ema=float(self._observed_variance_ema),
            last_observed_variance=float(self._last_observed_variance),
            last_observed_std=float(self._last_observed_std),
            last_audit_tag=str(self._last_audit_tag),
        )

    def apply(
        self,
        tensor,
        *,
        peer_id: str | None = None,
        request_id: str | None = None,
        stage_index: int | None = None,
        shared_secret_seed: str | None = None,
    ):
        if not self.enabled:
            return tensor
        randn_kwargs = {"device": tensor.device, "dtype": tensor.dtype}
        if str(getattr(tensor.device, "type", "cpu")) == "cpu":
            randn_kwargs["generator"] = self._generator
        noise = self._torch.randn(tensor.shape, **randn_kwargs) * self._std
        self._applied_payloads += 1
        noise_f32 = noise.detach().to(dtype=self._torch.float32)
        observed_variance = float(noise_f32.pow(2).mean().item())
        observed_std = math.sqrt(max(0.0, observed_variance))
        alpha = 0.20
        if self._applied_payloads == 1:
            self._observed_variance_ema = observed_variance
        else:
            self._observed_variance_ema = (
                (1.0 - alpha) * float(self._observed_variance_ema)
                + alpha * observed_variance
            )
        self._last_observed_variance = observed_variance
        self._last_observed_std = observed_std
        self._last_audit_tag = ""
        if (
            peer_id
            and request_id
            and stage_index is not None
            and shared_secret_seed
        ):
            try:
                from peer.crypto import build_privacy_audit_tag

                self._last_audit_tag = build_privacy_audit_tag(
                    peer_id=str(peer_id),
                    request_id=str(request_id),
                    stage_index=int(stage_index),
                    payload_index=int(self._applied_payloads),
                    configured_variance=float(self.variance),
                    observed_variance=float(observed_variance),
                    observed_std=float(observed_std),
                    shared_secret_seed=str(shared_secret_seed),
                )
            except Exception:
                self._last_audit_tag = ""
        return tensor + noise
