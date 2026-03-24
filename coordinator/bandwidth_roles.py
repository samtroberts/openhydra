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

from dataclasses import dataclass


PREFILL_MIN_BANDWIDTH_MBPS = 500.0
DECODE_MAX_BANDWIDTH_MBPS = 50.0


@dataclass(frozen=True)
class RoleThresholds:
    prefill_min_mbps: float = PREFILL_MIN_BANDWIDTH_MBPS
    decode_max_mbps: float = DECODE_MAX_BANDWIDTH_MBPS


def classify_role(bandwidth_mbps: float, thresholds: RoleThresholds | None = None) -> str:
    t = thresholds or RoleThresholds()
    bw = max(0.0, float(bandwidth_mbps))

    if bw >= t.prefill_min_mbps:
        return "prefill_capable"
    if bw <= t.decode_max_mbps:
        return "decode_only"
    return "balanced"


def estimate_prompt_tokens(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 1
    return max(1, len(stripped.split()))


def role_counts_from_bandwidth(bandwidths: list[float], thresholds: RoleThresholds | None = None) -> dict[str, int]:
    counts = {"prefill_capable": 0, "balanced": 0, "decode_only": 0}
    for bw in bandwidths:
        role = classify_role(bw, thresholds=thresholds)
        counts[role] = counts.get(role, 0) + 1
    return counts
