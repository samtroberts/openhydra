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
