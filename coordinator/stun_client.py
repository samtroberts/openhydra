from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NatProfile:
    reachable: bool
    nat_type: str


def probe_nat() -> NatProfile:
    """Tier 3 placeholder for STUN/TURN integration."""
    return NatProfile(reachable=True, nat_type="unknown")
