from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RedundantCheck:
    compared: bool
    match: bool


def compare_outputs(primary: str, secondary: str, tolerance: float = 0.0) -> RedundantCheck:
    del tolerance
    return RedundantCheck(compared=True, match=primary.strip() == secondary.strip())
