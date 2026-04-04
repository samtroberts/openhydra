# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Decentralized Speculative Decoding (DSD) for Swarm Mode.

Generates K draft tokens locally during network stalls, then verifies
all K in a single pipeline pass.  Converts idle network wait time into
useful GPU computation.

Reference: arxiv.org/abs/2511.11733
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from coordinator.speculative import select_verified_token_ids


@dataclass(frozen=True)
class SwarmSpecConfig:
    """Configuration for the swarm speculative decoder."""
    draft_tokens: int = 4
    adaptive_enabled: bool = True
    min_draft_tokens: int = 2
    max_draft_tokens: int = 8
    acceptance_low_watermark: float = 0.55
    acceptance_high_watermark: float = 0.80


@dataclass
class SwarmSpecStats:
    """Running statistics for the speculative decoder."""
    rounds: int = 0
    draft_tokens: int = 0
    accepted_tokens: int = 0

    @property
    def acceptance_rate(self) -> float | None:
        if self.draft_tokens == 0:
            return None
        return self.accepted_tokens / self.draft_tokens


class SwarmSpeculativeDecoder:
    """DSD controller: propose draft tokens, verify, accept/reject.

    Args:
        config: Speculative decoding configuration.
        draft_fn: Callable that takes (context_token_ids, k) and returns
            k draft token IDs.  This is the local "draft model" — can be
            a small MLX model, PyTorchDraftModel, or a ToyRuntime stub.
    """

    def __init__(
        self,
        config: SwarmSpecConfig,
        draft_fn: Callable[[list[int], int], list[int]],
    ) -> None:
        self._config = config
        self._draft_fn = draft_fn
        self._current_k = config.draft_tokens
        self._stats = SwarmSpecStats()

    @property
    def current_k(self) -> int:
        return self._current_k

    @property
    def stats(self) -> SwarmSpecStats:
        return self._stats

    def propose(self, context: list[int], k: int) -> list[int]:
        """Generate K draft tokens from the given context."""
        return list(self._draft_fn(context, k))

    def accept_reject(
        self,
        draft_ids: list[int],
        verified_ids: list[int],
    ) -> tuple[list[int], bool]:
        """Compare draft tokens against verified tokens.

        Uses the existing ``select_verified_token_ids`` from
        ``coordinator/speculative.py`` for the prefix-match logic.

        Returns:
            (accepted_token_ids, all_matched)
        """
        result = select_verified_token_ids(verified_ids, draft_ids)

        # Update stats
        self._stats.rounds += 1
        self._stats.draft_tokens += len(draft_ids)
        self._stats.accepted_tokens += len(result.accepted_token_ids)

        all_matched = not result.mismatch and len(result.accepted_token_ids) == len(draft_ids)

        # Adaptive K
        if self._config.adaptive_enabled:
            self._adapt_k(all_matched, len(result.accepted_token_ids), len(draft_ids))

        return list(result.accepted_token_ids), all_matched

    def _adapt_k(self, all_matched: bool, accepted: int, total: int) -> None:
        """Adjust K based on acceptance rate."""
        if total == 0:
            return
        rate = accepted / total
        if rate >= self._config.acceptance_high_watermark:
            self._current_k = min(self._current_k + 1, self._config.max_draft_tokens)
        elif rate < self._config.acceptance_low_watermark:
            self._current_k = max(self._current_k - 1, self._config.min_draft_tokens)
