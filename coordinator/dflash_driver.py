# Copyright 2026 OpenHydra contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Phase 2b — DFlash Topology A driver.

Coordinator-side driver loop for ``--draft-location local``:

    while not done:
        draft_ids       = drafter.draft(prefix)            # block_size tokens
        hidden_states   = ring.verify(prefix, draft_ids)   # block_size+1 positions
        accepted, bonus = sampler.verify_block(hidden_states, draft_ids)
        emit prefix-extension(draft_ids[:accepted] + [bonus])
        prefix         += draft_ids[:accepted] + [bonus]
        next iteration carries kv_rollback_to = len(prefix)

Pure orchestration. Receives Drafter, BlockVerifier, RingTransport,
emit callback. Does not touch model weights, gRPC stubs, KV state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time
from typing import Any, Callable, Iterable, Protocol

logger = logging.getLogger(__name__)

__all__ = [
    "BlockVerifier",
    "BlockVerifyResult",
    "DFlashTopologyADriver",
    "Drafter",
    "DriverStats",
    "RingVerifyTransport",
]


class Drafter(Protocol):
    def draft(self, prefix_token_ids: list[int]) -> list[int]: ...


class RingVerifyTransport(Protocol):
    def verify(
        self,
        *,
        prefix_token_ids: list[int],
        draft_token_ids: list[int],
        kv_rollback_to: int,
        request_id: str,
        kv_session_id: str,
    ) -> Any: ...


class BlockVerifier(Protocol):
    def __call__(
        self,
        hidden_states_block: Any,
        draft_token_ids: list[int],
    ) -> tuple[int, int]: ...


@dataclass
class BlockVerifyResult:
    block_index: int
    draft_token_ids: list[int]
    accepted_len: int
    bonus_token: int
    new_prefix_len: int

    @property
    def emitted(self) -> list[int]:
        return list(self.draft_token_ids[: self.accepted_len]) + [self.bonus_token]

    @property
    def acceptance_rate(self) -> float:
        n = len(self.draft_token_ids) or 1
        return self.accepted_len / n


@dataclass
class DriverStats:
    blocks: int = 0
    tokens_emitted: int = 0
    drafts_total: int = 0
    drafts_accepted: int = 0
    draft_ms_total: float = 0.0
    verify_ms_total: float = 0.0
    sampler_ms_total: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        n = self.drafts_total or 1
        return self.drafts_accepted / n

    @property
    def avg_block_size_emitted(self) -> float:
        return (self.tokens_emitted / self.blocks) if self.blocks else 0.0


@dataclass
class DFlashTopologyADriver:
    """Coordinator-side DFlash driver for Topology A."""

    drafter: Drafter
    verifier: BlockVerifier
    transport: RingVerifyTransport
    emit: Callable[[int], None]
    block_size: int = 16
    max_tokens: int = 256
    stop_token_ids: frozenset = field(default_factory=frozenset)
    request_id: str = ""
    kv_session_id: str = ""

    def run(self, prompt_token_ids: Iterable[int]) -> DriverStats:
        stats = DriverStats()
        prefix: list[int] = list(prompt_token_ids)
        prefix_len_at_block_start = len(prefix)
        block_index = 0

        while stats.tokens_emitted < self.max_tokens:
            # Draft.
            t0 = time.monotonic()
            draft_ids = list(self.drafter.draft(prefix))
            stats.draft_ms_total += (time.monotonic() - t0) * 1000.0
            if len(draft_ids) != self.block_size:
                raise ValueError(
                    f"drafter returned {len(draft_ids)} tokens; "
                    f"expected block_size={self.block_size}"
                )

            # Verify (ring round trip).
            t0 = time.monotonic()
            hidden_states_block = self.transport.verify(
                prefix_token_ids=prefix,
                draft_token_ids=draft_ids,
                # Peers roll their KV cache back to the position we
                # ENDED with last block (prefix length before this
                # block's drafts). On the first iteration that's the
                # prompt length; subsequent iterations advance by the
                # accepted+bonus tokens of the previous block.
                kv_rollback_to=prefix_len_at_block_start,
                request_id=self.request_id,
                kv_session_id=self.kv_session_id,
            )
            stats.verify_ms_total += (time.monotonic() - t0) * 1000.0

            # Sampler — accept-prefix walk.
            t0 = time.monotonic()
            accepted, bonus = self.verifier(hidden_states_block, draft_ids)
            stats.sampler_ms_total += (time.monotonic() - t0) * 1000.0

            result = BlockVerifyResult(
                block_index=block_index,
                draft_token_ids=draft_ids,
                accepted_len=int(accepted),
                bonus_token=int(bonus),
                new_prefix_len=len(prefix) + int(accepted) + 1,
            )
            block_index += 1
            stats.blocks += 1
            stats.drafts_total += len(draft_ids)
            stats.drafts_accepted += int(accepted)

            for tok in result.emitted:
                if stats.tokens_emitted >= self.max_tokens:
                    break
                self.emit(int(tok))
                prefix.append(int(tok))
                stats.tokens_emitted += 1
                if int(tok) in self.stop_token_ids:
                    logger.info(
                        "dflash_driver_stop_token: req=%s session=%s "
                        "block=%d tok=%d emitted=%d",
                        self.request_id, self.kv_session_id,
                        result.block_index, int(tok),
                        stats.tokens_emitted,
                    )
                    return stats

            prefix_len_at_block_start = len(prefix)

            if result.accepted_len == 0 and stats.blocks > 1:
                logger.debug(
                    "dflash_driver_zero_accept: req=%s block=%d",
                    self.request_id, result.block_index,
                )

        logger.info(
            "dflash_driver_run_complete: req=%s session=%s blocks=%d "
            "tokens_emitted=%d acceptance_rate=%.3f avg_block_emitted=%.2f",
            self.request_id, self.kv_session_id, stats.blocks,
            stats.tokens_emitted, stats.acceptance_rate,
            stats.avg_block_size_emitted,
        )
        return stats
