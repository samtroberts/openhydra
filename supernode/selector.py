from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Any

from .manifest import SupernodeManifest

logger = logging.getLogger(__name__)

NEAR_EQUAL_THRESHOLD = 0.9
MAX_FAILOVER_ATTEMPTS = 3


@dataclass
class ScoredCandidate:
    manifest: SupernodeManifest
    score: float
    latency_ms: float = 0.0


def score_candidate(manifest: SupernodeManifest) -> float:
    """Basic scoring for MVP — trust tier + model warm status + concurrent capacity.

    Full scoring (measured TPS, load, queue, reputation) is Phase 3b.
    """
    trust_weights = {"attested": 1.0, "supervised": 0.7, "unverified": 0.3}
    score = trust_weights.get(manifest.trust_tier, 0.3)

    warm_count = sum(1 for m in manifest.models if m.warm)
    if warm_count > 0:
        score += 0.2

    if manifest.max_concurrent > 0:
        score += 0.1

    if manifest.nat_status == "public":
        score += 0.05

    return score


def select_supernode(
    candidates: list[ScoredCandidate],
) -> ScoredCandidate | None:
    """Randomized near-equal selection (§4.3.1).

    Candidates within 10% of the best score form the eligible set.
    Weighted-random by score spreads load while preferring better nodes.
    """
    if not candidates:
        return None

    candidates.sort(key=lambda c: c.score, reverse=True)
    best = candidates[0].score

    eligible = [c for c in candidates if c.score >= best * NEAR_EQUAL_THRESHOLD and c.score > 0]
    if not eligible:
        return candidates[0]

    total = sum(c.score for c in eligible)
    if total <= 0:
        return eligible[0]

    r = random.random() * total
    acc = 0.0
    for c in eligible:
        acc += c.score
        if r <= acc:
            return c
    return eligible[-1]


class PromptRouter:
    """Routes a prompt request through discovery → scoring → selection → adapter.

    Handles fail-fast failover: pre-first-token failures retry the next
    candidate; post-first-token failures terminate with finish_reason="error".
    """

    def __init__(self, discovery, adapter_registry: dict[str, Any] | None = None):
        self._discovery = discovery
        self._adapters = adapter_registry or {}
        self._failed_peers: dict[str, float] = {}

    def register_adapter(self, libp2p_peer_id: str, adapter) -> None:
        self._adapters[libp2p_peer_id] = adapter

    def route(self, model_id: str) -> list[ScoredCandidate]:
        """Discover and score candidates for a model. Returns sorted list."""
        manifests = self._discovery.discover_supernodes(model_id)
        if not manifests:
            return []

        scored = []
        for m in manifests:
            penalty = 0.0
            if m.libp2p_peer_id in self._failed_peers:
                last_fail = self._failed_peers[m.libp2p_peer_id]
                if time.monotonic() - last_fail < 60:
                    penalty = 0.1
                else:
                    del self._failed_peers[m.libp2p_peer_id]

            s = score_candidate(m) - penalty
            scored.append(ScoredCandidate(manifest=m, score=max(s, 0.01)))

        scored.sort(key=lambda c: c.score, reverse=True)
        return scored

    def select(self, model_id: str) -> ScoredCandidate | None:
        """Discover, score, and select a single candidate."""
        candidates = self.route(model_id)
        return select_supernode(candidates)

    def record_failure(self, libp2p_peer_id: str) -> None:
        self._failed_peers[libp2p_peer_id] = time.monotonic()
        logger.info("supernode_failure_recorded peer=%s", libp2p_peer_id)

    def select_with_failover(
        self, model_id: str, max_attempts: int = MAX_FAILOVER_ATTEMPTS,
    ) -> list[ScoredCandidate]:
        """Return an ordered failover list (primary + backups).

        The caller tries each in order; on pre-first-token failure,
        moves to the next. No new DHT query needed.
        """
        candidates = self.route(model_id)
        if not candidates:
            return []

        result: list[ScoredCandidate] = []
        used: set[str] = set()

        for _ in range(min(max_attempts, len(candidates))):
            remaining = [c for c in candidates if c.manifest.libp2p_peer_id not in used]
            selected = select_supernode(remaining)
            if selected is None:
                break
            result.append(selected)
            used.add(selected.manifest.libp2p_peer_id)

        return result
