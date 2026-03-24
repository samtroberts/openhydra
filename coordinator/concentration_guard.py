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

from collections import defaultdict
from dataclasses import dataclass
import logging
import math

from coordinator.path_finder import PeerEndpoint

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConcentrationMetrics:
    total_peers: int
    operator_counts: dict[str, int]
    operator_shares: dict[str, float]
    max_operator: str | None
    max_share: float
    over_cap_operators: list[str]


def _operator_id(peer: PeerEndpoint) -> str:
    return peer.operator_id or peer.peer_id


def operator_counts(peers: list[PeerEndpoint]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for peer in peers:
        counts[_operator_id(peer)] += 1
    return dict(counts)


def concentration_metrics(peers: list[PeerEndpoint], cap_fraction: float = 0.33) -> ConcentrationMetrics:
    counts = operator_counts(peers)
    total = len(peers)
    if total <= 0:
        return ConcentrationMetrics(
            total_peers=0,
            operator_counts={},
            operator_shares={},
            max_operator=None,
            max_share=0.0,
            over_cap_operators=[],
        )

    shares = {op: count / total for op, count in counts.items()}
    max_operator = max(shares, key=shares.get)
    max_share = shares[max_operator]
    over_cap = sorted([op for op, share in shares.items() if share > (cap_fraction + 1e-9)])

    return ConcentrationMetrics(
        total_peers=total,
        operator_counts=counts,
        operator_shares=shares,
        max_operator=max_operator,
        max_share=max_share,
        over_cap_operators=over_cap,
    )


def _can_place(
    existing: list[PeerEndpoint],
    candidate: PeerEndpoint,
    diversity_window: int,
    max_per_window: int,
) -> bool:
    if diversity_window <= 1:
        return True

    recent = existing[-(diversity_window - 1) :]
    op = _operator_id(candidate)
    recent_count = sum(1 for peer in recent if _operator_id(peer) == op)
    return (recent_count + 1) <= max_per_window


def enforce_operator_caps(
    peers: list[PeerEndpoint],
    pipeline_width: int,
    max_fraction: float = (1.0 / 3.0),
) -> list[PeerEndpoint]:
    """Tier 2 guardrail: cap operator concentration.

    The returned list may be shorter than ``pipeline_width`` when diversity
    constraints cannot be satisfied.
    """
    if pipeline_width <= 0:
        return []

    max_per_operator = max(1, math.floor(pipeline_width * max_fraction))
    selected: list[PeerEndpoint] = []
    counts: dict[str, int] = defaultdict(int)

    for peer in peers:
        operator_id = _operator_id(peer)
        if counts[operator_id] >= max_per_operator:
            continue
        selected.append(peer)
        counts[operator_id] += 1
        if len(selected) >= pipeline_width:
            break

    return selected


def enforce_pipeline_diversity(
    peers: list[PeerEndpoint],
    diversity_window: int = 3,
    max_per_window: int = 1,
) -> list[PeerEndpoint]:
    """Attempt to reorder pipeline to minimize same-operator clustering.

    If strict diversity cannot be satisfied, falls back to preserving remaining order.
    """
    if len(peers) <= 1:
        return peers

    pool = list(peers)
    arranged: list[PeerEndpoint] = []

    while pool:
        placed = False
        for idx, peer in enumerate(pool):
            if _can_place(arranged, peer, diversity_window=diversity_window, max_per_window=max_per_window):
                arranged.append(peer)
                pool.pop(idx)
                placed = True
                break

        if not placed:
            arranged.extend(pool)
            break

    return arranged


def assemble_pipeline(
    peers: list[PeerEndpoint],
    pipeline_width: int,
    max_fraction: float = (1.0 / 3.0),
    enforce_diversity: bool = True,
    diversity_window: int = 3,
    max_per_window: int = 1,
) -> list[PeerEndpoint]:
    capped = enforce_operator_caps(peers, pipeline_width=pipeline_width, max_fraction=max_fraction)
    if len(capped) < max(1, int(pipeline_width)):
        logger.warning(
            "operator_cap_enforced: pipeline assembled with %s/%s peers",
            len(capped),
            max(1, int(pipeline_width)),
        )
    if enforce_diversity:
        capped = enforce_pipeline_diversity(
            capped,
            diversity_window=diversity_window,
            max_per_window=max_per_window,
        )
    return capped[:pipeline_width]
