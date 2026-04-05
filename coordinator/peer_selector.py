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

from coordinator.path_finder import PeerEndpoint, PeerHealth


@dataclass(frozen=True)
class ScoredPeer:
    peer: PeerEndpoint
    score: float
    latency_ms: float
    load_pct: float
    reputation: float
    bandwidth_mbps: float


def compute_routing_score(
    latency_ms: float,
    load_pct: float,
    reputation: float,
    bandwidth_mbps: float,
    tier: int,
    s2s_rtt_ms: float = 0.0,
) -> float:
    """Compute a routing score for peer pipeline ranking.

    Higher scores are better.  Incorporates 5 factors:
    1. Ping latency (inverse — lower is better)
    2. Load headroom (higher available capacity is better)
    3. Reputation (verification history)
    4. Bandwidth (higher is better)
    5. S2S RTT (server-to-server measured latency to downstream peers)

    S2S RTT penalizes peers that have high measured latency to their
    downstream neighbors in the pipeline.  This prevents the coordinator
    from building pipelines through peers with poor inter-node connectivity.
    """
    latency_ms = max(latency_ms, 1.0)
    headroom = max(1.0, 100.0 - load_pct)
    rep_norm = max(0.0, min(100.0, reputation)) / 100.0
    bw_norm = max(0.0, bandwidth_mbps) / 1000.0
    # S2S RTT: 0 means no measurement available (neutral). Higher = worse.
    s2s_penalty = 1.0 / max(1.0, s2s_rtt_ms) if s2s_rtt_ms > 0 else 0.5

    if tier <= 2:
        # Latency-focused: 40% ping, 25% load, 20% reputation, 5% bandwidth, 10% S2S
        w1, w2, w3, w4, w5 = 0.40, 0.25, 0.20, 0.05, 0.10
    else:
        # Balanced: 25% ping, 20% load, 25% reputation, 15% bandwidth, 15% S2S
        w1, w2, w3, w4, w5 = 0.25, 0.20, 0.25, 0.15, 0.15

    return (
        (w1 * (1.0 / latency_ms))
        + (w2 * (headroom / 100.0))
        + (w3 * rep_norm)
        + (w4 * bw_norm)
        + (w5 * s2s_penalty)
    )


def rank_peers(
    health: list[PeerHealth],
    tier: int = 1,
    reputation_by_peer: dict[str, float] | None = None,
    bandwidth_by_peer: dict[str, float] | None = None,
    default_reputation: float = 50.0,
    default_bandwidth_mbps: float = 0.0,
) -> list[ScoredPeer]:
    reputation_by_peer = reputation_by_peer or {}
    bandwidth_by_peer = bandwidth_by_peer or {}

    # Pre-compute average S2S RTT per peer (mean of all downstream RTTs)
    s2s_rtt_by_peer: dict[str, float] = {}
    for item in health:
        rtts = getattr(item.peer, "next_hop_rtts", None) or {}
        if rtts:
            s2s_rtt_by_peer[item.peer.peer_id] = sum(rtts.values()) / len(rtts)

    scored: list[ScoredPeer] = []
    for item in health:
        reputation = float(reputation_by_peer.get(item.peer.peer_id, default_reputation))
        bandwidth = float(
            bandwidth_by_peer.get(
                item.peer.peer_id,
                item.peer.bandwidth_mbps if item.peer.bandwidth_mbps > 0 else default_bandwidth_mbps,
            )
        )

        s2s_rtt = s2s_rtt_by_peer.get(item.peer.peer_id, 0.0)

        if tier == 1:
            # Tier 1: pure latency + S2S RTT penalty
            base = 1.0 / max(1.0, item.latency_ms)
            s2s_penalty = 1.0 / max(1.0, s2s_rtt) if s2s_rtt > 0 else 0.5
            score = 0.85 * base + 0.15 * s2s_penalty
        else:
            score = compute_routing_score(
                latency_ms=item.latency_ms,
                load_pct=item.load_pct,
                reputation=reputation,
                bandwidth_mbps=bandwidth,
                tier=tier,
                s2s_rtt_ms=s2s_rtt,
            )

        scored.append(
            ScoredPeer(
                peer=item.peer,
                score=score,
                latency_ms=item.latency_ms,
                load_pct=item.load_pct,
                reputation=reputation,
                bandwidth_mbps=bandwidth,
            )
        )

    return sorted(scored, key=lambda x: x.score, reverse=True)
