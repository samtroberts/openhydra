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
) -> float:
    latency_ms = max(latency_ms, 1.0)
    headroom = max(1.0, 100.0 - load_pct)
    rep_norm = max(0.0, min(100.0, reputation)) / 100.0
    bw_norm = max(0.0, bandwidth_mbps) / 1000.0

    if tier <= 2:
        w1, w2, w3, w4 = 0.45, 0.30, 0.25, 0.00
    else:
        w1, w2, w3, w4 = 0.30, 0.25, 0.30, 0.15

    return (w1 * (1.0 / latency_ms)) + (w2 * (headroom / 100.0)) + (w3 * rep_norm) + (w4 * bw_norm)


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

    scored: list[ScoredPeer] = []
    for item in health:
        reputation = float(reputation_by_peer.get(item.peer.peer_id, default_reputation))
        bandwidth = float(
            bandwidth_by_peer.get(
                item.peer.peer_id,
                item.peer.bandwidth_mbps if item.peer.bandwidth_mbps > 0 else default_bandwidth_mbps,
            )
        )

        if tier == 1:
            score = 1.0 / max(1.0, item.latency_ms)
        else:
            score = compute_routing_score(
                latency_ms=item.latency_ms,
                load_pct=item.load_pct,
                reputation=reputation,
                bandwidth_mbps=bandwidth,
                tier=tier,
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
