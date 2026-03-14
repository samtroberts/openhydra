from coordinator.path_finder import PeerEndpoint, PeerHealth
from coordinator.peer_selector import compute_routing_score, rank_peers


def test_tier2_score_prefers_low_latency_and_low_load():
    score_fast = compute_routing_score(latency_ms=20, load_pct=15, reputation=50, bandwidth_mbps=0, tier=2)
    score_slow = compute_routing_score(latency_ms=80, load_pct=15, reputation=50, bandwidth_mbps=0, tier=2)
    assert score_fast > score_slow


def test_rank_peers_tier1_latency_ordering():
    p1 = PeerEndpoint(peer_id="a", host="127.0.0.1", port=1)
    p2 = PeerEndpoint(peer_id="b", host="127.0.0.1", port=2)
    health = [
        PeerHealth(peer=p1, healthy=True, latency_ms=30.0, load_pct=20.0, daemon_mode="polite"),
        PeerHealth(peer=p2, healthy=True, latency_ms=10.0, load_pct=80.0, daemon_mode="polite"),
    ]
    ranked = rank_peers(health, tier=1)
    assert ranked[0].peer.peer_id == "b"


def test_rank_peers_tier2_uses_reputation_signal():
    p1 = PeerEndpoint(peer_id="a", host="127.0.0.1", port=1, bandwidth_mbps=50)
    p2 = PeerEndpoint(peer_id="b", host="127.0.0.1", port=2, bandwidth_mbps=50)
    health = [
        PeerHealth(peer=p1, healthy=True, latency_ms=20.0, load_pct=20.0, daemon_mode="polite"),
        PeerHealth(peer=p2, healthy=True, latency_ms=20.0, load_pct=20.0, daemon_mode="polite"),
    ]

    ranked = rank_peers(
        health,
        tier=2,
        reputation_by_peer={"a": 20.0, "b": 90.0},
    )
    assert ranked[0].peer.peer_id == "b"
