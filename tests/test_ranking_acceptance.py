# Copyright 2026 OpenHydra contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Ranking acceptance test (protocol.md §4/§5, M1.2).

Proves the router orders a seeded multi-peer set by live **health (load)**, **RTT
(latency)**, **throughput**, and **queue depth**. Each dimension is isolated (all
other factors held equal) so the assertion pins the behaviour to that one signal,
plus a combined realistic seed.
"""

from __future__ import annotations

from coordinator.path_finder import PeerEndpoint, PeerHealth
from coordinator.peer_selector import rank_peers


def _peer(peer_id: str, **fields) -> PeerEndpoint:
    data = {"peer_id": peer_id, "host": "h", "port": 1, "model_id": "openhydra-qwen3.5-2b"}
    data.update(fields)
    return PeerEndpoint.from_dict(data)


def _health(
    peer_id: str,
    *,
    latency_ms: float,
    load_pct: float = 0.0,
    throughput: float = 0.0,
    queue: int = 0,
    bandwidth: float = 0.0,
) -> PeerHealth:
    ep = _peer(
        peer_id,
        throughput_tok_s=throughput,
        queue_depth=queue,
        bandwidth_mbps=bandwidth,
    )
    return PeerHealth(
        peer=ep, healthy=True, latency_ms=latency_ms, load_pct=load_pct, daemon_mode="polite"
    )


def _order(health, tier=2):
    return [s.peer.peer_id for s in rank_peers(health, tier=tier)]


# --- one dimension at a time (all else equal) ---


def test_ranks_by_rtt():
    health = [_health("far", latency_ms=80.0), _health("near", latency_ms=5.0)]
    assert _order(health)[0] == "near"


def test_ranks_by_health_load():
    health = [_health("loaded", latency_ms=10.0, load_pct=90.0), _health("free", latency_ms=10.0, load_pct=5.0)]
    assert _order(health)[0] == "free"


def test_ranks_by_throughput():
    health = [_health("slow", latency_ms=10.0, throughput=5.0), _health("fast", latency_ms=10.0, throughput=45.0)]
    assert _order(health) == ["fast", "slow"]


def test_ranks_by_queue_depth():
    health = [_health("busy", latency_ms=10.0, queue=8), _health("idle", latency_ms=10.0, queue=0)]
    assert _order(health) == ["idle", "busy"]


# --- combined realistic seed ---


def test_seeded_multi_peer_overall_order():
    health = [
        _health("mid", latency_ms=20.0, load_pct=40.0, throughput=20.0, queue=2),
        _health("best", latency_ms=5.0, load_pct=10.0, throughput=48.0, queue=0),
        _health("worst", latency_ms=90.0, load_pct=85.0, throughput=4.0, queue=9),
    ]
    order = _order(health)
    assert order[0] == "best"
    assert order[-1] == "worst"


def test_scored_peer_exposes_live_throughput_and_queue():
    (scored,) = rank_peers([_health("p", latency_ms=10.0, throughput=30.0, queue=3)], tier=2)
    assert scored.throughput_tok_s == 30.0
    assert scored.queue_depth == 3


def test_tier1_ignores_throughput_and_queue():
    # Tier 1 is pure latency + S2S RTT; the §4 throughput/queue signals apply only at
    # tier >= 2, so two peers equal on latency tie regardless of throughput/queue.
    health = [
        _health("a", latency_ms=10.0, throughput=5.0, queue=9),
        _health("b", latency_ms=10.0, throughput=50.0, queue=0),
    ]
    scores = {s.peer.peer_id: s.score for s in rank_peers(health, tier=1)}
    assert scores["a"] == scores["b"]
