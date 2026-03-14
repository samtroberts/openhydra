import math

from coordinator.concentration_guard import (
    assemble_pipeline,
    concentration_metrics,
    enforce_operator_caps,
    enforce_pipeline_diversity,
)
from coordinator.path_finder import PeerEndpoint


def _peer(peer_id: str, operator_id: str) -> PeerEndpoint:
    return PeerEndpoint(peer_id=peer_id, host="127.0.0.1", port=1, operator_id=operator_id)


def test_concentration_metrics_flags_over_cap():
    peers = [_peer("a1", "op-a"), _peer("a2", "op-a"), _peer("b1", "op-b")]
    metrics = concentration_metrics(peers, cap_fraction=0.5)

    assert metrics.total_peers == 3
    assert metrics.max_operator == "op-a"
    assert metrics.max_share > 0.5
    assert metrics.over_cap_operators == ["op-a"]


def test_diversity_reorders_pipeline_when_possible():
    peers = [
        _peer("a1", "op-a"),
        _peer("a2", "op-a"),
        _peer("b1", "op-b"),
        _peer("c1", "op-c"),
    ]
    arranged = enforce_pipeline_diversity(peers, diversity_window=3, max_per_window=1)
    ops = [p.operator_id for p in arranged]

    for i in range(2, len(ops)):
        window = ops[i - 2 : i + 1]
        assert len(set(window)) == len(window)


def test_assemble_pipeline_applies_caps_and_diversity():
    peers = [
        _peer("a1", "op-a"),
        _peer("a2", "op-a"),
        _peer("a3", "op-a"),
        _peer("b1", "op-b"),
        _peer("c1", "op-c"),
    ]

    pipeline = assemble_pipeline(
        peers,
        pipeline_width=3,
        max_fraction=0.33,
        enforce_diversity=True,
        diversity_window=3,
        max_per_window=1,
    )

    assert len(pipeline) == 3
    ops = [p.operator_id for p in pipeline]
    assert ops.count("op-a") <= 1


def test_enforce_operator_caps_returns_short_list_when_single_operator():
    peers = [
        _peer("a1", "op-a"),
        _peer("a2", "op-a"),
        _peer("a3", "op-a"),
        _peer("a4", "op-a"),
    ]
    pipeline_width = 4
    max_fraction = 0.33

    pipeline = enforce_operator_caps(
        peers,
        pipeline_width=pipeline_width,
        max_fraction=max_fraction,
    )

    expected = max(1, math.floor(pipeline_width * max_fraction))
    assert len(pipeline) == expected
