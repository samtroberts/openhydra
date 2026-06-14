# Copyright 2026 OpenHydra contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Integration test: the live reputation feedback loop (M2.2).

Proves the full path through the FFI without a live swarm: a verification outcome
recorded via ``node.record_reputation_outcome`` lands in the node's per-peer
reputation store, and the router's rank stage (``node.rank_discovered`` — the same
``effective_reputation`` override the live ``resolve_and_route`` discover closure
applies) then ranks the penalized peer below an otherwise-identical baseline peer.

The pure score/decay math is covered by the Rust unit tests in ``network/src/verify.rs``;
this asserts the state store + FFI feedback + router override behave end to end.
"""

from __future__ import annotations

import pytest

ohn = pytest.importorskip("openhydra_network")


def _two_identical_peers() -> list[dict]:
    # Same throughput / load / queue — they differ ONLY by whatever reputation the
    # local store assigns, so any rank change is attributable to reputation alone.
    common = {"throughput_tok_s": 20.0, "load_pct": 10.0, "queue_depth": 0, "reputation_score": 0.0}
    return [{"peer_id": "peerA", **common}, {"peer_id": "peerB", **common}]


def test_unknown_peers_score_neutral(tmp_path):
    node = ohn.P2PNode(identity_key_path=str(tmp_path / "n.key"))
    # A peer we've never recorded an outcome for sits at the neutral baseline (50.0).
    assert node.reputation_score("never-seen") == 50.0


def test_outcomes_move_the_stored_score(tmp_path):
    node = ohn.P2PNode(identity_key_path=str(tmp_path / "n.key"))
    assert node.reputation_score("p") == 50.0
    honored = node.record_reputation_outcome("p", "honored")
    assert honored > 50.0  # additive increase
    node.record_reputation_outcome("p", "failed")
    assert node.reputation_score("p") < honored  # multiplicative drop


def test_unknown_outcome_string_rejected(tmp_path):
    node = ohn.P2PNode(identity_key_path=str(tmp_path / "n.key"))
    with pytest.raises(ValueError, match="unknown reputation outcome"):
        node.record_reputation_outcome("p", "bogus")


def test_failed_receipt_downranks_peer_in_routing(tmp_path):
    # The headline M2.2 loop: a failed receipt recorded over the FFI makes the router
    # rank that peer below an identical, un-penalized peer.
    node = ohn.P2PNode(identity_key_path=str(tmp_path / "n.key"))
    peers = _two_identical_peers()

    # Baseline: identical peers, neither yet penalized → stable order keeps input order.
    base = node.rank_discovered(peers, 2)
    assert [p["peer_id"] for p in base] == ["peerA", "peerB"]

    # Report a failed receipt against peerA through the FFI feedback loop.
    node.record_reputation_outcome("peerA", "failed")
    assert node.reputation_score("peerA") < node.reputation_score("peerB")  # B still neutral 50

    # Subsequent ranking: the penalized peer is now actively ranked lower.
    after = node.rank_discovered(peers, 2)
    assert [p["peer_id"] for p in after] == ["peerB", "peerA"]


def test_honored_receipts_promote_a_peer(tmp_path):
    # The mirror case: repeated honored receipts lift a peer above an identical baseline.
    node = ohn.P2PNode(identity_key_path=str(tmp_path / "n.key"))
    peers = _two_identical_peers()
    for _ in range(5):
        node.record_reputation_outcome("peerB", "honored")
    assert node.reputation_score("peerB") > node.reputation_score("peerA")
    ranked = node.rank_discovered(peers, 2)
    assert ranked[0]["peer_id"] == "peerB"


def test_tier1_ignores_reputation(tmp_path):
    # Tier 1 is pure latency + S2S; a reputation penalty must NOT reorder a tier-1 rank
    # (guards against the override leaking into the latency-only profile).
    node = ohn.P2PNode(identity_key_path=str(tmp_path / "n.key"))
    peers = _two_identical_peers()
    node.record_reputation_outcome("peerA", "failed")
    ranked = node.rank_discovered(peers, 1)
    # equal latency + no reputation term ⇒ stable order preserved despite A's penalty.
    assert [p["peer_id"] for p in ranked] == ["peerA", "peerB"]
