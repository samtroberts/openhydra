"""Tests for supernode.selector — scoring, near-equal selection, failover."""

import collections
import time

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from supernode.manifest import SupernodeManifest, ModelCapability, HardwareInfo
from supernode.discovery import SupernodeDiscovery
from supernode.selector import (
    ScoredCandidate,
    score_candidate,
    select_supernode,
    PromptRouter,
    NEAR_EQUAL_THRESHOLD,
)


def _make_manifest(
    peer_id: str,
    model_ids: list[str] | None = None,
    trust_tier: str = "unverified",
    warm: bool = False,
    max_concurrent: int = 4,
    nat_status: str = "unknown",
) -> SupernodeManifest:
    model_ids = model_ids or ["llama3:8b"]
    key = Ed25519PrivateKey.generate()
    m = SupernodeManifest(
        peer_id=peer_id,
        libp2p_peer_id=f"12D3KooW{peer_id}",
        backend_type="ollama",
        version="0.1.0",
        integration_level=1,
        trust_tier=trust_tier,
        models=[
            ModelCapability(
                model_id=mid,
                model_family="llama",
                parameter_count=8000,
                quantization="Q4_0",
                context_length=8192,
                warm=warm,
            )
            for mid in model_ids
        ],
        max_concurrent=max_concurrent,
        max_context_length=8192,
        hardware=HardwareInfo(),
        listen_addrs=["/ip4/127.0.0.1/tcp/4001"],
        nat_status=nat_status,
        region="",
    )
    m.sign(key)
    return m


class TestScoreCandidate:
    def test_unverified_baseline(self):
        m = _make_manifest("a")
        s = score_candidate(m)
        assert 0.3 <= s <= 0.5

    def test_attested_higher(self):
        u = score_candidate(_make_manifest("a", trust_tier="unverified"))
        a = score_candidate(_make_manifest("b", trust_tier="attested"))
        assert a > u

    def test_supervised_middle(self):
        u = score_candidate(_make_manifest("a", trust_tier="unverified"))
        s = score_candidate(_make_manifest("b", trust_tier="supervised"))
        a = score_candidate(_make_manifest("c", trust_tier="attested"))
        assert u < s < a

    def test_warm_bonus(self):
        cold = score_candidate(_make_manifest("a", warm=False))
        hot = score_candidate(_make_manifest("b", warm=True))
        assert hot > cold

    def test_public_nat_bonus(self):
        priv = score_candidate(_make_manifest("a", nat_status="private"))
        pub = score_candidate(_make_manifest("b", nat_status="public"))
        assert pub > priv

    def test_no_concurrent_no_bonus(self):
        zero = score_candidate(_make_manifest("a", max_concurrent=0))
        four = score_candidate(_make_manifest("b", max_concurrent=4))
        assert four > zero


class TestSelectSupernode:
    def test_empty(self):
        assert select_supernode([]) is None

    def test_single(self):
        c = ScoredCandidate(manifest=_make_manifest("a"), score=1.0)
        assert select_supernode([c]) is c

    def test_deterministic_single(self):
        c = ScoredCandidate(manifest=_make_manifest("a"), score=1.0)
        for _ in range(50):
            assert select_supernode([c]) is c

    def test_spread_among_equal(self):
        """3 equal-score candidates should all get selected sometimes."""
        candidates = [
            ScoredCandidate(manifest=_make_manifest(f"p{i}"), score=1.0)
            for i in range(3)
        ]
        counts: dict[str, int] = collections.Counter()
        for _ in range(600):
            picked = select_supernode(list(candidates))
            counts[picked.manifest.peer_id] += 1

        for pid in ["p0", "p1", "p2"]:
            assert counts[pid] > 50, f"{pid} selected only {counts[pid]} times"

    def test_near_equal_threshold(self):
        """Candidate at 91% of best should be eligible; at 89% should not."""
        best = ScoredCandidate(manifest=_make_manifest("best"), score=1.0)
        near = ScoredCandidate(manifest=_make_manifest("near"), score=0.91)
        far = ScoredCandidate(manifest=_make_manifest("far"), score=0.50)

        counts: dict[str, int] = collections.Counter()
        for _ in range(500):
            picked = select_supernode([best, near, far])
            counts[picked.manifest.peer_id] += 1

        assert counts["best"] > 0
        assert counts["near"] > 0
        assert counts.get("far", 0) == 0

    def test_weighted_toward_higher(self):
        """Higher-scored candidate should be selected more often."""
        high = ScoredCandidate(manifest=_make_manifest("high"), score=1.0)
        low = ScoredCandidate(manifest=_make_manifest("low"), score=0.91)

        counts: dict[str, int] = collections.Counter()
        for _ in range(1000):
            picked = select_supernode([high, low])
            counts[picked.manifest.peer_id] += 1

        assert counts["high"] > counts["low"]


class TestPromptRouter:
    @pytest.fixture
    def discovery(self):
        return SupernodeDiscovery()

    @pytest.fixture
    def manifests(self):
        return [
            _make_manifest("a", model_ids=["llama3:8b"], trust_tier="attested", warm=True),
            _make_manifest("b", model_ids=["llama3:8b"], trust_tier="supervised"),
            _make_manifest("c", model_ids=["llama3:8b", "qwen:2b"], trust_tier="unverified"),
        ]

    @pytest.fixture
    def router(self, discovery, manifests):
        for m in manifests:
            discovery.register_manifest(m)
        return PromptRouter(discovery)

    def test_route_returns_scored(self, router):
        scored = router.route("llama3:8b")
        assert len(scored) == 3
        assert all(isinstance(s, ScoredCandidate) for s in scored)

    def test_route_sorted_descending(self, router):
        scored = router.route("llama3:8b")
        scores = [s.score for s in scored]
        assert scores == sorted(scores, reverse=True)

    def test_route_unknown_model(self, router):
        assert router.route("nonexistent:1b") == []

    def test_select_returns_candidate(self, router):
        result = router.select("llama3:8b")
        assert result is not None
        assert isinstance(result, ScoredCandidate)

    def test_select_unknown_model(self, router):
        assert router.select("nonexistent:1b") is None

    def test_failover_list_ordered(self, router):
        failover = router.select_with_failover("llama3:8b")
        assert len(failover) == 3
        peer_ids = {f.manifest.peer_id for f in failover}
        assert len(peer_ids) == 3

    def test_failover_max_attempts(self, router):
        failover = router.select_with_failover("llama3:8b", max_attempts=2)
        assert len(failover) == 2

    def test_failover_no_duplicates(self, router):
        failover = router.select_with_failover("llama3:8b")
        peer_ids = [f.manifest.libp2p_peer_id for f in failover]
        assert len(peer_ids) == len(set(peer_ids))

    def test_failure_penalty(self, router):
        scores_before = {s.manifest.peer_id: s.score for s in router.route("llama3:8b")}

        router.record_failure("12D3KooWa")
        scores_after = {s.manifest.peer_id: s.score for s in router.route("llama3:8b")}

        assert scores_after["a"] < scores_before["a"]

    def test_failure_expires(self, router):
        router._failed_peers["12D3KooWa"] = time.monotonic() - 120
        scored = router.route("llama3:8b")
        a_score = next(s.score for s in scored if s.manifest.peer_id == "a")
        fresh_score = score_candidate(_make_manifest("x", trust_tier="attested", warm=True))
        assert abs(a_score - fresh_score) < 0.01

    def test_register_adapter(self, router):
        router.register_adapter("12D3KooWa", "mock_adapter")
        assert router._adapters["12D3KooWa"] == "mock_adapter"

    def test_model_specific_routing(self, router):
        qwen = router.route("qwen:2b")
        assert len(qwen) == 1
        assert qwen[0].manifest.peer_id == "c"
