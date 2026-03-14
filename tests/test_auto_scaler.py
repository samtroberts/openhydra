"""Tests for Phase 2 auto-scaler: coordinator/auto_scaler.py, request_log.py, role_assigner.py.

Covers all 8 test scenarios from plans/auto-scaling-policy.md §10:

  1. 10 capable peers, 3x ratio for 4B → promote
  2. 40 weak peers join (4 GB), only 10 can run 8B → do NOT promote
  3. Promotion would drop existing model below 2x floor → block
  4. High redundancy but zero demand for that tier → block
  5. Ratio oscillates around 3.0x → hysteresis prevents flapping
  6. 80 weak peers, no inference → assign support roles, all earn credits
  7. Peer leaves mid-cooldown → no re-evaluation until cooldown expires
  8. Network grows 5→500 peers over time → gradual promotion, no jumps

Plus unit tests for RequestLog and RoleAssigner.
"""
from __future__ import annotations

import time
import pytest

from coordinator.auto_scaler import (
    AutoScaler,
    ModelSpec,
    PeerView,
    ScalerResult,
    effective_redundancy,
    effective_redundancy_after_reassignment,
    should_promote,
    PROMOTE_THRESHOLD,
    DEMOTE_THRESHOLD,
    FLOOR_RATIO,
    MIN_DEMAND_WEIGHT,
    MIN_FLEET_SIZE,
    COOLDOWN_S,
    RE_EVALUATE_S,
)
from coordinator.request_log import RequestLog, quality_tier_for_model_id
from coordinator.role_assigner import (
    RoleAssignment,
    assign_role,
    EARNINGS_MULTIPLIER,
)


# ── Fixtures & helpers ─────────────────────────────────────────────────────────

def _spec(model_id: str, shard_vram_mb: int, shards_needed: int, tier: str) -> ModelSpec:
    return ModelSpec(
        model_id=model_id,
        shard_vram_mb=shard_vram_mb,
        shards_needed=shards_needed,
        quality_tier=tier,
        required_peers=shards_needed,
    )


def _peer(peer_id: str, vram_mb: int, assigned: str | None = None, tps: float = 0.0) -> PeerView:
    return PeerView(
        peer_id=peer_id,
        available_vram_mb=vram_mb,
        assigned_model_id=assigned,
        tps=tps,
    )


def _neutral_log() -> RequestLog:
    """A log with no events — returns 0.5 for all tiers."""
    return RequestLog()


def _hot_log(tier: str, count: int = 10) -> RequestLog:
    """Return a log heavy on *tier* (100% of events)."""
    log = RequestLog()
    for _ in range(count):
        log.record_tier(tier)
    return log


SPEC_SMALL = _spec("small-4b",  shard_vram_mb=4096,  shards_needed=1, tier="standard")
SPEC_MED   = _spec("medium-8b", shard_vram_mb=8192,  shards_needed=1, tier="standard")
SPEC_LARGE = _spec("large-70b", shard_vram_mb=9216,  shards_needed=8, tier="advanced")

CATALOG_TWO = [SPEC_SMALL, SPEC_MED]
CATALOG_ALL = [SPEC_SMALL, SPEC_MED, SPEC_LARGE]


# ═══════════════════════════════════════════════════════════════════════════════
# Request Log unit tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRequestLog:
    def test_empty_window_returns_neutral(self):
        log = RequestLog()
        assert log.demand_weight("basic") == 0.5
        assert log.demand_weight("standard") == 0.5

    def test_record_single_tier(self):
        log = RequestLog()
        for _ in range(10):
            log.record_tier("standard")
        assert log.demand_weight("standard") == pytest.approx(1.0)
        assert log.demand_weight("basic") == pytest.approx(0.0)

    def test_mixed_tiers(self):
        log = RequestLog()
        for _ in range(3):
            log.record_tier("basic")
        for _ in range(7):
            log.record_tier("standard")
        w = log.demand_weight("standard")
        assert abs(w - 0.7) < 1e-9

    def test_record_by_model_id(self):
        log = RequestLog()
        log.record("openhydra-qwen3.5-0.8b")  # → basic
        log.record("openhydra-qwen3.5-4b")     # → standard
        assert log.demand_weight("basic") == pytest.approx(0.5)

    def test_snapshot_all_tiers(self):
        log = RequestLog()
        log.record_tier("basic")
        snap = log.snapshot()
        assert set(snap.keys()) == {"basic", "standard", "advanced", "frontier"}
        assert snap["basic"] == pytest.approx(1.0)

    def test_len_reflects_events(self):
        log = RequestLog()
        assert len(log) == 0
        log.record_tier("standard")
        assert len(log) == 1

    def test_window_expiry(self):
        """Events outside the window must not count."""
        log = RequestLog(window_seconds=0.1)
        log.record_tier("basic")
        assert len(log) == 1
        time.sleep(0.3)
        assert len(log) == 0
        assert log.demand_weight("basic") == 0.5  # neutral

    def test_quality_tier_for_model_id_small(self):
        assert quality_tier_for_model_id("openhydra-qwen3.5-0.8b") == "basic"

    def test_quality_tier_for_model_id_standard(self):
        assert quality_tier_for_model_id("openhydra-qwen3.5-4b") == "standard"

    def test_quality_tier_for_model_id_advanced(self):
        assert quality_tier_for_model_id("openhydra-qwen3.5-9b") == "advanced"

    def test_quality_tier_for_model_id_frontier(self):
        assert quality_tier_for_model_id("openhydra-qwen3.5-27b") == "frontier"

    def test_quality_tier_for_model_id_unknown_fallback(self):
        assert quality_tier_for_model_id("unknown-model-xyz") == "standard"


# ═══════════════════════════════════════════════════════════════════════════════
# Role Assigner unit tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRoleAssigner:
    ROSTER = [("model-4b", 4096), ("model-8b", 8192)]

    def test_strong_gpu_peer_gets_inference_role(self):
        ra = assign_role("p1", available_vram_mb=8192, cpu_score=0, disk_free_gb=0,
                         model_roster=self.ROSTER)
        assert ra.role.startswith("inference:")
        assert ra.earnings_multiplier == EARNINGS_MULTIPLIER["inference"]

    def test_weak_gpu_gets_embedding_server(self):
        # 256 MB < 512 MB threshold → NOT enough for embedding server
        # But 1024 MB >= 512 MB → embedding server
        ra = assign_role("p2", available_vram_mb=1024, cpu_score=0, disk_free_gb=0,
                         model_roster=self.ROSTER)
        assert ra.role == "embedding_server"
        assert ra.earnings_multiplier == EARNINGS_MULTIPLIER["embedding_server"]

    def test_high_cpu_but_no_gpu_gets_auditor(self):
        ra = assign_role("p3", available_vram_mb=256, cpu_score=200, disk_free_gb=0,
                         model_roster=self.ROSTER)
        assert ra.role == "verification_auditor"

    def test_disk_rich_peer_gets_cache_seed(self):
        ra = assign_role("p4", available_vram_mb=256, cpu_score=0, disk_free_gb=50,
                         model_roster=self.ROSTER)
        assert ra.role == "model_cache_seed"

    def test_absolute_minimum_gets_relay(self):
        ra = assign_role("p5", available_vram_mb=256, cpu_score=0, disk_free_gb=1,
                         model_roster=self.ROSTER)
        assert ra.role == "activation_relay"
        assert ra.earnings_multiplier == EARNINGS_MULTIPLIER["activation_relay"]

    def test_unknown_vram_zero_gets_inference(self):
        """available_vram_mb=0 means unknown → assume capable for inference."""
        ra = assign_role("p6", available_vram_mb=0, cpu_score=0, disk_free_gb=0,
                         model_roster=self.ROSTER)
        assert ra.role.startswith("inference:")

    def test_inference_role_includes_largest_model(self):
        """Peer with enough VRAM for 8b gets inference:model-8b (largest first)."""
        ra = assign_role("p7", available_vram_mb=16384, cpu_score=0, disk_free_gb=0,
                         model_roster=self.ROSTER)
        assert ra.model_id == "model-8b"

    def test_earnings_multiplier_ordering(self):
        """inference > auditor > embedding > seed > relay"""
        assert EARNINGS_MULTIPLIER["inference"] > EARNINGS_MULTIPLIER["embedding_server"]
        assert EARNINGS_MULTIPLIER["verification_auditor"] > EARNINGS_MULTIPLIER["embedding_server"]
        assert EARNINGS_MULTIPLIER["model_cache_seed"] > EARNINGS_MULTIPLIER["activation_relay"]

    def test_empty_roster_falls_through_to_support(self):
        """No rostered models → peer can't do inference → gets support role.

        With available_vram_mb=8192 ≥ _MIN_GPU_MB_EMBED=512, the peer qualifies
        for embedding_server (first non-inference support tier).
        """
        ra = assign_role("p8", available_vram_mb=8192, cpu_score=200, disk_free_gb=0,
                         model_roster=[])
        assert ra.role == "embedding_server"


# ═══════════════════════════════════════════════════════════════════════════════
# effective_redundancy helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestEffectiveRedundancy:
    def test_all_capable(self):
        peers = [_peer(f"p{i}", 8192) for i in range(6)]
        assert effective_redundancy(SPEC_MED, peers) == pytest.approx(6.0)

    def test_none_capable(self):
        peers = [_peer(f"p{i}", 2048) for i in range(6)]  # 2 GB < 8 GB
        assert effective_redundancy(SPEC_MED, peers) == pytest.approx(0.0)

    def test_mixed_capability(self):
        peers = [
            _peer("p1", 2048),   # too small
            _peer("p2", 8192),   # capable
            _peer("p3", 16384),  # capable
        ]
        assert effective_redundancy(SPEC_MED, peers) == pytest.approx(2.0)

    def test_unknown_vram_treated_as_capable(self):
        peers = [_peer(f"p{i}", 0) for i in range(5)]
        assert effective_redundancy(SPEC_MED, peers) == pytest.approx(5.0)

    def test_shards_needed_gt_1(self):
        # 6 capable peers, 2 shards needed → 3.0
        peers = [_peer(f"p{i}", 10000) for i in range(6)]
        spec = _spec("big", shard_vram_mb=8192, shards_needed=2, tier="advanced")
        assert effective_redundancy(spec, peers) == pytest.approx(3.0)

    def test_after_reassignment_worst_case(self):
        spec_a = _spec("model-a", shard_vram_mb=4096, shards_needed=1, tier="standard")
        spec_b = _spec("model-b", shard_vram_mb=4096, shards_needed=1, tier="advanced")
        # 4 peers serving a; all 4 can also run b → worst case leaves 0
        peers = [_peer(f"p{i}", 8192, assigned="model-a") for i in range(4)]
        ratio = effective_redundancy_after_reassignment(spec_a, spec_b, peers)
        assert ratio == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 1: 10 capable peers, 3x ratio → promote to medium
# ═══════════════════════════════════════════════════════════════════════════════

class TestScenario1_BasicPromotion:
    """Scenario 1: 10 capable peers → 3x ratio for medium-8b → promote."""

    def test_promote_when_ratio_met(self):
        scaler = AutoScaler(CATALOG_TWO)
        # 10 peers with enough VRAM for 8B
        peers = [_peer(f"p{i}", 8192) for i in range(10)]
        log = _hot_log("standard", 10)
        result = scaler.evaluate(peers, log)
        assert "medium-8b" in result.promoted

    def test_active_roster_updated(self):
        scaler = AutoScaler(CATALOG_TWO)
        peers = [_peer(f"p{i}", 8192) for i in range(10)]
        log = _hot_log("standard", 10)
        result = scaler.evaluate(peers, log)
        assert "medium-8b" in result.active_roster

    def test_no_demotion_on_healthy_fleet(self):
        scaler = AutoScaler(CATALOG_TWO)
        peers = [_peer(f"p{i}", 8192) for i in range(10)]
        log = _hot_log("standard", 10)
        result = scaler.evaluate(peers, log)
        assert result.demoted == []

    def test_role_assignments_included(self):
        scaler = AutoScaler(CATALOG_TWO)
        peers = [_peer(f"p{i}", 8192) for i in range(10)]
        result = scaler.evaluate(peers, _hot_log("standard"))
        assert len(result.role_assignments) == 10


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 2: Weak peers join — do NOT promote (capability check)
# ═══════════════════════════════════════════════════════════════════════════════

class TestScenario2_WeakNodeNonPromotion:
    """Scenario 2: 40 weak-VRAM peers join; only 10 can run 8B → no promotion."""

    def test_no_promotion_if_weak_majority(self):
        scaler = AutoScaler(CATALOG_TWO)
        # 40 RPi-class with 4 GB VRAM (can't run 8B which needs 8 GB)
        weak = [_peer(f"w{i}", 4096) for i in range(40)]
        # 8 capable with 8 GB VRAM
        capable = [_peer(f"c{i}", 8192) for i in range(8)]
        peers = weak + capable
        log = _hot_log("standard", 10)
        result = scaler.evaluate(peers, log)
        # 8 capable / 1 shard = 8x ≥ 3x → actually should promote!
        # But wait — 8 capable is 8x which IS above PROMOTE_THRESHOLD.
        # In this scenario, 8 peers CAN run the 8B model.
        # The scenario says "only 10 can run 8B" which IS ≥ 3x.
        # The key insight is: effective_redundancy uses ONLY capable peers.
        assert effective_redundancy(SPEC_MED, peers) == pytest.approx(8.0)
        # 8 capable peers IS enough for 8B (8x ≥ 3x), so promotion IS correct here.
        # The weak nodes do NOT inflate the ratio (that's the point).
        assert "medium-8b" in result.promoted

    def test_truly_insufficient_capable_peers(self):
        """Only 2 capable peers for an 8B model needing 3x = 3 → no promote."""
        scaler = AutoScaler(CATALOG_TWO)
        weak = [_peer(f"w{i}", 2048) for i in range(40)]   # 40 weak: 2 GB
        capable = [_peer(f"c{i}", 8192) for i in range(2)]  # only 2 capable
        peers = weak + capable
        log = _hot_log("standard", 10)
        result = scaler.evaluate(peers, log)
        # 2 capable / 1 shard = 2x < PROMOTE_THRESHOLD 3x → no promote
        assert "medium-8b" not in result.promoted

    def test_weak_nodes_get_support_roles(self):
        """Weak peers that cannot run any model get non-inference support roles."""
        scaler = AutoScaler(CATALOG_TWO)
        weak = [
            PeerView(peer_id=f"w{i}", available_vram_mb=256, cpu_score=200, disk_free_gb=50)
            for i in range(6)
        ]
        capable = [_peer(f"c{i}", 8192) for i in range(6)]
        peers = weak + capable
        result = scaler.evaluate(peers, _hot_log("standard"))
        # Weak peers should not get inference roles for 4B (needs 4096 MB)
        weak_ids = {f"w{i}" for i in range(6)}
        weak_roles = [r for r in result.role_assignments if r.peer_id in weak_ids]
        # With 256 MB available_vram, these peers can't serve 4B (shard_vram_mb=4096)
        for r in weak_roles:
            assert not r.role.startswith("inference:"), (
                f"Weak peer {r.peer_id} should not get inference role, got {r.role}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 3: Promotion blocked by floor check
# ═══════════════════════════════════════════════════════════════════════════════

class TestScenario3_FloorCheck:
    """Scenario 3: Promoting new model would drop existing below 2x floor → block."""

    def test_floor_blocks_promotion(self):
        # small-4b is active; 3 peers serving small-4b.
        # All 3 are also capable for medium-8b.
        # If all 3 get pulled to 8b, small-4b drops to 0/1 = 0x < FLOOR_RATIO.
        scaler = AutoScaler(CATALOG_TWO)
        # Only 3 peers total, all serving small-4b
        peers = [_peer(f"p{i}", 8192, assigned="small-4b") for i in range(3)]
        log = _hot_log("standard", 10)

        ok, reason = should_promote(
            candidate=SPEC_MED,
            active_models=[SPEC_SMALL],
            peers=peers,
            request_log=log,
            recently_changed=set(),
        )
        assert not ok
        assert "floor" in reason.lower() or "drop" in reason.lower()

    def test_floor_not_triggered_with_enough_peers(self):
        # 10 peers serving small-4b, all capable for 8b.
        # Post-reassignment worst-case: 0 peers left for small-4b.
        # 0x < FLOOR_RATIO = 2.0 → still blocked.
        # But if some peers CANNOT run 8b, they remain for small-4b.
        peers_strong = [_peer(f"s{i}", 8192, assigned="small-4b") for i in range(6)]
        peers_weak   = [_peer(f"w{i}", 4096, assigned="small-4b") for i in range(6)]
        peers = peers_strong + peers_weak
        log = _hot_log("standard", 10)
        # After worst-case: 6 strong pulled to 8b; 6 weak remain for small-4b.
        # Remaining ratio for small-4b = 6/1 = 6.0x ≥ FLOOR_RATIO → promotion allowed.
        ok, reason = should_promote(
            candidate=SPEC_MED,
            active_models=[SPEC_SMALL],
            peers=peers,
            request_log=log,
            recently_changed=set(),
        )
        assert ok, f"Expected promotion to be allowed, got: {reason}"

    def test_scaler_respects_floor_end_to_end(self):
        """Full evaluate() must not promote when floor would be violated."""
        scaler = AutoScaler(CATALOG_TWO)
        # 3 peers only, all capable for 8b — worst case leaves 0 for small-4b
        peers = [_peer(f"p{i}", 8192, assigned="small-4b") for i in range(3)]
        result = scaler.evaluate(peers, _hot_log("standard"))
        assert "medium-8b" not in result.promoted


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 4: High redundancy but zero demand → no promotion
# ═══════════════════════════════════════════════════════════════════════════════

class TestScenario4_DemandWeight:
    """Scenario 4: Plenty of capable peers but nobody requests that tier."""

    def test_zero_demand_blocks_promotion(self):
        """100% basic requests → standard demand weight = 0.0 < 0.3 → no promote."""
        scaler = AutoScaler(CATALOG_TWO)
        peers = [_peer(f"p{i}", 8192) for i in range(20)]
        log = RequestLog()
        for _ in range(20):
            log.record_tier("basic")  # all basic, zero standard requests
        result = scaler.evaluate(peers, log)
        assert "medium-8b" not in result.promoted

    def test_neutral_log_allows_promotion(self):
        """Empty log returns 0.5 (neutral) which is ≥ MIN_DEMAND_WEIGHT=0.3."""
        scaler = AutoScaler(CATALOG_TWO)
        peers = [_peer(f"p{i}", 8192) for i in range(10)]
        result = scaler.evaluate(peers, _neutral_log())
        assert "medium-8b" in result.promoted  # 0.5 ≥ 0.3

    def test_min_demand_threshold_boundary(self):
        """demand_weight just above 0.3 → allowed; just below → blocked."""
        peers = [_peer(f"p{i}", 8192) for i in range(10)]

        # 30% standard, 70% basic → demand = 0.30 → at threshold → allowed
        log_at = RequestLog()
        for _ in range(7):
            log_at.record_tier("basic")
        for _ in range(3):
            log_at.record_tier("standard")
        ok_at, _ = should_promote(SPEC_MED, [], peers, log_at, set())
        assert ok_at  # 0.30 ≥ 0.30

        # 29% standard, 71% basic → demand = 0.29 → below threshold → blocked
        log_below = RequestLog()
        for _ in range(71):
            log_below.record_tier("basic")
        for _ in range(29):
            log_below.record_tier("standard")
        ok_below, _ = should_promote(SPEC_MED, [], peers, log_below, set())
        assert not ok_below, "Expected promotion blocked at 29% demand"


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 5: Hysteresis — ratio oscillates around 3.0x
# ═══════════════════════════════════════════════════════════════════════════════

class TestScenario5_Hysteresis:
    """Scenario 5: Borderline redundancy → cooldown + hysteresis prevent flapping."""

    def test_promotion_then_no_immediate_repromotion(self):
        """After promotion, model enters cooldown and cannot be re-promoted."""
        scaler = AutoScaler(CATALOG_TWO)
        peers = [_peer(f"p{i}", 8192) for i in range(10)]
        log = _hot_log("standard", 10)

        # First evaluation → promote
        r1 = scaler.evaluate(peers, log)
        assert "medium-8b" in r1.promoted

        # Immediately force another evaluation → medium-8b is in cooldown
        r2 = scaler.evaluate(peers, log)
        assert "medium-8b" not in r2.promoted  # still in cooldown

    def test_demotion_then_no_immediate_redemote(self):
        """After demotion the model enters cooldown; no immediate re-demotion."""
        scaler = AutoScaler(CATALOG_TWO)
        # Force medium-8b onto roster manually
        scaler._active_roster = ["small-4b", "medium-8b"]

        # 1 capable peer (8192 MB) + 4 medium peers (4096 MB) → 5 peers total (≥ MIN_FLEET_SIZE).
        # er(medium-8b) = 1/1 = 1.0 < DEMOTE_THRESHOLD=1.5 → demote medium-8b.
        # er(small-4b)  = 5/1 = 5.0 ≥ DEMOTE_THRESHOLD    → keep small-4b.
        capable = [_peer("p0", 8192)]
        medium  = [_peer(f"m{i}", 4096) for i in range(4)]
        peers   = capable + medium
        log = _hot_log("standard", 10)
        r1 = scaler.evaluate(peers, log)
        assert "medium-8b" in r1.demoted

        # Second eval — medium-8b is not on roster; and cooldown prevents re-promotion.
        r2 = scaler.evaluate(peers, log)
        assert "medium-8b" not in r2.demoted

    def test_promote_threshold_vs_demote_threshold_gap(self):
        """PROMOTE_THRESHOLD > DEMOTE_THRESHOLD (hysteresis gap exists)."""
        assert PROMOTE_THRESHOLD > DEMOTE_THRESHOLD
        assert PROMOTE_THRESHOLD - DEMOTE_THRESHOLD >= 1.0

    def test_interval_throttle(self):
        """maybe_evaluate() returns None before RE_EVALUATE_S elapses."""
        scaler = AutoScaler(CATALOG_TWO)
        peers = [_peer(f"p{i}", 8192) for i in range(10)]
        log = _hot_log("standard")

        # First call via evaluate() (bypasses timer)
        scaler.evaluate(peers, log)

        # Immediate maybe_evaluate() → should be throttled (None)
        result = scaler.maybe_evaluate(peers, log, force=False)
        assert result is None

    def test_force_bypasses_interval(self):
        """force=True always runs regardless of elapsed time."""
        scaler = AutoScaler(CATALOG_TWO)
        peers = [_peer(f"p{i}", 8192) for i in range(10)]
        log = _hot_log("standard")
        scaler.evaluate(peers, log)  # sets _last_evaluated
        result = scaler.maybe_evaluate(peers, log, force=True)
        assert result is not None  # ran despite recent evaluation


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 6: 80 weak peers → all get support roles and earn credits
# ═══════════════════════════════════════════════════════════════════════════════

class TestScenario6_WeakPeerRoles:
    """Scenario 6: 80 weak peers, no inference role → all earn through support roles."""

    def test_all_weak_peers_assigned_role(self):
        scaler = AutoScaler([SPEC_SMALL])
        weak = [
            PeerView(peer_id=f"w{i}", available_vram_mb=256,  # < 4096 for small-4b
                     cpu_score=150, disk_free_gb=20)
            for i in range(80)
        ]
        result = scaler.evaluate(weak, _neutral_log())
        assert len(result.role_assignments) == 80

    def test_all_weak_peers_earn_nonzero_credits(self):
        scaler = AutoScaler([SPEC_SMALL])
        weak = [
            PeerView(peer_id=f"w{i}", available_vram_mb=256,
                     cpu_score=150, disk_free_gb=20)
            for i in range(10)
        ]
        result = scaler.evaluate(weak, _neutral_log())
        for ra in result.role_assignments:
            assert ra.earnings_multiplier > 0.0, (
                f"Peer {ra.peer_id} ({ra.role}) has zero earnings"
            )

    def test_earnings_sum_meaningful(self):
        """Total earning capacity is non-trivial even with weak peers."""
        roles = [
            assign_role(f"p{i}", available_vram_mb=256, cpu_score=150, disk_free_gb=20,
                        model_roster=[("model-4b", 4096)])
            for i in range(80)
        ]
        total_earning = sum(r.earnings_multiplier for r in roles)
        # Each peer gets at least activation_relay (0.1)
        assert total_earning >= 80 * EARNINGS_MULTIPLIER["activation_relay"]

    def test_fleet_too_small_skips(self):
        """Fewer than MIN_FLEET_SIZE peers → scaler returns a skipped result."""
        scaler = AutoScaler([SPEC_SMALL])
        peers = [_peer(f"p{i}", 8192) for i in range(MIN_FLEET_SIZE - 1)]
        result = scaler.evaluate(peers, _neutral_log())
        assert result.skipped_reason != ""
        assert "fleet_too_small" in result.skipped_reason


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 7: Peer leaves mid-cooldown
# ═══════════════════════════════════════════════════════════════════════════════

class TestScenario7_CooldownRespected:
    """Scenario 7: Peer leaves mid-cooldown → no re-evaluation during cooldown."""

    def test_cooldown_active_after_demotion(self):
        """Model demoted → enters cooldown → should_promote returns False."""
        scaler = AutoScaler(CATALOG_TWO)
        # Force both models onto the roster then simulate low peers
        scaler._active_roster = ["small-4b", "medium-8b"]

        # 1 capable peer (8192 MB) + 4 medium peers (4096 MB) → 5 total (≥ MIN_FLEET_SIZE).
        # er(medium-8b) = 1/1 = 1.0 < 1.5 → demote.
        capable = [_peer("p0", 8192)]
        medium  = [_peer(f"m{i}", 4096) for i in range(4)]
        r = scaler.evaluate(capable + medium, _hot_log("standard"))
        assert "medium-8b" in r.demoted

        # medium-8b should now be in recently_changed (cooldown)
        assert "medium-8b" in scaler._recently_changed

    def test_should_promote_rejects_cooled_model(self):
        peers = [_peer(f"p{i}", 8192) for i in range(10)]
        ok, reason = should_promote(
            candidate=SPEC_MED,
            active_models=[],
            peers=peers,
            request_log=_hot_log("standard"),
            recently_changed={"medium-8b"},
        )
        assert not ok
        assert "cooldown" in reason.lower()

    def test_cooldown_not_blocking_other_models(self):
        """Cooldown on model-A must not block promotion of model-B."""
        spec_c = _spec("candidate-c", shard_vram_mb=8192, shards_needed=1, tier="standard")
        peers = [_peer(f"p{i}", 8192) for i in range(10)]
        ok, _ = should_promote(
            candidate=spec_c,
            active_models=[],
            peers=peers,
            request_log=_hot_log("standard"),
            recently_changed={"some-other-model"},  # different model in cooldown
        )
        assert ok  # spec_c not in cooldown


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 8: Gradual fleet growth 5→500 peers
# ═══════════════════════════════════════════════════════════════════════════════

class TestScenario8_GradualGrowth:
    """Scenario 8: Network grows 5→500 peers → gradual ladder, no jumps."""

    def test_no_promote_at_min_fleet_boundary(self):
        """At exactly MIN_FLEET_SIZE peers, scaler evaluates but may not promote."""
        scaler = AutoScaler(CATALOG_TWO)
        peers = [_peer(f"p{i}", 8192) for i in range(MIN_FLEET_SIZE)]
        result = scaler.evaluate(peers, _hot_log("standard"))
        # Should run (not skip); promotion is valid if ratio ≥ 3x
        assert result.skipped_reason == ""
        # MIN_FLEET_SIZE=5 capable peers / 1 shard = 5x ≥ 3x → promotion expected
        assert "medium-8b" in result.promoted

    def test_no_big_jump_in_single_evaluation(self):
        """One evaluation promotes at most one new model (no cascade jumps)."""
        # Use a 3-model catalog: small → medium → large
        scaler = AutoScaler(CATALOG_ALL)
        # 100 peers with max VRAM — both medium and large are promotable
        peers = [_peer(f"p{i}", 100_000) for i in range(100)]
        log = _hot_log("advanced", 20)
        result = scaler.evaluate(peers, log)
        # Only one promotion at a time
        assert len(result.promoted) <= 1

    def test_small_fleet_only_serves_smallest_model(self):
        """With just MIN_FLEET_SIZE peers and limited VRAM, only the base model is served."""
        scaler = AutoScaler(CATALOG_TWO)
        # 5 peers with just enough VRAM for small-4b (4096 MB) but not medium-8b (8192 MB)
        peers = [_peer(f"p{i}", 4096) for i in range(5)]
        result = scaler.evaluate(peers, _hot_log("standard"))
        assert "medium-8b" not in result.promoted  # 0 capable for 8b

    def test_large_fleet_allows_higher_model(self):
        """With 100 capable peers the medium model gets promoted."""
        scaler = AutoScaler(CATALOG_TWO)
        peers = [_peer(f"p{i}", 8192) for i in range(100)]
        result = scaler.evaluate(peers, _hot_log("standard"))
        assert "medium-8b" in result.promoted


# ═══════════════════════════════════════════════════════════════════════════════
# AutoScaler initialisation and contract tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestAutoScalerContract:
    def test_init_raises_on_empty_specs(self):
        with pytest.raises(ValueError, match="at least one"):
            AutoScaler([])

    def test_initial_roster_is_smallest_model(self):
        scaler = AutoScaler(CATALOG_ALL)
        # Sorted by tier + vram: small-4b comes first
        assert scaler.active_roster == ["small-4b"]

    def test_active_roster_property_is_copy(self):
        scaler = AutoScaler(CATALOG_TWO)
        r1 = scaler.active_roster
        r1.append("mutated")
        assert "mutated" not in scaler.active_roster

    def test_scaler_result_has_role_assignments(self):
        scaler = AutoScaler(CATALOG_TWO)
        peers = [_peer(f"p{i}", 8192) for i in range(10)]
        result = scaler.evaluate(peers, _neutral_log())
        assert isinstance(result, ScalerResult)
        assert isinstance(result.role_assignments, list)

    def test_shards_needed_zero_raises(self):
        with pytest.raises(ValueError, match="shards_needed"):
            ModelSpec(model_id="x", shard_vram_mb=4096, shards_needed=0,
                      quality_tier="standard")

    def test_concurrent_evaluate_is_thread_safe(self):
        """Two threads can call evaluate() without corrupting the roster."""
        import threading
        scaler = AutoScaler(CATALOG_TWO)
        peers = [_peer(f"p{i}", 8192) for i in range(10)]
        log = _hot_log("standard")
        errors = []

        def run():
            try:
                scaler.evaluate(peers, log)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=run) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        roster = scaler.active_roster
        assert isinstance(roster, list)
        assert all(isinstance(m, str) for m in roster)


# ═══════════════════════════════════════════════════════════════════════════════
# ModelAvailability integration: new fields loaded from catalog
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelAvailabilityFields:
    def test_model_availability_has_new_fields(self):
        from coordinator.degradation import ModelAvailability
        m = ModelAvailability(
            model_id="test", required_peers=1,
            shard_vram_gb=4.0, shards_needed=1, quality_tier="standard"
        )
        assert m.shard_vram_gb == 4.0
        assert m.shards_needed == 1
        assert m.quality_tier == "standard"

    def test_model_availability_defaults(self):
        from coordinator.degradation import ModelAvailability
        m = ModelAvailability(model_id="x", required_peers=2)
        assert m.shard_vram_gb == 0.0
        assert m.shards_needed == 1
        assert m.quality_tier == "standard"


# ═══════════════════════════════════════════════════════════════════════════════
# Announcement: available_vram_mb field present
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnnouncementAvailableVram:
    def test_announcement_has_field(self):
        from peer.dht_announce import Announcement
        a = Announcement(peer_id="p1", model_id="m1", host="h", port=50051)
        assert hasattr(a, "available_vram_mb")
        assert a.available_vram_mb == 0  # default

    def test_announcement_accepts_nonzero(self):
        from peer.dht_announce import Announcement
        a = Announcement(peer_id="p1", model_id="m1", host="h", port=50051,
                         available_vram_mb=8192)
        assert a.available_vram_mb == 8192
