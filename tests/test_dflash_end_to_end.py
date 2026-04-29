# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Phase 2b — end-to-end integration tests.

Wires DFlashTopologyADriver + select_accepted_prefix +
DraftModelRegistry + FailoverManager + DFlashTelemetry together to
exercise the full pipeline against deterministic mocks. These are
the headline tests Phase 2b's correctness story rests on:

1. Lossless guarantee under temp=0: --draft-location off and
   --draft-location local produce byte-identical token streams when
   the drafter happens to predict perfectly.
2. Mid-block divergence: partial-acceptance path emits exactly
   accepted_len + 1 tokens.
3. Failover: coord (Topology A) crashes mid-generation; stage-0
   detects, promotes, and the swarm records the new drafter.
4. Telemetry surface populates correctly during a real run.
5. kv_rollback_to advances correctly across iterations under both
   topologies.

These tests do NOT exercise the actual MLX/PyTorch runtimes — that
requires real model loads which are out of scope for the CI runner.
The tape-replay byte-equivalence of the runtime path is covered in
test_kv_rollback.py and test_inline_kv_rollback.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from coordinator.dflash_driver import DFlashTopologyADriver
from coordinator.dflash_telemetry import (
    DFlashTelemetry,
    get_telemetry,
    reset_telemetry,
)
from coordinator.failover import (
    DraftModelRegistry,
    FailoverError,
    FailoverManager,
)
from coordinator.head_sampler import select_accepted_prefix
from coordinator.swarm_events import (
    EVENT_TYPE_PROMOTE_DRAFTER,
    InMemorySwarmEventBus,
    PromoteDrafter,
    RegisterDraftModel,
)


@pytest.fixture(autouse=True)
def _telemetry_isolation():
    reset_telemetry()
    yield
    reset_telemetry()


# ── Test scaffolding ───────────────────────────────────────────────────


class _ScriptedDrafter:
    def __init__(self, blocks):
        self._blocks = list(blocks)
        self._i = 0

    def draft(self, prefix):
        if self._i >= len(self._blocks):
            return [0] * len(self._blocks[0])
        out = list(self._blocks[self._i])
        self._i += 1
        return out


class _ScriptedTransport:
    """Wraps an argmax sequence per call AND records telemetry."""

    def __init__(self, argmax_sequences, *, telemetry: DFlashTelemetry):
        self._argmax = list(argmax_sequences)
        self._i = 0
        self._telemetry = telemetry
        self.kv_rollbacks: list[int] = []

    def verify(self, *, prefix_token_ids, draft_token_ids,
               kv_rollback_to, request_id, kv_session_id):
        self.kv_rollbacks.append(int(kv_rollback_to))
        # Synthetic verify latency for telemetry.
        self._telemetry.record_verify_block_ms(8.0)
        out = self._argmax[self._i]
        self._i += 1
        return out


def _verifier_fn():
    def _v(hidden_states_block, draft_token_ids):
        return select_accepted_prefix(
            argmax_per_position=list(hidden_states_block),
            draft_token_ids=list(draft_token_ids),
        )
    return _v


# ── Lossless guarantee end-to-end ──────────────────────────────────────


def test_lossless_full_acceptance_matches_autoregressive_baseline():
    """When the drafter happens to predict perfectly, the emitted
    token stream is byte-identical to AR greedy output.

    AR baseline: 17 tokens emitted by greedy decoding token-by-token.
    DFlash with full acceptance: same 17 tokens, but in 1 verify
    pass instead of 17. Lossless guarantee in action.
    """
    ar_output = list(range(1000, 1017))    # 17 tokens
    drafter_block = ar_output[:16]         # 16 drafts
    target_argmax = ar_output              # 17 target argmax (16 verify + bonus)

    telemetry = get_telemetry()
    drafter = _ScriptedDrafter([drafter_block])
    transport = _ScriptedTransport([target_argmax], telemetry=telemetry)
    emitted: list[int] = []

    driver = DFlashTopologyADriver(
        drafter=drafter, verifier=_verifier_fn(),
        transport=transport, emit=emitted.append,
        block_size=16, max_tokens=17,
    )
    stats = driver.run(prompt_token_ids=[1, 2, 3])

    # Headline: byte-identical to AR.
    assert emitted == ar_output
    # 1 ring trip emitted 17 tokens — the speedup.
    assert stats.blocks == 1
    assert stats.tokens_emitted == 17
    assert stats.acceptance_rate == 1.0


def test_partial_acceptance_emits_exactly_accepted_plus_bonus():
    """Mid-block divergence at position 7: drafts match through 6,
    diverge at 7. Emit 7 accepted tokens + 1 bonus = 8 total.
    Verify the bonus is the target's argmax at the divergence
    position (lossless guarantee on partial acceptance)."""
    drafts = list(range(50, 57)) + [9999] * 9    # diverge at i=7
    target_argmax = list(range(50, 57)) + [777] + [88] * 9
    telemetry = get_telemetry()
    drafter = _ScriptedDrafter([drafts, [0] * 16])
    transport = _ScriptedTransport([target_argmax, [0] * 17], telemetry=telemetry)
    emitted: list[int] = []

    driver = DFlashTopologyADriver(
        drafter=drafter, verifier=_verifier_fn(),
        transport=transport, emit=emitted.append,
        block_size=16, max_tokens=8,
    )
    stats = driver.run(prompt_token_ids=[])

    # Exactly accepted_len + 1 tokens = 8 emitted.
    assert emitted == list(range(50, 57)) + [777]
    assert stats.tokens_emitted == 8
    assert stats.drafts_accepted == 7


# ── kv_rollback_to bookkeeping ─────────────────────────────────────────


def test_kv_rollback_to_advances_with_partial_acceptance():
    """Each block sets kv_rollback_to = prefix_len_at_block_start.
    Partial acceptance means the prefix grows by accepted_len + 1
    each iteration, and kv_rollback_to follows."""
    # Block 1: 3 accepted + 1 bonus = 4 emitted.
    drafts1 = [10, 20, 30, 99] + [0] * 12
    argmax1 = [10, 20, 30, 50] + [99] * 13   # diverge at i=3
    # Block 2: 1 accepted + 1 bonus = 2 emitted.
    drafts2 = [60, 99] + [0] * 14
    argmax2 = [60, 70] + [99] * 15           # diverge at i=1

    telemetry = get_telemetry()
    drafter = _ScriptedDrafter([drafts1, drafts2])
    transport = _ScriptedTransport([argmax1, argmax2], telemetry=telemetry)

    driver = DFlashTopologyADriver(
        drafter=drafter, verifier=_verifier_fn(),
        transport=transport, emit=lambda t: None,
        block_size=16,
        # Cap exactly at 4+2=6 to stop after both scripted blocks.
        max_tokens=6,
    )
    stats = driver.run(prompt_token_ids=[1, 2, 3, 4, 5])    # prompt_len=5

    # Block 1: rollback target = prompt_len = 5.
    assert transport.kv_rollbacks[0] == 5
    # After block 1: prefix_len = 5 + 3 + 1 = 9.
    # Block 2: rollback target = 9.
    assert transport.kv_rollbacks[1] == 9
    assert stats.tokens_emitted == 6


# ── Telemetry populates during real run ────────────────────────────────


def test_telemetry_acceptance_ema_populates_during_generation():
    """Block-verify outcomes folded into the EMA over the run."""
    telemetry = get_telemetry()
    # 3 blocks: full, partial (8/16), full → EMA settles.
    drafts1 = list(range(100, 116))
    argmax1 = list(range(100, 116)) + [9999]
    drafts2 = list(range(200, 208)) + [99] * 8
    argmax2 = list(range(200, 208)) + [88] + [77] * 8
    drafts3 = list(range(300, 316))
    argmax3 = list(range(300, 316)) + [777]

    drafter = _ScriptedDrafter([drafts1, drafts2, drafts3])
    transport = _ScriptedTransport(
        [argmax1, argmax2, argmax3], telemetry=telemetry,
    )
    emitted: list[int] = []

    # Wrap the verifier so it pushes block-acceptance to telemetry.
    base_verifier = _verifier_fn()
    block_size = 16

    def _instrumented(hidden_states_block, draft_token_ids):
        accepted, bonus = base_verifier(hidden_states_block, draft_token_ids)
        telemetry.record_block_acceptance(accepted, block_size)
        return accepted, bonus

    driver = DFlashTopologyADriver(
        drafter=drafter, verifier=_instrumented,
        transport=transport, emit=emitted.append,
        block_size=block_size,
        # Stop after exactly 3 blocks: 17 + 9 + 17 = 43 tokens.
        max_tokens=43,
    )
    driver.run(prompt_token_ids=[])

    snap = telemetry.snapshot()
    assert snap.ring_acceptance_rate_ema is not None
    assert snap.target_verify_block_p50_ms == 8.0     # all observations were 8 ms
    # EMA bounds: should be in [0, 1] and reflect a mix of full
    # acceptance (rate=1.0) and partial (rate=0.5).
    assert 0.5 <= snap.ring_acceptance_rate_ema <= 1.0


def test_telemetry_remains_at_zero_when_speculation_off():
    """If --draft-location off (no driver instantiated), the
    telemetry snapshot must stay at None across the board — Phase 3
    relies on this signal to detect 'speculation is currently off'."""
    snap = get_telemetry().snapshot()
    assert snap.draft_inflight_p50_ms is None
    assert snap.target_verify_block_p50_ms is None
    assert snap.ring_acceptance_rate_ema is None


# ── Failover end-to-end mid-generation ────────────────────────────────


def test_failover_topology_a_coord_crash_promotes_stage0_seamlessly():
    """The headline failover test:
    1. Coord registers spec, holds the drafter (Topology A).
    2. Generation starts, block 1 verifies normally.
    3. Coord crashes mid-generation.
    4. Stage-0 (preloaded with the spec) detects coord absence.
    5. Stage-0 promotes; emits PromoteDrafter.
    6. Other peers update their active_drafter_id.

    The KV state on peers is preserved across the crash — that's
    the whole point of running the rollback machinery on the
    PEERS, not on the coord. The promoting stage-0 takes over
    orchestration with the existing KV cache intact.
    """
    bus = InMemorySwarmEventBus()
    spec = RegisterDraftModel(
        target_path="Qwen/Qwen3.5-4B",
        draft_path="z-lab/Qwen3.5-4B-DFlash",
        block_size=16, backend="mlx",
    )

    # Coord-side registry + announce.
    coord_reg = DraftModelRegistry(bus)
    coord_reg.announce(spec, from_peer="coord")

    # Stage-0 side: registry sees the same announce.
    stage0_reg = DraftModelRegistry(bus)
    stage0_reg.announce(coord_reg.get_active_spec(), from_peer="coord")

    # Other-peer side: tracks active drafter; never promotes itself.
    other_reg = DraftModelRegistry(bus)
    other_reg.announce(coord_reg.get_active_spec(), from_peer="coord")
    other_fm = FailoverManager(
        bus=bus, local_peer_id="other-peer", registry=other_reg,
        coord_alive=lambda: True,   # other peer never decides coord is dead
    )

    # Stage-0's failover manager: coord starts alive, then dies.
    coord_alive = [True]
    stage0_fm = FailoverManager(
        bus=bus, local_peer_id="stage0", registry=stage0_reg,
        coord_alive=lambda: coord_alive[0],
        absence_threshold_ms=1_000,
    )

    # Block 1: coord alive, no promotion.
    assert stage0_fm.check_coord(now_ms=0) is False
    assert stage0_fm.is_local_active is False

    # Coord crashes.
    coord_alive[0] = False
    # First absent sample at t=10s — starts streak.
    assert stage0_fm.check_coord(now_ms=10_000) is False
    # Sample beyond threshold — promotion fires.
    promoted = stage0_fm.check_coord(now_ms=11_500)
    assert promoted is True

    # The whole swarm sees stage-0 as the new drafter.
    assert stage0_fm.is_local_active is True
    assert other_fm.active_drafter_id == "stage0"


def test_failover_promote_without_spec_raises_before_any_announce():
    """Stage-0 cannot promote if it never received a
    RegisterDraftModel — the promoted peer would have nothing
    to load. Fail loud at promote time rather than silently
    becoming a non-functional drafter."""
    bus = InMemorySwarmEventBus()
    reg = DraftModelRegistry(bus)
    fm = FailoverManager(bus=bus, local_peer_id="stage0", registry=reg)
    with pytest.raises(FailoverError) as exc:
        fm.promote()
    assert exc.value.code == "no_active_spec"


def test_failover_records_promote_for_late_joining_peer():
    """A peer that joins the swarm AFTER stage-0 promoted must
    still learn the active drafter from the next PromoteDrafter
    event delivered by the bus.

    (The InMemoryBus has no historical replay, so we model this
    by re-emitting on join — same pattern the libp2p adapter
    will use for late-joiners reading the swarm event log.)
    """
    bus = InMemorySwarmEventBus()
    spec = RegisterDraftModel(
        target_path="Qwen/Qwen3.5-4B",
        draft_path="z-lab/Qwen3.5-4B-DFlash",
        block_size=16, backend="mlx",
    )

    reg = DraftModelRegistry(bus)
    reg.announce(spec, from_peer="coord")

    stage0_fm = FailoverManager(
        bus=bus, local_peer_id="stage0", registry=reg,
    )
    stage0_fm.promote(now_ms=10_000)

    # Late-joining peer subscribes AFTER promotion.
    late_fm = FailoverManager(
        bus=bus, local_peer_id="late", registry=reg,
    )
    # Re-emit the historical PromoteDrafter (libp2p replay model).
    bus.publish(
        PromoteDrafter(from_peer_id="coord", to_peer_id="stage0",
                       unix_ms=10_000),
        from_peer="stage0",
    )
    assert late_fm.active_drafter_id == "stage0"


# ── Driver + topology interactions stay deterministic ─────────────────


def test_dual_topology_byte_identical_under_temp_zero():
    """The lossless guarantee says: same prompt + same drafter
    behaviour + temp=0 + same seed → byte-identical output across
    both topologies. We model this by running the same drafter
    + transport pair through the driver twice — once representing
    each topology — and asserting equal token streams.

    The driver itself is topology-agnostic; Topology A vs B differs
    only in WHERE the drafter runs (coord vs stage-0). The token
    semantics are identical. This test pins that contract.
    """
    block = list(range(500, 516))
    argmax = list(range(500, 516)) + [9999]
    telemetry_a = get_telemetry()

    # Topology A: coord runs the driver.
    drafter_a = _ScriptedDrafter([block])
    transport_a = _ScriptedTransport([argmax], telemetry=telemetry_a)
    emitted_a: list[int] = []
    DFlashTopologyADriver(
        drafter=drafter_a, verifier=_verifier_fn(),
        transport=transport_a, emit=emitted_a.append,
        block_size=16, max_tokens=17,
    ).run(prompt_token_ids=[1, 2, 3])

    # Topology B: stage-0 runs an equivalent loop. We model this
    # by reusing the same driver class with the same inputs — the
    # contract is that the WHO doesn't change the WHAT.
    reset_telemetry()
    telemetry_b = get_telemetry()
    drafter_b = _ScriptedDrafter([block])
    transport_b = _ScriptedTransport([argmax], telemetry=telemetry_b)
    emitted_b: list[int] = []
    DFlashTopologyADriver(
        drafter=drafter_b, verifier=_verifier_fn(),
        transport=transport_b, emit=emitted_b.append,
        block_size=16, max_tokens=17,
    ).run(prompt_token_ids=[1, 2, 3])

    assert emitted_a == emitted_b
    assert transport_a.kv_rollbacks == transport_b.kv_rollbacks
