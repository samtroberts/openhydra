# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Phase 2b live-bench Item 3 — DFlash integration into inference_service.

Exercises the integration layer end-to-end:
  * dflash_eligible(config) gate semantics.
  * setup_dflash_session() builds the bundle, registers the
    spec, sets up failover.
  * run_dflash_generation drives the driver and produces the
    expected token stream.
  * InProcessRingVerifyTransport delegates to a caller-provided
    target callable.
  * MultiPeerRingVerifyTransport stub raises actionably.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest


# ── dflash_eligible ────────────────────────────────────────────────────


def test_eligible_off_returns_false():
    from coordinator.dflash_integration import dflash_eligible

    @dataclass
    class _Cfg:
        draft_location: str = "off"
        draft_model_path: str = ""
    assert dflash_eligible(_Cfg()) is False


def test_eligible_local_with_draft_path_returns_true():
    from coordinator.dflash_integration import dflash_eligible

    @dataclass
    class _Cfg:
        draft_location: str = "local"
        draft_model_path: str = "z-lab/Qwen3.5-4B-DFlash"
    assert dflash_eligible(_Cfg()) is True


def test_eligible_stage0_with_draft_path_returns_true():
    from coordinator.dflash_integration import dflash_eligible

    @dataclass
    class _Cfg:
        draft_location: str = "stage-0"
        draft_model_path: str = "z-lab/Qwen3.5-4B-DFlash"
    assert dflash_eligible(_Cfg()) is True


def test_eligible_local_without_draft_path_returns_false():
    """A misconfigured deployment (--draft-location=local without
    --draft-model) falls back to off rather than crashing at
    drafter-load time."""
    from coordinator.dflash_integration import dflash_eligible

    @dataclass
    class _Cfg:
        draft_location: str = "local"
        draft_model_path: str = ""
    assert dflash_eligible(_Cfg()) is False


def test_eligible_unknown_location_returns_false():
    from coordinator.dflash_integration import dflash_eligible

    @dataclass
    class _Cfg:
        draft_location: str = "moon"
        draft_model_path: str = "x"
    assert dflash_eligible(_Cfg()) is False


# ── setup_dflash_session ──────────────────────────────────────────────


@dataclass
class _MiniCfg:
    draft_location: str = "local"
    draft_model_path: str = "z-lab/Qwen3.5-4B-DFlash"
    draft_block_size: int = 16
    default_target_model: str = "Qwen/Qwen3.5-4B"


class _FakeHeadSampler:
    """HeadSampler stand-in — exposes ``verify_block`` matching the
    real signature and returns a canned (accepted, bonus)."""

    def __init__(self, accepted_len: int = 0, bonus: int = 42):
        self._accepted = accepted_len
        self._bonus = bonus
        self.calls: list = []

    def verify_block(self, *, hidden_states_block, draft_token_ids, decode):
        self.calls.append((hidden_states_block, list(draft_token_ids)))
        return self._accepted, self._bonus


def test_setup_session_off_raises_draft_off():
    from coordinator.dflash_integration import (
        DFlashIntegrationError, setup_dflash_session,
    )
    from coordinator.swarm_events import InMemorySwarmEventBus

    @dataclass
    class _Off:
        draft_location: str = "off"
        draft_model_path: str = ""

    bus = InMemorySwarmEventBus()
    with pytest.raises(DFlashIntegrationError) as exc:
        setup_dflash_session(
            config=_Off(), head_sampler=_FakeHeadSampler(),
            bus=bus, local_peer_id="me",
        )
    assert exc.value.code == "draft_off"


def test_setup_session_missing_head_raises():
    from coordinator.dflash_integration import (
        DFlashIntegrationError, setup_dflash_session,
    )
    from coordinator.swarm_events import InMemorySwarmEventBus

    bus = InMemorySwarmEventBus()
    with pytest.raises(DFlashIntegrationError) as exc:
        setup_dflash_session(
            config=_MiniCfg(), head_sampler=None,
            bus=bus, local_peer_id="me",
        )
    assert exc.value.code == "missing_head"


def test_setup_session_announces_spec_to_registry():
    """Calling setup_dflash_session must emit a RegisterDraftModel
    event so the swarm registry records the active spec."""
    from coordinator.dflash_integration import setup_dflash_session
    from coordinator.dflash_telemetry import reset_telemetry
    from coordinator.swarm_events import (
        EVENT_TYPE_REGISTER_DRAFT_MODEL,
        InMemorySwarmEventBus,
    )

    reset_telemetry()
    bus = InMemorySwarmEventBus()
    received = []
    bus.subscribe(EVENT_TYPE_REGISTER_DRAFT_MODEL, received.append)

    session = setup_dflash_session(
        config=_MiniCfg(), head_sampler=_FakeHeadSampler(),
        bus=bus, local_peer_id="coord-id",
    )
    assert len(received) == 1
    spec = received[0].payload
    assert spec.target_path == "Qwen/Qwen3.5-4B"
    assert spec.draft_path == "z-lab/Qwen3.5-4B-DFlash"
    assert spec.block_size == 16
    assert spec.backend == "mlx"

    # Registry sees it too.
    assert session.registry.get_active_spec() is not None


def test_setup_session_attaches_failover_manager():
    from coordinator.dflash_integration import setup_dflash_session
    from coordinator.dflash_telemetry import reset_telemetry
    from coordinator.swarm_events import InMemorySwarmEventBus

    reset_telemetry()
    bus = InMemorySwarmEventBus()
    session = setup_dflash_session(
        config=_MiniCfg(), head_sampler=_FakeHeadSampler(),
        bus=bus, local_peer_id="me",
    )
    assert session.failover is not None
    # No active drafter yet — promotion only fires on coord absence.
    assert session.failover.active_drafter_id is None


def test_setup_session_drafter_uses_mock_backend_when_lazy_load_skipped():
    """The drafter is constructed lazily — load_dflash_drafter does
    NOT call ensure_loaded(), so a setup against an MLX backend with
    no MLX installed must succeed at setup time. The drafter blows
    up only on the first .draft() call."""
    from coordinator.dflash_integration import setup_dflash_session
    from coordinator.dflash_telemetry import reset_telemetry
    from coordinator.swarm_events import InMemorySwarmEventBus

    reset_telemetry()
    bus = InMemorySwarmEventBus()
    # Setup must not raise.
    session = setup_dflash_session(
        config=_MiniCfg(), head_sampler=_FakeHeadSampler(),
        bus=bus, local_peer_id="me",
    )
    assert session.drafter is not None
    assert session.block_size == 16


# ── InProcessRingVerifyTransport ─────────────────────────────────────


def test_inprocess_transport_invokes_run_target_callable():
    from coordinator.dflash_integration import InProcessRingVerifyTransport
    from coordinator.dflash_telemetry import reset_telemetry, get_telemetry

    reset_telemetry()
    telemetry = get_telemetry()
    captured: list = []

    def _run_target(prefix, drafts):
        captured.append((list(prefix), list(drafts)))
        return [99] * (len(drafts) + 1)   # synthetic argmax

    transport = InProcessRingVerifyTransport(
        run_target_block=_run_target, telemetry=telemetry,
    )
    out = transport.verify(
        prefix_token_ids=[1, 2, 3],
        draft_token_ids=[10, 20, 30, 40],
        kv_rollback_to=3,
        request_id="r1",
        kv_session_id="s1",
    )
    assert out == [99, 99, 99, 99, 99]
    assert captured == [([1, 2, 3], [10, 20, 30, 40])]
    # Telemetry recorded a verify-block latency.
    snap = telemetry.snapshot()
    assert snap.target_verify_block_p50_ms is not None


def test_inprocess_transport_requires_callable():
    from coordinator.dflash_integration import InProcessRingVerifyTransport

    with pytest.raises(ValueError, match="callable"):
        InProcessRingVerifyTransport(run_target_block="not callable")


# ── MultiPeerRingVerifyTransport stub ─────────────────────────────────


def test_multipeer_stub_raises_with_actionable_error():
    from coordinator.dflash_integration import (
        DFlashIntegrationError, MultiPeerRingVerifyTransport,
    )
    transport = MultiPeerRingVerifyTransport()
    with pytest.raises(DFlashIntegrationError) as exc:
        transport.verify()
    assert exc.value.code == "multipeer_unsupported"
    assert "InProcessRingVerifyTransport" in str(exc.value)


# ── run_dflash_generation end-to-end ──────────────────────────────────


def test_run_dflash_generation_produces_full_acceptance_stream():
    """Wire all the pieces together against an in-process transport
    that returns full-acceptance argmax. Driver emits 17 tokens
    in 1 block — the headline DFlash speedup case."""
    from coordinator.dflash_integration import (
        InProcessRingVerifyTransport,
        run_dflash_generation,
        setup_dflash_session,
    )
    from coordinator.dflash_telemetry import reset_telemetry
    from coordinator.swarm_events import InMemorySwarmEventBus

    reset_telemetry()
    bus = InMemorySwarmEventBus()

    # MockDFlashDrafter is deterministic — for prefix [1,2,3] it
    # produces a specific block. We capture that block, then construct
    # a transport that returns argmax matching it (full acceptance).
    @dataclass
    class _Cfg:
        draft_location: str = "local"
        draft_model_path: str = "z-lab/Qwen3.5-4B-DFlash"
        draft_block_size: int = 16
        default_target_model: str = "Qwen/Qwen3.5-4B"

    # The drafter loaded by setup_dflash_session is the real MLX/PyTorch
    # one — but we override with a mock for this test.
    from coordinator.dflash_draft import DFlashConfig, MockDFlashDrafter

    head = _FakeHeadSampler()
    session = setup_dflash_session(
        config=_Cfg(), head_sampler=head,
        bus=bus, local_peer_id="me",
    )
    # Replace drafter with the mock.
    session.drafter = MockDFlashDrafter(
        DFlashConfig(backend="mock", block_size=16),
    )

    # Capture what the drafter emits given prompt [1,2,3].
    expected_block = session.drafter.draft([1, 2, 3])

    def _full_acceptance_target(prefix, drafts):
        # Return argmax = drafts + bonus → full acceptance.
        return list(drafts) + [777]

    transport = InProcessRingVerifyTransport(
        run_target_block=_full_acceptance_target,
        telemetry=session.telemetry,
    )

    # The fake HeadSampler returns canned (accepted, bonus). To make
    # this test exercise the FULL pipeline including the verify path
    # we swap the verifier directly to use select_accepted_prefix
    # against the argmax the transport returns. (The real HeadSampler
    # does the same, just routed through apply_final_head_block.)
    from coordinator.head_sampler import select_accepted_prefix

    def _real_verifier(hidden_states_block, draft_token_ids):
        return select_accepted_prefix(
            argmax_per_position=list(hidden_states_block),
            draft_token_ids=list(draft_token_ids),
        )
    session.verifier = _real_verifier

    out = run_dflash_generation(
        session=session,
        transport=transport,
        prompt_token_ids=[1, 2, 3],
        max_tokens=17,
        request_id="r1",
        kv_session_id="s1",
    )

    # Full acceptance: 16 drafts accepted + 1 bonus = 17 tokens.
    assert out["tokens_emitted"] == 17
    assert out["blocks"] == 1
    assert out["acceptance_rate"] == 1.0
    assert out["tokens"] == list(expected_block) + [777]


def test_run_dflash_generation_requires_transport():
    from coordinator.dflash_integration import (
        DFlashIntegrationError, run_dflash_generation,
        setup_dflash_session,
    )
    from coordinator.dflash_telemetry import reset_telemetry
    from coordinator.swarm_events import InMemorySwarmEventBus

    reset_telemetry()
    bus = InMemorySwarmEventBus()
    session = setup_dflash_session(
        config=_MiniCfg(), head_sampler=_FakeHeadSampler(),
        bus=bus, local_peer_id="me",
    )
    with pytest.raises(DFlashIntegrationError) as exc:
        run_dflash_generation(
            session=session, transport=None,
            prompt_token_ids=[1], max_tokens=1,
        )
    assert exc.value.code == "transport_required"


def test_run_dflash_generation_pushes_telemetry_during_run():
    """Verify-block latency, draft latency, and acceptance EMA all
    get populated by the time run_dflash_generation returns."""
    from coordinator.dflash_draft import DFlashConfig, MockDFlashDrafter
    from coordinator.dflash_integration import (
        InProcessRingVerifyTransport,
        run_dflash_generation,
        setup_dflash_session,
    )
    from coordinator.dflash_telemetry import get_telemetry, reset_telemetry
    from coordinator.head_sampler import select_accepted_prefix
    from coordinator.swarm_events import InMemorySwarmEventBus

    reset_telemetry()
    bus = InMemorySwarmEventBus()
    session = setup_dflash_session(
        config=_MiniCfg(), head_sampler=_FakeHeadSampler(),
        bus=bus, local_peer_id="me",
    )
    session.drafter = MockDFlashDrafter(
        DFlashConfig(backend="mock", block_size=16),
    )

    def _t(prefix, drafts):
        return list(drafts) + [42]

    transport = InProcessRingVerifyTransport(
        run_target_block=_t, telemetry=session.telemetry,
    )

    def _v(h, d):
        return select_accepted_prefix(
            argmax_per_position=list(h), draft_token_ids=list(d),
        )
    session.verifier = _v

    run_dflash_generation(
        session=session, transport=transport,
        prompt_token_ids=[1], max_tokens=17,
    )

    snap = get_telemetry().snapshot()
    assert snap.target_verify_block_p50_ms is not None
    assert snap.draft_inflight_p50_ms is not None
    assert snap.ring_acceptance_rate_ema is not None
