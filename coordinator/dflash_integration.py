# Copyright 2026 OpenHydra contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Phase 2b live-bench Item 3 — DFlash integration into the inference service.

Wires the components shipped in Phase 2b's earlier commits
(``DFlashTopologyADriver`` + ``HeadSampler.verify_block`` +
``DraftModelRegistry`` + ``FailoverManager`` + ``DFlashTelemetry``)
into the coordinator's generation API. Two main exports:

* ``setup_dflash_session()`` — given an ``EngineConfig`` and a
  ``HeadSampler``, instantiate the drafter, registry, failover
  manager, and telemetry hooks. Returns a ``DFlashSession``
  bundle that the inference service holds for the duration of
  the generation.

* ``run_dflash_generation()`` — drive the
  ``DFlashTopologyADriver`` against an injected
  ``RingVerifyTransport`` and stream emitted tokens to a queue
  the existing HTTP streaming layer drains. Mirrors the contract
  of the existing ring path so the surrounding code in
  ``inference_service.py::infer`` stays minimal.

Two transport implementations:

* ``InProcessRingVerifyTransport`` — drafter, target, and
  verifier all in one process. Runs the verify forward over the
  block locally (no libp2p, no peers) and returns the hidden
  states. Useful for single-Mac demos, integration tests, and
  any deployment where the operator wants to skip the ring.

* ``MultiPeerRingVerifyTransport`` — interface stub. Concrete
  libp2p ring transport lands in a follow-up alongside the
  PushResult routing changes that distinguish block-verify from
  single-token responses. The stub raises ``NotImplementedError``
  with an actionable error so deployments that try to use it
  without the follow-up code know where the work is.

Eligibility gate: ``dflash_eligible(config) → bool`` checks the
config has all required fields. The inference service uses this
to branch ``infer()`` into the DFlash path; ``draft_location ==
'off'`` always returns False.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "DFlashSession",
    "InProcessRingVerifyTransport",
    "MultiPeerRingVerifyTransport",
    "dflash_eligible",
    "run_dflash_generation",
    "setup_dflash_session",
    "DFlashIntegrationError",
]


class DFlashIntegrationError(RuntimeError):
    """Raised when the DFlash setup cannot proceed.

    Carries ``code``:
      * ``"draft_off"``           — caller invoked DFlash setup
                                    when --draft-location=off.
      * ``"missing_drafter"``     — no draft model path configured.
      * ``"missing_head"``        — no HeadSampler registered (Path A
                                    not active).
      * ``"transport_required"``  — driver invoked without a
                                    transport.
      * ``"multipeer_unsupported"`` — MultiPeerRingVerifyTransport
                                      called before the libp2p ring
                                      transport lands.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


# ── Eligibility ────────────────────────────────────────────────────────


def dflash_eligible(config: Any) -> bool:
    """Return True iff the engine config selects DFlash speculation.

    Specifically: ``draft_location`` is ``"local"`` or ``"stage-0"``
    AND ``draft_model_path`` is non-empty.

    The inference service uses this to short-circuit into the
    DFlash branch; when False, falls through to the existing ring
    + autoregressive path unchanged.
    """
    location = getattr(config, "draft_location", "off") or "off"
    if location == "off":
        return False
    if location not in ("local", "stage-0"):
        logger.warning(
            "dflash_eligible: unrecognised draft_location=%r; treating "
            "as 'off' for safety", location,
        )
        return False
    path = str(getattr(config, "draft_model_path", "") or "").strip()
    if not path:
        logger.warning(
            "dflash_eligible: draft_location=%r requires a non-empty "
            "draft_model_path; falling back to 'off'", location,
        )
        return False
    return True


# ── Session bundle ────────────────────────────────────────────────────


@dataclass
class DFlashSession:
    """Bundle of DFlash components the inference service holds for
    one generation.

    Built once per generation (or once per process for the registry
    and failover manager — those are durable across requests).
    """

    drafter: Any                                  # DFlashDrafter
    verifier: Callable[[Any, list[int]], tuple[int, int]]
    registry: Any                                 # DraftModelRegistry
    failover: Optional[Any] = None                # FailoverManager
    telemetry: Optional[Any] = None               # DFlashTelemetry
    block_size: int = 16
    backend: str = "mlx"


def setup_dflash_session(
    *,
    config: Any,
    head_sampler: Any,
    bus: Any,
    local_peer_id: str,
) -> DFlashSession:
    """Construct the DFlash session bundle.

    Args:
        config: ``EngineConfig`` with the Phase 2b fields populated.
        head_sampler: A registered ``HeadSampler`` (Path A) — required
            because verify_block routes through it.
        bus: A ``SwarmEventBus`` (in-memory for tests, libp2p for
            production).
        local_peer_id: This peer's libp2p id; used by FailoverManager
            for promotion bookkeeping.

    Raises:
        DFlashIntegrationError on any precondition violation. Each
        error code points at a specific config or runtime gap.
    """
    if not dflash_eligible(config):
        raise DFlashIntegrationError(
            "draft_off",
            "setup_dflash_session: dflash_eligible(config) is False; "
            "check --draft-location and --draft-model are set",
        )
    if head_sampler is None:
        raise DFlashIntegrationError(
            "missing_head",
            "setup_dflash_session: HeadSampler is None; Phase 2b "
            "requires Path A (sample_on_coordinator) so the coord "
            "owns lm_head + verify_block. Re-launch with "
            "--sample-on-coordinator.",
        )

    from coordinator.dflash_draft import (
        DFlashConfig, load_dflash_drafter,
    )
    from coordinator.dflash_telemetry import get_telemetry
    from coordinator.failover import DraftModelRegistry, FailoverManager
    from coordinator.swarm_events import RegisterDraftModel

    backend = "mlx"   # default; CUDA peers override at construction time
    target_path = str(
        getattr(config, "default_target_model", "")
        or getattr(config, "default_model", "Qwen/Qwen3.5-4B")
    )
    draft_path = str(getattr(config, "draft_model_path", "") or "")
    block_size = max(1, min(32, int(getattr(config, "draft_block_size", 16) or 16)))

    drafter_cfg = DFlashConfig(
        target_model_path=target_path,
        draft_model_path=draft_path,
        block_size=block_size,
        backend=backend,
    )
    drafter = load_dflash_drafter(drafter_cfg)

    # Verifier: bind HeadSampler.verify_block as a callable matching
    # the BlockVerifier protocol.
    from coordinator.head_sampler import DecodeConfig

    decode_cfg = DecodeConfig(
        do_sample=False,
        temperature=0.0,
    )

    def _verifier(hidden_states_block, draft_token_ids):
        return head_sampler.verify_block(
            hidden_states_block=hidden_states_block,
            draft_token_ids=list(draft_token_ids),
            decode=decode_cfg,
        )

    # Registry + failover.
    registry = DraftModelRegistry(bus)
    spec = RegisterDraftModel(
        target_path=target_path,
        draft_path=draft_path,
        block_size=block_size,
        backend=backend,
    )
    registry.announce(spec, from_peer=local_peer_id)
    failover = FailoverManager(
        bus=bus,
        local_peer_id=local_peer_id,
        registry=registry,
    )

    telemetry = get_telemetry()
    if hasattr(drafter, "memory_mb"):
        try:
            telemetry.record_draft_ram_mb(int(drafter.memory_mb()))
        except Exception:
            pass

    logger.info(
        "dflash_session_setup: target=%s draft=%s block=%d backend=%s "
        "local_peer=%s",
        target_path, draft_path, block_size, backend, local_peer_id,
    )

    return DFlashSession(
        drafter=drafter,
        verifier=_verifier,
        registry=registry,
        failover=failover,
        telemetry=telemetry,
        block_size=block_size,
        backend=backend,
    )


# ── Transports ─────────────────────────────────────────────────────────


class InProcessRingVerifyTransport:
    """Single-process verify transport.

    Used when the operator runs DFlash on a single Mac (no peers)
    or for integration tests that don't want to stand up a libp2p
    ring. The transport invokes a caller-provided
    ``run_target_block(prompt_ids, draft_ids) → hidden_states_block``
    callable to do the actual target forward — typically wraps the
    local PyTorch / MLX runtime's prefill+block-decode path.

    Args:
        run_target_block: Callable that runs the target model's
            forward over (prefix + draft_block) and returns the
            ``[block_size + 1, hidden_size]`` hidden states. Called
            once per ``verify()`` invocation.
        telemetry: Optional ``DFlashTelemetry`` instance — receives
            ``record_verify_block_ms`` per call.
    """

    def __init__(
        self,
        *,
        run_target_block: Callable[[list[int], list[int]], Any],
        telemetry: Optional[Any] = None,
    ):
        if not callable(run_target_block):
            raise ValueError("run_target_block must be callable")
        self._run_target = run_target_block
        self._telemetry = telemetry

    def verify(
        self,
        *,
        prefix_token_ids: list[int],
        draft_token_ids: list[int],
        kv_rollback_to: int,
        request_id: str,
        kv_session_id: str,
    ) -> Any:
        """Run the target forward over (prefix + drafts) locally and
        return the hidden states.

        ``kv_rollback_to`` is honoured by ``run_target_block`` —
        this transport is single-process, so the same runtime that
        records the recurrent tape during the previous block also
        applies the rollback for the current block. No race window.
        """
        t0 = time.monotonic()
        out = self._run_target(list(prefix_token_ids), list(draft_token_ids))
        ms = (time.monotonic() - t0) * 1000.0
        if self._telemetry is not None:
            try:
                self._telemetry.record_verify_block_ms(ms)
            except Exception:
                pass
        logger.debug(
            "dflash_inprocess_verify: req=%s session=%s rollback=%d "
            "drafts=%d ms=%.1f",
            request_id, kv_session_id, kv_rollback_to,
            len(draft_token_ids), ms,
        )
        return out


class MultiPeerRingVerifyTransport:
    """Ring-based verify transport (multi-peer libp2p).

    Stub. Full implementation requires:
      1. A coord-side block-verify response queue keyed by
         (request_id, block_index).
      2. PushResult handler in peer/server.py distinguishing
         block-verify hidden-state responses from single-token ones.
      3. ForwardRequest path that fires the draft block as
         ``prompt_token_ids`` with ``draft_block=True``.

    Lands in a follow-up. Today the constructor is callable so
    deployments can wire the type, but ``verify()`` raises with
    an actionable error.
    """

    def __init__(self, **kwargs: Any):
        self._kwargs = dict(kwargs)

    def verify(self, **_kwargs: Any) -> Any:
        raise DFlashIntegrationError(
            "multipeer_unsupported",
            "MultiPeerRingVerifyTransport.verify: the libp2p ring "
            "transport for block-verify is not yet implemented. Use "
            "InProcessRingVerifyTransport for single-process DFlash "
            "or set --draft-location=off until the multi-peer ring "
            "follow-up lands.",
        )


# ── Driver harness ────────────────────────────────────────────────────


def run_dflash_generation(
    *,
    session: DFlashSession,
    transport: Any,
    prompt_token_ids: list[int],
    max_tokens: int,
    stop_token_ids: frozenset = frozenset(),
    request_id: str = "",
    kv_session_id: str = "",
    on_token: Optional[Callable[[int], None]] = None,
) -> dict:
    """Drive ``DFlashTopologyADriver`` end-to-end.

    Each emitted token is delivered via ``on_token`` (default: no-op).
    Returns a stats dict matching the existing ring-completion log
    shape so the inference service's logging code can be reused
    without changes.

    Telemetry: per-block draft latency and acceptance rate are
    pushed into the singleton.
    """
    if transport is None:
        raise DFlashIntegrationError(
            "transport_required",
            "run_dflash_generation: transport is None; pass an "
            "InProcessRingVerifyTransport (single-process) or a "
            "MultiPeerRingVerifyTransport (multi-peer libp2p)",
        )

    from coordinator.dflash_driver import DFlashTopologyADriver

    emitted: list[int] = []

    def _emit(tok: int) -> None:
        emitted.append(int(tok))
        if on_token is not None:
            try:
                on_token(int(tok))
            except Exception:
                logger.exception("dflash_on_token_callback_failed")

    # Wrap drafter to push per-call latency into telemetry.
    base_drafter = session.drafter
    telemetry = session.telemetry

    class _InstrumentedDrafter:
        def draft(self, prefix_token_ids: list[int]) -> list[int]:
            t0 = time.monotonic()
            out = base_drafter.draft(prefix_token_ids)
            ms = (time.monotonic() - t0) * 1000.0
            if telemetry is not None:
                try:
                    telemetry.record_draft_inflight_ms(ms)
                except Exception:
                    pass
            return out

    # Wrap verifier to record acceptance.
    base_verifier = session.verifier

    def _instrumented_verifier(hidden_states_block, draft_token_ids):
        accepted, bonus = base_verifier(hidden_states_block, draft_token_ids)
        if telemetry is not None:
            try:
                telemetry.record_block_acceptance(
                    int(accepted), len(draft_token_ids),
                )
            except Exception:
                pass
        return accepted, bonus

    driver = DFlashTopologyADriver(
        drafter=_InstrumentedDrafter(),
        verifier=_instrumented_verifier,
        transport=transport,
        emit=_emit,
        block_size=session.block_size,
        max_tokens=max_tokens,
        stop_token_ids=stop_token_ids,
        request_id=request_id,
        kv_session_id=kv_session_id,
    )

    t0 = time.monotonic()
    stats = driver.run(prompt_token_ids=prompt_token_ids)
    total_ms = (time.monotonic() - t0) * 1000.0

    logger.info(
        "dflash_generation_done: req=%s tokens=%d blocks=%d "
        "acceptance=%.3f draft_ms=%.1f verify_ms=%.1f sampler_ms=%.1f "
        "total_ms=%.1f",
        request_id, stats.tokens_emitted, stats.blocks,
        stats.acceptance_rate, stats.draft_ms_total,
        stats.verify_ms_total, stats.sampler_ms_total, total_ms,
    )

    return {
        "tokens": list(emitted),
        "blocks": stats.blocks,
        "tokens_emitted": stats.tokens_emitted,
        "acceptance_rate": stats.acceptance_rate,
        "avg_block_size_emitted": stats.avg_block_size_emitted,
        "draft_ms_total": stats.draft_ms_total,
        "verify_ms_total": stats.verify_ms_total,
        "sampler_ms_total": stats.sampler_ms_total,
        "total_ms": total_ms,
    }
