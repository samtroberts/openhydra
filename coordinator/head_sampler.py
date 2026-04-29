# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Coordinator-side final head + sampler.

Phase 1 of the client-terminated pipeline refactor (Path A, flag-gated).

When ``ForwardRequest.sample_on_coordinator`` is true, the last peer skips
``final_norm`` + ``lm_head`` + sampling and returns the raw post-last-layer
hidden state via the existing ``PushResult`` RPC with
``ForwardResponse.is_hidden_state=True``. This module owns the coordinator
side of that contract: it borrows the head weights from a co-located peer
(zero-copy, avoids duplicating the ``vocab × hidden`` embedding in unified
memory on Mac) or, as a future extension, loads its own copy when the
coordinator runs on a different process than any peer.

Phase 1 scope is intentionally narrow — this module defines the registry
and the borrow-or-load protocol. The actual ``sample(hidden_state)`` call,
token re-injection, and runtime hooks land in Phase 2 + Phase 3. Keeping
the surface small now means the PushResult branch in ``peer/server.py``
can already dispatch here without the sampling implementation being ready.

Design contract
---------------
- ``register_head_source(peer_id, runtime)`` — called by ``peer/server.py``
  at startup when a peer loads its shard. If ``runtime`` holds the last
  shard (``final_norm`` + ``lm_head`` / tied embeddings), its weights are
  now reachable from the coordinator-owned ``HeadSampler`` via a single
  process-local reference. No copy, no IPC.
- ``get_head_sampler()`` — returns the active ``HeadSampler`` or ``None``
  if no last-shard peer is co-located. The ``PushResult`` handler uses
  this to decide whether ``is_hidden_state=True`` responses can be
  sampled in-process.
- ``HeadSampler.sample(hidden_state_packed, decode_cfg)`` — Phase 2.
  Will dispatch to the borrowed runtime's final_norm + lm_head, apply
  the sampler (greedy/temperature/top_p/top_k), and return ``int``
  token id. Today it raises ``NotImplementedError`` so any caller that
  skips the flag check will fail loudly rather than silently sample zeros.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class DecodeConfig:
    """Sampler parameters carried on each ring-step request.

    Mirrors the ``decode_*`` fields of ``ForwardRequest``.
    """

    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 0.0
    top_k: int = 0
    seed: Optional[int] = None


class HeadSampler:
    """Applies ``final_norm`` + ``lm_head`` + sampler on the coordinator.

    Phase 1: skeleton. Holds a reference to a co-located peer's last-shard
    runtime so Phase 2 can call into its head weights without copying.

    The ``runtime`` duck-typed object is expected (Phase 2) to expose:
      * ``apply_final_head(hidden_state) -> logits`` — runs ``final_norm``
        followed by ``lm_head`` (or tied-embed linear) on the hidden state.
      * ``_tie_word_embeddings`` — bool (already on MLX runtime).
      * ``_is_last_shard`` — bool (already on MLX runtime).

    For PyTorch, the equivalent attributes are on ``_DecoderArchitecture``
    via the ``PyTorchRuntime._model`` handle. Phase 2 adds ``apply_final_head``
    to both runtimes.
    """

    def __init__(self, runtime: Any, peer_id: str) -> None:
        self._runtime = runtime
        self._peer_id = peer_id
        self._lock = threading.Lock()

    @property
    def peer_id(self) -> str:
        return self._peer_id

    @property
    def runtime(self) -> Any:
        return self._runtime

    def sample(
        self,
        hidden_state: Any,
        decode: DecodeConfig,
        *,
        packed_bytes: bytes | None = None,
    ) -> int:
        """Apply final head + sampler to a hidden state. Returns token id.

        Thin wrapper over the borrowed runtime's ``apply_final_head``
        method (implemented in Phase 2 on both MLX and PyTorch runtimes).
        Takes a process-local lock so concurrent ring steps don't collide
        on MLX's lazy graph — MLX ``mx.eval`` is thread-safe, but
        ``make_sampler`` / ``stream_generate`` globals can race on older
        mlx-lm versions. Cheap lock, no measured contention.
        """
        apply = getattr(self._runtime, "apply_final_head", None)
        if not callable(apply):
            raise RuntimeError(
                "head_sampler: borrowed runtime lacks apply_final_head; "
                "this runtime predates the client-terminated-pipeline "
                "refactor (Phase 2 of Path A)."
            )
        with self._lock:
            return int(apply(
                hidden_state,
                packed_bytes=packed_bytes,
                decode_do_sample=decode.do_sample,
                decode_temperature=decode.temperature,
                decode_top_p=decode.top_p,
                decode_top_k=decode.top_k,
                decode_seed=decode.seed,
            ))

    def verify_block(
        self,
        hidden_states_block: Any,
        draft_token_ids: list[int],
        decode: DecodeConfig,
        *,
        packed_bytes: bytes | None = None,
    ) -> tuple[int, int]:
        """Phase 2b — DFlash block-verify.

        Given the target's hidden states for ``len(draft_token_ids) + 1``
        positions (the +1 is the next-token position past the last
        draft, used to sample the bonus token) and the corresponding
        draft tokens, return ``(accepted_len, bonus_token)``:

        * ``accepted_len`` — the longest prefix where the target's
          greedy argmax matches the draft. Range [0, len(drafts)].
        * ``bonus_token`` — sampled from the target's logits at
          position ``accepted_len``. Always emitted, even on full
          rejection (then it's the target's argmax at position 0).

        The total tokens emitted per verify pass is therefore always
        ``accepted_len + 1`` — that's the per-block speedup factor
        (1 if the draft is fully wrong, up to ``block_size + 1`` when
        every draft matches).

        Lossless guarantee: under ``decode.temperature == 0`` every
        emitted token is the target's greedy argmax — byte-identical
        to non-speculative greedy decoding. Under ``temperature > 0``
        we accept while greedy-equivalent and sample the bonus from
        the target's distribution at position ``accepted_len``; output
        is sampling-equivalent to single-token decoding within the
        same temperature schedule.

        Args:
            hidden_states_block: Backend-tensor of shape
                ``[len(draft_token_ids) + 1, hidden_size]``. Caller
                guarantees fp16/fp32 contiguous layout the runtime
                expects — the runtime's
                ``apply_final_head_block(hidden_states_block) ->
                List[int]`` returns one argmax per position. Wired in
                Commit 9 (the Topology A driver loop).
            draft_token_ids: The candidate tokens the drafter emitted.
                Length == ``EngineConfig.draft_block_size``, default 16.
            decode: Decode config. Phase 2b only fully supports
                ``temperature == 0`` (lossless); ``temperature > 0``
                accepts on greedy-equivalence and samples the bonus
                from the target's distribution.
            packed_bytes: Optional pre-serialised hidden states, same
                contract as ``sample()`` — caller provides this for
                MLX runtimes that already produced the packed wire
                form during ring assembly.

        Returns:
            ``(accepted_len, bonus_token)`` — both ints, both safe to
            wire back to the swarm via VerifyResult / kv_rollback_to.

        Raises:
            RuntimeError: If the borrowed runtime lacks
            ``apply_final_head_block`` (runtime predates Phase 2b).
            ValueError: If ``draft_token_ids`` is empty.
        """
        if not draft_token_ids:
            raise ValueError("verify_block requires non-empty draft_token_ids")

        apply_block = getattr(self._runtime, "apply_final_head_block", None)
        if not callable(apply_block):
            raise RuntimeError(
                "head_sampler: borrowed runtime lacks apply_final_head_block; "
                "this runtime predates Phase 2b. Update the peer to a build "
                "that ships the block-decode path (Phase 2b Commit 7)."
            )

        with self._lock:
            argmax_per_position: list[int] = list(apply_block(
                hidden_states_block,
                packed_bytes=packed_bytes,
                decode_do_sample=decode.do_sample,
                decode_temperature=decode.temperature,
                decode_top_p=decode.top_p,
                decode_top_k=decode.top_k,
                decode_seed=decode.seed,
            ))

        accepted_len, bonus_token = select_accepted_prefix(
            argmax_per_position=argmax_per_position,
            draft_token_ids=list(draft_token_ids),
        )
        return int(accepted_len), int(bonus_token)


# ── Phase 2b pure algorithm — testable without a runtime ────────────────


def select_accepted_prefix(
    *,
    argmax_per_position: list[int],
    draft_token_ids: list[int],
) -> tuple[int, int]:
    """Compute (accepted_len, bonus_token) from argmax + draft sequences.

    Pure integer arithmetic — no tensors, no runtime dispatch. Lives
    here so ``HeadSampler.verify_block`` is just a thin orchestrator
    around it, AND so the algorithm gets unit-tested without pulling
    any backend.

    Contract:
        * ``argmax_per_position`` must have length ``len(draft_token_ids) + 1``.
          Index ``i`` for ``i < len(drafts)`` is the target's prediction
          for what should appear at position ``i`` — i.e., the value
          ``draft_token_ids[i]`` is being checked against. Index
          ``len(drafts)`` is the bonus-token position.
        * Returns ``accepted_len`` in ``[0, len(drafts)]`` — exclusive
          upper bound matches the "longest matching prefix length"
          convention.
        * Returns ``bonus_token`` = ``argmax_per_position[accepted_len]``.
          Total emitted = ``accepted_len + 1`` tokens.

    Lossless under temp=0: every emitted token is exactly what the
    target model's greedy argmax would have produced step-by-step
    in single-token decoding. The block-verify path is mathematically
    equivalent to autoregressive greedy decoding when each draft
    matches the argmax.

    Examples:
        Full acceptance (all 16 drafts match) — returns
        ``(16, argmax[16])``, emitting 17 tokens.

        First-position rejection — returns ``(0, argmax[0])``,
        emitting just 1 token (the target's argmax at pos 0). Note
        the rejected drafts are NOT emitted; the caller must roll
        back peer KV state by exactly ``len(drafts) - accepted_len``
        positions before the next forward.
    """
    n_drafts = len(draft_token_ids)
    if n_drafts <= 0:
        raise ValueError("draft_token_ids must be non-empty")
    if len(argmax_per_position) != n_drafts + 1:
        raise ValueError(
            f"argmax_per_position must have length {n_drafts + 1} "
            f"(== len(drafts)+1 for the bonus position); "
            f"got {len(argmax_per_position)}"
        )

    # Walk the draft, accept while the target agrees position-by-position.
    accepted_len = 0
    while accepted_len < n_drafts and (
        int(argmax_per_position[accepted_len]) == int(draft_token_ids[accepted_len])
    ):
        accepted_len += 1

    bonus_token = int(argmax_per_position[accepted_len])
    return accepted_len, bonus_token


# ── Process-local registry ──────────────────────────────────────────────
#
# The coordinator and its co-located peer live in the same process (the
# common OpenHydra deployment). The last-shard peer registers its runtime
# here at startup; the PushResult handler looks it up when routing
# ``is_hidden_state=True`` responses.

_ACTIVE_HEAD: Optional[HeadSampler] = None
_ACTIVE_LOCK = threading.Lock()


def register_head_source(peer_id: str, runtime: Any) -> None:
    """Register a co-located peer's runtime as the coordinator's head source.

    Idempotent: the last caller wins (handles reload_shard cases where
    the runtime object is swapped).

    Only peers whose shard includes the last transformer layer should
    call this — callers are expected to check ``runtime._is_last_shard``
    (MLX) or the PyTorch equivalent before registering.
    """
    global _ACTIVE_HEAD
    with _ACTIVE_LOCK:
        _ACTIVE_HEAD = HeadSampler(runtime=runtime, peer_id=str(peer_id))
    logger.info(
        "head_sampler_registered: peer=%s runtime=%s",
        peer_id, type(runtime).__name__,
    )


def unregister_head_source(peer_id: str) -> None:
    """Drop the active head source if it matches ``peer_id``."""
    global _ACTIVE_HEAD
    with _ACTIVE_LOCK:
        if _ACTIVE_HEAD is not None and _ACTIVE_HEAD.peer_id == str(peer_id):
            _ACTIVE_HEAD = None
            logger.info("head_sampler_unregistered: peer=%s", peer_id)


def get_head_sampler() -> Optional[HeadSampler]:
    """Return the active coordinator-side head sampler, or ``None``.

    ``None`` means no co-located last-shard peer is available; the
    ``PushResult`` handler must fall back to treating the response as
    a sampled-token payload (today's ring-on-peer behaviour).
    """
    with _ACTIVE_LOCK:
        return _ACTIVE_HEAD


def clear_head_source() -> None:
    """Test helper — wipe the registry between tests."""
    global _ACTIVE_HEAD
    with _ACTIVE_LOCK:
        _ACTIVE_HEAD = None


# ── Ring session state ──────────────────────────────────────────────────
#
# Phase 3 of the client-terminated pipeline refactor. When
# ``sample_on_coordinator=True``, the coordinator — not the last peer —
# owns the ring loop. It must remember the per-request ring state
# (remaining tokens, EOS set, decode config, full route) across the
# peer0 → peerN-1 → coordinator → peer0 cycle. We stash that state here
# at ring-launch time (``coordinator/chain.py::run_push_ring``) and look
# it up in the ``PushResult`` handler when the last peer returns a
# hidden state.
#
# The session is keyed by ``request_id`` — the same id the last peer
# carries in ``final_callback_request_id`` via the PushResult envelope.


# ── Phase 2a: per-slot state ────────────────────────────────────────────


# Slot lifecycle states. The compound transition (state mutation +
# in-flight count check + next-slot reservation) MUST run under
# ``RingSession.lock`` — see the docstring on ``RingSession.lock``.
SLOT_STATE_DISPATCHED = "dispatched"        # coord fired, peer-0 not yet started
SLOT_STATE_IN_FLIGHT_P0 = "in_flight_p0"     # peer-0 computing
SLOT_STATE_IN_FLIGHT_P1 = "in_flight_p1"     # peer-1 computing
SLOT_STATE_AWAITING_SAMPLE = "awaiting_sample"  # PushResult arrived, sampler running
SLOT_STATE_SAMPLED = "sampled"               # token id assigned, slot finalised
SLOT_STATE_COMMITTED = "committed"           # reserved for Phase 2b speculation
SLOT_STATE_ABORTED = "aborted"               # reserved for Phase 2b rollback

# Set of "in-flight" states for the depth-throttle check inside
# _coordinator_handle_push_result. A slot counts toward the in-flight
# budget while it's anywhere in the pipeline before final sampling.
SLOT_STATES_IN_FLIGHT = frozenset({
    SLOT_STATE_DISPATCHED,
    SLOT_STATE_IN_FLIGHT_P0,
    SLOT_STATE_IN_FLIGHT_P1,
    SLOT_STATE_AWAITING_SAMPLE,
})

# Set of "finalised" states — the early-return idempotency check uses
# this to drop late-arriving duplicate PushResults silently.
SLOT_STATES_FINAL = frozenset({
    SLOT_STATE_SAMPLED,
    SLOT_STATE_COMMITTED,
    SLOT_STATE_ABORTED,
})


@dataclass
class SlotState:
    """One per-token slot in a pipelined ring.

    Created when the coord fires a ForwardRequest with ``slot_id=N``;
    transitions through ``dispatched → in_flight_p0 → in_flight_p1 →
    awaiting_sample → sampled`` as the request progresses through the
    ring and the PushResult comes back. ``committed`` and ``aborted``
    are reserved for Phase 2b's draft-and-verify path.

    All field mutations MUST run under the parent ``RingSession.lock``.
    """

    slot_id: int
    state: str = SLOT_STATE_DISPATCHED
    dispatched_at_ms: float = 0.0
    last_update_ms: float = 0.0
    token_id: Optional[int] = None        # set after sampling
    # Reserved for Phase 2b — the draft token stage-0 predicted before
    # peer-1 returned its hidden state. ``None`` in Phase 2a.
    draft_token_id: Optional[int] = None


@dataclass
class RingSession:
    """Per-request ring state owned by the coordinator (Path A).

    Everything needed to (a) sample the returned hidden state with the
    same decode config the initial request specified, and (b) re-inject
    the sampled token into stage 0 for the next ring step.
    """

    request_id: str
    # Routing (so re-injection hits stage 0 regardless of NAT topology).
    ring_first_hop_address: str = ""
    ring_first_hop_peer_id: str = ""
    ring_first_hop_libp2p_id: str = ""
    ring_full_route: list = field(default_factory=list)  # list[PeerHop]
    next_hop_address: str = ""
    next_hop_peer_id: str = ""
    # Termination controls.
    ring_tokens_remaining: int = 0
    ring_eos_ids: set = field(default_factory=set)
    ring_generated_ids: list = field(default_factory=list)
    # Decode config (carried verbatim from the initial ForwardRequest).
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    # KV + session metadata.
    kv_session_id: str = ""
    total_stages: int = 1
    # Callback so the last peer knows to route hidden states here, and
    # so the ring-queue `emit_ring_token` binds to the right cb_id.
    final_callback_address: str = ""
    final_callback_libp2p_peer_id: str = ""
    callback_request_id: str = ""
    # Layer ranges for stage-0 re-injection.
    stage0_layer_start: int = 0
    stage0_layer_end: int = 0
    stage0_total_layers: int = 0
    # ── Phase 2a: pipelined ring state ─────────────────────────────
    # ``pipeline_depth`` of 1 (default) preserves today's serial ring
    # exactly — the slots dict stays empty, the lock is uncontended,
    # every code site short-circuits to the legacy path. 2+ enables
    # multiple in-flight tokens; the coord-side worker pool calls
    # ``_coordinator_handle_push_result`` from N concurrent threads
    # which hold ``lock`` for the compound state-transition + reinject-
    # decision op (see plan section 6).
    pipeline_depth: int = 1
    slots: dict[int, SlotState] = field(default_factory=dict)
    next_slot_id: int = 0
    # Per-session mutex protecting the COMPOUND ops on slots /
    # next_slot_id / ring_tokens_remaining / ring_generated_ids.
    # Granularity rationale: per-session, not per-slot (the in-flight
    # count iterates over all slots, so per-slot locks cannot make the
    # read-decide-write atomic) and not global (would over-serialise
    # independent requests on different request_ids).
    #
    # Lock-ordering discipline (deadlock prevention):
    #     NEVER acquire ``_RING_SESSION_LOCK`` while holding
    #     ``session.lock``. The reverse is fine — see the helpers
    #     below which always release ``_RING_SESSION_LOCK`` before
    #     returning the session object to a caller that may then
    #     acquire ``session.lock``.
    #
    # ``threading.Lock`` is not pickle-safe but ``RingSession`` is
    # process-local (lives in the in-memory ``_RING_SESSIONS`` dict),
    # so this is fine. ``compare=False`` keeps it out of dataclass
    # equality / repr — locks aren't comparable.
    lock: threading.Lock = field(
        default_factory=threading.Lock,
        compare=False,
        repr=False,
    )


_RING_SESSIONS: dict[str, RingSession] = {}
_RING_SESSION_LOCK = threading.Lock()


def register_ring_session(session: RingSession) -> None:
    """Store a ring session at the coordinator's ring-launch time.

    Called once per user request from ``coordinator/chain.py::run_push_ring``
    right before the initial ForwardRequest fires. Keyed by
    ``session.request_id``; re-registering with the same id overwrites —
    useful for test harnesses.
    """
    with _RING_SESSION_LOCK:
        _RING_SESSIONS[session.request_id] = session
    logger.info(
        "ring_session_registered: req=%s remaining=%d stages=%d",
        session.request_id, session.ring_tokens_remaining, session.total_stages,
    )


def get_ring_session(request_id: str) -> Optional[RingSession]:
    """Look up an active ring session. Returns ``None`` if absent."""
    with _RING_SESSION_LOCK:
        return _RING_SESSIONS.get(str(request_id))


def unregister_ring_session(request_id: str) -> None:
    """Drop a ring session — called when ``remaining == 0`` or on EOS."""
    with _RING_SESSION_LOCK:
        popped = _RING_SESSIONS.pop(str(request_id), None)
    if popped is not None:
        logger.info(
            "ring_session_unregistered: req=%s remaining=%d generated=%d",
            request_id, popped.ring_tokens_remaining, len(popped.ring_generated_ids),
        )


def clear_ring_sessions() -> None:
    """Test helper — wipe all ring sessions."""
    with _RING_SESSION_LOCK:
        _RING_SESSIONS.clear()
