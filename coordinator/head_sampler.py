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
