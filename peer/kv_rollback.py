# Copyright 2026 OpenHydra contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Phase 2b — KV cache rollback strategies (per-layer-type).

Phase 2b §5 (the seven sharpenings — point 3): rollback strategy is
dispatched **per-layer-type**, NOT per-backend. Qwen3.5 is hybrid —
a single shard owns a mix of standard self-attention layers and
GatedDeltaNet (Mamba-style linear-attention) layers — so the
"all attention" or "all Mamba" assumption a per-backend dispatch
would bake in is wrong from the start.

Two strategies, both backend-agnostic:

* ``TruncateAttentionRollback`` — for standard self-attention KV.
  Truncates the K/V tensors' sequence axis to ``target_len``. O(1):
  it's a slice. Works on any tensor library that supports basic
  indexing (PyTorch, MLX, even raw numpy in tests).

* ``TapeReplayRecurrentRollback`` — for GatedDeltaNet/Mamba layers.
  The recurrent state is NOT sliceable — it's been mixed by every
  preceding token. Strategy:
    1. During the verify forward, record an "innovation tape" — the
       per-step state delta the layer applied for each position.
    2. On rollback to ``target_len < N``, start from the
       ``target_len``-th tape entry's *pre-state* and replay only
       those innovations to reconstruct the post-rollback state.
    Equivalent to "scan-with-checkpoint." Cost is O(K) where K =
    accepted_len, ≪ 16 in practice. Reference: dflash-mlx's
    ``RecurrentRollbackCache`` (Apache 2.0-compatible MIT).

The dispatcher ``rollback_session_kv`` walks the cached layer
states for one session and invokes the appropriate strategy per
layer.

This module is **pure rollback semantics** — it does not run forward
passes, does not allocate tensors, does not interact with the gRPC
layer. Phase 2b Commit 8 wires the dispatcher into the runtime
forward path so ``ForwardRequest.kv_rollback_to`` is honoured INLINE
before the next forward starts (race-free against the next
``ForwardRequest``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "KVKind",
    "RollbackStrategy",
    "TruncateAttentionRollback",
    "TapeReplayRecurrentRollback",
    "LayerRollbackEntry",
    "SessionRollbackPlan",
    "classify_layer_kind",
    "rollback_session_kv",
    "RollbackError",
]


class KVKind:
    """String constants for layer KV-cache classification.

    Not an Enum so it serialises cleanly through the JSON
    swarm-event channel without needing a custom encoder.
    """

    ATTENTION = "attention"           # standard self-attention K/V
    RECURRENT = "recurrent"           # GatedDeltaNet / Mamba state
    NONE = "none"                     # layer holds no KV (norms, lm_head)


# ── Layer classification ────────────────────────────────────────────────


def classify_layer_kind(layer: Any) -> str:
    """Return the KVKind constant for a transformer layer instance.

    Inspection rules, in order:
        1. Explicit ``kv_kind`` attribute on the layer (forward-compat
           hook — runtimes that build their own layers can stamp this
           and skip the heuristic).
        2. Class-name substring match: "GatedDeltaNet" or "Mamba" →
           recurrent; "Attention" → attention.
        3. Fallback: ATTENTION (matches the dominant Qwen3.5 layer
           type and is the safer-but-slower default — truncate is
           cheap and never wrong, it just doesn't compose with the
           recurrent state for Mamba layers).

    The fallback is *deliberately* attention rather than NONE: a
    layer we can't identify but that does carry KV must still get
    rolled back, otherwise the next verify pass sees stale state.
    """
    explicit = getattr(layer, "kv_kind", None)
    if isinstance(explicit, str) and explicit in {
        KVKind.ATTENTION, KVKind.RECURRENT, KVKind.NONE,
    }:
        return explicit

    name = type(layer).__name__
    lower = name.lower()
    if "gateddeltanet" in lower or "mamba" in lower:
        return KVKind.RECURRENT
    if "attention" in lower:
        return KVKind.ATTENTION
    # Safer default: assume attention. Truncate is O(1) and cannot
    # produce wrong output (worst case it's a no-op when the layer
    # holds no KV), whereas tape-replay on a non-recurrent layer
    # would panic on missing-tape lookups.
    return KVKind.ATTENTION


# ── Strategy ABC + implementations ──────────────────────────────────────


class RollbackError(RuntimeError):
    """Raised when rollback fails (e.g. tape-replay on a layer with
    no tape recorded). Carries a ``layer_index`` for diagnostics."""

    def __init__(self, message: str, *, layer_index: Optional[int] = None):
        super().__init__(message)
        self.layer_index = layer_index


class RollbackStrategy(ABC):
    """ABC — concrete subclasses implement ``rollback_to``."""

    @abstractmethod
    def rollback_to(self, kv_state: Any, target_len: int) -> Any:
        """Return the KV state truncated to ``target_len`` entries.

        Args:
            kv_state: Backend-specific KV representation. Strategies
                are duck-typed against the runtime's KV format —
                attention truncate works on anything sliceable;
                tape-replay needs an "innovation tape" attached.
            target_len: Absolute sequence position to roll back TO
                (kept entries: ``[0, target_len)``).

        Returns:
            The new KV state. May be the same object mutated in place
            (attention truncate prefers this for zero-copy) or a new
            object (tape-replay reconstructs from scratch).

        Raises:
            RollbackError on any state-shape inconsistency.
        """


class TruncateAttentionRollback(RollbackStrategy):
    """Slice the sequence axis of K and V to ``target_len``.

    Works on:
        * PyTorch tensors via ``tensor[..., :target_len, :]``.
        * MLX arrays via the same slicing syntax.
        * dict-shaped caches: expects ``{"keys": K, "values": V}``
          with the sequence axis at index ``seq_axis`` (default -2,
          the convention HuggingFace transformers and MLX use).
        * tuple-shaped caches: ``(K, V)``.

    O(1) — the strategy is a slice; no copy, no allocation. If the
    tensor lib materialises lazily (MLX), the slice stays lazy
    until the next forward evaluates it.
    """

    def __init__(self, *, seq_axis: int = -2):
        self._seq_axis = seq_axis

    def rollback_to(self, kv_state: Any, target_len: int) -> Any:
        if target_len < 0:
            raise RollbackError(
                f"target_len must be >= 0; got {target_len}"
            )
        return _slice_kv_along_seq_axis(
            kv_state, target_len, seq_axis=self._seq_axis,
        )


class TapeReplayRecurrentRollback(RollbackStrategy):
    """Replay accepted innovations from a recorded tape.

    Phase 2b's port of dflash-mlx's ``RecurrentRollbackCache`` pattern,
    backend-neutral.

    Expected ``kv_state`` shape:
        ``{"state": <recurrent state>,
           "tape": [innovation_0, innovation_1, ...],
           "pre_states": [pre_state_0, pre_state_1, ...]}``

    where each ``innovation_i`` is the per-step delta the layer
    applied at step ``i`` (after seeing the prefix + drafts[0..i-1]),
    and ``pre_states[i]`` is a snapshot of the recurrent state BEFORE
    step ``i`` ran. ``pre_states[0]`` therefore equals the state at
    the prefix end, before any drafts; ``pre_states[target_len]``
    is the state we want to roll back TO.

    The runtime's verify forward is responsible for populating
    ``tape`` and ``pre_states``. Commit 7 (the block-decode path)
    adds the ``record_innovation`` hook that does this.

    Cost per layer: O(1) — we just read the right pre-state. The
    "replay" name is a holdover from dflash-mlx where the kernel
    actually re-runs the accepted innovations through a fused step;
    here we precomputed the pre-states so the rollback is a lookup,
    not a re-scan. Both are correct; the lookup is simpler.
    """

    def rollback_to(self, kv_state: Any, target_len: int) -> Any:
        if target_len < 0:
            raise RollbackError(
                f"target_len must be >= 0; got {target_len}"
            )
        if not isinstance(kv_state, dict):
            raise RollbackError(
                "TapeReplayRecurrentRollback requires dict-shaped KV "
                f"with 'state', 'tape', 'pre_states'; got "
                f"{type(kv_state).__name__}"
            )
        pre_states = kv_state.get("pre_states")
        if not isinstance(pre_states, list):
            raise RollbackError(
                "TapeReplayRecurrentRollback requires kv_state['pre_states'] "
                "to be a list of per-step pre-state snapshots"
            )
        if target_len > len(pre_states):
            raise RollbackError(
                f"target_len={target_len} exceeds tape length "
                f"{len(pre_states)} — verify pass didn't record enough "
                f"innovations"
            )
        # Reconstruct: the rollback'd state is the pre-state at
        # position target_len, with the tape truncated to match.
        new_state = pre_states[target_len] if target_len < len(pre_states) else kv_state["state"]
        new_tape = list(kv_state.get("tape", []))[:target_len]
        new_pre_states = list(pre_states)[:target_len + 1]
        return {
            "state": new_state,
            "tape": new_tape,
            "pre_states": new_pre_states,
        }


# ── Per-session dispatcher ──────────────────────────────────────────────


@dataclass(frozen=True)
class LayerRollbackEntry:
    """One layer's rollback descriptor.

    Bound up at session-start time so the dispatcher doesn't
    re-classify on every rollback call.
    """

    layer_index: int
    kind: str                                # one of KVKind.*
    strategy: RollbackStrategy


@dataclass
class SessionRollbackPlan:
    """The rollback playbook for a single inference session.

    Built once when the session is created (peers see the first
    ForwardRequest carrying a session_id) and reused for every
    ``kv_rollback_to`` value the coord sends.
    """

    session_id: str
    entries: list[LayerRollbackEntry] = field(default_factory=list)

    def add(self, layer_index: int, kind: str, strategy: RollbackStrategy) -> None:
        self.entries.append(LayerRollbackEntry(
            layer_index=layer_index, kind=kind, strategy=strategy,
        ))


def rollback_session_kv(
    plan: SessionRollbackPlan,
    layer_kv: dict[int, Any],
    target_len: int,
) -> dict[int, Any]:
    """Apply ``plan`` to ``layer_kv``, rolling every layer back to
    ``target_len`` absolute sequence position.

    Args:
        plan: The session's rollback playbook (built once at
            session start).
        layer_kv: ``{layer_index: kv_state}`` mapping. Layers absent
            from the plan are passed through untouched (defence in
            depth; signals a bug but doesn't drop data).
        target_len: Absolute sequence position. ``0`` = clear the
            cache entirely (next forward starts from scratch);
            equal to current length = no-op.

    Returns:
        New ``{layer_index: kv_state}`` dict with the rollback
        applied. Strategies that mutate in place return the same
        object reference; tape-replay returns a fresh dict per layer.
        Either is correct — the caller doesn't rely on identity.

    Raises:
        RollbackError on any single-layer rollback failure. The
        partial state is NOT committed back; callers should treat
        this as session-fatal and tear the KV cache down for that
        session_id.
    """
    if target_len < 0:
        raise RollbackError(
            f"target_len must be >= 0; got {target_len}",
        )

    out: dict[int, Any] = {}
    for entry in plan.entries:
        idx = entry.layer_index
        if idx not in layer_kv:
            # Layer not yet warmed up (e.g., first request after
            # session creation). Skip — strategy has no state to
            # roll back yet.
            continue
        kv_state = layer_kv[idx]
        if entry.kind == KVKind.NONE:
            # Layer carries no KV — pass through.
            out[idx] = kv_state
            continue
        try:
            out[idx] = entry.strategy.rollback_to(kv_state, target_len)
        except RollbackError as exc:
            exc.layer_index = idx
            logger.error(
                "kv_rollback_layer_failed: session=%s layer=%d kind=%s "
                "target_len=%d err=%s",
                plan.session_id, idx, entry.kind, target_len, exc,
            )
            raise

    # Pass through any layers in layer_kv that aren't in the plan
    # (shouldn't happen in normal operation; defence in depth).
    for idx, kv in layer_kv.items():
        if idx not in out:
            out[idx] = kv

    logger.debug(
        "kv_rollback_session: session=%s layers=%d target_len=%d",
        plan.session_id, len(plan.entries), target_len,
    )
    return out


# ── Internal helpers ────────────────────────────────────────────────────


def _slice_kv_along_seq_axis(kv_state: Any, target_len: int, *, seq_axis: int) -> Any:
    """Truncate K/V along the sequence axis. Handles dict, tuple,
    and (k, v) -tuple-of-list shapes used by HF + MLX transformers.

    Supports the four common cache shapes:
        * dict   {"keys": tensor, "values": tensor}   → per-key slice
        * dict   {"k": tensor, "v": tensor}           → alias of above
        * tuple  (keys_tensor, values_tensor)         → tuple slice
        * raw tensor                                  → just slice
    """
    if isinstance(kv_state, dict):
        out = dict(kv_state)
        for key in ("keys", "values", "k", "v"):
            if key in out:
                out[key] = _slice_seq(out[key], target_len, seq_axis)
        return out
    if isinstance(kv_state, tuple) and len(kv_state) == 2:
        return (
            _slice_seq(kv_state[0], target_len, seq_axis),
            _slice_seq(kv_state[1], target_len, seq_axis),
        )
    # Raw tensor / array.
    return _slice_seq(kv_state, target_len, seq_axis)


def _slice_seq(tensor: Any, target_len: int, seq_axis: int) -> Any:
    """Slice ``tensor`` along ``seq_axis`` to length ``target_len``.

    Works on anything that supports ``__getitem__`` with tuple
    indices — torch.Tensor, mlx.array, numpy.ndarray, plain list.
    The slicer constructs ``(slice(None), ..., slice(0, target_len),
    ..., slice(None))`` with the ``slice(0, target_len)`` placed at
    the resolved positive axis.
    """
    if tensor is None:
        return None
    # Resolve negative axes against the tensor's rank.
    rank = _tensor_rank(tensor)
    axis = seq_axis if seq_axis >= 0 else rank + seq_axis
    if axis < 0 or axis >= rank:
        raise RollbackError(
            f"seq_axis={seq_axis} out of range for tensor rank {rank}",
        )
    idx = tuple(
        slice(None) if i != axis else slice(0, target_len)
        for i in range(rank)
    )
    return tensor[idx]


def _tensor_rank(tensor: Any) -> int:
    """Return ndim/rank of any tensor-like. Falls back to ``len(shape)``
    for libraries that expose only a ``shape`` tuple."""
    if hasattr(tensor, "ndim"):
        return int(tensor.ndim)
    if hasattr(tensor, "shape"):
        return len(tensor.shape)
    if isinstance(tensor, list):
        # Best-effort for nested lists in tests.
        d = 0
        cur: Any = tensor
        while isinstance(cur, list):
            d += 1
            cur = cur[0] if cur else None
        return d
    raise RollbackError(
        f"cannot determine rank of {type(tensor).__name__}",
    )
