# Copyright 2026 OpenHydra contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Phase 2b — recurrent tape recorder for live model layers.

Captures the per-layer (pre_block_state, innovations) tape that
``TapeReplayRecurrentRollback`` (peer/kv_rollback.py) consumes on
rollback. Designed to instrument REAL Qwen3.5 GatedDeltaNet layers
without forking ``transformers`` — uses ``register_forward_pre_hook``
+ ``register_forward_hook`` on the layer module so the upstream
library stays a clean ``pip install`` dependency.

Architecture overview (Qwen3.5-4B / Qwen3Next as reference):

  * The actual layer class is ``Qwen3NextGatedDeltaNet`` from
    ``transformers.models.qwen3_next.modeling_qwen3_next``.
  * Recurrent state lives at ``cache_params.recurrent_states[layer_idx]``.
  * Conv state (matters for replay correctness — it precedes the
    recurrent update) lives at ``cache_params.conv_states[layer_idx]``.
  * The layer's forward signature is
    ``(hidden_states, cache_params, cache_position, attention_mask)``.

Replay strategy (the simpler-and-correct variant from the math
discussion):

  * Innovation per step = the per-position ``hidden_states`` slice
    that drove the verify forward.
  * Replay = re-invoke the layer's own forward over that slice with
    the cache pre-set to the pre-block snapshot. The layer recomputes
    its own (q, k, v, β, a) from the same input with the same weights
    → bit-identical state. No need to capture the projection
    intermediates.

This is more general than capturing (β, k, v) explicitly because:

  * It works for ANY recurrent layer architecture without changing
    the recorder — the replay function is just "re-run this layer's
    forward over this input."
  * It survives upstream library refactors that rename / reshape
    the projection sub-modules.
  * Memory cost is the same order: one ``hidden_states`` slice per
    step (block_size × hidden_size floats) vs one (β, k, v) triple
    per step. For Qwen3.5-4B with hidden=2560, block_size=16, fp16:
    ~80 KB per layer per session, well within the 5 MB budget.

Probe-and-fail-loud discipline: the recorder validates the layer's
attribute layout on install. Layers that don't match the canonical
Qwen3Next shape raise ``RecorderInstallError`` with a structured
``reason`` so the diagnostic log is actionable.

The recorder does NOT install hooks itself in this commit — that
happens during the verify forward path (lands with the API endpoint
integration in Item 3). This module ships:

  * ``RecurrentTapeRecorder`` — per-(session_id, layer_idx) record
    builder.
  * ``probe_layer_kind`` / ``probe_qwen3_next_layer`` — attribute
    detection helpers that produce a structured error on mismatch.
  * ``replay_step_qwen3_next`` — the closure factory that produces
    a ``replay_step`` callable for a given (layer, cache, layer_idx).
  * ``LayerLayoutError`` — raised on unsupported layer shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "LayerLayoutError",
    "RecurrentTapeRecorder",
    "probe_qwen3_next_layer",
    "probe_layer_kind",
    "replay_step_qwen3_next",
]


# ── Exceptions ──────────────────────────────────────────────────────────


class LayerLayoutError(RuntimeError):
    """Raised when a layer's attribute shape does not match any
    supported architecture.

    Carries ``reason``:
      * ``"unsupported_class"``    — class name not recognised.
      * ``"missing_projections"``  — expected projection sub-modules
                                      not present.
      * ``"missing_cache"``        — cache_params not threaded through
                                      the forward call.
      * ``"missing_layer_idx"``    — layer has no layer_idx attribute.
      * ``"cache_shape_mismatch"`` — cache.recurrent_states /
                                      conv_states absent or wrong shape.
    """

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


# ── Layer probing ──────────────────────────────────────────────────────


_QWEN3_NEXT_GATEDDELTANET_CLASSES = frozenset({
    "Qwen3NextGatedDeltaNet",
    # Future: add MLX equivalents when porting recorder to MLX
    # (Qwen3HybridGatedDeltaNet on the mlx-lm side).
})


def probe_layer_kind(layer: Any) -> str:
    """Return ``"qwen3_next"`` | ``"unknown"``.

    Cheap classification used by callers to decide which probe /
    recorder shape to use. Does not raise — caller can fall back
    to ``probe_layer_kind(...) == "unknown"`` and skip
    instrumentation rather than aborting startup.
    """
    name = type(layer).__name__
    if name in _QWEN3_NEXT_GATEDDELTANET_CLASSES:
        return "qwen3_next"
    return "unknown"


def probe_qwen3_next_layer(layer: Any) -> None:
    """Validate the layer matches the Qwen3Next GatedDeltaNet shape.

    Raises ``LayerLayoutError`` on any structural mismatch. Run
    once at recorder install time so the failure surface is
    startup-time rather than mid-generation.

    Concrete checks:

      1. Class name is in the recognised set.
      2. ``in_proj_qkvz`` and ``in_proj_ba`` projection sub-modules
         exist (these are the canonical Qwen3Next combined
         projections — see ``transformers/models/qwen3_next/
         modeling_qwen3_next.py::Qwen3NextGatedDeltaNet.__init__``).
      3. ``layer_idx`` attribute present (the layer's own index in
         ``model.layers`` — needed to address ``cache.recurrent_states``).
    """
    name = type(layer).__name__
    if name not in _QWEN3_NEXT_GATEDDELTANET_CLASSES:
        raise LayerLayoutError(
            "unsupported_class",
            f"probe_qwen3_next_layer: class {name!r} is not in the "
            f"recognised Qwen3Next set {sorted(_QWEN3_NEXT_GATEDDELTANET_CLASSES)}. "
            f"If this is a new transformers version with renamed "
            f"classes, extend _QWEN3_NEXT_GATEDDELTANET_CLASSES.",
        )

    for proj_name in ("in_proj_qkvz", "in_proj_ba"):
        if not hasattr(layer, proj_name):
            raise LayerLayoutError(
                "missing_projections",
                f"probe_qwen3_next_layer: layer {name!r} missing "
                f"sub-module {proj_name!r}. The Qwen3Next "
                f"GatedDeltaNet uses combined Q/K/V/z and β/α "
                f"projections; if upstream renamed these the "
                f"recorder needs an update.",
            )

    if not hasattr(layer, "layer_idx"):
        raise LayerLayoutError(
            "missing_layer_idx",
            f"probe_qwen3_next_layer: layer {name!r} has no "
            f"``layer_idx`` attribute; the recorder needs it to "
            f"address cache.recurrent_states[layer_idx]",
        )


def _resolve_cache_state(cache: Any, layer_idx: int) -> tuple[Any, Any]:
    """Read (recurrent_state, conv_state) for ``layer_idx`` from a
    Qwen3Next cache. Both must be present — Qwen3Next's recurrent
    update is preceded by a causal conv1d, so replay correctness
    requires restoring BOTH states.

    Raises ``LayerLayoutError(reason='cache_shape_mismatch')`` if the
    expected attribute paths are absent — common cause is passing the
    wrong cache type into the verify forward.
    """
    recurrent_states = getattr(cache, "recurrent_states", None)
    conv_states = getattr(cache, "conv_states", None)
    if recurrent_states is None or conv_states is None:
        raise LayerLayoutError(
            "cache_shape_mismatch",
            f"_resolve_cache_state: cache {type(cache).__name__} "
            f"missing recurrent_states or conv_states attribute. "
            f"Phase 2b expects a Qwen3NextDynamicCache-shaped object.",
        )
    try:
        rec = recurrent_states[layer_idx]
        conv = conv_states[layer_idx]
    except (IndexError, KeyError, TypeError) as exc:
        raise LayerLayoutError(
            "cache_shape_mismatch",
            f"_resolve_cache_state: cannot index cache state at "
            f"layer_idx={layer_idx}: {exc}",
        ) from exc
    return rec, conv


def _clone_cache_state(rec: Any, conv: Any) -> tuple[Any, Any]:
    """Deep-clone (recurrent_state, conv_state). Backend-neutral."""
    def _clone(t: Any) -> Any:
        if t is None:
            return None
        if hasattr(t, "clone"):
            return t.clone()
        if hasattr(t, "copy"):
            return t.copy()
        import copy as _copy
        return _copy.deepcopy(t)

    return _clone(rec), _clone(conv)


# ── Replay step factory ────────────────────────────────────────────────


def replay_step_qwen3_next(
    *,
    layer: Any,
    cache: Any,
    layer_idx: int,
) -> Callable[[Any, Any], Any]:
    """Build a ``replay_step(state, innovation) → state`` closure for
    a Qwen3Next GatedDeltaNet layer.

    The returned callable matches the signature
    ``TapeReplayRecurrentRollback`` expects:

        S' = replay_step(S, innovation)

    where:
      * ``S`` is a dict ``{"recurrent": <tensor>, "conv": <tensor>}``
        — the joint snapshot state.
      * ``innovation`` is the per-step ``hidden_states`` slice
        (a ``[1, 1, hidden_size]`` tensor).
      * ``S'`` is the post-step state dict.

    The closure restores the cache to ``S``, runs the layer's forward
    over the single-position innovation, then reads the cache back.
    Bit-identical to the verify forward at that position because the
    layer recomputes (q, k, v, β, a) from the same input with the
    same weights.

    Important: the closure mutates ``cache`` in place because that's
    how Qwen3Next's forward writes back state. Multi-rollback usage
    (e.g. Phase 2c tree speculation) must serialise calls to this
    closure or maintain separate cache instances per branch.
    """

    def _step(state: Any, innovation: Any) -> Any:
        if not isinstance(state, dict) or not all(
            k in state for k in ("recurrent", "conv")
        ):
            raise LayerLayoutError(
                "cache_shape_mismatch",
                "replay_step_qwen3_next: state must be a dict with "
                "'recurrent' and 'conv' keys",
            )

        # Restore cache to ``state``.
        cache.recurrent_states[layer_idx] = _clone_cache_state(
            state["recurrent"], state["conv"],
        )[0]
        cache.conv_states[layer_idx] = _clone_cache_state(
            state["recurrent"], state["conv"],
        )[1]

        # Re-run the layer's forward over the innovation. The layer
        # mutates the cache in place; we read the new state back.
        # cache_position must match the absolute sequence position
        # this innovation corresponds to — caller's responsibility
        # to set ``state['cache_position']`` if the layer's forward
        # depends on it. For Qwen3Next, single-position decode uses
        # ``cache_position`` to pick the conv-update vs prefill code
        # path; we always run the single-position path during replay.
        layer(innovation, cache_params=cache)

        rec, conv = _resolve_cache_state(cache, layer_idx)
        rec_clone, conv_clone = _clone_cache_state(rec, conv)
        return {"recurrent": rec_clone, "conv": conv_clone}

    return _step


# ── Recorder ────────────────────────────────────────────────────────────


@dataclass
class RecurrentTapeRecorder:
    """Per-(session_id, layer_idx) tape builder.

    Lifecycle:

        recorder = RecurrentTapeRecorder(
            session_id="sess-1", layer_idx=7,
            block_start_pos=128,
        )
        recorder.snapshot_pre_state(layer, cache)        # at block start
        recorder.append_innovation(hidden_slice_i)       # for each draft
        ...
        kv_state = recorder.to_kv_state(
            replay_step=replay_step_qwen3_next(
                layer=layer, cache=cache, layer_idx=7,
            ),
        )
        # Hand kv_state to TapeReplayRecurrentRollback for rollback.

    The recorder is intentionally state-mutation-light (no hooks
    held internally) so it composes cleanly with whatever forward-
    hook strategy the runtime uses to invoke
    ``snapshot_pre_state`` / ``append_innovation``. Two integration
    paths the runtime can choose:

      A. Forward-pre-hook captures the layer's ``hidden_states``
         input and feeds it to ``append_innovation``. One pre-hook
         per layer; cheap.
      B. Monkey-patch ``layer.forward`` to call into the recorder
         around the original. More invasive; only use if (A)
         doesn't fit (e.g. when the recorder needs intermediate
         results).

    For Qwen3Next we use (A): the recorder's ``append_innovation``
    receives the per-step ``hidden_states`` slice from the pre-hook,
    and at install time we ``snapshot_pre_state`` from the cache.
    """

    session_id: str
    layer_idx: int
    block_start_pos: int
    pre_block_state: Optional[dict] = None
    innovations: list = field(default_factory=list)

    def snapshot_pre_state(self, layer: Any, cache: Any) -> None:
        """Capture (recurrent, conv) state at block start. Idempotent
        — calling twice from the same block keeps the FIRST snapshot
        (mid-block re-snapshot would corrupt the rollback baseline).
        """
        if self.pre_block_state is not None:
            return
        rec, conv = _resolve_cache_state(cache, self.layer_idx)
        rec_clone, conv_clone = _clone_cache_state(rec, conv)
        self.pre_block_state = {"recurrent": rec_clone, "conv": conv_clone}
        logger.debug(
            "tape_recorder_pre_snapshot: session=%s layer=%d block_start=%d",
            self.session_id, self.layer_idx, self.block_start_pos,
        )

    def append_innovation(self, hidden_slice: Any) -> None:
        """Record the per-step hidden_states input. Replay re-runs
        the layer over each appended slice in order.

        Must be called once per draft position during the verify
        forward. Out-of-order calls would produce incorrect replay.
        """
        # Detach + clone so the recording doesn't keep gradient or
        # tie the lifetime of the innovation to the autograd graph.
        if hasattr(hidden_slice, "detach"):
            innovation = hidden_slice.detach().clone()
        elif hasattr(hidden_slice, "copy"):
            innovation = hidden_slice.copy()
        else:
            import copy as _copy
            innovation = _copy.deepcopy(hidden_slice)
        self.innovations.append(innovation)

    def to_kv_state(self, *, replay_step: Callable[[Any, Any], Any]) -> dict:
        """Materialise the kv_state dict ``TapeReplayRecurrentRollback``
        consumes.

        Raises if ``snapshot_pre_state`` was never called — that
        would mean the verify forward never entered this layer,
        which is a bug worth surfacing loudly rather than producing
        a tape with ``pre_block_state=None``.
        """
        if self.pre_block_state is None:
            raise LayerLayoutError(
                "missing_cache",
                f"to_kv_state: pre_block_state was never snapshotted "
                f"for session={self.session_id!r} "
                f"layer_idx={self.layer_idx}; verify forward did not "
                f"enter this layer",
            )
        return {
            "live_state": self.pre_block_state,   # placeholder; updated below
            "pre_block_state": self.pre_block_state,
            "innovations": list(self.innovations),
            "block_start_pos": int(self.block_start_pos),
            "replay_step": replay_step,
        }
