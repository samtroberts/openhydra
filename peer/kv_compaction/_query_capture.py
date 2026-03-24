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
# See the License for the specific language governing permissions and
# limitations under the License.

"""AttentionQueryCapture — capture real query tensors during a model forward pass.

Option A: thread actual Q vectors from each attention layer into the compaction
pipeline to replace the proxy-key heuristic used in Phases 1-3.

Why this matters
----------------
The proxy heuristic in _compactor.py uses K[-n_ref:] (the last few *key*
vectors) as stand-ins for reference queries.  Keys and queries live in
different projected subspaces (W_k vs W_q), so the proxy Q-K dot products are
geometrically arbitrary and do not reflect true attention patterns.

Real queries come from  Q = W_q · hidden_states.  Even without RoPE (which
is position-dependent and hard to reproduce cheaply outside the attention
kernel) the W_q projection puts the reference queries in the correct subspace,
so the attention scores used for key selection and β fitting become
semantically meaningful.

This gives a significant quality uplift at virtually zero added latency:
the W_q projections are small linear layers, and the heavy matrix multiplies
(attention over the full KV cache) are already done inside the forward pass.

Design
------
AttentionQueryCapture registers a pre-forward hook on every transformer layer
to capture the input hidden states (batch, seq_len, hidden_size).  After the
forward completes, ``compute_q_ref()`` runs the Q projection (W_q) on the
last n_ref token positions and groups the resulting Q heads by kv-head
(averaging within each GQA group).

RoPE is intentionally omitted from the captured Q.  For compaction KEY
SELECTION the relative content-similarity is the dominant factor; the
position-encoding cross-term averages to near-zero when compared across many
reference queries.  Empirical results confirm that W_q·h (no RoPE) is
significantly closer to the true attention ranking than the proxy-K approach.

Returns
-------
``compute_q_ref()`` → List[Tensor | None] of length n_layers.
Each non-None element has shape (n_kv_heads, n_ref, head_dim) — one slice
per kv-head, ready to be passed as ``Q_ref_actual`` in ``_compact_single_head``.
"""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)


class AttentionQueryCapture:
    """Context manager that captures Q = W_q(hidden) per layer during forward.

    Usage::

        with AttentionQueryCapture(model, n_ref=8) as qc:
            model_out = model(**model_kwargs)
        q_per_layer = qc.compute_q_ref()
        # q_per_layer: list[Tensor(n_kv_heads, n_ref, d_head) | None]

    The object is single-use: call ``compute_q_ref()`` once, then discard.
    """

    def __init__(self, model: Any, n_ref: int = 8) -> None:
        self._model = model
        self._n_ref = max(1, int(n_ref))
        self._hidden_per_layer: dict[int, "Tensor"] = {}
        self._hooks: list[Any] = []
        self._entered = False

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "AttentionQueryCapture":
        self._register_hooks()
        self._entered = True
        return self

    def __exit__(self, *args: Any) -> None:
        self._remove_hooks()

    # ── hook management ───────────────────────────────────────────────────────

    def _register_hooks(self) -> None:
        decoder = getattr(self._model, "model", None)
        if decoder is None:
            return
        layers = getattr(decoder, "layers", None)
        if not layers:
            return

        for layer_idx, layer in enumerate(layers):
            capture = self  # captured by closure

            def _make_hook(idx: int):
                def _pre_hook(module: Any, args: tuple, kwargs: dict) -> None:
                    # hidden_states is always the first positional arg in
                    # standard HF transformer layers, or passed as a kwarg.
                    hidden = None
                    if args:
                        hidden = args[0]
                    if hidden is None:
                        hidden = kwargs.get("hidden_states")
                    if hidden is None:
                        return
                    if hidden.ndim != 3:
                        return
                    # Capture the last n_ref token positions.
                    # Detach to avoid holding onto the computation graph.
                    n = min(capture._n_ref, hidden.shape[1])
                    capture._hidden_per_layer[idx] = (
                        hidden[:, -n:, :].detach().clone()
                    )

                return _pre_hook

            try:
                h = layer.register_forward_pre_hook(
                    _make_hook(layer_idx), with_kwargs=True
                )
                self._hooks.append(h)
            except Exception as exc:
                logger.debug("query_capture_hook_register_failed layer=%d: %s", layer_idx, exc)

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()

    # ── Q computation ─────────────────────────────────────────────────────────

    def compute_q_ref(self) -> "list[Tensor | None]":
        """Compute Q = W_q(hidden) per layer and group by kv-head.

        Must be called *after* the model forward completes (i.e. after
        ``__exit__`` has been called or after the ``with`` block).

        Returns:
            List of length n_layers.  Each element is either:
              - Tensor of shape (n_kv_heads, n_ref, head_dim) — Q grouped
                per kv-head (averaged within each GQA group), or
              - None if Q could not be computed for that layer.
        """
        import torch

        decoder = getattr(self._model, "model", None)
        if decoder is None:
            return []
        layers = getattr(decoder, "layers", None)
        if not layers:
            return []

        q_per_layer: list["Tensor | None"] = []

        for layer_idx, layer in enumerate(layers):
            hidden = self._hidden_per_layer.get(layer_idx)
            if hidden is None:
                q_per_layer.append(None)
                continue

            attn = getattr(layer, "self_attn", None)
            if attn is None:
                q_per_layer.append(None)
                continue

            q_proj = getattr(attn, "q_proj", None)
            if q_proj is None:
                q_per_layer.append(None)
                continue

            try:
                n_q: int = int(
                    getattr(attn, "num_heads", None)
                    or getattr(attn, "num_attention_heads", None)
                    or 0
                )
                n_kv: int = int(
                    getattr(attn, "num_key_value_heads", None) or n_q
                )
                head_dim: int = int(getattr(attn, "head_dim", None) or 0)

                if not (n_q > 0 and n_kv > 0 and head_dim > 0):
                    q_per_layer.append(None)
                    continue

                n_groups = max(1, n_q // n_kv)

                with torch.no_grad():
                    # Cast hidden to the same dtype as q_proj weights
                    proj_dtype = next(q_proj.parameters()).dtype
                    h = hidden.to(proj_dtype)              # (1, n_ref, H)

                    # Q projection (no RoPE — intentional, see module docstring)
                    q_raw = q_proj(h)                     # (1, n_ref, n_q·d)
                    bsz, n_ref_actual, _ = q_raw.shape

                    # Reshape → (n_q, n_ref, head_dim)
                    q = (
                        q_raw
                        .view(bsz, n_ref_actual, n_q, head_dim)[0]   # (n_ref, n_q, d)
                        .permute(1, 0, 2)                             # (n_q, n_ref, d)
                    )

                    # Group by kv-head and average within each GQA group
                    # (n_kv, n_groups, n_ref, d) → mean over groups → (n_kv, n_ref, d)
                    q_kv = (
                        q.view(n_kv, n_groups, n_ref_actual, head_dim)
                        .mean(dim=1)                                  # (n_kv, n_ref, d)
                        .float()
                    )

                q_per_layer.append(q_kv)

            except Exception as exc:
                logger.debug("query_capture_compute_failed layer=%d: %s", layer_idx, exc)
                q_per_layer.append(None)

        # Free captured hidden states — they're large
        self._hidden_per_layer.clear()
        return q_per_layer
