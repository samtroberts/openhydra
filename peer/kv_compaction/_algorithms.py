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

"""Core KV compaction algorithms: HAK and OMP key selection + Phase 2 fitting.

All functions operate on single (head, layer) slices:
  K : (T, d_head) key matrix
  V : (T, d_head) value matrix
  Q_ref : (R, d_head) reference query matrix (proxy queries for scoring)
  t : target token count after compaction

No external dependencies beyond torch.  scipy is an *optional* improvement
for β fitting (falls back gracefully if not installed).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — key selection
# ─────────────────────────────────────────────────────────────────────────────

def select_hak(K: "Tensor", Q_ref: "Tensor", t: int) -> "Tensor":
    """Highest Attention Keys — select *t* indices by RMS-aggregated attention.

    For each reference query the attention distribution over K is computed.
    Tokens are ranked by RMS of their attention weight across all reference
    queries; the top *t* are returned.

    Args:
        K:     (T, d_head) key matrix.
        Q_ref: (R, d_head) reference query matrix.
        t:     Number of tokens to keep.

    Returns:
        Sorted index tensor of shape (min(t, T),).
    """
    import torch

    T = K.shape[0]
    t = min(t, T)
    if t >= T:
        return torch.arange(T, device=K.device)

    scale = K.shape[-1] ** 0.5
    logits = Q_ref.to(K.dtype) @ K.T / scale       # (R, T)
    attn = logits.float().softmax(dim=-1)           # (R, T)

    # RMS importance across reference queries
    importance = attn.pow(2).mean(dim=0).sqrt()     # (T,)
    return importance.topk(t).indices.sort().values


def select_omp(K: "Tensor", Q_ref: "Tensor", t: int) -> "Tensor":
    """OMP-style greedy key selection for attention-mass matching.

    At each step selects the token that maximally reduces the *uncovered*
    attention mass (greedy residual pursuit).  More accurate than HAK at the
    cost of O(T·t) iterations.

    Args:
        K:     (T, d_head) key matrix.
        Q_ref: (R, d_head) reference query matrix.
        t:     Number of tokens to keep.

    Returns:
        Sorted index tensor of shape (min(t, T),).
    """
    import torch

    T = K.shape[0]
    t = min(t, T)
    if t >= T:
        return torch.arange(T, device=K.device)

    scale = K.shape[-1] ** 0.5
    A = (Q_ref.to(K.dtype) @ K.T / scale).float().softmax(dim=-1)  # (R, T)

    # residual = uncovered attention mass per *reference query* (R,).
    # Starts at 1.0 for each query (softmax total mass = 1).
    residual = torch.ones(Q_ref.shape[0], device=K.device)          # (R,)
    available = torch.ones(T, dtype=torch.bool, device=K.device)
    selected: list[int] = []

    for _ in range(t):
        # score(k) = how much does token k cover the remaining residual?
        scores = A.T @ residual                                      # (T,)
        scores[~available] = float("-inf")
        k = int(scores.argmax())
        selected.append(k)
        available[k] = False
        # Subtract mass covered by token k from each query's residual
        residual = (residual - A[:, k]).clamp(min=0.0)              # (R,)

    return torch.tensor(sorted(selected), dtype=torch.long, device=K.device)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — β and Cv fitting
# ─────────────────────────────────────────────────────────────────────────────

def fit_beta_and_cv(
    K: "Tensor",
    V: "Tensor",
    Ck: "Tensor",
    Q_ref: "Tensor",
    indices: "Tensor",
) -> "tuple[Tensor, Tensor]":
    """Fit β (scalar log-space biases) and Cv (compact values).

    β corrects the attention mass underestimation that arises when only *t*
    compact keys are attended to instead of all T original keys.

    Cv minimises ‖attn_compact(+β) @ Cv − O_ref‖² via least squares.

    Uses scipy.optimize.nnls for β if available; falls back to a log-ratio
    approximation otherwise.

    Args:
        K:       (T, d_head) original key matrix.
        V:       (T, d_head) original value matrix.
        Ck:      (t, d_head) compact keys (rows of K at positions *indices*).
        Q_ref:   (R, d_head) reference queries.
        indices: (t,) positions of Ck in K.

    Returns:
        (beta, Cv) where beta: (t,) float32 log-space bias, Cv: (t, d_head).
    """
    import torch

    scale = K.shape[-1] ** 0.5
    eps = 1e-8

    A_orig = (Q_ref.to(K.dtype) @ K.T / scale).float().softmax(dim=-1)  # (R, T)
    A_sel = A_orig[:, indices]                                            # (R, t)
    A_cmp_logits = (Q_ref.to(K.dtype) @ Ck.T / scale).float()           # (R, t)
    A_cmp_sm = A_cmp_logits.softmax(dim=-1)                              # (R, t)

    # ── β fitting ────────────────────────────────────────────────────────────
    try:
        import scipy.optimize
        import numpy as np
        # NNLS formulation: find exp(β) ≥ 0 such that A_cmp_sm @ exp(β) ≈ mass_target
        # where mass_target[r] = fraction of original mass on selected tokens
        mass_target = A_sel.sum(dim=-1).numpy()          # (R,)
        A_np = A_cmp_sm.numpy()                          # (R, t)
        exp_beta_np, _ = scipy.optimize.nnls(A_np, mass_target)
        # Convert to log-space; clamp to prevent extreme values
        beta = torch.from_numpy(
            np.log(np.clip(exp_beta_np, eps, None))
        ).float().clamp(-10.0, 10.0)
    except Exception:
        # Fallback: per-token log-ratio averaged across reference queries
        beta = (A_sel / (A_cmp_sm + eps) + eps).log().mean(dim=0).clamp(-10.0, 10.0)

    beta = beta.to(K.device)

    # ── Cv fitting ───────────────────────────────────────────────────────────
    O_ref = A_orig @ V.float()                                           # (R, d_head)
    attn_corr = (A_cmp_logits + beta.unsqueeze(0)).softmax(dim=-1)      # (R, t)

    Cv = V[indices]  # default: just the selected values
    try:
        result = torch.linalg.lstsq(attn_corr, O_ref)
        solution = result.solution                                        # (t, d_head)
        if solution.shape == Cv.shape:
            Cv = solution.to(V.dtype)
    except Exception as exc:
        logger.debug("lstsq_failed_for_cv: %s — using selected values", exc)

    return beta, Cv
