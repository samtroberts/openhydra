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

"""MLX Pipeline Parallelism — Phase 4A.

Distributes transformer layers across multiple Apple Silicon devices (ranks)
connected via the MLX distributed backend (``mx.distributed``).

Architecture
------------
``PipelineParallelMLX`` wraps a loaded MLX model and assigns each rank a
contiguous slice of transformer layers using largest-remainder allocation.
During forward passes, hidden states are passed between ranks using
``mx.distributed.send()`` / ``mx.distributed.recv_like()``.

Overlapped prefill (Phase 4B):
    ``mx.async_eval()`` is used to overlap compute and network transfer.
    Rank 0 can begin sending its output tensor to Rank 1 while the Metal
    GPU is still finishing its evaluation — eliminating the synchronous
    barrier between pipeline stages.

Usage
-----
::

    from peer.mlx_parallel import PipelineParallelMLX

    model, tokenizer = mlx_load("Qwen/Qwen3.5-0.8B")
    parallel = PipelineParallelMLX(model, world_size=4, rank=0)
    hidden = parallel.forward_layers(input_hidden_states)
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["PipelineParallelMLX", "assign_layers"]


def assign_layers(
    total_layers: int,
    world_size: int,
) -> list[tuple[int, int]]:
    """Distribute layers across ranks using largest-remainder allocation.

    Ensures each rank gets at least ``floor(total_layers / world_size)``
    layers.  Remaining layers are distributed one each to the first
    ``total_layers % world_size`` ranks (largest remainder method).

    Args:
        total_layers: Total number of transformer layers in the model.
        world_size: Number of ranks (devices).

    Returns:
        List of ``(layer_start, layer_end)`` tuples, one per rank.
        Each range is ``[start, end)`` (exclusive end).

    Raises:
        ValueError: If ``world_size`` < 1 or ``total_layers`` < 1.

    Examples:
        >>> assign_layers(32, 4)
        [(0, 8), (8, 16), (16, 24), (24, 32)]

        >>> assign_layers(7, 3)
        [(0, 3), (3, 5), (5, 7)]

        >>> assign_layers(5, 2)
        [(0, 3), (3, 5)]
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if total_layers < 1:
        raise ValueError(f"total_layers must be >= 1, got {total_layers}")

    base = total_layers // world_size
    remainder = total_layers % world_size
    assignments: list[tuple[int, int]] = []
    offset = 0
    for rank in range(world_size):
        count = base + (1 if rank < remainder else 0)
        assignments.append((offset, offset + count))
        offset += count
    return assignments


class PipelineParallelMLX:
    """Pipeline-parallel wrapper for MLX models.

    Assigns a contiguous slice of transformer layers to each rank and
    provides ``forward_layers()`` for inter-rank hidden state passing.

    Args:
        model: A loaded MLX model (e.g. from ``mlx_lm.load``).
        world_size: Total number of ranks (devices).
        rank: This device's rank (0-indexed).
        async_eval: If ``True`` (default), use ``mx.async_eval()`` for
            overlapped pipeline prefill (Phase 4B).

    Attributes:
        layer_start: First layer this rank handles (inclusive).
        layer_end: Last layer this rank handles (exclusive).
        is_first: Whether this rank is the first in the pipeline.
        is_last: Whether this rank is the last in the pipeline.
    """

    def __init__(
        self,
        model: Any,
        world_size: int = 1,
        rank: int = 0,
        async_eval: bool = True,
    ) -> None:
        self.model = model
        self.world_size = max(1, int(world_size))
        self.rank = max(0, min(int(rank), self.world_size - 1))
        self.async_eval = bool(async_eval)

        # Detect transformer layers.
        self._blocks = self._find_transformer_blocks(model)
        self.total_layers = len(self._blocks)

        if self.total_layers == 0:
            logger.warning("mlx_parallel: no transformer blocks found in model")
            self.layer_start = 0
            self.layer_end = 0
            self._my_layers = []
        else:
            assignments = assign_layers(self.total_layers, self.world_size)
            self.layer_start, self.layer_end = assignments[self.rank]
            self._my_layers = self._blocks[self.layer_start:self.layer_end]

        self.is_first = self.rank == 0
        self.is_last = self.rank == self.world_size - 1

        logger.info(
            "mlx_parallel_init: rank=%d/%d layers=[%d,%d) total=%d async=%s",
            self.rank, self.world_size, self.layer_start, self.layer_end,
            self.total_layers, self.async_eval,
        )

    @staticmethod
    def _find_transformer_blocks(model: Any) -> list[Any]:
        """Extract the list of transformer blocks from the model.

        Tries common attribute paths used by HuggingFace MLX models:
        ``model.layers``, ``model.model.layers``, ``model.transformer.h``.

        Args:
            model: An MLX model instance.

        Returns:
            List of transformer block modules.
        """
        for path in [
            ("model", "layers"),
            ("layers",),
            ("model", "model", "layers"),
            ("transformer", "h"),
        ]:
            obj = model
            try:
                for attr in path:
                    obj = getattr(obj, attr)
                if hasattr(obj, "__len__") and len(obj) > 0:
                    return list(obj)
            except (AttributeError, TypeError):
                continue
        return []

    def forward_layers(
        self,
        hidden_states: Any,
        mask: Any = None,
        cache: Any = None,
    ) -> Any:
        """Run this rank's layer slice and exchange hidden states.

        For ``world_size == 1``, runs all layers locally (no communication).
        For ``world_size > 1``, receives from the previous rank (if not first),
        processes local layers, and sends to the next rank (if not last).

        Phase 4B: Uses ``mx.async_eval()`` to overlap compute and send,
        so the next rank can start receiving while this rank is still
        evaluating.

        Args:
            hidden_states: Input hidden states tensor.
            mask: Optional attention mask.
            cache: Optional KV cache.

        Returns:
            Output hidden states after processing this rank's layers.
        """
        try:
            import mlx.core as mx
        except ImportError:
            # Fallback for testing without MLX.
            return self._forward_layers_cpu(hidden_states, mask, cache)

        # ── Receive from previous rank ───────────────────────────────────
        if not self.is_first and self.world_size > 1:
            try:
                dist = mx.distributed
                hidden_states = dist.recv_like(hidden_states, src=self.rank - 1)
                mx.eval(hidden_states)
            except Exception as exc:
                logger.warning("mlx_recv_failed: rank=%d src=%d err=%s",
                              self.rank, self.rank - 1, exc)

        # ── Run local layers ─────────────────────────────────────────────
        for layer in self._my_layers:
            if cache is not None:
                hidden_states = layer(hidden_states, mask=mask, cache=cache)
            else:
                hidden_states = layer(hidden_states, mask=mask)

        # ── Phase 4B: Overlapped prefill via async_eval ──────────────────
        if self.async_eval and hasattr(mx, "async_eval"):
            mx.async_eval(hidden_states)
        else:
            mx.eval(hidden_states)

        # ── Send to next rank ────────────────────────────────────────────
        if not self.is_last and self.world_size > 1:
            try:
                dist = mx.distributed
                dist.send(hidden_states, dst=self.rank + 1)
            except Exception as exc:
                logger.warning("mlx_send_failed: rank=%d dst=%d err=%s",
                              self.rank, self.rank + 1, exc)

        return hidden_states

    def _forward_layers_cpu(
        self,
        hidden_states: Any,
        mask: Any = None,
        cache: Any = None,
    ) -> Any:
        """CPU fallback for testing — processes layers sequentially.

        Args:
            hidden_states: Input tensor (or mock).
            mask: Optional attention mask.
            cache: Optional KV cache.

        Returns:
            Output after processing this rank's layers.
        """
        for layer in self._my_layers:
            if cache is not None:
                hidden_states = layer(hidden_states, mask=mask, cache=cache)
            else:
                hidden_states = layer(hidden_states, mask=mask)
        return hidden_states

    @property
    def num_local_layers(self) -> int:
        """Number of layers assigned to this rank."""
        return len(self._my_layers)

    @property
    def layer_range(self) -> tuple[int, int]:
        """The ``(start, end)`` layer range for this rank."""
        return (self.layer_start, self.layer_end)
