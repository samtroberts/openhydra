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

"""DFlash drafter wrapper for OpenHydra Phase 2b.

Wraps the two upstream DFlash implementations behind one Python
interface so the rest of the coordinator can be backend-agnostic:

* ``z-lab/dflash`` — PyTorch reference (Transformers backend with
  ``AutoModel.spec_generate``). Used on CUDA peers.
* ``bstnxbt/dflash-mlx`` — Apple Silicon implementation. Used when the
  coordinator (or stage-0 peer in Topology B) is a Mac.

Contract:

    drafter = load_dflash_drafter(cfg)
    draft_ids = drafter.draft(prefix_token_ids)   # len == cfg.block_size

Both DFlash packages are optional dependencies. The skeleton imports
lazily; if neither is available the loader raises
``DFlashNotAvailableError`` with a clear installation hint instead of
exploding at import time. This keeps Phase 2a installs working without
DFlash being on the path.

A ``MockDFlashDrafter`` is exposed for tests so the rest of Phase 2b
(verify_block, KV rollback, dual-topology driver, telemetry) can be
exercised end-to-end without pulling either upstream package.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "DFlashConfig",
    "DFlashDrafter",
    "DFlashNotAvailableError",
    "MockDFlashDrafter",
    "load_dflash_drafter",
]


# ── Phase 2b launch defaults (per ARCHITECTURE_ROADMAP_v1 + plan) ─────
_DEFAULT_TARGET_CUDA = "Qwen/Qwen3.5-4B"
_DEFAULT_TARGET_MLX = "mlx-community/Qwen3.5-4B-MLX-4bit"
_DEFAULT_DRAFT = "z-lab/Qwen3.5-4B-DFlash"


class DFlashNotAvailableError(RuntimeError):
    """Raised when the requested DFlash backend isn't importable.

    Carries a structured ``hint`` field telling the operator the exact
    pip extra to install. Surfaced verbatim by the coordinator startup
    path so the failure mode is self-documenting.
    """

    def __init__(self, backend: str, *, install_hint: str):
        super().__init__(
            f"dflash_backend_unavailable: backend={backend!r}; "
            f"install with: {install_hint}"
        )
        self.backend = backend
        self.hint = install_hint


@dataclass(frozen=True)
class DFlashConfig:
    """Configuration for a DFlash drafter.

    Attributes:
        target_model_path: HF path or local dir of the target model.
            On CUDA peers this is ``Qwen/Qwen3.5-4B``; on MLX peers it
            is ``mlx-community/Qwen3.5-4B-MLX-4bit``. Phase 2b
            launch defaults — see ARCHITECTURE_ROADMAP_v1.md "Pinned
            models" for the full matrix.
        draft_model_path: HF path or local dir of the DFlash draft.
            Phase 2b launch default: ``z-lab/Qwen3.5-4B-DFlash``.
        block_size: Tokens drafted per call to ``draft()``. Capped to
            32 by ``EngineConfig.draft_block_size``.
        backend: ``"mlx"`` for Apple Silicon, ``"pytorch"`` for
            CUDA/CPU, ``"mock"`` for tests.
        sliding_window_size: dflash-mlx experimental flag; bounds the
            committed draft KV history. ``None`` = disabled (default).
        device: Backend-specific device string. CUDA: ``"cuda:0"``,
            ``"cuda:1"``, etc. PyTorch CPU: ``"cpu"``. MLX: ignored
            (always Metal). Phase 2b uses the first available device
            unless overridden.
    """

    target_model_path: str = _DEFAULT_TARGET_CUDA
    draft_model_path: str = _DEFAULT_DRAFT
    block_size: int = 16
    backend: str = "mlx"           # "mlx" | "pytorch" | "mock"
    sliding_window_size: Optional[int] = None
    device: str = ""

    def __post_init__(self) -> None:
        if self.backend not in {"mlx", "pytorch", "mock"}:
            raise ValueError(
                f"DFlashConfig.backend must be 'mlx', 'pytorch', or "
                f"'mock'; got {self.backend!r}"
            )
        if self.block_size < 1 or self.block_size > 32:
            raise ValueError(
                f"DFlashConfig.block_size must be in [1, 32]; "
                f"got {self.block_size}"
            )


class DFlashDrafter:
    """Abstract base — wrappers for each backend subclass this."""

    def __init__(self, cfg: DFlashConfig):
        self._cfg = cfg
        self._loaded = False

    @property
    def config(self) -> DFlashConfig:
        return self._cfg

    @property
    def is_loaded(self) -> bool:
        """True after the first successful ``draft()`` (lazy load) or
        after an explicit ``ensure_loaded()`` call."""
        return self._loaded

    def ensure_loaded(self) -> None:
        """Materialise model weights eagerly. Default: no-op (subclass
        implementations should override to perform the load)."""
        self._loaded = True

    def draft(self, prefix_token_ids: list[int]) -> list[int]:
        """Generate ``self._cfg.block_size`` candidate tokens.

        Subclasses MUST return a list of length ``self._cfg.block_size``
        regardless of acceptance pattern — block diffusion is parallel,
        not autoregressive, so the full block is always emitted.
        """
        raise NotImplementedError  # pragma: no cover

    def reload(self, cfg: DFlashConfig) -> None:
        """Replace the loaded weights. Used during failover (Phase 2b
        Topology A → stage-0 promotion). Default: re-init self with
        the new config."""
        self._cfg = cfg
        self._loaded = False
        self.ensure_loaded()

    def memory_mb(self) -> int:
        """Approximate RAM/VRAM footprint of loaded weights. Used by
        the Phase 3 auto-negotiator. Default: 0 when not loaded."""
        return 0


class MockDFlashDrafter(DFlashDrafter):
    """Deterministic drafter for tests.

    Produces ``[seed, seed+1, …, seed+block_size-1]`` where ``seed``
    is a hash of ``prefix_token_ids``. Acceptance behaviour can be
    controlled by callers because the output is fully predictable.

    Used by:
      * ``tests/test_block_verify.py`` — synthetic drafts to exercise
        the accept-prefix logic.
      * ``tests/test_dual_topology.py`` — full ring under both
        topologies without pulling DFlash itself.
    """

    def __init__(self, cfg: DFlashConfig):
        super().__init__(cfg)
        self._loaded = True   # mock has nothing to load

    def ensure_loaded(self) -> None:
        self._loaded = True

    def draft(self, prefix_token_ids: list[int]) -> list[int]:
        # Deterministic seed: low bits of sum(prefix). Stable under
        # serialise/deserialise so tests can assert exact draft bodies.
        seed = (sum(int(t) for t in prefix_token_ids) & 0xFFFF) or 1
        return [(seed + i) & 0xFFFF for i in range(self._cfg.block_size)]


class _PyTorchDFlashDrafter(DFlashDrafter):
    """PyTorch backend via ``z-lab/dflash`` Transformers entrypoint.

    Lazy-imports ``dflash`` on first ``draft()`` so installs without
    the optional dep don't pay an import cost. The actual
    ``spec_generate`` wiring is implemented in a follow-up commit
    (Commit 9 — Topology A driver loop) so this skeleton lands green
    without requiring the upstream package on the test runner.
    """

    _INSTALL_HINT = "pip install 'openhydra[speculative-pytorch]'"

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            import dflash  # noqa: F401  — presence check
        except ImportError as exc:
            raise DFlashNotAvailableError(
                backend="pytorch",
                install_hint=self._INSTALL_HINT,
            ) from exc
        # Real load implemented in Commit 9.
        self._loaded = True
        logger.info(
            "dflash_pytorch_drafter_loaded target=%s draft=%s block=%d device=%s",
            self._cfg.target_model_path, self._cfg.draft_model_path,
            self._cfg.block_size, self._cfg.device or "auto",
        )

    def draft(self, prefix_token_ids: list[int]) -> list[int]:  # pragma: no cover - covered by integration tests
        self.ensure_loaded()
        raise NotImplementedError(
            "PyTorch DFlash draft() lands in Commit 9 (Topology A driver)"
        )


class _MLXDFlashDrafter(DFlashDrafter):
    """MLX backend via ``bstnxbt/dflash-mlx``.

    Lazy-imports ``dflash`` (the dflash-mlx package also ships the
    ``dflash`` namespace with an MLX submodule). Same lazy-load pattern
    as the PyTorch backend.
    """

    _INSTALL_HINT = "pip install 'openhydra[speculative-mlx]'"

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            from dflash import model_mlx  # noqa: F401  — presence check
        except ImportError as exc:
            raise DFlashNotAvailableError(
                backend="mlx",
                install_hint=self._INSTALL_HINT,
            ) from exc
        # Real load implemented in Commit 9.
        self._loaded = True
        logger.info(
            "dflash_mlx_drafter_loaded target=%s draft=%s block=%d sliding_window=%s",
            self._cfg.target_model_path, self._cfg.draft_model_path,
            self._cfg.block_size, self._cfg.sliding_window_size,
        )

    def draft(self, prefix_token_ids: list[int]) -> list[int]:  # pragma: no cover - covered by integration tests
        self.ensure_loaded()
        raise NotImplementedError(
            "MLX DFlash draft() lands in Commit 9 (Topology A driver)"
        )


def load_dflash_drafter(cfg: DFlashConfig) -> DFlashDrafter:
    """Factory: instantiate the right drafter subclass for ``cfg.backend``.

    Does NOT call ``ensure_loaded``; that's deferred to first ``draft()``
    so a coord with ``--draft-location off`` doesn't pay any DFlash
    import cost. To force eager load, call ``drafter.ensure_loaded()``
    explicitly after construction.
    """
    if cfg.backend == "mock":
        return MockDFlashDrafter(cfg)
    if cfg.backend == "pytorch":
        return _PyTorchDFlashDrafter(cfg)
    if cfg.backend == "mlx":
        return _MLXDFlashDrafter(cfg)
    raise ValueError(f"unknown DFlash backend: {cfg.backend!r}")
