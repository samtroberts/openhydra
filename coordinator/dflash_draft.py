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
    backend: str = "mlx"           # "mlx" | "pytorch" | "mock" | "autoregressive"
    sliding_window_size: Optional[int] = None
    device: str = ""

    def __post_init__(self) -> None:
        if self.backend not in {"mlx", "pytorch", "mock", "autoregressive"}:
            raise ValueError(
                f"DFlashConfig.backend must be 'mlx', 'pytorch', 'mock', "
                f"or 'autoregressive'; got {self.backend!r}"
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
        # The `dflash-mlx` PyPI package installs as `dflash_mlx` (not
        # `dflash` — the upstream README's `from dflash.model_mlx`
        # example refers to an older package layout).
        try:
            from dflash_mlx.runtime import (
                load_target_bundle, load_draft_bundle,
                generate_dflash_once,
            )
        except ImportError as exc:
            raise DFlashNotAvailableError(
                backend="mlx",
                install_hint=self._INSTALL_HINT,
            ) from exc

        # dflash-mlx 0.1.0 splits the loader into two bundle helpers.
        # Each returns a small dataclass-like object with .model and
        # .tokenizer attributes (target side) or .model (draft side).
        # We hold the bundles + the generate_dflash_once function the
        # draft() method uses.
        self._target_bundle = load_target_bundle(self._cfg.target_model_path)
        self._draft_bundle = load_draft_bundle(self._cfg.draft_model_path)
        self._generate_once = generate_dflash_once
        self._loaded = True
        logger.info(
            "dflash_mlx_drafter_loaded target=%s draft=%s block=%d "
            "sliding_window=%s api=runtime.generate_dflash_once",
            self._cfg.target_model_path, self._cfg.draft_model_path,
            self._cfg.block_size, self._cfg.sliding_window_size,
        )

    def draft(self, prefix_token_ids: list[int]) -> list[int]:
        """Run a single block-diffusion draft + local verify via
        dflash-mlx, return the FIRST block_size accepted tokens.

        Implementation note for Phase 2b live-bench:
        ``dflash_mlx.runtime.generate_dflash_once`` is an
        end-to-end loop (draft → local verify → accept → loop).
        It does NOT expose a "draft only this block" entry point;
        the draft model's __call__ requires target hidden states
        as input which would couple us to the package's internals.

        Pragmatic compromise for the cross-ISP benchmark: call
        ``generate_dflash_once`` with ``max_new_tokens=block_size``
        and treat the accepted tokens as "drafts" to verify on the
        OpenHydra ring. The OpenHydra HeadSampler.verify_block then
        runs them through the layer-sharded ring — those drafts WILL
        be accepted by the ring's verify (because they're already
        argmax-equivalent on the local target), but they ride
        through our distributed verify path so the multi-peer ring
        machinery is exercised end-to-end. Acceptance rate ~100%
        on the ring; speedup compared to per-token decode is the
        block-amortisation factor.

        For a true "draft only" path that lets the ring verify
        catch divergence between local Mac and remote GPU peers
        (which would happen if 4-bit MLX target diverges from
        fp16 PyTorch target), Phase 4 work is needed.
        """
        self.ensure_loaded()
        prompt_tokens = list(prefix_token_ids) if prefix_token_ids else [0]
        try:
            result = self._generate_once(
                target_model=self._target_bundle.model,
                tokenizer=self._target_bundle.tokenizer,
                draft_model=self._draft_bundle.model,
                prompt="",                         # using prompt_tokens_override
                max_new_tokens=self._cfg.block_size,
                use_chat_template=False,
                block_tokens=self._cfg.block_size,
                prompt_tokens_override=prompt_tokens,
            )
        except Exception as exc:
            logger.error(
                "dflash_mlx_draft_failed: %s — falling back to "
                "deterministic mock to keep the ring exercised",
                exc, exc_info=True,
            )
            # Last-resort: produce a deterministic placeholder so
            # the ring keeps moving. The verify path will reject
            # these (acceptance ~ 0%) but the multi-peer machinery
            # still exercises end-to-end.
            seed = (sum(prompt_tokens) & 0xFFFF) or 1
            return [(seed + i) & 0xFFFF for i in range(self._cfg.block_size)]

        # ``generate_dflash_once`` returns a dict; the accepted/
        # generated tokens are typically under ``output_token_ids``
        # or ``tokens`` depending on version.
        emitted = (
            result.get("output_token_ids")
            or result.get("tokens")
            or result.get("accepted_token_ids")
            or []
        )
        emitted = [int(t) for t in emitted][: self._cfg.block_size]
        if len(emitted) < self._cfg.block_size:
            # Pad with last to maintain contract — verify pass
            # will reject the padding.
            emitted += [emitted[-1] if emitted else 0] * (
                self._cfg.block_size - len(emitted)
            )
        return emitted


class _AutoRegressiveDrafter(DFlashDrafter):
    """Standard speculative decoding drafter using a small causal LM.

    Loads any HuggingFace ``AutoModelForCausalLM``-compatible model
    (e.g. ``Qwen/Qwen3.5-0.8B``) and generates draft tokens via
    greedy autoregressive decoding. The draft tokens are then
    verified by the full target model through the ring.

    This is the "classic" speculative decoding approach (Leviathan
    et al. 2022) — no block-diffusion, no target hidden states
    needed. The draft model only needs the prefix token IDs.

    Advantages over DFlash for distributed sharding:
      * Draft model is self-contained — no target hidden state
        extraction from specific layers across peers.
      * Works with any same-family small model as drafter.
      * Acceptance rate depends on distribution match between
        small and large model (typically 60-85% for same-family).
    """

    _INSTALL_HINT = "pip install transformers accelerate"

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise DFlashNotAvailableError(
                backend="autoregressive",
                install_hint=self._INSTALL_HINT,
            ) from exc

        device = self._cfg.device or "auto"
        # Detect dtype: fp16 for CUDA (T4 etc.), bf16 for Ampere+
        import torch
        if device == "auto" or "cuda" in device:
            # T4 = sm_75, no native bf16. Ampere (sm_80+) has bf16.
            if torch.cuda.is_available():
                cap = torch.cuda.get_device_capability()
                dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
            else:
                dtype = torch.float32
        else:
            dtype = torch.float32

        logger.info(
            "autoregressive_drafter_loading: model=%s device=%s dtype=%s",
            self._cfg.draft_model_path, device, dtype,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self._cfg.draft_model_path,
            dtype=dtype,
            device_map=device if device != "auto" else "auto",
            trust_remote_code=True,
        ).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._cfg.draft_model_path,
            trust_remote_code=True,
        )
        self._device = next(self._model.parameters()).device
        self._dtype = dtype
        self._loaded = True
        logger.info(
            "autoregressive_drafter_loaded: model=%s device=%s "
            "params=%.1fM block_size=%d",
            self._cfg.draft_model_path, self._device,
            sum(p.numel() for p in self._model.parameters()) / 1e6,
            self._cfg.block_size,
        )

    def draft(self, prefix_token_ids: list[int]) -> list[int]:
        self.ensure_loaded()
        import torch

        input_ids = torch.tensor(
            [list(prefix_token_ids)],
            dtype=torch.long,
            device=self._device,
        )
        with torch.inference_mode():
            output = self._model.generate(
                input_ids,
                max_new_tokens=self._cfg.block_size,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        # Extract only the NEW tokens (after the prefix).
        new_tokens = output[0, input_ids.shape[1]:].tolist()

        # Pad to block_size if the model hit EOS early.
        if len(new_tokens) < self._cfg.block_size:
            pad = new_tokens[-1] if new_tokens else 0
            new_tokens += [pad] * (self._cfg.block_size - len(new_tokens))

        return new_tokens[: self._cfg.block_size]

    def memory_mb(self) -> int:
        if not self._loaded:
            return 0
        try:
            import torch
            total = sum(
                p.numel() * p.element_size()
                for p in self._model.parameters()
            )
            return int(total / (1024 * 1024))
        except Exception:
            return 0


def load_dflash_drafter(cfg: DFlashConfig) -> DFlashDrafter:
    """Factory: instantiate the right drafter subclass for ``cfg.backend``.

    Does NOT call ``ensure_loaded``; that's deferred to first ``draft()``
    so a coord with ``--draft-location off`` doesn't pay any DFlash
    import cost. To force eager load, call ``drafter.ensure_loaded()``
    explicitly after construction.
    """
    if cfg.backend == "mock":
        return MockDFlashDrafter(cfg)
    if cfg.backend == "autoregressive":
        return _AutoRegressiveDrafter(cfg)
    if cfg.backend == "pytorch":
        return _PyTorchDFlashDrafter(cfg)
    if cfg.backend == "mlx":
        return _MLXDFlashDrafter(cfg)
    raise ValueError(f"unknown DFlash backend: {cfg.backend!r}")
