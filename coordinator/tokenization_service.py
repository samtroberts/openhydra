"""Tokenization and draft-model loading service.

Centralises HuggingFace tokenizer caching, PyTorch draft-model caching,
and the logic that maps user-facing model IDs to runtime (HuggingFace)
model IDs via the model catalogue.
"""

from __future__ import annotations

from typing import Any, Callable

from coordinator.path_finder import PeerEndpoint
from coordinator.speculative import PyTorchDraftModel


def _default_trust_remote_code(model_id: str) -> bool:
    return "qwen" in str(model_id or "").strip().lower()


class TokenizationService:
    """Extracted from ``CoordinatorEngine``.

    Parameters
    ----------
    config:
        An ``EngineConfig`` (or duck-typed equivalent) that exposes at least
        ``pytorch_generation_model_id`` and ``pytorch_speculative_draft_model_id``.
    tokenizer_cache:
        Mutable dict used to cache loaded tokenizers by normalised model ID.
    draft_model_cache:
        Mutable dict used to cache ``PyTorchDraftModel`` instances by
        ``(draft_model_id, tokenizer_model_id)`` key.
    catalog_hf_model_id:
        A callable ``(model_id: str) -> str | None`` that looks up the
        HuggingFace model ID from the model catalogue.
    """

    def __init__(
        self,
        config: Any,
        tokenizer_cache: dict[str, Any],
        draft_model_cache: dict[tuple[str, str], PyTorchDraftModel],
        catalog_hf_model_id: Callable[[str], str | None],
    ):
        self.config = config
        self._tokenizer_cache = tokenizer_cache
        self._pytorch_draft_model_cache = draft_model_cache
        self._catalog_hf_model_id = catalog_hf_model_id

    # ------------------------------------------------------------------
    # Tokenizer loading
    # ------------------------------------------------------------------

    def _load_generation_tokenizer(self, model_id: str) -> Any:
        normalized = str(model_id or self.config.pytorch_generation_model_id).strip() or "gpt2"
        cached = self._tokenizer_cache.get(normalized)
        if cached is not None:
            return cached
        try:
            from transformers import AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "pytorch_generation_tokenizer_unavailable: install optional dependency 'transformers'"
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(
            normalized,
            trust_remote_code=_default_trust_remote_code(normalized),
        )
        self._tokenizer_cache[normalized] = tokenizer
        return tokenizer

    # ------------------------------------------------------------------
    # Draft model loading
    # ------------------------------------------------------------------

    def _load_pytorch_draft_model(self, *, tokenizer_model_id: str) -> PyTorchDraftModel:
        draft_model_id = str(self.config.pytorch_speculative_draft_model_id or "sshleifer/tiny-gpt2").strip() or "sshleifer/tiny-gpt2"
        tokenizer_model = str(tokenizer_model_id or self.config.pytorch_generation_model_id or "gpt2").strip() or "gpt2"
        key = (draft_model_id, tokenizer_model)
        cached = self._pytorch_draft_model_cache.get(key)
        if cached is not None:
            return cached
        model = PyTorchDraftModel(
            model_id=draft_model_id,
            tokenizer_model_id=tokenizer_model,
            target="cpu",
        )
        self._pytorch_draft_model_cache[key] = model
        return model

    # ------------------------------------------------------------------
    # Runtime model ID resolution
    # ------------------------------------------------------------------

    def _resolve_runtime_model_id(self, model_id: str) -> str:
        requested = str(model_id or "").strip()
        if not requested:
            return str(self.config.pytorch_generation_model_id or "gpt2").strip() or "gpt2"
        catalog_hf = self._catalog_hf_model_id(requested)
        if catalog_hf:
            return catalog_hf
        if "/" in requested:
            return requested
        return str(self.config.pytorch_generation_model_id or "gpt2").strip() or "gpt2"

    def _resolve_pipeline_runtime_model_id(self, pipeline: list[PeerEndpoint], served_model: str) -> str:
        for peer in pipeline:
            runtime_model_id = str(getattr(peer, "runtime_model_id", "") or "").strip()
            if runtime_model_id:
                return runtime_model_id
        return self._resolve_runtime_model_id(served_model)
