# Copyright 2026 OpenHydra contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Read-only accessor for ``models.catalog.json``.

The authoritative map from a user-facing ``model_id``
(e.g. ``openhydra-qwen3.5-2b``) to a canonical HuggingFace
``hf_model_id`` (e.g. ``Qwen/Qwen3.5-2B``). Extracted from
``coordinator/engine.py`` so peers and tests can resolve the HF id
without pulling the full coordinator engine context.

The primary consumer is the MLX ↔ PyTorch tokenizer alignment path:
MLX peers must discard the tokenizer bundled with an
``mlx-community/*`` checkpoint and load the canonical HF tokenizer
instead, so token IDs crossing the wire are valid indices in any
downstream PyTorch peer's embedding table.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

__all__ = [
    "resolve_hf_model_id",
    "resolve_mlx_model_id",
    "resolve_dflash_draft_model_id",
]


def resolve_hf_model_id(
    model_id: str,
    *,
    catalog_path: str | Path | None = None,
    runtime_model_id: str | None = None,
) -> str:
    """Return the canonical HF model id for ``model_id``.

    Resolution order:
      1. Catalog lookup (``hf_model_id`` field for the matching entry).
      2. ``runtime_model_id`` if it looks like an HF id (contains ``/``)
         and is NOT an ``mlx-community/*`` fork.
      3. ``model_id`` itself if it looks like an HF id.
      4. Empty string when nothing suitable is found.

    Args:
        model_id: The user-facing model identifier.
        catalog_path: Path to ``models.catalog.json``. Defaults to
            ``models.catalog.json`` in the current working directory.
        runtime_model_id: Optional runtime override — if the user
            supplied ``--runtime-model-id`` on the CLI, we honour it as
            a fallback but only when it's an HF-shaped id.

    Returns:
        The canonical HF model id or ``""`` if unresolvable.
    """
    mid = str(model_id or "").strip()
    # 1. Catalog lookup.
    cp = Path(catalog_path) if catalog_path is not None else Path("models.catalog.json")
    try:
        if cp.exists():
            raw = json.loads(cp.read_text())
            if isinstance(raw, list):
                for entry in raw:
                    if not isinstance(entry, dict):
                        continue
                    if str(entry.get("model_id", "")).strip() == mid:
                        value = str(entry.get("hf_model_id", "") or "").strip()
                        if value:
                            return value
                        break  # found entry but empty hf_model_id → fall through
    except Exception as exc:  # pragma: no cover — logged, non-fatal
        logging.warning("hf_model_id_catalog_lookup_failed: path=%s err=%s", cp, exc)

    # 2. runtime_model_id fallback (only if HF-shaped and not mlx-community).
    rmi = str(runtime_model_id or "").strip()
    if "/" in rmi and not rmi.startswith("mlx-community/"):
        return rmi

    # 3. model_id itself when HF-shaped.
    if "/" in mid and not mid.startswith("mlx-community/"):
        return mid

    # 4. Unresolved.
    return ""


def _read_catalog_entry(
    model_id: str,
    *,
    catalog_path: str | Path | None = None,
) -> dict | None:
    """Return the catalog entry dict for ``model_id`` or ``None``."""
    mid = str(model_id or "").strip()
    if not mid:
        return None
    cp = Path(catalog_path) if catalog_path is not None else Path("models.catalog.json")
    try:
        if cp.exists():
            raw = json.loads(cp.read_text())
            if isinstance(raw, list):
                for entry in raw:
                    if isinstance(entry, dict) and str(entry.get("model_id", "")).strip() == mid:
                        return entry
    except Exception as exc:  # pragma: no cover — logged, non-fatal
        logging.warning("catalog_lookup_failed: path=%s id=%s err=%s", cp, mid, exc)
    return None


def resolve_mlx_model_id(
    model_id: str,
    *,
    catalog_path: str | Path | None = None,
) -> str:
    """Return the canonical MLX-community model id for a target model.

    Phase 2b: peers running MLX should load the ``mlx-community/*``
    fork specified by the catalog's ``mlx_model_id`` field rather than
    the canonical HF ``hf_model_id`` (which is unquantised fp16). This
    keeps Mac peers within unified-memory budget on 4B+ targets and
    matches the Phase 2b launch matrix in ARCHITECTURE_ROADMAP_v1.md.

    Returns the empty string when the catalog has no MLX-specific id.
    Callers must fall back to ``hf_model_id`` (or the operator's
    explicit ``--runtime-model-id``) when this returns empty.
    """
    entry = _read_catalog_entry(model_id, catalog_path=catalog_path)
    if entry is None:
        return ""
    return str(entry.get("mlx_model_id", "") or "").strip()


def resolve_dflash_draft_model_id(
    model_id: str,
    *,
    catalog_path: str | Path | None = None,
) -> str:
    """Return the DFlash draft model id paired with a target model.

    Distinct from ``draft_model_id`` (the legacy SpecPipe draft, used
    by ``coordinator/specpipe_scheduler.py``). DFlash drafts are
    block-diffusion models specifically trained against a target —
    e.g. ``z-lab/Qwen3.5-4B-DFlash`` for ``Qwen/Qwen3.5-4B``. The
    legacy and DFlash drafts coexist in the catalog so existing
    SpecPipe deployments keep working while Phase 2b ships.

    Returns the empty string when the catalog has no DFlash draft for
    this target. Callers should treat that as "DFlash unsupported for
    this model; pass --draft-model explicitly or use --draft-location off."
    """
    entry = _read_catalog_entry(model_id, catalog_path=catalog_path)
    if entry is None:
        return ""
    return str(entry.get("dflash_draft_model_id", "") or "").strip()
