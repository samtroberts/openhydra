# Copyright 2026 OpenHydra contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Resolve a canonical model id at the model-load site (protocol.md §4).

The canonical id (`family/params/quant/template_hash`) is the protocol's
equivalent of a BitTorrent infohash. It must be computed **where the model /
tokenizer / external engine is loaded** — that is the only site with the chat
template — and then passed downstream as a plain string. The network layer
(``dht_announce``) stays a dumb carrier; it never sees a tokenizer or engine
handle. The hashing/parsing itself lives in Rust (``openhydra_network``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from peer.model_catalog import resolve_hf_model_id

_log = logging.getLogger(__name__)


def resolve_canonical_model_id(
    shard: Any,
    model_id: str,
    runtime_profile: dict[str, Any] | None,
    *,
    catalog_path: str | Path | None = None,
) -> str:
    """Compute the canonical model id from a loaded shard's tokenizer + runtime.

    Reads the chat template from the loaded tokenizer (``shard._runtime._tokenizer``),
    the runtime quant from ``runtime_profile``, and resolves the canonical HF id
    from the catalog. Returns ``""`` when unresolvable — e.g. a toy runtime with no
    tokenizer, a model with no chat template, or the Rust extension unavailable.
    Callers treat ``""`` as "not advertised", which is backward-compatible: legacy
    peers simply don't carry a canonical id and are never refused on that basis.
    """
    try:
        import openhydra_network as _ohn
    except Exception as exc:  # pragma: no cover — extension missing in some envs
        _log.debug("canonical_id: openhydra_network unavailable: %s", exc)
        return ""

    runtime = getattr(shard, "_runtime", None)
    tokenizer = getattr(runtime, "_tokenizer", None)
    chat_template = getattr(tokenizer, "chat_template", None)
    if not chat_template:
        # No template (e.g. ToyRuntime, or a base model without one) → no
        # meaningful canonical id. Not an error.
        return ""

    profile = runtime_profile or {}
    runtime_model_id = str(profile.get("runtime_model_id", "") or "")
    hf_model_id = resolve_hf_model_id(
        model_id, runtime_model_id=runtime_model_id, catalog_path=catalog_path
    )
    if not hf_model_id:
        hf_model_id = runtime_model_id or str(model_id or "")
    quant = str(profile.get("quantization_mode", "fp32") or "fp32")

    try:
        return str(_ohn.canonical_id_from_hf(hf_model_id, quant, str(chat_template)))
    except Exception as exc:  # malformed inputs → treat as not advertised
        _log.debug("canonical_id: resolve failed for model_id=%s: %s", model_id, exc)
        return ""
