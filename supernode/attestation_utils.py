from __future__ import annotations

import hashlib
from pathlib import Path

import cbor2


def sha256_file(path: str | Path) -> str:
    """Streaming SHA-256 hex digest (1 MB chunks)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_request_hash(
    *,
    model_id: str,
    prompt: str | None = None,
    messages: list[dict] | None = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    system_prompt: str | None = None,
) -> str:
    """CBOR canonical encode → SHA-256. Deterministic request fingerprint."""
    obj: dict = {"model_id": model_id, "max_tokens": max_tokens}
    if prompt is not None:
        obj["prompt"] = prompt
    if messages is not None:
        obj["messages"] = messages
    if system_prompt is not None:
        obj["system_prompt"] = system_prompt
    obj["temperature"] = temperature
    obj["top_p"] = top_p
    obj["top_k"] = top_k
    raw = cbor2.dumps(obj, canonical=True)
    return hashlib.sha256(raw).hexdigest()


def multi_file_manifest_hash(
    file_entries: list[tuple[str, int, str]],
) -> str:
    """SHA-256 over sorted (filename, size, sha256) tuples.

    Deterministic weight-manifest hash for multi-shard models.
    """
    sorted_entries = sorted(file_entries, key=lambda e: e[0])
    obj = [(name, size, digest) for name, size, digest in sorted_entries]
    raw = cbor2.dumps(obj, canonical=True)
    return hashlib.sha256(raw).hexdigest()
