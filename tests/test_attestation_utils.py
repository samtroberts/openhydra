from __future__ import annotations

import hashlib
from pathlib import Path

from supernode.attestation_utils import (
    canonical_request_hash,
    multi_file_manifest_hash,
    sha256_file,
)


def test_sha256_file_matches_hashlib(tmp_path: Path):
    f = tmp_path / "data.bin"
    content = b"hello world" * 1000
    f.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    assert sha256_file(f) == expected


def test_sha256_file_empty(tmp_path: Path):
    f = tmp_path / "empty"
    f.write_bytes(b"")
    expected = hashlib.sha256(b"").hexdigest()
    assert sha256_file(f) == expected


def test_sha256_file_large_chunked(tmp_path: Path):
    f = tmp_path / "big.bin"
    content = b"\x42" * (3 * 1024 * 1024)
    f.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    assert sha256_file(f) == expected


def test_canonical_request_hash_deterministic():
    h1 = canonical_request_hash(model_id="llama-7b", prompt="hello")
    h2 = canonical_request_hash(model_id="llama-7b", prompt="hello")
    assert h1 == h2
    assert len(h1) == 64


def test_canonical_request_hash_differs_on_prompt():
    h1 = canonical_request_hash(model_id="llama-7b", prompt="hello")
    h2 = canonical_request_hash(model_id="llama-7b", prompt="world")
    assert h1 != h2


def test_canonical_request_hash_differs_on_model():
    h1 = canonical_request_hash(model_id="llama-7b", prompt="hello")
    h2 = canonical_request_hash(model_id="llama-13b", prompt="hello")
    assert h1 != h2


def test_canonical_request_hash_with_messages():
    msgs = [{"role": "user", "content": "hi"}]
    h1 = canonical_request_hash(model_id="m", messages=msgs)
    h2 = canonical_request_hash(model_id="m", prompt="hi")
    assert h1 != h2


def test_multi_file_manifest_hash_deterministic():
    entries = [
        ("model.safetensors", 1000, "abc123"),
        ("config.json", 200, "def456"),
    ]
    h1 = multi_file_manifest_hash(entries)
    h2 = multi_file_manifest_hash(entries)
    assert h1 == h2
    assert len(h1) == 64


def test_multi_file_manifest_hash_order_independent():
    e1 = [("a.bin", 100, "aaa"), ("b.bin", 200, "bbb")]
    e2 = [("b.bin", 200, "bbb"), ("a.bin", 100, "aaa")]
    assert multi_file_manifest_hash(e1) == multi_file_manifest_hash(e2)


def test_multi_file_manifest_hash_differs_on_content():
    e1 = [("a.bin", 100, "aaa")]
    e2 = [("a.bin", 100, "bbb")]
    assert multi_file_manifest_hash(e1) != multi_file_manifest_hash(e2)


def test_multi_file_manifest_hash_empty():
    h = multi_file_manifest_hash([])
    assert len(h) == 64
