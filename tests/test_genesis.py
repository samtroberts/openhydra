import hashlib
import json

import pytest

from torrent.genesis import IntegrityError, bootstrap_weights, sha256_file


def test_bootstrap_weights_local_copy_and_manifest(tmp_path):
    src = tmp_path / "src.safetensors"
    src.write_bytes(b"abc123" * 100)

    result = bootstrap_weights(
        "model-x",
        cache_dir=str(tmp_path / "cache"),
        local_path=str(src),
        piece_bytes=64,
    )

    artifact = tmp_path / "cache" / "model-x" / "model-x.safetensors"
    manifest = tmp_path / "cache" / "model-x" / "manifest.json"
    torrent_meta = tmp_path / "cache" / "model-x" / "genesis.torrent.json"

    assert artifact.exists()
    assert manifest.exists()
    assert torrent_meta.exists()

    manifest_payload = json.loads(manifest.read_text())
    torrent_payload = json.loads(torrent_meta.read_text())

    assert result.source == "local-copy"
    assert manifest_payload["artifact"]["sha256"] == sha256_file(artifact)
    assert torrent_payload["piece_count"] == len(torrent_payload["pieces"])
    assert len(torrent_payload["pieces"]) >= 1


def test_bootstrap_weights_integrity_error(tmp_path):
    src = tmp_path / "src.safetensors"
    src.write_bytes(b"openhydra")

    with pytest.raises(IntegrityError):
        bootstrap_weights(
            "model-y",
            cache_dir=str(tmp_path / "cache"),
            local_path=str(src),
            expected_sha256="0" * 64,
        )


def test_bootstrap_weights_http_download(monkeypatch, tmp_path):
    payload = b"downloaded-weight-bytes"

    class _FakeResponse:
        def __init__(self, data: bytes):
            self._data = data

        def read(self, n=-1):
            if n == -1:
                n = len(self._data)
            chunk = self._data[:n]
            self._data = self._data[n:]
            return chunk

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("torrent.genesis.request.urlopen", lambda req, timeout=0: _FakeResponse(payload))

    expected = hashlib.sha256(payload).hexdigest()
    result = bootstrap_weights(
        "model-z",
        cache_dir=str(tmp_path / "cache"),
        source_url="https://example.com/model.safetensors",
        expected_sha256=expected,
    )

    assert result.source == "http-download"
    assert result.artifact_sha256 == expected
