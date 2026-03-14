"""Tests for Phase 5 — P2P Model Distribution.

Four test groups:

A. TestModelSeedServer
   Real HTTP server (port 0) + real temp dir.
   Validates full-file serving, Range requests, path-traversal blocking,
   directory listing, and /health endpoint.

B. TestHFManifest
   Mocks urllib.request.urlopen to simulate HF Hub API responses.
   Validates manifest fetching, 24 h disk cache, TTL expiry, and error handling.

C. TestP2PModelCache
   End-to-end: ModelSeedServer (seeder) + P2PModelCache (leecher).
   Validates local cache hit, P2P download + SHA-256 verification, hash mismatch
   rejection, resume from .part file, and graceful no-peers path.

D. TestDHTAnnouncement
   Unit-level dataclass tests for the new seeder fields in Announcement and
   PeerEndpoint, plus P2PModelCache.announce_cached_models() filtering.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import URLError
from urllib.request import Request
import urllib.request as _urllib_request

import pytest

from peer.dht_announce import Announcement
from peer.p2p_model_cache import HFManifest, P2PModelCache, _looks_complete, _sha256_file
from peer.seeder_http import ModelSeedServer, _is_safe_path, _parse_range
from coordinator.path_finder import PeerEndpoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_file(path: Path, data: bytes) -> str:
    """Write *data* to *path* and return its SHA-256 hex."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _make_fake_model_dir(root: Path, model_id: str, filenames: list[str]) -> dict[str, str]:
    """
    Write random model files to ``root/{model_id_safe}/``.
    Returns {filename: sha256_hex}.
    """
    model_safe = model_id.replace("/", "--")
    model_dir = root / model_safe
    manifest: dict[str, str] = {}
    for fname in filenames:
        data = os.urandom(512)  # 512 B of random bytes
        sha256 = _write_file(model_dir / fname, data)
        manifest[fname] = sha256
    return manifest


def _start_seeder(root: Path) -> ModelSeedServer:
    seeder = ModelSeedServer(cache_root=root, port=0)
    seeder.start()
    return seeder


def _fetch(url: str, headers: dict[str, str] | None = None) -> tuple[int, bytes, dict]:
    req = Request(url, headers=headers or {})
    try:
        with _urllib_request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except Exception as exc:
        code = getattr(getattr(exc, "code", None), "__class__", type(exc))
        status = getattr(exc, "code", 0)
        return status, b"", {}


# ---------------------------------------------------------------------------
# Group A: ModelSeedServer
# ---------------------------------------------------------------------------

class TestModelSeedServer:
    """HTTP seeder — real server on port 0, real temp directory."""

    def test_health_endpoint(self, tmp_path: Path) -> None:
        """GET /health → 200 JSON {"status": "ok"}."""
        seeder = _start_seeder(tmp_path)
        try:
            status, body, _ = _fetch(f"http://127.0.0.1:{seeder.port}/health")
            assert status == 200
            data = json.loads(body)
            assert data["status"] == "ok"
        finally:
            seeder.stop()

    def test_file_served_fully(self, tmp_path: Path) -> None:
        """GET full model file → 200, correct bytes returned."""
        model_safe = "org--model"
        content = b"A" * 4096
        file_path = tmp_path / model_safe / "model.safetensors"
        file_path.parent.mkdir(parents=True)
        file_path.write_bytes(content)

        seeder = _start_seeder(tmp_path)
        try:
            status, body, _ = _fetch(
                f"http://127.0.0.1:{seeder.port}/v1/p2p/models/{model_safe}/model.safetensors"
            )
            assert status == 200
            assert body == content
        finally:
            seeder.stop()

    def test_range_request(self, tmp_path: Path) -> None:
        """GET with Range header → 206 Partial Content, correct slice."""
        model_safe = "test--rangemodel"
        content = bytes(range(256)) * 40  # 10 240 B
        file_path = tmp_path / model_safe / "weights.bin"
        file_path.parent.mkdir(parents=True)
        file_path.write_bytes(content)

        seeder = _start_seeder(tmp_path)
        try:
            url = f"http://127.0.0.1:{seeder.port}/v1/p2p/models/{model_safe}/weights.bin"
            status, body, headers = _fetch(url, headers={"Range": "bytes=1000-1999"})
            assert status == 206
            assert body == content[1000:2000]
            assert "206" in str(status) or len(body) == 1000
        finally:
            seeder.stop()

    def test_path_traversal_blocked(self, tmp_path: Path) -> None:
        """GET with traversal sequence → 403 Forbidden."""
        seeder = _start_seeder(tmp_path)
        try:
            url = f"http://127.0.0.1:{seeder.port}/v1/p2p/models/foo/../../etc/passwd"
            status, _, _ = _fetch(url)
            # Server returns 404 for unmatched routes or 403 for traversal.
            # The regex won't match "../../etc/passwd" as a safe filename so we get 404,
            # but the realpath check also fires with 403 on valid-looking paths.
            assert status in (403, 404)
        finally:
            seeder.stop()

    def test_path_traversal_unsafe_filename_blocked(self, tmp_path: Path) -> None:
        """Filename with '..' is rejected with 403."""
        seeder = _start_seeder(tmp_path)
        try:
            # The regex _RE_FILE matches, but _RE_SAFE_NAME rejects ".." in filename.
            url = f"http://127.0.0.1:{seeder.port}/v1/p2p/models/model/..%2Fpwd"
            status, _, _ = _fetch(url)
            assert status in (403, 404)
        finally:
            seeder.stop()

    def test_directory_listing(self, tmp_path: Path) -> None:
        """GET directory URL → 200 JSON list of filenames."""
        model_safe = "org--listtest"
        model_dir = tmp_path / model_safe
        model_dir.mkdir()
        (model_dir / "config.json").write_bytes(b"{}")
        (model_dir / "model.safetensors").write_bytes(b"fake")

        seeder = _start_seeder(tmp_path)
        try:
            url = f"http://127.0.0.1:{seeder.port}/v1/p2p/models/{model_safe}/"
            status, body, _ = _fetch(url)
            assert status == 200
            files = json.loads(body)
            assert isinstance(files, list)
            assert "config.json" in files
            assert "model.safetensors" in files
        finally:
            seeder.stop()

    def test_missing_file_returns_404(self, tmp_path: Path) -> None:
        """GET non-existent file → 404."""
        seeder = _start_seeder(tmp_path)
        try:
            url = f"http://127.0.0.1:{seeder.port}/v1/p2p/models/nosuchmodel/weights.bin"
            status, _, _ = _fetch(url)
            assert status == 404
        finally:
            seeder.stop()


# ---------------------------------------------------------------------------
# Group B: HF Manifest fetching
# ---------------------------------------------------------------------------

_HF_SIBLINGS_FIXTURE = [
    {"rfilename": "config.json", "sha256": "aabbccdd" * 8},
    {"rfilename": "model.safetensors", "sha256": "11223344" * 8},
]

_HF_API_RESPONSE = json.dumps({"siblings": _HF_SIBLINGS_FIXTURE}).encode("utf-8")


class _MockResponse:
    """Minimal mock for urllib.request.urlopen return value."""

    def __init__(self, data: bytes, status: int = 200) -> None:
        self._data = data
        self.status = status

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> "_MockResponse":
        return self

    def __exit__(self, *args: object) -> None:
        pass


class TestHFManifest:
    """HF Hub manifest fetching — mocked network, real disk cache."""

    def _make_cache(self, tmp_path: Path) -> P2PModelCache:
        return P2PModelCache(
            cache_root=tmp_path / "cache",
            manifest_cache_dir=tmp_path / "manifests",
            dht_urls=[],
            hf_api_base="https://fake.hf.example",
        )

    def test_manifest_fetched_and_cached(self, tmp_path: Path) -> None:
        """First call fetches from HF API; second call reads disk cache."""
        cache = self._make_cache(tmp_path)
        call_count = 0

        def fake_urlopen(req: object, timeout: float = 10) -> _MockResponse:
            nonlocal call_count
            call_count += 1
            return _MockResponse(_HF_API_RESPONSE)

        with patch("peer.p2p_model_cache.urllib_request.urlopen", side_effect=fake_urlopen):
            m1 = cache._fetch_hf_manifest("org/model")
            m2 = cache._fetch_hf_manifest("org/model")

        assert call_count == 1, "Second call should have hit the disk cache"
        assert m1 is not None
        assert m2 is not None
        assert m1.files == m2.files
        assert "config.json" in m1.files
        assert "model.safetensors" in m1.files

    def test_manifest_ttl_expiry(self, tmp_path: Path) -> None:
        """Manifest past TTL is re-fetched from HF API."""
        cache = self._make_cache(tmp_path)
        call_count = 0

        def fake_urlopen(req: object, timeout: float = 10) -> _MockResponse:
            nonlocal call_count
            call_count += 1
            return _MockResponse(_HF_API_RESPONSE)

        with patch("peer.p2p_model_cache.urllib_request.urlopen", side_effect=fake_urlopen):
            # Fetch and write cache.
            cache._fetch_hf_manifest("org/model")

        # Manually age the cache file beyond TTL.
        cache_file = (tmp_path / "manifests" / "org--model.json")
        data = json.loads(cache_file.read_text())
        data["fetched_at"] = time.time() - 100_000  # 27 hours ago
        cache_file.write_text(json.dumps(data))

        with patch("peer.p2p_model_cache.urllib_request.urlopen", side_effect=fake_urlopen):
            cache._fetch_hf_manifest("org/model")

        assert call_count == 2, "Expired cache should trigger a re-fetch"

    def test_manifest_network_error_returns_none(self, tmp_path: Path) -> None:
        """Network error while fetching manifest → resolve() returns None."""
        cache = self._make_cache(tmp_path)

        def fail_urlopen(req: object, timeout: float = 10) -> _MockResponse:
            raise URLError("connection refused")

        with patch("peer.p2p_model_cache.urllib_request.urlopen", side_effect=fail_urlopen):
            result = cache._fetch_hf_manifest("org/model")

        assert result is None


# ---------------------------------------------------------------------------
# Group C: End-to-end P2P download
# ---------------------------------------------------------------------------

def _make_mock_peer(host: str, port: int, model_id: str) -> MagicMock:
    """Return a minimal mock PeerEndpoint for use with P2PModelCache."""
    peer = MagicMock()
    peer.host = host
    peer.seeder_http_port = port
    peer.cached_model_ids = (model_id,)
    return peer


class TestP2PModelCache:
    """End-to-end tests combining a real seeder and a real leecher cache."""

    def _make_leecher(
        self,
        tmp_path: Path,
        peers: list[Any] | None = None,
    ) -> P2PModelCache:
        """Create a P2PModelCache with patched _find_seeder_peers."""
        cache = P2PModelCache(
            cache_root=tmp_path / "leecher_cache",
            manifest_cache_dir=tmp_path / "manifests",
            dht_urls=[],
        )
        if peers is not None:
            cache._find_seeder_peers = lambda model_id: peers  # type: ignore[assignment]
        return cache

    def _manifest_for(self, file_map: dict[str, str]) -> HFManifest:
        """Build an HFManifest from {filename: sha256} map."""
        return HFManifest(model_id="test/model", files=file_map, fetched_at=time.time())

    # ------------------------------------------------------------------

    def test_local_cache_hit_no_download(self, tmp_path: Path) -> None:
        """Pre-populated cache dir returns path without any HTTP requests."""
        leecher = self._make_leecher(tmp_path)
        # Manually populate the leecher cache.
        model_safe = "test--model"
        model_dir = leecher._cache_root / model_safe
        _write_file(model_dir / "config.json", b"{}")
        _write_file(model_dir / "model.safetensors", b"fake_weights")

        fetch_count = [0]

        def fake_urlopen(req: object, timeout: float = 10) -> _MockResponse:
            fetch_count[0] += 1
            return _MockResponse(b"{}")

        with patch("peer.p2p_model_cache.urllib_request.urlopen", side_effect=fake_urlopen):
            result = leecher.resolve("test/model")

        assert result == leecher._cache_root / model_safe
        assert fetch_count[0] == 0, "No HTTP request should happen on cache hit"

    def test_download_from_peer_and_verify(self, tmp_path: Path) -> None:
        """Full P2P download: seeder → leecher with SHA-256 verification."""
        seeder_root = tmp_path / "seeder_cache"
        file_map = _make_fake_model_dir(
            seeder_root,
            "test/model",
            ["config.json", "model.safetensors"],
        )
        manifest = self._manifest_for(file_map)
        seeder = _start_seeder(seeder_root)
        peer = _make_mock_peer("127.0.0.1", seeder.port, "test/model")

        leecher = self._make_leecher(tmp_path, peers=[peer])

        def fake_fetch_manifest(model_id: str) -> HFManifest:
            return manifest

        try:
            leecher._fetch_hf_manifest = fake_fetch_manifest  # type: ignore[method-assign]
            result = leecher.resolve("test/model")
        finally:
            seeder.stop()

        assert result is not None
        assert result.is_dir()
        assert (result / "config.json").exists()
        assert (result / "model.safetensors").exists()
        # Verify the downloaded files match the expected SHA-256.
        for fname, expected_sha in file_map.items():
            assert _sha256_file(result / fname) == expected_sha

    def test_hash_mismatch_rejects_file(self, tmp_path: Path) -> None:
        """Wrong SHA-256 in manifest causes leecher to reject and return None."""
        seeder_root = tmp_path / "seeder_cache"
        file_map = _make_fake_model_dir(
            seeder_root,
            "test/model",
            ["model.safetensors"],
        )
        # Corrupt the manifest with a wrong SHA-256.
        bad_manifest = HFManifest(
            model_id="test/model",
            files={"model.safetensors": "deadbeef" * 8},
            fetched_at=time.time(),
        )
        seeder = _start_seeder(seeder_root)
        peer = _make_mock_peer("127.0.0.1", seeder.port, "test/model")
        leecher = self._make_leecher(tmp_path, peers=[peer])
        leecher._fetch_hf_manifest = lambda model_id: bad_manifest  # type: ignore[method-assign]

        try:
            result = leecher.resolve("test/model")
        finally:
            seeder.stop()

        assert result is None, "Hash mismatch should cause resolve() to return None"

    def test_resume_partial_download(self, tmp_path: Path) -> None:
        """Leecher resumes an existing .part file using Range requests."""
        seeder_root = tmp_path / "seeder_cache"
        model_safe = "test--model"
        full_data = os.urandom(8192)
        seeder_dir = seeder_root / model_safe
        seeder_dir.mkdir(parents=True)
        (seeder_dir / "model.safetensors").write_bytes(full_data)
        expected_sha = hashlib.sha256(full_data).hexdigest()

        manifest = HFManifest(
            model_id="test/model",
            files={"model.safetensors": expected_sha},
            fetched_at=time.time(),
        )

        seeder = _start_seeder(seeder_root)
        peer = _make_mock_peer("127.0.0.1", seeder.port, "test/model")
        leecher = self._make_leecher(tmp_path, peers=[peer])
        leecher._fetch_hf_manifest = lambda model_id: manifest  # type: ignore[method-assign]

        # Pre-create a .part file with the first 4096 bytes.
        leecher_model_dir = leecher._cache_root / model_safe
        leecher_model_dir.mkdir(parents=True)
        (leecher_model_dir / "model.safetensors.part").write_bytes(full_data[:4096])

        try:
            result = leecher.resolve("test/model")
        finally:
            seeder.stop()

        assert result is not None
        final_file = result / "model.safetensors"
        assert final_file.exists()
        assert final_file.read_bytes() == full_data

    def test_no_peers_returns_none(self, tmp_path: Path) -> None:
        """No seeder peers found → resolve() returns None immediately."""
        leecher = self._make_leecher(tmp_path, peers=[])
        manifest = HFManifest(
            model_id="test/model",
            files={"model.safetensors": "aa" * 32},
            fetched_at=time.time(),
        )
        leecher._fetch_hf_manifest = lambda model_id: manifest  # type: ignore[method-assign]
        result = leecher.resolve("test/model")
        assert result is None


# ---------------------------------------------------------------------------
# Group D: DHT announcement fields
# ---------------------------------------------------------------------------

class TestDHTAnnouncement:
    """Unit tests for the new seeder fields in Announcement and PeerEndpoint."""

    def test_announcement_seeder_fields_serialised(self) -> None:
        """Announcement with seeder fields round-trips through dataclasses.asdict()."""
        ann = Announcement(
            peer_id="test-peer",
            model_id="org/model",
            host="127.0.0.1",
            port=50051,
            seeder_http_port=9000,
            cached_model_ids=("org/model", "other/model"),
        )
        d = dataclasses.asdict(ann)
        assert d["seeder_http_port"] == 9000
        assert d["cached_model_ids"] == ("org/model", "other/model")

    def test_announcement_seeder_fields_default_off(self) -> None:
        """Seeder fields default to 0 / empty tuple (P2P disabled by default)."""
        ann = Announcement(
            peer_id="p", model_id="m", host="h", port=1
        )
        assert ann.seeder_http_port == 0
        assert ann.cached_model_ids == ()

    def test_peer_endpoint_parses_seeder_fields(self) -> None:
        """PeerEndpoint carries the seeder fields from DHT parsing."""
        ep = PeerEndpoint(
            peer_id="p",
            host="1.2.3.4",
            port=50051,
            seeder_http_port=8080,
            cached_model_ids=("foo/bar",),
        )
        assert ep.seeder_http_port == 8080
        assert "foo/bar" in ep.cached_model_ids

    def test_peer_endpoint_seeder_fields_default_off(self) -> None:
        """PeerEndpoint seeder fields default to 0 / empty — no breaking change."""
        ep = PeerEndpoint(peer_id="p", host="h", port=1)
        assert ep.seeder_http_port == 0
        assert ep.cached_model_ids == ()

    def test_announce_cached_models_lists_complete_dirs(self, tmp_path: Path) -> None:
        """announce_cached_models() returns model IDs for complete cache dirs only."""
        cache = P2PModelCache(
            cache_root=tmp_path,
            manifest_cache_dir=tmp_path / "manifests",
            dht_urls=[],
        )
        # Complete model dir (has a .safetensors file, no .part).
        complete_dir = tmp_path / "org--complete"
        complete_dir.mkdir()
        (complete_dir / "model.safetensors").write_bytes(b"weights")

        # Incomplete model dir (has a .part file).
        incomplete_dir = tmp_path / "org--incomplete"
        incomplete_dir.mkdir()
        (incomplete_dir / "model.safetensors.part").write_bytes(b"partial")

        # Empty dir (no weight files).
        empty_dir = tmp_path / "org--empty"
        empty_dir.mkdir()

        cached = cache.announce_cached_models()
        # Only the complete model should be listed.
        assert "org/complete" in cached
        assert "org/incomplete" not in cached
        assert "org/empty" not in cached

    def test_find_seeder_peers_filters_by_port(self, tmp_path: Path) -> None:
        """_find_seeder_peers returns only peers with seeder_http_port > 0."""
        cache = P2PModelCache(
            cache_root=tmp_path,
            manifest_cache_dir=tmp_path / "manifests",
            dht_urls=["http://fake-dht:8468"],
        )

        p_seeding = PeerEndpoint(peer_id="a", host="1.1.1.1", port=50051, seeder_http_port=9000)
        p_not_seeding = PeerEndpoint(peer_id="b", host="2.2.2.2", port=50051, seeder_http_port=0)

        with patch(
            "peer.p2p_model_cache.load_peers_from_dht",
            return_value=[p_seeding, p_not_seeding],
        ):
            seeders = cache._find_seeder_peers("test/model")

        assert len(seeders) == 1
        assert seeders[0].peer_id == "a"


# ---------------------------------------------------------------------------
# Sanity: _parse_range and _is_safe_path helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_parse_range_full(self) -> None:
        assert _parse_range("", 1000) == (None, None)

    def test_parse_range_open_end(self) -> None:
        start, end = _parse_range("bytes=500-", 1000)
        assert start == 500
        assert end == 999

    def test_parse_range_closed(self) -> None:
        start, end = _parse_range("bytes=0-99", 1000)
        assert start == 0
        assert end == 99

    def test_parse_range_suffix(self) -> None:
        start, end = _parse_range("bytes=-200", 1000)
        assert start == 800
        assert end == 999

    def test_parse_range_multi_range_returns_none(self) -> None:
        assert _parse_range("bytes=0-99,200-299", 1000) == (None, None)

    def test_is_safe_path_allows_subpath(self, tmp_path: Path) -> None:
        sub = tmp_path / "models" / "foo" / "bar.bin"
        assert _is_safe_path(tmp_path / "models", sub)

    def test_is_safe_path_blocks_parent(self, tmp_path: Path) -> None:
        parent = tmp_path.parent
        assert not _is_safe_path(tmp_path, parent)
