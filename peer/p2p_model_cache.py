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

"""P2P model cache for Phase 5 peer-to-peer model distribution.

Peers can download model weight files from each other instead of each hitting
HuggingFace independently.  Every downloaded file is SHA-256 verified against
the HuggingFace Hub API manifest before being moved to its final location,
so a malicious peer cannot poison the cache with corrupt weights.

Flow (P2PModelCache.resolve):
    1. Local cache hit? → return path immediately.
    2. Fetch HF manifest (24 h TTL on disk) to get expected SHA-256 per file.
    3. Discover seeder peers via DHT lookup (filter: seeder_http_port > 0).
    4. Try each peer: download all files → verify SHA-256 → atomic rename.
    5. If all peers fail → return None; caller falls back to HuggingFace Hub.

Thread-safety: per-model Lock prevents duplicate concurrent downloads.
Crash-safety: files are downloaded as *.part and renamed atomically on success.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib import request as urllib_request
from urllib.error import URLError
from urllib.request import Request

from coordinator.path_finder import load_peers_from_dht

logger = logging.getLogger(__name__)

# 24 hours between HF manifest re-fetches.
_MANIFEST_TTL_S: float = 86_400.0

# Download in 64 KB chunks.
_CHUNK_SIZE: int = 64 * 1024

# Model files we must find before declaring a cache directory "complete".
_MODEL_FILE_SUFFIXES = {".safetensors", ".bin", ".gguf"}


# ---------------------------------------------------------------------------
# HF Hub manifest
# ---------------------------------------------------------------------------

@dataclass
class HFManifest:
    """SHA-256 manifest fetched from the HuggingFace Hub API."""

    model_id: str
    files: dict[str, str]   # filename → sha256 hex
    fetched_at: float       # time.time() at fetch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    """Return hex-encoded SHA-256 digest of *path* (streaming, 1 MB chunks)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _model_id_to_safe(model_id: str) -> str:
    """Convert a HuggingFace model ID to a filesystem-safe directory name."""
    return model_id.replace("/", "--")


def _looks_complete(path: Path) -> bool:
    """
    Return True if *path* looks like a complete, usable model directory.

    Criteria:
      - No *.part temporary files (would indicate an interrupted download).
      - At least one model weight file (*.safetensors, *.bin, *.gguf) OR
        a config.json (for tokenizer-only or mlx-format checkpoints).
    """
    if not path.is_dir():
        return False
    has_weight = False
    for child in path.iterdir():
        if child.suffix == ".part":
            return False  # incomplete download
        if child.suffix in _MODEL_FILE_SUFFIXES or child.name == "config.json":
            has_weight = True
    return has_weight


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class P2PModelCache:
    """
    Resolve HuggingFace model IDs to locally-cached paths via P2P download.

    Args:
        cache_root: Root directory for P2P-downloaded model files.
            Each model occupies ``cache_root/{model_id_safe}/``.
        manifest_cache_dir: Directory for JSON HF manifest cache files (24 h TTL).
        dht_urls: DHT bootstrap node URLs used to discover seeder peers.
        request_timeout_s: HTTP timeout for peer downloads and HF API calls.
        hf_api_base: HuggingFace Hub API base URL (overridable for tests).
        dht_lookup_timeout_s: Per-node timeout for DHT peer discovery lookups.
    """

    def __init__(
        self,
        cache_root: str | Path,
        manifest_cache_dir: str | Path,
        dht_urls: list[str],
        request_timeout_s: float = 30.0,
        hf_api_base: str = "https://huggingface.co/api/models",
        dht_lookup_timeout_s: float = 3.0,
    ) -> None:
        self._cache_root = Path(cache_root)
        self._manifest_cache_dir = Path(manifest_cache_dir)
        self._dht_urls = list(dht_urls)
        self._request_timeout_s = float(request_timeout_s)
        self._hf_api_base = hf_api_base.rstrip("/")
        self._dht_lookup_timeout_s = float(dht_lookup_timeout_s)
        # Per-model locks to prevent duplicate concurrent downloads.
        self._model_locks: dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, model_id: str) -> Path | None:
        """
        Try to resolve *model_id* to a locally-cached model directory.

        Returns the directory path on success, or ``None`` when no peer can
        serve the model (caller should fall back to HuggingFace Hub download).
        """
        lock = self._get_model_lock(model_id)
        with lock:
            return self._resolve_locked(model_id)

    def announce_cached_models(self) -> list[str]:
        """
        Return a list of model IDs whose files are fully cached in *cache_root*.

        Called every announce cycle to populate :attr:`Announcement.cached_model_ids`.
        """
        self._cache_root.mkdir(parents=True, exist_ok=True)
        result: list[str] = []
        for child in self._cache_root.iterdir():
            if not child.is_dir():
                continue
            if not _looks_complete(child):
                continue
            # Reverse the safe-name encoding: "--" → "/"
            model_id = child.name.replace("--", "/", 1)
            result.append(model_id)
        return sorted(result)

    # ------------------------------------------------------------------
    # Internal: resolve (called under per-model lock)
    # ------------------------------------------------------------------

    def _resolve_locked(self, model_id: str) -> Path | None:
        local_path = self._cache_root / _model_id_to_safe(model_id)

        # Step 1: local cache hit.
        if _looks_complete(local_path):
            logger.info("p2p_cache_hit model=%s path=%s", model_id, local_path)
            return local_path

        # Step 2: fetch HF manifest for SHA-256 verification.
        manifest = self._fetch_hf_manifest(model_id)
        if manifest is None:
            logger.warning(
                "p2p_manifest_unavailable model=%s; falling back to HF Hub", model_id
            )
            return None

        if not manifest.files:
            logger.warning("p2p_manifest_empty model=%s; falling back to HF Hub", model_id)
            return None

        # Step 3: discover seeder peers.
        peers = self._find_seeder_peers(model_id)
        if not peers:
            logger.info("p2p_no_seeders model=%s; falling back to HF Hub", model_id)
            return None

        # Step 4: try each peer (shuffled for load distribution).
        random.shuffle(peers)
        for peer in peers:
            logger.info(
                "p2p_attempting_download model=%s peer=%s:%d",
                model_id,
                peer.host,
                peer.seeder_http_port,
            )
            try:
                ok = self._download_from_peer(peer, model_id, manifest, local_path)
            except Exception as exc:
                logger.warning(
                    "p2p_peer_download_error model=%s peer=%s:%d error=%s",
                    model_id,
                    peer.host,
                    peer.seeder_http_port,
                    exc,
                )
                ok = False
            if ok:
                logger.info("p2p_download_complete model=%s path=%s", model_id, local_path)
                return local_path

        # Step 5: all peers failed.
        logger.warning("p2p_all_peers_failed model=%s; falling back to HF Hub", model_id)
        return None

    # ------------------------------------------------------------------
    # HF manifest
    # ------------------------------------------------------------------

    def _fetch_hf_manifest(self, model_id: str) -> HFManifest | None:
        """
        Return the HF Hub manifest for *model_id*, using a 24 h disk cache.
        Returns None when the HF API is unreachable.
        """
        self._manifest_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self._manifest_cache_dir / f"{_model_id_to_safe(model_id)}.json"

        # Check disk cache.
        if cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                fetched_at = float(cached.get("fetched_at", 0.0))
                if time.time() - fetched_at < _MANIFEST_TTL_S:
                    return HFManifest(
                        model_id=model_id,
                        files=dict(cached.get("files", {})),
                        fetched_at=fetched_at,
                    )
            except Exception:
                pass  # stale or corrupt cache → re-fetch

        # Fetch from HF API.
        url = f"{self._hf_api_base}/{model_id}"
        try:
            req = Request(url, headers={"User-Agent": "openhydra-p2p/1.0"})
            with urllib_request.urlopen(req, timeout=self._request_timeout_s) as resp:
                data: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        except URLError as exc:
            logger.warning("p2p_hf_api_error model=%s url=%s error=%s", model_id, url, exc)
            return None
        except Exception as exc:
            logger.warning("p2p_hf_api_error model=%s url=%s error=%s", model_id, url, exc)
            return None

        siblings = data.get("siblings", [])
        files: dict[str, str] = {}
        for sibling in siblings:
            fname = str(sibling.get("rfilename", "")).strip()
            sha = str(sibling.get("sha256", "")).strip()
            if fname and sha:
                files[fname] = sha

        now = time.time()
        manifest = HFManifest(model_id=model_id, files=files, fetched_at=now)

        # Persist to disk cache.
        try:
            cache_file.write_text(
                json.dumps({"fetched_at": now, "files": files}, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("p2p_manifest_cache_write_error path=%s error=%s", cache_file, exc)

        return manifest

    # ------------------------------------------------------------------
    # Peer discovery
    # ------------------------------------------------------------------

    def _find_seeder_peers(self, model_id: str) -> list[Any]:
        """
        Look up peers from DHT that serve *model_id* AND have a seeder port.

        Returns PeerEndpoint objects with seeder_http_port > 0.
        """
        if not self._dht_urls:
            return []
        try:
            peers = load_peers_from_dht(
                model_id=model_id,
                dht_urls=self._dht_urls,
                timeout_s=self._dht_lookup_timeout_s,
            )
            return [p for p in peers if p.seeder_http_port > 0]
        except Exception as exc:
            logger.warning("p2p_peer_discovery_failed model=%s error=%s", model_id, exc)
            return []

    # ------------------------------------------------------------------
    # Download from peer
    # ------------------------------------------------------------------

    def _download_from_peer(
        self,
        peer: Any,
        model_id: str,
        manifest: HFManifest,
        dest_path: Path,
    ) -> bool:
        """
        Download all manifest files from *peer* into *dest_path*.

        Files are downloaded as ``.part`` temporaries and atomically renamed
        on SHA-256 verification success.  Supports resume via HTTP Range.

        Returns True iff all files downloaded and verified successfully.
        """
        dest_path.mkdir(parents=True, exist_ok=True)
        model_safe = _model_id_to_safe(model_id)

        for filename, expected_sha256 in manifest.files.items():
            final_path = dest_path / filename
            part_path = dest_path / (filename + ".part")

            # Skip files that are already verified.
            if final_path.exists():
                actual = _sha256_file(final_path)
                if actual == expected_sha256:
                    continue
                # Corrupt / stale final file — re-download.
                try:
                    final_path.unlink()
                except OSError:
                    pass

            # Resume from existing .part file.
            resume_offset = 0
            if part_path.exists():
                resume_offset = part_path.stat().st_size

            url = (
                f"http://{peer.host}:{peer.seeder_http_port}"
                f"/v1/p2p/models/{model_safe}/{filename}"
            )
            headers: dict[str, str] = {}
            if resume_offset > 0:
                headers["Range"] = f"bytes={resume_offset}-"

            try:
                req = Request(url, headers=headers)
                with urllib_request.urlopen(req, timeout=self._request_timeout_s) as resp:
                    mode = "ab" if resume_offset > 0 else "wb"
                    with open(part_path, mode) as fh:
                        while True:
                            chunk = resp.read(_CHUNK_SIZE)
                            if not chunk:
                                break
                            fh.write(chunk)
            except URLError as exc:
                logger.warning(
                    "p2p_download_network_error model=%s file=%s peer=%s:%d error=%s",
                    model_id,
                    filename,
                    peer.host,
                    peer.seeder_http_port,
                    exc,
                )
                return False
            except Exception as exc:
                logger.warning(
                    "p2p_download_error model=%s file=%s peer=%s:%d error=%s",
                    model_id,
                    filename,
                    peer.host,
                    peer.seeder_http_port,
                    exc,
                )
                return False

            # Verify SHA-256.
            actual_sha256 = _sha256_file(part_path)
            if actual_sha256 != expected_sha256:
                logger.warning(
                    "p2p_hash_mismatch model=%s file=%s expected=%s actual=%s peer=%s:%d",
                    model_id,
                    filename,
                    expected_sha256,
                    actual_sha256,
                    peer.host,
                    peer.seeder_http_port,
                )
                try:
                    part_path.unlink()
                except OSError:
                    pass
                return False

            # Atomic rename: .part → final.
            try:
                os.replace(str(part_path), str(final_path))
            except OSError as exc:
                logger.warning(
                    "p2p_rename_error src=%s dst=%s error=%s", part_path, final_path, exc
                )
                return False

        return True

    # ------------------------------------------------------------------
    # Locking helpers
    # ------------------------------------------------------------------

    def _get_model_lock(self, model_id: str) -> threading.Lock:
        with self._locks_lock:
            if model_id not in self._model_locks:
                self._model_locks[model_id] = threading.Lock()
            return self._model_locks[model_id]
