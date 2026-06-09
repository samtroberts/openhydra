from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .manifest import (
    SupernodeManifest,
    normalize_model_id,
    MANIFEST_TTL_MS,
)

logger = logging.getLogger(__name__)

CACHE_TTL_S = 120


@dataclass
class CachedManifest:
    manifest: SupernodeManifest
    cached_at: float = field(default_factory=time.monotonic)

    def is_stale(self) -> bool:
        return (time.monotonic() - self.cached_at) > CACHE_TTL_S


class SupernodeDiscovery:
    """Discovers supernodes by model ID using a manifest cache.

    MVP: manifests are registered locally (single-node or from
    the SupernodeRouter's adapters). Phase 1d adds DHT-backed
    GET_PROVIDERS lookups via the Rust libp2p layer.
    """

    def __init__(self):
        self._manifests: dict[str, CachedManifest] = {}
        self._lock = threading.Lock()
        self._model_index: dict[str, set[str]] = {}

    def register_manifest(self, manifest: SupernodeManifest) -> bool:
        if not manifest.verify_signature():
            logger.warning(
                "supernode_manifest_rejected peer=%s reason=bad_signature",
                manifest.peer_id,
            )
            return False

        now_ms = int(time.time() * 1000)
        if not manifest.is_fresh(now_ms):
            logger.warning(
                "supernode_manifest_rejected peer=%s reason=stale ts=%d now=%d",
                manifest.peer_id, manifest.timestamp, now_ms,
            )
            return False

        with self._lock:
            self._manifests[manifest.libp2p_peer_id] = CachedManifest(manifest=manifest)
            for model in manifest.models:
                norm = normalize_model_id(model.model_id)
                if norm not in self._model_index:
                    self._model_index[norm] = set()
                self._model_index[norm].add(manifest.libp2p_peer_id)

        logger.info(
            "supernode_manifest_registered peer=%s models=%s",
            manifest.peer_id, manifest.model_ids(),
        )
        return True

    def remove_manifest(self, libp2p_peer_id: str) -> None:
        with self._lock:
            cached = self._manifests.pop(libp2p_peer_id, None)
            if cached:
                for model in cached.manifest.models:
                    norm = normalize_model_id(model.model_id)
                    peers = self._model_index.get(norm)
                    if peers:
                        peers.discard(libp2p_peer_id)
                        if not peers:
                            del self._model_index[norm]

    def discover_supernodes(self, model_id: str) -> list[SupernodeManifest]:
        norm = normalize_model_id(model_id)
        results: list[SupernodeManifest] = []

        with self._lock:
            peer_ids = self._model_index.get(norm, set()).copy()

        for pid in peer_ids:
            with self._lock:
                cached = self._manifests.get(pid)
            if cached is None:
                continue
            if cached.is_stale():
                self.remove_manifest(pid)
                continue
            if not cached.manifest.is_fresh():
                self.remove_manifest(pid)
                continue
            results.append(cached.manifest)

        return results

    def all_manifests(self) -> list[SupernodeManifest]:
        with self._lock:
            entries = list(self._manifests.values())
        return [
            e.manifest for e in entries
            if not e.is_stale() and e.manifest.is_fresh()
        ]

    def known_models(self) -> list[str]:
        models: set[str] = set()
        for m in self.all_manifests():
            for cap in m.models:
                models.add(cap.model_id)
        return sorted(models)

    def prune_stale(self) -> int:
        with self._lock:
            stale = [
                pid for pid, cached in self._manifests.items()
                if cached.is_stale() or not cached.manifest.is_fresh()
            ]
        for pid in stale:
            self.remove_manifest(pid)
        return len(stale)
