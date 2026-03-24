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

"""KV-cache affinity service.

Manages session-based KV-cache affinity mappings so that repeat requests
for the same (session_id, model_id) pair are routed to the peer that
already holds the prefill KV-cache, avoiding redundant recomputation.
"""

from __future__ import annotations

import time
from typing import Any


class KvAffinityService:
    """Extracted from ``CoordinatorEngine`` -- owns the ``_kv_affinity`` dict
    and every method that reads / writes it.

    Parameters
    ----------
    config:
        An ``EngineConfig`` (or duck-typed equivalent) that exposes at least
        ``kv_affinity_enabled: bool`` and ``kv_affinity_ttl_seconds: int``.
    kv_affinity:
        The shared mutable dict keyed by ``(session_id, model_id)`` tuples.
    """

    def __init__(self, config: Any, kv_affinity: dict[tuple[str, str], dict[str, Any]]):
        self.config = config
        self._kv_affinity = kv_affinity

    # ------------------------------------------------------------------
    # Key helper
    # ------------------------------------------------------------------

    def _kv_affinity_key(self, session_id: str, model_id: str) -> tuple[str, str]:
        return session_id, model_id

    # ------------------------------------------------------------------
    # Expiry
    # ------------------------------------------------------------------

    def _purge_expired_kv_affinity(self) -> None:
        """Remove all KV affinity entries whose TTL has expired."""
        if not self._kv_affinity:
            return
        now = time.time()
        expired = [
            key
            for key, item in self._kv_affinity.items()
            if float(item.get("expires_at", 0.0)) < now
        ]
        for key in expired:
            self._kv_affinity.pop(key, None)

    # ------------------------------------------------------------------
    # Peer affinity
    # ------------------------------------------------------------------

    def _get_kv_affinity_peer(self, session_id: str | None, model_id: str) -> str | None:
        """Look up the preferred prefill peer for a session/model pair.

        Args:
            session_id: The client session identifier (``None`` disables lookup).
            model_id: The model being requested.

        Returns:
            The peer ID that holds the KV cache, or ``None`` if no affinity
            exists or the feature is disabled.
        """
        if not self.config.kv_affinity_enabled or not session_id:
            return None
        self._purge_expired_kv_affinity()
        item = self._kv_affinity.get(self._kv_affinity_key(session_id, model_id))
        if item is None:
            return None
        peer_id = str(item.get("prefill_peer_id", "")).strip()
        return peer_id or None

    def _set_kv_affinity_peer(self, session_id: str | None, model_id: str, peer_id: str | None) -> bool:
        """Record or update the prefill peer affinity for a session/model pair.

        Preserves any existing activation cache entry while refreshing the TTL.

        Args:
            session_id: The client session identifier.
            model_id: The model being served.
            peer_id: The peer that now holds the KV cache.

        Returns:
            ``True`` if the affinity was stored, ``False`` if skipped.
        """
        if not self.config.kv_affinity_enabled or not session_id or not peer_id:
            return False
        now = time.time()
        ttl = max(1, int(self.config.kv_affinity_ttl_seconds))
        key = self._kv_affinity_key(session_id, model_id)
        previous = self._kv_affinity.get(key, {})
        activation_cache = previous.get("activation")
        activation_peer_id = previous.get("activation_peer_id")
        self._kv_affinity[key] = {
            "prefill_peer_id": peer_id,
            "updated_at": now,
            "expires_at": now + ttl,
            "activation": activation_cache,
            "activation_peer_id": activation_peer_id,
            "activation_updated_at": previous.get("activation_updated_at"),
        }
        return True

    # ------------------------------------------------------------------
    # Activation affinity
    # ------------------------------------------------------------------

    def _get_kv_affinity_activation(self, session_id: str | None, model_id: str) -> list[float] | None:
        """Retrieve the cached activation seed for a session/model pair.

        Args:
            session_id: The client session identifier.
            model_id: The model being requested.

        Returns:
            A list of floats representing the cached activation, or ``None``
            if unavailable.
        """
        if not self.config.kv_affinity_enabled or not session_id:
            return None
        self._purge_expired_kv_affinity()
        item = self._kv_affinity.get(self._kv_affinity_key(session_id, model_id))
        if item is None:
            return None
        raw = item.get("activation")
        if not isinstance(raw, list) or not raw:
            return None
        out: list[float] = []
        for value in raw:
            try:
                out.append(float(value))
            except (TypeError, ValueError):
                return None
        return out or None

    def _get_kv_affinity_activation_peer(self, session_id: str | None, model_id: str) -> str | None:
        """Return the peer ID that produced the cached activation for this session.

        Args:
            session_id: The client session identifier.
            model_id: The model being requested.

        Returns:
            Peer ID string, or ``None`` if no activation cache exists.
        """
        if not self.config.kv_affinity_enabled or not session_id:
            return None
        self._purge_expired_kv_affinity()
        item = self._kv_affinity.get(self._kv_affinity_key(session_id, model_id))
        if item is None:
            return None
        peer_id = str(item.get("activation_peer_id", "")).strip()
        return peer_id or None

    def _set_kv_affinity_activation(
        self,
        session_id: str | None,
        model_id: str,
        activation: list[float] | None,
    ) -> bool:
        """Store an activation seed in the KV affinity cache.

        Updates the entry's TTL and records the producing peer so that
        cross-peer relay can be detected on subsequent lookups.

        Args:
            session_id: The client session identifier.
            model_id: The model being served.
            activation: The activation vector to cache.

        Returns:
            ``True`` if stored, ``False`` if skipped.
        """
        if not self.config.kv_affinity_enabled or not session_id or not activation:
            return False
        key = self._kv_affinity_key(session_id, model_id)
        now = time.time()
        ttl = max(1, int(self.config.kv_affinity_ttl_seconds))
        item = dict(self._kv_affinity.get(key, {}))
        item["activation"] = [float(v) for v in activation]
        item["activation_updated_at"] = now
        item["activation_peer_id"] = str(item.get("prefill_peer_id", "")).strip() or None
        item["updated_at"] = now
        item["expires_at"] = now + ttl
        self._kv_affinity[key] = item
        return True
