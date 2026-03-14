"""dht.hivemind_bridge — dual-stack adapter for Hivemind Kademlia DHT.

Wraps ``hivemind.DHT`` to provide announce/lookup operations compatible with
OpenHydra's peer discovery protocol.  When ``hivemind`` is not installed, all
operations gracefully degrade to no-ops with logged warnings.

Key design
----------
* **key** = ``model_id`` (e.g. ``"openhydra-qwen3.5-0.8b"``)
* **subkey** = ``peer_id`` (e.g. ``"hxp-…"``)
* **value** = announcement dict (same as HTTP DHT payload)

This lets the coordinator retrieve all peers for a model via a single
Hivemind ``get()`` call and merge results with the legacy HTTP DHT.

Usage
-----
::

    adapter = HivemindDHTAdapter(initial_peers=["/ip4/…/tcp/…/p2p/…"])
    adapter.start()
    adapter.announce(announcement_dict, ttl_seconds=300)
    peers = adapter.lookup(model_id="openhydra-qwen3.5-0.8b")
    adapter.shutdown()
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from typing import Any

logger = logging.getLogger(__name__)

# Guard: hivemind is an optional dependency.
_hivemind_available = False
try:
    import hivemind
    _hivemind_available = True
except ImportError:
    hivemind = None  # type: ignore[assignment]


def hivemind_available() -> bool:
    """Return ``True`` if the ``hivemind`` package is importable."""
    return _hivemind_available


class HivemindDHTAdapter:
    """Dual-stack adapter wrapping ``hivemind.DHT``.

    Parameters
    ----------
    initial_peers:
        List of Hivemind multiaddr strings for bootstrap nodes, e.g.
        ``["/ip4/172.105.69.49/tcp/38751/p2p/QmXyz…"]``.
    host_maddrs:
        Multiaddr(s) to bind the local Hivemind node to.  Defaults to
        ``["/ip4/0.0.0.0/tcp/0"]`` (OS-assigned port).
    start:
        If ``True`` (default), start the DHT node immediately.

    When ``hivemind`` is not installed, the adapter operates in **stub mode**:
    ``announce()`` and ``lookup()`` are no-ops that log warnings.
    """

    def __init__(
        self,
        initial_peers: list[str] | None = None,
        host_maddrs: list[str] | None = None,
        start: bool = True,
    ) -> None:
        self._initial_peers = list(initial_peers or [])
        self._host_maddrs = list(host_maddrs or ["/ip4/0.0.0.0/tcp/0"])
        self._dht: Any = None
        self._started = False
        if start:
            self.start()

    def start(self) -> bool:
        """Start the underlying Hivemind DHT node.

        Returns ``True`` on success, ``False`` if hivemind is unavailable.
        """
        if self._started:
            return True

        if not _hivemind_available:
            logger.warning("hivemind_bridge: hivemind not installed — stub mode")
            return False

        try:
            self._dht = hivemind.DHT(
                initial_peers=self._initial_peers or None,
                host_maddrs=self._host_maddrs,
                start=True,
            )
            self._started = True
            logger.info(
                "hivemind_bridge: started peer_id=%s",
                getattr(self._dht, "peer_id", "unknown"),
            )
            return True
        except Exception as exc:
            logger.warning("hivemind_bridge: failed to start: %s", exc)
            return False

    @property
    def is_alive(self) -> bool:
        return self._started and self._dht is not None

    @property
    def peer_id(self) -> str:
        """Return the Hivemind peer ID (libp2p identity), or empty string."""
        if self._dht is None:
            return ""
        return str(getattr(self._dht, "peer_id", ""))

    def announce(
        self,
        announcement: dict[str, Any],
        ttl_seconds: int = 300,
    ) -> bool:
        """Store a peer announcement in the Hivemind DHT.

        Uses ``dht.store(key=model_id, subkey=peer_id, value=json)``
        with Hivemind's subkey mechanism for multi-peer-per-key storage.

        Parameters
        ----------
        announcement: Dict with at least ``model_id`` and ``peer_id`` keys.
        ttl_seconds:  Time-to-live in the DHT.

        Returns ``True`` on success.
        """
        if not self._started or self._dht is None:
            logger.debug("hivemind_bridge: announce skipped — not started")
            return False

        model_id = str(announcement.get("model_id", "")).strip()
        peer_id = str(announcement.get("peer_id", "")).strip()
        if not model_id or not peer_id:
            logger.debug("hivemind_bridge: announce skipped — missing model/peer id")
            return False

        # Ensure updated_unix_ms is set.
        payload = dict(announcement)
        if "updated_unix_ms" not in payload:
            payload["updated_unix_ms"] = int(time.time() * 1000)

        try:
            value_json = json.dumps(payload)
            success = self._dht.store(
                key=model_id,
                subkey=peer_id,
                value=value_json,
                expiration_time=hivemind.get_dht_time() + float(ttl_seconds),
            )
            if success:
                logger.debug(
                    "hivemind_announce: model=%s peer=%s ttl=%ds",
                    model_id, peer_id, ttl_seconds,
                )
            return bool(success)
        except Exception as exc:
            logger.debug("hivemind_announce_error: %s", exc)
            return False

    def lookup(
        self,
        model_id: str,
        timeout_s: float = 5.0,
    ) -> list[dict[str, Any]]:
        """Retrieve all peer announcements for *model_id* from the Hivemind DHT.

        Returns a list of announcement dicts (one per peer).
        """
        if not self._started or self._dht is None:
            return []

        try:
            result = self._dht.get(
                key=model_id,
                latest=True,
            )
            if result is None:
                return []

            # Result is a DHTValue or dict of subkeys → (value, expiration).
            # Hivemind returns: {subkey: (value_bytes, expiration_time)}
            peers: list[dict[str, Any]] = []
            if isinstance(result, dict):
                entries = result
            elif hasattr(result, "value") and isinstance(result.value, dict):
                entries = result.value
            else:
                return []

            for subkey, entry in entries.items():
                try:
                    if isinstance(entry, tuple) and len(entry) >= 1:
                        value_str = entry[0]
                    else:
                        value_str = entry
                    if isinstance(value_str, bytes):
                        value_str = value_str.decode("utf-8")
                    peer_dict = json.loads(str(value_str))
                    if isinstance(peer_dict, dict):
                        peers.append(peer_dict)
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    logger.debug("hivemind_lookup_parse_error: subkey=%s err=%s", subkey, exc)

            return peers
        except Exception as exc:
            logger.debug("hivemind_lookup_error: model=%s err=%s", model_id, exc)
            return []

    def shutdown(self) -> None:
        """Shut down the Hivemind DHT node."""
        if self._dht is not None:
            try:
                self._dht.shutdown()
            except Exception:
                pass
        self._dht = None
        self._started = False
        logger.info("hivemind_bridge: shutdown")


def merge_peer_lists(
    http_peers: list[dict[str, Any]],
    hivemind_peers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge peer announcements from HTTP DHT and Hivemind DHT.

    Deduplicates by ``peer_id``, keeping the entry with the newest
    ``updated_unix_ms`` timestamp.

    Parameters
    ----------
    http_peers:     Peer dicts from the legacy HTTP DHT.
    hivemind_peers: Peer dicts from the Hivemind DHT.

    Returns
    -------
    Merged list of peer dicts, sorted by ``updated_unix_ms`` descending.
    """
    best: dict[str, dict[str, Any]] = {}

    for peer_dict in list(http_peers) + list(hivemind_peers):
        peer_id = str(peer_dict.get("peer_id", "")).strip()
        if not peer_id:
            continue
        existing = best.get(peer_id)
        existing_ts = int((existing or {}).get("updated_unix_ms", 0) or 0)
        incoming_ts = int(peer_dict.get("updated_unix_ms", 0) or 0)
        if existing is None or incoming_ts >= existing_ts:
            best[peer_id] = dict(peer_dict)

    return sorted(
        best.values(),
        key=lambda d: int(d.get("updated_unix_ms", 0) or 0),
        reverse=True,
    )
