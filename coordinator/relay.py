# Copyright 2026 OpenHydra contributors — Apache 2.0

"""gRPC relay for NAT-traversal — Petals parity Phase C.

Peers behind NAT cannot accept inbound gRPC connections.  The relay
runs on a public-IP node (e.g. the DHT bootstrap server) and proxies
``Forward`` calls between the coordinator and NATted peers.

Architecture::

    1. NATted peer connects OUTBOUND to the relay via ``register_relay()``.
       This establishes a persistent gRPC channel that the relay holds.
    2. The peer announces ``relay_address=<relay_host>:<relay_port>`` and
       ``requires_relay=True`` in its DHT announcement.
    3. When the coordinator needs to call ``Forward()`` on this peer, it
       routes to the relay address.  The relay looks up the peer's
       registered channel and proxies the request.

The relay is intentionally simple — it's just a dict of
``peer_id → outbound_channel`` with a ``Forward()`` proxy method.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RelayRegistration:
    """A registered NATted peer."""
    peer_id: str
    channel: Any  # gRPC channel to the peer (outbound from peer side)
    stub: Any     # PeerStub on that channel
    registered_at: float = 0.0
    last_heartbeat: float = 0.0
    model_id: str = ""


class RelayServer:
    """Manages registrations from NATted peers and proxies gRPC calls.

    The relay itself exposes a ``Forward()`` gRPC handler (via the
    standard ``PeerService``) that looks up the target peer's
    registered channel and forwards the request.

    Args:
        max_registrations: Maximum number of concurrent NATted peers.
        heartbeat_timeout_s: Seconds before a registration expires.
    """

    def __init__(
        self,
        max_registrations: int = 1024,
        heartbeat_timeout_s: float = 300.0,
    ) -> None:
        self.max_registrations = max_registrations
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self._peers: dict[str, RelayRegistration] = {}
        self._lock = threading.Lock()

    def register(
        self,
        peer_id: str,
        channel: Any,
        stub: Any,
        model_id: str = "",
    ) -> bool:
        """Register a NATted peer's outbound channel.

        Called when a peer behind NAT connects to the relay.

        Returns:
            True if registration succeeded, False if at capacity.
        """
        with self._lock:
            self._reap_expired()
            if len(self._peers) >= self.max_registrations and peer_id not in self._peers:
                logger.warning("relay_full: max=%d peer=%s rejected", self.max_registrations, peer_id)
                return False
            now = time.monotonic()
            self._peers[peer_id] = RelayRegistration(
                peer_id=peer_id,
                channel=channel,
                stub=stub,
                registered_at=now,
                last_heartbeat=now,
                model_id=model_id,
            )
        logger.info("relay_registered: peer=%s model=%s", peer_id, model_id)
        return True

    def heartbeat(self, peer_id: str) -> bool:
        """Update the heartbeat timestamp for a registered peer."""
        with self._lock:
            reg = self._peers.get(peer_id)
            if reg is None:
                return False
            reg.last_heartbeat = time.monotonic()
            return True

    def unregister(self, peer_id: str) -> None:
        """Remove a peer's registration."""
        with self._lock:
            reg = self._peers.pop(peer_id, None)
        if reg is not None:
            try:
                reg.channel.close()
            except Exception:
                pass
            logger.info("relay_unregistered: peer=%s", peer_id)

    def forward_to_peer(self, peer_id: str, request: Any, timeout_s: float = 60.0) -> Any:
        """Forward a ForwardRequest to a registered NATted peer.

        Args:
            peer_id: The target peer.
            request: A ForwardRequest protobuf message.
            timeout_s: gRPC timeout.

        Returns:
            The ForwardResponse from the peer.

        Raises:
            RuntimeError: If the peer is not registered.
        """
        with self._lock:
            reg = self._peers.get(peer_id)
        if reg is None:
            raise RuntimeError(f"relay_peer_not_found: {peer_id}")
        try:
            response = reg.stub.Forward(request, timeout=timeout_s)
            with self._lock:
                reg.last_heartbeat = time.monotonic()
            return response
        except Exception as exc:
            logger.warning("relay_forward_failed: peer=%s err=%s", peer_id, exc)
            raise RuntimeError(f"relay_forward_failed: {peer_id}: {exc}") from exc

    def registered_peers(self) -> list[str]:
        """Return list of currently registered peer IDs."""
        with self._lock:
            self._reap_expired()
            return list(self._peers.keys())

    def _reap_expired(self) -> int:
        """Remove registrations that have not heartbeated recently."""
        now = time.monotonic()
        expired = [
            pid for pid, reg in self._peers.items()
            if now - reg.last_heartbeat > self.heartbeat_timeout_s
        ]
        for pid in expired:
            reg = self._peers.pop(pid, None)
            if reg:
                try:
                    reg.channel.close()
                except Exception:
                    pass
                logger.info("relay_expired: peer=%s", pid)
        return len(expired)


def connect_to_relay(
    relay_address: str,
    peer_id: str,
    grpc_port: int,
    model_id: str = "",
) -> tuple[Any, str]:
    """Connect to a relay node as a NATted peer.

    Establishes an outbound gRPC channel to the relay and registers
    this peer.  The relay can then proxy incoming Forward requests
    through this channel.

    Args:
        relay_address: "host:port" of the relay node.
        peer_id: This peer's identifier.
        grpc_port: This peer's local gRPC port (for the relay to call back).
        model_id: Model being served.

    Returns:
        (channel, relay_peer_id) — the channel to keep alive and the
        relay's peer_id for announcement.
    """
    import grpc
    from peer import peer_pb2, peer_pb2_grpc

    channel = grpc.insecure_channel(
        relay_address,
        options=[
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
            ("grpc.keepalive_time_ms", 30000),
            ("grpc.keepalive_timeout_ms", 10000),
        ],
    )
    stub = peer_pb2_grpc.PeerStub(channel)

    # Ping the relay to verify connectivity
    try:
        ping = stub.Ping(
            peer_pb2.PingRequest(sent_unix_ms=int(time.time() * 1000)),
            timeout=10.0,
        )
        relay_peer_id = str(ping.peer_id)
    except Exception as exc:
        channel.close()
        raise RuntimeError(f"relay_connect_failed: {relay_address}: {exc}") from exc

    logger.info(
        "relay_connected: peer=%s relay=%s relay_peer=%s",
        peer_id, relay_address, relay_peer_id,
    )
    return channel, relay_peer_id
