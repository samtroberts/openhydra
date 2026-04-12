# Copyright 2026 OpenHydra contributors — Apache 2.0

"""STUN client for NAT type detection — Petals parity Phase C.

Implements a minimal RFC 5389 STUN Binding Request to discover the
peer's external IP/port and classify the NAT type.  Uses raw UDP
sockets — no external dependencies.

NAT classification:
- ``open``: Peer has a public IP, directly reachable.
- ``full_cone``: Peer is behind NAT but the mapping is consistent.
  Hole-punching may work.
- ``restricted``: Port-restricted cone NAT.  Relay recommended.
- ``symmetric``: Different mapping per destination.  Relay required.
- ``unknown``: STUN probe failed or timed out.

Usage::

    profile = probe_nat()
    if profile.requires_relay:
        # Connect to a relay node
"""

from __future__ import annotations

import logging
import os
import socket
import struct
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Public STUN servers (UDP port 3478)
_DEFAULT_STUN_SERVERS = [
    ("stun.l.google.com", 19302),
    ("stun1.l.google.com", 19302),
    ("stun.cloudflare.com", 3478),
]

# STUN message constants (RFC 5389)
_STUN_BINDING_REQUEST = 0x0001
_STUN_BINDING_RESPONSE = 0x0101
_STUN_MAGIC_COOKIE = 0x2112A442
_ATTR_XOR_MAPPED_ADDRESS = 0x0020
_ATTR_MAPPED_ADDRESS = 0x0001


@dataclass(frozen=True)
class NatProfile:
    """Result of a NAT probe.

    Attributes:
        reachable: Whether the peer appears reachable from the internet.
        nat_type: One of ``open``, ``full_cone``, ``restricted``,
            ``symmetric``, ``unknown``.
        external_ip: The peer's external IP as seen by the STUN server.
        external_port: The peer's external port.
        requires_relay: Whether the peer should use a relay node.
    """
    reachable: bool
    nat_type: str
    external_ip: str = ""
    external_port: int = 0
    requires_relay: bool = False


def _build_binding_request() -> tuple[bytes, bytes]:
    """Build a STUN Binding Request and return (packet, transaction_id)."""
    txn_id = os.urandom(12)  # 96-bit transaction ID
    # Header: type (2) + length (2) + magic cookie (4) + txn_id (12) = 20 bytes
    header = struct.pack(
        "!HHI",
        _STUN_BINDING_REQUEST,
        0,  # message length (no attributes)
        _STUN_MAGIC_COOKIE,
    )
    return header + txn_id, txn_id


def _parse_binding_response(data: bytes, txn_id: bytes) -> tuple[str, int] | None:
    """Parse a STUN Binding Response and extract the mapped address.

    Returns ``(ip, port)`` or ``None`` on failure.
    """
    if len(data) < 20:
        return None
    msg_type, msg_len, cookie = struct.unpack("!HHI", data[:8])
    if msg_type != _STUN_BINDING_RESPONSE:
        return None
    if cookie != _STUN_MAGIC_COOKIE:
        return None
    resp_txn = data[8:20]
    if resp_txn != txn_id:
        return None

    # Parse attributes
    offset = 20
    while offset + 4 <= len(data):
        attr_type, attr_len = struct.unpack("!HH", data[offset:offset + 4])
        attr_data = data[offset + 4:offset + 4 + attr_len]
        offset += 4 + attr_len
        # Pad to 4-byte boundary
        if attr_len % 4:
            offset += 4 - (attr_len % 4)

        if attr_type == _ATTR_XOR_MAPPED_ADDRESS and len(attr_data) >= 8:
            family = attr_data[1]
            xor_port = struct.unpack("!H", attr_data[2:4])[0] ^ (_STUN_MAGIC_COOKIE >> 16)
            if family == 0x01:  # IPv4
                xor_ip_int = struct.unpack("!I", attr_data[4:8])[0] ^ _STUN_MAGIC_COOKIE
                ip = socket.inet_ntoa(struct.pack("!I", xor_ip_int))
                return ip, xor_port

        if attr_type == _ATTR_MAPPED_ADDRESS and len(attr_data) >= 8:
            family = attr_data[1]
            port = struct.unpack("!H", attr_data[2:4])[0]
            if family == 0x01:  # IPv4
                ip = socket.inet_ntoa(attr_data[4:8])
                return ip, port

    return None


def _stun_query(
    server: tuple[str, int],
    timeout_s: float = 3.0,
    local_port: int = 0,
) -> tuple[str, int] | None:
    """Send a single STUN Binding Request and return (external_ip, external_port).

    Returns ``None`` on timeout or failure.
    """
    packet, txn_id = _build_binding_request()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.settimeout(timeout_s)
        if local_port:
            sock.bind(("", local_port))
        sock.sendto(packet, server)
        data, _ = sock.recvfrom(2048)
        return _parse_binding_response(data, txn_id)
    except (socket.timeout, OSError) as exc:
        logger.debug("stun_query_failed: server=%s:%d err=%s", server[0], server[1], exc)
        return None
    finally:
        sock.close()


def probe_nat(
    stun_servers: list[tuple[str, int]] | None = None,
    timeout_s: float = 3.0,
) -> NatProfile:
    """Probe the NAT type using STUN Binding Requests.

    Sends requests to two different STUN servers and compares the
    reflected external address to classify the NAT type.

    Args:
        stun_servers: List of ``(host, port)`` STUN servers to query.
            Defaults to Google + Cloudflare STUN servers.
        timeout_s: Timeout per STUN query in seconds.

    Returns:
        A ``NatProfile`` describing the NAT type and external address.
    """
    servers = stun_servers or _DEFAULT_STUN_SERVERS
    if len(servers) < 2:
        servers = list(servers) + list(_DEFAULT_STUN_SERVERS)

    # Query two different servers from the same local port
    # to detect NAT mapping behavior.
    try:
        local_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        local_sock.bind(("", 0))
        local_port = local_sock.getsockname()[1]
        local_sock.close()
    except OSError:
        local_port = 0

    result1 = _stun_query(servers[0], timeout_s=timeout_s, local_port=local_port)
    result2 = _stun_query(servers[1], timeout_s=timeout_s, local_port=local_port)

    if result1 is None and result2 is None:
        logger.info("nat_probe: no STUN response — assuming unknown NAT")
        return NatProfile(
            reachable=False, nat_type="unknown", requires_relay=True,
        )

    # Use whichever responded
    primary = result1 or result2
    if primary is None:
        return NatProfile(reachable=False, nat_type="unknown", requires_relay=True)

    ext_ip, ext_port = primary

    # Check if the external IP looks like a private address
    # (shouldn't happen with STUN, but guard against local reflectors)
    _private_prefixes = ("10.", "172.16.", "172.17.", "172.18.", "172.19.",
                         "172.20.", "172.21.", "172.22.", "172.23.", "172.24.",
                         "172.25.", "172.26.", "172.27.", "172.28.", "172.29.",
                         "172.30.", "172.31.", "192.168.", "127.")
    if ext_ip.startswith(_private_prefixes):
        return NatProfile(
            reachable=False, nat_type="unknown",
            external_ip=ext_ip, external_port=ext_port,
            requires_relay=True,
        )

    # If both servers returned the same external address → consistent mapping
    if result1 and result2:
        ip1, port1 = result1
        ip2, port2 = result2
        if ip1 == ip2 and port1 == port2:
            # Same mapping to both servers → full cone or open
            nat_type = "full_cone"
        elif ip1 == ip2 and port1 != port2:
            # Same IP but different ports → symmetric NAT
            nat_type = "symmetric"
        else:
            # Different IPs → very unusual, treat as symmetric
            nat_type = "symmetric"
    else:
        # Only one server responded — can't determine type reliably
        nat_type = "unknown"

    # For gRPC (TCP), ALL NAT types except "open" need a relay.
    # full_cone allows consistent UDP hole-punching but TCP connections
    # to the external IP:port won't reach the peer's gRPC server
    # (the STUN-reflected port is for the probe socket, not port 50051).
    # Only peers with a genuine public IP (local == external) are
    # directly reachable without relay.
    requires_relay = True
    try:
        local_ip = socket.gethostbyname(socket.gethostname())
        if local_ip == ext_ip:
            nat_type = "open"
            requires_relay = False
    except OSError:
        pass

    logger.info(
        "nat_probe: type=%s external=%s:%d relay=%s",
        nat_type, ext_ip, ext_port, requires_relay,
    )
    return NatProfile(
        reachable=not requires_relay,
        nat_type=nat_type,
        external_ip=ext_ip,
        external_port=ext_port,
        requires_relay=requires_relay,
    )
