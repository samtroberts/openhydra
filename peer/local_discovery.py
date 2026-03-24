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

"""peer.local_discovery — local peer detection for fast-path tensor transfer.

Phase A of the Local Clusters feature.  Provides subnet-awareness utilities
so peers on the same LAN can bypass gRPC and stream raw tensor bytes over
a direct TCP socket.

Key functions
-------------
``is_same_lan(addr_a, addr_b, prefix_len=24)``
    Check if two IPv4 addresses share the same /prefix_len subnet.
``is_loopback(addr)``
    Check if an address is localhost.
``is_private(addr)``
    Check if an address is in a private (RFC 1918) range.
``parse_ipv4(addr)``
    Extract the IPv4 address from a host string (strip port, resolve hostname).
"""
from __future__ import annotations

import ipaddress
import logging
import socket
from typing import Any

logger = logging.getLogger(__name__)


def parse_ipv4(addr: str) -> str | None:
    """Extract an IPv4 address from *addr*, returning ``None`` on failure.

    Handles:
    * Plain IPs: ``"192.168.1.5"``
    * Host:port: ``"192.168.1.5:50051"``
    * Hostnames:  resolved via ``socket.getaddrinfo`` (first A record).
    * IPv6-mapped IPv4: ``"::ffff:192.168.1.5"``
    """
    raw = str(addr or "").strip()
    if not raw:
        return None

    # Strip port if present (naive: last colon with digits after).
    if ":" in raw and not raw.startswith("["):
        parts = raw.rsplit(":", 1)
        if parts[-1].isdigit():
            raw = parts[0]

    # Try parsing as an IP directly.
    try:
        ip = ipaddress.ip_address(raw)
        if isinstance(ip, ipaddress.IPv4Address):
            return str(ip)
        # IPv6-mapped IPv4.
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            return str(ip.ipv4_mapped)
        return None
    except ValueError:
        pass

    # Try DNS resolution.
    try:
        results = socket.getaddrinfo(raw, None, socket.AF_INET, socket.SOCK_STREAM)
        if results:
            return str(results[0][4][0])
    except (socket.gaierror, OSError):
        pass

    return None


def is_loopback(addr: str) -> bool:
    """Return ``True`` if *addr* resolves to a loopback address (127.x.x.x)."""
    ip_str = parse_ipv4(addr)
    if ip_str is None:
        return False
    try:
        return ipaddress.ip_address(ip_str).is_loopback
    except ValueError:
        return False


def is_private(addr: str) -> bool:
    """Return ``True`` if *addr* is in an RFC 1918 private range or loopback."""
    ip_str = parse_ipv4(addr)
    if ip_str is None:
        return False
    try:
        return ipaddress.ip_address(ip_str).is_private
    except ValueError:
        return False


def is_same_lan(
    addr_a: str,
    addr_b: str,
    prefix_len: int = 24,
) -> bool:
    """Return ``True`` when two addresses share the same IPv4 subnet.

    Parameters
    ----------
    addr_a, addr_b:
        IPv4 addresses or host:port strings.
    prefix_len:
        Subnet prefix length (default 24 → /24, i.e. 255.255.255.0).

    Both loopback addresses (127.x.x.x) are treated as same-LAN with each
    other and with any other loopback address.

    Returns ``False`` if either address cannot be parsed as IPv4.
    """
    ip_a_str = parse_ipv4(addr_a)
    ip_b_str = parse_ipv4(addr_b)
    if ip_a_str is None or ip_b_str is None:
        return False

    try:
        ip_a = ipaddress.ip_address(ip_a_str)
        ip_b = ipaddress.ip_address(ip_b_str)
    except ValueError:
        return False

    # Both loopback → same LAN.
    if ip_a.is_loopback and ip_b.is_loopback:
        return True

    # Same exact IP → same LAN.
    if ip_a == ip_b:
        return True

    # Subnet comparison.
    prefix = max(1, min(32, int(prefix_len)))
    net = ipaddress.ip_network(f"{ip_a_str}/{prefix}", strict=False)
    return ip_b in net


def get_local_ipv4() -> str | None:
    """Best-effort detection of the machine's LAN IPv4 address.

    Uses a UDP connect trick (no actual packet sent) to determine the
    outbound interface address.  Returns ``None`` if detection fails.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
            return str(ip)
        finally:
            s.close()
    except Exception:
        return None
