# Copyright 2026 OpenHydra contributors — Apache 2.0

"""LAN-first routing helpers for peer-to-peer hops.

Problem
-------
``peer/server.py`` historically picked libp2p (relay-mediated) whenever a
``next_hop_libp2p_id`` was present, even when the next hop's
``next_address`` was a private RFC1918 IP that the sender could reach
directly via gRPC. On infrastructure where two peers share a VPC subnet
but live behind a global DHT advertising their public-relay reservation
(e.g. two Lightning AI studios in the same ``10.192.x.x`` subnet),
this forced every hop through a transcontinental Linode relay — and
when DCUtR couldn't hole-punch the symmetric NAT pair, the connection
silently stalled, breaking the 2026-04-24 cross-ISP benchmark.

The fix
-------
Before falling into the libp2p path, classify ``next_address``:

* If it's an RFC1918 private IP **and** at least one of the sender's
  local interfaces lives in the same ``/16``, prefer **direct gRPC**
  unconditionally. The libp2p_peer_id is ignored for that hop. The
  loopback case (127.x) and link-local case (169.254.x) are also
  treated as "definitely LAN, definitely reachable".
* Otherwise (public IP, or a private IP we can't reach via any local
  interface), fall through to the existing libp2p-then-gRPC fallback.

The check is intentionally local to the sender: the same
ForwardRequest can be routed via libp2p on one hop and via direct gRPC
on the next, depending on each peer's network position. No central
coordination, no extra wire fields, no protocol change.

This module is dependency-free (stdlib only) and side-effect-free
beyond the local-interface enumeration done once per process. The
``local_lan_prefixes`` cache is computed eagerly the first time it's
needed; if interfaces change at runtime, call ``_invalidate_cache()``
(test helper).
"""

from __future__ import annotations

import ipaddress
import logging
import socket
import threading
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


# ── Address classification ─────────────────────────────────────────────


def is_private_lan_address(host: str) -> bool:
    """Return True iff ``host`` is an IPv4/IPv6 address in a non-routable
    private range (RFC1918, loopback, link-local). Hostnames and public
    IPs return False.

    A private IP is a strong signal that direct gRPC could work IF the
    sender shares the same LAN — checked separately by
    :func:`is_reachable_lan`.
    """
    if not host:
        return False
    try:
        ip = ipaddress.ip_address(str(host).strip())
    except ValueError:
        return False
    return bool(ip.is_private) and not ip.is_unspecified


def _enumerate_local_ipv4_addresses() -> set[str]:
    """Best-effort enumeration of this host's IPv4 addresses.

    Uses two cheap mechanisms:
        1. ``socket.getaddrinfo(socket.gethostname(), None)``
        2. The "connect to 8.8.8.8 by UDP and read sockname" trick to
           identify the default-route-egress IP (works behind NAT).

    Both are wrapped in try/except — interface enumeration is famously
    flaky across OSes; failure here just means the LAN-first
    optimisation is disabled for this hop, not a hard error.
    """
    addrs: set[str] = set()
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None):
            try:
                a = info[4][0]
                # Strip IPv6 zone-id if present.
                if "%" in a:
                    a = a.split("%", 1)[0]
                ip = ipaddress.ip_address(a)
                if isinstance(ip, ipaddress.IPv4Address):
                    addrs.add(str(ip))
            except (ValueError, IndexError):
                continue
    except (OSError, socket.gaierror):
        pass

    # Default-route egress IP — catches Lightning-style NAT'd VMs whose
    # gethostname returns a public name but whose actual interface IP
    # is 10.x.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(0.5)
            s.connect(("8.8.8.8", 80))  # never actually sends
            addrs.add(s.getsockname()[0])
    except (OSError, socket.timeout):
        pass

    return addrs


# Cached prefixes. Recomputed once per process — interfaces don't change
# often enough to justify a TTL on the hot push path.
_LOCAL_PREFIXES_LOCK = threading.Lock()
_LOCAL_PREFIXES_V4: Optional[set[ipaddress.IPv4Network]] = None


def local_lan_prefixes() -> set[ipaddress.IPv4Network]:
    """Return the set of /16 IPv4 networks our local interfaces live in.

    /16 is the right granularity for VPC-style "all hosts in 10.192.0.0/16
    can talk to each other" deployments (Lightning studios, AWS VPCs,
    most Kubernetes pod networks). Tighter (/24) misses cross-subnet
    intra-VPC routing; looser (/8) over-claims and would try direct
    gRPC to genuinely unreachable hosts.

    Result is memoised — see ``_invalidate_cache`` for the test hook.
    """
    global _LOCAL_PREFIXES_V4
    with _LOCAL_PREFIXES_LOCK:
        if _LOCAL_PREFIXES_V4 is not None:
            return _LOCAL_PREFIXES_V4
        prefixes: set[ipaddress.IPv4Network] = set()
        for addr in _enumerate_local_ipv4_addresses():
            try:
                ip = ipaddress.IPv4Address(addr)
            except ValueError:
                continue
            if not ip.is_private and not ip.is_loopback:
                continue
            prefixes.add(ipaddress.IPv4Network(f"{addr}/16", strict=False))
        _LOCAL_PREFIXES_V4 = prefixes
        if prefixes:
            logger.info(
                "lan_routing_local_prefixes: %s",
                sorted(str(p) for p in prefixes),
            )
        else:
            logger.info(
                "lan_routing_local_prefixes: (none — LAN-first disabled, "
                "all hops will use existing libp2p/gRPC fallback)"
            )
        return prefixes


def is_reachable_lan(target_host: str) -> bool:
    """Return True iff ``target_host`` is a private IP **and** falls in
    a /16 that any of our local interfaces also covers.

    This is the predicate the routing layer queries before deciding
    whether to send a ForwardRequest via direct gRPC instead of via the
    libp2p relay path. Conservative: returns False on any error,
    on hostnames (no DNS round-trip), and on public IPs.

    The "same /16" rule is empirically a good fit for VPC topologies.
    For tighter or looser policies, override the prefixes via
    :func:`set_local_lan_prefixes` (test/operator hook).
    """
    if not is_private_lan_address(target_host):
        return False
    try:
        target = ipaddress.IPv4Address(str(target_host).strip())
    except ValueError:
        return False
    for prefix in local_lan_prefixes():
        if target in prefix:
            return True
    return False


def parse_host_from_address(address: str) -> str:
    """Extract the host portion of an ``address`` like ``"10.0.0.1:50051"``
    or ``"[::1]:50051"``. Returns the address unchanged if no port is
    present. Returns empty string for falsy input.
    """
    if not address:
        return ""
    s = str(address).strip()
    if not s:
        return ""
    # IPv6 bracket form.
    if s.startswith("["):
        end = s.find("]")
        if end > 0:
            return s[1:end]
    # IPv4 host:port.
    if ":" in s:
        # Don't split bare IPv6 addresses.
        if s.count(":") == 1:
            return s.split(":", 1)[0]
        # If multiple colons and no brackets, it's bare IPv6 — return as-is.
        try:
            ipaddress.ip_address(s)
            return s
        except ValueError:
            return s.rsplit(":", 1)[0]
    return s


# ── Test / operator hooks ─────────────────────────────────────────────


def _invalidate_cache() -> None:
    """Wipe the memoised local prefixes — for tests that mock
    ``_enumerate_local_ipv4_addresses`` and need a fresh read."""
    global _LOCAL_PREFIXES_V4
    with _LOCAL_PREFIXES_LOCK:
        _LOCAL_PREFIXES_V4 = None


def set_local_lan_prefixes(prefixes: Iterable[str]) -> None:
    """Operator override — explicitly declare which /16s count as LAN.

    Useful when the auto-detection misses an aliased interface (rare on
    Lightning, common on baremetal with multi-NIC setups). Pass strings
    like ``["10.192.0.0/16", "192.168.1.0/24"]``.
    """
    global _LOCAL_PREFIXES_V4
    parsed: set[ipaddress.IPv4Network] = set()
    for p in prefixes:
        try:
            parsed.add(ipaddress.IPv4Network(p, strict=False))
        except ValueError:
            logger.warning("lan_routing_invalid_prefix_override: %s", p)
            continue
    with _LOCAL_PREFIXES_LOCK:
        _LOCAL_PREFIXES_V4 = parsed
    logger.info(
        "lan_routing_prefix_override: %s",
        sorted(str(p) for p in parsed),
    )
