# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Tests for peer.lan_routing — LAN-first routing helpers.

These verify the address classification + local-interface-prefix logic
in isolation. Wiring tests for the four routing decision sites in
``peer/server.py`` (``_push_to_next_hop``, ``_push_final_result``, the
legacy ring loopback, and ``_coordinator_reinject_ring_step``) are
exercised live in the cross-VPC benchmark — no clean way to mock
``threading.Thread(target=_fire)`` + grpc.insecure_channel without
ending up with a brittle integration mock.
"""

from __future__ import annotations

import ipaddress

import pytest

from peer.lan_routing import (
    _invalidate_cache,
    is_private_lan_address,
    is_reachable_lan,
    local_lan_prefixes,
    parse_host_from_address,
    set_local_lan_prefixes,
)


@pytest.fixture(autouse=True)
def _wipe_prefix_cache():
    """Each test starts from a clean cache so prefix overrides don't leak."""
    _invalidate_cache()
    yield
    _invalidate_cache()


# ── is_private_lan_address ──────────────────────────────────────────────


@pytest.mark.parametrize("host,expected", [
    ("10.0.0.1", True),
    ("10.192.11.74", True),       # Lightning VPC sample
    ("172.16.0.1", True),         # RFC1918 172.16/12
    ("172.31.255.255", True),
    ("172.32.0.1", False),        # outside 172.16/12
    ("192.168.0.1", True),
    ("192.168.255.255", True),
    ("127.0.0.1", True),          # loopback counts as private
    ("169.254.1.1", True),        # link-local counts as private
    ("8.8.8.8", False),           # public
    ("1.1.1.1", False),
    ("45.79.190.172", False),     # the actual Linode US relay IP
    ("openhydra.co", False),      # hostname, not IP
    ("", False),
    ("not-an-ip", False),
])
def test_is_private_lan_address(host, expected):
    assert is_private_lan_address(host) is expected


def test_is_private_lan_address_handles_whitespace():
    assert is_private_lan_address("  10.0.0.1  ") is True


# ── parse_host_from_address ─────────────────────────────────────────────


@pytest.mark.parametrize("addr,expected", [
    ("10.192.11.74:50052", "10.192.11.74"),
    ("127.0.0.1:7050", "127.0.0.1"),
    ("[::1]:50051", "::1"),
    ("10.0.0.1", "10.0.0.1"),     # no port
    ("", ""),
    ("  ", ""),
])
def test_parse_host_from_address(addr, expected):
    assert parse_host_from_address(addr) == expected


# ── set_local_lan_prefixes (operator override) ──────────────────────────


def test_set_local_lan_prefixes_replaces_cache():
    set_local_lan_prefixes(["10.192.0.0/16"])
    prefixes = local_lan_prefixes()
    assert ipaddress.IPv4Network("10.192.0.0/16") in prefixes
    # Cache is hot — calling again returns the same set.
    assert local_lan_prefixes() is prefixes or local_lan_prefixes() == prefixes


def test_set_local_lan_prefixes_accepts_non_strict_form():
    """``10.192.11.221/16`` is the IP-with-prefix form a peer would
    write directly — strict=False lets us parse it without
    pre-computing the network address."""
    set_local_lan_prefixes(["10.192.11.221/16", "192.168.1.42/24"])
    prefixes = local_lan_prefixes()
    assert ipaddress.IPv4Network("10.192.0.0/16") in prefixes
    assert ipaddress.IPv4Network("192.168.1.0/24") in prefixes


def test_set_local_lan_prefixes_skips_garbage():
    set_local_lan_prefixes(["10.0.0.0/16", "garbage", "not-cidr/whatever"])
    prefixes = local_lan_prefixes()
    assert ipaddress.IPv4Network("10.0.0.0/16") in prefixes
    assert len(prefixes) == 1


# ── is_reachable_lan (the main predicate the routing layer queries) ─────


def test_is_reachable_lan_with_matching_prefix_returns_true():
    set_local_lan_prefixes(["10.192.0.0/16"])
    # GPU2 from GPU1's perspective in the same Lightning VPC.
    assert is_reachable_lan("10.192.11.74") is True
    # Same /16 boundary.
    assert is_reachable_lan("10.192.0.1") is True
    assert is_reachable_lan("10.192.255.255") is True


def test_is_reachable_lan_with_different_prefix_returns_false():
    """A /16 in 10.192.x is NOT reachable from a host whose prefix is
    10.193.x (different VPC subnet)."""
    set_local_lan_prefixes(["10.193.0.0/16"])
    assert is_reachable_lan("10.192.11.74") is False


def test_is_reachable_lan_rejects_public_ip_even_when_prefixes_loaded():
    """Public IPs always return False — we never try direct gRPC to them."""
    set_local_lan_prefixes(["10.0.0.0/16"])
    assert is_reachable_lan("8.8.8.8") is False
    assert is_reachable_lan("45.79.190.172") is False


def test_is_reachable_lan_with_no_prefixes_returns_false():
    """If we have no LAN interfaces (or auto-detection failed), every
    LAN-routing query falls through — disables the optimisation
    rather than risking a wrong direct dial."""
    set_local_lan_prefixes([])
    assert is_reachable_lan("10.192.11.74") is False
    assert is_reachable_lan("192.168.1.5") is False


def test_is_reachable_lan_handles_invalid_input():
    """Hostnames, malformed strings, empty: all return False."""
    set_local_lan_prefixes(["10.0.0.0/16"])
    assert is_reachable_lan("openhydra.co") is False
    assert is_reachable_lan("") is False
    assert is_reachable_lan("not.an.ip.address") is False


def test_is_reachable_lan_respects_loopback():
    """127.x is a private range; if our local prefixes include 127.x
    (which they typically do), 127.0.0.1 must be reachable. Critical
    for single-host integration tests where coord and peer share lo0."""
    set_local_lan_prefixes(["127.0.0.0/16"])
    assert is_reachable_lan("127.0.0.1") is True


# ── End-to-end intent check: documents the Lightning scenario ───────────


def test_lightning_two_gpu_vpc_scenario():
    """Reproduces the 2026-04-24 cross-VPC benchmark scenario.

    GPU1 has interface IP 10.192.11.221.
    GPU2 has interface IP 10.192.11.74.
    Mac coordinator has interface IPs 127.0.0.1 + 192.168.x (home Wi-Fi).

    From GPU1's perspective, GPU2 must be reachable via LAN.
    From Mac's perspective, neither GPU is reachable (different subnet).
    """
    # GPU1's perspective.
    set_local_lan_prefixes(["10.192.11.221/16"])
    assert is_reachable_lan("10.192.11.74") is True, (
        "GPU1 must classify GPU2 as LAN-reachable — this is the whole "
        "point of the LAN-first routing fix"
    )
    # Mac's perspective.
    set_local_lan_prefixes(["127.0.0.0/16", "192.168.1.0/24"])
    assert is_reachable_lan("10.192.11.74") is False, (
        "Mac must NOT try direct gRPC to a Lightning VPC IP — that would "
        "race-fail the public-internet path the libp2p relay covers"
    )
