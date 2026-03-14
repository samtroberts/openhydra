"""Tests for peer.local_discovery and peer.local_fast_path — Phase A Local Clusters.

Groups:
    A — is_same_lan() subnet detection (5 tests): same subnet, different subnet,
        loopback, host:port stripping, cross-class rejection.
    B — parse_ipv4 / is_loopback / is_private helpers (4 tests).
    C — FastPathServer + send_fast_path raw TCP transport (4 tests): roundtrip,
        empty activation, large payload, server shutdown.
    D — DHT Announcement & PeerEndpoint integration (2 tests): field existence,
        DHT parsing.
"""
from __future__ import annotations

import socket
import threading
import time
import unittest
from dataclasses import asdict
from typing import Any
from unittest.mock import patch

from peer.local_discovery import (
    get_local_ipv4,
    is_loopback,
    is_private,
    is_same_lan,
    parse_ipv4,
)
from peer.local_fast_path import (
    FastPathServer,
    send_fast_path,
    _pack_message,
    _unpack_message,
    HEADER_SIZE,
    MAGIC,
    VERSION,
)


# ═════════════════════════════════════════════════════════════════════════════
# Group A — is_same_lan() subnet detection
# ═════════════════════════════════════════════════════════════════════════════


class TestIsSameLan(unittest.TestCase):
    """Group A: same-LAN detection via subnet comparison."""

    def test_same_subnet_24(self):
        """Two IPs in the same /24 → True."""
        self.assertTrue(is_same_lan("192.168.1.10", "192.168.1.200"))

    def test_different_subnet_24(self):
        """IPs in different /24 subnets → False."""
        self.assertFalse(is_same_lan("192.168.1.10", "192.168.2.10"))

    def test_loopback_same_lan(self):
        """Both loopback addresses → True (same machine)."""
        self.assertTrue(is_same_lan("127.0.0.1", "127.0.0.2"))

    def test_host_port_stripping(self):
        """Addresses with port suffixes are handled correctly."""
        self.assertTrue(is_same_lan("192.168.1.10:50051", "192.168.1.20:8080"))

    def test_cross_class_rejection(self):
        """10.x.x.x vs 192.168.x.x → False (different private ranges)."""
        self.assertFalse(is_same_lan("10.0.0.5", "192.168.1.5"))

    def test_same_ip_same_lan(self):
        """Identical IPs → True."""
        self.assertTrue(is_same_lan("192.168.1.42", "192.168.1.42"))

    def test_invalid_address_returns_false(self):
        """Unparseable addresses → False (no crash)."""
        self.assertFalse(is_same_lan("not-an-ip", "192.168.1.1"))
        self.assertFalse(is_same_lan("", ""))

    def test_custom_prefix_length(self):
        """Different prefix lengths: /16 captures wider subnets."""
        # Same /16 but different /24.
        self.assertTrue(is_same_lan("192.168.1.10", "192.168.2.10", prefix_len=16))
        # Different /16.
        self.assertFalse(is_same_lan("192.168.1.10", "192.169.1.10", prefix_len=16))


# ═════════════════════════════════════════════════════════════════════════════
# Group B — parse_ipv4 / is_loopback / is_private helpers
# ═════════════════════════════════════════════════════════════════════════════


class TestParseIpv4(unittest.TestCase):
    """Group B: IPv4 parsing and classification."""

    def test_parse_plain_ip(self):
        self.assertEqual(parse_ipv4("192.168.1.5"), "192.168.1.5")

    def test_parse_host_port(self):
        self.assertEqual(parse_ipv4("10.0.0.1:8080"), "10.0.0.1")

    def test_parse_invalid_returns_none(self):
        self.assertIsNone(parse_ipv4("not-an-ip-at-all"))

    def test_parse_empty_returns_none(self):
        self.assertIsNone(parse_ipv4(""))


class TestIsLoopback(unittest.TestCase):
    def test_localhost(self):
        self.assertTrue(is_loopback("127.0.0.1"))

    def test_non_loopback(self):
        self.assertFalse(is_loopback("192.168.1.1"))


class TestIsPrivate(unittest.TestCase):
    def test_private_range(self):
        self.assertTrue(is_private("192.168.1.1"))
        self.assertTrue(is_private("10.0.0.1"))
        self.assertTrue(is_private("172.16.0.1"))

    def test_public_range(self):
        self.assertFalse(is_private("8.8.8.8"))


# ═════════════════════════════════════════════════════════════════════════════
# Group C — FastPathServer + send_fast_path raw TCP transport
# ═════════════════════════════════════════════════════════════════════════════


class TestFastPathTransport(unittest.TestCase):
    """Group C: raw TCP tensor transport roundtrip."""

    @classmethod
    def setUpClass(cls):
        # Echo handler: returns the received activation with values doubled.
        def _handler(activation: list[float]) -> list[float]:
            return [v * 2.0 for v in activation]

        cls._server = FastPathServer(handler=_handler, bind_host="127.0.0.1", port=0)
        cls._server.start()
        cls._port = cls._server.port

    @classmethod
    def tearDownClass(cls):
        cls._server.stop()

    def test_roundtrip(self):
        """Send activation, receive doubled values."""
        activation = [1.0, 2.0, 3.0, -0.5]
        result = send_fast_path("127.0.0.1", self._port, activation, timeout_s=5.0)
        self.assertEqual(len(result), 4)
        self.assertAlmostEqual(result[0], 2.0)
        self.assertAlmostEqual(result[1], 4.0)
        self.assertAlmostEqual(result[2], 6.0)
        self.assertAlmostEqual(result[3], -1.0)

    def test_empty_activation(self):
        """Empty activation list → empty response."""
        result = send_fast_path("127.0.0.1", self._port, [], timeout_s=5.0)
        self.assertEqual(result, [])

    def test_large_payload(self):
        """64-element activation (typical ACTIVATION_SIZE)."""
        activation = [float(i) * 0.01 for i in range(64)]
        result = send_fast_path("127.0.0.1", self._port, activation, timeout_s=5.0)
        self.assertEqual(len(result), 64)
        for i, v in enumerate(result):
            self.assertAlmostEqual(v, float(i) * 0.02, places=10)

    def test_connection_refused_when_stopped(self):
        """After server stops, send_fast_path raises an error."""
        server = FastPathServer(
            handler=lambda a: a,
            bind_host="127.0.0.1",
            port=0,
        )
        server.start()
        port = server.port
        # Verify it works.
        result = send_fast_path("127.0.0.1", port, [1.0], timeout_s=5.0)
        self.assertEqual(len(result), 1)
        # Stop and verify connection fails.
        server.stop()
        time.sleep(0.1)  # Brief settle.
        with self.assertRaises((ConnectionError, ConnectionRefusedError, OSError)):
            send_fast_path("127.0.0.1", port, [1.0], timeout_s=1.0)


class TestFastPathProtocol(unittest.TestCase):
    """Group C (cont.): protocol encoding/decoding."""

    def test_pack_unpack_roundtrip(self):
        """Pack then unpack via a socket pair."""
        activation = [1.5, -2.5, 0.0, 3.14159]
        data = _pack_message(activation)
        self.assertEqual(len(data), HEADER_SIZE + 4 * 8)

        # Create a socket pair for in-process test.
        server_sock, client_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            client_sock.sendall(data)
            result = _unpack_message(server_sock)
            self.assertEqual(len(result), 4)
            self.assertAlmostEqual(result[0], 1.5)
            self.assertAlmostEqual(result[3], 3.14159, places=5)
        finally:
            server_sock.close()
            client_sock.close()


# ═════════════════════════════════════════════════════════════════════════════
# Group D — DHT Announcement & PeerEndpoint integration
# ═════════════════════════════════════════════════════════════════════════════


class TestDhtFastPathField(unittest.TestCase):
    """Group D: local_fast_path_port in Announcement and PeerEndpoint."""

    def test_announcement_has_field(self):
        from peer.dht_announce import Announcement
        a = Announcement(
            peer_id="p1", model_id="m1", host="127.0.0.1", port=50051,
            local_fast_path_port=9999,
        )
        self.assertEqual(a.local_fast_path_port, 9999)
        d = asdict(a)
        self.assertEqual(d["local_fast_path_port"], 9999)

    def test_announcement_default_zero(self):
        from peer.dht_announce import Announcement
        a = Announcement(peer_id="p1", model_id="m1", host="127.0.0.1", port=50051)
        self.assertEqual(a.local_fast_path_port, 0)

    def test_peer_endpoint_has_field(self):
        from coordinator.path_finder import PeerEndpoint
        pe = PeerEndpoint(
            peer_id="p1", host="192.168.1.10", port=50051,
            local_fast_path_port=12345,
        )
        self.assertEqual(pe.local_fast_path_port, 12345)

    def test_peer_endpoint_default_zero(self):
        from coordinator.path_finder import PeerEndpoint
        pe = PeerEndpoint(peer_id="p1", host="192.168.1.10", port=50051)
        self.assertEqual(pe.local_fast_path_port, 0)


# ═════════════════════════════════════════════════════════════════════════════
# Group E — gRPC fallback logic
# ═════════════════════════════════════════════════════════════════════════════


class TestGrpcFallbackLogic(unittest.TestCase):
    """Group E: is_same_lan + local_fast_path_port → use fast path or gRPC."""

    def test_same_lan_with_fast_path_port(self):
        """Same LAN + fast path port > 0 → should use fast path."""
        from coordinator.path_finder import PeerEndpoint
        local_host = "192.168.1.10"
        peer = PeerEndpoint(
            peer_id="p2", host="192.168.1.20", port=50051,
            local_fast_path_port=9999,
        )
        use_fast_path = (
            is_same_lan(local_host, peer.host)
            and peer.local_fast_path_port > 0
        )
        self.assertTrue(use_fast_path)

    def test_different_lan_falls_back_to_grpc(self):
        """Different LAN → should NOT use fast path."""
        from coordinator.path_finder import PeerEndpoint
        local_host = "192.168.1.10"
        peer = PeerEndpoint(
            peer_id="p2", host="10.0.0.5", port=50051,
            local_fast_path_port=9999,
        )
        use_fast_path = (
            is_same_lan(local_host, peer.host)
            and peer.local_fast_path_port > 0
        )
        self.assertFalse(use_fast_path)

    def test_same_lan_no_fast_path_port_falls_back(self):
        """Same LAN but fast_path_port == 0 → should NOT use fast path."""
        from coordinator.path_finder import PeerEndpoint
        local_host = "192.168.1.10"
        peer = PeerEndpoint(
            peer_id="p2", host="192.168.1.20", port=50051,
            local_fast_path_port=0,
        )
        use_fast_path = (
            is_same_lan(local_host, peer.host)
            and peer.local_fast_path_port > 0
        )
        self.assertFalse(use_fast_path)


# ═════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    unittest.main()
