"""peer.local_fast_path — raw TCP socket for LAN tensor transfer.

Phase A of the Local Clusters feature.  When two peers are on the same LAN,
this module bypasses gRPC protobuf serialisation and streams tensor bytes
directly over a raw TCP socket, cutting ~2 ms of ser/deser overhead per hop.

Protocol (v1)
-------------
1. **Header** (12 bytes, big-endian):
   - ``magic`` (4 bytes): ``b'OHFP'`` ("OpenHydra Fast Path")
   - ``version`` (2 bytes): ``uint16 = 1``
   - ``payload_len`` (4 bytes): ``uint32`` = number of float64 values
   - ``flags`` (2 bytes): reserved, must be 0

2. **Payload**: ``payload_len × 8`` bytes of IEEE-754 float64 values,
   big-endian.

3. **Response**: Same header+payload format for the activation response.

Server
------
``FastPathServer(handler, bind_host, port)`` — stdlib TCP server on a
background thread.  ``handler`` is called with the decoded activation list
and must return a list of floats (the response activation).

Client
------
``send_fast_path(host, port, activation)`` — sends an activation list and
returns the response activation list.
"""
from __future__ import annotations

import logging
import socket
import socketserver
import struct
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Protocol constants.
MAGIC = b"OHFP"
VERSION = 1
HEADER_FORMAT = "!4sHIH"  # magic(4) + version(u16) + payload_len(u32) + flags(u16)
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # = 12


def _pack_message(activation: list[float]) -> bytes:
    """Encode an activation list into a header + payload bytes."""
    count = len(activation)
    header = struct.pack(HEADER_FORMAT, MAGIC, VERSION, count, 0)
    payload = struct.pack(f"!{count}d", *activation)
    return header + payload


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    """Read exactly *n* bytes from *sock*."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("fast_path: connection closed mid-read")
        buf.extend(chunk)
    return bytes(buf)


def _unpack_message(sock: socket.socket) -> list[float]:
    """Read and decode a header + payload from *sock*."""
    raw_header = _recv_exactly(sock, HEADER_SIZE)
    magic, version, payload_len, flags = struct.unpack(HEADER_FORMAT, raw_header)
    if magic != MAGIC:
        raise ValueError(f"fast_path: invalid magic {magic!r}")
    if version != VERSION:
        raise ValueError(f"fast_path: unsupported version {version}")
    if payload_len == 0:
        return []
    raw_payload = _recv_exactly(sock, payload_len * 8)
    return list(struct.unpack(f"!{payload_len}d", raw_payload))


# ── Server ──────────────────────────────────────────────────────────────────


class _FastPathHandler(socketserver.BaseRequestHandler):
    """TCP request handler for one fast-path activation exchange."""

    def handle(self) -> None:
        try:
            activation = _unpack_message(self.request)
            handler_fn: Callable = self.server._fast_path_handler  # type: ignore[attr-defined]
            result = handler_fn(activation)
            self.request.sendall(_pack_message(result))
        except Exception as exc:
            logger.debug("fast_path_handler_error: %s", exc)
            # Send an empty response on error so the client doesn't hang.
            try:
                self.request.sendall(_pack_message([]))
            except Exception:
                pass


class FastPathServer:
    """Raw TCP server for same-LAN tensor transfer.

    Parameters
    ----------
    handler:
        Callable that takes ``list[float]`` (input activation) and returns
        ``list[float]`` (output activation).
    bind_host:
        Interface to bind to (default ``"0.0.0.0"``).
    port:
        Port to bind to (default ``0`` → OS-assigned ephemeral port).
    """

    def __init__(
        self,
        handler: Callable[[list[float]], list[float]],
        bind_host: str = "0.0.0.0",
        port: int = 0,
    ) -> None:
        self._handler = handler

        class _Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        self._server = _Server((bind_host, port), _FastPathHandler)
        self._server._fast_path_handler = handler  # type: ignore[attr-defined]
        self._port = self._server.server_address[1]
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self._port

    def start(self) -> None:
        """Start serving in a background daemon thread."""
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="fast-path-tcp",
        )
        self._thread.start()
        logger.info("fast_path_server: listening on port %d", self._port)

    def stop(self) -> None:
        """Shutdown the server."""
        self._server.shutdown()
        self._server.server_close()
        logger.info("fast_path_server: stopped")


# ── Client ──────────────────────────────────────────────────────────────────


def send_fast_path(
    host: str,
    port: int,
    activation: list[float],
    timeout_s: float = 5.0,
) -> list[float]:
    """Send an activation to a peer's fast-path port and return the response.

    Parameters
    ----------
    host:   Target peer's IP address.
    port:   Target peer's fast-path TCP port.
    activation: Tensor activation as a list of floats.
    timeout_s:  Socket timeout in seconds.

    Returns
    -------
    list[float]: Response activation from the peer.

    Raises
    ------
    ConnectionError, TimeoutError, ValueError:
        On transport or protocol errors.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout_s)
    try:
        sock.connect((host, port))
        sock.sendall(_pack_message(activation))
        return _unpack_message(sock)
    finally:
        sock.close()
