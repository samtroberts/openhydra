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

"""HTTP model seeder for Phase 5 P2P model distribution.

ModelSeedServer exposes locally-cached model weight files over HTTP with RFC 7233
Range request support.  It runs in a daemon thread and requires no new dependencies
(uses stdlib http.server / socketserver / threading).

Security:
  All file paths are validated with os.path.realpath() prefix-check against
  cache_root to prevent any directory traversal.  Filenames must match the
  safe-name regex [A-Za-z0-9._-]+ to block double-encoded attacks.

API:
  GET /health
      → 200 {"status": "ok", "port": <port>}

  GET /v1/p2p/models/{model_id_safe}/
      → 200 JSON array of filename strings available for download

  GET /v1/p2p/models/{model_id_safe}/{filename}
      → 200 full file body
      → 206 Partial Content when a valid single-range Range header is present
      → 403 on path traversal or invalid filename
      → 404 when the file does not exist
"""

from __future__ import annotations

import http.server
import json
import logging
import os
import re
import socketserver
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Route patterns
_RE_FILE = re.compile(
    r"^/v1/p2p/models/(?P<model_safe>[^/]+)/(?P<filename>[^/]+)$"
)
_RE_DIR = re.compile(r"^/v1/p2p/models/(?P<model_safe>[^/]+)/?$")

# Only allow these characters in file names to prevent traversal even after URL-decode.
_RE_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")

# Stream 64 KB at a time.
_CHUNK = 64 * 1024


class _SeedHandler(http.server.BaseHTTPRequestHandler):
    """Internal request handler.  Requires server.cache_root to be set."""

    # Silence the default per-request log output (very chatty for large downloads).
    def log_message(self, fmt: str, *args: Any) -> None:  # type: ignore[override]
        pass

    # ------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802  (stdlib naming convention)
        path = self.path.split("?", 1)[0]  # strip query string if any

        if path == "/health":
            self._send_json(200, {"status": "ok", "port": self.server.server_address[1]})
            return

        m_dir = _RE_DIR.match(path)
        if m_dir:
            self._handle_dir(m_dir.group("model_safe"))
            return

        m_file = _RE_FILE.match(path)
        if m_file:
            self._handle_file(m_file.group("model_safe"), m_file.group("filename"))
            return

        self._send_text(404, "not found")

    # ------------------------------------------------------------------

    def _handle_dir(self, model_safe: str) -> None:
        """Return a JSON list of filenames in the model directory."""
        cache_root: Path = self.server.cache_root  # type: ignore[attr-defined]
        model_dir = cache_root / model_safe
        # Validate resolved path stays inside cache_root.
        if not _is_safe_path(cache_root, model_dir):
            self._send_text(403, "forbidden")
            return
        if not model_dir.is_dir():
            self._send_json(200, [])
            return
        files = sorted(
            f.name for f in model_dir.iterdir()
            if f.is_file() and _RE_SAFE_NAME.match(f.name)
        )
        self._send_json(200, files)

    def _handle_file(self, model_safe: str, filename: str) -> None:
        """Serve a file, supporting single-range RFC 7233 Range requests."""
        cache_root: Path = self.server.cache_root  # type: ignore[attr-defined]

        # Reject unsafe filenames immediately (defence-in-depth on top of realpath check).
        if not _RE_SAFE_NAME.match(filename):
            self._send_text(403, "forbidden")
            return

        file_path = cache_root / model_safe / filename
        if not _is_safe_path(cache_root, file_path):
            self._send_text(403, "forbidden")
            return
        if not file_path.is_file():
            self._send_text(404, "not found")
            return

        total_size = file_path.stat().st_size
        range_header = self.headers.get("Range", "")
        start, end = _parse_range(range_header, total_size)

        if start is None:
            # Serve full file.
            self._stream_file(file_path, 0, total_size - 1, total_size, status=200)
        else:
            # Serve partial content.
            self._stream_file(file_path, start, end, total_size, status=206)

    # ------------------------------------------------------------------

    def _stream_file(
        self, path: Path, start: int, end: int, total: int, status: int
    ) -> None:
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
        self.end_headers()
        try:
            with open(path, "rb") as fh:
                fh.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = fh.read(min(_CHUNK, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass  # Client disconnected mid-download — normal for large files.

    # ------------------------------------------------------------------

    def _send_json(self, status: int, data: Any) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, status: int, text: str) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_safe_path(base: Path, target: Path) -> bool:
    """Return True iff *target* (after realpath resolution) is inside *base*."""
    real_base = os.path.realpath(str(base))
    real_target = os.path.realpath(str(target))
    return real_target.startswith(real_base + os.sep) or real_target == real_base


def _parse_range(
    header: str, total: int
) -> tuple[int, int] | tuple[None, None]:
    """
    Parse a single-range ``Range: bytes=START-END`` header.

    Returns (start, end) inclusive, or (None, None) if the header is absent,
    malformed, or multi-range (multi-range → caller serves full file).
    """
    if not header:
        return None, None
    header = header.strip()
    if not header.lower().startswith("bytes="):
        return None, None
    ranges_part = header[6:].strip()
    # Reject multi-range requests — not needed for P2P resume.
    if "," in ranges_part:
        return None, None
    parts = ranges_part.split("-", 1)
    if len(parts) != 2:
        return None, None
    raw_start, raw_end = parts[0].strip(), parts[1].strip()
    try:
        if raw_start == "":
            # suffix-range: bytes=-N  → last N bytes
            n = int(raw_end)
            start = max(0, total - n)
            end = total - 1
        elif raw_end == "":
            # open-ended: bytes=N-  → from N to end
            start = int(raw_start)
            end = total - 1
        else:
            start = int(raw_start)
            end = int(raw_end)
    except ValueError:
        return None, None
    # Clamp and validate.
    start = max(0, start)
    end = min(end, total - 1)
    if start > end or start >= total:
        return None, None
    return start, end


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class _ThreadingHTTPServer(socketserver.ThreadingTCPServer):
    """ThreadingTCPServer with allow_reuse_address and a cache_root attribute."""
    allow_reuse_address = True
    daemon_threads = True
    cache_root: Path  # set by ModelSeedServer before start()


class ModelSeedServer:
    """
    Lightweight HTTP file server exposing cached model weight files for P2P download.

    Runs in a daemon thread.  Zero external dependencies (stdlib only).

    Args:
        cache_root: Root directory of the P2P model cache.  Each subdirectory is
            named ``{model_id.replace("/", "--")}`` and contains weight files.
        port: TCP port to listen on.  0 = OS-assigned ephemeral port (recommended
            for tests and when the node-operator does not need a fixed port).
        bind_address: Interface to bind to.  Defaults to all interfaces.
    """

    def __init__(
        self,
        cache_root: str | Path,
        port: int = 0,
        bind_address: str = "0.0.0.0",
    ) -> None:
        self._cache_root = Path(cache_root)
        self._port = port
        self._bind_address = bind_address
        self._server: _ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------

    def start(self) -> int:
        """
        Start the HTTP server in a daemon thread.

        Returns the actual bound port (useful when ``port=0`` was specified).
        """
        self._cache_root.mkdir(parents=True, exist_ok=True)

        server = _ThreadingHTTPServer(
            (self._bind_address, self._port),
            _SeedHandler,
        )
        server.cache_root = self._cache_root

        # Record the OS-assigned port before releasing the reference.
        actual_port = server.server_address[1]
        self._port = actual_port
        self._server = server

        thread = threading.Thread(
            target=server.serve_forever,
            daemon=True,
            name=f"p2p-seeder:{actual_port}",
        )
        thread.start()
        self._thread = thread

        logger.info("p2p_seeder_started bind=%s:%d cache=%s", self._bind_address, actual_port, self._cache_root)
        return actual_port

    def stop(self) -> None:
        """Shutdown the server and wait for the daemon thread to exit."""
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    @property
    def port(self) -> int:
        """The bound port (valid after :meth:`start` has been called)."""
        return self._port
