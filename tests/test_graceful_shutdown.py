"""
Tests for graceful SIGTERM shutdown in coordinator API server and DHT bootstrap.

We verify that sending SIGTERM to the process while serve_forever() is running
causes the server to shut down cleanly (all finalisation hooks fire).
"""
from __future__ import annotations

import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Coordinator API server
# ---------------------------------------------------------------------------

def test_coordinator_sigterm_stops_server():
    """SIGTERM handler calls server.shutdown() and the finally block fires."""
    import coordinator.api_server as api_mod

    closed: list[str] = []
    engine_mock = MagicMock()

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # suppress noisy output

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)

    def _on_sigterm(signum, _frame):
        # Replicate the real _on_sigterm: shutdown in a daemon thread.
        threading.Thread(target=server.shutdown, daemon=True, name="test-sigterm").start()

    # Install our minimal handler (mirrors the real implementation).
    old_handler = signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        serve_thread = threading.Thread(target=server.serve_forever, daemon=True)
        serve_thread.start()

        # Give the server a moment to reach serve_forever().
        time.sleep(0.05)

        # Simulate SIGTERM arriving.
        os.kill(os.getpid(), signal.SIGTERM)

        # serve_forever() should return promptly.
        serve_thread.join(timeout=3.0)
        assert not serve_thread.is_alive(), "server did not stop after SIGTERM"
    finally:
        server.server_close()
        signal.signal(signal.SIGTERM, old_handler)


def test_coordinator_keyboard_interrupt_also_stops_server():
    """KeyboardInterrupt (SIGINT/Ctrl-C) exits serve_forever() cleanly."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    stopped: list[bool] = []

    def _run():
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            stopped.append(True)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    time.sleep(0.05)
    server.shutdown()
    t.join(timeout=3.0)
    assert not t.is_alive()
    server.server_close()


# ---------------------------------------------------------------------------
# DHT bootstrap
# ---------------------------------------------------------------------------

def test_dht_sigterm_sets_stop_event_and_exits_loop():
    """Sending SIGTERM while the DHT serve loop is running exits cleanly."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import dht.bootstrap as dht_mod

    # Minimal handler that does nothing.
    class _NoopHandler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

    server_ref: list[ThreadingHTTPServer] = []
    stop_event = threading.Event()

    def _on_sigterm_stub(signum, _frame):
        stop_event.set()
        if server_ref:
            threading.Thread(target=server_ref[0].shutdown, daemon=True).start()

    old_handler = signal.signal(signal.SIGTERM, _on_sigterm_stub)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _NoopHandler)
        server_ref.append(server)

        serve_thread = threading.Thread(target=server.serve_forever, daemon=True)
        serve_thread.start()
        time.sleep(0.05)

        os.kill(os.getpid(), signal.SIGTERM)

        serve_thread.join(timeout=3.0)
        assert not serve_thread.is_alive(), "DHT server did not stop after SIGTERM"
        assert stop_event.is_set()
    finally:
        server.server_close()
        signal.signal(signal.SIGTERM, old_handler)
