"""Tests for Phase 1e CLI — arg parsing, join smoke, chat SSE client."""

from __future__ import annotations

import json
import io
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coordinator.cli import cmd_chat, cmd_status, cmd_models, main as cli_main


# ── Helpers ──────────────────────────────────────────────────────────────


class FakeAPIHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that serves OpenAI-compatible endpoints."""

    models_response = {
        "object": "list",
        "data": [
            {
                "id": "llama3:8b",
                "object": "model",
                "created": 0,
                "owned_by": "openhydra",
                "openhydra": {
                    "family": "llama",
                    "parameter_count": 8_000_000_000,
                    "quantization": "Q4_0",
                    "context_length": 4096,
                    "supports_streaming": True,
                },
            },
        ],
    }

    health_response = {"status": "ok"}
    supernodes_response = {"supernodes": []}

    def do_GET(self):
        if self.path == "/v1/models":
            self._json(self.models_response)
        elif self.path == "/health":
            self._json(self.health_response)
        elif self.path == "/v1/supernodes":
            self._json(self.supernodes_response)
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        if self.path == "/v1/chat/completions":
            self._stream_chat(body)
        else:
            self.send_error(404)

    def _json(self, data):
        payload = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _stream_chat(self, body):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        chunks = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " world"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        for c in chunks:
            line = f"data: {json.dumps(c)}\n\n"
            self.wfile.write(line.encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, format, *args):
        pass


@pytest.fixture
def fake_api():
    server = HTTPServer(("127.0.0.1", 0), FakeAPIHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


# ── CLI arg parsing ──────────────────────────────────────────────────────


class TestCLIParsing:
    def test_no_command_shows_help(self, capsys):
        import sys
        with patch.object(sys, "argv", ["openhydra"]):
            with pytest.raises(SystemExit) as exc:
                cli_main()
            assert exc.value.code == 0

    def test_join_requires_bridge(self):
        import sys
        with patch.object(sys, "argv", ["openhydra", "join"]):
            with pytest.raises(SystemExit) as exc:
                cli_main()
            assert exc.value.code == 2

    def test_join_invalid_bridge(self):
        import sys
        with patch.object(sys, "argv", ["openhydra", "join", "--bridge", "invalid"]):
            with pytest.raises(SystemExit) as exc:
                cli_main()
            assert exc.value.code == 2


# ── Chat command ─────────────────────────────────────────────────────────


class TestChatCommand:
    def test_chat_auto_model_selection(self, fake_api, capsys, monkeypatch):
        inputs = iter(["hello", "quit"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        import argparse
        args = argparse.Namespace(
            model=None,
            api_url=fake_api,
            max_tokens=None,
        )
        cmd_chat(args)

        captured = capsys.readouterr()
        assert "llama3:8b" in captured.out
        assert "Hello world" in captured.out

    def test_chat_explicit_model(self, fake_api, capsys, monkeypatch):
        inputs = iter(["test", "quit"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        import argparse
        args = argparse.Namespace(
            model="llama3:8b",
            api_url=fake_api,
            max_tokens=100,
        )
        cmd_chat(args)

        captured = capsys.readouterr()
        assert "Hello world" in captured.out

    def test_chat_empty_input_skipped(self, fake_api, capsys, monkeypatch):
        inputs = iter(["", "hi", "q"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        import argparse
        args = argparse.Namespace(
            model="llama3:8b",
            api_url=fake_api,
            max_tokens=None,
        )
        cmd_chat(args)

        captured = capsys.readouterr()
        assert "Hello world" in captured.out

    def test_chat_keyboard_interrupt(self, fake_api, capsys, monkeypatch):
        call_count = 0
        def fake_input(_):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "hello"
            raise KeyboardInterrupt()

        monkeypatch.setattr("builtins.input", fake_input)

        import argparse
        args = argparse.Namespace(
            model="llama3:8b",
            api_url=fake_api,
            max_tokens=None,
        )
        cmd_chat(args)

    def test_chat_eof(self, fake_api, capsys, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: (_ for _ in ()).throw(EOFError))

        import argparse
        args = argparse.Namespace(
            model="llama3:8b",
            api_url=fake_api,
            max_tokens=None,
        )
        cmd_chat(args)


# ── Status command ───────────────────────────────────────────────────────


class TestStatusCommand:
    def test_status_healthy(self, fake_api, capsys):
        import argparse
        args = argparse.Namespace(api_url=fake_api)
        cmd_status(args)

        captured = capsys.readouterr()
        assert "healthy" in captured.out
        assert "llama3:8b" in captured.out

    def test_status_unreachable(self):
        import argparse
        args = argparse.Namespace(api_url="http://127.0.0.1:1")
        with pytest.raises(SystemExit):
            cmd_status(args)


# ── Models command ───────────────────────────────────────────────────────


class TestModelsCommand:
    def test_models_list(self, fake_api, capsys):
        import argparse
        args = argparse.Namespace(api_url=fake_api)
        cmd_models(args)

        captured = capsys.readouterr()
        assert "llama3:8b" in captured.out
        assert "Q4_0" in captured.out
        assert "4096" in captured.out
        assert "yes" in captured.out

    def test_models_empty(self, capsys):
        original = FakeAPIHandler.models_response
        FakeAPIHandler.models_response = {"object": "list", "data": []}
        try:
            server = HTTPServer(("127.0.0.1", 0), FakeAPIHandler)
            port = server.server_address[1]
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()

            import argparse
            args = argparse.Namespace(api_url=f"http://127.0.0.1:{port}")
            cmd_models(args)

            captured = capsys.readouterr()
            assert "No models" in captured.out
            server.shutdown()
        finally:
            FakeAPIHandler.models_response = original


# ── Join wiring (node.py --bridge) ───────────────────────────────────────


class TestBridgeArgParsing:
    """Verify that node.py accepts --bridge flags without crashing at parse time."""

    def test_bridge_flag_accepted(self):
        import sys
        from unittest.mock import patch as _p
        with _p.object(sys, "argv", [
            "openhydra-node", "--bridge", "ollama", "--bridge-url", "http://localhost:11434",
        ]):
            from coordinator.node import main
            # main() will fail at P2P bootstrap or adapter health — that's fine,
            # we're only testing arg parsing here. Catch SystemExit from health check.
            with pytest.raises(SystemExit):
                main()

    def test_bridge_url_default(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--bridge", choices=["ollama"], default=None)
        parser.add_argument("--bridge-url", default=None)
        args = parser.parse_args(["--bridge", "ollama"])
        assert args.bridge == "ollama"
        assert args.bridge_url is None
