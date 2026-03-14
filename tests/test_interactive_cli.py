"""Tests for 6.2 — Rich Interactive CLI.

Verifies the coordinator/interactive_cli.py module including the
RichOpenHydraShell class (when prompt_toolkit is available) and the fallback
OpenHydraShell.
"""
from __future__ import annotations

import io
import json
import unittest.mock as mock
from unittest.mock import MagicMock, patch


class TestRichCLI:
    """Tests for 6.2 — Rich Interactive CLI."""

    def test_module_imports_without_error(self):
        """coordinator.interactive_cli imports without error."""
        import coordinator.interactive_cli  # noqa: F401

    def test_use_rich_flag_reflects_availability(self):
        """_USE_RICH is True when prompt_toolkit is available, else False."""
        import coordinator.interactive_cli as cli
        try:
            import prompt_toolkit  # noqa: F401
            assert cli._USE_RICH is True
        except ImportError:
            assert cli._USE_RICH is False

    def test_rich_shell_handle_dispatches_models(self):
        """RichOpenHydraShell._handle('/models ...') calls _cmd_models."""
        import coordinator.interactive_cli as cli
        if not cli._USE_RICH:
            return  # skip if prompt_toolkit not installed

        shell = cli.RichOpenHydraShell(url="http://localhost:8080")
        with patch.object(shell, "_cmd_models") as mock_cmd:
            shell._handle("/models")
            mock_cmd.assert_called_once()

    def test_stream_request_yields_chunks(self):
        """_stream_request yields text chunks from SSE payload."""
        import coordinator.interactive_cli as cli
        if not cli._USE_RICH:
            return  # skip if prompt_toolkit not installed

        shell = cli.RichOpenHydraShell(url="http://localhost:8080")

        # Build a fake SSE response
        sse_lines = [
            b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n',
            b'data: {"choices": [{"delta": {"content": " world"}}]}\n',
            b'data: [DONE]\n',
        ]
        fake_response = MagicMock()
        fake_response.__enter__ = lambda s: s
        fake_response.__exit__ = MagicMock(return_value=False)
        fake_response.__iter__ = lambda s: iter(sse_lines)

        with patch("urllib.request.urlopen", return_value=fake_response):
            chunks = list(shell._stream_request("/v1/chat/completions", {"model": None}))

        assert chunks == ["Hello", " world"]

    def test_main_function_exists_and_is_callable(self):
        """main() function exists and is callable."""
        from coordinator.interactive_cli import main
        assert callable(main)

    def test_fallback_shell_exists(self):
        """OpenHydraShell (fallback) is always importable."""
        from coordinator.interactive_cli import OpenHydraShell
        shell = OpenHydraShell(base_url="http://localhost:8080")
        assert shell._base_url == "http://localhost:8080"
        assert shell._session_id is None

    def test_commands_list_contains_expected_commands(self):
        """_COMMANDS includes the standard slash-commands."""
        from coordinator.interactive_cli import _COMMANDS
        for cmd in ("/chat", "/models", "/status", "/help", "/exit"):
            assert cmd in _COMMANDS

    def test_history_file_path_is_in_home(self):
        """_HISTORY_FILE points to ~/.openhydra_history."""
        from coordinator.interactive_cli import _HISTORY_FILE
        from pathlib import Path
        assert _HISTORY_FILE == Path.home() / ".openhydra_history"

    def test_rich_shell_cmd_compaction_sets_mode(self):
        """_cmd_compaction sets _kv_mode correctly."""
        import coordinator.interactive_cli as cli
        if not cli._USE_RICH:
            return
        shell = cli.RichOpenHydraShell(url="http://localhost:8080")
        assert shell._kv_mode == "off"
        with patch("builtins.print"):
            shell._cmd_compaction("auto")
        assert shell._kv_mode == "auto"
        with patch("builtins.print"):
            shell._cmd_compaction("on")
        assert shell._kv_mode == "on"
        with patch("builtins.print"):
            shell._cmd_compaction("off")
        assert shell._kv_mode == "off"
