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

"""coordinator.interactive_cli — OpenHydra interactive shell.

Two implementations are available:

* **RichOpenHydraShell** (preferred) — uses ``prompt_toolkit`` for persistent
  history, tab-completion, a live bottom toolbar, and SSE streaming output.
  Activated automatically when ``prompt_toolkit>=3.0`` is installed
  (``pip install "openhydra[shell]"``).

* **OpenHydraShell** (fallback) — the original ``cmd.Cmd``-based shell.
  Always available with zero extra dependencies.

Both implementations are started via :func:`main`.
"""
from __future__ import annotations

import cmd
import json
import os
import shutil
import textwrap
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


# ─── Capability flag ─────────────────────────────────────────────────────────

_USE_RICH = False
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.styles import Style
    _USE_RICH = True
except ImportError:
    pass

_HISTORY_FILE = Path.home() / ".openhydra_history"

_COMMANDS = [
    "/chat", "/ask", "/status", "/balance", "/models",
    "/model", "/session", "/compaction", "/help", "/exit", "/quit",
]

_STYLE_DICT = {
    "prompt":     "ansiblue bold",
    "cmd":        "ansigreen",
    "error":      "ansired bold",
    "model-name": "ansicyan",
}

# ─────────────────────────────────────────────────────────────────────────────
# Fallback: original cmd.Cmd shell (always available)
# ─────────────────────────────────────────────────────────────────────────────


class OpenHydraShell(cmd.Cmd):
    intro = "OpenHydra interactive shell. Type 'help' for commands.\n"
    prompt = "openhydra> "

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        client_id: str = "cli",
    ) -> None:
        super().__init__()
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client_id = client_id
        self._session_id: str | None = None
        self._model: str | None = None

    # --- HTTP helper ---

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        """Make an HTTP request to the coordinator. Returns parsed JSON."""
        url = f"{self._base_url}{path}"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            try:
                return json.loads(exc.read().decode())
            except Exception:
                return {"error": f"HTTP {exc.code}: {exc.reason}"}
        except urllib.error.URLError:
            print(
                f"Cannot connect to coordinator at {self._base_url} — is it running?"
            )
            return {}

    # --- Formatting helper ---

    def _print_response(self, text: str) -> None:
        """Print response text with blank lines, wrapping if terminal is narrow."""
        width = shutil.get_terminal_size(fallback=(120, 24)).columns
        print()
        if width < 100:
            print(textwrap.fill(text, width=width - 2))
        else:
            print(text)
        print()

    # --- Commands ---

    def do_chat(self, line: str) -> None:
        """chat <message> — Send a chat message (session maintained across turns)."""
        line = line.strip()
        if not line:
            print("Usage: chat <message text>")
            return
        is_new_session = self._session_id is None
        if is_new_session:
            self._session_id = str(uuid.uuid4())
        body: dict[str, Any] = {
            "messages": [{"role": "user", "content": line}],
            "client_id": self._client_id,
            "session_id": self._session_id,
        }
        if self._model:
            body["model"] = self._model
        result = self._request("POST", "/v1/chat/completions", body)
        if not result:
            return
        if "error" in result:
            print(f"Error: {json.dumps(result, indent=2)}")
            return
        try:
            text = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            print(f"Unexpected response: {json.dumps(result, indent=2)}")
            return
        self._print_response(text)
        if is_new_session:
            print(f"[session: {self._session_id}]")

    def do_ask(self, line: str) -> None:
        """ask <prompt> — Single prompt completion (no session tracking)."""
        line = line.strip()
        if not line:
            print("Usage: ask <prompt>")
            return
        body: dict[str, Any] = {
            "prompt": line,
            "client_id": self._client_id,
        }
        if self._model:
            body["model"] = self._model
        result = self._request("POST", "/v1/completions", body)
        if not result:
            return
        if "error" in result:
            print(f"Error: {json.dumps(result, indent=2)}")
            return
        try:
            text = result["choices"][0]["text"]
        except (KeyError, IndexError):
            print(f"Unexpected response: {json.dumps(result, indent=2)}")
            return
        self._print_response(text)

    def do_status(self, line: str) -> None:
        """status — Show a compact network status summary."""
        result = self._request("GET", "/v1/network/status")
        if not result:
            return
        if "error" in result:
            print(f"Error: {json.dumps(result, indent=2)}")
            return
        peers = result.get("peers", result.get("peer_count", "?"))
        models = result.get("models", result.get("model_count", "?"))
        replication = result.get("replication", result.get("replication_status", "?"))
        print(f"  Peers      : {peers}")
        print(f"  Models     : {models}")
        print(f"  Replication: {replication}")

    def do_balance(self, line: str) -> None:
        """balance [client_id] — Show balance, stake, and rewards for a client."""
        client_id = line.strip() or self._client_id
        result = self._request("GET", f"/v1/account/balance?client_id={client_id}")
        if not result:
            return
        if "error" in result:
            print(f"Error: {json.dumps(result, indent=2)}")
            return
        print(f"  Client ID : {client_id}")
        print(f"  Balance   : {result.get('balance', 'N/A')}")
        print(f"  Stake     : {result.get('stake', 'N/A')}")
        print(f"  Rewards   : {result.get('rewards', 'N/A')}")

    def do_models(self, line: str) -> None:
        """models — List available model IDs."""
        result = self._request("GET", "/v1/models")
        if not result:
            return
        if "error" in result:
            print(f"Error: {json.dumps(result, indent=2)}")
            return
        models: list[str] = []
        if isinstance(result, list):
            models = [str(m.get("id", m)) if isinstance(m, dict) else str(m) for m in result]
        elif isinstance(result, dict):
            raw = result.get("data", result.get("models", []))
            if isinstance(raw, list):
                models = [str(m.get("id", m)) if isinstance(m, dict) else str(m) for m in raw]
            else:
                models = [str(raw)]
        if not models:
            print("(no models found)")
            return
        for m in models:
            marker = " *" if m == self._model else ""
            print(f"  {m}{marker}")

    def do_session(self, line: str) -> None:
        """session reset | session show — Manage the current chat session."""
        cmd_arg = line.strip().lower()
        if cmd_arg == "reset":
            self._session_id = None
            print("Session cleared. Next chat turn starts a new session.")
        elif cmd_arg == "show":
            print(f"Session: {self._session_id or '(none)'}")
        else:
            print("Usage: session reset | session show")

    def do_model(self, line: str) -> None:
        """model <model_id> | model clear — Set or clear the default model."""
        arg = line.strip()
        if not arg:
            print("Usage: model <model_id> | model clear")
            return
        if arg.lower() == "clear":
            self._model = None
            print("Model cleared (using server default).")
        else:
            self._model = arg
            print(f"Model set to: {self._model}")

    def do_exit(self, line: str) -> bool:
        """exit — Exit the shell."""
        print("Bye.")
        return True

    def do_quit(self, line: str) -> bool:
        """quit — Exit the shell."""
        print("Bye.")
        return True

    def do_EOF(self, line: str) -> bool:
        """EOF (Ctrl-D) — Exit the shell."""
        print("\nBye.")
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Rich shell — only defined when prompt_toolkit is available
# ─────────────────────────────────────────────────────────────────────────────

if _USE_RICH:
    class RichOpenHydraShell:
        """Interactive shell using prompt_toolkit for a rich terminal experience.

        Features:
        - Persistent command history (``~/.openhydra_history``)
        - Tab-completion for ``/cmd`` commands
        - Live bottom toolbar showing model + session
        - SSE streaming for ``/chat`` responses
        """

        def __init__(
            self,
            url: str,
            api_key: str | None = None,
            client_id: str = "cli",
            model: str | None = None,
        ) -> None:
            self._url = url.rstrip("/")
            self._api_key = api_key
            self._client_id = client_id
            self._model: str | None = model
            self._session_id: str | None = None
            self._kv_mode: str = "off"

            style = Style.from_dict(_STYLE_DICT)
            self._session = PromptSession(
                history=FileHistory(str(_HISTORY_FILE)),
                completer=WordCompleter(_COMMANDS, ignore_case=True),
                complete_while_typing=True,
                style=style,
                bottom_toolbar=self._toolbar,
                enable_history_search=True,
            )

        # ── Helpers ─────────────────────────────────────────────────────────

        def _auth_header(self) -> dict[str, str]:
            if self._api_key:
                return {"Authorization": f"Bearer {self._api_key}"}
            return {}

        def _toolbar(self) -> str:
            sess = (self._session_id[:12] + "…") if self._session_id else "none"
            model = self._model or "default"
            return f" model: {model} | session: {sess} | /help for commands"

        def _prompt_text(self) -> HTML:
            model = self._model or "default"
            return HTML(
                f'<prompt>openhydra</prompt>'
                f' <model-name>[{model}]</model-name>'
                f' <prompt>❯</prompt> '
            )

        def _request(self, method: str, path: str, body: dict | None = None) -> dict:
            url = f"{self._url}{path}"
            headers: dict[str, str] = {
                "Content-Type": "application/json",
                **self._auth_header(),
            }
            data = json.dumps(body).encode() if body is not None else None
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as exc:
                try:
                    return json.loads(exc.read().decode())
                except Exception:
                    return {"error": f"HTTP {exc.code}: {exc.reason}"}
            except urllib.error.URLError:
                return {"error": f"Cannot connect to {self._url}"}

        def _stream_request(self, path: str, body: dict):
            """Yield text delta chunks from an SSE /v1/chat/completions response."""
            url = f"{self._url}{path}"
            headers: dict[str, str] = {
                "Content-Type": "application/json",
                **self._auth_header(),
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode(),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                for raw_line in resp:
                    line = raw_line.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except Exception:
                        pass

        # ── Command dispatch ─────────────────────────────────────────────────

        def _handle(self, line: str) -> bool:
            """Dispatch a line to the appropriate command. Returns True to exit."""
            line = line.strip()
            if not line:
                return False

            if line.startswith("/"):
                parts = line[1:].split(None, 1)
                cmd_name = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
            else:
                # Bare text → treat as /chat
                cmd_name = "chat"
                args = line

            dispatch = {
                "chat":       self._cmd_chat,
                "ask":        self._cmd_ask,
                "status":     self._cmd_status,
                "balance":    self._cmd_balance,
                "models":     self._cmd_models,
                "model":      self._cmd_model,
                "session":    self._cmd_session,
                "compaction": self._cmd_compaction,
                "help":       self._cmd_help,
                "exit":       lambda _: None,
                "quit":       lambda _: None,
            }

            if cmd_name in ("exit", "quit"):
                print("Bye.")
                return True

            fn = dispatch.get(cmd_name)
            if fn is None:
                print(f"\033[31merror: unknown command /{cmd_name} — type /help\033[0m")
            else:
                fn(args)
            return False

        # ── Individual commands ──────────────────────────────────────────────

        def _cmd_chat(self, args: str) -> None:
            text = args.strip()
            if not text:
                print("Usage: /chat <message>  (or just type your message)")
                return
            if not self._session_id:
                self._session_id = str(uuid.uuid4())
            body = {
                "model": self._model,
                "messages": [{"role": "user", "content": text}],
                "client_id": self._client_id,
                "session_id": self._session_id,
                "stream": True,
            }
            print()
            print("\033[36massistant>\033[0m ", end="", flush=True)
            try:
                for chunk in self._stream_request("/v1/chat/completions", body):
                    print(chunk, end="", flush=True)
            except Exception as e:
                print(f"\n\033[31merror: {e}\033[0m")
            print("\n")

        def _cmd_ask(self, args: str) -> None:
            text = args.strip()
            if not text:
                print("Usage: /ask <prompt>")
                return
            body: dict[str, Any] = {"prompt": text, "client_id": self._client_id}
            if self._model:
                body["model"] = self._model
            result = self._request("POST", "/v1/completions", body)
            if "error" in result:
                print(f"\033[31merror: {result['error']}\033[0m")
                return
            try:
                print(f"\n{result['choices'][0]['text']}\n")
            except (KeyError, IndexError):
                print(json.dumps(result, indent=2))

        def _cmd_status(self, args: str) -> None:
            result = self._request("GET", "/v1/network/status")
            if "error" in result:
                print(f"\033[31merror: {result['error']}\033[0m")
                return
            peers = result.get("peers", result.get("peer_count", "?"))
            models_val = result.get("models", result.get("model_count", "?"))
            replication = result.get("replication", result.get("replication_status", "?"))
            print(f"  Peers      : {peers}")
            print(f"  Models     : {models_val}")
            print(f"  Replication: {replication}")

        def _cmd_balance(self, args: str) -> None:
            client_id = args.strip() or self._client_id
            result = self._request("GET", f"/v1/account/balance?client_id={client_id}")
            if "error" in result:
                print(f"\033[31merror: {result['error']}\033[0m")
                return
            print(f"  Client ID : {client_id}")
            print(f"  Balance   : {result.get('balance', 'N/A')}")
            print(f"  Stake     : {result.get('stake', 'N/A')}")
            print(f"  Rewards   : {result.get('rewards', 'N/A')}")

        def _cmd_models(self, args: str) -> None:
            result = self._request("GET", "/v1/models")
            if "error" in result:
                print(f"\033[31merror: {result['error']}\033[0m")
                return
            raw = result.get("data", result) if isinstance(result, dict) else result
            if not isinstance(raw, list):
                print(json.dumps(result, indent=2))
                return
            if not raw:
                print("(no models found)")
                return
            # Pretty table
            col_id   = max((len(str(m.get("id", ""))) for m in raw), default=8)
            col_id   = max(col_id, 8)
            col_q    = max((len(str(m.get("recommended_quantization", ""))) for m in raw), default=5)
            col_q    = max(col_q, 5)
            hdr = f"  {'Model ID':<{col_id}}  {'Quant':<{col_q}}  {'Peers':>5}  {'Online':>6}  Tags"
            print(f"\n{hdr}")
            print(f"  {'─' * col_id}  {'─' * col_q}  {'─' * 5}  {'─' * 6}  ─────────")
            for m in raw:
                mid   = str(m.get("id", ""))
                quant = str(m.get("recommended_quantization", ""))
                req   = int(m.get("required_replicas", 1))
                healthy = int(m.get("healthy_peers", 0))
                online = "✓" if healthy >= req else "✗"
                tags  = ", ".join(m.get("tags", []))
                mark  = " *" if mid == self._model else ""
                print(f"  {mid:<{col_id}}  {quant:<{col_q}}  {req:>5}  {online:>6}  {tags}{mark}")
            print()

        def _cmd_model(self, args: str) -> None:
            arg = args.strip()
            if not arg:
                print(f"Current model: {self._model or '(default)'}")
                return
            if arg.lower() == "clear":
                self._model = None
                print("Model cleared (using server default).")
            else:
                self._model = arg
                print(f"Model set to: \033[36m{self._model}\033[0m")

        def _cmd_session(self, args: str) -> None:
            arg = args.strip().lower()
            if arg == "reset":
                self._session_id = None
                print("Session cleared.")
            else:
                print(f"Session: {self._session_id or '(none)'}")

        def _cmd_compaction(self, args: str) -> None:
            arg = args.strip().lower()
            if arg in {"off", "auto", "on"}:
                self._kv_mode = arg
                print(f"KV compaction mode set to: \033[36m{arg}\033[0m")
                print("(Takes effect on next node start)")
            else:
                print(f"Current compaction mode: {self._kv_mode}")
                print("Usage: /compaction off|auto|on")

        def _cmd_help(self, args: str) -> None:
            print("""
  /chat <message>        Send a chat message (streaming, session tracked)
  /ask <prompt>          Single completion (no session)
  /status                Network status summary
  /balance [client_id]   Account balance
  /models                List all models (pretty table)
  /model <id>|clear      Set or clear active model
  /session reset|show    Manage current chat session
  /compaction off|auto|on  Set KV compaction mode (local setting)
  /help                  Show this help
  /exit, /quit           Exit
  <message>              Shortcut for /chat <message>
""")

        # ── Main loop ────────────────────────────────────────────────────────

        def run(self) -> None:
            print("OpenHydra shell (rich mode). Type /help for commands, /exit to quit.\n")
            while True:
                try:
                    line = self._session.prompt(self._prompt_text())
                except (KeyboardInterrupt, EOFError):
                    print("\nBye.")
                    break
                if self._handle(line):
                    break


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="OpenHydra interactive shell")
    parser.add_argument(
        "--url", default="http://127.0.0.1:8080", help="Coordinator URL"
    )
    parser.add_argument(
        "--api-key", default=None, help="API key (or OPENHYDRA_API_KEY env)"
    )
    parser.add_argument(
        "--client-id", default="cli", help="Client ID for balance/economy"
    )
    parser.add_argument("--model", default=None, help="Default model to use")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENHYDRA_API_KEY")

    if _USE_RICH:
        RichOpenHydraShell(
            url=args.url,
            api_key=api_key,
            client_id=args.client_id,
            model=args.model,
        ).run()
    else:
        shell = OpenHydraShell(
            base_url=args.url,
            api_key=api_key,
            client_id=args.client_id,
        )
        if args.model:
            shell._model = args.model
        shell.cmdloop()


if __name__ == "__main__":
    main()
