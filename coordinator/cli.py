"""openhydra CLI — unified entry point for join, chat, status, models.

Usage:
    openhydra join --bridge ollama [--bridge-url URL] [--api-port PORT]
    openhydra chat [--model MODEL] [--api-url URL]
    openhydra status [--api-url URL]
    openhydra models [--api-url URL]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def _api_get(url: str) -> dict:
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        print(f"Error: cannot reach {url} — {e.reason}", file=sys.stderr)
        raise SystemExit(1)


def _api_post_stream(url: str, body: dict):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=300)
    except urllib.error.URLError as e:
        print(f"Error: cannot reach {url} — {e.reason}", file=sys.stderr)
        raise SystemExit(1)
    for line in resp:
        decoded = line.decode("utf-8", errors="replace").strip()
        if not decoded:
            continue
        if decoded.startswith("data: "):
            payload = decoded[6:]
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        print(content, end="", flush=True)
            except json.JSONDecodeError:
                continue
    print()


def cmd_join(args: argparse.Namespace) -> None:
    """Start a node in bridge mode."""
    import sys as _sys
    inject = ["--bridge", args.bridge]
    if args.bridge_url:
        inject += ["--bridge-url", args.bridge_url]
    if args.api_port != 8080:
        inject += ["--api-port", str(args.api_port)]
    if args.api_host != "127.0.0.1":
        inject += ["--api-host", args.api_host]
    if args.peer_id:
        inject += ["--peer-id", args.peer_id]

    _sys.argv = ["openhydra-node"] + inject
    from coordinator.node import main
    main()


def cmd_chat(args: argparse.Namespace) -> None:
    """Interactive chat session."""
    api_url = args.api_url.rstrip("/")

    if not args.model:
        models_resp = _api_get(f"{api_url}/v1/models")
        model_list = models_resp.get("data", [])
        if not model_list:
            print("No models available.", file=sys.stderr)
            raise SystemExit(1)
        model_id = model_list[0]["id"]
        print(f"Using model: {model_id}")
    else:
        model_id = args.model

    messages: list[dict[str, str]] = []
    print("Chat started. Type 'quit' or Ctrl-C to exit.\n")

    try:
        while True:
            try:
                user_input = input("> ")
            except EOFError:
                break
            if user_input.strip().lower() in ("quit", "exit", "q"):
                break
            if not user_input.strip():
                continue

            messages.append({"role": "user", "content": user_input})

            body = {
                "model": model_id,
                "messages": messages,
                "stream": True,
            }
            if args.max_tokens:
                body["max_tokens"] = args.max_tokens

            url = f"{api_url}/v1/chat/completions"

            data = json.dumps(body).encode()
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"},
            )
            try:
                resp = urllib.request.urlopen(req, timeout=300)
            except urllib.error.URLError as e:
                print(f"\nError: {e.reason}", file=sys.stderr)
                messages.pop()
                continue

            assistant_content = []
            for line in resp:
                decoded = line.decode("utf-8", errors="replace").strip()
                if not decoded or not decoded.startswith("data: "):
                    continue
                payload = decoded[6:]
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            print(content, end="", flush=True)
                            assistant_content.append(content)
                except json.JSONDecodeError:
                    continue
            print()
            messages.append({
                "role": "assistant",
                "content": "".join(assistant_content),
            })

    except KeyboardInterrupt:
        print()


def cmd_status(args: argparse.Namespace) -> None:
    """Show local node status."""
    api_url = args.api_url.rstrip("/")

    try:
        models = _api_get(f"{api_url}/v1/models")
        model_list = models.get("data", [])
        healthy = True
    except SystemExit:
        model_list = []
        healthy = False

    print(f"Node:    {'healthy' if healthy else 'unhealthy'}")
    print(f"Models:  {len(model_list)}")
    for m in model_list:
        mid = m.get("id", "?")
        oh = m.get("openhydra", {})
        quant = oh.get("quantization", "")
        ctx = oh.get("context_length", "")
        print(f"  - {mid} ({quant}, ctx={ctx})")

    try:
        supernodes = _api_get(f"{api_url}/v1/supernodes")
        sn_list = supernodes if isinstance(supernodes, list) else supernodes.get("supernodes", [])
        print(f"Supernodes: {len(sn_list)}")
        for sn in sn_list:
            name = sn.get("name", "?")
            sn_models = sn.get("models", [])
            sn_healthy = sn.get("healthy", False)
            print(f"  - {name}: {', '.join(sn_models)} ({'ok' if sn_healthy else 'down'})")
    except SystemExit:
        pass


def cmd_models(args: argparse.Namespace) -> None:
    """List available models."""
    api_url = args.api_url.rstrip("/")
    models = _api_get(f"{api_url}/v1/models")
    model_list = models.get("data", [])
    if not model_list:
        print("No models available.")
        return

    print(f"{'MODEL':<40} {'QUANT':<8} {'CTX':<8} {'STREAMING'}")
    print("-" * 70)
    for m in model_list:
        mid = m.get("id", "?")
        oh = m.get("openhydra", {})
        quant = oh.get("quantization", "?")
        ctx = str(oh.get("context_length", "?"))
        stream = "yes" if oh.get("supports_streaming", False) else "no"
        print(f"{mid:<40} {quant:<8} {ctx:<8} {stream}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="openhydra",
        description="OpenHydra — decentralised LLM inference",
    )
    sub = parser.add_subparsers(dest="command")

    # --- join ---
    join_p = sub.add_parser("join", help="Join the mesh as a supernode bridge")
    join_p.add_argument(
        "--bridge", required=True, choices=["ollama"],
        help="Runtime to bridge (currently: ollama)",
    )
    join_p.add_argument(
        "--bridge-url", default=None,
        help="URL of the bridged runtime (default: http://localhost:11434)",
    )
    join_p.add_argument("--api-port", type=int, default=8080)
    join_p.add_argument("--api-host", default="127.0.0.1")
    join_p.add_argument("--peer-id", default=None)

    # --- chat ---
    chat_p = sub.add_parser("chat", help="Interactive chat with a model")
    chat_p.add_argument("--model", default=None, help="Model ID to chat with")
    chat_p.add_argument(
        "--api-url", default="http://127.0.0.1:8080",
        help="URL of the OpenHydra API (default: http://127.0.0.1:8080)",
    )
    chat_p.add_argument("--max-tokens", type=int, default=None)

    # --- status ---
    status_p = sub.add_parser("status", help="Show node status")
    status_p.add_argument(
        "--api-url", default="http://127.0.0.1:8080",
    )

    # --- models ---
    models_p = sub.add_parser("models", help="List available models")
    models_p.add_argument(
        "--api-url", default="http://127.0.0.1:8080",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        raise SystemExit(0)

    dispatch = {
        "join": cmd_join,
        "chat": cmd_chat,
        "status": cmd_status,
        "models": cmd_models,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
