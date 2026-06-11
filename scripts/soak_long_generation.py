#!/usr/bin/env python3
# Copyright 2026 OpenHydra contributors — Apache 2.0
"""Long-run relay soak — validates the F4 tensor-stream fix on real hardware.

Before the F4 fix (split inbound read/write halves + no per-token response-
handle leak), ring sessions over a relay died on long generations (~800+
tokens) because unread ACK bytes stalled QUIC flow control until the 250 ms
write timeout tore the circuit down. This script drives a single long
generation through the coordinator API and asserts it completes, then scans
the coordinator log for circuit-death / stall markers.

Run on a GPU/relay deployment (NOT in CI — needs a live coordinator + peers):

    python3 scripts/soak_long_generation.py \
        --url http://127.0.0.1:8080 \
        --model openhydra-qwen3.5-2b \
        --max-tokens 2000 \
        --log /tmp/coordinator.log

Exit 0 = generation completed and no circuit-death markers seen.
Exit 1 = generation truncated / errored, or stall markers found in the log.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request


# Markers that indicate the pre-F4 failure mode (or other circuit collapse).
_DEATH_MARKERS = (
    "write timed out",
    "circuit died",
    "circuit closed",
    "circuit collapse",
    "max_circuit",
    "ring: tensor_stream re-inject failed",
    "reinjectfailed",
    "session timed out",
    "stream reset",
    "insufficientdata",
)


def _post(url: str, model: str, prompt: str, max_tokens: int, timeout: float) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _completion_tokens(resp: dict) -> int:
    usage = resp.get("usage") or {}
    if "completion_tokens" in usage:
        return int(usage["completion_tokens"])
    # Fall back to a rough word count of the content.
    try:
        return len(resp["choices"][0]["message"]["content"].split())
    except (KeyError, IndexError, AttributeError):
        return 0


def _scan_log(path: str) -> list[str]:
    hits: list[str] = []
    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                low = line.lower()
                for m in _DEATH_MARKERS:
                    if m.lower() in low:
                        hits.append(line.rstrip())
                        break
    except FileNotFoundError:
        print(f"[warn] log not found: {path} (skipping log scan)", file=sys.stderr)
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="openhydra-qwen3.5-2b")
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--min-tokens", type=int, default=0,
                    help="fail if fewer than this many tokens were generated "
                         "(default: 80%% of --max-tokens)")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--log", default="", help="coordinator log path to scan")
    ap.add_argument("--prompt", default=(
        "Write an extremely detailed, multi-section technical essay on the "
        "history and architecture of distributed systems, covering consensus, "
        "replication, partitioning, and fault tolerance. Be exhaustive."
    ))
    args = ap.parse_args()

    min_tokens = args.min_tokens or int(args.max_tokens * 0.8)

    print(f"[soak] {args.model} max_tokens={args.max_tokens} url={args.url}")
    t0 = time.perf_counter()
    try:
        resp = _post(args.url, args.model, args.prompt, args.max_tokens, args.timeout)
    except Exception as exc:  # noqa: BLE001 — soak tool, report any failure
        print(f"[FAIL] request errored (generation did not complete): {exc}")
        return 1
    dt = time.perf_counter() - t0

    toks = _completion_tokens(resp)
    tps = toks / dt if dt > 0 else 0.0
    print(f"[soak] completed: {toks} tokens in {dt:.1f}s ({tps:.2f} tok/s)")

    ok = True
    if toks < min_tokens:
        print(f"[FAIL] only {toks} tokens (< {min_tokens}); likely truncated "
              f"mid-stream (the pre-F4 circuit-death symptom).")
        ok = False

    if args.log:
        hits = _scan_log(args.log)
        if hits:
            ok = False
            print(f"[FAIL] {len(hits)} circuit-death/stall marker(s) in log:")
            for h in hits[:10]:
                print(f"    {h}")
        else:
            print("[soak] no circuit-death/stall markers in log")

    print("[PASS]" if ok else "[FAIL]")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
