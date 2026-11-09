#!/usr/bin/env python3
# Copyright 2026 OpenHydra contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Load-test an OpenHydra gateway (or any OpenAI-compatible endpoint) and check a latency target.

Sends chat completions, streamed so the time to the first token is measured where a client sees it.
Requests are grouped into conversations: each conversation's later turns resend its history, as a
coding agent does, which is what the gateway's cache affinity keys on. Standard library only.

  python3 ops/loadtest/chat_load.py --url http://127.0.0.1:16527 --model qwen2.5:1.5b \\
      --requests 200 --concurrency 16 --turns 4 --ttft-p95-target 3

Prints a JSON summary; exits 1 when the time-to-first-token p95 misses `--ttft-p95-target`, or any
request failed and `--allow-errors` wasn't given.
"""
import argparse
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

TOPICS = ["a sorting function", "a REST handler", "a unit test", "a SQL query", "a config parser",
          "a retry loop", "a CLI flag", "a log message", "a cache", "a rate limiter"]


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, max(0, int(round(p / 100 * len(xs))) - 1))], 3)


def one(url, headers, body, timeout):
    """One streamed completion: (ok, ttft_s, total_s, tokens, finish, error)."""
    req = urllib.request.Request(url, json.dumps(body).encode(), headers)
    t0 = time.monotonic()
    ttft, chunks, finish = None, 0, None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                ev = json.loads(data)
                if "error" in ev:
                    return (False, None, time.monotonic() - t0, chunks, None, str(ev["error"])[:120])
                for ch in ev.get("choices", []):
                    if ch.get("delta", {}).get("content"):
                        if ttft is None:
                            ttft = time.monotonic() - t0
                        chunks += 1
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
        ok = finish not in (None, "error")
        return (ok, ttft, time.monotonic() - t0, chunks, finish, None if ok else f"finish={finish}")
    except urllib.error.HTTPError as e:
        return (False, None, time.monotonic() - t0, 0, None, f"HTTP {e.code}: {e.read()[:120]!r}")
    except Exception as e:  # noqa: BLE001 - every failure is a counted error
        return (False, None, time.monotonic() - t0, 0, None, str(e)[:120])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="gateway base URL, e.g. http://127.0.0.1:16527")
    ap.add_argument("--model", required=True)
    ap.add_argument("--requests", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--turns", type=int, default=3, help="turns per conversation (history is resent)")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--ttft-p95-target", type=float, default=None, help="seconds; exit 1 when missed")
    ap.add_argument("--allow-errors", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    url = a.url.rstrip("/") + "/v1/chat/completions"
    headers = {"content-type": "application/json"}
    if a.api_key:
        headers["authorization"] = f"Bearer {a.api_key}"
    rng = random.Random(a.seed)
    conversations = max(1, a.requests // max(1, a.turns))
    # Each conversation runs its turns in order on one worker; conversations run side by side.
    results, lock = [], threading.Lock()

    def conversation(c):
        topic = rng.choice(TOPICS)
        messages = [{"role": "system", "content": f"You are a coding assistant. Conversation {c}."},
                    {"role": "user", "content": f"Write {topic} in Python, briefly."}]
        for _ in range(a.turns):
            body = {"model": a.model, "messages": messages, "max_tokens": a.max_tokens, "stream": True}
            r = one(url, headers, body, a.timeout)
            with lock:
                results.append(r)
            messages = messages + [{"role": "assistant", "content": "(reply)"},
                                   {"role": "user", "content": "Now make it shorter."}]

    t0 = time.monotonic()
    with ThreadPoolExecutor(a.concurrency) as pool:
        list(pool.map(conversation, range(conversations)))
    wall = time.monotonic() - t0
    ok = [r for r in results if r[0]]
    ttfts = [r[1] for r in ok if r[1] is not None]
    totals = [r[2] for r in ok]
    errors = {}
    for r in results:
        if not r[0]:
            errors[r[5]] = errors.get(r[5], 0) + 1
    finishes = {}
    for r in ok:
        finishes[r[4]] = finishes.get(r[4], 0) + 1
    summary = {
        "requests": len(results), "ok": len(ok), "errors": errors, "wall_s": round(wall, 2),
        "requests_per_s": round(len(results) / wall, 2) if wall else None,
        "ttft_s": {"p50": pct(ttfts, 50), "p95": pct(ttfts, 95), "p99": pct(ttfts, 99), "max": pct(ttfts, 100)},
        "total_s": {"p50": pct(totals, 50), "p95": pct(totals, 95), "max": pct(totals, 100)},
        "finish_reasons": finishes, "conversations": conversations, "turns": a.turns,
    }
    passed = True
    if a.ttft_p95_target is not None:
        p95 = summary["ttft_s"]["p95"]
        summary["ttft_p95_target_s"] = a.ttft_p95_target
        passed = p95 is not None and p95 <= a.ttft_p95_target
    if errors and not a.allow_errors:
        passed = False
    summary["pass"] = passed
    print(json.dumps(summary, indent=2))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
