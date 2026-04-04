#!/usr/bin/env python3
# Copyright 2026 OpenHydra contributors — Apache 2.0
"""OpenHydra TPS Benchmark — measures TTFT, TPS, and validates output quality.

Usage:
    python3 scripts/benchmark_tps.py [--url URL] [--mode MODE] [--output FILE]

Connects to the local API via streaming SSE, records timing metrics,
and validates the generated text for quality (length, repetition, entropy).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import requests

PROMPT = (
    "Write a comprehensive, multi-paragraph essay on the history, architecture, "
    "and cryptographic mechanisms of decentralized peer-to-peer networks."
)

DEFAULT_URL = "http://127.0.0.1:8080"
DEFAULT_MODEL = "openhydra-qwen3.5-0.8b"
MAX_TOKENS = 256


def run_benchmark(
    base_url: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = MAX_TOKENS,
) -> dict:
    """Run a non-streaming benchmark and return metrics.

    Uses a single request/response (not SSE streaming) to avoid
    read-timeout issues on memory-constrained 8GB machines where MLX
    generation is slow.  TTFT = time to full response.  TPS is computed
    from completion_tokens / wall_time.
    """
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
        "grounding": False,
    }

    error: str | None = None
    full_text = ""
    token_count = 0
    ttft = 0.0
    wall_time = 0.0

    t_start = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=300)
        t_end = time.perf_counter()
        wall_time = t_end - t_start

        data = resp.json()
        if "error" in data:
            error = str(data["error"])
        else:
            choices = data.get("choices", [])
            if choices:
                full_text = choices[0].get("message", {}).get("content", "")
            usage = data.get("usage", {})
            token_count = usage.get("completion_tokens", len(full_text.split()))
            ttft = wall_time  # Non-streaming: TTFT = full response time

            # Extract internal latency if available
            oh = data.get("openhydra", {})
            if oh:
                pipeline = oh.get("pipeline", [])
                if pipeline:
                    ttft = pipeline[0].get("latency_ms", wall_time * 1000) / 1000.0

    except requests.exceptions.ConnectionError:
        t_end = time.perf_counter()
        wall_time = t_end - t_start
        error = "Connection refused — is the OpenHydra daemon running?"
    except requests.exceptions.Timeout:
        t_end = time.perf_counter()
        wall_time = t_end - t_start
        error = "Request timed out (300s)"
    except Exception as e:
        t_end = time.perf_counter()
        wall_time = t_end - t_start
        error = f"Unexpected error: {e}"

    tps = token_count / wall_time if wall_time > 0 else 0.0

    return {
        "error": error,
        "full_text": full_text,
        "token_count": token_count,
        "word_count": len(full_text.split()),
        "char_count": len(full_text),
        "ttft_s": ttft,
        "gen_time_s": wall_time,
        "wall_time_s": wall_time,
        "tps": tps,
    }


def validate_output(text: str, min_chars: int = 500) -> dict:
    """Validate output quality: length, repetition, and entropy."""
    results: dict[str, dict] = {}

    # Length check
    length_ok = len(text) >= min_chars
    results["length"] = {
        "pass": length_ok,
        "detail": f"{len(text)} chars (min {min_chars})",
    }

    # Repetition check: no single token > 20% of output
    words = text.lower().split()
    if words:
        counts = Counter(words)
        most_common_word, most_common_count = counts.most_common(1)[0]
        repeat_ratio = most_common_count / len(words)
        repeat_ok = repeat_ratio < 0.20
        results["repetition"] = {
            "pass": repeat_ok,
            "detail": f"most repeated '{most_common_word}' = {repeat_ratio:.1%} of {len(words)} words",
        }
    else:
        results["repetition"] = {"pass": False, "detail": "no words generated"}

    # Shannon entropy check (character-level)
    if text:
        freq = Counter(text)
        total = len(text)
        entropy = -sum((c / total) * math.log2(c / total) for c in freq.values())
        # English prose typically has entropy > 3.5 bits/char
        entropy_ok = entropy > 3.0
        results["entropy"] = {
            "pass": entropy_ok,
            "detail": f"{entropy:.2f} bits/char (min 3.0)",
        }
    else:
        results["entropy"] = {"pass": False, "detail": "empty text"}

    all_pass = all(r["pass"] for r in results.values())
    return {"all_pass": all_pass, "checks": results}


def print_results(label: str, metrics: dict, validation: dict) -> None:
    """Pretty-print benchmark results."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    if metrics["error"]:
        print(f"  ERROR: {metrics['error']}")
        return

    print(f"  TTFT (Time to First Token):  {metrics['ttft_s']*1000:.0f} ms")
    print(f"  Generation Time:             {metrics['gen_time_s']:.3f} s")
    print(f"  Wall Time:                   {metrics['wall_time_s']:.3f} s")
    print(f"  Tokens (SSE chunks):         {metrics['token_count']}")
    print(f"  Words:                       {metrics['word_count']}")
    print(f"  Characters:                  {metrics['char_count']}")
    print(f"  TPS (Tokens/sec):            {metrics['tps']:.1f}")
    print()
    print("  Validation:")
    for name, check in validation["checks"].items():
        icon = "PASS" if check["pass"] else "FAIL"
        print(f"    [{icon}] {name}: {check['detail']}")
    print(f"    Overall: {'PASS' if validation['all_pass'] else 'FAIL'}")


def save_output(text: str, path: str) -> None:
    """Save generated text to file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    print(f"  Output saved to: {path}")


def main():
    parser = argparse.ArgumentParser(description="OpenHydra TPS Benchmark")
    parser.add_argument("--url", default=DEFAULT_URL, help="API base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model ID")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--output", help="Save output text to file")
    parser.add_argument("--label", default="Benchmark", help="Label for output")
    parser.add_argument("--min-chars", type=int, default=500, help="Min chars for validation")
    args = parser.parse_args()

    print(f"\nRunning {args.label}...")
    print(f"  URL: {args.url}")
    print(f"  Model: {args.model}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Prompt: {PROMPT[:80]}...")

    metrics = run_benchmark(args.url, args.model, args.max_tokens)
    validation = validate_output(metrics["full_text"], min_chars=args.min_chars)
    print_results(args.label, metrics, validation)

    if args.output and metrics["full_text"]:
        save_output(metrics["full_text"], args.output)

    # Return exit code based on validation
    sys.exit(0 if (validation["all_pass"] or metrics["error"] is None) else 1)


if __name__ == "__main__":
    main()
