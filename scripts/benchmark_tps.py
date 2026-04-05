#!/usr/bin/env python3
# Copyright 2026 OpenHydra contributors — Apache 2.0
"""OpenHydra TPS Benchmark — measures TTFT, TPS, and validates output quality.

Supports Local Mode, Swarm Mode, and Chunked Prefill benchmarking.

Usage:
    # Local mode (default)
    python3 scripts/benchmark_tps.py --label "Local"

    # Swarm mode on custom port
    python3 scripts/benchmark_tps.py --url http://127.0.0.1:8090 --model openhydra-smollm2-360m

    # Short prompt (<256 words)
    python3 scripts/benchmark_tps.py --prompt-mode short

    # Long prompt (>256 words, triggers chunked prefill)
    python3 scripts/benchmark_tps.py --prompt-mode long

    # Full comparison: short vs long
    python3 scripts/benchmark_tps.py --prompt-mode compare
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

SHORT_PROMPT = "Once upon a time there was a young girl who loved to explore the forest near her village."

LONG_PROMPT = (
    "Write a comprehensive, multi-paragraph essay on the history, architecture, "
    "and cryptographic mechanisms of decentralized peer-to-peer networks. "
    "Cover the evolution from early file sharing systems like Napster and Gnutella "
    "through the BitTorrent protocol and its use of distributed hash tables for "
    "peer discovery and content addressing. Discuss how the InterPlanetary File "
    "System builds on content-addressable storage and Merkle DAG structures to "
    "create a permanent web of linked data. Explain the role of Kademlia DHT in "
    "modern peer-to-peer overlay networks, including how XOR-based distance metrics "
    "enable efficient routing with logarithmic lookup complexity. Analyze the "
    "challenges of NAT traversal, including STUN, TURN, and ICE protocols that "
    "allow peers behind firewalls to establish direct connections. Cover onion "
    "routing and its privacy guarantees, explaining how layered encryption through "
    "multiple relay nodes prevents any single node from knowing both the source and "
    "destination of a message. Discuss proof-of-work and proof-of-stake consensus "
    "mechanisms, comparing their energy efficiency, security guarantees, and "
    "finality properties. Explain how economic incentive structures like token "
    "economies and reputation systems motivate volunteer node operators to "
    "contribute compute, bandwidth, and storage resources without a central "
    "authority coordinating payments. Compare pipeline parallelism and tensor "
    "parallelism for distributed inference, analyzing the latency and throughput "
    "tradeoffs when splitting large language models across heterogeneous volunteer "
    "hardware connected over the public internet. Finally, discuss the emerging "
    "field of decentralized AI inference networks and how techniques like speculative "
    "decoding, activation compression, and adaptive routing can enable volunteer "
    "laptops to collectively run frontier models exceeding the memory of any "
    "single machine, democratizing access to artificial intelligence."
)

DEFAULT_URL = "http://127.0.0.1:8080"
DEFAULT_MODEL = "openhydra-qwen3.5-0.8b"
MAX_TOKENS = 64


def run_benchmark(
    base_url: str,
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = MAX_TOKENS,
    endpoint: str = "/v1/completions",
) -> dict:
    """Run a non-streaming benchmark and return metrics."""
    url = f"{base_url}{endpoint}"

    if endpoint == "/v1/completions":
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": False,
            "grounding": False,
        }
    else:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
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
    pipeline_mode = "unknown"
    pipeline_hops = 0
    pipeline_traces = []
    openhydra_meta = {}

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
                choice = choices[0]
                if "text" in choice:
                    full_text = choice["text"]
                elif "message" in choice:
                    full_text = choice["message"].get("content", "")

            usage = data.get("usage", {})
            token_count = usage.get("completion_tokens", len(full_text.split()))
            ttft = wall_time

            oh = data.get("openhydra", {})
            openhydra_meta = oh
            pipeline_mode = oh.get("pipeline_mode", "unknown")
            pipeline_traces = oh.get("pipeline", [])
            pipeline_hops = len(pipeline_traces)

            if pipeline_traces:
                ttft = pipeline_traces[0].get("latency_ms", wall_time * 1000) / 1000.0

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
        "prompt_word_count": len(prompt.split()),
        "ttft_s": ttft,
        "gen_time_s": wall_time,
        "wall_time_s": wall_time,
        "tps": tps,
        "pipeline_mode": pipeline_mode,
        "pipeline_hops": pipeline_hops,
        "pipeline_traces": pipeline_traces,
        "openhydra": openhydra_meta,
    }


def validate_output(text: str, min_chars: int = 10) -> dict:
    """Validate output quality."""
    results: dict[str, dict] = {}

    length_ok = len(text) >= min_chars
    results["length"] = {
        "pass": length_ok,
        "detail": f"{len(text)} chars (min {min_chars})",
    }

    words = text.lower().split()
    if words:
        counts = Counter(words)
        most_common_word, most_common_count = counts.most_common(1)[0]
        repeat_ratio = most_common_count / len(words)
        repeat_ok = repeat_ratio < 0.30
        results["repetition"] = {
            "pass": repeat_ok,
            "detail": f"most repeated '{most_common_word}' = {repeat_ratio:.0%} of {len(words)} words",
        }
    else:
        results["repetition"] = {"pass": False, "detail": "no words generated"}

    if text:
        freq = Counter(text)
        total = len(text)
        entropy = -sum((c / total) * math.log2(c / total) for c in freq.values())
        entropy_ok = entropy > 2.5
        results["entropy"] = {
            "pass": entropy_ok,
            "detail": f"{entropy:.2f} bits/char (min 2.5)",
        }
    else:
        results["entropy"] = {"pass": False, "detail": "empty text"}

    all_pass = all(r["pass"] for r in results.values())
    return {"all_pass": all_pass, "checks": results}


def print_results(label: str, metrics: dict, validation: dict) -> None:
    """Pretty-print benchmark results."""
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")

    if metrics["error"]:
        print(f"  ERROR: {metrics['error']}")
        return

    print(f"  Prompt:              {metrics['prompt_word_count']} words")
    print(f"  Pipeline:            {metrics['pipeline_mode']} ({metrics['pipeline_hops']} hops)")
    print(f"  TTFT:                {metrics['ttft_s']*1000:.0f} ms")
    print(f"  Wall Time:           {metrics['wall_time_s']:.3f} s")
    print(f"  Tokens:              {metrics['token_count']}")
    print(f"  Words:               {metrics['word_count']}")
    print(f"  TPS:                 {metrics['tps']:.1f} tok/s")

    if metrics["pipeline_traces"]:
        print(f"  Per-stage latency:")
        for trace in metrics["pipeline_traces"]:
            print(f"    Stage {trace.get('stage_index','?')}: "
                  f"peer={trace.get('peer_id','?')}, "
                  f"latency={trace.get('latency_ms',0):.0f}ms")

    print()
    print("  Validation:")
    for name, check in validation["checks"].items():
        icon = "PASS" if check["pass"] else "FAIL"
        print(f"    [{icon}] {name}: {check['detail']}")
    overall = "PASS" if validation["all_pass"] else "FAIL"
    print(f"    Overall: {overall}")

    print(f"\n  Generated text:")
    print(f"    {metrics['full_text'][:300]}")


def save_output(text: str, path: str) -> None:
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
    parser.add_argument("--min-chars", type=int, default=10, help="Min chars for validation")
    parser.add_argument("--prompt", default=None, help="Custom prompt text")
    parser.add_argument("--prompt-mode", default="short",
                        choices=["short", "long", "compare"],
                        help="short (<256 words), long (>256 words), compare (both)")
    parser.add_argument("--endpoint", default="/v1/completions",
                        choices=["/v1/completions", "/v1/chat/completions"])
    args = parser.parse_args()

    prompts_to_run = []
    if args.prompt:
        prompts_to_run.append(("custom", args.prompt))
    elif args.prompt_mode == "compare":
        prompts_to_run.append(("SHORT (<256 words)", SHORT_PROMPT))
        prompts_to_run.append(("LONG (>256 words, chunked prefill)", LONG_PROMPT))
    elif args.prompt_mode == "long":
        prompts_to_run.append(("LONG (>256 words)", LONG_PROMPT))
    else:
        prompts_to_run.append(("SHORT (<256 words)", SHORT_PROMPT))

    all_results = []
    for prompt_label, prompt_text in prompts_to_run:
        full_label = f"{args.label} — {prompt_label}" if len(prompts_to_run) > 1 else args.label
        print(f"\nRunning {full_label}...")
        print(f"  URL: {args.url}")
        print(f"  Model: {args.model}")
        print(f"  Max tokens: {args.max_tokens}")
        print(f"  Prompt: {prompt_text[:80]}...")
        print(f"  Prompt length: {len(prompt_text.split())} words")

        metrics = run_benchmark(
            args.url, prompt_text, args.model, args.max_tokens, args.endpoint,
        )
        validation = validate_output(metrics["full_text"], min_chars=args.min_chars)
        print_results(full_label, metrics, validation)

        if args.output and metrics["full_text"]:
            suffix = f"_{prompt_label.split()[0].lower()}" if len(prompts_to_run) > 1 else ""
            save_output(metrics["full_text"], f"{args.output}{suffix}.txt")

        all_results.append((full_label, metrics, validation))

    # Summary comparison table
    if len(all_results) > 1:
        print(f"\n{'='*65}")
        print(f"  COMPARISON SUMMARY")
        print(f"{'='*65}")
        print(f"  {'Label':<40} {'Words':>6} {'Tokens':>7} {'Wall':>7} {'TPS':>6} {'Valid':>5}")
        print(f"  {'-'*40} {'-'*6} {'-'*7} {'-'*7} {'-'*6} {'-'*5}")
        for label, m, v in all_results:
            valid = "PASS" if v["all_pass"] else "FAIL"
            print(f"  {label:<40} {m['prompt_word_count']:>6} {m['token_count']:>7} "
                  f"{m['wall_time_s']:>6.1f}s {m['tps']:>5.1f} {valid:>5}")

    sys.exit(0 if all(v["all_pass"] for _, _, v in all_results) else 1)


if __name__ == "__main__":
    main()
