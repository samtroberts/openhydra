# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Throughput self-benchmarking — Petals parity Phase D.

Measures actual compute throughput at startup instead of using static
estimates.  Results are cached to ``~/.openhydra/throughput_cache.json``
so restarts don't re-benchmark if the config hasn't changed.

Petals equivalent: ``measure_compute_rps()`` + ``measure_network_rps()``
in ``petals/server/throughput.py``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CACHE_DIR = os.path.expanduser("~/.openhydra")
_CACHE_FILE = os.path.join(_CACHE_DIR, "throughput_cache.json")
_CACHE_TTL_HOURS = 24


@dataclass
class ThroughputResult:
    """Result of a throughput benchmark.

    Attributes:
        compute_tps: Measured tokens per second (forward pass).
        network_mbps: Measured upload bandwidth in Mbps (0 = not measured).
        model_id: Model used for the benchmark.
        device: "cpu" or "cuda".
        layer_count: Number of layers benchmarked.
        quantization: Quantization mode used.
        measured_at: Unix timestamp of the measurement.
    """
    compute_tps: float = 0.0
    network_mbps: float = 0.0
    model_id: str = ""
    device: str = "cpu"
    layer_count: int = 0
    quantization: str = "fp32"
    measured_at: float = 0.0


def _cache_key(model_id: str, device: str, layer_count: int, quantization: str) -> str:
    """Generate a cache key for this configuration."""
    return f"{model_id}|{device}|{layer_count}|{quantization}"


def load_cached_throughput(
    model_id: str,
    device: str,
    layer_count: int,
    quantization: str,
) -> ThroughputResult | None:
    """Load a cached benchmark result if fresh enough.

    Returns None if no cache exists or it's older than 24 hours.
    """
    try:
        if not os.path.exists(_CACHE_FILE):
            return None
        with open(_CACHE_FILE, "r") as f:
            cache = json.load(f)
        key = _cache_key(model_id, device, layer_count, quantization)
        entry = cache.get(key)
        if entry is None:
            return None
        measured_at = float(entry.get("measured_at", 0))
        if time.time() - measured_at > _CACHE_TTL_HOURS * 3600:
            return None  # Expired
        return ThroughputResult(**entry)
    except Exception:
        return None


def save_cached_throughput(result: ThroughputResult) -> None:
    """Save a benchmark result to the cache."""
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        cache: dict[str, Any] = {}
        if os.path.exists(_CACHE_FILE):
            with open(_CACHE_FILE, "r") as f:
                cache = json.load(f)
        key = _cache_key(result.model_id, result.device, result.layer_count, result.quantization)
        cache[key] = asdict(result)
        with open(_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as exc:
        logger.debug("throughput_cache_save_failed: %s", exc)


def measure_compute_tps(
    model: Any,
    tokenizer: Any,
    device: str = "cpu",
    n_warmup: int = 2,
    n_measure: int = 5,
) -> float:
    """Measure actual forward-pass throughput by running real inference.

    Generates a short sequence and measures wall-clock time per token.
    Uses greedy decoding for deterministic results.

    Args:
        model: A loaded HuggingFace model (AutoModelForCausalLM).
        tokenizer: The model's tokenizer.
        device: "cpu" or "cuda".
        n_warmup: Number of warmup runs to discard.
        n_measure: Number of measurement runs to average.

    Returns:
        Tokens per second (float). Returns 0.0 on failure.
    """
    try:
        import torch

        prompt = "The quick brown fox"
        max_new = 8
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        if device != "cpu" and torch.cuda.is_available():
            input_ids = input_ids.to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model.generate(input_ids, max_new_tokens=max_new, do_sample=False)
                if device != "cpu" and torch.cuda.is_available():
                    torch.cuda.synchronize()

        # Measure
        times: list[float] = []
        with torch.no_grad():
            for _ in range(n_measure):
                t0 = time.perf_counter()
                out = model.generate(input_ids, max_new_tokens=max_new, do_sample=False)
                if device != "cpu" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                n_tokens = out.shape[1] - input_ids.shape[1]
                if n_tokens > 0 and elapsed > 0:
                    times.append(n_tokens / elapsed)

        if not times:
            return 0.0

        # Drop the slowest run (may include JIT/compilation)
        if len(times) > 2:
            times.sort()
            times = times[:-1]

        avg_tps = sum(times) / len(times)
        logger.info("throughput_bench: compute_tps=%.1f (n=%d runs, device=%s)", avg_tps, len(times), device)
        return round(avg_tps, 3)
    except Exception as exc:
        logger.warning("throughput_bench_failed: %s", exc)
        return 0.0


def measure_network_mbps(
    target_url: str = "https://speed.cloudflare.com/__down?bytes=1000000",
    timeout_s: float = 10.0,
) -> float:
    """Measure download bandwidth by fetching a test payload.

    Returns Mbps (megabits per second). Returns 0.0 on failure.
    """
    try:
        import urllib.request
        t0 = time.perf_counter()
        req = urllib.request.Request(target_url)
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = resp.read()
        elapsed = time.perf_counter() - t0
        if elapsed <= 0:
            return 0.0
        mbps = (len(data) * 8) / (elapsed * 1_000_000)
        logger.info("throughput_bench: network_mbps=%.1f (%d bytes in %.1fs)", mbps, len(data), elapsed)
        return round(mbps, 1)
    except Exception as exc:
        logger.debug("network_bench_failed: %s", exc)
        return 0.0


def benchmark_and_cache(
    model: Any,
    tokenizer: Any,
    model_id: str,
    device: str,
    layer_count: int,
    quantization: str,
    skip_network: bool = True,
) -> ThroughputResult:
    """Run benchmark (or load from cache) and return the result.

    This is the main entry point. Call it during PyTorchRuntime.__init__()
    to get an accurate TPS estimate.

    Args:
        model: Loaded model.
        tokenizer: Model tokenizer.
        model_id: HuggingFace model ID.
        device: "cpu" or "cuda".
        layer_count: Number of layers this shard runs.
        quantization: "fp32", "int8", "int4".
        skip_network: If True, skip the network bandwidth test.

    Returns:
        ThroughputResult with measured (or cached) metrics.
    """
    # Check cache first
    cached = load_cached_throughput(model_id, device, layer_count, quantization)
    if cached is not None:
        logger.info(
            "throughput_bench: using cached result: %.1f TPS (measured %.0fs ago)",
            cached.compute_tps, time.time() - cached.measured_at,
        )
        return cached

    # Run benchmark
    compute_tps = measure_compute_tps(model, tokenizer, device=device)
    network_mbps = 0.0 if skip_network else measure_network_mbps()

    result = ThroughputResult(
        compute_tps=compute_tps,
        network_mbps=network_mbps,
        model_id=model_id,
        device=device,
        layer_count=layer_count,
        quantization=quantization,
        measured_at=time.time(),
    )

    # Cache it
    save_cached_throughput(result)
    return result
