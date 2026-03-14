#!/usr/bin/env python3
"""bench_kv_compaction.py — quality benchmark for KV cache compaction.

Measures how closely the output logits from a compacted KV cache match
those of the full (uncompacted) KV cache across three methods:

  Method 0 — Baseline (no compaction):
      Full KV cache used as-is.  Upper bound on quality.

  Method 1 — Phase 1 + Proxy-K:
      HAK / OMP key selection using the last ``n_ref`` KEY vectors as
      stand-in reference queries (wrong W_k subspace).  Phase 1 baseline.

  Method 2 — Phase 1 + Real-Q (Option A):
      HAK / OMP key selection using W_q·hidden as reference queries
      (correct W_q subspace, AttentionQueryCapture).  Expected quality uplift.

  Method 3 — Phase 2 + Real-Q (Option A + β + Cv):
      Adds log-space bias corrections β and refits Cv via least-squares.
      Best quality at highest cost.

Quality metrics (computed at the last-token logit vector):
  logit_cos_sim    cosine similarity to baseline (↑ better, 1.0 = perfect)
  top1_match       argmax matches baseline (fraction, ↑ better)
  top5_overlap     fraction of top-5 tokens that overlap with baseline (↑ better)
  rank_corr        Spearman rank correlation over top-100 logits (↑ better)

Usage::

    # Minimal smoke test (uses gpt2 which is tiny)
    python scripts/bench_kv_compaction.py --model gpt2 --seq-len 64 --n-prompts 5

    # Proper Qwen3 benchmark
    python scripts/bench_kv_compaction.py \\
        --model Qwen/Qwen3.5-0.8B \\
        --seq-len 256 \\
        --n-prompts 50 \\
        --target-ratio 0.10 \\
        --method hak

    # Compare HAK vs OMP at multiple budgets
    python scripts/bench_kv_compaction.py \\
        --model Qwen/Qwen3.5-0.8B \\
        --seq-len 512 \\
        --n-prompts 20 \\
        --target-ratio 0.05 0.10 0.20 0.50 \\
        --method hak omp

Notes
-----
* The benchmark uses the model's OWN KV cache from a prefill pass as the
  "full cache" baseline.  Compaction is applied to this in-memory cache and
  the next-token logits are compared.  No re-generation is required.
* Requires ``transformers``, ``torch``, and optionally ``scipy`` (for Phase 2
  NNLS β fitting).
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from typing import Any

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("bench_kv_compaction")


# ─────────────────────────────────────────────────────────────────────────────
# Prompts used for benchmarking (diverse topics for generalised coverage)
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_PROMPTS = [
    "The history of the Roman Empire spans over a thousand years and encompasses",
    "Quantum mechanics describes the behaviour of matter and energy at atomic scales.",
    "In the field of machine learning, attention mechanisms allow models to",
    "The Amazon rainforest produces approximately 20 percent of the world's oxygen",
    "Shakespeare's play Hamlet explores themes of revenge, mortality, and betrayal",
    "The development of modern cryptography began in earnest during World War II",
    "Climate change refers to long-term shifts in global temperatures and weather",
    "The human brain contains approximately 86 billion neurons interconnected by",
    "Einstein's theory of general relativity describes gravity as a curvature of",
    "The first programmable electronic computer, ENIAC, was completed in 1945 and",
    "Ocean currents play a crucial role in regulating the Earth's climate by",
    "The discovery of DNA's double-helix structure by Watson and Crick in 1953",
    "Blockchain technology enables decentralised record-keeping without the need",
    "The Silk Road was an ancient network of trade routes connecting East Asia to",
    "Photosynthesis is the process by which plants convert sunlight, water, and",
    "The International Space Station orbits Earth at an altitude of approximately",
    "The French Revolution of 1789 fundamentally transformed European politics",
    "Neural networks are computational models loosely inspired by the structure of",
    "The periodic table organises chemical elements by their atomic number and",
    "Free trade agreements between countries reduce tariffs and barriers to allow",
]


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_similarity(a: "Tensor", b: "Tensor") -> float:
    import torch
    a = a.float().flatten()
    b = b.float().flatten()
    dot = float((a * b).sum())
    norm = float(a.norm() * b.norm())
    return dot / (norm + 1e-12)


def _top1_match(a: "Tensor", b: "Tensor") -> bool:
    return int(a.argmax()) == int(b.argmax())


def _topk_overlap(a: "Tensor", b: "Tensor", k: int = 5) -> float:
    import torch
    ta = set(a.topk(k).indices.tolist())
    tb = set(b.topk(k).indices.tolist())
    return len(ta & tb) / k


def _rank_corr(a: "Tensor", b: "Tensor", k: int = 100) -> float:
    """Spearman rank correlation over top-k tokens from baseline."""
    import torch
    k = min(k, int(a.shape[-1]))
    _, top_idx = a.topk(k)
    a_vals = a[top_idx].float().tolist()
    b_vals = b[top_idx].float().tolist()
    # Simple rank-based Spearman
    def _rank(lst):
        sorted_pairs = sorted(enumerate(lst), key=lambda x: x[1], reverse=True)
        ranks = [0.0] * len(lst)
        for rank, (orig_idx, _) in enumerate(sorted_pairs, 1):
            ranks[orig_idx] = float(rank)
        return ranks
    ra, rb = _rank(a_vals), _rank(b_vals)
    n = len(ra)
    d2 = sum((ri - rj) ** 2 for ri, rj in zip(ra, rb))
    rho = 1.0 - (6 * d2) / (n * (n * n - 1) + 1e-12)
    return float(rho)


# ─────────────────────────────────────────────────────────────────────────────
# Core benchmark function
# ─────────────────────────────────────────────────────────────────────────────

def _get_full_cache_and_logits(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_len: int,
    device: Any,
    decoder_family: str,
) -> "tuple[Any, Tensor, Any]":
    """Prefill *prompt* with use_cache=True.

    Returns
    -------
    (past_key_values, last_token_logits, input_ids)
        The full KV cache, the logit vector at the last prompt token, and
        the tokenised prompt ids (for the continuation forward pass later).
    """
    import torch
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"].to(device)
    model_kwargs: dict[str, Any] = {"input_ids": input_ids, "use_cache": True}
    total_len = int(input_ids.shape[1])
    if decoder_family in {"qwen_llama", "llama"}:
        model_kwargs["attention_mask"] = torch.ones(
            (1, total_len), dtype=torch.bool, device=device
        )
    with torch.no_grad():
        out = model(**model_kwargs)
    logits = getattr(out, "logits", None)
    if logits is None and isinstance(out, tuple):
        logits = out[0]
    past = getattr(out, "past_key_values", None)
    if past is None and isinstance(out, tuple) and len(out) > 1:
        past = out[1]
    # Logits: (1, seq, vocab) → take the last token
    last_logits = logits[0, -1, :].detach()
    return past, last_logits, input_ids


def _continuation_logits(
    model: Any,
    past_key_values: Any,
    input_ids: Any,
    device: Any,
    decoder_family: str,
    q_ref_per_layer: "list | None" = None,
) -> "Tensor":
    """Run a one-token decode step from a (possibly compacted) KV cache.

    Returns the logit vector at the continuation token.
    """
    import torch
    # Use the last token of the prompt as the "new" input_ids
    new_ids = input_ids[:, -1:].to(device)
    past_len: int
    get_seq = getattr(past_key_values, "get_seq_length", None)
    if callable(get_seq):
        past_len = int(get_seq())
    elif isinstance(past_key_values, (tuple, list)) and past_key_values:
        try:
            first = past_key_values[0]
            if isinstance(first, (tuple, list)):
                past_len = int(first[0].shape[-2])
            else:
                past_len = int(past_key_values.key_cache[0].shape[-2])
        except Exception:
            past_len = 0
    else:
        past_len = 0

    model_kwargs: dict[str, Any] = {
        "input_ids": new_ids,
        "past_key_values": past_key_values,
        "use_cache": False,
    }
    total_len = past_len + 1
    if decoder_family in {"qwen_llama", "llama"}:
        model_kwargs["attention_mask"] = torch.ones(
            (1, total_len), dtype=torch.bool, device=device
        )

    with torch.no_grad():
        out = model(**model_kwargs)
    logits = getattr(out, "logits", None)
    if logits is None and isinstance(out, tuple):
        logits = out[0]
    return logits[0, -1, :].detach()


def _compact_with_proxy_k(past_key_values, config, budgets_data):
    """Phase 1 compaction using proxy-K heuristic (no Option A)."""
    from peer.kv_compaction import compact_past_key_values
    return compact_past_key_values(
        past_key_values,
        config,
        budgets_data=budgets_data,
        q_ref_per_layer=None,   # explicitly no real Q
    )


def _compact_with_real_q(past_key_values, model, config, budgets_data, n_ref,
                         input_ids, device, decoder_family):
    """Phase 1 compaction using real Q tensors from AttentionQueryCapture (Option A)."""
    import torch
    from peer.kv_compaction import compact_past_key_values, AttentionQueryCapture

    # We need to re-run a lightweight forward pass to capture Q.
    # Use input_ids for the capture forward (same tokens as original prefill).
    model_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "use_cache": False,   # we already have past_key_values; just capture Q
    }
    total_len = int(input_ids.shape[1])
    if decoder_family in {"qwen_llama", "llama"}:
        model_kwargs["attention_mask"] = torch.ones(
            (1, total_len), dtype=torch.bool, device=device
        )

    q_per_layer = None
    try:
        qc = AttentionQueryCapture(model, n_ref=n_ref)
        with qc:
            with torch.no_grad():
                model(**model_kwargs)
        q_per_layer = qc.compute_q_ref()
    except Exception as exc:
        logger.warning("real_q_capture_failed: %s", exc)

    return compact_past_key_values(
        past_key_values,
        config,
        budgets_data=budgets_data,
        q_ref_per_layer=q_per_layer,
    )


def _detect_decoder_family(model: Any) -> str:
    arch = type(model).__name__.lower()
    model_type = str(getattr(getattr(model, "config", None), "model_type", "")).lower()
    if "qwen" in arch or "qwen" in model_type:
        return "qwen_llama"
    if "llama" in arch or "llama" in model_type:
        return "llama"
    return "gpt"


# ─────────────────────────────────────────────────────────────────────────────
# Per-ratio benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_ratio(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    target_ratio: float,
    method: str,
    run_beta: bool,
    seq_len: int,
    n_ref: int,
    device: Any,
    decoder_family: str,
) -> dict[str, Any]:
    """Benchmark a single target_ratio value.  Returns aggregated metrics."""
    from peer.kv_compaction import CompactionConfig

    cfg_proxy = CompactionConfig(
        enabled=True,
        method=method,
        target_ratio=target_ratio,
        min_source_tokens=16,
        n_ref_queries=n_ref,
        beta_enabled=False,
    )
    cfg_real_q = CompactionConfig(
        enabled=True,
        method=method,
        target_ratio=target_ratio,
        min_source_tokens=16,
        n_ref_queries=n_ref,
        beta_enabled=False,
    )
    cfg_beta = CompactionConfig(
        enabled=True,
        method=method,
        target_ratio=target_ratio,
        min_source_tokens=16,
        n_ref_queries=n_ref,
        beta_enabled=True,
    )

    results: dict[str, list[float]] = {
        "proxy_cos": [], "proxy_top1": [], "proxy_top5": [], "proxy_rk": [],
        "realq_cos": [], "realq_top1": [], "realq_top5": [], "realq_rk": [],
        "beta_cos":  [], "beta_top1":  [], "beta_top5":  [], "beta_rk":  [],
        "tokens_kept_proxy": [], "tokens_kept_realq": [],
    }

    for i, prompt in enumerate(prompts):
        try:
            # ── Prefill (full cache, baseline logits) ─────────────────────────
            full_pkv, baseline_logits, input_ids = _get_full_cache_and_logits(
                model, tokenizer, prompt, seq_len, device, decoder_family,
            )
            T = None
            # Determine actual seq_len of the full cache for reporting
            try:
                if hasattr(full_pkv, "key_cache"):
                    T = int(full_pkv.key_cache[0].shape[-2])
                elif isinstance(full_pkv, (tuple, list)):
                    T = int(full_pkv[0][0].shape[-2])
            except Exception:
                T = 0
            if T is None or T < 16:
                logger.debug("prompt %d: seq_len=%s too short, skipping", i, T)
                continue

            # ── Method 1: Proxy-K ─────────────────────────────────────────────
            pkv_proxy = _compact_with_proxy_k(full_pkv, cfg_proxy, None)
            t_proxy = 0
            try:
                if hasattr(pkv_proxy, "key_cache"):
                    t_proxy = int(pkv_proxy.key_cache[0].shape[-2])
                elif isinstance(pkv_proxy, (tuple, list)):
                    t_proxy = int(pkv_proxy[0][0].shape[-2])
            except Exception:
                pass
            logits_proxy = _continuation_logits(
                model, pkv_proxy, input_ids, device, decoder_family,
            )
            results["proxy_cos"].append(_cosine_similarity(baseline_logits, logits_proxy))
            results["proxy_top1"].append(float(_top1_match(baseline_logits, logits_proxy)))
            results["proxy_top5"].append(_topk_overlap(baseline_logits, logits_proxy, k=5))
            results["proxy_rk"].append(_rank_corr(baseline_logits, logits_proxy))
            results["tokens_kept_proxy"].append(float(t_proxy))

            # ── Method 2: Real Q (Option A) ───────────────────────────────────
            pkv_realq = _compact_with_real_q(
                full_pkv, model, cfg_real_q, None,
                n_ref, input_ids, device, decoder_family,
            )
            t_realq = 0
            try:
                if hasattr(pkv_realq, "key_cache"):
                    t_realq = int(pkv_realq.key_cache[0].shape[-2])
                elif isinstance(pkv_realq, (tuple, list)):
                    t_realq = int(pkv_realq[0][0].shape[-2])
            except Exception:
                pass
            logits_realq = _continuation_logits(
                model, pkv_realq, input_ids, device, decoder_family,
            )
            results["realq_cos"].append(_cosine_similarity(baseline_logits, logits_realq))
            results["realq_top1"].append(float(_top1_match(baseline_logits, logits_realq)))
            results["realq_top5"].append(_topk_overlap(baseline_logits, logits_realq, k=5))
            results["realq_rk"].append(_rank_corr(baseline_logits, logits_realq))
            results["tokens_kept_realq"].append(float(t_realq))

            # ── Method 3: Phase 2 + β + Cv (Option A + beta) ─────────────────
            if run_beta:
                try:
                    pkv_beta = _compact_with_real_q(
                        full_pkv, model, cfg_beta, None,
                        n_ref, input_ids, device, decoder_family,
                    )
                    # Phase 2 returns CompactedKVCache — get the standard cache for inference
                    _cache_for_infer = pkv_beta
                    logits_beta = _continuation_logits(
                        model, _cache_for_infer, input_ids, device, decoder_family,
                    )
                    results["beta_cos"].append(_cosine_similarity(baseline_logits, logits_beta))
                    results["beta_top1"].append(float(_top1_match(baseline_logits, logits_beta)))
                    results["beta_top5"].append(_topk_overlap(baseline_logits, logits_beta, k=5))
                    results["beta_rk"].append(_rank_corr(baseline_logits, logits_beta))
                except Exception as exc:
                    logger.debug("beta_failed prompt %d: %s", i, exc)

        except Exception as exc:
            logger.warning("prompt %d failed: %s", i, exc)
            continue

        if (i + 1) % 5 == 0:
            logger.info("  … %d/%d prompts done", i + 1, len(prompts))

    def _mean(lst):
        return statistics.mean(lst) if lst else float("nan")

    return {
        "target_ratio": target_ratio,
        "n_prompts": len(results["proxy_cos"]),
        "proxy_k": {
            "logit_cos_sim": _mean(results["proxy_cos"]),
            "top1_match":    _mean(results["proxy_top1"]),
            "top5_overlap":  _mean(results["proxy_top5"]),
            "rank_corr":     _mean(results["proxy_rk"]),
            "tokens_kept":   _mean(results["tokens_kept_proxy"]),
        },
        "real_q": {
            "logit_cos_sim": _mean(results["realq_cos"]),
            "top1_match":    _mean(results["realq_top1"]),
            "top5_overlap":  _mean(results["realq_top5"]),
            "rank_corr":     _mean(results["realq_rk"]),
            "tokens_kept":   _mean(results["tokens_kept_realq"]),
        },
        "real_q_plus_beta": {
            "logit_cos_sim": _mean(results["beta_cos"]),
            "top1_match":    _mean(results["beta_top1"]),
            "top5_overlap":  _mean(results["beta_top5"]),
            "rank_corr":     _mean(results["beta_rk"]),
        } if run_beta and results["beta_cos"] else None,
        "uplift_cos": (
            _mean(results["realq_cos"]) - _mean(results["proxy_cos"])
        ),
        "uplift_top1": (
            _mean(results["realq_top1"]) - _mean(results["proxy_top1"])
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _print_table(all_results: list[dict[str, Any]]) -> None:
    header = (
        f"{'ratio':>6}  {'n':>4}  "
        f"{'proxy_cos':>10}  {'realq_cos':>10}  {'Δcos':>8}  "
        f"{'proxy_t1':>9}  {'realq_t1':>9}  {'Δtop1':>8}  "
        f"{'proxy_rk':>9}  {'realq_rk':>9}"
    )
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))
    for r in all_results:
        ratio = r["target_ratio"]
        n = r["n_prompts"]
        pc = r["proxy_k"]["logit_cos_sim"]
        rc = r["real_q"]["logit_cos_sim"]
        pt = r["proxy_k"]["top1_match"]
        rt = r["real_q"]["top1_match"]
        pr = r["proxy_k"]["rank_corr"]
        rr = r["real_q"]["rank_corr"]
        dc = r["uplift_cos"]
        dt = r["uplift_top1"]
        print(
            f"{ratio:>6.2f}  {n:>4d}  "
            f"{pc:>10.4f}  {rc:>10.4f}  {dc:>+8.4f}  "
            f"{pt:>9.4f}  {rt:>9.4f}  {dt:>+8.4f}  "
            f"{pr:>9.4f}  {rr:>9.4f}"
        )
    print("─" * len(header))
    print()

    # Headline: overall mean uplift
    mean_uplift_cos = statistics.mean(r["uplift_cos"] for r in all_results)
    mean_uplift_top1 = statistics.mean(r["uplift_top1"] for r in all_results)
    print(
        f"  ▶  Mean Quality Uplift (Real-Q vs Proxy-K)\n"
        f"     Δlogit_cos_sim = {mean_uplift_cos:+.4f}\n"
        f"     Δtop1_match    = {mean_uplift_top1:+.4f}\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark KV cache compaction quality: proxy-K vs real-Q.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="gpt2",
                        help="HuggingFace model id or local path.")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Max token length for each prompt (prefill length).")
    parser.add_argument("--n-prompts", type=int, default=20,
                        help="Number of prompts to evaluate.")
    parser.add_argument("--target-ratio", type=float, nargs="+", default=[0.10],
                        metavar="RATIO",
                        help="Compaction target ratio(s), e.g. 0.05 0.10 0.20.")
    parser.add_argument("--method", choices=["hak", "omp"], nargs="+", default=["hak"],
                        help="Key-selection algorithm(s) to benchmark.")
    parser.add_argument("--n-ref", type=int, default=8,
                        help="Number of reference queries for HAK/OMP scoring.")
    parser.add_argument("--beta", action="store_true",
                        help="Also benchmark Phase 2 (β + Cv fitting).")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device to run on.")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Pass trust_remote_code=True to AutoModel (required for Qwen).")
    parser.add_argument("--output-json", default=None,
                        help="Write full results to a JSON file.")
    parser.add_argument("--custom-prompts", default=None,
                        help="Path to a .txt file with one prompt per line.")
    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────────
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        logger.error("Required: torch + transformers.  Install with: pip install torch transformers")
        raise SystemExit(1) from exc

    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    logger.info("device=%s  model=%s", device_str, args.model)

    trust = args.trust_remote_code or bool("qwen" in args.model.lower())
    logger.info("loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=trust)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("loading model …")
    dtype = torch.float16 if device_str == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=trust,
    ).to(device).eval()

    decoder_family = _detect_decoder_family(model)
    logger.info("decoder_family=%s", decoder_family)

    # ── Prompts ───────────────────────────────────────────────────────────────
    if args.custom_prompts:
        with open(args.custom_prompts, "r", encoding="utf-8") as fh:
            prompts = [ln.strip() for ln in fh if ln.strip()]
    else:
        prompts = _DEFAULT_PROMPTS

    # Repeat / truncate to n_prompts
    while len(prompts) < args.n_prompts:
        prompts = prompts * 2
    prompts = prompts[: args.n_prompts]
    logger.info("running %d prompts, seq_len=%d", len(prompts), args.seq_len)

    # ── Benchmark loop ────────────────────────────────────────────────────────
    all_results = []
    t0 = time.perf_counter()

    for method in args.method:
        logger.info("=== method=%s ===", method)
        for ratio in sorted(args.target_ratio):
            logger.info("--- target_ratio=%.2f ---", ratio)
            row = _run_ratio(
                model, tokenizer, prompts,
                target_ratio=ratio,
                method=method,
                run_beta=args.beta,
                seq_len=args.seq_len,
                n_ref=args.n_ref,
                device=device,
                decoder_family=decoder_family,
            )
            row["method"] = method
            all_results.append(row)

    elapsed = time.perf_counter() - t0
    logger.info("benchmark completed in %.1fs", elapsed)

    # ── Print ─────────────────────────────────────────────────────────────────
    print(f"\nBenchmark: {args.model}  |  seq_len={args.seq_len}  "
          f"|  n_prompts={len(prompts)}  |  n_ref={args.n_ref}")
    _print_table(all_results)

    # ── JSON export ───────────────────────────────────────────────────────────
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(all_results, fh, indent=2)
        logger.info("results written to %s", args.output_json)


if __name__ == "__main__":
    main()
