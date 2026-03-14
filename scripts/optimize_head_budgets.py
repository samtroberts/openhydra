"""
scripts/optimize_head_budgets.py — calibrate per-head KV budget ratios from attention entropy.

Usage
-----
    python scripts/optimize_head_budgets.py \
        --model Qwen/Qwen3-4B \
        --output peer/kv_compaction/head_budgets/qwen3_4b_calibrated.json \
        --target-ratio 0.10

The script:
  1. Loads the model with ``output_attentions=True``.
  2. Runs a set of calibration prompts through the model.
  3. For each layer, computes per-head attention entropy on the *last* token's
     attention row (i.e. how focused vs. diffuse is each head's attention pattern).
  4. Groups query-head entropies into kv-head groups (GQA support) by averaging.
  5. Allocates token-budget ratios proportional to entropy — high-entropy heads get
     larger budgets because they attend broadly and need more context.
  6. Normalises so the mean ratio across all heads equals ``--target-ratio``.
  7. Writes a JSON file in the exact schema consumed by Phase 3 of KV compaction
     (``_load_head_budgets()`` / ``_get_budget_for_layer_head()``).

Output JSON schema
------------------
    {
      "model":         "<model-id>",
      "n_layers":      <int>,
      "n_kv_heads":    <int>,
      "source":        "calibrated_entropy_v1",
      "layer_budgets": [
        [ratio_kv_head_0, ratio_kv_head_1, ...],   // layer 0
        ...
      ]
    }
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import pathlib
import sys
from typing import Any

logger = logging.getLogger(__name__)

# ─── Built-in calibration prompts ────────────────────────────────────────────
_DEFAULT_PROMPTS: list[str] = [
    "The distributed inference system routes each token through multiple peer nodes.",
    "Attention mechanisms in large language models weigh every prior token.",
    "Scientists discovered a new protein structure using cryo-electron microscopy.",
    "The recipe calls for two cups of flour, one cup of sugar, and a pinch of salt.",
    "In the year 2045, autonomous vehicles outnumbered human-driven cars globally.",
    "She carefully examined the ancient manuscript under a bright magnifying lens.",
    "The central bank raised interest rates to combat persistent inflationary pressure.",
    "A peer-to-peer network distributes computation across thousands of volunteer nodes.",
    "Mount Everest stands at 8,848 metres above sea level, the highest on Earth.",
    "The compiler optimises nested loops by hoisting invariant expressions outward.",
    "Quantum entanglement allows correlated measurements across arbitrarily large distances.",
    "He solved the differential equation by applying Laplace transforms to both sides.",
    "The surgeon performed a minimally invasive procedure using robotic-assisted tools.",
    "Language models predict the next token given all previously observed context tokens.",
    "The cargo ship navigated through heavy fog using GPS and radar-assisted guidance.",
    "Gradient descent iteratively adjusts model weights to minimise the training loss.",
    "The parliament voted to extend the emergency energy subsidies for another quarter.",
    "Her watercolour painting captured the soft light of dusk over the Italian hills.",
    "Database transactions must satisfy atomicity, consistency, isolation, and durability.",
    "The team celebrated after their pull request finally passed the continuous integration checks.",
]


# ─── Core maths ──────────────────────────────────────────────────────────────

def _compute_head_entropy(attn_row: "Any", n_kv_heads: int) -> "Any":
    """Compute per-KV-head entropy from a last-token attention row.

    Parameters
    ----------
    attn_row:
        Tensor of shape ``(n_heads, seq_len)`` — the attention weights for the
        last token, already softmaxed (values sum to 1 over seq_len per head).
    n_kv_heads:
        Number of key-value heads (for GQA, ``n_kv_heads ≤ n_heads``).

    Returns
    -------
    Tensor of shape ``(n_kv_heads,)`` — mean entropy per kv-head group.
    """
    import torch

    n_heads, seq_len = attn_row.shape
    # Shannon entropy per head: H = -sum(p * log(p + eps))
    eps = 1e-9
    log_p = torch.log(attn_row + eps)
    entropy_per_head = -(attn_row * log_p).sum(dim=-1)   # (n_heads,)

    if n_kv_heads == n_heads:
        return entropy_per_head

    # GQA: average within each kv-head group
    n_groups = n_heads // n_kv_heads
    # reshape to (n_kv_heads, n_groups) then mean over groups
    grouped = entropy_per_head.view(n_kv_heads, n_groups)
    return grouped.mean(dim=1)    # (n_kv_heads,)


def _allocate_budgets(
    mean_entropy: "Any",
    target_ratio: float,
    min_ratio: float,
    max_ratio: float,
) -> list[float]:
    """Convert per-KV-head mean entropy into budget ratios.

    Ratios are proportional to entropy, clipped to [min_ratio, max_ratio],
    then renormalised so their mean equals ``target_ratio``.

    Parameters
    ----------
    mean_entropy:
        1-D tensor of shape ``(n_kv_heads,)``.
    target_ratio, min_ratio, max_ratio:
        Floats in (0, 1).

    Returns
    -------
    list[float] of length ``n_kv_heads``.
    """
    import torch

    n_kv = len(mean_entropy)
    total = float(mean_entropy.sum())
    if total < 1e-12:
        # All heads have zero entropy — fall back to uniform
        raw = [target_ratio] * n_kv
    else:
        raw = [float(e / total) * target_ratio * n_kv for e in mean_entropy.tolist()]

    # Clip
    clipped = [max(min_ratio, min(max_ratio, r)) for r in raw]

    # Renormalise so mean == target_ratio
    mean_clipped = sum(clipped) / n_kv
    if mean_clipped > 1e-12:
        scale = target_ratio / mean_clipped
        normalised = [max(min_ratio, min(max_ratio, r * scale)) for r in clipped]
    else:
        normalised = clipped

    return [round(r, 4) for r in normalised]


# ─── Model utilities ─────────────────────────────────────────────────────────

def _load_model(model_id: str, device: str, trust_remote_code: bool) -> "tuple[Any, Any]":
    """Return (model, tokenizer) loaded from ``model_id``."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        sys.exit(f"ERROR: torch and transformers are required. {exc}")

    logger.info("Loading tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model: %s → device=%s", model_id, device)
    dtype = torch.float16 if ("cuda" in device) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )
    model.to(torch.device(device))
    model.eval()
    return model, tokenizer


def _run_calibration(
    model: "Any",
    tokenizer: "Any",
    prompts: list[str],
    device: str,
    max_length: int = 512,
) -> "list[list[Any]]":
    """Run each prompt and collect per-layer last-token attention rows.

    Returns
    -------
    per_layer_rows:
        List of length ``n_layers``.  Each element is a list of tensors, one
        per prompt, each of shape ``(n_heads, seq_len)``.
    """
    import torch

    all_rows: list[list[Any]] = []   # [layer][prompt] → (n_heads, seq_len)
    n_layers_seen = 0

    for i, prompt in enumerate(prompts, 1):
        logger.info("  Calibrating prompt %d/%d …", i, len(prompts))
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(torch.device(device)) for k, v in inputs.items()}

        with torch.no_grad():
            out = model(**inputs, output_attentions=True)

        attentions = out.attentions   # tuple of (batch, n_heads, seq, seq) per layer
        if attentions is None:
            sys.exit("ERROR: model returned attentions=None. Ensure output_attentions is supported.")

        n_layers_seen = len(attentions)
        if not all_rows:
            all_rows = [[] for _ in range(n_layers_seen)]

        for layer_idx, attn in enumerate(attentions):
            # attn: (1, n_heads, seq, seq) — take last token's attention row
            last_row = attn[0, :, -1, :].float().cpu()   # (n_heads, seq_len)
            all_rows[layer_idx].append(last_row)

    return all_rows


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="optimize_head_budgets",
        description=(
            "Calibrate per-head KV budget ratios from attention entropy. "
            "Outputs a JSON file for Phase 3 of OpenHydra KV compaction."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        required=True,
        help="HuggingFace model id (e.g. Qwen/Qwen3-4B).",
    )
    p.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help=(
            "Output JSON path.  Defaults to "
            "peer/kv_compaction/head_budgets/<model_slug>.json "
            "where <model_slug> is the model id with / replaced by _."
        ),
    )
    p.add_argument(
        "--target-ratio",
        type=float,
        default=0.10,
        help="Mean budget ratio across all heads (0–1).  E.g. 0.10 = keep 10%% of tokens.",
    )
    p.add_argument(
        "--min-ratio",
        type=float,
        default=0.02,
        help="Minimum per-head budget ratio.",
    )
    p.add_argument(
        "--max-ratio",
        type=float,
        default=0.40,
        help="Maximum per-head budget ratio.",
    )
    p.add_argument(
        "--n-prompts",
        type=int,
        default=20,
        help="Number of built-in calibration prompts to use (1–20).",
    )
    p.add_argument(
        "--prompts-file",
        default=None,
        metavar="PATH",
        help="Path to a text file with one calibration prompt per line.  "
             "Overrides --n-prompts.",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="Device: 'auto', 'cpu', or 'cuda'.  'auto' selects cuda if available.",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Pass trust_remote_code=True to from_pretrained (required for some models).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Resolve device
    device = args.device.strip().lower()
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    logger.info("Device: %s", device)

    # Resolve output path
    if args.output:
        output_path = pathlib.Path(args.output)
    else:
        slug = args.model.replace("/", "_").replace("\\", "_")
        output_path = pathlib.Path("peer/kv_compaction/head_budgets") / f"{slug}_calibrated.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve prompts
    if args.prompts_file:
        prompts = [
            line.strip()
            for line in pathlib.Path(args.prompts_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not prompts:
            sys.exit("ERROR: --prompts-file is empty.")
    else:
        n = max(1, min(args.n_prompts, len(_DEFAULT_PROMPTS)))
        prompts = _DEFAULT_PROMPTS[:n]
    logger.info("Using %d calibration prompts", len(prompts))

    # Load model
    trust = args.trust_remote_code or ("qwen" in args.model.lower())
    model, tokenizer = _load_model(args.model, device, trust)

    # Read model config
    cfg = model.config
    n_layers: int = int(
        getattr(cfg, "num_hidden_layers", None)
        or getattr(cfg, "n_layer", None)
        or getattr(cfg, "num_layers", 0)
    )
    n_heads: int = int(
        getattr(cfg, "num_attention_heads", None)
        or getattr(cfg, "n_head", 0)
    )
    n_kv_heads: int = int(
        getattr(cfg, "num_key_value_heads", None)
        or n_heads
    )
    if n_layers <= 0 or n_heads <= 0:
        sys.exit("ERROR: Could not read num_hidden_layers / num_attention_heads from model config.")
    logger.info("Model: %s  layers=%d  n_heads=%d  n_kv_heads=%d", args.model, n_layers, n_heads, n_kv_heads)

    # Calibrate
    logger.info("Running calibration …")
    per_layer_rows = _run_calibration(model, tokenizer, prompts, device)

    # Sanity
    if len(per_layer_rows) != n_layers:
        logger.warning(
            "Model returned %d attention layers but config says %d; "
            "using actual count from attentions.", len(per_layer_rows), n_layers,
        )
        n_layers = len(per_layer_rows)

    import torch

    # Compute mean entropy per layer per kv-head
    layer_budgets: list[list[float]] = []
    for layer_idx, rows in enumerate(per_layer_rows):
        # Stack rows from all prompts: each row is (n_heads, seq_len)
        # Different prompts have different seq_len — process separately
        entropy_sum = torch.zeros(n_kv_heads)
        count = 0
        for row in rows:
            if row.shape[0] != n_heads:
                logger.warning(
                    "Layer %d: expected %d heads but got %d — skipping this prompt.",
                    layer_idx, n_heads, row.shape[0],
                )
                continue
            h = _compute_head_entropy(row, n_kv_heads)
            entropy_sum += h
            count += 1

        if count == 0:
            # No valid prompts for this layer — uniform
            layer_budgets.append([round(args.target_ratio, 4)] * n_kv_heads)
            continue

        mean_entropy = entropy_sum / count
        budgets = _allocate_budgets(
            mean_entropy,
            target_ratio=args.target_ratio,
            min_ratio=args.min_ratio,
            max_ratio=args.max_ratio,
        )
        layer_budgets.append(budgets)

    # Build output
    result = {
        "model": args.model,
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "source": "calibrated_entropy_v1",
        "layer_budgets": layer_budgets,
    }

    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote budget file: %s", output_path)

    # Summary stats
    all_ratios = [r for layer in layer_budgets for r in layer]
    mean_r = sum(all_ratios) / len(all_ratios) if all_ratios else 0.0
    min_r = min(all_ratios) if all_ratios else 0.0
    max_r = max(all_ratios) if all_ratios else 0.0
    logger.info(
        "Budget summary: mean=%.4f  min=%.4f  max=%.4f  (target=%.4f)",
        mean_r, min_r, max_r, args.target_ratio,
    )


if __name__ == "__main__":
    main()
