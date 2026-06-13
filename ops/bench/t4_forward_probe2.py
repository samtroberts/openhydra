#!/usr/bin/env python3
"""T4 probe #2 — does OpenHydra's conv1d fallback fix the seq_len=1 stall?

Probe #1 proved the raw seq_len=1 forward fast-FAILS (cuDNN "GET ... engine")
and that cudnn.enabled=False works (0.23s). OpenHydra's
``_patch_depthwise_conv1d_t4_fallback`` catches that error and retries with
cuDNN disabled. This probe replicates that exact patch (and the Turing SDPA
math routing) on a pure-transformers model and confirms the failing shape
becomes fast — and tests whether the "FIND" variant (benchmark=True) slips
past the fallback's "GET"-only match.
"""
from __future__ import annotations

import argparse
import time

import torch
from torch import nn


def apply_openhydra_conv1d_patch(model):
    """Exact replica of model_shard.py _patch_depthwise_conv1d_t4_fallback."""
    patched = 0
    for mod in model.modules():
        if not isinstance(mod, nn.Conv1d):
            continue
        if mod.groups != mod.in_channels:
            continue
        _original = mod._conv_forward

        def _fallback_conv_forward(inp, weight, bias, _orig=_original):
            try:
                return _orig(inp, weight, bias)
            except RuntimeError as exc:
                if "unable to find an engine" not in str(exc):
                    raise
                with torch.backends.cudnn.flags(enabled=False):
                    return _orig(inp, weight, bias)

        mod._conv_forward = _fallback_conv_forward
        patched += 1
    return patched


def timed(model, ids, label, n=3):
    for i in range(n):
        t0 = time.perf_counter()
        err = None
        try:
            with torch.no_grad():
                model(ids, use_cache=False)
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            err = exc
        dt = time.perf_counter() - t0
        st = "OK" if err is None else f"RAISED {type(err).__name__}: {str(err)[:120]}"
        print(f"  [{label} #{i+1}] {dt:7.2f}s  {st}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    args = ap.parse_args()

    dev = "cuda"
    from transformers import AutoModelForCausalLM
    print(f"loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(dev).eval()

    n = apply_openhydra_conv1d_patch(model)
    print(f"patched {n} depthwise Conv1d modules with OpenHydra fallback\n")

    ids1 = torch.tensor([[100]], device=dev)

    print("With OpenHydra conv1d fallback (cudnn.benchmark=False):")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    timed(model, ids1, "seq1 patched")

    print("\nWith cudnn.benchmark=True (error text becomes 'FIND', not 'GET'):")
    torch.backends.cudnn.benchmark = True
    timed(model, ids1, "seq1 patched bench=T", n=2)
    torch.backends.cudnn.benchmark = False

    print("\nDONE.")


if __name__ == "__main__":
    main()
