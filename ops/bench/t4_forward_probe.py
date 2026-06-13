#!/usr/bin/env python3
"""Standalone T4 single-token forward probe — isolates the ~150s decode
stall from OpenHydra.

Loads a Qwen3.5 model with *pure* transformers (no OpenHydra shard code,
no patches) and times forwards at the exact shapes OpenHydra hits:

  * seq_len=1, past_len=0   (the shape the prod probe stalled on)
  * seq_len=8               (prefill — for contrast)
  * seq_len=1 with KV cache (decode step with past)

For each forward it records wall time and samples GPU utilisation in a
background thread, so we can tell a CPU-bound stall (0% GPU) from a slow
GPU kernel. It then re-runs the seq_len=1 case under cuDNN toggles to
localise the cause:

  A. baseline (cudnn enabled, benchmark=False)
  B. cudnn.benchmark = True   (autotune — can be slow first call)
  C. cudnn DISABLED globally  (forces aten native conv kernels)

Hypothesis under test: the depthwise Conv1d in Qwen3.5 ``linear_attn``
hits cuDNN's "GET was unable to find an engine" search at seq_len<4 on
Turing (SM 7.5), which is CPU-bound and one-time per shape.

Usage:
    python3 t4_forward_probe.py --model Qwen/Qwen3.5-2B
"""
from __future__ import annotations

import argparse
import threading
import time
import traceback

import torch


def _gpu_sampler(stop_evt, samples, idx=0):
    """Sample GPU util% + mem via pynvml (falls back to nvidia-smi parse)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(idx)
        while not stop_evt.is_set():
            u = pynvml.nvmlDeviceGetUtilizationRates(h)
            m = pynvml.nvmlDeviceGetMemoryInfo(h)
            samples.append((u.gpu, m.used // (1024 * 1024)))
            stop_evt.wait(0.25)
    except Exception:
        import shutil
        import subprocess
        if not shutil.which("nvidia-smi"):
            return
        while not stop_evt.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi",
                     "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    text=True, timeout=5,
                ).strip().splitlines()[idx]
                g, mm = (x.strip() for x in out.split(","))
                samples.append((int(g), int(mm)))
            except Exception:
                pass
            stop_evt.wait(0.5)


def _summarise(samples):
    if not samples:
        return "no gpu samples"
    gus = [s[0] for s in samples]
    mus = [s[1] for s in samples]
    return (f"gpu_util min/avg/max={min(gus)}/{sum(gus)//len(gus)}/{max(gus)}% "
            f"mem max={max(mus)}MiB n={len(samples)}")


def timed_forward(model, input_ids, *, past=None, label, use_cache=False):
    """Run one forward, sampling GPU util concurrently."""
    stop = threading.Event()
    samples: list = []
    th = threading.Thread(target=_gpu_sampler, args=(stop, samples), daemon=True)
    th.start()
    t0 = time.perf_counter()
    err = None
    out = None
    try:
        with torch.no_grad():
            kw = {"use_cache": use_cache}
            if past is not None:
                kw["past_key_values"] = past
            out = model(input_ids, **kw)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        err = exc
    dt = time.perf_counter() - t0
    stop.set()
    th.join(timeout=2)
    status = "OK" if err is None else f"RAISED: {type(err).__name__}: {str(err)[:200]}"
    print(f"  [{label}] {dt:8.2f}s  {status}")
    print(f"           {_summarise(samples)}")
    if err is not None and "GET was unable to find an engine" in str(err):
        print("           >>> cuDNN GET-engine failure (the known T4 conv1d path)")
    return dt, out, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = ap.parse_args()

    print("=" * 70)
    print("ENVIRONMENT")
    print("=" * 70)
    print(f"torch          {torch.__version__}")
    print(f"cuda runtime   {torch.version.cuda}")
    print(f"cudnn          {torch.backends.cudnn.version()}")
    print(f"cuda available {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability()
        print(f"device         {torch.cuda.get_device_name(0)} (sm_{cc[0]}{cc[1]})")
        print(f"is_turing      {cc == (7, 5)}")
    print(f"cudnn.enabled  {torch.backends.cudnn.enabled}")
    print(f"cudnn.benchmark {torch.backends.cudnn.benchmark}")
    try:
        import causal_conv1d  # noqa: F401
        print("causal_conv1d  INSTALLED")
    except Exception:
        print("causal_conv1d  NOT installed (Qwen3.5 falls back to cuDNN/aten conv)")
    print()

    dtype = getattr(torch, args.dtype)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print(f"LOADING {args.model} dtype={args.dtype} device={dev}")
    print("=" * 70)
    from transformers import AutoModelForCausalLM
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(dev).eval()
    print(f"loaded + moved to {dev} in {time.perf_counter() - t0:.1f}s")
    p0 = next(model.parameters())
    print(f"param device={p0.device} dtype={p0.dtype}")
    print()

    ids1 = torch.tensor([[100]], device=dev)        # seq_len=1
    ids8 = torch.tensor([[100] * 8], device=dev)     # seq_len=8 (prefill)

    print("=" * 70)
    print("FORWARDS (pure transformers, NO OpenHydra patches)")
    print("=" * 70)

    print("seq_len=1, past=None (the shape the prod probe stalled on):")
    timed_forward(model, ids1, label="seq1 #1 (cold)", use_cache=False)
    timed_forward(model, ids1, label="seq1 #2 (warm)", use_cache=False)
    timed_forward(model, ids1, label="seq1 #3 (warm)", use_cache=False)

    print("\nseq_len=8, past=None (prefill shape, conv1d seq>=4):")
    timed_forward(model, ids8, label="seq8 #1 (cold)", use_cache=False)
    timed_forward(model, ids8, label="seq8 #2 (warm)", use_cache=False)

    print("\nKV-aware decode: prefill seq=8 then decode seq=1 with past:")
    stop = threading.Event(); samples: list = []
    th = threading.Thread(target=_gpu_sampler, args=(stop, samples), daemon=True); th.start()
    t0 = time.perf_counter()
    with torch.no_grad():
        pf = model(ids8, use_cache=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"  [prefill seq8 use_cache] {time.perf_counter()-t0:8.2f}s  {_summarise(samples)}")
    stop.set()
    nxt = torch.tensor([[101]], device=dev)
    timed_forward(model, nxt, past=pf.past_key_values, label="decode seq1 +past", use_cache=True)

    print("\n" + "=" * 70)
    print("cuDNN TOGGLE MATRIX on seq_len=1 (localise the stall)")
    print("=" * 70)

    print("A. cudnn.enabled=True, benchmark=False (baseline):")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    timed_forward(model, ids1, label="A enabled/bench=F", use_cache=False)

    print("B. cudnn.enabled=True, benchmark=True (autotune):")
    torch.backends.cudnn.benchmark = True
    timed_forward(model, ids1, label="B enabled/bench=T #1", use_cache=False)
    timed_forward(model, ids1, label="B enabled/bench=T #2", use_cache=False)
    torch.backends.cudnn.benchmark = False

    print("C. cudnn.enabled=False (force aten native conv):")
    torch.backends.cudnn.enabled = False
    timed_forward(model, ids1, label="C disabled #1", use_cache=False)
    timed_forward(model, ids1, label="C disabled #2", use_cache=False)
    torch.backends.cudnn.enabled = True

    print("\nDONE.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
