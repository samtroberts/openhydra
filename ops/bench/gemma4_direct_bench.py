"""Direct Gemma 4 E4B-it benchmark on a single T4 (non-sharded).

Loads the full text decoder (vision/audio towers stripped) and runs
``model.generate()`` with warm and cold measurements. Written to let us
produce real TPS numbers for Gemma 4 on T4 without requiring OpenHydra's
sharded ``_run_layers`` path, which does not yet support Gemma 4's
layer-type-aware rotary + per_layer_input wiring.
"""
import argparse
import logging
import os
import subprocess
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def gpu_used_mib() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        ).decode().strip().split("\n")[0]
        return int(out)
    except Exception:
        return -1


def strip_aux(model) -> None:
    inner = getattr(model, "model", None)
    if inner is None:
        return
    for attr in ("vision_tower", "audio_tower", "embed_vision", "embed_audio", "multi_modal_projector"):
        if hasattr(inner, attr) and isinstance(getattr(inner, attr), torch.nn.Module):
            setattr(inner, attr, None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/teamspace/studios/this_studio/openhydra/models/gemma-4-E4B-it")
    ap.add_argument("--prompt", default="Write a single short sentence about the ocean.")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    logging.info("gpu_before_load: %d MiB", gpu_used_mib())
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        dtype=torch.float16,
        trust_remote_code=False,
    )
    t1 = time.time()
    logging.info("cpu_load: %.1fs type=%s", t1 - t0, type(model).__name__)

    strip_aux(model)
    import gc
    gc.collect()
    logging.info("after_strip_aux: %d MiB (should still be 0 — on CPU)", gpu_used_mib())

    # Move text decoder + lm_head to GPU
    inner = model.model
    text_decoder = inner.language_model  # Gemma4TextModel
    t2 = time.time()
    text_decoder.to("cuda:0", dtype=torch.float16)
    model.lm_head.to("cuda:0", dtype=torch.float16)
    gc.collect()
    t3 = time.time()
    logging.info("gpu_move: %.1fs vram=%d MiB", t3 - t2, gpu_used_mib())

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build chat-formatted prompt
    messages = [{"role": "user", "content": args.prompt}]
    try:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        input_text = args.prompt
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda:0")
    prompt_len = inputs["input_ids"].shape[1]
    logging.info("prompt_tokens: %d", prompt_len)

    # Warmup (cold TTFT) — generate 1 token
    logging.info("warmup_run: generating 1 token...")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    warmup_s = time.time() - t0
    logging.info("warmup_complete: %.2fs", warmup_s)

    # Measured runs — warm path
    results = []
    for i in range(args.runs):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        generated = out.shape[1] - prompt_len
        tps = generated / elapsed if elapsed > 0 else 0.0
        results.append((elapsed, generated, tps))
        text = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
        logging.info(
            "run_%d: %.2fs gen=%d tps=%.2f text=%r",
            i + 1, elapsed, generated, tps, text[:120],
        )

    elapsed_avg = sum(r[0] for r in results) / len(results)
    gen_avg = sum(r[1] for r in results) / len(results)
    tps_avg = sum(r[2] for r in results) / len(results)
    logging.info(
        "SUMMARY: runs=%d avg_elapsed=%.2fs avg_gen=%.1f avg_tps=%.2f peak_vram=%d MiB",
        len(results), elapsed_avg, gen_avg, tps_avg, gpu_used_mib(),
    )
    print(f"RESULT tps_avg={tps_avg:.2f} elapsed={elapsed_avg:.2f} gen={gen_avg:.1f} vram={gpu_used_mib()}")


if __name__ == "__main__":
    main()
