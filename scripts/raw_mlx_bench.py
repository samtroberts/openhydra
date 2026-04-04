#!/usr/bin/env python3
"""Raw MLX-LM benchmark — bypasses OpenHydra entirely.

Loads a 4-bit quantized model directly via mlx_lm and measures
generation TPS with no coordinator, no gRPC, no HTTP overhead.
"""

import time
from mlx_lm import load, stream_generate

MODEL_ID = "prism-ml/Bonsai-1.7B-mlx-1bit"
PROMPT = "Write a short story about Harry Potter."
MAX_TOKENS = 128

print(f"Loading {MODEL_ID}...")
t0 = time.perf_counter()
model, tokenizer = load(MODEL_ID)
t_load = time.perf_counter() - t0
print(f"Model loaded in {t_load:.2f}s")

print(f"\nGenerating {MAX_TOKENS} tokens...")
print(f"Prompt: {PROMPT}\n")
print("--- OUTPUT ---")

tokens = []
t_first = None
t_start = time.perf_counter()

for response in stream_generate(
    model,
    tokenizer,
    prompt=PROMPT,
    max_tokens=MAX_TOKENS,
):
    if t_first is None:
        t_first = time.perf_counter()
    tokens.append(response.token)
    print(tokenizer.decode([response.token]), end="", flush=True)
    if response.finish_reason is not None:
        break

t_end = time.perf_counter()
print("\n--- END ---\n")

ttft = (t_first - t_start) * 1000 if t_first else 0
gen_time = t_end - (t_first or t_start)
tps = len(tokens) / gen_time if gen_time > 0 else 0

print(f"Tokens generated: {len(tokens)}")
print(f"TTFT:             {ttft:.0f} ms")
print(f"Generation time:  {gen_time:.3f} s")
print(f"TPS:              {tps:.1f} tok/s")
