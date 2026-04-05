#!/usr/bin/env python3
"""Local sharded pipeline test — verifies position_embeddings fix.

Splits tinyllama-15M into 2 stages and compares output with full-model inference.
Both should produce the same token, proving hidden state + position_embeddings
transfer works correctly.

Usage:
    python3 scripts/test_sharded_local.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "nickypro/tinyllama-15M"
PROMPT = "Once upon a time there was"


def main():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()

    # Encode prompt
    inputs = tokenizer(PROMPT, return_tensors="pt")
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]
    print(f"Prompt: {PROMPT!r} ({seq_len} tokens)")

    # --- Full model reference ---
    with torch.no_grad():
        full_out = model(input_ids)
        full_logits = full_out.logits[0, -1, :]
        full_token = torch.argmax(full_logits).item()
        full_text = tokenizer.decode(full_token)
    print(f"\nFull model next token: {full_token} = {full_text!r}")

    # --- Sharded: Stage 0 (embed + layers 0-5) → Stage 1 (layers 6-11 + lm_head) ---
    layers = list(model.model.layers)
    total_layers = len(layers)
    split = total_layers // 2
    print(f"\nTotal layers: {total_layers}, split at {split}")
    print(f"Stage 0: layers [0, {split})")
    print(f"Stage 1: layers [{split}, {total_layers})")

    rotary_emb = model.model.rotary_emb
    embed_tokens = model.model.embed_tokens
    norm = model.model.norm

    with torch.no_grad():
        # Stage 0: embed + first half of layers
        position_ids = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0)
        hidden = embed_tokens(input_ids)

        for i, block in enumerate(layers[:split]):
            pos_emb = rotary_emb(hidden, position_ids)
            block_out = block(
                hidden,
                position_ids=position_ids,
                position_embeddings=pos_emb,
            )
            hidden = block_out[0] if isinstance(block_out, tuple) else block_out

        # Serialize hidden state (simulates wire transfer)
        payload = hidden.flatten().tolist()
        h_size = hidden.shape[-1]
        wire = [float(seq_len), float(h_size)] + payload
        print(f"Wire payload: {len(wire)} floats ({seq_len}×{h_size} + 2 header)")

        # Stage 1: deserialize + second half of layers + norm + lm_head
        r_seq = int(wire[0])
        r_h = int(wire[1])
        hidden2 = torch.tensor(wire[2:], dtype=torch.float32).view(1, r_seq, r_h)
        recon_error = (hidden - hidden2).abs().max().item()
        print(f"Reconstruction error: {recon_error}")

        for i, block in enumerate(layers[split:]):
            pos_emb = rotary_emb(hidden2, position_ids)
            block_out = block(
                hidden2,
                position_ids=position_ids,
                position_embeddings=pos_emb,
            )
            hidden2 = block_out[0] if isinstance(block_out, tuple) else block_out

        hidden2 = norm(hidden2)
        logits = model.lm_head(hidden2)
        sharded_logits = logits[0, -1, :]
        sharded_token = torch.argmax(sharded_logits).item()
        sharded_text = tokenizer.decode(sharded_token)

    print(f"Sharded pipeline next token: {sharded_token} = {sharded_text!r}")

    # Compare
    logit_diff = (full_logits - sharded_logits).abs().max().item()
    print(f"\nLogit max diff: {logit_diff:.6f}")
    if full_token == sharded_token:
        print("✅ PASS — sharded pipeline produces identical output")
        return 0
    else:
        print(f"❌ FAIL — full={full_token} ({full_text!r}), sharded={sharded_token} ({sharded_text!r})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
