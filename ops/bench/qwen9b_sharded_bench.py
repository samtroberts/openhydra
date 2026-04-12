"""Direct sharded benchmark for Qwen3.5-9B across 2 Lightning T4s.

Bypasses the OpenHydra coordinator entirely and drives both peers via gRPC
manually. This isolates the sharded forward pipeline from coordinator-level
quirks (specpipe, KV cache, grounding) and produces clean TPS numbers.

Flow per decode step:
    step 0: tokenize prompt → context_ids
    step 1: gRPC Forward(stage=0, total_stages=2, activation=context_ids)
            → peer 1 runs layers 0-15, returns hidden state
    step 2: gRPC Forward(stage=1, total_stages=2, activation=hidden)
            → peer 2 runs layers 16-31 + lm_head, returns next token id
    step 3: append token to context, go to step 1 (full re-prefill —
            no KV cache, so each decode step re-runs the full prompt
            through both peers).

This is O(max_tokens²) cost (re-prefill per step) but correct without any
KV cache coordination. For small max_tokens (8-32) it's still usable as a
proof-of-work and apples-to-apples TPS benchmark.
"""
import argparse
import os
import sys
import time

import grpc

sys.path.insert(0, "/Users/sam/Documents/New project 2")
from peer import peer_pb2, peer_pb2_grpc


def _forward(stub, *, prompt: str, activation, stage: int, total_stages: int,
             max_tokens: int, request_id: str) -> peer_pb2.ForwardResponse:
    req = peer_pb2.ForwardRequest(
        request_id=request_id,
        prompt=prompt,
        activation=list(activation),
        stage_index=stage,
        total_stages=total_stages,
        max_tokens=max_tokens,
        shard_layer_start=0 if stage == 0 else 16,
        shard_layer_end=16 if stage == 0 else 32,
        shard_total_layers=32,
    )
    return stub.Forward(req, timeout=300)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--peer1", default="127.0.0.1:50099",
                    help="host:port of the stage-0 peer (layers 0-15)")
    ap.add_argument("--peer2", default="127.0.0.1:50098",
                    help="host:port of the stage-1 peer (layers 16-31)")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-9B",
                    help="HuggingFace tokenizer id (must match the peers)")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    print(f"loading tokenizer: {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    eos_ids = set()
    if isinstance(tok.eos_token_id, int):
        eos_ids.add(int(tok.eos_token_id))
    # Qwen3.5 uses <|im_end|> + <|endoftext|> as stop tokens
    for tok_name in ("<|im_end|>", "<|endoftext|>"):
        tid = tok.convert_tokens_to_ids(tok_name)
        if isinstance(tid, int) and tid >= 0:
            eos_ids.add(tid)
    print(f"eos_ids: {sorted(eos_ids)}")

    ch1 = grpc.insecure_channel(
        args.peer1,
        options=[("grpc.max_send_message_length", 64 * 1024 * 1024),
                 ("grpc.max_receive_message_length", 64 * 1024 * 1024)],
    )
    ch2 = grpc.insecure_channel(
        args.peer2,
        options=[("grpc.max_send_message_length", 64 * 1024 * 1024),
                 ("grpc.max_receive_message_length", 64 * 1024 * 1024)],
    )
    stub1 = peer_pb2_grpc.PeerStub(ch1)
    stub2 = peer_pb2_grpc.PeerStub(ch2)

    # Ensure peers are reachable
    for i, (stub, name) in enumerate(((stub1, "peer1 (layers 0-15)"), (stub2, "peer2 (layers 16-31)"))):
        try:
            _ = stub.Forward(peer_pb2.ForwardRequest(
                request_id="ping", prompt="hi",
                activation=[], stage_index=0, total_stages=2,
                max_tokens=1,
            ), timeout=5)
        except grpc.RpcError as e:
            print(f"peer {i+1} ping: {e.code()} — {e.details()[:200]}")

    # Build chat-formatted prompt
    try:
        formatted = tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        formatted = args.prompt
    context_ids: list[int] = tok.encode(formatted, add_special_tokens=True)
    prompt_len = len(context_ids)
    print(f"prompt tokens: {prompt_len}")
    print(f"prompt text: {formatted[:160]}")

    for run_idx in range(args.runs):
        generated: list[int] = []
        t0 = time.time()
        for step in range(args.max_new_tokens):
            # Stage 0: prefill-or-decode through peer 1
            #   activation = full context token IDs → peer 1 tokenizes them
            #   as input_ids, runs layers 0-15, returns hidden state
            resp1 = _forward(
                stub1,
                prompt="",
                activation=[float(t) for t in context_ids + generated],
                stage=0, total_stages=2, max_tokens=1,
                request_id=f"bench-{run_idx}-{step}",
            )
            if resp1.error:
                print(f"[step {step}] peer1 error: {resp1.error[:200]}")
                return
            hidden = list(resp1.activation)

            # Stage 1: run hidden state through peer 2's layers 16-31 +
            #          lm_head, sample next token, return [token_id]
            resp2 = _forward(
                stub2,
                prompt="",
                activation=hidden,
                stage=1, total_stages=2, max_tokens=1,
                request_id=f"bench-{run_idx}-{step}",
            )
            if resp2.error:
                print(f"[step {step}] peer2 error: {resp2.error[:200]}")
                return
            out = list(resp2.activation)
            if not out:
                print(f"[step {step}] peer2 returned empty activation")
                break
            next_id = int(round(float(out[0])))
            generated.append(next_id)
            if next_id in eos_ids:
                print(f"[step {step}] hit eos {next_id}")
                break
        elapsed = time.time() - t0
        text = tok.decode(generated, skip_special_tokens=True)
        n = len(generated)
        tps = n / elapsed if elapsed > 0 else 0.0
        print(f"run {run_idx + 1}: gen={n} elapsed={elapsed:.2f}s tps={tps:.3f}")
        print(f"text: {text!r}")


if __name__ == "__main__":
    main()
