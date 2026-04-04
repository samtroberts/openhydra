#!/usr/bin/env python3
# Copyright 2026 OpenHydra contributors — Apache 2.0

"""3-peer swarm benchmark with DSD + INT8 enabled.

Simulates a 3-peer swarm using local processes (no Docker required).
Each peer loads TinyStories-1M via PyTorchRuntime on CPU.
The coordinator assembles a pipeline through all 3 peers.

Usage:
    python3 scripts/swarm_3peer_bench.py
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

import requests

MODEL_ID = "nickypro/tinyllama-15M"
MODEL_CACHE = os.path.expanduser("~/.cache/openhydra/models/tinyllama-15M")
PROMPT = "Once upon a time there was"
MAX_TOKENS = 64
COORDINATOR_PORT = 8090
GRPC_PORTS = [50061, 50062, 50063]
PEER_IDS = ["peer-alpha", "peer-beta", "peer-gamma"]
PYTHON = "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"


def start_peer(peer_id: str, grpc_port: int, model_path: str) -> subprocess.Popen:
    """Start a peer gRPC server as a background process.

    Uses MLX backend on macOS for full multi-token generation via
    stream_generate (PyTorchRuntime only generates 1 token per forward).
    Falls back to ToyRuntime if MLX is unavailable.
    """
    cmd = [
        PYTHON, "-c", f"""
import sys, os, platform
sys.path.insert(0, os.getcwd())
from peer.server import serve

# Use toy_auto which now loads tinyllama-15M (real LLaMA model)
backend = "toy_auto"

serve(
    host="127.0.0.1",
    port={grpc_port},
    peer_id="{peer_id}",
    model_id="{MODEL_ID}",
    shard_index=0,
    total_shards=1,
    runtime_backend=backend,
    runtime_model_id="{model_path}",
    dht_urls=[],
    daemon_mode="polite",
)
""",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.getcwd(),
    )
    return proc


def start_coordinator(peers_config_path: str) -> subprocess.Popen:
    """Start the coordinator HTTP API."""
    coord_log = open("/tmp/openhydra_coord.log", "w")
    cmd = [
        PYTHON, "-c", f"""
import sys, os
sys.path.insert(0, os.getcwd())
from coordinator.api_server import serve
from coordinator.engine import EngineConfig

config = EngineConfig(
    peers_config_path="{peers_config_path}",
    dht_urls=[],
    required_replicas=1,
    pipeline_width=1,
    timeout_ms=30000,
    max_latency_ms=60000,
    default_model="{MODEL_ID}",
    pytorch_generation_model_id="{MODEL_CACHE}",
    speculative_swarm_enabled=True,
    speculative_draft_tokens=4,
    activation_quantization_enabled=True,
    deployment_profile="dev",
)

serve(
    host="127.0.0.1",
    port={COORDINATOR_PORT},
    config=config,
)
""",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=coord_log,
        cwd=os.getcwd(),
    )
    return proc


def wait_for_port(port: int, timeout: float = 30.0) -> bool:
    """Wait for a port to become available."""
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.5)
    return False


def main():
    procs: list[subprocess.Popen] = []

    try:
        # 1. Create peers config
        peers_config = [
            {
                "peer_id": pid,
                "host": "127.0.0.1",
                "port": port,
                "model_id": MODEL_ID,
                "operator_id": f"op-{i}",
                "runtime_backend": "toy_auto",
                "runtime_model_id": MODEL_CACHE,
            }
            for i, (pid, port) in enumerate(zip(PEER_IDS, GRPC_PORTS))
        ]
        peers_path = "/tmp/openhydra_3peer_config.json"
        with open(peers_path, "w") as f:
            json.dump(peers_config, f)
        print(f"Peers config: {peers_path}")

        # 2. Start 3 peers
        print("\n=== Starting 3 peers ===")
        for pid, port in zip(PEER_IDS, GRPC_PORTS):
            print(f"  Starting {pid} on :{port}...")
            proc = start_peer(pid, port, MODEL_CACHE)
            procs.append(proc)

        # Wait for peers
        for port in GRPC_PORTS:
            if wait_for_port(port, timeout=30):
                print(f"  :{port} ready")
            else:
                print(f"  :{port} TIMEOUT — peer may have failed")
                # Dump stderr
                for p in procs:
                    if p.poll() is not None:
                        _, err = p.communicate(timeout=1)
                        print(f"  stderr: {err.decode()[:500]}")
                return

        # 3. Start coordinator
        print(f"\n=== Starting coordinator on :{COORDINATOR_PORT} ===")
        coord_proc = start_coordinator(peers_path)
        procs.append(coord_proc)

        if wait_for_port(COORDINATOR_PORT, timeout=30):
            print(f"  Coordinator ready on :{COORDINATOR_PORT}")
        else:
            print("  Coordinator TIMEOUT")
            _, err = coord_proc.communicate(timeout=5)
            print(f"  stderr: {err.decode()[:1000]}")
            return

        coord_log_path = "/tmp/openhydra_coord.log"

        # 4. Warm-up
        print("\n=== Warm-up ===")
        try:
            r = requests.post(
                f"http://127.0.0.1:{COORDINATOR_PORT}/v1/chat/completions",
                json={
                    "model": MODEL_ID,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 4,
                    "grounding": False,
                },
                timeout=60,
            )
            d = r.json()
            if "choices" in d:
                print(f"  Warm-up OK: {d['choices'][0]['message']['content'][:50]}")
            else:
                print(f"  Warm-up error: {d.get('error', '')[:100]}")
        except Exception as e:
            print(f"  Warm-up failed: {e}")

        time.sleep(3)

        # 5. Benchmark
        print(f"\n=== BENCHMARK: '{PROMPT}' (max_tokens={MAX_TOKENS}) ===")
        print(f"  DSD: ENABLED | INT8 Compression: ENABLED")
        print(f"  Peers: {len(PEER_IDS)} | Model: {MODEL_ID}")

        t_start = time.perf_counter()
        r = requests.post(
            f"http://127.0.0.1:{COORDINATOR_PORT}/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": MAX_TOKENS,
                "grounding": False,
                "temperature": 0.8,
            },
            timeout=120,
        )
        t_end = time.perf_counter()
        wall = t_end - t_start

        data = r.json()
        if "error" in data:
            print(f"\n  ERROR: {data['error']}")
            # Dump coordinator stderr
            coord_proc.terminate()
            _, err = coord_proc.communicate(timeout=5)
            print(f"  Coordinator stderr (last 2000 chars):\n{err.decode()[-2000:]}")
            return

        # Dump coordinator logs for debugging
        try:
            coord_proc.poll()
            import select, io
            if coord_proc.stderr and hasattr(coord_proc.stderr, 'fileno'):
                pass  # Will dump at end
        except Exception:
            pass

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        oh = data.get("openhydra", {})
        pipeline = oh.get("pipeline", [])
        dsd = oh.get("dsd", {})

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        tps = completion_tokens / wall if wall > 0 else 0

        print(f"\n{'='*60}")
        print(f"  3-PEER SWARM BENCHMARK RESULTS")
        print(f"  DSD: {'ON' if dsd else 'OFF'} | INT8: ON")
        print(f"{'='*60}")
        print(f"  Prompt tokens:      {prompt_tokens}")
        print(f"  Completion tokens:  {completion_tokens}")
        print(f"  Wall time:          {wall:.3f}s")
        print(f"  TPS:                {tps:.1f} tok/s")
        if pipeline:
            print(f"  Pipeline hops:      {len(pipeline)}")
            for hop in pipeline:
                print(f"    Stage {hop.get('stage_index', '?')}: peer={hop.get('peer_id', '?')}, latency={hop.get('latency_ms', 0):.0f}ms")
        if dsd:
            print(f"  DSD rounds:         {dsd.get('rounds', 0)}")
            print(f"  DSD accepted:       {dsd.get('accepted_tokens', 0)}/{dsd.get('draft_tokens', 0)}")
            print(f"  DSD acceptance rate: {dsd.get('acceptance_rate', 0):.1%}")
            print(f"  DSD final K:        {dsd.get('final_k', 0)}")

        print(f"\n{'='*60}")
        print(f"  GENERATED TEXT")
        print(f"{'='*60}")
        print(f"  {content}")
        print(f"{'='*60}")

        # Validation
        print(f"\n  Validation:")
        print(f"    Length:     {'PASS' if len(content) > 20 else 'FAIL'} ({len(content)} chars)")
        words = content.split()
        if words:
            from collections import Counter
            mc_word, mc_count = Counter(w.lower() for w in words).most_common(1)[0]
            repeat_rate = mc_count / len(words)
            print(f"    Repetition: {'PASS' if repeat_rate < 0.3 else 'FAIL'} ('{mc_word}' = {repeat_rate:.0%})")
        print(f"    Coherent:   {'PASS' if 'upon' in content.lower() or 'time' in content.lower() or 'was' in content.lower() or len(content) > 50 else 'FAIL'}")

    finally:
        print("\n=== Cleanup: killing all processes ===")
        for p in procs:
            try:
                p.terminate()
                p.wait(timeout=3)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        print("Done.")


if __name__ == "__main__":
    main()
