# OpenHydra Session Memory

> Updated: 2026-04-10 (Petals parity complete, Gemma 4 + Qwen 3.5, autonomous rebalancing, GPU benchmarks)
> Purpose: Gives Claude the context needed to resume work without re-reading everything.

---

## Where We Stand

### Completed Passes

| Pass | Scope | Status |
|------|-------|--------|
| 1-6 | Core infra, DHT, TCP, tests, MkDocs, Tauri desktop, Ollama API | Done |
| Equalization | Engine decomp, 6-factor routing, gRPC streaming, MLX parallelism | Done |
| Pass 8 QA | RAM fix, CI codesign, Golden Path docs, 8GB benchmark | Done |
| v1.1 | Hybrid Local/Swarm Mode (4 pillars), 75 TPS Local, 20 TPS Swarm | Done |
| v1.2 | Swarm Optimization: DSD, SpecPipe, INT8, TOPLOC, Chunked Prefill | Done |
| **Petals Parity** | 4 phases: push mode, streaming sessions, NAT relay, throughput bench | **Done** |
| **GPU Benchmarks** | Lightning.ai T4: 18.8 TPS localhost, 0.54 WAN, push 2.66x | **Done** |
| **New Models** | Gemma 4 + Qwen 3.5 (transformers 5.5.3, multimodal arch detection) | **Done** |
| **Rebalancing** | Autonomous peer-driven layer assignment | **Done** |

### Current State

- **1103 tests pass, 0 failures** (transformers 5.5.3)
- **Model catalog: 21 entries** — Qwen 2.5/3/3.5, Gemma 3/4, SmolLM2, TinyLLaMA (base + instruct)
- **Petals parity achieved** — push mode, streaming, NAT relay, throughput bench (+1,327 lines)
- **Selective weight loading** — 14GB model on 8GB Mac (peak 1.5GB)
- **Push mode verified** — 2.66x TPS (4.95→13.16 on localhost)
- **Autonomous rebalancing** — peers self-optimize layer positions

### Key Files (recent additions)

| File | What it is |
|------|------------|
| `coordinator/push_receiver.py` | PushResult callback registry for push mode |
| `coordinator/relay.py` | gRPC relay for NATted peers |
| `coordinator/stun_client.py` | Real RFC 5389 STUN (was stub) |
| `peer/throughput_bench.py` | Actual forward-pass TPS measurement + cache |
| `peer/autonomous_rebalancer.py` | Peer-autonomous layer rebalancing algorithm |
| `coordinator/specpipe_scheduler.py` | SpecPipe + run_pipelined() |
| `peer/activation_codec.py` | INT8 activation compression (now default on) |

### Test Suite

- **1103 passed, 9 skipped** (pytest)

### Production Bootstrap Nodes

- EU: 172.105.69.49:8468
- US: 45.79.190.172:8468
- AP: 172.104.164.98:8468
- Peer nanodes: **DELETED** (snapshots at `ops/nanode-snapshots/`, gitignored)

### Critical Architecture Facts

- Entry point: `openhydra-node` → `coordinator/node.py`
- gRPC: peer ↔ coordinator on port 50051
- **Push mode**: peers forward activations directly to next peer (no coordinator round-trip)
- **Streaming sessions**: ForwardStream RPC with StreamPool + InferenceSession history replay
- **NAT relay**: STUN probe → relay registration → proxied Forward calls
- **Selective loading**: `_build_selective_device_map()` maps unused layers to "disk"
- **Accelerate hooks**: removed after selective load for native speed
- **Throughput bench**: real model.generate() measurement, cached 24h
- **Autonomous rebalancing**: `should_rebalance()` runs every 6 announce cycles
- Layer sharding: FULLY ACTIVATED — DHT announces layer ranges, Dijkstra routing
- position_embeddings: computed per-layer via `rotary_emb()` — MUST NOT be wrapped in try/except
- `accelerate>=1.13.0` + `transformers>=5.5.0` required

---

## What to Do Next

1. **On-chain HYDRA integration** — replace mock_mode=True with real Solidity contracts
2. **Multi-GPU demo** — 7B instruct model across 4+ GPU peers with push mode (>5 TPS coherent output)
3. **Agent GUI** — point Open WebUI at OpenHydra's OpenAI-compatible API (zero code changes)
4. **Combine SpecPipe + push mode** — speculative tokens through the push pipeline
5. **Launch prep** — HN post, demo video, public endpoint

Always update this file and `progress.md` at the end of each session.
