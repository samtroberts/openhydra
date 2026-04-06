# OpenHydra Session Memory

> Updated: 2026-04-06 (v1.2 Swarm Optimization complete, 1103 tests, nanodes deleted)
> Purpose: Gives Claude the context needed to resume work without re-reading everything.

---

## Where We Stand

### Completed Passes

| Pass | Scope | Status |
|------|-------|--------|
| 1-4 | Core infra, DHT fixes, TCP pooling, test stabilisation | Done |
| 5 | Live inference test, Petals/Exo analysis, beta launch strategy | Done |
| 6.x | MkDocs, Tauri desktop, Ollama API, GitHub publish, security audit | Done |
| Equalization | Engine decomp, 6-factor routing, gRPC streaming, MLX parallelism | Done |
| Pass 8 QA | RAM fix, CI codesign, Golden Path docs, 8GB benchmark | Done |
| v1.1 | Hybrid Local/Swarm Mode (4 pillars), 75 TPS Local, 20 TPS Swarm | Done |
| v1.2 | Swarm Optimization: DSD, SpecPipe, INT8, TOPLOC, Chunked Prefill, 5-node WAN pipeline | Done |

### Current State

- **v1.2 complete** — all swarm inference optimization techniques implemented, tested on real WAN pipeline, nanodes documented and deleted.
- **1103 tests pass, 9 skipped, 0 failures**
- **5-node WAN sharded pipeline** tested end-to-end (Bangalore→Chennai→Mumbai→Singapore×2) producing coherent multi-token output through SpecPipe.
- **ToyRuntime** replaced with real tinyllama-15M model.
- **Model catalog**: 9 entries (Qwen3.5 family + Qwen2.5-0.5B + SmolLM2-360M + Gemma-3-270m + TinyLLaMA-15M).

### Key Files (v1.2 additions)

| File | What it is |
|------|------------|
| `coordinator/specpipe_scheduler.py` | SpecPipe: concurrent speculative pipeline filling (~220 lines) |
| `coordinator/chunked_prefill.py` | Chunked prefill for long prompt interleaving (~160 lines) |
| `coordinator/speculative_swarm.py` | DSD accept/reject logic |
| `coordinator/local_engine.py` | LocalInferenceEngine for offline mode |
| `peer/activation_codec.py` | INT8 activation compression |
| `verification/toploc.py` | SHA-256 activation hashing |
| `ops/nanode-snapshots/SETUP_GUIDE.md` | Complete nanode recreation guide (9 errors documented) |
| `scripts/benchmarks/wan5_qwen25_05b_benchmark_2026-04-06.md` | Full WAN benchmark results |
| `scripts/test_sharded_local.py` | Local 2-stage sharded pipeline verification |

### Test Suite

- **1103 passed, 9 skipped** (pytest)

### Production Bootstrap Nodes

- EU: 172.105.69.49:8468
- US: 45.79.190.172:8468
- AP: 172.104.164.98:8468
- Peer nanodes: **DELETED** (snapshots at `ops/nanode-snapshots/`)

### Critical Architecture Facts

- Entry point: `openhydra-node` → `coordinator/node.py`
- gRPC: peer ↔ coordinator on port 50051
- DHT cache TTL: 120s coordinator / 60s peer re-announce / 300s expiry
- `dht_lookup_timeout_s`: 3.0 (was 0.5)
- Pipeline width: 3 (configurable, max 16)
- Ed25519 identity, X25519 ECDH + AES-GCM encryption (0.02% overhead)
- Layer sharding: **FULLY ACTIVATED** — DHT announces layer ranges, coordinator builds sharded pipelines via Dijkstra
- position_embeddings: computed per-layer via `rotary_emb(hidden, position_ids)` — MUST NOT be wrapped in try/except
- Peer dedup: preserves layer_start/layer_end from static config when DHT peers lack it
- `accelerate>=1.13.0` required for `transformers>=5.3` + `device_map`

---

## What to Do Next

All beta launch phases (0-5) and v1.1/v1.2 are complete. Priorities:

1. **Improve sharded pipeline TPS** — 0.43 TPS on 5-stage WAN needs optimization (parallel draft eval, SpecPipe tuning, KV persistence).
2. **Larger model testing** — 7B+ across 8+ peers for coherent output quality.
3. **Rich interactive CLI** — rewrite `coordinator/interactive_cli.py` with `prompt_toolkit`.
4. **Tauri UI/UX polish** — onboarding wizard, real-time peer map, model browser, earnings widget.
5. **On-chain integration** — replace `mock_mode=True` in `ledger_bridge.py` with real Solidity contracts.

Always update this file and `progress.md` at the end of each session.
