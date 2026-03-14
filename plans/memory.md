# OpenHydra Session Memory

> Updated: 2026-03-10 (Phase 1 complete)
> Purpose: Gives Claude the context needed to resume work without re-reading everything.

---

## Where We Stand

### Completed Passes (from plan file)

| Pass | Scope | Status |
|------|-------|--------|
| 1-4 | Core infra, DHT fixes, TCP pooling, test stabilisation | Done |
| 5 | Live inference test, Petals/Exo analysis, beta launch strategy | Done |
| 6.5 | MkDocs documentation site (4 Mermaid diagrams, CI job) | Done |
| 6.6 | Tauri desktop: 5 UX features (theme, earnings, palette, history, peer map) | Done |

### Current Work

- **Phase 1 complete** — MLX backend fully implemented, wired, tested, and benchmarked.
  - `peer/mlx_runtime.py` — full `MLXRuntime` class: DLPack bridges, `encode_prompt`, `_warmup`, `forward`, `runtime_profile`, Phase 3 stubs.
  - `"mlx"` added to `--runtime-backend` choices in `peer/server.py`.
  - `ModelShard.__init__` now routes `runtime_backend="mlx"` to `MLXRuntime` (lazy import).
  - `mlx = ["mlx>=0.20", "mlx-lm>=0.20"]` optional dep group added to `pyproject.toml`.
  - 31 new tests in `tests/test_mlx_runtime.py` (skip guard if mlx not installed); full suite: **454 passed, 9 skipped**.
  - **Live benchmark on Qwen3.5-0.8B (Apple Silicon, Metal):**
    - Load: 5.67 s | Warmup: 0.61 s
    - TTFT (warm run 1): **0.919 s** | TTFT (run 2, cached): **0.366 s**
    - Avg TPS: **252 tok/s** (was 1.3 tok/s with pytorch_auto/MPS) — **194× speedup**
  - API fixes discovered during testing:
    - `model.parameters()` in MLX returns nested dict → use `mlx.utils.tree_flatten`
    - `TokenizerWrapper` not callable → use `.encode(text)` directly
    - `stream_generate` no longer accepts `temp/top_p/top_k` kwargs → use `make_sampler()` from `mlx_lm.sample_utils`
- **Phase 0 complete** — all four quick-win tasks implemented and tested.
- **Plans directory + auto-scaling sub-plan** — capability-aware scaling design complete, master plan updated.

### Key Files

| File | What it is |
|------|------------|
| `docs/beta-launch-strategy.md` | Master plan (8 sections, roadmap through Phase 6) |
| `plans/README.md` | Sub-plan index |
| `plans/auto-scaling-policy.md` | Detailed auto-scaling design |
| `plans/progress.md` | Milestone tracker |
| `plans/memory.md` | This file |
| `peer/mlx_runtime.py` | MLX inference backend (Phase 1) — 194× speedup on Apple Silicon |
| `tests/test_mlx_runtime.py` | 31 MLX tests (skip-guarded for non-Apple machines) |
| `desktop/src/app.js` | Tauri frontend (1439 lines, all monkey-patching bugs fixed) |
| `docs/architecture.md` | Architecture reference (4 Mermaid diagrams added) |

### Test Suite Baseline

- **454 passed, 9 skipped** (pytest) — was 423; Phase 1 adds 31 tests
- **mkdocs build --strict** passes (0 warnings)

### Background Analyses Completed

| Topic | Verdict | Detail |
|-------|---------|--------|
| DLPack | Skip for beta | Solves same-machine tensor interchange, not network |
| Hivemind | High value, defer to v1.0 | Current HTTP DHT works for <1000 peers |
| Exo deep-dive | Borrow MLX + shard-aware downloads | Fixed ring topology, RDMA over Thunderbolt 5, memory-weighted partitioning |

### Production Bootstrap Nodes

- EU: 172.105.69.49:8468
- US: 45.79.190.172:8468
- AP: 172.104.164.98:8468

### Critical Architecture Facts

- Entry point: `openhydra-node` → `coordinator/node.py`
- gRPC: peer <-> coordinator on port 50051
- DHT cache TTL: 120s coordinator / 60s peer re-announce / 300s expiry
- `dht_lookup_timeout_s`: 3.0 (was 0.5)
- Pipeline width: 3 (configurable, max 16)
- Ed25519 identity, X25519 ECDH + AES-GCM encryption (0.02% overhead)
- Layer sharding infrastructure EXISTS in `model_shard.py` (not yet wired to DHT/coordinator)

---

## What to Do Next

Phases 0 and 1 are complete. Next phase in order:

1. **Phase 2: Auto-Scaling** (`coordinator/auto_promoter.py`) — implement capability-aware `AutoPromoter` per `plans/auto-scaling-policy.md`. Wire into `CoordinatorEngine`. Add Qwen 3.5 family to `models.catalog.json`.
2. **Phase 3: Layer Sharding** — wire DHT layer-range announcements + coordinator pipeline assembly (infra already in `model_shard.py`; Phase 3 stubs in `mlx_runtime.py` ready).
3. **Phase 4: NF4 Quantization + Request Coalescing** — bitsandbytes + `BatchingQueue`.
4. Any new design conversation → create a new sub-plan file.

Always update this file and `progress.md` at the end of each session.
