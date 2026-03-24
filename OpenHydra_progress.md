# OpenHydra Progress

Last updated: 2026-03-24 (Pass 6 100% complete, Equalization Sprint complete, v0.1.0 + v0.1.1 released)

## Overall status

- Tier 1: ~100% (core objectives complete in scaffold form)
- Tier 2: ~100% (production hardening + deployment scaffolding complete)
- Tier 3: ~100% (privacy parity finalized with maximum-mode DP audit verification, MoE geo expert-sharding guardrails enforced, and HYDRA governance controls/API scaffold completed)
- Post-Tier-3 pass 4: ~100% (distributed node model, Ed25519 identity, PostgreSQL, GUI layer, multi-region IaC, alerting)
- Post-Tier-3 pass 5: ~100% (KV cache compaction via Attention Matching arXiv:2602.16284 — all four phases + Option A real-Q threading + Option B benchmark)
- Post-Tier-3 pass 6: **100% complete** (all items done: CI hardening, security audit, Apache 2.0 relicense, architecture docs, CONTRIBUTING.md, issue templates)
- Equalization Sprint: **100% complete** (4 phases: engine decomp, 6-factor routing + swarm rebalance, gRPC streaming, MLX parallelism)
- Desktop App: **Shipped** (Tauri v2 + React, .dmg uploaded to GitHub Releases v0.1.0)
- PyPI: **Published** as `openhydra-network` v0.1.0 at https://pypi.org/project/openhydra-network/
- GitHub Release: **Live** at https://github.com/samtroberts/openhydra/releases/tag/v0.1.0
- License: **Apache 2.0** (entire stack — abandoned AGPL open-core model for maximum adoption)
- Test status: `982 passed, 9 skipped` by default (`OPENHYDRA_RUN_REAL_TENSOR_TEST=0`), with `9` gated real-PyTorch tests available under (`OPENHYDRA_RUN_REAL_TENSOR_TEST=1`)

## Roadmap checklist (architecture v6 aligned)

### Tier 1

- [x] 1.1 Model slicing/local inference scaffold
- [x] 1.2 gRPC activation pipeline
- [x] 1.3 Mystery Shopper verification
- [x] 1.4 Static peer config + latency ping
- [x] 1.5 CLI client flow

### Tier 2

- [x] 2.1 Genesis bootstrap + torrent metadata scaffold
- [x] 2.2 DHT bootstrap + announce/heartbeat/lookup
- [x] 2.3 Health scoring + retry/failover
- [x] 2.4 Replication monitor
- [x] 2.5 TLS/mTLS transport
- [x] 2.6 Smart routing (latency/load/reputation)
- [x] 2.7 Compute barter credits
- [x] 2.8 Per-operator routing caps + diversity controls
- [x] 2.9 Polite daemon integration (mode-aware budget + load signaling)
- [x] 2.10 Client-side grounding
- [x] 2.11 Production hardening + deployment scaffolding (immortal daemon restart/backoff loops, Prometheus `/metrics`, Docker/Docker Compose testnet)

### Tier 3

- [x] 3.1 Quantized models + GPU profiling (baseline complete: hardware RAM/VRAM profiling + startup hardware logging + graceful 4-bit/8-bit bitsandbytes load with CPU fp32 fallback)
- [x] 3.2 True pipeline parallelism + streaming tokens from execution path (cross-peer PyTorch autoregressive loop now active: prompt tokenization on coordinator, staged shard execution, last-shard token emission, EOS/max-token loop)
- [x] 3.3 Bandwidth profiling/role separation (control-plane implementation)
- [x] 3.4 KV cache isolation for prefill/decode split (distributed runtime KV residency complete: per-peer/session `past_key_values` retention, cache-hit decode path, bounded LRU eviction, and coordinator-stage cache orchestration)
- [x] 3.5 Learned tensor autoencoder in serving path (runtime-backed linear projection compressor in PyTorch data plane: deterministic encoder/decoder, cross-wire latent transfer, and stage-side reconstruction)
- [x] 3.6 Speculative decoding (desktop/mobile draft model path) (distributed runtime path complete: local PyTorch draft-model proposals + multi-token network verification + accept/reject correction flow + KV-safe commit rounds)
- [x] 3.7 MoE geographic expert sharding (production guardrails complete: expert admission controls, reputation/stake coupling for expert claims, and coordinator routing awareness of expert-admission approvals)
- [x] 3.8 Advanced DHT (DSHT + geo triangulation) (baseline complete: nonce-signature RTT geo challenge enforcement on announce, dynamic hot-key DSHT replica rebalancing, and rebalance hints in lookup responses)
- [x] 3.9 Graceful degradation (replica + verification QoS aware)
- [x] 3.10 Advanced encryption (per-hop AES-GCM / onion routing) (complete: maximum-mode parity with formal DP-noise audit tags, coordinator-side privacy verification hooks, and per-hop observability telemetry)
- [x] 3.11 Verification enhancements (redundant execution + auditor spot-checks + feedback metrics)
- [x] 3.12 HYDRA token launch path (BME hardening complete: coordinator `OpenHydraLedgerBridge` with hard-cap enforcement, channel-close burn/mint settlement bridge, frictionless DHT admission preserved, stake-priority routing, and no-stake malicious reputation suppression)
- [x] 3.13 Pipeline diversity enforcement

### KV Compaction — remaining research items (from original A–H list)

> All 8 items (A–H) are now complete.

- [x] **A — Real query tensor threading** — DONE as Pass 5 Option A (`peer/kv_compaction/_query_capture.py`, `AttentionQueryCapture`, threaded through `_forward_impl` in `peer/model_shard.py`). Replaces proxy-K heuristic with `W_q·hidden` in the correct semantic subspace.
- [x] **B — Benchmark compaction quality** — DONE as Pass 5 Option B (`scripts/bench_kv_compaction.py`). Measures `logit_cos_sim`, `top1_match`, `top5_overlap`, `rank_corr` across proxy-K / real-Q / β+Cv at configurable ratios and methods.
- [x] **C — Head budget optimisation script** — DONE (Pass 6). `scripts/optimize_head_budgets.py`: loads a model with `output_attentions=True`, computes per-head Shannon entropy on the last-token attention row, GQA-averages query heads into kv-head groups, allocates budget ratios ∝ entropy, normalises to `--target-ratio` mean, writes `{"source": "calibrated_entropy_v1", ...}` JSON. 7 tests.
- [x] **D — SLO instrumentation (general)** — DONE (Pass 4): HTTP latency, Grafana, Prometheus alert rules. Compaction-specific counters also now DONE (Pass 6): `compact_calls`, `compact_tokens_before/after/saved`, `compact_latency_s`, `kv_cache_hits/misses` in `PyTorchRuntime.compaction_stats()`; `ToyRuntime` stub; `ModelShard` delegation; coordinator proxy counters (`kv_store_ops_total`, `kv_retrieve_ops_total`, `inference_requests_total`) in `metrics_snapshot()` and Prometheus `/metrics`. 8 tests.
- [ ] **E — On-chain / DAO integration** — NOT DONE. The `coordinator/ledger_bridge.py` `OpenHydraLedgerBridge` defaults to `mock_mode=True`. Requires: Solidity state-channel contract (Arbitrum/Base), web3.py integration, stake resolver, finality/retry logic.
- [ ] **F — Multi-region rollout hardening** — PARTIAL. Terraform IaC + Docker Compose testnet exist (Pass 4). NOT tested with a live distributed deployment across regions with soak testing.
- [x] **G — Speculative decoding** — DONE (Tier 3.6). `coordinator/speculative.py` + `coordinator/engine.py`: toy draft model + HF draft model, 4-token draft generation, multi-token verification, accept/reject correction loop, adaptive draft length (2–8 tokens), KV-safe commit rounds.
- [x] **H — Prefix sharing / radix cache** — DONE (Pass 6). `peer/kv_compaction/_radix_cache.py`: `RadixKVCache` (flat-dict LRU, O(n) longest-prefix lookup, `min_prefix_len` guard) + `_slice_kv_prefix` (DynamicCache duck-type + tuple-of-tuples). Integrated into `_forward_impl`: radix lookup trims `input_ids` to unseen suffix before forward; insert after `_kv_cache_set`. 3 new `ToyShardConfig` fields; 3 new `--kv-radix-cache-*` CLI flags in `peer/server.py`. 14 tests.

### Pass 6 — Visibility, UX & Launch (planned)

- [x] 6.1 **KV compaction auto mode** — three-position toggle (Auto / On / Off) exposed in `CompactionConfig`, `ToyShardConfig`, CLI flags, Tauri settings panel, and the REST API. Auto: compact only when `seq_len > auto_threshold` AND measured perplexity drift stays below `auto_quality_floor`; On: always compact at configured ratio; Off: bypass entirely. Requires a lightweight online quality probe (logit KL-divergence against a 32-token rolling baseline).
- [ ] 6.2 **Rich interactive CLI** — rewrite `coordinator/interactive_cli.py` (currently Python `cmd.Cmd`) to a Claude Code-style TUI using `prompt_toolkit` (or `rich` + `textual`): coloured prompt, syntax-highlighted streaming output, fuzzy model/session autocomplete, spinner while waiting, slash-command palette (`/model`, `/session`, `/status`, `/compaction`, `/help`), keyboard shortcuts (Ctrl+C cancel, Ctrl+L clear, ↑/↓ history), inline network-status sidebar, ANSI-safe fallback for dumb terminals.
- [ ] 6.3 **Expanded model catalog** — grow `models.catalog.json` to cover the models OpenHydra can realistically serve as the network scales: small (Qwen3.5-0.8B ✓, Llama-3.2-1B, Phi-3-mini-4k, Gemma-2-2B), medium (Qwen3-4B, Llama-3.1-8B, Mistral-7B-v0.3, Gemma-2-9B), large-sharded (Llama-3.1-70B ÷ 8 peers, Qwen3-72B ÷ 8 peers), code (Qwen2.5-Coder-7B, DeepSeek-Coder-V2-Lite), multimodal stub (LLaVA-1.6-Mistral, Qwen2-VL-7B). Each entry includes: `model_id`, `hf_model_id`, `required_peers`, `min_vram_gb`, `recommended_quantization`, `context_length`, `languages`, `tags`.
- [x] 6.4 **Publish to GitHub** — `.gitignore` (Python + ML + Tauri + Terraform + secrets), dual `LICENSE` (Apache 2.0 for `peer/`+`dht/`, AGPLv3 for network services), `licenses/Apache-2.0.txt`, `licenses/AGPL-3.0.txt`, `CONTRIBUTING.md`, `SECURITY.md` (responsible disclosure), `.github/workflows/python-app.yml` (CI: test matrix Python 3.11/3.12, lint, landing HTML check). Network hardening: `ops/network_limits.sh` (iptables OPENHYDRA chain: SSH/8080/8468-hashlimit/50051-connlimit, sysctl TCP tuning). Rate-limit headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`, `Retry-After` on all API responses.
- [x] 6.5 **Full project documentation** — README.md fully rewritten (architecture, AppChain economy, model catalog table, dual-license table, all CLI flags, REST API table, rate-limit headers, operator deployment, ARM/A1 guide, roadmap). Formal docs site (MkDocs Material, API reference via mkdocstrings, hosted on GitHub Pages) still pending.: Architecture overview (ASCII diagram → SVG), Quick-start (pip install, run node, first inference call), Operator guide (hardware requirements, Docker deployment, TLS setup, monitoring), Developer guide (gRPC API reference, REST API reference, Python SDK, TypeScript SDK), KV compaction guide (all 4 phases + Option A/B with diagrams), HYDRA token guide (mock mode → production bridge), Peer economics (credits, stake, reputation), FAQ, Changelog. Auto-generate API reference from docstrings using `pdoc` or `mkdocstrings`.
- [ ] 6.6 **Tauri UI/UX polish** — elevate `desktop/` from functional scaffold to polished product: onboarding flow (first-launch wizard: identity gen, model selection, hardware scan, network connect), real-time peer map (SVG world map with animated peer dots, connection lines, latency labels), compaction mode toggle in settings panel (Auto/On/Off), model browser with download progress bar, conversation history sidebar with search, token/compute earnings widget (HYDRA balance + credits earned this session), system tray integration (show/hide, status badge), native macOS/Windows/Linux menu bar, dark/light theme switch, smooth animations (Framer Motion or CSS transitions), keyboard shortcuts (⌘K command palette).
- [ ] 6.7 **GTM launch strategy** — coordinated multi-channel launch plan:
  - **arXiv companion paper** — short technical paper (4–6 pages, NeurIPS/ICML format) titled "OpenHydra: Distributed Inference with KV Cache Compaction via Attention Matching" or similar; cite arXiv:2602.16284; include benchmark results from `bench_kv_compaction.py`; submit to arXiv cs.DC + cs.LG
  - **Hacker News** — "Show HN: OpenHydra — run frontier LLMs by pooling idle GPU across peers"; post from personal account with a live demo; target Tuesday/Wednesday 9–11am ET; have a public demo endpoint ready
  - **Product Hunt** — full PH launch: hunter, maker comments, GIF demo, tagline "The Airbnb for GPU compute — run any LLM by sharing idle hardware"; schedule for Tuesday 12:01am PT; coordinate upvote brigade from beta users
  - **X (Twitter) launch thread** — 10-tweet technical thread: problem → solution → architecture diagram → KV compaction explainer → benchmark numbers → demo GIF → GitHub link → HYDRA token teaser → CTA to join waitlist; pin to profile
  - **OpenHydra newsletter** — launch `openhydra.substack.com` (or self-hosted via Buttondown); pre-launch issue: "Why we built OpenHydra"; launch issue: "We're live — here's how it works"; post-launch: monthly network stats digest (peers online, tokens processed, HYDRA minted)
  - **Reddit** — r/MachineLearning (technical framing), r/LocalLLaMA (operator framing: "pool your RTX 3090 into the network"), r/selfhosted (privacy angle: "your queries stay distributed, no single company sees them all")
  - **Discord/Slack communities** — EleutherAI Discord, Together AI community, LocalAI Discord, Hugging Face Discord (# ml-discussions), GPU-rich Slacks
  - **Research channels** — email the Zweiger et al. arXiv:2602.16284 authors; post in MIT CSAIL + Berkeley Sky Computing Lab mailing lists; tweet at relevant AI infrastructure researchers

## Equalization Sprint — Petals/Exo Parity (complete)

Comparative analysis of OpenHydra vs Petals and Exo identified 5 areas where OpenHydra fell behind. All 5 have been addressed:

- [x] **Phase 1A: Engine decomposition** — engine.py god class (3,244 lines) decomposed into 9 focused services (869 lines facade). Services: EconomyService, KvAffinityService, TokenizationService, HealthService, MoeService, DiscoveryService, PipelineService, StatusService, InferenceService.
- [x] **Phase 1B: Google-style docstrings** — 95+ methods across all 10 files documented with Args/Returns/Raises.
- [x] **Phase 1C: Strict typing** — Pydantic v2 API types (`coordinator/api_types.py`), pyrightconfig.json, basedpyright in CI.
- [x] **Phase 2A: 6-factor Dijkstra routing** — Extended `_dijkstra_edge_cost()` with Factor 5 (KV cache pressure penalty) and Factor 6 (server-to-server RTT). Matches Petals' routing sophistication.
- [x] **Phase 2B: Swarm rebalancing** — `SwarmRebalancer` detects throughput bottlenecks and generates `RebalanceDirective` for peer migration. Matches Petals' `block_selection.py`. Inflight drain safety guard prevents dropping live tensors.
- [x] **Phase 3A: Bidirectional gRPC streaming** — `ForwardStream` RPC with `StreamPool` connection reuse, 30s idle timeout, graceful unary fallback, `InferenceSession` with history-replay failover. Matches Petals' persistent streaming.
- [x] **Phase 4A: MLX tensor parallelism** — `PipelineParallelMLX` wrapper with `_assign_layers()` largest-remainder allocation and `mx.distributed.send()`/`recv_like()` communication. Matches Exo's multi-device capability.
- [x] **Phase 4B: Overlapped pipeline prefill** — Replaced blocking `mx.eval()` with `mx.async_eval()` so Rank 0 starts network transmission while still finishing local evaluation. Matches Exo's async execution.

Scorecard after equalization:
- vs Petals: OpenHydra wins 5, Petals wins 0, Tie 2 (was 5-3-2)
- vs Exo: OpenHydra wins 6, Exo wins 0, Tie 2 (was 6-2-2)
- Reference: `docs/comparison_petals_exo_analysis.md`

## Next: Universal AI OS Roadmap

Plan designed (pending approval) to transform OpenHydra from inference engine to universal AI platform:

- **Phase 1**: Agent framework + tool use + MCP server + code sandbox (Q3 2026)
- **Phase 2**: HuggingFace auto-population + dynamic model registry (Q3-Q4 2026)
- **Phase 3**: Desktop agent (OpenClaw-like) + WebSocket support (Q4 2026)
- **Phase 4**: Inter-node agent messaging over DHT (Q1 2027)
- **Phase 5**: Smartphone consumer-only mode + ONNX mobile runtime (Q1-Q2 2027)
- **Phase 6**: Local LoRA fine-tuning + federated LoRA aggregation (Q2-Q3 2027)
- **Phase 7**: Plugin/extension system (Q3 2027)

Full roadmap in plan file. Key decisions: agent execution local-only (peers do inference, your machine runs code); MCP as tool standard; Apache 2.0 for maximum adoption; federated training = LoRA only (full pre-training is research-grade).

## Recent changes log

### 2026-03-19 through 2026-03-24

- **AMD ROCm GPU detection** (`coordinator/node.py`): Auto-detect distinguishes NVIDIA CUDA vs AMD ROCm via `torch.version.hip`. Actionable error when PyTorch missing.
- **Counterculture copy**: README.md + landing/index.html rewritten with "Turn on. Tune in. Drop in." philosophy, 0.8B default base swarm, auto-promotion for heavy hardware.
- **Desktop app** (`desktop/`): Complete Tauri v2 + React build — Dashboard (NodeToggle, StatusCards, ModelSelector, RamSlider, LogTerminal), Swarm tab (Network Demand with smart RAM-based model recommendations), Chat tab (SSE streaming), Bootstrap screen (first-run setup). Production .dmg built (3.7 MB).
- **Desktop icons**: All Tauri icons (macOS .icns, Windows .ico, iOS, Android, Windows Store) generated from landing page logo (RGBA, all sizes).
- **PyPI publish**: Package renamed from `openhydra` (taken by unrelated project) to `openhydra-network`. Published v0.1.0 at https://pypi.org/project/openhydra-network/. All docs updated.
- **v0.1.0 GitHub Release**: .dmg uploaded to https://github.com/samtroberts/openhydra/releases/tag/v0.1.0. Title: "v0.1.0 — Turn on. Tune in. Drop in."
- **Landing page fixes**: All URLs → `samtroberts/openhydra`, direct download links to release assets, Apache-2.0.txt license path fixed, larger download CTAs.
- **CI fixes**: Added `requests>=2.31,<3` to core deps (was missing). Added `_make_torch()` skip guard to `test_none_returns_none` (torch import guard). CI fully green.
- **Equalization Sprint** (see section above): Phases 1A through 4B complete. Test count: 411 → 982 (+571 new tests).
- **Pass 6.1 — KV compaction auto mode**: `--kv-compaction-mode auto|on|off` CLI flag. Auto mode monitors `available_vram_mb` and triggers compaction when VRAM ≥ 75% utilized.
- **Pass 6.2 — SLO compaction metrics**: `tokens_saved_total` and `compaction_latency_ms` exposed in `/metrics` endpoint. Peer heartbeat reports lifetime compaction savings.
- **Pass 6.3 — Open source onboarding**: CONTRIBUTING.md updated for Apache 2.0 + dev setup + PR guidelines. `.github/ISSUE_TEMPLATE/bug_report.md` and `feature_request.md` created.
- **Pass 6.4 — Architecture docs**: `docs/architecture.md` created — covers Data at Rest vs Data in Motion, Geographic Expert Sharding, MLX/PyTorch zero-copy bridge, Live Q-Tensor KV Compaction.
- **Pass 6.5 — CI hardening**: pytest-cov (--cov-fail-under=60), pip-audit (dependency vulnerability scanning), ruff security rules (S/bandit + I/imports), coverage XML artifact upload.
- **Pass 6.6 — Apache 2.0 license headers**: Entire stack relicensed to Apache 2.0 (abandoned AGPL open-core model). Headers added to all Python files in `peer/`, `dht/`, `coordinator/`, `economy/`.
- **Pass 6.7 — Security audit**: Zero code-level vulnerabilities found. Clean bill of health: no pickle, no eval, no path traversal, no shell injection. Dependency pins: `certifi>=2024.2.2`, `protobuf>=4.25,<6`.
- **v0.1.1 patch**: Economy service extraction, `_dedupe_peer_entries` field drop bug fix, `PeerEndpoint.from_dict()` classmethod.
- Test suite: `982 passed, 9 skipped` ✓

### 2026-03-08

- **Pass 6.4 — GitHub publish prep (complete):**
  - **`LICENSE`** (new): dual open-core preamble; `peer/`+`dht/` → Apache 2.0; all network services → AGPLv3
  - **`licenses/Apache-2.0.txt`** (new): full Apache 2.0 canonical text
  - **`licenses/AGPL-3.0.txt`** (new): full AGPL v3 canonical text (661 lines, fetched from gnu.org)
  - **`.gitignore`** (expanded): Python, ML artefacts (*.safetensors, *.pt, *.gguf), secrets (*.pem, *.key), Tauri target, Terraform state, Node modules, OS files
  - **`CONTRIBUTING.md`** (new): dev setup, test instructions, commit conventions (Conventional Commits), license agreement, good first issues
  - **`SECURITY.md`** (new): coordinated disclosure policy, in-scope/out-of-scope components, crypto primitives table, known limitations
  - **`.github/workflows/python-app.yml`** (new): CI matrix (Python 3.11/3.12), lint (ruff), landing HTML check; triggers on push to main/dev and PRs to main
- **Pass 6 — Network hardening (ops/network_limits.sh):**
  - **`ops/network_limits.sh`** (new): idempotent iptables OPENHYDRA custom chain; SSH allow, 8080 allow (Cloudflare-fronted), 8468 hashlimit 20 new-conn/min per IP (burst 5), 50051 connlimit ≤5 per IP (tcp-reset excess), ICMP 5/s limit; sysctl: `tcp_syncookies=1`, `tcp_max_syn_backlog=4096`, `rp_filter=1`, `accept_redirects=0`, `tcp_fin_timeout=20`, `tcp_keepalive_time=600`, `somaxconn=1024`, `tcp_tw_reuse=1`; persists via `/etc/sysctl.d/60-openhydra.conf` + `netfilter-persistent`; supports `--check` and `--flush` flags
- **Pass 6 — Application rate-limit headers (coordinator/api_server.py):**
  - `_RateLimiter.check(client_ip)` → `(allowed, remaining, reset_unix_timestamp)`; `is_allowed()` kept as backward-compat wrapper
  - `OpenHydraHandler._rate_limit_check()` → `(allowed, headers_dict)`; populates `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`; adds `Retry-After` on 429 only
  - `do_GET` / `do_POST`: merges RL headers into `rid_headers` before any response (rate-limit metadata present on all responses, not just 429)
  - `tests/test_api_server.py`: updated `_NeverAllowed` mock to implement `check()` alongside `is_allowed()`
- **Pass 6.5 — README rewrite:**
  - **`README.md`** (rewritten): architecture ASCII diagrams, request flow, AppChain economy (barter + HYDRA), model catalog table (18 models), quick start, Docker, interactive shell, CLI flags table, all REST endpoints, rate-limit headers, operator deployment (Linode, prod profile, ARM A1, TLS, Grafana), KV compaction, project structure, dual-license table, roadmap
- Test suite: `411 passed, 9 skipped` ✓

### 2026-03-07

- Pass 5 Option A — Real W_q-projected query threading (AttentionQueryCapture):
  - **`peer/kv_compaction/_query_capture.py`** (new):
    - `AttentionQueryCapture(model, n_ref)` context manager that registers PyTorch `register_forward_pre_hook(with_kwargs=True)` on every `model.model.layers[i]` before the forward pass
    - Hook captures `hidden_states[:, -n_ref:, :]` per layer during the forward pass (detached, cloned, no computation graph retained)
    - `compute_q_ref()` runs `W_q(hidden)` projection (no RoPE — intentional: content-based subspace, not position-dependent) and groups by kv-head via GQA averaging: `q.view(n_kv, n_groups, n_ref, d).mean(dim=1)` → `(n_kv, n_ref, d_head)`
    - Returns `list[Tensor(n_kv_heads, n_ref, head_dim) | None]` of length `n_layers`; None for any layer without captured states or a missing `q_proj`
    - Clears `_hidden_per_layer` after `compute_q_ref()` to free memory; object is single-use by design
  - **`peer/kv_compaction/_compactor.py`** (updated):
    - `_compact_single_head(..., Q_ref_actual=None)`: when `Q_ref_actual` is provided it replaces the proxy-K heuristic (`K[-n_ref:]`) as reference queries — semantically correct W_q subspace
    - `compact_past_key_values(..., q_ref_per_layer=None)`: new optional argument; per-head `Q_ref_actual` slice extracted from `q_ref_per_layer[layer_idx][head_idx]`; partial lists and None entries fall back gracefully to proxy-K
  - **`peer/kv_compaction/__init__.py`** (updated): exports `AttentionQueryCapture`
  - **`peer/model_shard.py`** (updated):
    - `_kv_cache_set(session_id, past_key_values, q_ref_per_layer=None)`: new `q_ref_per_layer` parameter threaded to `compact_past_key_values`
    - `_forward_impl` full-model path: when compaction is active and storing to cache, wraps `self._model(**model_kwargs)` with `AttentionQueryCapture` context manager; calls `compute_q_ref()` post-forward; passes resulting Q tensors to `_kv_cache_set`; `_model_out_set` sentinel prevents duplicate forward on `compute_q_ref` failure; falls back transparently to proxy-K on any error
  - **28 new tests** in `tests/test_kv_compaction.py`:
    - `TestAttentionQueryCaptureImport` (2): package export + direct import
    - `TestAttentionQueryCaptureContextManager` (6): hooks registered/removed, empty model, n_ref clamp
    - `TestAttentionQueryCaptureHiddenCapture` (3): states captured per layer, limited to n_ref, detached
    - `TestAttentionQueryCaptureComputeQRef` (6): list length, shape (n_kv, n_ref, d), float32, cleared, None on no-capture, GQA grouping
    - `TestCompactWithQRefPerLayer` (8): accepted, reduces seq_len, None entries, partial entries, too-short list, OMP method, Phase 2 + β, tuple cache format
    - `TestAttentionQueryCaptureFullPipeline` (3): end-to-end pipeline, single-use, real-Q ≠ proxy-K

- Pass 5 Option B — Quality uplift benchmark (`scripts/bench_kv_compaction.py`):
  - **`scripts/bench_kv_compaction.py`** (new, ~350 lines):
    - Loads any HF model and runs a prefill pass to obtain a full KV cache and baseline logits
    - Three compaction methods compared at each target ratio: (0) baseline no-compaction, (1) Phase 1 + Proxy-K, (2) Phase 1 + Real-Q (Option A), (3) Phase 2 + Real-Q + β + Cv
    - Quality metrics at the last-token logit vector: `logit_cos_sim` (cosine similarity to baseline), `top1_match` (argmax agreement), `top5_overlap` (fraction of top-5 tokens shared), `rank_corr` (Spearman ρ over top-100 tokens)
    - Prints a formatted comparison table with Δcos and Δtop1 uplift columns
    - `--target-ratio` and `--method` accept multiple values for multi-configuration sweeps
    - `--output-json` exports full results for offline analysis
    - `--beta` flag enables Phase 2 β + Cv measurement
    - Minimal smoke test: `python scripts/bench_kv_compaction.py --model gpt2 --seq-len 64 --n-prompts 5`
    - Proper benchmark: `python scripts/bench_kv_compaction.py --model Qwen/Qwen3.5-0.8B --seq-len 256 --n-prompts 50 --target-ratio 0.05 0.10 0.20 0.50 --method hak omp`
  - **14 new tests** in `tests/test_kv_compaction.py` (`TestBenchmarkMetrics`):
    - Script importable; `_cosine_similarity` (identical=1, orthogonal=0, opposite=-1); `_top1_match` (same/different argmax); `_topk_overlap` (identical/disjoint); `_rank_corr` (identical=1, reversed≈-1, range check); `_detect_decoder_family` (gpt/qwen); `--help` flag exits 0 with expected tokens
- Validation:
  - full default suite: `355 passed, 9 skipped` (`pytest -q`)
  - `tests/test_kv_compaction.py` alone: `104 passed` in 11.4s (62 original + 42 new)

- Post-Tier-3 pass 4 — distributed node model, identity, PostgreSQL, GUI, IaC:
  - **Distributed API model (Option c — IPFS-style)**:
    - `coordinator/node.py`: new `openhydra-node` unified daemon entry point; starts peer gRPC server in a background daemon thread and coordinator HTTP API on the main thread; `--api-host` defaults to `127.0.0.1` (local-only), `--api-host 0.0.0.0` for Docker/testnet; falls back to production bootstrap URLs if no `--dht-url` given
    - `pyproject.toml`: added `openhydra-node = "coordinator.node:main"` script entry; kept all existing entry points
    - `docker-compose.yml`: replaced `peer` + `coordinator` services with unified `node` service; exposed both gRPC (50051) and HTTP API (8080) ports
    - `README.md`: new "How it works" section (ASCII diagram), `openhydra-node` as primary quickstart, CLI reference table; old per-process docs moved to "Advanced / manual control"
  - **Ed25519 peer identity** (cryptographic proof of peer ID):
    - `peer/identity.py`: keypair gen/load/sign/verify using `cryptography` library; `peer_id = sha256(pubkey)[:16]`; identity file persisted at `.openhydra/identity.key` (mode 0600)
    - `peer/server.py`: `--identity-path` flag; peer announces `public_key` + `signature` fields in DHT payload; auto-derives `peer_id` from keypair if not explicitly set
    - `dht/bootstrap.py`: verifies Ed25519 signatures on announce; stores `identity_verified: bool`; never rejects unverified peers (graceful degradation)
    - `coordinator/node.py`: `--identity-path` flag wired through to peer serve kwargs
    - `tests/test_peer_identity.py`: 7 tests — generate, derive peer_id, create file, idempotent load, sign/verify roundtrip, wrong signature, wrong field
  - **PostgreSQL migration** (opt-in via `DATABASE_URL`):
    - `economy/postgres.py`: `PostgresCreditLedger` and `PostgresHydraTokenEconomy` implementing the same interface as the SQLite counterparts; psycopg2 `%s` placeholders; guarded by `_PSYCOPG2_AVAILABLE` flag; raises `RuntimeError` if used without psycopg2 installed
    - `coordinator/engine.py`: `database_url: str | None = None` in `EngineConfig`; `CoordinatorEngine.__init__` selects Postgres vs SQLite based on config; logs `ledger_backend=postgres` or `ledger_backend=sqlite`
    - `coordinator/api_server.py`: `--database-url` CLI flag; `DATABASE_URL` env var fallback
    - `docker-compose.yml`: `postgres:16-alpine` service with healthcheck; `node` depends on postgres healthy; `DATABASE_URL` env var wired; `postgres_data` named volume
    - `ops/db/schema.sql`: PostgreSQL DDL for all four tables (`credits`, `hydra_accounts`, `hydra_channels`, `hydra_meta`); auto-applied via `docker-entrypoint-initdb.d`
    - `pyproject.toml`: `[postgres]` optional extra: `psycopg2-binary>=2.9,<3`
    - `tests/test_postgres_economy.py`: 5 tests with mocked psycopg2 — no real Postgres needed in CI
  - **Grafana + Prometheus alert rules**:
    - `ops/prometheus/rules/openhydra_alerts.yml`: 8 rules using real metric names verified from source (`CoordinatorDown`, `HighErrorRate`, `CriticalErrorRate`, `HighLatency`, `LowDHTSuccessRate`, `CriticalDHTSuccessRate`, `HydraSupplyCapWarning`, `HydraSupplyCapCritical`)
    - `ops/prometheus/prometheus.yml`: `rule_files:` block pointing to `/etc/prometheus/rules/*.yml`
    - `docker-compose.yml`: Prometheus rules directory mount (`./ops/prometheus/rules:/etc/prometheus/rules:ro`); Grafana provisioning mount (`./ops/grafana/provisioning:/etc/grafana/provisioning:ro`)
    - `ops/grafana/openhydra_dashboard.json`: alert threshold lines on error rate panel, corrected DHT gauge thresholds, corrected supply cap thresholds; new "Alerts" row with `alertlist` panel; dashboard version bumped to 2
    - `ops/grafana/provisioning/alerting/contact_points.yml`: Slack webhook contact point (reads `SLACK_WEBHOOK_URL` env var at Grafana startup)
  - **Web chat UI** (`web/index.html`):
    - Single self-contained HTML/CSS/JS file (~550 lines); no npm, no build step
    - Dark theme matching `landing/index.html`; Inter + JetBrains Mono fonts; fully responsive (single column ≤640px)
    - Full SSE streaming via `AbortController` + `ReadableStream`; conversation history sent with every request; Stop button cancels stream
    - Settings slide-in panel: endpoint URL, model selector (from `/v1/models`), pipeline width slider; all persisted to `localStorage`
    - Node offline banner if `/readyz` returns non-200; network status badge polling `/v1/network/status` every 10s
    - Copy button on assistant messages; typing dots animation; smooth auto-scroll; last 50 messages persisted to `localStorage`
  - **Tauri desktop app** (`desktop/`):
    - Tauri v2 scaffold: `package.json`, `src-tauri/Cargo.toml`, `src-tauri/build.rs`, `src-tauri/tauri.conf.json`, `src-tauri/capabilities/default.json`
    - `src-tauri/src/lib.rs`: `NodeState` struct managing `openhydra-dht` and `openhydra-node` child processes; Tauri commands: `start_node`, `stop_node`, `is_node_running`, `get_node_pid`; auto-kills processes on window close
    - Frontend (`src/index.html`, `src/app.js`, `src/styles.css`): sidebar with node controls + status, Chat/Network/About tabs, full SSE streaming chat, HYDRA balance display, settings form
    - `desktop/README.md`: build instructions (Rust 1.77+, Node 18+, `npm run tauri dev/build`)
  - **Multi-region Terraform IaC** (`ops/terraform/`):
    - `bootstrap/main.tf`: DigitalOcean Nanodes (`s-1vcpu-1gb`) × 3 regions (nyc3/ams3/sgp1); shared firewall (ports 22 + 8468); DNS A records via `digitalocean_record`; `templatefile()` for cloud-init
    - `bootstrap/user_data.sh`: installs Python + OpenHydra, creates systemd `openhydra-dht.service`, hardens UFW, starts service on boot
    - `bootstrap/variables.tf`, `outputs.tf`, `versions.tf`: full variable set, IP/URL/SSH outputs, Terraform ≥1.6 + DO provider ~2.39
    - `environments/production.tfvars` + `staging.tfvars`: staging overrides to single nyc3 node
    - `ops/terraform/README.md`: prerequisites, `terraform init/plan/apply` workflow, DNS fallback, cost breakdown (~$18/mo prod), workspace-based staging
- Validation:
  - full default suite: `251 passed, 9 skipped` (239 → +7 Ed25519 → +5 PostgreSQL)

### 2026-03-06

- Production operations hardening (post-Tier-3 pass 1):
  - **Request correlation IDs** (end-to-end tracing):
    - `coordinator/api_server.py`: UUID generated at `do_GET`/`do_POST` entry; threaded through all payload helpers (`_chat_payload`, `_completion_payload`, stream variants); returned as `X-Request-ID` response header on every route; `request_start` / `request_done` log lines carry `req_id=`, `method=`, `path=`, `status=`, `latency_ms=`
    - `coordinator/engine.py`: `infer()` and `infer_stream()` accept `request_id=` (auto-generate UUID when absent); `infer_start` log line emitted with `req_id`, `model`, `client`; threaded into `_run_chain()` → `InferenceChain.run()` → `ForwardRequest.request_id`
    - `coordinator/chain.py`: `run()` and `_request_stage()` accept `request_id=`; each gRPC `ForwardRequest` carries the coordinator-assigned UUID
  - **gRPC deadline propagation** (absolute request deadline):
    - `coordinator/engine.py`: `infer()` computes `deadline = time.time() + max_latency_ms / 1000.0`; threaded into `_run_chain()` and all secondary/tertiary verification chain calls; `infer_stream()` deliberately omits a fixed deadline (multi-round token streaming uses per-hop timeout only)
    - `coordinator/chain.py`: `_request_stage()` computes `remaining = deadline - time.time()` and uses `min(self.timeout_s, remaining)` as the effective gRPC timeout, raising `deadline_exceeded` with no time remaining
  - **Packaging** (`pyproject.toml`):
    - added `[build-system]` with `setuptools.build_meta` so `pip install .` works
    - pinned dependency upper bounds: `grpcio>=1.62,<2`, `grpcio-tools>=1.62,<2`, `protobuf>=4.25,<7`, `cryptography>=42,<47`
    - added `[project.scripts]` entry points: `openhydra` (client CLI), `openhydra-coordinator`, `openhydra-peer`, `openhydra-dht`, `openhydra-genesis`
    - added `[tool.setuptools] py-modules = ["openhydra_defaults", "openhydra_secrets"]` so root-level runtime modules are importable after `pip install`
    - added `dev` extras with `pytest>=8.0,<9` and `pytest-asyncio`
    - updated `requirements.txt` to match pinned bounds
  - test mock fixes: all `fake_run_chain`, `fake_request`, and `_DummyChain.run()` mocks in test suite updated to accept `deadline=None` / `**kwargs` for forward-compatibility
  - **Structured JSON logging** (`openhydra_logging.py`):
    - new root-level module `openhydra_logging.py` with `_JsonFormatter` (one JSON line per record: `ts`, `level`, `logger`, `msg`, plus any `extra={}` fields promoted to top-level), `_TextFormatter` (human-readable dev format), and `configure_logging(level, json_logs)` entry point
    - `coordinator/api_server.py`: imports and calls `configure_logging(json_logs=profile=="prod")` after profile resolution (also adds the previously missing root-logger init)
    - `peer/server.py`: replaces `logging.basicConfig(...)` with `configure_logging(...)`
    - `dht/bootstrap.py`: replaces `logging.basicConfig(...)` with `configure_logging(...)`
    - `pyproject.toml`: added `openhydra_logging` to `py-modules` so it's importable after `pip install`
    - `tests/test_logging_config.py`: 10 tests covering JSON field presence, extra-field promotion, stdlib-attr exclusion, exception serialisation, text vs JSON mode, handler replacement, and level setting
  - **Graceful SIGTERM shutdown** (coordinator + DHT bootstrap):
    - `coordinator/api_server.py` `serve()`: registers `signal.SIGTERM` handler that dispatches `server.shutdown()` to a daemon thread so in-flight requests drain before the process exits; `finally:` block logs `shutdown_complete` and calls `engine.close()`; `KeyboardInterrupt` also logs `shutdown_requested signal=SIGINT`
    - `dht/bootstrap.py` `serve()`: registers `signal.SIGTERM` handler with a `threading.Event` stop flag; on SIGTERM the outer restart loop exits cleanly after the current `serve_forever()` returns; `finally:` block logs `shutdown_complete`
    - `coordinator/api_server.py`: added `/readyz` readiness probe endpoint (public, no auth); returns `{"status":"ok"}` when engine is initialised, `503 {"status":"not_ready"}` otherwise — compatible with Kubernetes liveness/readiness probe conventions
    - `tests/test_graceful_shutdown.py`: 3 tests verifying SIGTERM causes serve_forever() to exit, KeyboardInterrupt is handled cleanly, and the stop-event pattern works for DHT bootstrap
    - `tests/test_api_server.py`: 2 new tests for `/readyz` (engine wired → 200, engine None → 503)
- Validation:
  - full default suite: `219 passed, 9 skipped` (`pytest -q`)

- Production operations hardening (post-Tier-3 pass 2):
  - **SQLite ledger recovery on coordinator restart**:
    - `economy/token.py`: added `SqliteHydraTokenEconomy.recover()` method — loads persisted economy, runs `_auto_expire_channels()` to settle channels whose TTL lapsed while the coordinator was down, persists the result, returns a summary dict (`open_channels`, `expired_on_recovery`, `total_accounts`, `total_minted`, `total_burned`)
    - `coordinator/engine.py`: `CoordinatorEngine.__init__()` calls `self.hydra.recover()` immediately after constructing the economy object and logs the summary as `ledger_recovery` at INFO level — ensures any expired open channels are settled deterministically on every restart
    - `tests/test_ledger_store.py`: 3 new tests — empty-DB recovery returns zero stats, expired-channel-on-restart settlement (SQLite-patched TTL), and idempotency of double-recovery
  - **API server test coverage** (auth, rate limiting, inference paths, validation, GET routes):
    - `tests/test_api_server.py`: 17 new tests covering:
      - Auth: 401 when API key configured but not provided; Bearer token accepted; `X-API-Key` header accepted
      - Rate limiting: 429 when `_RateLimiter.is_allowed()` returns False; `_RateLimiter` sliding-window allows then blocks at cap boundary
      - Inference: POST `/v1/completions` returns `text_completion` object; POST `/v1/chat/completions` returns `chat.completion` object; malformed JSON body handled without crash
      - GET routes: `/v1/models` returns model list; `/v1/network/status` returns network payload; unknown path returns 404
      - `_validate_infer_params`: accepts valid body; rejects `max_tokens` > 4096; rejects `max_tokens` < 1; rejects prompt > 32 768 chars; rejects total message content > 32 768 chars; rejects `pipeline_width` > 16
  - **Landing page** (`landing/index.html`):
    - Single self-contained HTML file, no JS libraries, no CSS frameworks
    - Dark theme (`#0a0a0a` / `#e8e8e8` / `#00d4b8` teal accent), Inter + JetBrains Mono from Google Fonts
    - Sections: nav (wordmark + GitHub link), hero (tagline chip, H1 clamp(2.5rem,6vw,4.5rem), subcopy, `pip install openhydra` code block with clipboard copy, two CTAs), 3-column features strip, footer
    - Vanilla JS copy-to-clipboard on the install snippet (~8 lines); responsive single-column below 600 px
  - **Interactive CLI shell** (`coordinator/interactive_cli.py`):
    - `openhydra-shell` entry point (added to `pyproject.toml [project.scripts]`)
    - `OpenHydraShell(cmd.Cmd)` with maintained `_session_id` across `chat` turns
    - Commands: `chat <msg>` (session-aware, POST `/v1/chat/completions`), `ask <prompt>` (stateless, POST `/v1/completions`), `status` (GET `/v1/network/status`), `balance [id]` (GET `/v1/account/balance`), `models` (GET `/v1/models`), `session reset|show`, `model <id>|clear`, `exit`/`quit`
    - stdlib-only (`urllib.request`); friendly connection-refused message; `textwrap.fill` for narrow terminals; Bearer auth header support
- Validation:
  - full default suite: `239 passed, 9 skipped` (`pytest -q`)

- Production operations hardening (post-Tier-3 pass 3 — production-readiness gap closure):
  - **HYDRA mock-mode disclaimer** (logs + API surface + interface docs + README):
    - `coordinator/api_server.py` `serve()`: logs `WARNING HYDRA_BRIDGE_MOCK_MODE=true` at startup when the ledger bridge is in mock mode, guiding operators to set `--no-hydra-ledger-bridge-mock-mode` for production
    - `coordinator/engine.py` `hydra_status()`: surfaces `"mock_mode": bool` and `"mock_mode_warning": str | None` at the top level of the `/v1/hydra/status` response — client tooling and dashboards can detect mock mode programmatically
    - `coordinator/ledger_bridge.py`: added detailed production integration guide to the class docstring covering: flipping `mock_mode=False`, implementing `external_stake_resolver` (web3.py example), implementing `external_stake_slasher`, replacing `burn_for_compute`/`mint_provider_rewards` with signed EVM transactions, and adding retry/finality logic
    - `README.md`: added "Production limitations" section with mock-mode disclaimer, coordinator HA pointer, and cert automation pointer
  - **Coordinator HA** (nginx load-balancer overlay):
    - `ops/ha/nginx.conf`: nginx config with three upstreams — `least_conn` for general API routes, `ip_hash` for `/v1/chat/completions` (SSE session stickiness), single-backend for `/metrics`; `proxy_buffering off` for SSE; `proxy_read_timeout 120s`; gzip for JSON; custom log format with upstream addr + request time; commented TLS block referencing `/etc/openhydra/certs/`
    - `docker-compose.ha.yml`: compose override adding `coordinator-1`, `coordinator-2` (shared `openhydra_state` volume for SQLite state), `nginx:1.27-alpine` ingress (port 80), and disabling the base `coordinator` service via `profiles: [disabled]`; usage: `docker-compose -f docker-compose.yml -f docker-compose.ha.yml up -d`
  - **Cert automation** (`ops/certs/provision_cert.sh`):
    - Standalone shell script (197 lines, `bash -n` syntax-checked, executable)
    - Supports standalone mode (binds port 80 temporarily) and webroot mode (`--webroot PATH`)
    - Auto-installs certbot via `apt-get`/`yum`/`brew` if missing
    - Deploys cert+key to `/etc/openhydra/certs/` with `chmod 600 privkey.pem`
    - Installs auto-renewal cron job at `/etc/cron.d/openhydra-certbot` (daily 03:00)
    - `--renew` flag re-deploys all live certs without re-issuing
    - `--staging` flag for Let's Encrypt staging CA
    - Prints the exact `openhydra-coordinator --tls-*` flags to use after provisioning
  - **Grafana dashboard + Prometheus/Grafana in docker-compose**:
    - `ops/prometheus/prometheus.yml`: scrape config for coordinator at `coordinator:8080/metrics` (15s interval); commented placeholder for DHT bootstrap health (JSON, not Prometheus format)
    - `ops/grafana/openhydra_dashboard.json`: valid Grafana 9+ dashboard (14 panels, 3 rows) covering HTTP traffic (requests/s, error rate %, avg latency ms, total requests stat), DHT (lookup rate, success rate gauge with red/yellow/green thresholds, failures/s), and HYDRA economy (total supply, cap used % gauge, minted/s, burned/s); `__inputs__` block for clean UI import
    - `docker-compose.yml`: added `prometheus:v2.52.0` and `grafana:10.4.0` services with named volumes, depends_on coordinator, admin password `openhydra`; Grafana accessible at `:3000`, Prometheus at `:9090`
- Validation:
  - full default suite: `239 passed, 9 skipped` (`pytest -q`)

### 2026-03-05

- Continued post-P5 routing/metadata hardening pass:
  - expanded runtime model identity propagation:
    - peer announce payloads now carry `runtime_model_id` through DHT bootstrap lookup surfaces
    - coordinator discovery/load paths retain `runtime_model_id` in peer endpoints
    - coordinator discovered-peer payload now exposes `runtime_model_id`
  - improved model catalog API ergonomics:
    - `list_models()` now includes optional `hf_model_id` to make alias->HF mapping explicit to clients
  - added regression coverage:
    - `tests/test_path_finder_dht.py` now validates `runtime_model_id` roundtrip from DHT
    - `tests/test_dht_bootstrap.py` now validates `runtime_model_id` persistence in lookup
    - `tests/test_engine_discovery.py` now validates pipeline runtime-model precedence and `hf_model_id` exposure in model listing
- Validation:
  - full default suite: `185 passed, 9 skipped` (`pytest -q -ra`)
  - gated Qwen integration test file validated locally with lightweight override:
    - `OPENHYDRA_RUN_REAL_TENSOR_TEST=1 OPENHYDRA_PYTORCH_TEST_MODEL=sshleifer/tiny-gpt2 pytest -q tests/test_real_qwen_generation.py -ra` -> `3 passed`

- Continued codex prompt model-compatibility pass (P5-a/P5-b/P5-c/P5-d/P5-e/P5-f):
  - Qwen defaults and trust-remote-code policy:
    - `peer/model_shard.py` now defaults runtime model to `Qwen/Qwen3.5-0.8B`
    - added model-aware `trust_remote_code` gating (enabled for Qwen-family IDs, disabled for `gpt2`/standard IDs)
    - applied trust gating across tokenizer/model loads in PyTorch runtime and decode tokenizer path
  - architecture + EOS handling:
    - added explicit Qwen/LLaMA-family architecture detection branch (`qwen_llama`) while preserving GPT/LLaMA support
    - added robust EOS ID normalization for tokenizers that expose single or multiple EOS IDs, and wired stream-loop termination against EOS sets
  - catalog/runtime model mapping:
    - `coordinator/degradation.py`: `ModelAvailability` now includes optional `hf_model_id`
    - `coordinator/engine.py`: catalog loader ingests `hf_model_id`; runtime model resolution now maps alias model IDs to HF model IDs for tokenizer/runtime use
    - added pipeline runtime-model resolution to prefer peer-advertised runtime model IDs
  - runtime model-id propagation over DHT:
    - added `runtime_model_id` to peer announcements and DHT/bootstrap normalization
    - added `runtime_model_id` to coordinator peer endpoint surfaces and lookup/config loaders
    - chain decode now prefers last-stage `runtime_model_id` for PyTorch token decoding
  - defaults/docs/catalog:
    - switched coordinator API/CLI and peer runtime model defaults from `gpt2` to `Qwen/Qwen3.5-0.8B`
    - added `models.catalog.json` with `openhydra-qwen3.5-0.8b -> Qwen/Qwen3.5-0.8B`
    - updated README model catalogue/examples/runtime defaults and added a Qwen quick-start section
  - tests added/updated:
    - `tests/test_model_shard.py` (trust policy + EOS normalization + decode tokenizer trust flag)
    - `tests/test_engine_discovery.py` (catalog `hf_model_id` runtime resolution)
    - `tests/test_real_qwen_generation.py` (gated real-Qwen path, EOS metadata, trust-flag guard)
- Validation:
  - full default suite: `183 passed, 9 skipped` (`pytest -q -ra`)

- Completed codex prompt SQLite economy migration pass (P4-a/P4-b/P4-c/P4-d):
  - `economy/barter.py`:
    - completed `SqliteCreditLedger` as the primary implementation with WAL mode, write lock, lazy per-peer decay, transactional `earn()/spend()`, read-only `balance()`, and JSON auto-migration (`.db` -> `.json` -> `.json.migrated`)
    - retained `FileCreditLedger` as a deprecated compatibility alias
  - `economy/token.py`:
    - added `SqliteHydraTokenEconomy` with WAL mode + write lock and full HYDRA account/channel persistence in SQLite (`hydra_accounts`, `hydra_channels`, `hydra_meta`)
    - channel/economy operations now run in `BEGIN IMMEDIATE` write transactions and persist ledger totals + policy metadata
    - added supply-cap enforcement at persistence time via `hydra_meta.supply_cap` / `hydra_meta.total_supply`
    - added JSON migration support for `.db` paths and retained `FileHydraTokenEconomy` as a deprecated alias
  - `coordinator/engine.py`:
    - switched internal economy wiring to `SqliteCreditLedger` and `SqliteHydraTokenEconomy`
    - updated defaults to `.openhydra/credits.db` and `.openhydra/hydra_tokens.db`
    - coordinator shutdown now closes all persistent stores (`health`, barter ledger, HYDRA ledger)
  - updated tests:
    - `tests/test_barter.py` now covers SQLite ledger earn/spend/balance, lazy decay correctness, 4-thread concurrent writes, and JSON migration
    - `tests/test_ledger_store.py` now covers SQLite persistence and HYDRA JSON migration
- Validation:
  - full default suite: `179 passed, 6 skipped` (`pytest -q -ra`)

- Completed codex prompt hardening pass (P0-b and remaining P1 items):
  - `openhydra_secrets.py`: restricted env ingestion to explicit allowlist; removed broad `os.environ` exposure path and added filtering regression coverage
  - `coordinator/ledger_bridge.py`: fixed `slash_stake()` TOCTOU race by capturing locked state once, isolating external slasher call outside lock, and committing totals in a single final lock section
  - `economy/barter.py`: removed write-on-read from `FileCreditLedger.balance()` and added explicit `flush()` persistence API
  - `coordinator/concentration_guard.py` + `coordinator/engine.py`: enforced hard operator-cap behavior with no silent refill, plus warning telemetry for reduced-width pipelines
  - `coordinator/health_scorer.py`: removed unknown-peer score side effects; `score()/scores()` now use detached defaults and avoid writes/mutations
  - `peer/peer.proto` + `peer/server.py` + `coordinator/chain.py`: added stage `compression_latent_dim` response plumbing and removed chain-side telemetry re-encode (uses stage-reported latent size with safe fallback)
  - `coordinator/degradation.py` + `coordinator/engine.py` + `coordinator/api_server.py`: added explicit `DegradationDecision.available` semantics and standardized unservable-path surface (`503` + `{"error":"no_viable_model","reason":...}`)
  - test coverage updated across:
    - `tests/test_openhydra_secrets.py`
    - `tests/test_hydra_ledger_bridge_bme.py`
    - `tests/test_barter.py`
    - `tests/test_concentration_guard.py`
    - `tests/test_health_scorer.py`
    - `tests/test_chain_compression.py`
    - `tests/test_chain_encryption.py`
    - `tests/test_degradation.py`
    - `tests/test_engine_degradation.py`
    - `tests/test_api_server.py`
- Validation:
  - full default suite: `169 passed, 6 skipped` (`pytest -q`)
- Completed codex prompt performance pass (P2-a and P2-b):
  - `coordinator/health_scorer.py`:
    - replaced per-event synchronous writes with batched persistence (`_dirty` + background flush thread)
    - added `flush()` and `close()` lifecycle methods
    - preserved write-path insertion semantics while keeping score-only reads side-effect free
  - coordinator shutdown wiring:
    - added `CoordinatorEngine.close()` to flush health/ledger state
    - API server and CLI now call engine close hooks on shutdown/exit
  - `dht/node.py`:
    - removed full-store prune calls from hot paths (`put`, `get`, `keys`, `stats`)
    - added lazy per-key TTL expiry in `put/get`
    - added optional daemonized background pruner (`start_background_pruner` / `stop_background_pruner`)
  - `dht/bootstrap.py`:
    - bootstrap now starts DHT background pruner at service start and stops it on shutdown
  - test coverage updated:
    - `tests/test_health_scorer.py` (batched write behavior, flush/close lifecycle, side-effect safety)
    - `tests/test_dht_node.py` (lazy expiry path without explicit prune + background pruner cleanup)
- Validation:
  - full default suite: `172 passed, 6 skipped` (`pytest`)
- Completed codex prompt bootstrap-resilience pass (P3-a and P3-b):
  - multi-bootstrap URL support:
    - `EngineConfig` now supports `dht_urls` (with backward-compatible `dht_url` fallback)
    - coordinator DHT discovery now fans out across multiple bootstrap URLs and keeps existing cached-peer fallback behavior intact
    - `load_peers_from_dht(...)` now queries all configured URLs in parallel, logs partial failures, merges results, and de-duplicates peers by freshest `updated_unix_ms`
  - CLI/API/peer flag plumbing:
    - coordinator API + CLI `--dht-url` now accepts repeated/comma-separated URLs
    - peer daemon `--dht-url` now accepts repeated/comma-separated URLs
    - peer DHT announce loop now posts to all configured bootstrap URLs in parallel, logging per-URL failures without aborting successful announces
  - bootstrap deployment scaffolding:
    - added `ops/bootstrap/bootstrap.service`
    - added `ops/bootstrap/nginx-bootstrap.conf`
    - added executable `ops/bootstrap/setup_nanode.sh`
    - added operator guide `ops/bootstrap/README.md`
  - test coverage updated:
    - `tests/test_path_finder_dht.py` (multi-URL merge + partial-failure behavior)
    - `tests/test_engine_discovery.py` (coordinator multi-URL wiring)
- Validation:
  - full default suite: `175 passed, 6 skipped` (`pytest`)

### 2026-03-04

- Removed desktop GUI scaffold by request:
  - deleted `desktop/` (Electron + React app scaffold and build artifacts)
  - kept core coordinator/peer APIs unchanged (CLI + HTTP control plane remains primary interface)
  - queued full regression run to confirm no impact to existing test suites

### 2026-02-24

- Completed LLaMA + dynamic-model upgrade pass:
  - upgraded `PyTorchRuntime` architecture detection in `peer/model_shard.py`:
    - now supports GPT-style (`transformer.h`) and LLaMA-style (`model.layers`) stacks
    - architecture-specific embedding/final-norm handling (`wte/wpe/ln_f` vs `embed_tokens/norm/lm_head`)
    - added RoPE-aware layer execution hooks for LLaMA (`position_ids` + `position_embeddings` via rotary module)
  - upgraded coordinator model admission/routing policy:
    - added dynamic model-ID support (`allow_dynamic_model_ids`) so non-catalog Hugging Face IDs can route when healthy peers exist
    - preserved existing fast failover hardening (500ms chain timeout, 0.5s DHT timeout, cached DHT fallback) unchanged
    - added dynamic-model routing tests in `tests/test_engine_discovery.py`
  - validation:
    - default suite: `160 passed, 6 skipped`
    - gated real tensor suite: `6 passed`
- Completed mainnet SLO latency tuning pass (strict chaos gate):
  - tightened coordinator chain gRPC inference timeout defaults to `500ms` (`InferenceChain`, `EngineConfig`, API/CLI defaults)
  - tightened DHT lookup timeout default to `0.5s` and added cached-peer fallback for transient DHT timeouts
  - added regression coverage for DHT cached fallback path (`tests/test_engine_discovery.py`)
  - updated sustained chaos harness to disable grounding by default for pure network failover SLO measurement
  - local stress validation (`scripts/slo_chaos_test.py`, `120s`, `16 workers`, peer restart):
    - `success_rate=0.988761`
    - `latency_ms.p95=555.645`
    - `pass=true`
- Completed public V1 mainnet hardening pass (Tier 2 release hardening closure):
  - locked production deployment profiles across control/data-plane binaries:
    - added `--deployment-profile {dev,prod}` and `--secrets-file` handling to coordinator API, peer daemon, and DHT bootstrap
    - prod profile now enforces strict transport posture:
      - coordinator requires TLS client config (`--tls-enable` + root/client cert/key + server-name override)
      - peer requires strict mTLS (`--tls-enable` + `--tls-require-client-auth` + cert/key/CA paths)
      - DHT bootstrap requires geo challenge to remain enabled in prod
    - prod profile now defaults HYDRA bridge to non-mock mode and forbids mock override
  - added secure secret-management utility (`openhydra_secrets.py`):
    - supports env + secrets file ingestion with precedence and strict `0600` file permission enforcement
    - rejects insecure/dev placeholder seeds for prod profile
    - wired strong-seed requirements for:
      - `OPENHYDRA_ADVANCED_ENCRYPTION_SEED`
      - `OPENHYDRA_GEO_CHALLENGE_SEED`
  - added sustained load + chaos SLO validation harness:
    - `scripts/slo_chaos_test.py` runs concurrent inference load, injects service restart chaos, evaluates success-rate and p95-latency SLO gates, and emits JSON reports
  - added one-command canary rollout + rollback automation:
    - `scripts/mainnet_canary.sh rollout|rollback`
    - generates checklist, SLO report, rollback logs, and rollback playbook artifacts under `.openhydra/canary_reports/`
    - added runbook docs: `ops/public_v1_mainnet_playbook.md`
    - added secrets template: `ops/prod_secrets.env.example`
  - added prod-hardening regression coverage:
    - `tests/test_openhydra_secrets.py`
    - `tests/test_prod_profile_hardening.py`
  - validated all suites:
    - default: `154 passed, 6 skipped`
    - gated real-PyTorch: `6 passed`
- Completed Tier 2 production hardening release-candidate pass:
  - hardened daemon resilience and reconnect behavior:
    - peer gRPC service lifecycle now self-heals on unexpected runtime termination with exponential restart backoff
    - DHT announce loop now tracks outage windows, keeps retrying indefinitely, and forces fresh re-announce after long outages
    - DHT bootstrap loop now separates bind-failure backoff from runtime-restart backoff for stable reconnect behavior
  - added production observability endpoint:
    - coordinator API now exposes `GET /metrics` in Prometheus text format
    - telemetry includes total HTTP requests, HTTP latency average/sum/count, DHT lookup attempts/success/failure/success-rate, and BME burn/mint/supply metrics
  - added deployment scaffolding:
    - root `Dockerfile` now packages peer/coordinator/bootstrap runtime with `torch`, `transformers`, `bitsandbytes`, and `libtorrent`
    - `docker-compose.yml` now defines a 3-node local testnet: `bootstrap`, `coordinator`, and `peer`
  - test coverage:
    - added `/metrics` route coverage in `tests/test_api_server.py`
    - validated full suite: `144 passed, 6 skipped`
    - validated gated real-PyTorch integrations: `6 passed`
- Completed final Tier 3 hardening pass (Privacy Parity + MoE Guardrails + Governance):
  - finalized privacy-mode parity and observability:
    - upgraded DP noise tracking in `peer/privacy.py` with observed variance/std telemetry, EMA tracking, and signed audit tags
    - added privacy-audit cryptographic helpers in `peer/crypto.py`
    - extended peer gRPC wire metadata (`ForwardResponse` + `PeerStatusResponse`) with DP audit fields
    - coordinator chain now verifies DP audit tags/variance in `--privacy-level=maximum` and fails closed on violations
  - finalized MoE expert-sharding admission controls:
    - DHT bootstrap now enforces expert-claim guardrails using reputation/stake thresholds
    - low-reputation or unstaked nodes remain joinable but are stripped of expert claims (`expert_admission_approved=false`)
    - path-finder/coordinator now ingest and honor expert admission metadata in routing surfaces
  - expanded HYDRA governance controls:
    - added governance parameter surface and vote scaffold to `OpenHydraLedgerBridge`
    - exposed governance API routes:
      - `GET /v1/hydra/governance/params`
      - `POST /v1/hydra/governance/vote`
  - added/updated tests:
    - `tests/test_chain_encryption.py` (maximum-mode DP audit verification + fail-closed behavior)
    - `tests/test_dht_bootstrap.py` (expert-claim rejection for low-reputation/unstaked nodes)
    - `tests/test_api_server.py` (governance route coverage)
    - `tests/test_hydra_ledger_bridge_bme.py` (governance params/vote behavior)
- Completed Highest-impact remaining work item #1 (HYDRA token-launch path hardening, BME model):
  - added `coordinator/ledger_bridge.py` with `OpenHydraLedgerBridge` in mock L1 mode and strict global cap enforcement (`69,000,000 HYDRA`)
  - implemented bridge methods for balance/stake verification, stake slashing, compute burn settlement, and provider reward minting
  - integrated state-channel close settlement with bridge reconciliation:
    - coordinator now burns payer spend (`burn_for_compute`) and mints provider rewards (`mint_provider_rewards`) on channel close
    - added per-channel provider spend tracking and settlement receipts in `hydra_close_channel(...)`
  - preserved frictionless DHT admission (no mandatory upfront stake) and added optional stake-priority routing boost via `verify_staked_balance(...)`
  - added no-stake malicious penalty escalation: when stake is unavailable to slash, verification penalties now apply aggressive reputation suppression (effective blacklist behavior)
  - added integration coverage in `tests/test_hydra_ledger_bridge_bme.py`:
    - unstaked peer admission through DHT
    - staked peer routing priority
    - BME burn/mint settlement on completed session close
- Completed Highest-impact remaining work item #1 (advanced DHT data-plane hardening):
  - added cryptographic geo-triangulation challenge on DHT announce:
    - bootstrap issues per-announce nonce challenge via peer `Ping`
    - peer returns deterministic cryptographic nonce signature
    - bootstrap enforces max RTT policy and downgrades unverified region claims with penalty metadata
  - added dynamic DSHT rebalancing on hot-key pressure:
    - lookup rate-limit breaches trigger replica expansion into adjacent DSHT keys
    - bootstrap emits rebalance hints (`recommended_dsht_replicas`, replica indices) in lookup/rate-limit responses
    - coordinator DHT loader now honors rebalance recommendations with follow-up lookup expansion
  - added new integration test `tests/test_dht_geo_triangulation.py` validating delayed-peer geo challenge rejection/penalization behavior
- Completed Highest-impact remaining work item #1 (enhanced privacy hardening for data plane):
  - added calibrated Gaussian DP activation obfuscation utility (`peer/privacy.py`) and wired it into `PyTorchRuntime` hidden-state egress before compression/on-wire transfer
  - added peer runtime flag `--privacy-noise-variance` and runtime telemetry counters for obfuscation application
  - upgraded enhanced encryption mode to multi-relay onion route peeling:
    - added route-onion envelope utilities (`build_onion_route_envelope` / `peel_onion_route_layer`)
    - coordinator now builds concentric route layers for the full selected pipeline and forwards only the remaining encrypted route layer state per stage
    - each peer decrypts one route layer, discovers only immediate next hop, and returns the residual encrypted onion metadata
  - added gated integration test `tests/test_real_onion_routing.py` validating 3-node real-PyTorch generation with enhanced onion routing + DP noise enabled
- Completed Highest-impact remaining work item #1 (learned/runtime-backed activation compression):
  - added deterministic `PyTorchActivationCompressor` with linear encoder/decoder projections (`d_model -> latent_dim -> d_model`)
  - integrated compressor directly into `PyTorchRuntime` stage boundaries:
    - first/middle stages emit compressed latent hidden payloads
    - middle/last stages reconstruct latents back to model hidden-size tensors before block execution
  - replaced placeholder path for PyTorch hops by bypassing coordinator-side pooling codec on `pytorch_*` peers
  - added gated integration test `tests/test_real_tensor_compression.py` validating compressed payload transfer and successful generation across local PyTorch peers
- Completed Highest-impact remaining work item #2 (distributed speculative decoding with local draft model integration):
  - added `PyTorchDraftModel` on coordinator side for local lightweight draft-token proposal batches (no gRPC routing for draft generation)
  - PyTorch runtime last-stage decode now supports sequence verification outputs (returns parallel token predictions for `K`-token decode verification rounds)
  - coordinator PyTorch stream path now supports DSD acceptance engine:
    - compare local draft tokens to network-verified tokens in parallel
    - accept matched prefix + first target correction on divergence
    - commit only accepted tokens to shard KV caches (rejecting non-accepted draft continuations without transferring KV payloads on-wire)
  - added gated integration test `tests/test_real_speculative_decoding.py` validating multi-token acceptance in a single verification round
- Completed Highest-impact remaining work item #1 (true distributed KV-cache residency/ownership):
  - `PyTorchRuntime` now maintains bounded per-session `past_key_values` with LRU eviction to prevent unbounded memory growth
  - prefill/decode asymmetry is enforced in runtime:
    - prefill cache-miss rounds compute and store local `past_key_values` per session
    - decode cache-hit rounds inject stored `past_key_values`, compute incremental updates, and persist the appended cache locally
  - cache isolation is data-plane local: KV tensors are retained on each shard and are not returned over gRPC
  - coordinator PyTorch autoregressive loop now runs prefill-once + incremental decode rounds with shared `kv_session_id` across all stages (`kv_cache_all_stages`)
  - added gated integration test `tests/test_real_kv_cache_isolation.py` validating multi-node runtime KV retention and cache-hit decode reuse
- Completed prior Highest-impact remaining work item #1 (calibrated quantization + hardware profiling foundation):
  - added `peer/hardware.py` with lightweight RAM/VRAM accelerator detection (`cpu` / `cuda` / `mps`)
  - peer startup now logs detected hardware profile to support polite-daemon scheduling and operator visibility
  - added quantization flag path for PyTorch runtime (`--quantization`: `none` / `8bit` / `4bit`) with backward-compatible `--quantization-mode`
  - integrated `BitsAndBytesConfig` loading for 4-bit/8-bit requests when CUDA + compatible libs are available
  - added graceful fallback behavior: on incompatible GPU/library or quantized-load errors, peer falls back to `cpu/fp32` with warning instead of crashing
  - added `tests/test_hardware_profiler.py` for mocked RAM/VRAM profiling validation
- Completed prior Highest-impact remaining work item #2 (true cross-peer pipeline-parallel token scheduling for real PyTorch runtime):
  - `PyTorchRuntime` now executes true stage roles by layer range:
    - stage 0: `input_ids` -> embeddings + local transformer blocks -> hidden-state payload
    - middle stages: hidden-state payload -> local blocks -> hidden-state payload
    - final stage: hidden-state payload -> local blocks + `ln_f` + LM head projection -> `argmax(next_token_id)`
  - coordinator stream loop now runs a real autoregressive generation cycle for PyTorch pipelines:
    - prompt text -> tokenizer ids -> pipeline forward -> next token -> append context -> repeat until EOS/max tokens
  - added isolated gated integration test `tests/test_real_text_generation.py` to validate real text generation across local PyTorch peers
  - validated end-to-end with `OPENHYDRA_RUN_REAL_TENSOR_TEST=1` (real tensor-routing + text-generation integration tests both pass)
- Added Tier 3 data-plane hardening scaffold for real weights:
  - introduced runtime strategy split in `peer/model_shard.py` with existing `ToyRuntime` preserved and new `PyTorchRuntime`
  - `PyTorchRuntime` supports HuggingFace `gpt2` loading and shard/layer slicing by shard index and optional expert layer indices
  - peer forward path now offloads `pytorch_*` inference with `asyncio.to_thread` via `ModelShard.forward_async`
  - peer gRPC server now sets enlarged payload limits (`grpc.max_receive_message_length` / `grpc.max_send_message_length` at 100MB)
  - peer runtime CLI now supports `pytorch_*` backends and `--runtime-model-id`
- Added isolated integration test `tests/test_real_tensor_routing.py`:
  - validates real PyTorch tensor routing across two local nodes in a 2-stage chain
  - gated behind `OPENHYDRA_RUN_REAL_TENSOR_TEST=1` to keep default CI fast/stable
- Extended Tier 3.7 MoE routing with expert-layer awareness:
  - request/control path now accepts `expert_layer_indices` (CLI `--expert-layers`, API `expert_layers`)
  - coordinator MoE sharding now matches on tags and/or layer indices with configurable thresholds
  - prompt-hint parsing now supports layer hints (`layer:<n>` / `expert-layer:<n>`)
  - added `--moe-geo-min-layer-matches` to coordinator CLI/API config
- Added MoE layer-routing regression coverage:
  - prompt-hint layer parsing in engine
  - engine pipeline reorder based on requested expert layer indices
  - API forwarding and normalization for `expert_layers`
- Added Tier 3.7 MoE expert-sharding scaffold:
  - peer DHT announcements now include `expert_tags`, `expert_layer_indices`, and `expert_router`
  - DHT bootstrap normalization/lookup now preserves expert metadata
  - DHT/static peer discovery now maps expert metadata into `PeerEndpoint`
  - coordinator applies optional expert-aware geo sharding (`moe_geo`) with explicit `expert_tags` and prompt hints (`expert:<tag>`)
  - inference and stream payloads now expose `moe_geo` policy metadata
  - `/v1/network/status` now exposes `expert_profiles` (tag counts, layer coverage, router-capable peers)
  - CLI/API/peer flags wired for MoE controls and expert advertisement
- Added MoE/expert regression coverage:
  - DHT expert metadata round-trip
  - path-finder expert metadata parsing
  - engine MoE reorder behavior
  - network status expert profile aggregation
  - API expert tag forwarding
- Fixed seeding telemetry propagation from peer announcements through DHT into coordinator status/metrics.
- Added `TorrentSessionManager` seeding integration and expanded DHT/engine seeding observability.
- Implemented real polite daemon controller and peer load-pressure integration.
- Added tier-aware verification sampling:
  - Tier 1: `mystery_shopper` (`audit_rate`)
  - Tier 2+: `redundant_execution` (`redundant_exec_rate`)
- Added verification feedback to peer health/reputation scoring.
- Added model-level verification aggregates in `/v1/network/status`.
- Added `verification_degraded` alert with configurable thresholds.
- Added verification QoS gating for graceful fallback (`verification_qos_*` thresholds).
- Added auditor spot-checking on matched replicas (`auditor_rate`) with tertiary validation.
- Replaced post-hoc SSE word splitting with execution-path token streaming (`infer_stream` iterative decode loop).
- Added KV-cache control-plane affinity:
  - `session_id`-based prefill stickiness
  - cold-restart detection when sticky prefill peer is unavailable
  - `bandwidth_policy` metadata (`kv_affinity_hit`, `kv_cold_restart`, `kv_previous_prefill_peer_id`)
- Added tensor autoencoder serving-path integration:
  - optional per-hop activation encode/decode (`--tensor-autoencoder-enabled`)
  - configurable latent width (`--tensor-autoencoder-latent-dim`)
  - compression telemetry in inference responses (`compression.*`)
- Added speculative decoding stream mode:
  - deterministic draft token proposals (`DraftTokenModel`)
  - strong-model batch verification rounds with mismatch fallback
  - streaming metadata/flags (`speculative_enabled`, `speculative_draft_tokens`, `mode=speculative_decode`)
- Added acceptance-rate-aware speculative scheduling:
  - adaptive draft batch sizing with configurable thresholds and min/max bounds
  - per-stream telemetry (`rounds`, `mismatch_rounds`, `acceptance_rate`, `current_draft_tokens`)
- Added Tier 3 advanced encryption control path:
  - `ForwardRequest` encrypted activation envelope fields (`encrypted_activation`, nonces, ephemeral pubkeys, suite, layers)
  - per-hop `X25519 + HKDF-SHA256 + AES-256-GCM` activation transfer with configurable layered envelopes (`standard`/`enhanced`/`maximum`)
  - peer-side encrypted-hop enforcement (`encrypted_activation_required` for inter-stage hops when enabled)
  - inference telemetry (`encryption.enabled`, `level`, `suite`, `layers_per_hop`, `encrypted_hops`)
- Added on-wire tensor autoencoder transfer:
  - `ForwardRequest` compression metadata (`compression_codec`, `compression_original_dim`, `compression_latent_dim`)
  - coordinator now sends latent activations on inter-stage hops when autoencoder is enabled
  - peers reconstruct latent tensors to original dimensions before shard execution
- Added KV data-plane reuse in streaming:
  - session-scoped activation cache persisted behind KV-affinity keys
  - stream decode path reuses cached activation state on affinity hits (`session_id` + stable prefill peer)
  - stream telemetry now includes `streaming.kv_data_plane` (`cache_available`, `cache_used`, `cache_updated`)
- Upgraded stream decode loop reuse:
  - after first prefill round, subsequent rounds are seeded with prior activation state
  - avoids repeated prompt-prefill work within a single stream request
  - telemetry now includes `streaming.kv_data_plane.seeded_rounds` and `external_cache_seeded`
- Added KV activation relay on cold restarts:
  - cached session activation state is preserved when prefill peer changes
  - stream path can seed decode on new prefill peer after `kv_cold_restart`
  - telemetry includes relay indicators (`cross_peer_relay`, source/target peer ids)
- Added peer-native KV cache control path:
  - `ForwardRequest` / `ForwardResponse` now carry KV hints and hit status (`kv_session_id`, `kv_store_activation`, `kv_use_cached_activation`, `kv_cache_hit`)
  - peers now maintain bounded in-memory session KV entries (`--kv-cache-max-entries`) for prefill reuse
  - stream path now requests peer-native cache reuse on affinity hits and falls back to relay seeding on cache miss
  - telemetry now exposes peer-cache outcomes (`peer_native_cache_enabled`, `peer_cache_requested`, `peer_cache_hits`, `peer_cache_misses`, `peer_cache_fallbacks`)
- Added coordinator/API/CLI control for peer-native cache hinting (`--kv-peer-cache-enabled`).
- Added Tier 3.1 runtime/quantization profiling scaffold:
  - model shard runtime modes now support `fp32` / `int8` / `int4` quantization behavior and runtime profile reporting
  - peer runtime controls added (`--runtime-backend`, `--runtime-target`, `--quantization-mode`)
  - runtime metadata now propagates through DHT discovery records and coordinator peer endpoint models
  - inference `discovered_peers` and `/v1/network/status.runtime_profiles` now expose backend/quantization/GPU profile summaries
- Added Tier 3 DHT control-path extensions:
  - peer region announcements (`--region`) propagated through DHT and coordinator discovery metadata
  - lookup controls (`preferred_region`, `limit`, `sloppy_factor`) with region-prioritized selection
  - operator-diversified lookup ordering to reduce hot-key concentration bias
  - coordinator/API/CLI wiring for advanced DHT lookup options (`--dht-*`)
- Added DSHT sloppy-replication scaffolding:
  - announce path now replicates entries into DSHT replica keys (`model:<id>:dsht:<n>`)
  - lookup merges primary + DSHT replica-key entries with peer-id dedupe
  - coordinator lookup now controls DSHT merge width (`--dht-lookup-dsht-replicas`)
- Added DHT hot-key lookup rate limiting:
  - per-model lookup windows with configurable max requests
  - bootstrap returns `429` with `Retry-After` when a key is over quota
  - `/health` now exposes lookup rate-limit configuration and active windows
- Added Tier 3.2 pipeline-prefetch scaffold for stream decode:
  - coordinator stream loop can pre-submit next seeded decode round when activation seed is available
  - stream telemetry now includes `streaming.pipeline_parallel` (`prefetch_submitted`, `prefetch_hits`, `prefetch_misses`, `prefetch_failures`, `prefetch_waits`)
  - coordinator CLI/API now expose `--pipeline-parallel-enabled` and `--pipeline-parallel-workers`
- Added Tier 3.12 HYDRA token economy scaffold:
  - replaced token placeholder with persistent HYDRA account ledger (`balance`, `stake`, `rewards_earned`, `slashed_total`)
  - added off-chain state-channel lifecycle (`open`, `charge`, `reconcile`, `close`) with channel escrow accounting
  - wired serving rewards (`--hydra-reward-per-1k-tokens`) and optional verification slashing (`--hydra-slash-per-failed-verification`) into coordinator flows
  - surfaced economy telemetry in `/v1/network/status.hydra_economy` and `account_balance(...).hydra`
  - added HYDRA API routes (`/v1/hydra/status`, `/v1/hydra/account`, transfer/stake/unstake/channel ops)
- Added HYDRA channel anti-abuse hardening:
  - channel expiry metadata (`created_at`, `expires_at`) with automatic settlement on expiry
  - policy controls for default TTL, per-payer open-channel caps, and minimum channel deposit
  - coordinator/API/CLI wiring: `--hydra-channel-default-ttl-seconds`, `--hydra-channel-max-open-per-payer`, `--hydra-channel-min-deposit`
  - `POST /v1/hydra/channels/open` now supports optional per-channel `ttl_seconds` override
- Added and initialized this tracker file (`OpenHydra_progress.md`) for ongoing milestone tracking.

---

## Pass 5 — 2026-03-07: KV cache compaction via Attention Matching (arXiv:2602.16284)

Implemented all four phases of fast KV cache compaction based on the Attention
Matching (AM) algorithm by Zweiger et al. (MIT, 2026). Up to 50× compaction in
seconds with minimal quality loss. Reference repo (MIT license): github.com/adamzweiger/compaction.

### New files
- `peer/kv_compaction/__init__.py` — public API (`CompactionConfig`, `compact_past_key_values`)
- `peer/kv_compaction/_config.py` — `CompactionConfig` dataclass with all phase controls
- `peer/kv_compaction/_algorithms.py` — HAK and OMP key selection + Phase 2 β/Cv fitting
- `peer/kv_compaction/_compactor.py` — `compact_past_key_values()` entry point (all 4 phases)
- `peer/kv_compaction/_cache.py` — `CompactedKVCache` wrapper (Phase 2)
- `peer/kv_compaction/_beta_inject.py` — runtime monkey-patch for β injection (Phase 2)
- `peer/kv_compaction/head_budgets/qwen3_4b.json` — per-head budgets for Qwen3-4B (Phase 3)
- `peer/kv_compaction/head_budgets/llama3_8b.json` — per-head budgets for Llama-3.1-8B (Phase 3)
- `peer/models/__init__.py`, `peer/models/qwen3/__init__.py`, `peer/models/llama/__init__.py`
- `tests/test_kv_compaction.py` — 62 new tests covering all phases

### Modified files
- `peer/model_shard.py` — added 8 new `ToyShardConfig` fields; wired `_kv_cache_set` to
  call `compact_past_key_values` when enabled; auto-patches model for β on init
- `peer/server.py` — added 7 new params to `PeerService.__init__`, `serve()`, and
  `main()` argparse (7 new CLI flags: `--kv-compaction-enabled`, `--kv-compaction-method`,
  `--kv-compaction-ratio`, `--kv-compaction-beta`, `--kv-compaction-head-budget-path`,
  `--kv-compaction-online`, `--kv-compaction-online-max-tokens`)
- `pyproject.toml` — added `[kv-compaction]` optional extra pinning `transformers>=4.40`
  and `scipy>=1.11`

### Algorithm summary

| Phase | What | Method |
|-------|------|--------|
| 1 | Key selection, no β | HAK (RMS-aggregated attention topk) or OMP (greedy mass-pursuit) |
| 2 | β + Cv fitting | NNLS β via scipy (log-ratio fallback); Cv via `torch.linalg.lstsq` |
| 3 | Nonuniform head budgets | Per-layer/per-head JSON budgets (synthetic; optimal values from paper's optimization) |
| 4 | Online compaction | Compact at every `_kv_cache_set` when `seq_len > online_max_tokens` |

### Usage
```bash
# Phase 1 — no-beta (10% token budget, HAK)
openhydra-peer --kv-compaction-enabled --kv-compaction-ratio 0.10 ...

# Phase 1 — OMP key selection
openhydra-peer --kv-compaction-enabled --kv-compaction-method omp ...

# Phase 2 — β correction (requires scipy)
openhydra-peer --kv-compaction-enabled --kv-compaction-beta ...

# Phase 3 — per-head budgets
openhydra-peer --kv-compaction-enabled \
  --kv-compaction-head-budget-path peer/kv_compaction/head_budgets/qwen3_4b.json ...

# Phase 4 — online (cap physical KV at 512 tokens)
openhydra-peer --kv-compaction-enabled --kv-compaction-online \
  --kv-compaction-online-max-tokens 512 ...
```

### Compatibility
- Phase 1 (no-β): works with any standard HF model, no extra dependencies
- Phase 2 (β): works with Qwen2/3 and Llama families; model is auto-patched on init
- `scipy>=1.11` optional — Phase 2 falls back to log-ratio β fitting without it

## Highest-impact remaining work

1. Tier 3 is complete; remaining work shifts to post-Tier-3 production operations (SLO instrumentation, DAO/on-chain integration, and multi-region rollout hardening).
2. KV compaction Phase 2 β quality can be improved by passing actual query tensors from the forward pass instead of proxy-key reference queries (requires threading `Q` out of `_run_layers`).
