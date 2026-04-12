# OpenHydra — Post-HN Launch Plan

**Launch date**: March 17, 2026 (Monday)
**Current state**: Public repo at `samtroberts/openhydra`, landing page at `openhydra.co`, 867 tests passing, 3 bootstrap nodes live (EU/US/AP), MLX + CUDA + ROCm auto-detect working.

---

## Week 1: Launch Week (March 17–23)

### Day 0 — HN Post (Monday March 17)

**Post format**: "Show HN: OpenHydra — BitTorrent for LLMs, run frontier models across volunteer laptops"

**Monitoring checklist:**
- [ ] Watch bootstrap node health (all 3 Linodes: EU/US/AP)
- [ ] Monitor nginx logs for traffic spikes (`journalctl -u bootstrap -f`)
- [ ] Watch GitHub issues/stars/forks in real-time
- [ ] Reply to every HN comment within 1 hour
- [ ] Have `openhydra.co` up and responsive (Cloudflare Pages — auto-scaling)

**Prepared answers for likely HN questions:**
| Question | Answer |
|----------|--------|
| "How does this compare to Petals?" | Trust layer (verification, economy, privacy), KV compaction, graceful degradation — see `docs/petals-comparison.md` |
| "How does this compare to Exo?" | Exo = local cluster (same network), OpenHydra = public internet (strangers). We add verification, privacy, incentives |
| "What prevents a malicious peer?" | Three-tier verification (mystery shopper, redundant exec, auditor), reputation scoring, Sybil resistance via geo-challenge |
| "How do you handle privacy?" | Onion routing + AES-256-GCM activation encryption + DP noise injection. No single peer sees your full query |
| "What's the token economics?" | Barter credits (immediate, decay 5%/day) + HYDRA token (capped 69M supply, burn-and-mint). See economy section |
| "Can I actually use this today?" | Yes: `pip install -e . && openhydra-node --peer-id my-node --model-id Qwen/Qwen3.5-0.8B` |

### Days 1–3 — Triage & Quick Fixes (Tue–Thu)

**Priority**: Fix any blocking issues reported by real users trying to install and run.

Likely issues:
1. **Install failures** — missing system deps, platform-specific pip issues
2. **gRPC/protobuf version mismatches** — pin versions in pyproject.toml
3. **MLX not found on Intel Macs** — better error message, guide to PyTorch fallback
4. **Bootstrap nodes overloaded** — add rate limiting, scale if needed
5. **Model download too slow** — P2P model cache needs seeders; seed from personal machines

**Response SLA**: Critical bugs (crashes, can't install) → fix within 4 hours. UX issues → fix within 24 hours.

### Days 4–7 — Metrics & Momentum (Fri–Sun)

- Track: GitHub stars, forks, clones, npm downloads, unique IPs hitting bootstrap nodes
- Write a follow-up blog post if traction is strong ("What we learned launching OpenHydra on HN")
- Cross-post to Reddit r/LocalLLaMA, r/selfhosted, r/MachineLearning
- Submit to Product Hunt if HN reception is positive

---

## Week 2–3: Stabilization (March 24 – April 6)

### Developer Experience Polish

| Task | Priority | Why |
|------|----------|-----|
| **One-line install script** (`curl -sSL openhydra.co/install | sh`) | P0 | Removes pip/git friction for non-Python users |
| **Docker image on Docker Hub** | P0 | `docker run openhydra/node` — zero-dependency onboarding |
| **Pre-built wheels** on PyPI | P1 | `pip install openhydra` instead of git clone |
| **Homebrew formula** | P1 | `brew install openhydra` for Mac users |
| **ARM Linux support** (Raspberry Pi, Jetson) | P2 | Large hobbyist audience on HN |

### Reliability Improvements

| Task | Priority |
|------|----------|
| **Automated bootstrap node monitoring** (Prometheus alerts → PagerDuty/Slack) | P0 |
| **Bootstrap node auto-scaling** (Terraform, spin up more Linodes if load > threshold) | P1 |
| **Connection retry with exponential backoff** in coordinator | P1 |
| **Graceful handling of peer churn** (peer disconnects mid-inference → retry on another peer) | P1 |
| **Integration test suite** (real model, real gRPC, real DHT — not just mocks) | P2 |

### Documentation

| Task | Priority |
|------|----------|
| **MkDocs Material site** at `docs.openhydra.co` | P0 |
| **Quickstart video** (2-min screencast: install → run → chat) | P0 |
| **API reference** (OpenAPI spec auto-generated) | P1 |
| **Architecture deep-dive** blog post | P2 |
| **Contributing guide** with "good first issue" labels | P1 |

---

## Month 2: Community & Growth (April 7 – May 4)

### Desktop App (Tauri v2)

Ship the native desktop client (see `plans/openhydra-desktop.md`):
- One-click node management (Start/Stop toggle)
- Model selector with VRAM-aware filtering
- Live terminal logs
- HYDRA balance dashboard
- **Target**: macOS DMG on GitHub Releases by April 15

### SDK v1

| SDK | What | Target |
|-----|------|--------|
| **Python SDK** | `pip install openhydra-sdk` — `client.chat("Hello")` with streaming, retry, type hints | April 10 |
| **TypeScript SDK** | `npm install @openhydra/sdk` — same API, works in Node.js and browser | April 20 |

### Community Building

| Channel | Action |
|---------|--------|
| **Discord server** | Launch with #general, #support, #dev, #showcase channels |
| **GitHub Discussions** | Enable for Q&A, feature requests, show-and-tell |
| **Twitter/X** | Weekly updates, demo videos, community highlights |
| **Blog** (openhydra.co/blog) | Technical deep-dives, benchmark results, comparison posts |

### Model Catalog Expansion

| Model | HuggingFace ID | Priority |
|-------|----------------|----------|
| Llama 3.2 1B | `meta-llama/Llama-3.2-1B` | P0 |
| Llama 3.1 8B | `meta-llama/Llama-3.1-8B` | P0 |
| Mistral 7B | `mistralai/Mistral-7B-v0.3` | P1 |
| Qwen 2.5 Coder 7B | `Qwen/Qwen2.5-Coder-7B` | P1 |
| Llama 3.1 70B | `meta-llama/Llama-3.1-70B` | P2 (needs 8-peer pipeline) |

---

## Month 3: Economy & Incentives (May 5 – June 1)

### On-Chain Integration

| Task | Description |
|------|-------------|
| **Solidity state-channel contract** | Deploy on Arbitrum or Base (L2 for low gas). Enable HYDRA token staking/slashing on-chain |
| **Wallet connect** | MetaMask/WalletConnect integration in desktop app |
| **Token claim flow** | Peers earn off-chain credits → periodic on-chain settlement |
| **Governance DAO** | Token holders vote on model catalog, fee structure, protocol upgrades |

### Sustainability

| Revenue Stream | How |
|----------------|-----|
| **Priority routing** | Pay HYDRA tokens for guaranteed low-latency inference |
| **Commercial API** | Hosted coordinator endpoint for enterprises (no node required) |
| **Enterprise support** | SLA-backed support contracts |
| **Desktop Pro** | Premium desktop features (multi-model, priority queue, analytics) |

---

## Success Metrics

### Week 1 Targets
| Metric | Target |
|--------|--------|
| GitHub stars | 500+ |
| HN upvotes | 100+ |
| Unique nodes on network | 50+ |
| Successful inference requests | 1,000+ |

### Month 1 Targets
| Metric | Target |
|--------|--------|
| GitHub stars | 2,000+ |
| Monthly active nodes | 200+ |
| Models in catalog | 10+ |
| Discord members | 300+ |
| PyPI downloads | 1,000+ |

### Month 3 Targets
| Metric | Target |
|--------|--------|
| GitHub stars | 5,000+ |
| Daily active nodes | 500+ |
| Desktop app downloads | 2,000+ |
| HYDRA tokens in circulation | 100,000+ |

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| **Bootstrap nodes go down** | Auto-scaling Terraform, multi-cloud (Linode + Hetzner), health checks every 30s |
| **Low peer count → bad UX** | Seed network with own machines (3-5 always-on nodes running popular models) |
| **Security vulnerability reported** | `SECURITY.md` with responsible disclosure policy, 48-hour response SLA |
| **"This is just Petals"** | Clear differentiation doc ready; emphasize trust layer, economy, privacy |
| **Model download bottleneck** | P2P model cache (already built), HuggingFace CDN fallback, pre-seeded popular models |
| **Regulatory concerns** | Decentralized = no single operator. Open-source = auditable. DP noise = plausible deniability |
