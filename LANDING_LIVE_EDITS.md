# Suggested edits for the LIVE openhydra.co

For the private repo `samtroberts/openhydra-landing`. The hero stays
exactly as live ("Run AI in a herd, not a data centre" + BitTorrent
tagline). Edits below target stale facts and missing milestones only.

---

## Edit 1 — Install prerequisites (hero install row / footnote)

**Why:** the live page's macOS/Linux hints omit two steps that have
cost new users hours since 2026-04-23: pinning `numpy<2` on managed
Python envs (Lightning AI / Modal ship numpy 2.x with scipy pinned
against 1.x → import cascades through transformers), and uninstalling
torchvision on Linux CUDA studios (torchvision 0.23 vs torch 2.11 ABI
mismatch → `operator torchvision::nms does not exist` → every HF
`AutoModelForCausalLM` load fails with a misleading
`ModuleNotFoundError: Qwen3_5ForCausalLM`).

**Find** (in the hero footnote under the install command):

```
macOS: requires xcode-select --install first.
Linux: apt install build-essential libssl-dev, then pip install torch separately for NVIDIA/AMD.
```

**Replace with:**

```
macOS: xcode-select --install, then brew install rust.
Linux: sudo apt install build-essential libssl-dev protobuf-compiler, then curl https://sh.rustup.rs | sh.
On managed Python envs (Lightning AI / Modal) also run: pip install "numpy<2" --force-reinstall && pip uninstall -y torchvision.
See Troubleshooting → numpy / torchvision.
```

---

## Edit 2 — DHT terminology (Features → Dual-stack DHT routing)

**Why:** "Hivemind Kademlia" was replaced with Rust libp2p in 2026-03.
The OpenHydra codebase no longer has a Hivemind dependency. Keeping
the old name on the landing is misleading.

**Find:**

```
HTTP DHT + Hivemind Kademlia across three continents.
```

**Replace with:**

```
HTTP DHT + Rust libp2p Kademlia across three continents,
with DCUtR hole-punching and Circuit Relay v2 fallback.
```

---

## Edit 3 — Catalog size ("What runs on it" section footnote)

**Why:** the live page says "5 models in the default catalog". As of
2026-04 there are 21. Undercounting the catalog in a marketing page
misses a real selling point (breadth).

**Find:**

```
5 models in the default catalog.
Add any HuggingFace model to models.catalog.json.
```

**Replace with:**

```
21 models in the default catalog (Qwen 2.5/3/3.5, Gemma 3/4,
SmolLM2, TinyLLaMA — base + instruct variants).
Add any HuggingFace model to models.catalog.json.
If the requested model lacks peers, the coordinator gracefully
degrades to the nearest available smaller model.
```

---

## Edit 4 — Benchmarks ("Measured, not promised" section)

**Why:** the live page's benchmark table shows 6.9 / 9.8 / 7.3 / 1.09
TPS — all from before 2026-04-17. The 2026-04-24 milestone (**3.76
TPS on a 3-node True Petals swarm**, 3.87× the 2-node cross-ISP
baseline) is the single biggest proof point we have for the
"nobody needs to own the whole model" pitch, and it's missing.

**Find** the existing benchmarks section, however it's structured
(table / grid / bullet list). **Add** a new cell/row for the
3-node result, visually distinguished as the newest/most impressive
number.

**Suggested cell copy** (to match the other cells' tone):

```
Qwen 3.5 2B · 3-node True Petals · new (2026-04-24)
3.76 tok/s
Pure coord + 2 GPU shards on a shared LAN.
3.87× the 2-node cross-ISP baseline.
Coordinator holds only lm_head + embeddings — no transformer layers.
```

**Optional longer-form callout block** for before/after the table (if
the landing style supports it):

```
In April 2026 the OpenHydra swarm hit its design target: run a
language model end-to-end where no single node owns the whole
thing. On a three-Lightning-AI-studio VPC, one Linux node runs
the coordinator with just the lm_head + embeddings (~500 MB),
and two NVIDIA T4 peers shard the 24 transformer layers between
them. The coordinator samples every token locally and re-injects
it into the ring — classic Petals client-terminated pipeline.
3.76 tokens/sec, 3.87× faster than the 2-node cross-ISP ring
that held for two months.
```

---

## Edit 5 — Download button version

**Why:** the live page links to `v0.1.0` DMG + EXE releases. Verify
that release asset still exists on GitHub
(`samtroberts/openhydra/releases/download/v0.1.0/...`). If a newer
Tauri release has shipped, bump. If `v0.1.0` still resolves, no change
needed.

**Suggested check:** `curl -IL https://github.com/samtroberts/openhydra/releases/download/v0.1.0/OpenHydra_0.1.0_aarch64.dmg`
should return 302 → 200. If 404, replace with the latest Releases
landing URL: `https://github.com/samtroberts/openhydra/releases/latest`.

---

## Edit 6 (optional) — "What you get" card: new 3-node Petals

**Why:** the Features grid has cards for "Drop-in OpenAI API", "KV
cache compaction", "Dual-stack DHT routing", "Desktop app + Local
Mode", "Onion routing", "Python & TS SDKs coming soon". A sixth card
for the True Petals topology makes the technical selling point
concrete.

**Suggested new card:**

```
Emoji: 🔗 or 🧬
Title: True Petals topology
Description: Coordinator runs no transformer layers — just
lm_head + embeddings. Peers shard the layers between them.
Zero trust required; every token is re-sampled on the client.
Classic Petals client-terminated pipeline, but on WebSockets and
libp2p instead of daemon sockets.
```

Or, if the grid is already full, skip this — Edit 4 carries the
message via benchmarks alone.

---

## Edits NOT proposed

- **Hero copy stays.** "Run AI in a herd, not a data centre" is
  preferred and will not be changed.
- **HYDRA tokens / earnings claims stay.** Product-strategy decision.
- **"5 models" → "21 models"** is Edit 3; no broader restructure of
  "What runs on it".
- **Section order stays.** No reshuffle proposed.

---

## Execution checklist for the user

When ready to land these on the live site:

1. Clone `samtroberts/openhydra-landing` locally.
2. Apply Edits 1-4 (required) + 5 (verify) + 6 (optional).
3. `git commit -m "docs(landing): 3.76 TPS + libp2p + 21 models + numpy/torchvision note" --author="Claude <noreply@anthropic.com>"` if you want the attribution.
4. Push; the live site deploy picks up via whatever CI is wired
   (Cloudflare Pages / Vercel / Netlify).
