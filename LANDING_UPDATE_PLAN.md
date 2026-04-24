# Landing Page Update Plan

After parsing the live [openhydra.co](https://openhydra.co/) and
reviewing the local copy in `/Users/sam/Documents/New project 2/landing/`
(which is in `.gitignore` — canonical published copy lives in the
private repo `samtroberts/openhydra-landing`).

## Live-vs-local divergence summary

| Area | Live openhydra.co | Local `landing/index.html` | Plan |
|---|---|---|---|
| Hero headline | "Run AI in a herd, not a data centre" | "Turn on. Tune in. *Drop in.*" | **Do not change** — the local's creative direction is distinct from live; user owns that brand choice. Flag in plan but don't touch without explicit sign-off. |
| Hero tagline | "OpenHydra splits big language models across volunteer laptops — BitTorrent-style…" | "Open the app. Your Mac, NVIDIA, or AMD GPU auto-joins…" | **Keep local**, same reason as hero. |
| Install command | `git clone … && pip install -r requirements.txt` | Same (already fixed 2026-04-24) | ✓ Aligned. |
| DHT terminology | "HTTP DHT + Hivemind Kademlia" | "HTTP DHT + Rust libp2p Kademlia" (already fixed 2026-04-24) | ✓ Local correct; live is stale. |
| Catalog count | "5 models in the default catalog" | "21 models in the default catalog" (already fixed 2026-04-24) | ✓ Local correct; live is stale. |
| Benchmark numbers | 6.9 / 9.8 / 7.3 / 1.09 TPS (pre-2026-04-17) | Same + new freshness banner for 3.76 TPS True Petals 3-node | Partial — need to also refresh the in-place "Measured, not promised" benchmark table so the 3-node result is *in the table*, not just a pinned banner. |
| Path A / StandaloneHead / True Petals | None mentioned | Only in the freshness banner at top | Add a proper "3-node True Petals" mini-section so it reads as part of the narrative, not an afterthought. |
| Install prerequisites | `xcode-select`, `apt install build-essential libssl-dev`, `pip install torch` | Same + `protobuf-compiler`, `rust`, numpy<2, torchvision-uninstall | ✓ Local correct and more comprehensive; live is stale. |
| Download buttons | `v0.1.0` DMG + EXE (release from 2026-04-ish) | Same URLs | Verify v0.1.0 release assets still exist; no change proposed unless broken. |

## What this plan WILL do (inside `/Users/sam/Documents/New project 2/landing/`)

1. **Update the "Measured, not promised" benchmark block** to include the 2026-04-24 3-node True Petals row (3.76 TPS, 3.87× baseline) alongside the existing four rows. Not replacing anything — adding one cell so the table reads as "progress over time".
2. **Weave Path A into "How it works"** — one extra subsection/step explaining that the coordinator can *itself* run no layers (true Petals topology). Keeps the narrative honest without turning the landing into a changelog.
3. **Tighten the freshness banner copy** so it reads as a headline result, not a footnote. Shorter, bigger number, one-line explanation.
4. **No hero changes.** Preserved.
5. **No download-URL changes.** Preserved.
6. **Fixes that were already applied in my 2026-04-24 pass stay as-is**: Hivemind→libp2p, 5→21 models, updated install one-liner, extended prerequisite hints.

## What this plan will NOT do

- Touch the live `samtroberts/openhydra-landing` private repo. My edits are local only. The user pushes to that repo at their discretion.
- Rewrite the hero. "Turn on. Tune in. Drop in." stays unless the user wants it aligned to the live "Run AI in a herd".
- Change the HYDRA tokens / earning claim copy. That's a product-strategy decision.
- Change the download link target versions unless they 404.

## Delivery

After the landing edits, a brief summary of what changed in the
response, then the response pivots to the README verification task
(run the cross-ISP Mac↔GPU1 benchmark, correct README on any
deviation).
