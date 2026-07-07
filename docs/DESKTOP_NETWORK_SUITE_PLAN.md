# OpenHydra Network Suite — BiglyBT views, engine onboarding, models board, geo dashboard

**Status:** PLAN (2026-07-03, no code yet). Covers four requested feature areas for the
desktop app + network:

1. BiglyBT-grade network introspection (DHT settings, peers view, swarm view)
2. First-run engine selector with one-click engine installs
3. Top-models board + model-request polls (shown at startup)
4. Geographic node/usage dashboard, in-app and at `status.openhydra.co`

**Verdict up front:** all four make sense. Each has one hard consideration this plan
resolves: (1) needs an **agent introspection API** first — the app supervises child
processes and currently cannot see inside their swarms; (2) is UX + package-manager
plumbing, very feasible; (3) needs a small **central control plane** (the network's first
non-bootstrap service) + a content-policy decision on abliterated models (recommendation:
protocol-neutral, catalog-curated — see §4.4); (4) is the same control plane plus an
**opt-in telemetry beacon**, with privacy designed in from the start.

---

## 0. The enabler everything hangs off: the agent introspection API

**Problem.** BiglyBT can render peers/swarm/DHT views because the UI lives in the same
process as the network stack. OpenHydra Desktop deliberately does not: the app supervises
`openhydra-agent` **child processes** and today sees only their log lines. Peer tables,
DHT routing-table state, and connection paths live inside the child's libp2p event loop.

**Solution.** A read-only, loopback-only **status endpoint** on the agent
(`--status-bind 127.0.0.1:<port>`, off by default; the app always enables it for its
children). One JSON surface, versioned, serving:

- `/status/peers` — connected peers: peer id, direction, transport (QUIC/TCP), path
  (direct / relay / dcutr-punched — the `proxy_forward` truthful-path data we already
  log), multiaddr, agent version (identify), connection age, per-peer bytes/requests,
  reputation score (M2.2), credit/throttle state (M2.3), country (see §5 privacy).
- `/status/dht` — routing-table size per bucket, server/client mode, AutoNAT verdict +
  confidence, external addrs, provider records held, records we announce, last announce
  age, relay reservations (which relays, TTL), DCUtR punch attempts/successes (v4/v6 —
  counters exist from #43-W1), rebootstrap/generation counters (#42), PEX peers learned.
- `/status/swarm?model=<id>` — per-model view: providers known for the model (from the
  router's last discovery), their rank inputs (reputation, liveness), which one served
  last, our own announce state for that model.
- `/status/transfers` — rolling counters: requests served/consumed, tokens in/out,
  per-model TPS aggregates, receipts co-signed, audits agreed/inconclusive (#39), AUP
  refusals, throttle events. In-memory ring of recent requests (model, peer, tokens,
  wall ms, path) for the History view.
- `/metrics` — the Prometheus surface (#33) already exists gateway-side; provider role
  gets the same treatment. The status endpoint is the JSON twin.

**Why HTTP on loopback** (vs pipes): both roles already run HTTP stacks; the same
surface later feeds Prometheus scrapers and the telemetry beacon (§5) without new
plumbing; and `curl`-ability keeps it debuggable. Auth: loopback bind + a per-launch
bearer token the app passes via env, so another local user/process can't read peer data.

**Where the data comes from:** most items already exist internally (event-loop state,
reputation store, credit map, dispatch logs, #43 counters) — this is an exposure task,
not new distributed-systems work. The event loop gains one `StatusSnapshot` request
message; no hot-path changes.

---

## 1. BiglyBT suite: Peers view, Swarm view, DHT view, and the settings surface

### 1.1 Does it make sense?
Yes — this is the most natural feature transplant in the whole plan. OpenHydra *is*
BitTorrent-shaped (DHT, PEX, relays, reciprocity, per-"content" swarms where content =
models). BiglyBT's power-user views map almost 1:1. What does **not** map: torrent-file
mechanics (pieces/blocks/trackers/disk-ops views) — their OpenHydra analogue is the
request/receipt stream, covered by Transfers/History.

### 1.2 Peers view (BiglyBT "Peers" + country stats)
Dense sortable table (we already have the BiglyBT table style):

| BiglyBT column | OpenHydra column |
|---|---|
| IP / client | Peer id (short) / agent version via identify |
| Flags (encryption, source) | Transport (QUIC/TCP) · Path (direct/relay/punched) · Source (DHT/PEX/mDNS/bootstrap) |
| % complete | Models announced by that peer |
| Down/Up speed | Tokens consumed from / served to that peer (rates + totals) |
| Choked/interested | Throttle state (M2.3 credit multiplier) |
| Country flag | Country (only if peer opted into telemetry, else "—"; never IP-derived client-side — see §5.3) |
| — | Reputation score (M2.2), audits agree/disagree with this peer |

Row context menu: disconnect, forget, pin as bootstrap, view swarms in common.

### 1.3 Swarm view (BiglyBT's animated swarm graph)
Per-model force graph: our node center; providers of the selected model as nodes sized
by reputation, colored by path (direct green / relay amber / punched teal); edges pulse
on live requests. Data = `/status/swarm` + `/status/peers`, animated client-side (SVG/
canvas; no heavy deps). A secondary "whole-mesh" mode shows every connected peer grouped
by role (bootstrap/relay/provider/consumer). This is partly eye-candy — but it is *the*
BiglyBT signature, it makes relay-vs-direct legible at a glance (our #1 support
question), and it demos the network better than any table.

### 1.4 DHT view (BiglyBT DHT stats)
Read-only diagnostics panel: routing-table occupancy per bucket, client/server mode +
AutoNAT verdict history, external addr candidates, our provider records (model → last
announce age, TTL countdown vs the relays' 300 s expiry), relay reservations with
expiries, DCUtR v4/v6 attempt/success counters, rebootstrap events, PEX intake rate.
Everything here has bitten us live at least once (stale announces, zombie conns, v6
dial-back) — this panel turns those from log-archaeology into a glance.

### 1.5 Settings surface (BiglyBT Options → Connection/Transfer/Queue)
The CLI already exposes nearly all of it; the app surfaces it as **Settings → Advanced**
(basic Settings stays as-is: bootstraps, gateway port, autostart, search, memory):

| Category | Existing flags to surface |
|---|---|
| Network | listen ports, `--peer-relay` (be a relay for others), `--connection-reversal`, `--pcp-gateway`, identity file path (+ "reset identity" with a scary confirm) |
| DHT | bootstrap list editor with liveness check, reannounce interval (bounded < 300 s TTL with an explanatory note) |
| Serving | `--max-concurrency`, AUP floor (max messages / prompt chars / completion tokens / deny-substrings — the closest thing to BiglyBT rate limits, already implemented as #40) |
| Gateway | rate-limit levers (#41: max-inflight, rps, burst, trusted-proxy), API key |
| BYOK | Anthropic/Gemini/embeddings model routes + keys (#34) |
| Telemetry | opt-in/out + what's shared (see §5.3) |

Each control maps to a child-process flag → applying = restart the affected role
(the app already supervises restarts; a "pending restart" chip keeps it honest).

### 1.6 Effort
Introspection API (§0) is the bulk: ~1–2 sessions agent-side. Views: peers/DHT tables
~1 session; swarm graph ~1 session; advanced settings ~1 session.

---

## 2. First-run engine selector + one-click engine install

### 2.1 Does it make sense?
Yes — it completes the "non-dev installs OpenHydra" story. Today the app auto-detects
running engines, but a fresh machine has **none**, and the current experience (empty
models table + a banner) makes the user solve the hardest problem alone. A first-run
wizard turns that into a menu.

### 2.2 Flow
Trigger: app launch with **no engine detected and no prior dismissal** (state in
`desktop.json`). Steps:

1. **Detect** — run the existing probe; anything already running is pre-selected
   ("Found Ollama with 2 models ✓").
2. **Choose** — engine cards (logo, one-liner, disk/RAM needs, platform fit — hide vLLM
   on Macs, hide LM Studio on headless Linux): Ollama (recommended default), LM Studio,
   llama.cpp, vLLM, Exo.
3. **Install** — one click per engine, with a live progress log panel:

| Engine | macOS | Windows | Linux |
|---|---|---|---|
| Ollama | `brew install ollama` (or official pkg script) | `winget install Ollama.Ollama` | official `curl \| sh` script |
| LM Studio | `brew install --cask lm-studio` | `winget install LMStudio` (verify id) | AppImage download |
| llama.cpp | `brew install llama.cpp` | winget/choco or release zip | release binary / package |
| vLLM | n/a (hide) | n/a (WSL note) | `uv tool install vllm` (NVIDIA note) |
| Exo | guided (git + uv) — "advanced" badge, docs link | n/a | guided |

   Fallback for every row: "Open download page" if the package manager is missing or
   the command fails. Install commands run visibly (streamed output), never silently.
4. **First model** — for Ollama/llama.cpp offer a starter pull (e.g. `llama3.2:1b`,
   ~1.3 GB) with a progress bar; LM Studio hands off to its own model browser.
5. **Go** — enable engine-autostart, start Share (+ optionally Gateway), land on the
   Dashboard with the models table filling in. Wizard re-runnable from Settings.

### 2.3 Design notes
- **Consent & transparency:** each install shows the exact command before running;
  nothing runs without the click. Package managers do the verification/signing work —
  we never curl arbitrary binaries ourselves except official vendor scripts, shown
  verbatim first.
- **Reuse:** detection = existing `detect_engines()`; process-running = the autostart
  module's pattern; the wizard is UI + a per-OS command table, not new architecture.
- This also ties into §3: the wizard's "first model" step can offer the **currently
  most-requested poll model** ("the network is asking for X — host it?").

### 2.4 Effort
~1–2 sessions (the matrix needs per-OS verification; macOS first).

---

## 3. Models board: top models + request polls (startup surface)

### 3.1 Does it make sense?
Yes, and it's strategically important: it converts the network's core coordination
problem — **supply (what providers host) meeting demand (what consumers want)** — into
a visible loop. BitTorrent had no way to ask seeders to seed something; OpenHydra can.

### 3.2 What it needs that we don't have: a small control plane
Aggregated "top models" and tamper-resistant polls can't come from the DHT alone
(aggregation + sybil-resistant voting over Kademlia is a research project, not a v1).
Pragmatic v1: a tiny **network-services API** (`api.openhydra.co`) — one small service
(Cloudflare Worker + KV/D1, or a service on the netcup box beside the bootstrap):

- `GET /v1/models/top` — ranked models: provider count, regions, avg native TPS,
  tokens served (7d), trend. Fed by the telemetry beacon (§5).
- `GET /v1/polls/current` + `POST /v1/polls/vote` — vote = `{model_id, peer_id,
  ed25519 signature}`; one vote per peer id per poll; optionally weighted by earned
  reputation/served-tokens later (raises the cost of sybil farms).
- `POST /v1/polls/pledge` — "I'll host this": returns the exact engine command
  (`ollama pull …`) and, once the beacon sees the model announced from that peer,
  marks the pledge fulfilled (a satisfying loop, and later a reciprocity bonus).

This is consistent with the existing direction of a centralized *service* layer on top
of a neutral protocol (same posture as the monetization control plane). The protocol
never depends on it; the app degrades gracefully to "board unavailable".

### 3.3 In-app surface
- **Models board view**: two panels — *Top on the network* (ranked table: model,
  providers, avg TPS, tokens 7d, "chat now" button) and *Requested* (poll: model,
  votes, "vote" + "host this" buttons wired to the pledge flow + engine pull).
- **Startup prompt**: a dismissible card on the Dashboard (not a modal — respect the
  user): "Most-requested this week: X (N votes) — host it with one click." Frequency-
  capped (once per poll cycle), permanently dismissible.

### 3.4 Which models should the list/poll carry?
Curation principle: **models people actually run locally in 2026, across the hardware
tiers the network really has** (potato laptop → single GPU → Mac unified memory →
sharded clusters):

- **Small/default tier (≤4 GB):** Qwen 3.5 0.6B/2B, Llama 3.2 1B/3B, Gemma 3 2B-class,
  SmolLM-class, TinyLlama (the LAN-test classic).
- **Single-GPU / 8–16 GB tier:** Qwen 3.5 4B/9B, Llama 3.1/3.3 8B, Gemma 3 9B-class,
  Mistral Small, DeepSeek-R1 distills (7B/8B) — reasoning models poll extremely well.
- **Coder models** (the gateway's OpenAI-compat + the app's Code view make these
  high-demand): Qwen3-Coder tiers, DeepSeek-Coder-V2-class, Codestral-class.
- **Mac unified-memory / MLX tier:** the mlx-community 4-bit conversions (Llama 3.2/3.3,
  Qwen 3.5) — they're what LM Studio/Exo users actually have.
- **Sharded/frontier tier (the aspirational pull):** Llama 3.3 70B, Qwen3-Next 80B,
  GLM-4.7, Qwen3-Coder 480B — listed with a "needs N peers" badge; these are the
  poll items that motivate cluster formation (Exo/llama.cpp-RPC).
- **Embeddings** (quietly important for RAG consumers): nomic-embed, bge-class.
- Poll candidates beyond the curated seed: free-text suggestion box, admin-approved
  into the ballot (prevents ballot spam/squatting).

### 3.5 Abliterated models — honest answer
Context: "abliterated" = refusal behavior surgically removed from open-weight models;
popular in the local-AI scene, and they *would* poll well.

Recommendation: **protocol-neutral, catalog-curated — don't feature them in the
official board/poll; don't pretend the protocol can ban them.**

- *Protocol level:* OpenHydra routes by model id; operators choose what their engines
  serve. A protocol-level ban is neither enforceable nor philosophically consistent
  with BYO-engine. Providers already have the per-operator AUP floor (#40) as their
  own control, plus the ability to simply not host a model.
- *Catalog/poll level (the part we curate):* featuring abliterated models on the
  official startup surface makes every volunteer provider — who may not understand
  what they're pulling — the serving endpoint for uncensored output to strangers,
  routed through infrastructure with your name on it. That's provider legal exposure
  (jurisdiction-dependent), a reputational tail-risk while the network is young, a
  direct conflict with the planned payments layer (processors are unforgiving about
  exactly this), and a likely app-store/distribution problem for the desktop app.
- *Practical line:* the board lists mainstream open models; the poll ballot is
  curated; nothing stops an operator hosting whatever their local law allows, and
  discovery still works for any model id — it just isn't *promoted* by the network's
  official surfaces. Revisit when there's a moderation/verification story (and a
  legal entity) that can carry it.

### 3.6 Effort
Control-plane API ~1 session (Worker + KV is small); board view + startup card ~1
session; pledge-fulfillment loop depends on the beacon (§5) landing first.

---

## 4. Geographic dashboard — in-app + status.openhydra.co

### 4.1 Does the map make sense?
Yes — with the privacy design below. Precedent is strong (BiglyBT itself ships
country-based transfer stats; every serious P2P network publishes a node map) and the
demand is real: a public status page is the single best growth/trust artifact a young
network can have ("is this thing alive?" answered with a map and counters). The one
non-negotiable: **node locations are volunteers' homes.** City-level coarsening +
consent, or we don't ship it.

### 4.2 Architecture
- **Beacon (agent-side, opt-in):** every N minutes the agent POSTs a signed heartbeat
  to `telemetry.openhydra.co`: peer id (as a salted hash — the raw id is already
  public on the DHT, but hashing decouples the two datasets), role(s), agent version,
  models announced (ids only), rolling counters (requests, tokens in/out, per-model
  native-TPS aggregates, path mix direct/relay/punched, punch successes), engine kinds
  (no URLs). **No prompts, no completions, no IPs in the payload.**
- **Ingest:** the service derives coarse geo from the connecting IP **server-side**,
  stores only `{country, region, city-centroid lat/lon}`, and discards the IP. Worker +
  Analytics Engine/D1 (or netcup + DuckDB) — heartbeat volume is trivial.
- **Status site (`status.openhydra.co`):** static page + the API, in the landing-page
  brand: the map, headline counters, model table, trends. The **app's Network view
  embeds the same API** (one data source, two skins; app adds "you are here").
- Aggregates only on the public surface: no per-node drill-down below city cluster;
  cities with <3 nodes render at country level (k-anonymity floor).

### 4.3 Metrics
Requested: **active nodes · models served · avg TPS per model · total tokens.** All
four come straight from the beacon. Worth adding (grouped, all cheap once the beacon
exists):

- *Network health:* nodes by role (providers/gateways/relays/bootstraps), churn
  (joins/leaves 24 h), version distribution (upgrade telemetry), relay-vs-direct-vs-
  punched ratio (the connectivity story in one number), DCUtR success rate v4/v6,
  median relay reservation count.
- *Serving quality:* p50/p95 TTFT and wall-time per model, native vs pipeline TPS
  spread (network overhead made visible), requests/min network-wide, error/timeout
  rate, AUP-refusal count.
- *Supply/demand:* unique models on the network, top models by tokens (feeds §3's
  board), **requested-but-unserved models** (poll votes with zero providers — the
  gap chart is the single most actionable panel), capacity proxy (sum of announced
  model sizes), regional coverage per top model (latency story).
- *Trust layer:* receipts co-signed (24 h), audit outcomes agree/inconclusive/disagree
  (#39 — a public integrity signal no centralized API can offer; genuinely
  differentiating), reputation distribution histogram.
- *Fun/growth:* tokens-served odometer, "herd size" over time, opt-in provider
  leaderboard (pseudonymous, peer-id-hash handles).

### 4.4 Privacy defaults (decide before building)
Recommendation: beacon **off by default in the CLI agent**, and presented as a clear
opt-in toggle in the desktop app's first-run wizard ("Count me on the map — city-level
only, never your address or prompts"), with the payload documented and inspectable
(`openhydra-agent telemetry --show-last`). The map's growth value is real but consent
is worth more; make opting in feel like joining, not being tracked.

### 4.5 Effort
Beacon ~half a session (counters exist once §0 lands); ingest+API ~1 session; status
site ~1 session (reuse landing styles); in-app view ~half. The map itself: static
world SVG + dot clusters — no heavy mapping deps.

---

## 5. Phasing & dependencies

| Phase | Delivers | Depends on |
|---|---|---|
| **P0** | Agent introspection API (§0) — status endpoint on both roles | — (pure exposure work) |
| **P1** | Peers view + DHT view + Transfers/History; Settings → Advanced | P0 |
| **P2** | First-run engine wizard + one-click installs (macOS first) | — (parallel to P0/P1) |
| **P3** | Swarm graph view | P0 (+ P1 tables for drill-down) |
| **P4** | Telemetry beacon + `status.openhydra.co` + in-app Network map | P0 (counters); infra decision (Worker vs netcup) |
| **P5** | Models board + polls + pledge loop + startup card | P4's control plane (shares the service) |

Rationale for the order: P0 unlocks three of the four asks; the wizard (P2) is
independent and the biggest onboarding win, so it can run in parallel; the board (P5)
without the beacon (P4) would have no "top models" data — polls alone could ship
earlier if desired by seeding the service with manual data.

## 6. Open questions / risks

1. **Control-plane hosting** — Cloudflare Worker (fast, free tier, no new box) vs
   netcup (self-hosted ethos, already paid for). Leaning Worker for ingest + static
   status page; the DE box stays protocol-only.
2. **Beacon consent default** for the *desktop app* specifically: wizard opt-in
   (recommended) vs default-on-with-banner. CLI stays off-by-default either way.
3. **Vote weighting** (sybil resistance): start 1-peer-1-vote + curated ballot;
   upgrade to reputation-weighted once M2.2 scores are beacon-visible.
4. **Settings-apply UX**: role restart on change is honest but interrupts serving;
   batch-apply with a "restart now" chip; hot-reload is a later agent feature.
5. **Windows/Linux wizard rows** need real verification (winget ids, cask names) —
   macOS ships first, other rows land with the release-CI platforms.
6. **Status-page abuse**: the ingest must rate-limit and require valid peer-id
   signatures so the map can't be inflated by a script (same key = same node).
