<p align="center">
  <h1 align="center">OpenHydra</h1>
  <p align="center"><strong>Run AI in a herd, not a data center.</strong></p>
  <p align="center"><em>An open, peer-to-peer network for AI inference — bring the engine you already run.</em></p>
</p>

<p align="center">
  <a href="https://github.com/samtroberts/openhydra/releases/latest"><img src="https://img.shields.io/github/v/release/samtroberts/openhydra" alt="Latest release"></a>
  <a href="https://github.com/samtroberts/openhydra/actions/workflows/rust-network.yml"><img src="https://github.com/samtroberts/openhydra/actions/workflows/rust-network.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/platforms-macOS%20(Apple%20Silicon)%20%7C%20Linux%20x86--64%20%26%20arm64%20%7C%20Windows%20x64-lightgrey" alt="Platforms">
</p>

- **Bring your own engine.** Ollama, LM Studio, llama.cpp, vLLM, SGLang, Exo or ComfyUI. OpenHydra
  doesn't run models itself; it shares the ones your engine already serves.
- **One local endpoint.** OpenAI and Anthropic APIs on `http://127.0.0.1:16527`, wired into
  Claude Code, OpenCode, Hermes and Pi by one command.
- **Checked, not trusted.** Co-signed receipts, audits, token recounts and optional log-prob
  checks, and private to your own machines when you want. [What is and isn't checked](#trust-and-its-limits).

## Quick start

**The app.** Download it from [openhydra.co/downloads](https://openhydra.co/downloads) or the
[latest release](https://github.com/samtroberts/openhydra/releases/latest): `.dmg` (macOS,
Apple Silicon), `-setup.exe` / `.msi` (Windows x64), `.AppImage` / `.deb` / `.rpm` (Linux x86-64).

**The command line** (macOS Apple Silicon, Linux x86-64 and Linux arm64):

```bash
curl -fsSL https://openhydra.co/install.sh | sh
```

```bash
openhydra provide          # share the models your local engine serves
openhydra serve            # local gateway on http://127.0.0.1:16527
openhydra launch claude    # run Claude Code against that gateway (start `serve` first)
```

---

## How it works

One binary, `openhydra-agent` (installed as `openhydra`), plays two roles, alone or together:

- **`provide`** finds the engines running on this machine (`--engine-kind auto`, the default,
  probes every supported engine on its standard port), announces the models you choose to
  share on a Kademlia DHT, and answers requests by passing them to your engine.
- **`serve`** is the local gateway your tools talk to. For each request it finds providers of
  the requested model, picks one by measured round-trip time and reported load (keeping a
  conversation's later turns on the same provider, whose engine has its prompt cached), streams
  the answer back over libp2p, and both sides sign a receipt.

Peers find each other through the DHT, mDNS and a few bootstrap nodes; libp2p handles NAT
(AutoNAT, UPnP, DCUtR hole-punching) and falls back to Circuit Relay v2 when no direct path
works. Connections are encrypted in transit (Noise over TCP, TLS 1.3 over QUIC). A relay
forwards encrypted bytes; **the provider that serves a request reads it in plain text**,
because its engine has to.

The network is free today: there is no token, and payments aren't live. Serving gives your
requests priority on other providers (give-to-get: a short delay for heavy consumers who
don't serve, never a refusal). The repo also contains draft settlement contracts in
[`onchain/`](onchain/); they are not audited, not deployed, and not used by the app.

## Gateway endpoints

| Route | |
|---|---|
| `POST /v1/chat/completions` | OpenAI Chat Completions, streaming and non-streaming |
| `POST /v1/messages` | Anthropic Messages API (what Claude Code speaks) |
| `POST /v1/embeddings` | OpenAI embeddings |
| `GET /v1/models` | models found on the network, plus `openhydra/auto` |
| `GET /health` | liveness, no key needed |
| `GET /metrics` | Prometheus metrics, no key needed |

`model: "openhydra/auto"` lets the gateway pick a live model. The gateway binds
`127.0.0.1:16527` by default.

**Keys.** `openhydra serve` is open on loopback unless you set `--api-key` (or
`OPENHYDRA_API_KEY`). Since v0.3.21 the desktop app's gateway requires its own key, created on
first run at `~/.openhydra/gateway.key`; Connect and `openhydra launch` write it into your
tools for you. Send it as `Authorization: Bearer <key>` or `x-api-key: <key>`.

```bash
curl http://127.0.0.1:16527/v1/chat/completions \
  -H "Authorization: Bearer $(cat ~/.openhydra/gateway.key)" \
  -H 'Content-Type: application/json' \
  -d '{"model": "openhydra/auto", "messages": [{"role": "user", "content": "Hello!"}], "stream": true}'
```

## Connect your tool

```bash
openhydra launch claude      # Claude Code, through the gateway's own Anthropic API
openhydra launch opencode    # also: hermes, pi
openhydra connect <tool>     # only write the config;  openhydra disconnect <tool> restores it
```

`launch` writes the tool's config (keeping a `.openhydra.bak` backup) and starts the tool. It
doesn't start a gateway; run `openhydra serve` or the app first. In the app, the same thing is
**Connectors → Connect**.

| Tool | Config written | What it points at |
|---|---|---|
| Claude Code | `~/.claude/settings.json` (`env.ANTHROPIC_BASE_URL`, `env.ANTHROPIC_API_KEY`) | `http://127.0.0.1:16527` |
| OpenCode | `~/.config/opencode/opencode.json` (provider `openhydra`) | `http://127.0.0.1:16527/v1` |
| Hermes | `~/.hermes/config.yaml` (`model:` block, `provider: custom`) | `http://127.0.0.1:16527/v1` |
| Pi | `~/.pi/agent/models.json` (provider `openhydra`) | `http://127.0.0.1:16527/v1` |
| Continue (`connect` only, no `launch`) | `~/.continue/config.yaml` | `http://127.0.0.1:16527/v1` |

Any other OpenAI-compatible tool works with base URL `http://127.0.0.1:16527/v1`, your gateway
key, and model `openhydra/auto`.

## Share your machine

```bash
openhydra provide                                   # auto: every engine it finds
openhydra provide --engine-kind ollama              # or one engine
openhydra provide --engine-kind exo --engine http://127.0.0.1:52415
openhydra provide --share-models llama3.2,qwen3     # only these models
```

Useful flags: `--max-concurrency` (parallel requests, default 8), `--share-models` /
`--share-policy-file` (what to share), `--db <path>` (keep the receipt ledger across restarts),
`--aup-*` (optional request limits, off by default). In the app, sharing is per model, from the
Share screen; the app has no contribution cap yet.

Sharing runs whenever the agent runs. It doesn't detect whether you're using the machine.

## Private groups and private deployments

- **Swarms** (private groups). Create one, admit members by exchanging a copy-paste request and
  a signed credential, and mark models "Only my swarms". Your provider serves those models only
  to members and never announces them on the public DHT. `openhydra swarm` does the same on the
  CLI (`create`, `request`, `approve`, `accept`, `list`, `revoke`, `leave`, `forget`).
- **`.openhydra` cards.** A signed, secret-free description of one provider and one model.
  `openhydra card export` / `verify` / `import` on the CLI; the app opens card files and
  `openhydra:` links. A card for a private model is useless without a swarm credential, and a
  request for it goes only to that swarm's providers (and your own machine): never to a public
  provider, even one announcing the same model name; if none can serve it, the request fails.
  For a public card, `card fallback` sets what happens when its provider is offline: `ask`,
  `same-version` (the default: another provider of the exact same version) or `never`.
  *Limit:* a model requested by name, without a card, is looked up on the public network.
- **Private deployments.** `--public-relays off` (app: Settings → Network → "Use OpenHydra's
  public network" off) uses only bootstrap peers you list, and refuses OpenHydra's own servers.
  To keep outsiders off the network altogether:
  - `openhydra network-key --out swarm.key`, then `--private-network-key-file swarm.key` on
    every node: TCP connections and relay circuits are encrypted with the shared key, so a node
    without it can't connect or relay. QUIC can't carry the key and is off unless you add
    `--private-network-quic` with a peer allowlist.
  - `--allow-peers-file <file>` (peer ids, one per line) and `--allow-swarm-peers` (the peers
    your swarm credentials name, kept current) refuse every other peer on TCP, QUIC and relay
    circuits. Your `--bootstrap` peers are always allowed.
  - `--network-id <name>` keeps nodes given different subsets of your bootstraps on one network.

## Trust and its limits

What the gateway checks:

- **Receipts.** Every request to another provider ends in a receipt both sides sign (model,
  token count, keys, nonce, time).
- **Token recount.** The gateway recounts the delivered answer with the model's own tokenizer
  and signs for the lower count, so a provider can't over-bill on the text it sent.
- **Audits.** A sample of providers (2–50%, by reputation) is checked by sending the same
  greedy request to other providers and comparing answers. Reputation is kept per gateway.
- **Log-prob checks** (opt-in) compare a provider's token probabilities on reference texts with
  the claimed model's.

What it doesn't check: that a provider doesn't read or log your prompt; that it runs exactly the
weights or quantization it claims (except through the opt-in log-prob check, on challenge
requests); and whether answers are good. A low-reputation provider is ranked lower, not banned.

## Privacy

- **The provider that serves your request reads it in plain text.** Don't send what you
  wouldn't show a stranger, or keep it on your own machines (private deployments above).
- **Audit replay.** In `openhydra serve` it is **on by default**: a request with
  `temperature: 0` to a public model may be sent again, as is, to up to three other providers
  of that model (usually one) so their answers can be compared. Private-swarm requests are never
  replayed. Turn it off with `--audit-replay off`. **The desktop app has it off** (Settings →
  Network → "Check providers with my own requests").
- **Token counts.** The gateway downloads the tokenizer index from `dl.openhydra.co` once a day
  (the whole list, so it says nothing about your models) and, the first time you use a listed
  model, that model's tokenizer from huggingface.co, which tells Hugging Face that this gateway
  uses that model. `--no-tokenizer-downloads` (app: "Check billed token counts") turns that off.
- **Network metadata.** Bootstrap nodes and relays see your peer id and IP address; providers
  announce their model names on the DHT.
- **No accounts and no telemetry.** Your identity is an ed25519 key in `~/.openhydra/` that
  never leaves the agent; receipts are signed in-process. The app keeps chat history in
  `~/.openhydra/sessions.json` (not encrypted) and checks `dl.openhydra.co` for updates.

Full policy: [openhydra.co/privacy](https://openhydra.co/privacy).

## Supported engines

| `--engine-kind` | Engine | Default port | Detection |
|---|---|---|---|
| `auto` (default) | every engine below that answers (and moex, port 8765) | — | probes each standard port, re-probes on re-announce |
| `ollama` | Ollama (native `/api/*`) | 11434 | `/api/tags` |
| `lm-studio` | LM Studio | 1234 | `/v1/models` |
| `llama-cpp` | `llama-server` | 8080 | `/props` → GGUF model id |
| `vllm` | vLLM | 8000 | `/v1/models` |
| `sglang` | SGLang | 30000 | `/v1/models` |
| `exo` | Exo cluster | 52415 | `/state` (placed, ready models only) |
| `comfyui` | ComfyUI (image / audio / video) | 8188 | `/object_info/CheckpointLoaderSimple`; `--comfyui-workflow-dir` for your own workflows |
| `openai` | any OpenAI-compatible server | 8000 | `/v1/models` |

## Downloads and signing

- **Desktop app:** macOS (Apple Silicon) `.dmg`, Windows x64 `-setup.exe` and `.msi`, Linux
  x86-64 `.AppImage`, `.deb` and `.rpm`. The macOS app is signed and notarized by Apple. The
  Windows build isn't code-signed yet, so SmartScreen may warn on first run. The app updates
  itself from `dl.openhydra.co`.
- **CLI:** `openhydra-agent-<target>` for `aarch64-apple-darwin`, `x86_64-unknown-linux-gnu`,
  `aarch64-unknown-linux-gnu` and `x86_64-pc-windows-msvc`, each with `openhydra-scorer-<target>`,
  the scorer the opt-in log-prob check runs (and `openhydra-mlx-scorer` on the Mac). The binaries
  themselves aren't code-signed; since 0.3.22 each
  platform's `SHA256SUMS` file is signed with minisign. `install.sh` always checks the SHA-256
  and also checks the signature when `minisign` is installed. Public key:
  `RWT1WCE7c96xPsN2V6HZHspfB29Ucvh6rePhj+Y6bmCSzl54zcIGVMn0`.
- Linux arm64 gets the CLI only, no desktop app yet. No Intel Mac builds; build from source.

## Build from source

Needs [Rust](https://rustup.rs) 1.88 or newer and `protoc`, the Protocol Buffers compiler
(`brew install protobuf`, `apt install protobuf-compiler` or `choco install protoc`). No Python.

```bash
git clone https://github.com/samtroberts/openhydra && cd openhydra
cargo build --release -p openhydra-agent
# → ./target/release/openhydra-agent
```

## Repository map

- `agent/` — the `openhydra-agent` binary: engine adapters, provider, gateway, connectors.
- `network/` — libp2p swarm (Kademlia, mDNS, relay, DCUtR, AutoNAT, UPnP) and the
  `openhydra-bootstrap` binary. `network/patched-deps/` holds the libp2p crates we carry fixes
  for; each one's changes from upstream are noted in its CHANGELOG.md (in the code, marked
  "OpenHydra", for libp2p-quic).
- `protocol/` — model ids, routing, receipts, verification, the redb ledger.
- `desktop/` — the Tauri desktop app.
- `engine-mlx/` — the built-in Mac engine (Apple MLX in-process; a build feature, not in the
  release builds yet).
- `scorer/` — the log-prob scorer builds (patched llama.cpp and mistral.rs).
- `ops/` — bootstrap deploy scripts. `onchain/` — draft contracts, not used.
- `benchmarks/` — historical numbers from before the bring-your-own-engine design.

## Security

Report vulnerabilities privately; see [SECURITY.md](SECURITY.md).

## License

Apache 2.0 — see [LICENSE](LICENSE).
