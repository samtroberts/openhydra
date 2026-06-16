# OpenHydra — Mac Install Guide (Apple Silicon)

Share the model you already run locally — and consume models other peers serve —
over a peer-to-peer network. No servers, no cloud. OpenHydra does **not** run a
model itself; it sits in front of a local inference engine (Ollama is easiest on a
Mac) and routes requests across the swarm.

## Prerequisites

- macOS 14+ (Sonoma or newer) on Apple Silicon (M1/M2/M3/M4)
- 8 GB+ RAM
- A Rust toolchain (we install it below)
- A local inference engine — this guide uses **[Ollama](https://ollama.com)**
- IPv6 connectivity helps for direct cross-NAT links but is not required (the swarm
  falls back to relays). Check with `ifconfig | grep inet6`.

## Install

Open Terminal and run these one by one:

```bash
# 1. Install Homebrew (skip if already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install Rust and an inference engine
brew install rust ollama

# 3. Clone and build the agent (links no Python; ~2–4 min the first time)
git clone https://github.com/samtroberts/openhydra.git
cd openhydra
cargo build --release -p openhydra-agent
# → ./target/release/openhydra-agent

# 4. Start Ollama and pull a small model
ollama serve &                 # runs on 127.0.0.1:11434
ollama pull llama3.2           # ~2 GB
```

## Run

OpenHydra has two roles, each a separate process. On a single Mac you can run both;
across two Macs, one provides and the other consumes (or both do both).

### Provider — share your Ollama models

```bash
./target/release/openhydra-agent provide --engine-kind ollama
```

You should see the node come up, detect Ollama, and announce its models:

```
openhydra-agent: node up — libp2p=12D3KooW… openhydra=…
openhydra-agent: announced 1 model(s) from http://127.0.0.1:11434
openhydra-agent: serving inbound requests, re-announcing every 120s (Ctrl-C to stop)
```

Add `--db ~/.openhydra/ledger.redb` to persist the receipt ledger across restarts.

### Consumer — run the OpenAI-compatible gateway

In a second terminal (same Mac or another):

```bash
./target/release/openhydra-agent serve            # binds 127.0.0.1:8080
```

On a LAN, mDNS discovers the provider automatically. Across networks, point the
gateway (or provider) at a known peer/relay with `--bootstrap <multiaddr>`; the
swarm hole-punches (DCUtR) or falls back to Circuit Relay v2 behind NAT.

## Sending prompts

Point any OpenAI-compatible client at the gateway:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2",
    "messages": [{"role": "user", "content": "Explain P2P networking in 3 sentences"}],
    "stream": true
  }'
```

The gateway also serves `GET /v1/models` (model ids discovered on the swarm) and
`GET /health`. The model id must match what a provider advertises — check
`/v1/models` if a request returns `503 no_provider`.

## Notes on performance

End-to-end throughput is your engine's native tokens/sec plus transport overhead.
A direct LAN or IPv6 link adds only a few milliseconds; a relayed cross-NAT path
adds one hop. The engine (Ollama/MLX/llama.cpp) — not OpenHydra — determines raw
generation speed.

## Troubleshooting

| Problem | Fix |
|---|---|
| `cargo: command not found` | `brew install rust`, then reopen the terminal |
| Build fails with a linker error about `ring` | `MACOSX_DEPLOYMENT_TARGET=14.0 cargo build --release -p openhydra-agent` |
| `503 no_provider` from the gateway | No provider is serving that model id — start a `provide` agent and confirm the id in `GET /v1/models` |
| Provider logs `0 models` | Pull a model in your engine (`ollama pull …`) and restart the provider so it re-detects |
| Peer not discovered on a LAN | Some Wi-Fi routers block mDNS multicast — pass `--bootstrap` with the other node's multiaddr |
| No global IPv6 address | `ifconfig en0 \| grep inet6` shows only `fe80::` (link-local) → rely on relay mode (the default fallback) |
| `MallocStackLogging` warnings | Harmless macOS diagnostic messages — ignore them |
