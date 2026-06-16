<p align="center">
  <h1 align="center">OpenHydra</h1>
  <p align="center"><strong>Run AI in a herd, not a data center.</strong></p>
  <p align="center"><em>A peer-to-peer protocol for free, distributed AI inference.</em></p>
</p>

<p align="center">
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-stable-orange.svg" alt="Rust"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green.svg" alt="License"></a>
</p>

---

OpenHydra is a peer-to-peer **protocol** for distributed AI inference. It does not
run a model itself. Instead it sits in front of whatever inference engine you
already run — **Ollama, vLLM, LM Studio, llama.cpp, or any OpenAI-compatible
server** — and joins a global swarm that routes each request to a peer that has
the requested model, streams the completion back, and settles a co-signed
receipt. No central server. No API keys. No subscription. Think BitTorrent, for
AI inference.

The whole thing is a single Rust binary, `openhydra-agent`. There is no Python
in any role.

**Why OpenHydra?**

- **Bring your own engine.** Already running Ollama, vLLM, LM Studio, or
  llama.cpp? Join with a thin adapter and keep your stack — any Mac, NVIDIA, or
  AMD GPU, any platform. Requests run on the provider at native local speed.
- **No central server.** Peers discover each other over a Kademlia DHT + mDNS,
  with AutoNAT, UPnP, DCUtR hole-punching, and Circuit Relay v2 fallback for
  NAT'd nodes. The network is the computer.
- **Free, by reciprocity.** Give-to-get: serve inference to build standing when
  you consume. No tokens, no crypto, no monthly bill.
- **OpenAI-compatible.** The consumer gateway speaks the OpenAI Chat Completions
  API (streaming and non-streaming), so existing SDKs and tools point at it
  unchanged.
- **Keys stay put.** Each node's ed25519 identity key never leaves the Rust
  daemon — receipts are co-signed in-process; only signatures cross the wire.

See [docs/protocol.md](docs/protocol.md) for the protocol design and
[docs/ENGINE_COMPATIBILITY.md](docs/ENGINE_COMPATIBILITY.md) for the engine
support matrix.

---

## How it works

Two roles, one binary:

- **`provide`** — joins the swarm, detects the models your local engine serves,
  advertises their canonical ids on the DHT, and serves inbound inference by
  proxying to that engine. It never runs a model itself.
- **`serve`** — the consumer front door: an OpenAI-compatible HTTP/SSE gateway.
  For each request it discovers a provider for the requested model, streams the
  completion back over libp2p, and settles a co-signed receipt at EOS.

A request never touches a coordinator: the gateway talks to the provider
directly (or through a relay when both sides are NAT'd). Throughput is your
engine's native speed plus the routing/transport overhead.

---

## Quick Start

### Prerequisites

- A **Rust toolchain** (stable) — <https://rustup.rs>.
- An **inference engine** running locally that you want to share (Ollama, vLLM,
  LM Studio, or llama.cpp). On Apple Silicon, see
  [docs/INSTALL_MAC.md](docs/INSTALL_MAC.md).

### Build

```bash
git clone <repo-url> openhydra && cd openhydra
cargo build --release -p openhydra-agent
# → ./target/release/openhydra-agent
```

The whole workspace is pure Rust — no Python toolchain is required.

### Share your engine (provider)

Start your engine, then point a provider at it. Ollama is the default:

```bash
# Ollama on its standard port (11434)
./target/release/openhydra-agent provide --engine-kind ollama

# vLLM / LM Studio / llama.cpp / any OpenAI-compatible server
./target/release/openhydra-agent provide --engine-kind vllm       --engine http://127.0.0.1:8000
./target/release/openhydra-agent provide --engine-kind lm-studio  --engine http://127.0.0.1:1234
./target/release/openhydra-agent provide --engine-kind llama-cpp  --engine http://127.0.0.1:8080
./target/release/openhydra-agent provide --engine-kind openai     --engine http://127.0.0.1:8000
```

Useful flags: `--db <path>` persists the receipt ledger (redb) across restarts;
`--bootstrap <multiaddr>` dials a known peer/relay when mDNS can't reach the
swarm; `--reannounce-secs` tunes the DHT re-announce interval (default 120s).

### Consume (gateway)

Run the OpenAI-compatible gateway on another machine (or the same one, second
process):

```bash
./target/release/openhydra-agent serve            # binds 127.0.0.1:8080
# optional: require a key on /v1/* —
#   --api-key <key>   or   OPENHYDRA_API_KEY=<key>
```

Then point any OpenAI client at it:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "llama3.2",
    "messages": [{"role": "user", "content": "Write a haiku about peer-to-peer AI."}],
    "stream": true
  }'
```

The gateway also serves `GET /v1/models` (model ids discovered on the swarm) and
`GET /health`.

---

## Supported engines

| `--engine-kind` | Engine                         | Default port | Detection |
|-----------------|--------------------------------|--------------|-----------|
| `ollama`        | Ollama (native `/api/*`)       | 11434        | `/api/tags` — full canonical ids |
| `vllm`          | vLLM (OpenAI `/v1/*`)          | 8000         | `/v1/models` |
| `lm-studio`     | LM Studio (OpenAI `/v1/*`)     | 1234         | `/v1/models` |
| `llama-cpp`     | `llama-server` (`/props`)      | 8080         | `/props` → GGUF canonical id |
| `openai`        | Any OpenAI-compatible (Exo, LocalAI, …) | 8000 | `/v1/models` |

Details and per-engine notes: [docs/ENGINE_COMPATIBILITY.md](docs/ENGINE_COMPATIBILITY.md).

---

## Architecture

A three-crate Rust workspace:

- **`network/`** — the libp2p swarm: Kademlia DHT, mDNS, Circuit Relay v2,
  DCUtR, AutoNAT v2, UPnP, and the request/response proxy. Exposes a synchronous
  `NetworkHandle` to the agent and the `openhydra-bootstrap` binary.
- **`protocol/`** — pure protocol logic: canonical model-id resolution, capability
  records, the router (resolve → rank → route), co-signed receipts, and the redb
  receipt ledger.
- **`agent/`** — the runnable host (`openhydra-agent`): engine adapters, the
  provider loop, and the OpenAI-compatible consumer gateway.

Public **bootstrap nodes** (a small Rust `openhydra-bootstrap` binary on a few
VPS instances, dual-stack IPv4+IPv6 on :4001) seed DHT discovery and act as
relays for NAT'd peers. Deploy scripts live in [`ops/bootstrap/`](ops/bootstrap/).

---

## Benchmarks

Historical throughput data from earlier builds lives in
[`benchmarks/`](benchmarks/). Because OpenHydra now proxies to an external
engine, end-to-end throughput is the engine's own tokens/sec plus the transport
overhead (LAN direct ≈ a few ms; relayed cross-NAT paths add a hop). See
[docs/protocol.md](docs/protocol.md) for the current measurement methodology.

---

## Documentation

- [Protocol design](docs/protocol.md)
- [Engine compatibility](docs/ENGINE_COMPATIBILITY.md)
- [macOS install notes](docs/INSTALL_MAC.md)
- [Implementation plan](docs/PROTOCOL_IMPLEMENTATION_PLAN.md)
- [DHT robustness remediation](docs/DHT_ROBUSTNESS_REMEDIATION.md)

---

## Security & Privacy

- Each node's **ed25519 identity key never leaves the Rust daemon**. Receipts are
  co-signed in-process; only the resulting signatures and public keys cross the
  wire.
- Transport is authenticated and encrypted by libp2p (Noise + TLS over QUIC/TCP).
- Provider operators choose what their engine exposes; the protocol routes by
  model id and does not require sharing prompts with any third party beyond the
  serving provider.

See [SECURITY.md](SECURITY.md) to report vulnerabilities.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
