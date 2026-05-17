# OpenHydra — Mac Install Guide (Apple Silicon)

Run a sharded LLM across multiple Macs over the internet. No servers, no cloud — just peer-to-peer.

## Prerequisites

- macOS 14+ (Sonoma or newer) on Apple Silicon (M1/M2/M3/M4)
- 8GB+ RAM
- ~10GB free disk space (model weights + dependencies)
- IPv6 connectivity (most ISPs provide this; check with `ifconfig | grep inet6`)

## Install

Open Terminal and run these commands one by one:

```bash
# 1. Install Homebrew (skip if already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install system dependencies
brew install python@3.12 rust protobuf
xcode-select --install   # skip if Xcode CLI tools already installed

# 3. Clone OpenHydra and checkout the right branch
git clone https://github.com/samtroberts/openhydra.git
cd openhydra
git checkout feat/unified-libp2p-transport

# 4. Create a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 5. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-mlx.txt

# 6. Build the P2P networking module (Rust → Python, takes ~2-5 min)
pip install maturin
cd network && maturin develop --release && cd ..

# 7. Verify everything works
python3 -c "import openhydra_network; print('P2P networking: OK')"
python3 -c "import mlx.core; print('MLX (Metal GPU): OK')"
python3 -c "import torch; print('PyTorch: OK')"
```

## Run

OpenHydra shards a model's layers across two Macs. Mac1 runs layers 0-12 (and the coordinator), Mac2 runs layers 12-24.

### Mac2 — the second peer (layers 12-24)

Replace `<MAC1_IPV6>` and `<MAC1_PEER_ID>` with values Mac1 gives you after it starts.

```bash
cd openhydra
source .venv/bin/activate

python3 -m coordinator.node \
  --peer-id mac2 \
  --p2p-enabled \
  --push-mode \
  --pipeline-depth 1 \
  --runtime-model-id mlx-community/Qwen3.5-2B-MLX-8bit \
  --hf-model-id mlx-community/Qwen3.5-2B-MLX-8bit \
  --layer-start 12 --layer-end 24 \
  --p2p-bootstrap "/ip6/<MAC1_IPV6>/tcp/4001/p2p/<MAC1_PEER_ID>"
```

The first run downloads ~2.5GB of model weights from HuggingFace. After that, weights are cached locally.

**What you should see:**

```
INFO  announced to kademlia model_id=openhydra-qwen3.5-2b peer_id=mac2
INFO  peer_discovered peer_id=mac1 ...
INFO  connection_established peer_id=... endpoint=Dialer
```

Once you see `peer_discovered`, the two Macs are connected and sharding inference.

### If direct IPv6 doesn't work

If the direct `--p2p-bootstrap` connection fails (e.g. firewall blocks port 4001), omit that flag entirely — both Macs will discover each other via the public bootstrap DHT and connect through a relay:

```bash
python3 -m coordinator.node \
  --peer-id mac2 \
  --p2p-enabled \
  --push-mode \
  --pipeline-depth 1 \
  --runtime-model-id mlx-community/Qwen3.5-2B-MLX-8bit \
  --hf-model-id mlx-community/Qwen3.5-2B-MLX-8bit \
  --layer-start 12 --layer-end 24
```

This routes through Circuit Relay v2 nodes (US/EU/AP). Slower (~3 TPS vs ~12 TPS direct) but works behind any NAT.

## Sending prompts

Once both Macs show `peer_discovered`, send prompts to **Mac1** (the coordinator):

```bash
curl http://<MAC1_IP>:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openhydra-qwen3.5-2b",
    "messages": [{"role": "user", "content": "Explain P2P networking in 3 sentences"}],
    "max_tokens": 128
  }'
```

From Mac1 itself, use `http://127.0.0.1:8080/v1/chat/completions`.

## Expected performance

| Transport | TPS (Qwen3.5-2B, 2×Mac M1) |
|---|---|
| LAN (same WiFi/ethernet) | 12-17 tokens/sec |
| Direct IPv6 (cross-ISP) | 7-8 tokens/sec |
| Relay (fallback) | ~3 tokens/sec |

## Troubleshooting

| Problem | Fix |
|---|---|
| `protoc: command not found` | `brew install protobuf` |
| `maturin: command not found` | `pip install maturin` (make sure venv is active) |
| Rust build fails with linker error about `ring` | Try `MACOSX_DEPLOYMENT_TARGET=14.0 cd network && maturin develop --release && cd ..` |
| Rust build fails with "cargo lock" | `find ~/.cargo/registry -name '.cargo-lock' -delete` then retry |
| `import openhydra_network` fails | `cd network && maturin develop --release && cd ..` |
| No global IPv6 address | Check `ifconfig en0 \| grep inet6` — if only `fe80::` (link-local), your router/ISP doesn't provide IPv6. Use relay mode instead. |
| Peer not discovered after 60s | Check that both Macs have outbound access on TCP+UDP port 4001. Try relay mode (omit `--p2p-bootstrap`). |
| `ConnectionClosed` during generation | Relay hiccup — just retry the prompt |
| Very slow first response | Model is downloading from HuggingFace (~2.5GB for 8-bit). Check `~/.cache/huggingface/` |
| `MallocStackLogging` warnings | Harmless macOS diagnostic messages — ignore them |
