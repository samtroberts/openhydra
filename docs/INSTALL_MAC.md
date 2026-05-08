# OpenHydra — Mac Install Guide (Apple Silicon)

Run a sharded LLM across multiple Macs over the internet. No servers, no cloud — just peer-to-peer.

## Prerequisites

- macOS 14+ (Sonoma or newer) on Apple Silicon (M1/M2/M3/M4)
- 8GB+ RAM
- ~10GB free disk space (model weights + dependencies)

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

# 6. Build the P2P networking module (Rust → Python, takes ~2 min)
pip install maturin
cd network && maturin develop --release && cd ..

# 7. Verify everything works
python3 -c "import openhydra_network; print('P2P networking: OK')"
python3 -c "import mlx.core; print('MLX (Metal GPU): OK')"
python3 -c "import torch; print('PyTorch: OK')"
```

## Run (as the second peer — layers 12-24)

```bash
cd openhydra
source .venv/bin/activate

python3 -m coordinator.node \
  --peer-id mac2 \
  --model-id openhydra-qwen3.5-2b \
  --runtime-model-id mlx-community/Qwen3.5-2B-MLX-8bit \
  --hf-model-id mlx-community/Qwen3.5-2B-MLX-8bit \
  --layer-start 12 --layer-end 24 \
  --shard-index 1 --total-shards 2
```

The first run downloads ~5GB of model weights from HuggingFace. After that, weights are cached locally.

You should see logs like:
```
INFO  peer_announced model=openhydra-qwen3.5-2b layers=[12,24]
INFO  peer_discovered peer_id=mac1 ...
```

Once you see `peer_discovered`, the two Macs are connected and ready.

## Troubleshooting

| Problem | Fix |
|---|---|
| `protoc: command not found` | `brew install protobuf` |
| `maturin: command not found` | `pip install maturin` (make sure venv is active) |
| Rust build fails with "cargo lock" | `find ~/.cargo/registry -name '.cargo-lock' -delete` then retry |
| `import openhydra_network` fails | `cd network && maturin develop --release && cd ..` |
| Peer not discovered after 60s | Check internet connection; both Macs need outbound UDP+TCP on port 4001 |
| `ConnectionClosed` during generation | Relay hiccup — just retry the prompt |
| Very slow first response | Model is downloading from HuggingFace (~5GB). Check `~/.cache/huggingface/` |
