# Quick Start

Get OpenHydra running in under 5 minutes.

---

## Prerequisites

- Python 3.11+
- `make` and `protobuf-compiler` (`apt install protobuf-compiler` on Ubuntu)
- A locally hosted LLM (llama.cpp, Ollama, or a Hugging Face model)

---

## Single-node setup

### 1. Clone and install

```bash
git clone https://github.com/openhydra-ai/openhydra.git
cd openhydra

# Create virtualenv and install everything
make venv
source .venv/bin/activate
make install
make proto
```

### 2. Configure your node

```bash
# Copy and edit the defaults
cp openhydra_defaults.py my_node_config.py
```

Key settings in `openhydra_defaults.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_PEER_ID` | `"peer-local"` | Unique name for this node |
| `DEFAULT_MODEL_ID` | `"llama3-8b"` | Model your node serves |
| `DEFAULT_DHT_URL` | `"http://localhost:8468"` | Bootstrap node URL |
| `DEFAULT_COORDINATOR_HOST` | `"0.0.0.0"` | Coordinator listen address |
| `DEFAULT_COORDINATOR_PORT` | `8080` | Coordinator HTTP port |

### 3. Start the DHT bootstrap node

```bash
openhydra-dht --host 0.0.0.0 --port 8468
```

### 4. Start a peer node

```bash
openhydra-peer \
  --peer-id my-laptop \
  --model-id llama3-8b \
  --dht-url http://localhost:8468 \
  --api-host 0.0.0.0 \
  --api-port 50051
```

### 5. Start the coordinator

```bash
openhydra-coordinator \
  --dht-url http://localhost:8468 \
  --host 0.0.0.0 \
  --port 8080
```

### 6. Send your first request

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3-8b",
    "messages": [{"role": "user", "content": "Hello, OpenHydra!"}]
  }'
```

---

## Multi-node setup (3 bootstrap nodes)

For a production-grade cluster use the provided Docker Compose file:

```bash
# Single-machine multi-node (development)
docker compose up

# High-availability multi-node
docker compose -f docker-compose.ha.yml up
```

Or deploy to Linode nanodes — see the [Operator Guide](operator-guide.md) for full instructions.

---

## Desktop node (macOS / Windows / Linux)

The **OpenHydra Desktop** app lets you participate in the network from your laptop and earn HYDRA tokens.

1. Download the latest release from [GitHub Releases](https://github.com/openhydra-ai/openhydra/releases)
2. Open the app and complete the onboarding wizard
3. Select a model and click **Start Node**

Your node will appear in the network peer map within ~30 seconds.

---

## Interactive shell

```bash
pip install -e ".[shell]"
openhydra-shell
```

The interactive shell provides tab completion, command history, and a streaming chat interface against your local coordinator.

---

## Next steps

- Read the [API Reference](api-reference.md) to integrate OpenHydra into your application
- Deploy to the cloud with the [Operator Guide](operator-guide.md)
- Use the [Python SDK](sdk/python.md) or [TypeScript SDK](sdk/typescript.md) for programmatic access
