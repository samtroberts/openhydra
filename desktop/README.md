# OpenHydra Desktop

A Tauri v2 desktop application that lets non-technical users join the OpenHydra
peer-to-peer LLM inference network with a single button click — no terminal required.

## Prerequisites

| Dependency | Version | Install |
|---|---|---|
| Rust | 1.77+ | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |
| Node.js | 18+ | https://nodejs.org |
| OpenHydra | latest | `pip install openhydra` or `pip install -e ..` (from repo root) |

### macOS extras
```bash
xcode-select --install
```

### Linux extras (Debian/Ubuntu)
```bash
sudo apt update
sudo apt install libwebkit2gtk-4.1-dev libxdo-dev libssl-dev libayatana-appindicator3-dev librsvg2-dev
```

## Development

```bash
cd desktop
npm install
npm run dev
```

`npm run dev` launches the Tauri dev server with hot-reload for the frontend.
The Rust backend recompiles automatically on save.

## Build

```bash
cd desktop
npm install
npm run build
```

Distributable bundles are written to `src-tauri/target/release/bundle/`:

| Platform | Output |
|---|---|
| macOS | `macos/OpenHydra.app` + `.dmg` |
| Linux | `.deb`, `.rpm`, `.AppImage` |
| Windows | `.msi`, `.exe` (NSIS) |

## Icons

Before building for distribution, add a 1024x1024 PNG icon:

```bash
cp /path/to/your/icon.png src-tauri/icons/icon.png
npx tauri icon src-tauri/icons/icon.png
```

See `src-tauri/icons/README.txt` for full instructions.

## Architecture

```
desktop/
├── src/                        # Vanilla JS + CSS frontend (no framework)
│   ├── index.html              # Three-panel UI: sidebar, tabs, content
│   ├── app.js                  # Node lifecycle, chat SSE, network polling
│   └── styles.css              # Dark design system matching openhydra.co
└── src-tauri/                  # Rust Tauri v2 backend
    ├── src/
    │   ├── main.rs             # Entry point
    │   └── lib.rs              # Tauri commands: start_node, stop_node, …
    ├── Cargo.toml
    └── tauri.conf.json
```

### Tauri commands (Rust → JS bridge)

| Command | Args | Returns | Description |
|---|---|---|---|
| `start_node` | `peer_id, dht_url, model_id` | `Result<String>` | Spawns `openhydra-dht` (if localhost) then `openhydra-node` |
| `stop_node` | — | `Result<()>` | Kills both child processes |
| `is_node_running` | — | `bool` | Checks if node process is still alive |
| `get_node_pid` | — | `Option<u32>` | Returns the node process PID |

### Frontend panels

- **Chat** — OpenAI-compatible streaming chat via `POST /v1/chat/completions` (SSE)
- **Network** — Live peer table from `GET /v1/network/status`
- **About** — Version info and links

### Node API endpoints used

| Endpoint | Description |
|---|---|
| `POST /v1/chat/completions` | Streaming inference (SSE) |
| `GET /v1/network/status` | Peer list with model/region/reputation |
| `GET /v1/account/balance?peer_id=…` | HYDRA token balance |

## Settings

Settings are persisted in `localStorage` under the key `openhydra_settings`:

| Key | Default | Description |
|---|---|---|
| `peerId` | auto-generated | Unique identifier for this node |
| `modelId` | `openhydra-qwen3.5-0.8b` | Model to load and serve |
| `dhtUrl` | `http://127.0.0.1:8468` | DHT bootstrap URL |

## Troubleshooting

**"Failed to start DHT"**
Ensure `openhydra-dht` is on your PATH. Run `pip install openhydra` or install from
the repo root with `pip install -e .`.

**"Failed to start node"**
Same as above — ensure `openhydra-node` is on PATH.

**Chat returns no response**
The node takes 1-3 seconds to start. Wait for the status dot to turn green and
try again.

**Build fails — missing system libraries (Linux)**
Install the webkit/gtk development libraries listed in the Prerequisites section.
