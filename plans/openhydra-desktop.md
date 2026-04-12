# OpenHydra Desktop — Architecture Plan (LLMule-style)

## Goal

Build a native desktop client at `/Users/sam/Documents/openhydra-desktop` using **Electron + React (Vite) + TailwindCSS**, matching LLMule's architecture exactly. The app wraps `openhydra-node` as a child process, with IPC bridge for process control and streaming logs.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│  Renderer Process (React + Vite + Tailwind)     │
│  ┌───────────────────────────────────────────┐  │
│  │  App.jsx                                  │  │
│  │  ├── NodeProvider  (node state/control)   │  │
│  │  └── ChatProvider  (conversations)        │  │
│  │      ├── Dashboard (toggle, config, logs) │  │
│  │      └── ChatView  (inference UI)         │  │
│  └───────────────────────────────────────────┘  │
│       │  window.electron.*  (context-isolated)  │
│       │  ipcRenderer.invoke / .on               │
├───────┼─────────────────────────────────────────┤
│  Preload (preload.js)                           │
│       │  contextBridge.exposeInMainWorld         │
├───────┼─────────────────────────────────────────┤
│  Main Process (electron/main.js)                │
│  ├── electron-store (settings persistence)      │
│  ├── node/manager.js  (spawn/kill openhydra)    │
│  ├── node/handlers.js (IPC: start/stop/status)  │
│  ├── llm/handlers.js  (IPC: chat, models)       │
│  └── tray.js (system tray toggle)               │
│       │                                         │
│       │  child_process.spawn("openhydra-node")  │
│       ▼                                         │
│  openhydra-node (Python subprocess)             │
│  ├── gRPC peer on :50051                        │
│  ├── HTTP coordinator on :8080                  │
│  └── stdout/stderr → piped to renderer          │
└─────────────────────────────────────────────────┘
```

---

## 1. Project Structure (mirrors LLMule exactly)

```
openhydra-desktop/
├── package.json              # ESM, Electron 33+, Vite 6+
├── vite.config.js            # React + vite-plugin-electron
├── tailwind.config.js        # @tailwindcss/typography
├── postcss.config.js
├── index.html                # Splash screen + React root
├── electron-builder.json     # macOS DMG, Windows NSIS, Linux AppImage
├── .env                      # API endpoints
│
├── electron/                 # Main process (Node.js)
│   ├── main.js               # BrowserWindow, IPC hub, power monitor
│   ├── preload.js            # contextBridge → window.electron
│   ├── config.js             # Dev/prod API URLs
│   ├── tray.js               # System tray (Start/Stop node)
│   ├── updater.js            # electron-updater auto-updates
│   │
│   ├── node/                 # OpenHydra node management
│   │   ├── manager.js        # NodeManager class (spawn/kill child_process)
│   │   └── handlers.js       # IPC: node:start, node:stop, node:status
│   │
│   ├── llm/                  # LLM inference layer
│   │   ├── handlers.js       # IPC: llm:chat, llm:cancel, llm:models
│   │   └── client.js         # OpenHydraClient (HTTP to localhost:8080)
│   │
│   └── auth/                 # Future: DHT identity / HYDRA auth
│       └── handlers.js       # IPC: auth:getBalance, auth:getStatus
│
├── src/                      # Renderer process (React)
│   ├── main.jsx              # ReactDOM.createRoot
│   ├── App.jsx               # Provider stack + view routing
│   ├── styles/
│   │   └── index.css         # Tailwind imports + custom styles
│   │
│   ├── contexts/
│   │   ├── NodeContext.jsx    # Node state (running/stopped, logs, config)
│   │   └── ChatContext.jsx   # Conversations, streaming, model selection
│   │
│   ├── components/
│   │   ├── layouts/
│   │   │   └── MainLayout.jsx    # Header + nav tabs + status bar
│   │   ├── dashboard/
│   │   │   ├── Dashboard.jsx     # Main dashboard view
│   │   │   ├── NodeToggle.jsx    # Start/Stop button
│   │   │   ├── StatusCards.jsx   # Status, API Port, HYDRA earned
│   │   │   ├── ModelSelector.jsx # Model dropdown + VRAM badge
│   │   │   ├── RamSlider.jsx     # RAM allocation slider
│   │   │   └── LogTerminal.jsx   # Live stdout/stderr terminal
│   │   ├── chat/
│   │   │   ├── ChatView.jsx      # Chat interface wrapper
│   │   │   ├── ChatInterface.jsx # Messages + input
│   │   │   ├── ChatMessage.jsx   # Message bubble (markdown)
│   │   │   ├── ChatInput.jsx     # Text input + send
│   │   │   └── ChatSidebar.jsx   # Conversation list
│   │   └── NetworkStats.jsx      # Footer: node uptime, tokens earned
│   │
│   └── assets/
│       └── logo.svg
│
├── build/
│   ├── icon.png
│   ├── icon.icns
│   └── entitlements.mac.plist
│
└── assets/
    └── icon.png
```

---

## 2. Electron Main Process

### 2a — `electron/main.js` (mirrors LLMule's main.js)

```javascript
import { app, BrowserWindow, ipcMain } from 'electron';
import Store from 'electron-store';
import { setupNodeHandlers } from './node/handlers.js';
import { setupLLMHandlers } from './llm/handlers.js';
import { setupAuthHandlers } from './auth/handlers.js';
import { SystemTray } from './tray.js';

const store = new Store();
let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200, height: 800,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
    },
    titleBarStyle: 'hiddenInset',  // macOS native look
  });
  // Load Vite dev server or built index.html
}

app.whenReady().then(() => {
  createWindow();

  // Persistent settings via electron-store
  ipcMain.handle('store:get', (_, key) => store.get(key));
  ipcMain.handle('store:set', (_, key, val) => store.set(key, val));

  // Platform info
  ipcMain.handle('system:getPlatformInfo', () => ({
    os: process.platform, arch: process.arch
  }));

  // Setup domain-specific IPC handlers
  setupNodeHandlers(ipcMain, mainWindow, store);
  setupLLMHandlers(ipcMain, mainWindow, store);
  setupAuthHandlers(ipcMain, mainWindow, store);

  // System tray
  new SystemTray(mainWindow);

  // Power monitor (suspend/resume)
  powerMonitor.on('suspend', () => mainWindow.webContents.send('system:suspend'));
  powerMonitor.on('resume', () => mainWindow.webContents.send('system:resume'));
});
```

### 2b — `electron/preload.js` (context-isolated API bridge)

Exposes `window.electron` to the renderer — **exact same pattern as LLMule**:

```javascript
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electron', {
  store: {
    get: (key) => ipcRenderer.invoke('store:get', key),
    set: (key, val) => ipcRenderer.invoke('store:set', key, val),
  },
  node: {
    start: (config) => ipcRenderer.invoke('node:start', config),
    stop: () => ipcRenderer.invoke('node:stop'),
    getStatus: () => ipcRenderer.invoke('node:status'),
    onLog: (cb) => ipcRenderer.on('node:log', (_, data) => cb(data)),
    onStatusChange: (cb) => ipcRenderer.on('node:status-change', (_, s) => cb(s)),
  },
  llm: {
    chat: (params) => ipcRenderer.invoke('llm:chat', params),
    cancel: (reqId) => ipcRenderer.invoke('llm:cancel', reqId),
    getModels: () => ipcRenderer.invoke('llm:models'),
    onStream: (reqId, cb) => ipcRenderer.on(`llm:stream:${reqId}`, (_, d) => cb(d)),
  },
  auth: {
    getBalance: () => ipcRenderer.invoke('auth:getBalance'),
    getStatus: () => ipcRenderer.invoke('auth:getStatus'),
  },
  system: {
    getPlatformInfo: () => ipcRenderer.invoke('system:getPlatformInfo'),
    onSuspend: (cb) => ipcRenderer.on('system:suspend', cb),
    onResume: (cb) => ipcRenderer.on('system:resume', cb),
  },
  shell: {
    openExternal: (url) => ipcRenderer.invoke('shell:openExternal', url),
  },
});
```

### 2c — `electron/node/manager.js` (child process management)

This is the OpenHydra-specific piece (LLMule doesn't have this — they talk to external services via HTTP). We spawn `openhydra-node` as a child process:

```javascript
import { spawn } from 'child_process';

class NodeManager {
  constructor() {
    this.process = null;
    this.status = 'stopped'; // stopped | starting | running | error
  }

  start(config, onLog, onStatusChange) {
    const args = [
      '--peer-id', config.peerId || 'desktop-node',
      '--model-id', config.modelId || 'Qwen/Qwen3.5-0.8B',
      '--api-port', String(config.apiPort || 8080),
    ];

    // Resolve the openhydra-node binary:
    // In dev: use the Python venv directly
    // In production: use bundled PyInstaller binary from app resources
    const nodeBin = app.isPackaged
      ? path.join(process.resourcesPath, 'openhydra-node')
      : 'openhydra-node';  // assumes on PATH from pip install -e .

    this.process = spawn(nodeBin, args, { stdio: ['ignore', 'pipe', 'pipe'] });
    this.status = 'starting';
    onStatusChange('starting');

    this.process.stdout.on('data', (chunk) => {
      const line = chunk.toString().trim();
      onLog({ stream: 'stdout', line, timestamp: Date.now() });
      // Detect when coordinator is ready
      if (line.includes('Coordinator API listening')) {
        this.status = 'running';
        onStatusChange('running');
      }
    });

    this.process.stderr.on('data', (chunk) => {
      onLog({ stream: 'stderr', line: chunk.toString().trim(), timestamp: Date.now() });
    });

    this.process.on('close', (code) => {
      this.status = 'stopped';
      this.process = null;
      onStatusChange('stopped');
    });

    this.process.on('error', (err) => {
      this.status = 'error';
      onStatusChange('error');
    });
  }

  stop() {
    if (this.process) {
      this.process.kill('SIGTERM');  // graceful shutdown
    }
  }

  getStatus() {
    return { status: this.status, pid: this.process?.pid || null };
  }
}
```

### 2d — `electron/node/handlers.js` (IPC handlers for node control)

```javascript
export function setupNodeHandlers(ipcMain, mainWindow, store) {
  const manager = new NodeManager();

  ipcMain.handle('node:start', (_, config) => {
    manager.start(config,
      (log) => mainWindow.webContents.send('node:log', log),
      (status) => mainWindow.webContents.send('node:status-change', status),
    );
  });

  ipcMain.handle('node:stop', () => manager.stop());
  ipcMain.handle('node:status', () => manager.getStatus());

  // Kill on app quit
  app.on('before-quit', () => manager.stop());
}
```

### 2e — `electron/llm/client.js` (HTTP client to local coordinator)

```javascript
import axios from 'axios';

class OpenHydraClient {
  constructor(baseUrl = 'http://127.0.0.1:8080') {
    this.api = axios.create({ baseURL: baseUrl, timeout: 60000 });
  }

  async chat(model, messages, options = {}) {
    const res = await this.api.post('/v1/chat/completions', {
      model, messages,
      temperature: options.temperature || 0.7,
      max_tokens: options.maxTokens || 1024,
      stream: options.stream || false,
    });
    return res.data;
  }

  async chatStream(model, messages, options = {}) {
    const res = await this.api.post('/v1/chat/completions', {
      model, messages,
      temperature: options.temperature || 0.7,
      max_tokens: options.maxTokens || 1024,
      stream: true,
    }, { responseType: 'stream' });
    return res.data;  // readable stream of SSE chunks
  }

  async getModels() {
    const res = await this.api.get('/v1/models');
    return res.data;
  }

  async getHealth() {
    const res = await this.api.get('/health');
    return res.data;
  }

  async getBalance() {
    const res = await this.api.get('/v1/account/balance');
    return res.data;
  }
}
```

### 2f — `electron/llm/handlers.js` (chat IPC — mirrors LLMule's handlers.js)

```javascript
export function setupLLMHandlers(ipcMain, mainWindow, store) {
  const client = new OpenHydraClient();
  const activeRequests = new Map();  // reqId → AbortController

  ipcMain.handle('llm:chat', async (event, { model, messages, requestId, stream, ...opts }) => {
    const controller = new AbortController();
    activeRequests.set(requestId, controller);

    try {
      if (stream) {
        const dataStream = await client.chatStream(model, messages, opts);
        // Parse SSE and send chunks to renderer
        for await (const chunk of dataStream) {
          const lines = chunk.toString().split('\n').filter(l => l.startsWith('data: '));
          for (const line of lines) {
            const json = line.slice(6);
            if (json === '[DONE]') continue;
            const parsed = JSON.parse(json);
            const content = parsed.choices?.[0]?.delta?.content || '';
            if (content) {
              event.sender.send(`llm:stream:${requestId}`, { content });
            }
          }
        }
        return { done: true };
      } else {
        return await client.chat(model, messages, opts);
      }
    } finally {
      activeRequests.delete(requestId);
    }
  });

  ipcMain.handle('llm:cancel', (_, requestId) => {
    activeRequests.get(requestId)?.abort();
    activeRequests.delete(requestId);
  });

  ipcMain.handle('llm:models', () => client.getModels());
}
```

### 2g — `electron/tray.js` (system tray — mirrors LLMule)

Context menu with:
- **Start Node / Stop Node** — sends `node:start` or `node:stop` via IPC
- **Show Window** — `mainWindow.show()`
- **Quit** — `app.quit()`

---

## 3. React Renderer

### 3a — `src/App.jsx` (mirrors LLMule's provider stack + tab navigation)

```jsx
<NodeProvider>
  <ChatProvider>
    <MainLayout>
      {activeView === 'dashboard' && <Dashboard />}
      {activeView === 'chat' && <ChatView />}
    </MainLayout>
  </ChatProvider>
</NodeProvider>
```

Two tabs in header: **Dashboard** (node control + logs) and **Chat** (inference UI).

### 3b — `src/contexts/NodeContext.jsx` (OpenHydra-specific, replaces LLMule's NetworkContext)

```jsx
// State:
// - status: 'stopped' | 'starting' | 'running' | 'error'
// - logs: [{ stream, line, timestamp }]  (capped at 500)
// - config: { modelId, peerId, apiPort, ramAllocation }
// - balance: { hydra, credits }

// Methods:
// - startNode(config) → window.electron.node.start(config)
// - stopNode()       → window.electron.node.stop()

// Effects:
// - Listen to window.electron.node.onLog → append to logs
// - Listen to window.electron.node.onStatusChange → update status
// - Poll balance every 10s when running
// - system:suspend → auto-stop node
// - system:resume  → prompt to restart
```

### 3c — `src/contexts/ChatContext.jsx` (mirrors LLMule's ChatContext exactly)

Same patterns:
- Conversations array persisted to localStorage
- `sendMessage()` → `window.electron.llm.chat()` with streaming callback
- Request cancellation via `window.electron.llm.cancel(requestId)`
- Markdown rendering in chat messages
- Model selection tied to NodeContext's active model

### 3d — Components

| Component | LLMule Equivalent | What it does |
|-----------|-------------------|-------------|
| `MainLayout.jsx` | `layouts/MainLayout.jsx` | Header with tabs (Dashboard / Chat), status badge, app version |
| `Dashboard.jsx` | — (new) | Wraps NodeToggle + StatusCards + ModelSelector + RamSlider + LogTerminal |
| `NodeToggle.jsx` | — (new, inspired by LLMule connect/disconnect) | Green Start / Red Stop / Amber Starting button with lucide-react icons |
| `StatusCards.jsx` | `NetworkStats.jsx` | Three cards: Node Status, API Port :8080, HYDRA Earned |
| `ModelSelector.jsx` | `chat/ModelSelector.jsx` | Dropdown with HuggingFace model IDs + VRAM badges |
| `RamSlider.jsx` | — (new) | Range slider, greys out models exceeding allocation |
| `LogTerminal.jsx` | — (new) | Dark monospace terminal, auto-scroll, colour-coded log levels |
| `ChatView.jsx` | `chat/ChatView.jsx` | Chat interface wrapper |
| `ChatInterface.jsx` | `chat/ChatInterface.jsx` | Message list + input area |
| `ChatMessage.jsx` | `chat/ChatMessage.jsx` | Markdown-rendered message bubble |
| `ChatInput.jsx` | `chat/ChatInput.jsx` | Text input + send button |

---

## 4. Python Binary Bundling

### Development Mode
No binary needed. The app spawns `openhydra-node` directly from the system PATH (installed via `pip install -e .` in the main repo). `app.isPackaged === false` triggers this path in NodeManager.

### Production Builds
Use **PyInstaller** to compile into a standalone binary:

```bash
pyinstaller --onefile --name openhydra-node \
  --hidden-import coordinator --hidden-import peer \
  --hidden-import mlx --hidden-import mlx_lm --hidden-import grpc \
  coordinator/node.py
```

The binary is placed in `build/` and included via `electron-builder.json`:

```json
{
  "extraResources": [
    { "from": "build/openhydra-node", "to": "openhydra-node" }
  ]
}
```

At runtime, NodeManager resolves the binary path:
```javascript
const nodeBin = app.isPackaged
  ? path.join(process.resourcesPath, 'openhydra-node')
  : 'openhydra-node';
```

Platform builds:
| Platform | PyInstaller output | electron-builder target |
|----------|-------------------|------------------------|
| macOS ARM | `openhydra-node` (Mach-O arm64) | DMG (arm64) |
| macOS x64 | `openhydra-node` (Mach-O x64) | DMG (x64) |
| Linux x64 | `openhydra-node` (ELF x64) | AppImage + .deb |
| Windows x64 | `openhydra-node.exe` (PE x64) | NSIS installer |

---

## 5. Build & Distribution (mirrors LLMule's electron-builder.json)

```json
{
  "appId": "co.openhydra.desktop",
  "productName": "OpenHydra",
  "directories": { "output": "dist_electron" },
  "files": ["dist/**/*", "dist-electron/**/*"],
  "extraResources": [
    { "from": "build/openhydra-node", "to": "openhydra-node" }
  ],
  "mac": {
    "target": [{ "target": "dmg", "arch": ["arm64", "x64"] }],
    "icon": "assets/icon.icns",
    "hardenedRuntime": true,
    "entitlements": "build/entitlements.mac.plist"
  },
  "win": {
    "target": [{ "target": "nsis", "arch": ["x64"] }]
  },
  "linux": {
    "target": ["AppImage", "deb"],
    "category": "Utility"
  },
  "publish": {
    "provider": "github",
    "owner": "samtroberts",
    "repo": "openhydra-desktop"
  }
}
```

---

## 6. Key Dependencies (matching LLMule)

```json
{
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "axios": "^1.7.9",
    "electron-store": "^10.0.0",
    "uuid": "^11.0.5",
    "react-hot-toast": "^2.5.1",
    "react-markdown": "^9.0.1",
    "react-syntax-highlighter": "^15.5.0",
    "lucide-react": "^0.460.0"
  },
  "devDependencies": {
    "electron": "^33.2.1",
    "electron-builder": "^25.1.8",
    "vite": "^6.0.5",
    "vite-plugin-electron": "^0.29.0",
    "@vitejs/plugin-react": "^4.3.1",
    "tailwindcss": "^3.4.17",
    "@tailwindcss/typography": "^0.5.0",
    "postcss": "^8.4.49",
    "autoprefixer": "^10.4.20"
  }
}
```

---

## 7. Execution Order

| Step | Action |
|------|--------|
| 1 | Create `openhydra-desktop/` directory, initialize npm project |
| 2 | Install all dependencies (Electron, React, Vite, Tailwind, etc.) |
| 3 | Create `vite.config.js`, `tailwind.config.js`, `postcss.config.js`, `index.html` |
| 4 | Create `electron/main.js` + `electron/preload.js` (IPC bridge) |
| 5 | Create `electron/config.js` + `electron/tray.js` |
| 6 | Create `electron/node/manager.js` + `electron/node/handlers.js` (spawn/kill) |
| 7 | Create `electron/llm/client.js` + `electron/llm/handlers.js` (chat/models) |
| 8 | Create `src/main.jsx` + `src/App.jsx` + `src/styles/index.css` |
| 9 | Create `src/contexts/NodeContext.jsx` + `src/contexts/ChatContext.jsx` |
| 10 | Create dashboard components: NodeToggle, StatusCards, ModelSelector, RamSlider, LogTerminal |
| 11 | Create chat components: ChatView, ChatInterface, ChatMessage, ChatInput |
| 12 | Create `MainLayout.jsx` + `NetworkStats.jsx` |
| 13 | Create `electron-builder.json` + build icons |
| 14 | Test full flow: `npm run electron:dev` → start node → see logs → chat → stop |
