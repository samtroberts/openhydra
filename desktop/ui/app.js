// OpenHydra Desktop — the shadcn wireframe (docs/openhydra_wireframe_shadcn.html) wired to the
// Rust backend via Tauri IPC. The DOM + CSS are the wireframe verbatim; this file swaps the
// wireframe's demo data for live network state. In a plain browser it renders a demo mock.
(function () {
  const $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const app = $("#app"), root = document.documentElement;
  if (/Mac/.test(navigator.platform)) document.body.classList.add("is-mac");
  document.addEventListener("contextmenu", (e) => {
    const t = e.target; if (!(t.matches?.("input, textarea") || t.isContentEditable)) e.preventDefault();
  });

  // ── Tauri bridge ──
  const tauri = window.__TAURI__?.core;
  async function call(cmd, args) { if (tauri) return tauri.invoke(cmd, args); return mock(cmd, args); }

  // ── browser-preview mock ──
  const mk = { provider: false, gateway: false };
  function mock(cmd, args) {
    if (cmd === "start_provider") { mk.provider = true; return null; }
    if (cmd === "stop_provider") { mk.provider = false; return null; }
    if (cmd === "start_gateway") { mk.gateway = true; return null; }
    if (cmd === "stop_gateway") { mk.gateway = false; return null; }
    if (cmd === "save_settings") { if (args?.settings) mk.sharedModels = args.settings.shared_models || []; return null; }
    if (cmd === "gateway_health") return mk.gateway;
    if (cmd === "connector_status") {
      const c = (o) => ({ declared_models: [], ...o, connected: !!(mk.connected && mk.connected[o.key]) });
      return [
        c({ key: "opencode", label: "OpenCode", kind: "opencode", installed: true, detail: "/usr/local/bin/opencode", surfaces: ["terminal", "app"], declares_models: true, has_gui: true, natural_verb: "launch", declared_models: ["qwen2.5:7b"] }),
        c({ key: "claude", label: "Claude Code", kind: "claude", installed: true, detail: "/usr/local/bin/claude", surfaces: ["terminal", "editor"], declares_models: false, has_gui: true, natural_verb: "launch" }),
        c({ key: "continue", label: "Continue", kind: "continue", installed: false, detail: null, surfaces: ["editor"], declares_models: true, has_gui: true, natural_verb: "connect" }),
        c({ key: "pi", label: "Pi", kind: "pi", installed: true, detail: "~/.pi/agent", surfaces: ["terminal"], declares_models: true, has_gui: false, natural_verb: "launch" }),
        c({ key: "hermes", label: "Hermes", kind: "hermes", installed: false, detail: null, surfaces: ["terminal", "app"], declares_models: false, has_gui: true, natural_verb: "launch" }),
      ];
    }
    if (cmd === "connector_preview") {
      const key = args?.key;
      const paths = { opencode: "~/.config/opencode/opencode.json", claude: "~/.claude/settings.json", continue: "~/.continue/config.yaml", pi: "~/.pi/agent/models.json", hermes: "~/.hermes/config.yaml" };
      const previews = {
        opencode: '{\n  "$schema": "https://opencode.ai/config.json",\n  "provider": {\n    "openhydra": {\n      "npm": "@ai-sdk/openai-compatible",\n      "name": "OpenHydra",\n      "options": { "baseURL": "http://127.0.0.1:16527/v1", "apiKey": "oh-local" },\n      "models": { "openhydra/auto": { "name": "OpenHydra Auto" } }\n    }\n  },\n  "model": "openhydra/openhydra/auto"\n}',
        claude: '{\n  "env": {\n    "ANTHROPIC_BASE_URL": "http://127.0.0.1:16527",\n    "ANTHROPIC_API_KEY": "oh-local"\n  }\n}',
        continue: "models:\n- name: OpenHydra\n  provider: openai\n  model: openhydra/auto\n  apiBase: http://127.0.0.1:16527/v1\n  apiKey: oh-local\n  roles:\n  - chat\n  - edit\n  - apply\n",
        pi: '{\n  "providers": {\n    "openhydra": {\n      "baseUrl": "http://127.0.0.1:16527/v1",\n      "apiKey": "oh-local",\n      "api": "openai-completions",\n      "models": [{ "id": "openhydra/auto", "name": "OpenHydra Auto" }]\n    }\n  }\n}',
        hermes: "model:\n  provider: custom\n  base_url: http://127.0.0.1:16527/v1\n  api_key: oh-local\n  name: openhydra/auto\n",
      };
      return { key, kind: key, path: paths[key], action: "create", preview: previews[key] || "{}", warning: key === "continue" ? "Continue's config.yaml will be reformatted (comments/spacing are not preserved). The original is backed up." : (key === "hermes" ? "Hermes' config.yaml will be reformatted (comments/spacing are not preserved). The original is backed up." : null) };
    }
    if (cmd === "connector_apply") { (mk.connected || (mk.connected = {}))[args?.key] = true; return { key: args?.key, path: "~/(config)", backup: null, action: "added" }; }
    if (cmd === "connector_disconnect") { if (mk.connected) mk.connected[args?.key] = false; return { key: args?.key, path: "~/(config)", action: "restored" }; }
    if (cmd === "open_gui") return null;
    if (cmd === "connector_test") return mk.gateway ? "granite4.1:3b" : Promise.reject("gateway unreachable on :16527 — is OpenHydra sharing/serving?");
    if (cmd === "get_state") return {
      provider: { status: { running: mk.provider, pid: 42, peer_id: "12D3KooWQvXm4cAsusDEuXRH", engines: "ollama", announced: mk.provider ? 2 : 0, relays: 2, exited: null }, logs: ["node up", "announced tinyllama:latest", "announced llama3.2:1b"] },
      gateway: { status: { running: mk.gateway, pid: 43, peer_id: "12D3KooWQvXm4cAsusDEuXRH", engines: null, announced: null, relays: 2, exited: null }, logs: ["gateway listening 127.0.0.1:16527"] },
      settings: { bootstraps: ["/dns4/bootstrap-us.openhydra.co/tcp/4001"], gateway_port: 16527, engine_autostart: true, search_url: "", shared_models: mk.sharedModels || [], sharing_enabled: !!mk.provider, resume_on_launch: true, schema_version: 2 },
      agent_found: true, gateway_url: "http://127.0.0.1:16527/v1", resumed_on_launch: !!mk.resumedOnLaunch,
    };
    if (cmd === "system_info") return { os: "macos", arch: "aarch64", cpu: "Apple M1", ram_bytes: 8589934592, gpus: [{ name: "Apple M1 (7-core GPU)", vram_bytes: 8589934592, unified: true }] };
    if (cmd === "detect_engines_now") return (mk.runningEngines || mk.installedEngines || ["ollama"]).map((l) => ({ label: l, url: "http://127.0.0.1:11434", models: l === "ollama" ? ["tinyllama:latest", "llama3.2:1b"] : [] }));
    if (cmd === "installed_engines") return mk.installedEngines || ["ollama"];
    if (cmd === "run_engine") { mk.runningEngines = [...new Set([...(mk.runningEngines || ["ollama"]), args.engine])]; return null; }
    if (cmd === "install_plan") {
      const cli = args.variant === "cli";
      // vLLM on this mock host (macOS) = the community vllm-metal plugin (MLX, prebuilt wheels).
      if (args.engine === "vllm") return { engine: "vllm", supported: true, verified: false,
        already_installed: (mk.installedEngines || ["ollama"]).includes("vllm"), cli_available: false,
        summary: "Install vLLM for Apple Silicon via the community vllm-metal plugin (MLX backend, prebuilt wheels — no source compile). Runs install.sh into ~/.venv-vllm-metal.", blocker: null };
      const supported = ["ollama", "llama.cpp", "lm-studio", "comfyui", "exo"].includes(args.engine);
      const installed = (mk.installedEngines || ["ollama"]).includes(args.engine);
      const cliAvailable = ["comfyui", "exo"].includes(args.engine); // macOS: app default + CLI option
      const summaries = {
        "ollama": "Install Ollama via the official method → serves on :11434 (pulls qwen2.5:7b)",
        "llama.cpp": "Install llama.cpp via Homebrew (brew install llama.cpp) — installs llama-server; then point it at a GGUF model to serve on :8080.",
        "lm-studio": "Download the official LM Studio installer (~570 MB .dmg, Apple Silicon) and install it to /Applications, then it auto-starts the local server on :1234.",
        "comfyui": cli
          ? "Headless CLI: install ComfyUI via comfy-cli (uv) → clones ComfyUI + the Metal PyTorch into ~/.openhydra/engines/comfyui. Run `comfy launch` (:8188)."
          : "Download the official ComfyUI desktop app (signed .dmg, ~170 MB) → /Applications, like LM Studio. Then launch it and add a checkpoint model.",
        "exo": cli
          ? "Headless CLI: install Exo from source: git clone → uv venv (Python 3.13) → uv pip install. Then run `exo`."
          : "Download the native EXO macOS app (signed .dmg) → /Applications, like LM Studio. Then launch EXO to serve + join the cluster.",
      };
      return { engine: args.engine, supported, verified: args.engine !== "llama.cpp", already_installed: installed, cli_available: cliAvailable,
        summary: supported ? summaries[args.engine] : "No Tier-1 installer for this engine yet — use a guided install.", blocker: null };
    }
    if (cmd === "install_engine") {
      const eng = args.engine;
      const steps = eng === "comfyui"
        ? [["phase", "running: uvx --from comfy-cli comfy --skip-prompt install"], ["log", "cloning ComfyUI…"], ["log", "detected Metal (Apple Silicon) → installing torch"], ["log", "installing ComfyUI-Manager"], ["done", "ComfyUI installed. Launch it (comfy launch, serves :8188) and add a checkpoint model; this card flips to running once it serves."]]
        : eng === "exo"
        ? [["phase", "running: git clone https://github.com/exo-explore/exo.git"], ["phase", "running: uv venv --python 3.12"], ["log", "installed Python 3.12"], ["phase", "running: uv pip install ."], ["log", "resolving dependencies…"], ["done", "Exo installed. Run exo to serve an OpenAI API + join the P2P cluster; this card flips to running once it serves."]]
        : eng === "lm-studio"
        ? [["phase", "downloading installer → ~/.openhydra/engines/lm-studio/LM-Studio.dmg (can take a minute)"], ["log", "downloaded 570 MB"], ["phase", "mounting the disk image"], ["phase", "copying LM Studio → /Applications"], ["log", "installed LM Studio to /Applications"], ["phase", "launching LM Studio"], ["done", "LM Studio installed to /Applications and launched. Enable Developer → Start Server (:1234); this card flips to running once it serves."]]
        : eng === "llama.cpp"
        ? [["phase", "running: brew install llama.cpp"], ["log", "==> Fetching llama.cpp"], ["log", "==> Pouring llama.cpp bottle"], ["phase", "confirming llama-server is installed"], ["done", "llama.cpp installed — ready to configure"]]
        : [["phase", "running: ollama installer"], ["log", "downloading ollama…"], ["log", "installing to /usr/local/bin"], ["phase", "health-check http://127.0.0.1:11434/api/version"], ["phase", "pulling qwen2.5:7b (first time can take minutes)"], ["log", "pulling manifest"], ["log", "verifying sha256 digest"], ["done", "ollama ready — qwen2.5:7b pulled and warm"]];
      let i = 0;
      const tick = () => { if (i >= steps.length) { mk.installedEngines = [...new Set([...(mk.installedEngines || ["ollama"]), eng])]; return; } mockEmitInstall({ engine: eng, kind: steps[i][0], message: steps[i][1] }); i++; setTimeout(tick, 450); };
      setTimeout(tick, 300);
      return null;
    }
    if (cmd === "status_snapshot") {
      if (!mk.provider && !mk.gateway) return null;
      return {
        libp2p_peer_id: "12D3KooWQvXm4cAsusDEuXRH",
        uptime_secs: 8123,
        network: {
          nat: { nat_type: "cone", is_public: false }, autonat_private: true, ipv6_capable: true, kad_server_mode: false, kad_routing_peers: 12, network_generation: 0,
          listen_addrs: ["/ip4/0.0.0.0/udp/4111/quic-v1"], external_addrs: ["/ip4/49.36.1.2/udp/4111/quic-v1"], relay_reservations: ["/ip4/45.79.190.172/tcp/4001/p2p/12D3KooWEL/p2p-circuit"],
          peers: [
            { peer_id: "12D3KooWEL", quic_direct_v4: 0, quic_direct_v6: 0, tcp_direct: 0, tcp_relay: 1, failure_streak: 0, path: "relay" }, // bootstrap → filtered out
            { peer_id: "12D3KooWM2qsVg5WbR6Asusn2XN7", quic_direct_v4: 1, quic_direct_v6: 0, tcp_direct: 0, tcp_relay: 0, failure_streak: 0, path: "direct" },
            { peer_id: "12D3KooWEzegXr4qcjEW3WT", quic_direct_v4: 0, quic_direct_v6: 0, tcp_direct: 0, tcp_relay: 1, failure_streak: 0, path: "relay" },
            ...Array.from({ length: 10 }, (_, i) => ({ peer_id: "12D3KooWPeer" + String(i).padStart(2, "0") + "abcd", quic_direct_v4: i % 2, quic_direct_v6: (i + 1) % 2, tcp_direct: 0, tcp_relay: i % 3 === 0 ? 1 : 0, failure_streak: 0, path: i % 3 === 0 ? "relay" : i % 3 === 1 ? "direct" : "mixed" })),
          ],
          known_models: ["tinyllama:latest", "llama3:latest", "qwen2.5:7b"],
          known_providers: [
            { model_id: "tinyllama:latest", openhydra_peer_id: "oh_asus", libp2p_peer_id: "12D3KooWM2qsVg5WbR6Asusn2XN7" },
            { model_id: "tinyllama:latest", openhydra_peer_id: "oh_mac2", libp2p_peer_id: "12D3KooWSecondTiny99xyz" },
            { model_id: "qwen2.5:7b", openhydra_peer_id: "oh_gpu3", libp2p_peer_id: "12D3KooWEzegXr4qcjEW3WT" },
          ],
          counters: { dcutr_successes: 1, dcutr_failures: 0, reversal_dials: 0, reversal_successes: 0, tier_connect_success: { direct_quic_v4: 2, relay: 1 } },
        },
        transfers: {
          requests_served: 3, tokens_served: 1280, serve_errors: 0, aup_refusals: 0, receipts_ledgered: 3,
          requests_consumed: 2, tokens_consumed: 722,
          per_model: { "tinyllama:latest": { requests: 3, tokens: 1280, avg_native_tps: 73.2 } },
          consumed_per_model: { "qwen2.5:7b": { requests: 2, tokens: 722, avg_native_tps: 0 } },
          recent: [
            { ts_ms: Date.now() - 120000, kind: "served", model: "tinyllama:latest", counterparty: "12D3KooWEzegXr4qcjEW3WT", tokens: 128 },
            { ts_ms: Date.now() - 840000, kind: "used", model: "qwen2.5:7b", counterparty: "12D3KooWM2qsVg5WbR6Asusn2XN7", tokens: 512 },
            { ts_ms: Date.now() - 1860000, kind: "served", model: "tinyllama:latest", counterparty: "12D3KooWaBcSXB", tokens: 64 },
          ],
        },
        economy: {
          role: "consumer",
          reputation: [{ openhydra_peer_id: "oh_asus", score: 72.5 }, { openhydra_peer_id: "oh_mac2", score: 64.0 }, { openhydra_peer_id: "oh_gpu3", score: 58.0 }],
          credit: [{ libp2p_peer_id: "12D3KooWM2qsVg5WbR6Asusn2XN7", balance: 5123.4, rate_cap: 1.0 }, { libp2p_peer_id: "12D3KooWEzegXr4qcjEW3WT", balance: 4880.1 }],
          avg_reputation: 65.3, total_credit: 10003.5,
        },
      };
    }
    if (cmd === "chat_completion") {
      // vary by prompt so the reasoning-model render paths are testable in the browser mock:
      // "think" → inline <think> + answer, "empty" → empty content (thinking-mode strip)
      const q = (args?.messages?.slice(-1)[0]?.content || "").toLowerCase();
      let content = "The herd answers: reciprocity beats rent-seeking.\n\n```rust\nfn hi(){ println!(\"herd\"); }\n```";
      if (/empty/.test(q)) content = "";
      else if (/think/.test(q)) content = "<think>\nThe user greeted me. I should respond warmly and briefly.\n</think>\nHey! How can I help?";
      return new Promise((r) => setTimeout(() => r({
        choices: [{ message: { content, role: "assistant" } }],
        usage: { completion_tokens: 42 }, openhydra: { engine: { native_tps: 61 }, hops_ms: { network_rtt: 12 } }, model: "qwen3.5-4b-mlx",
      }), 400));
    }
    if (cmd === "web_search") return [];
    if (cmd === "load_sessions") return "";                       // #1 (mock: nothing persisted)
    if (cmd === "save_sessions") return null;                     // #1
    if (cmd === "load_stats") {                                   // #7/#10 seeded history for preview
      const now = Date.now(), HR = 3600000, hk = (ms) => Math.floor(ms / HR), models = {};
      const mk = (id, sDay, uDay, days) => {
        const m = { firstServed: now - days * 86400000, firstUsed: now - days * 86400000, servedTotal: 0, usedTotal: 0, lastServed: 0, lastUsed: 0, buckets: {} };
        for (let d = 0; d < days; d++) for (let h = 0; h < 24; h += 3) {
          const k = hk(now - d * 86400000 - h * HR);
          const s = Math.round(sDay / 8 * (0.4 + Math.random())), u = Math.round(uDay / 8 * (0.4 + Math.random()));
          const b = (m.buckets[k] ||= { s: 0, u: 0 }); b.s += s; b.u += u; m.servedTotal += s; m.usedTotal += u;
        }
        models[id] = m;
      };
      mk("tinyllama:latest", 1800, 150, 9); mk("llama3.2:1b", 900, 500, 6); mk("qwen2.5:7b", 0, 1300, 5);
      mk("phi3:mini", 600, 0, 12);   // previously served, no longer loaded → shows as "inactive"
      return JSON.stringify({ models });
    }
    if (cmd === "save_stats") return null;                        // #7
    if (cmd === "device_hostname") return "Sam’s MacBook Air";    // #9
    if (cmd === "app_version") return "0.3.10";                   // mock bundle version
    if (cmd === "appimage_status") return { is_appimage: localStorage.getItem("oh_mock_appimage") === "1", integrated: mk.integrated || false };
    if (cmd === "integrate_appimage") { mk.integrated = true; return null; }
    if (cmd === "export_logs") return "~/.openhydra/openhydra-logs-1785200000.txt"; // #4
    return null;
  }

  // ── icons (wireframe verbatim) ──
  const S = {
    zap:'<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    command:'<path d="M15 6v12a3 3 0 1 0 3-3H6a3 3 0 1 0 3 3V6a3 3 0 1 0-3 3h12a3 3 0 1 0-3-3"/>',
    theme:'<path d="M12 8a2.83 2.83 0 0 0 4 4 4 4 0 1 1-4-4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.9 4.9 1.4 1.4"/><path d="m17.7 17.7 1.4 1.4"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.3 17.7-1.4 1.4"/><path d="m19.1 4.9-1.4 1.4"/>',
    users:'<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    network:'<rect x="16" y="16" width="6" height="6" rx="1"/><rect x="2" y="16" width="6" height="6" rx="1"/><rect x="9" y="2" width="6" height="6" rx="1"/><path d="M5 16v-3a1 1 0 0 1 1-1h12a1 1 0 0 1 1 1v3"/><path d="M12 12V8"/>',
    activity:'<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    list:'<path d="M8 6h13"/><path d="M8 12h13"/><path d="M8 18h13"/><path d="M3 6h.01"/><path d="M3 12h.01"/><path d="M3 18h.01"/>',
    chat:'<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    monitor:'<rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8"/><path d="M12 17v4"/>',
    package:'<path d="M11 21.73a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73z"/><path d="M3.3 7 12 12l8.7-5"/><path d="M12 22V12"/>',
    plug:'<path d="M12 22v-5"/><path d="M9 8V2"/><path d="M15 8V2"/><path d="M18 8v5a4 4 0 0 1-4 4h-4a4 4 0 0 1-4-4V8Z"/>',
    settings:'<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1Z"/>',
    search:'<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    send:'<path d="M22 2 11 13"/><path d="M22 2 15 22l-4-9-9-4Z"/>',
    plus:'<path d="M5 12h14"/><path d="M12 5v14"/>',
    chev:'<path d="m6 9 6 6 6-6"/>', chevl:'<path d="m15 18-6-6 6-6"/>', chevr:'<path d="m9 18 6-6-6-6"/>',
    homeic:'<path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>',
    refresh:'<path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M3 21v-5h5"/>',
    sliders:'<line x1="4" x2="4" y1="21" y2="14"/><line x1="4" x2="4" y1="10" y2="3"/><line x1="12" x2="12" y1="21" y2="12"/><line x1="12" x2="12" y1="8" y2="3"/><line x1="20" x2="20" y1="21" y2="16"/><line x1="20" x2="20" y1="12" y2="3"/><line x1="2" x2="6" y1="14" y2="14"/><line x1="10" x2="14" y1="8" y2="8"/><line x1="18" x2="22" y1="16" y2="16"/>',
    x:'<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
    more:'<circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/><circle cx="5" cy="12" r="1"/>',
    paperclip:'<path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48"/>',
    mic:'<path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/>',
    panelleft:'<rect width="18" height="18" x="3" y="3" rx="2"/><path d="M9 3v18"/>',
  };
  function injectIcons(rt = document) { $$("[data-i]", rt).forEach((e) => { const n = e.dataset.i; if (S[n]) e.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${S[n]}</svg>`; }); }
  injectIcons();

  // ── toast + popover menu (wireframe verbatim) ──
  let toastEl, toastT;
  function toast(m) { if (!toastEl) { toastEl = document.createElement("div"); toastEl.className = "toast"; document.body.appendChild(toastEl); } toastEl.textContent = m; toastEl.classList.add("show"); clearTimeout(toastT); toastT = setTimeout(() => toastEl.classList.remove("show"), 1600); }
  let menuFor = null;
  function closeMenus() { $$(".menu").forEach((m) => m.remove()); menuFor = null; }
  function menu(anchor, items) {
    if (menuFor === anchor) { closeMenus(); return; }
    closeMenus(); menuFor = anchor;
    const m = document.createElement("div"); m.className = "menu";
    items.forEach((it) => { if (it.sep) { const s = document.createElement("div"); s.className = "msep"; m.appendChild(s); return; } const mi = document.createElement("div"); mi.className = "mi"; mi.innerHTML = `<span class="ck">${it.on ? "✓" : ""}</span>${esc(it.label)}`; mi.onclick = (e) => { e.stopPropagation(); closeMenus(); it.fn && it.fn(); }; m.appendChild(mi); });
    document.body.appendChild(m);
    const r = anchor.getBoundingClientRect(); m.style.left = Math.min(r.left, innerWidth - m.offsetWidth - 10) + "px"; let t = r.bottom + 5; if (t + m.offsetHeight > innerHeight - 8) t = r.top - m.offsetHeight - 5; m.style.top = Math.max(8, t) + "px";
  }
  document.addEventListener("click", closeMenus); addEventListener("resize", closeMenus);
  function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
  function shortPeer(p) { return p && p.length > 20 ? `${p.slice(0, 10)}…${p.slice(-6)}` : p || "—"; }
  function peerShort(p) { return p && p.length > 18 ? `${p.slice(0, 12)}…${p.slice(-4)}` : p || "—"; }

  // ── model family badge ──
  const FAMILY = [[/qwen/i, "qwen"], [/llama|tinyllama/i, "llama"], [/gemma/i, "gemma"], [/mi(s|x)tral/i, "mistral"], [/phi/i, "phi"], [/deepseek/i, "deepseek"], [/flux|sdxl|stable/i, "flux"], [/nomic|embed/i, "nomic"]];
  const FAMSTYLE = { qwen: ["Q", "#6d49c4"], llama: ["L", "#0866ff"], gemma: ["G", "#1a73e8"], mistral: ["M", "#fa5210"], phi: ["φ", "#12a3a6"], deepseek: ["DS", "#4d6bfe"], flux: ["F", "#111418"], nomic: ["N", "#127a6b"] };
  function modelIcon(id) { const k = (FAMILY.find(([rx]) => rx.test(id || "")) || [])[1]; const s = k && FAMSTYLE[k]; if (s) return `<span class="micon" style="background:${s[1]}">${s[0]}</span>`; const hues = ["#64748b", "#0891b2", "#7c3aed", "#db2777", "#ca8a04", "#059669"]; return `<span class="micon" style="background:${hues[(String(id).charCodeAt(0) || 0) % hues.length]}">${esc((String(id)[0] || "?").toUpperCase())}</span>`; }
  const modelCat = (m) => /coder|code/i.test(m) ? "code" : /flux|sdxl|stable|image/i.test(m) ? "image" : /embed|nomic/i.test(m) ? "embeddings" : "chat";

  // ── persistence ──
  const store = { get(k, d) { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } }, set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} } };
  // Coerce persisted sessions into the canonical `{ id: {t, m} }` object. A legacy build stored
  // them as an ARRAY of `{ id, name, history, preset }`; if that array reaches `sessions`, every
  // new chat added under a string key is silently dropped by JSON.stringify on save (arrays only
  // serialize numeric indices) — so no chat ever persists across relaunch. Normalizing here fixes
  // it and migrates the old shape (name→t, history→m) in passing.
  function coerceSessions(raw) {
    if (Array.isArray(raw)) {
      const obj = {};
      raw.forEach((s, i) => {
        if (!s) return;
        const id = s.id || ("mig" + i);               // synthesize an id rather than DROP the chat
        const src = Array.isArray(s.m) ? s.m : Array.isArray(s.history) ? s.history : [];
        const msgs = src.map((h) => {
          if (Array.isArray(h)) return h;             // already [role, content, …] — keep verbatim
          if (typeof h === "string") return ["me", h]; // bare-string message
          if (h && typeof h === "object") {
            const text = typeof h.content === "string" ? h.content : (h.text ?? (h.content != null ? JSON.stringify(h.content) : ""));
            return [h.role === "assistant" || h.role === "ai" ? "ai" : "me", text];
          }
          return ["me", String(h ?? "")];
        });
        obj[id] = { t: s.t || s.name || "New chat", m: msgs };
      });
      return obj;
    }
    return raw && typeof raw === "object" ? raw : {};
  }
  let sessions = coerceSessions(store.get("oh_sessions", {})), sessionOrder = store.get("oh_order", []);
  let deviceName = store.get("oh_device", "");   // #9: derived from the OS on boot if unset
  let usedTokens = store.get("oh_used", 0);
  root.dataset.theme = store.get("oh_theme", "light");
  if (store.get("oh_adv", false)) app.setAttribute("data-adv", "");
  function saveSessions() {
    if (Array.isArray(sessions)) sessions = coerceSessions(sessions); // never persist an array (see coerceSessions)
    store.set("oh_sessions", sessions); store.set("oh_order", sessionOrder);
    // #1: durable write-through to the Tauri backend file (WebView localStorage isn't durable
    // across restarts on any platform). Fire-and-forget; localStorage stays as a fast cache.
    try { call("save_sessions", { data: JSON.stringify({ sessions, order: sessionOrder, device: deviceName, used: usedTokens }) }); } catch {}
  }

  // ── live state ──
  let state = null, engines = [], installedEngines = [], snap = null, activeView = "home", curChat = null, attachments = [];
  // Rolling per-chat telemetry the agent only emits per-request (there's no aggregated RTT on
  // the status API) — we average the last N replies client-side for the Activity view.
  const rttSamples = store.get("oh_rtt", []), tpsSamples = store.get("oh_tps", []);
  const ROLL_MAX = 30;
  function pushSample(arr, v, key) { if (v == null || !isFinite(v)) return; arr.push(v); while (arr.length > ROLL_MAX) arr.shift(); store.set(key, arr); }
  const mean = (a) => a.length ? a.reduce((x, y) => x + y, 0) / a.length : null;

  // ── #7/#10: lifetime served/consumed model stats + hourly time-series ────────────────────
  // The agent's per-model counters reset to zero every process restart, so the DURABLE lifetime
  // totals + timeline live here. On each poll we diff the agent's cumulative counters and fold
  // the delta into hourly buckets (a Prometheus-style counter→rate), anchored at each model's
  // first token. Persisted to ~/.openhydra/stats.json via the Tauri backend (localStorage isn't
  // durable in the WebView). served = provider's `per_model`; used = gateway's `consumed_per_model`.
  const HOUR_MS = 3600000, STATS_KEEP_HOURS = 24 * 31;   // ~31 days of hourly buckets
  let statsDB = { models: {} };                          // hydrated on boot from load_stats
  let statsDirty = false, statsSaveT = 0;
  function hourKey(ms) { return Math.floor(ms / HOUR_MS); }
  function statModel(id) {
    return (statsDB.models[id] ||= { firstServed: null, firstUsed: null, servedTotal: 0, usedTotal: 0, lastServed: 0, lastUsed: 0, buckets: {} });
  }
  // Fold one cumulative counter reading into a per-model lifetime total + current-hour bucket.
  // `raw` is the agent's since-boot counter; a value < the last reading means the agent restarted
  // (counter reset to 0), so the whole `raw` is the fresh delta.
  function foldCounter(m, raw, field, bucketField, now) {
    raw = Math.max(0, Math.round(raw || 0));
    const last = m[field];
    const delta = raw >= last ? raw - last : raw;
    m[field] = raw;
    if (delta <= 0) return;
    const totalField = field === "lastServed" ? "servedTotal" : "usedTotal";
    const firstField = field === "lastServed" ? "firstServed" : "firstUsed";
    m[totalField] += delta;
    if (m[firstField] == null) m[firstField] = now;       // first-token anchor (no synthetic back-fill)
    const b = (m.buckets[hourKey(now)] ||= { s: 0, u: 0 });
    b[bucketField] += delta;
  }
  function accumulateStats() {
    if (!snap?.transfers) return;
    const now = Date.now(), t = snap.transfers;
    for (const [id, pm] of Object.entries(t.per_model || {})) foldCounter(statModel(id), pm.tokens, "lastServed", "s", now);
    for (const [id, pm] of Object.entries(t.consumed_per_model || {})) foldCounter(statModel(id), pm.tokens, "lastUsed", "u", now);
    // Prune buckets older than the retention window (keeps stats.json bounded).
    const floor = hourKey(now) - STATS_KEEP_HOURS;
    for (const m of Object.values(statsDB.models)) for (const k in m.buckets) if (+k < floor) delete m.buckets[k];
    statsDirty = true;
    const nowT = Date.now();
    if (nowT - statsSaveT > 8000) { statsSaveT = nowT; statsDirty = false; try { call("save_stats", { data: JSON.stringify(statsDB) }); } catch {} }
  }
  async function loadStats() {
    try { const blob = await call("load_stats"); if (blob) { const d = JSON.parse(blob); if (d && d.models) statsDB = d; } } catch {}
  }
  const lifetimeServed = (id) => statsDB.models[id]?.servedTotal || 0;
  const lifetimeUsed = (id) => statsDB.models[id]?.usedTotal || 0;
  const statModels = () => Object.keys(statsDB.models);
  // Durable lifetime totals across all models. Fall back to the agent's since-boot counter (or the
  // legacy desktop-chat counter) until the accumulator has data — so a fresh install still shows
  // something. `used` now derives from the gateway's per-model consumed tracking, so it counts
  // tokens consumed by external OpenAI clients (e.g. a coding agent), not just in-app chats.
  const totalServed = () => Object.values(statsDB.models).reduce((a, m) => a + (m.servedTotal || 0), 0) || (snap?.transfers?.tokens_served ?? 0);
  const totalUsed = () => Object.values(statsDB.models).reduce((a, m) => a + (m.usedTotal || 0), 0) || (snap?.transfers?.tokens_consumed ?? usedTokens);
  // Aggregate buckets for a range into ordered time-slots, each a per-model {s,u}. 24h → hourly
  // (24 slots), 7d/30d → daily. Returns { slots:[{label,models:{id:{s,u}}}], models:Set }.
  function statsSeries(range) {
    const now = Date.now();
    const spans = { "24h": { n: 24, ms: HOUR_MS }, "7d": { n: 7, ms: 86400000 }, "30d": { n: 30, ms: 86400000 } };
    const sp = spans[range] || spans["24h"];
    const slots = [], models = new Set();
    for (let i = sp.n - 1; i >= 0; i--) {
      const end = now - i * sp.ms, start = end - sp.ms;
      const slot = { end, models: {} };
      const kLo = hourKey(start), kHi = hourKey(end);
      for (const [id, m] of Object.entries(statsDB.models)) {
        let s = 0, u = 0;
        for (let k = kLo; k < kHi; k++) { const b = m.buckets[k]; if (b) { s += b.s; u += b.u; } }
        if (s || u) { slot.models[id] = { s, u }; models.add(id); }
      }
      slots.push(slot);
    }
    return { slots, models, span: sp };
  }

  // ── economy (M2.2 reputation + M2.3 credit), surfaced by the agent status endpoint ──
  // reputation is keyed by OpenHydra peer id; credit by libp2p peer id. known_providers
  // carries both, so we can join either back to a model row or a peer row.
  function econ() { return snap?.economy || {}; }
  function repByOpenhydra() { const m = {}; (econ().reputation || []).forEach((r) => m[r.openhydra_peer_id] = r.score); return m; }
  function creditByLibp2p() { const m = {}; (econ().credit || []).forEach((c) => m[c.libp2p_peer_id] = c); return m; }
  // libp2p peer id → earned reputation, resolved through the provider directory.
  function repByLibp2p() {
    const byOh = repByOpenhydra(), out = {};
    (snap?.network?.known_providers || []).forEach((p) => { if (p.openhydra_peer_id in byOh) out[p.libp2p_peer_id] = byOh[p.openhydra_peer_id]; });
    return out;
  }
  function providersForModel(model) { return (snap?.network?.known_providers || []).filter((p) => p.model_id === model); }
  // mean earned reputation across the providers serving `model` (null if none rated yet).
  function modelReputation(model, byOh) { const s = providersForModel(model).map((p) => byOh[p.openhydra_peer_id]).filter((x) => x != null); return s.length ? Math.round(s.reduce((a, b) => a + b, 0) / s.length) : null; }
  // provider role publishes per-model serve TPS; only present for models THIS node serves.
  function modelAvgTps(model) { const pm = snap?.transfers?.per_model?.[model]; return pm && pm.avg_native_tps > 0 ? Math.round(pm.avg_native_tps) : null; }
  function fmtUptime(s) { if (s == null) return "—"; s = Math.floor(s); if (s < 60) return s + "s"; const m = Math.floor(s / 60); if (m < 60) return m + "m"; const h = Math.floor(m / 60); if (h < 24) return `${h}h ${m % 60}m`; const d = Math.floor(h / 24); return `${d}d ${h % 24}h`; }
  function repBadge(v) { if (v == null) return '<span class="mut">—</span>'; v = Math.round(v); const cls = v >= 66 ? "ok" : v >= 40 ? "warn" : "secondary"; return `<span class="badge ${cls}">${v}</span>`; }

  // ── nav / workspace switcher / history (wireframe verbatim + header-hide + renderView) ──
  const titles = { home: "Home", chat: "Chat", activity: "Activity", connectors: "Connectors", providers: "Models", share: "Share", engines: "Engines", ledger: "Ledger", peers: "Diagnostics and Stats", settings: "Settings" };
  const searchable = { providers: 1, peers: 1 };
  const VIEWMODE = { home: "home", chat: "home", activity: "home", connectors: "home", providers: "network", share: "network", engines: "network", ledger: "network", peers: "network" };
  function setMode(m) { app.dataset.mode = m; $$("#modeswitch button").forEach((b) => b.toggleAttribute("data-on", b.dataset.m === m)); }
  let hist = ["home"], hi = 0;
  function updNavBtns() { $("#navback").classList.toggle("dis", hi <= 0); $("#navfwd").classList.toggle("dis", hi >= hist.length - 1); }
  function go(v, noHist) {
    activeView = v; const vm = VIEWMODE[v]; if (vm) setMode(vm);
    $$(".nav").forEach((x) => x.classList.toggle("on", x.dataset.v === v || (v === "chat" && x.dataset.chat === curChat)));
    $$(".view").forEach((x) => x.classList.toggle("on", x.id === "v-" + v));
    $("#htitle").textContent = titles[v];
    $(".header").style.display = v === "home" ? "none" : "flex";   // Home has no title bar
    $("#becomeprov").style.display = v === "providers" ? "" : "none";
    $("#searchwrap").style.visibility = searchable[v] ? "visible" : "hidden"; $("#search").value = "";
    if (!noHist && hist[hi] !== v) { hist = hist.slice(0, hi + 1); hist.push(v); hi = hist.length - 1; }
    updNavBtns(); renderView(); refreshStatus();
  }
  $$(".nav[data-v]").forEach((n) => n.onclick = () => go(n.dataset.v));
  $("#modeswitch").onclick = (e) => { const b = e.target.closest("button"); if (b) go(b.dataset.m === "home" ? "home" : "providers"); };
  $("#becomeprov").onclick = () => go("share");
  $("#navback").onclick = () => { if (hi > 0) { hi--; go(hist[hi], true); } };
  $("#navfwd").onclick = () => { if (hi < hist.length - 1) { hi++; go(hist[hi], true); } };
  $("#navtoggle").onclick = () => { const c = app.classList.toggle("navcol"); $("#navtoggle").title = c ? "Expand sidebar" : "Collapse sidebar"; };
  $("#recsearchbtn").onclick = () => { app.classList.remove("navcol"); $("#recfilter").focus(); };
  $("#gosettings").onclick = () => go("settings");

  // ── theme ──
  function setTheme(m) { m = (m || "").toLowerCase(); if (m === "system") m = matchMedia("(prefers-color-scheme:dark)").matches ? "dark" : "light"; if (m !== "dark" && m !== "light") m = root.dataset.theme === "dark" ? "light" : "dark"; root.dataset.theme = m; store.set("oh_theme", m); }
  $("#theme").onclick = () => setTheme();

  // ── generic dropdowns (wireframe verbatim) ──
  $$(".drop").forEach((d) => d.onclick = (e) => {
    e.stopPropagation();
    const opts = (d.dataset.opts || "").split("|").filter(Boolean);
    const cur = d.querySelector("span").textContent;
    menu(d, opts.length ? opts.map((o) => ({ label: o, on: o === cur, fn: () => { d.querySelector("span").textContent = o; if (d.dataset.act === "theme") setTheme(o); if (d.id === "enginedrop") updateEngineEndpoint(); if (d.id === "modeldrop" || d.id === "homedrop") { setChatMode(o); const other = $(d.id === "modeldrop" ? "#homedrop" : "#modeldrop"); if (other) other.querySelector("span").textContent = o; } } })) : [{ label: "No models on the network yet", fn: () => {} }]);
  });
  // #3: the Settings › Engine endpoint follows the selected engine (was pinned to engines[0]).
  const ENGINE_ENDPOINTS = { "auto-detect": "", ollama: "http://127.0.0.1:11434", "vLLM": "http://127.0.0.1:8000", "LM Studio": "http://127.0.0.1:1234", "llama.cpp": "http://127.0.0.1:8080", "Exo": "http://127.0.0.1:52415", "ComfyUI": "http://127.0.0.1:8188" };
  function updateEngineEndpoint() {
    const drop = $("#enginedrop"), field = $("#engineendpoint"); if (!drop || !field) return;
    const sel = (drop.querySelector("span")?.textContent || "auto-detect").trim();
    // auto-detect → the first detected engine's live URL; a specific pick → that engine's live URL
    // if it's currently running, else its standard endpoint.
    const live = engines.find((e) => (e.label || "").toLowerCase() === sel.toLowerCase());
    const url = sel === "auto-detect"
      ? (engines[0]?.url || "http://127.0.0.1:11434")
      : (live?.url || ENGINE_ENDPOINTS[sel] || "http://127.0.0.1:11434");
    field.textContent = url;
  }

  // ── inference range sliders (wireframe verbatim) ──
  $$(".range").forEach((rg) => {
    const mn = +rg.dataset.min, mx = +rg.dataset.max, isInt = rg.dataset.int, fill = rg.querySelector(".fill"), knob = rg.querySelector(".knob"), lab = $("#" + rg.dataset.label);
    function set(v) { v = Math.max(mn, Math.min(mx, v)); const p = (v - mn) / (mx - mn) * 100; fill.style.width = p + "%"; knob.style.left = p + "%"; lab.textContent = rg.dataset.name + " · " + (isInt ? Math.round(v) : Math.round(v * 100) / 100); rg.dataset.val = v; }
    set(+rg.dataset.val); let drag = 0;
    function fe(e) { const r = rg.getBoundingClientRect(); set(mn + ((e.clientX - r.left) / r.width) * (mx - mn)); }
    rg.onmousedown = (e) => { drag = 1; fe(e); e.preventDefault(); }; addEventListener("mousemove", (e) => { if (drag) fe(e); }); addEventListener("mouseup", () => drag = 0);
  });
  $("#inftoggle").onclick = () => $("#v-chat .three").classList.toggle("noinf");

  // ── recents (live sessions) ──
  function renderRecents() {
    const box = $("#recents"), q = ($("#recfilter").value || "").toLowerCase();
    box.innerHTML = "";
    for (const id of sessionOrder) {
      const s = sessions[id]; if (!s) continue;
      const el = document.createElement("div");
      el.className = "nav recent" + (id === curChat && activeView === "chat" ? " on" : "");
      el.dataset.chat = id;
      el.innerHTML = `<span class="icon" data-i="chat"></span><span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1">${esc(s.t)}</span>`;
      injectIcons(el);
      el.style.display = q && !s.t.toLowerCase().includes(q) ? "none" : "";
      box.appendChild(el);
    }
  }
  $("#recents").onclick = (e) => { const r = e.target.closest(".recent"); if (r) openChat(r.dataset.chat); };
  $("#recfilter").oninput = renderRecents;
  function newSession(title) { const id = "c" + Date.now().toString(36); sessions[id] = { t: title || "New chat", m: [] }; sessionOrder.unshift(id); saveSessions(); return id; }
  function openChat(id) { curChat = id; renderChat(); go("chat"); renderRecents(); }

  // ── chat: model list = network-routable models (fixes the 504: can't pick an unservable model) ──
  // DHT provider records expire (~300s TTL) and re-propagate on their own schedule, so the raw
  // known_models snapshot flickers. Keep a model "sticky" for STICKY_MS after we last saw it so
  // the Providers list and model picker stay calm instead of blinking rows in and out.
  // W2: hold a model listed up to 3 min after we last saw it. This rides transient gossip gaps /
  // relay blips (a live provider that briefly drops out of known_models on a connection loss)
  // WITHOUT ever exceeding the network's own ~300s record TTL — so the UI never claims a provider
  // is alive longer than the network itself does. A provider that genuinely stops is dropped once
  // it's been absent past STICKY_MS, and the real liveness gate is request-time discovery (a dead
  // pick returns a clean "no provider", never a hang).
  const STICKY_MS = 180000;
  const IDLE_MS = 130000;  // seen within this = "live"; IDLE_MS..STICKY_MS = "idle" (dimmed).
  const seenModels = {};   // model -> last-seen ms
  const seenCount = {};    // model -> last-known provider count
  function noteSeen() {
    const now = Date.now();
    (snap?.network?.known_models || []).forEach((m) => seenModels[m] = now);
    const provs = snap?.network?.known_providers || [];
    const c = {}; for (const p of provs) c[p.model_id] = (c[p.model_id] || 0) + 1;
    for (const m in c) seenCount[m] = c[m];
  }
  // A model still listed only because of stickiness (seen, but not within the last IDLE_MS) — shown
  // dimmed/"idle", an honest "unconfirmed this instant" signal. Locally-served models are always live.
  function modelIdle(m) {
    if (state?.provider?.status?.running && engines.some((e) => e.models.includes(m))) return false;
    const t = seenModels[m];
    return t != null && (Date.now() - t) > IDLE_MS;
  }
  function netModels() {
    const now = Date.now();
    const net = Object.keys(seenModels).filter((m) => now - seenModels[m] < STICKY_MS);
    const sharingLocal = state?.provider?.status?.running ? engines.flatMap((e) => e.models) : [];
    return [...new Set([...net, ...sharingLocal])].sort();
  }
  function curModel() { const m = $("#modeldrop span").textContent; return m && m !== "—" && !/no models/i.test(m) ? m : ""; }
  function renderModels() {
    const models = netModels(); const opts = models.join("|");
    $("#homedrop").dataset.opts = opts; $("#modeldrop").dataset.opts = opts;
    const label = models[0] || "— no models yet";
    for (const d of ["#homedrop", "#modeldrop"]) { const sp = $(d + " span"); const cur = sp.textContent; if (!models.includes(cur)) sp.textContent = label; }
    $("#mcount").textContent = models.length; $("#homelive").textContent = models.length;
    $("#provcount") && ($("#provcount").textContent = models.length);
    $("#sbmodels").textContent = models.length + " models";
  }

  // reply rendering: fenced code cards + inline media (image/video/audio) + metadata line
  // #3/#4: classify a markdown media src as image | video | audio (data: URL or media-file URL),
  // "" if it isn't media (a regular link stays as text). Drives which element we render.
  function mediaKind(src) {
    const s = (src || "").trim();
    const dm = s.match(/^data:([a-z]+)\/[a-z0-9.+-]+/i);
    let m = dm ? dm[1].toLowerCase() : "";
    if (!m) { const ext = (s.match(/\.([a-z0-9]+)(?:[?#]|$)/i) || [, ""])[1].toLowerCase();
      m = ["png","jpg","jpeg","gif","webp","svg"].includes(ext) ? "image"
        : ["mp4","webm","mov","m4v"].includes(ext) ? "video"
        : ["mp3","wav","flac","ogg","m4a"].includes(ext) ? "audio" : ""; }
    return (m === "image" || m === "video" || m === "audio") ? m : "";
  }
  // #4: render inline media; #3: copy + download controls under it. Returns null for non-media.
  function mediaEl(src, label) {
    const kind = mediaKind(src); if (!kind) return null;
    const wrap = document.createElement("div"); wrap.style.cssText = "margin:6px 0;max-width:100%";
    const el = document.createElement(kind === "image" ? "img" : kind);
    el.src = src; if (kind !== "image") el.controls = true;
    el.style.cssText = "max-width:100%;border-radius:10px;border:1px solid hsl(var(--border));display:block;width:auto;height:auto";
    wrap.appendChild(el);
    const bar = document.createElement("div"); bar.style.cssText = "display:flex;gap:12px;margin-top:4px;font-size:12px";
    const cp = document.createElement("button"); cp.textContent = "⎘ copy";
    cp.style.cssText = "cursor:pointer;color:hsl(var(--muted));background:none;border:none;padding:0;font:inherit";
    cp.onclick = async () => {
      try { // copy the image itself to the clipboard when possible; else copy the src/data-URL
        if (kind === "image" && src.startsWith("data:") && window.ClipboardItem) {
          const blob = await (await fetch(src)).blob();
          await navigator.clipboard.write([new ClipboardItem({ [blob.type]: blob })]);
        } else { await navigator.clipboard.writeText(src); }
      } catch { try { await navigator.clipboard.writeText(src); } catch {} }
      cp.textContent = "✓ copied"; setTimeout(() => cp.textContent = "⎘ copy", 1400);
    };
    const ext = kind === "image" ? "png" : kind === "video" ? "mp4" : "flac";
    const dl = document.createElement("a"); dl.textContent = "⭳ download";
    dl.href = src; dl.download = (label && label.trim()) || `openhydra-${kind}.${ext}`;
    dl.style.cssText = "cursor:pointer;color:hsl(var(--muted));text-decoration:none";
    bar.appendChild(cp); bar.appendChild(dl); wrap.appendChild(bar);
    return wrap;
  }
  const KW = new Set("fn let mut pub use impl struct enum trait match if else for while loop return async await move def class import from as with try except lambda yield pass raise function const var new typeof export default package func go defer chan interface map range type switch case break continue static void int float double char bool".split(" "));
  function hl(code) { let out = "", i = 0; const push = (c, s) => out += c ? `<span class="${c}">${esc(s)}</span>` : esc(s); while (i < code.length) { const rest = code.slice(i); let m; if ((m = rest.match(/^(\/\/|#(?!\[)|--)[^\n]*/))) push("tk-com", m[0]); else if ((m = rest.match(/^\/\*[\s\S]*?\*\//))) push("tk-com", m[0]); else if ((m = rest.match(/^("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|`(?:[^`\\]|\\.)*`)/))) push("tk-str", m[0]); else if ((m = rest.match(/^\b\d[\d_]*(\.\d+)?\b/))) push("tk-num", m[0]); else if ((m = rest.match(/^[A-Za-z_]\w*/))) { if (KW.has(m[0])) push("tk-kw", m[0]); else if (code[i + m[0].length] === "(") push("tk-fn", m[0]); else push(null, m[0]); } else { push(null, code[i]); i++; continue; } i += m[0].length; } return out; }
  function parseFences(t) { const p = [], re = /```([\w+-]*)\n([\s\S]*?)```/g; let last = 0, m; while ((m = re.exec(t))) { if (m.index > last) p.push({ prose: t.slice(last, m.index).trim() }); p.push({ lang: m[1] || "code", code: m[2] }); last = re.lastIndex; } if (last < t.length) p.push({ prose: t.slice(last).trim() }); return p.filter((x) => x.code != null || x.prose); }
  function metaRow(m) { return `<div class="msgmeta"><span>${esc(m.model)}</span><span class="num">${m.tok} tok</span><span class="num" title="Engine generation speed — decode tokens per second, measured on the provider">${m.tps} tok/s</span><span class="num">${m.rtt} ms RTT</span><span>${m.at}</span></div>`; }
  // Reasoning models (Qwen3, DeepSeek-R1, …) emit a chain-of-thought. Some inline it as
  // <think>…</think> inside content; some engines (LM Studio serving MLX) strip it and hand
  // back an EMPTY content. Split any inline thinking out of the answer so the chat shows the
  // real answer (or a clear note) instead of a blank bubble.
  function splitThink(raw) {
    raw = raw || ""; let reasoning = "";
    // Well-formed <think>…</think> pairs.
    raw = raw.replace(/<think>([\s\S]*?)<\/think>/gi, (_, t) => { reasoning += t + "\n"; return ""; });
    // Reasoning models (Qwen3, DeepSeek-R1, …) whose chat template PRE-FILLS the opening
    // <think> in the prompt emit a completion that *starts* with the chain-of-thought and
    // closes it with a lone </think> (no opening tag). If a </think> remains with no <think>
    // before it, treat that leading block as reasoning — else it (and the stray tag) leak.
    const close = raw.search(/<\/think>/i);
    if (close !== -1 && !/<think>/i.test(raw.slice(0, close))) {
      reasoning += raw.slice(0, close) + "\n";
      raw = raw.slice(close).replace(/<\/think>/i, "");
    }
    // Unclosed <think>… (still thinking / truncated stream).
    raw = raw.replace(/<think>([\s\S]*)$/i, (_, t) => { reasoning += t; return ""; });
    return { answer: raw.trim(), reasoning: reasoning.trim() };
  }
  function botEl(content, meta, reasoning) {
    const d = document.createElement("div"); d.className = "msg ai";
    if (reasoning) {
      const det = document.createElement("details"); det.style.cssText = "margin:0 0 6px;font-size:12px";
      det.innerHTML = `<summary style="cursor:pointer;color:hsl(var(--muted))">🧠 Thinking</summary>`;
      const b = document.createElement("div"); b.style.cssText = "white-space:pre-wrap;margin-top:4px;padding:6px 10px;border-left:2px solid hsl(var(--border));color:hsl(var(--muted))"; b.textContent = reasoning; det.appendChild(b); d.appendChild(det);
    }
    if (content && content.trim()) {
      for (const part of parseFences(content)) {
        if (part.code != null) { const c = document.createElement("div"); c.className = "code-card"; c.innerHTML = `<div class="code-card-head"><span>${esc(part.lang)}</span><button class="code-copy">⎘ copy</button></div><pre>${hl(part.code)}</pre>`; c.querySelector(".code-copy").onclick = (ev) => { navigator.clipboard?.writeText(part.code); ev.target.textContent = "✓ copied"; setTimeout(() => ev.target.textContent = "⎘ copy", 1400); }; d.appendChild(c); }
        else { const pr = document.createElement("div"); pr.style.whiteSpace = "pre-wrap"; const re = /!?\[([^\]]*)\]\(([^)]+)\)/g; let last = 0, m; while ((m = re.exec(part.prose))) { if (m.index > last) pr.appendChild(document.createTextNode(part.prose.slice(last, m.index))); const mel = mediaEl(m[2].trim(), m[1]); if (mel) pr.appendChild(mel); else pr.appendChild(document.createTextNode(m[0])); last = re.lastIndex; } if (last < part.prose.length) pr.appendChild(document.createTextNode(part.prose.slice(last))); d.appendChild(pr); }
      }
    } else {
      // No visible answer — reasoning-model / thinking-mode case. Say so instead of a blank bubble.
      const note = document.createElement("div"); note.className = "mut"; note.style.fontSize = "12.5px";
      note.textContent = reasoning
        ? "The model produced only reasoning — expand “Thinking” above, or ask it to answer directly."
        : "The model returned no visible text — it may be in “thinking” mode. Ask it to answer directly, or turn off reasoning in the provider’s engine (e.g. add /no_think to the prompt).";
      d.appendChild(note);
    }
    if (meta) { const mm = document.createElement("div"); mm.innerHTML = metaRow(meta); d.appendChild(mm.firstChild); }
    return d;
  }
  function renderChat() {
    const s = sessions[curChat]; const th = $("#thread"); if (!s) { th.innerHTML = ""; return; }
    $("#chattitle").textContent = s.t; th.innerHTML = "";
    if (!s.m.length) { th.innerHTML = `<div class="mut" style="margin:auto;text-align:center;font-size:12.5px">Ask anything — requests route through the network to whichever provider serves the model.</div>`; return; }
    for (const x of s.m) { if (x[0] === "me") { const d = document.createElement("div"); d.className = "msg me"; d.textContent = x[1]; th.appendChild(d); } else th.appendChild(botEl(x[1], x[2], x[3])); }
    th.scrollTop = th.scrollHeight;
  }

  // ── sharing = provider role; announce switches reflect + drive it ──
  let sharingBusy = false;
  async function setSharing(on) {
    if (sharingBusy) return; sharingBusy = true;
    try {
      if (on) {
        if (!engines.some((e) => e.models.length)) { go("engines"); toast("No local models found — start or install an engine to share"); return; }
        await call("start_provider");
      } else {
        await call("stop_provider");
      }
      await refresh();
    } catch (e) { toast(`${on ? "start" : "stop"} sharing failed: ${e}`); }
    finally { sharingBusy = false; }
  }
  async function toggleSharing() { await setSharing(!state?.provider?.status?.running); }
  $("#sharecta").onclick = () => go("share");   // #1: title-bar CTA routes to the Share tab
  $("#sharetoggle").onclick = toggleSharing;    // start/stop sharing now lives in the Share view

  // ── gateway lifecycle (Local API) ──
  async function ensureGateway() {
    if (state?.gateway?.status?.running) return true;
    try { await call("start_gateway"); } catch (e) { toast(`Local API failed to start: ${e}`); return false; }
    for (let i = 0; i < 24; i++) { await new Promise((r) => setTimeout(r, 300)); try { if (await call("gateway_health")) break; } catch {} }
    await refresh(); return true;
  }

  // ── send ──
  async function doSend(text, fromHome) {
    const model = curModel();
    if (!text) return;
    if (!model) { toast("No routable model yet — connecting to the network…"); return; }
    if (fromHome || !curChat) { curChat = newSession(text.slice(0, 34)); }
    const s = sessions[curChat];
    let content = text;
    if (attachments.length) { content = attachments.map((a) => `--- file: ${a.name} ---\n${a.text}`).join("\n\n") + `\n\n${text}`; attachments = []; }
    s.m.push(["me", content]); if (s.t === "New chat") s.t = text.slice(0, 34); saveSessions();
    if (fromHome) go("chat");
    renderRecents(); renderChat();
    const th = $("#thread"); const wait = document.createElement("div"); wait.className = "mut"; wait.style.fontSize = "12.5px"; wait.textContent = "thinking…"; th.appendChild(wait); th.scrollTop = th.scrollHeight;
    if (!(await ensureGateway())) { wait.remove(); return; }
    const messages = s.m.map((x) => ({ role: x[0] === "me" ? "user" : "assistant", content: x[1] }));
    const t0 = Date.now();
    try {
      // Tauri v2 maps camelCase JS args → snake_case Rust params, so `maxTokens` binds to the
      // command's `max_tokens`. Send the slider's current value; if it's somehow missing, omit it
      // and let the Rust command apply its generous default (a reasoning model needs well over the
      // old 512 or it spends the whole budget on hidden thinking and returns nothing).
      const tokLimit = Math.round(+($("[data-label='tokk']")?.dataset.val)) || undefined;
      const resp = await call("chat_completion", { model, messages, maxTokens: tokLimit });
      wait.remove();
      const msg = resp?.choices?.[0]?.message || {};
      const split = splitThink(msg.content || "");                 // pull inline <think> out of the answer
      const reasoning = [msg.reasoning_content, split.reasoning].filter(Boolean).join("\n").trim();
      const reply = split.answer;                                  // may be "" — botEl shows a clear note
      const oh = resp?.openhydra || {};
      const meta = { model: resp?.model || model, tok: resp?.usage?.completion_tokens ?? "—", tps: oh.engine?.native_tps ? Math.round(oh.engine.native_tps) : "—", rtt: oh.hops_ms?.network_rtt ?? "—", at: new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }) };
      pushSample(tpsSamples, oh.engine?.native_tps, "oh_tps");   // rolling throughput for Activity
      pushSample(rttSamples, oh.hops_ms?.network_rtt, "oh_rtt"); // rolling latency for Activity
      s.m.push(["ai", reply, meta, reasoning || null]); saveSessions(); renderChat();
      if (resp?.usage?.completion_tokens) { usedTokens += resp.usage.completion_tokens; store.set("oh_used", usedTokens); renderStatusbar(); }
    } catch (e) {
      wait.remove(); const secs = Math.round((Date.now() - t0) / 1000);
      const err = document.createElement("div"); err.className = "msg ai"; err.style.color = "hsl(var(--danger))"; err.style.fontSize = "12.5px";
      err.textContent = /504|timeout|timed out/i.test(String(e)) ? `The provider didn't respond in time (${secs}s). Cold model loads can be slow — try again; it warms up.`
        : /control character|expected value|invalid|parse|json/i.test(String(e)) ? `The provider's response couldn't be read — some reasoning models emit raw formatting that breaks parsing. Try turning off thinking on the provider (e.g. /no_think).`
        : `${e}`;
      th.appendChild(err);
    }
  }
  function send() { const c = $("#composer"); const t = c.textContent.trim(); c.textContent = ""; doSend(t, false); }
  $("#send").onclick = send;
  $("#composer").onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } };
  function homeSend() { const p = $("#homeprompt"); const t = p.textContent.trim(); p.textContent = ""; doSend(t, true); }
  $("#homesend").onclick = homeSend;
  $("#homeprompt").onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); homeSend(); } };
  // #2: paste PLAIN text only — a whole rich-HTML page (e.g. Wikipedia) into a contenteditable
  // injects huge markup and broke the app. Strip formatting + cap the size.
  const PASTE_CAP = 100 * 1024; // 100 KB
  function plainPaste(e) {
    e.preventDefault();
    let t = (e.clipboardData || window.clipboardData)?.getData("text/plain") || "";
    if (t.length > PASTE_CAP) { t = t.slice(0, PASTE_CAP); toast("Pasted text truncated to 100 KB"); }
    document.execCommand("insertText", false, t); // inserts as plain text at the caret, replacing any selection
  }
  ["#homeprompt", "#composer"].forEach((s) => $(s)?.addEventListener("paste", plainPaste));

  // image mode (wireframe) — the picker sets it; the network decides if an image model exists
  let chatMode = "text";
  function setChatMode(model) { chatMode = /flux|sdxl|(^|[^a-z])sd($|[^a-z])/i.test(model || "") ? "image" : "text"; const img = chatMode === "image"; $("#composer").dataset.ph = img ? "Describe an image…" : "Message…"; $("#sendlbl").textContent = img ? "Generate" : "Send"; $("#txtparams").style.display = img ? "none" : ""; $("#imgparams").style.display = img ? "" : "none"; }

  // attachments (paperclip → hidden file input)
  $$('button[title="Attach files to context"]').forEach((b) => b.onclick = () => $("#attachfile").click());
  $("#attachfile").onchange = async (e) => { for (const f of e.target.files) { if (f.size > 512 * 1024) { toast(`${f.name}: too large (512 KB max)`); continue; } attachments.push({ name: f.name, text: await f.text() }); } e.target.value = ""; if (attachments.length) toast(`${attachments.length} file(s) attached to next message`); };

  // ── view renderers (populate the wireframe's exact tables/cards with live data) ──
  function renderView() {
    if (activeView === "share") renderShare();
    else if (activeView === "providers") renderProviders();
    else if (activeView === "engines") renderEngines();
    else if (activeView === "activity") renderActivity();
    else if (activeView === "ledger") renderLedger();
    else if (activeView === "connectors") renderConnectors();
    else if (activeView === "peers") renderPeers();
    else if (activeView === "settings") renderSettings();
    else if (activeView === "chat") renderChat();
  }
  // #7: hide previously-served-but-now-inactive models from the serve list.
  let hideInactive = store.get("oh_hideinactive", false);
  // #10: selected timeline range per chart host (persisted).
  const chartRange = store.get("oh_chartrange", { share: "7d", activity: "7d" });
  // Stable per-model hue for the timeline + legend.
  function modelColor(id) { let h = 0; for (const c of id) h = (h * 31 + c.charCodeAt(0)) % 360; return `hsl(${h} 60% 55%)`; }
  function fmtNum(n) { n = Math.round(n); return n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1) + "k" : String(n); }
  // #10: inline-SVG served-vs-used timeline, stacked by model, with a 24h/7d/30d range selector.
  // Self-contained (no chart lib — CSP-safe). `hostKey` = "share" | "activity".
  function renderChart(hostSel, hostKey) {
    const host = $(hostSel); if (!host) return;
    const range = chartRange[hostKey] || "7d";
    const { slots, models } = statsSeries(range);
    // Rank models by lifetime volume in-window; keep top 6, fold the rest into "other".
    const vol = {}; for (const s of slots) for (const [id, v] of Object.entries(s.models)) vol[id] = (vol[id] || 0) + v.s + v.u;
    const top = Object.keys(vol).sort((a, b) => vol[b] - vol[a]).slice(0, 6);
    const topSet = new Set(top);
    const colorOf = (id) => topSet.has(id) ? modelColor(id) : "hsl(var(--muted-foreground))";
    const hasData = Object.keys(vol).length > 0;
    // Chart geometry (viewBox units; scales responsively).
    const W = 640, H = 150, padL = 8, padR = 8, padB = 18, padT = 8, plotH = H - padB - padT;
    const n = slots.length, groupW = (W - padL - padR) / n, barW = Math.min(14, groupW * 0.36);
    let maxV = 1; for (const s of slots) { let sv = 0, uv = 0; for (const v of Object.values(s.models)) { sv += v.s; uv += v.u; } maxV = Math.max(maxV, sv, uv); }
    const y = (v) => padT + plotH - (v / maxV) * plotH;
    const rangeLabel = { "24h": "last 24 hours", "7d": "last 7 days", "30d": "last 30 days" }[range];
    const xLabels = () => {
      const step = range === "24h" ? 6 : range === "7d" ? 1 : 5, out = [];
      for (let i = 0; i < n; i++) if (i % step === 0) {
        const cx = padL + i * groupW + groupW / 2;
        const end = slots[i].end, d = new Date(end);
        const lbl = range === "24h" ? d.getHours() + "h" : (d.getMonth() + 1) + "/" + d.getDate();
        out.push(`<text x="${cx.toFixed(1)}" y="${H - 5}" text-anchor="middle" font-size="9" fill="hsl(var(--muted-foreground))">${lbl}</text>`);
      }
      return out.join("");
    };
    let bars = "";
    slots.forEach((s, i) => {
      const gx = padL + i * groupW;
      // served bar (left), used bar (right); each stacked by model.
      const stack = (entries, x0, key) => {
        let acc = 0, seg = "";
        // stable order: top models first, then others
        const ids = Object.keys(entries).sort((a, b) => (vol[b] || 0) - (vol[a] || 0));
        for (const id of ids) {
          const val = entries[id][key]; if (!val) continue;
          const h = (val / maxV) * plotH, yTop = y(acc + val);
          seg += `<rect x="${x0.toFixed(1)}" y="${yTop.toFixed(1)}" width="${barW.toFixed(1)}" height="${Math.max(0.5, h).toFixed(1)}" fill="${colorOf(id)}" opacity="${key === "u" ? 0.55 : 1}"><title>${esc(id)} · ${key === "s" ? "served" : "used"} ${val}</title></rect>`;
          acc += val;
        }
        return seg;
      };
      const gap = 2, cx = gx + groupW / 2;
      bars += stack(s.models, cx - barW - gap / 2, "s");
      bars += stack(s.models, cx + gap / 2, "u");
    });
    const legend = hasData
      ? top.map((id) => `<span style="display:inline-flex;align-items:center;gap:5px;font-size:11px;color:hsl(var(--muted-foreground))"><span style="width:9px;height:9px;border-radius:2px;background:${modelColor(id)}"></span>${esc(id)}</span>`).join("")
        + (Object.keys(vol).length > 6 ? `<span style="font-size:11px;color:hsl(var(--muted-foreground))">+${Object.keys(vol).length - 6} more</span>` : "")
      : "";
    const chip = (r) => `<span class="btn ${range === r ? "outline" : "ghost"} sm chartrange" data-r="${r}" data-host="${hostKey}" style="padding:2px 8px;font-size:11px">${r}</span>`;
    host.innerHTML = `
      <div class="row" style="align-items:center;margin-bottom:6px"><div class="ctitle" style="font-size:13px">Served vs used · ${rangeLabel}</div><div class="grow"></div>${chip("24h")}${chip("7d")}${chip("30d")}</div>
      ${hasData
        ? `<svg viewBox="0 0 ${W} ${H}" width="100%" preserveAspectRatio="xMidYMid meet" style="display:block">
             <line x1="${padL}" y1="${padT + plotH}" x2="${W - padR}" y2="${padT + plotH}" stroke="hsl(var(--border))" stroke-width="1"/>
             ${bars}${xLabels()}
           </svg>
           <div style="display:flex;flex-wrap:wrap;gap:12px;margin-top:8px;align-items:center">
             <span style="display:inline-flex;align-items:center;gap:5px;font-size:11px;color:hsl(var(--muted-foreground))"><span style="width:9px;height:9px;border-radius:2px;background:hsl(var(--foreground))"></span>served (solid) · used (faded)</span>
             ${legend}
           </div>`
        : `<div class="mut" style="padding:28px 0;text-align:center;font-size:12.5px">No served or used tokens in the ${rangeLabel} yet.<br/>Serve a model or run a chat — the timeline fills in from your node's activity.</div>`}
    `;
    $$(`${hostSel} .chartrange`).forEach((c) => c.onclick = () => { chartRange[hostKey] = c.dataset.r; store.set("oh_chartrange", chartRange); renderChart(hostSel, hostKey); });
  }
  const engineFor = (m) => { for (const e of engines) if (e.models.includes(m)) return e.label; return null; };

  // ── per-model share allowlist (Share view toggles) ──
  // settings.shared_models empty ⇒ share ALL detected models (the default). Non-empty ⇒ only those.
  // A per-model toggle edits this list; the provider announces AND serves only shared models (the
  // gate is enforced agent-side). The list is read at provider start, so a change applies on the
  // next Sharing restart — we don't auto-restart a running provider mid-serve.
  function shareActiveModels() { return [...new Set(engines.flatMap((e) => e.models))]; }
  function isModelShared(m) { const s = state?.settings?.shared_models || []; return s.length === 0 || s.includes(m); }
  async function saveSharedModels(next) {
    const settings = Object.assign({}, state?.settings || {}, { shared_models: next });
    if (state?.settings) state.settings.shared_models = next; // optimistic local update
    try { await call("save_settings", { settings }); } catch (e) { toast(`Save failed: ${e}`); return; }
    toast(state?.provider?.status?.running ? "Saved — restart Sharing to apply" : "Saved");
    renderShare();
  }
  async function toggleShareModel(m) {
    const all = shareActiveModels();
    const running = !!state?.provider?.status?.running;
    // The switch means "serving this model right now" = sharing is on AND m is in the share set.
    const servingNow = running && isModelShared(m);
    if (servingNow) {
      // Turning a served model OFF. If it's the LAST one still served, the intent is "share
      // nothing" — which the allowlist can't encode (empty = share all), so stop sharing entirely.
      const othersServed = all.filter((x) => x !== m && isModelShared(x)).length;
      if (othersServed === 0) { await setSharing(false); return; }
      // Otherwise just narrow the allowlist (applies on the next Sharing restart).
      let list = (state?.settings?.shared_models || []).slice();
      if (list.length === 0) list = all.slice();               // materialize "share all" first
      list = list.filter((x) => x !== m);
      if (all.length && all.every((x) => list.includes(x))) list = []; // all ⇒ default
      await saveSharedModels(list);
    } else {
      // Turning a model ON.
      let list = (state?.settings?.shared_models || []).slice();
      if (!running && list.length === 0) list = [m];           // from a stopped/default state, share just this one
      else if (!list.includes(m)) list.push(m);
      if (all.length && all.every((x) => list.includes(x))) list = []; // all ⇒ default
      await saveSharedModels(list);
      if (!running) await setSharing(true);                    // begin serving
    }
  }
  function renderShare() {
    const p = state?.provider?.status, running = !!p?.running, t = snap?.transfers;
    const head = $("#v-share .row .ctitle").parentElement;
    head.querySelector(".badge").innerHTML = running ? '<span class="dot ok"></span>provider running' : '<span class="dot"></span>not sharing';
    head.querySelector(".badge").className = "badge " + (running ? "ok" : "secondary"); head.querySelector(".badge").style.marginLeft = "8px";
    // #1: start/stop lives here now (the title-bar control became a CTA that routes here).
    const stb = $("#sharetoggle");
    if (stb) { stb.textContent = running ? "Stop sharing" : "Start sharing"; stb.className = "btn sm " + (running ? "outline" : "brand"); stb.style.marginLeft = "12px"; }
    const up = snap?.uptime_secs != null ? ` · up ${fmtUptime(snap.uptime_secs)}` : "";
    head.querySelector(".mut").textContent = `${running ? (p.announced ?? 0) : 0} models announced · gateway :16527${up}`;
    const k = $$("#v-share .g4 .kpi .val");
    // #7: durable lifetime totals (survive restart; `used` counts external clients too).
    const served = totalServed(), used = totalUsed();
    // #3 fix: "Credits earned" = net CONTRIBUTION (served − used), which RISES with serving and
    // falls with using — the earlier `economy.total_credit` summed counterparty balances and
    // moved the wrong way (provider's number dropped on serve). Contribution-based per decision.
    const netCredits = served - used;
    const ratio = used > 0 ? (served / used) : (served > 0 ? null : 0);
    k[0].textContent = fmtNum(served);
    k[1].textContent = (netCredits >= 0 ? "+" : "") + Math.round(netCredits).toLocaleString();
    k[2].textContent = ratio == null ? "∞" : ratio ? ratio.toFixed(1) + "×" : "—";
    k[3].textContent = "—";   // own reputation is held by the peers we serve — not locally knowable
    $$("#v-share .g4 .kpi .sub")[0].innerHTML = `${t?.requests_served ?? 0} requests served`;
    $$("#v-share .g4 .kpi .sub")[1].textContent = "contribution · served − used (not money)";
    $$("#v-share .g4 .kpi .sub")[2].textContent = "served ÷ used";
    $$("#v-share .g4 .kpi .sub")[3].innerHTML = `<span class="mut">earned on the peers you serve</span>`;
    // #7: LIFETIME served-models list. Active = a model an installed engine can serve right now;
    // inactive = previously served (lifetime tokens on record) but not currently loaded — shown
    // dimmed, hideable. Lifetime tokens come from the durable accumulator so they survive restart.
    const per = t?.per_model || {};
    const active = new Set(engines.flatMap((e) => e.models));
    const lifetime = statModels().filter((id) => lifetimeServed(id) > 0);
    const allModels = [...new Set([...engines.flatMap((e) => e.models), ...lifetime])]
      .sort((a, b) => (lifetimeServed(b) || 0) - (lifetimeServed(a) || 0));
    const rows = [];
    for (const m of allModels) {
      const isActive = active.has(m);
      if (!isActive && hideInactive) continue;
      const pm = per[m] || {};
      const tokens = lifetimeServed(m) || pm.tokens || 0;
      const reqs = pm.requests ?? "—";
      const tps = pm.avg_native_tps ? Math.round(pm.avg_native_tps) : "—";
      const status = isActive ? `<span class="badge ${running ? "ok" : "secondary"}">${running ? "live" : "ready"}</span>` : `<span class="badge secondary">inactive</span>`;
      const ann = isActive ? `<div class="switch ${running && isModelShared(m) ? "on" : ""}" data-share="${esc(m)}" title="${running ? "Serve this model on the network" : "Start sharing to serve this model"}"></div>` : `<span class="mut" style="font-size:11px">—</span>`;
      rows.push(`<tr${isActive ? "" : ' style="opacity:.5"'}><td>${modelIcon(m)}<b>${esc(m)}</b></td><td>${esc(engineFor(m) || "—")}</td><td class="num">${reqs}</td><td class="num">${fmtNum(tokens)}</td><td class="num">${tps}</td><td>${status}</td><td>${ann}</td></tr>`);
    }
    $("#servetable tbody").innerHTML = rows.join("") || `<tr><td colspan="7" class="mut">No engines answering — start Ollama, LM Studio, vLLM, llama.cpp, or Exo, then rescan.</td></tr>`;
    const hb = $("#hideinactive");
    if (hb) { const anyInactive = allModels.some((m) => !active.has(m)); hb.style.display = anyInactive ? "" : "none"; hb.classList.toggle("outline", hideInactive); hb.classList.toggle("ghost", !hideInactive); hb.textContent = hideInactive ? "Show inactive" : "Hide inactive"; hb.onclick = () => { hideInactive = !hideInactive; store.set("oh_hideinactive", hideInactive); renderShare(); }; }
    renderChart("#sharechart", "share");   // #10 timeline
    $("#v-share .badge.ok, #v-share .badge").parentElement && ($$("#v-share .card .row .badge")[0]);
    const ann = $("#v-share .card .row .badge.ok") || $("#servetable").closest(".card").querySelector(".badge");
    if (ann) { ann.textContent = `${running ? engines.reduce((n, e) => n + e.models.length, 0) : 0} announced`; ann.className = "badge " + (running && engines.length ? "ok" : "secondary"); }
    $$("#servetable [data-share]").forEach((sw) => sw.onclick = () => toggleShareModel(sw.dataset.share));
    // incoming strip
    // Honest "Incoming" strip: real cumulative serve activity while sharing; hidden when idle.
    // There is no live concurrent-in-flight telemetry, so we don't fabricate one (the old card
    // showed a hardcoded "2 requests in flight from 12D3…" mockup).
    const incCard = $("#incomingcard"), incText = $("#incomingtext");
    if (incCard && incText) {
      if (running) {
        const r = t?.requests_served ?? 0;
        incText.textContent = r > 0
          ? `${r} request${r === 1 ? "" : "s"} served · ${fmtNum(t?.tokens_served ?? 0)} tokens so far`
          : "Waiting for requests…";
        incCard.style.display = "flex";
      } else {
        incCard.style.display = "none";
      }
    }
    // wanted table — honest empty state until demand telemetry lands
    $("#wanttable tbody").innerHTML = `<tr><td colspan="4" class="mut">Fills in as network demand telemetry lands.</td></tr>`;
  }
  // #11b: which model rows are expanded to show their per-provider breakdown. A Set so the
  // 2.5s poll re-render doesn't collapse an open row (mirrors the peers-filter persistence rule).
  const expandedProviders = new Set();
  function renderProviders() {
    const models = netModels();
    const local = new Set(state?.provider?.status?.running ? engines.flatMap((e) => e.models) : []);
    const strip = $("#v-providers .card.pad");
    strip.querySelectorAll("b")[0].textContent = (snap?.transfers?.tokens_served ?? 0);  // tokens
    strip.querySelectorAll("b")[1].textContent = models.length;                          // models
    strip.querySelectorAll("b")[2].textContent = snap?.network?.peers?.length ?? 0;       // peers
    // Labels are now correct & honest in the template ("tokens served · models · peers · from your
    // node") — these are all LOCAL counts, so no "network-wide" claim until Tier-C stats land.
    $("#provcount").textContent = models.length;
    const q = ($("#search").value || "").toLowerCase();
    const byOh = repByOpenhydra();
    // Group the announced providers by model so a model served by several peers can expand (#11b).
    const provsByModel = {};
    for (const p of (snap?.network?.known_providers || [])) (provsByModel[p.model_id] ||= []).push(p);
    const rows = models.filter((m) => !q || m.toLowerCase().includes(q)).map((m) => {
      const remote = provsByModel[m] || [];
      const cnt = (seenCount[m] || 0) + (local.has(m) ? 1 : 0);   // last-known count, sticky
      const tps = modelAvgTps(m);                                  // only for models we serve
      const rep = modelReputation(m, byOh);                        // earned rep of its providers
      const canExpand = remote.length > 1;                         // >1 provider → disclosure
      const open = expandedProviders.has(m);
      const caret = canExpand ? `<span class="prowtog" data-m="${esc(m)}" style="cursor:pointer;display:inline-block;width:14px;color:hsl(var(--muted-foreground));transition:transform .12s;transform:rotate(${open ? 90 : 0}deg)">▸</span>` : '<span style="display:inline-block;width:14px"></span>';
      const idle = modelIdle(m);   // W2: seen but quiet → dim, don't drop (rides gossip gaps)
      let html = `<tr class="prov" data-cat="${modelCat(m)}" data-m="${esc(m)}"${idle ? ' style="opacity:.5"' : ""}><td>${caret}${modelIcon(m)}<b>${esc(m)}</b>${local.has(m) ? ' <span class="mut">· your machine</span>' : idle ? ' <span class="mut" style="font-size:10.5px">· idle</span>' : ""}</td><td class="num">${cnt || "—"}</td><td class="num${tps == null ? " mut" : ""}">${tps == null ? "—" : tps}</td><td>${repBadge(rep)}</td><td class="num mut">—</td></tr>`;
      if (canExpand && open) {
        html += remote.map((p) => {
          const prep = byOh[p.openhydra_peer_id];
          return `<tr class="provsub" data-cat="${modelCat(m)}" data-for="${esc(m)}"><td style="padding-left:34px"><span class="mono mut">${peerShort(p.libp2p_peer_id)}</span></td><td class="num mut">1</td><td class="num mut">—</td><td>${repBadge(prep)}</td><td class="num mut">—</td></tr>`;
        }).join("");
      }
      return html;
    });
    $("#provtable tbody").innerHTML = rows.join("") || `<tr><td colspan="5" class="mut">${snap ? "No models discovered yet — they appear as peers announce." : "Connecting…"}</td></tr>`;
    const cat = $("#provchips .chip.on")?.dataset.cat || "all";
    $$("#provtable .prov, #provtable .provsub").forEach((r) => r.style.display = (cat === "all" || r.dataset.cat === cat) ? "" : "none");
    $$("#provtable .prowtog").forEach((tog) => tog.onclick = (e) => {
      e.stopPropagation(); const m = tog.dataset.m;
      if (expandedProviders.has(m)) expandedProviders.delete(m); else expandedProviders.add(m);
      renderProviders();
    });
  }
  // ── Tier-1 engine installer (B5): consent → stream install://progress → refresh ──
  const tauriEvent = window.__TAURI__?.event;
  const mockInstallCbs = [];
  // Subscribe to install progress; returns an unlisten fn. Real Tauri event bus if present,
  // else the in-page mock bus the browser-preview install driver feeds.
  function onInstallProgress(cb) {
    if (tauriEvent?.listen) {
      let un = null, dead = false;
      tauriEvent.listen("install://progress", (e) => cb(e.payload)).then((u) => { if (dead) u(); else un = u; });
      return () => { dead = true; if (un) un(); };
    }
    mockInstallCbs.push(cb);
    return () => { const i = mockInstallCbs.indexOf(cb); if (i >= 0) mockInstallCbs.splice(i, 1); };
  }
  function mockEmitInstall(ev) { mockInstallCbs.slice().forEach((cb) => cb(ev)); }

  function installOverlay() {
    let el = $("#installov");
    if (!el) {
      el = document.createElement("div");
      el.id = "installov";
      el.style.cssText = "position:fixed;inset:0;background:rgba(0,0,0,.45);display:none;align-items:center;justify-content:center;z-index:9999";
      el.innerHTML = `<style>@keyframes instIndet{0%{transform:translateX(-100%)}100%{transform:translateX(280%)}} #instBarFill.indet{width:36%!important;animation:instIndet 1.15s ease-in-out infinite}</style>
        <div class="card pad" style="width:min(560px,92vw);max-height:82vh;display:flex;flex-direction:column;gap:10px">
        <div class="row"><b id="instTitle" style="font-size:14px">Install</b><button id="instX" class="btn outline sm" style="margin-left:auto">✕</button></div>
        <div id="instBody" style="font-size:12.5px;line-height:1.5"></div>
        <div id="instBar" style="display:none;height:8px;border-radius:6px;background:rgba(127,127,127,.15);overflow:hidden"><div id="instBarFill" style="height:100%;width:0%;background:var(--brand,#2f9e6f);transition:width .35s ease"></div></div>
        <pre id="instLog" class="mono" style="display:none;background:rgba(127,127,127,.1);border-radius:8px;padding:10px;overflow:auto;max-height:42vh;font-size:11px;margin:0;white-space:pre-wrap"></pre>
        <div class="row" id="instActions"></div></div>`;
      document.body.appendChild(el);
      el.querySelector("#instX").onclick = () => el.style.display = "none";
      el.addEventListener("click", (e) => { if (e.target === el) el.style.display = "none"; });
    }
    return el;
  }

  async function startInstall(label, name) {
    const ov = installOverlay();
    const body = $("#instBody"), log = $("#instLog"), actions = $("#instActions");
    $("#instTitle").textContent = `Install ${name}`;
    log.style.display = "none"; log.textContent = ""; actions.innerHTML = ""; body.textContent = "Checking…"; ov.style.display = "flex";
    const okBtn = (t, fn) => { actions.innerHTML = `<button class="btn brand sm" id="instOk" style="margin-left:auto">${t}</button>`; $("#instOk").onclick = fn || (() => ov.style.display = "none"); };
    let variant = null; // null = platform default (desktop app where both exist)
    // (Re)fetch + render the plan for the current variant — lets the app/CLI toggle swap the summary.
    const render = async () => {
      body.textContent = "Checking…"; actions.innerHTML = "";
      let plan;
      try { plan = await call("install_plan", { engine: label, accel: null, variant }); }
      catch (e) { body.textContent = `Could not plan the install: ${e}`; okBtn("Close"); return; }
      if (plan.already_installed) { body.textContent = `${name} is already installed. Nothing to do.`; okBtn("OK"); return; }
      if (plan.blocker) { body.innerHTML = `<b>Can't install ${name} here</b><br><span class="mut">${esc(plan.blocker)}</span>${plan.summary ? `<br><br><span class="mut">${esc(plan.summary)}</span>` : ""}`; okBtn("OK"); return; }
      if (!plan.supported) { body.innerHTML = `No one-click installer for ${name} yet.<br><span class="mut">${esc(plan.summary)}</span>`; okBtn("OK"); return; }
      const warn = plan.verified ? "" : `<br><br><span class="badge warn">unverified on your OS</span> <span class="mut">This recipe isn't yet end-to-end tested here — it may take a few minutes and could fail.</span>`;
      const cur = variant === "cli" ? "cli" : "app";
      // App-vs-CLI toggle where both exist (ComfyUI / Exo on macOS: desktop app vs headless CLI).
      const toggle = plan.cli_available
        ? `<div class="row" style="gap:6px;margin-bottom:8px"><button class="btn ${cur === "app" ? "brand" : "outline"} sm" id="varApp">Desktop app</button><button class="btn ${cur === "cli" ? "brand" : "outline"} sm" id="varCli">Headless CLI</button></div>`
        : "";
      body.innerHTML = `${toggle}${esc(plan.summary)}${warn}`;
      if (plan.cli_available) { $("#varApp").onclick = () => { variant = "app"; render(); }; $("#varCli").onclick = () => { variant = "cli"; render(); }; }
      actions.innerHTML = `<button class="btn outline sm" id="instCancel">Cancel</button><button class="btn brand sm" id="instGo" style="margin-left:auto">Install ${name}${cur === "cli" ? " · CLI" : ""}</button>`;
      $("#instCancel").onclick = () => ov.style.display = "none";
      $("#instGo").onclick = () => runInstall(label, name, variant);
    };
    render();
  }

  async function runInstall(label, name, variant) {
    const body = $("#instBody"), log = $("#instLog"), actions = $("#instActions");
    const bar = $("#instBar"), fill = $("#instBarFill");
    body.textContent = `Installing ${name}…`; log.style.display = "block"; log.textContent = ""; actions.innerHTML = "";
    bar.style.display = "none"; fill.style.width = "0%";
    const append = (s) => { log.textContent += s + "\n"; log.scrollTop = log.scrollHeight; };
    // Bar states: determinate (we know %, our own downloads) vs indeterminate (vendor tool is
    // working — brew/uv/git/pip manage their own download, so we show an animated "working" bar).
    const setDet = (pct) => { bar.style.display = "block"; fill.classList.remove("indet"); fill.style.opacity = "1"; fill.style.width = pct + "%"; };
    const setIndet = () => { bar.style.display = "block"; fill.classList.add("indet"); fill.style.opacity = "1"; fill.style.width = ""; };
    const hideBar = () => { bar.style.display = "none"; fill.classList.remove("indet"); };
    const un = onInstallProgress((ev) => {
      if (ev.engine !== label) return;
      if (ev.kind === "phase") { setIndet(); body.textContent = ev.message === "downloading" ? "Downloading…" : ev.message === "installing" ? "Installing…" : ev.message; append("▸ " + ev.message); }
      else if (ev.kind === "download") { if (typeof ev.percent === "number") setDet(ev.percent); else setIndet(); body.textContent = "Downloading — " + ev.message; }
      else if (ev.kind === "log") append(ev.message);
      else if (ev.kind === "done") { un(); hideBar(); body.innerHTML = `<span class="badge ok">done</span> ${esc(ev.message)}`; append("✓ " + ev.message); actions.innerHTML = `<button class="btn brand sm" id="instFin" style="margin-left:auto">Done</button>`; $("#instFin").onclick = () => { $("#installov").style.display = "none"; refreshEngines(); }; refreshEngines(); }
      else if (ev.kind === "error") { un(); hideBar(); body.innerHTML = `<span class="badge warn">failed</span> ${esc(ev.message)}`; append("✗ " + ev.message); actions.innerHTML = `<button class="btn outline sm" id="instFin" style="margin-left:auto">Close</button>`; $("#instFin").onclick = () => $("#installov").style.display = "none"; }
    });
    try { await call("install_engine", { engine: label, accel: null, variant: variant || null }); }
    catch (e) { /* backend already emitted an error event; guard the mock/no-event path */ mockEmitInstall({ engine: label, kind: "error", message: String(e) }); }
  }

  // ── Host hardware panel (system_info) — shown in the Engines header, LM-Studio style ──
  let sysInfo = null;
  function fmtGB(bytes) { if (!bytes) return "—"; const gb = bytes / 1073741824; const s = gb >= 10 ? String(Math.round(gb)) : gb.toFixed(1).replace(/\.0$/, ""); return s + " GB"; }
  function fmtSystem(s) {
    if (!s || !s.cpu) return "";
    const bits = [s.cpu, fmtGB(s.ram_bytes)];
    if (s.gpus && s.gpus.length) {
      const g = s.gpus[0];
      if (g.unified) { const m = (g.name.match(/\((\d+-core GPU)\)/) || [])[1]; bits.push(m ? m + " · unified" : "unified memory"); }
      else { bits.push(g.name + (g.vram_bytes ? " " + fmtGB(g.vram_bytes) : "")); }
    }
    return bits.join(" · ");
  }
  async function refreshSystem() { try { sysInfo = await call("system_info"); } catch { sysInfo = null; } const el = $("#syshw"); if (el) el.textContent = fmtSystem(sysInfo) || `${sysInfo?.os || ""} · ${sysInfo?.arch || ""}`.trim(); }

  // Labels MUST match `detect_engines` (agent adapter `engine_name`) so the "running" badge
  // resolves — e.g. `llama.cpp`, not `llama-cpp` (the latter silently never matched detection).
  const ENGINES = [["Ollama", "ollama", "General-purpose local LLMs."], ["LM Studio", "lm-studio", "MLX-optimised models on Apple silicon."], ["llama.cpp", "llama.cpp", "Lightweight GGUF runtime."], ["ComfyUI", "comfyui", "Image generation — Stable Diffusion, Flux."], ["vLLM", "vllm", "High-throughput serving. Needs Python + GPU."], ["Exo", "exo", "Shard big models across your devices."]];
  // Engines OpenHydra can start on demand (self-serving; no model/cluster arg) → get a "Run" CTA.
  const RUNNABLE = new Set(["ollama", "lm-studio"]);
  function renderEngines() {
    const det = Object.fromEntries(engines.map((e) => [e.label, e]));
    const cards = $$("#v-engines .grid.g3 .card");
    ENGINES.forEach(([name, label, desc], i) => {
      // All six engines now have real (probe-then-install) recipes → none is a plain "guided" toast.
      const c = cards[i]; if (!c) return;
      const d = det[label];                                // serving right now
      const installed = installedEngines.includes(label);  // present on disk (may be idle)
      const runnable = RUNNABLE.has(label);                // can self-serve (no model/cluster arg)
      c.querySelector("b").textContent = name;
      const badge = c.querySelector(".badge"); badge.style.marginLeft = "auto";
      badge.className = "badge " + (d ? "ok" : "");
      badge.innerHTML = d ? '<span class="dot ok"></span>running' : installed ? "installed" : "not installed";
      c.querySelectorAll(".mut")[0].textContent = desc;
      const foot = c.querySelectorAll(".row")[1]; const btn = foot.querySelector(".enginst"); const def = foot.querySelector(".badge");
      if (def) def.style.display = d && label === "ollama" ? "" : "none";
      // Three-state CTA: running → Manage; installed-but-idle → Run (self-serving) or Installed
      // (needs a model to serve); not installed → Install (drives the Tier-1 installer).
      let txt, cls, act;
      if (d) { txt = "Manage"; cls = "outline"; act = () => toast(`${name} is running — manage models from Share`); }
      else if (installed && runnable) { txt = "Run"; cls = "brand"; act = () => runEngine(label, name); }
      else if (installed) { txt = "Installed"; cls = "outline"; act = () => toast(`${name} is installed — start it with a model to serve`); }
      else { txt = "Install"; cls = "brand"; act = () => startInstall(label, name); }
      btn.textContent = txt; btn.className = "btn " + cls + " sm enginst"; btn.style.marginLeft = "auto"; btn.onclick = act;
    });
    // recommended: honest until a model store lands
    $("#rectable tbody").innerHTML = `<tr><td colspan="5" class="mut">One-click downloads land with the model store — for now, pull with your engine (e.g. <span class="mono">ollama pull</span>) and it appears in Share.</td></tr>`;
  }
  function renderActivity() {
    const t = snap?.transfers, k = $$("#v-activity .g4 .kpi .val");
    const served = totalServed(), used = totalUsed();   // #7 durable lifetime totals
    const netCredits = served - used;   // #3: net balance rises with serving, falls with using
    const ratio = used > 0 ? (served / used) : (served > 0 ? null : 0);
    k[0].textContent = fmtNum(served);
    k[1].textContent = fmtNum(used);
    k[2].textContent = (netCredits >= 0 ? "+" : "") + Math.round(netCredits).toLocaleString();
    k[3].textContent = ratio == null ? "∞" : ratio ? ratio.toFixed(1) + "×" : "—";
    $$("#v-activity .g4 .kpi .sub")[0].innerHTML = `<span class="dot ok"></span>${t?.receipts_ledgered ?? 0} receipts co-signed`;
    $$("#v-activity .g4 .kpi .sub")[1].textContent = "this device";
    $$("#v-activity .g4 .kpi .sub")[2].textContent = "net contribution · served − used";
    $$("#v-activity .g4 .kpi .sub")[3].textContent = "served ÷ used";
    // Rolling per-chat throughput/latency — the only place an aggregated TPS/RTT is honest,
    // since the agent emits these per-request. Uptime rounds out the "your node" picture.
    const note = $("#v-activity .mut");
    if (note) {
      const at = mean(tpsSamples), ar = mean(rttSamples);
      const parts = [];
      if (at != null) parts.push(`avg ${Math.round(at)} t/s`);
      if (ar != null) parts.push(`${Math.round(ar)} ms RTT`);
      if (snap?.uptime_secs != null) parts.push(`node up ${fmtUptime(snap.uptime_secs)}`);
      note.textContent = (parts.length ? `Your recent chats: ${parts.join(" · ")}. ` : "") + "Full transaction history lives in Network › Ledger.";
    }
    renderChart("#actchart", "activity");   // #10 timeline
  }
  // Relative "Nm ago" timestamp for the ledger rows.
  function relTime(ms) {
    const s = Math.max(0, (Date.now() - ms) / 1000);
    if (s < 60) return Math.round(s) + "s ago";
    const m = s / 60; if (m < 60) return Math.round(m) + "m ago";
    const h = m / 60; if (h < 24) return Math.round(h) + "h ago";
    return Math.round(h / 24) + "d ago";
  }
  // #5: real ledger rows from the agent's recent-transaction ring (served rows from the provider
  // process, used rows from the gateway; merged + newest-first by the desktop). The credit column
  // is a signed contribution unit (served +, used −; "not money"), not a wallet balance.
  function renderLedger() {
    const t = snap?.transfers, rows = t?.recent || [];
    $("#v-ledger .row .mut").textContent = rows.length
      ? `${rows.length} recent · ${t?.receipts_ledgered ?? 0} co-signed · ${t?.tokens_served ?? 0} served / ${t?.tokens_consumed ?? 0} used tokens`
      : `${t?.receipts_ledgered ?? 0} receipts · ${t?.tokens_served ?? 0} tokens served`;
    $("#ledgertable tbody").innerHTML = rows.length
      ? rows.slice(0, 100).map((r) => {
          const served = r.kind === "served", cr = (served ? "+" : "−") + (r.tokens / 100).toFixed(1);
          return `<tr><td class="mut">${relTime(r.ts_ms)}</td><td><span class="badge ${served ? "ok" : "secondary"}">${served ? "served" : "used"}</span></td><td>${modelIcon(r.model)}${esc(r.model)}</td><td class="mono">${peerShort(r.counterparty)}</td><td class="num">${r.tokens}</td><td class="num ${served ? "up" : ""}"${served ? "" : ' style="color:hsl(var(--danger))"'}>${cr}</td></tr>`;
        }).join("")
      : `<tr><td colspan="6" class="mut">No transactions yet — serve a model or run a chat, and co-signed receipts appear here. Recent activity is kept in memory (launch the agent with a ledger DB for full history).</td></tr>`;
  }
  function renderConnectors() {
    $$("#v-connectors .cp").forEach((b) => b.onclick = (e) => { e.stopPropagation(); const sn = b.closest(".conncard")?.querySelector(".snippet"); navigator.clipboard?.writeText((sn?.textContent || "").trim()); toast("Copied"); });
    wireConnectors();
  }

  const AUTO_MODEL = "openhydra/auto";
  function surfLabel(s) { return s === "terminal" ? "Terminal" : s === "editor" ? "Editor" : "App"; }
  // Live network model ids for the selector (sticky-smoothed; may be empty before the first snapshot).
  function liveModels() {
    return [...new Set([...(snap?.network?.known_models || []), ...Object.keys(seenModels || {})])]
      .filter(Boolean).sort();
  }

  // Detect installed tools + render the surface switcher, live snippet, model selector, and
  // state-driven actions on the actionable connector cards.
  async function wireConnectors() {
    let statuses = [];
    try { statuses = await call("connector_status"); } catch {}
    const byKey = {}; statuses.forEach((s) => (byKey[s.key] = s));
    $$("#v-connectors .conncard[data-key]").forEach((card) => {
      const st = byKey[card.dataset.key];
      if (!st) return;
      if (!card.dataset.surface || !(st.surfaces || []).includes(card.dataset.surface)) {
        card.dataset.surface = (st.surfaces && st.surfaces[0]) || "terminal";
      }
      renderSurfaceSwitcher(card, st);
      renderModelSelector(card, st);
      renderSnippet(card, st);
      renderActions(card, st);
    });
  }

  // The repurposed "direct" slot: a segmented switcher for dual-surface tools, a static chip otherwise.
  function renderSurfaceSwitcher(card, st) {
    const slot = $(".csurf", card); if (!slot) return;
    const surfaces = st.surfaces && st.surfaces.length ? st.surfaces : ["terminal"];
    if (surfaces.length < 2) { slot.innerHTML = `<span class="solo">${surfLabel(surfaces[0])}</span>`; return; }
    slot.innerHTML = surfaces.map((s) => `<button data-s="${s}"${s === card.dataset.surface ? ' class="on"' : ""}>${surfLabel(s)}</button>`).join("");
    $$("button", slot).forEach((b) => b.onclick = () => {
      card.dataset.surface = b.dataset.s;
      renderSurfaceSwitcher(card, st); renderSnippet(card, st); renderActions(card, st);
    });
  }

  // Model selector (picker tools only: opencode/pi/continue) — injected once, before the actions row.
  // Searchable multi-select: type-to-filter dropdown of all network model ids (or a custom typed id),
  // selections become removable chips. `openhydra/auto` is a permanent chip. Selection lives on
  // `card._models` (a Set) so it survives the periodic refresh() re-render.
  function renderModelSelector(card, st) {
    let box = $(".cmodels", card);
    if (!st.declares_models) { if (box) box.remove(); return; }
    // Build ONCE — rebuilding the whole selector every 2.5s (refresh) collapsed the open dropdown and
    // wiped focus/typed text. On later renders, only live-refresh the OPEN dropdown's option list.
    if (box) { if (box._liveRefresh) box._liveRefresh(); return; }
    // Seed from models ALREADY declared in the tool's config, so a re-Connect keeps them (the writers
    // replace the declared set, and this Set is what Connect sends).
    if (!card._models) card._models = new Set(st.declared_models || []);
    box = document.createElement("details"); box.className = "cmodels";
    card.insertBefore(box, $(".connact", card));
    box.innerHTML = `<summary>Models to declare in ${esc(st.label)} ▾</summary>`
      + `<div class="mchips"></div>`
      + `<div class="mcombo"><input class="mfilter" type="text" placeholder="add a model — scroll or type to filter…"><div class="mopts" hidden></div></div>`;
    const chipsEl = $(".mchips", box), input = $(".mfilter", box), opts = $(".mopts", box);

    const renderChips = () => {
      chipsEl.innerHTML = `<span class="mchip auto">${AUTO_MODEL} <span class="mut">· always</span></span>`
        + [...card._models].map((m) => `<span class="mchip">${esc(m)} <button class="mrm" data-v="${esc(m)}" title="Remove">×</button></span>`).join("");
      $$(".mrm", chipsEl).forEach((b) => b.onclick = () => { card._models.delete(b.dataset.v); renderChips(); renderSnippet(card, st); if (!opts.hidden) renderOpts(); });
    };
    const add = (v) => { v = (v || "").trim(); if (!v || v === AUTO_MODEL) return; card._models.add(v); input.value = ""; renderChips(); renderOpts(); renderSnippet(card, st); };
    const renderOpts = () => {
      const q = input.value.trim().toLowerCase();
      const avail = liveModels().filter((m) => !card._models.has(m) && m.toLowerCase().includes(q));
      let html = avail.map((m) => `<div class="mopt" data-v="${esc(m)}">${esc(m)}</div>`).join("");
      const typed = input.value.trim();
      if (typed && !liveModels().some((m) => m.toLowerCase() === typed.toLowerCase()) && !card._models.has(typed)) {
        html += `<div class="mopt add" data-v="${esc(typed)}">+ Add “${esc(typed)}”</div>`;
      }
      opts.innerHTML = html || `<div class="mopt mut">No matching models on the network</div>`;
      // mousedown (not click) + preventDefault so the option registers before the input's blur hides it.
      $$(".mopt[data-v]", opts).forEach((o) => o.onmousedown = (e) => { e.preventDefault(); add(o.dataset.v); });
      box._lastSig = liveModels().join(""); // remember the network's model set at this render
    };
    input.onfocus = () => { opts.hidden = false; renderOpts(); };
    input.oninput = () => { opts.hidden = false; renderOpts(); };
    input.onblur = () => setTimeout(() => { opts.hidden = true; }, 150);
    input.onkeydown = (e) => { if (e.key === "Enter") { e.preventDefault(); add(input.value); } };
    // Called on each 2.5s poll (via the build-once early-return): if the dropdown is OPEN and the
    // network's model set changed since the last render, refresh just the option list — preserving the
    // input's focus + typed text. Unchanged ⇒ skip, so an idle open list never reshuffles or loses scroll.
    box._liveRefresh = () => { if (!opts.hidden && liveModels().join("") !== box._lastSig) renderOpts(); };
    renderChips();
  }

  function selectedModels(card) { return card._models ? [...card._models] : []; }

  // The Terminal snippet is the live `openhydra <verb> …` command (reflecting the model selection);
  // the App/Editor snippet is a short instruction, since there's no command to copy.
  function renderSnippet(card, st) {
    const sn = $(".snippet", card); if (!sn) return;
    const models = selectedModels(card);
    if (card.dataset.surface === "terminal") {
      const lines = [];
      if (models.length) lines.push(`openhydra connect ${st.key}` + models.map((m) => ` --model ${m}`).join(""));
      lines.push(`openhydra ${st.natural_verb} ${st.key}`);
      sn.textContent = lines.join("\n");
    } else {
      const where = card.dataset.surface === "editor" ? "your editor" : "the app";
      const declare = models.length ? `\n# picker models: ${models.join(", ")}` : "";
      sn.textContent = `# Connect writes ${st.kind}'s config, then open ${where} — it reads it.${declare}\n# model: ${AUTO_MODEL}`;
    }
  }

  function renderActions(card, st) {
    const act = $(".connact", card); if (!act) return;
    const isTerminal = card.dataset.surface === "terminal";
    // Skip the rebuild when nothing that affects the buttons changed — otherwise the 2.5s refresh
    // would destroy the button DOM mid-interaction (e.g. clobbering a Test's "Testing…" state and
    // re-enabling it while the request is still in flight).
    const sig = `${st.installed}|${st.connected}|${isTerminal}`;
    if (act._sig === sig) return;
    act._sig = sig;
    const runLabel = st.connected ? (isTerminal ? "Run" : "Open") : (isTerminal ? "Connect &amp; Run" : "Connect &amp; Open");
    const statusHtml = st.connected
      ? `<span class="cconnected">● Connected</span>`
      : `<span class="cstat ${st.installed ? "on" : "off"}">●</span><span class="cstat-t">${st.installed ? "installed" : "not detected"}</span>`;
    act.innerHTML = statusHtml
      + (isTerminal ? `<button class="btn ghost sm ccopy" style="margin-left:auto">Copy</button>` : `<span style="margin-left:auto"></span>`)
      + `<button class="btn ghost sm ctest">Test</button>`
      + (st.connected ? `<button class="btn ghost sm cdisc">Disconnect</button>` : `<button class="btn ghost sm cwire">Connect</button>`)
      + `<button class="btn sm crun">${runLabel}</button>`;
    const on = (sel, fn) => { const b = $(sel, act); if (b) b.onclick = fn; };
    on(".ccopy", () => { navigator.clipboard?.writeText(($(".snippet", card)?.textContent || "").trim()); toast("Copied"); });
    on(".ctest", (e) => testGateway(e.target));
    on(".cwire", () => connectTool(card, st, false));
    on(".crun", () => connectTool(card, st, true));
    on(".cdisc", () => disconnectTool(card, st));
  }

  // Wire the tool (persist config via apply, declaring any selected models), then optionally run it:
  // Terminal → copy `openhydra launch <tool>`; App/Editor → open the GUI. `apply` backs up the original.
  async function connectTool(card, st, run) {
    const models = selectedModels(card), surf = card.dataset.surface;
    let rep;
    try { rep = await call("connector_apply", { key: st.key, models }); }
    catch (e) { toast("Connect failed: " + e); return; }
    if (!run) {
      toast(`Connected — ${rep.action} config` + (rep.backup ? " (original backed up)" : ""));
      wireConnectors(); return;
    }
    if (surf === "terminal") {
      const cmd = `openhydra launch ${st.key}`;
      try { await navigator.clipboard.writeText(cmd); toast(`Wired ✓ — copied "${cmd}". Paste in your terminal to run.`); }
      catch { toast(`Wired ✓ — run: ${cmd}`); }
    } else {
      try { await call("open_gui", { key: st.key }); toast(`Wired ✓ — opening ${surfLabel(surf)}…`); }
      catch (e) { toast(`Wired ✓ — couldn't open ${surfLabel(surf)} (${e}). Open it manually; it reads the config.`); }
    }
    wireConnectors();
  }

  async function disconnectTool(card, st) {
    if (!(await confirmModal(`Disconnect ${st.label}`, `Restores ${st.label}'s original config (removes the OpenHydra block).`, "Disconnect"))) return;
    try {
      const rep = await call("connector_disconnect", { key: st.key });
      const label = { restored: "original restored", stripped: "removed our block, kept your config", removed: "removed our config", "not-connected": "nothing to undo" }[rep.action] || rep.action;
      toast("Disconnected — " + label);
    }
    catch (e) { toast("Disconnect failed: " + e); return; }
    wireConnectors();
  }

  async function testGateway(btn) {
    const prev = btn.textContent; btn.textContent = "Testing…"; btn.disabled = true;
    try { const model = await call("connector_test"); toast("Gateway OK → " + model); }
    catch (e) { toast("Test failed: " + e); }
    finally { btn.textContent = prev; btn.disabled = false; }
  }

  function escapeHtml(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

  // Generic in-app confirm — window.confirm() is a no-op in the Tauri webview, so we can't gate on it.
  // Resolves true/false.
  function confirmModal(title, body, okLabel) {
    return new Promise((resolve) => {
      const back = document.createElement("div");
      back.className = "cmodal-back";
      back.innerHTML =
        `<div class="cmodal"><div class="cmodal-h"><b>${escapeHtml(title)}</b></div>` +
        `<div class="cmodal-b"><div class="cmodal-path">${escapeHtml(body)}</div></div>` +
        `<div class="cmodal-f"><button class="btn ghost sm cx">Cancel</button><button class="btn sm cok">${escapeHtml(okLabel || "OK")}</button></div></div>`;
      document.body.appendChild(back);
      const done = (v) => { back.remove(); resolve(v); };
      $(".cx", back).onclick = () => done(false);
      $(".cok", back).onclick = () => done(true);
      back.onclick = (e) => { if (e.target === back) done(false); };
    });
  }

  // libp2p ids of the infrastructure we're connected to (bootstraps + circuit relays) — these
  // aren't "peers" a user cares about, so we hide them from the Peers list.
  function infraPeerIds() {
    const s = new Set();
    (snap?.network?.relay_reservations || []).forEach((a) => { const m = a.match(/\/p2p\/([^/]+)\/p2p-circuit/); if (m) s.add(m[1]); });
    (state?.settings?.bootstraps || []).forEach((a) => { const m = a.match(/\/p2p\/([^/]+)/); if (m) s.add(m[1]); });
    return s;
  }
  let peerLimit = 10;   // "View more" bumps this
  function renderPeers() {
    if (!snap) { $("#peertable tbody").innerHTML = `<tr><td colspan="5" class="mut">Turn on Sharing or chat to connect, then peers appear here.</td></tr>`; const pc = $("#peercount"); if (pc) pc.textContent = "0 peers"; const pm = $("#peermore"); if (pm) pm.style.display = "none"; return; }
    const n = snap.network;
    const infra = infraPeerIds();
    const peers = (n.peers || []).filter((p) => !infra.has(p.peer_id));   // hide bootstraps/relays
    const repL = repByLibp2p();
    const shown = peers.slice(0, peerLimit);
    $("#peertable tbody").innerHTML = shown.length ? shown.map((p) => `<tr data-p="${p.path}"><td class="mono">${peerShort(p.peer_id)}</td><td><span class="badge ${p.path === "direct" ? "ok" : p.path === "relay" ? "warn" : "secondary"}">${p.path}</span></td><td class="num">${p.quic_direct_v6}</td><td>${repBadge(repL[p.peer_id])}</td><td class="rowmenu mut"><span class="icon" data-i="more"></span></td></tr>`).join("") : `<tr><td colspan="5" class="mut">${n.peers.length ? "Only infrastructure connected — waiting for network peers." : "No peers connected yet — connecting."}</td></tr>`;
    injectIcons($("#peertable"));
    // dynamic count + View more
    const pc = $("#peercount"); if (pc) pc.textContent = peers.length <= peerLimit ? `${peers.length} peer${peers.length === 1 ? "" : "s"}` : `${shown.length} of ${peers.length} peers`;
    const pm = $("#peermore"); if (pm) { pm.style.display = peers.length > peerLimit ? "" : "none"; pm.onclick = () => { peerLimit += 10; renderPeers(); }; }
    // re-apply the active Direct/Relay/Mixed chip so the 2.5s poll doesn't reset it to All
    const pp = $("#peerchips .chip.on")?.dataset.p || "all";
    $$("#peertable tbody tr").forEach((r) => r.style.display = (pp === "all" || r.dataset.p === pp) ? "" : "none");
    $("#actchips .chip .num") && ($("#actchips .chip .num").textContent = peers.length);
    $$("#peertable .rowmenu").forEach((cell) => cell.onclick = (e) => { e.stopPropagation(); menu(cell, [{ label: "Copy peer id", fn: () => toast("Copied") }, { sep: 1 }, { label: "Drop connection", fn: () => toast("Dropped") }]); });
    // DHT
    $$("#v-peers .acttab")[1].querySelector("tbody").innerHTML = (n.known_models || []).length ? (snap.network.known_models).map((m) => `<tr><td class="mono">/oh/model/${esc(m)}</td><td><span class="badge secondary">provider</span></td><td class="num">${(snap.network.known_providers || []).filter((p) => p.model_id === m).length || 1}</td><td class="num">—</td></tr>`).join("") : `<tr><td colspan="4" class="mut">No records yet.</td></tr>`;
    $$("#v-peers .acttab")[1].querySelector(".card.pad").innerHTML = `kad_routing_peers: <span class="num">${n.kad_routing_peers}</span> · server mode: ${n.kad_server_mode ? "yes" : "no"}`;
    // Swarm
    const sw = $$("#v-peers .acttab")[2].querySelectorAll(".kpi .val"); sw[0].textContent = n.listen_addrs.length; sw[1].textContent = n.relay_reservations.length; sw[2].textContent = n.counters.dcutr_successes; sw[3].textContent = n.autonat_private ? "private" : "public";
    // Logs
    renderLogs();
  }
  let logTab = "provider";
  function renderLogs() { const logs = (logTab === "provider" ? state?.provider?.logs : state?.gateway?.logs) || []; $("#logbody").innerHTML = logs.length ? logs.map(esc).join("<br>") : "—"; }
  function renderSettings() {
    const p = state; if (!p) return;
    // #9: don't clobber the device-name field while the user is editing it (the 2.5s poll
    // re-renders Settings; overwriting a focused field was why it "couldn't be changed").
    const id = $('.setpanel[data-p="identity"]'); const dn = id.querySelector('[contenteditable]');
    if (document.activeElement !== dn) dn.textContent = deviceName;
    id.querySelectorAll(".input")[1].childNodes[0].textContent = (p.provider.status.peer_id || p.gateway.status.peer_id || "—");
    const netp = $('.setpanel[data-p="network"]');
    const gwp = netp.querySelector('#gwport'); if (gwp && document.activeElement !== gwp) gwp.textContent = p.settings.gateway_port;
    const bsEl = netp.querySelector('#bootstraps'); if (bsEl && document.activeElement !== bsEl) bsEl.textContent = (p.settings.bootstraps || []).join("\n");
    const eng = $('.setpanel[data-p="engine"]');
    updateEngineEndpoint(); // #3: endpoint follows the selected engine (auto-detect ⇒ engines[0])
    $("#engineautostartsw").classList.toggle("on", !!p.settings.engine_autostart);
    $("#resumelaunchsw") && $("#resumelaunchsw").classList.toggle("on", p.settings.resume_on_launch !== false);
    $("#advsw").classList.toggle("on", app.hasAttribute("data-adv"));
    $("#verboselogsw") && $("#verboselogsw").classList.toggle("on", !!p.settings.verbose_logs);   // #4
  }

  // ── status bar + lifecycle ──
  function renderStatusbar() {
    const p = state?.provider?.status, g = state?.gateway?.status, peers = snap?.network?.peers?.length ?? 0;
    const anyOn = !!(p?.running || g?.running);
    let dot = "warn pulse", label = "Initializing…";
    if (state) { if (!anyOn) { dot = ""; label = "Ready — connecting…"; } else if (peers > 0) { dot = "ok pulse"; label = "Connected"; } else { dot = "warn pulse"; label = "Connecting to network…"; } }
    $("#netdot").className = "dot " + dot; $("#netlabel").textContent = label;
    $("#sbpeers").textContent = `${peers} peers`;
    $("#sbserved").textContent = fmtNum(totalServed()); $("#sbused").textContent = fmtNum(totalUsed());
    $("#apiendpoint").textContent = (state?.gateway_url || "http://127.0.0.1:16527/v1").replace(/^https?:\/\//, "") + (g?.running ? "" : " · off");
    // #1: the title-bar CTA reflects state. Off = a brand-filled call-to-action ("Share your
    // models"). Live = a calm green "active" chip with a pulsing dot ("Sharing N models") — it reads
    // as status, not an unclicked CTA. Reuses the app's shared `.btn.on` active treatment.
    const shb = $("#sharecta");
    if (shb) {
      const n = p?.announced ?? 0, live = !!p?.running;
      shb.classList.toggle("brand", !live);
      shb.classList.toggle("on", live);
      shb.title = live
        ? `Sharing ${n} model${n === 1 ? "" : "s"} to the network — click to manage`
        : "Share your machine's models to the network";
      shb.innerHTML = live
        ? `<span class="dot ok pulse"></span>${n > 0 ? `Sharing ${n} model${n === 1 ? "" : "s"}` : "Sharing…"}`
        : "Share your models";
    }
    $("#sidepeer").textContent = `${deviceName} · ${shortPeer(p?.peer_id || g?.peer_id)}`;
    renderModels();
    const connected = anyOn && (peers > 0 || netModels().length > 0);
    $("#homeconnecting").style.display = connected ? "none" : "inline-flex";
    $("#homeready").style.display = connected ? "" : "none";
  }
  $("#apiendpoint").onclick = async () => { if (!state?.gateway?.status?.running) { await ensureGateway(); toast("Local API started"); return; } navigator.clipboard?.writeText(state?.gateway_url || "http://127.0.0.1:16527/v1"); toast("Copied"); };

  // ── chips ──
  $("#provchips").onclick = (e) => { const c = e.target.closest(".chip[data-cat]"); if (!c) return; $$("#provchips .chip[data-cat]").forEach((x) => x.classList.toggle("on", x === c)); const cat = c.dataset.cat; $$("#provtable .prov").forEach((r) => r.style.display = (cat === "all" || r.dataset.cat === cat) ? "" : "none"); };
  $("#actchips").onclick = (e) => { const c = e.target.closest(".chip[data-act]"); if (!c) return; $$("#actchips .chip").forEach((x) => x.classList.toggle("on", x === c)); $$("#v-peers .acttab").forEach((t) => t.classList.toggle("on", t.dataset.act === c.dataset.act)); };
  $("#peerchips").onclick = (e) => { const c = e.target.closest(".chip"); if (!c) return; $$("#peerchips .chip").forEach((x) => x.classList.toggle("on", x === c)); const pp = c.dataset.p; $$("#peertable tbody tr").forEach((r) => r.style.display = (pp === "all" || r.dataset.p === pp) ? "" : "none"); };
  $("#logchips").onclick = (e) => { const c = e.target.closest(".chip"); if (!c) return; $$("#logchips .chip").forEach((x) => x.classList.toggle("on", x === c)); logTab = c.dataset.log === "gateway" ? "gateway" : "provider"; renderLogs(); };
  $("#search").oninput = () => { if (activeView === "providers") renderProviders(); else if (activeView === "peers") { const q = $("#search").value.toLowerCase(); $$("#peertable tbody tr").forEach((r) => r.style.display = r.textContent.toLowerCase().includes(q) ? "" : "none"); } };
  $("#cmdk").onclick = (e) => { e.stopPropagation(); menu($("#cmdk"), Object.keys(titles).map((v) => ({ label: "Go to " + titles[v], on: v === activeView, fn: () => go(v) }))); };
  $("#traymark").onclick = (e) => { e.stopPropagation(); menu($("#traymark"), [{ label: "Launch OpenHydra", fn: () => {} }, { sep: 1 }, { label: "Sharing", on: !!state?.provider?.status?.running, fn: toggleSharing }, { label: "Model · " + (netModels()[0] || "—"), fn: () => {} }, { sep: 1 }, { label: `▲ ${fmtNum(totalServed())} served`, fn: () => {} }, { label: `▼ ${fmtNum(totalUsed())} used`, fn: () => {} }, { sep: 1 }, { label: "Quit OpenHydra", fn: () => call("quit") }]); };
  $("#addmodel") && ($("#addmodel").onclick = (e) => { e.stopPropagation(); const opts = engines.flatMap((e) => e.models.map((m) => ({ label: `${m} · ${e.label}`, fn: () => toggleSharing() }))); menu($("#addmodel"), opts.length ? opts : [{ label: "No engine models — start an engine", fn: () => go("engines") }]); });

  // ── settings ──
  $("#setnav").onclick = (e) => { const s = e.target.closest(".s"); if (!s) return; $$("#setnav .s").forEach((x) => x.classList.toggle("on", x === s)); $$(".setpanel").forEach((pnl) => pnl.classList.toggle("on", pnl.dataset.p === s.dataset.p)); };
  $$("[data-sw]").forEach((sw) => { if (sw.closest("#servetable")) return; sw.onclick = () => sw.classList.toggle("on"); });
  $$(".save").forEach((b) => b.onclick = async () => {
    deviceName = ($('.setpanel[data-p="identity"] [contenteditable]').textContent || deviceName).trim(); store.set("oh_device", deviceName);
    saveSessions();   // #9: persist the (edited) device name to the durable file too
    const netp = $('.setpanel[data-p="network"]');
    let gwPort = parseInt((netp.querySelector('#gwport')?.textContent || "").trim(), 10);
    if (!Number.isInteger(gwPort) || gwPort < 1024 || gwPort > 65535) gwPort = state?.settings?.gateway_port || 16527;
    const bootstraps = (netp.querySelector('#bootstraps')?.textContent || "").split("\n").map((x) => x.trim()).filter(Boolean);
    const settings = { bootstraps, gateway_port: gwPort, engine_autostart: $("#engineautostartsw").classList.contains("on"), resume_on_launch: $("#resumelaunchsw") ? $("#resumelaunchsw").classList.contains("on") : true, search_url: state?.settings?.search_url || "", verbose_logs: $("#verboselogsw")?.classList.contains("on") || false, device_name: deviceName, shared_models: state?.settings?.shared_models || [] };
    try { await call("save_settings", { settings }); toast("Settings saved"); await refresh(); } catch (e) { toast(`Save failed: ${e}`); }
  });
  $('.setpanel[data-p="identity"] .cp')?.addEventListener("click", () => { navigator.clipboard?.writeText($('.setpanel[data-p="identity"] .input.mono').textContent.replace("Copy", "").trim()); toast("Peer ID copied"); });
  $("#advsw").onclick = () => { const on = !app.hasAttribute("data-adv"); app.toggleAttribute("data-adv", on); $("#advsw").classList.toggle("on", on); store.set("oh_adv", on); if (!on && activeView === "peers") go("providers"); };
  // #4: verbose-logs toggle (persist on Save) + Send-logs export
  $("#verboselogsw") && ($("#verboselogsw").onclick = () => $("#verboselogsw").classList.toggle("on"));
  $("#sendlogsbtn") && ($("#sendlogsbtn").onclick = async () => { try { const path = await call("export_logs"); toast(path ? `Logs saved: ${path}` : "No logs yet"); } catch (e) { toast(`Export failed: ${e}`); } });

  // ── connectors copy (wireframe .cp) ──
  $$(".cp").forEach((b) => b.onclick = (e) => { e.stopPropagation(); navigator.clipboard?.writeText(b.parentElement.textContent.replace(/Copy$/, "").trim()); toast("Copied to clipboard"); });

  // ── updater → silent download, apply on next restart ──
  // Policy: auto-check (hourly), then silently download + stage a signed update in the
  // BACKGROUND. We deliberately do NOT relaunch — on macOS the new bundle is swapped in place
  // without touching the running process or its sidecar agents, so an active provider keeps
  // serving and the update takes effect the next time the app is restarted ("apply on quit").
  // The card becomes an optional "restart now" accelerator, not a forced interruption.
  let updateReady = null;   // the checked Update handle
  let updateStaged = false; // true once bytes are downloaded + installed on disk (pending restart)
  $("#relaunch").style.display = "none";
  async function checkUpdates() {
    const u = window.__TAURI__?.updater; if (!u?.check) return;
    if (updateStaged) return; // already downloaded + staged this session; waiting on a restart
    try {
      const up = await u.check();
      if (!up) return;
      updateReady = up;
      // Silent staging is a clean in-place swap on macOS (.app) and Linux (AppImage): the running
      // process is untouched and the update applies on next restart. On Windows the update is an
      // NSIS installer that CLOSES the app to replace the binary — auto-installing would interrupt
      // an active serve — so there we fall back to the one-click card and let the user pick when.
      // Fail SAFE: only silent-stage when we've CONFIRMED a non-Windows OS. If sysInfo hasn't
      // loaded yet (cold-start can delay it past this check) OR the OS is Windows, fall back to the
      // manual card — so an unknown OS never triggers the NSIS installer that closes the app.
      const canSilentStage = !!sysInfo && (sysInfo.os || "").toLowerCase() !== "windows";
      if (canSilentStage) {
        try {
          await up.downloadAndInstall();   // download + stage; no relaunch → applies on next restart
          updateStaged = true;
          $("#relaunchver").textContent = "v" + up.version + " · restart to apply";
        } catch (e) {
          // Staging failed (e.g. offline mid-download) — fall back to a manual one-click apply.
          console.warn("update staging failed", e);
          $("#relaunchver").textContent = "v" + up.version;
        }
      } else {
        $("#relaunchver").textContent = "v" + up.version; // Windows: manual apply via the card
      }
      $("#relaunch").style.display = "flex";
    } catch (e) { console.warn("update check", e); }
  }
  setTimeout(checkUpdates, 3000);
  setInterval(checkUpdates, 60 * 60 * 1000); // #13: re-check hourly so a running instance stages a new release in the background
  let relaunchBusy = false;
  $("#relaunch").onclick = async () => {
    if (relaunchBusy) return; // guard against a double-click firing two downloadAndInstall calls
    relaunchBusy = true;
    try {
      // Staged already → just relaunch into it. Not staged (fallback) → download+install first.
      if (!updateStaged && updateReady) await updateReady.downloadAndInstall();
      await window.__TAURI__?.process?.relaunch?.();
    } catch (e) { relaunchBusy = false; toast(`Update failed: ${e}`); }
  };

  // ── first-run coachmark tour (spotlight; first launch + after updates only) ──
  const COACH = [
    { v: "home", a: "#homecard", t: "Chat with the network", d: "Ask anything — requests route to models served by peers. The first connection takes a few seconds; watch the status bar fill in." },
    { v: "providers", a: "#modeswitch", t: "Two sides of the app", d: "Home is where you use AI. Network is where you browse models, share your machine, and manage engines." },
    { v: "engines", a: '.nav[data-v="engines"]', t: "Engines & models", d: "OpenHydra wraps any engine already on your machine — whatever it can run, you can share." },
    { v: "share", a: "#sharecta", t: "Share when you're ready", d: "Click ‘Share your models’ to open the Share tab and pick what to serve. You're always connected; sharing is your choice." },
  ];
  const TOUR_KEY = "oh_tour_v2";
  let coachEl = null, coachRing = null, coachOv = [];
  function coachEnd() { coachEl?.remove(); coachEl = null; coachRing?.classList.remove("coachring"); coachRing = null; coachOv.forEach((d) => d.remove()); coachOv = []; store.set(TOUR_KEY, true); }
  function coachSpot(a, r) {
    const p = 5, W = innerWidth, H = innerHeight;
    if (!coachOv.length) { const s = document.createElement("div"); s.className = "covspot"; document.body.appendChild(s); coachOv.push(s); for (let i = 0; i < 4; i++) { const d = document.createElement("div"); d.className = "covstrip"; document.body.appendChild(d); coachOv.push(d); } }
    const x1 = Math.max(0, r.left - p), y1 = Math.max(0, r.top - p), x2 = Math.min(W, r.right + p), y2 = Math.min(H, r.bottom + p);
    const set = (d, l, t, w, h) => { d.style.left = l + "px"; d.style.top = t + "px"; d.style.width = Math.max(0, w) + "px"; d.style.height = Math.max(0, h) + "px"; };
    const rad = a ? (parseFloat(getComputedStyle(a).borderRadius) || 8) : 8;
    set(coachOv[0], x1, y1, x2 - x1, y2 - y1); coachOv[0].style.borderRadius = (rad + p) + "px";
    set(coachOv[1], 0, 0, W, y1); set(coachOv[2], 0, y2, W, H - y2); set(coachOv[3], 0, y1, x1, y2 - y1); set(coachOv[4], x2, y1, W - x2, y2 - y1);
  }
  function coachShow(i) {
    const s = COACH[i]; go(s.v);
    coachRing?.classList.remove("coachring"); const a = $(s.a); coachRing = a; a?.classList.add("coachring");
    if (!coachEl) { coachEl = document.createElement("div"); coachEl.className = "coach"; document.body.appendChild(coachEl); }
    coachEl.innerHTML = `<button class="iconbtn cx" id="coachx" title="Close"><span class="icon" data-i="x"></span></button><div style="font-weight:600;margin-bottom:4px;padding-right:24px">${s.t}</div><div class="mut" style="font-size:12px;line-height:1.55">${s.d}</div><div class="row" style="margin-top:11px;gap:8px">${i > 0 ? '<button class="btn outline sm" id="coachback">Back</button>' : ""}<span class="mut num" style="font-size:11px">${i + 1} / ${COACH.length}</span><div class="grow"></div><button class="btn primary sm" id="coachnext">${i < COACH.length - 1 ? "Next" : "Done"}</button></div>`;
    injectIcons(coachEl);
    const r = a ? a.getBoundingClientRect() : { left: innerWidth / 2 - 136, top: innerHeight / 2, right: innerWidth / 2 + 136, bottom: innerHeight / 2 };
    coachSpot(a, r);
    let left = Math.min(Math.max(10, r.left), innerWidth - 292), top = r.bottom + 14; if (top + 160 > innerHeight) top = Math.max(10, r.top - 166);
    coachEl.style.left = left + "px"; coachEl.style.top = top + "px";
    $("#coachx").onclick = coachEnd; const bk = $("#coachback"); if (bk) bk.onclick = () => coachShow(i - 1); $("#coachnext").onclick = () => (i < COACH.length - 1 ? coachShow(i + 1) : coachEnd());
  }
  if (!store.get(TOUR_KEY, false)) setTimeout(() => coachShow(0), 900);
  $("#obreplay").onclick = () => coachShow(0);

  // ── polling ──
  async function refresh() {
    try { state = await call("get_state"); } catch {}
    // Informed opt-out: while this launch's auto-resume is flagged, keep a non-blocking notice up.
    if (state?.resumed_on_launch) ensureResumeNotice();
    renderStatusbar();
    if (["share", "settings", "connectors", "engines", "activity", "ledger"].includes(activeView)) renderView();
    if (activeView === "peers") renderLogs();
  }

  // Non-blocking "Resuming your shared models…" banner (informed opt-out). RE-ASSERTED each refresh so a
  // first-run / after-update coachmark DOM rebuild can't kill it (that's exactly the update case where
  // this matters). Clears on "Don't resume", ×, or after a short show-window.
  let resumeNoticeDismissed = false, resumeNoticeDeadline = 0;
  function ensureResumeNotice() {
    if (resumeNoticeDismissed) return;
    if (!resumeNoticeDeadline) resumeNoticeDeadline = Date.now() + 18000;
    if (Date.now() > resumeNoticeDeadline) { resumeNoticeDismissed = true; document.getElementById("resumenotice")?.remove(); return; }
    if (document.getElementById("resumenotice")) return;
    const el = document.createElement("div");
    el.id = "resumenotice"; el.className = "resumenotice";
    el.innerHTML = `<span class="rn-dot">●</span><span>Resuming your shared models…</span>`
      + `<button class="btn ghost sm rn-stop">Don't resume</button><button class="rn-x" title="Dismiss">×</button>`;
    document.body.appendChild(el);
    const dismiss = () => { resumeNoticeDismissed = true; el.remove(); };
    el.querySelector(".rn-stop").onclick = async () => { dismiss(); try { await setSharing(false); toast("Sharing stopped"); } catch (e) { toast("Couldn't stop: " + e); } };
    el.querySelector(".rn-x").onclick = dismiss;
  }
  async function refreshEngines() { try { engines = await call("detect_engines_now"); } catch { engines = []; } try { installedEngines = await call("installed_engines"); } catch { installedEngines = []; } renderStatusbar(); if (["share", "engines", "providers", "settings"].includes(activeView)) renderView(); }
  // Start an installed-but-idle engine's server, then refresh so the card flips to "running".
  async function runEngine(label, name) {
    toast(`Starting ${name}…`);
    try { await call("run_engine", { engine: label }); toast(`${name} started`); }
    catch (e) { toast(String(e)); }
    await refreshEngines();
  }
  async function refreshStatus() { try { snap = await call("status_snapshot"); } catch { snap = null; } noteSeen(); accumulateStats(); renderStatusbar(); if (["peers", "providers", "activity", "ledger", "share"].includes(activeView)) renderView(); }
  $$(".enginst, #refreshEngines").forEach(() => {});

  // ── boot ──
  $(".header").style.display = "none"; // Home landing has no header
  $("#homelogo").src = "/logo-mark.png";
  (async () => {
    // #1: hydrate chat sessions from the durable backend file (localStorage is only a cache).
    // `loadOk` gates the clean-shape rewrite below: on a transient load failure we must NOT write
    // the (possibly stale) localStorage cache back over the durable file — that would delete chats
    // the file holds but the cache lost. `migrated` = the on-disk shape actually needed fixing.
    let loadOk = false, migrated = false;
    try {
      const blob = await call("load_sessions");
      loadOk = true;
      if (blob) {
        const d = JSON.parse(blob);
        if (d.sessions) { const co = coerceSessions(d.sessions); migrated = co !== d.sessions; sessions = co; }
        if (Array.isArray(d.order)) sessionOrder = d.order;
        if (d.device) deviceName = d.device;
        if (typeof d.used === "number") { usedTokens = d.used; store.set("oh_used", usedTokens); }
      }
    } catch { loadOk = false; }
    // Repair order/sessions drift: drop order ids with no session, and append any session missing
    // from order — otherwise legacy orphaned ids render nothing and recovered chats stay hidden.
    const orderBefore = sessionOrder.join("");
    sessionOrder = sessionOrder.filter((id) => sessions[id]);
    for (const id in sessions) if (!sessionOrder.includes(id)) sessionOrder.push(id);
    const orderRepaired = sessionOrder.join("") !== orderBefore;
    // Rewrite the durable file ONLY when the load succeeded AND coercion/repair actually changed the
    // shape — never overwrite it with stale cache after a failed/empty load.
    if (loadOk && (migrated || orderRepaired) && Object.keys(sessions).length) saveSessions();
    await loadStats();   // #7/#10: hydrate lifetime model stats + timeline buckets from disk
    // #9: default device name from the OS if the user hasn't set/restored one.
    if (!deviceName) { try { deviceName = await call("device_hostname"); } catch {} if (!deviceName) deviceName = /Mac/.test(navigator.platform) ? "This Mac" : "This machine"; store.set("oh_device", deviceName); }
    // Show the running OpenHydra version (authoritative bundle version, not a guess) in the
    // statusbar + Settings › About.
    try { const v = await call("app_version"); if (v) { const sv = $("#sbver"); if (sv) sv.textContent = "v" + v; const av = $("#aboutver"); if (av) av.textContent = "OpenHydra v" + v; } } catch {}
    // First-run "Add to Applications" — only when running from a Linux AppImage that hasn't been
    // integrated yet (an AppImage is double-clickable but doesn't add itself to the app menu).
    try {
      const st = await call("appimage_status");
      if (st?.is_appimage && !st.integrated && !store.get("oh_appimage_dismissed", false)) {
        const card = $("#appimageintegrate");
        card.style.display = "block";
        $("#appimageadd").onclick = async () => {
          try { await call("integrate_appimage"); toast("Added to Applications ✓"); }
          catch (e) { toast(`Couldn't add: ${e}`); }
          card.style.display = "none";
        };
        $("#appimagedismiss").onclick = () => { store.set("oh_appimage_dismissed", true); card.style.display = "none"; };
      }
    } catch {}
    renderRecents(); renderStatusbar();
    await refresh(); await refreshEngines(); await refreshSystem();
    await ensureGateway();  // eager: warm discovery so the first chat isn't a cold 504
    await refreshStatus();
  })();
  setInterval(refresh, 2500);
  setInterval(refreshEngines, 10000);
  setInterval(refreshStatus, 2500);
})();
