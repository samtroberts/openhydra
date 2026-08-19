// Tauri IPC bridge + browser-preview mock. call() routes to the real backend when running inside
// Tauri (window.__TAURI__), else to the in-page mock that renders demo data in a plain browser.
// The mock install bus (mockInstallCbs / mockEmitInstall) is shared with the installer UI.
const tauri = window.__TAURI__?.core;
export async function call(cmd, args) { if (tauri) return tauri.invoke(cmd, args); return mock(cmd, args); }

export const mockInstallCbs = [];
export function mockEmitInstall(ev) { mockInstallCbs.slice().forEach((cb) => cb(ev)); }

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
