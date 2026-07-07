// OpenHydra Desktop UI. Talks to the Rust backend via Tauri IPC; when opened in a plain
// browser (no __TAURI__, e.g. layout preview), it renders harmless demo state instead.

const tauri = window.__TAURI__?.core;

async function call(cmd, args) {
  if (tauri) return tauri.invoke(cmd, args);
  return mock(cmd);
}

// ── demo state for browser preview only ──
function mock(cmd) {
  if (cmd === "get_state")
    return {
      provider: {
        status: {
          running: true, pid: 4242,
          peer_id: "12D3KooWEVGKuH5uEqhR7PfkV4k8RrZbwivLedrY6cGQDKDEuXRH",
          engines: "auto-detected 3 engine(s): ollama(2 models) @ http://127.0.0.1:11434, lm-studio(3 models) @ http://127.0.0.1:1234, exo(1 model) @ http://127.0.0.1:52415",
          announced: 6, relays: 3, exited: null,
        },
        logs: ["openhydra-agent: node up — libp2p=12D3KooW…", "openhydra-agent: announced 6 model(s) from auto"],
      },
      gateway: {
        status: { running: false, pid: null, peer_id: null, engines: null, announced: null, relays: 0, exited: null },
        logs: [],
      },
      settings: { bootstraps: [], gateway_port: 8080, engine_autostart: true },
      agent_found: true,
      gateway_url: "http://127.0.0.1:8080/v1",
    };
  if (cmd === "detect_engines_now")
    return [
      { label: "ollama", url: "http://127.0.0.1:11434", models: ["tinyllama:latest", "llama3.2:1b"] },
      { label: "lm-studio", url: "http://127.0.0.1:1234", models: ["qwen3-0.6b-mlx", "qwen3.5-2b-mlx"] },
      { label: "exo", url: "http://127.0.0.1:52415", models: ["mlx-community/Llama-3.2-1B-Instruct-4bit"] },
    ];
  if (cmd === "gateway_health") return false;
  return null;
}

// ── elements ──
const $ = (id) => document.getElementById(id);
let state = null;
let activeTab = "provider";
let logsPinned = true; // autoscroll unless the user scrolled up

// ── rendering ──
function chip(el, text, cls) {
  el.textContent = text;
  el.className = `chip ${cls}`;
}

function shortPeer(p) {
  return p && p.length > 20 ? `${p.slice(0, 10)}…${p.slice(-6)}` : p || "—";
}

function renderRole(prefix, role, startBtn, startLabel, stopLabel) {
  const s = role.status;
  const btn = $(startBtn);
  if (s.running) {
    chip($(`${prefix}Status`), "running", "chip-on");
    btn.textContent = stopLabel;
    btn.classList.remove("btn-primary");
    btn.classList.add("btn-danger");
  } else {
    chip($(`${prefix}Status`), s.exited ? s.exited : "stopped", s.exited ? "chip-err" : "chip-idle");
    btn.textContent = startLabel;
    btn.classList.add("btn-primary");
    btn.classList.remove("btn-danger");
  }
  $(`${prefix}Peer`).textContent = shortPeer(s.peer_id);
  $(`${prefix}Peer`).title = s.peer_id || "";
}

function render() {
  if (!state) return;
  const p = state.provider, g = state.gateway;

  renderRole("prov", p, "provBtn", "Start sharing", "Stop sharing");
  const ps = p.status;
  $("provEngines").textContent = ps.engines
    ? ps.engines.replace(/^auto-detected\s+/, "").replace(/ @ http[^,]+/g, "")
    : "—";
  $("provAnnounced").textContent = ps.announced ?? "—";
  $("provRelays").textContent = ps.running ? String(ps.relays) : "—";

  renderRole("gw", g, "gwBtn", "Start gateway", "Stop gateway");
  $("gwUrl").textContent = state.gateway_url;
  $("gwSnippet").textContent =
    `export OPENAI_BASE_URL=${state.gateway_url}\n` +
    `curl ${state.gateway_url}/chat/completions -d '{"model":"<model>","messages":[…]}'`;

  // banners + overall chip
  $("bootstrapBanner").classList.toggle("hidden", state.settings.bootstraps.length > 0);
  $("agentBanner").classList.toggle("hidden", state.agent_found);
  const anyOn = ps.running || g.status.running;
  chip($("netChip"), anyOn ? "online" : "idle", anyOn ? "chip-on" : "chip-idle");

  // logs
  const logs = activeTab === "provider" ? p.logs : g.logs;
  const box = $("logBox");
  const text = logs.length ? logs.join("\n") : "—";
  if (box.textContent !== text) {
    box.textContent = text;
    if (logsPinned) box.scrollTop = box.scrollHeight;
  }
}

function renderEngines(engines) {
  const el = $("engineList");
  if (!engines.length) {
    el.innerHTML = `<div class="muted pad">No engines answering on the standard ports — start Ollama, LM Studio, vLLM, llama.cpp, or Exo.</div>`;
    return;
  }
  el.innerHTML = engines
    .map(
      (e) => `
    <div class="engine">
      <div class="engine-head">
        <span class="engine-name">${e.label}</span>
        <span class="engine-url">${e.url}</span>
      </div>
      <div class="models">${e.models.map((m) => `<span class="model">${m}</span>`).join("") || '<span class="muted">no models loaded</span>'}</div>
    </div>`
    )
    .join("");
}

// ── actions ──
async function toggleRole(which) {
  const running = which === "provider" ? state?.provider.status.running : state?.gateway.status.running;
  const cmd = running
    ? which === "provider" ? "stop_provider" : "stop_gateway"
    : which === "provider" ? "start_provider" : "start_gateway";
  try {
    await call(cmd);
  } catch (e) {
    alert(`${cmd} failed: ${e}`);
  }
  await refresh();
}

function openSettings() {
  $("setBootstraps").value = state.settings.bootstraps.join("\n");
  $("setPort").value = state.settings.gateway_port;
  $("setAutostart").checked = state.settings.engine_autostart;
  $("settingsModal").classList.remove("hidden");
}
function closeSettings() {
  $("settingsModal").classList.add("hidden");
}
window.openSettings = openSettings;
window.closeSettings = closeSettings;

async function saveSettings() {
  const settings = {
    bootstraps: $("setBootstraps").value.split("\n").map((s) => s.trim()).filter(Boolean),
    gateway_port: parseInt($("setPort").value, 10) || 8080,
    engine_autostart: $("setAutostart").checked,
  };
  try {
    await call("save_settings", { settings });
    closeSettings();
    await refresh();
  } catch (e) {
    alert(`save failed: ${e}`);
  }
}

// ── polling ──
async function refresh() {
  try {
    state = await call("get_state");
    render();
  } catch (_) { /* backend briefly busy — next tick catches up */ }
}

async function refreshEngines() {
  try {
    renderEngines(await call("detect_engines_now"));
  } catch (_) {}
}

// ── wiring ──
$("provBtn").addEventListener("click", () => toggleRole("provider"));
$("gwBtn").addEventListener("click", () => toggleRole("gateway"));
$("settingsBtn").addEventListener("click", openSettings);
$("saveSettings").addEventListener("click", saveSettings);
$("refreshEngines").addEventListener("click", refreshEngines);
$("copyUrl").addEventListener("click", () => navigator.clipboard?.writeText(state?.gateway_url || ""));
$("tabProv").addEventListener("click", () => {
  activeTab = "provider";
  $("tabProv").classList.add("tab-active");
  $("tabGw").classList.remove("tab-active");
  render();
});
$("tabGw").addEventListener("click", () => {
  activeTab = "gateway";
  $("tabGw").classList.add("tab-active");
  $("tabProv").classList.remove("tab-active");
  render();
});
$("logBox").addEventListener("scroll", () => {
  const b = $("logBox");
  logsPinned = b.scrollTop + b.clientHeight >= b.scrollHeight - 8;
});

refresh();
refreshEngines();
setInterval(refresh, 2000);
setInterval(refreshEngines, 10000);
