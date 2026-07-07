// OpenHydra Desktop UI — BiglyBT-style shell. Talks to the Rust backend via Tauri IPC;
// in a plain browser (layout preview) it renders demo state instead.

const tauri = window.__TAURI__?.core;
async function call(cmd, args) {
  if (tauri) return tauri.invoke(cmd, args);
  return mock(cmd, args);
}

// ── demo state for browser preview only ──
function mock(cmd, args) {
  if (cmd === "get_state")
    return {
      provider: {
        status: { running: true, pid: 4242, peer_id: "12D3KooWEVGKuH5uEqhR7PfkV4k8RrZbwivLedrY6cGQDKDEuXRH",
          engines: "auto-detected 3 engine(s): ollama(2 models) @ http://127.0.0.1:11434, lm-studio(3 models) @ http://127.0.0.1:1234, exo(1 model) @ http://127.0.0.1:52415",
          announced: 6, relays: 3, exited: null },
        logs: ["openhydra-agent: node up — libp2p=12D3KooWEVGK…", "openhydra-agent: auto-detected 3 engine(s): …", "openhydra-agent: announced 6 model(s) from auto", "INFO relay reservation accepted ×3"],
      },
      gateway: { status: { running: true, pid: 4243, peer_id: "12D3KooWKzuVb8tc…", engines: null, announced: null, relays: 2, exited: null }, logs: ["openhydra-agent: gateway listening on http://127.0.0.1:8080"] },
      settings: { bootstraps: ["/ip4/45.79.190.172/tcp/4001/p2p/12D3KooWEL…"], gateway_port: 8080, engine_autostart: true },
      agent_found: true, gateway_url: "http://127.0.0.1:8080/v1",
    };
  if (cmd === "detect_engines_now")
    return [
      { label: "ollama", url: "http://127.0.0.1:11434", models: ["tinyllama:latest", "llama3.2:1b"] },
      { label: "lm-studio", url: "http://127.0.0.1:1234", models: ["qwen3-0.6b-mlx", "qwen3.5-2b-mlx", "text-embedding-nomic-embed-text-v1.5"] },
      { label: "exo", url: "http://127.0.0.1:52415", models: ["mlx-community/Llama-3.2-1B-Instruct-4bit"] },
    ];
  if (cmd === "gateway_health") return true;
  if (cmd === "chat_completion") {
    const isCode = (args?.messages || []).some((m) => m.role === "system");
    return new Promise((r) => setTimeout(() => r({
      choices: [{ message: { content: isCode
        ? "Here is a minimal example:\n```rust\n// parse a multiaddr string\nfn parse(addr: &str) -> Result<Multiaddr, Error> {\n    let ma: Multiaddr = addr.parse()?;\n    Ok(ma)\n}\n```\nCall `parse(\"/ip4/1.2.3.4/tcp/4001\")` to try it."
        : "Hello! I'm being served by a provider on the OpenHydra swarm — this reply routed gateway → discover → provider → engine." } }],
      usage: { completion_tokens: 42 },
      openhydra: { engine: { native_tps: 91.4 }, hops_ms: { network_rtt: 12, discover: 2 } },
      model: args?.model || "llama3.2:1b",
    }), 700));
  }
  return null;
}

const $ = (id) => document.getElementById(id);
let state = null;
let engines = [];
let activeView = "dashboard";
let activeLogTab = "provider";
let logsPinned = true;
let lastTokens = 0, lastTps = "—";

// ── view switching ──
document.querySelectorAll(".nav-item").forEach((el) => {
  el.addEventListener("click", () => {
    activeView = el.dataset.view;
    document.querySelectorAll(".nav-item").forEach((n) => n.classList.toggle("nav-active", n === el));
    document.querySelectorAll(".view").forEach((v) => v.classList.toggle("hidden", v.id !== `view-${activeView}`));
    render();
  });
});

// ── rendering ──
function shortPeer(p) { return p && p.length > 20 ? `${p.slice(0, 10)}…${p.slice(-6)}` : p || "—"; }
function esc(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }

function roleBtn(btn, running, startLabel, stopLabel) {
  btn.textContent = running ? `■ ${stopLabel}` : `▶ ${startLabel}`;
  btn.classList.toggle("tbtn-start", !running);
  btn.classList.toggle("tbtn-stop", running);
}

function render() {
  if (!state) return;
  const p = state.provider.status, g = state.gateway.status;

  roleBtn($("provBtn"), p.running, "Share", "Sharing");
  roleBtn($("gwBtn"), g.running, "Gateway", "Gateway");
  const anyOn = p.running || g.running;
  $("netChip").textContent = anyOn ? "online" : "idle";
  $("netChip").className = `chip ${anyOn ? "chip-on" : "chip-idle"}`;

  // dashboard strip
  $("dashProv").textContent = p.running ? "sharing" : p.exited || "stopped";
  $("dashGw").textContent = g.running ? "serving" : g.exited || "stopped";
  $("dashAnnounced").textContent = p.announced ?? "—";
  $("dashRelays").textContent = p.running ? String(p.relays) : "—";
  $("dashUrl").textContent = state.gateway_url;
  $("bootstrapBanner").classList.toggle("hidden", state.settings.bootstraps.length > 0);
  $("agentBanner").classList.toggle("hidden", state.agent_found);

  // sidebar + status bar
  const peer = p.peer_id || g.peer_id;
  $("sidePeer").textContent = shortPeer(peer);
  $("stPeer").textContent = peer || "—";
  $("stProvDot").className = `dot ${p.running ? "dot-on" : p.exited ? "dot-err" : ""}`;
  $("stGwDot").className = `dot ${g.running ? "dot-on" : g.exited ? "dot-err" : ""}`;
  $("stModels").textContent = p.announced ?? engines.reduce((n, e) => n + e.models.length, 0);
  $("stRelays").textContent = p.relays || 0;
  $("stTokens").textContent = lastTokens;
  $("stTps").textContent = lastTps;

  // chat gate hint
  $("chatGateHint").classList.toggle("hidden", g.running);

  // logs
  if (activeView === "logs") {
    const logs = activeLogTab === "provider" ? state.provider.logs : state.gateway.logs;
    const box = $("logBox");
    const text = logs.length ? logs.join("\n") : "—";
    if (box.textContent !== text) {
      box.textContent = text;
      if (logsPinned) box.scrollTop = box.scrollHeight;
    }
  }
}

function renderModels() {
  const rows = [];
  const shared = state?.provider.status.running;
  for (const e of engines) {
    for (const m of e.models) {
      rows.push(`<tr><td class="mono">${esc(m)}</td><td>${esc(e.label)}</td>
        <td class="mono">${esc(e.url)}</td>
        <td><span class="pill ${shared ? "pill-shared" : "pill-local"}">${shared ? "shared" : "local"}</span></td></tr>`);
    }
  }
  $("modelRows").innerHTML = rows.length
    ? rows.join("")
    : `<tr><td colspan="4" class="dim">No engines answering — start Ollama, LM Studio, vLLM, llama.cpp, or Exo, then rescan.</td></tr>`;
  $("navModels").textContent = rows.length;
  // model picker suggestions
  const models = engines.flatMap((e) => e.models);
  $("modelOptions").innerHTML = models.map((m) => `<option value="${esc(m)}">`).join("");
  if (!$("chatModel").value && models[0]) $("chatModel").value = models[0];
  if (!$("codeModel").value && models[0]) $("codeModel").value = models[0];
}

// ── chat ──
const chatHistory = [];
function bubble(cls, text) {
  const d = document.createElement("div");
  d.className = `msg ${cls}`;
  d.textContent = text;
  $("chatEmpty")?.remove();
  $("chatScroll").appendChild(d);
  $("chatScroll").scrollTop = $("chatScroll").scrollHeight;
  return d;
}

function metaLine(resp) {
  const oh = resp.openhydra || {};
  const parts = [];
  if (resp.usage?.completion_tokens != null) parts.push(`▼ ${resp.usage.completion_tokens} tok`);
  if (oh.engine?.native_tps) parts.push(`${oh.engine.native_tps} tok/s engine`);
  if (oh.hops_ms?.network_rtt != null) parts.push(`rtt ${oh.hops_ms.network_rtt} ms`);
  if (oh.hops_ms?.discover != null) parts.push(`discover ${oh.hops_ms.discover} ms`);
  if (resp.model) parts.push(esc(resp.model));
  return parts.join(" · ");
}

async function sendChat() {
  const text = $("chatInput").value.trim();
  const model = $("chatModel").value.trim();
  if (!text || !model) return;
  $("chatInput").value = "";
  bubble("msg-user", text);
  chatHistory.push({ role: "user", content: text });
  const wait = document.createElement("div");
  wait.className = "msg-wait"; wait.textContent = "thinking";
  $("chatScroll").appendChild(wait);
  try {
    const resp = await call("chat_completion", { model, messages: chatHistory, maxTokens: 1024 });
    wait.remove();
    const content = resp?.choices?.[0]?.message?.content ?? "(empty reply)";
    chatHistory.push({ role: "assistant", content });
    const b = bubble("msg-bot", content);
    const meta = metaLine(resp);
    if (meta) {
      const m = document.createElement("div");
      m.className = "msg-meta"; m.textContent = meta;
      b.appendChild(m);
    }
    if (resp.usage?.completion_tokens) lastTokens += resp.usage.completion_tokens;
    if (resp.openhydra?.engine?.native_tps) lastTps = resp.openhydra.engine.native_tps;
    render();
  } catch (e) {
    wait.remove();
    bubble("msg-err", `${e}`);
    chatHistory.pop(); // don't poison the next turn with an unanswered message
  }
}

// ── code view ──
const KEYWORDS = new Set(("fn let mut pub use impl struct enum trait match if else for while loop return async await move " +
  "def class import from as with try except lambda yield pass raise global nonlocal " +
  "function const var new typeof instanceof extends super this export default " +
  "package func go defer chan interface map range type switch case break continue static void int float double char bool").split(" "));

function highlight(code) {
  let out = "", i = 0;
  const push = (cls, s) => { out += cls ? `<span class="${cls}">${esc(s)}</span>` : esc(s); };
  while (i < code.length) {
    const rest = code.slice(i);
    let m;
    if ((m = rest.match(/^(\/\/|#(?!\[)|--)[^\n]*/)))       { push("tk-com", m[0]); }
    else if ((m = rest.match(/^\/\*[\s\S]*?\*\//)))          { push("tk-com", m[0]); }
    else if ((m = rest.match(/^("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|`(?:[^`\\]|\\.)*`)/))) { push("tk-str", m[0]); }
    else if ((m = rest.match(/^\b\d[\d_]*(\.\d+)?\b/)))      { push("tk-num", m[0]); }
    else if ((m = rest.match(/^[A-Za-z_][A-Za-z0-9_]*/))) {
      if (KEYWORDS.has(m[0])) push("tk-kw", m[0]);
      else if (code[i + m[0].length] === "(") push("tk-fn", m[0]);
      else push(null, m[0]);
    } else { push(null, code[i]); i += 1; continue; }
    i += m[0].length;
  }
  return out;
}

/// Split a model reply into prose + fenced code blocks.
function parseFences(text) {
  const parts = [];
  const re = /```([\w+-]*)\n([\s\S]*?)```/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) parts.push({ prose: text.slice(last, m.index).trim() });
    parts.push({ lang: m[1] || "code", code: m[2] });
    last = re.lastIndex;
  }
  if (last < text.length) parts.push({ prose: text.slice(last).trim() });
  return parts.filter((p) => p.code != null || p.prose);
}

async function sendCode() {
  const text = $("codeInput").value.trim();
  const model = $("codeModel").value.trim();
  if (!text || !model) return;
  $("codeInput").value = "";
  $("codeEmpty")?.remove();
  const turn = document.createElement("div");
  turn.className = "code-turn";
  turn.innerHTML = `<div class="code-prompt"><b>›</b> ${esc(text)}</div>`;
  $("codeScroll").appendChild(turn);
  const wait = document.createElement("div");
  wait.className = "msg-wait"; wait.textContent = "generating";
  turn.appendChild(wait);
  $("codeScroll").scrollTop = $("codeScroll").scrollHeight;
  try {
    const resp = await call("chat_completion", {
      model,
      messages: [
        { role: "system", content: "You are a concise coding assistant. Always put code in fenced ``` blocks with a language tag; keep prose minimal." },
        { role: "user", content: text },
      ],
      maxTokens: 2048,
    });
    wait.remove();
    const content = resp?.choices?.[0]?.message?.content ?? "";
    for (const part of parseFences(content)) {
      if (part.code != null) {
        const card = document.createElement("div");
        card.className = "code-card";
        card.innerHTML = `<div class="code-card-head"><span>${esc(part.lang)}</span>
          <button class="code-copy">⎘ copy</button></div><pre>${highlight(part.code)}</pre>`;
        card.querySelector(".code-copy").addEventListener("click", (ev) => {
          navigator.clipboard?.writeText(part.code);
          ev.target.textContent = "✓ copied";
          setTimeout(() => (ev.target.textContent = "⎘ copy"), 1400);
        });
        turn.appendChild(card);
      } else {
        const pr = document.createElement("div");
        pr.className = "code-prose"; pr.textContent = part.prose;
        turn.appendChild(pr);
      }
    }
    const meta = metaLine(resp);
    if (meta) {
      const m = document.createElement("div");
      m.className = "msg-meta"; m.textContent = meta;
      turn.appendChild(m);
    }
    if (resp.usage?.completion_tokens) lastTokens += resp.usage.completion_tokens;
    if (resp.openhydra?.engine?.native_tps) lastTps = resp.openhydra.engine.native_tps;
    render();
  } catch (e) {
    wait.remove();
    const err = document.createElement("div");
    err.className = "msg msg-err"; err.textContent = `${e}`;
    turn.appendChild(err);
  }
  $("codeScroll").scrollTop = $("codeScroll").scrollHeight;
}

// ── actions ──
async function toggleRole(which) {
  const running = which === "provider" ? state?.provider.status.running : state?.gateway.status.running;
  const cmd = running ? (which === "provider" ? "stop_provider" : "stop_gateway")
                      : (which === "provider" ? "start_provider" : "start_gateway");
  try { await call(cmd); } catch (e) { alert(`${cmd} failed: ${e}`); }
  await refresh();
}

function openSettings() {
  $("setBootstraps").value = state.settings.bootstraps.join("\n");
  $("setPort").value = state.settings.gateway_port;
  $("setAutostart").checked = state.settings.engine_autostart;
  $("settingsModal").classList.remove("hidden");
}
function closeSettings() { $("settingsModal").classList.add("hidden"); }
window.openSettings = openSettings;
window.closeSettings = closeSettings;

async function saveSettings() {
  const settings = {
    bootstraps: $("setBootstraps").value.split("\n").map((s) => s.trim()).filter(Boolean),
    gateway_port: parseInt($("setPort").value, 10) || 8080,
    engine_autostart: $("setAutostart").checked,
  };
  try { await call("save_settings", { settings }); closeSettings(); await refresh(); }
  catch (e) { alert(`save failed: ${e}`); }
}

// ── polling ──
async function refresh() {
  try { state = await call("get_state"); render(); } catch (_) {}
}
async function refreshEngines() {
  try { engines = await call("detect_engines_now"); renderModels(); } catch (_) {}
}

// ── wiring ──
$("provBtn").addEventListener("click", () => toggleRole("provider"));
$("gwBtn").addEventListener("click", () => toggleRole("gateway"));
$("chatStartGw").addEventListener("click", () => toggleRole("gateway"));
$("settingsBtn").addEventListener("click", openSettings);
$("saveSettings").addEventListener("click", saveSettings);
$("refreshEngines").addEventListener("click", refreshEngines);
$("chatSend").addEventListener("click", sendChat);
$("codeSend").addEventListener("click", sendCode);
$("chatClear").addEventListener("click", () => { chatHistory.length = 0; $("chatScroll").innerHTML = ""; });
$("codeClear").addEventListener("click", () => { $("codeScroll").innerHTML = ""; });
for (const [input, send] of [["chatInput", sendChat], ["codeInput", sendCode]]) {
  $(input).addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  });
}
$("tabProv").addEventListener("click", () => { activeLogTab = "provider"; $("tabProv").classList.add("tab-active"); $("tabGw").classList.remove("tab-active"); render(); });
$("tabGw").addEventListener("click", () => { activeLogTab = "gateway"; $("tabGw").classList.add("tab-active"); $("tabProv").classList.remove("tab-active"); render(); });
$("logBox").addEventListener("scroll", () => {
  const b = $("logBox");
  logsPinned = b.scrollTop + b.clientHeight >= b.scrollHeight - 8;
});

refresh();
refreshEngines();
setInterval(refresh, 2000);
setInterval(refreshEngines, 10000);
