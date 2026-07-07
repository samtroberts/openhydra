// OpenHydra Desktop UI — BiglyBT-style shell + AI-workspace tools (sessions, presets,
// memory, attachments, web-augmented chat, blind council). Talks to the Rust backend via
// Tauri IPC; in a plain browser (layout preview) it renders demo state instead.

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
          engines: "auto-detected 3 engine(s)…", announced: 6, relays: 3, exited: null },
        logs: ["openhydra-agent: node up — libp2p=12D3KooWEVGK…", "openhydra-agent: announced 6 model(s) from auto"],
      },
      gateway: { status: { running: true, pid: 4243, peer_id: "12D3KooWKzuVb8tc…", engines: null, announced: null, relays: 2, exited: null }, logs: [] },
      settings: { bootstraps: ["/ip4/45.79.190.172/tcp/4001/p2p/12D3KooWEL…"], gateway_port: 8080, engine_autostart: true, search_url: "http://127.0.0.1:8888" },
      agent_found: true, gateway_url: "http://127.0.0.1:8080/v1",
    };
  if (cmd === "detect_engines_now")
    return [
      { label: "ollama", url: "http://127.0.0.1:11434", models: ["tinyllama:latest", "llama3.2:1b"] },
      { label: "lm-studio", url: "http://127.0.0.1:1234", models: ["qwen3-0.6b-mlx", "qwen3.5-2b-mlx"] },
      { label: "exo", url: "http://127.0.0.1:52415", models: ["mlx-community/Llama-3.2-1B-Instruct-4bit"] },
    ];
  if (cmd === "gateway_health") return true;
  if (cmd === "status_snapshot")
    return {
      role: "provider", agent_version: "0.1.0",
      libp2p_peer_id: "12D3KooWEVGKuH5uEqhR7PfkV4k8RrZbwivLedrY6cGQDKDEuXRH",
      network: {
        nat: { nat_type: "unknown", is_public: false }, autonat_private: false, ipv6_capable: true,
        kad_server_mode: false, kad_routing_peers: 8, network_generation: 0,
        listen_addrs: ["/ip4/0.0.0.0/tcp/4111", "/ip4/0.0.0.0/udp/4111/quic-v1"],
        external_addrs: ["/ip4/49.36.1.2/udp/4111/quic-v1"],
        relay_reservations: ["/ip4/45.79.190.172/tcp/4001/p2p/12D3KooWEL…/p2p-circuit", "/ip4/172.105.69.49/tcp/4001/p2p/12D3KooWEz…/p2p-circuit"],
        peers: [
          { peer_id: "12D3KooWM2qsVg5WbR6ukzDL7dvThvr1X5JsVU6h6Sfz7hnn2XN7", quic_direct_v4: 0, quic_direct_v6: 0, tcp_direct: 1, tcp_relay: 0, failure_streak: 0, path: "direct" },
          { peer_id: "12D3KooWHNQ9nMedAbcd1234efGh5678ijKl90mnOpQrStUvWxYz", quic_direct_v4: 1, quic_direct_v6: 0, tcp_direct: 0, tcp_relay: 0, failure_streak: 0, path: "direct" },
          { peer_id: "12D3KooWEzegXr4qcj37EWF2aQo9vp121MGrCaCwYcJF2oTkW3WT", quic_direct_v4: 0, quic_direct_v6: 0, tcp_direct: 0, tcp_relay: 1, failure_streak: 0, path: "relay" },
          { peer_id: "12D3KooWPgqZBgLZ1f94AQ7sbeyEz5UJ4jiT4d3zuQp2t61VLPZo", quic_direct_v4: 0, quic_direct_v6: 0, tcp_direct: 0, tcp_relay: 1, failure_streak: 2, path: "relay" },
        ],
        known_models: ["llama3.2:1b"],
        known_providers: [{ model_id: "llama3.2:1b", libp2p_peer_id: "12D3KooWEVGK…", openhydra_peer_id: "7a13…" }],
        counters: { dcutr_successes: 0, dcutr_failures: 0, reversal_dials: 0, reversal_successes: 0, tier_connect_success: { direct_quic_v6: 7, direct_quic_v4: 2, direct_tcp_v4: 3, relay: 3 } },
      },
      transfers: { requests_served: 0, tokens_served: 0, serve_errors: 0, aup_refusals: 0, receipts_ledgered: 0, per_model: {} },
    };
  if (cmd === "web_search")
    return [{ title: "Example result", url: "https://example.com", snippet: "A relevant snippet about " + (args?.query || "") }];
  if (cmd === "chat_completion") {
    const model = args?.model || "llama3.2:1b";
    const sys = (args?.messages || []).some((m) => m.role === "system" && m.content.includes("coding"));
    const judge = (args?.messages || []).some((m) => m.content?.includes("SEAT A"));
    return new Promise((r) => setTimeout(() => r({
      choices: [{ message: { content: judge
        ? "Verdict: SEAT B gave the most complete answer.\n\nSynthesis: combining the seats — the herd agrees the answer is 42."
        : sys ? "```rust\nfn hello() { println!(\"herd\"); }\n```"
        : `(${model}) The swarm answers: reciprocity beats rent-seeking.` } }],
      usage: { completion_tokens: 24 + model.length },
      openhydra: { engine: { native_tps: 60 + model.length }, hops_ms: { network_rtt: 9, discover: 2 } },
      model,
    }), 500 + Math.random() * 600));
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
let attachments = []; // [{name, text}]

// ── persistence (workspace state lives client-side) ──
const store = {
  get(k, d) { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } },
  set(k, v) { localStorage.setItem(k, JSON.stringify(v)); },
};
const DEFAULT_PRESETS = [
  { name: "Default", text: "" },
  { name: "Concise", text: "Answer as briefly as possible. No preamble." },
  { name: "Researcher", text: "Be rigorous. Distinguish facts from speculation; say when you are unsure." },
  { name: "Socratic", text: "Answer, then pose one sharp follow-up question." },
];
let sessions = store.get("oh_sessions", []);
let activeSession = store.get("oh_active", null);
let presets = store.get("oh_presets", DEFAULT_PRESETS);
let memory = store.get("oh_memory", "");
if (document.documentElement) document.documentElement.dataset.theme = store.get("oh_theme", "light");

function saveSessions() { store.set("oh_sessions", sessions); store.set("oh_active", activeSession); }
function currentSession() {
  let s = sessions.find((s) => s.id === activeSession);
  if (!s) {
    s = { id: Date.now().toString(36), name: "New chat", history: [], preset: "Default" };
    sessions.unshift(s); activeSession = s.id; saveSessions();
  }
  return s;
}

// ── view switching ──
document.querySelectorAll(".nav-item").forEach((el) => {
  el.addEventListener("click", () => {
    activeView = el.dataset.view;
    document.querySelectorAll(".nav-item").forEach((n) => n.classList.toggle("nav-active", n === el));
    document.querySelectorAll(".view").forEach((v) => v.classList.toggle("hidden", v.id !== `view-${activeView}`));
    render();
    refreshStatus(); // network views fetch immediately on entry
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

  $("dashProv").textContent = p.running ? "sharing" : p.exited || "stopped";
  $("dashGw").textContent = g.running ? "serving" : g.exited || "stopped";
  $("dashAnnounced").textContent = p.announced ?? "—";
  $("dashRelays").textContent = p.running ? String(p.relays) : "—";
  $("dashUrl").textContent = state.gateway_url;
  $("bootstrapBanner").classList.toggle("hidden", state.settings.bootstraps.length > 0);
  $("agentBanner").classList.toggle("hidden", state.agent_found);

  const peer = p.peer_id || g.peer_id;
  $("sidePeer").textContent = shortPeer(peer);
  $("stPeer").textContent = peer || "—";
  $("stProvDot").className = `dot ${p.running ? "dot-on" : p.exited ? "dot-err" : ""}`;
  $("stGwDot").className = `dot ${g.running ? "dot-on" : g.exited ? "dot-err" : ""}`;
  $("stModels").textContent = p.announced ?? engines.reduce((n, e) => n + e.models.length, 0);
  $("stRelays").textContent = p.relays || 0;
  $("stTokens").textContent = lastTokens;
  $("stTps").textContent = lastTps;

  $("chatGateHint").classList.toggle("hidden", g.running);
  $("webToggleWrap").classList.toggle("hidden", !(state.settings.search_url || "").trim());

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
  for (const e of engines)
    for (const m of e.models)
      rows.push(`<tr><td class="mono">${esc(m)}</td><td>${esc(e.label)}</td>
        <td class="mono">${esc(e.url)}</td>
        <td><span class="pill ${shared ? "pill-shared" : "pill-local"}">${shared ? "shared" : "local"}</span></td></tr>`);
  $("modelRows").innerHTML = rows.length
    ? rows.join("")
    : `<tr><td colspan="4" class="dim">No engines answering — start Ollama, LM Studio, vLLM, llama.cpp, or Exo, then rescan.</td></tr>`;
  $("navModels").textContent = rows.length;
  const models = engines.flatMap((e) => e.models);
  $("modelOptions").innerHTML = models.map((m) => `<option value="${esc(m)}">`).join("");
  for (const id of ["chatModel", "codeModel", "cMod1", "judgeModel"])
    if (!$(id).value && models[0]) $(id).value = models[0];
  if (!$("cMod2").value && models[1]) $("cMod2").value = models[1];
}

function renderPresets() {
  const s = currentSession();
  $("chatPreset").innerHTML = presets.map((p) => `<option ${p.name === s.preset ? "selected" : ""}>${esc(p.name)}</option>`).join("");
}

// ── sessions ──
function renderSessions() {
  const list = $("sessionList");
  list.innerHTML = "";
  for (const s of sessions) {
    const item = document.createElement("div");
    item.className = `rail-item ${s.id === activeSession ? "rail-active" : ""}`;
    item.innerHTML = `<span class="rail-name">${esc(s.name)}</span><button class="rail-del" title="Delete">✕</button>`;
    item.querySelector(".rail-name").addEventListener("click", () => { activeSession = s.id; saveSessions(); renderSessions(); renderChat(); renderPresets(); });
    item.addEventListener("click", (e) => { if (!e.target.classList.contains("rail-del")) { activeSession = s.id; saveSessions(); renderSessions(); renderChat(); renderPresets(); } });
    item.querySelector(".rail-del").addEventListener("click", (e) => {
      e.stopPropagation();
      sessions = sessions.filter((x) => x.id !== s.id);
      if (activeSession === s.id) activeSession = sessions[0]?.id ?? null;
      saveSessions(); renderSessions(); renderChat(); renderPresets();
    });
    list.appendChild(item);
  }
}

function bubbleEl(cls, text) {
  const d = document.createElement("div");
  d.className = `msg ${cls}`;
  d.textContent = text;
  return d;
}

function metaLine(resp) {
  const oh = resp.openhydra || {};
  const parts = [];
  if (resp.usage?.completion_tokens != null) parts.push(`▼ ${resp.usage.completion_tokens} tok`);
  if (oh.engine?.native_tps) parts.push(`${oh.engine.native_tps} tok/s engine`);
  if (oh.hops_ms?.network_rtt != null) parts.push(`rtt ${oh.hops_ms.network_rtt} ms`);
  if (oh.hops_ms?.discover != null) parts.push(`discover ${oh.hops_ms.discover} ms`);
  if (resp.model) parts.push(resp.model);
  return parts.join(" · ");
}

function renderChat() {
  const s = currentSession();
  const scroll = $("chatScroll");
  scroll.innerHTML = "";
  if (!s.history.length) {
    scroll.innerHTML = `<div class="dim center">Ask anything — requests route through the swarm to whichever provider serves the model.</div>`;
    return;
  }
  for (const m of s.history) {
    if (m.role === "user") scroll.appendChild(bubbleEl("msg-user", m.content));
    else if (m.role === "assistant") {
      const b = bubbleEl("msg-bot", m.content);
      if (m.meta) { const mm = document.createElement("div"); mm.className = "msg-meta"; mm.textContent = m.meta; b.appendChild(mm); }
      scroll.appendChild(b);
    }
  }
  scroll.scrollTop = scroll.scrollHeight;
}

/// System prompt = preset + memory (the Odysseus-style always-on context).
function systemPrompt(presetName) {
  const preset = presets.find((p) => p.name === presetName)?.text || "";
  const mem = memory.trim() ? `Persistent user memory (honor it):\n${memory.trim()}` : "";
  return [preset, mem].filter(Boolean).join("\n\n");
}

async function sendChat() {
  const s = currentSession();
  const text = $("chatInput").value.trim();
  const model = $("chatModel").value.trim();
  if (!text || !model) return;
  $("chatInput").value = "";

  // attachments fold into this turn's content
  let content = text;
  if (attachments.length) {
    const files = attachments.map((a) => `--- file: ${a.name} ---\n${a.text}`).join("\n\n");
    content = `${files}\n\n${text}`;
    attachments = []; renderAttachments();
  }

  s.history.push({ role: "user", content });
  if (s.name === "New chat") { s.name = text.slice(0, 34); renderSessions(); }
  renderChat();
  const wait = document.createElement("div");
  wait.className = "msg-wait"; wait.textContent = "thinking";
  $("chatScroll").appendChild(wait);

  // optional web augmentation for this turn
  const messages = [];
  const sys = systemPrompt(s.preset);
  if (sys) messages.push({ role: "system", content: sys });
  if ($("webToggle").checked && (state?.settings.search_url || "").trim()) {
    wait.textContent = "searching";
    try {
      const hits = await call("web_search", { query: text });
      if (hits?.length) {
        messages.push({ role: "system", content: "Fresh web results for the user's question:\n" +
          hits.map((h, i) => `${i + 1}. ${h.title} — ${h.snippet} (${h.url})`).join("\n") +
          "\nUse them where relevant and cite by number." });
      }
    } catch (e) { /* search down → answer without it */ }
    wait.textContent = "thinking";
  }
  messages.push(...s.history.map(({ role, content }) => ({ role, content })));

  try {
    const resp = await call("chat_completion", { model, messages, maxTokens: 1024 });
    wait.remove();
    const reply = resp?.choices?.[0]?.message?.content ?? "(empty reply)";
    s.history.push({ role: "assistant", content: reply, meta: metaLine(resp) });
    saveSessions(); renderChat();
    if (resp.usage?.completion_tokens) lastTokens += resp.usage.completion_tokens;
    if (resp.openhydra?.engine?.native_tps) lastTps = resp.openhydra.engine.native_tps;
    render();
  } catch (e) {
    wait.remove();
    s.history.pop(); saveSessions(); renderChat();
    const err = bubbleEl("msg-err", `${e}`);
    $("chatScroll").appendChild(err);
  }
}

// ── attachments ──
function renderAttachments() {
  const strip = $("attachStrip");
  strip.classList.toggle("hidden", !attachments.length);
  strip.innerHTML = attachments.map((a, i) =>
    `<span class="attach-chip">📄 ${esc(a.name)} <button data-i="${i}">✕</button></span>`).join("");
  strip.querySelectorAll("button").forEach((b) =>
    b.addEventListener("click", () => { attachments.splice(+b.dataset.i, 1); renderAttachments(); }));
}

$("attachBtn").addEventListener("click", () => $("attachFile").click());
$("attachFile").addEventListener("change", async (e) => {
  for (const f of e.target.files) {
    if (f.size > 512 * 1024) { alert(`${f.name}: too large (512 KB max)`); continue; }
    attachments.push({ name: f.name, text: await f.text() });
  }
  e.target.value = "";
  renderAttachments();
});

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

function codeCards(container, content) {
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
      container.appendChild(card);
    } else {
      const pr = document.createElement("div");
      pr.className = "code-prose"; pr.textContent = part.prose;
      container.appendChild(pr);
    }
  }
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
    codeCards(turn, resp?.choices?.[0]?.message?.content ?? "");
    const meta = metaLine(resp);
    if (meta) { const m = document.createElement("div"); m.className = "msg-meta"; m.textContent = meta; turn.appendChild(m); }
    if (resp.usage?.completion_tokens) lastTokens += resp.usage.completion_tokens;
    if (resp.openhydra?.engine?.native_tps) lastTps = resp.openhydra.engine.native_tps;
    render();
  } catch (e) {
    wait.remove();
    const err = bubbleEl("msg msg-err", `${e}`);
    turn.appendChild(err);
  }
  $("codeScroll").scrollTop = $("codeScroll").scrollHeight;
}

// ── council (Odysseus "compare": blind side-by-side + synthesis) ──
let council = null; // { prompt, seats: [{model, content, meta}] } — seat order pre-shuffled

async function convene() {
  const prompt = $("councilInput").value.trim();
  const models = ["cMod1", "cMod2", "cMod3"].map((id) => $(id).value.trim()).filter(Boolean);
  if (!prompt || models.length < 2) { alert("Pick at least two models."); return; }
  $("councilInput").value = "";
  const board = $("councilBoard");
  board.innerHTML = "";
  $("councilActions").classList.add("hidden");
  $("councilVerdict").classList.add("hidden");
  $("councilVerdict").innerHTML = "";

  // blind: shuffle seat order so position ≠ model
  const shuffled = [...models].sort(() => Math.random() - 0.5);
  const seats = shuffled.map((model, i) => ({ model, letter: String.fromCharCode(65 + i) }));
  const els = seats.map((seat) => {
    const el = document.createElement("div");
    el.className = "seat";
    el.innerHTML = `<div class="seat-head"><span>Seat ${seat.letter}</span><span class="seat-model hidden"></span></div>
      <div class="seat-body"><span class="msg-wait">deliberating</span></div>`;
    board.appendChild(el);
    return el;
  });

  const results = await Promise.all(seats.map(async (seat, i) => {
    try {
      const resp = await call("chat_completion", {
        model: seat.model,
        messages: [{ role: "user", content: prompt }],
        maxTokens: 1024,
      });
      const content = resp?.choices?.[0]?.message?.content ?? "(empty)";
      els[i].querySelector(".seat-body").textContent = content;
      const meta = document.createElement("div");
      meta.className = "msg-meta";
      meta.textContent = metaLine({ ...resp, model: undefined }); // keep it blind
      els[i].appendChild(meta);
      els[i].querySelector(".seat-model").textContent = seat.model;
      if (resp.usage?.completion_tokens) lastTokens += resp.usage.completion_tokens;
      return { ...seat, content };
    } catch (e) {
      els[i].querySelector(".seat-body").textContent = `${e}`;
      els[i].querySelector(".seat-body").classList.add("msg-err");
      return { ...seat, content: null };
    }
  }));

  council = { prompt, seats: results.filter((s) => s.content != null) };
  if (council.seats.length >= 2) $("councilActions").classList.remove("hidden");
  render();
}

function reveal() {
  document.querySelectorAll(".seat-model").forEach((el) => el.classList.remove("hidden"));
}

async function synthesize() {
  if (!council) return;
  const judge = $("judgeModel").value.trim();
  if (!judge) { alert("Pick a judge model."); return; }
  const verdictEl = $("councilVerdict");
  verdictEl.className = "verdict";
  verdictEl.innerHTML = `<div class="verdict-title">Synthesis</div><span class="msg-wait">the judge deliberates</span>`;
  const brief = council.seats.map((s) => `SEAT ${s.letter}:\n${s.content}`).join("\n\n");
  try {
    const resp = await call("chat_completion", {
      model: judge,
      messages: [
        { role: "system", content: "You are a strict judge. Given several anonymous answers to the same prompt, briefly say which seat answered best and why, then produce one improved, synthesized answer." },
        { role: "user", content: `PROMPT:\n${council.prompt}\n\n${brief}` },
      ],
      maxTokens: 1024,
    });
    verdictEl.innerHTML = `<div class="verdict-title">Synthesis · judged by ${esc(judge)}</div>`;
    const body = document.createElement("div");
    body.textContent = resp?.choices?.[0]?.message?.content ?? "(empty)";
    verdictEl.appendChild(body);
    if (resp.usage?.completion_tokens) lastTokens += resp.usage.completion_tokens;
    render();
  } catch (e) {
    verdictEl.innerHTML = `<div class="verdict-title">Synthesis failed</div>${esc(String(e))}`;
  }
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
  $("setSearch").value = state.settings.search_url || "";
  $("setMemory").value = memory;
  $("setAutostart").checked = state.settings.engine_autostart;
  $("settingsModal").classList.remove("hidden");
}
function closeSettings() { $("settingsModal").classList.add("hidden"); }
window.openSettings = openSettings;
window.closeSettings = closeSettings;

async function saveSettingsFn() {
  memory = $("setMemory").value;
  store.set("oh_memory", memory);
  const settings = {
    bootstraps: $("setBootstraps").value.split("\n").map((s) => s.trim()).filter(Boolean),
    gateway_port: parseInt($("setPort").value, 10) || 8080,
    engine_autostart: $("setAutostart").checked,
    search_url: $("setSearch").value.trim(),
  };
  try { await call("save_settings", { settings }); closeSettings(); await refresh(); }
  catch (e) { alert(`save failed: ${e}`); }
}

// ── network views (P1): peers / DHT / swarm, fed by the agent's status endpoint ──
let snap = null;

function peerShort(p) { return p.length > 18 ? `${p.slice(0, 12)}…${p.slice(-4)}` : p; }

function renderNetworkViews() {
  const offline = !snap;
  for (const id of ["peersOffline", "dhtOffline", "swarmOffline"]) $(id).classList.toggle("hidden", !offline);
  $("navPeers").textContent = snap ? snap.network.peers.length : 0;
  if (!snap) {
    $("peerRows").innerHTML = `<tr><td colspan="6" class="dim">—</td></tr>`;
    $("swarmSvg").innerHTML = "";
    return;
  }
  const n = snap.network;

  // Peers table
  $("peersCount").textContent = n.peers.length;
  $("peerRows").innerHTML = n.peers.length
    ? n.peers.map((p) => `<tr>
        <td class="mono">${esc(peerShort(p.peer_id))}</td>
        <td><span class="path-dot path-${p.path}"></span>${p.path}</td>
        <td>${p.quic_direct_v4}/${p.quic_direct_v6}</td>
        <td>${p.tcp_direct}</td>
        <td>${p.tcp_relay}</td>
        <td class="${p.failure_streak > 0 ? "streak-bad" : ""}">${p.failure_streak}</td></tr>`).join("")
    : `<tr><td colspan="6" class="dim">No peers connected yet — the node is dialing bootstraps.</td></tr>`;

  // DHT view
  $("dhtMode").textContent = n.kad_server_mode ? "server (reachable)" : "client (NAT'd)";
  $("dhtKadPeers").textContent = n.kad_routing_peers;
  $("dhtNat").textContent = `${n.nat.nat_type}${n.nat.is_public ? " · public" : ""}`;
  $("dhtV6").textContent = n.ipv6_capable ? "yes" : "no";
  $("dhtGen").textContent = n.network_generation;
  const addrList = (arr) => arr.length ? arr.map((a) => `<div title="${esc(a)}">${esc(a)}</div>`).join("") : "";
  $("dhtReservations").innerHTML = addrList(n.relay_reservations);
  $("dhtResvCount").textContent = n.relay_reservations.length;
  $("dhtExternal").innerHTML = addrList(n.external_addrs);
  $("dhtExtCount").textContent = n.external_addrs.length;
  const c = n.counters;
  const counters = [
    ["DCUtR ✓", c.dcutr_successes], ["DCUtR ✗", c.dcutr_failures],
    ["Reversal dials", c.reversal_dials], ["Reversal ✓", c.reversal_successes],
    ...Object.entries(c.tier_connect_success || {}).map(([k, v]) => [k.replace(/_/g, " "), v]),
  ];
  $("dhtCounters").innerHTML = counters.map(([k, v]) =>
    `<div class="counter"><div class="counter-k">${esc(k)}</div><div class="counter-v">${v}</div></div>`).join("");

  if (activeView === "swarm") drawSwarm(n);
}

/// Radial swarm graph: self at center, connected peers on a ring, edges colored by path.
function drawSwarm(n) {
  const svg = $("swarmSvg");
  const W = 800, H = 460, cx = W / 2, cy = H / 2;
  const peers = n.peers;
  const R = Math.min(cx, cy) - 70;
  let out = "";
  // edges first (under nodes)
  peers.forEach((p, i) => {
    const a = (i / Math.max(peers.length, 1)) * Math.PI * 2 - Math.PI / 2;
    const x = cx + R * Math.cos(a), y = cy + R * Math.sin(a);
    const cls = p.path === "relay" ? "swarm-edge-relay" : "swarm-edge-direct";
    out += `<line class="swarm-edge ${cls}" x1="${cx}" y1="${cy}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}"/>`;
  });
  // peer nodes
  peers.forEach((p, i) => {
    const a = (i / Math.max(peers.length, 1)) * Math.PI * 2 - Math.PI / 2;
    const x = cx + R * Math.cos(a), y = cy + R * Math.sin(a);
    const cls = p.path === "relay" ? "swarm-node-relay" : "swarm-node-direct";
    const r = p.failure_streak > 0 ? 7 : 10;
    out += `<circle class="swarm-node ${cls}" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r}"><title>${esc(p.peer_id)} (${p.path})</title></circle>`;
    out += `<text class="swarm-label" x="${x.toFixed(1)}" y="${(y + 22).toFixed(1)}" text-anchor="middle">${esc(peerShort(p.peer_id).slice(0, 10))}</text>`;
  });
  // self
  out += `<circle class="swarm-node swarm-node-self" cx="${cx}" cy="${cy}" r="14"><title>this node</title></circle>`;
  out += `<text class="swarm-label" x="${cx}" y="${cy + 30}" text-anchor="middle">you</text>`;
  svg.innerHTML = out;
}

// ── polling ──
async function refresh() {
  try { state = await call("get_state"); render(); } catch (_) {}
}
async function refreshEngines() {
  try { engines = await call("detect_engines_now"); renderModels(); } catch (_) {}
}
async function refreshStatus() {
  // Only fetch when a network view is visible (peers/dht/swarm) — cheap otherwise.
  if (!["peers", "dht", "swarm"].includes(activeView)) return;
  try { snap = await call("status_snapshot"); renderNetworkViews(); } catch (_) { snap = null; renderNetworkViews(); }
}

// ── wiring ──
$("provBtn").addEventListener("click", () => toggleRole("provider"));
$("gwBtn").addEventListener("click", () => toggleRole("gateway"));
$("chatStartGw").addEventListener("click", () => toggleRole("gateway"));
$("peersStart").addEventListener("click", () => toggleRole("provider"));
$("settingsBtn").addEventListener("click", openSettings);
$("saveSettings").addEventListener("click", saveSettingsFn);
$("refreshEngines").addEventListener("click", refreshEngines);
$("chatSend").addEventListener("click", sendChat);
$("codeSend").addEventListener("click", sendCode);
$("councilAsk").addEventListener("click", convene);
$("councilReveal").addEventListener("click", reveal);
$("councilSynth").addEventListener("click", synthesize);
$("newSession").addEventListener("click", () => { activeSession = null; currentSession(); renderSessions(); renderChat(); renderPresets(); });
$("chatClear").addEventListener("click", () => { const s = currentSession(); s.history = []; saveSessions(); renderChat(); });
$("codeClear").addEventListener("click", () => { $("codeScroll").innerHTML = ""; });
$("chatPreset").addEventListener("change", () => { currentSession().preset = $("chatPreset").value; saveSessions(); });
$("themeBtn").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  store.set("oh_theme", next);
});
for (const [input, send] of [["chatInput", sendChat], ["codeInput", sendCode], ["councilInput", convene]]) {
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

currentSession();
renderSessions();
renderChat();
renderPresets();
refresh();
refreshEngines();
setInterval(refresh, 2000);
setInterval(refreshEngines, 10000);
setInterval(refreshStatus, 2500);
