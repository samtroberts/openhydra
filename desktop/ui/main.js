// OpenHydra Desktop — the shadcn wireframe (docs/openhydra_wireframe_shadcn.html) wired to the
// Rust backend via Tauri IPC. The DOM + CSS are the wireframe verbatim; this file swaps the
// wireframe's demo data for live network state. In a plain browser it renders a demo mock.
import { $, $$, esc, shortPeer } from "./dom";
import { injectIcons } from "./icons";
import { store } from "./storage";
import { modelIcon, fmtUptime, fmtNum, fmtGB } from "./format";
import { hl, parseFences, splitThink, metaRow, mediaKind, mediaEl } from "./text";
import { call } from "./bridge";
import { toast, menu, closeMenus } from "./chrome";
import { accumulateStats, loadStats, totalServed, totalUsed } from "./stats";
import { noteSeen, netModels, curModel, renderModels } from "./models";
import { renderConnectors } from "./connectors";
import { renderChart } from "./chart";
import { renderProviders } from "./providers";
import { renderLedger, renderPeers, renderLogs, setLogTab } from "./network-tables";
import { renderEngines } from "./installer";
import { renderShare, setSharing, toggleSharing, ensureGateway } from "./share";
import { coachShow, maybeStartTour } from "./coach";
import { renderActivity } from "./activity";
import { renderSettings, updateEngineEndpoint } from "./settings";
import { rttSamples, tpsSamples, pushSample } from "./telemetry";
import { on } from "./bus";
import {
  state, snap, engines, installedEngines, sessions, sessionOrder, curChat, activeView, deviceName, usedTokens,
  setState, setSnap, setEngines, setInstalledEngines, setSessions, setSessionOrder, setCurChat, setActiveView, setDeviceName, setUsedTokens,
} from "./state";

const app = $("#app"), root = document.documentElement;
if (/Mac/.test(navigator.platform)) document.body.classList.add("is-mac");
document.addEventListener("contextmenu", (e) => {
  const t = e.target; if (!(t.matches?.("input, textarea") || t.isContentEditable)) e.preventDefault();
});


injectIcons();

// toast + popover menu live in chrome.js; dismiss-open-menus on any click / resize stays wired here:
document.addEventListener("click", closeMenus); addEventListener("resize", closeMenus);

// ── model family badge ──

// ── persistence ──
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
setSessions(coerceSessions(store.get("oh_sessions", {}))); setSessionOrder(store.get("oh_order", []));
setDeviceName(store.get("oh_device", ""));   // #9: derived from the OS on boot if unset
setUsedTokens(store.get("oh_used", 0));
root.dataset.theme = store.get("oh_theme", "light");
if (store.get("oh_adv", false)) app.setAttribute("data-adv", "");
function saveSessions() {
  if (Array.isArray(sessions)) setSessions(coerceSessions(sessions)); // never persist an array (see coerceSessions)
  store.set("oh_sessions", sessions); store.set("oh_order", sessionOrder);
  // #1: durable write-through to the Tauri backend file (WebView localStorage isn't durable
  // across restarts on any platform). Fire-and-forget; localStorage stays as a fast cache.
  try { call("save_sessions", { data: JSON.stringify({ sessions, order: sessionOrder, device: deviceName, used: usedTokens }) }); } catch {}
}

// ── live state ──
let attachments = [];   // chat-local: file attachments queued for the next message
// Rolling per-chat telemetry the agent only emits per-request (there's no aggregated RTT on
// the status API) — we average the last N replies client-side for the Activity view.



// ── nav / workspace switcher / history (wireframe verbatim + header-hide + renderView) ──
const titles = { home: "Home", chat: "Chat", activity: "Activity", connectors: "Connectors", providers: "Models", share: "Share", engines: "Engines", ledger: "Ledger", peers: "Diagnostics and Stats", settings: "Settings" };
const searchable = { providers: 1, peers: 1 };
const VIEWMODE = { home: "home", chat: "home", activity: "home", connectors: "home", providers: "network", share: "network", engines: "network", ledger: "network", peers: "network" };
function setMode(m) { app.dataset.mode = m; $$("#modeswitch button").forEach((b) => b.toggleAttribute("data-on", b.dataset.m === m)); }
let hist = ["home"], hi = 0;
function updNavBtns() { $("#navback").classList.toggle("dis", hi <= 0); $("#navfwd").classList.toggle("dis", hi >= hist.length - 1); }
function go(v, noHist) {
  setActiveView(v); const vm = VIEWMODE[v]; if (vm) setMode(vm);
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
function openChat(id) { setCurChat(id); renderChat(); go("chat"); renderRecents(); }


// reply rendering: fenced code cards + inline media (image/video/audio) + metadata line
// #3/#4: classify a markdown media src as image | video | audio (data: URL or media-file URL),
// "" if it isn't media (a regular link stays as text). Drives which element we render.
// #4: render inline media; #3: copy + download controls under it. Returns null for non-media.
// Reasoning models (Qwen3, DeepSeek-R1, …) emit a chain-of-thought. Some inline it as
// <think>…</think> inside content; some engines (LM Studio serving MLX) strip it and hand
// back an EMPTY content. Split any inline thinking out of the answer so the chat shows the
// real answer (or a clear note) instead of a blank bubble.
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

$("#sharecta").onclick = () => go("share");   // #1: title-bar CTA routes to the Share tab
$("#sharetoggle").onclick = toggleSharing;    // start/stop sharing now lives in the Share view


// ── send ──
async function doSend(text, fromHome) {
  const model = curModel();
  if (!text) return;
  if (!model) { toast("No routable model yet — connecting to the network…"); return; }
  if (fromHome || !curChat) { setCurChat(newSession(text.slice(0, 34))); }
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
    if (resp?.usage?.completion_tokens) { setUsedTokens(usedTokens + resp.usage.completion_tokens); store.set("oh_used", usedTokens); renderStatusbar(); }
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
// ── Host hardware panel (system_info) — shown in the Engines header, LM-Studio style ──
let sysInfo = null;
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
$("#logchips").onclick = (e) => { const c = e.target.closest(".chip"); if (!c) return; $$("#logchips .chip").forEach((x) => x.classList.toggle("on", x === c)); setLogTab(c.dataset.log === "gateway" ? "gateway" : "provider"); renderLogs(); };
$("#search").oninput = () => { if (activeView === "providers") renderProviders(); else if (activeView === "peers") { const q = $("#search").value.toLowerCase(); $$("#peertable tbody tr").forEach((r) => r.style.display = r.textContent.toLowerCase().includes(q) ? "" : "none"); } };
$("#cmdk").onclick = (e) => { e.stopPropagation(); menu($("#cmdk"), Object.keys(titles).map((v) => ({ label: "Go to " + titles[v], on: v === activeView, fn: () => go(v) }))); };
$("#traymark").onclick = (e) => { e.stopPropagation(); menu($("#traymark"), [{ label: "Launch OpenHydra", fn: () => {} }, { sep: 1 }, { label: "Sharing", on: !!state?.provider?.status?.running, fn: toggleSharing }, { label: "Model · " + (netModels()[0] || "—"), fn: () => {} }, { sep: 1 }, { label: `▲ ${fmtNum(totalServed())} served`, fn: () => {} }, { label: `▼ ${fmtNum(totalUsed())} used`, fn: () => {} }, { sep: 1 }, { label: "Quit OpenHydra", fn: () => call("quit") }]); };
$("#addmodel") && ($("#addmodel").onclick = (e) => { e.stopPropagation(); const opts = engines.flatMap((e) => e.models.map((m) => ({ label: `${m} · ${e.label}`, fn: () => toggleSharing() }))); menu($("#addmodel"), opts.length ? opts : [{ label: "No engine models — start an engine", fn: () => go("engines") }]); });

// ── settings ──
$("#setnav").onclick = (e) => { const s = e.target.closest(".s"); if (!s) return; $$("#setnav .s").forEach((x) => x.classList.toggle("on", x === s)); $$(".setpanel").forEach((pnl) => pnl.classList.toggle("on", pnl.dataset.p === s.dataset.p)); };
$$("[data-sw]").forEach((sw) => { if (sw.closest("#servetable")) return; sw.onclick = () => sw.classList.toggle("on"); });
$$(".save").forEach((b) => b.onclick = async () => {
  setDeviceName(($('.setpanel[data-p="identity"] [contenteditable]').textContent || deviceName).trim()); store.set("oh_device", deviceName);
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

maybeStartTour();                              // coach.js: first-run/after-update spotlight tour
$("#obreplay").onclick = () => coachShow(0);   // replay from Settings

// ── polling ──
async function refresh() {
  try { setState(await call("get_state")); } catch {}
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
async function refreshEngines() { try { setEngines(await call("detect_engines_now")); } catch { setEngines([]); } try { setInstalledEngines(await call("installed_engines")); } catch { setInstalledEngines([]); } renderStatusbar(); if (["share", "engines", "providers", "settings"].includes(activeView)) renderView(); }
on("refresh-engines", refreshEngines);   // installer.js fires this after an install completes
on("nav", (v) => go(v));                  // feature modules request navigation via the bus
on("refresh", () => refresh());           // …and a full state refresh
async function refreshStatus() { try { setSnap(await call("status_snapshot")); } catch { setSnap(null); } noteSeen(); accumulateStats(); renderStatusbar(); if (["peers", "providers", "activity", "ledger", "share"].includes(activeView)) renderView(); }
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
      if (d.sessions) { const co = coerceSessions(d.sessions); migrated = co !== d.sessions; setSessions(co); }
      if (Array.isArray(d.order)) setSessionOrder(d.order);
      if (d.device) setDeviceName(d.device);
      if (typeof d.used === "number") { setUsedTokens(d.used); store.set("oh_used", usedTokens); }
    }
  } catch { loadOk = false; }
  // Repair order/sessions drift: drop order ids with no session, and append any session missing
  // from order — otherwise legacy orphaned ids render nothing and recovered chats stay hidden.
  const orderBefore = sessionOrder.join("");
  setSessionOrder(sessionOrder.filter((id) => sessions[id]));
  for (const id in sessions) if (!sessionOrder.includes(id)) sessionOrder.push(id);
  const orderRepaired = sessionOrder.join("") !== orderBefore;
  // Rewrite the durable file ONLY when the load succeeded AND coercion/repair actually changed the
  // shape — never overwrite it with stale cache after a failed/empty load.
  if (loadOk && (migrated || orderRepaired) && Object.keys(sessions).length) saveSessions();
  await loadStats();   // #7/#10: hydrate lifetime model stats + timeline buckets from disk
  // #9: default device name from the OS if the user hasn't set/restored one.
  if (!deviceName) { try { setDeviceName(await call("device_hostname")); } catch {} if (!deviceName) setDeviceName(/Mac/.test(navigator.platform) ? "This Mac" : "This machine"); store.set("oh_device", deviceName); }
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
