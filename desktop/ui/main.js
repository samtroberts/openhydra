// OpenHydra Desktop — the shadcn wireframe (docs/openhydra_wireframe_shadcn.html) wired to the
// Rust backend via Tauri IPC. The DOM + CSS are the wireframe verbatim; this file swaps the
// wireframe's demo data for live network state. In a plain browser it renders a demo mock.
import { $, $$, shortPeer } from "./dom";
import { html, setHTML } from "./html";
import { injectIcons } from "./icons";
import { store } from "./storage";
import { fmtNum, fmtGB, fmtSigned, CREDITS_NOTE } from "./format";
import { hl, parseFences, splitThink, metaRow, mediaEl } from "./text";
import { call } from "./bridge";
import { renderMarkdownMarkup } from "./markdown";
import { toast, menu, closeMenus, confirmModal, failToast } from "./chrome";
import { accumulateStats, loadStats, totalServed, totalUsed, totalCreditsEarned, totalCreditsSpent } from "./stats";
import { noteSeen, netModels, curModel, renderModels, displayModelName, modelIdle, AUTO_MODEL } from "./models";
import { pickerLabel, AUTO_LABEL } from "./picker.js";
import { reportUrl, KNOWN_ISSUES_URL } from "./feedback.js";
import { renderConnectors } from "./connectors";
import { renderProviders, setProviderSort } from "./providers";
import { renderLedger, renderPeers, renderLogs, setLogTab } from "./network-tables";
import { renderEngines } from "./installer";
import { renderShare, setSharing, toggleSharing, ensureGateway, openShareDialog } from "./share";
import { wireCardDeepLinks } from "./cards";
import { renderSwarms } from "./swarms";
import { coachShow, maybeStartTour } from "./coach";
import { renderActivity } from "./activity";
import { renderSettings, updateEngineEndpoint, pendingSettings, engineEndpointValue } from "./settings";
import { tokenizerDownloadsKey } from "./netmode.js";
import { wireSwitches } from "./switches.js";
import { rttSamples, tpsSamples, pushSample } from "./telemetry";
import { on, emit } from "./bus";
import { cliState, refreshCliStatus, installCli } from "./cliui";
import {
  state, snap, engines, sessions, sessionOrder, curChat, activeView, deviceName, usedTokens, infraPeerIds,
  setState, setSnap, setEngines, setInstalledEngines, setSessions, setSessionOrder, setCurChat, setActiveView, setDeviceName, setUsedTokens,
} from "./state";

const app = $("#app"), root = document.documentElement;
if (/Mac/.test(navigator.platform)) document.body.classList.add("is-mac");
document.addEventListener("contextmenu", (e) => {
  const t = e.target; if (!(t.matches?.("input, textarea") || t.isContentEditable)) e.preventDefault();
});


injectIcons();

// toast + popover menu live in chrome.js; dismiss-open-menus on any click / resize stays wired here:
document.addEventListener("click", closeMenus);
// Links in answers open in the default browser, never inside the app window.
document.addEventListener("click", (e) => {
  const a = e.target.closest?.(".msg .md a[href]:not([download])"); if (!a) return;
  e.preventDefault(); e.stopPropagation();
  call("open_external", { url: a.getAttribute("href") }).catch((err) => toast(`Couldn't open the link: ${err}`));
}, true); addEventListener("resize", closeMenus);

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
// Theme choice: "system" follows the OS (live), "light"/"dark" are fixed. An older store holds only the
// applied theme ("oh_theme"), which reads as that fixed choice.
let themeChoice = store.get("oh_theme_choice", null) || store.get("oh_theme", "system");
const osDark = matchMedia("(prefers-color-scheme:dark)");
function applyTheme() { root.dataset.theme = themeChoice === "system" ? (osDark.matches ? "dark" : "light") : themeChoice; }
applyTheme();
osDark.addEventListener?.("change", () => { if (themeChoice === "system") applyTheme(); });
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
const titles = { home: "Home", chat: "Chat", activity: "Activity", connectors: "Connectors", providers: "Models", share: "Share", swarms: "Swarms", engines: "Engines", ledger: "Ledger", peers: "Diagnostics and Stats", settings: "Settings" };
const searchable = { providers: 1, peers: 1 };
const VIEWMODE = { home: "home", chat: "home", activity: "home", connectors: "home", providers: "network", share: "network", swarms: "network", engines: "network", ledger: "network", peers: "network" };
function setMode(m) { app.dataset.mode = m; $$("#modeswitch button").forEach((b) => b.toggleAttribute("data-on", b.dataset.m === m)); }
let hist = ["home"], hi = 0;
function updNavBtns() { $("#navback").classList.toggle("dis", hi <= 0); $("#navfwd").classList.toggle("dis", hi >= hist.length - 1); }
function go(v, noHist) {
  const changed = v !== activeView;
  setActiveView(v); const vm = VIEWMODE[v]; if (vm) setMode(vm);
  // SL4: a new screen starts at its top, not wherever the last one was scrolled to.
  if (changed) { const c = $(".content"); if (c) c.scrollTop = 0; }
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
// m: "system" | "light" | "dark" (any case); no argument flips between light and dark (top-bar button).
function setTheme(m) {
  m = (m || "").toLowerCase();
  themeChoice = m === "system" || m === "light" || m === "dark" ? m : (root.dataset.theme === "dark" ? "light" : "dark");
  store.set("oh_theme_choice", themeChoice); store.set("oh_theme", themeChoice === "system" ? null : themeChoice);
  applyTheme();
  const lbl = $('[data-act="theme"] span'); if (lbl) lbl.textContent = themeChoice[0].toUpperCase() + themeChoice.slice(1);
}
$("#theme").onclick = () => setTheme();
{ const lbl = $('[data-act="theme"] span'); if (lbl) lbl.textContent = themeChoice[0].toUpperCase() + themeChoice.slice(1); }

// ── generic dropdowns (wireframe verbatim) ──
$$(".drop").forEach((d) => d.onclick = (e) => {
  e.stopPropagation();
  const opts = (d.dataset.opts || "").split("|").filter(Boolean);
  // Model dropdowns show a clean label but must route on the RAW id: store it in the span's
  // dataset.value and read that (not the prettified text) as the current selection.
  const isModel = d.id === "modeldrop" || d.id === "homedrop";
  const sp = d.querySelector("span");
  const cur = isModel ? (sp.dataset.value || sp.textContent) : sp.textContent;
  const setSel = (el, o) => { if (isModel) { el.dataset.value = o; el.textContent = pickerLabel(o, displayModelName); } else { el.textContent = o; } };
  const modelLabel = (o) => o === AUTO_MODEL ? AUTO_LABEL + " · picks a trusted model for you" : displayModelName(o) + (modelIdle(o) ? "  · not seen recently" : "");
  menu(d, opts.length ? opts.map((o) => ({ label: isModel ? modelLabel(o) : o, on: o === cur, fn: () => { setSel(sp, o); if (d.dataset.act === "theme") setTheme(o); if (d.id === "enginedrop") { updateEngineEndpoint(); saveSetting({ engine_kind: ENGINE_KIND_OF[o] || "auto", engine_endpoint: "" }); } if (d.id === "presetdrop") applyPreset(o); if (isModel) { setChatMode(o); const other = $(d.id === "modeldrop" ? "#homedrop" : "#modeldrop"); if (other) setSel(other.querySelector("span"), o); } } })) : [{ label: "No models on the network yet", fn: () => {} }]);
});

// ── inference range sliders (wireframe verbatim) ──
$$(".range").forEach((rg) => {
  const mn = +rg.dataset.min, mx = +rg.dataset.max, isInt = rg.dataset.int, fill = rg.querySelector(".fill"), knob = rg.querySelector(".knob"), lab = $("#" + rg.dataset.label);
  function set(v) { v = Math.max(mn, Math.min(mx, v)); const p = (v - mn) / (mx - mn) * 100; fill.style.width = p + "%"; knob.style.left = p + "%"; lab.textContent = rg.dataset.name + " · " + (isInt ? Math.round(v) : Math.round(v * 100) / 100); rg.dataset.val = v; }
  set(+rg.dataset.val); rg._set = (v) => { rg.dataset.touched = "1"; set(v); }; let drag = 0;
  // Until the user sets the temperature, the provider's default for the model applies.
  if (rg.dataset.label === "tempk") lab.textContent = "Temperature · model default";
  // Dragging a slider by hand leaves the preset: show "Custom".
  function fe(e) { const r = rg.getBoundingClientRect(); rg.dataset.touched = "1"; set(mn + ((e.clientX - r.left) / r.width) * (mx - mn)); const ps = $("#presetdrop span"); if (ps) ps.textContent = "Custom"; }
  rg.onmousedown = (e) => { drag = 1; fe(e); e.preventDefault(); }; addEventListener("mousemove", (e) => { if (drag) fe(e); }); addEventListener("mouseup", () => drag = 0);
});
$("#inftoggle").onclick = () => $("#v-chat .three").classList.toggle("noinf");
// Presets set both sliders (temperature, max tokens).
const PRESETS = { "Concise · temp 0.3": [0.3, 512], "Creative · temp 1.2": [1.2, 2048], "Deterministic · temp 0": [0, 2048] };
function applyPreset(name) {
  const p = PRESETS[name]; if (!p) return;
  $("[data-label='tempk']")?._set?.(p[0]);
  $("[data-label='tokk']")?._set?.(p[1]);
}
// Settings › Engine labels → the agent's --engine-kind values.
const ENGINE_KIND_OF = { "auto-detect": "auto", ollama: "ollama", vLLM: "vllm", "LM Studio": "lm-studio", "llama.cpp": "llama-cpp", Exo: "exo", ComfyUI: "comfyui" };

// ── recents (live sessions) ──
function renderRecents() {
  const box = $("#recents"), q = ($("#recfilter").value || "").toLowerCase();
  box.replaceChildren();
  for (const id of sessionOrder) {
    const s = sessions[id]; if (!s) continue;
    const el = document.createElement("div");
    el.className = "nav recent" + (id === curChat && activeView === "chat" ? " on" : "");
    el.dataset.chat = id;
    setHTML(el, html`<span class="icon" data-i="chat"></span><span class="rt" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1">${s.t}</span><span class="recmenu" title="Rename or delete">⋯</span>`);
    injectIcons(el);
    el.style.display = q && !s.t.toLowerCase().includes(q) ? "none" : "";
    box.appendChild(el);
  }
}
$("#recents").onclick = (e) => {
  const r = e.target.closest(".recent"); if (!r) return;
  if (e.target.closest(".recmenu")) { e.stopPropagation(); recentMenu(e.target.closest(".recmenu"), r.dataset.chat); return; }
  openChat(r.dataset.chat);
};
// Rename or delete a chat (the ⋯ on hover, or right-click).
$("#recents").addEventListener("contextmenu", (e) => { const r = e.target.closest(".recent"); if (!r) return; e.preventDefault(); recentMenu(r.querySelector(".recmenu") || r, r.dataset.chat); });
function recentMenu(anchor, id) {
  menu(anchor, [
    { label: "Rename", fn: () => renameChat(id) },
    { label: "Delete", fn: () => deleteChat(id) },
  ]);
}
function renameChat(id) {
  const row = $(`#recents .recent[data-chat="${id}"] .rt`), s = sessions[id]; if (!row || !s) return;
  row.contentEditable = "true"; row.spellcheck = false; row.focus();
  document.getSelection()?.selectAllChildren(row);
  let finished = false;
  const done = (save) => {
    if (finished) return; finished = true;
    row.onblur = null;
    row.contentEditable = "false";
    const t = (row.textContent || "").trim();
    if (save && t) { s.t = t.slice(0, 80); saveSessions(); }
    renderRecents(); if (curChat === id) renderChat();
  };
  row.onkeydown = (e) => { if (e.key === "Enter") { e.preventDefault(); done(true); } else if (e.key === "Escape") { e.preventDefault(); done(false); } };
  row.onblur = () => done(true);
  row.onclick = (e) => e.stopPropagation();
}
async function deleteChat(id) {
  const s = sessions[id]; if (!s) return;
  if (!(await confirmModal(`Delete "${s.t}"?`, "The conversation is removed from this device. This can't be undone.", "Delete"))) return;
  delete sessions[id];
  setSessionOrder(sessionOrder.filter((x) => x !== id));
  saveSessions();
  if (curChat === id) { setCurChat(null); if (activeView === "chat") go("home"); }
  renderRecents();
}
$("#recfilter").oninput = renderRecents;
function newSession(title) { const id = "c" + Date.now().toString(36); sessions[id] = { t: title || "New chat", m: [] }; sessionOrder.unshift(id); saveSessions(); return id; }
function openChat(id) { setCurChat(id); renderChat(); go("chat"); renderRecents(); }


// reply rendering: fenced code cards + inline media (image/video/audio) + metadata line
// #3/#4: classify a markdown media src as image | video | audio (inline data: URLs only),
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
    setHTML(det, html`<summary style="cursor:pointer;color:hsl(var(--muted-foreground))">🧠 Thinking</summary>`);
    const b = document.createElement("div"); b.style.cssText = "white-space:pre-wrap;margin-top:4px;padding:6px 10px;border-left:2px solid hsl(var(--border));color:hsl(var(--muted-foreground))"; b.textContent = reasoning; det.appendChild(b); d.appendChild(det);
  }
  if (content && content.trim()) {
    for (const part of parseFences(content)) {
      if (part.code != null) { const c = document.createElement("div"); c.className = "code-card"; setHTML(c, html`<div class="code-card-head"><span>${part.lang}</span><button class="code-copy">⎘ copy</button></div><pre>${hl(part.code)}</pre>`); c.querySelector(".code-copy").onclick = (ev) => { navigator.clipboard?.writeText(part.code); ev.target.textContent = "✓ copied"; setTimeout(() => ev.target.textContent = "⎘ copy", 1400); }; d.appendChild(c); }
      else {
        // Prose: inline media (images, audio, video) as players; everything else as markdown.
        const pr = document.createElement("div"); pr.className = "md";
        // Bounded so a long run of brackets can't make this scan quadratic (data URLs have no "[", "(" or spaces).
        const re = /!?\[([^\]\n]{0,500})\]\(([^)\s[(]+)\)/g; let last = 0, m, buf = "";
        const flush = () => { if (buf) { const md = document.createElement("div"); setHTML(md, renderMarkdownMarkup(buf)); pr.appendChild(md); buf = ""; } };
        while ((m = re.exec(part.prose))) {
          const mel = mediaEl(m[2].trim(), m[1]);
          buf += part.prose.slice(last, m.index);
          if (mel) { flush(); pr.appendChild(mel); } else buf += m[0];
          last = re.lastIndex;
        }
        buf += part.prose.slice(last); flush();
        d.appendChild(pr);
      }
    }
  } else {
    // No visible answer — reasoning-model / thinking-mode case. Say so instead of a blank bubble.
    const note = document.createElement("div"); note.className = "mut"; note.style.fontSize = "12.5px";
    // A reply stopped (or cut off) before any text says that, not that the model is "thinking".
    note.textContent = meta?.stopped
      ? (meta.stopped === true || meta.stopped === "stopped" ? "Stopped before any reply arrived." : `No reply — it ${meta.stopped}.`)
      : reasoning
      ? "The model produced only reasoning — expand “Thinking” above, or ask it to answer directly."
      : "The model returned no visible text — it may be in “thinking” mode. Ask it to answer directly, or turn off reasoning in the provider’s engine (e.g. add /no_think to the prompt).";
    d.appendChild(note);
  }
  if (meta) { const mm = document.createElement("div"); setHTML(mm, metaRow(meta)); d.appendChild(mm.firstChild); }
  return d;
}
// The reply being streamed, and the chat it belongs to: a redraw of that chat (opening it from
// Recents, a rename) re-attaches it instead of dropping it until the reply ends (ADVERSARIAL item 8).
let liveView = null;
function renderChat() {
  const s = sessions[curChat]; const th = $("#thread"); if (!s) { th.replaceChildren(); return; }
  $("#chattitle").textContent = s.t; th.replaceChildren();
  if (!s.m.length) { setHTML(th, html`<div class="mut" style="margin:auto;text-align:center;font-size:12.5px">Ask anything — requests route through the network to whichever provider serves the model.</div>`); return; }
  for (const x of s.m) {
    if (x[0] === "me") {
      // A message with files shows the file names, not the inlined file text (the model gets both).
      const d = document.createElement("div"); d.className = "msg me";
      const files = x[4];
      if (files?.length) {
        const chips = document.createElement("div"); chips.className = "msgfiles"; chips.textContent = files.map((n) => `📎 ${n}`).join("  ");
        d.appendChild(chips);
        const marker = x[1].lastIndexOf("\n\n"); d.appendChild(document.createTextNode(x[5] ?? (marker >= 0 ? x[1].slice(marker + 2) : x[1])));
      } else d.textContent = x[1];
      th.appendChild(d);
    } else th.appendChild(botEl(x[1], x[2], x[3]));
  }
  // A reply still streaming into this chat: put its bubble (or the wait note) back after a redraw.
  if (liveView && liveView.chat === curChat) th.appendChild(liveView.started ? liveView.live : liveView.wait);
  th.scrollTop = th.scrollHeight;
}

$("#sharecta").onclick = () => go("share");   // #1: title-bar CTA routes to the Share tab
$("#sharetoggle").onclick = toggleSharing;    // start/stop sharing now lives in the Share view


// ── send ──
let chatBusy = false;   // one request at a time, from Home or Chat
async function doSend(text, fromHome) {
  if (chatBusy) return;
  chatBusy = true;
  try { await sendOne(text, fromHome); } finally { chatBusy = false; }
}
async function sendOne(text, fromHome) {
  const model = curModel();
  if (!text) return;
  if (!model) { toast("No routable model yet — connecting to the network…"); return; }
  if (fromHome || !curChat) { setCurChat(newSession(text.slice(0, 34))); }
  const s = sessions[curChat];
  let content = text;
  let files = null;
  if (attachments.length) { content = attachments.map((a) => `--- file: ${a.name} ---\n${a.text}`).join("\n\n") + `\n\n${text}`; files = attachments.map((a) => a.name); attachments = []; renderAttachChips(); }
  s.m.push(["me", content, null, null, files, files ? text : undefined]); if (s.t === "New chat") s.t = text.slice(0, 34); saveSessions();
  if (fromHome) go("chat");
  renderRecents(); renderChat();
  const th = $("#thread"); const wait = document.createElement("div"); wait.className = "mut"; wait.style.fontSize = "12.5px"; wait.textContent = "thinking…"; th.appendChild(wait); th.scrollTop = th.scrollHeight;
  // A provider loading the model can take ~20 s on the first request: say so instead of a bare wait.
  const slow = setTimeout(() => { if (wait.isConnected) wait.textContent = "Waiting for the provider — the first request to a model can take up to 20 s while it loads…"; }, 3000);
  if (!(await ensureGateway())) { wait.remove(); return; }
  const messages = s.m.map((x) => ({ role: x[0] === "me" ? "user" : "assistant", content: x[1] }));
  const t0 = Date.now();
  try {
    // Tauri v2 maps camelCase JS args → snake_case Rust params, so `maxTokens` binds to the
    // command's `max_tokens`. Send the slider's current value; if it's somehow missing, omit it
    // and let the Rust command apply its generous default (a reasoning model needs well over the
    // old 512 or it spends the whole budget on hidden thinking and returns nothing).
    const tokLimit = Math.round(+($("[data-label='tokk']")?.dataset.val)) || undefined;
    // Temperature: sent once the user has set it (slider or preset); otherwise the provider's own
    // default for the model applies.
    const tr = $("[data-label='tempk']"), tv = +(tr?.dataset.val);
    const temperature = tr?.dataset.touched && Number.isFinite(tv) ? Math.round(tv * 100) / 100 : undefined;
    const resp = await completeChat({ model, messages, maxTokens: tokLimit, temperature }, th, wait, slow);
    if (resp?.stopped) { s.m.push(["ai", resp.text || "", { model, tok: "—", tps: "—", rtt: "—", tier: null, at: new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }), stopped: resp.why || "stopped" }, resp.reasoning || null]); saveSessions(); renderChat(); return; }
    clearTimeout(slow); wait.remove();
    const msg = resp?.choices?.[0]?.message || {};
    const split = splitThink(msg.content || "");                 // pull inline <think> out of the answer
    const reasoning = [msg.reasoning_content, split.reasoning].filter(Boolean).join("\n").trim();
    const reply = split.answer;                                  // may be "" — botEl shows a clear note
    const oh = resp?.openhydra || {};
    const meta = { model: resp?.model || model, tok: resp?.usage?.completion_tokens ?? "—", claimed: Number.isFinite(oh.claimed_tokens) ? oh.claimed_tokens : null, tps: oh.engine?.native_tps ? Math.round(oh.engine.native_tps) : "—", rtt: oh.hops_ms?.network_rtt ?? "—", tier: oh.provider_authenticity || null, at: new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }) };
    pushSample(tpsSamples, oh.engine?.native_tps, "oh_tps");   // rolling throughput for Activity
    pushSample(rttSamples, oh.hops_ms?.network_rtt, "oh_rtt"); // rolling latency for Activity
    s.m.push(["ai", reply, meta, reasoning || null]); saveSessions(); renderChat();
    // A reply from another machine counts as used; one from this Mac (no provider tier) doesn't.
    if (resp?.usage?.completion_tokens && oh.provider_authenticity) { setUsedTokens(usedTokens + resp.usage.completion_tokens); store.set("oh_used", usedTokens); renderStatusbar(); }
  } catch (e) {
    clearTimeout(slow); wait.remove(); const secs = Math.round((Date.now() - t0) / 1000);
    const err = document.createElement("div"); err.className = "msg ai"; err.style.color = "hsl(var(--danger))"; err.style.fontSize = "12.5px";
    // A gateway error reply arrives as "gateway <status> <type>: <message>"; anything else is ours.
    const es = String(e), gw = /^gateway (\d{3}) ([a-z_]*): ([\s\S]*)$/.exec(es);
    err.textContent = /is verified at A\d/.test(es) ? "No provider meets your \u2018Only use trusted providers\u2019 setting yet. Lower it in Settings, or keep using providers so they build a track record."
      : /upstream_overloaded|server_overloaded/.test(es) ? "Providers are busy right now. Try again in a moment."
      : /504|timeout|timed out/i.test(es) ? `The provider didn't respond in time (${secs}s). Cold model loads can be slow — try again; it warms up.`
      : gw ? gw[3]   // the gateway's own words (a refusal, a bad request): shown as they are
      : /control character|expected value|invalid|parse|json/i.test(es) ? `The provider's response couldn't be read — some reasoning models emit raw formatting that breaks parsing. Try turning off thinking on the provider (e.g. /no_think).`
      : es;
    th.appendChild(err);
  }
}
// Run one chat request. In the app it streams: the answer appears as it is generated and Send turns
// into Stop. Resolves to the same shape as a non-streamed reply (choices / usage / openhydra), or
// { stopped, text } when the user stopped it. The browser mock (no Tauri channel) asks in one go.
let activeStream = null;
// What the live bubble shows while a reply streams: the answer so far ("Thinking…" while only
// reasoning has arrived). Media arriving as a data URL, or a very long reply, shows its progress
// instead of megabytes of raw text.
function livePreview(text, reasoning) {
  const mb = (n) => (n / (1024 * 1024)).toFixed(1);
  if (/\]\(data:(image|video|audio)\//i.test(text.slice(0, 4096)) || /\]\(data:(image|video|audio)\//i.test(text.slice(-4096))) return `Receiving media… ${mb(text.length)} MB`;
  if (text.length > 512 * 1024) return `Receiving a long reply… ${mb(text.length)} MB so far`;
  const sp = splitThink(text);
  return sp.answer || (sp.reasoning || reasoning ? "Thinking…" : "");
}
async function completeChat(args, th, wait, slow) {
  const Channel = window.__TAURI__?.core?.Channel;
  if (!Channel) return call("chat_completion", args);
  const streamId = "s" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
  const live = document.createElement("div"); live.className = "msg ai"; live.style.whiteSpace = "pre-wrap";
  const view = { chat: curChat, live, wait, started: false };
  liveView = view;
  let text = "", reasoning = "", done = null, error = null, stopped = false, incomplete = false;
  // The last event is "end": large messages (an image as a data URL) can arrive after the command
  // itself has returned, so the reply is complete only once "end" is seen.
  let ended; const endSeen = new Promise((r) => (ended = r));
  const ch = new Channel();
  // Redraw the live bubble at most once a frame, not on every fragment: re-reading the whole reply
  // per fragment is quadratic and froze the window on a long one (ADVERSARIAL item 11).
  let frame = 0;
  const paint = () => {
    frame = 0;
    if (!live.isConnected && curChat === view.chat) th.appendChild(live);
    live.textContent = livePreview(text, reasoning);
    th.scrollTop = th.scrollHeight;
  };
  ch.onmessage = (ev) => {
    if (ev.kind === "delta" || ev.kind === "reasoning") {
      if (ev.kind === "delta") text += ev.text; else reasoning += ev.text;
      if (!view.started) { view.started = true; clearTimeout(slow); wait.remove(); }
      if (!frame) frame = requestAnimationFrame(paint);
    } else if (ev.kind === "done") done = ev;
    else if (ev.kind === "error") error = ev.message;
    else if (ev.kind === "stopped") stopped = true;
    else if (ev.kind === "incomplete") incomplete = true;
    else if (ev.kind === "end") ended();
  };
  activeStream = streamId; setSendMode(true);
  try {
    await call("chat_completion_stream", { streamId, model: args.model, messages: args.messages, maxTokens: args.maxTokens, temperature: args.temperature, onEvent: ch });
    await Promise.race([endSeen, new Promise((r) => setTimeout(r, 30000))]);
  } finally {
    if (frame) cancelAnimationFrame(frame);
    if (activeStream === streamId) { activeStream = null; setSendMode(false); }
    if (liveView === view) liveView = null;
    live.remove();
  }
  const sp = splitThink(text);
  // A stopped, failed or cut-off reply keeps what arrived (and says so) instead of losing it.
  if (stopped || ((error || incomplete) && (sp.answer || sp.reasoning || reasoning))) {
    return { stopped: true, why: stopped ? "stopped" : error ? `ended with an error: ${error}` : "ended early", text: sp.answer, reasoning: [reasoning, sp.reasoning].filter(Boolean).join("\n") };
  }
  if (error) throw error;
  return {
    choices: [{ message: { content: text, reasoning_content: reasoning || undefined } }],
    usage: done?.usage || {}, openhydra: done?.openhydra || {}, model: done?.model || args.model,
  };
}
// Send ↔ Stop while a reply is streaming.
function setSendMode(streaming) {
  $("#sendlbl").textContent = streaming ? "Stop" : (chatMode === "image" ? "Generate" : "Send");
  $("#send").classList.toggle("stop", streaming);
}
function send(fromKey) {
  if (activeStream || chatBusy) {
    if (fromKey) { toast("Wait for the reply, or press Stop"); return; }
    if (activeStream) { $("#sendlbl").textContent = "Stopping…"; call("chat_stop", { streamId: activeStream }).catch(() => {}); }
    return;
  }
  const c = $("#composer"); const t = c.textContent.trim(); c.textContent = ""; doSend(t, false); }
$("#composer").onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(true); } };
$("#send").onclick = () => send(false);
function homeSend() { if (chatBusy) { toast("Wait for the reply, or press Stop"); return; } const p = $("#homeprompt"); const t = p.textContent.trim(); p.textContent = ""; doSend(t, true); }
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
function setChatMode(model) { chatMode = /flux|sdxl|(^|[^a-z])sd($|[^a-z])/i.test(model || "") ? "image" : "text"; const img = chatMode === "image"; $("#composer").dataset.ph = img ? "Describe an image…" : "Message…"; setSendMode(!!activeStream); $("#txtparams").style.display = img ? "none" : ""; $("#imgparams").style.display = img ? "" : "none"; }

// attachments (paperclip → hidden file input)
$$('button[title="Attach files to context"]').forEach((b) => b.onclick = () => $("#attachfile").click());
$("#attachfile").onchange = async (e) => {
  for (const f of e.target.files) {
    if (f.size > 512 * 1024) { toast(`${f.name}: too large (512 KB max)`); continue; }
    const text = await f.text();
    if (text.includes("\u0000")) { toast(`${f.name}: not a text file`); continue; }
    attachments.push({ name: f.name, text });
  }
  e.target.value = ""; renderAttachChips();
};
// The files queued for the next message, as removable chips above both composers.
function renderAttachChips() {
  for (const host of [$("#homeattach"), $("#chatattach")]) {
    if (!host) continue;
    setHTML(host, html`${attachments.map((a, i) => html`<span class="achip" title="${a.name}">📎 ${a.name}<button data-i="${i}" title="Remove">×</button></span>`)}`);
    host.style.display = attachments.length ? "" : "none";
    host.querySelectorAll("button").forEach((b) => b.onclick = () => { attachments.splice(+b.dataset.i, 1); renderAttachChips(); });
  }
}

// ── view renderers (populate the wireframe's exact tables/cards with live data) ──
function renderView() {
  if (activeView === "share") renderShare();
  else if (activeView === "swarms") renderSwarms();
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
  const p = state?.provider?.status, g = state?.gateway?.status;
  // Real peers = network participants, excluding infrastructure (bootstraps + relay hops) — matches
  // the Diagnostics peer list, which filters the same set. The footer used to count infra too, so it
  // read e.g. "4 peers" while Diagnostics listed 1. `onNetwork` keeps the connection STATUS on ANY
  // live link, since a bootstrap connection still means we're on the network.
  const conns = snap?.network?.peers || [], infra = infraPeerIds();
  const peers = conns.filter((pr) => !infra.has(pr.peer_id)).length, onNetwork = conns.length > 0;
  const anyOn = !!(p?.running || g?.running);
  let dot = "warn pulse", label = "Initializing…";
  if (state) { if (!anyOn) { dot = ""; label = "Ready — connecting…"; } else if (onNetwork) { dot = "ok pulse"; label = "Connected"; } else { dot = "warn pulse"; label = "Connecting to network…"; } }
  $("#netdot").className = "dot " + dot; $("#netlabel").textContent = label;
  $("#sbpeers").textContent = `${peers} peer${peers === 1 ? "" : "s"}`;
  $("#sbserved").textContent = fmtNum(totalServed()); $("#sbused").textContent = fmtNum(totalUsed());
  // Net credits, the same figure as Activity's net balance.
  // Net tokens (served − used) shown; the cost-weighted credits in the tooltip.
  if (snap) {
    const el = $("#sbcredits"), credits = Math.round(totalCreditsEarned() - totalCreditsSpent());
    el.textContent = `${fmtSigned(totalServed() - totalUsed())} tokens net`;
    el.title = `Tokens served minus tokens used: ${fmtSigned(credits)} credits. ${CREDITS_NOTE}`;
  }
  $("#apiendpoint").textContent = (state?.gateway_url || "http://127.0.0.1:16527/v1").replace(/^https?:\/\//, "") + (g?.running ? "" : " · off");
  // Every place that names the local API uses the configured port.
  const port = state?.settings?.gateway_port || 16527;
  $$(".gwbase").forEach((el) => { const t = `http://127.0.0.1:${port}${el.dataset.suffix || ""}`; if (el.textContent !== t) el.textContent = t; });
  $$("#v-connectors .snippet").forEach((sn) => { const t = sn.textContent.replace(/127\.0\.0\.1:\d+/g, `127.0.0.1:${port}`); if (t !== sn.textContent) sn.textContent = t; });
  // #1: the title-bar CTA reflects state. Off = a brand-filled call-to-action ("Share your
  // models"). Live = a calm green "active" chip with a pulsing dot ("Sharing N models") — it reads
  // as status, not an unclicked CTA. Reuses the app's shared `.btn.on` active treatment.
  const shb = $("#sharecta");
  if (shb) {
    // Models being shared, public or swarm-only (the intended set from the live share view; the
    // announced count alone left swarm-only sharing reading "0 models shared").
    const sv = snap?.share, running = !!p?.running;
    const engineModels = new Set(engines.flatMap((e) => e.models));
    const n = !sv ? (p?.announced ?? 0) : sv.share_mode === "all" ? engineModels.size : (sv.shared_models || []).filter((m) => engineModels.has(m)).length;
    const sharing = running && n > 0;   // actually announcing ≥1 model — not merely "provider process up"
    shb.classList.toggle("brand", !running);   // brand CTA only when there's no provider at all
    shb.classList.toggle("on", sharing);       // green "active" ONLY when a model is really shared
    shb.title = running
      ? (n > 0
          ? `Sharing ${n} model${n === 1 ? "" : "s"} to the network — click to manage`
          : "Provider on, but 0 models shared — click to choose what to share")
      : "Share your machine's models to the network";
    setHTML(shb, running
      ? (n > 0
          ? html`<span class="dot ok pulse"></span>Sharing ${n} model${n === 1 ? "" : "s"}`
          : html`<span class="dot"></span>0 models shared`)   // idle: neutral dot, honest label — not "Sharing…"
      : html`Share your models`);
  }
  $("#sidepeer").textContent = `${deviceName} · ${shortPeer(p?.peer_id || g?.peer_id)}`;
  renderModels();
  const connected = anyOn && (onNetwork || netModels().length > 0);
  $("#homeconnecting").style.display = connected ? "none" : "inline-flex";
  $("#homeready").style.display = connected ? "" : "none";
}
$("#apiendpoint").onclick = async () => { if (!state?.gateway?.status?.running) { await ensureGateway(); toast("Local API started"); return; } navigator.clipboard?.writeText(state?.gateway_url || "http://127.0.0.1:16527/v1"); toast("Copied"); };

// ── chips ──
$("#provchips").onclick = (e) => { const c = e.target.closest(".chip[data-cat]"); if (!c) return; $$("#provchips .chip[data-cat]").forEach((x) => x.classList.toggle("on", x === c)); const cat = c.dataset.cat; $$("#provtable .prov").forEach((r) => r.style.display = (cat === "all" || r.dataset.cat === cat) ? "" : "none"); };
$("#actchips").onclick = (e) => { const c = e.target.closest(".chip[data-act]"); if (!c) return; $$("#actchips .chip").forEach((x) => x.classList.toggle("on", x === c)); $$("#v-peers .acttab").forEach((t) => t.classList.toggle("on", t.dataset.act === c.dataset.act)); };
$("#peerchips").onclick = (e) => { const c = e.target.closest(".chip"); if (!c) return; $$("#peerchips .chip").forEach((x) => x.classList.toggle("on", x === c)); const pp = c.dataset.p; $$("#peertable tbody tr").forEach((r) => r.style.display = (pp === "all" || r.dataset.p === pp) ? "" : "none"); };
$("#logchips").onclick = (e) => { const c = e.target.closest(".chip"); if (!c) return; $$("#logchips .chip").forEach((x) => x.classList.toggle("on", x === c)); setLogTab(c.dataset.log === "gateway" ? "gateway" : "provider"); renderLogs(); };
$("#provsort") && ($("#provsort").onclick = (e) => { e.stopPropagation(); menu($("#provsort"), ["reputation", "providers", "name"].map((v) => ({ label: `Sort by ${v}`, fn: () => setProviderSort(v) }))); });
// Keys, peer ids, magnets and addresses (the monospace fields) are typed or pasted exactly: never
// auto-capitalised, auto-corrected or spell-checked by the OS.
document.addEventListener("focusin", (e) => {
  const t = e.target;
  if (t instanceof HTMLElement && t.matches("input.mono, textarea.mono")) {
    t.setAttribute("autocapitalize", "off");
    t.setAttribute("autocorrect", "off");
    t.setAttribute("spellcheck", "false");
  }
});
$("#search").oninput = () => { if (activeView === "providers") renderProviders(); else if (activeView === "peers") { const q = $("#search").value.toLowerCase(); $$("#peertable tbody tr").forEach((r) => r.style.display = r.textContent.toLowerCase().includes(q) ? "" : "none"); renderLogs(); } };
// ⌘K (Ctrl+K elsewhere) or the button: a "Go to …" menu. Diagnostics is listed only when advanced
// views are on (it's hidden otherwise).
function openGoto() { menu($("#cmdk"), Object.keys(titles).filter((v) => v !== "peers" || app.hasAttribute("data-adv")).map((v) => ({ label: "Go to " + titles[v], on: v === activeView, fn: () => go(v) })), { filter: true }); }
$("#cmdk").onclick = (e) => { e.stopPropagation(); openGoto(); };
document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && !e.altKey && !e.shiftKey && e.key.toLowerCase() === "k") {
    e.preventDefault();
    if (document.querySelector(".cmodal-back, #installov[style*='flex']")) return;   // not over a dialog
    document.activeElement?.blur?.();   // so the arrow keys and Enter drive the menu
    openGoto();
  }
});
// The logo goes Home. (It used to open a "tray menu preview" whose Quit called a command that
// doesn't exist; the real system tray has Sharing and Quit.)
$("#traymark").onclick = () => go("home");
// SL12: feedback paths — the status bar's "Feedback" and Settings › Appearance › Send feedback.
const openLink = (url) => call("open_external", { url }).catch((err) => toast(`Couldn't open the link: ${err}`));
const appVersion = () => ($("#sbver")?.textContent || "").trim();
const feedbackItems = () => [
  { label: "Report a problem on GitHub", fn: () => openLink(reportUrl(appVersion())) },
  { label: "Known issues", fn: () => openLink(KNOWN_ISSUES_URL) },
];
$("#feedback").onclick = (e) => { e.stopPropagation(); menu($("#feedback"), feedbackItems()); };
// While its menu is open, keys go to the menu (↑/↓, Enter picks), not back to this opener.
$("#feedback").onkeydown = (e) => { if (document.querySelector(".menu")) return; if (e.key === "Enter" || e.key === " ") { e.preventDefault(); e.stopPropagation(); menu($("#feedback"), feedbackItems()); } };
$("#fbreport").onclick = () => openLink(reportUrl(appVersion()));
$("#fbknown").onclick = () => openLink(KNOWN_ISSUES_URL);
$("#addmodel") && ($("#addmodel").onclick = (e) => { e.stopPropagation(); const opts = engines.flatMap((e) => e.models.map((m) => ({ label: `${m} · ${e.label}`, fn: () => openShareDialog([m]) }))); menu($("#addmodel"), opts.length ? opts : [{ label: "No engine models — start an engine", fn: () => go("engines") }]); });

// ── settings ──
$("#setnav").onclick = (e) => { const s = e.target.closest(".s"); if (!s) return; $$("#setnav .s").forEach((x) => x.classList.toggle("on", x === s)); $$(".setpanel").forEach((pnl) => pnl.classList.toggle("on", pnl.dataset.p === s.dataset.p)); };
// Settings save as they change: switches and selects at once, text fields on Enter or when they
// lose focus. `pendingSettings` keeps the new value on screen until the save lands, so the 2.5 s poll
// can't flip it back. Each toast says when the change takes effect.
const SETTING_SWITCHES = { engineautostartsw: "engine_autostart", authchecksw: "authenticity_checks", auditreplaysw: "audit_replay", tokdlsw: "tokenizer_downloads", publicnetsw: "public_network", resumelaunchsw: "resume_on_launch", verboselogsw: "verbose_logs", relaysw: "peer_relay" };
const APPLIES = { gateway_port: "gateway", bootstraps: "both", audit_replay: "gateway", tokenizer_downloads: "gateway", private_tokenizer_downloads: "gateway", public_network: "both", min_authenticity: "gateway", authenticity_checks: "sharing", verbose_logs: "both", peer_relay: "both", engine_kind: "sharing", engine_endpoint: "sharing", engine_autostart: "launch" };
async function saveSetting(patch) {
  // Never save before the first state load: the payload would reset every other setting to defaults.
  if (!state?.settings) { toast("Still loading — try again in a moment"); renderSettings(); return; }
  Object.assign(pendingSettings, patch);
  const settings = { ...(state?.settings || {}), ...pendingSettings, device_name: deviceName };
  try {
    await call("save_settings", { settings });
    const when = new Set(Object.keys(patch).map((k) => APPLIES[k]).filter(Boolean));
    const sharing = !!state?.provider?.status?.running;
    const parts = [];
    if (when.has("gateway") || when.has("both")) parts.push("restart the gateway");
    if ((when.has("sharing") || when.has("both")) && sharing) parts.push("restart sharing");
    toast(when.has("launch") ? "Saved · applies next launch" : parts.length ? `Saved · ${parts.join(" and ")} to apply` : "Saved");
  } catch (e) { failToast("Save failed", e); }
  await refresh().catch(() => {});
  for (const k of Object.keys(patch)) if (pendingSettings[k] === patch[k]) delete pendingSettings[k];
  renderSettings();
}
$$("[data-sw]").forEach((sw) => {
  if (sw.closest("#servetable")) return;
  const fixed = SETTING_SWITCHES[sw.id];
  sw.onclick = () => {
    const on = !sw.classList.contains("on"); sw.classList.toggle("on", on);
    // The token-count switch edits the setting for the saved mode (a private deployment has its own).
    const key = sw.id === "tokdlsw" ? tokenizerDownloadsKey({ ...(state?.settings || {}), ...pendingSettings }) : fixed;
    if (key) saveSetting({ [key]: on });
  };
});
wireSwitches(document); // keyboard and screen reader (L17)
$("#minauthsel")?.addEventListener("change", (e) => saveSetting({ min_authenticity: e.target.value }));
// Text fields: Enter commits single-line fields; blur saves when the value changed.
function wireTextSetting(el, { multiline = false, read, key, valid, skip }) {
  if (!el) return;
  let cancelled = false;
  el.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !multiline) { e.preventDefault(); el.blur(); }
    if (e.key === "Escape") { cancelled = true; el.blur(); }
  });
  el.addEventListener("blur", () => {
    if (cancelled) { cancelled = false; renderSettings(); return; }   // Escape: drop the edit
    if (skip && skip(el)) return;
    const v = read(el);
    if (valid && !valid(v)) { toast("Not saved: " + valid.msg); renderSettings(); return; }
    const cur = key === "device_name" ? deviceName : ({ ...(state?.settings || {}), ...pendingSettings })[key];
    if (JSON.stringify(v) === JSON.stringify(cur)) return;
    if (key === "device_name") { setDeviceName(v); store.set("oh_device", v); saveSessions(); }
    saveSetting({ [key]: v });
  });
}
const portOk = (v) => Number.isInteger(v) && v >= 1024 && v <= 65535; portOk.msg = "the port must be a number from 1024 to 65535";
wireTextSetting($("#gwport"), { key: "gateway_port", read: (el) => parseInt((el.textContent || "").trim(), 10), valid: portOk });
wireTextSetting($("#bootstraps"), { key: "bootstraps", multiline: true, read: (el) => (el.innerText || el.textContent || "").split("\n").map((x) => x.trim()).filter(Boolean) });
wireTextSetting($("#engineendpoint"), { key: "engine_endpoint", skip: (el) => el.contentEditable !== "true", read: engineEndpointValue });
wireTextSetting($('.setpanel[data-p="identity"] [contenteditable]'), { key: "device_name", read: (el) => (el.textContent || "").trim() || deviceName });
// Restart buttons: apply saved settings to the running roles.
$("#restartgw")?.addEventListener("click", async (e) => {
  const b = e.currentTarget; b.disabled = true; b.textContent = "Restarting…";
  try { await call("stop_gateway"); await call("start_gateway"); toast("Gateway restarted"); } catch (err) { toast(`Restart failed: ${err}`); }
  finally { b.disabled = false; b.textContent = "Restart"; await refresh().catch(() => {}); }
});
$("#restartshare")?.addEventListener("click", async (e) => {
  if (!state?.provider?.status?.running) { toast("Not sharing — changes apply when you start sharing"); return; }
  const b = e.currentTarget; b.disabled = true; b.textContent = "Restarting…";
  try { await call("restart_provider"); toast("Sharing restarted"); } catch (err) { failToast("Restart failed", err); }
  finally { b.disabled = false; b.textContent = "Restart"; await refresh().catch(() => {}); }
});
// Reset sharing preferences → clean share-nothing default (hot-reloads; no restart). Non-destructive.
$("#resetsharebtn")?.addEventListener("click", async () => {
  if (!(await confirmModal("Reset sharing preferences?", "Stops announcing every model. Nothing is shared until you choose again.", "Reset"))) return;
  try {
    await call("reset_share_policy");
    emit("share-policy-reset");   // invalidate the Share view's cached policy (works even when stopped)
    toast("Sharing preferences reset — nothing is shared until you choose.");
    await refreshStatus();
    if (activeView === "share") renderView();
  } catch (e) { toast(`Reset failed: ${e}`); }
});
$("#advsw").onclick = () => { const on = !app.hasAttribute("data-adv"); app.toggleAttribute("data-adv", on); $("#advsw").classList.toggle("on", on); store.set("oh_adv", on); if (!on && activeView === "peers") go("providers"); };
// Send-logs export
$("#sendlogsbtn") && ($("#sendlogsbtn").onclick = async () => {
  try {
    const path = await call("export_logs");
    if (!path) { toast("No logs yet"); return; }
    toast("Logs saved — showing the file");
    await call("reveal_path", { path }).catch((e) => toast(`Logs saved: ${path} (${e})`));
  } catch (e) { toast(`Export failed: ${e}`); }
});
// CLI-on-PATH (Layer 1): the Settings "Command-line tool" row. Status comes from the shared cliui
// source, so an install here — or from the Connectors banner — keeps both in sync via "cli-status".
function renderCliRow() {
  const btn = $("#clibtn"), st = $("#clistatus"); if (!btn) return;
  const s = cliState();
  if (!s) { if (st) st.textContent = ""; return; }                                   // not fetched yet
  if (s.error) { btn.textContent = "Install"; if (st) st.textContent = "couldn't check status"; return; }
  if (s.on_path) { btn.textContent = "Reinstall"; if (st) st.textContent = `✓ installed — ${s.resolved || "openhydra"}`; }
  else if (s.managed_broken) { btn.textContent = "Repair"; if (st) st.textContent = "installed but broken — click Repair"; }
  else { btn.textContent = "Install"; if (st) st.textContent = "not installed"; }
}
on("cli-status", renderCliRow);
$("#clibtn") && ($("#clibtn").onclick = async () => {
  const btn = $("#clibtn"); btn.disabled = true; btn.textContent = "Installing…";
  await installCli();   // toasts + refreshes; the "cli-status" handler re-renders the row
  btn.disabled = false;
});
refreshCliStatus();     // initial fetch (broadcasts to both surfaces)

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
// Returns "staged", "found", "none" or "" (nothing checked); `manual` = the user asked (Settings).
async function checkUpdates(manual = false) {
  const u = window.__TAURI__?.updater; if (!u?.check) return "";
  if (updateStaged) return "staged"; // already downloaded + staged this session; waiting on a restart
  // A private deployment doesn't contact OpenHydra's update server unless the user asks (M1, D2);
  // before the settings have loaded we don't know which this is, so look again shortly.
  if (!manual) {
    if (!state?.settings) { setTimeout(() => checkUpdates(), 5000); return ""; }
    if (state.settings.public_network === false) return "";
  }
  try {
    const up = await u.check();
    if (!up) return "none";
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
    return "found";
  } catch (e) {
    // A release that didn't build for this platform has no entry for it: nothing to update to yet.
    if (/found in the response `platforms` object/.test(String(e))) return "none";
    console.warn("update check", e);
    if (manual) throw e;
    return "";
  }
}
setTimeout(() => checkUpdates(), 3000);
setInterval(() => checkUpdates(), 60 * 60 * 1000); // #13: re-check hourly so a running instance stages a new release in the background
$("#checkupd")?.addEventListener("click", async (e) => {
  const b = e.currentTarget; b.disabled = true; b.textContent = "Checking…";
  try {
    const r = await checkUpdates(true);
    toast(r === "none" ? "OpenHydra is up to date" : r === "" ? "Updates aren't available in this build" : "An update is ready: see the button at the top");
  } catch (err) { failToast("Update check failed", err); }
  finally { b.disabled = false; b.textContent = "Check now"; }
});
// Settings › Network: restart what still runs with the old network mode.
$("#netrestartnow")?.addEventListener("click", async (e) => {
  const b = e.currentTarget; b.disabled = true; b.textContent = "Restarting…";
  try {
    if (state?.gateway?.status?.running) { await call("stop_gateway"); await call("start_gateway"); }
    if (state?.provider?.status?.running) await call("restart_provider");
    toast("Restarted with the new network settings");
  } catch (err) { failToast("Restart failed", err); }
  finally { b.disabled = false; b.textContent = "Restart now"; await refresh().catch(() => {}); renderSettings(); }
});
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
  let st;
  try { st = await call("get_state"); setState(st); } catch {}
  // One-shot: the share policy was corrupt and self-healed to a safe default (backend read-and-clear).
  if (st?.share_policy_reset) toast("Your sharing preferences couldn't be read and were reset — pick which models to share, or use “Share everything”.");
  // Informed opt-out: while this launch's auto-resume is flagged, keep a non-blocking notice up.
  if (state?.resumed_on_launch) ensureResumeNotice();
  renderStatusbar();
  if (["share", "settings", "connectors", "engines", "activity", "ledger"].includes(activeView)) renderView();
  if (activeView === "peers") renderLogs();
}

// Non-blocking "Resuming sharing of N models…" banner (informed opt-out). RE-ASSERTED each refresh so a
// first-run / after-update coachmark DOM rebuild can't kill it (that's exactly the update case where
// this matters). Clears on "Don't resume", ×, or after a short show-window.
let resumeNoticeDismissed = false, resumeNoticeDeadline = 0, resumeLabel = null;
// B1: name the count in the resume toast — "all models" for share-all, "N model(s)" for a list.
function shareCountLabel(pol) {
  if (!pol || !pol.mode) return null;
  if (pol.mode === "all") return "all models";
  const n = (pol.models || []).length;
  if (n === 0) return null;   // share-nothing → fall back to generic text (no "Resuming sharing of 0 models")
  return `${n} model${n === 1 ? "" : "s"}`;
}
function resumeText() { return resumeLabel ? `Resuming sharing of ${resumeLabel}…` : "Resuming your shared models…"; }
function ensureResumeNotice() {
  if (resumeNoticeDismissed) return;
  if (!resumeNoticeDeadline) resumeNoticeDeadline = Date.now() + 18000;
  if (Date.now() > resumeNoticeDeadline) { resumeNoticeDismissed = true; document.getElementById("resumenotice")?.remove(); return; }
  // B1: fetch the resumed policy once to name the count; update the banner in place when it lands.
  if (resumeLabel === null) {
    resumeLabel = "";
    call("read_share_policy").then((r) => {
      resumeLabel = shareCountLabel(r) || "";
      const s = document.querySelector("#resumenotice .rn-text");
      if (s) s.textContent = resumeText();
    }).catch(() => {});
  }
  if (document.getElementById("resumenotice")) return;
  const el = document.createElement("div");
  el.id = "resumenotice"; el.className = "resumenotice";
  setHTML(el, html`<span class="rn-dot">●</span><span class="rn-text">${resumeText()}</span><button class="btn ghost sm rn-stop">Don't resume</button><button class="rn-x" title="Dismiss">×</button>`);
  document.body.appendChild(el);
  const dismiss = () => { resumeNoticeDismissed = true; el.remove(); };
  el.querySelector(".rn-stop").onclick = async () => { dismiss(); try { await setSharing(false); toast("Sharing stopped"); } catch (e) { toast("Couldn't stop: " + e); } };
  el.querySelector(".rn-x").onclick = dismiss;
}
async function refreshEngines() { try { setEngines(await call("detect_engines_now")); } catch { setEngines([]); } try { setInstalledEngines(await call("installed_engines")); } catch { setInstalledEngines([]); } renderStatusbar(); if (["share", "engines", "providers", "settings"].includes(activeView)) renderView(); }
on("refresh-engines", refreshEngines);   // installer.js fires this after an install completes
on("nav", (v) => go(v));                  // feature modules request navigation via the bus
// The tray's Sharing item asks here when the user hasn't chosen what to share yet.
try { window.__TAURI__?.event?.listen?.("open-share-dialog", () => { go("share"); openShareDialog(); }); } catch {}
on("refresh", () => refresh());           // …and a full state refresh
async function refreshStatus() { try { setSnap(await call("status_snapshot")); } catch { setSnap(null); } noteSeen(); accumulateStats(); renderStatusbar(); if (["peers", "providers", "activity", "ledger", "share"].includes(activeView)) renderView(); }
on("refresh-status", () => refreshStatus());  // share.js nudges the /status/share poll after a policy save
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
  wireCardDeepLinks();    // M2.1: route a `.openhydra` file / `openhydra:` link into the import flow
})();
setInterval(refresh, 2500);
setInterval(refreshEngines, 10000);
setInterval(refreshStatus, 2500);
