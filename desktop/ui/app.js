// OpenHydra Desktop — the shadcn wireframe (docs/openhydra_wireframe_shadcn.html) wired to the
// Rust backend via Tauri IPC. The DOM + CSS are the wireframe verbatim; this file swaps the
// wireframe's demo data for live network state. In a plain browser it renders a demo mock.
import { $, $$, esc, shortPeer, peerShort } from "./dom";
import { injectIcons } from "./icons";
import { store } from "./storage";
import { modelIcon, modelCat, fmtUptime, repBadge, fmtNum, modelColor, relTime, fmtGB } from "./format";
import { hl, parseFences, splitThink, metaRow, mediaKind, mediaEl } from "./text";
import { call, mockEmitInstall, mockInstallCbs } from "./bridge";
import { toast, menu, closeMenus } from "./chrome";
import { accumulateStats, loadStats, totalServed, totalUsed, statsSeries, lifetimeServed, statModels } from "./stats";
import { repByOpenhydra, repByLibp2p, modelReputation, modelAvgTps } from "./econ";
import { noteSeen, modelIdle, netModels, curModel, renderModels, seenCount } from "./models";
import { renderConnectors } from "./connectors";
import {
  state, snap, engines, installedEngines, sessions, sessionOrder, curChat, activeView, deviceName, usedTokens,
  setState, setSnap, setEngines, setInstalledEngines, setSessions, setSessionOrder, setCurChat, setActiveView, setDeviceName, setUsedTokens,
} from "./state";

(function () {
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
  const rttSamples = store.get("oh_rtt", []), tpsSamples = store.get("oh_tps", []);
  const ROLL_MAX = 30;
  function pushSample(arr, v, key) { if (v == null || !isFinite(v)) return; arr.push(v); while (arr.length > ROLL_MAX) arr.shift(); store.set(key, arr); }
  const mean = (a) => a.length ? a.reduce((x, y) => x + y, 0) / a.length : null;



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
  let hideInactive = store.get("oh_hideinactive", false);
  // #10: selected timeline range per chart host (persisted).
  const chartRange = store.get("oh_chartrange", { share: "7d", activity: "7d" });
  // Stable per-model hue for the timeline + legend.
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
  // Start an installed-but-idle engine's server, then refresh so the card flips to "running".
  async function runEngine(label, name) {
    toast(`Starting ${name}…`);
    try { await call("run_engine", { engine: label }); toast(`${name} started`); }
    catch (e) { toast(String(e)); }
    await refreshEngines();
  }
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
})();
