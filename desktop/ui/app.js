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
  function mock(cmd) {
    if (cmd === "start_provider") { mk.provider = true; return null; }
    if (cmd === "stop_provider") { mk.provider = false; return null; }
    if (cmd === "start_gateway") { mk.gateway = true; return null; }
    if (cmd === "stop_gateway") { mk.gateway = false; return null; }
    if (cmd === "save_settings") return null;
    if (cmd === "gateway_health") return mk.gateway;
    if (cmd === "get_state") return {
      provider: { status: { running: mk.provider, pid: 42, peer_id: "12D3KooWQvXm4cAsusDEuXRH", engines: "ollama", announced: mk.provider ? 2 : 0, relays: 2, exited: null }, logs: ["node up", "announced tinyllama:latest", "announced llama3.2:1b"] },
      gateway: { status: { running: mk.gateway, pid: 43, peer_id: "12D3KooWQvXm4cAsusDEuXRH", engines: null, announced: null, relays: 2, exited: null }, logs: ["gateway listening 127.0.0.1:8080"] },
      settings: { bootstraps: ["/dns4/bootstrap-us.openhydra.co/tcp/4001"], gateway_port: 8080, engine_autostart: true, search_url: "" },
      agent_found: true, gateway_url: "http://127.0.0.1:8080/v1",
    };
    if (cmd === "detect_engines_now") return [{ label: "ollama", url: "http://127.0.0.1:11434", models: ["tinyllama:latest", "llama3.2:1b"] }];
    if (cmd === "status_snapshot") {
      if (!mk.provider && !mk.gateway) return null;
      return {
        libp2p_peer_id: "12D3KooWQvXm4cAsusDEuXRH",
        uptime_secs: 8123,
        network: {
          nat: { nat_type: "cone", is_public: false }, autonat_private: true, ipv6_capable: true, kad_server_mode: false, kad_routing_peers: 12, network_generation: 0,
          listen_addrs: ["/ip4/0.0.0.0/udp/4111/quic-v1"], external_addrs: ["/ip4/49.36.1.2/udp/4111/quic-v1"], relay_reservations: ["/ip4/45.79.190.172/tcp/4001/p2p/12D3KooWEL/p2p-circuit"],
          peers: [
            { peer_id: "12D3KooWM2qsVg5WbR6Asusn2XN7", quic_direct_v4: 1, quic_direct_v6: 0, tcp_direct: 0, tcp_relay: 0, failure_streak: 0, path: "direct" },
            { peer_id: "12D3KooWEzegXr4qcjEW3WT", quic_direct_v4: 0, quic_direct_v6: 0, tcp_direct: 0, tcp_relay: 1, failure_streak: 0, path: "relay" },
          ],
          known_models: ["tinyllama:latest", "llama3:latest", "qwen2.5:7b"],
          known_providers: [
            { model_id: "tinyllama:latest", openhydra_peer_id: "oh_asus", libp2p_peer_id: "12D3KooWM2qsVg5WbR6Asusn2XN7" },
            { model_id: "qwen2.5:7b", openhydra_peer_id: "oh_gpu3", libp2p_peer_id: "12D3KooWEzegXr4qcjEW3WT" },
          ],
          counters: { dcutr_successes: 1, dcutr_failures: 0, reversal_dials: 0, reversal_successes: 0, tier_connect_success: { direct_quic_v4: 2, relay: 1 } },
        },
        transfers: { requests_served: 3, tokens_served: 1280, serve_errors: 0, aup_refusals: 0, receipts_ledgered: 3, per_model: { "tinyllama:latest": { requests: 3, tokens: 1280, avg_native_tps: 73.2 } } },
        economy: {
          role: "consumer",
          reputation: [{ openhydra_peer_id: "oh_asus", score: 72.5 }, { openhydra_peer_id: "oh_gpu3", score: 58.0 }],
          credit: [{ libp2p_peer_id: "12D3KooWM2qsVg5WbR6Asusn2XN7", balance: 5123.4, rate_cap: 1.0 }, { libp2p_peer_id: "12D3KooWEzegXr4qcjEW3WT", balance: 4880.1 }],
          avg_reputation: 65.3, total_credit: 10003.5,
        },
      };
    }
    if (cmd === "chat_completion") return new Promise((r) => setTimeout(() => r({
      choices: [{ message: { content: "The herd answers: reciprocity beats rent-seeking.\n\n```rust\nfn hi(){ println!(\"herd\"); }\n```" } }],
      usage: { completion_tokens: 42 }, openhydra: { engine: { native_tps: 61 }, hops_ms: { network_rtt: 12 } }, model: "tinyllama:latest",
    }), 500));
    if (cmd === "web_search") return [];
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
  let sessions = store.get("oh_sessions", {}), sessionOrder = store.get("oh_order", []);
  let deviceName = store.get("oh_device", /Mac/.test(navigator.platform) ? "MacBook Pro" : "This machine");
  let usedTokens = store.get("oh_used", 0);
  root.dataset.theme = store.get("oh_theme", "light");
  if (store.get("oh_adv", false)) app.setAttribute("data-adv", "");
  function saveSessions() { store.set("oh_sessions", sessions); store.set("oh_order", sessionOrder); }

  // ── live state ──
  let state = null, engines = [], snap = null, activeView = "home", curChat = null, attachments = [];
  // Rolling per-chat telemetry the agent only emits per-request (there's no aggregated RTT on
  // the status API) — we average the last N replies client-side for the Activity view.
  const rttSamples = store.get("oh_rtt", []), tpsSamples = store.get("oh_tps", []);
  const ROLL_MAX = 30;
  function pushSample(arr, v, key) { if (v == null || !isFinite(v)) return; arr.push(v); while (arr.length > ROLL_MAX) arr.shift(); store.set(key, arr); }
  const mean = (a) => a.length ? a.reduce((x, y) => x + y, 0) / a.length : null;

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
  const titles = { home: "Home", chat: "Chat", activity: "Activity", connectors: "Connectors", providers: "Providers", share: "Share", engines: "Engines", ledger: "Ledger", peers: "Diagnostics and Stats", settings: "Settings" };
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
    menu(d, opts.length ? opts.map((o) => ({ label: o, on: o === cur, fn: () => { d.querySelector("span").textContent = o; if (d.dataset.act === "theme") setTheme(o); if (d.id === "modeldrop" || d.id === "homedrop") { setChatMode(o); const other = $(d.id === "modeldrop" ? "#homedrop" : "#modeldrop"); if (other) other.querySelector("span").textContent = o; } } })) : [{ label: "No models on the network yet", fn: () => {} }]);
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
  function openChat(id) { curChat = id; renderChat(); go("chat"); renderRecents(); }

  // ── chat: model list = network-routable models (fixes the 504: can't pick an unservable model) ──
  // DHT provider records expire (~300s TTL) and re-propagate on their own schedule, so the raw
  // known_models snapshot flickers. Keep a model "sticky" for STICKY_MS after we last saw it so
  // the Providers list and model picker stay calm instead of blinking rows in and out.
  const STICKY_MS = 90000;
  const seenModels = {};   // model -> last-seen ms
  const seenCount = {};    // model -> last-known provider count
  function noteSeen() {
    const now = Date.now();
    (snap?.network?.known_models || []).forEach((m) => seenModels[m] = now);
    const provs = snap?.network?.known_providers || [];
    const c = {}; for (const p of provs) c[p.model_id] = (c[p.model_id] || 0) + 1;
    for (const m in c) seenCount[m] = c[m];
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

  // reply rendering: fenced code cards + ComfyUI images + metadata line
  const IMG_SRC = /^(data:image\/[a-z.+-]+;base64,[A-Za-z0-9+/=]+|https?:\/\/\S+\.(?:png|jpe?g|gif|webp))$/i;
  const KW = new Set("fn let mut pub use impl struct enum trait match if else for while loop return async await move def class import from as with try except lambda yield pass raise function const var new typeof export default package func go defer chan interface map range type switch case break continue static void int float double char bool".split(" "));
  function hl(code) { let out = "", i = 0; const push = (c, s) => out += c ? `<span class="${c}">${esc(s)}</span>` : esc(s); while (i < code.length) { const rest = code.slice(i); let m; if ((m = rest.match(/^(\/\/|#(?!\[)|--)[^\n]*/))) push("tk-com", m[0]); else if ((m = rest.match(/^\/\*[\s\S]*?\*\//))) push("tk-com", m[0]); else if ((m = rest.match(/^("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|`(?:[^`\\]|\\.)*`)/))) push("tk-str", m[0]); else if ((m = rest.match(/^\b\d[\d_]*(\.\d+)?\b/))) push("tk-num", m[0]); else if ((m = rest.match(/^[A-Za-z_]\w*/))) { if (KW.has(m[0])) push("tk-kw", m[0]); else if (code[i + m[0].length] === "(") push("tk-fn", m[0]); else push(null, m[0]); } else { push(null, code[i]); i++; continue; } i += m[0].length; } return out; }
  function parseFences(t) { const p = [], re = /```([\w+-]*)\n([\s\S]*?)```/g; let last = 0, m; while ((m = re.exec(t))) { if (m.index > last) p.push({ prose: t.slice(last, m.index).trim() }); p.push({ lang: m[1] || "code", code: m[2] }); last = re.lastIndex; } if (last < t.length) p.push({ prose: t.slice(last).trim() }); return p.filter((x) => x.code != null || x.prose); }
  function metaRow(m) { return `<div class="msgmeta"><span>${esc(m.model)}</span><span class="num">${m.tok} tok</span><span class="num" title="End-to-end throughput — engine TPS with network latency">${m.tps} t/s herd</span><span class="num">${m.rtt} ms RTT</span><span>${m.at}</span></div>`; }
  function botEl(content, meta) {
    const d = document.createElement("div"); d.className = "msg ai";
    for (const part of parseFences(content)) {
      if (part.code != null) { const c = document.createElement("div"); c.className = "code-card"; c.innerHTML = `<div class="code-card-head"><span>${esc(part.lang)}</span><button class="code-copy">⎘ copy</button></div><pre>${hl(part.code)}</pre>`; c.querySelector(".code-copy").onclick = (ev) => { navigator.clipboard?.writeText(part.code); ev.target.textContent = "✓ copied"; setTimeout(() => ev.target.textContent = "⎘ copy", 1400); }; d.appendChild(c); }
      else { const pr = document.createElement("div"); pr.style.whiteSpace = "pre-wrap"; const re = /!\[[^\]]*\]\(([^)]+)\)/g; let last = 0, m; while ((m = re.exec(part.prose))) { if (m.index > last) pr.appendChild(document.createTextNode(part.prose.slice(last, m.index))); const src = m[1].trim(); if (IMG_SRC.test(src)) { const img = document.createElement("img"); img.className = "micon-img"; img.style.cssText = "max-width:100%;border-radius:10px;border:1px solid hsl(var(--border));margin:6px 0;display:block;width:auto;height:auto"; img.src = src; pr.appendChild(img); } else pr.appendChild(document.createTextNode(m[0])); last = re.lastIndex; } if (last < part.prose.length) pr.appendChild(document.createTextNode(part.prose.slice(last))); d.appendChild(pr); }
    }
    if (meta) { const mm = document.createElement("div"); mm.innerHTML = metaRow(meta); d.appendChild(mm.firstChild); }
    return d;
  }
  function renderChat() {
    const s = sessions[curChat]; const th = $("#thread"); if (!s) { th.innerHTML = ""; return; }
    $("#chattitle").textContent = s.t; th.innerHTML = "";
    if (!s.m.length) { th.innerHTML = `<div class="mut" style="margin:auto;text-align:center;font-size:12.5px">Ask anything — requests route through the network to whichever provider serves the model.</div>`; return; }
    for (const x of s.m) { if (x[0] === "me") { const d = document.createElement("div"); d.className = "msg me"; d.textContent = x[1]; th.appendChild(d); } else th.appendChild(botEl(x[1], x[2])); }
    th.scrollTop = th.scrollHeight;
  }

  // ── sharing = provider role; announce switches reflect + drive it ──
  let sharingBusy = false;
  async function toggleSharing() {
    if (sharingBusy) return; sharingBusy = true;
    const running = state?.provider?.status?.running;
    try {
      if (running) await call("stop_provider");
      else { if (!engines.some((e) => e.models.length)) { go("engines"); toast("No local models found — start or install an engine to share"); sharingBusy = false; return; } await call("start_provider"); }
    } catch (e) { toast(`${running ? "stop" : "start"} failed: ${e}`); }
    await refresh(); sharingBusy = false;
  }
  $("#sharingsw").onclick = toggleSharing;

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
      const resp = await call("chat_completion", { model, messages, maxTokens: +($("[data-label='tokk']")?.dataset.val) || 1024 });
      wait.remove();
      const reply = resp?.choices?.[0]?.message?.content ?? "(empty reply)";
      const oh = resp?.openhydra || {};
      const meta = { model: resp?.model || model, tok: resp?.usage?.completion_tokens ?? "—", tps: oh.engine?.native_tps ? Math.round(oh.engine.native_tps) : "—", rtt: oh.hops_ms?.network_rtt ?? "—", at: new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }) };
      pushSample(tpsSamples, oh.engine?.native_tps, "oh_tps");   // rolling throughput for Activity
      pushSample(rttSamples, oh.hops_ms?.network_rtt, "oh_rtt"); // rolling latency for Activity
      s.m.push(["ai", reply, meta]); saveSessions(); renderChat();
      if (resp?.usage?.completion_tokens) { usedTokens += resp.usage.completion_tokens; store.set("oh_used", usedTokens); renderStatusbar(); }
    } catch (e) {
      wait.remove(); const secs = Math.round((Date.now() - t0) / 1000);
      const err = document.createElement("div"); err.className = "msg ai"; err.style.color = "hsl(var(--danger))"; err.style.fontSize = "12.5px";
      err.textContent = /504|timeout|timed out/i.test(String(e)) ? `The provider didn't respond in time (${secs}s). Cold model loads can be slow — try again; it warms up.` : `${e}`;
      th.appendChild(err);
    }
  }
  function send() { const c = $("#composer"); const t = c.textContent.trim(); c.textContent = ""; doSend(t, false); }
  $("#send").onclick = send;
  $("#composer").onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } };
  function homeSend() { const p = $("#homeprompt"); const t = p.textContent.trim(); p.textContent = ""; doSend(t, true); }
  $("#homesend").onclick = homeSend;
  $("#homeprompt").onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); homeSend(); } };

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
  function renderShare() {
    const p = state?.provider?.status, running = !!p?.running, t = snap?.transfers;
    const head = $("#v-share .row .ctitle").parentElement;
    head.querySelector(".badge").innerHTML = running ? '<span class="dot ok"></span>provider running' : '<span class="dot"></span>not sharing';
    head.querySelector(".badge").className = "badge " + (running ? "ok" : "secondary"); head.querySelector(".badge").style.marginLeft = "8px";
    const up = snap?.uptime_secs != null ? ` · up ${fmtUptime(snap.uptime_secs)}` : "";
    head.querySelector(".mut").textContent = `${running ? (p.announced ?? 0) : 0} models announced · gateway :8080${up}`;
    const k = $$("#v-share .g4 .kpi .val");
    const served = t?.tokens_served ?? 0, credit = econ().total_credit;
    const ratio = usedTokens > 0 ? (served / usedTokens) : (served > 0 ? null : 0);
    k[0].textContent = served;
    k[1].textContent = credit != null ? Math.round(credit).toLocaleString() : "—";
    k[2].textContent = ratio == null ? "∞" : ratio ? ratio.toFixed(1) + "×" : "—";
    k[3].textContent = "—";   // own reputation is held by the peers we serve — not locally knowable
    $$("#v-share .g4 .kpi .sub")[0].innerHTML = `${t?.requests_served ?? 0} requests served`;
    $$("#v-share .g4 .kpi .sub")[1].textContent = credit != null ? "give-to-get credit (not money)" : "starts once you serve a peer";
    $$("#v-share .g4 .kpi .sub")[2].textContent = "served ÷ used";
    $$("#v-share .g4 .kpi .sub")[3].innerHTML = `<span class="mut">earned on the peers you serve</span>`;
    const per = t?.per_model || {}, rows = [];
    for (const e of engines) for (const m of e.models) { const pm = per[m] || {}; rows.push(`<tr><td>${modelIcon(m)}<b>${esc(m)}</b></td><td>${esc(e.label)}</td><td class="num">${pm.requests ?? "—"}</td><td class="num">${pm.tokens ?? "—"}</td><td class="num">${pm.avg_native_tps ? Math.round(pm.avg_native_tps) : "—"}</td><td><span class="badge ${running ? "ok" : "secondary"}">${running ? "live" : "ready"}</span></td><td><div class="switch ${running ? "on" : ""}" data-announce></div></td></tr>`); }
    $("#servetable tbody").innerHTML = rows.join("") || `<tr><td colspan="7" class="mut">No engines answering — start Ollama, LM Studio, vLLM, llama.cpp, or Exo, then rescan.</td></tr>`;
    $("#v-share .badge.ok, #v-share .badge").parentElement && ($$("#v-share .card .row .badge")[0]);
    const ann = $("#v-share .card .row .badge.ok") || $("#servetable").closest(".card").querySelector(".badge");
    if (ann) { ann.textContent = `${running ? engines.reduce((n, e) => n + e.models.length, 0) : 0} announced`; ann.className = "badge " + (running && engines.length ? "ok" : "secondary"); }
    $$("#servetable [data-announce]").forEach((sw) => sw.onclick = toggleSharing);
    // incoming strip
    const inflight = t?.requests_served != null ? "" : "";
    const inc = $("#v-share .card.pad .mut");
    if (inc) inc.textContent = running ? `Serving live · ${t?.requests_served ?? 0} requests, ${t?.tokens_served ?? 0} tokens so far` : "Turn on Sharing (title bar) to start serving.";
    // wanted table — honest empty state until demand telemetry lands
    $("#wanttable tbody").innerHTML = `<tr><td colspan="4" class="mut">Fills in as network demand telemetry lands.</td></tr>`;
  }
  function renderProviders() {
    const models = netModels();
    const local = new Set(state?.provider?.status?.running ? engines.flatMap((e) => e.models) : []);
    const strip = $("#v-providers .card.pad");
    strip.querySelectorAll("b")[0].textContent = (snap?.transfers?.tokens_served ?? 0);
    strip.querySelectorAll("b")[1].textContent = models.length;
    strip.querySelectorAll("b")[2].textContent = snap?.network?.peers?.length ?? 0;
    strip.querySelector(".mut").textContent = "from your node's view of the network";
    $("#provcount").textContent = models.length;
    const q = ($("#search").value || "").toLowerCase();
    const byOh = repByOpenhydra();
    const rows = models.filter((m) => !q || m.toLowerCase().includes(q)).map((m) => {
      const cnt = (seenCount[m] || 0) + (local.has(m) ? 1 : 0);   // last-known count, sticky
      const tps = modelAvgTps(m);                                  // only for models we serve
      const rep = modelReputation(m, byOh);                        // earned rep of its providers
      return `<tr class="prov" data-cat="${modelCat(m)}"><td>${modelIcon(m)}<b>${esc(m)}</b>${local.has(m) ? ' <span class="mut">· your machine</span>' : ""}</td><td class="num">${cnt || "—"}</td><td class="num${tps == null ? " mut" : ""}">${tps == null ? "—" : tps}</td><td class="num mut">—</td><td>${repBadge(rep)}</td><td class="num mut">—</td></tr>`;
    });
    $("#provtable tbody").innerHTML = rows.join("") || `<tr><td colspan="6" class="mut">${snap ? "No models discovered yet — they appear as peers announce." : "Connecting…"}</td></tr>`;
    const cat = $("#provchips .chip.on")?.dataset.cat || "all";
    $$("#provtable .prov").forEach((r) => r.style.display = (cat === "all" || r.dataset.cat === cat) ? "" : "none");
  }
  const ENGINES = [["Ollama", "ollama", "General-purpose local LLMs."], ["LM Studio", "lm-studio", "MLX-optimised models on Apple silicon."], ["llama.cpp", "llama-cpp", "Lightweight GGUF runtime."], ["ComfyUI", "comfyui", "Image generation — Stable Diffusion, Flux."], ["vLLM", "vllm", "High-throughput serving. Needs Python + GPU."], ["Exo", "exo", "Shard big models across your devices."]];
  function renderEngines() {
    const det = Object.fromEntries(engines.map((e) => [e.label, e]));
    const cards = $$("#v-engines .grid.g3 .card");
    ENGINES.forEach(([name, label, desc], i) => {
      const c = cards[i]; if (!c) return; const d = det[label], guided = ["comfyui", "vllm", "exo"].includes(label);
      c.querySelector("b").textContent = name;
      const badge = c.querySelector(".badge"); badge.className = "badge " + (d ? "ok" : ""); badge.style.marginLeft = "auto"; badge.innerHTML = d ? '<span class="dot ok"></span>running' : "not installed";
      c.querySelectorAll(".mut")[0].textContent = desc;
      const foot = c.querySelectorAll(".row")[1]; const btn = foot.querySelector(".enginst"); const def = foot.querySelector(".badge");
      if (def) def.style.display = d && label === "ollama" ? "" : "none";
      btn.textContent = d ? "Manage" : guided ? "Guided install" : "Install"; btn.className = "btn " + (d || guided ? "outline" : "brand") + " sm enginst"; btn.style.marginLeft = "auto";
    });
    $$("#v-engines .enginst").forEach((b) => b.onclick = () => toast("One-click installs land with the engine store — start or install it manually for now"));
    // recommended: honest until a model store lands
    $("#rectable tbody").innerHTML = `<tr><td colspan="5" class="mut">One-click downloads land with the model store — for now, pull with your engine (e.g. <span class="mono">ollama pull</span>) and it appears in Share.</td></tr>`;
  }
  function renderActivity() {
    const t = snap?.transfers, k = $$("#v-activity .g4 .kpi .val");
    const served = t?.tokens_served ?? 0, credit = econ().total_credit;
    const ratio = usedTokens > 0 ? (served / usedTokens) : (served > 0 ? null : 0);
    k[0].textContent = served;
    k[1].textContent = usedTokens;
    k[2].textContent = credit != null ? (credit >= 0 ? "+" : "") + Math.round(credit).toLocaleString() : "—";
    k[3].textContent = ratio == null ? "∞" : ratio ? ratio.toFixed(1) + "×" : "—";
    $$("#v-activity .g4 .kpi .sub")[0].innerHTML = `<span class="dot ok"></span>${t?.receipts_ledgered ?? 0} receipts co-signed`;
    $$("#v-activity .g4 .kpi .sub")[1].textContent = "this device";
    $$("#v-activity .g4 .kpi .sub")[2].textContent = credit != null ? "give-to-get credit standing" : "credits · once you transact";
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
  }
  function renderLedger() {
    const t = snap?.transfers;
    $("#v-ledger .row .mut").textContent = `${t?.receipts_ledgered ?? 0} receipts · ${t?.tokens_served ?? 0} tokens served`;
    $("#ledgertable tbody").innerHTML = `<tr><td colspan="6" class="mut">Receipts are co-signed and stored by the agent — the per-transaction view lands with the credit ledger.</td></tr>`;
  }
  function renderConnectors() {
    $("#v-connectors .cp") && $$("#v-connectors .cp").forEach((b) => b.onclick = (e) => { e.stopPropagation(); navigator.clipboard?.writeText(b.parentElement.textContent.replace(/Copy$/, "").trim()); toast("Copied"); });
  }
  function renderPeers() {
    if (!snap) { $("#peertable tbody").innerHTML = `<tr><td colspan="6" class="mut">Turn on Sharing or chat to connect, then peers appear here.</td></tr>`; return; }
    const n = snap.network;
    const repL = repByLibp2p();
    $("#peertable tbody").innerHTML = n.peers.length ? n.peers.map((p) => `<tr data-p="${p.path}"><td class="mono">${peerShort(p.peer_id)}</td><td><span class="badge ${p.path === "direct" ? "ok" : p.path === "relay" ? "warn" : "secondary"}">${p.path}</span></td><td class="num">${p.quic_direct_v6}</td><td class="num">${p.failure_streak > 0 ? "—" : "·"}</td><td>${repBadge(repL[p.peer_id])}</td><td class="rowmenu mut"><span class="icon" data-i="more"></span></td></tr>`).join("") : `<tr><td colspan="6" class="mut">No peers connected yet — dialing bootstraps.</td></tr>`;
    injectIcons($("#peertable"));
    $("#actchips .chip .num") && ($("#actchips .chip .num").textContent = n.peers.length);
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
    const id = $('.setpanel[data-p="identity"]'); id.querySelector('[contenteditable]').textContent = deviceName;
    id.querySelectorAll(".input")[1].childNodes[0].textContent = (p.provider.status.peer_id || p.gateway.status.peer_id || "—");
    $('.setpanel[data-p="network"] .input.mono').textContent = `127.0.0.1:${p.settings.gateway_port}`;
    const eng = $('.setpanel[data-p="engine"]'); eng.querySelectorAll(".input.mono")[0].textContent = engines[0]?.url || "http://127.0.0.1:11434";
    eng.querySelector('.switch').classList.toggle("on", !!p.settings.engine_autostart);
    $("#advsw").classList.toggle("on", app.hasAttribute("data-adv"));
  }

  // ── status bar + lifecycle ──
  function renderStatusbar() {
    const p = state?.provider?.status, g = state?.gateway?.status, peers = snap?.network?.peers?.length ?? 0;
    const anyOn = !!(p?.running || g?.running);
    let dot = "warn pulse", label = "Initializing…";
    if (state) { if (!anyOn) { dot = ""; label = "Ready — connecting…"; } else if (peers > 0) { dot = "ok pulse"; label = "Connected"; } else { dot = "warn pulse"; label = "Connecting to network…"; } }
    $("#netdot").className = "dot " + dot; $("#netlabel").textContent = label;
    $("#sbpeers").textContent = `${peers} peers`;
    $("#sbserved").textContent = snap?.transfers?.tokens_served ?? 0; $("#sbused").textContent = usedTokens;
    $("#apiendpoint").textContent = (state?.gateway_url || "http://127.0.0.1:8080/v1").replace(/^https?:\/\//, "") + (g?.running ? "" : " · off");
    $("#sharingsw").classList.toggle("on", !!p?.running);
    $("#sidepeer").textContent = `${deviceName} · ${shortPeer(p?.peer_id || g?.peer_id)}`;
    renderModels();
    const connected = anyOn && (peers > 0 || netModels().length > 0);
    $("#homeconnecting").style.display = connected ? "none" : "inline-flex";
    $("#homeready").style.display = connected ? "" : "none";
  }
  $("#apiendpoint").onclick = async () => { if (!state?.gateway?.status?.running) { await ensureGateway(); toast("Local API started"); return; } navigator.clipboard?.writeText(state?.gateway_url || "http://127.0.0.1:8080/v1"); toast("Copied"); };

  // ── chips ──
  $("#provchips").onclick = (e) => { const c = e.target.closest(".chip[data-cat]"); if (!c) return; $$("#provchips .chip[data-cat]").forEach((x) => x.classList.toggle("on", x === c)); const cat = c.dataset.cat; $$("#provtable .prov").forEach((r) => r.style.display = (cat === "all" || r.dataset.cat === cat) ? "" : "none"); };
  $("#actchips").onclick = (e) => { const c = e.target.closest(".chip[data-act]"); if (!c) return; $$("#actchips .chip").forEach((x) => x.classList.toggle("on", x === c)); $$("#v-peers .acttab").forEach((t) => t.classList.toggle("on", t.dataset.act === c.dataset.act)); };
  $("#peerchips").onclick = (e) => { const c = e.target.closest(".chip"); if (!c) return; $$("#peerchips .chip").forEach((x) => x.classList.toggle("on", x === c)); const pp = c.dataset.p; $$("#peertable tbody tr").forEach((r) => r.style.display = (pp === "all" || r.dataset.p === pp) ? "" : "none"); };
  $("#logchips").onclick = (e) => { const c = e.target.closest(".chip"); if (!c) return; $$("#logchips .chip").forEach((x) => x.classList.toggle("on", x === c)); logTab = c.dataset.log === "gateway" ? "gateway" : "provider"; renderLogs(); };
  $("#search").oninput = () => { if (activeView === "providers") renderProviders(); else if (activeView === "peers") { const q = $("#search").value.toLowerCase(); $$("#peertable tbody tr").forEach((r) => r.style.display = r.textContent.toLowerCase().includes(q) ? "" : "none"); } };
  $("#cmdk").onclick = (e) => { e.stopPropagation(); menu($("#cmdk"), Object.keys(titles).map((v) => ({ label: "Go to " + titles[v], on: v === activeView, fn: () => go(v) }))); };
  $("#traymark").onclick = (e) => { e.stopPropagation(); menu($("#traymark"), [{ label: "Launch OpenHydra", fn: () => {} }, { sep: 1 }, { label: "Sharing", on: !!state?.provider?.status?.running, fn: toggleSharing }, { label: "Model · " + (netModels()[0] || "—"), fn: () => {} }, { sep: 1 }, { label: `▲ ${snap?.transfers?.tokens_served ?? 0} served`, fn: () => {} }, { label: `▼ ${usedTokens} used`, fn: () => {} }, { sep: 1 }, { label: "Quit OpenHydra", fn: () => call("quit") }]); };
  $("#addmodel") && ($("#addmodel").onclick = (e) => { e.stopPropagation(); const opts = engines.flatMap((e) => e.models.map((m) => ({ label: `${m} · ${e.label}`, fn: () => toggleSharing() }))); menu($("#addmodel"), opts.length ? opts : [{ label: "No engine models — start an engine", fn: () => go("engines") }]); });

  // ── settings ──
  $("#setnav").onclick = (e) => { const s = e.target.closest(".s"); if (!s) return; $$("#setnav .s").forEach((x) => x.classList.toggle("on", x === s)); $$(".setpanel").forEach((pnl) => pnl.classList.toggle("on", pnl.dataset.p === s.dataset.p)); };
  $$("[data-sw]").forEach((sw) => { if (sw.closest("#servetable")) return; sw.onclick = () => sw.classList.toggle("on"); });
  $$(".save").forEach((b) => b.onclick = async () => {
    deviceName = ($('.setpanel[data-p="identity"] [contenteditable]').textContent || deviceName).trim(); store.set("oh_device", deviceName);
    const settings = { bootstraps: state?.settings?.bootstraps || [], gateway_port: state?.settings?.gateway_port || 8080, engine_autostart: $('.setpanel[data-p="engine"] .switch').classList.contains("on"), search_url: state?.settings?.search_url || "" };
    try { await call("save_settings", { settings }); toast("Settings saved"); await refresh(); } catch (e) { toast(`Save failed: ${e}`); }
  });
  $('.setpanel[data-p="identity"] .cp')?.addEventListener("click", () => { navigator.clipboard?.writeText($('.setpanel[data-p="identity"] .input.mono').textContent.replace("Copy", "").trim()); toast("Peer ID copied"); });
  $("#advsw").onclick = () => { const on = !app.hasAttribute("data-adv"); app.toggleAttribute("data-adv", on); $("#advsw").classList.toggle("on", on); store.set("oh_adv", on); if (!on && activeView === "peers") go("providers"); };

  // ── connectors copy (wireframe .cp) ──
  $$(".cp").forEach((b) => b.onclick = (e) => { e.stopPropagation(); navigator.clipboard?.writeText(b.parentElement.textContent.replace(/Copy$/, "").trim()); toast("Copied to clipboard"); });

  // ── updater → relaunch card ──
  let updateReady = null;
  $("#relaunch").style.display = "none";
  async function checkUpdates() { const u = window.__TAURI__?.updater; if (!u?.check) return; try { const up = await u.check(); if (up) { updateReady = up; $("#relaunchver").textContent = "v" + up.version; $("#relaunch").style.display = "flex"; } } catch (e) { console.warn("update check", e); } }
  setTimeout(checkUpdates, 3000);
  $("#relaunch").onclick = async () => { if (!updateReady) return; try { await updateReady.downloadAndInstall(); await window.__TAURI__?.process?.relaunch?.(); } catch (e) { toast(`Update failed: ${e}`); } };

  // ── first-run coachmark tour (spotlight; first launch + after updates only) ──
  const COACH = [
    { v: "home", a: "#homecard", t: "Chat with the network", d: "Ask anything — requests route to models served by peers. The first connection takes a few seconds; watch the status bar fill in." },
    { v: "providers", a: "#modeswitch", t: "Two sides of the app", d: "Home is where you use AI. Network is where you browse providers, share your models, and manage engines." },
    { v: "engines", a: '.nav[data-v="engines"]', t: "Engines & models", d: "OpenHydra wraps any engine already on your machine — whatever it can run, you can share." },
    { v: "share", a: "#sharingsw", t: "Share when you're ready", d: "Flip Sharing to announce your models and earn your place. Off means connected, not serving." },
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
  async function refresh() { try { state = await call("get_state"); } catch {} renderStatusbar(); if (["share", "settings", "connectors", "engines", "activity", "ledger"].includes(activeView)) renderView(); if (activeView === "peers") renderLogs(); }
  async function refreshEngines() { try { engines = await call("detect_engines_now"); } catch { engines = []; } renderStatusbar(); if (["share", "engines", "providers", "settings"].includes(activeView)) renderView(); }
  async function refreshStatus() { try { snap = await call("status_snapshot"); } catch { snap = null; } noteSeen(); renderStatusbar(); if (["peers", "providers", "activity", "ledger", "share"].includes(activeView)) renderView(); }
  $$(".enginst, #refreshEngines").forEach(() => {});

  // ── boot ──
  $(".header").style.display = "none"; // Home landing has no header
  $("#homelogo").src = "logo.png";
  renderRecents();
  (async () => {
    await refresh(); await refreshEngines();
    await ensureGateway();  // eager: warm discovery so the first chat isn't a cold 504
    await refreshStatus();
  })();
  setInterval(refresh, 2500);
  setInterval(refreshEngines, 10000);
  setInterval(refreshStatus, 2500);
})();
