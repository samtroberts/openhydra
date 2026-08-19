// Tier-1 engine installer (B5): consent -> stream install://progress -> refresh. Self-contained
// modal overlay; on completion it fires the "refresh-engines" bus signal (the controller re-detects
// engines) rather than importing the controller's refreshEngines (which would cycle).
import { $, $$, esc } from "./dom";
import { engines, installedEngines } from "./state";
import { call, mockEmitInstall, mockInstallCbs } from "./bridge";
import { toast } from "./chrome";
import { emit } from "./bus";

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

  export async function startInstall(label, name) {
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
      else if (ev.kind === "done") { un(); hideBar(); body.innerHTML = `<span class="badge ok">done</span> ${esc(ev.message)}`; append("✓ " + ev.message); actions.innerHTML = `<button class="btn brand sm" id="instFin" style="margin-left:auto">Done</button>`; $("#instFin").onclick = () => { $("#installov").style.display = "none"; emit("refresh-engines"); }; emit("refresh-engines"); }
      else if (ev.kind === "error") { un(); hideBar(); body.innerHTML = `<span class="badge warn">failed</span> ${esc(ev.message)}`; append("✗ " + ev.message); actions.innerHTML = `<button class="btn outline sm" id="instFin" style="margin-left:auto">Close</button>`; $("#instFin").onclick = () => $("#installov").style.display = "none"; }
    });
    try { await call("install_engine", { engine: label, accel: null, variant: variant || null }); }
    catch (e) { /* backend already emitted an error event; guard the mock/no-event path */ mockEmitInstall({ engine: label, kind: "error", message: String(e) }); }
  }


  const ENGINES = [["Ollama", "ollama", "General-purpose local LLMs."], ["LM Studio", "lm-studio", "MLX-optimised models on Apple silicon."], ["llama.cpp", "llama.cpp", "Lightweight GGUF runtime."], ["ComfyUI", "comfyui", "Image generation — Stable Diffusion, Flux."], ["vLLM", "vllm", "High-throughput serving. Needs Python + GPU."], ["Exo", "exo", "Shard big models across your devices."]];
  // Engines OpenHydra can start on demand (self-serving; no model/cluster arg) → get a "Run" CTA.
  const RUNNABLE = new Set(["ollama", "lm-studio"]);
  export function renderEngines() {
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

  // Start an installed-but-idle engine's server, then refresh so the card flips to "running".
  async function runEngine(label, name) {
    toast(`Starting ${name}…`);
    try { await call("run_engine", { engine: label }); toast(`${name} started`); }
    catch (e) { toast(String(e)); }
    emit("refresh-engines");
  }
