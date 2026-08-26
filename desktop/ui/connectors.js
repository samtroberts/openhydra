// Connectors view: detect installed coding tools, render the surface switcher, live command
// snippet, searchable model selector, and state-driven actions on each actionable card; wire
// Connect / Connect&Run / Disconnect / Test. renderConnectors is the entry the view dispatcher calls.
import { $, $$, esc, escapeHtml } from "./dom";
import { call } from "./bridge";
import { toast } from "./chrome";
import { liveModels, displayModelName } from "./models";
import { on } from "./bus";
import { cliState, refreshCliStatus, installCli } from "./cliui";

  export function renderConnectors() {
    $$("#v-connectors .cp").forEach((b) => b.onclick = (e) => { e.stopPropagation(); const sn = b.closest(".conncard")?.querySelector(".snippet"); navigator.clipboard?.writeText((sn?.textContent || "").trim()); toast("Copied"); });
    wireConnectors();
    if (cliState() === null) refreshCliStatus();   // lazy first fetch if boot hasn't run it yet
    renderCliBanner();
  }

  // Layer 2: the terminal snippets/`openhydra launch` only work once the `openhydra` CLI is on PATH.
  // When it isn't, show a banner offering the one-click install instead of letting the user copy a
  // command that fails with command-not-found. Status comes from the shared cliui source (one probe,
  // broadcast via "cli-status"), so an install here OR from the Settings row updates both surfaces.
  on("cli-status", renderCliBanner);
  function renderCliBanner() {
    const b = $("#cliinstallbanner"); if (!b) return;
    const s = cliState();
    // Show ONLY when we've CONFIRMED the CLI is missing/broken — never on error/unknown. A failed check
    // doesn't nag here (the always-present Settings row is the reliable install path).
    const show = s && !s.error && (!s.on_path || s.managed_broken);
    if (!show) { b.style.display = "none"; b.innerHTML = ""; return; }
    const broken = s.managed_broken;
    b.style.display = "flex";
    b.innerHTML = `<span class="clib-ic">⌨</span><span class="clib-t">${broken
        ? `The <span class="mono">openhydra</span> command is installed but broken (the app moved) — repair it so the terminal commands below work.`
        : `The <span class="mono">openhydra</span> command isn't on your PATH yet — the terminal snippets below (and <span class="mono">openhydra launch</span>) won't run until it's installed.`
      }</span><button class="btn brand sm clib-go">${broken ? "Repair" : "Install the CLI"}</button>`;
    const go = $(".clib-go", b);
    if (go) go.onclick = async () => {
      go.disabled = true; go.textContent = "Installing…";
      await installCli();   // toasts + refreshes; the "cli-status" handler re-renders (hides on success)
    };
  }

  const AUTO_MODEL = "openhydra/auto";
  function surfLabel(s) { return s === "terminal" ? "Terminal" : s === "editor" ? "Editor" : "App"; }

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
        + [...card._models].map((m) => `<span class="mchip" title="${esc(m)}">${esc(displayModelName(m))} <button class="mrm" data-v="${esc(m)}" title="Remove">×</button></span>`).join("");
      $$(".mrm", chipsEl).forEach((b) => b.onclick = () => { card._models.delete(b.dataset.v); renderChips(); renderSnippet(card, st); if (!opts.hidden) renderOpts(); });
    };
    const add = (v) => { v = (v || "").trim(); if (!v || v === AUTO_MODEL) return; card._models.add(v); input.value = ""; renderChips(); renderOpts(); renderSnippet(card, st); };
    const renderOpts = () => {
      const q = input.value.trim().toLowerCase();
      const avail = liveModels().filter((m) => !card._models.has(m) && m.toLowerCase().includes(q));
      let html = avail.map((m) => `<div class="mopt" data-v="${esc(m)}" title="${esc(m)}">${esc(displayModelName(m))}</div>`).join("");
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
