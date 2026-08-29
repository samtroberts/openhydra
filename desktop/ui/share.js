// Share view (provider role) + the per-model share allowlist + the sharing/gateway lifecycle.
// setSharing / ensureGateway signal the controller via the bus (emit "nav"/"refresh") instead of
// importing its go()/refresh() (which would cycle). renderShare is the view-dispatcher entry.
import { $, $$, esc, escapeHtml } from "./dom";
import { state, snap, engines } from "./state";
import { call } from "./bridge";
import { toast } from "./chrome";
import { emit, on } from "./bus";
import { store } from "./storage";
import { totalServed, totalUsed, lifetimeServed, statModels } from "./stats";
import { renderChart } from "./chart";
import { fmtNum, fmtUptime, modelIcon } from "./format";
import { displayModelName } from "./models";
import { exportCardModal, ensureImportSection } from "./cards";

  let sharingBusy = false;
  export async function setSharing(on) {
    if (sharingBusy) return; sharingBusy = true;
    try {
      if (on) {
        if (!engines.some((e) => e.models.length)) { emit("nav", "engines"); toast("No local models found — start or install an engine to share"); return; }
        await call("start_provider");
      } else {
        await call("stop_provider");
      }
      emit("refresh");
    } catch (e) { toast(`${on ? "start" : "stop"} sharing failed: ${e}`); }
    finally { sharingBusy = false; }
  }
  export async function toggleSharing() { await setSharing(!state?.provider?.status?.running); }
  // ── gateway lifecycle (Local API) ──
  export async function ensureGateway() {
    if (state?.gateway?.status?.running) return true;
    try { await call("start_gateway"); } catch (e) { toast(`Local API failed to start: ${e}`); return false; }
    for (let i = 0; i < 24; i++) { await new Promise((r) => setTimeout(r, 300)); try { if (await call("gateway_health")) break; } catch {} }
    emit("refresh"); return true;
  }
  let hidePast = store.get("oh_hidepast", false);   // B4: collapse the "previously served" table
  const engineFor = (m) => { for (const e of engines) if (e.models.includes(m)) return e.label; return null; };

  // ── per-model share policy (Share view toggles) ──
  // The intended policy lives in ~/.openhydra/share-policy.json (mode "all" | "list"). The running
  // provider HOT-RELOADS it, so a toggle applies without a restart. `snap.share` carries the agent's
  // live view — mode + intended list + the REAL announced set — which we render from; when the
  // provider is stopped we seed the intended state from `read_share_policy`. New models default OFF
  // in "list" mode (they must be explicitly shared); the "Share everything" switch sets "all".
  let policy = null;                    // { mode: "all"|"list", models: Set<string> } — intended
  let policyLoading = false, savingPolicy = false;
  // M4: after a save, keep the optimistic policy authoritative until the agent's live view actually
  // reflects it — otherwise a stale snapshot poll would revert the just-toggled switch (bounce). A
  // timeout releases the hold so a permanently-diverging agent can't freeze the UI.
  let pendingSig = null, pendingSince = 0;
  const RECONCILE_TIMEOUT_MS = 10000;
  const policySig = (pol) => pol ? `${pol.mode}|${[...pol.models].sort().join(",")}` : "";
  // The policy file was changed OUTSIDE this view (the Settings "Reset" button): drop the cached
  // policy + any pending hold so the next render re-reads it. Without this, a reset while the
  // provider is STOPPED would leave the toggles showing the pre-reset selection (review F1/F2).
  on("share-policy-reset", () => { policy = null; pendingSig = null; scopeLoaded = false; scopeLoadPromise = null; });
  function shareActiveModels() { return [...new Set(engines.flatMap((e) => e.models))]; }

  // ── per-model REACH (scope): device / private / global (M1) ──
  // Orthogonal to mode/models (which decides *whether* a model is shared) — scope decides *how far*
  // it reaches. Only `global` (with recorded consent) is announced to the public DHT. The live
  // status view (`snap.share`) carries no scope, so scope lives in its own state, loaded once from
  // `read_share_policy` and preserved across snapshot reconciles. `def` (default_scope) stays as the
  // file has it — `global` for a legacy file, so upgrading NEVER silently un-shares (the agent's
  // migration materialises the matching consent record); a newly-shared model is set to `private`
  // explicitly (privacy-first), leaving already-global models on their existing reach.
  let scope = { def: "global", byModel: new Map(), consent: new Map() };
  let scopeLoaded = false, scopeLoadPromise = null;
  function loadScope(r) {
    scope = {
      def: r?.default_scope || "global",
      byModel: new Map(Object.entries(r?.scopes || {})),
      consent: new Map(Object.entries(r?.global_consent || {})),
      // Policy-level "share everything globally" consent (consent-hardening). Preserved verbatim so
      // a save never clears the migration-materialised record (which would un-announce default-Global
      // models). Phase 3 sets it on a real "Share everything globally" choice; here we only carry it.
      defConsent: r?.default_global_consent ?? null,
    };
    scopeLoaded = true;
  }
  // Resolves once scope state is loaded from the file (once). It MUST be awaited by every
  // scope-mutating action: while the provider is running, `ensurePolicy` reconciles mode/models
  // from `snap.share` and never loads scope — only this async read does. Without awaiting it, a
  // toggle/scope change fired before the initial read would persist a TRUNCATED policy: dropping
  // the file's `scopes`/`global_consent`/`default_global_consent` — which the airtight backend then
  // reads as un-consented, silently un-announcing every previously-global model — and clobbering
  // `default_scope`. See the M1 race fix.
  function ensureScope() {
    if (scopeLoaded) return Promise.resolve();
    if (!scopeLoadPromise) {
      scopeLoadPromise = call("read_share_policy")
        .then((r) => loadScope(r))
        .catch(() => { scopeLoaded = true; })   // read failed → keep the safe default; don't loop
        .finally(() => { renderShare(); });
    }
    return scopeLoadPromise;
  }
  const effectiveScope = (m) => scope.byModel.get(m) || scope.def;
  // UI reach is BINARY: global vs not-global. `device` (a deferred, not-yet-offered loopback tier) and
  // any unknown value collapse to the Private presentation. `isGlobal` is the one enforced distinction.
  const isGlobal = (m) => effectiveScope(m) === "global";
  // Binary picker offers Private ↔ Global only. A legacy `device` value still READS/round-trips (it
  // collapses to the Private presentation via `isGlobal`), but it is never offered as a choice.
  // M4-base: Private is now access-controlled — the provider serves a Private model only to a member
  // of a swarm you own (see the Swarms tab), not merely un-announced. Copy reflects that.
  const SCOPE_META = {
    private: { label: "Private", icon: "🔒", title: "Off the global network — served only to members of a swarm you own (set up in Swarms)." },
    global:  { label: "Global",  icon: "🌐", title: "Announced to the global network — others can discover and route to it." },
  };
  // Consent gate for Global publish. Returns true only if the operator confirms. Reuses the shared
  // `.cmodal` chrome (window.confirm is inert in the Tauri webview).
  function globalConsentModal(m) {
    return new Promise((resolve) => {
      const back = document.createElement("div");
      back.className = "cmodal-back";
      back.innerHTML =
        `<div class="cmodal"><div class="cmodal-h"><b>Publish “${esc(displayModelName(m))}” to the global network?</b></div>` +
        `<div class="cmodal-b">` +
        `<div class="cmodal-path">Anyone on the OpenHydra network will be able to <b>discover</b> this model, <b>route inference</b> to it, and <b>earn or spend credits</b> against it. Your machine serves the requests.</div>` +
        `<div class="cmodal-warn">This exposes an <b>offer</b> for this model (its clean handle + capability) to the public network — not your prompts, files, or identity. You can switch it back to Private at any time.</div>` +
        `</div>` +
        `<div class="cmodal-f"><button class="btn ghost sm cx">Cancel</button><button class="btn sm brand cok">Publish globally</button></div></div>`;
      document.body.appendChild(back);
      const done = (v) => { back.remove(); resolve(v); };
      $(".cx", back).onclick = () => done(false);
      $(".cok", back).onclick = () => done(true);
      back.onclick = (e) => { if (e.target === back) done(false); };
    });
  }
  // Flip a shared model's reach (binary: `private` or `global`). Global opens the consent gate;
  // declining reverts (a re-render resets the switch). Private clears any recorded consent.
  async function setModelScope(m, next) {
    ensurePolicy(); await ensureScope();   // never mutate scope on a half-loaded state (race fix)
    if (next === effectiveScope(m)) return;
    if (next === "global") {
      const ok = await globalConsentModal(m);
      if (!ok) { renderShare(); return; }                 // declined → revert the switch
      scope.byModel.set(m, "global"); scope.consent.set(m, Date.now());
    } else {
      scope.byModel.set(m, next); scope.consent.delete(m); // Private never carries consent
    }
    await savePolicy(policy.mode, [...policy.models]);      // scope fields ride along (see savePolicy)
  }
  function policyFromSnap() {
    const s = snap?.share;
    return s?.share_mode ? { mode: s.share_mode, models: new Set(s.shared_models || []) } : null;
  }
  // Resolve the intended policy. When running, the agent's live view (`snap.share`) is ground truth —
  // EXCEPT while a save is pending and the view hasn't caught up yet: then hold the optimistic copy.
  // When stopped, lazily load the file.
  function ensurePolicy() {
    const running = !!state?.provider?.status?.running;
    if (running && !savingPolicy) {
      const p = policyFromSnap();
      if (p) {
        if (pendingSig !== null) {
          if (policySig(p) === pendingSig || Date.now() - pendingSince > RECONCILE_TIMEOUT_MS) {
            pendingSig = null; policy = p;      // confirmed (or timed out) → adopt ground truth
          }
          // else: keep the local optimistic `policy` — don't reseed to a stale snapshot
        } else {
          policy = p;                            // normal reconcile (also reflects external edits)
        }
        return;
      }
    }
    if (policy) return;
    const p2 = policyFromSnap(); if (p2) { policy = p2; return; }
    if (!policyLoading) {
      policyLoading = true;
      call("read_share_policy")
        .then((r) => { policy = { mode: r.mode, models: new Set(r.models || []) }; loadScope(r); })
        .catch(() => { policy = { mode: "all", models: new Set() }; })
        .finally(() => { policyLoading = false; renderShare(); });
    }
    if (!policy) policy = { mode: "all", models: new Set() }; // provisional until the load resolves
  }
  function isModelShared(m) { ensurePolicy(); return policy.mode === "all" || policy.models.has(m); }
  async function savePolicy(mode, models) {
    const prev = policy;                                    // L2: rollback target on failure
    // Snapshot scope too so a failed save reverts reach/consent, not just mode/models (M1).
    const prevScope = { def: scope.def, byModel: new Map(scope.byModel), consent: new Map(scope.consent), defConsent: scope.defConsent };
    policy = { mode, models: new Set(models) };             // optimistic — the row reflects intent at once
    pendingSig = policySig(policy); pendingSince = Date.now();
    savingPolicy = true; renderShare();
    // The scope maps ride along from module state; version 3 carries the reach/consent shape.
    const payload = {
      version: 3, mode, models,
      default_scope: scope.def,
      scopes: Object.fromEntries(scope.byModel),
      global_consent: Object.fromEntries(scope.consent),
      default_global_consent: scope.defConsent,   // preserve the policy-level consent verbatim
    };
    try { await call("save_share_policy", { policy: payload }); }
    catch (e) { policy = prev; scope = prevScope; pendingSig = null; toast(`Save failed: ${e}`); }   // L2: roll back
    finally { savingPolicy = false; }
    emit("refresh-status");   // pull the fresh /status/share; the sticky hold clears once it matches
    renderShare();
  }
  async function toggleShareModel(m) {
    ensurePolicy(); await ensureScope();   // scope must be loaded before we mutate/persist it (race fix)
    const wasShared = policy.mode === "all" || policy.models.has(m);
    const running = !!state?.provider?.status?.running;
    let mode = policy.mode, models = new Set(policy.models);
    if (wasShared) {
      // Turning OFF. From "all", materialize the current active set first, then drop this one — so
      // the others stay shared. "Share nothing" is now a valid state (list + empty), so we never
      // have to stop the whole provider just because the last model was de-selected.
      if (mode === "all") { mode = "list"; models = new Set(shareActiveModels().filter((x) => x !== m)); }
      else models.delete(m);
      scope.byModel.delete(m); scope.consent.delete(m);   // un-shared → forget its reach + consent
    } else if (mode === "list") {
      models.add(m);   // (in "all" mode it's already shared)
      // Privacy-first: a newly-shared model reaches only your trust domain until you opt it Global.
      if (!scope.byModel.has(m)) scope.byModel.set(m, "private");
    }
    await savePolicy(mode, [...models]);
    if (!running && !wasShared) await setSharing(true); // toggled a model on from a stopped node → begin serving
  }
  // "Share everything" master switch: ON ⇒ mode "all" (new models auto-share); OFF ⇒ freeze to the
  // current active set as an explicit "list" (nothing stops, but future models won't auto-share).
  async function toggleShareAll() {
    ensurePolicy();
    const running = !!state?.provider?.status?.running;
    if (policy.mode === "all") { await savePolicy("list", shareActiveModels()); }
    else { await savePolicy("all", []); if (!running) await setSharing(true); }
  }
  export function renderShare() {
    const p = state?.provider?.status, running = !!p?.running, t = snap?.transfers;
    ensurePolicy(); ensureScope();
    // The REAL advertised set from /status/share (falls back to the log-parsed count pre-M2 agents).
    const hasShareView = !!snap?.share;   // L5: a pre-M2 agent has no per-model announced set
    const announcedSet = new Set(snap?.share?.announced_models || []);
    const announcedCount = hasShareView ? announcedSet.size : (running ? (p.announced ?? 0) : 0);
    const head = $("#v-share .row .ctitle").parentElement;
    head.querySelector(".badge").innerHTML = running ? '<span class="dot ok"></span>provider running' : '<span class="dot"></span>not sharing';
    head.querySelector(".badge").className = "badge " + (running ? "ok" : "secondary"); head.querySelector(".badge").style.marginLeft = "8px";
    // #1: start/stop lives here now (the title-bar control became a CTA that routes here).
    const stb = $("#sharetoggle");
    if (stb) { stb.textContent = running ? "Stop sharing" : "Start sharing"; stb.className = "btn sm " + (running ? "outline" : "brand"); stb.style.marginLeft = "12px"; }
    const up = snap?.uptime_secs != null ? ` · up ${fmtUptime(snap.uptime_secs)}` : "";
    head.querySelector(".mut").textContent = `${running ? announcedCount : 0} models announced · gateway :16527${up}`;
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
    // B4: two tables. Primary = models an engine serves RIGHT NOW, sorted shared-first then by
    // lifetime tokens. Secondary = models with lifetime tokens on record but no longer detected
    // (deleted / engine offline). Lifetime tokens come from the durable accumulator (survive restart).
    const per = t?.per_model || {};
    const active = new Set(engines.flatMap((e) => e.models));
    const activeModels = [...active].sort((a, b) => {
      const sa = isModelShared(a) ? 1 : 0, sb = isModelShared(b) ? 1 : 0;
      if (sa !== sb) return sb - sa;                                   // shared first
      return (lifetimeServed(b) || 0) - (lifetimeServed(a) || 0);      // then by lifetime tokens
    });
    const rows = [];
    for (const m of activeModels) {
      const pm = per[m] || {};
      const tokens = lifetimeServed(m) || pm.tokens || 0;
      const reqs = pm.requests ?? "—";
      const tps = pm.avg_native_tps ? Math.round(pm.avg_native_tps) : "—";
      // Real state, from the agent's announced set (not detection/optimism): shared → "live" once
      // actually announced, "pending" in the brief window before the announce lands.
      const sharedIntent = isModelShared(m);
      const gl = isGlobal(m);
      // Status = the ANNOUNCEMENT state (reach is owned by the pill, so we never repeat "private"
      // here): a `global` model shows live/pending; a shared non-global model shows "not announced"
      // (served on the trust domain, absent from the public DHT). `device`/unknown ⇒ non-global.
      let status;
      if (!running) status = `<span class="badge secondary">ready</span>`;
      else if (!sharedIntent) status = `<span class="badge secondary">off</span>`;
      else if (gl) status = (announcedSet.has(m) || !hasShareView)
        ? `<span class="badge ok">live</span>` : `<span class="badge warn">pending</span>`;
      else status = `<span class="badge secondary" title="Shared and served on your trust domain, but not announced to the global network.">not announced</span>`;
      const swTitle = !running ? "Share this model when you start sharing"
        : sharedIntent ? "Sharing — toggle off to stop" : "Toggle on to share this model";
      // Reach control (binary), only for a shared model: a Private | Global segmented pill. Choosing
      // Global opens the consent gate; Private clears the model's consent. Distinct from the on/off
      // share switch so the two are never confused. `device`/unknown highlights Private.
      const reachPill = sharedIntent
        ? `<span class="reach-pill" title="How far this model reaches">`
          + `<button class="rp ${gl ? "" : "sel"}" data-reach="${escapeHtml(m)}" data-to="private" title="${SCOPE_META.private.title}">🔒 Private</button>`
          + `<button class="rp ${gl ? "sel" : ""}" data-reach="${escapeHtml(m)}" data-to="global" title="${SCOPE_META.global.title}">🌐 Global</button>`
          + `</span>`
        : "";
      // M2: export a signed `.openhydra` card — only for a globally-shared model (a card is a public
      // pointer, so the export gate refuses anything else anyway).
      const cardBtn = (sharedIntent && gl)
        ? `<button class="cardbtn" data-export-card="${escapeHtml(m)}" title="Export a signed .openhydra card — share a link to this model">🔗</button>`
        : "";
      rows.push(`<tr><td>${modelIcon(m)}<b title="${escapeHtml(displayModelName(m))}">${esc(displayModelName(m))}</b></td><td>${esc(engineFor(m) || "—")}</td><td class="num">${reqs}</td><td class="num">${fmtNum(tokens)}</td><td class="num">${tps}</td><td>${status}</td><td class="shcell"><div class="switch ${sharedIntent ? "on" : ""}" data-share="${escapeHtml(m)}" title="${swTitle}"></div>${reachPill}${cardBtn}</td></tr>`);
    }
    $("#servetable tbody").innerHTML = rows.join("") || `<tr><td colspan="7" class="mut">No engines answering — start Ollama, LM Studio, vLLM, llama.cpp, or Exo, then rescan.</td></tr>`;
    // "Previously served" — lifetime tokens on record but NOT currently detected.
    const pastModels = statModels().filter((id) => lifetimeServed(id) > 0 && !active.has(id))
      .sort((a, b) => (lifetimeServed(b) || 0) - (lifetimeServed(a) || 0));
    const pastCard = $("#pastcard");
    if (pastCard) {
      pastCard.style.display = pastModels.length ? "" : "none";
      $("#pasttable tbody").innerHTML = pastModels.map((m) =>
        `<tr><td>${modelIcon(m)}<b title="${esc(displayModelName(m))}">${esc(displayModelName(m))}</b></td><td class="num">${fmtNum(lifetimeServed(m))}</td><td><span class="badge secondary">not on device</span></td></tr>`
      ).join("");
      const tbl = $("#pasttable"), foot = pastCard.querySelector(".pager");
      if (tbl) tbl.style.display = hidePast ? "none" : "";
      if (foot) foot.style.display = hidePast ? "none" : "";
      const pt = $("#pasttoggle");
      if (pt) { pt.textContent = hidePast ? "Show" : "Hide"; pt.onclick = () => { hidePast = !hidePast; store.set("oh_hidepast", hidePast); renderShare(); }; }
    }
    renderChart("#sharechart", "share");   // #10 timeline
    // The card-header pill = the REAL announced count (was mislabeled: it counted detected models).
    const annBadge = $("#servetable").closest(".card").querySelector(".row .badge");
    if (annBadge) { annBadge.textContent = `${running ? announcedCount : 0} announced`; annBadge.className = "badge " + (running && announcedCount ? "ok" : "secondary"); }
    // "Share everything" master switch — tri-state (B3): ON = mode "all" (every model incl. future);
    // INDETERMINATE = a specific selection (some shared, not "all" mode); OFF = nothing shared.
    const saw = $("#shareallwrap"), sasw = $("#shareallsw");
    if (saw && sasw) {
      const allActive = shareActiveModels();
      saw.style.display = allActive.length ? "inline-flex" : "none";
      const masterOn = policy?.mode === "all";
      const someShared = allActive.some((m) => isModelShared(m));
      sasw.classList.toggle("on", masterOn);
      sasw.classList.toggle("indeterminate", !masterOn && someShared);
      sasw.title = masterOn
        ? "Sharing every model, including ones you add later — click to keep only your current selection"
        : someShared
          ? "Sharing a selection — click to share everything (including models you add later)"
          : "Click to share every model, including ones you add later";
      sasw.onclick = () => toggleShareAll();
    }
    $$("#servetable [data-share]").forEach((sw) => sw.onclick = () => toggleShareModel(sw.dataset.share));
    // Reach pill: each segment declares its target scope; clicking the active one is a no-op (guarded
    // in setModelScope), clicking Global opens the consent gate. The whole segment is the hit target.
    $$("#servetable [data-reach]").forEach((b) => b.onclick = () => setModelScope(b.dataset.reach, b.dataset.to));
    // M2: export a `.openhydra` card for a globally-shared model.
    $$("#servetable [data-export-card]").forEach((b) => b.onclick = () => exportCardModal(b.dataset.exportCard));
    // M2: build the "Add a model by card" import section once (persists across status refreshes).
    ensureImportSection();
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
