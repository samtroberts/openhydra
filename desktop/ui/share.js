// Share view (provider role) + the per-model share allowlist + the sharing/gateway lifecycle.
// setSharing / ensureGateway signal the controller via the bus (emit "nav"/"refresh") instead of
// importing its go()/refresh() (which would cycle). renderShare is the view-dispatcher entry.
import { $, $$, esc, escapeHtml } from "./dom";
import { state, snap, engines } from "./state";
import { call } from "./bridge";
import { toast, confirmModal } from "./chrome";
import { emit, on } from "./bus";
import { store } from "./storage";
import { totalServed, totalUsed, totalCreditsEarned, totalCreditsSpent, lifetimeServed, statModels } from "./stats";
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
        // Ask what to share first when nothing is chosen yet (a fresh install), or when the user has
        // never confirmed it in this dialog (an older build may have left "share everything" behind).
        if (!(await anythingShared()) || !state?.settings?.sharing_confirmed) { sharingBusy = false; await openShareDialog(); return; }
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
  // Re-read reach and consent from the file before changing them: the policy file can change outside
  // this view (a swarm member with rig control flips a model's scope), and saving a stale copy would
  // undo that — e.g. re-publish a model its owner just made private.
  async function reloadScope() {
    if (savingPolicy) return ensureScope();
    try { loadScope(await call("read_share_policy")); } catch { await ensureScope(); }
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
      const onKey = (e) => { if (e.key === "Escape") { e.stopPropagation(); done(false); } };
      const done = (v) => { document.removeEventListener("keydown", onKey, true); back.remove(); resolve(v); };
      document.addEventListener("keydown", onKey, true);
      $(".cx", back).onclick = () => done(false);
      $(".cok", back).onclick = () => done(true);
      back.onclick = (e) => { if (e.target === back) done(false); };
      $(".cx", back).focus();
    });
  }
  // Flip a shared model's reach (binary: `private` or `global`). Global opens the consent gate;
  // declining reverts (a re-render resets the switch). Private clears any recorded consent.
  async function setModelScope(m, next) {
    ensurePolicy(); await reloadScope();   // never mutate scope on a half-loaded or stale state
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
        .catch(() => { policy = { mode: "list", models: new Set() }; })
        .finally(() => { policyLoading = false; renderShare(); });
    }
    if (!policy) policy = { mode: "list", models: new Set() }; // provisional (shows nothing shared) until the load resolves
  }
  // The policy as it is on disk (or live, while sharing), awaited — for decisions, not just display.
  async function policyNow() {
    if (state?.provider?.status?.running && policyFromSnap()) { ensurePolicy(); await reloadScope(); return policy; }
    try { const r = await call("read_share_policy"); if (!savingPolicy) policy = { mode: r.mode, models: new Set(r.models || []) }; loadScope(r); }
    catch { ensurePolicy(); }
    return policy;
  }
  async function anythingShared() {
    const pol = await policyNow();
    return pol.mode === "all" || pol.models.size > 0;
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
    let ok = true;
    try { await call("save_share_policy", { policy: payload }); }
    catch (e) { ok = false; policy = prev; scope = prevScope; pendingSig = null; toast(`Save failed: ${e}`); }   // L2: roll back
    finally { savingPolicy = false; }
    emit("refresh-status");   // pull the fresh /status/share; the sticky hold clears once it matches
    renderShare();
    return ok;   // callers stop (no success toast, no start) when the save failed
  }
  async function toggleShareModel(m) {
    ensurePolicy(); await reloadScope();   // scope must be loaded (and current) before we mutate/persist it
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
    if (!(await savePolicy(mode, [...models]))) return;
    if (!running && !wasShared) await setSharing(true); // toggled a model on from a stopped node → begin serving
  }
  // "Share everything" master switch: ON ⇒ mode "all" (new models auto-share); OFF ⇒ freeze to the
  // current active set as an explicit "list" (nothing stops, but future models won't auto-share).
  async function toggleShareAll() {
    ensurePolicy(); await reloadScope();
    const running = !!state?.provider?.status?.running;
    if (policy.mode === "all") { await savePolicy("list", shareActiveModels()); return; }
    if (!state?.settings?.sharing_confirmed) { await openShareDialog(); return; }   // first time: choose explicitly
    const ok = await confirmModal(
      "Share every model, including ones you add later?",
      "Anyone on the OpenHydra network will be able to discover these models and send requests to them, and your machine serves the requests. Models you've set to Private stay private. You can turn this off at any time.",
      "Share everything",
    );
    if (!ok) { renderShare(); return; }
    scope.def = "global"; scope.defConsent = Date.now();
    if (!(await savePolicy("all", []))) return;
    if (!running) await setSharing(true);
  }

  // Choose what to share and how far it reaches, in one step, with consent recorded only here.
  // `only` limits the dialog to those models and ADDS them to what's already shared ("Add model");
  // without it, the dialog sets the whole selection (first "Start sharing").
  let shareDialogOpen = false;
  export async function openShareDialog(only) {
    if (shareDialogOpen) return;
    shareDialogOpen = true;
    try { await shareDialog(only); } finally { shareDialogOpen = false; }
  }
  async function shareDialog(only) {
    const all = shareActiveModels();
    if (!all.length) { emit("nav", "engines"); toast("No local models found — start or install an engine to share"); return; }
    await policyNow(); await reloadScope();
    const list = only ? only.filter((m) => all.includes(m)) : all;
    if (!list.length) { toast("That model isn't available from your engine right now"); return; }
    const checked = new Set(only ? list : list.filter((m) => isModelShared(m)));
    const one = list.length === 1;
    // Who can use them: pre-select only what the user already chose (a model already shared keeps
    // its reach); for new models there's no default — the user picks.
    const shared = list.filter((m) => isModelShared(m));
    const reaches = new Set(shared.map((m) => (isGlobal(m) ? "global" : "private")));
    const preReach = only && shared.length === list.length && reaches.size === 1 ? [...reaches][0] : null;
    const back = document.createElement("div");
    back.className = "cmodal-back";
    const rows = list.map((m) => `<label class="sdrow" style="display:flex;align-items:center;gap:8px;padding:4px 0;cursor:pointer"><input type="checkbox" data-m="${escapeHtml(m)}"${checked.has(m) ? " checked" : ""}/>${modelIcon(m)}<span>${esc(displayModelName(m))}</span></label>`).join("");
    back.innerHTML =
      `<div class="cmodal"><div class="cmodal-h"><b>${only ? `Share ${esc(displayModelName(list[0]))}${list.length > 1 ? ` and ${list.length - 1} more` : ""}` : "Choose what to share"}</b></div>` +
      `<div class="cmodal-b">` +
      (only && list.length === 1 ? `<input type="checkbox" data-m="${escapeHtml(list[0])}" checked hidden/>` : `<div style="max-height:220px;overflow:auto;margin-bottom:12px">${rows}</div>`) +
      `<div style="font-weight:600;font-size:12.5px;margin-bottom:4px">Who can use ${one ? "it" : "them"}</div>` +
      `<label style="display:flex;gap:8px;align-items:flex-start;padding:4px 0;cursor:pointer"><input type="radio" name="sdreach" value="global"${preReach === "global" ? " checked" : ""}/><span><b>Everyone</b> — anyone on the OpenHydra network can discover ${one ? "this model and send requests to it" : "these models and send requests to them"}. Your machine serves the requests; your prompts, files and identity aren't shared.</span></label>` +
      `<label style="display:flex;gap:8px;align-items:flex-start;padding:4px 0;cursor:pointer"><input type="radio" name="sdreach" value="private"${preReach === "private" ? " checked" : ""}/><span><b>Only my swarms</b> — not announced publicly; only members of a swarm you own can use ${one ? "it" : "them"} (set up in Swarms).</span></label>` +
      `</div>` +
      `<div class="cmodal-f"><button class="btn ghost sm cx">Cancel</button><button class="btn sm brand cok">Share</button></div></div>`;
    document.body.appendChild(back);
    const picked = await new Promise((resolve) => {
      const onKey = (e) => { if (e.key === "Escape") { e.stopPropagation(); done(null); } };
      const done = (v) => { document.removeEventListener("keydown", onKey, true); back.remove(); resolve(v); };
      document.addEventListener("keydown", onKey, true);
      $(".cx", back).onclick = () => done(null);
      back.onclick = (e) => { if (e.target === back) done(null); };
      $(".cok", back).onclick = () => {
        const ms = $$("input[type=checkbox]", back).filter((c) => c.checked).map((c) => c.dataset.m);
        const reach = $("input[name=sdreach]:checked", back)?.value;
        if (!ms.length) { toast("Pick at least one model"); return; }
        if (!reach) { toast(`Choose who can use ${ms.length === 1 ? "it" : "them"}`); return; }
        done({ models: ms, reach });
      };
      ($("input[type=checkbox]:not([hidden])", back) || $("input[type=radio]", back))?.focus();
    });
    if (!picked) { renderShare(); return; }
    const running = !!state?.provider?.status?.running;
    let mode = policy.mode, models = new Set(policy.mode === "all" ? all : policy.models);
    if (!only) {
      // The dialog's selection replaces the old one.
      mode = "list";
      for (const m of all) if (!picked.models.includes(m)) { models.delete(m); scope.byModel.delete(m); scope.consent.delete(m); }
    }
    for (const m of picked.models) {
      models.add(m);
      scope.byModel.set(m, picked.reach);
      if (picked.reach === "global") scope.consent.set(m, Date.now()); else scope.consent.delete(m);
    }
    if (!(await savePolicy(mode, mode === "all" ? [] : [...models]))) return;
    await call("confirm_sharing").catch(() => {});
    if (state?.settings) state.settings.sharing_confirmed = true;   // until the next poll brings it
    toast(`Sharing ${picked.models.length} model${picked.models.length === 1 ? "" : "s"} with ${picked.reach === "global" ? "everyone" : "your swarms"}`);
    if (!running) await setSharing(true);
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
    head.querySelector(".mut").textContent = `${running ? announcedCount : 0} models announced · gateway :${state?.settings?.gateway_port || 16527}${up}`;
    const k = $$("#v-share .g4 .kpi .val");
    // #7: durable lifetime totals (survive restart; `used` counts external clients too).
    const served = totalServed(), used = totalUsed();
    // #3 fix: "Credits earned" = net CONTRIBUTION, which RISES with serving and falls with using
    // (the earlier `economy.total_credit` summed counterparty balances and moved the wrong way).
    // Credit-tiering: this is now COST-WEIGHTED (earned − spent), so a 30B serve counts for far more
    // than a 1B one — not the flat served − used tokens.
    const netCredits = totalCreditsEarned() - totalCreditsSpent();
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
      else status = `<span class="badge secondary" title="Not announced publicly. Only members of a swarm you own can use it (set up in Swarms).">swarms only</span>`;
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
      // M2/M4: export a signed `.openhydra` card. A GLOBAL model → a public card (a link anyone can
      // use). A PRIVATE model (M4) → a private card bound to a swarm you own (only members can serve
      // from it). Shown for any shared model; the private branch prompts for the swarm.
      const cardBtn = sharedIntent
        ? `<button class="cardbtn" data-export-card="${escapeHtml(m)}" data-private="${gl ? "0" : "1"}" title="${gl ? "Export a public .openhydra card — a link to this model" : "Export a private .openhydra card for a swarm — only members can serve from it"}">🔗</button>`
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
    // M2/M4: export a `.openhydra` card — public for a global model, or private (swarm-bound) for a
    // private one.
    $$("#servetable [data-export-card]").forEach((b) => b.onclick = () => exportCardModal(b.dataset.exportCard, b.dataset.private === "1"));
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
    $("#wanttable tbody").innerHTML = `<tr><td colspan="4" class="mut">Not available yet: this fills in once the network reports demand.</td></tr>`;
  }
