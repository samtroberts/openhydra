// Share view (provider role) + the per-model share allowlist + the sharing/gateway lifecycle.
// setSharing / ensureGateway signal the controller via the bus (emit "nav"/"refresh") instead of
// importing its go()/refresh() (which would cycle). renderShare is the view-dispatcher entry.
import { $, $$, esc } from "./dom";
import { state, snap, engines } from "./state";
import { call } from "./bridge";
import { toast } from "./chrome";
import { emit, on } from "./bus";
import { store } from "./storage";
import { totalServed, totalUsed, lifetimeServed, statModels } from "./stats";
import { renderChart } from "./chart";
import { fmtNum, fmtUptime, modelIcon } from "./format";

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
  on("share-policy-reset", () => { policy = null; pendingSig = null; });
  function shareActiveModels() { return [...new Set(engines.flatMap((e) => e.models))]; }
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
        .then((r) => { policy = { mode: r.mode, models: new Set(r.models || []) }; })
        .catch(() => { policy = { mode: "all", models: new Set() }; })
        .finally(() => { policyLoading = false; renderShare(); });
    }
    if (!policy) policy = { mode: "all", models: new Set() }; // provisional until the load resolves
  }
  function isModelShared(m) { ensurePolicy(); return policy.mode === "all" || policy.models.has(m); }
  async function savePolicy(mode, models) {
    const prev = policy;                                    // L2: rollback target on failure
    policy = { mode, models: new Set(models) };             // optimistic — the row reflects intent at once
    pendingSig = policySig(policy); pendingSince = Date.now();
    savingPolicy = true; renderShare();
    try { await call("save_share_policy", { policy: { version: 1, mode, models } }); }
    catch (e) { policy = prev; pendingSig = null; toast(`Save failed: ${e}`); }   // L2: roll back
    finally { savingPolicy = false; }
    emit("refresh-status");   // pull the fresh /status/share; the sticky hold clears once it matches
    renderShare();
  }
  async function toggleShareModel(m) {
    ensurePolicy();
    const wasShared = policy.mode === "all" || policy.models.has(m);
    const running = !!state?.provider?.status?.running;
    let mode = policy.mode, models = new Set(policy.models);
    if (wasShared) {
      // Turning OFF. From "all", materialize the current active set first, then drop this one — so
      // the others stay shared. "Share nothing" is now a valid state (list + empty), so we never
      // have to stop the whole provider just because the last model was de-selected.
      if (mode === "all") { mode = "list"; models = new Set(shareActiveModels().filter((x) => x !== m)); }
      else models.delete(m);
    } else if (mode === "list") {
      models.add(m);   // (in "all" mode it's already shared)
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
    ensurePolicy();
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
      let status;
      if (!running) status = `<span class="badge secondary">ready</span>`;
      else if (sharedIntent && (announcedSet.has(m) || !hasShareView)) status = `<span class="badge ok">live</span>`;
      else if (sharedIntent) status = `<span class="badge warn">pending</span>`;
      else status = `<span class="badge secondary">off</span>`;
      const swTitle = !running ? "Share this model when you start sharing"
        : sharedIntent ? "Sharing on the network — toggle off to stop" : "Toggle on to share this model";
      rows.push(`<tr><td>${modelIcon(m)}<b>${esc(m)}</b></td><td>${esc(engineFor(m) || "—")}</td><td class="num">${reqs}</td><td class="num">${fmtNum(tokens)}</td><td class="num">${tps}</td><td>${status}</td><td><div class="switch ${sharedIntent ? "on" : ""}" data-share="${esc(m)}" title="${swTitle}"></div></td></tr>`);
    }
    $("#servetable tbody").innerHTML = rows.join("") || `<tr><td colspan="7" class="mut">No engines answering — start Ollama, LM Studio, vLLM, llama.cpp, or Exo, then rescan.</td></tr>`;
    // "Previously served" — lifetime tokens on record but NOT currently detected.
    const pastModels = statModels().filter((id) => lifetimeServed(id) > 0 && !active.has(id))
      .sort((a, b) => (lifetimeServed(b) || 0) - (lifetimeServed(a) || 0));
    const pastCard = $("#pastcard");
    if (pastCard) {
      pastCard.style.display = pastModels.length ? "" : "none";
      $("#pasttable tbody").innerHTML = pastModels.map((m) =>
        `<tr><td>${modelIcon(m)}<b>${esc(m)}</b></td><td class="num">${fmtNum(lifetimeServed(m))}</td><td><span class="badge secondary">not on device</span></td></tr>`
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
