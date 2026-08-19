// Share view (provider role) + the per-model share allowlist + the sharing/gateway lifecycle.
// setSharing / ensureGateway signal the controller via the bus (emit "nav"/"refresh") instead of
// importing its go()/refresh() (which would cycle). renderShare is the view-dispatcher entry.
import { $, $$, esc } from "./dom";
import { state, snap, engines } from "./state";
import { call } from "./bridge";
import { toast } from "./chrome";
import { emit } from "./bus";
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
  let hideInactive = store.get("oh_hideinactive", false);
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
  export function renderShare() {
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
