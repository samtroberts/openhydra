// Network › Models view: the announced-provider table grouped by model, with sticky provider
// counts, avg TPS + earned reputation, and a per-model expand (#11b) to the individual providers.
import { $, $$, esc, peerShort } from "./dom";
import { snap } from "./state";
import { netModels, modelIdle, seenCount, displayModelName, localSharedModels } from "./models";
import { repByPeerModel, repKey, modelReputation, modelAvgTps, trustByPeerModel } from "./econ";
import { modelIcon, modelCat, repBadge, tierOf, trustCell } from "./format";

  const expandedProviders = new Set();
  let sortBy = "reputation";
  export function setProviderSort(v) { sortBy = v; renderProviders(); }
  export function renderProviders() {
    const models = netModels();
    const local = new Set(localSharedModels());   // only models actually shared count as "· your machine"
    const strip = $("#v-providers .card.pad");
    strip.querySelectorAll("b")[0].textContent = (snap?.transfers?.tokens_served ?? 0);  // tokens
    strip.querySelectorAll("b")[1].textContent = models.length;                          // models
    strip.querySelectorAll("b")[2].textContent = snap?.network?.peers?.length ?? 0;       // peers
    // Labels are now correct & honest in the template ("tokens served · models · peers · from your
    // node") — these are all LOCAL counts, so no "network-wide" claim until Tier-C stats land.
    $("#provcount").textContent = models.length;
    const q = ($("#search").value || "").toLowerCase();
    const byPM = repByPeerModel();
    const trust = trustByPeerModel();
    const trustOf = (p) => trust[repKey(p.openhydra_peer_id, p.model_id)] || {};
    // One provider: its badge (and model check). Several: how many your gateway has verified.
    const modelTrust = (remote) => {
      if (!remote.length) return '<span class="mut">—</span>';
      if (remote.length === 1) { const t = trustOf(remote[0]); return trustCell(t.tier, t.check); }
      const verified = remote.filter((p) => tierOf(trustOf(p).tier) !== "A0").length;
      return `<span class="${verified ? "" : "mut"}" title="Providers your gateway has verified (A1 or higher)">${verified} of ${remote.length} verified</span>`;
    };
    // Group the announced providers by model so a model served by several peers can expand (#11b).
    const provsByModel = {};
    for (const p of (snap?.network?.known_providers || [])) (provsByModel[p.model_id] ||= []).push(p);
    // Sort: by reputation (best first), by provider count (most first), or by name.
    const repOf = (m) => modelReputation(m, byPM) ?? -1;
    const cntOf = (m) => (seenCount[m] || 0) + (local.has(m) ? 1 : 0);
    const shown = models.filter((m) => !q || m.toLowerCase().includes(q) || displayModelName(m).toLowerCase().includes(q));
    if (sortBy === "reputation") shown.sort((a, b) => repOf(b) - repOf(a) || a.localeCompare(b));
    else if (sortBy === "providers") shown.sort((a, b) => cntOf(b) - cntOf(a) || a.localeCompare(b));
    const rows = shown.map((m) => {
      const remote = provsByModel[m] || [];
      const cnt = (seenCount[m] || 0) + (local.has(m) ? 1 : 0);   // last-known count, sticky
      const tps = modelAvgTps(m);                                  // only for models we serve
      const rep = modelReputation(m, byPM);                        // earned rep of its providers (per (peer,model))
      const canExpand = remote.length > 1;                         // >1 provider → disclosure
      const open = expandedProviders.has(m);
      const caret = canExpand ? `<span class="prowtog" data-m="${esc(m)}" style="cursor:pointer;display:inline-block;width:14px;color:hsl(var(--muted-foreground));transition:transform .12s;transform:rotate(${open ? 90 : 0}deg)">▸</span>` : '<span style="display:inline-block;width:14px"></span>';
      const idle = modelIdle(m);   // W2: seen but quiet → dim, don't drop (rides gossip gaps)
      let html = `<tr class="prov" data-cat="${modelCat(m)}" data-m="${esc(m)}"${idle ? ' style="opacity:.5"' : ""}><td>${caret}${modelIcon(m)}<b title="${esc(displayModelName(m))}">${esc(displayModelName(m))}</b>${local.has(m) ? ' <span class="mut">· your machine</span>' : idle ? ' <span class="mut" style="font-size:10.5px">· idle</span>' : ""}</td><td class="num">${cnt || "—"}</td><td class="num${tps == null ? " mut" : ""}">${tps == null ? "—" : tps}</td><td>${repBadge(rep)}</td><td>${modelTrust(remote)}</td></tr>`;
      if (canExpand && open) {
        html += remote.map((p) => {
          const prep = byPM[repKey(p.openhydra_peer_id, p.model_id)];
          const t = trustOf(p);
          return `<tr class="provsub" data-cat="${modelCat(m)}" data-for="${esc(m)}"><td style="padding-left:34px"><span class="mono mut">${peerShort(p.libp2p_peer_id)}</span></td><td class="num mut">1</td><td class="num mut">—</td><td>${repBadge(prep)}</td><td>${trustCell(t.tier, t.check)}</td></tr>`;
        }).join("");
      }
      return html;
    });
    $("#provtable tbody").innerHTML = rows.join("") || `<tr><td colspan="5" class="mut">${snap ? (q ? "No models match your search." : "No models discovered yet — they appear as peers announce.") : "Connecting…"}</td></tr>`;
    const cat = $("#provchips .chip.on")?.dataset.cat || "all";
    $$("#provtable .prov, #provtable .provsub").forEach((r) => r.style.display = (cat === "all" || r.dataset.cat === cat) ? "" : "none");
    // A category with nothing in it says so instead of showing an empty table.
    if (rows.length && cat !== "all" && !$$("#provtable .prov").some((r) => r.dataset.cat === cat)) {
      $("#provtable tbody").insertAdjacentHTML("beforeend", `<tr><td colspan="5" class="mut">No ${esc(cat)} models on the network right now.</td></tr>`);
    }
    const sc = $("#provsort"); if (sc) sc.firstChild.textContent = `Sort: ${sortBy} `;
    $$("#provtable .prowtog").forEach((tog) => tog.onclick = (e) => {
      e.stopPropagation(); const m = tog.dataset.m;
      if (expandedProviders.has(m)) expandedProviders.delete(m); else expandedProviders.add(m);
      renderProviders();
    });
  }
