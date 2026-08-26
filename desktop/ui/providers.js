// Network › Models view: the announced-provider table grouped by model, with sticky provider
// counts, avg TPS + earned reputation, and a per-model expand (#11b) to the individual providers.
import { $, $$, esc, peerShort } from "./dom";
import { snap } from "./state";
import { netModels, modelIdle, seenCount, displayModelName, localSharedModels } from "./models";
import { repByOpenhydra, modelReputation, modelAvgTps } from "./econ";
import { modelIcon, modelCat, repBadge } from "./format";

  const expandedProviders = new Set();
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
      let html = `<tr class="prov" data-cat="${modelCat(m)}" data-m="${esc(m)}"${idle ? ' style="opacity:.5"' : ""}><td>${caret}${modelIcon(m)}<b title="${esc(m)}">${esc(displayModelName(m))}</b>${local.has(m) ? ' <span class="mut">· your machine</span>' : idle ? ' <span class="mut" style="font-size:10.5px">· idle</span>' : ""}</td><td class="num">${cnt || "—"}</td><td class="num${tps == null ? " mut" : ""}">${tps == null ? "—" : tps}</td><td>${repBadge(rep)}</td><td class="num mut">—</td></tr>`;
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
