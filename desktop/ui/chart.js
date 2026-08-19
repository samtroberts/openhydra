// #10: self-contained inline-SVG served-vs-used timeline, stacked by model, with a 24h/7d/30d
// range selector (no chart lib — CSP-safe). Shared by the Share and Activity views; hostKey is
// "share" | "activity". The selected range per host is persisted in chartRange (module-local).
import { $, $$, esc } from "./dom";
import { store } from "./storage";
import { statsSeries } from "./stats";
import { modelColor } from "./format";

  const chartRange = store.get("oh_chartrange", { share: "7d", activity: "7d" });
  // Stable per-model hue for the timeline + legend.
  // #10: inline-SVG served-vs-used timeline, stacked by model, with a 24h/7d/30d range selector.
  // Self-contained (no chart lib — CSP-safe). `hostKey` = "share" | "activity".
  export function renderChart(hostSel, hostKey) {
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
