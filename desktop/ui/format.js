// Pure formatters + model-family badge. No app state.
import { esc } from "./dom";

// ── model family badge ──
export const FAMILY = [[/qwen/i, "qwen"], [/llama|tinyllama/i, "llama"], [/gemma/i, "gemma"], [/mi(s|x)tral/i, "mistral"], [/phi/i, "phi"], [/deepseek/i, "deepseek"], [/flux|sdxl|stable/i, "flux"], [/nomic|embed/i, "nomic"]];
export const FAMSTYLE = { qwen: ["Q", "#6d49c4"], llama: ["L", "#0866ff"], gemma: ["G", "#1a73e8"], mistral: ["M", "#fa5210"], phi: ["φ", "#12a3a6"], deepseek: ["DS", "#4d6bfe"], flux: ["F", "#111418"], nomic: ["N", "#127a6b"] };
export function modelIcon(id) { const k = (FAMILY.find(([rx]) => rx.test(id || "")) || [])[1]; const s = k && FAMSTYLE[k]; if (s) return `<span class="micon" style="background:${s[1]}">${s[0]}</span>`; const hues = ["#64748b", "#0891b2", "#7c3aed", "#db2777", "#ca8a04", "#059669"]; return `<span class="micon" style="background:${hues[(String(id).charCodeAt(0) || 0) % hues.length]}">${esc((String(id)[0] || "?").toUpperCase())}</span>`; }
export const modelCat = (m) => /coder|code/i.test(m) ? "code" : /flux|sdxl|stable|image/i.test(m) ? "image" : /embed|nomic/i.test(m) ? "embeddings" : "chat";

export function fmtUptime(s) { if (s == null) return "—"; s = Math.floor(s); if (s < 60) return s + "s"; const m = Math.floor(s / 60); if (m < 60) return m + "m"; const h = Math.floor(m / 60); if (h < 24) return `${h}h ${m % 60}m`; const d = Math.floor(h / 24); return `${d}d ${h % 24}h`; }
export function repBadge(v) { if (v == null) return '<span class="mut">—</span>'; v = Math.round(v); const cls = v >= 66 ? "ok" : v >= 40 ? "warn" : "secondary"; return `<span class="badge ${cls}">${v}</span>`; }
export function fmtNum(n) { n = Math.round(n); return n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1) + "k" : String(n); }
// Stable per-model hue for the timeline + legend.
export function modelColor(id) { let h = 0; for (const c of id) h = (h * 31 + c.charCodeAt(0)) % 360; return `hsl(${h} 60% 55%)`; }
// Relative "Nm ago" timestamp for the ledger rows.
export function relTime(ms) {
  const s = Math.max(0, (Date.now() - ms) / 1000);
  if (s < 60) return Math.round(s) + "s ago";
  const m = s / 60; if (m < 60) return Math.round(m) + "m ago";
  const h = m / 60; if (h < 24) return Math.round(h) + "h ago";
  return Math.round(h / 24) + "d ago";
}
export function fmtGB(bytes) { if (!bytes) return "—"; const gb = bytes / 1073741824; const s = gb >= 10 ? String(Math.round(gb)) : gb.toFixed(1).replace(/\.0$/, ""); return s + " GB"; }
