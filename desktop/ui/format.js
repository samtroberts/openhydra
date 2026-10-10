// Pure formatters + model-family badge. No app state.
import { esc, escapeHtml } from "./dom";

// ── model family badge ──
const FAMILY = [[/qwen/i, "qwen"], [/llama|tinyllama/i, "llama"], [/gemma/i, "gemma"], [/mi(s|x)tral/i, "mistral"], [/phi/i, "phi"], [/deepseek/i, "deepseek"], [/flux|sdxl|stable/i, "flux"], [/nomic|embed/i, "nomic"]];
const FAMSTYLE = { qwen: ["Q", "#6d49c4"], llama: ["L", "#0866ff"], gemma: ["G", "#1a73e8"], mistral: ["M", "#fa5210"], phi: ["φ", "#12a3a6"], deepseek: ["DS", "#4d6bfe"], flux: ["F", "#111418"], nomic: ["N", "#127a6b"] };
export function modelIcon(id) { const k = (FAMILY.find(([rx]) => rx.test(id || "")) || [])[1]; const s = k && FAMSTYLE[k]; if (s) return `<span class="micon" style="background:${s[1]}">${s[0]}</span>`; const hues = ["#64748b", "#0891b2", "#7c3aed", "#db2777", "#ca8a04", "#059669"]; return `<span class="micon" style="background:${hues[(String(id).charCodeAt(0) || 0) % hues.length]}">${esc((String(id)[0] || "?").toUpperCase())}</span>`; }
export const modelCat = (m) => /coder|code/i.test(m) ? "code" : /flux|sdxl|stable|image/i.test(m) ? "image" : /embed|nomic/i.test(m) ? "embeddings" : "chat";

export function fmtUptime(s) { if (s == null) return "—"; s = Math.floor(s); if (s < 60) return s + "s"; const m = Math.floor(s / 60); if (m < 60) return m + "m"; const h = Math.floor(m / 60); if (h < 24) return `${h}h ${m % 60}m`; const d = Math.floor(h / 24); return `${d}d ${h % 24}h`; }
export function repBadge(v) { if (v == null) return '<span class="mut">—</span>'; v = Math.round(v); const cls = v >= 66 ? "ok" : v >= 40 ? "warn" : "secondary"; return `<span class="badge ${cls}">${v}</span>`; }
// Authenticity tiers in plain language, from the agent's label (e.g. "A1-reputation-redundant").
// A missing or unrecognised label reads as not verified: no track record with this gateway yet.
const TIERS = {
  A0: ["Not verified yet", "secondary", "No track record with your gateway yet: every provider starts here."],
  A1: ["Track record", "ok", "Agreed with other providers when your gateway checked it, and has built a reputation with you."],
  A2: ["Model checked", "ok", "Also passed a check that it runs the model it names."],
  A3: ["Hardware attested", "ok", "Its hardware attested to the model it runs."],
};
export function tierOf(label) { const m = /^A([0-3])(?![0-9])/.exec(String(label || "")); return m ? "A" + m[1] : "A0"; }
export function tierBadge(label) { const t = tierOf(label); const [txt, cls, tip] = TIERS[t]; return `<span class="badge ${cls}" title="${escapeHtml(t + ": " + tip)}">${esc(txt)}</span>`; }
// A provider's trust: its tier, plus its latest model check when that adds something (a failed one,
// or a pass that hasn't yet earned A2; A2 already says the model was checked).
export function trustCell(tier, check) {
  const passedA2 = tierOf(tier) === "A2" && /^pass/i.test(check || "");
  return passedA2 || !check ? tierBadge(tier) : `${tierBadge(tier)} ${checkBadge(check)}`;
}
// The latest log-prob ("model check") verdict, e.g. "pass 0.97x" / "fail 1.62x"; nothing when none.
export function checkBadge(verdict) {
  if (!verdict) return "";
  const pass = /^pass/i.test(verdict);
  return `<span class="badge ${pass ? "ok" : "bad"}" title="${escapeHtml("Model check: " + verdict)}">${pass ? "Model check passed" : "Model check failed"}</span>`;
}
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
