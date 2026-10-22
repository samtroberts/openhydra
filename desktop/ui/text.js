// Reply rendering helpers: syntax highlight, fenced-code split, inline media, reasoning split,
// metadata row. Pure (build DOM / return html`` markup); no app state.
import { html } from "./html.js";
import { tierBadge } from "./format.js";

// #3/#4: classify a markdown media src as image | video | audio, "" if it isn't media (it then
// stays as text). Only inline `data:` media counts: answers come from network providers, and a
// remote URL would load on its own (revealing your IP) or carry another scheme into the download link.
export function mediaKind(src) {
  const dm = (src || "").trim().match(/^data:(image|video|audio)\/[a-z0-9.+-]+[;,]/i);
  return dm ? dm[1].toLowerCase() : "";
}
// #4: render inline media; #3: copy + download controls under it. Returns null for non-media.
export function mediaEl(src, label) {
  const kind = mediaKind(src); if (!kind) return null;
  const wrap = document.createElement("div"); wrap.style.cssText = "margin:6px 0;max-width:100%";
  const el = document.createElement(kind === "image" ? "img" : kind);
  el.src = src; if (kind !== "image") el.controls = true;
  el.style.cssText = "max-width:100%;border-radius:10px;border:1px solid hsl(var(--border));display:block;width:auto;height:auto";
  wrap.appendChild(el);
  const bar = document.createElement("div"); bar.style.cssText = "display:flex;gap:12px;margin-top:4px;font-size:12px";
  const cp = document.createElement("button"); cp.textContent = "⎘ copy";
  cp.style.cssText = "cursor:pointer;color:hsl(var(--muted-foreground));background:none;border:none;padding:0;font:inherit";
  cp.onclick = async () => {
    try { // copy the image itself to the clipboard when possible; else copy the src/data-URL
      if (kind === "image" && src.startsWith("data:") && window.ClipboardItem) {
        const blob = await (await fetch(src)).blob();
        await navigator.clipboard.write([new ClipboardItem({ [blob.type]: blob })]);
      } else { await navigator.clipboard.writeText(src); }
    } catch { try { await navigator.clipboard.writeText(src); } catch {} }
    cp.textContent = "✓ copied"; setTimeout(() => cp.textContent = "⎘ copy", 1400);
  };
  const ext = kind === "image" ? "png" : kind === "video" ? "mp4" : "flac";
  const dl = document.createElement("a"); dl.textContent = "⭳ download";
  dl.href = src; dl.download = (label && label.trim()) || `openhydra-${kind}.${ext}`;
  dl.style.cssText = "cursor:pointer;color:hsl(var(--muted-foreground));text-decoration:none";
  bar.appendChild(cp); bar.appendChild(dl); wrap.appendChild(bar);
  return wrap;
}
const KW = new Set("fn let mut pub use impl struct enum trait match if else for while loop return async await move def class import from as with try except lambda yield pass raise function const var new typeof export default package func go defer chan interface map range type switch case break continue static void int float double char bool".split(" "));
export function hl(code) { const out = []; let i = 0; const push = (c, s) => out.push(c ? html`<span class="${c}">${s}</span>` : s); while (i < code.length) { const rest = code.slice(i); let m; if ((m = rest.match(/^(\/\/|#(?!\[)|--)[^\n]*/))) push("tk-com", m[0]); else if ((m = rest.match(/^\/\*[\s\S]*?\*\//))) push("tk-com", m[0]); else if ((m = rest.match(/^("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|`(?:[^`\\]|\\.)*`)/))) push("tk-str", m[0]); else if ((m = rest.match(/^\b\d[\d_]*(\.\d+)?\b/))) push("tk-num", m[0]); else if ((m = rest.match(/^[A-Za-z_]\w*/))) { if (KW.has(m[0])) push("tk-kw", m[0]); else if (code[i + m[0].length] === "(") push("tk-fn", m[0]); else push(null, m[0]); } else { push(null, code[i]); i++; continue; } i += m[0].length; } return html`${out}`; }
// Fenced code is split out by the markdown module's parser (tested there).
export { parseFences } from "./markdown.js";
// Under an answer (SL3): the model, the time and the provider's trust; the numbers (tokens, speed,
// network time) appear on hover, and a value that's missing isn't shown at all. A corrected bill is
// always shown, in words.
export function metaRow(m) {
  if (m.stopped) return html`<div class="msgmeta"><span>${m.model}</span><span>${m.stopped === true ? "stopped" : m.stopped}</span><span>${m.at}</span></div>`;
  const known = (v) => v != null && v !== "" && v !== "—";
  const details = [known(m.tok) && `${m.tok} tokens`, known(m.tps) && `${m.tps} tok/s`, known(m.rtt) && `${m.rtt} ms network`].filter(Boolean).join(" · ");
  const charged = m.claimed != null && known(m.tok)
    ? html`<span class="num" title="The provider billed more tokens than its answer holds; your gateway pays only what was delivered">Charged ${m.tok} tokens (provider asked for ${m.claimed}, corrected)</span>`
    : "";
  return html`<div class="msgmeta"><span>${m.model}</span>${known(m.at) ? html`<span>${m.at}</span>` : ""}${m.tier ? html`<span title="How far your gateway has verified the provider that answered">${tierBadge(m.tier)}</span>` : ""}${charged}${details ? html`<span class="num det">${details}</span>` : ""}</div>`;
}
// Reasoning models (Qwen3, DeepSeek-R1, …) emit a chain-of-thought. Some inline it as
// <think>…</think> inside content; some engines (LM Studio serving MLX) strip it and hand
// back an EMPTY content. Split any inline thinking out of the answer so the chat shows the
// real answer (or a clear note) instead of a blank bubble.
export function splitThink(raw) {
  raw = raw || ""; let reasoning = "";
  // Well-formed <think>…</think> pairs.
  raw = raw.replace(/<think>([\s\S]*?)<\/think>/gi, (_, t) => { reasoning += t + "\n"; return ""; });
  // Reasoning models (Qwen3, DeepSeek-R1, …) whose chat template PRE-FILLS the opening
  // <think> in the prompt emit a completion that *starts* with the chain-of-thought and
  // closes it with a lone </think> (no opening tag). If a </think> remains with no <think>
  // before it, treat that leading block as reasoning — else it (and the stray tag) leak.
  const close = raw.search(/<\/think>/i);
  if (close !== -1 && !/<think>/i.test(raw.slice(0, close))) {
    reasoning += raw.slice(0, close) + "\n";
    raw = raw.slice(close).replace(/<\/think>/i, "");
  }
  // Unclosed <think>… (still thinking / truncated stream).
  raw = raw.replace(/<think>([\s\S]*)$/i, (_, t) => { reasoning += t; return ""; });
  return { answer: raw.trim(), reasoning: reasoning.trim() };
}
