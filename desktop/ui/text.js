// Reply rendering helpers: syntax highlight, fenced-code split, inline media, reasoning split,
// metadata row. Pure (build DOM / return strings); no app state.
import { esc } from "./dom";

// #3/#4: classify a markdown media src as image | video | audio (data: URL or media-file URL),
// "" if it isn't media (a regular link stays as text). Drives which element we render.
export function mediaKind(src) {
  const s = (src || "").trim();
  const dm = s.match(/^data:([a-z]+)\/[a-z0-9.+-]+/i);
  let m = dm ? dm[1].toLowerCase() : "";
  if (!m) { const ext = (s.match(/\.([a-z0-9]+)(?:[?#]|$)/i) || [, ""])[1].toLowerCase();
    m = ["png","jpg","jpeg","gif","webp","svg"].includes(ext) ? "image"
      : ["mp4","webm","mov","m4v"].includes(ext) ? "video"
      : ["mp3","wav","flac","ogg","m4a"].includes(ext) ? "audio" : ""; }
  return (m === "image" || m === "video" || m === "audio") ? m : "";
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
  cp.style.cssText = "cursor:pointer;color:hsl(var(--muted));background:none;border:none;padding:0;font:inherit";
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
  dl.style.cssText = "cursor:pointer;color:hsl(var(--muted));text-decoration:none";
  bar.appendChild(cp); bar.appendChild(dl); wrap.appendChild(bar);
  return wrap;
}
const KW = new Set("fn let mut pub use impl struct enum trait match if else for while loop return async await move def class import from as with try except lambda yield pass raise function const var new typeof export default package func go defer chan interface map range type switch case break continue static void int float double char bool".split(" "));
export function hl(code) { let out = "", i = 0; const push = (c, s) => out += c ? `<span class="${c}">${esc(s)}</span>` : esc(s); while (i < code.length) { const rest = code.slice(i); let m; if ((m = rest.match(/^(\/\/|#(?!\[)|--)[^\n]*/))) push("tk-com", m[0]); else if ((m = rest.match(/^\/\*[\s\S]*?\*\//))) push("tk-com", m[0]); else if ((m = rest.match(/^("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|`(?:[^`\\]|\\.)*`)/))) push("tk-str", m[0]); else if ((m = rest.match(/^\b\d[\d_]*(\.\d+)?\b/))) push("tk-num", m[0]); else if ((m = rest.match(/^[A-Za-z_]\w*/))) { if (KW.has(m[0])) push("tk-kw", m[0]); else if (code[i + m[0].length] === "(") push("tk-fn", m[0]); else push(null, m[0]); } else { push(null, code[i]); i++; continue; } i += m[0].length; } return out; }
export function parseFences(t) { const p = [], re = /```([\w+-]*)\n([\s\S]*?)```/g; let last = 0, m; while ((m = re.exec(t))) { if (m.index > last) p.push({ prose: t.slice(last, m.index).trim() }); p.push({ lang: m[1] || "code", code: m[2] }); last = re.lastIndex; } if (last < t.length) p.push({ prose: t.slice(last).trim() }); return p.filter((x) => x.code != null || x.prose); }
export function metaRow(m) { return `<div class="msgmeta"><span>${esc(m.model)}</span><span class="num">${m.tok} tok</span><span class="num" title="Engine generation speed — decode tokens per second, measured on the provider">${m.tps} tok/s</span><span class="num">${m.rtt} ms RTT</span><span>${m.at}</span></div>`; }
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
