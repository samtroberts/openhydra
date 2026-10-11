// A small markdown renderer for chat answers. Escape-first: the text is HTML-escaped before any
// pattern runs, so only the tags produced here can appear. Covers what models commonly emit:
// headings, bold, italic, inline code, links, bullet and numbered lists, block quotes, tables and
// horizontal rules. Fenced code blocks are split out before this (parseFences); maths stays as text.
// Pure string → string, so it is unit-tested with `node --test`; renderMarkdownMarkup wraps the
// result as html`` markup (this renderer is one of the trusted producers).
import { trustedMarkup } from "./html.js";

const escHtml = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

const unescHtml = (s) => s.replace(/&quot;/g, '"').replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&");

// Links: only http(s) and mailto; anything else (javascript:, data:, file:) stays as plain text.
function safeHref(raw) {
  const u = raw.trim();
  return /^(https?:\/\/|mailto:)/i.test(u) ? u : null;
}

// Inline markup on already-escaped text. Inline code is cut out first so nothing inside it is
// touched, then restored at the end.
export function inline(escaped) {
  // Code spans and links are cut out first (as numbered placeholders) so emphasis never reaches
  // inside them — not into code, and not into a link's URL or attributes — then put back.
  const held = [];
  const hold = (html) => { held.push(html); return `\u0000${held.length - 1}\u0000`; };
  let s = escaped.replace(/`([^`\n]{1,2000})`/g, (_, c) => hold(`<code>${c}</code>`));
  // Bounded lengths, and no "[" or "(" inside, so a long run of brackets can't make this quadratic.
  s = s.replace(/\[([^\]\n[]{1,500})\]\(([^)\s[(]{1,2048})\)/g, (m, text, href) => {
    const h = safeHref(unescHtml(href));
    return h ? hold(`<a href="${escHtml(h)}" target="_blank" rel="noopener noreferrer">`) + text + hold("</a>") : m;
  });
  s = s.replace(/\*\*(?=\S)([^*\n]+?)\*\*/g, "<strong>$1</strong>").replace(/__(?=\S)([^_\n]+?)__/g, "<strong>$1</strong>");
  s = s.replace(/(^|[^*\w])\*(?=\S)([^*\n]+?)\*(?!\w)/g, "$1<em>$2</em>").replace(/(^|[^_\w])_(?=\S)([^_\n]+?)_(?!\w)/g, "$1<em>$2</em>");
  s = s.replace(/~~(?=\S)([^~\n]+?)~~/g, "<del>$1</del>");
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => held[+i]);
}

// A table's header separator (| --- | :---: |): checked cell by cell, without a nested regex.
const isTableSep = (l) => {
  const t = l.trim(); if (!t.includes("-")) return false;
  const cs = t.replace(/^\|/, "").replace(/\|$/, "").split("|");
  return cs.length > 0 && cs.every((c) => /^:?-{2,}:?$/.test(c.trim()));
};
const cells = (l) => l.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());

export function renderMarkdown(text) {
  // NUL is the placeholder marker in inline(); a model's own NULs are dropped.
  const lines = escHtml(String(text || "").replace(/\u0000/g, "")).replace(/\r\n?/g, "\n").split("\n");
  const out = [];
  let i = 0, para = [];
  const flushPara = () => { if (para.length) { out.push(`<p>${para.map(inline).join("<br>")}</p>`); para = []; } };
  while (i < lines.length) {
    const l = lines[i];
    if (!l.trim()) { flushPara(); i++; continue; }
    let m;
    if ((m = /^(#{1,6})\s+(.*)$/.exec(l))) { flushPara(); const n = Math.min(m[1].length + 2, 6); out.push(`<h${n}>${inline(m[2])}</h${n}>`); i++; continue; }
    if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(l)) { flushPara(); out.push("<hr>"); i++; continue; }
    if (l.includes("|") && i + 1 < lines.length && isTableSep(lines[i + 1])) {
      flushPara();
      const head = cells(l); i += 2;
      const rows = [];
      while (i < lines.length && lines[i].includes("|") && lines[i].trim()) { rows.push(cells(lines[i])); i++; }
      out.push(`<table><thead><tr>${head.map((c) => `<th>${inline(c)}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${head.map((_, k) => `<td>${inline(r[k] ?? "")}</td>`).join("")}</tr>`).join("")}</tbody></table>`);
      continue;
    }
    if (/^\s*&gt;\s?/.test(l)) {
      flushPara();
      const q = [];
      while (i < lines.length && /^\s*&gt;\s?/.test(lines[i])) { q.push(lines[i].replace(/^\s*&gt;\s?/, "")); i++; }
      out.push(`<blockquote>${q.map(inline).join("<br>")}</blockquote>`);
      continue;
    }
    if ((m = /^\s*([-*+]|\d+[.)])\s+/.exec(l))) {
      flushPara();
      const ordered = /\d/.test(m[1]);
      const re = ordered ? /^\s*\d+[.)]\s+(.*)$/ : /^\s*[-*+]\s+(.*)$/;
      const items = [];
      while (i < lines.length && re.test(lines[i])) {
        let item = re.exec(lines[i])[1]; i++;
        // Indented continuation lines belong to the item.
        while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !/^\s*([-*+]|\d+[.)])\s+/.test(lines[i])) { item += " " + lines[i].trim(); i++; }
        items.push(`<li>${inline(item)}</li>`);
      }
      const start = ordered ? parseInt(m[1], 10) : 1;
      out.push(ordered ? `<ol${start !== 1 ? ` start="${start}"` : ""}>${items.join("")}</ol>` : `<ul>${items.join("")}</ul>`);
      continue;
    }
    para.push(l); i++;
  }
  flushPara();
  return out.join("");
}

/** renderMarkdown as html`` markup, for setHTML. */
export const renderMarkdownMarkup = (text) => trustedMarkup(renderMarkdown(text));
