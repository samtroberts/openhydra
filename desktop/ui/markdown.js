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
  // A double-backtick span first (it may hold a single backtick: ``a `b` c``), then single ones.
  let s = escaped.replace(/``(?!`) ?((?:[^`\n]|`(?!`)){1,1000}?) ?``(?!`)/g, (_, c) => hold(`<code>${c}</code>`));
  s = s.replace(/`([^`\n]{1,2000})`/g, (_, c) => hold(`<code>${c}</code>`));
  // Bounded lengths, and no "[" inside, so a long run of brackets can't make this quadratic. The URL
  // may hold one level of balanced parentheses (Wikipedia links); an optional "title" follows it.
  s = s.replace(/\[([^\]\n[]{1,500})\]\(((?:[^()\s[]|\([^()\s[]{0,500}\)){1,2048})(?:\s+&quot;((?:(?!&quot;)[^\n]){0,300})&quot;)?\)/g, (m, text, href, title) => {
    const h = safeHref(unescHtml(href));
    const t = title ? ` title="${escHtml(unescHtml(title))}"` : "";
    return h ? hold(`<a href="${escHtml(h)}"${t} target="_blank" rel="noopener noreferrer">`) + text + hold("</a>") : m;
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
// A row's cells: split on "|" outside code spans (a "|" in `a | b` is part of the code).
const cells = (l) => {
  const t = l.trim().replace(/^\|/, "").replace(/\|$/, "");
  const out = []; let cur = "", inCode = false;
  for (const ch of t) {
    if (ch === "`") inCode = !inCode;
    if (ch === "|" && !inCode) { out.push(cur.trim()); cur = ""; continue; }
    cur += ch;
  }
  out.push(cur.trim());
  return out;
};

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
    // A closing run of #s ("## Title ##") isn't part of the heading.
    if ((m = /^(#{1,6})\s+(.*)$/.exec(l))) { flushPara(); const n = Math.min(m[1].length + 2, 6); out.push(`<h${n}>${inline(m[2].replace(/(^|\s+)#+\s*$/, ""))}</h${n}>`); i++; continue; }
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

// Split a reply into prose and fenced code blocks, CommonMark-style: a fence is three or more
// backticks or tildes (up to three spaces in), with an optional info string whose first word is the
// language ("``` python" too); it closes on a line of the same character, at least as long, and
// nothing else. A fence that never closes runs to the end, as in a reply that was stopped or cut off,
// so its text stays code instead of being read as markdown (`# comment` as a heading, `__name__` as
// bold). CRLF line ends count as line ends.
export function parseFences(t) {
  const lines = (t || "").replace(/\r\n?/g, "\n").split("\n");
  const parts = [];
  let prose = [], code = null, fence = null, lang = "";
  const flushProse = () => { const p = prose.join("\n").trim(); if (p) parts.push({ prose: p }); prose = []; };
  for (const line of lines) {
    if (fence) {
      const close = line.match(/^ {0,3}(`{3,}|~{3,})\s*$/);
      if (close && close[1][0] === fence[0] && close[1].length >= fence.length) {
        parts.push({ lang: lang || "code", code: code.join("\n") + "\n" });
        fence = null; code = null;
      } else {
        code.push(line);
      }
      continue;
    }
    const open = line.match(/^ {0,3}(`{3,}|~{3,})\s*([^\s`]*)[^`]*$/) || line.match(/^ {0,3}(~{3,})\s*(\S*).*$/);
    if (open) {
      flushProse();
      fence = open[1]; lang = (open[2] || "").replace(/[^\w+#.-]/g, ""); code = [];
      continue;
    }
    prose.push(line);
  }
  if (fence) parts.push({ lang: lang || "code", code: code.join("\n") + (code.length ? "\n" : "") });
  else flushProse();
  return parts.filter((x) => x.code != null || x.prose);
}
