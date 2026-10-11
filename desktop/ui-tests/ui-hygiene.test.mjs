// Source rules that keep the UI safe by construction. Text in the UI comes from network peers (model
// names, answers, peer ids), so markup is built only with html`` (desktop/ui/html.js), which escapes
// every value. Run with `node --test desktop/ui-tests/*.test.mjs`.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const uiDir = join(here, "..", "ui");
// bridge.js is the browser-preview mock (never part of the app's data path); html.js is the one
// module that writes markup.
const files = readdirSync(uiDir).filter((n) => n.endsWith(".js") && n !== "bridge.js");
const src = Object.fromEntries(files.map((f) => [f, readFileSync(join(uiDir, f), "utf8")]));
const lineOf = (text, index) => text.slice(0, index).split("\n").length;

function findAll(re, { except = [] } = {}) {
  const hits = [];
  for (const [f, text] of Object.entries(src)) {
    if (except.includes(f)) continue;
    for (const m of text.matchAll(re)) hits.push(`${f}:${lineOf(text, m.index)}: ${m[0].slice(0, 80)}`);
  }
  return hits;
}

// The literal parts of every html`…` template in the UI (the text between ${…}), with file and line.
function htmlTemplates() {
  const out = [];
  for (const [f, text] of Object.entries(src)) {
    let i = 0;
    while ((i = text.indexOf("html`", i)) !== -1) {
      const start = i + 5;
      // Walk to the closing backtick, skipping ${…} (with nested templates inside).
      let j = start, depth = 0, literal = "";
      while (j < text.length) {
        const c = text[j];
        if (depth === 0 && c === "`") break;
        if (depth === 0 && c === "$" && text[j + 1] === "{") { depth = 1; literal += "${}"; j += 2; continue; }
        if (depth > 0) {
          if (c === "{") depth++;
          else if (c === "}") depth--;
          else if (c === "`") { // a nested template inside ${…}: skip it whole
            j++;
            let d2 = 0;
            while (j < text.length && !(text[j] === "`" && d2 === 0)) {
              if (text[j] === "$" && text[j + 1] === "{") { d2++; j += 2; continue; }
              if (d2 > 0 && text[j] === "}") d2--;
              j++;
            }
          }
          j++;
          continue;
        }
        literal += c;
        j++;
      }
      out.push({ where: `${f}:${lineOf(text, i)}`, literal });
      i = j + 1;
    }
  }
  return out;
}

test("nothing but html.js writes markup", () => {
  const hits = findAll(/\.(innerHTML|outerHTML)\s*=|insertAdjacentHTML\s*\(|document\.write/g, { except: ["html.js"] });
  assert.deepEqual(hits, [], `use setHTML / insertHTML with html\`\` markup instead:\n${hits.join("\n")}`);
});

test("no hand-rolled escaping is left to forget", () => {
  const hits = findAll(/\b(esc|escapeHtml)\s*\(/g, { except: ["html.js"] });
  assert.deepEqual(hits, [], hits.join("\n"));
});

test("only the trusted producers wrap raw strings as markup", () => {
  // markdown.js (escape-first renderer, fuzzed in markdown.test.mjs) and icons.js (constant SVGs).
  const hits = findAll(/\btrustedMarkup\s*\(/g, { except: ["html.js", "markdown.js", "icons.js"] });
  assert.deepEqual(hits, [], hits.join("\n"));
});

test("templates have no inline handlers, unquoted values or unreviewed URLs", () => {
  const bad = [];
  const tpls = htmlTemplates();
  assert.ok(tpls.length > 50, `found ${tpls.length} html templates; the scanner may be broken`);
  for (const { where, literal } of tpls) {
    // An inline event handler would run whatever the attribute holds (the CSP blocks it too).
    if (/\son[a-z]+\s*=/i.test(literal)) bad.push(`${where}: inline on*= handler`);
    // A value must sit inside quotes: escaping can't stop a space ending an unquoted attribute.
    if (/=\s*\$\{\}/.test(literal)) bad.push(`${where}: unquoted attribute value`);
    // A URL attribute takes schemes escaping can't neutralise (javascript:); none are built here.
    if (/\b(href|src|action|formaction|srcdoc)\s*=\s*["']?[^"'>]*\$\{\}/i.test(literal)) bad.push(`${where}: interpolated URL attribute`);
    // A value inside <script>/<style> is code, not text.
    if (/<(script|style)\b/i.test(literal)) bad.push(`${where}: <script>/<style> in a template`);
  }
  assert.deepEqual(bad, [], bad.join("\n"));
});

test("markup is never concatenated as a string", () => {
  // `"<p>" + html\`…\`` would throw at runtime (markup refuses to become a string); catch it here.
  const hits = findAll(/\+\s*html`|`\s*\+\s*(?=html`)/g);
  assert.deepEqual(hits, [], hits.join("\n"));
});
