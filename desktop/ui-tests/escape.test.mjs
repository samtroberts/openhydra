// HTML escaping. Model names and other strings in the UI come from network peers, so every value
// put into markup must be escaped, including inside quoted attributes (`title="${esc(x)}"`), where
// an unescaped `"` would let the value add its own attributes. Run with `node --test desktop/ui-tests`.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { esc, escapeHtml } from "../ui/dom.js";

const here = dirname(fileURLToPath(import.meta.url));
const uiDir = join(here, "..", "ui");

test("esc and escapeHtml escape all five HTML-special characters", () => {
  for (const f of [esc, escapeHtml]) {
    assert.equal(f(`a&b<c>d"e'f`), "a&amp;b&lt;c&gt;d&quot;e&#39;f");
    assert.equal(f("plain-model:7b"), "plain-model:7b");
    assert.equal(f(42), "42");
  }
});

test("an escaped hostile model name can't leave its attribute", () => {
  const hostile = `x" onclick="alert(1)" data-x="`;
  const html = `<tr data-m="${esc(hostile)}">`;
  assert.equal((html.match(/"/g) || []).length, 2, "only the attribute's own quotes remain");
  assert.ok(!/"\s+on\w+=/.test(html), "no handler attribute after a real quote");
});

// Expressions put into a quoted attribute without esc(): each was checked to be a number or a
// constant string chosen by the UI itself, never data from the backend or a peer. A new
// interpolation must use esc() or be reviewed and added here.
const SAFE_ATTR = new Set([
  "chart.js|cx.toFixed(1)", "chart.js|H - 5", "chart.js|x0.toFixed(1)", "chart.js|yTop.toFixed(1)",
  "chart.js|barW.toFixed(1)", "chart.js|Math.max(0.5, h).toFixed(1)", "chart.js|colorOf(id)",
  'chart.js|key === "u" ? 0.55 : 1', "chart.js|r", "chart.js|hostKey",
  "chart.js|padL", "chart.js|padT + plotH", "chart.js|W - padR",
  "connectors.js|s", // a connector surface id ("terminal" | "app") from the app's own connector table
  "providers.js|modelCat(m)", // returns one of four constant category names
  "share.js|SCOPE_META.private.title", "share.js|SCOPE_META.global.title", 'share.js|gl ? "0" : "1"',
  'share.js|gl ? "Export a public .openhydra card — a link to this model" : "Export a private .openhydra card for a swarm — only members can serve from it"',
  "share.js|swTitle",
  'swarms.js|owner ? "1" : "0"', 'swarms.js|owner ? "Delete this swarm and its group key" : "Leave this swarm"',
  "swarms.js|name", // a segmented control's constant name ("grant" | "role")
  "text.js|c", // a constant token class from hl()
]);

test("every quoted-attribute interpolation in the UI is escaped or known-safe", () => {
  const bad = [];
  for (const f of readdirSync(uiDir).filter((n) => n.endsWith(".js") && n !== "bridge.js")) {
    const src = readFileSync(join(uiDir, f), "utf8");
    const re = /=\s*(["'])\$\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\1/g;
    let m;
    while ((m = re.exec(src))) {
      const expr = m[2].trim();
      if (/^(esc|escapeHtml)\(/.test(expr)) continue;
      if (SAFE_ATTR.has(`${f}|${expr}`)) continue;
      bad.push(`${f}:${src.slice(0, m.index).split("\n").length}: ${expr}`);
    }
  }
  assert.deepEqual(bad, [], `unescaped attribute values:\n${bad.join("\n")}`);
});
