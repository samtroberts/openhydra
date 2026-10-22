// SL4: text in the app meets WCAG AA (4.5:1) in both themes, read from the colours in styles.css.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const css = readFileSync(new URL("../ui/styles.css", import.meta.url), "utf8");
const block = (start, end) => css.slice(css.indexOf(start), css.indexOf(end));
const vars = (b) => Object.fromEntries([...b.matchAll(/--([a-z-]+):([0-9][^;]*);/g)].map((m) => [m[1], m[2].trim()]));
const THEMES = {
  light: vars(block(":root{", "@media (prefers-color-scheme:dark)")),
  dark: vars(block(':root[data-theme="dark"]{', ':root[data-theme="light"]{')),
};

function hsl(v) {
  const [h, s, l] = v.split(/\s+/).map(parseFloat);
  const S = s / 100, L = l / 100, a = S * Math.min(L, 1 - L);
  const f = (n) => { const k = (n + h / 30) % 12; return L - a * Math.max(-1, Math.min(k - 3, 9 - k, 1)); };
  return [f(0), f(8), f(4)];
}
const blend = (fg, bg, a) => fg.map((f, i) => a * f + (1 - a) * bg[i]);
const lum = (c) => { const [r, g, b] = c.map((x) => (x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4)); return 0.2126 * r + 0.7152 * g + 0.0722 * b; };
const ratio = (a, b) => { const [hi, lo] = [lum(a), lum(b)].sort((x, y) => y - x); return (hi + 0.05) / (lo + 0.05); };

// The pairs the stylesheet draws text with (rule names in brackets).
for (const [name, t] of Object.entries(THEMES)) {
  test(`${name} theme: text is at least 4.5:1`, () => {
    const V = (k) => hsl(t[k]);
    const bg = V("background"), card = V("card");
    const pairs = {
      "brand button text [.btn.brand, .reach-pill]": [V("brand-fg"), V("brand")],
      "brand links [.md a]": [V("brand"), bg],
      "brand on its tint [.avatar, .iconbtn.on]": [V("brand"), blend(V("brand"), bg, 0.14)],
      "success badge [.badge.ok]": [V("success"), blend(V("success"), card, 0.1)],
      "warning badge [.badge.warn]": [V("warning"), blend(V("warning"), card, 0.1)],
      "danger badge [.badge.bad]": [V("danger"), blend(V("danger"), card, 0.1)],
      "success text [.up]": [V("success"), bg],
      "warning text [.nav .soon]": [V("warning"), V("sidebar")],
      "Stop button [.btn.stop]": [V("danger-fg"), V("danger")],
      "muted text": [V("muted-foreground"), bg],
      "muted text on muted": [V("muted-foreground"), V("muted")],
      "muted text in the sidebar": [V("muted-foreground"), V("sidebar")],
    };
    const low = Object.entries(pairs).map(([k, [f, b]]) => [k, ratio(f, b)]).filter(([, r]) => r < 4.5);
    assert.deepEqual(low.map(([k, r]) => `${k}: ${r.toFixed(2)}`), []);
  });
}

test("text on a coloured fill uses the theme's own foreground, not hard-coded white", () => {
  assert.match(css, /\.btn\.stop[^{]*\{[^}]*color:hsl\(var\(--danger-fg\)\)/);
  assert.match(css, /\.reach-pill \.rp\.sel\[data-to="global"\]\{[^}]*color:hsl\(var\(--brand-fg\)\)/);
});
