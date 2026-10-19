// Switches a keyboard and a screen reader can use (review 2026-10-14, L17).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { syncSwitch, switchName, isToggleKey, wireSwitches } from "../ui/switches.js";

// Just enough of an element for the switch code.
class El {
  constructor(classes = [], attrs = {}, parent = null) {
    this.nodeType = 1; this.cls = new Set(classes); this.attrs = { ...attrs }; this.parent = parent;
    this.children = []; this.clicks = 0; this.text = "";
    if (parent) parent.children.push(this);
    this.classList = { contains: (c) => this.cls.has(c), toggle: (c, on) => (on ? this.cls.add(c) : this.cls.delete(c)) };
  }
  hasAttribute(k) { return k in this.attrs; }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  get textContent() { return this.text; }
  matches(sel) { return sel === "label" ? this.tag === "label" : sel.startsWith(".") && this.cls.has(sel.slice(1)); }
  closest(sel) { for (let n = this; n; n = n.parent) if (n.matches(sel)) return n; return null; }
  all() { return this.children.flatMap((c) => [c, ...c.all()]); }
  querySelector(sel) { return this.all().find((n) => n.matches(sel)) || null; }
  querySelectorAll(sel) { return this.all().filter((n) => n.matches(sel)); }
  click() { this.clicks++; }
}

test("a two-state switch: role switch, a tab stop, its state, its row's label as name", () => {
  const row = new El(["field"]);
  const label = new El([], {}, row); label.tag = "label"; label.text = " Act as a relay ";
  const sw = new El(["switch"], {}, row);
  syncSwitch(sw);
  assert.equal(sw.getAttribute("role"), "switch");
  assert.equal(sw.getAttribute("tabindex"), "0");
  assert.equal(sw.getAttribute("aria-checked"), "false");
  assert.equal(sw.getAttribute("aria-label"), "Act as a relay");
  sw.classList.toggle("on", true); syncSwitch(sw);
  assert.equal(sw.getAttribute("aria-checked"), "true");
});

test("the share-all switch has three states, so it is a checkbox that can be mixed", () => {
  const wrap = new El(); wrap.tag = "label"; wrap.text = "Share everything";
  const sw = new El(["switch", "indeterminate"], { "data-tristate": "", title: "Sharing a selection — click…" }, wrap);
  syncSwitch(sw);
  assert.equal(sw.getAttribute("role"), "checkbox");
  assert.equal(sw.getAttribute("aria-checked"), "mixed");
  assert.equal(switchName(sw), "Share everything");
  // "mixed" never on a two-state switch.
  const two = new El(["switch", "indeterminate"], { title: "x" });
  syncSwitch(two);
  assert.equal(two.getAttribute("aria-checked"), "false");
});

test("a switch with no label is named by its tooltip; an existing tabindex is kept", () => {
  const sw = new El(["switch"], { title: "Announce on the network", tabindex: "-1" });
  syncSwitch(sw);
  assert.equal(sw.getAttribute("aria-label"), "Announce on the network");
  assert.equal(sw.getAttribute("tabindex"), "-1");
});

test("Space and Enter toggle the focused switch; other keys and other elements don't", () => {
  const body = new El();
  const sw = new El(["switch"], {}, body);
  const other = new El(["btn"], {}, body);
  let keydown;
  globalThis.MutationObserver = class { observe() {} };
  const doc = { body, querySelectorAll: (s) => body.querySelectorAll(s), addEventListener: (t, f) => { if (t === "keydown") keydown = f; } };
  wireSwitches(doc);
  assert.equal(sw.getAttribute("role"), "switch", "existing switches are set up at once");
  let prevented = 0;
  const press = (target, key) => keydown({ target, key, preventDefault: () => prevented++ });
  press(sw, " "); press(sw, "Enter"); press(sw, "a"); press(other, " ");
  assert.equal(sw.clicks, 2);
  assert.equal(other.clicks, 0);
  assert.equal(prevented, 2, "Space doesn't scroll the page");
  assert.ok(isToggleKey({ key: " " }) && isToggleKey({ key: "Enter" }) && !isToggleKey({ key: "Tab" }));
});

test("every switch in the page is a .switch, which the wiring covers, and the share-all one is tri-state", () => {
  const ui = new URL("../ui/", import.meta.url);
  const page = readFileSync(new URL("index.html", ui), "utf8");
  // Anything styled as a switch carries the class the wiring looks for.
  for (const m of page.matchAll(/<div[^>]*\b(data-sw|id="[a-z]*sw")[^>]*>/g)) assert.match(m[0], /class="switch[ "]/, m[0]);
  const share = readFileSync(new URL("share.js", ui), "utf8");
  assert.match(share, /sasw\.setAttribute\("data-tristate", ""\)/);
  assert.doesNotMatch(share, /setAttribute\("role", "switch"\)/, "role comes from syncSwitch");
  const main = readFileSync(new URL("main.js", ui), "utf8");
  assert.match(main, /wireSwitches\(document\)/);
  // The mono fields aren't auto-corrected.
  for (const id of ["gwport", "bootstraps", "engineendpoint"]) {
    assert.match(page, new RegExp(`id="${id}" autocapitalize="off" autocorrect="off" spellcheck="false"`), id);
  }
});
