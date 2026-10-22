// SL4: the ⌘K menu filters as you type.
import { test } from "node:test";
import assert from "node:assert/strict";
import { nextQuery, matches } from "../ui/menufilter.js";

const key = (k, mods = {}) => ({ key: k, metaKey: false, ctrlKey: false, altKey: false, ...mods });

test("typing builds the query; Backspace takes a letter back; other keys aren't for the filter", () => {
  let q = "";
  for (const k of "set") q = nextQuery(q, key(k));
  assert.equal(q, "set");
  assert.equal(nextQuery(q, key("Backspace")), "se");
  assert.equal(nextQuery("", key(" ")), null, "a leading space");
  assert.equal(nextQuery("go", key(" ")), "go ");
  for (const k of ["Enter", "ArrowDown", "Escape", "Tab"]) assert.equal(nextQuery("x", key(k)), null, k);
  assert.equal(nextQuery("x", key("k", { metaKey: true })), null, "⌘K itself");
});

test("a label matches when it holds every word of the query, in any case", () => {
  assert.ok(matches("Go to Settings", "set"));
  assert.ok(matches("Go to Connect apps", "go con"));
  assert.ok(!matches("Go to Settings", "share"));
  assert.ok(matches("Go to Settings", ""));
});
