// SL2 + SL3: the model picker starts on Best available, and the line under an answer is plain.
import { test } from "node:test";
import assert from "node:assert/strict";
import { AUTO_MODEL, AUTO_LABEL, pickerSelection, pickerLabel } from "../ui/picker.js";
import { metaRow } from "../ui/text.js";
import { tierBadge } from "../ui/format.js";
import { markupString } from "../ui/html.js";

test("the picker starts on, and falls back to, Best available — never the first model alphabetically", () => {
  const live = ["aaa-tiny:0.5b", "qwen3:8b"];
  assert.equal(pickerSelection("", live), AUTO_MODEL, "first models arrive: Best available, not aaa-tiny");
  assert.equal(pickerSelection("gone:1b", live), AUTO_MODEL, "a model that dropped out");
  assert.equal(pickerSelection("qwen3:8b", live), "qwen3:8b", "a live choice is kept");
  assert.equal(pickerSelection(AUTO_MODEL, live), AUTO_MODEL);
  assert.equal(pickerSelection(AUTO_MODEL, []), "", "nothing live: nothing selected");
  assert.equal(pickerLabel(AUTO_MODEL, (x) => x), AUTO_LABEL);
  // A peer can name its model "Best available", but not with the mark (model ids are plain ASCII).
  assert.match(AUTO_LABEL, /^[^\x00-\x7f] Best available$/);
  assert.notEqual(pickerLabel("Best available", (x) => x), AUTO_LABEL);
  assert.equal(pickerLabel("qwen3:8b", (x) => x.toUpperCase()), "QWEN3:8B");
});

test("under an answer: model, time, trust; numbers on hover; nothing for a missing value", () => {
  const m = markupString(metaRow({ model: "qwen3:8b", tok: 42, claimed: null, tps: 61, rtt: 12, tier: "A1-reputation-redundant", at: "9:25 PM" }));
  assert.match(m, /<span>qwen3:8b<\/span><span>9:25 PM<\/span>/);
  assert.match(m, />Trusted</);
  assert.match(m, /class="num det">42 tokens · 61 tok\/s · 12 ms network</);
  // A reply from this machine: no tier, no speed, no network time — and no "— tok/s".
  const local = markupString(metaRow({ model: "llama3.2:1b", tok: 5, claimed: null, tps: "—", rtt: "—", tier: null, at: "9:26 PM" }));
  assert.doesNotMatch(local, /—|tok\/s|network|badge/);
  assert.match(local, /5 tokens/);
});

test("a corrected bill reads in words", () => {
  const m = markupString(metaRow({ model: "qwen3:8b", tok: 61, claimed: 120, tps: 50, rtt: 9, tier: "A0", at: "9:27 PM" }));
  assert.match(m, /Charged 61 tokens \(provider asked for 120, corrected\)/);
  assert.doesNotMatch(m, /provider claimed/);
});

test("trust labels: New, Trusted, Verified model, Verified hardware", () => {
  const label = (t) => /<span[^>]*>([^<]*)<\/span>/.exec(markupString(tierBadge(t)))[1];
  assert.deepEqual(["A0", "A1-reputation-redundant", "A2-logprob", "A3-hardware", undefined].map(label), ["New", "Trusted", "Verified model", "Verified hardware", "New"]);
});
