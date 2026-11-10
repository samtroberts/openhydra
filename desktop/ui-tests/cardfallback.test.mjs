// The app's side of a card's provider that can't serve: what it sends, when it asks, what it says.
import test from "node:test";
import assert from "node:assert/strict";
import { fallbackToSend, shouldOfferFallback, offerText, fellBackNote, cardVersionLine, selectedFallback, declinedText, FALLBACK_OPTIONS } from "../ui/cardfallback.js";

test("the app asks unless the user saved a choice", () => {
  assert.equal(fallbackToSend("qwen", {}), "ask");
  assert.equal(fallbackToSend("qwen", undefined), "ask");
  assert.equal(fallbackToSend("qwen", { qwen: "never" }), "never");
  assert.equal(fallbackToSend("qwen", { qwen: "same-version" }), "same-version");
  assert.equal(fallbackToSend("qwen", { qwen: "always" }), "ask", "an unknown value is not sent");
});

test("another provider is offered only for a public card with same-version providers, when asking", () => {
  assert.equal(shouldOfferFallback({ private: false, alternatives: 2, fallback: "ask", reason: "offline" }), true);
  assert.equal(shouldOfferFallback({ private: true, alternatives: 0, fallback: "never" }), false, "private: never");
  assert.equal(shouldOfferFallback({ private: true, alternatives: 3, fallback: "ask" }), false, "private even if a count came back");
  assert.equal(shouldOfferFallback({ private: false, alternatives: 0, fallback: "ask" }), false, "nobody else serves it");
  assert.equal(shouldOfferFallback({ private: false, alternatives: 1, fallback: "never" }), false, "the user said never");
  assert.equal(shouldOfferFallback(undefined), false);
});

test("the question says why, how many, and that another machine reads the prompt", () => {
  const t = offerText("Qwen2.5-7B", { reason: "overloaded", alternatives: 1 });
  assert.match(t.body, /is busy/);
  assert.match(t.body, /1 other provider announces/);
  assert.match(t.body, /their own claim/);
  assert.match(t.body, /can read your prompt/);
  assert.match(offerText("m", { reason: "offline", alternatives: 3 }).body, /is offline[\s\S]*3 other providers announce/);
});

test("a reply another provider served says so; a card's version reads plainly", () => {
  assert.equal(fellBackNote(null), "");
  assert.equal(fellBackNote({ fell_back: true, reason: "offline" }), "Served by another provider of the same version (the card's provider is offline).");
  assert.equal(cardVersionLine("qwen2.5/7b/q4_k_m/abcd"), "qwen2.5 · 7B · Q4_K_M");
  assert.equal(cardVersionLine(""), "");
  assert.equal(cardVersionLine("weird"), "");
});

test("the selector shows the real default, and saying no is said plainly", () => {
  assert.equal(selectedFallback("qwen", {}), "", "nothing saved: the default, not 'Ask me'");
  assert.equal(selectedFallback("qwen", { qwen: "ask" }), "ask");
  assert.equal(selectedFallback("qwen", { qwen: "bogus" }), "");
  assert.match(FALLBACK_OPTIONS[0][1], /other apps use another provider/, "the default says what other apps get");
  assert.equal(declinedText("Qwen", { reason: "overloaded" }), "Qwen: the provider from your card is busy. Not sent to another provider.");
});
