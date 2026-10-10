import { test } from "node:test";
import assert from "node:assert/strict";
import { HOUR_MS, hourKey, slotBuckets } from "../ui/series.js";

// Mirror of statsSeries' slot loop: which slot (0 = oldest) each bucket key falls in.
function slotOf(key, now, n, ms) {
  for (let i = n - 1; i >= 0; i--) {
    const end = now - i * ms, start = end - ms, { lo, hi } = slotBuckets(start, end);
    if (key >= lo && key < hi) return n - 1 - i;
  }
  return -1;
}

test("the newest slot includes the current hour", () => {
  const now = 1791645000000; // mid-hour
  assert.equal(slotOf(hourKey(now), now, 24, HOUR_MS), 23);
  assert.equal(slotOf(hourKey(now), now, 7, 24 * HOUR_MS), 6);
});

test("every recent bucket lands in exactly one slot", () => {
  const now = 1791645000000;
  for (const [n, ms] of [[24, HOUR_MS], [7, 24 * HOUR_MS], [30, 24 * HOUR_MS]]) {
    const span = (n * ms) / HOUR_MS;
    for (let k = hourKey(now) - span + 1; k <= hourKey(now); k++) {
      let hits = 0;
      for (let i = n - 1; i >= 0; i--) {
        const end = now - i * ms, start = end - ms, { lo, hi } = slotBuckets(start, end);
        if (k >= lo && k < hi) hits++;
      }
      assert.equal(hits, 1, `bucket ${k} in range ${n}×${ms}`);
    }
    assert.equal(slotOf(hourKey(now) - span, now, n, ms), -1, "older than the window is left out");
  }
});
