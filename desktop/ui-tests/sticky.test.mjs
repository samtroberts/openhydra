// The sticky model list drops a model the network says just left (30 Oct: it stayed 180 s).
import { test } from "node:test";
import assert from "node:assert/strict";
import { noteSnapshot } from "../ui/sticky.js";

test("a model the network reports gone leaves the list at once", () => {
  const seen = {};
  noteSnapshot(seen, ["a", "b"], [], 1000);
  assert.deepEqual(Object.keys(seen).sort(), ["a", "b"]);
  // b's provider was killed: b is no longer known and is reported withdrawn.
  assert.deepEqual(noteSnapshot(seen, ["a"], ["b"], 2000), ["b"]);
  assert.deepEqual(Object.keys(seen), ["a"]);
  assert.equal(seen.a, 2000);
});

test("only withdrawn models skip the sticky timer, and never one still listed", () => {
  const seen = { a: 1000, c: 1000 };
  // c merely dropped out of this snapshot (flicker): it stays, on its timer.
  assert.deepEqual(noteSnapshot(seen, ["a"], [], 2000), []);
  assert.equal(seen.c, 1000);
  // a is reported withdrawn by one role but another provider still lists it: it stays.
  assert.deepEqual(noteSnapshot(seen, ["a"], ["a"], 3000), []);
  assert.equal(seen.a, 3000);
  // An older agent reports no withdrawals at all.
  assert.deepEqual(noteSnapshot(seen, ["a"], undefined, 4000), []);
  // A withdrawn model never seen here changes nothing.
  assert.deepEqual(noteSnapshot(seen, ["a"], ["zzz"], 5000), []);
});
