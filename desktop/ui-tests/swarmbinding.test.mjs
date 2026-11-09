// The one-time question about Private models with no swarm binding (enterprise phase 0.2).
import test from "node:test";
import assert from "node:assert/strict";
import { modelsToAskAbout, bindingsAfter, swarmChoices } from "../ui/swarmbinding.js";

const owned = [{ swarm_public_key: "aa" }, { swarm_public_key: "bb" }];
const isPrivate = (m) => m !== "pub" && m !== "dev";

test("asks about shared Private models with no binding, once, and only with a swarm to bind to", () => {
  const bindings = new Map([["bound", ["aa"]], ["empty", []]]);
  const shared = ["pub", "dev", "bound", "free", "empty", "free"];
  assert.deepEqual(modelsToAskAbout({ shared, isPrivate, bindings, owned, asked: false }), ["empty", "free"]);
  assert.deepEqual(modelsToAskAbout({ shared, isPrivate, bindings, owned, asked: true }), [], "asked before");
  assert.deepEqual(modelsToAskAbout({ shared, isPrivate, bindings, owned: [], asked: false }), [], "no swarm owned");
  assert.deepEqual(modelsToAskAbout({ shared: ["pub", "bound"], isPrivate, bindings, owned, asked: false }), [], "nothing unbound");
});

test("limiting binds each model to the swarms owned now; keeping changes nothing", () => {
  const bindings = new Map([["bound", ["aa"]]]);
  const limited = bindingsAfter("current", ["free"], owned, bindings);
  assert.deepEqual(limited.get("free"), ["aa", "bb"]);
  assert.deepEqual(limited.get("bound"), ["aa"], "an existing binding is left alone");
  assert.equal(bindings.has("free"), false, "the input map isn't changed");
  assert.deepEqual([...bindingsAfter("keep", ["free"], owned, bindings)], [...bindings]);
});

test("a Device model is never asked about: it is for this machine, not for a swarm", () => {
  assert.deepEqual(modelsToAskAbout({ shared: ["dev"], isPrivate, bindings: new Map(), owned, asked: false }), []);
});

test("the chooser shows a forgotten swarm a model is still bound to, ticked", () => {
  const rows = swarmChoices([{ swarm_public_key: "aa", label: "Team" }], ["aa", "deadbeefcafe0011"]);
  assert.deepEqual(rows.map((r) => [r.swarm_public_key, r.checked, r.gone]), [["aa", true, false], ["deadbeefcafe0011", true, true]]);
  assert.equal(rows[1].fingerprint, "…cafe0011");
  assert.deepEqual(swarmChoices([{ swarm_public_key: "aa" }], []).map((r) => r.checked), [false]);
});
