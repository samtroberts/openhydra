// The one-time question about Private models with no swarm binding (enterprise phase 0.2).
import test from "node:test";
import assert from "node:assert/strict";
import { modelsToAskAbout, bindingsAfter } from "../ui/swarmbinding.js";

const owned = [{ swarm_public_key: "aa" }, { swarm_public_key: "bb" }];
const isGlobal = (m) => m === "pub";

test("asks about shared Private models with no binding, once, and only with a swarm to bind to", () => {
  const bindings = new Map([["bound", ["aa"]], ["empty", []]]);
  const shared = ["pub", "bound", "free", "empty", "free"];
  assert.deepEqual(modelsToAskAbout({ shared, isGlobal, bindings, owned, asked: false }), ["empty", "free"]);
  assert.deepEqual(modelsToAskAbout({ shared, isGlobal, bindings, owned, asked: true }), [], "asked before");
  assert.deepEqual(modelsToAskAbout({ shared, isGlobal, bindings, owned: [], asked: false }), [], "no swarm owned");
  assert.deepEqual(modelsToAskAbout({ shared: ["pub", "bound"], isGlobal, bindings, owned, asked: false }), [], "nothing unbound");
});

test("limiting binds each model to the swarms owned now; keeping changes nothing", () => {
  const bindings = new Map([["bound", ["aa"]]]);
  const limited = bindingsAfter("current", ["free"], owned, bindings);
  assert.deepEqual(limited.get("free"), ["aa", "bb"]);
  assert.deepEqual(limited.get("bound"), ["aa"], "an existing binding is left alone");
  assert.equal(bindings.has("free"), false, "the input map isn't changed");
  assert.deepEqual([...bindingsAfter("keep", ["free"], owned, bindings)], [...bindings]);
});
