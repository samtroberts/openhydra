// The one-time question about Private models shared before swarm binding existed (enterprise
// phase 0.2). Such a model has no swarm list, which means "every swarm you own, including ones you
// create later": the behaviour it always had, kept on upgrade. The app asks once whether to keep
// that or limit each model. Pure, so it can be tested without the webview.

export const SWARM_BINDING_ASKED = "oh.swarmBindingAsked.v1";

// The shared Private models open to every owned swarm (no binding), when the question should be
// asked now: not asked before, at least one swarm owned, at least one such model. Otherwise [].
//   shared   — the models being shared
//   isGlobal — (model) => whether it is Global (a Global model isn't swarm-gated)
//   bindings — Map(model → [swarm public keys]); no entry (or an empty one) = every owned swarm
//   owned    — the swarms this node owns
//   asked    — whether the question was answered before
export function modelsToAskAbout({ shared, isGlobal, bindings, owned, asked }) {
  if (asked || !owned?.length) return [];
  return [...new Set(shared)].filter((m) => !isGlobal(m) && !(bindings?.get(m)?.length)).sort();
}

// The bindings after the user's answer: "keep" leaves them open to every owned swarm (no change);
// "current" binds each to the swarms owned now, so a swarm created later doesn't get them.
export function bindingsAfter(answer, models, owned, bindings) {
  const next = new Map(bindings);
  if (answer === "current") {
    const pks = owned.map((s) => s.swarm_public_key);
    for (const m of models) next.set(m, [...pks]);
  }
  return next;
}
