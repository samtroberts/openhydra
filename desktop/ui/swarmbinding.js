// The one-time question about Private models shared before swarm binding existed (enterprise
// phase 0.2). Such a model has no swarm list, which means "every swarm you own, including ones you
// create later": the behaviour it always had, kept on upgrade. The app asks once whether to keep
// that or limit each model. Pure, so it can be tested without the webview.

export const SWARM_BINDING_ASKED = "oh.swarmBindingAsked.v1";

// The shared Private models open to every owned swarm (no binding), when the question should be
// asked now: not asked before, at least one swarm owned, at least one such model. Otherwise [].
//   shared    — the models being shared
//   isPrivate — (model) => whether its reach is exactly Private (a Global model isn't swarm-gated,
//               and a Device one is for this machine only, never for a swarm)
//   bindings  — Map(model → [swarm public keys]); no entry (or an empty one) = every owned swarm
//   owned     — the swarms this node owns
//   asked     — whether the question was answered before
export function modelsToAskAbout({ shared, isPrivate, bindings, owned, asked }) {
  if (asked || !owned?.length) return [];
  return [...new Set(shared)].filter((m) => isPrivate(m) && !(bindings?.get(m)?.length)).sort();
}

// The rows of the swarm chooser for a model: every swarm owned now, then any it is still bound to
// that is gone (forgotten), so the user can see why nobody can use it and untick them.
//   owned — the swarms this node owns ({ swarm_public_key, label, fingerprint })
//   bound — the model's swarm public keys
export function swarmChoices(owned, bound) {
  const known = new Set(owned.map((s) => s.swarm_public_key));
  const rows = owned.map((s) => ({ ...s, checked: bound.includes(s.swarm_public_key), gone: false }));
  for (const pk of bound) {
    if (!known.has(pk)) rows.push({ swarm_public_key: pk, label: "Forgotten swarm", fingerprint: `…${pk.slice(-8)}`, checked: true, gone: true });
  }
  return rows;
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
