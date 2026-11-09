// The sticky model list's rule, kept apart from the DOM so it can be tested (desktop/ui-tests).
// `seen` maps model -> last-seen ms. Every model in this snapshot's `known` list is seen now; a model
// the network reports as just left (`withdrawn`: its last provider withdrew it, or the bootstraps
// report that provider gone) is dropped at once unless something still lists it — the sticky timer
// smooths discovery flicker, it must not keep a model the network has said is gone (30 Oct: a gone
// model stayed on screen for 180 s). Returns the models dropped.
export function noteSnapshot(seen, known, withdrawn, now) {
  const live = new Set(known || []);
  for (const m of live) seen[m] = now;
  const dropped = [];
  for (const m of withdrawn || []) {
    if (!live.has(m) && Object.prototype.hasOwnProperty.call(seen, m)) {
      delete seen[m];
      dropped.push(m);
    }
  }
  return dropped;
}
