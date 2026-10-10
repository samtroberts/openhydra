// Economy helpers (M2.2 reputation + M2.3 credit), surfaced by the agent status endpoint.
// Reputation is keyed per (OpenHydra peer id, model id) since credit-tiering Phase 4a — trust a
// provider earns on one model grants nothing on another (anti-Sybil). Credit stays keyed by libp2p
// peer id. known_providers carries model_id + both peer ids, so we can join either back to a model
// row or a peer row. All read the live `snap`.
import { snap } from "./state";

function econ() { return snap?.economy || {}; }

// The composite (peer, model) reputation key — must match the model half the agent keyed on
// (rep_model_key: the provider's canonical model id, or its raw handle when it advertised none).
// NUL delimiter, matching the agent's reputation_key().
export function repKey(openhydra_peer_id, model_id) { return `${openhydra_peer_id}\u0000${model_id || ""}`; }

// (peer, model) → earned reputation, one entry per model a provider serves. The Providers view
// joins on it via repKey(p.openhydra_peer_id, p.model_id).
export function repByPeerModel() {
  const m = {};
  (econ().reputation || []).forEach((r) => { m[repKey(r.openhydra_peer_id, r.model_id)] = r.score; });
  return m;
}

// (peer, model) → what your gateway has verified about it: its authenticity tier label and latest
// model-check (log-prob) verdict. No entry: never used, so not verified.
export function trustByPeerModel() {
  const m = {};
  (econ().reputation || []).forEach((r) => { m[repKey(r.openhydra_peer_id, r.model_id)] = { tier: r.authenticity, check: r.logprob_check }; });
  return m;
}

// OpenHydra peer id → a single earned reputation, aggregated (mean) across that peer's models —
// for per-peer rows (the Network view) that show one number per connection, not per model.
export function repByOpenhydra() {
  const acc = {};
  (econ().reputation || []).forEach((r) => { (acc[r.openhydra_peer_id] ||= []).push(r.score); });
  const m = {};
  for (const [peer, xs] of Object.entries(acc)) m[peer] = xs.reduce((a, b) => a + b, 0) / xs.length;
  return m;
}

// libp2p peer id → earned reputation (per-peer aggregate), resolved through the provider directory.
export function repByLibp2p() {
  const byOh = repByOpenhydra(), out = {};
  (snap?.network?.known_providers || []).forEach((p) => { if (p.openhydra_peer_id in byOh) out[p.libp2p_peer_id] = byOh[p.openhydra_peer_id]; });
  return out;
}

function providersForModel(model) { return (snap?.network?.known_providers || []).filter((p) => p.model_id === model); }

// mean earned reputation across the providers serving `model`, each looked up per (peer, model)
// (null if none rated yet). `byPM` is repByPeerModel().
export function modelReputation(model, byPM) {
  const s = providersForModel(model)
    .map((p) => byPM[repKey(p.openhydra_peer_id, p.model_id)])
    .filter((x) => x != null);
  return s.length ? Math.round(s.reduce((a, b) => a + b, 0) / s.length) : null;
}

// provider role publishes per-model serve TPS; only present for models THIS node serves.
export function modelAvgTps(model) { const pm = snap?.transfers?.per_model?.[model]; return pm && pm.avg_native_tps > 0 ? Math.round(pm.avg_native_tps) : null; }
