// Economy helpers (M2.2 reputation + M2.3 credit), surfaced by the agent status endpoint.
// reputation is keyed by OpenHydra peer id; credit by libp2p peer id. known_providers carries
// both, so we can join either back to a model row or a peer row. All read the live `snap`.
import { snap } from "./state";

function econ() { return snap?.economy || {}; }
export function repByOpenhydra() { const m = {}; (econ().reputation || []).forEach((r) => m[r.openhydra_peer_id] = r.score); return m; }
export function creditByLibp2p() { const m = {}; (econ().credit || []).forEach((c) => m[c.libp2p_peer_id] = c); return m; }
// libp2p peer id → earned reputation, resolved through the provider directory.
export function repByLibp2p() {
  const byOh = repByOpenhydra(), out = {};
  (snap?.network?.known_providers || []).forEach((p) => { if (p.openhydra_peer_id in byOh) out[p.libp2p_peer_id] = byOh[p.openhydra_peer_id]; });
  return out;
}
export function providersForModel(model) { return (snap?.network?.known_providers || []).filter((p) => p.model_id === model); }
// mean earned reputation across the providers serving `model` (null if none rated yet).
export function modelReputation(model, byOh) { const s = providersForModel(model).map((p) => byOh[p.openhydra_peer_id]).filter((x) => x != null); return s.length ? Math.round(s.reduce((a, b) => a + b, 0) / s.length) : null; }
// provider role publishes per-model serve TPS; only present for models THIS node serves.
export function modelAvgTps(model) { const pm = snap?.transfers?.per_model?.[model]; return pm && pm.avg_native_tps > 0 ? Math.round(pm.avg_native_tps) : null; }
