// Sticky network-routable model list — smooths DHT/gossip flicker. DHT provider records expire
// (~300s TTL) and re-propagate on their own schedule, so the raw known_models snapshot blinks.
// Keep a model "sticky" for STICKY_MS after last seen; IDLE_MS..STICKY_MS shows it dimmed ("idle").
// Never exceeds the network's ~300s TTL, so the UI never claims a provider is alive longer than the
// network does; the real liveness gate stays request-time discovery. seenModels is module-local;
// seenCount is read by the Providers view.
import { $ } from "./dom";
import { state, snap, engines } from "./state";

const STICKY_MS = 180000;
const IDLE_MS = 130000;  // seen within this = "live"; IDLE_MS..STICKY_MS = "idle" (dimmed).
const seenModels = {};   // model -> last-seen ms
export const seenCount = {};    // model -> last-known provider count
export function noteSeen() {
  const now = Date.now();
  (snap?.network?.known_models || []).forEach((m) => seenModels[m] = now);
  const provs = snap?.network?.known_providers || [];
  const c = {}; for (const p of provs) c[p.model_id] = (c[p.model_id] || 0) + 1;
  for (const m in c) seenCount[m] = c[m];
}
// A model still listed only because of stickiness (seen, but not within the last IDLE_MS) — shown
// dimmed/"idle", an honest "unconfirmed this instant" signal. Locally-served models are always live.
export function modelIdle(m) {
  if (state?.provider?.status?.running && engines.some((e) => e.models.includes(m))) return false;
  const t = seenModels[m];
  return t != null && (Date.now() - t) > IDLE_MS;
}
export function netModels() {
  const now = Date.now();
  const net = Object.keys(seenModels).filter((m) => now - seenModels[m] < STICKY_MS);
  const sharingLocal = state?.provider?.status?.running ? engines.flatMap((e) => e.models) : [];
  return [...new Set([...net, ...sharingLocal])].sort();
}
export function curModel() { const m = $("#modeldrop span").textContent; return m && m !== "—" && !/no models/i.test(m) ? m : ""; }
export function renderModels() {
  const models = netModels(); const opts = models.join("|");
  $("#homedrop").dataset.opts = opts; $("#modeldrop").dataset.opts = opts;
  const label = models[0] || "— no models yet";
  for (const d of ["#homedrop", "#modeldrop"]) { const sp = $(d + " span"); const cur = sp.textContent; if (!models.includes(cur)) sp.textContent = label; }
  $("#mcount").textContent = models.length; $("#homelive").textContent = models.length;
  $("#provcount") && ($("#provcount").textContent = models.length);
  $("#sbmodels").textContent = models.length + " models";
}
// Live network model ids for the selector (sticky-smoothed; may be empty before the first snapshot).
export function liveModels() {
  return [...new Set([...(snap?.network?.known_models || []), ...Object.keys(seenModels || {})])]
    .filter(Boolean).sort();
}
