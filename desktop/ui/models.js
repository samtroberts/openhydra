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
// Human-readable model name. Path-addressed engines (llama.cpp) report the model as an absolute
// path (`/home/alice/models/Qwen3.5-9B-Q4_K_M.gguf`); show the GGUF basename minus the extension so
// the operator's home dir / OS username never appears and the row stays readable. Ollama-style tags
// (`llama3.2:1b`) and other clean ids pass through untouched. Display-only — the raw id remains the
// routing key everywhere it matters (see `curModel`, which reads the stored value, not the label).
export function displayModelName(id) {
  if (!id) return id;
  // Only rewrite genuine filesystem paths — a GGUF file or an absolute/home/drive path. Namespaced
  // logical ids (`openhydra/auto`, HF-style `Qwen/Qwen2.5-7B`) contain a slash but are NOT paths and
  // must pass through unchanged.
  const isPath = /\.gguf$/i.test(id) || /^([/~]|[A-Za-z]:[/\\])/.test(id);
  if (isPath) {
    const base = id.split(/[/\\]/).pop() || id;
    return base.replace(/\.gguf$/i, "");
  }
  return id;
}
// The selected model to route on: the RAW id stored in the span's dataset, not the (prettified)
// label. Falls back to the label for safety before the first render sets the value.
export function curModel() { const sp = $("#modeldrop span"); const m = (sp && (sp.dataset.value || sp.textContent)) || ""; return m && m !== "—" && !/no models/i.test(m) ? m : ""; }
export function renderModels() {
  const models = netModels(); const opts = models.join("|");
  $("#homedrop").dataset.opts = opts; $("#modeldrop").dataset.opts = opts;
  for (const d of ["#homedrop", "#modeldrop"]) {
    const sp = $(d + " span");
    let cur = sp.dataset.value || sp.textContent;            // raw current selection
    if (!models.includes(cur)) cur = models[0] || "";        // dropped out → first model
    sp.dataset.value = cur;                                  // keep the raw id for routing
    sp.textContent = cur ? displayModelName(cur) : "— no models yet"; // show the clean name
  }
  $("#mcount").textContent = models.length; $("#homelive").textContent = models.length;
  $("#provcount") && ($("#provcount").textContent = models.length);
  $("#sbmodels").textContent = models.length + " models";
}
// Live network model ids for the selector (sticky-smoothed; may be empty before the first snapshot).
export function liveModels() {
  return [...new Set([...(snap?.network?.known_models || []), ...Object.keys(seenModels || {})])]
    .filter(Boolean).sort();
}
