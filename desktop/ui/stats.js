// #7/#10: durable lifetime served/consumed model stats + hourly time-series. The agent's per-model
// counters reset each restart, so the lifetime totals + timeline live here, diffed from the poll
// and folded into hourly buckets, persisted to ~/.openhydra/stats.json via the Tauri backend.
// statsDB (+ dirty/saveT) is module-local; readers import the aggregate getters.
import { snap, usedTokens } from "./state";
import { call } from "./bridge";

  const HOUR_MS = 3600000, STATS_KEEP_HOURS = 24 * 31;   // ~31 days of hourly buckets
  let statsDB = { models: {} };                          // hydrated on boot from load_stats
  let statsDirty = false, statsSaveT = 0;
  function hourKey(ms) { return Math.floor(ms / HOUR_MS); }
  function statModel(id) {
    return (statsDB.models[id] ||= { firstServed: null, firstUsed: null, servedTotal: 0, usedTotal: 0, lastServed: 0, lastUsed: 0, buckets: {} });
  }
  // Fold one cumulative counter reading into a per-model lifetime total + current-hour bucket.
  // `raw` is the agent's since-boot counter; a value < the last reading means the agent restarted
  // (counter reset to 0), so the whole `raw` is the fresh delta.
  function foldCounter(m, raw, field, bucketField, now) {
    raw = Math.max(0, Math.round(raw || 0));
    const last = m[field];
    const delta = raw >= last ? raw - last : raw;
    m[field] = raw;
    if (delta <= 0) return;
    const totalField = field === "lastServed" ? "servedTotal" : "usedTotal";
    const firstField = field === "lastServed" ? "firstServed" : "firstUsed";
    m[totalField] += delta;
    if (m[firstField] == null) m[firstField] = now;       // first-token anchor (no synthetic back-fill)
    const b = (m.buckets[hourKey(now)] ||= { s: 0, u: 0 });
    b[bucketField] += delta;
  }
  export function accumulateStats() {
    if (!snap?.transfers) return;
    const now = Date.now(), t = snap.transfers;
    for (const [id, pm] of Object.entries(t.per_model || {})) foldCounter(statModel(id), pm.tokens, "lastServed", "s", now);
    for (const [id, pm] of Object.entries(t.consumed_per_model || {})) foldCounter(statModel(id), pm.tokens, "lastUsed", "u", now);
    // Prune buckets older than the retention window (keeps stats.json bounded).
    const floor = hourKey(now) - STATS_KEEP_HOURS;
    for (const m of Object.values(statsDB.models)) for (const k in m.buckets) if (+k < floor) delete m.buckets[k];
    statsDirty = true;
    const nowT = Date.now();
    if (nowT - statsSaveT > 8000) { statsSaveT = nowT; statsDirty = false; try { call("save_stats", { data: JSON.stringify(statsDB) }); } catch {} }
  }
  export async function loadStats() {
    try { const blob = await call("load_stats"); if (blob) { const d = JSON.parse(blob); if (d && d.models) statsDB = d; } } catch {}
  }
  export const lifetimeServed = (id) => statsDB.models[id]?.servedTotal || 0;
  export const lifetimeUsed = (id) => statsDB.models[id]?.usedTotal || 0;
  export const statModels = () => Object.keys(statsDB.models);
  // Durable lifetime totals across all models. Fall back to the agent's since-boot counter (or the
  // legacy desktop-chat counter) until the accumulator has data — so a fresh install still shows
  // something. `used` now derives from the gateway's per-model consumed tracking, so it counts
  // tokens consumed by external OpenAI clients (e.g. a coding agent), not just in-app chats.
  export const totalServed = () => Object.values(statsDB.models).reduce((a, m) => a + (m.servedTotal || 0), 0) || (snap?.transfers?.tokens_served ?? 0);
  export const totalUsed = () => Object.values(statsDB.models).reduce((a, m) => a + (m.usedTotal || 0), 0) || (snap?.transfers?.tokens_consumed ?? usedTokens);
  // Aggregate buckets for a range into ordered time-slots, each a per-model {s,u}. 24h → hourly
  // (24 slots), 7d/30d → daily. Returns { slots:[{label,models:{id:{s,u}}}], models:Set }.
  export function statsSeries(range) {
    const now = Date.now();
    const spans = { "24h": { n: 24, ms: HOUR_MS }, "7d": { n: 7, ms: 86400000 }, "30d": { n: 30, ms: 86400000 } };
    const sp = spans[range] || spans["24h"];
    const slots = [], models = new Set();
    for (let i = sp.n - 1; i >= 0; i--) {
      const end = now - i * sp.ms, start = end - sp.ms;
      const slot = { end, models: {} };
      const kLo = hourKey(start), kHi = hourKey(end);
      for (const [id, m] of Object.entries(statsDB.models)) {
        let s = 0, u = 0;
        for (let k = kLo; k < kHi; k++) { const b = m.buckets[k]; if (b) { s += b.s; u += b.u; } }
        if (s || u) { slot.models[id] = { s, u }; models.add(id); }
      }
      slots.push(slot);
    }
    return { slots, models, span: sp };
  }
