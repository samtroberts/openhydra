// localStorage JSON wrapper — a fast cache only. The durable copy of sessions/stats/settings
// lives in the Tauri backend files; this survives reloads but not necessarily reinstalls.
export const store = {
  get(k, d) { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};
