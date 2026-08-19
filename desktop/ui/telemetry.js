// Rolling per-chat throughput/latency samples. The agent emits TPS/RTT per request; we keep the
// last ROLL_MAX client-side (persisted) and average them for the Activity view. Shared by the chat
// send path (writer) and renderActivity (reader) — the arrays are mutated in place.
import { store } from "./storage";
export const rttSamples = store.get("oh_rtt", []);
export const tpsSamples = store.get("oh_tps", []);
const ROLL_MAX = 30;
export function pushSample(arr, v, key) { if (v == null || !isFinite(v)) return; arr.push(v); while (arr.length > ROLL_MAX) arr.shift(); store.set(key, arr); }
export const mean = (a) => a.length ? a.reduce((x, y) => x + y, 0) / a.length : null;
