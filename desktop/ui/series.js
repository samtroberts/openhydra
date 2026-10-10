// Pure time-series helpers for the Activity/Share charts (no DOM, no app state), so they can be
// unit-tested with `node --test`.
export const HOUR_MS = 3600000;
export const hourKey = (ms) => Math.floor(ms / HOUR_MS);

// The hourly buckets that belong to a chart slot covering the time span (start, end]: every bucket
// after the one holding `start`, up to and including the one holding `end`. Adjacent slots share an
// edge (one's end is the next one's start), so each bucket lands in exactly one slot, and the
// newest slot (end = now) includes the current hour.
export function slotBuckets(start, end) {
  return { lo: hourKey(start) + 1, hi: hourKey(end) + 1 }; // [lo, hi)
}
