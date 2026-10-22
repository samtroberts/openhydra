// Type-to-filter for a menu opened with `{ filter: true }` (SL4: ⌘K filters as you type). Pure, no
// imports, so the UI tests can load it.

/** The query after a key: a printable character appends, Backspace removes the last; any other key
 * leaves it alone (`null`: not a filter key). */
export function nextQuery(q, e) {
  if (e.metaKey || e.ctrlKey || e.altKey) return null;
  if (e.key === "Backspace") return q.slice(0, -1);
  if (e.key === " ") return q ? q + " " : null; // a leading space isn't a filter
  if (e.key.length === 1) return q + e.key;
  return null;
}

/** Whether an item's label matches the query: every word of it, anywhere, ignoring case. */
export function matches(label, q) {
  const l = label.toLowerCase();
  return q.toLowerCase().split(/\s+/).filter(Boolean).every((w) => l.includes(w));
}
