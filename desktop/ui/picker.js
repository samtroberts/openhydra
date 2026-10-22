// The chat model picker's rule (SL2): it starts on, and falls back to, "Best available"
// (`openhydra/auto`, the gateway picks a trusted model) — never the alphabetically first model on the
// network, which anyone can announce. No imports, so the UI tests can load it.
export const AUTO_MODEL = "openhydra/auto";

/** The selection to keep, given the current one and the models live now: nothing when none is
 * live; the current one while it's still live (or is Best available); otherwise Best available. */
export function pickerSelection(cur, models) {
  if (!models.length) return "";
  return cur === AUTO_MODEL || models.includes(cur) ? cur : AUTO_MODEL;
}

/** What the picker shows for `id`: "Best available" for auto, else the model's display name. */
export function pickerLabel(id, display) {
  return id === AUTO_MODEL ? "Best available" : display(id);
}
