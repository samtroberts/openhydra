// The chat model picker's rule (SL2): it starts on, and falls back to, "Best available"
// (`openhydra/auto`, the gateway picks a trusted model) — never the alphabetically first model on the
// network, which anyone can announce. No imports, so the UI tests can load it.
export const AUTO_MODEL = "openhydra/auto";
// Its label starts with a mark no model id can contain (ids are plain ASCII, checked on the way
// in), so a peer that names its model "Best available" can't pass for it.
export const AUTO_LABEL = "\u2726 Best available";

/** The selection to keep, given the current one and the models live now: nothing when none is
 * live; the current one while it's still live (or is Best available); otherwise Best available. */
export function pickerSelection(cur, models) {
  if (!models.length) return "";
  return cur === AUTO_MODEL || models.includes(cur) ? cur : AUTO_MODEL;
}

/** What the picker shows for `id`: [`AUTO_LABEL`] for auto, else the model's display name. */
export function pickerLabel(id, display) {
  return id === AUTO_MODEL ? AUTO_LABEL : display(id);
}
