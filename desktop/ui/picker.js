// The chat model picker's rule (SL2): it starts on, and falls back to, "Best available"
// (`openhydra/auto`, the gateway picks a trusted model) — never the alphabetically first model on the
// network, which anyone can announce. No imports, so the UI tests can load it.
export const AUTO_MODEL = "openhydra/auto";
// Its label starts with a mark no model id can contain (ids are plain ASCII, checked on the way
// in), so a peer that names its model "Best available" can't pass for it.
export const AUTO_LABEL = "\u2726 Best available";

/** Whether a model name reads like the auto choice ("best available", "auto", "openhydra/auto",
 * in any case or punctuation): a peer's model named so is left out of the pickers, as it could
 * only be there to pass for it. */
export function looksLikeAuto(id) {
  const n = String(id).toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
  return n.startsWith("best available") || n === "auto" || n === "openhydra auto";
}

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
