// What the app does when an imported card's provider can't serve (offline, busy, or it refused).
// The gateway never falls back silently for the app: the app sends "ask" unless the user saved a
// choice for that model, and on a public card's failure asks whether to use another provider of
// the same exact version. A private card never falls back. Pure, so it can be tested without the
// webview.

export const FALLBACK_CHOICES = ["ask", "same-version", "never"];

// The choices a card's selector offers: "" is no saved choice (the default: the app asks; other
// apps on the gateway, which can't be asked, use another provider of the same version).
export const FALLBACK_OPTIONS = [
  ["", "Default: ask me here; other apps use another provider of the same version"],
  ["ask", "Ask me here; other apps get an error"],
  ["same-version", "Use another provider of the same version, everywhere"],
  ["never", "Never use another provider"],
];

// The selector's value for a model's saved choice ("" when none, or one this app doesn't know).
export function selectedFallback(model, prefs) {
  const saved = prefs && prefs[model];
  return FALLBACK_CHOICES.includes(saved) ? saved : "";
}

// The choice the app sends for `model`: the one saved for it, else "ask" (a person is here to ask).
export function fallbackToSend(model, prefs) {
  const saved = prefs && prefs[model];
  return FALLBACK_CHOICES.includes(saved) ? saved : "ask";
}

// Whether to offer another provider for a failed request: a public card, providers of the same
// version exist, and the gateway was told to ask.
export function shouldOfferFallback(details) {
  return !!details && details.private !== true && Number(details.alternatives) > 0 && details.fallback === "ask";
}

const REASON = { offline: "is offline", overloaded: "is busy", refused: "refused the request" };

// The question for the user: title, body, button.
export function offerText(model, details) {
  const why = REASON[details?.reason] || "couldn't serve it";
  const n = Number(details?.alternatives) || 0;
  return {
    title: "Use another provider?",
    body: `${model}: the provider from your card ${why}.\n\n${n === 1 ? "1 other provider announces" : `${n} other providers announce`} the same model and version ` +
      "(their own claim). It is a different machine, which can read your prompt and the reply.",
    ok: "Use another provider",
  };
}

// What the chat says when the user said no: nothing was sent anywhere else.
export function declinedText(model, details) {
  const why = REASON[details?.reason] || "couldn't serve it";
  return `${model}: the provider from your card ${why}. Not sent to another provider.`;
}

// The note under a reply another provider served, from its `openhydra.card` block; "" if none.
export function fellBackNote(card) {
  if (!card || !card.fell_back) return "";
  const why = REASON[card.reason] || "couldn't serve it";
  return `Served by another provider of the same version (the card's provider ${why}).`;
}

// A card's version line from its canonical id (`family/params/quant/template_hash`), e.g.
// "qwen2.5 · 7B · Q4_K_M"; "" when the card states none.
export function cardVersionLine(canonical) {
  const parts = String(canonical || "").split("/");
  if (parts.length < 3 || !parts[0] || !parts[1] || !parts[2]) return "";
  return `${parts[0]} · ${parts[1].toUpperCase()} · ${parts[2].toUpperCase()}`;
}
