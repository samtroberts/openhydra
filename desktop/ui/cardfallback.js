// What the app does when an imported card's provider can't serve (offline, busy, or it refused).
// The gateway never falls back silently for the app: the app sends "ask" unless the user saved a
// choice for that model, and on a public card's failure asks whether to use another provider of
// the same exact version. A private card never falls back. Pure, so it can be tested without the
// webview.

export const FALLBACK_CHOICES = ["ask", "same-version", "never"];

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
    body: `${model}: the provider from your card ${why}.\n\n${n === 1 ? "1 other provider serves" : `${n} other providers serve`} the same model and version. ` +
      "It is a different machine, which can read your prompt and the reply.",
    ok: "Use another provider",
  };
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
