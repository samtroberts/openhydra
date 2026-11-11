// The `/model` choices active in the connected tools, shown on the Connectors view (finding 5 of the
// 11 Nov audit: a choice made with `/model` was invisible in the app). Pure, no imports, so the UI
// tests can load it.

// Client names (the first word of a tool's User-Agent) the gateway reports, and what to call them.
// Anything else is shown as the tool sent it.
const NAMES = {
  "claude-cli": "Claude Code",
  "opencode": "OpenCode",
  "continue": "Continue",
  "pi": "Pi",
  "hermes": "Hermes",
};

/** What to call the tool behind a pin's client name. */
export function toolName(client) {
  if (!client) return "A tool that doesn't send its name";
  return NAMES[client] || client;
}

/** When a pin lapses: it lasts while the tool keeps using it, and ends after an hour unused. */
export function lapseText(secs) {
  const m = Math.max(1, Math.round((Number(secs) || 0) / 60));
  return `ends if unused for ${m} min`;
}

/** The rows to show, from the gateway's `{ pins: [{ client, model, expires_in_secs }] }`. */
export function pinRows(resp) {
  const pins = Array.isArray(resp?.pins) ? resp.pins : [];
  return pins
    .filter((p) => p && typeof p.model === "string" && p.model)
    .map((p) => {
      const client = typeof p.client === "string" ? p.client : "";
      return { client, tool: toolName(client), model: p.model, lapse: lapseText(p.expires_in_secs) };
    });
}
