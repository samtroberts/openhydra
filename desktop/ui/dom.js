// DOM query + HTML-escaping + peer-id shortening helpers. Pure — no app state, no imports.
export const $ = (s, r = document) => r.querySelector(s);
export const $$ = (s, r = document) => [...r.querySelectorAll(s)];

// Escapes all five HTML-special characters, so the result is safe both as text and inside a quoted
// attribute (`title="${esc(x)}"`). Model names and other strings come from network peers.
const HTML_ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
export function esc(s) { return String(s).replace(/[&<>"']/g, (c) => HTML_ESC[c]); }
export const escapeHtml = esc;

export function shortPeer(p) { return p && p.length > 20 ? `${p.slice(0, 10)}…${p.slice(-6)}` : p || "—"; }
export function peerShort(p) { return p && p.length > 18 ? `${p.slice(0, 12)}…${p.slice(-4)}` : p || "—"; }
