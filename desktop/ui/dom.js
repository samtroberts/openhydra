// DOM query + peer-id shortening helpers. Pure — no app state, no imports.
export const $ = (s, r = document) => r.querySelector(s);
export const $$ = (s, r = document) => [...r.querySelectorAll(s)];

// Markup is built with html`` (./html.js), which escapes every value; there is no separate escape
// helper to forget.

export function shortPeer(p) { return p && p.length > 20 ? `${p.slice(0, 10)}…${p.slice(-6)}` : p || "—"; }
export function peerShort(p) { return p && p.length > 18 ? `${p.slice(0, 12)}…${p.slice(-4)}` : p || "—"; }
