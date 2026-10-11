// Markup is built only through this module. Text in the UI comes from network peers (model names,
// answers, peer ids), so it must never be able to become tags or attributes:
//
//   html`<b title="${name}">${name}</b>`   escapes every interpolated value…
//   html`<ul>${items.map((i) => html`<li>${i}</li>`)}</ul>`   …unless it is itself markup made here.
//
// Arrays are joined; null, undefined and false render as nothing. Writing to the page goes through
// setHTML / appendHTML, which accept only markup from this module, so a plain string can't reach
// innerHTML by mistake. (The UI hygiene test fails on innerHTML anywhere else.)

const MARKUP = Symbol("markup");
const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };

/** Escape a value for HTML text or a quoted attribute. */
export function escapeText(v) {
  return String(v).replace(/[&<>"']/g, (c) => ESC[c]);
}

function render(v) {
  if (v == null || v === false) return "";
  if (typeof v === "object" && v[MARKUP] !== undefined) return v[MARKUP];
  if (Array.isArray(v)) return v.map(render).join("");
  return escapeText(v);
}

// Markup used as a plain string (`"<p>" + m`, or `${m}` in an untagged template) is a bug: it would
// render "[object Object]" or lose the markup. It throws instead, so the mistake is loud.
function misuse() {
  throw new TypeError("markup used as a string: compose it with html`` instead");
}

function markup(s) {
  return Object.freeze({ [MARKUP]: s, toString: misuse });
}

/** The template tag: every `${value}` is escaped unless it is markup from this module. */
export function html(strings, ...values) {
  let out = strings[0];
  for (let i = 0; i < values.length; i++) out += render(values[i]) + strings[i + 1];
  return markup(out);
}

/** Is `v` markup made by this module? */
export function isMarkup(v) {
  return v != null && typeof v === "object" && v[MARKUP] !== undefined;
}

/**
 * Wrap a string that a trusted producer built with its own escape-first rules: the markdown
 * renderer, the syntax highlighter, the bundled icon SVGs. Every caller is listed in the UI hygiene
 * test; nothing that takes text from outside may use this.
 */
export function trustedMarkup(s) {
  return markup(String(s));
}

/** The markup's HTML string (for tests and for a producer composing trusted markup). */
export function markupString(m) {
  if (!isMarkup(m)) throw new TypeError("expected markup from html``");
  return m[MARKUP];
}

/** Replace `el`'s content with `m` (markup from html`` only). */
export function setHTML(el, m) {
  el.innerHTML = markupString(m);
}

/** Insert `m` (markup from html`` only) at `position` ("beforeend", …) of `el`. */
export function insertHTML(el, position, m) {
  el.insertAdjacentHTML(position, markupString(m));
}

/** Build a detached element from markup: the first element `m` produces. */
export function fromHTML(m) {
  const t = document.createElement("template");
  t.innerHTML = markupString(m);
  return t.content.firstElementChild;
}
