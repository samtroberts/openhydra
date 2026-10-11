// The html`` template tag: values are escaped unless they are markup made by the tag itself.
// Run with `node --test desktop/ui-tests/*.test.mjs`.
import { test } from "node:test";
import assert from "node:assert/strict";
import { html, escapeText, isMarkup, markupString, trustedMarkup } from "../ui/html.js";

const s = (m) => markupString(m);

test("interpolated strings are escaped, as text and inside quoted attributes", () => {
  const name = `x" onclick="alert(1)" data-x='<b>'&`;
  assert.equal(s(html`<b title="${name}">${name}</b>`),
    `<b title="x&quot; onclick=&quot;alert(1)&quot; data-x=&#39;&lt;b&gt;&#39;&amp;">x&quot; onclick=&quot;alert(1)&quot; data-x=&#39;&lt;b&gt;&#39;&amp;</b>`);
});

test("nested markup and arrays compose without double escaping", () => {
  const items = ["a<", "b"];
  assert.equal(s(html`<ul>${items.map((i) => html`<li>${i}</li>`)}</ul>`), "<ul><li>a&lt;</li><li>b</li></ul>");
  assert.equal(s(html`${null}${undefined}${false}${0}${""}`), "0");
});

test("a plain string that looks like markup stays text", () => {
  const fake = "<span>not markup</span>";
  assert.equal(s(html`<div>${fake}</div>`), "<div>&lt;span&gt;not markup&lt;/span&gt;</div>");
  // A forged object can't pass as markup: the brand is a module-private symbol.
  assert.equal(isMarkup({ markup: "<b>" }), false);
  assert.throws(() => markupString("<b>x</b>"), TypeError);
});

test("trustedMarkup passes a producer's own output through unchanged", () => {
  assert.equal(s(html`<div>${trustedMarkup("<em>ok</em>")}</div>`), "<div><em>ok</em></div>");
  assert.equal(escapeText(`&<>"'`), "&amp;&lt;&gt;&quot;&#39;");
});

test("markup can't be used as a plain string by mistake", () => {
  const m = html`<b>x</b>`;
  assert.throws(() => "<p>" + m, TypeError);
  assert.throws(() => `${m}`, TypeError);
});
