import { test } from "node:test";
import assert from "node:assert/strict";
import { renderMarkdown, parseFences } from "../ui/markdown.js";

test("common formatting", () => {
  assert.equal(renderMarkdown("91 is **not** prime"), "<p>91 is <strong>not</strong> prime</p>");
  assert.equal(renderMarkdown("use `s[::-1]` here"), "<p>use <code>s[::-1]</code> here</p>");
  assert.equal(renderMarkdown("an *aside* and a_b_c"), "<p>an <em>aside</em> and a_b_c</p>");
  assert.equal(renderMarkdown("## Title"), "<h4>Title</h4>");
  assert.equal(renderMarkdown("- one\n- two"), "<ul><li>one</li><li>two</li></ul>");
  assert.equal(renderMarkdown("1. a\n2. b"), "<ol><li>a</li><li>b</li></ol>");
  assert.equal(renderMarkdown("line one\nline two"), "<p>line one<br>line two</p>");
  assert.equal(renderMarkdown("> quoted"), "<blockquote>quoted</blockquote>");
  assert.match(renderMarkdown("| a | b |\n|---|---|\n| 1 | 2 |"), /^<table><thead><tr><th>a<\/th><th>b<\/th><\/tr><\/thead><tbody><tr><td>1<\/td><td>2<\/td><\/tr><\/tbody><\/table>$/);
});

test("maths and stray markers stay as text", () => {
  assert.equal(renderMarkdown("$7 \\times 13$"), "<p>$7 \\times 13$</p>");
  assert.equal(renderMarkdown("2 * 3 * 4"), "<p>2 * 3 * 4</p>");
});

test("HTML is escaped and only safe links become links", () => {
  assert.equal(renderMarkdown("<script>alert(1)</script>"), "<p>&lt;script&gt;alert(1)&lt;/script&gt;</p>");
  assert.equal(renderMarkdown("<img src=x onerror=alert(1)>"), "<p>&lt;img src=x onerror=alert(1)&gt;</p>");
  assert.equal(renderMarkdown("[x](javascript:alert(1))"), "<p>[x](javascript:alert(1))</p>");
  assert.equal(renderMarkdown("[x](data:text/html,hi)"), "<p>[x](data:text/html,hi)</p>");
  assert.equal(
    renderMarkdown('[docs](https://example.com/a?b=1&c="2")'),
    '<p><a href="https://example.com/a?b=1&amp;c=&quot;2&quot;" target="_blank" rel="noopener noreferrer">docs</a></p>',
  );
  assert.equal(renderMarkdown('**<b onclick="x">**'), "<p><strong>&lt;b onclick=&quot;x&quot;&gt;</strong></p>");
  assert.equal(renderMarkdown("`<i>`"), "<p><code>&lt;i&gt;</code></p>");
});

test("emphasis never reaches into links or code", () => {
  assert.equal(
    renderMarkdown("[x](https://a.com) and foo_ bar"),
    '<p><a href="https://a.com" target="_blank" rel="noopener noreferrer">x</a> and foo_ bar</p>',
  );
  assert.equal(
    renderMarkdown("[x](https://a.com/**b**)"),
    '<p><a href="https://a.com/**b**" target="_blank" rel="noopener noreferrer">x</a></p>',
  );
  assert.equal(renderMarkdown("a\u00000\u0000b"), "<p>a0b</p>");
});

test("long hostile inputs render quickly", () => {
  const inputs = [
    "[".repeat(100000),
    "[a](".repeat(25000),
    "a|b\n" + " ".repeat(100000) + "x",
    "**a ".repeat(25000),
    "`".repeat(100000),
    "_x ".repeat(30000),
  ];
  for (const s of inputs) {
    const t0 = Date.now();
    renderMarkdown(s);
    assert.ok(Date.now() - t0 < 500, `took ${Date.now() - t0} ms for ${JSON.stringify(s.slice(0, 12))}…`);
  }
});

test("code fences: the forms models write, and ones left open", () => {
  const code = (t) => parseFences(t).filter((p) => p.code != null);
  assert.deepEqual(parseFences("hi\n```py\nx = 1\n```\nbye"), [{ prose: "hi" }, { lang: "py", code: "x = 1\n" }, { prose: "bye" }]);
  // Unclosed (a stopped reply): code to the end, never markdown.
  assert.deepEqual(code("```python\n# comment\nif __name__ == '__main__':"), [{ lang: "python", code: "# comment\nif __name__ == '__main__':\n" }]);
  // Tildes, a spaced info string, CRLF, a longer fence holding a shorter one.
  assert.deepEqual(code("~~~\nA\n~~~"), [{ lang: "code", code: "A\n" }]);
  assert.deepEqual(code("``` python\nB\n```"), [{ lang: "python", code: "B\n" }]);
  assert.deepEqual(code("```js\r\nC\r\n```\r\n"), [{ lang: "js", code: "C\n" }]);
  assert.deepEqual(code("````md\n```\ninner\n```\n````"), [{ lang: "md", code: "```\ninner\n```\n" }]);
  // A closing fence must be the same character and at least as long; text after it isn't a close.
  assert.deepEqual(code("```\na\n~~~\n``` not\n```"), [{ lang: "code", code: "a\n~~~\n``` not\n" }]);
  // Inline backticks in prose aren't fences.
  assert.deepEqual(parseFences("use ```x``` inline"), [{ prose: "use ```x``` inline" }]);
  assert.deepEqual(parseFences(""), []);
});
