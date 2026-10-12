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

test("headings, double-backtick spans and links with parentheses or titles", () => {
  assert.equal(renderMarkdown("## Title ##"), "<h4>Title</h4>");
  assert.equal(renderMarkdown("# C# notes"), "<h3>C# notes</h3>", "a # inside the text stays");
  assert.equal(renderMarkdown("use ``a `b` c`` here"), "<p>use <code>a `b` c</code> here</p>");
  assert.equal(renderMarkdown("[wiki](https://en.wikipedia.org/wiki/Rust_(programming_language))"),
    '<p><a href="https://en.wikipedia.org/wiki/Rust_(programming_language)" target="_blank" rel="noopener noreferrer">wiki</a></p>');
  assert.equal(renderMarkdown('[docs](https://openhydra.co/docs "The \'docs\' & more")'),
    '<p><a href="https://openhydra.co/docs" title="The \'docs\' &amp; more" target="_blank" rel="noopener noreferrer">docs</a></p>');
  // A title can't break out of its attribute, and a javascript: link with a title stays text.
  assert.ok(renderMarkdown('[x](https://a.b "q&quot; onmouseover=&quot;alert(1)")').includes('title="q&amp;quot; onmouseover=&amp;quot;alert(1)" target='));
  assert.equal(renderMarkdown('[x](javascript:alert(1) "t")'), '<p>[x](javascript:alert(1) &quot;t&quot;)</p>');
});

test("long runs of backticks and brackets stay fast", () => {
  for (const s of ["``a".repeat(20000), "`".repeat(40000), "[a](".repeat(10000) + "(".repeat(5000), "[x](h" + "(a)".repeat(20000)]) {
    const t0 = Date.now(); renderMarkdown(s); assert.ok(Date.now() - t0 < 500, `${s.slice(0, 12)}… took ${Date.now() - t0} ms`);
  }
});

test("fuzz: only the expected tags and attributes, and only safe links", () => {
  const alphabet = ["[", "]", "(", ")", '"', "'", "<", ">", "&", "`", "``", "*", "_", "#", "|", "~", "\n", " ", "a", "javascript:", "https://x.y", "mailto:a@b", "onerror=", "&quot;", "title", "data:"];
  const tags = new Set(["p", "br", "strong", "em", "del", "code", "a", "h3", "h4", "h5", "h6", "hr", "ul", "ol", "li", "blockquote", "table", "thead", "tbody", "tr", "th", "td"]);
  let seed = 7; const rnd = (n) => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed % n; };
  for (let i = 0; i < 20000; i++) {
    let t = ""; const len = 1 + rnd(40);
    for (let j = 0; j < len; j++) t += alphabet[rnd(alphabet.length)];
    const out = renderMarkdown(t);
    for (const m of out.matchAll(/<\/?([a-z0-9]+)([^>]*)>/gi)) {
      assert.ok(tags.has(m[1].toLowerCase()), `tag ${m[1]} from ${JSON.stringify(t)}`);
      const attrs = [...m[2].matchAll(/\s([a-z-]+)="[^"]*"/gi)].map((a) => a[1]);
      const rest = m[2].replace(/\s[a-z-]+="[^"]*"/gi, "").trim();
      assert.equal(rest, "", `stray attribute text in ${m[0]} from ${JSON.stringify(t)}`);
      for (const a of attrs) assert.ok(["href", "title", "target", "rel"].includes(a), `attribute ${a} from ${JSON.stringify(t)}`);
      const href = /\shref="([^"]*)"/.exec(m[2]);
      if (href) assert.match(href[1], /^(https?:\/\/|mailto:)/i, `href ${href[1]} from ${JSON.stringify(t)}`);
    }
  }
});

test("a pipe inside a table cell's code stays in the cell", () => {
  assert.equal(renderMarkdown("| op | use |\n|---|---|\n| `a | b` | or |"),
    "<table><thead><tr><th>op</th><th>use</th></tr></thead><tbody><tr><td><code>a | b</code></td><td>or</td></tr></tbody></table>");
});
