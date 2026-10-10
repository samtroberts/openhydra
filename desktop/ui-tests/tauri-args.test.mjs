// Checks every `call("cmd", { … })` in desktop/ui against the Rust `#[tauri::command]` signatures in
// desktop/src-tauri/src/main.rs. Tauri v2 maps a Rust parameter `swarm_public_key` to the JS key
// `swarmPublicKey`; a snake_case key is ignored, so a required parameter fails with "missing required
// key" and an optional one is silently dropped. Run with `node --test desktop/ui-tests`.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const uiDir = join(here, "..", "ui");
const mainRs = readFileSync(join(here, "..", "src-tauri", "src", "main.rs"), "utf8");

const camel = (s) => s.replace(/_([a-z0-9])/g, (_, c) => c.toUpperCase());

// Rust commands: name → [{ key, optional }], skipping injected params (State, AppHandle, Window).
function rustCommands() {
  const out = new Map();
  const re = /#\[tauri::command[^\]]*\]\s*(?:#\[[^\]]*\]\s*)*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\(([^)]*)\)/g;
  let m;
  while ((m = re.exec(mainRs))) {
    const params = [];
    for (const raw of m[2].split(",")) {
      const p = raw.trim();
      const i = p.indexOf(":");
      if (i < 0) continue;
      const name = p.slice(0, i).replace(/^mut\s+/, "").trim();
      const ty = p.slice(i + 1).trim();
      if (/State<|AppHandle|Window|Webview/.test(ty) || name.startsWith("_")) continue;
      params.push({ key: camel(name), optional: ty.startsWith("Option") });
    }
    out.set(m[1], params);
  }
  return out;
}

// The commands registered with the invoke handler (a defined-but-unregistered command fails too).
function registered() {
  const i = mainRs.indexOf("generate_handler![");
  const body = mainRs.slice(i, mainRs.indexOf("]", i));
  return new Set(body.replace("generate_handler![", "").split(",").map((s) => s.trim()).filter(Boolean));
}

// Top-level keys of an object literal starting at `text[start] === "{"`, or null for non-literals.
function literalKeys(text, start) {
  let depth = 0, i = start, cur = "", keys = [];
  for (; i < text.length; i++) {
    const ch = text[i];
    if ("{[(".includes(ch)) { depth++; if (depth === 1) continue; }
    if ("}])".includes(ch)) { depth--; if (depth === 0) break; }
    if (depth === 1 && ch === ",") { keys.push(cur); cur = ""; } else if (depth >= 1) cur += ch;
  }
  keys.push(cur);
  return keys.map((k) => (/^\s*(?:\.\.\.)?\s*([A-Za-z_$][\w$]*)/.exec(k) || [])[1]).filter(Boolean);
}

function jsCalls() {
  const calls = [];
  for (const f of readdirSync(uiDir).filter((f) => f.endsWith(".js") && f !== "bridge.js")) {
    const text = readFileSync(join(uiDir, f), "utf8");
    const re = /\bcall\(\s*"(\w+)"\s*(,\s*)?/g;
    let m;
    while ((m = re.exec(text))) {
      const line = text.slice(0, m.index).split("\n").length;
      const after = m.index + m[0].length;
      const keys = m[2] && text[after] === "{" ? literalKeys(text, after) : m[2] ? null : [];
      calls.push({ where: `${f}:${line}`, cmd: m[1], keys });
    }
  }
  return calls;
}

test("every UI call names a registered command with the right argument keys", () => {
  const cmds = rustCommands(), reg = registered(), problems = [];
  const calls = jsCalls();
  assert.ok(calls.length > 30, "found the UI's backend calls");
  for (const c of calls) {
    if (!cmds.has(c.cmd) || !reg.has(c.cmd)) { problems.push(`${c.where}: "${c.cmd}" is not a registered command`); continue; }
    if (c.keys === null) continue; // args passed as a variable: not checkable statically
    const params = cmds.get(c.cmd), known = new Set(params.map((p) => p.key));
    for (const k of c.keys) if (!known.has(k)) problems.push(`${c.where}: ${c.cmd} gets unknown key "${k}" (expected one of ${[...known].join(", ") || "none"})`);
    for (const p of params) if (!p.optional && !c.keys.includes(p.key)) problems.push(`${c.where}: ${c.cmd} is missing required key "${p.key}"`);
  }
  assert.deepEqual(problems, []);
});

test("no window.confirm/alert/prompt (they do nothing in the Tauri webview)", () => {
  const problems = [];
  for (const f of readdirSync(uiDir).filter((f) => f.endsWith(".js"))) {
    readFileSync(join(uiDir, f), "utf8").split("\n").forEach((l, i) => {
      const code = l.replace(/\/\/.*$/, "");
      if (/(^|[^\w.$])(window\.)?(confirm|alert|prompt)\s*\(/.test(code)) problems.push(`${f}:${i + 1}: ${l.trim()}`);
    });
  }
  assert.deepEqual(problems, []);
});
