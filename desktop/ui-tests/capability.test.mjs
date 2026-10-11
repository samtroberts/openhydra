// The command allowlist. build.rs gives every app command an allow-<command> permission; the main
// window's capability grants only the commands the UI calls, plus the few core APIs it uses. Page
// script (even injected script) can then call nothing else. Run with
// `node --test desktop/ui-tests/*.test.mjs`.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const tauriDir = join(here, "..", "src-tauri");
const uiDir = join(here, "..", "ui");
const mainRs = readFileSync(join(tauriDir, "src", "main.rs"), "utf8");
const buildRs = readFileSync(join(tauriDir, "build.rs"), "utf8");
const cap = JSON.parse(readFileSync(join(tauriDir, "capabilities", "default.json"), "utf8"));

const registered = mainRs.match(/generate_handler!\[([\s\S]*?)\]/)[1].split(",").map((s) => s.trim()).filter(Boolean);
const manifest = [...buildRs.matchAll(/^\s*"([a-z_]+)",$/gm)].map((m) => m[1]);
const called = new Set();
for (const f of readdirSync(uiDir).filter((n) => n.endsWith(".js") && n !== "bridge.js")) {
  for (const m of readFileSync(join(uiDir, f), "utf8").matchAll(/call\(\s*"([a-z_]+)"/g)) called.add(m[1]);
}
const kebab = (c) => "allow-" + c.replace(/_/g, "-");
const appGrants = cap.permissions.filter((p) => typeof p === "string" && !p.includes(":"));

test("build.rs lists exactly the registered commands", () => {
  assert.deepEqual([...manifest].sort(), [...registered].sort());
});

test("the capability grants exactly the commands the UI calls", () => {
  assert.deepEqual([...appGrants].sort(), [...called].map(kebab).sort());
  for (const c of called) assert.ok(registered.includes(c), `${c} is called but not registered`);
});

test("commands only Rust uses are not reachable from the page", () => {
  for (const c of ["connector_preview", "uninstall_cli", "web_search"]) {
    assert.ok(!appGrants.includes(kebab(c)), `${c} must not be granted`);
  }
});

test("no broad core grant", () => {
  const core = cap.permissions.filter((p) => typeof p === "string" && p.includes(":"));
  assert.ok(!core.some((p) => /^core:default$|^core:[a-z]+:default$/.test(p)), core.join(", "));
  assert.deepEqual(cap.windows, ["main"]);
  assert.ok(!cap.remote, "no remote (non-app) origins");
});
