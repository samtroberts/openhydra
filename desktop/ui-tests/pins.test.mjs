// Audit 11 Nov, finding 5: the app shows each tool's active `/model` choice.
import { test } from "node:test";
import assert from "node:assert/strict";
import { pinRows, toolName, lapseText } from "../ui/pins.js";

test("known tools get their names; others are shown as sent", () => {
  assert.equal(toolName("claude-cli"), "Claude Code");
  assert.equal(toolName("opencode"), "OpenCode");
  assert.equal(toolName("my-agent"), "my-agent");
  assert.match(toolName(""), /doesn't send its name/);
});

test("rows come from the gateway's reply; junk is dropped", () => {
  const rows = pinRows({ pins: [
    { client: "claude-cli", model: "qwen2.5:7b", expires_in_secs: 3480 },
    { client: "x", model: "" },
    null,
    { client: 7, model: "llama3.2:3b", expires_in_secs: 10 },
  ] });
  assert.deepEqual(rows.map((r) => [r.tool, r.model, r.lapse]), [
    ["Claude Code", "qwen2.5:7b", "ends if unused for 58 min"],
    ["A tool that doesn't send its name", "llama3.2:3b", "ends if unused for 1 min"],
  ]);
  assert.deepEqual(pinRows(undefined), []);
  assert.deepEqual(pinRows({ pins: "nope" }), []);
});

test("the lapse is in whole minutes, at least one", () => {
  assert.equal(lapseText(0), "ends if unused for 1 min");
  assert.equal(lapseText(3600), "ends if unused for 60 min");
});
