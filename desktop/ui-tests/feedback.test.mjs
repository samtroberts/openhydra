// SL12: "Feedback" opens the issue form with the version filled in, and the known-issues page.
import { test } from "node:test";
import assert from "node:assert/strict";
import { reportUrl, KNOWN_ISSUES_URL, REPO } from "../ui/feedback.js";

test("the report link fills in a real version, and only that", () => {
  assert.equal(reportUrl("v0.3.23"), `${REPO}/issues/new?template=bug_report.yml&version=0.3.23`);
  assert.equal(reportUrl("0.3.23-rc.1"), `${REPO}/issues/new?template=bug_report.yml&version=0.3.23-rc.1`);
  assert.equal(reportUrl("v—"), `${REPO}/issues/new?template=bug_report.yml`, "not loaded yet");
  assert.equal(reportUrl("1&x=<b>"), `${REPO}/issues/new?template=bug_report.yml`);
  assert.equal(KNOWN_ISSUES_URL, `${REPO}/blob/main/KNOWN_ISSUES.md`);
});
