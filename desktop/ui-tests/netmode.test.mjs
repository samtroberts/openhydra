// Settings › Network mode helpers (review 2026-10-14, M1/M2, D1).
import { test } from "node:test";
import assert from "node:assert/strict";
import { tokenizerDownloadsOn, tokenizerDownloadsKey, networkRestartPending } from "../ui/netmode.js";

test("tokenizer downloads: on by default on the public network, off by default in a private one", () => {
  assert.equal(tokenizerDownloadsOn({}), true, "an older config: the public network, downloads on");
  assert.equal(tokenizerDownloadsOn({ public_network: true, tokenizer_downloads: false }), false);
  assert.equal(tokenizerDownloadsOn({ public_network: false, tokenizer_downloads: true }), false, "private: its own setting, off");
  assert.equal(tokenizerDownloadsOn({ public_network: false, private_tokenizer_downloads: true }), true);
  assert.equal(tokenizerDownloadsKey({}), "tokenizer_downloads");
  assert.equal(tokenizerDownloadsKey({ public_network: false }), "private_tokenizer_downloads");
});

test("a restart is pending while a running role uses another network mode", () => {
  const running = (pub, dl) => ({ running: true, started_public_network: pub, started_tokenizer_downloads: dl });
  const priv = { public_network: false };
  // Saved private, both roles still on the public network.
  assert.match(networkRestartPending({ gateway: { status: running(true, true) }, provider: { status: running(true, true) } }, priv),
    /gateway and sharing still run on the public network/);
  // Only the gateway runs (not sharing): only it is named.
  assert.match(networkRestartPending({ gateway: { status: running(true, true) }, provider: { status: { running: false } } }, priv),
    /^The gateway still runs on the public network\.$/);
  // Restarted: nothing pending.
  assert.equal(networkRestartPending({ gateway: { status: running(false, false) }, provider: { status: running(false, false) } }, priv), "");
  // Back to public while sharing still runs privately.
  assert.match(networkRestartPending({ gateway: { status: running(true, true) }, provider: { status: running(false, false) } }, {}),
    /^Sharing still runs as a private deployment\.$/);
  // Only the tokenizer setting changed.
  assert.match(networkRestartPending({ gateway: { status: running(true, true) } }, { tokenizer_downloads: false }),
    /tokenizer downloads on/);
  // A role started by an older app (no record): nothing claimed.
  assert.equal(networkRestartPending({ gateway: { status: { running: true } } }, priv), "");
});
