// Single source of truth for the `openhydra` CLI-on-PATH status + install action, shared by both
// surfaces (the Settings "Command-line tool" row and the Connectors PATH banner). Refreshing after an
// install on EITHER surface broadcasts "cli-status" on the bus, so both re-render in sync.
//
// State shape: null = not fetched yet; { error: true } = the check FAILED (a distinct "couldn't
// check" state — we never silently assume installed); otherwise the cli_status object
// { on_path, resolved, source, target, managed_broken }.
import { call } from "./bridge";
import { toast } from "./chrome";
import { emit } from "./bus";

let state = null;
export function cliState() { return state; }

export async function refreshCliStatus() {
  try { state = await call("cli_status"); }
  catch { state = { error: true }; }
  emit("cli-status", state);
  return state;
}

// Install (or reinstall/repair), then refresh — which broadcasts "cli-status" so every surface updates.
export async function installCli() {
  try {
    const r = await call("install_cli");
    toast(r.note ? `openhydra installed → ${r.path}. ${r.note}` : `openhydra installed → ${r.path}`);
  } catch (e) {
    toast(/cancel/i.test(String(e)) ? "CLI install cancelled" : `Install failed: ${e}`);
  }
  await refreshCliStatus();
}
