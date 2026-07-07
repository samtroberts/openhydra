#!/usr/bin/env bash
# Build the release openhydra-agent and stage it as the Tauri sidecar for the current
# host target. MUST run before a local `tauri build`, or the bundle ships a stale agent
# (symptom: the app's roles exit with clap "unexpected argument" / exit status 2 when
# main.rs gains a new flag the bundled agent doesn't know). The release CI stages the
# sidecar in its own step, so this is for local bundling.
set -euo pipefail
cd "$(dirname "$0")/.."
TRIPLE="$(rustc -vV | awk '/host:/{print $2}')"
EXT=""; [[ "$TRIPLE" == *windows* ]] && EXT=".exe"
cargo build --release -p openhydra-agent
cp "target/release/openhydra-agent$EXT" "desktop/src-tauri/binaries/openhydra-agent-$TRIPLE$EXT"
echo "staged sidecar: openhydra-agent-$TRIPLE$EXT ($(cd target/release && md5 -q openhydra-agent 2>/dev/null || echo built))"
