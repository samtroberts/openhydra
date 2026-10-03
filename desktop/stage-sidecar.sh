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
# The prompt scorer (patched llama-server) the agent runs for Ollama models' authenticity checks.
# Rebuilt only when the pinned llama.cpp commit or the patch changes (a build takes minutes).
SCORER="desktop/src-tauri/binaries/openhydra-scorer-$TRIPLE$EXT"
STAMP="$(cat scorer/LLAMACPP_COMMIT scorer/llamacpp-prompt-scoring.patch scorer/build.sh | shasum -a 256 | cut -c1-16)"
if [[ ! -x "$SCORER" || "$(cat "$SCORER.stamp" 2>/dev/null)" != "$STAMP" ]]; then
  scorer/build.sh "$SCORER" && echo "$STAMP" > "$SCORER.stamp"
fi
echo "staged sidecar: openhydra-scorer-$TRIPLE$EXT"
# macOS also ships the MLX scorer (LM Studio's MLX models), bundled via tauri.macos.conf.json.
if [[ "$TRIPLE" == *apple-darwin ]]; then
  scorer/mlx/build.sh scorer/mlx/target/stage
  cp scorer/mlx/target/stage/openhydra-mlx-scorer "desktop/src-tauri/binaries/openhydra-mlx-scorer-$TRIPLE"
  cp scorer/mlx/target/stage/mlx.metallib desktop/src-tauri/binaries/mlx.metallib   # must sit next to it (MacOS/)
  echo "staged sidecar: openhydra-mlx-scorer-$TRIPLE + mlx.metallib"
fi
