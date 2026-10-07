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
# Rebuilt only when the pinned llama.cpp commit or the patch changes (a build takes minutes). A failed
# build stops this script (set -e; not inside an && list, where bash would carry on).
# A stamp records what a staged scorer was built from AND the SHA-256 of the binary and its notices
# as staged: a binary or notices file replaced or edited since (a stale copy, a partial write) no
# longer matches its stamp, and is rebuilt (review N6).
# A file that can't be hashed gets a value no stamp holds, so it is rebuilt rather than trusted
# (review round 4, R4-7).
command -v shasum >/dev/null || { echo "stage-sidecar.sh: shasum is needed to check the staged scorers" >&2; exit 1; }
sha() { shasum -a 256 "$1" 2>/dev/null | cut -d' ' -f1 | grep . || echo "unhashed-$RANDOM$RANDOM"; }
stamped() { # <binary> <notices> <inputs hash>: whether the stamp vouches for exactly these files
  [[ -x "$1" && -f "$2" && "$(cat "$1.stamp" 2>/dev/null)" == "$3 $(sha "$1") $(sha "$2")" ]]
}
stamp() { echo "$3 $(sha "$1") $(sha "$2")" > "$1.stamp"; }
SCORER="desktop/src-tauri/binaries/openhydra-scorer-$TRIPLE$EXT"
STAMP="$(cat scorer/LLAMACPP_COMMIT scorer/llamacpp-prompt-scoring.patch scorer/llamacpp-ollama-compat.patch scorer/llamacpp-token-count.patch scorer/build.sh scorer/notices.py $(find scorer/licenses -type f | sort) | shasum -a 256 | cut -c1-16)"
if ! stamped "$SCORER" "${SCORER%.exe}.notices.txt" "$STAMP"; then
  scorer/build.sh "$SCORER"
  stamp "$SCORER" "${SCORER%.exe}.notices.txt" "$STAMP"
fi
echo "staged sidecar: openhydra-scorer-$TRIPLE$EXT"
# macOS also ships the MLX scorer (patched mistral.rs, for LM Studio's MLX models), bundled via
# tauri.macos.conf.json; rebuilt only when its pinned commit or patch changes (Rust 1.94+).
if [[ "$TRIPLE" == *apple-darwin ]]; then
  MLX_SCORER="desktop/src-tauri/binaries/openhydra-mlx-scorer-$TRIPLE"
  MLX_STAMP="$(cat scorer/mistralrs/MISTRALRS_COMMIT scorer/mistralrs/mistralrs-openhydra.patch scorer/mistralrs/build.sh scorer/notices.py $(find scorer/licenses -type f | sort) | shasum -a 256 | cut -c1-16)"
  if ! stamped "$MLX_SCORER" "$MLX_SCORER.notices.txt" "$MLX_STAMP"; then
    scorer/mistralrs/build.sh "$MLX_SCORER"
    stamp "$MLX_SCORER" "$MLX_SCORER.notices.txt" "$MLX_STAMP"
  fi
  echo "staged sidecar: openhydra-mlx-scorer-$TRIPLE"
fi
# The licence notices the app bundles (resources/THIRD_PARTY_NOTICES.txt).
desktop/stage-notices.sh "$TRIPLE"
