#!/usr/bin/env bash
# Assemble desktop/src-tauri/resources/THIRD_PARTY_NOTICES.txt: the licence notices of the third-party
# code in everything the app ships — the desktop app and the agent (their crates, from cargo tree) and
# each staged scorer (the <binary>.notices.txt its build wrote). Bundled as an app resource
# (tauri.conf.json); the release also publishes it beside the CLI binaries.
#
#   desktop/stage-notices.sh [target-triple]     (default: this host)
set -euo pipefail
cd "$(dirname "$0")/.."
TRIPLE="${1:-$(rustc -vV | awk '/host:/{print $2}')}"
PY="$(command -v python3 || command -v python)"
ROOT="$(pwd -W 2>/dev/null || pwd)"  # a Windows path under Git Bash, as cargo prints them
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
"$PY" scorer/notices.py cargo desktop/src-tauri/Cargo.toml openhydra-desktop "$WORK/desktop.txt" --target "$TRIPLE" --own-root "$ROOT"
"$PY" scorer/notices.py cargo Cargo.toml openhydra-agent "$WORK/agent.txt" --target "$TRIPLE" --own-root "$ROOT"
PARTS=("$WORK/desktop.txt" "$WORK/agent.txt")
for bin in openhydra-scorer openhydra-mlx-scorer; do
  B="desktop/src-tauri/binaries/$bin-$TRIPLE"
  [[ -f "$B" || -f "$B.exe" ]] || continue
  # A staged scorer must not ship without its notices.
  [[ -f "$B.notices.txt" ]] || { echo "stage-notices.sh: $B has no $B.notices.txt (rebuild it with its build.sh)" >&2; exit 1; }
  PARTS+=("$B.notices.txt")
done
mkdir -p desktop/src-tauri/resources
"$PY" scorer/notices.py merge desktop/src-tauri/resources/THIRD_PARTY_NOTICES.txt "${PARTS[@]}"
echo "staged THIRD_PARTY_NOTICES.txt ($(wc -c < desktop/src-tauri/resources/THIRD_PARTY_NOTICES.txt | tr -d ' ') bytes, ${#PARTS[@]} parts)"
