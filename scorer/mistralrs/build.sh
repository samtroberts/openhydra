#!/usr/bin/env bash
# Build openhydra-mlx-scorer: mistral.rs (MISTRALRS_TAG, pinned to MISTRALRS_COMMIT) with OpenHydra's
# patch — POST /v1/prompt_logprobs (teacher-forced prompt scoring), loading of MLX-converted Qwen3.5 /
# Qwen3-Next checkpoints (mlx_lm moves their conv1d weights channels-last and bakes +1 into their
# (1 + w) norms), and `serve --exit-on-stdin-eof`. The agent's side-scorer runs it for LM Studio's MLX
# models (agent/src/side_scorer.rs).
#
#   scorer/mistralrs/build.sh <out-path>
#
# Built without code execution (a scorer has no use for it), Metal on macOS, symbols stripped. Needs
# Rust 1.94+ (RUSTUP_TOOLCHAIN=1.94 selects it if the default is older). MISTRALRS_SRC=<pristine
# checkout at that commit> skips the fetch.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:?usage: scorer/mistralrs/build.sh <out-path>}"
TAG="$(tr -d ' \r\n' < "$HERE/MISTRALRS_TAG")"
COMMIT="$(tr -d ' \r\n' < "$HERE/MISTRALRS_COMMIT")"
PATCH="$HERE/mistralrs-openhydra.patch"
WORK="${SCORER_WORK:-$HERE/.build}"
SRC="$WORK/mistral.rs"
mkdir -p "$WORK"

if [[ -n "${MISTRALRS_SRC:-}" ]]; then
  rm -rf "$SRC"
  git clone -q --no-hardlinks "$MISTRALRS_SRC" "$SRC"
  git -C "$SRC" checkout -q --force "$COMMIT"
else
  if [[ ! -d "$SRC/.git" ]]; then
    git init -q "$SRC"
    git -C "$SRC" remote add origin https://github.com/EricLBuehler/mistral.rs
  fi
  git -C "$SRC" fetch -q --depth 1 origin "refs/tags/$TAG:refs/tags/$TAG"
  git -C "$SRC" checkout -q --force "$TAG"
fi
[[ "$(git -C "$SRC" rev-parse HEAD)" == "$COMMIT" ]] || { echo "scorer/mistralrs/build.sh: $TAG is not $COMMIT" >&2; exit 1; }
git -C "$SRC" clean -qfdx -e target
git -C "$SRC" apply "$PATCH"

FEATURES=()
[[ "$(uname -s)" == Darwin ]] && FEATURES=(--features metal)
(cd "$SRC" && cargo build --release -p mistralrs-cli --no-default-features "${FEATURES[@]}") > "$WORK/build.log" 2>&1 \
  || { tail -20 "$WORK/build.log" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"
cp "$SRC/target/release/mistralrs" "$OUT"
strip -x "$OUT" 2>/dev/null || true
# The patch must be in: the agent starts it with these flags.
"$OUT" serve --help 2>&1 | grep -q -- '--exit-on-stdin-eof' || { echo "scorer/mistralrs/build.sh: $OUT lacks the OpenHydra patch" >&2; exit 1; }
echo "scorer/mistralrs/build.sh: built $OUT (mistral.rs $TAG + OpenHydra patch)"
