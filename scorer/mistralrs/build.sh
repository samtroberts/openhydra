#!/usr/bin/env bash
# Build openhydra-mlx-scorer: mistral.rs (MISTRALRS_TAG, pinned to MISTRALRS_COMMIT) with OpenHydra's
# patch — POST /v1/prompt_logprobs (teacher-forced prompt scoring), loading of MLX-converted Qwen3.5 /
# Qwen3-Next checkpoints (mlx_lm moves their conv1d weights channels-last and bakes +1 into their
# (1 + w) norms), and `serve --exit-on-stdin-eof`. The agent's side-scorer runs it for LM Studio's MLX
# models (agent/src/side_scorer.rs).
#
#   scorer/mistralrs/build.sh <out-path>
#
# Built without code execution (a scorer has no use for it), Metal on macOS, symbols stripped; writes
# <out>.notices.txt beside it (the licences of every crate compiled in). Needs
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
# mistral.rs's own .cargo/config.toml builds for target-cpu=native, which would tie the binary to the
# build machine's CPU: target the oldest Apple Silicon instead (RUSTFLAGS overrides that config). Its
# Metal kernels are precompiled for macOS only — the iOS/tvOS ones would never be loaded.
if [[ "$(uname -s)" == Darwin ]]; then
  FEATURES=(--features metal)
  export RUSTFLAGS="-C target-cpu=apple-m1" MISTRALRS_METAL_PLATFORMS=macos
fi
# --locked: exactly mistral.rs's own lockfile at the pinned commit (a stale one fails, not silently re-resolved).
(cd "$SRC" && cargo build --locked --release -p mistralrs-cli --no-default-features ${FEATURES[@]+"${FEATURES[@]}"}) > "$WORK/build.log" 2>&1 \
  || { tail -20 "$WORK/build.log" >&2; exit 1; }
# Licence notices of every crate compiled in (same target and features as the build).
(cd "$SRC" && "$(command -v python3 || command -v python)" "$HERE/../notices.py" cargo Cargo.toml mistralrs-cli "$WORK/notices.txt" \
  --target "$(rustc -vV | awk '/host:/{print $2}')" --no-default-features --locked ${FEATURES[@]+"${FEATURES[@]}"})
BIN="$WORK/openhydra-mlx-scorer"
cp "$SRC/target/release/mistralrs" "$BIN"
strip -x "$BIN" 2>/dev/null || true
# The patch must be in (the agent starts it with these flags) before it replaces $OUT.
# Captured first: piped straight into `grep -q`, the binary can die of SIGPIPE once grep has matched,
# and pipefail would then fail the check of a binary that passes it.
HELP="$("$BIN" serve --help 2>&1 || true)"
grep -q -- '--exit-on-stdin-eof' <<<"$HELP" || { echo "scorer/mistralrs/build.sh: $BIN lacks the OpenHydra patch" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"
mv "$BIN" "$OUT"
cp "$WORK/notices.txt" "$OUT.notices.txt"
echo "scorer/mistralrs/build.sh: built $OUT (mistral.rs $TAG + OpenHydra patch)"
