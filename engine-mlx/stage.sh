#!/usr/bin/env bash
# Put the engine's Metal kernels beside an executable as `mlx.metallib`, where MLX looks first and where
# the agent checks them against the digest pinned at build time. Run it for every directory a
# `--features mlx-swift` binary runs from (target/debug, target/release). Where it goes inside a packaged
# .app (codesigning keeps non-code files out of Contents/MacOS) is decided with desktop bundling.
#
#   engine-mlx/stage.sh <dir> [<dir>…]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="${OHMLX_PRODUCTS:-$HERE/.xcbuild/Build/Products/Release}/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"
[[ -f "$SRC" ]] || { echo "engine-mlx/stage.sh: no $SRC — run engine-mlx/build.sh first" >&2; exit 1; }
[[ $# -gt 0 ]] || { echo "usage: engine-mlx/stage.sh <dir> [<dir>…]" >&2; exit 1; }
WANT="$(shasum -a 256 "$SRC" | cut -d' ' -f1)"
EXPECTED="$(cut -d' ' -f1 < "$HERE/METALLIB_SHA256")"
if [[ "$WANT" != "$EXPECTED" ]]; then
  [[ "${OHMLX_ALLOW_UNPINNED_METALLIB:-}" == 1 ]] \
    || { echo "engine-mlx/stage.sh: default.metallib is $WANT, not the committed $EXPECTED (set OHMLX_ALLOW_UNPINNED_METALLIB=1 for development only)" >&2; exit 1; }
  echo "engine-mlx/stage.sh: warning: default.metallib is $WANT, not the committed $EXPECTED (development only)" >&2
fi
for dir in "$@"; do
  mkdir -p "$dir"
  cp "$SRC" "$dir/mlx.metallib"
  GOT="$(shasum -a 256 "$dir/mlx.metallib" | cut -d' ' -f1)"
  [[ "$GOT" == "$WANT" ]] || { echo "engine-mlx/stage.sh: copy to $dir does not match" >&2; exit 1; }
  echo "engine-mlx/stage.sh: $dir/mlx.metallib ($GOT)"
done
