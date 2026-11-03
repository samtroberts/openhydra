#!/usr/bin/env bash
# Build OpenHydra's built-in Mac engine (Apple MLX via mlx-swift-lm) with xcodebuild, which compiles MLX's
# Metal shaders into default.metallib (the SwiftPM command line can't). The `openhydra-mlx-engine` crate links
# the resulting objects into openhydra-agent when it is built with `--features mlx-swift`.
#
#   engine-mlx/build.sh            # Release build into engine-mlx/.xcbuild
#
# Needs Xcode (not just the command-line tools) with the Metal toolchain, on Apple Silicon.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || { echo "engine-mlx/build.sh: needs an Apple Silicon Mac" >&2; exit 1; }
LOG="$HERE/.xcbuild.log"
xcodebuild build -scheme OHMLX -destination 'generic/platform=macOS' -configuration Release \
  -derivedDataPath "$HERE/.xcbuild" ARCHS=arm64 ONLY_ACTIVE_ARCH=YES > "$LOG" 2>&1 \
  || { tail -30 "$LOG" >&2; echo "engine-mlx/build.sh: xcodebuild failed (log: $LOG)" >&2; exit 1; }
PRODUCTS="$HERE/.xcbuild/Build/Products/Release"
[[ -f "$PRODUCTS/OHMLX.o" && -f "$PRODUCTS/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib" ]] \
  || { echo "engine-mlx/build.sh: expected OHMLX.o and default.metallib in $PRODUCTS" >&2; exit 1; }
echo "engine-mlx/build.sh: built $PRODUCTS"
