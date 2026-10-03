#!/usr/bin/env bash
# Build openhydra-mlx-scorer (macOS, Apple Silicon) and put it next to its Metal kernel library:
#
#   scorer/mlx/build.sh <out-dir>     → <out-dir>/openhydra-mlx-scorer + <out-dir>/mlx.metallib
#
# MLX loads mlx.metallib from next to the binary, so the two must ship side by side. MLX is built
# with JIT Metal kernels (.cargo/config.toml → jit.cmake), keeping the library at ~1.4 MB.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:?usage: scorer/mlx/build.sh <out-dir>}"
cargo build --release --manifest-path "$HERE/Cargo.toml"
LIB="$(ls -t "$HERE"/target/release/build/mlx-sys-*/out/build/lib/mlx.metallib 2>/dev/null | head -n 1)"
[[ -n "$LIB" ]] || { echo "scorer/mlx/build.sh: no mlx.metallib was built" >&2; exit 1; }
mkdir -p "$OUT"
cp "$HERE/target/release/openhydra-mlx-scorer" "$OUT/openhydra-mlx-scorer"
cp "$LIB" "$OUT/mlx.metallib"
echo "scorer/mlx/build.sh: built $OUT/openhydra-mlx-scorer + mlx.metallib ($(du -h "$LIB" | cut -f1))"
