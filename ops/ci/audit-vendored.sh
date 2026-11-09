#!/bin/bash
# Audit the crates we vendor (network/patched-deps/*) against the RustSec advisory database.
#
# `cargo audit` skips path dependencies, so a published advisory for the upstream version we copied
# would never be reported (checked 2026-10-29 with a planted advisory: reported for a registry crate,
# silent for the vendored libp2p-mdns). This lists each vendored crate's upstream name and version as
# a crates.io package in a throwaway lock file and audits that, so cargo-audit's own version matching
# decides. Extra arguments go to `cargo audit` (e.g. --db DIR --no-fetch).
set -euo pipefail
cd "$(dirname "$0")/../.."
lock=$(mktemp -d)/Cargo.lock
trap 'rm -rf "$(dirname "$lock")"' EXIT
echo 'version = 3' > "$lock"
n=0
for manifest in network/patched-deps/*/Cargo.toml; do
  # The [package] table's own name and version (the first ones in the file).
  name=$(sed -n 's/^name = "\(.*\)"$/\1/p' "$manifest" | head -1)
  version=$(sed -n 's/^version = "\(.*\)"$/\1/p' "$manifest" | head -1)
  [ -n "$name" ] && [ -n "$version" ] || { echo "audit-vendored: no name/version in $manifest" >&2; exit 2; }
  printf '\n[[package]]\nname = "%s"\nversion = "%s"\nsource = "registry+https://github.com/rust-lang/crates.io-index"\n' \
    "$name" "$version" >> "$lock"
  echo "vendored: $name $version"
  n=$((n + 1))
done
[ "$n" -gt 0 ] || { echo "audit-vendored: no vendored crates found" >&2; exit 2; }
cargo audit --file "$lock" "$@"
