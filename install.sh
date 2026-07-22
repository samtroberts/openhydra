#!/bin/sh
# OpenHydra CLI installer — downloads the latest `openhydra-agent` for your platform.
#
# Usage:
#   curl -fsSL https://openhydra.co/install.sh | sh
#
# Environment overrides:
#   OPENHYDRA_VERSION    release tag to install                (default: latest)
#   OPENHYDRA_BIN_DIR    install directory                     (default: /usr/local/bin, else ~/.local/bin)
#   OPENHYDRA_REPO       GitHub owner/repo for releases        (default: samtroberts/openhydra)
#   OPENHYDRA_DIST_BASE  base URL hosting the per-platform assets
#                        (default: https://dl.openhydra.co/latest — Cloudflare R2).
#                        Set this to pin a different origin; GitHub Releases is used as
#                        an automatic fallback only when this is unset.
#
# Windows: use the .msi / -setup.exe installer or the desktop app instead.
set -eu

REPO="${OPENHYDRA_REPO:-samtroberts/openhydra}"
VERSION="${OPENHYDRA_VERSION:-latest}"
BIN_NAME="openhydra-agent"

info() { printf '  %s\n' "$*"; }
err()  { printf 'error: %s\n' "$*" >&2; exit 1; }

# --- detect platform ------------------------------------------------------
os="$(uname -s)"
arch="$(uname -m)"

case "$os" in
  Darwin) os_name="macOS" ;;
  Linux)  os_name="Linux" ;;
  *) err "unsupported OS '$os'. On Windows, use the .msi/-setup.exe installer or the desktop app." ;;
esac

asset=""
case "$os:$arch" in
  Darwin:arm64|Darwin:aarch64)
    asset="openhydra-agent-aarch64-apple-darwin" ;;
  Darwin:x86_64)
    err "Intel Macs have no prebuilt binary (Apple Silicon only). Build from source: https://github.com/$REPO#build-from-source" ;;
  Linux:x86_64|Linux:amd64)
    asset="openhydra-agent-x86_64-unknown-linux-gnu" ;;
  Linux:aarch64|Linux:arm64)
    err "Linux arm64 has no prebuilt binary yet. Build from source: https://github.com/$REPO#build-from-source" ;;
  *)
    err "unsupported platform '$os/$arch'. Build from source: https://github.com/$REPO#build-from-source" ;;
esac

# --- resolve download URL(s) ---------------------------------------------
# Primary origin: Cloudflare R2, which serves the binaries whether or not the
# source repo is public. Override the whole base with OPENHYDRA_DIST_BASE.
DIST_BASE="${OPENHYDRA_DIST_BASE:-https://dl.openhydra.co/latest}"
primary="${DIST_BASE%/}/$asset"
# Fallback: GitHub Releases — only resolves while the repo/release is public.
if [ "$VERSION" = "latest" ]; then
  fallback="https://github.com/$REPO/releases/latest/download/$asset"
else
  fallback="https://github.com/$REPO/releases/download/$VERSION/$asset"
fi

# --- pick a downloader ----------------------------------------------------
if command -v curl >/dev/null 2>&1; then
  dl() { curl -fsSL "$1" -o "$2"; }
elif command -v wget >/dev/null 2>&1; then
  dl() { wget -qO "$2" "$1"; }
else
  err "need curl or wget to download."
fi

# --- choose install dir ---------------------------------------------------
if [ -n "${OPENHYDRA_BIN_DIR:-}" ]; then
  bindir="$OPENHYDRA_BIN_DIR"
elif [ -d /usr/local/bin ] && [ -w /usr/local/bin ]; then
  bindir="/usr/local/bin"
else
  bindir="$HOME/.local/bin"
fi
mkdir -p "$bindir" || err "cannot create install dir '$bindir'."

# --- download + install ---------------------------------------------------
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT INT TERM
info "Downloading $BIN_NAME for $os_name ($arch)…"
if dl "$primary" "$tmp" && [ -s "$tmp" ]; then
  :
elif [ -z "${OPENHYDRA_DIST_BASE:-}" ] && dl "$fallback" "$tmp" && [ -s "$tmp" ]; then
  info "(primary origin unavailable — used GitHub fallback)"
else
  err "download failed. Tried: $primary"
fi

chmod +x "$tmp"
target="$bindir/$BIN_NAME"
mv "$tmp" "$target" 2>/dev/null || err "cannot write '$target'. Retry with a writable dir, e.g.: OPENHYDRA_BIN_DIR=\"\$HOME/.local/bin\" sh install.sh"
trap - EXIT INT TERM

# macOS: clear any quarantine flag so Gatekeeper won't block the unsigned binary.
if [ "$os" = "Darwin" ]; then
  xattr -d com.apple.quarantine "$target" >/dev/null 2>&1 || true
fi

info "Installed → $target"

# --- PATH hint ------------------------------------------------------------
case ":$PATH:" in
  *":$bindir:"*) : ;;
  *)
    info "NOTE: $bindir is not on your PATH. Add it, e.g.:"
    info "  echo 'export PATH=\"$bindir:\$PATH\"' >> ~/.profile && . ~/.profile" ;;
esac

# --- best-effort verify ---------------------------------------------------
if "$target" --version >/dev/null 2>&1; then
  info "Verified: $("$target" --version 2>/dev/null | head -n1)"
fi

cat <<EOF

OpenHydra CLI is ready. Next:

  # Share your local engine (e.g. Ollama on :11434):
  $BIN_NAME provide --engine-kind ollama

  # …or run the OpenAI-compatible gateway to consume from the network:
  $BIN_NAME serve            # → http://127.0.0.1:8080

Docs: https://github.com/$REPO#quick-start
EOF
