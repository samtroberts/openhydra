#!/bin/sh
# OpenHydra CLI installer — installs the `openhydra` command for your platform (the binary is
# distributed as `openhydra-agent-<target>`; it's installed as `openhydra`, with an `openhydra-agent`
# compatibility symlink), the prompt scorers beside it (what a provider answers authenticity checks
# with; the agent looks for them next to itself), and the third-party licence notices.
#
# Usage:
#   curl -fsSL https://openhydra.co/install.sh | sh
#
# Environment overrides:
#   OPENHYDRA_VERSION    release to install, e.g. v0.4.0       (default: latest)
#   OPENHYDRA_BIN_DIR    install directory                     (default: /usr/local/bin, else ~/.local/bin)
#   OPENHYDRA_REPO       GitHub owner/repo for releases        (default: samtroberts/openhydra)
#   OPENHYDRA_DIST_BASE  base URL hosting the per-platform assets
#                        (default: https://dl.openhydra.co/latest, or …/v<VERSION> for a
#                        pinned version — Cloudflare R2). Set this to pin a different
#                        origin; GitHub Releases is used as an automatic fallback only
#                        when this is unset.
#   OPENHYDRA_NO_SCORER  set to 1 to skip the prompt scorers (a provider then can't answer
#                        authenticity checks, and stays unverified)
#
# Windows: use the .msi / -setup.exe installer or the desktop app instead.
set -eu

REPO="${OPENHYDRA_REPO:-samtroberts/openhydra}"
VERSION="${OPENHYDRA_VERSION:-latest}"
# The command users type is `openhydra`. The release artifact is still named `openhydra-agent-<target>`
# (see $asset below); it's installed as `openhydra`, plus an `openhydra-agent` compat symlink.
BIN_NAME="openhydra"

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

triple=""
case "$os:$arch" in
  Darwin:arm64|Darwin:aarch64)
    triple="aarch64-apple-darwin" ;;
  Darwin:x86_64)
    err "Intel Macs have no prebuilt binary (Apple Silicon only). Build from source: https://github.com/$REPO#build-from-source" ;;
  Linux:x86_64|Linux:amd64)
    triple="x86_64-unknown-linux-gnu" ;;
  Linux:aarch64|Linux:arm64)
    err "Linux arm64 has no prebuilt binary yet. Build from source: https://github.com/$REPO#build-from-source" ;;
  *)
    err "unsupported platform '$os/$arch'. Build from source: https://github.com/$REPO#build-from-source" ;;
esac

asset="openhydra-agent-$triple"

# --- resolve download URL(s) ---------------------------------------------
# Primary origin: Cloudflare R2, which serves the binaries whether or not the
# source repo is public: latest/, or the immutable v<VERSION>/ of a pinned release.
# Override the whole base with OPENHYDRA_DIST_BASE.
if [ "$VERSION" = "latest" ]; then
  DIST_BASE="${OPENHYDRA_DIST_BASE:-https://dl.openhydra.co/latest}"
  gh_base="https://github.com/$REPO/releases/latest/download"
else
  DIST_BASE="${OPENHYDRA_DIST_BASE:-https://dl.openhydra.co/v${VERSION#v}}"
  gh_base="https://github.com/$REPO/releases/download/v${VERSION#v}"
fi
primary="${DIST_BASE%/}/$asset"
# Fallback: GitHub Releases — only resolves while the repo/release is public.
fallback="$gh_base/$asset"

# --- pick a downloader ----------------------------------------------------
if command -v curl >/dev/null 2>&1; then
  dl() { curl -fsSL "$1" -o "$2"; }
elif command -v wget >/dev/null 2>&1; then
  dl() { wget -qO "$2" "$1"; }
else
  err "need curl or wget to download."
fi

# --- choose install dir ---------------------------------------------------
# HOME can be unset (a minimal container, a service account): never rely on it being there.
home="${HOME:-}"
if [ -n "${OPENHYDRA_BIN_DIR:-}" ]; then
  bindir="$OPENHYDRA_BIN_DIR"
elif [ -d /usr/local/bin ] && [ -w /usr/local/bin ]; then
  bindir="/usr/local/bin"
elif [ -n "$home" ]; then
  bindir="$home/.local/bin"
else
  err "no writable install dir: /usr/local/bin isn't writable and HOME is unset. Set OPENHYDRA_BIN_DIR."
fi
mkdir -p "$bindir" || err "cannot create install dir '$bindir'."

# Fetch release file $1 into $2: the primary origin, else (when it isn't pinned) GitHub.
fetch() {
  if dl "${DIST_BASE%/}/$1" "$2" && [ -s "$2" ]; then
    return 0
  elif [ -z "${OPENHYDRA_DIST_BASE:-}" ] && dl "$gh_base/$1" "$2" && [ -s "$2" ]; then
    info "(primary origin unavailable — used GitHub fallback for $1)"
    return 0
  fi
  return 1
}

# --- download + install ---------------------------------------------------
# Temporary files, removed however the script ends (a moved one is already gone).
tmp="$(mktemp)"; stmp=""; ntmp=""
# Keeps the script's exit status (a cleanup that succeeded mustn't turn a failure into success).
trap 'rc=$?; rm -f "$tmp" "$stmp" "$ntmp"; exit $rc' EXIT
trap 'exit 130' INT TERM
info "Downloading $BIN_NAME for $os_name ($arch)…"
tried="$primary"
[ -z "${OPENHYDRA_DIST_BASE:-}" ] && tried="$tried, $fallback"
fetch "$asset" "$tmp" || err "download failed. Tried: $tried"

# mktemp makes files only their owner can read: readable by all, as installed binaries are (the
# agent hashes its own binary for attestation, which needs read access).
chmod 755 "$tmp"
target="$bindir/$BIN_NAME"
mv "$tmp" "$target" 2>/dev/null || err "cannot write '$target'. Retry with a writable dir, e.g.: OPENHYDRA_BIN_DIR=\"\$HOME/.local/bin\" sh install.sh"

# Compatibility: some scripts/docs invoke the binary as `openhydra-agent` (the desktop sidecar's
# name). Expose that alias next to the primary `openhydra` command — but NEVER clobber a real
# `openhydra-agent` binary the user already has (only create/refresh it when absent or already a symlink).
if [ ! -e "$bindir/openhydra-agent" ] || [ -L "$bindir/openhydra-agent" ]; then
  ln -sf "$BIN_NAME" "$bindir/openhydra-agent" 2>/dev/null || true
fi

# macOS: clear any quarantine flag so Gatekeeper won't block the unsigned binary.
if [ "$os" = "Darwin" ]; then
  xattr -d com.apple.quarantine "$target" >/dev/null 2>&1 || true
fi

info "Installed → $target"

# --- prompt scorers --------------------------------------------------------
# What a provider answers authenticity (log-prob) checks with: patched llama.cpp for GGUF models
# (Ollama, LM Studio, llama.cpp), and on macOS patched mistral.rs for LM Studio's MLX models. The
# agent finds them next to itself. Optional: without them the CLI works, but a provider's models
# can't be verified.
scorers="openhydra-scorer"
if [ "$os" = "Darwin" ]; then scorers="$scorers openhydra-mlx-scorer"; fi
# Said when a scorer for this release couldn't be installed: an older one left in place still runs,
# but may not match this agent.
no_scorer() {
  if [ -e "$bindir/$1" ]; then
    info "WARNING: $2; kept the existing $bindir/$1, which may not match this version."
  else
    info "WARNING: $2; authenticity checks of your models will be refused."
  fi
}
if [ "${OPENHYDRA_NO_SCORER:-}" != "1" ]; then
  for s in $scorers; do
    stmp="$(mktemp)"
    if fetch "$s-$triple" "$stmp" 2>/dev/null; then
      chmod 755 "$stmp"
      if mv "$stmp" "$bindir/$s" 2>/dev/null; then
        if [ "$os" = "Darwin" ]; then xattr -d com.apple.quarantine "$bindir/$s" >/dev/null 2>&1 || true; fi
        info "Installed → $bindir/$s"
      else
        rm -f "$stmp"
        no_scorer "$s" "couldn't write $bindir/$s"
      fi
    else
      rm -f "$stmp"
      no_scorer "$s" "no $s for this release"
    fi
  done
else
  for s in $scorers; do
    if [ -e "$bindir/$s" ]; then info "OPENHYDRA_NO_SCORER=1: left the existing $bindir/$s as it was."; fi
  done
fi

# --- licence notices -------------------------------------------------------
# The third-party code compiled into the binaries above, and its licences.
# Beside the binaries' prefix when that is a standard one (/usr/local/bin → /usr/local/share), else
# in the user's own share directory (not a stray ~/share for OPENHYDRA_BIN_DIR=~/bin).
prefix="$(dirname "$bindir")"
# The user's own share directory; without a HOME, the one beside the binaries.
if [ -n "$home" ]; then usershare="$home/.local/share/openhydra"; else usershare="$prefix/share/openhydra"; fi
case "$prefix" in
  /usr/local|/usr|/opt/homebrew|/opt/local) sharedir="$prefix/share/openhydra" ;;
  *) if [ -n "$home" ] && [ "$prefix" = "$home/.local" ]; then sharedir="$prefix/share/openhydra"; else sharedir="$usershare"; fi ;;
esac
mkdir -p "$sharedir" 2>/dev/null && [ -w "$sharedir" ] || sharedir="$usershare"
mkdir -p "$sharedir" 2>/dev/null || true
ntmp="$(mktemp)"
if fetch "THIRD_PARTY_NOTICES-$triple.txt" "$ntmp" 2>/dev/null && chmod 644 "$ntmp" && mv "$ntmp" "$sharedir/THIRD_PARTY_NOTICES.txt" 2>/dev/null; then
  info "Licence notices → $sharedir/THIRD_PARTY_NOTICES.txt"
else
  rm -f "$ntmp"
fi

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
  $BIN_NAME serve            # → http://127.0.0.1:16527

  # …or wire a coding tool (Claude Code, OpenCode, Hermes, Pi) to the gateway and run it:
  $BIN_NAME launch claude

Docs: https://github.com/$REPO#quick-start
EOF
