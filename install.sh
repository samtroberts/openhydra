#!/bin/sh
# OpenHydra CLI installer — installs the `openhydra` command for your platform (the binary is
# distributed as `openhydra-agent-<target>`; it's installed as `openhydra`, with an `openhydra-agent`
# compatibility symlink), the prompt scorers beside it (what a provider answers authenticity checks
# with; the agent looks for them next to itself), and the third-party licence notices.
#
# Usage:
#   curl -fsSL https://openhydra.co/install.sh | sh
#
# Every file is checked against the release's SHA256SUMS-<target> before anything is installed. The
# checksums come from the GitHub release, a different origin from the downloads; if `minisign` is
# installed, their signature is checked too. A release from 0.3.22 on that has no checksums is refused.
#
# Environment overrides:
#   OPENHYDRA_VERSION    release to install, e.g. v0.4.0       (default: latest)
#   OPENHYDRA_BIN_DIR    install directory                     (default: /usr/local/bin, else ~/.local/bin)
#   OPENHYDRA_REPO       GitHub owner/repo for releases        (default: samtroberts/openhydra)
#   OPENHYDRA_DIST_BASE  base URL hosting the per-platform assets
#                        (default: https://dl.openhydra.co/v<VERSION> — Cloudflare R2 — with the
#                        latest version read from https://dl.openhydra.co/latest.json). Set this to
#                        pin a different origin; GitHub Releases is used as an automatic fallback
#                        only when this is unset.
#   OPENHYDRA_NO_SCORER  set to 1 to skip the prompt scorers (a provider then can't answer
#                        authenticity checks, and stays unverified)
#   OPENHYDRA_DIST_HOST, OPENHYDRA_GITHUB_URL
#                        the download host and GitHub's address, for a mirror
#                        (default: https://dl.openhydra.co, https://github.com)
#
# Windows: use the .msi / -setup.exe installer or the desktop app instead.
#
# Everything runs from `main`, called on the last line: a download cut off part way runs nothing.
set -eu

REPO="${OPENHYDRA_REPO:-samtroberts/openhydra}"
VERSION="${OPENHYDRA_VERSION:-latest}"
DIST_HOST="${OPENHYDRA_DIST_HOST:-https://dl.openhydra.co}"
GITHUB_URL="${OPENHYDRA_GITHUB_URL:-https://github.com}"
# The command users type is `openhydra`. The release artifact is still named `openhydra-agent-<target>`
# (see $asset below); it's installed as `openhydra`, plus an `openhydra-agent` compat symlink.
BIN_NAME="openhydra"
# The first release published with SHA256SUMS: from it on, a release without them is refused (so
# deleting the checksums can't turn the check off).
SUMS_SINCE="0.3.22"
# The CLI release key (minisign) the checksums are signed with. Empty until the key exists: then only
# the checksums are checked.
MINISIGN_PUBKEY=""

info() { printf '  %s\n' "$*"; }
err()  { printf 'error: %s\n' "$*" >&2; exit 1; }

# Whether version $1 is at least $2 (numeric x.y.z; a pre-release counts as its release).
ver_ge() {
  awk -v a="${1%%-*}" -v b="${2%%-*}" 'BEGIN {
    split(a, x, "."); split(b, y, ".")
    for (i = 1; i <= 3; i++) { if (x[i] + 0 > y[i] + 0) exit 0; if (x[i] + 0 < y[i] + 0) exit 1 }
    exit 0 }'
}

# The version latest.json ($1) lists for platform $2 (its "versions" entry, else its "version").
manifest_version() {
  awk -v k="\"$2\"" '
    /"versions"[ \t]*:/ { inv = 1; next }
    inv && /}/ { inv = 0 }
    inv && index($0, k) && match($0, /:[ \t]*"[^"]+"/) { v = substr($0, RSTART, RLENGTH); gsub(/[: \t"]/, "", v); print v; found = 1; exit }
    !inv && /^[ \t]*"version"[ \t]*:/ && top == "" && match($0, /:[ \t]*"[^"]+"/) { top = substr($0, RSTART, RLENGTH); gsub(/[: \t"]/, "", top) }
    END { if (!found && top != "") print top }' "$1"
}

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'
  fi
}

main() {
  # --- detect platform ------------------------------------------------------
  os="$(uname -s)"
  arch="$(uname -m)"

  case "$os" in
    Darwin) os_name="macOS"; os_key="macos" ;;
    Linux)  os_name="Linux"; os_key="linux" ;;
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

  # --- pick a downloader ----------------------------------------------------
  # HTTPS only, redirects included, TLS 1.2 or newer.
  if command -v curl >/dev/null 2>&1; then
    dl() { curl -fsSL --proto '=https' --proto-redir '=https' --tlsv1.2 "$1" -o "$2"; }
  elif command -v wget >/dev/null 2>&1; then
    dl() { case "$1" in https://*) wget -q --https-only --secure-protocol=TLSv1_2 -O "$2" "$1" ;; *) return 1 ;; esac; }
  else
    err "need curl or wget to download."
  fi
  if ! command -v sha256sum >/dev/null 2>&1 && ! command -v shasum >/dev/null 2>&1; then
    err "need sha256sum or shasum to check the downloads."
  fi

  # Temporary files, removed however the script ends (a moved one is already gone).
  tmpdir="$(mktemp -d)"
  # Keeps the script's exit status (a cleanup that succeeded mustn't turn a failure into success).
  trap 'rc=$?; rm -rf "$tmpdir"; exit $rc' EXIT
  trap 'exit 130' INT TERM

  # --- resolve the version, then the download URLs --------------------------
  # One version for every file: "latest" is read once from latest.json (or GitHub's latest release),
  # and everything then comes from that version's own immutable v<VERSION>/ — never from latest/,
  # which a publish in progress could have half replaced.
  ver=""
  if [ "$VERSION" != "latest" ]; then
    ver="${VERSION#v}"
  elif [ -z "${OPENHYDRA_DIST_BASE:-}" ]; then
    if dl "${DIST_HOST%/}/latest.json" "$tmpdir/latest.json" 2>/dev/null; then
      ver="$(manifest_version "$tmpdir/latest.json" "$os_key")"
    fi
    if [ -z "$ver" ] && command -v curl >/dev/null 2>&1; then
      # GitHub redirects …/releases/latest to …/releases/tag/v<VERSION>.
      loc="$(curl -fsSI --proto '=https' --tlsv1.2 -o /dev/null -w '%{redirect_url}' "${GITHUB_URL%/}/$REPO/releases/latest" 2>/dev/null || true)"
      case "$loc" in */tag/v*) ver="${loc##*/tag/v}" ;; esac
    fi
    [ -n "$ver" ] || err "couldn't find the latest version (tried ${DIST_HOST%/}/latest.json and GitHub). Set OPENHYDRA_VERSION."
  fi
  case "$ver" in
    *[!0-9A-Za-z.-]*) err "not a version: '$ver'" ;;
  esac

  if [ -n "$ver" ]; then
    DIST_BASE="${OPENHYDRA_DIST_BASE:-${DIST_HOST%/}/v$ver}"
    gh_base="${GITHUB_URL%/}/$REPO/releases/download/v$ver"
  else
    # A pinned origin with the latest version: its files as they are, checked against its own
    # checksums when it has them.
    DIST_BASE="$OPENHYDRA_DIST_BASE"
    gh_base=""
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
    if dl "${DIST_BASE%/}/$1" "$2" 2>/dev/null && [ -s "$2" ]; then
      return 0
    elif [ -z "${OPENHYDRA_DIST_BASE:-}" ] && [ -n "$gh_base" ] && dl "$gh_base/$1" "$2" 2>/dev/null && [ -s "$2" ]; then
      info "(primary origin unavailable — used GitHub fallback for $1)"
      return 0
    fi
    return 1
  }

  # --- checksums --------------------------------------------------------------
  # From the GitHub release (a different origin from the files), else from the download origin.
  sums="$tmpdir/SHA256SUMS"
  sums_from=""
  if [ -n "$gh_base" ] && dl "$gh_base/SHA256SUMS-$triple" "$sums" 2>/dev/null && [ -s "$sums" ]; then
    sums_from="$gh_base"
  elif dl "${DIST_BASE%/}/SHA256SUMS-$triple" "$sums" 2>/dev/null && [ -s "$sums" ]; then
    sums_from="${DIST_BASE%/}"
    if [ -n "$gh_base" ]; then info "(GitHub unavailable — checksums taken from the download origin)"; fi
  fi
  if [ -z "$sums_from" ]; then
    if [ -n "$ver" ] && ver_ge "$ver" "$SUMS_SINCE"; then
      err "release $ver has no checksums (SHA256SUMS-$triple): refusing to install it."
    fi
    info "WARNING: this release has no checksums (it predates them): its files can't be checked."
  elif [ -n "$MINISIGN_PUBKEY" ]; then
    if command -v minisign >/dev/null 2>&1; then
      dl "$sums_from/SHA256SUMS-$triple.minisig" "$tmpdir/SHA256SUMS.minisig" 2>/dev/null \
        || err "the checksums' signature (SHA256SUMS-$triple.minisig) is missing: refusing to install."
      minisign -V -q -P "$MINISIGN_PUBKEY" -m "$sums" -x "$tmpdir/SHA256SUMS.minisig" >/dev/null 2>&1 \
        || err "the checksums' signature doesn't verify: refusing to install."
      info "Checksums signed by the OpenHydra release key."
    else
      info "(install minisign to also check the release signature)"
    fi
  fi

  # Check downloaded file $1 (published as $2) against the checksums; refuse on any doubt.
  check() {
    [ -n "$sums_from" ] || return 0
    want="$(awk -v n="$2" '$2 == n || $2 == "*" n { print $1; exit }' "$sums")"
    [ -n "$want" ] || err "$2 isn't in the release's checksums: refusing to install it."
    [ "$(sha256_of "$1")" = "$want" ] || err "$2 doesn't match the release's checksum (download corrupted or tampered with): refusing to install."
  }

  # --- download and check everything before installing anything ------------
  where="${DIST_BASE%/}/$asset"
  [ -z "${OPENHYDRA_DIST_BASE:-}" ] && [ -n "$gh_base" ] && where="$where, $gh_base/$asset"
  info "Downloading $BIN_NAME${ver:+ $ver} for $os_name ($arch)…"
  fetch "$asset" "$tmpdir/agent" || err "download failed. Tried: $where"
  check "$tmpdir/agent" "$asset"

  # The prompt scorers: what a provider answers authenticity (log-prob) checks with — patched llama.cpp
  # for GGUF models (Ollama, LM Studio, llama.cpp), and on macOS patched mistral.rs for LM Studio's MLX
  # models. The agent finds them next to itself. Optional: without them the CLI works, but a provider's
  # models can't be verified.
  scorers="openhydra-scorer"
  if [ "$os" = "Darwin" ]; then scorers="$scorers openhydra-mlx-scorer"; fi
  got_scorers=""
  if [ "${OPENHYDRA_NO_SCORER:-}" != "1" ]; then
    for s in $scorers; do
      if fetch "$s-$triple" "$tmpdir/$s"; then
        check "$tmpdir/$s" "$s-$triple"
        got_scorers="$got_scorers $s"
      fi
    done
  fi
  got_notices=""
  if fetch "THIRD_PARTY_NOTICES-$triple.txt" "$tmpdir/notices"; then
    check "$tmpdir/notices" "THIRD_PARTY_NOTICES-$triple.txt"
    got_notices=1
  fi
  if [ -n "$sums_from" ]; then info "Checksums match."; fi

  # --- install ----------------------------------------------------------------
  # mktemp makes files only their owner can read: readable by all, as installed binaries are (the
  # agent hashes its own binary for attestation, which needs read access).
  chmod 755 "$tmpdir/agent"
  target="$bindir/$BIN_NAME"
  mv "$tmpdir/agent" "$target" 2>/dev/null || err "cannot write '$target'. Retry with a writable dir, e.g.: OPENHYDRA_BIN_DIR=\"\$HOME/.local/bin\" sh install.sh"

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
      case " $got_scorers " in
        *" $s "*)
          chmod 755 "$tmpdir/$s"
          if mv "$tmpdir/$s" "$bindir/$s" 2>/dev/null; then
            if [ "$os" = "Darwin" ]; then xattr -d com.apple.quarantine "$bindir/$s" >/dev/null 2>&1 || true; fi
            info "Installed → $bindir/$s"
          else
            no_scorer "$s" "couldn't write $bindir/$s"
          fi ;;
        *) no_scorer "$s" "no $s for this release" ;;
      esac
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
  if [ -n "$got_notices" ]; then
    prefix="$(dirname "$bindir")"
    # The user's own share directory; without a HOME, the one beside the binaries.
    if [ -n "$home" ]; then usershare="$home/.local/share/openhydra"; else usershare="$prefix/share/openhydra"; fi
    case "$prefix" in
      /usr/local|/usr|/opt/homebrew|/opt/local) sharedir="$prefix/share/openhydra" ;;
      *) if [ -n "$home" ] && [ "$prefix" = "$home/.local" ]; then sharedir="$prefix/share/openhydra"; else sharedir="$usershare"; fi ;;
    esac
    mkdir -p "$sharedir" 2>/dev/null && [ -w "$sharedir" ] || sharedir="$usershare"
    mkdir -p "$sharedir" 2>/dev/null || true
    if chmod 644 "$tmpdir/notices" && mv "$tmpdir/notices" "$sharedir/THIRD_PARTY_NOTICES.txt" 2>/dev/null; then
      info "Licence notices → $sharedir/THIRD_PARTY_NOTICES.txt"
    fi
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
}

main "$@"
