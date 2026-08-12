#!/usr/bin/env bash
# Build signed APT + YUM/DNF repositories from the release's .deb/.rpm artifacts so Linux users
# get auto-updates through their own package manager (`apt upgrade` / `dnf upgrade`) — the
# Tauri updater only self-updates the AppImage, never system packages.
#
# WHERE THIS RUNS: the Linux release job in CI (ubuntu-latest), AFTER the .deb/.rpm are built and
# BEFORE the R2 upload. NOT on macOS — it needs dpkg-dev/apt-utils/createrepo_c/rpm/gnupg.
#
# WHAT THE MAINTAINER MUST PROVIDE (one-time):
#   * A GPG signing keypair dedicated to package signing. Generate once, locally:
#       gpg --quick-generate-key "OpenHydra Packages <packages@openhydra.co>" default default never
#       gpg --armor --export-secret-keys <KEYID>   > private.asc   # -> CI secret APT_GPG_PRIVATE_KEY
#       gpg --armor --export        <KEYID>        > openhydra.gpg  # the PUBLIC key users import
#     Store the PRIVATE key + its passphrase as CI secrets; commit/publish only the PUBLIC key.
#   * CI secrets: APT_GPG_PRIVATE_KEY, APT_GPG_PASSPHRASE, and the existing R2 upload creds.
#
# INPUTS (env):
#   IN_DIR   dir holding the built *.deb and *.rpm            (default: ./dist)
#   OUT_DIR  where the repo trees are written for R2 upload   (default: ./repo-out)
#   GPG_KEY_ID  signing key id/email (already imported into the keyring)
#   REPO_ORIGIN  apt Origin/Label                             (default: OpenHydra)
# The caller uploads $OUT_DIR/apt -> r2:dl.openhydra.co/apt and $OUT_DIR/rpm -> r2:.../rpm.
set -euo pipefail

IN_DIR="${IN_DIR:-dist}"
OUT_DIR="${OUT_DIR:-repo-out}"
REPO_ORIGIN="${REPO_ORIGIN:-OpenHydra}"
: "${GPG_KEY_ID:?set GPG_KEY_ID to the imported signing key id/email}"
: "${APT_GPG_PASSPHRASE:=}"   # empty is allowed (no-passphrase key)

# ⚠️ UNTESTED IN CI — the signing wiring below (loopback for gpg, agent-preset for rpm) is the
# well-known non-interactive pattern but MUST be validated in a CI dry-run before a real release;
# non-interactive `rpm --addsign` in particular is finicky across distros.

# Prime gpg-agent so BOTH gpg (loopback) and `rpm --addsign` (which drives gpg-agent) can sign with
# no tty. Without this, a passphrase-protected key makes gpg/rpm fail with a pinentry/ioctl error.
mkdir -p "$HOME/.gnupg" && chmod 700 "$HOME/.gnupg"
{ echo "allow-loopback-pinentry"; echo "allow-preset-passphrase"; } >> "$HOME/.gnupg/gpg-agent.conf"
gpgconf --kill gpg-agent 2>/dev/null || true
gpg-connect-agent reloadagent /bye >/dev/null 2>&1 || true
if [ -n "$APT_GPG_PASSPHRASE" ]; then
  KEYGRIP=$(gpg --batch --with-colons --with-keygrip --list-secret-keys "$GPG_KEY_ID" | awk -F: '/^grp:/{print $10; exit}')
  PRESET=$(command -v gpg-preset-passphrase || echo /usr/lib/gnupg/gpg-preset-passphrase)
  "$PRESET" --preset --passphrase "$APT_GPG_PASSPHRASE" "$KEYGRIP" \
    || echo "warn: could not preset passphrase into agent — rpm --addsign may prompt/fail" >&2
fi

# Sign with loopback pinentry so our own gpg calls take the passphrase non-interactively.
gpg_sign() {
  gpg --batch --yes --pinentry-mode loopback --passphrase "$APT_GPG_PASSPHRASE" \
      --default-key "$GPG_KEY_ID" "$@"
}

APT="$OUT_DIR/apt"
RPM="$OUT_DIR/rpm"
rm -rf "$OUT_DIR"; mkdir -p "$APT/pool/main" "$APT/dists/stable/main/binary-amd64" "$RPM/x86_64"
built_any=0   # set when at least one repo (apt or rpm) is actually produced; guards against
              # publishing an empty/Release-less tree when a build produced no artifacts.

# Public key both ecosystems import (same key, two conventional filenames/locations).
gpg --armor --export "$GPG_KEY_ID" > "$APT/openhydra.gpg"
cp "$APT/openhydra.gpg" "$RPM/RPM-GPG-KEY-openhydra"

# ── APT (Debian/Ubuntu) ─────────────────────────────────────────────────────────────────────
cp "$IN_DIR"/*.deb "$APT/pool/main/" 2>/dev/null || { echo "no .deb in $IN_DIR"; }
if ls "$APT"/pool/main/*.deb >/dev/null 2>&1; then
  ( cd "$APT"
    dpkg-scanpackages --arch amd64 pool/ > dists/stable/main/binary-amd64/Packages
    gzip -9c dists/stable/main/binary-amd64/Packages > dists/stable/main/binary-amd64/Packages.gz
    # Release file over the component, then detach- and clear-sign it (Release.gpg + InRelease).
    apt-ftparchive \
      -o APT::FTPArchive::Release::Origin="$REPO_ORIGIN" \
      -o APT::FTPArchive::Release::Label="$REPO_ORIGIN" \
      -o APT::FTPArchive::Release::Suite=stable \
      -o APT::FTPArchive::Release::Codename=stable \
      -o APT::FTPArchive::Release::Architectures=amd64 \
      -o APT::FTPArchive::Release::Components=main \
      release dists/stable > dists/stable/Release
    gpg_sign -abs -o dists/stable/Release.gpg dists/stable/Release
    gpg_sign --clearsign -o dists/stable/InRelease dists/stable/Release
  )
  built_any=1
  echo "APT repo built at $APT"
fi

# ── YUM/DNF (Fedora/RHEL) ───────────────────────────────────────────────────────────────────
cp "$IN_DIR"/*.rpm "$RPM/x86_64/" 2>/dev/null || { echo "no .rpm in $IN_DIR"; }
if ls "$RPM"/x86_64/*.rpm >/dev/null 2>&1; then
  # Sign each package (needs ~/.rpmmacros pointing %_gpg_name at $GPG_KEY_ID).
  printf '%%_gpg_name %s\n' "$GPG_KEY_ID" > "$HOME/.rpmmacros"
  rpm --addsign "$RPM"/x86_64/*.rpm
  createrepo_c "$RPM/x86_64"
  # Detach-sign the repo metadata so dnf can verify repomd.xml.
  gpg_sign --detach-sign --armor "$RPM/x86_64/repodata/repomd.xml"
  built_any=1
  # The .repo file users drop into /etc/yum.repos.d/.
  cat > "$RPM/openhydra.repo" <<EOF
[openhydra]
name=OpenHydra
baseurl=https://dl.openhydra.co/rpm/x86_64
enabled=1
gpgcheck=1
repo_gpgcheck=1
gpgkey=https://dl.openhydra.co/rpm/RPM-GPG-KEY-openhydra
EOF
  echo "RPM repo built at $RPM"
fi

# Refuse to "succeed" with nothing: a build that produced no .deb/.rpm would otherwise upload an
# empty, Release-less tree that users' apt/dnf then error on. Fail loudly so CI catches it.
if [ "$built_any" -eq 0 ]; then
  echo "ERROR: no .deb or .rpm found in $IN_DIR — refusing to publish an empty repo." >&2
  exit 1
fi

echo "Done. Upload $APT -> dl.openhydra.co/apt and $RPM -> dl.openhydra.co/rpm (preserve paths)."
