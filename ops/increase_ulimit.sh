#!/usr/bin/env bash
# ops/increase_ulimit.sh
# ─────────────────────────────────────────────────────────────────────────────
# Permanently raises the open-file-descriptor limit to 65 535 on an Ubuntu
# Linode running the OpenHydra DHT bootstrap service.
#
# What it changes
# ───────────────
# 1. /etc/security/limits.conf   — user-space soft + hard nofile for every
#                                  user (including root and the openhydra svc
#                                  account when accessed outside systemd).
# 2. /etc/sysctl.conf            — kernel-level fs.file-max cap raised to
#                                  500 000 so the system can honour the above.
# 3. /etc/systemd/system.conf    — systemd manager-level DefaultLimitNOFILE so
#                                  every new service inherits a sane default.
# NOTE: The bootstrap.service unit already contains LimitNOFILE=65536, so the
#       DHT service itself is already protected.  This script covers the rest.
#
# How to run
# ──────────
# Copy this script to one of the Linodes (or run via the deploy_all.sh SSH
# loop) and execute as root:
#
#   scp ops/increase_ulimit.sh root@<node-ip>:/tmp/
#   ssh root@<node-ip> "bash /tmp/increase_ulimit.sh"
#
# A reboot or at minimum a `systemctl daemon-reexec` is required for systemd
# changes to take effect.  For the PAM limits to apply to *new* SSH sessions
# no reboot is strictly required — log out and back in.
#
# Idempotent: re-running the script is safe; it checks before appending.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

NOFILE_LIMIT=65535
KERNEL_FILEMAX=500000

# ── Guard: must run as root ──────────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
    echo "ERROR: this script must be run as root (sudo bash $0)" >&2
    exit 1
fi

echo "==> [1/3] /etc/security/limits.conf — raising nofile to ${NOFILE_LIMIT}"
LIMITS_CONF=/etc/security/limits.conf
LIMITS_MARKER="# OpenHydra: raised nofile"

if grep -qF "${LIMITS_MARKER}" "${LIMITS_CONF}" 2>/dev/null; then
    echo "    already configured, skipping."
else
    cat >> "${LIMITS_CONF}" <<EOF

${LIMITS_MARKER}
*         soft nofile ${NOFILE_LIMIT}
*         hard nofile ${NOFILE_LIMIT}
root      soft nofile ${NOFILE_LIMIT}
root      hard nofile ${NOFILE_LIMIT}
EOF
    echo "    written."
fi

# ── sysctl kernel cap ────────────────────────────────────────────────────────
echo "==> [2/3] /etc/sysctl.conf — setting fs.file-max=${KERNEL_FILEMAX}"
SYSCTL_CONF=/etc/sysctl.conf
SYSCTL_MARKER="# OpenHydra: raised fs.file-max"

if grep -qF "${SYSCTL_MARKER}" "${SYSCTL_CONF}" 2>/dev/null; then
    echo "    already configured, skipping."
else
    cat >> "${SYSCTL_CONF}" <<EOF

${SYSCTL_MARKER}
fs.file-max = ${KERNEL_FILEMAX}
EOF
    sysctl -p "${SYSCTL_CONF}"
    echo "    written and applied."
fi

# ── systemd manager default ──────────────────────────────────────────────────
echo "==> [3/3] /etc/systemd/system.conf — DefaultLimitNOFILE=${NOFILE_LIMIT}"
SYSTEMD_CONF=/etc/systemd/system.conf
SYSTEMD_MARKER="# OpenHydra: raised DefaultLimitNOFILE"

if grep -qF "${SYSTEMD_MARKER}" "${SYSTEMD_CONF}" 2>/dev/null; then
    echo "    already configured, skipping."
else
    # Comment out any existing (uncommented) DefaultLimitNOFILE line first
    sed -i 's/^DefaultLimitNOFILE=.*$/# & (replaced by OpenHydra)/' "${SYSTEMD_CONF}"
    cat >> "${SYSTEMD_CONF}" <<EOF

${SYSTEMD_MARKER}
DefaultLimitNOFILE=${NOFILE_LIMIT}
EOF
    echo "    written."
fi

# ── Reload systemd so new services inherit the limit ────────────────────────
systemctl daemon-reexec
echo ""
echo "Done. Verify with:"
echo "  ulimit -n                        (current shell — log out/in first)"
echo "  cat /proc/sys/fs/file-max        (kernel cap)"
echo "  systemctl show openhydra-bootstrap | grep LimitNOFILE"
echo ""
echo "NOTE: The bootstrap.service unit already has LimitNOFILE=65536 on line 24."
echo "      No service restart is needed for that process."
