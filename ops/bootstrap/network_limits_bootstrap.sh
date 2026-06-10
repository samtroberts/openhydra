#!/usr/bin/env bash
# ops/bootstrap/network_limits_bootstrap.sh
#
# iptables + sysctl hardening for OpenHydra libp2p bootstrap nodes.
# Run as root on Ubuntu 24.04. Replaces UFW with raw iptables managed by
# iptables-persistent so rules survive reboots without conflicts.
#
# Ports managed:
#   22    SSH
#   4001  libp2p (TCP + QUIC/UDP) — Kademlia DHT, Circuit Relay, AutoNAT
#
# Usage:
#   sudo bash ops/bootstrap/network_limits_bootstrap.sh          # apply
#   sudo bash ops/bootstrap/network_limits_bootstrap.sh --check  # show rules
#   sudo bash ops/bootstrap/network_limits_bootstrap.sh --flush  # remove rules

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fatal() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || fatal "Must be run as root (use sudo)"

ACTION="${1:-apply}"
case "$ACTION" in
  --check|-c)
    info "Current iptables INPUT chain:"
    iptables -L INPUT -n -v --line-numbers
    echo
    info "OPENHYDRA chain (if present):"
    iptables -L OPENHYDRA -n -v --line-numbers 2>/dev/null || warn "Chain OPENHYDRA not found"
    echo
    info "Relevant sysctl values:"
    for key in \
      net.ipv4.tcp_syncookies \
      net.ipv4.tcp_max_syn_backlog \
      net.ipv4.conf.all.rp_filter \
      net.ipv4.conf.all.accept_redirects \
      net.ipv4.conf.all.send_redirects \
      net.ipv4.tcp_fin_timeout \
      net.ipv4.tcp_keepalive_time \
      net.core.somaxconn; do
      printf "  %-45s = %s\n" "$key" "$(sysctl -n "$key" 2>/dev/null || echo 'n/a')"
    done
    exit 0
    ;;
  --flush|-f)
    info "Removing OPENHYDRA iptables rules..."
    iptables -D INPUT -j OPENHYDRA 2>/dev/null && info "Removed jump rule" || warn "Jump rule not present"
    iptables -F OPENHYDRA 2>/dev/null && info "Flushed OPENHYDRA chain" || warn "Chain not present"
    iptables -X OPENHYDRA 2>/dev/null && info "Deleted OPENHYDRA chain" || true
    info "Done. Sysctl values are NOT reverted (persistent via /etc/sysctl.d/)."
    exit 0
    ;;
  apply|"")
    : # fall through
    ;;
  *)
    fatal "Unknown argument: $ACTION  (use --check, --flush, or no argument)"
    ;;
esac

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — Disable UFW (conflicts with raw iptables on reboot)
# ─────────────────────────────────────────────────────────────────────────────
if command -v ufw &>/dev/null; then
  ufw disable 2>/dev/null || true
  info "UFW disabled (raw iptables takes over)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Kernel TCP hardening via sysctl
# ─────────────────────────────────────────────────────────────────────────────
info "Applying sysctl TCP hardening..."

SYSCTL_CONF=/etc/sysctl.d/60-openhydra.conf
cat > "$SYSCTL_CONF" <<'SYSCTL'
# OpenHydra bootstrap node TCP hardening
# Applied by ops/bootstrap/network_limits_bootstrap.sh

# SYN cookie protection against SYN flood attacks
net.ipv4.tcp_syncookies = 1

# Increase SYN backlog for connection bursts during peer storms
net.ipv4.tcp_max_syn_backlog = 4096

# Reverse path filtering — drop packets with spoofed source IPs
net.ipv4.conf.all.rp_filter = 1
net.ipv4.conf.default.rp_filter = 1

# Do not accept or send ICMP redirects (prevents routing attacks)
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.default.accept_redirects = 0

# Shorten FIN_WAIT_2 timeout to reclaim sockets faster
net.ipv4.tcp_fin_timeout = 20

# Keepalive: detect dead connections after 10 min (default 2 h)
net.ipv4.tcp_keepalive_time = 600
net.ipv4.tcp_keepalive_intvl = 30
net.ipv4.tcp_keepalive_probes = 5

# Increase listen() backlog for libp2p under load
net.core.somaxconn = 1024

# Allow TIME_WAIT socket reuse for fast port recycling
net.ipv4.tcp_tw_reuse = 1
SYSCTL

sysctl -p "$SYSCTL_CONF" > /dev/null
info "sysctl values applied and persisted to $SYSCTL_CONF"

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — iptables: OPENHYDRA chain (bootstrap profile)
# ─────────────────────────────────────────────────────────────────────────────
info "Setting up iptables OPENHYDRA chain (bootstrap profile)..."

# Remove old chain cleanly (idempotent)
iptables -D INPUT -j OPENHYDRA 2>/dev/null || true
iptables -F OPENHYDRA 2>/dev/null || true
iptables -X OPENHYDRA 2>/dev/null || true

iptables -N OPENHYDRA

# ── 2a. Always allow established / related traffic ───────────────────────────
iptables -A OPENHYDRA -m state --state ESTABLISHED,RELATED -j ACCEPT

# ── 2b. SSH (port 22) — always allow ────────────────────────────────────────
iptables -A OPENHYDRA -p tcp --dport 22 -j ACCEPT
info "SSH (22): ACCEPT"

# ── 2c. Port 4001 TCP — libp2p (Kademlia, Relay, Identify) ─────────────────
# Limit concurrent TCP connections per source IP to 20. Bootstrap nodes
# legitimately serve many peers, but no single IP needs more than 20
# connections (each peer typically maintains 1-2 connections).
iptables -A OPENHYDRA -p tcp --dport 4001 \
  -m connlimit --connlimit-above 20 --connlimit-mask 32 \
  -j REJECT --reject-with tcp-reset
iptables -A OPENHYDRA -p tcp --dport 4001 -j ACCEPT
info "Port 4001 TCP: connlimit ≤20 per IP; excess RST"

# ── 2d. Port 4001 UDP — libp2p QUIC (AutoNAT, DCUtR, relay) ────────────────
# Rate-limit NEW UDP flows to 100/min per source IP (burst 20). QUIC uses
# connection IDs so established flows are not affected by this rule.
iptables -A OPENHYDRA -p udp --dport 4001 \
  -m hashlimit \
  --hashlimit-name libp2p_quic \
  --hashlimit 100/minute \
  --hashlimit-mode srcip \
  --hashlimit-burst 20 \
  -j ACCEPT
iptables -A OPENHYDRA -p udp --dport 4001 -j DROP
info "Port 4001 UDP: hashlimit 100/min per IP (burst 20), excess DROP"

# ── 2e. ICMP rate limiting ──────────────────────────────────────────────────
iptables -A OPENHYDRA -p icmp \
  -m limit --limit 5/second --limit-burst 10 \
  -j ACCEPT
iptables -A OPENHYDRA -p icmp -j DROP
info "ICMP: rate-limited 5/s (burst 10), excess DROP"

# ── 2f. Loopback — always allow ─────────────────────────────────────────────
iptables -A OPENHYDRA -i lo -j ACCEPT

# ── 2g. Default DROP for everything else ────────────────────────────────────
iptables -A OPENHYDRA -j DROP
info "Default: DROP all other inbound"

# ── 2h. Jump INPUT → OPENHYDRA ──────────────────────────────────────────────
iptables -I INPUT 1 -j OPENHYDRA
info "OPENHYDRA chain inserted at INPUT position 1"

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Persist rules across reboots via iptables-persistent
# ─────────────────────────────────────────────────────────────────────────────
if ! dpkg -l iptables-persistent >/dev/null 2>&1; then
  info "Installing iptables-persistent..."
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq iptables-persistent
fi

if command -v netfilter-persistent &>/dev/null; then
  netfilter-persistent save
  info "Rules persisted via netfilter-persistent"
else
  RULES_FILE=/etc/iptables/rules.v4
  mkdir -p /etc/iptables
  iptables-save > "$RULES_FILE"
  info "Rules saved to $RULES_FILE"
fi

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Summary
# ─────────────────────────────────────────────────────────────────────────────
echo
info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
info "OpenHydra bootstrap firewall applied."
info ""
info "  Port 22   TCP  — SSH (ACCEPT)"
info "  Port 4001 TCP  — libp2p (connlimit ≤20/IP)"
info "  Port 4001 UDP  — QUIC (hashlimit 100/min/IP)"
info "  ICMP           — 5/s burst 10"
info "  Default        — DROP"
info ""
info "  UFW: disabled"
info "  Persistence: iptables-persistent"
info ""
info "To verify: sudo bash $0 --check"
info "To remove: sudo bash $0 --flush"
info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
