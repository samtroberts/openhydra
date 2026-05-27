#!/usr/bin/env bash
# ops/network_limits.sh — OpenHydra bootstrap node connection-stability hardening
#
# Applies iptables + ip6tables connection limits and sysctl TCP tuning for
# OpenHydra bootstrap nodes running the Rust openhydra-bootstrap binary.
#
# Ports managed:
#   22    SSH
#   4001  libp2p (Kademlia DHT + Circuit Relay v2 + QUIC) — TCP + UDP
#
# Usage:
#   sudo bash ops/network_limits.sh          # apply rules
#   sudo bash ops/network_limits.sh --check  # show current rules & sysctl values
#   sudo bash ops/network_limits.sh --flush  # remove OpenHydra rules only
#
# Rules are idempotent: re-running the script is safe — it flushes and
# re-inserts the OpenHydra chain before applying.

set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fatal() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── Root check ────────────────────────────────────────────────────────────────
[[ $EUID -eq 0 ]] || fatal "Must be run as root (use sudo)"

# ── Argument handling ─────────────────────────────────────────────────────────
ACTION="${1:-apply}"
case "$ACTION" in
  --check|-c)
    info "Current iptables INPUT chain:"
    iptables -L INPUT -n -v --line-numbers
    echo
    info "OpenHydra IPv4 chain (if present):"
    iptables -L OPENHYDRA -n -v --line-numbers 2>/dev/null || warn "Chain OPENHYDRA not found"
    echo
    info "OpenHydra IPv6 chain (if present):"
    ip6tables -L OPENHYDRA -n -v --line-numbers 2>/dev/null || warn "IPv6 chain OPENHYDRA not found"
    echo
    info "IPv6 INPUT default policy:"
    ip6tables -L INPUT -n | head -1
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
    info "Removing OpenHydra iptables rules..."
    iptables -D INPUT -j OPENHYDRA 2>/dev/null && info "Removed IPv4 jump rule" || warn "IPv4 jump rule not present"
    iptables -F OPENHYDRA 2>/dev/null && info "Flushed IPv4 OPENHYDRA chain" || warn "IPv4 chain not present"
    iptables -X OPENHYDRA 2>/dev/null && info "Deleted IPv4 OPENHYDRA chain" || true
    ip6tables -D INPUT -j OPENHYDRA 2>/dev/null && info "Removed IPv6 jump rule" || warn "IPv6 jump rule not present"
    ip6tables -F OPENHYDRA 2>/dev/null && info "Flushed IPv6 OPENHYDRA chain" || warn "IPv6 chain not present"
    ip6tables -X OPENHYDRA 2>/dev/null && info "Deleted IPv6 OPENHYDRA chain" || true
    ip6tables -P INPUT ACCEPT && info "Reset IPv6 INPUT policy to ACCEPT"
    info "Done. Kernel sysctl values are NOT reverted (persistent via /etc/sysctl.d/)."
    exit 0
    ;;
  apply|"")
    : # fall through to main logic
    ;;
  *)
    fatal "Unknown argument: $ACTION  (use --check, --flush, or no argument to apply)"
    ;;
esac

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Kernel TCP hardening via sysctl
# ─────────────────────────────────────────────────────────────────────────────
info "Applying sysctl TCP hardening..."

SYSCTL_CONF=/etc/sysctl.d/60-openhydra.conf
cat > "$SYSCTL_CONF" <<'SYSCTL'
# OpenHydra TCP hardening — applied by ops/network_limits.sh
# Do not edit manually; re-run the script to update.

# SYN cookie protection against SYN flood attacks
net.ipv4.tcp_syncookies = 1

# Increase SYN backlog to handle connection bursts during peer storms
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
# SECTION 2 — iptables: IPv4 OPENHYDRA chain
# ─────────────────────────────────────────────────────────────────────────────
info "Setting up IPv4 iptables OPENHYDRA chain..."

# Remove old chain cleanly (idempotent)
iptables -D INPUT -j OPENHYDRA 2>/dev/null || true
iptables -F OPENHYDRA 2>/dev/null || true
iptables -X OPENHYDRA 2>/dev/null || true

# Create a dedicated chain so rules are easy to audit and flush
iptables -N OPENHYDRA

# ── 2a. Always allow established / related traffic ────────────────────────────
iptables -A OPENHYDRA -m state --state ESTABLISHED,RELATED -j ACCEPT

# ── 2b. SSH (port 22) — always allow ─────────────────────────────────────────
iptables -A OPENHYDRA -p tcp --dport 22 -j ACCEPT
info "SSH (22): ALLOW all"

# ── 2c. Port 4001 — libp2p (Kademlia DHT + Circuit Relay + QUIC) ─────────────
# TCP: Kademlia, relay circuits, direct TCP connections.
# UDP: QUIC transport, AutoNAT probing, DCUtR hole-punching.
iptables -A OPENHYDRA -p tcp --dport 4001 -j ACCEPT
iptables -A OPENHYDRA -p udp --dport 4001 -j ACCEPT
info "Port 4001 (libp2p): ALLOW TCP + UDP"

# ── 2d. ICMP rate limiting ────────────────────────────────────────────────────
iptables -A OPENHYDRA -p icmp \
  -m limit --limit 5/second --limit-burst 10 \
  -j ACCEPT
iptables -A OPENHYDRA -p icmp -j DROP
info "ICMP: rate-limited to 5/s (burst 10), excess dropped"

# ── 2e. Loopback — always allow ──────────────────────────────────────────────
iptables -A OPENHYDRA -i lo -j ACCEPT

# ── 2f. Jump INPUT → OPENHYDRA ───────────────────────────────────────────────
iptables -I INPUT 1 -j OPENHYDRA
info "IPv4 OPENHYDRA chain inserted at INPUT position 1"

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — ip6tables: IPv6 OPENHYDRA chain
# ─────────────────────────────────────────────────────────────────────────────
info "Setting up IPv6 ip6tables OPENHYDRA chain..."

# Remove old chain cleanly (idempotent)
ip6tables -D INPUT -j OPENHYDRA 2>/dev/null || true
ip6tables -F OPENHYDRA 2>/dev/null || true
ip6tables -X OPENHYDRA 2>/dev/null || true

ip6tables -N OPENHYDRA

# ── 3a. Always allow established / related traffic ────────────────────────────
ip6tables -A OPENHYDRA -m state --state ESTABLISHED,RELATED -j ACCEPT

# ── 3b. SSH (port 22) ────────────────────────────────────────────────────────
ip6tables -A OPENHYDRA -p tcp --dport 22 -j ACCEPT

# ── 3c. Port 4001 — libp2p (TCP + UDP) ───────────────────────────────────────
ip6tables -A OPENHYDRA -p tcp --dport 4001 -j ACCEPT
ip6tables -A OPENHYDRA -p udp --dport 4001 -j ACCEPT

# ── 3d. ICMPv6 — MUST allow fully (NDP, path MTU discovery) ──────────────────
# Unlike IPv4 ICMP, ICMPv6 carries Neighbor Discovery Protocol messages.
# Dropping NDP breaks IPv6 connectivity entirely.
ip6tables -A OPENHYDRA -p icmpv6 -j ACCEPT

# ── 3e. Loopback ─────────────────────────────────────────────────────────────
ip6tables -A OPENHYDRA -i lo -j ACCEPT

# ── 3f. Jump INPUT → OPENHYDRA + set default DROP ────────────────────────────
ip6tables -I INPUT 1 -j OPENHYDRA
ip6tables -P INPUT DROP
info "IPv6 OPENHYDRA chain applied, default INPUT policy DROP"

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Persist rules across reboots
# ─────────────────────────────────────────────────────────────────────────────
if command -v netfilter-persistent &>/dev/null; then
  netfilter-persistent save
  info "Rules persisted via netfilter-persistent"
elif command -v iptables-save &>/dev/null; then
  RULES_FILE=/etc/iptables/rules.v4
  RULES6_FILE=/etc/iptables/rules.v6
  mkdir -p /etc/iptables
  iptables-save > "$RULES_FILE"
  ip6tables-save > "$RULES6_FILE"
  info "IPv4 rules saved to $RULES_FILE"
  info "IPv6 rules saved to $RULES6_FILE"
  warn "Install 'iptables-persistent' to auto-restore on reboot: apt install iptables-persistent"
else
  warn "Cannot auto-persist rules. Run 'iptables-save > /etc/iptables/rules.v4' manually."
fi

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Summary
# ─────────────────────────────────────────────────────────────────────────────
echo
info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
info "OpenHydra network limits applied successfully."
info ""
info "  Layer 1: Linode Cloud Firewall (configure separately in Cloud Manager)"
info "  Layer 2: iptables + ip6tables OPENHYDRA chain — active now ✓"
info "  Layer 3: libp2p Noise (TCP) + TLS 1.3 (QUIC) encryption ✓"
info ""
info "  IPv4: SSH(22) + libp2p TCP+UDP(4001) + ICMP + loopback"
info "  IPv6: SSH(22) + libp2p TCP+UDP(4001) + ICMPv6 + loopback, default DROP"
info ""
info "To verify: sudo bash ops/network_limits.sh --check"
info "To remove: sudo bash ops/network_limits.sh --flush"
info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
