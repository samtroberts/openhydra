#!/usr/bin/env bash
# ops/network_limits.sh — OpenHydra Ubuntu node connection-stability hardening
#
# Applies iptables connection limits and sysctl TCP tuning for OpenHydra
# bootstrap / peer nodes.  Run as root on Ubuntu 22.04+.
#
# Ports managed:
#   22    SSH
#   8080  Coordinator HTTP API (sits behind Cloudflare; accept all, CF filters L7)
#   8468  DHT bootstrap HTTP  (public; hashlimit 20 new conns/min per IP)
#   50051 Peer gRPC           (connlimit ≤5 concurrent per IP)
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
    info "OpenHydra custom chain (if present):"
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
    info "Removing OpenHydra iptables rules..."
    iptables -D INPUT -j OPENHYDRA 2>/dev/null && info "Removed jump rule" || warn "Jump rule not present"
    iptables -F OPENHYDRA 2>/dev/null && info "Flushed OPENHYDRA chain" || warn "Chain not present"
    iptables -X OPENHYDRA 2>/dev/null && info "Deleted OPENHYDRA chain" || true
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

# Increase listen() backlog for gRPC server under load
net.core.somaxconn = 1024

# Allow TIME_WAIT socket reuse for fast port recycling
net.ipv4.tcp_tw_reuse = 1
SYSCTL

sysctl -p "$SYSCTL_CONF" > /dev/null
info "sysctl values applied and persisted to $SYSCTL_CONF"

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — iptables: custom OPENHYDRA chain
# ─────────────────────────────────────────────────────────────────────────────
info "Setting up iptables OPENHYDRA chain..."

# Remove old chain cleanly (idempotent)
iptables -D INPUT -j OPENHYDRA 2>/dev/null || true
iptables -F OPENHYDRA 2>/dev/null || true
iptables -X OPENHYDRA 2>/dev/null || true

# Create a dedicated chain so rules are easy to audit and flush
iptables -N OPENHYDRA

# ── 2a. Always allow established / related traffic ────────────────────────────
iptables -A OPENHYDRA -m state --state ESTABLISHED,RELATED -j ACCEPT

# ── 2b. SSH (port 22) — always allow ─────────────────────────────────────────
# Tip: further restrict with: -s <your-management-ip-cidr>
iptables -A OPENHYDRA -p tcp --dport 22 -j ACCEPT
info "SSH (22): ALLOW all"

# ── 2c. Port 8080 — Coordinator API (Cloudflare-fronted) ─────────────────────
# Accept all; Cloudflare's edge handles L7 DDoS, bot detection, and rate
# limiting before traffic ever reaches the nanode.  Optionally restrict to
# Cloudflare IP ranges only (see https://www.cloudflare.com/ips/).
iptables -A OPENHYDRA -p tcp --dport 8080 -j ACCEPT
info "Port 8080 (coordinator API): ALLOW all (Cloudflare fronted)"

# ── 2d. Port 8468 — DHT bootstrap HTTP ───────────────────────────────────────
# Allow established DHT connections (already matched by 2a above).
# For NEW connections: hashlimit to 20/minute per source IP, burst of 5.
# Excess new connections are silently dropped.
iptables -A OPENHYDRA -p tcp --dport 8468 -m state --state NEW \
  -m hashlimit \
  --hashlimit-name dht_new_conn \
  --hashlimit 20/minute \
  --hashlimit-mode srcip \
  --hashlimit-burst 5 \
  -j ACCEPT
iptables -A OPENHYDRA -p tcp --dport 8468 -m state --state NEW -j DROP
info "Port 8468 (DHT): NEW conns hashlimit 20/min per IP (burst 5), excess dropped"

# ── 2e. Port 50051 — Peer gRPC ───────────────────────────────────────────────
# Limit concurrent connections per source IP to 5.
# Legitimate peers make very few long-lived connections; this blocks
# connection-exhaustion attacks while not affecting real peers.
iptables -A OPENHYDRA -p tcp --dport 50051 \
  -m connlimit --connlimit-above 5 --connlimit-mask 32 \
  -j REJECT --reject-with tcp-reset
iptables -A OPENHYDRA -p tcp --dport 50051 -j ACCEPT
info "Port 50051 (gRPC): connlimit ≤5 per IP; excess RST"

# ── 2f. ICMP rate limiting ────────────────────────────────────────────────────
iptables -A OPENHYDRA -p icmp \
  -m limit --limit 5/second --limit-burst 10 \
  -j ACCEPT
iptables -A OPENHYDRA -p icmp -j DROP
info "ICMP: rate-limited to 5/s (burst 10), excess dropped"

# ── 2g. Loopback — always allow ──────────────────────────────────────────────
iptables -A OPENHYDRA -i lo -j ACCEPT

# ── 2h. Jump INPUT → OPENHYDRA ───────────────────────────────────────────────
iptables -I INPUT 1 -j OPENHYDRA
info "OPENHYDRA chain inserted at INPUT position 1"

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Persist rules across reboots
# ─────────────────────────────────────────────────────────────────────────────
if command -v netfilter-persistent &>/dev/null; then
  netfilter-persistent save
  info "Rules persisted via netfilter-persistent"
elif command -v iptables-save &>/dev/null; then
  RULES_FILE=/etc/iptables/rules.v4
  mkdir -p /etc/iptables
  iptables-save > "$RULES_FILE"
  info "Rules saved to $RULES_FILE"
  warn "Install 'iptables-persistent' to auto-restore on reboot: apt install iptables-persistent"
else
  warn "Cannot auto-persist rules. Run 'iptables-save > /etc/iptables/rules.v4' manually."
fi

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Summary
# ─────────────────────────────────────────────────────────────────────────────
echo
info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
info "OpenHydra network limits applied successfully."
info ""
info "  Layer 1: Linode Cloud Firewall (configure separately in Cloud Manager)"
info "  Layer 2: Cloudflare (point 8080/8468 DNS through CF proxy)"
info "  Layer 3: iptables OPENHYDRA chain — active now ✓"
info "  Layer 4: Application rate limiter (coordinator/api_server.py) — built-in ✓"
info ""
info "To verify: sudo bash ops/network_limits.sh --check"
info "To remove: sudo bash ops/network_limits.sh --flush"
info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
