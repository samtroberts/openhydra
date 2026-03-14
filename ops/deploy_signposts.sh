#!/usr/bin/env bash
#
# deploy_signposts.sh — Push dual-stack (HTTP + Hivemind) DHT to 3 Linode Nanodes.
#
# Prerequisites:
#   export LINODE_PERSONAL_ACCESS_TOKEN="<your-token>"
#   export CLOUDFLARE_ZONE_TOKEN="<your-zone-token>"
#
# Usage:
#   bash ops/deploy_signposts.sh
#
# What this script does:
#   1. Fetches Nanode IPs via the Linode API (tagged "openhydra-bootstrap").
#   2. Verifies Cloudflare DNS A records for bootstrap.openhydra.ai.
#   3. SCPs the updated codebase to each Nanode.
#   4. SSH: installs hivemind, copies systemd units, restarts services.
#
# SECURITY: This script reads secrets ONLY from environment variables.
#           It NEVER writes them to disk or embeds them in files.
set -euo pipefail

# ── Validate environment ──────────────────────────────────────────────────

if [[ -z "${LINODE_PERSONAL_ACCESS_TOKEN:-}" ]]; then
    echo "ERROR: LINODE_PERSONAL_ACCESS_TOKEN not set."
    echo "Export it before running this script:"
    echo '  export LINODE_PERSONAL_ACCESS_TOKEN="your-token-here"'
    exit 1
fi

if [[ -z "${CLOUDFLARE_ZONE_TOKEN:-}" ]]; then
    echo "WARNING: CLOUDFLARE_ZONE_TOKEN not set — skipping DNS verification."
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SSH_USER="${SSH_USER:-root}"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"
DEPLOY_DIR="/opt/openhydra"
HIVEMIND_PORT=38751

echo "=== OpenHydra Signpost Deployment ==="
echo "Repo: $REPO_ROOT"
echo ""

# ── 1. Fetch Nanode IPs from Linode API ───────────────────────────────────

echo "--- Fetching Nanode IPs from Linode API ---"
NANODE_IPS=$(curl -s -H "Authorization: Bearer ${LINODE_PERSONAL_ACCESS_TOKEN}" \
    "https://api.linode.com/v4/linode/instances" \
    | python3 -c "
import sys, json
data = json.load(sys.stdin)
for inst in data.get('data', []):
    tags = [t.lower() for t in inst.get('tags', [])]
    if 'openhydra-bootstrap' in tags:
        ipv4s = inst.get('ipv4', [])
        if ipv4s:
            print(ipv4s[0])
" 2>/dev/null || true)

if [[ -z "$NANODE_IPS" ]]; then
    echo "WARNING: No Nanodes found via API (tag: openhydra-bootstrap)."
    echo "Falling back to known production IPs."
    NANODE_IPS="172.105.69.49
45.79.190.172
172.104.164.98"
fi

echo "Nanodes:"
echo "$NANODE_IPS" | while read -r ip; do echo "  $ip"; done
echo ""

# ── 2. Verify Cloudflare DNS (optional) ──────────────────────────────────

if [[ -n "${CLOUDFLARE_ZONE_TOKEN:-}" ]]; then
    echo "--- Verifying Cloudflare DNS records ---"
    # This is informational only — does NOT modify DNS.
    ZONE_ID="${CLOUDFLARE_ZONE_ID:-}"
    if [[ -n "$ZONE_ID" ]]; then
        curl -s -H "Authorization: Bearer ${CLOUDFLARE_ZONE_TOKEN}" \
            "https://api.cloudflare.com/client/v4/zones/${ZONE_ID}/dns_records?name=bootstrap.openhydra.ai&type=A" \
            | python3 -c "
import sys, json
data = json.load(sys.stdin)
for rec in data.get('result', []):
    print(f\"  DNS: {rec['name']} -> {rec['content']} (proxied={rec.get('proxied', False)})\")
" 2>/dev/null || echo "  (DNS check skipped — could not query Cloudflare)"
    else
        echo "  CLOUDFLARE_ZONE_ID not set — skipping DNS verification."
    fi
    echo ""
fi

# ── 3. Create signpost.service unit file ──────────────────────────────────

SIGNPOST_SERVICE=$(cat <<'SVCEOF'
[Unit]
Description=OpenHydra Hivemind DHT Signpost
After=network.target openhydra-bootstrap.service
Wants=openhydra-bootstrap.service

[Service]
Type=simple
User=openhydra
Group=openhydra
WorkingDirectory=/opt/openhydra
ExecStart=/opt/openhydra/.venv/bin/python3 -m dht.signpost --host 0.0.0.0 --port 38751 --identity-path /opt/openhydra/.hivemind_identity.key
Restart=always
RestartSec=5
LimitNOFILE=65536
Environment=PYTHONPATH=/opt/openhydra
NoNewPrivileges=true
ProtectSystem=strict
ReadWritePaths=/opt/openhydra
PrivateTmp=true

[Install]
WantedBy=multi-user.target
SVCEOF
)

# ── 4. Deploy to each Nanode ──────────────────────────────────────────────

for IP in $NANODE_IPS; do
    [[ -z "$IP" ]] && continue
    echo "=== Deploying to $IP ==="

    # 4a. Sync codebase.
    echo "  [1/7] Syncing codebase..."
    rsync -az --delete \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='.pytest_cache' \
        --exclude='*.pyc' \
        --exclude='.openhydra' \
        --exclude='.venv' \
        --exclude='.hivemind_identity.key' \
        -e "ssh $SSH_OPTS" \
        "$REPO_ROOT/" "${SSH_USER}@${IP}:${DEPLOY_DIR}/" \
        2>/dev/null || {
            echo "  WARNING: rsync failed for $IP — trying scp fallback"
            ssh $SSH_OPTS "${SSH_USER}@${IP}" "mkdir -p ${DEPLOY_DIR}"
            scp $SSH_OPTS -r "$REPO_ROOT"/{coordinator,dht,peer,openhydra_defaults.py,openhydra_logging.py,openhydra_secrets.py} \
                "${SSH_USER}@${IP}:${DEPLOY_DIR}/" 2>/dev/null || true
        }

    # 4b. Ensure swap + build deps (Nanodes have 1GB RAM; hivemind needs swap to compile).
    echo "  [2/7] Ensuring swap space and build dependencies..."
    ssh $SSH_OPTS "${SSH_USER}@${IP}" bash -s <<'SWAP_EOF'
        # Add 2GB swap if not already present.
        if [ ! -f /swapfile ]; then
            echo "    Creating 2GB swapfile..."
            fallocate -l 2G /swapfile 2>/dev/null || dd if=/dev/zero of=/swapfile bs=1M count=2048 status=none
            chmod 600 /swapfile
            mkswap /swapfile >/dev/null 2>&1
            swapon /swapfile
            echo '/swapfile none swap sw 0 0' >> /etc/fstab
            echo "    Swap enabled."
        elif ! swapon --show | grep -q /swapfile; then
            swapon /swapfile 2>/dev/null || true
            echo "    Swap re-enabled."
        else
            echo "    Swap already active."
        fi
        # Install build dependencies for hivemind's C extensions.
        if ! dpkg -s python3-dev >/dev/null 2>&1; then
            echo "    Installing build dependencies..."
            apt-get update -qq >/dev/null 2>&1
            apt-get install -y -qq python3-dev build-essential >/dev/null 2>&1
            echo "    Build deps installed."
        else
            echo "    Build deps already present."
        fi
SWAP_EOF

    # 4c. Create venv if missing + install hivemind.
    echo "  [3/7] Installing hivemind..."
    ssh $SSH_OPTS "${SSH_USER}@${IP}" bash -s <<'PIP_EOF'
        # Ensure venv exists.
        if [ ! -f /opt/openhydra/.venv/bin/python3 ]; then
            echo "    Creating virtualenv..."
            python3 -m venv /opt/openhydra/.venv
        fi
        # Install hivemind inside the venv.
        echo "    Installing hivemind (this may take a few minutes on first run)..."
        /opt/openhydra/.venv/bin/python3 -m pip install --no-cache-dir 'hivemind>=1.1.0' 2>&1 \
            || { echo "WARNING: hivemind install failed"; exit 1; }
        echo "    hivemind installed."
PIP_EOF

    # 4d. Install signpost.service.
    echo "  [4/7] Installing signpost.service..."
    echo "$SIGNPOST_SERVICE" | ssh $SSH_OPTS "${SSH_USER}@${IP}" \
        "cat > /etc/systemd/system/openhydra-signpost.service"

    # 4e. Ensure openhydra user exists + owns /opt/openhydra.
    ssh $SSH_OPTS "${SSH_USER}@${IP}" bash -s <<'USER_EOF'
        id -u openhydra >/dev/null 2>&1 || useradd -r -s /bin/false -d /opt/openhydra openhydra
        chown -R openhydra:openhydra /opt/openhydra
USER_EOF

    # 4f. Open port 38751 in firewall (iptables + ufw if present).
    echo "  [5/7] Configuring firewall for port 38751..."
    ssh $SSH_OPTS "${SSH_USER}@${IP}" bash -s <<'FW_EOF'
        # iptables: allow TCP 38751 with rate-limiting (20 new/min per source IP).
        if ! iptables -C INPUT -p tcp --dport 38751 -j ACCEPT 2>/dev/null; then
            iptables -A INPUT -p tcp --dport 38751 \
                -m hashlimit --hashlimit-name hivemind \
                --hashlimit-above 20/min --hashlimit-burst 5 \
                --hashlimit-mode srcip -j DROP 2>/dev/null || true
            iptables -A INPUT -p tcp --dport 38751 -j ACCEPT
            echo "    iptables: port 38751 opened with rate limiting."
        else
            echo "    iptables: port 38751 already open."
        fi
        # ufw (if installed and active).
        if command -v ufw >/dev/null 2>&1 && ufw status | grep -q "active"; then
            ufw allow 38751/tcp >/dev/null 2>&1
            echo "    ufw: port 38751 allowed."
        fi
FW_EOF

    # 4g. Reload and restart services.
    echo "  [6/7] Restarting services..."
    ssh $SSH_OPTS "${SSH_USER}@${IP}" bash -s <<'REMOTE_EOF'
        systemctl daemon-reload
        systemctl enable openhydra-signpost.service 2>/dev/null || true
        systemctl restart openhydra-bootstrap.service 2>/dev/null || echo "  bootstrap restart skipped (not found)"
        systemctl restart openhydra-signpost.service 2>/dev/null || echo "  signpost restart skipped"
        echo "  Services restarted."
        systemctl --no-pager status openhydra-signpost.service 2>/dev/null | head -5 || true
REMOTE_EOF

    # 4h. Verify signpost is running properly.
    echo "  [7/7] Verifying signpost..."
    sleep 3
    ssh $SSH_OPTS "${SSH_USER}@${IP}" \
        "journalctl -u openhydra-signpost -n 5 --no-pager 2>/dev/null | tail -2" \
        || true

    echo "  Done: $IP"
    echo ""
done

echo "=== Deployment complete ==="
echo ""
echo "Signpost nodes are running on port $HIVEMIND_PORT."
echo "Peers can connect with:"
echo "  --hivemind-initial-peers /ip4/<NANODE_IP>/tcp/$HIVEMIND_PORT/p2p/<PEER_ID>"
echo ""
echo "To get peer IDs, SSH to each Nanode and check the signpost logs:"
echo "  journalctl -u openhydra-signpost -n 20"
