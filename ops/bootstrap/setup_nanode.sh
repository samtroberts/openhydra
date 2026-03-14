#!/usr/bin/env bash
# setup_nanode.sh <domain>
#
# Idempotent provisioning script for an OpenHydra DHT bootstrap node.
# Run as root on a fresh Linode Nanode 1 GB (Ubuntu 24.04 LTS).
#
# Usage:
#   bash setup_nanode.sh bootstrap-us.openhydra.example

set -euo pipefail

DOMAIN="${1:?usage: setup_nanode.sh <domain>}"
REPO_URL="${OPENHYDRA_REPO_URL:-https://github.com/your-org/openhydra.git}"
VENV=/opt/openhydra/.venv
SECRETS_FILE=/etc/openhydra/secrets.env
SERVICE_FILE=/etc/systemd/system/openhydra-bootstrap.service
NGINX_CONF=/etc/nginx/sites-available/openhydra-bootstrap.conf
# Script directory — used to locate sibling files when the script is invoked
# from a staging path (e.g. /tmp/openhydra-bootstrap/) before the git clone.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


echo "==> Provisioning bootstrap node: $DOMAIN"

# --- System packages ---
# Ubuntu 24.04 ships Python 3.12 as default — no PPA needed.
apt-get update -qq
apt-get install -y --no-install-recommends \
    python3 python3-venv \
    nginx certbot python3-certbot-nginx \
    git ufw curl

# --- System user ---
id -u openhydra >/dev/null 2>&1 || useradd -r -s /bin/false -d /opt/openhydra openhydra
mkdir -p /opt/openhydra /etc/openhydra
chown openhydra:openhydra /opt/openhydra

# --- Repository ---
if [ -d /opt/openhydra/.git ]; then
    echo "==> Updating existing repo"
    git -C /opt/openhydra pull --ff-only
elif [ -d /opt/openhydra/dht ] || [ "${REPO_URL}" = "RSYNCED" ]; then
    echo "==> Project tree already present (rsynced), skipping git clone"
else
    echo "==> Cloning repo"
    git clone "$REPO_URL" /opt/openhydra
fi

# --- Python venv (bootstrap-only deps — no ML packages) ---
# python3 resolves to 3.12 on Ubuntu 24.04.
python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet grpcio protobuf cryptography

# --- Secrets file (create skeleton if absent) ---
if [ ! -f "$SECRETS_FILE" ]; then
    cat > "$SECRETS_FILE" <<'EOS'
# OpenHydra bootstrap node secrets
# All three bootstrap nodes MUST share the same GEO_CHALLENGE_SEED value.
# Generate with: openssl rand -hex 32
OPENHYDRA_GEO_CHALLENGE_SEED=REPLACE_WITH_SHARED_SECRET
EOS
    # Must be owned by the service user and mode 600 (no group/world bits)
    # so openhydra_secrets.py's permission check passes.
    chown openhydra:openhydra "$SECRETS_FILE"
    chmod 600 "$SECRETS_FILE"
    echo "!!! Edit $SECRETS_FILE and set OPENHYDRA_GEO_CHALLENGE_SEED before starting the service !!!"
fi

# --- Firewall ---
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

# --- TLS certificate ---
# Requires DNS A record for $DOMAIN to already point at this node's IP.
certbot --nginx -d "$DOMAIN" \
    --non-interactive --agree-tos \
    -m "admin@${DOMAIN#*.}" \
    --redirect || echo "WARNING: certbot failed — re-run after DNS propagates"

# --- nginx config ---
# Prefer the repo copy; fall back to the staged copy if the clone just ran
# the template from the staging directory (works both on first deploy and
# on re-runs after the repo is already in place).
NGINX_TEMPLATE=/opt/openhydra/ops/bootstrap/nginx-bootstrap.conf.template
if [ ! -f "$NGINX_TEMPLATE" ]; then
    NGINX_TEMPLATE="${SCRIPT_DIR}/nginx-bootstrap.conf.template"
fi
sed "s/%%DOMAIN%%/${DOMAIN}/g" "$NGINX_TEMPLATE" > "$NGINX_CONF"
ln -sf "$NGINX_CONF" /etc/nginx/sites-enabled/openhydra-bootstrap.conf
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl reload nginx

# --- systemd service ---
BOOTSTRAP_SERVICE=/opt/openhydra/ops/bootstrap/bootstrap.service
if [ ! -f "$BOOTSTRAP_SERVICE" ]; then
    BOOTSTRAP_SERVICE="${SCRIPT_DIR}/bootstrap.service"
fi
install -m 644 "$BOOTSTRAP_SERVICE" "$SERVICE_FILE"
systemctl daemon-reload
systemctl enable openhydra-bootstrap.service

echo ""
echo "=== Bootstrap setup complete for $DOMAIN ==="
echo ""
echo "Manual steps required before starting the service:"
echo "  1. Edit $SECRETS_FILE"
echo "     Set OPENHYDRA_GEO_CHALLENGE_SEED to the shared secret"
echo "     (must be identical on all bootstrap nodes)"
echo "  2. sudo systemctl start openhydra-bootstrap.service"
echo "  3. Verify: curl -s https://${DOMAIN}/health | python3 -m json.tool"
