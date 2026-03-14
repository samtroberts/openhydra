#!/usr/bin/env bash
# ops/certs/provision_cert.sh
#
# Provision a Let's Encrypt TLS certificate for OpenHydra using certbot.
# Installs the cert + key to /etc/openhydra/certs/ and generates the paths
# needed by openhydra-coordinator --tls-* flags.
#
# Usage:
#   sudo ./ops/certs/provision_cert.sh --domain api.example.com --email you@example.com
#   sudo ./ops/certs/provision_cert.sh --domain api.example.com --email you@example.com --webroot /var/www/html
#   sudo ./ops/certs/provision_cert.sh --renew
#
# Flags:
#   -d, --domain DOMAIN    Domain name to issue the certificate for (required unless --renew)
#   -e, --email  EMAIL     Email for Let's Encrypt account registration (required unless --renew)
#   -w, --webroot PATH     Use webroot mode (serve ACME challenge from PATH)
#                          Default: standalone mode (temporarily binds port 80)
#   --staging              Use Let's Encrypt staging CA (for testing; cert will NOT be trusted)
#   --renew                Renew all existing certbot certificates and re-deploy
#   -h, --help             Print this help and exit
#
# After provisioning, the coordinator can be started with:
#   openhydra-coordinator \
#     --tls-enable \
#     --tls-root-cert-path   /etc/openhydra/certs/ca-bundle.pem \
#     --tls-client-cert-path /etc/openhydra/certs/fullchain.pem \
#     --tls-client-key-path  /etc/openhydra/certs/privkey.pem \
#     --tls-server-name-override <domain>

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DOMAIN=""
EMAIL=""
WEBROOT=""
STAGING=0
RENEW=0
CERT_DIR="/etc/openhydra/certs"
CERTBOT_DIR="/etc/letsencrypt/live"
LE_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt"  # system CA bundle

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
usage() {
    sed -n 's/^# //p' "$0" | head -40
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--domain)    DOMAIN="$2"; shift 2 ;;
        -e|--email)     EMAIL="$2";  shift 2 ;;
        -w|--webroot)   WEBROOT="$2"; shift 2 ;;
        --staging)      STAGING=1; shift ;;
        --renew)        RENEW=1; shift ;;
        -h|--help)      usage ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if [[ $EUID -ne 0 ]]; then
    echo "Error: this script must be run as root (sudo)." >&2
    exit 1
fi

if ! command -v certbot &>/dev/null; then
    echo "certbot not found — installing..."
    if command -v apt-get &>/dev/null; then
        apt-get update -qq && apt-get install -y certbot
    elif command -v yum &>/dev/null; then
        yum install -y certbot
    elif command -v brew &>/dev/null; then
        brew install certbot
    else
        echo "Cannot auto-install certbot. Please install it manually: https://certbot.eff.org" >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Renew mode
# ---------------------------------------------------------------------------
if [[ $RENEW -eq 1 ]]; then
    echo "==> Renewing all Let's Encrypt certificates..."
    certbot renew --quiet
    echo "==> Re-deploying certificates..."
    # Re-run deploy for each live domain
    for domain_dir in "$CERTBOT_DIR"/*/; do
        dom=$(basename "$domain_dir")
        echo "    Deploying $dom"
        mkdir -p "$CERT_DIR/$dom"
        cp -L "$domain_dir/fullchain.pem" "$CERT_DIR/$dom/fullchain.pem"
        cp -L "$domain_dir/privkey.pem"   "$CERT_DIR/$dom/privkey.pem"
        chmod 600 "$CERT_DIR/$dom/privkey.pem"
        # Symlink primary domain to the unversioned paths
        if [[ "$dom" == "${DOMAIN:-}" ]]; then
            _deploy_symlinks "$dom"
        fi
    done
    echo "==> Renewal complete."
    exit 0
fi

# ---------------------------------------------------------------------------
# Provision mode
# ---------------------------------------------------------------------------
if [[ -z "$DOMAIN" || -z "$EMAIL" ]]; then
    echo "Error: --domain and --email are required for provisioning." >&2
    echo "Run with --help for usage." >&2
    exit 1
fi

STAGING_FLAG=""
if [[ $STAGING -eq 1 ]]; then
    STAGING_FLAG="--staging"
    echo "==> WARNING: Using Let's Encrypt STAGING CA — certificate will NOT be browser-trusted."
fi

echo "==> Requesting certificate for $DOMAIN (email: $EMAIL)..."

if [[ -n "$WEBROOT" ]]; then
    certbot certonly \
        --non-interactive \
        --agree-tos \
        --email "$EMAIL" \
        --webroot \
        --webroot-path "$WEBROOT" \
        -d "$DOMAIN" \
        $STAGING_FLAG
else
    certbot certonly \
        --non-interactive \
        --agree-tos \
        --email "$EMAIL" \
        --standalone \
        -d "$DOMAIN" \
        $STAGING_FLAG
fi

# ---------------------------------------------------------------------------
# Deploy to /etc/openhydra/certs/
# ---------------------------------------------------------------------------
echo "==> Deploying certificate to $CERT_DIR ..."
mkdir -p "$CERT_DIR"

LIVE="$CERTBOT_DIR/$DOMAIN"
cp -L "$LIVE/fullchain.pem" "$CERT_DIR/fullchain.pem"
cp -L "$LIVE/privkey.pem"   "$CERT_DIR/privkey.pem"
chmod 600 "$CERT_DIR/privkey.pem"

# Copy system CA bundle (used as --tls-root-cert-path)
cp "$LE_CA_BUNDLE" "$CERT_DIR/ca-bundle.pem" 2>/dev/null || \
    cp /etc/ssl/cert.pem "$CERT_DIR/ca-bundle.pem" 2>/dev/null || \
    echo "Warning: could not copy system CA bundle — set --tls-root-cert-path manually."

# ---------------------------------------------------------------------------
# Install auto-renewal cron job (if not already present)
# ---------------------------------------------------------------------------
CRON_ENTRY="0 3 * * * certbot renew --quiet && cp -L $LIVE/fullchain.pem $CERT_DIR/fullchain.pem && cp -L $LIVE/privkey.pem $CERT_DIR/privkey.pem"
CRON_FILE="/etc/cron.d/openhydra-certbot"
if [[ ! -f "$CRON_FILE" ]]; then
    echo "$CRON_ENTRY" > "$CRON_FILE"
    chmod 644 "$CRON_FILE"
    echo "==> Auto-renewal cron job installed at $CRON_FILE (runs daily at 03:00)."
fi

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
echo ""
echo "==> Certificate provisioned successfully!"
echo ""
echo "    Domain:     $DOMAIN"
echo "    Cert dir:   $CERT_DIR"
echo "    fullchain:  $CERT_DIR/fullchain.pem"
echo "    privkey:    $CERT_DIR/privkey.pem"
echo "    CA bundle:  $CERT_DIR/ca-bundle.pem"
echo ""
echo "Start the coordinator with TLS:"
echo ""
echo "    openhydra-coordinator \\"
echo "      --tls-enable \\"
echo "      --tls-root-cert-path   $CERT_DIR/ca-bundle.pem \\"
echo "      --tls-client-cert-path $CERT_DIR/fullchain.pem \\"
echo "      --tls-client-key-path  $CERT_DIR/privkey.pem \\"
echo "      --tls-server-name-override $DOMAIN \\"
echo "      --dht-url http://bootstrap:8468 \\"
echo "      --host 0.0.0.0 --port 8080"
echo ""
echo "Or use the nginx HA overlay (recommended for production):"
echo "    docker-compose -f docker-compose.yml -f docker-compose.ha.yml up -d"
