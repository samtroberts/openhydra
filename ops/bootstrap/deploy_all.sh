#!/usr/bin/env bash
# deploy_all.sh
#
# Deploys the OpenHydra bootstrap service to all three nodes in parallel.
#
# Environment variables (set before running):
#   BOOTSTRAP_US_IP   — IPv4 of the US-East Nanode
#   BOOTSTRAP_EU_IP   — IPv4 of the EU-West Nanode
#   BOOTSTRAP_AP_IP   — IPv4 of the AP-South Nanode
#   BOOTSTRAP_US_DOMAIN — hostname for US node (e.g. bootstrap-us.openhydra.example)
#   BOOTSTRAP_EU_DOMAIN — hostname for EU node
#   BOOTSTRAP_AP_DOMAIN — hostname for AP node
#   OPENHYDRA_REPO_URL  — git clone URL for the repo

set -euo pipefail

: "${BOOTSTRAP_US_IP:?set BOOTSTRAP_US_IP}"
: "${BOOTSTRAP_EU_IP:?set BOOTSTRAP_EU_IP}"
: "${BOOTSTRAP_AP_IP:?set BOOTSTRAP_AP_IP}"
: "${BOOTSTRAP_US_DOMAIN:?set BOOTSTRAP_US_DOMAIN}"
: "${BOOTSTRAP_EU_DOMAIN:?set BOOTSTRAP_EU_DOMAIN}"
: "${BOOTSTRAP_AP_DOMAIN:?set BOOTSTRAP_AP_DOMAIN}"
# OPENHYDRA_REPO_URL is optional. When omitted the full project tree is
# rsynced to /opt/openhydra on each node instead of git-cloned.
USE_RSYNC=false
if [ -z "${OPENHYDRA_REPO_URL:-}" ]; then
    USE_RSYNC=true
    echo "OPENHYDRA_REPO_URL not set — will rsync project tree to nodes."
fi

deploy_node() {
    local ip="$1"
    local domain="$2"
    echo "[${domain}] Starting deploy..."
    # setup_nanode.sh was staged to /tmp/openhydra-bootstrap on the remote so that
    # git clone /opt/openhydra finds an empty directory (clone fails into non-empty dirs).
    ssh -o StrictHostKeyChecking=accept-new root@"$ip" \
        "OPENHYDRA_REPO_URL='${OPENHYDRA_REPO_URL:-RSYNCED}' bash /tmp/openhydra-bootstrap/setup_nanode.sh '${domain}'"
    echo "[${domain}] Deploy complete."
}

# Stage ops scripts to a temp location on each remote node.
# We deliberately do NOT pre-populate /opt/openhydra because setup_nanode.sh
# runs `git clone $REPO_URL /opt/openhydra`, which fails if the directory is
# already non-empty.
for IP in "$BOOTSTRAP_US_IP" "$BOOTSTRAP_EU_IP" "$BOOTSTRAP_AP_IP"; do
    ssh -o StrictHostKeyChecking=accept-new root@"$IP" \
        "mkdir -p /tmp/openhydra-bootstrap" &
done
wait

for IP in "$BOOTSTRAP_US_IP" "$BOOTSTRAP_EU_IP" "$BOOTSTRAP_AP_IP"; do
    rsync -az ops/bootstrap/ root@"$IP":/tmp/openhydra-bootstrap/ &
done
wait

# If no git URL was provided, rsync the full project tree directly.
# setup_nanode.sh will skip the git clone step when code is already present.
if [ "$USE_RSYNC" = "true" ]; then
    echo "Rsyncing project tree to /opt/openhydra on all nodes..."
    REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || echo "$(dirname "$0")/../..")"
    for IP in "$BOOTSTRAP_US_IP" "$BOOTSTRAP_EU_IP" "$BOOTSTRAP_AP_IP"; do
        rsync -az --exclude='.git' --exclude='.venv' --exclude='__pycache__' \
            --exclude='*.pyc' --exclude='*.egg-info' --exclude='.pytest_cache' \
            --exclude='models/' \
            "$REPO_ROOT/" root@"$IP":/opt/openhydra/ &
    done
    wait
    echo "Project tree synced."
fi

# Run setup in parallel (each invocation clones the full repo, then the
# production ops files from the repo supersede the staged copies in /tmp).
deploy_node "$BOOTSTRAP_US_IP" "$BOOTSTRAP_US_DOMAIN" &
deploy_node "$BOOTSTRAP_EU_IP" "$BOOTSTRAP_EU_DOMAIN" &
deploy_node "$BOOTSTRAP_AP_IP" "$BOOTSTRAP_AP_DOMAIN" &
wait

echo ""
echo "All nodes deployed. Checking health..."
sleep 3
for DOMAIN in "$BOOTSTRAP_US_DOMAIN" "$BOOTSTRAP_EU_DOMAIN" "$BOOTSTRAP_AP_DOMAIN"; do
    echo -n "  $DOMAIN: "
    curl -sf "https://${DOMAIN}/health" | python3 -m json.tool --no-ensure-ascii 2>/dev/null \
        || echo "UNREACHABLE (DNS/TLS may still be propagating)"
done
