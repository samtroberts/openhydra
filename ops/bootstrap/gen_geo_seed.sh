#!/usr/bin/env bash
# gen_geo_seed.sh [node1-ip] [node2-ip] [node3-ip]
#
# Generates a shared geo-challenge seed and prints distribution commands.
# Run this once on your local machine, then paste the output into your terminal.
#
# Usage:
#   bash gen_geo_seed.sh 192.0.2.1 198.51.100.1 203.0.113.1

set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "WARNING: no node IPs provided — seed will be generated but no distribution commands printed."
    echo "Usage: bash gen_geo_seed.sh <node1-ip> [node2-ip] [node3-ip]"
    echo ""
fi

GEO_SEED="$(openssl rand -hex 32)"

echo "Generated geo seed: $GEO_SEED"
echo ""
echo "Distribution commands (paste and run each):"
echo ""

for IP in "$@"; do
    echo "# Node: $IP"
    echo "ssh root@$IP \"sed -i 's/^OPENHYDRA_GEO_CHALLENGE_SEED=.*/OPENHYDRA_GEO_CHALLENGE_SEED=${GEO_SEED}/' /etc/openhydra/secrets.env && chmod 600 /etc/openhydra/secrets.env\""
    echo ""
done

echo "After distributing, start or restart the service on each node:"
for IP in "$@"; do
    echo "  ssh root@$IP 'systemctl restart openhydra-bootstrap.service'"
done
