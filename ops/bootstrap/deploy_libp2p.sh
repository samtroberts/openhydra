#!/usr/bin/env bash
# Deploy the openhydra-bootstrap binary to all three Linode bootstrap servers.
#
# Prerequisites:
#   1. Cross-compile the binary for Linux x86_64:
#      cd network && cross build --release --target x86_64-unknown-linux-gnu --bin openhydra-bootstrap
#      OR: cargo build --release --bin openhydra-bootstrap  (if building on Linux)
#
#   2. Ensure SSH access to deploy@<linode> for all three servers.
#
# Usage:
#   ./ops/bootstrap/deploy_libp2p.sh [path/to/binary]

set -euo pipefail

BINARY="${1:-network/target/x86_64-unknown-linux-gnu/release/openhydra-bootstrap}"
SERVICE_FILE="ops/bootstrap/libp2p-bootstrap.service"

# Production bootstrap nodes.
SERVERS=(
    "deploy@172.105.69.49"   # EU (London)
    "deploy@45.79.190.172"   # US (Dallas)
    "deploy@172.104.164.98"  # AP (Singapore)
)

if [[ ! -f "$BINARY" ]]; then
    echo "ERROR: Binary not found at $BINARY"
    echo ""
    echo "Build it first:"
    echo "  # Option A: Cross-compile from macOS (requires 'cross' or 'cargo-zigbuild')"
    echo "  cargo install cross"
    echo "  cd network && cross build --release --target x86_64-unknown-linux-gnu --bin openhydra-bootstrap"
    echo ""
    echo "  # Option B: Build on a Linux machine"
    echo "  cd network && cargo build --release --bin openhydra-bootstrap"
    exit 1
fi

echo "Binary: $BINARY ($(du -h "$BINARY" | cut -f1))"
echo "Deploying to ${#SERVERS[@]} servers..."
echo ""

for server in "${SERVERS[@]}"; do
    echo "── $server ──"

    # Create dirs.
    ssh "$server" "sudo mkdir -p /opt/openhydra/bin"

    # Stop service before uploading (binary may be in use).
    ssh "$server" "sudo systemctl stop openhydra-libp2p 2>/dev/null || true"

    # Upload binary to /tmp first, then sudo move into place.
    scp "$BINARY" "$server:/tmp/openhydra-bootstrap"
    ssh "$server" "sudo mv /tmp/openhydra-bootstrap /opt/openhydra/bin/openhydra-bootstrap && sudo chmod +x /opt/openhydra/bin/openhydra-bootstrap"

    # Upload systemd service.
    scp "$SERVICE_FILE" "$server:/tmp/openhydra-libp2p.service"
    ssh "$server" "sudo mv /tmp/openhydra-libp2p.service /etc/systemd/system/openhydra-libp2p.service"

    # Generate identity key if it doesn't exist.
    ssh "$server" "
        if [ ! -f /opt/openhydra/.libp2p_identity.key ]; then
            sudo /opt/openhydra/bin/openhydra-bootstrap \
                --identity /opt/openhydra/.libp2p_identity.key \
                --listen /ip4/0.0.0.0/tcp/0 &
            BGPID=\$!
            sleep 2
            sudo kill \$BGPID 2>/dev/null || true
            echo 'Generated new libp2p identity key'
        else
            echo 'Identity key already exists'
        fi
    "

    # Enable and (re)start the service.
    ssh "$server" "
        sudo systemctl daemon-reload
        sudo systemctl enable openhydra-libp2p
        sudo systemctl restart openhydra-libp2p
        sleep 1
        sudo systemctl status openhydra-libp2p --no-pager -l | head -20
    "

    echo ""
done

echo "Deployment complete. Verify with:"
echo "  ssh deploy@45.79.190.172 sudo journalctl -u openhydra-libp2p -f"
echo ""
echo "Get the PeerId for --p2p-bootstrap flags:"
for server in "${SERVERS[@]}"; do
    echo "  ssh $server sudo journalctl -u openhydra-libp2p | grep peer_id | head -1"
done
