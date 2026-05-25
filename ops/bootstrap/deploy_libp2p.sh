#!/usr/bin/env bash
# Deploy the openhydra-bootstrap binary to all three Linode bootstrap servers.
#
# Prerequisites:
#   1. Cross-compile the binary for Linux x86_64:
#      cd network && cross build --release --target x86_64-unknown-linux-gnu --bin openhydra-bootstrap
#      OR: cargo build --release --bin openhydra-bootstrap  (if building on Linux)
#
#   2. Ensure SSH access to root@<linode> for all three servers.
#
# Usage:
#   ./ops/bootstrap/deploy_libp2p.sh [path/to/binary]

set -euo pipefail

BINARY="${1:-network/target/x86_64-unknown-linux-gnu/release/openhydra-bootstrap}"
SERVICE_FILE="ops/bootstrap/libp2p-bootstrap.service"

# Production bootstrap nodes.
SERVERS=(
    "root@172.105.69.49"   # EU (London)
    "root@45.79.190.172"   # US (Dallas)
    "root@172.104.164.98"  # AP (Singapore)
)

# Bootstrap peer multiaddrs (each server connects to the other two).
EU_PEER="/ip4/172.105.69.49/tcp/4001/p2p/12D3KooWEzegXr4qcj37EWF2aQo9vp121MGrCaCwYcJF2oTkW3WT"
US_PEER="/ip4/45.79.190.172/tcp/4001/p2p/12D3KooWEL5wEL3foSWUk1E1rXHLbveqTahoHKhAsEYhDsLUkyWb"
AP_PEER="/ip4/172.104.164.98/tcp/4001/p2p/12D3KooWPgqZBgLZ1f94AQ7sbeyEz5UJ4jiT4d3zuQp2t61VLPZo"

# Map server IP → the OTHER two peers (for --peer flags).
declare -A PEER_FLAGS
PEER_FLAGS["172.105.69.49"]="--peer ${US_PEER} --peer ${AP_PEER}"   # EU connects to US + AP
PEER_FLAGS["45.79.190.172"]="--peer ${EU_PEER} --peer ${AP_PEER}"   # US connects to EU + AP
PEER_FLAGS["172.104.164.98"]="--peer ${EU_PEER} --peer ${US_PEER}"  # AP connects to EU + US

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
    ssh "$server" "mkdir -p /opt/openhydra/bin"

    # Upload binary.
    scp "$BINARY" "$server:/opt/openhydra/bin/openhydra-bootstrap"
    ssh "$server" "chmod +x /opt/openhydra/bin/openhydra-bootstrap"

    # Upload systemd service.
    scp "$SERVICE_FILE" "$server:/etc/systemd/system/openhydra-libp2p.service"

    # Generate identity key if it doesn't exist.
    ssh "$server" "
        if [ ! -f /opt/openhydra/.libp2p_identity.key ]; then
            /opt/openhydra/bin/openhydra-bootstrap \
                --identity /opt/openhydra/.libp2p_identity.key \
                --listen /ip4/127.0.0.1/tcp/0 &
            BGPID=\$!
            # Poll for key file instead of sleep (fixes race condition).
            for i in \$(seq 1 20); do
                [ -f /opt/openhydra/.libp2p_identity.key ] && break
                sleep 0.5
            done
            kill \$BGPID 2>/dev/null || true
            echo 'Generated new libp2p identity key'
        else
            echo 'Identity key already exists'
        fi
    "

    # Extract this server's IP to look up peer flags.
    SERVER_IP="\${server#root@}"

    # Generate systemd drop-in override with --peer flags for the other two
    # bootstraps + full dual-stack listen addresses (IPv4+IPv6, TCP+QUIC).
    PEERS="${PEER_FLAGS[$SERVER_IP]:-}"
    ssh "$server" "
        mkdir -p /etc/systemd/system/openhydra-libp2p.service.d
        cat > /etc/systemd/system/openhydra-libp2p.service.d/peers.conf <<DROPIN
[Service]
ExecStart=
ExecStart=/opt/openhydra/bin/openhydra-bootstrap \\\\
    --identity /opt/openhydra/.libp2p_identity.key \\\\
    --listen /ip4/0.0.0.0/tcp/4001 \\\\
    --listen /ip4/0.0.0.0/udp/4001/quic-v1 \\\\
    --listen /ip6/::/tcp/4001 \\\\
    --listen /ip6/::/udp/4001/quic-v1 \\\\
    ${PEERS}
DROPIN
        echo 'Drop-in override written with peer flags + dual-stack listen'
    "

    # Enable and (re)start the service.
    ssh "$server" "
        systemctl daemon-reload
        systemctl enable openhydra-libp2p
        systemctl restart openhydra-libp2p
        sleep 1
        systemctl status openhydra-libp2p --no-pager -l | head -20
    "

    echo ""
done

echo "Deployment complete. Verify with:"
echo "  ssh root@45.79.190.172 journalctl -u openhydra-libp2p -f"
echo ""
echo "Get the PeerId for --p2p-bootstrap flags:"
for server in "${SERVERS[@]}"; do
    echo "  ssh $server journalctl -u openhydra-libp2p | grep peer_id | head -1"
done
