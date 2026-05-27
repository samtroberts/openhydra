#!/usr/bin/env bash
# Deploy the openhydra-bootstrap binary to all three Linode bootstrap servers.
#
# Prerequisites:
#   1. Cross-compile the binary for Linux x86_64:
#      cd network && cargo zigbuild --release --target x86_64-unknown-linux-gnu \
#          --bin openhydra-bootstrap --no-default-features
#      OR: cargo build --release --bin openhydra-bootstrap  (if building on Linux)
#
#   2. Ensure SSH access to root@<linode> for all three servers.
#
# Usage:
#   ./ops/bootstrap/deploy_libp2p.sh [path/to/binary]

set -euo pipefail

BINARY="${1:-network/target/x86_64-unknown-linux-gnu/release/openhydra-bootstrap}"
SERVICE_FILE="ops/bootstrap/libp2p-bootstrap.service"
FIREWALL_SCRIPT="ops/network_limits.sh"

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

if [[ ! -f "$BINARY" ]]; then
    echo "ERROR: Binary not found at $BINARY"
    echo ""
    echo "Build it first:"
    echo "  cd network && cargo zigbuild --release --target x86_64-unknown-linux-gnu \\"
    echo "      --bin openhydra-bootstrap --no-default-features"
    exit 1
fi

echo "Binary: $BINARY ($(du -h "$BINARY" | cut -f1))"
echo "Deploying to ${#SERVERS[@]} servers..."
echo ""

for server in "${SERVERS[@]}"; do
    echo "── $server ──"

    # Extract IP for peer flag lookup.
    SERVER_IP="${server#root@}"

    # ── Upload binary ───────────────────────────────────────────────
    ssh "$server" "systemctl stop openhydra-libp2p 2>/dev/null || true"
    ssh "$server" "mkdir -p /opt/openhydra/bin"
    scp "$BINARY" "$server:/opt/openhydra/bin/openhydra-bootstrap"
    ssh "$server" "chmod +x /opt/openhydra/bin/openhydra-bootstrap"

    # Upload systemd service.
    scp "$SERVICE_FILE" "$server:/etc/systemd/system/openhydra-libp2p.service"

    # ── Generate identity key if it doesn't exist ───────────────────
    ssh "$server" "
        if [ ! -f /opt/openhydra/.libp2p_identity.key ]; then
            /opt/openhydra/bin/openhydra-bootstrap \
                --identity /opt/openhydra/.libp2p_identity.key \
                --listen /ip4/127.0.0.1/tcp/0 &
            BGPID=\$!
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

    # ── Legacy service cleanup ──────────────────────────────────────
    echo "  Cleaning up legacy services..."
    ssh "$server" "
        for svc in openhydra-bootstrap openhydra-relay openhydra-signpost; do
            if systemctl is-active --quiet \${svc}.service 2>/dev/null; then
                systemctl stop \${svc}.service
                echo \"  Stopped \${svc}\"
            fi
            if systemctl is-enabled --quiet \${svc}.service 2>/dev/null; then
                systemctl disable \${svc}.service
                echo \"  Disabled \${svc}\"
            fi
        done
        # Stop nginx (only reverse-proxied the legacy Python DHT)
        if systemctl is-active --quiet nginx 2>/dev/null; then
            systemctl stop nginx && systemctl disable nginx
            echo '  Stopped and disabled nginx'
        fi
        # Kill any gRPC relay running as root
        # Use relay[.]relay_service so pkill -f doesn't match this shell's own argv
        pkill -f 'relay[.]relay_service' 2>/dev/null && echo '  Killed relay_service' || true
        # Kill hivemind p2pd daemon — spawned by signpost but may survive service stop
        pkill -9 p2pd 2>/dev/null && echo '  Killed p2pd' || true
    "

    # ── Disk cleanup ────────────────────────────────────────────────
    echo "  Freeing disk space..."
    ssh "$server" "
        # Preserve only the Rust binary and identity key.
        # Remove everything else: Python venv, source dirs, git repo, logs.
        find /opt/openhydra -mindepth 1 -maxdepth 1 \
            ! -name 'bin' \
            ! -name '.libp2p_identity.key' \
            -exec rm -rf {} + && echo '  Purged legacy files from /opt/openhydra'

        # Remove stray log files anywhere under /opt/openhydra
        find /opt/openhydra -name '*.log' -delete 2>/dev/null || true

        # Vacuum journal logs to 100 MB
        journalctl --vacuum-size=100M 2>&1 | tail -1

        # Full APT cleanup (remove orphaned packages + cache)
        apt-get autoremove -y -qq && apt-get clean -qq && echo '  APT cleaned'

        # Report
        echo '  Disk after cleanup:' && df -h / | tail -1
    "

    # ── Systemd drop-in override with peer flags ────────────────────
    # Determine which peers this server connects to (the other two).
    if [[ "$SERVER_IP" == "172.105.69.49" ]]; then
        PEERS="--peer ${US_PEER} --peer ${AP_PEER}"
    elif [[ "$SERVER_IP" == "45.79.190.172" ]]; then
        PEERS="--peer ${EU_PEER} --peer ${AP_PEER}"
    elif [[ "$SERVER_IP" == "172.104.164.98" ]]; then
        PEERS="--peer ${EU_PEER} --peer ${US_PEER}"
    else
        echo "WARNING: Unknown server IP $SERVER_IP, skipping peer flags"
        PEERS=""
    fi

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

    # ── Enable and (re)start the service ────────────────────────────
    ssh "$server" "
        systemctl daemon-reload
        systemctl enable openhydra-libp2p
        systemctl restart openhydra-libp2p
        sleep 1
        systemctl status openhydra-libp2p --no-pager -l | head -20
    "

    # ── Apply firewall rules ────────────────────────────────────────
    if [[ -f "$FIREWALL_SCRIPT" ]]; then
        echo "  Applying firewall rules..."
        scp "$FIREWALL_SCRIPT" "$server:/tmp/network_limits.sh"
        ssh "$server" "bash /tmp/network_limits.sh && rm /tmp/network_limits.sh"
    fi

    echo ""
done

echo "Deployment complete. Verify with:"
echo "  ssh root@45.79.190.172 journalctl -u openhydra-libp2p -f"
echo ""
echo "Get the PeerId for --p2p-bootstrap flags:"
for server in "${SERVERS[@]}"; do
    echo "  ssh $server journalctl -u openhydra-libp2p | grep peer_id | head -1"
done
