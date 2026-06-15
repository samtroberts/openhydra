#!/usr/bin/env bash
# Safe, staged redeploy of the `openhydra-bootstrap` binary to the 4-node
# bootstrap mesh (3 Linodes + 1 netcup).
#
# This replaces the old "loop over root@<linode>, purge disk, restart all at
# once" script, which no longer matches the fleet and was unsafe:
#   * root SSH login is DISABLED on the Linodes — access is `deploy@<ip>` with
#     ~/.ssh/openhydra_bootstrap + passwordless sudo.
#   * netcup (DE) is a 4th node (root@, service `openhydra-bootstrap`) that the
#     old script didn't know about.
#   * deploying to all nodes in one pass means a bad binary takes down the whole
#     DHT simultaneously, with no backup to roll back to.
#
# This version mirrors the verified 2026-06-15 manual procedure:
#   build → roll out node-by-node → back up the old binary → restart →
#   health-check (active + re-meshed) → auto-rollback + ABORT on any failure,
#   so at most one node is ever touched before a problem stops the rollout and
#   the remaining ≥3 nodes keep the DHT alive.
#
# Usage:
#   ./ops/bootstrap/deploy_libp2p.sh                 # build, then roll out
#   ./ops/bootstrap/deploy_libp2p.sh path/to/binary  # use a prebuilt binary
#   ./ops/bootstrap/deploy_libp2p.sh --reconfigure-peers
#                                                    # (re)write the 4-node mesh
#                                                    #  peers.conf drop-ins, then
#                                                    #  daemon-reload + restart
#   SSH_KEY=~/.ssh/other_key ./ops/bootstrap/deploy_libp2p.sh
#
# Out of scope (one-time provisioning, done by hand for the existing fleet):
# identity-key generation, firewall (ufw/iptables) setup, swap, disk prep.

set -euo pipefail

KEY="${SSH_KEY:-$HOME/.ssh/openhydra_bootstrap}"
SSH_OPTS=(-i "$KEY" -o ConnectTimeout=15 -o ServerAliveInterval=20 -o ServerAliveCountMax=6)
BIN_REMOTE="/opt/openhydra/bin/openhydra-bootstrap"
STAMP="$(date +%Y%m%d-%H%M%S)"

# ── Fleet ────────────────────────────────────────────────────────────────────
# label | ssh_target | service | sudo? (yes=deploy user, no=root) | owner
NODES=(
    "DE-netcup|root@85.209.48.209|openhydra-bootstrap|no|openhydra:openhydra"
    "US-linode|deploy@45.79.190.172|openhydra-libp2p|yes|root:root"
    "EU-linode|deploy@172.105.69.49|openhydra-libp2p|yes|root:root"
    "AP-linode|deploy@172.104.164.98|openhydra-libp2p|yes|root:root"
)

# Bootstrap peer multiaddrs — each node peers with all THREE others (full mesh).
US_PEER="/ip4/45.79.190.172/tcp/4001/p2p/12D3KooWEL5wEL3foSWUk1E1rXHLbveqTahoHKhAsEYhDsLUkyWb"
EU_PEER="/ip4/172.105.69.49/tcp/4001/p2p/12D3KooWEzegXr4qcj37EWF2aQo9vp121MGrCaCwYcJF2oTkW3WT"
AP_PEER="/ip4/172.104.164.98/tcp/4001/p2p/12D3KooWPgqZBgLZ1f94AQ7sbeyEz5UJ4jiT4d3zuQp2t61VLPZo"
DE_PEER="/ip4/85.209.48.209/tcp/4001/p2p/12D3KooWHNQ9nMedT3ZaMjj4xX7Mf4dxmTZ3cAX3tnJfyERV9Bua"

# The set of `--peer` flags a given node should carry (everyone but itself).
peers_for() {
    case "$1" in
        DE-netcup) echo "--peer ${US_PEER} --peer ${EU_PEER} --peer ${AP_PEER}" ;;
        US-linode) echo "--peer ${EU_PEER} --peer ${AP_PEER} --peer ${DE_PEER}" ;;
        EU-linode) echo "--peer ${US_PEER} --peer ${AP_PEER} --peer ${DE_PEER}" ;;
        AP-linode) echo "--peer ${US_PEER} --peer ${EU_PEER} --peer ${DE_PEER}" ;;
        *) echo "" ;;
    esac
}

# ── Build ────────────────────────────────────────────────────────────────────
BINARY="target/x86_64-unknown-linux-gnu/release/openhydra-bootstrap"
RECONFIGURE_PEERS=false

case "${1:-}" in
    --reconfigure-peers) RECONFIGURE_PEERS=true ;;
    "" ) ;;                       # default: build (below) then roll out
    * ) BINARY="$1" ;;            # caller supplied a prebuilt binary path
esac

build_binary() {
    echo "▶ Cross-compiling openhydra-bootstrap for linux x86_64 (zigbuild)…"
    cargo zigbuild --release --target x86_64-unknown-linux-gnu \
        -p openhydra-network --bin openhydra-bootstrap --no-default-features
}

if [[ "$RECONFIGURE_PEERS" == false ]]; then
    # Build only if we weren't handed an existing binary.
    if [[ "${1:-}" == "" && ! -f "$BINARY" ]]; then
        build_binary
    elif [[ ! -f "$BINARY" ]]; then
        echo "ERROR: binary not found at $BINARY" >&2
        exit 1
    fi
    # Sanity-check it's actually a Linux x86_64 ELF, not a host build.
    if ! file "$BINARY" | grep -q "ELF 64-bit.*x86-64"; then
        echo "ERROR: $BINARY is not a linux x86_64 ELF:" >&2
        file "$BINARY" >&2
        exit 1
    fi
    echo "Binary: $BINARY ($(du -h "$BINARY" | cut -f1))"
fi

# Run a command on a node, prefixing sudo when the login user isn't root.
remote() {  # remote <target> <sudo:yes|no> <command…>
    local target="$1" use_sudo="$2"; shift 2
    local prefix=""; [[ "$use_sudo" == "yes" ]] && prefix="sudo "
    ssh "${SSH_OPTS[@]}" "$target" "${prefix}$*"
}

# ── Per-node binary rollout (backup → install → restart → health-check) ──────
deploy_node() {
    local label="$1" target="$2" service="$3" use_sudo="$4" owner="$5"
    echo "── ${label} (${target}, ${service}) ──"

    echo "  uploading binary…"
    scp "${SSH_OPTS[@]}" "$BINARY" "${target}:/tmp/openhydra-bootstrap.new"

    echo "  backing up + installing…"
    remote "$target" "$use_sudo" "cp -f ${BIN_REMOTE} ${BIN_REMOTE}.bak-${STAMP}"
    remote "$target" "$use_sudo" "install -o ${owner%%:*} -g ${owner##*:} -m 0755 /tmp/openhydra-bootstrap.new ${BIN_REMOTE} && rm -f /tmp/openhydra-bootstrap.new"

    echo "  restarting ${service}…"
    remote "$target" "$use_sudo" "systemctl restart ${service}"
    sleep 6

    # ── Health check: service active AND re-meshed ──────────────────────────
    local active mesh
    active="$(remote "$target" "$use_sudo" "systemctl is-active ${service}" || true)"
    # `grep -c` exits 1 on zero matches; the trailing `|| true` keeps the pipeline
    # exit 0 so the captured value is just the count (0 when nothing re-meshed yet).
    mesh="$(remote "$target" "$use_sudo" "journalctl -u ${service} --no-pager -n 60 | grep -cE 'connection established|gossipsub: peer subscribed|external address confirmed' || true" || echo 0)"
    mesh="${mesh//[^0-9]/}"; mesh="${mesh:-0}"

    if [[ "$active" == "active" && "${mesh:-0}" -gt 0 ]]; then
        echo "  ✓ healthy (active, re-meshed: ${mesh} signal(s))"
        return 0
    fi

    # ── Failure → auto-rollback to the backup, then abort ───────────────────
    echo "  ✗ UNHEALTHY (active=${active}, mesh-signals=${mesh}). Rolling back…" >&2
    # Split: each command needs root, and the simple sudo-prefix only privileges
    # the first half of an `&&` chain.
    remote "$target" "$use_sudo" "cp -f ${BIN_REMOTE}.bak-${STAMP} ${BIN_REMOTE}" || true
    remote "$target" "$use_sudo" "systemctl restart ${service}" || true
    sleep 4
    active="$(remote "$target" "$use_sudo" "systemctl is-active ${service}" || true)"
    echo "  rollback done (service now: ${active}). ABORTING rollout — remaining nodes untouched." >&2
    return 1
}

# ── (Re)write the full-mesh peers.conf systemd drop-in for one node ──────────
reconfigure_peers_node() {
    local label="$1" target="$2" service="$3" use_sudo="$4"
    local peers; peers="$(peers_for "$label")"
    echo "── ${label}: writing 4-node mesh peers.conf drop-in ──"
    # Heredoc is expanded locally (peers/service interpolated) then piped to a
    # privileged `tee` on the remote. Backslash line-continuations are written
    # literally into the unit file.
    local dropin
    dropin="$(cat <<DROPIN
[Service]
ExecStart=
ExecStart=${BIN_REMOTE} \\
    --identity /opt/openhydra/.libp2p_identity.key \\
    --listen /ip4/0.0.0.0/tcp/4001 \\
    --listen /ip4/0.0.0.0/udp/4001/quic-v1 \\
    --listen /ip6/::/tcp/4001 \\
    --listen /ip6/::/udp/4001/quic-v1 \\
    ${peers}
DROPIN
)"
    local tee_cmd="tee /etc/systemd/system/${service}.service.d/peers.conf >/dev/null"
    [[ "$use_sudo" == "yes" ]] && tee_cmd="sudo ${tee_cmd}"
    remote "$target" "$use_sudo" "mkdir -p /etc/systemd/system/${service}.service.d"
    printf '%s\n' "$dropin" | ssh "${SSH_OPTS[@]}" "$target" "$tee_cmd"
    # Split: both commands need root (the simple sudo-prefix only covers the first).
    remote "$target" "$use_sudo" "systemctl daemon-reload"
    remote "$target" "$use_sudo" "systemctl restart ${service}"
    sleep 4
    echo "  ✓ peers.conf applied (active: $(remote "$target" "$use_sudo" "systemctl is-active ${service}" || true))"
}

# ── Main ─────────────────────────────────────────────────────────────────────
for entry in "${NODES[@]}"; do
    IFS='|' read -r label target service use_sudo owner <<< "$entry"
    if [[ "$RECONFIGURE_PEERS" == true ]]; then
        reconfigure_peers_node "$label" "$target" "$service" "$use_sudo"
    else
        if ! deploy_node "$label" "$target" "$service" "$use_sudo" "$owner"; then
            exit 1   # abort-on-failure: keep the remaining nodes (and DHT) up
        fi
    fi
    echo ""
done

echo "Done. Verify a node with:"
echo "  ssh -i ${KEY} deploy@45.79.190.172 sudo journalctl -u openhydra-libp2p -f"
echo "Rollback binary on each node (if needed): ${BIN_REMOTE}.bak-${STAMP}"
