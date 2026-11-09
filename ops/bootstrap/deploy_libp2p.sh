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
#   ./ops/bootstrap/deploy_libp2p.sh --apply-unit   # install ops/bootstrap/libp2p-bootstrap.service
#                                                    #  as each node's unit (+ its peers.conf drop-in),
#                                                    #  node by node, health-checked, restored on failure
#   ./ops/bootstrap/deploy_libp2p.sh --reconfigure-peers
#                                                    # (re)write the 4-node mesh
#                                                    #  peers.conf drop-ins, then
#                                                    #  daemon-reload + restart
#   SSH_KEY=~/.ssh/other_key ./ops/bootstrap/deploy_libp2p.sh
#
# Out of scope (one-time provisioning, done by hand for the existing fleet):
# identity-key generation, firewall (ufw/iptables) setup, swap, disk prep.

set -euo pipefail
# Keep a Mac awake for the run: sleeping mid-rollout looks like a dead node to the health check.
if [[ "$(uname)" == Darwin && -z "${OH_DEPLOY_AWAKE:-}" ]] && command -v caffeinate >/dev/null; then
    exec env OH_DEPLOY_AWAKE=1 caffeinate -i "$0" "$@"
fi
# A prebuilt binary given as a relative path means relative to where we were called from: make it
# absolute before moving to the repository root (git paths and target/ are relative to that).
case "${1:-}" in ""|--*) ;; /*) ;; *) set -- "$PWD/$1" "${@:2}" ;; esac
cd "$(cd "$(dirname "$0")/../.." && pwd -P)"

KEY="${SSH_KEY:-$HOME/.ssh/openhydra_bootstrap}"
# BatchMode: a prompt (a changed host key, a passphrase) fails the step instead of stalling a rollout
# with a node half done.
# One connection per node for the whole run (ControlMaster): a health check makes ~40 calls, and a
# fresh connection each time made one dropped handshake roll a healthy node back (review 2026-10-29).
# (Under /tmp, not $TMPDIR: on macOS that path is too long for the socket, which ssh refuses.)
CTL_DIR="$(mktemp -d /tmp/ohdep.XXXX)"
SSH_OPTS=(-i "$KEY" -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=20 -o ServerAliveCountMax=6
    -o ControlMaster=auto -o "ControlPath=${CTL_DIR}/%C" -o ControlPersist=120)
# How long a restarted node must stay up, on the new binary, before the next node is touched.
SOAK_SECS="${SOAK_SECS:-45}"
BIN_REMOTE="/opt/openhydra/bin/openhydra-bootstrap"
STAMP="$(date -u +%Y%m%d-%H%M%S)"  # UTC, like deployed_at

# ── Fleet ────────────────────────────────────────────────────────────────────
# label | ssh_target | service | sudo? (yes=deploy user, no=root). Binaries are installed root:root
# (not writable by the service user, which could otherwise replace its own binary or the backups).
NODES=(
    "DE-netcup|root@85.209.48.209|openhydra-bootstrap|no"
    "US-linode|deploy@45.79.190.172|openhydra-libp2p|yes"
    "EU-linode|deploy@172.105.69.49|openhydra-libp2p|yes"
    "AP-linode|deploy@172.104.164.98|openhydra-libp2p|yes"
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
APPLY_UNIT=false

case "${1:-}" in
    --reconfigure-peers) RECONFIGURE_PEERS=true ;;
    --apply-unit) APPLY_UNIT=true; RECONFIGURE_PEERS=true ;;
    "" ) ;;                       # default: build (below) then roll out
    * ) BINARY="$1" ;;            # caller supplied a prebuilt binary path
esac

# Build the bootstrap and set BINARY to the file cargo itself says it produced, so a build
# directory elsewhere (CARGO_TARGET_DIR) can never leave an old binary at a guessed path, and a
# rerun on the same commit, where cargo correctly reuses the binary, works (review 2026-10-29).
build_binary() {
    echo "▶ Cross-compiling openhydra-bootstrap for linux x86_64 (zigbuild)…"
    command -v python3 >/dev/null || { echo "ERROR: python3 is needed to read cargo's build report" >&2; exit 1; }
    local report
    report="$(mktemp)"
    cargo zigbuild --release --target x86_64-unknown-linux-gnu \
        -p openhydra-network --bin openhydra-bootstrap --no-default-features \
        --message-format=json-render-diagnostics > "$report"
    # The `compiler-artifact` message for the openhydra-bootstrap binary: its path, and whether it
    # was reused (fresh) or built now.
    local found
    found="$(python3 -I -c '
import json, sys
for line in open(sys.argv[1]):
    try:
        m = json.loads(line)
    except ValueError:
        continue
    if m.get("reason") == "compiler-artifact" and m.get("target", {}).get("name") == "openhydra-bootstrap" \
            and "bin" in m.get("target", {}).get("kind", []) and m.get("executable"):
        print(m["executable"], "reused" if m.get("fresh") else "built")
' "$report")"
    rm -f "$report"
    [[ -n "$found" ]] || { echo "ERROR: cargo reported no openhydra-bootstrap binary" >&2; exit 1; }
    BINARY="${found% *}"
    echo "  cargo ${found##* }: $BINARY"
}

if [[ "$RECONFIGURE_PEERS" == false ]]; then
    # Build unless we were handed a binary: an old build left in target/ would otherwise be
    # redeployed as is (2026-10-27: a rollout restarted all four nodes on the 19 Oct binary).
    if [[ "${1:-}" == "" ]]; then
        # The binary links the whole network crate: build only from committed code, so what runs
        # on the nodes matches a commit (ALLOW_DIRTY=1 to override, e.g. for a test fleet).
        dirty="$(git status --porcelain -- network protocol Cargo.toml Cargo.lock .cargo)" \
            || { echo "ERROR: git status failed; run from the openhydra checkout" >&2; exit 1; }
        if [[ -n "$dirty" && "${ALLOW_DIRTY:-}" != 1 ]]; then
            echo "ERROR: uncommitted changes the bootstrap is built from; commit them first:" >&2
            echo "$dirty" >&2
            exit 1
        fi
        # The clean-tree check above means the sources are the committed ones, so a binary cargo
        # reused is the right one too.
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
    # A binary handed in by path must at least be a bootstrap (not, say, the agent).
    if ! grep -q "openhydra_bootstrap" "$BINARY"; then
        echo "ERROR: $BINARY doesn't look like openhydra-bootstrap" >&2
        exit 1
    fi
    BIN_SHA="$(shasum -a 256 "$BINARY" | cut -d' ' -f1)"
    if [[ "${1:-}" == "" ]]; then
        SOURCE_DESC="$(git rev-parse --short HEAD)"
        # Hash every file the binary is built from, tracked or not (untracked crates and modules are
        # compiled in too; review 2026-10-29), so a dirty build can be matched to its tree later.
        [[ -n "${dirty:-}" ]] && SOURCE_DESC="${SOURCE_DESC} + uncommitted changes (tree sha256 $(git ls-files -co --exclude-standard -z -- network protocol Cargo.toml Cargo.lock .cargo | LC_ALL=C sort -z | xargs -0 shasum -a 256 | shasum -a 256 | cut -c1-16))"
    else
        SOURCE_DESC="prebuilt $(basename "$BINARY")"
    fi
    echo "Binary: $BINARY ($(du -h "$BINARY" | cut -f1)) sha256 ${BIN_SHA:0:16}"
    echo "Source: ${SOURCE_DESC}"
fi

# Run a command on a node, prefixing sudo when the login user isn't root.
remote() {  # remote <target> <sudo:yes|no> <command…>
    local target="$1" use_sudo="$2"; shift 2
    local prefix=""; [[ "$use_sudo" == "yes" ]] && prefix="sudo "
    # ssh's own failures (exit 255: no connection) are retried twice; the command's never are.
    local try rc
    for try in 1 2 3; do
        ssh "${SSH_OPTS[@]}" "$target" "${prefix}$*" && return 0
        rc=$?
        [[ $rc -eq 255 && $try -lt 3 ]] || return $rc
        sleep 3
    done
}

# The peer ids a node meshes with (the other three bootstraps).
other_peer_ids() {
    local p
    for p in $(peers_for "$1"); do
        [[ "$p" == /* ]] && echo "${p##*/p2p/}"
    done
}

# One service property (`systemctl show -p NAME`), its value only.
unit_prop() {  # unit_prop <target> <sudo> <service> <property>
    local out
    out="$(remote "$1" "$2" "systemctl show -p $4 $3")" || return 1
    out="${out#*=}"
    echo "${out//[$'\r\n']/}"
}

# Health of the node's service, judged on the process that runs now, not on journal lines an older
# one left (review 2026-10-29: the old check counted the last 60 journal lines, which the previous
# process's "connection established" lines always satisfied, so a binary that crash-looped passed
# on every node). Healthy = for SOAK_SECS the same main process keeps running (same PID and run,
# NRestarts unchanged, active), its executable is the binary we expect, and this run's own journal
# shows it connected to at least one of the other bootstraps.
health_check() {  # health_check <label> <target> <sudo> <service> <expected sha256>
    local label="$1" target="$2" use_sudo="$3" service="$4" want="$5"
    local pid inv restarts n state now_pid now_inv now_restarts exe ids lines id waited=0
    pid="$(unit_prop "$target" "$use_sudo" "$service" MainPID)" || { echo "  ✗ can't read the service state" >&2; return 1; }
    inv="$(unit_prop "$target" "$use_sudo" "$service" InvocationID)" || return 1
    restarts="$(unit_prop "$target" "$use_sudo" "$service" NRestarts)" || return 1
    [[ "$pid" =~ ^[1-9][0-9]*$ && -n "$inv" ]] || { echo "  ✗ no running main process (MainPID=${pid:-?})" >&2; return 1; }
    exe="$(remote "$target" "$use_sudo" "sha256sum /proc/${pid}/exe")" || { echo "  ✗ can't hash the running executable" >&2; return 1; }
    exe="${exe%% *}"
    [[ "$exe" == "$want" ]] || { echo "  ✗ the running executable is ${exe:0:16}, expected ${want:0:16}" >&2; return 1; }
    while (( waited < SOAK_SECS )); do
        sleep 5; waited=$((waited + 5))
        state="$(remote "$target" "$use_sudo" "systemctl is-active $service" || true)"
        now_pid="$(unit_prop "$target" "$use_sudo" "$service" MainPID || true)"
        now_inv="$(unit_prop "$target" "$use_sudo" "$service" InvocationID || true)"
        now_restarts="$(unit_prop "$target" "$use_sudo" "$service" NRestarts || true)"
        if [[ "$state" != active || "$now_pid" != "$pid" || "$now_inv" != "$inv" || "$now_restarts" != "$restarts" ]]; then
            echo "  ✗ the service didn't stay up (after ${waited}s: ${state:-?}, pid ${pid}→${now_pid:-?}, restarts ${restarts}→${now_restarts:-?})" >&2
            return 1
        fi
    done
    ids="$(other_peer_ids "$label")"
    lines="$(remote "$target" "$use_sudo" "journalctl -u $service _SYSTEMD_INVOCATION_ID=$inv --no-pager -o cat" || true)"
    n=0
    for id in $ids; do
        grep -q "connection established.*${id}" <<< "$lines" && n=$((n + 1))
    done
    if (( n == 0 )); then
        echo "  ✗ up for ${SOAK_SECS}s but not connected to any other bootstrap" >&2
        return 1
    fi
    echo "  ✓ healthy: pid ${pid} up ${SOAK_SECS}s on ${want:0:16}, meshed with ${n} of the other bootstraps"
}

# Put the node back on its backup and check it the same way. Says loudly if that fails too.
rollback_node() {  # rollback_node <label> <target> <service> <sudo> <old sha256>
    local label="$1" target="$2" service="$3" use_sudo="$4" old="$5"
    echo "  rolling ${label} back to ${old:0:16}…" >&2
    if remote "$target" "$use_sudo" "sh -c 'cp -f ${BIN_REMOTE}.bak-${STAMP} ${BIN_REMOTE}.new && mv -f ${BIN_REMOTE}.new ${BIN_REMOTE}'" \
        && remote "$target" "$use_sudo" "systemctl restart ${service}" \
        && health_check "$label" "$target" "$use_sudo" "$service" "$old"; then
        echo "  rollback done: ${label} is back on its previous binary. ABORTING rollout — remaining nodes untouched." >&2
    else
        echo "" >&2
        echo "  !!! ROLLBACK OF ${label} FAILED: the node needs attention now (${target}, ${service};" >&2
        echo "  !!! backup ${BIN_REMOTE}.bak-${STAMP}, sha ${old:0:16}). Remaining nodes untouched." >&2
    fi
}

# The node being changed, for the interrupt handler: "mode|label|target|service|sudo|sha" (mode:
# binary, peers or unit; sha: the binary to go back to, or the one the node runs).
CHANGING=""
on_interrupt() {
    trap - INT TERM HUP
    if [[ -n "$CHANGING" ]]; then
        local mode label target service use_sudo sha
        IFS='|' read -r mode label target service use_sudo sha <<< "$CHANGING"
        echo "" >&2; echo "interrupted while changing ${label}" >&2
        case "$mode" in
            binary) rollback_node "$label" "$target" "$service" "$use_sudo" "$sha" ;;
            peers) restore_peers_node "$label" "$target" "$service" "$use_sudo" "$sha" ;;
            unit) restore_unit_node "$label" "$target" "$service" "$use_sudo" "$sha" ;;
        esac
    fi
    exit 130
}

# ── Per-node binary rollout (upload → verify → backup → install → restart → soak) ──
# Every step is checked explicitly: this runs as the condition of an `if`, where bash ignores
# `set -e` (review 2026-10-29: a failed upload left the old binary in place and reported "healthy";
# a failed backup made the rollback a no-op).
deploy_node() {
    local label="$1" target="$2" service="$3" use_sudo="$4"
    local tmp got old
    echo "── ${label} (${target}, ${service}) ──"

    echo "  uploading binary…"
    # A fresh temporary file (not a fixed /tmp path someone else could have put a file at).
    tmp="$(remote "$target" no "mktemp /tmp/openhydra-bootstrap.XXXXXXXX")" || { echo "  ✗ couldn't reach ${label}; nothing changed. ABORTING." >&2; return 1; }
    tmp="${tmp//[$'\r\n']/}"
    [[ "$tmp" == /tmp/openhydra-bootstrap.* ]] || { echo "  ✗ unexpected temporary path '${tmp}'. ABORTING." >&2; return 1; }
    if ! scp "${SSH_OPTS[@]}" -q "$BINARY" "${target}:${tmp}"; then
        remote "$target" no "rm -f ${tmp}" || true
        echo "  ✗ upload failed; nothing changed. ABORTING." >&2; return 1
    fi
    got="$(remote "$target" no "sha256sum ${tmp}")" || got=""
    if [[ "${got%% *}" != "$BIN_SHA" ]]; then
        remote "$target" no "rm -f ${tmp}" || true
        echo "  ✗ the uploaded file is ${got:0:16}, not ${BIN_SHA:0:16}; nothing changed. ABORTING." >&2; return 1
    fi

    echo "  backing up the current binary…"
    old="$(remote "$target" "$use_sudo" "sha256sum ${BIN_REMOTE}")" || old=""
    old="${old%% *}"
    if [[ ! "$old" =~ ^[0-9a-f]{64}$ ]] \
        || ! remote "$target" "$use_sudo" "cp -f ${BIN_REMOTE} ${BIN_REMOTE}.bak-${STAMP}" \
        || [[ "$(remote "$target" "$use_sudo" "sha256sum ${BIN_REMOTE}.bak-${STAMP}" | cut -d' ' -f1)" != "$old" ]]; then
        remote "$target" no "rm -f ${tmp}" || true
        echo "  ✗ couldn't make a verified backup; nothing changed. ABORTING." >&2; return 1
    fi
    if [[ "$old" == "$BIN_SHA" ]]; then
        echo "  (already on ${BIN_SHA:0:16}: restarting it on the same binary)"
    fi

    echo "  installing + restarting ${service}…"
    CHANGING="binary|${label}|${target}|${service}|${use_sudo}|${old}"
    if ! remote "$target" "$use_sudo" "install -o root -g root -m 0755 ${tmp} ${BIN_REMOTE}" \
        || [[ "$(remote "$target" "$use_sudo" "sha256sum ${BIN_REMOTE}" | cut -d' ' -f1)" != "$BIN_SHA" ]] \
        || ! remote "$target" "$use_sudo" "systemctl restart ${service}"; then
        echo "  ✗ install or restart failed" >&2
        remote "$target" "$use_sudo" "rm -f ${tmp}" || true
        rollback_node "$label" "$target" "$service" "$use_sudo" "$old"; CHANGING=""; return 1
    fi
    remote "$target" "$use_sudo" "rm -f ${tmp}" || true

    if ! health_check "$label" "$target" "$use_sudo" "$service" "$BIN_SHA"; then
        rollback_node "$label" "$target" "$service" "$use_sudo" "$old"; CHANGING=""; return 1
    fi
    CHANGING=""
    # What runs here and where it came from, beside the binary.
    printf 'sha256=%s\nsource=%s\ndeployed_at=%s\nby=%s\nprevious_sha256=%s\nbackup=%s\n' \
        "$BIN_SHA" "$SOURCE_DESC" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(whoami)@$(hostname -s)" "$old" "${BIN_REMOTE}.bak-${STAMP}" \
        | ssh "${SSH_OPTS[@]}" "$target" "$([[ "$use_sudo" == yes ]] && echo "sudo ")tee ${BIN_REMOTE}.deployed >/dev/null" \
        || echo "  (couldn't write ${BIN_REMOTE}.deployed)" >&2
    return 0
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
    local conf="/etc/systemd/system/${service}.service.d/peers.conf" cur
    local tee_cmd="tee ${conf} >/dev/null"
    [[ "$use_sudo" == "yes" ]] && tee_cmd="sudo ${tee_cmd}"
    # The binary stays as it is: the health check after the restart expects it to be the one running.
    cur="$(remote "$target" "$use_sudo" "sha256sum ${BIN_REMOTE}" | cut -d' ' -f1)"
    [[ "$cur" =~ ^[0-9a-f]{64}$ ]] || { echo "  ✗ can't read the installed binary; nothing changed. ABORTING." >&2; return 1; }
    remote "$target" "$use_sudo" "mkdir -p /etc/systemd/system/${service}.service.d" || { echo "  ✗ unreachable; nothing changed. ABORTING." >&2; return 1; }
    # Keep the old drop-in (or note there was none) so a bad one can be undone.
    # (In `sh -c`: sudo runs a command, not shell syntax.)
    remote "$target" "$use_sudo" "sh -c 'if [ -f ${conf} ]; then cp -f ${conf} ${conf}.bak-${STAMP}; else : > ${conf}.none-${STAMP}; fi'" \
        || { echo "  ✗ couldn't back up ${conf}; nothing changed. ABORTING." >&2; return 1; }
    CHANGING="peers|${label}|${target}|${service}|${use_sudo}|${cur}"
    if printf '%s\n' "$dropin" | ssh "${SSH_OPTS[@]}" "$target" "$tee_cmd" \
        && remote "$target" "$use_sudo" "systemctl daemon-reload" \
        && remote "$target" "$use_sudo" "systemctl restart ${service}" \
        && health_check "$label" "$target" "$use_sudo" "$service" "$cur"; then
        CHANGING=""
        echo "  ✓ peers.conf applied"
        return 0
    fi
    echo "  ✗ the new peers.conf didn't come up healthy; restoring the old one…" >&2
    restore_peers_node "$label" "$target" "$service" "$use_sudo" "$cur"; CHANGING=""
    return 1
}

# Put a node's previous peers.conf back (the one backed up at STAMP, or none) and check it.
restore_peers_node() {  # restore_peers_node <label> <target> <service> <sudo> <binary sha>
    local label="$1" target="$2" service="$3" use_sudo="$4" cur="$5"
    local conf="/etc/systemd/system/${service}.service.d/peers.conf"
    remote "$target" "$use_sudo" "sh -c 'if [ -f ${conf}.bak-${STAMP} ]; then cp -f ${conf}.bak-${STAMP} ${conf}.new && mv -f ${conf}.new ${conf}; else rm -f ${conf}; fi'" || true
    if remote "$target" "$use_sudo" "systemctl daemon-reload" \
        && remote "$target" "$use_sudo" "systemctl restart ${service}" \
        && health_check "$label" "$target" "$use_sudo" "$service" "$cur"; then
        echo "  restored: ${label} runs its previous configuration. ABORTING — remaining nodes untouched." >&2
    else
        echo "  !!! RESTORING ${label}'s configuration FAILED: the node needs attention now (${target}, ${conf})." >&2
    fi
}

# ── Install the repository's unit file as one node's unit (+ its peers.conf) ──
UNIT_SRC="ops/bootstrap/libp2p-bootstrap.service"
apply_unit_node() {
    local label="$1" target="$2" service="$3" use_sudo="$4"
    local unit="/etc/systemd/system/${service}.service" dir="/etc/systemd/system/${service}.service.d"
    local ctl="/etc/systemd/system.control/${service}.service.d" cur peers dropin s
    s=""; [[ "$use_sudo" == "yes" ]] && s="sudo "
    echo "── ${label}: installing ${UNIT_SRC} as ${service}.service ──"
    cur="$(remote "$target" "$use_sudo" "sha256sum ${BIN_REMOTE}" | cut -d' ' -f1)"
    [[ "$cur" =~ ^[0-9a-f]{64}$ ]] || { echo "  ✗ can't read the installed binary; nothing changed. ABORTING." >&2; return 1; }
    peers="$(peers_for "$label")"
    dropin="$(printf '[Service]\nExecStart=\nExecStart=%s \\\n    --identity /opt/openhydra/.libp2p_identity.key \\\n    --listen /ip4/0.0.0.0/tcp/4001 \\\n    --listen /ip4/0.0.0.0/udp/4001/quic-v1 \\\n    --listen /ip6/::/tcp/4001 \\\n    --listen /ip6/::/udp/4001/quic-v1 \\\n    %s\n' "$BIN_REMOTE" "$peers")"
    # Back up the unit, its drop-ins and the set-property ones the unit now replaces.
    remote "$target" "$use_sudo" "sh -c 'mkdir -p /root/oh-unit-${STAMP} && cp -a ${unit} /root/oh-unit-${STAMP}/ && { [ ! -d ${dir} ] || cp -a ${dir} /root/oh-unit-${STAMP}/dropins; } && { [ ! -d ${ctl} ] || cp -a ${ctl} /root/oh-unit-${STAMP}/control; }'" \
        || { echo "  ✗ couldn't back up the unit; nothing changed. ABORTING." >&2; return 1; }
    CHANGING="unit|${label}|${target}|${service}|${use_sudo}|${cur}"
    # Write to a scratch name first and check it there.
    if ! ssh "${SSH_OPTS[@]}" "$target" "${s}tee /etc/systemd/system/${service}.service.new >/dev/null" < "$UNIT_SRC"; then
        echo "  ✗ couldn't upload the unit; nothing changed. ABORTING." >&2; return 1
    fi
    if ! remote "$target" "$use_sudo" "sh -c 'd=\$(mktemp -d) && cp /etc/systemd/system/${service}.service.new \$d/${service}.service && systemd-analyze verify \$d/${service}.service; r=\$?; rm -rf \$d; exit \$r'"; then
        remote "$target" "$use_sudo" "rm -f /etc/systemd/system/${service}.service.new" || true
        CHANGING=""
        echo "  ✗ systemd-analyze refused the unit; nothing changed. ABORTING." >&2; return 1
    fi
    if remote "$target" "$use_sudo" "mv -f /etc/systemd/system/${service}.service.new ${unit}" \
        && remote "$target" "$use_sudo" "mkdir -p ${dir}" \
        && printf '%s\n' "$dropin" | ssh "${SSH_OPTS[@]}" "$target" "${s}tee ${dir}/peers.conf >/dev/null" \
        && remote "$target" "$use_sudo" "rm -f ${ctl}/50-MemoryHigh.conf ${ctl}/50-MemoryMax.conf" \
        && remote "$target" "$use_sudo" "systemctl daemon-reload" \
        && remote "$target" "$use_sudo" "systemctl restart ${service}" \
        && health_check "$label" "$target" "$use_sudo" "$service" "$cur" \
        && [[ "$(unit_prop "$target" "$use_sudo" "$service" MemoryMax)" =~ ^[1-9][0-9]*$ ]]; then
        CHANGING=""
        echo "  ✓ unit applied (MemoryMax $(unit_prop "$target" "$use_sudo" "$service" MemoryMax), hardening on)"
        return 0
    fi
    echo "  ✗ the new unit didn't come up healthy; restoring the old one…" >&2
    restore_unit_node "$label" "$target" "$service" "$use_sudo" "$cur"; CHANGING=""
    return 1
}

# Put a node's previous unit, drop-ins and set-property drop-ins back (backed up at STAMP) and check it.
restore_unit_node() {  # restore_unit_node <label> <target> <service> <sudo> <binary sha>
    local label="$1" target="$2" service="$3" use_sudo="$4" cur="$5"
    local unit="/etc/systemd/system/${service}.service" dir="/etc/systemd/system/${service}.service.d"
    local ctl="/etc/systemd/system.control/${service}.service.d"
    remote "$target" "$use_sudo" "sh -c 'rm -f /etc/systemd/system/${service}.service.new; [ -f /root/oh-unit-${STAMP}/${service}.service ] || exit 0; cp -a /root/oh-unit-${STAMP}/${service}.service ${unit}.new && mv -f ${unit}.new ${unit}; rm -rf ${dir}; [ ! -d /root/oh-unit-${STAMP}/dropins ] || cp -a /root/oh-unit-${STAMP}/dropins ${dir}; [ ! -d /root/oh-unit-${STAMP}/control ] || { mkdir -p ${ctl} && cp -a /root/oh-unit-${STAMP}/control/. ${ctl}/; }'" || true
    if remote "$target" "$use_sudo" "systemctl daemon-reload" \
        && remote "$target" "$use_sudo" "systemctl restart ${service}" \
        && health_check "$label" "$target" "$use_sudo" "$service" "$cur"; then
        echo "  restored: ${label} runs its previous unit. ABORTING — remaining nodes untouched." >&2
    else
        echo "  !!! RESTORING ${label}'s unit FAILED: the node needs attention now (${target}; backup /root/oh-unit-${STAMP})." >&2
    fi
}

# ── Main ─────────────────────────────────────────────────────────────────────
# One rollout at a time: two at once could each have a different node down.
LOCK="${TMPDIR:-/tmp}/openhydra-bootstrap-deploy.lock"
mkdir "$LOCK" 2>/dev/null || { echo "ERROR: another rollout holds $LOCK (remove it if none is running)" >&2; exit 1; }
trap 'rmdir "$LOCK" 2>/dev/null || true; for c in "$CTL_DIR"/*; do [ -S "$c" ] && ssh -o ControlPath="$c" -O exit x 2>/dev/null; done; rm -rf "$CTL_DIR"' EXIT
trap on_interrupt INT TERM HUP
for entry in "${NODES[@]}"; do
    IFS='|' read -r label target service use_sudo <<< "$entry"
    if [[ "$RECONFIGURE_PEERS" == true ]]; then
        if [[ "$APPLY_UNIT" == true ]]; then
            apply_unit_node "$label" "$target" "$service" "$use_sudo" || exit 1
        else
            reconfigure_peers_node "$label" "$target" "$service" "$use_sudo" || exit 1
        fi
    else
        if ! deploy_node "$label" "$target" "$service" "$use_sudo"; then
            exit 1   # abort-on-failure: keep the remaining nodes (and DHT) up
        fi
    fi
    echo ""
done

echo "Done. Verify a node with:"
echo "  ssh -i ${KEY} deploy@45.79.190.172 sudo journalctl -u openhydra-libp2p -f"
echo "Rollback binary on each node (if needed): ${BIN_REMOTE}.bak-${STAMP}"
