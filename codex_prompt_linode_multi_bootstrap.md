# Codex Prompt — Multi-Node Linode Bootstrap Deployment

## Context

OpenHydra already has a working DHT bootstrap tracker (`dht.bootstrap`) and a
single-node deployment scaffold in `ops/bootstrap/`.  The bootstrap tracker is
architecturally identical to a BitTorrent tracker: it is a stateless HTTP
service that stores peer announcements in memory.  There is **no inter-node
replication** — each node is independent.  Redundancy is achieved by:

- **Peers announcing to all bootstrap URLs** (already implemented in
  `peer/server.py` — `_announce_loop` fans out via `announce_http_many()`).
- **Coordinators querying all bootstrap URLs concurrently** and merging by
  newest `updated_unix_ms` (already implemented in `coordinator/path_finder.py`
  — `load_peers_from_dht()` with `ThreadPoolExecutor`).

**No Python code changes are required.** This prompt covers ops files only:
scripts, systemd units, nginx configs, and documentation inside `ops/bootstrap/`.

---

## Goal

Extend `ops/bootstrap/` to support deploying **three independent Linode Nanode
1 GB nodes** in three geographic regions, producing a multi-tracker setup
equivalent to a BitTorrent swarm with three tracker URLs.

---

## Recommended regions and naming

| Slug         | Linode region          | Example hostname                  |
|---|---|---|
| `us-east`    | Newark, US             | `bootstrap-us.openhydra.example`  |
| `eu-west`    | London / Frankfurt, EU | `bootstrap-eu.openhydra.example`  |
| `ap-south`   | Singapore / Mumbai, AP | `bootstrap-ap.openhydra.example`  |

---

## Files to create or replace

### 1. `ops/bootstrap/nginx-bootstrap.conf.template`

Rename / replace the existing `nginx-bootstrap.conf` with a template that uses
a `%%DOMAIN%%` placeholder.  `setup_nanode.sh` will substitute the real domain
at deploy time using `sed`.

```nginx
server {
    listen 443 ssl http2;
    server_name %%DOMAIN%%;

    ssl_certificate     /etc/letsencrypt/live/%%DOMAIN%%/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/%%DOMAIN%%/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;
    ssl_session_cache   shared:SSL:10m;
    ssl_session_timeout 10m;

    # DHT bootstrap endpoints only — no large uploads, tight timeouts
    client_max_body_size 64k;
    proxy_read_timeout   10s;
    proxy_send_timeout   10s;

    location / {
        proxy_pass       http://127.0.0.1:8468;
        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

server {
    listen 80;
    server_name %%DOMAIN%%;
    return 301 https://$host$request_uri;
}
```

---

### 2. `ops/bootstrap/bootstrap.service`

**Replace** the existing unit file.  Key fixes:
- Use `--secrets-file` to load the geo seed from a file rather than passing it
  as a visible CLI argument (secrets in `ps aux` output is a security defect).
- Remove `EnvironmentFile` — the secrets file is now read by the Python process
  via `load_secret_store()`, not by systemd.
- Add `StandardOutput=journal` and `StandardError=journal` for proper logging.
- Add `PrivateTmp=true` and `NoNewPrivileges=true` for systemd hardening.

```ini
[Unit]
Description=OpenHydra DHT Bootstrap Node
Documentation=https://github.com/your-org/openhydra
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=openhydra
Group=openhydra
WorkingDirectory=/opt/openhydra

ExecStart=/opt/openhydra/.venv/bin/python -m dht.bootstrap \
    --host 0.0.0.0 \
    --port 8468 \
    --deployment-profile prod \
    --secrets-file /etc/openhydra/secrets.env \
    --ttl-seconds 300 \
    --geo-challenge-enabled \
    --geo-max-rtt-ms 80

Restart=always
RestartSec=5
LimitNOFILE=65536

# Systemd hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/opt/openhydra

StandardOutput=journal
StandardError=journal
SyslogIdentifier=openhydra-bootstrap

[Install]
WantedBy=multi-user.target
```

---

### 3. `ops/bootstrap/setup_nanode.sh`

**Replace** the existing script.  Requirements:
- Positional arg: domain name (e.g. `bootstrap-us.openhydra.example`).
- Fully idempotent — safe to re-run on an already-provisioned node.
- Substitutes `%%DOMAIN%%` in the nginx config template.
- Uses `--secrets-file` (not a CLI flag) for the geo seed.
- Installs **only** `grpcio protobuf cryptography` — no torch, no transformers.
- Sets `geo-max-rtt-ms 80` (relaxed from the 50 ms default to accommodate
  legitimate inter-continental jitter for some edge regions).
- Prints a clear checklist of what the operator must do manually after the
  script completes.

```bash
#!/usr/bin/env bash
# setup_nanode.sh <domain>
#
# Idempotent provisioning script for an OpenHydra DHT bootstrap node.
# Run as root on a fresh Linode Nanode 1 GB (Ubuntu 22.04).
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

echo "==> Provisioning bootstrap node: $DOMAIN"

# --- System packages ---
apt-get update -qq
apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv \
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
else
    echo "==> Cloning repo"
    git clone "$REPO_URL" /opt/openhydra
fi

# --- Python venv (bootstrap-only deps — no ML packages) ---
python3.11 -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet grpcio protobuf cryptography

# --- Secrets file (create skeleton if absent) ---
if [ ! -f "$SECRETS_FILE" ]; then
    cat > "$SECRETS_FILE" <<'EOF'
# OpenHydra bootstrap node secrets
# All three bootstrap nodes MUST share the same GEO_CHALLENGE_SEED value.
# Generate with: openssl rand -hex 32
OPENHYDRA_GEO_CHALLENGE_SEED=REPLACE_WITH_SHARED_SECRET
EOF
    chmod 600 "$SECRETS_FILE"
    chown root:root "$SECRETS_FILE"
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
sed "s/%%DOMAIN%%/${DOMAIN}/g" \
    /opt/openhydra/ops/bootstrap/nginx-bootstrap.conf.template \
    > "$NGINX_CONF"
ln -sf "$NGINX_CONF" /etc/nginx/sites-enabled/openhydra-bootstrap.conf
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl reload nginx

# --- systemd service ---
install -m 644 /opt/openhydra/ops/bootstrap/bootstrap.service "$SERVICE_FILE"
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
```

---

### 4. `ops/bootstrap/gen_geo_seed.sh`

New script.  Generates a single strong `OPENHYDRA_GEO_CHALLENGE_SEED` and
prints the `ssh` commands to distribute it to all three nodes.  The operator
runs this **once** locally and pastes the output into their terminal.

```bash
#!/usr/bin/env bash
# gen_geo_seed.sh [node1-ip] [node2-ip] [node3-ip]
#
# Generates a shared geo-challenge seed and prints distribution commands.
# Run this once on your local machine, then paste the output into your terminal.
#
# Usage:
#   bash gen_geo_seed.sh 192.0.2.1 198.51.100.1 203.0.113.1

set -euo pipefail

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
```

---

### 5. `ops/bootstrap/deploy_all.sh`

New script.  Runs `setup_nanode.sh` on each of the three nodes via SSH in
parallel, then tail-follows the service log on each node to confirm it is up.

```bash
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
: "${OPENHYDRA_REPO_URL:?set OPENHYDRA_REPO_URL}"

deploy_node() {
    local ip="$1"
    local domain="$2"
    echo "[${domain}] Starting deploy..."
    ssh -o StrictHostKeyChecking=accept-new root@"$ip" \
        "OPENHYDRA_REPO_URL='${OPENHYDRA_REPO_URL}' bash /opt/openhydra/ops/bootstrap/setup_nanode.sh '${domain}'"
    echo "[${domain}] Deploy complete."
}

# Upload the ops directory first (in case the repo isn't cloned yet)
for IP in "$BOOTSTRAP_US_IP" "$BOOTSTRAP_EU_IP" "$BOOTSTRAP_AP_IP"; do
    ssh -o StrictHostKeyChecking=accept-new root@"$IP" \
        "mkdir -p /opt/openhydra/ops/bootstrap" &
done
wait

for IP in "$BOOTSTRAP_US_IP" "$BOOTSTRAP_EU_IP" "$BOOTSTRAP_AP_IP"; do
    rsync -az ops/bootstrap/ root@"$IP":/opt/openhydra/ops/bootstrap/ &
done
wait

# Run setup in parallel
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
```

---

### 6. `ops/bootstrap/README.md`

**Replace** the existing file with comprehensive multi-node documentation.

The README must cover:

#### Purpose
- What the bootstrap node is and isn't (tracker only, no model weights, ~150 MB)
- Why three nodes (geographic redundancy, no single point of failure)
- The BitTorrent-tracker analogy: each node is independent; merge happens at the coordinator

#### Prerequisites
- Three Linode Nanode 1 GB instances provisioned in three regions
- DNS A records for each domain pointing to the respective instance IP
- SSH access as root

#### Step-by-step setup
1. Generate a shared geo seed with `gen_geo_seed.sh` and note it securely
2. Set env vars (`BOOTSTRAP_US_IP`, etc.) and run `deploy_all.sh`
3. On each node: edit `/etc/openhydra/secrets.env` with the shared geo seed
4. On each node: `systemctl start openhydra-bootstrap.service`
5. Verify all three with `curl https://<domain>/health`

#### Alternative: manual single-node setup
Show how to run `setup_nanode.sh` directly on one node.

#### Coordinator configuration
```bash
python -m coordinator.api_server \
  --dht-url https://bootstrap-us.openhydra.example \
  --dht-url https://bootstrap-eu.openhydra.example \
  --dht-url https://bootstrap-ap.openhydra.example \
  --model-catalog-path ./models.catalog.json
```
Explain: coordinator queries all three concurrently, merges results by
`peer_id` keeping the newest `updated_unix_ms`, raises only if **all three**
fail.

#### Peer configuration
```bash
python -m peer.server \
  --peer-id my-peer-01 \
  --host 0.0.0.0 --port 50051 \
  --advertise-host <YOUR_PUBLIC_IP> \
  --shard-index 0 --total-shards 1 \
  --dht-url https://bootstrap-us.openhydra.example \
  --dht-url https://bootstrap-eu.openhydra.example \
  --dht-url https://bootstrap-ap.openhydra.example \
  --runtime-backend pytorch_cpu \
  --runtime-model-id Qwen/Qwen3.5-0.8B \
  --quantization-mode int4 \
  --data-dir ~/.openhydra
```
Explain: peer announces to **all** URLs on each announce interval; a partial
failure (one node down) is logged as a warning but does not block the peer.

#### Service lifecycle
```bash
# On each bootstrap node:
sudo systemctl status openhydra-bootstrap.service
sudo systemctl restart openhydra-bootstrap.service
sudo journalctl -u openhydra-bootstrap.service -f
```

#### Health check
```bash
curl -s https://bootstrap-us.openhydra.example/health | python3 -m json.tool
```
Expected response contains `"status": "ok"` and a `peers_count` metric.

#### Upgrade procedure
```bash
# On each node:
sudo -u openhydra git -C /opt/openhydra pull --ff-only
sudo systemctl restart openhydra-bootstrap.service
```
Upgrade one node at a time.  Peers and coordinators tolerate one node being
briefly unavailable.

#### Geo seed rotation
Document how to generate a new seed and distribute it atomically across all
three nodes before restarting, ensuring coordinators and peers configured with
the same seed remain compatible.

#### Scaling guidance
- 2-3 nodes are sufficient for global production traffic (the in-memory DHT is
  trivially small — each peer record is ~1 KB).
- A single Nanode serves thousands of announce+lookup RPS at the typical
  OpenHydra cadence (60-second announce interval).
- No shared state or consensus protocol is required between nodes.
- If a node is lost, peer records expire naturally after `--ttl-seconds` (300 s
  default); peers re-announce to remaining nodes within one announce interval.

---

## Constraints

- **No changes to any Python source files.**
- All scripts must be POSIX-compatible `bash` (not `sh`).
- `setup_nanode.sh` must be idempotent — re-running it on an already-provisioned
  node must not break anything.
- The nginx config template must use `%%DOMAIN%%` (double-percent) as the
  placeholder to avoid conflicts with nginx's own `$variable` syntax.
- The `bootstrap.service` unit must **not** pass secrets as CLI arguments
  (they appear in `ps aux`); it must use `--secrets-file` instead.
- No `torch`, `transformers`, or any ML package must be installed on the
  bootstrap node.  The pip install line must list only:
  `grpcio protobuf cryptography`.
- All new shell scripts must have `set -euo pipefail` and executable permission
  (`chmod +x`).
