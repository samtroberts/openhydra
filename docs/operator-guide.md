# Operator Guide

This guide covers deploying and operating OpenHydra bootstrap nodes in production on Linode (recommended) and ARM-based hardware (Oracle Cloud A1, Raspberry Pi 5).

---

## Infrastructure overview

```
┌─────────────────────────────────────────────────┐
│  Cloudflare (Layer 2)                           │
│  Proxy + WAF + Rate limiting on 8080/8468        │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│  Linode Cloud Firewall (Layer 1)                │
│  DROP by default; ACCEPT 22/80/443/8080/8468    │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│  Ubuntu 22.04 nanode (1 GB RAM, 1 vCPU)         │
│  iptables OPENHYDRA chain (Layer 3)             │
│  openhydra-dht  :8468                           │
│  openhydra-coordinator  :8080                   │
└─────────────────────────────────────────────────┘
```

---

## Linode deployment

### 1. Provision nanodes

```bash
# Bootstrap nodes (3 regions recommended for HA)
linode-cli linodes create \
  --label bootstrap-us \
  --region us-central \
  --type g6-nanode-1 \
  --image linode/ubuntu22.04 \
  --root_pass "$(openssl rand -hex 20)" \
  --authorized_keys "$(cat ~/.ssh/openhydra_bootstrap.pub)"
```

Repeat for `bootstrap-eu` (`eu-central`) and `bootstrap-ap` (`ap-southeast`).

### 2. Create Linode Cloud Firewall

```bash
linode-cli firewalls create \
  --label openhydra-bootstrap \
  --rules.inbound_policy DROP \
  --rules.outbound_policy ACCEPT \
  --rules.inbound '[
    {"action":"ACCEPT","protocol":"TCP","ports":"22","addresses":{"ipv4":["0.0.0.0/0"]}},
    {"action":"ACCEPT","protocol":"TCP","ports":"80,443","addresses":{"ipv4":["0.0.0.0/0"]}},
    {"action":"ACCEPT","protocol":"TCP","ports":"8080","addresses":{"ipv4":["0.0.0.0/0"]}},
    {"action":"ACCEPT","protocol":"TCP","ports":"8468","addresses":{"ipv4":["0.0.0.0/0"]}},
    {"action":"ACCEPT","protocol":"TCP","ports":"50051","addresses":{"ipv4":["0.0.0.0/0"]}}
  ]'
```

### 3. Attach firewall to nodes

```bash
linode-cli firewalls device-create <firewall-id> \
  --type linode --id <linode-id>
```

### 4. Bootstrap node software

```bash
ssh root@<node-ip> "
  apt-get update -q && apt-get install -y -q \
    python3.11 python3-pip python3-venv protobuf-compiler iptables-persistent

  git clone https://github.com/openhydra-ai/openhydra.git /opt/openhydra
  cd /opt/openhydra
  python3.11 -m venv .venv
  source .venv/bin/activate
  pip install -e .
  make proto
"
```

### 5. Apply network hardening

```bash
sudo bash ops/network_limits.sh
```

This script (idempotent) applies:

- **sysctl TCP hardening** — SYN cookies, reverse path filtering, keepalive tuning
- **iptables OPENHYDRA chain** — per-IP rate limits on DHT (20 new conns/min), gRPC (≤5 concurrent), ICMP flood protection

Verify with:

```bash
sudo bash ops/network_limits.sh --check
```

---

## Cloudflare setup

### DNS records

| Name | Type | Value | Mode |
|------|------|-------|------|
| `openhydra.co` | A | `45.79.190.172` | 🟠 Proxied |
| `www.openhydra.co` | CNAME | `openhydra.co` | 🟠 Proxied |
| `bootstrap-us.openhydra.co` | A | `45.79.190.172` | ⚫ DNS-only |
| `bootstrap-eu.openhydra.co` | A | `172.105.69.49` | ⚫ DNS-only |
| `bootstrap-ap.openhydra.co` | A | `172.104.164.98` | ⚫ DNS-only |

`openhydra.co` and `www` should be **proxied** (orange cloud) — Cloudflare handles DDoS protection and TLS termination for the landing page and coordinator API.

Bootstrap subdomains **must be DNS-only** (grey cloud). The DHT service runs on port 8468 which Cloudflare cannot proxy; peers need to connect directly to the node IPs. Setting bootstrap records to proxied would break peer discovery.

### Recommended zone settings

```bash
CF_TOKEN="<your-token>"
ZONE_ID="<your-zone-id>"

# SSL Full mode
curl -X PATCH "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/settings/ssl" \
  -H "Authorization: Bearer $CF_TOKEN" -H "Content-Type: application/json" \
  -d '{"value":"full"}'

# Always HTTPS
curl -X PATCH "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/settings/always_use_https" \
  -H "Authorization: Bearer $CF_TOKEN" -H "Content-Type: application/json" \
  -d '{"value":"on"}'

# Security level: high
curl -X PATCH "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/settings/security_level" \
  -H "Authorization: Bearer $CF_TOKEN" -H "Content-Type: application/json" \
  -d '{"value":"high"}'
```

---

## TLS / HTTPS

For the coordinator API, terminate TLS at Cloudflare (Full mode). For gRPC peers on port 50051, use self-signed certificates or mTLS with the provided `ops/gen_certs.sh` script.

```bash
# Generate peer certificates
bash ops/gen_certs.sh --peer-id my-node --out certs/
```

---

## Systemd services

Create `/etc/systemd/system/openhydra-dht.service`:

```ini
[Unit]
Description=OpenHydra DHT Bootstrap
After=network.target

[Service]
Type=simple
User=nobody
WorkingDirectory=/opt/openhydra
ExecStart=/opt/openhydra/.venv/bin/openhydra-dht --host 0.0.0.0 --port 8468
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
systemctl daemon-reload
systemctl enable --now openhydra-dht
```

Repeat for `openhydra-coordinator` pointing to `openhydra-coordinator --port 8080`.

---

## Prometheus monitoring

The coordinator exposes metrics at `/metrics` (Prometheus text format). Scrape it from your Prometheus instance:

```yaml
scrape_configs:
  - job_name: openhydra
    static_configs:
      - targets:
          - openhydra.co:8080
```

Key metrics:

| Metric | Description |
|--------|-------------|
| `openhydra_requests_total` | Total inference requests served |
| `openhydra_peer_latency_ms` | gRPC round-trip latency by peer |
| `openhydra_peers_healthy` | Number of healthy registered peers |
| `openhydra_kv_compaction_evictions_total` | KV cache entries evicted |
| `openhydra_barter_credits_awarded_total` | Credits awarded to peers |

---

## ARM deployment (Oracle A1)

Oracle Cloud Free Tier A1 instances (4 vCPU / 24 GB RAM, ARM Ampere) are well-suited for running larger models:

```bash
# Install on Oracle Linux / Ubuntu ARM
sudo apt-get install -y python3.11 python3-pip protobuf-compiler

# GGUF models run natively via llama-cpp-python (ARM optimised)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

The `openhydra-peer` process auto-detects ARM and adjusts thread count to match the Ampere core layout.

---

## Backups

The coordinator SQLite ledger is stored at `~/.openhydra/ledger.db`. Back it up daily:

```bash
# Add to crontab
0 2 * * * sqlite3 ~/.openhydra/ledger.db ".backup /backups/ledger-$(date +%F).db"
```

For PostgreSQL ledger (production): use `pg_dump` with your standard backup tooling.
