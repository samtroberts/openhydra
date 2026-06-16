# OpenHydra Multi-Node Bootstrap Deployment (Linode)

## Purpose

OpenHydra bootstrap nodes run only the DHT tracker service (`dht.bootstrap`) behind nginx TLS termination.
They do **not** host model weights and do **not** execute inference.

- Typical node footprint: ~150 MB (Python service, nginx, in-memory DHT records)
- Recommended host size: Linode Nanode 1 GB

Use three independent bootstrap nodes for geographic redundancy:

- `us-east` (`bootstrap-us.openhydra.example`)
- `eu-west` (`bootstrap-eu.openhydra.example`)
- `ap-south` (`bootstrap-ap.openhydra.example`)

BitTorrent tracker analogy:

- Each bootstrap node is independent (no inter-node replication)
- Peers announce to all trackers
- Coordinators query all trackers and merge by latest `updated_unix_ms`

## Prerequisites

- Three Linode Nanode 1 GB instances (one per region, Ubuntu 24.04 LTS)
- DNS A records pointing each bootstrap domain to its node IP
- SSH access as `root` to each node

## Step-by-Step Setup

1. Generate a shared geo seed once:

```bash
bash ops/bootstrap/gen_geo_seed.sh <US_IP> <EU_IP> <AP_IP>
```

2. Export deploy variables and run parallel deployment:

```bash
export BOOTSTRAP_US_IP="<US_IP>"
export BOOTSTRAP_EU_IP="<EU_IP>"
export BOOTSTRAP_AP_IP="<AP_IP>"

export BOOTSTRAP_US_DOMAIN="bootstrap-us.openhydra.example"
export BOOTSTRAP_EU_DOMAIN="bootstrap-eu.openhydra.example"
export BOOTSTRAP_AP_DOMAIN="bootstrap-ap.openhydra.example"

export OPENHYDRA_REPO_URL="https://github.com/samtroberts/openhydra.git"

bash ops/bootstrap/deploy_all.sh
```

3. On each node, edit `/etc/openhydra/secrets.env` and set the same value for:

```bash
OPENHYDRA_GEO_CHALLENGE_SEED=<shared-secret>
```

4. On each node, start the service:

```bash
sudo systemctl start openhydra-bootstrap.service
```

5. Verify all three endpoints:

```bash
curl -s https://bootstrap-us.openhydra.example/healthcurl -s https://bootstrap-eu.openhydra.example/healthcurl -s https://bootstrap-ap.openhydra.example/health```

## Alternative: Manual Single-Node Setup

On one node:

```bash
bash /opt/openhydra/ops/bootstrap/setup_nanode.sh bootstrap-us.openhydra.example
```

Then edit `/etc/openhydra/secrets.env`, start the service, and verify `/health`.

## Using a bootstrap from an agent

These nodes are libp2p bootstrap/relay peers (the Rust `openhydra-bootstrap`
binary, dual-stack TCP+QUIC on :4001). An `openhydra-agent` (provider or
gateway) joins the swarm through them with `--bootstrap`:

```bash
openhydra-agent \
  --bootstrap /dns4/bootstrap-us.openhydra.example/udp/4001/quic-v1/p2p/<PEER_ID> \
  provide --engine-kind ollama
```

On a LAN, mDNS discovers peers without a bootstrap; across networks, pass one or
more `--bootstrap` multiaddrs. NAT'd agents reserve a relay slot on a reachable
bootstrap automatically (Circuit Relay v2) and hole-punch via DCUtR when possible.

## Service Lifecycle

On each bootstrap node:

```bash
sudo systemctl status openhydra-bootstrap.service
sudo systemctl restart openhydra-bootstrap.service
sudo journalctl -u openhydra-bootstrap.service -f
```

## Health Check

```bash
curl -s https://bootstrap-us.openhydra.example/health```

Expected response includes `"status": "ok"` and `peers_count`.

## Upgrade Procedure

Upgrade one node at a time:

```bash
# On each node:
sudo -u openhydra git -C /opt/openhydra pull --ff-only
sudo systemctl restart openhydra-bootstrap.service
```

OpenHydra peers/coordinators tolerate one bootstrap node being temporarily unavailable.

## Geo Seed Rotation

1. Generate a new seed value.
2. Distribute it to all three nodes (`/etc/openhydra/secrets.env`) before restart.
3. Restart bootstrap service on each node in a controlled rollout.

Use the same seed on all bootstrap nodes to keep geo challenge verification consistent.

## Scaling Guidance

- 2-3 bootstrap nodes are enough for global production traffic.
- Peer records are small (~1 KB each), so memory usage stays low.
- A single Nanode can handle high announce/lookup RPS at 60-second announce cadence.
- No shared state or consensus protocol is required between bootstrap nodes.
- If one node is lost, records on that node expire after `--ttl-seconds` (default 300s), and peers re-announce to surviving nodes on next interval.
