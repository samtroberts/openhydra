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

export OPENHYDRA_REPO_URL="https://github.com/your-org/openhydra.git"

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
curl -s https://bootstrap-us.openhydra.example/health | python3 -m json.tool
curl -s https://bootstrap-eu.openhydra.example/health | python3 -m json.tool
curl -s https://bootstrap-ap.openhydra.example/health | python3 -m json.tool
```

## Alternative: Manual Single-Node Setup

On one node:

```bash
bash /opt/openhydra/ops/bootstrap/setup_nanode.sh bootstrap-us.openhydra.example
```

Then edit `/etc/openhydra/secrets.env`, start the service, and verify `/health`.

## Coordinator Configuration

```bash
python -m coordinator.api_server \
  --dht-url https://bootstrap-us.openhydra.example \
  --dht-url https://bootstrap-eu.openhydra.example \
  --dht-url https://bootstrap-ap.openhydra.example \
  --model-catalog-path ./models.catalog.json
```

Coordinator behavior:

- Queries all bootstrap URLs concurrently
- Merges results by `peer_id`, keeping newest `updated_unix_ms`
- Raises only if all bootstrap lookups fail

## Peer Configuration

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

Peer behavior:

- Announces to all bootstrap URLs each interval
- Partial announce failures are warnings only
- Peer continues operating as long as at least one bootstrap is reachable

## Service Lifecycle

On each bootstrap node:

```bash
sudo systemctl status openhydra-bootstrap.service
sudo systemctl restart openhydra-bootstrap.service
sudo journalctl -u openhydra-bootstrap.service -f
```

## Health Check

```bash
curl -s https://bootstrap-us.openhydra.example/health | python3 -m json.tool
```

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
