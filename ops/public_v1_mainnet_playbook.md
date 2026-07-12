# OpenHydra Public V1 Mainnet Playbook

> **The previous contents of this file described the pre-pivot architecture and were
> removed (G7b).** They referenced a Docker-Compose application canary
> (`scripts/mainnet_canary.sh`, `scripts/slo_chaos_test.py`, `docker compose`, a `peer`
> service, `/v1/completions`) — none of which exist in the pure-Rust BYO-engine protocol.
> An operator following them hit `No such file or directory`. There is no central
> "mainnet" application to canary-deploy: providers run the `openhydra-agent` binary
> wrapping their own engine, and the only shared infrastructure is the bootstrap mesh.

## What actually gets deployed

| Component | Artifact | How |
|-----------|----------|-----|
| **Bootstrap mesh** (3 Linode + 1 netcup) | `openhydra-bootstrap` binary | [`ops/bootstrap/deploy_libp2p.sh`](bootstrap/deploy_libp2p.sh) — node-by-node rollout with per-node binary backup, health-check (active + re-meshed), and **auto-rollback + abort** on any failure, so at most one node is ever touched before a bad rollout stops. See [`ops/bootstrap/README.md`](bootstrap/README.md). |
| **Provider / gateway node** | `openhydra-agent` binary | `openhydra-agent provide` (wraps a local engine) / `openhydra-agent serve` (OpenAI-compatible gateway). Runs standalone; discovers the mesh via the bootstrap DNS names. |

## Bootstrap rollout (the one command that exists)

```bash
./ops/bootstrap/deploy_libp2p.sh                 # build, then roll out node-by-node
./ops/bootstrap/deploy_libp2p.sh path/to/binary  # deploy a prebuilt binary
./ops/bootstrap/deploy_libp2p.sh --reconfigure-peers
```

The script builds, rolls out one node at a time, backs up the old binary, restarts,
health-checks that the node is active **and** re-meshed, and auto-rolls-back + aborts on
any failure — the remaining ≥3 nodes keep the DHT alive throughout. One-time provisioning
(identity keys, firewall, swap, disk) is out of its scope and done by hand per node.

## Smoke test

The gateway speaks the OpenAI Chat Completions API. Verify a live node with:

```bash
curl -s http://<node>:<port>/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"<announced-model>","messages":[{"role":"user","content":"ping"}]}'
```

(The old `/v1/completions` legacy-text endpoint no longer exists.)

## TODO before a public V1

This file is a redirect, not a full runbook. A real V1 launch still needs a documented,
rehearsed procedure for: provider onboarding, mesh capacity/health monitoring
(`ops/prometheus` + `ops/grafana` are wired), incident rollback for the agent fleet, and
the SLO targets the removed canary harness used to assert. Track that as launch work.
