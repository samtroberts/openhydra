# OpenHydra Public V1 Mainnet Playbook

## One-command canary rollout

```bash
./scripts/mainnet_canary.sh rollout openhydra-mainnet-canary
```

The rollout command performs:

1. Preflight (`docker info`, `curl`, `python3`)
2. Canary deploy (`docker compose -p <project> up -d --build`)
3. Health gating (`/health`, `/metrics`)
4. Smoke inference request (`/v1/completions`)
5. Sustained load + chaos SLO test (`scripts/slo_chaos_test.py`)
6. Checklist + SLO report generation under `.openhydra/canary_reports/`

## One-command rollback

```bash
./scripts/mainnet_canary.sh rollback openhydra-mainnet-canary
```

Rollback performs:

1. Log capture (`docker compose logs --no-color`)
2. Stack teardown (`docker compose down -v --remove-orphans`)
3. Rollback report generation under `.openhydra/canary_reports/`

## Recommended production env overrides

```bash
export OPENHYDRA_CANARY_SLO_DURATION_S=300
export OPENHYDRA_CANARY_WORKERS=48
export OPENHYDRA_CANARY_MIN_SUCCESS_RATE=0.99
export OPENHYDRA_CANARY_MAX_P95_MS=900
export OPENHYDRA_CANARY_CHAOS_AT_S=60
export OPENHYDRA_CANARY_CHAOS_SERVICE=peer
```

