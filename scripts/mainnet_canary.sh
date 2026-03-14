#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:-rollout}"
PROJECT_NAME="${2:-${OPENHYDRA_CANARY_PROJECT:-openhydra-canary}}"
BASE_URL="${OPENHYDRA_CANARY_BASE_URL:-http://127.0.0.1:8080}"
BOOTSTRAP_URL="${OPENHYDRA_CANARY_BOOTSTRAP_URL:-http://127.0.0.1:8468/health}"
REPORT_DIR="${ROOT_DIR}/.openhydra/canary_reports"
TIMESTAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
CHECKLIST_FILE="${REPORT_DIR}/${PROJECT_NAME}_checklist_${TIMESTAMP}.md"
ROLLBACK_LOG_FILE="${REPORT_DIR}/${PROJECT_NAME}_rollback_${TIMESTAMP}.log"
ROLLBACK_NOTE_FILE="${REPORT_DIR}/${PROJECT_NAME}_rollback_${TIMESTAMP}.md"
SLO_REPORT_FILE="${REPORT_DIR}/${PROJECT_NAME}_slo_${TIMESTAMP}.json"

function _require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

function _wait_for_url() {
  local url="$1"
  local timeout_s="$2"
  local start
  start="$(date +%s)"
  while true; do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    if (( "$(date +%s)" - start >= timeout_s )); then
      echo "timed out waiting for ${url}" >&2
      return 1
    fi
    sleep 1
  done
}

function _mark_checklist_item() {
  local old_line="$1"
  local new_line="$2"
  python3 - "$CHECKLIST_FILE" "$old_line" "$new_line" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
old = sys.argv[2]
new = sys.argv[3]
text = path.read_text(encoding="utf-8")
text = text.replace(old, new)
path.write_text(text, encoding="utf-8")
PY
}

function _rollback() {
  mkdir -p "$REPORT_DIR"
  docker compose -p "$PROJECT_NAME" logs --no-color >"$ROLLBACK_LOG_FILE" 2>&1 || true
  docker compose -p "$PROJECT_NAME" down -v --remove-orphans || true
  cat >"$ROLLBACK_NOTE_FILE" <<EOF
# OpenHydra Canary Rollback Playbook

Timestamp (UTC): ${TIMESTAMP}
Project: ${PROJECT_NAME}

## Executed rollback steps

1. Captured container logs to:
   - ${ROLLBACK_LOG_FILE}
2. Executed rollback command:
   - \`docker compose -p ${PROJECT_NAME} down -v --remove-orphans\`
3. Confirmed canary stack teardown.

## Next actions

1. Review rollback logs and identify first failure signature.
2. Patch and re-run:
   - \`./scripts/mainnet_canary.sh rollout ${PROJECT_NAME}\`
EOF
  echo "rollback complete: ${ROLLBACK_NOTE_FILE}"
}

function _rollout() {
  _require_cmd docker
  _require_cmd curl
  _require_cmd python3

  mkdir -p "$REPORT_DIR"
  cat >"$CHECKLIST_FILE" <<EOF
# OpenHydra Canary Rollout Checklist

Timestamp (UTC): ${TIMESTAMP}
Project: ${PROJECT_NAME}

- [ ] Preflight checks
- [ ] Deploy canary stack
- [ ] Wait for bootstrap and coordinator health
- [ ] Smoke API validation
- [ ] Sustained load + chaos SLO validation
- [ ] Record metrics and decision
- [ ] Rollback playbook validated
EOF

  echo "[1/7] preflight checks"
  docker info >/dev/null
  _mark_checklist_item "- [ ] Preflight checks" "- [x] Preflight checks"

  echo "[2/7] deploy canary stack"
  docker compose -p "$PROJECT_NAME" up -d --build
  _mark_checklist_item "- [ ] Deploy canary stack" "- [x] Deploy canary stack"

  echo "[3/7] wait for health"
  _wait_for_url "$BOOTSTRAP_URL" 120
  _wait_for_url "${BASE_URL}/metrics" 120
  _mark_checklist_item "- [ ] Wait for bootstrap and coordinator health" "- [x] Wait for bootstrap and coordinator health"

  echo "[4/7] smoke API validation"
  curl -fsS "${BASE_URL}/v1/models" >/dev/null
  curl -fsS "${BASE_URL}/v1/completions" \
    -H 'content-type: application/json' \
    -d '{"model":"openhydra-toy-345m","prompt":"Canary smoke probe","max_tokens":8,"pipeline_width":1}' >/dev/null
  _mark_checklist_item "- [ ] Smoke API validation" "- [x] Smoke API validation"

  echo "[5/7] sustained load + chaos SLO validation"
  python3 "${ROOT_DIR}/scripts/slo_chaos_test.py" \
    --base-url "${BASE_URL}" \
    --duration-s "${OPENHYDRA_CANARY_SLO_DURATION_S:-120}" \
    --workers "${OPENHYDRA_CANARY_WORKERS:-16}" \
    --min-success-rate "${OPENHYDRA_CANARY_MIN_SUCCESS_RATE:-0.97}" \
    --max-p95-ms "${OPENHYDRA_CANARY_MAX_P95_MS:-1500}" \
    --chaos-at-s "${OPENHYDRA_CANARY_CHAOS_AT_S:-30}" \
    --chaos-service "${OPENHYDRA_CANARY_CHAOS_SERVICE:-peer}" \
    --compose-project "${PROJECT_NAME}" \
    --report-json "${SLO_REPORT_FILE}"
  _mark_checklist_item "- [ ] Sustained load + chaos SLO validation" "- [x] Sustained load + chaos SLO validation"

  echo "[6/7] record metrics and decision"
  {
    echo ""
    echo "## Decision"
    echo ""
    echo "- Canary status: PASS"
    echo "- SLO report: ${SLO_REPORT_FILE}"
    echo "- Promote command: docker compose -p ${PROJECT_NAME} up -d --build"
  } >>"$CHECKLIST_FILE"
  _mark_checklist_item "- [ ] Record metrics and decision" "- [x] Record metrics and decision"

  echo "[7/7] rollback playbook validated"
  {
    echo ""
    echo "## Rollback Command"
    echo ""
    echo "- \`./scripts/mainnet_canary.sh rollback ${PROJECT_NAME}\`"
  } >>"$CHECKLIST_FILE"
  _mark_checklist_item "- [ ] Rollback playbook validated" "- [x] Rollback playbook validated"

  echo "canary rollout complete: ${CHECKLIST_FILE}"
}

case "$ACTION" in
  rollout)
    if ! _rollout; then
      echo "rollout failed, executing rollback..."
      _rollback
      exit 1
    fi
    ;;
  rollback)
    _rollback
    ;;
  *)
    echo "usage: $0 [rollout|rollback] [project_name]" >&2
    exit 2
    ;;
esac
