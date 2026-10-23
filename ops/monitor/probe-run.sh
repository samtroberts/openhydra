#!/usr/bin/env bash
# OpenHydra network probe (SL10), run by openhydra-probe.timer every 5 minutes.
#
# Runs `openhydra-agent probe`, keeps its last report, and tells Healthchecks
# (https://healthchecks.io) how it went:
#   - all well                    → ping            (a missing ping also alerts: the probe box died)
#   - problems, first time        → /log            (recorded, no alert: one blip isn't an outage)
#   - problems, twice in a row    → /fail + problems (alert)
# Settings come from /etc/openhydra/monitor.env (see README.md): HC_URL, EXPECT_SEEDS.
set -uo pipefail
: "${HC_URL:?set HC_URL in /etc/openhydra/monitor.env}"
EXPECT_SEEDS=${EXPECT_SEEDS:-0}
AGENT=${AGENT:-/opt/openhydra-monitor/bin/openhydra-agent}
STATE=${STATE_DIRECTORY:-/var/lib/openhydra-monitor}

report=$(HOME="$STATE" "$AGENT" --identity "$STATE/probe.key" --listen /ip4/127.0.0.1/tcp/0 \
  probe --expect-seeds "$EXPECT_SEEDS" 2>"$STATE/probe.log")
code=$?
[ -n "$report" ] && printf '%s\n' "$report" > "$STATE/last.json"
problems=$(printf '%s' "$report" | python3 -I -c 'import json,sys; print("\n".join(json.load(sys.stdin)["problems"]))' 2>/dev/null)
[ "$code" -ne 0 ] && [ -z "$problems" ] && problems="probe failed to run (exit $code): $(tail -3 "$STATE/probe.log")"

ping() { curl -fsS -m 10 --retry 3 -o /dev/null --data-raw "${2:-}" "$HC_URL$1" || echo "couldn't reach Healthchecks ($HC_URL$1)" >&2; }
if [ "$code" -eq 0 ]; then
  rm -f "$STATE/failing"
  ping ""
elif [ -e "$STATE/failing" ]; then
  ping /fail "$problems"
else
  touch "$STATE/failing"
  ping /log "first failure, alerting if it repeats: $problems"
fi
exit 0
