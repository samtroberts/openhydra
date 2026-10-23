#!/usr/bin/env bash
# OpenHydra daily network count (SL10), run by openhydra-count.timer once a day.
#
# Appends one line of totals to daily.jsonl: when, and how many peers, models and providers the
# network showed the probe (gossip for 150 s, plus the bootstrap registry for every known model
# name). Aggregates only: no peer ids, no prompts. A lower bound: a peer-to-peer network can't be
# counted completely.
set -uo pipefail
AGENT=${AGENT:-/opt/openhydra-monitor/bin/openhydra-agent}
STATE=${STATE_DIRECTORY:-/var/lib/openhydra-monitor}

HOME="$STATE" "$AGENT" --identity "$STATE/count.key" --listen /ip4/127.0.0.1/tcp/0 \
  probe --count-models --observe-secs 150 --no-test-request 2>"$STATE/count.log" \
  | python3 -I -c 'import json,sys; r=json.load(sys.stdin); print(json.dumps({"at_ms": r["at_ms"], "version": r["version"], **r["totals"]}))' \
  >> "$STATE/daily.jsonl"
