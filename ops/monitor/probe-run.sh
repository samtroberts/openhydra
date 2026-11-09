#!/usr/bin/env bash
# OpenHydra network probe (SL10), run by openhydra-probe.timer every 5 minutes.
#
# Runs `openhydra-agent probe`, keeps its last report, and emails a person (through msmtp):
#   - problems twice in a row → "PROBLEM" (one blip is only logged), then a reminder every 6 hours
#     while they last, and "RECOVERED" when a run is clean again;
#   - once a week, "still watching" with the last report, so a probe that silently stopped is
#     noticed by the missing email.
# Two probes on different hosts watch each other: each checks every bootstrap, so a dead probe
# host shows up as an unreachable bootstrap in the other's email.
# Settings come from /etc/openhydra/monitor.env (see README.md): ALERT_TO, EXPECT_SEEDS.
set -uo pipefail
: "${ALERT_TO:?set ALERT_TO in /etc/openhydra/monitor.env}"
EXPECT_SEEDS=${EXPECT_SEEDS:-0}
AGENT=${AGENT:-/opt/openhydra-monitor/bin/openhydra-agent}
STATE=${STATE_DIRECTORY:-/var/lib/openhydra-monitor}
MAILER=${MAILER:-msmtp --file=/etc/openhydra/msmtprc -t}
REMIND_SECS=${REMIND_SECS:-21600}
WEEKLY_SECS=${WEEKLY_SECS:-604800}
HOST=$(hostname -s 2>/dev/null || hostname)
now=$(date +%s)

report=$(HOME="$STATE" "$AGENT" --identity "$STATE/probe.key" --listen /ip4/127.0.0.1/tcp/0 \
  probe --expect-seeds "$EXPECT_SEEDS" 2>"$STATE/probe.log")
code=$?
[ -n "$report" ] && printf '%s\n' "$report" > "$STATE/last.json"
problems=$(printf '%s' "$report" | python3 -I -c 'import json,sys; print("\n".join(json.load(sys.stdin)["problems"]))' 2>/dev/null)
[ "$code" -ne 0 ] && [ -z "$problems" ] && problems="the probe failed to run (exit $code): $(tail -3 "$STATE/probe.log")"

mail() { # subject, body; fails if the email wasn't sent, so it's tried again next run
  printf 'To: %s\nSubject: [OpenHydra probe %s] %s\nContent-Type: text/plain; charset=utf-8\n\n%s\n\n-- \nLast report: %s/last.json on %s\n' \
    "$ALERT_TO" "$HOST" "$1" "$2" "$STATE" "$HOST" | $MAILER && return 0
  echo "couldn't send the email: $1" >&2
  return 1
}
# Seconds since the time stored in $1; very large when the file is missing, empty, garbled (a write
# on a full disk) or in the future (a clock that ran ahead), so a broken state file errs towards
# alerting rather than going quiet.
age() {
  t=$(cat "$1" 2>/dev/null)
  case "$t" in ''|*[!0-9]*|???????????*) echo 999999999; return;; esac
  if [ $(( 10#$t )) -gt "$now" ]; then echo 999999999; else echo $(( now - 10#$t )); fi
}

if [ "$code" -eq 0 ]; then
  if [ ! -e "$STATE/alerted" ] || mail "RECOVERED" "The last run found no problems."; then
    rm -f "$STATE/failing" "$STATE/alerted"
  fi
elif [ ! -e "$STATE/failing" ]; then
  echo "$now" > "$STATE/failing"
  echo "first failure, emailing if it repeats: $problems" >&2
elif [ "$(age "$STATE/alerted")" -ge "$REMIND_SECS" ]; then
  subject="PROBLEM"; [ -e "$STATE/alerted" ] && subject="still failing since $(date -u -d "@$(cat "$STATE/failing")" +%FT%TZ 2>/dev/null || cat "$STATE/failing")"
  mail "$subject" "$problems" && echo "$now" > "$STATE/alerted"
fi
if [ "$(age "$STATE/weekly")" -ge "$WEEKLY_SECS" ]; then
  mail "still watching" "Weekly check-in: this probe runs every 5 minutes. If this email stops arriving, the probe on $HOST has stopped.

Last report:
$report" && echo "$now" > "$STATE/weekly"
fi
exit 0
