# Network monitoring (SL10)

Watches OpenHydra's public network from outside and emails a person when it needs one.

- **`openhydra-probe.timer`**, every 5 minutes: `openhydra-agent probe` joins the network and
  checks that every OpenHydra bootstrap answers and grants a relay reservation (a full or broken
  relay refuses one), that the signed seed list is served, and, with `EXPECT_SEEDS`, that that many
  seed models are online and answer a short test request. [`probe-run.sh`](probe-run.sh) emails:
  - **PROBLEM** when a run finds problems twice in a row (one blip is only logged), a reminder
    every 6 hours while they last, and **RECOVERED** when a run is clean again;
  - **still watching** once a week (and on the first run), with the last report: if it stops
    arriving, that probe has stopped.
- **`openhydra-count.timer`**, daily: [`count-run.sh`](count-run.sh) appends one line of totals
  (peers, models, providers the network showed it) to `/var/lib/openhydra-monitor/daily.jsonl`.
  Aggregates only, no peer ids, no prompts, and a lower bound: a peer-to-peer network can't be
  counted completely.

It runs on **two hosts**, which watch each other: the netcup bootstrap host, and
`openhydra-monitor-use1`, an AWS t4g.micro spot instance in us-east-1 (interruption behaviour
*stop*, so it resumes with its disk when capacity returns; SSH only from the operator). Each probe
checks every bootstrap, so a dead netcup shows in the AWS probe's email; a stopped AWS probe shows
as a missing weekly email. No outside service is involved: mail goes out through Google Workspace
with an app password.

The last report is kept in `/var/lib/openhydra-monitor/last.json`.

## Set up (on each host)

1. **Packages:** `apt-get install -y msmtp ca-certificates python3 curl`.
2. **Binary:** an `openhydra-agent` with the `probe` command (0.3.25 or later, or a build of `main`)
   at `/opt/openhydra-monitor/bin/openhydra-agent`; `probe-run.sh` and `count-run.sh` in
   `/opt/openhydra-monitor/`, executable.
3. **User and settings:**
   ```bash
   useradd --system --home-dir /var/lib/openhydra-monitor --shell /usr/sbin/nologin openhydra-monitor
   install -d -m 755 /etc/openhydra
   printf 'ALERT_TO=sam@openhydra.co\nEXPECT_SEEDS=0\n' > /etc/openhydra/monitor.env
   install -m 640 -g openhydra-monitor msmtprc.example /etc/openhydra/msmtprc
   ```
   Set `EXPECT_SEEDS` to the number of seed models that should be online (0 while seeds are off).
4. **Mail password (yourself, not over chat):** create an app password for sam@openhydra.co
   (Google Account › Security › App passwords; needs 2-step verification), then on the host:
   ```bash
   read -rs -p 'App password: ' P && printf '%s' "$P" > /etc/openhydra/smtp-password && unset P
   chown root:openhydra-monitor /etc/openhydra/smtp-password && chmod 640 /etc/openhydra/smtp-password
   ```
5. **Units:** copy the four `openhydra-{probe,count}.{service,timer}` files to `/etc/systemd/system/`, then
   ```bash
   systemctl daemon-reload && systemctl enable --now openhydra-probe.timer openhydra-count.timer
   ```
6. **Check:** `systemctl start openhydra-probe.service`; a "still watching" email arrives within a
   minute, and `/var/lib/openhydra-monitor/last.json` shows the report.

## By hand

```bash
openhydra-agent --identity /tmp/probe.key probe                  # alert run: JSON, exit 1 on problems
openhydra-agent --identity /tmp/probe.key probe --expect-seeds 1  # also require one seed model online
openhydra-agent --identity /tmp/probe.key probe --count-models --observe-secs 150 --no-test-request
```
