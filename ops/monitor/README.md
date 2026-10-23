# Network monitoring (SL10)

Watches OpenHydra's public network from outside and alerts a person when it needs one.

- **`openhydra-probe.timer`**, every 5 minutes: `openhydra-agent probe` joins the network and
  checks that every OpenHydra bootstrap answers and grants a relay reservation (a full or broken
  relay refuses one), that the signed seed list is served, and, with `EXPECT_SEEDS`, that that many
  seed models are online and answer a short test request. [`probe-run.sh`](probe-run.sh) reports to
  [Healthchecks](https://healthchecks.io), which emails:
  - when a run finds problems **twice in a row** (one blip is only logged);
  - when the runs **stop** (the probe box itself is down).
- **`openhydra-count.timer`**, daily: [`count-run.sh`](count-run.sh) appends one line of totals
  (peers, models, providers the network showed it) to `/var/lib/openhydra-monitor/daily.jsonl`.
  Aggregates only, no peer ids, no prompts, and a lower bound: a peer-to-peer network can't be
  counted completely.

The last report is kept in `/var/lib/openhydra-monitor/last.json`.

## Set up

On any Linux host with outbound internet (it opens no ports):

1. **Healthchecks:** create a check with period 5 minutes and grace 10 minutes, set its email, and
   copy its ping URL (`https://hc-ping.com/<uuid>`).
2. **Binary:** an `openhydra-agent` with the `probe` command (0.3.25 or later, or a build of `main`)
   at `/opt/openhydra-monitor/bin/openhydra-agent`; the two scripts beside `bin/`, executable.
3. **User and settings:**
   ```bash
   useradd --system --home-dir /var/lib/openhydra-monitor --shell /usr/sbin/nologin openhydra-monitor
   install -d -m 755 /etc/openhydra
   printf 'HC_URL=https://hc-ping.com/<uuid>\nEXPECT_SEEDS=0\n' > /etc/openhydra/monitor.env
   chmod 640 /etc/openhydra/monitor.env && chgrp openhydra-monitor /etc/openhydra/monitor.env
   ```
   Set `EXPECT_SEEDS` to the number of seed models that should be online (0 while seeds are off).
4. **Units:** copy the four `openhydra-{probe,count}.{service,timer}` files to `/etc/systemd/system/`, then
   ```bash
   systemctl daemon-reload && systemctl enable --now openhydra-probe.timer openhydra-count.timer
   ```
5. **Check:** `systemctl start openhydra-probe.service && cat /var/lib/openhydra-monitor/last.json`,
   and the check turns green in Healthchecks.

## By hand

```bash
openhydra-agent --identity /tmp/probe.key probe                  # alert run: JSON, exit 1 on problems
openhydra-agent --identity /tmp/probe.key probe --expect-seeds 1  # also require one seed model online
openhydra-agent --identity /tmp/probe.key probe --count-models --observe-secs 150 --no-test-request
```
