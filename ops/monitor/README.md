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
`openhydra-monitor-aps2`, an AWS spot instance in ap-south-2 (Hyderabad). Each probe checks every
bootstrap, so a dead netcup shows in the AWS probe's email. Mail goes out through Google Workspace
with an app password. (netcup's network blocks outgoing mail ports, so in practice the alerts come
from the AWS probe.)

The AWS probe is a one-instance spot **Auto Scaling group**, set up by
[`aws-asg.sh`](aws-asg.sh):
- It keeps one t4g.nano or t4g.micro instance, picked by AWS's price-capacity-optimized strategy:
  among the cheapest pools, the one least likely to be taken away (the lowest-price pool was taken
  away twice in 20 minutes on 28 Oct). When AWS takes the instance away anyway, the group starts
  another (measured: in service 23 s after a termination), and Capacity Rebalancing starts one as
  soon as AWS warns.
- Every instance boots from an image of a set-up probe box, so a replacement needs nothing by hand.
- AWS (SNS) emails `ALERT_TO` when the group can't launch an instance, and when it has had none in
  service for 15 minutes. Confirm the subscription once, from the email AWS sends.
- What a replacement doesn't keep: the alert state and `daily.jsonl` start again from the image's
  copy (the netcup host keeps its own daily counts).

To change the image after updating the probe box: make an image of a running instance
(`aws ec2 create-image`), then run `SOURCE_IMAGE=<new id> SOURCE_REGION=ap-south-2 ./aws-asg.sh`.
To move regions: run `./aws-asg.sh --region <new>`, check its instance, then
`./aws-asg.sh --region <old> --delete`.

What's left behind, and how to remove it:
- **Older images.** A new `SOURCE_IMAGE` leaves the previous image and its snapshot (cents a
  month). `aws-asg.sh` lists the images the group no longer boots (not in the launch template's
  latest version; never `SOURCE_IMAGE`), with the command to remove each; it never removes them
  itself.
- **After `--delete`:** the group, its alarm, the alert topic (with its email subscription) and the
  launch template go; the images and the security group stay (an image is the recovery point).
  `--delete --images` removes this group's image copies (those tagged
  `openhydra-managed-by=aws-asg`) and their snapshots too, but never `SOURCE_IMAGE`; older
  untagged copies named `openhydra-monitor` are only listed, with the command to remove them; the security group then goes with
  `aws ec2 delete-security-group --region <r> --group-name openhydra-monitor`.

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
