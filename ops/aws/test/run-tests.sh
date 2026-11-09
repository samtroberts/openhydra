#!/bin/bash
# Tests for launch-cheapest-spot.sh against a fake AWS CLI (fake_aws.py): nothing reaches AWS.
# usage: ops/aws/test/run-tests.sh            (macOS bash 3.2 or Linux bash)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
LAUNCH=$HERE/../launch-cheapest-spot.sh
PASS=0 FAIL=0

# setup SCENARIO_JSON: a fresh fake-AWS world in $T.
setup() {
  T=$(mktemp -d)
  mkdir -p "$T/bin" "$T/state" "$T/xdg"
  printf '#!/bin/bash\nexec python3 -I "%s" "$@"\n' "$HERE/fake_aws.py" > "$T/bin/aws"; chmod +x "$T/bin/aws"
  echo "$1" > "$T/scenario.json"
  : > "$T/calls"
}
# run ARGS...: the launcher in the fake world; output in $T/out, status in $RC.
run() {
  PATH="$T/bin:$PATH" FAKE_AWS_SCENARIO="$T/scenario.json" FAKE_AWS_LOG="$T/calls" FAKE_AWS_STATE="$T/state" \
    MYIP=198.51.100.7 TOKEN_WAITS="0 0 0" XDG_STATE_HOME="$T/xdg" AWS_PROFILE=test \
    python3 -I -c 'import os, signal, sys; signal.signal(signal.SIGINT, signal.SIG_DFL); os.execv(sys.argv[1], sys.argv[1:])' \
    "$LAUNCH" "$@" > "$T/out" 2>&1
  # (Through python, which puts Ctrl-C back to its default: a background job starts with it ignored,
  # and bash can't trap a signal it started with ignored, unlike a run from a terminal.)
  RC=$?
}
ok() { PASS=$((PASS + 1)); echo "ok   $1"; }
bad() { FAIL=$((FAIL + 1)); echo "FAIL $1"; sed 's/^/     | /' "$T/out" | tail -15; }
has() { grep -q -- "$1" "$T/out"; }
called() { grep -qE -- "$1" "$T/calls"; }
# The calls that change something in AWS.
mutated() { grep -E '^ec2 (import-key-pair|copy-image|authorize-|revoke-|delete-|deregister-image)' "$T/calls" | grep -v generate-cli-skeleton
            grep -E '^ec2 create-security-group' "$T/calls" | grep -v -- --dry-run
            grep -E '^ec2 run-instances' "$T/calls" | grep -v -- --dry-run; }

BASE='"regions":["r1","r2"],"prices":{"r1":[["0.0040","r1a","t4g.nano"]],"r2":[["0.0050","r2a","t4g.nano"]]}'
STD=(--name test-box --types t4g.nano --ami-id ami-src --ami-region r0)

# --- --check changes nothing, even where a launch would create and copy everything.
setup "{$BASE,\"sg\":null,\"key_exists\":false}"
run "${STD[@]}" --pubkey "$HERE/fake_aws.py" --check
if [ $RC = 0 ] && has "CHECK PASSED" && has "would create security group" && has "would copy image" \
   && has "would import the key pair" && [ -z "$(mutated)" ]; then ok "--check changes nothing"; else bad "--check changes nothing"; mutated; fi

# --- --check on an existing group with an unwanted rule: reported, not revoked.
setup "{$BASE,\"sg\":\"sg-old\",\"sg_managed\":\"launch-cheapest-spot\",\"sg_rules\":[[\"sgr-1\",\"tcp\",\"22\",\"22\",\"203.0.113.1/32\",\"None\"]],\"existing_copy\":\"ami-copy\"}"
run "${STD[@]}" --check
if [ $RC = 0 ] && has "would remove the rule tcp 22 22 203.0.113.1/32" && has "would add the rule tcp 22 22 198.51.100.7/32" \
   && [ -z "$(mutated)" ]; then ok "--check reports rule changes without making them"; else bad "--check reports rule changes without making them"; fi

# --- A group an instance uses only gains rules.
setup "{$BASE,\"sg\":\"sg-live\",\"sg_managed\":\"launch-cheapest-spot\",\"sg_in_use\":1,\"sg_rules\":[[\"sgr-9\",\"udp\",\"4001\",\"4001\",\"0.0.0.0/0\",\"None\"]],\"existing_copy\":\"ami-copy\"}"
run "${STD[@]}"
if [ $RC = 0 ] && has LAUNCHED && ! called revoke && called authorize && has "in use by 1"; then ok "a group in use is never revoked"; else bad "a group in use is never revoked"; fi

# --- A free group this script made is kept to exactly its rules.
setup "{$BASE,\"sg\":\"sg-free\",\"sg_managed\":\"launch-cheapest-spot\",\"sg_in_use\":0,\"sg_rules\":[[\"sgr-9\",\"tcp\",\"22\",\"22\",\"203.0.113.1/32\",\"None\"]],\"existing_copy\":\"ami-copy\"}"
run "${STD[@]}"
if [ $RC = 0 ] && called "revoke-security-group-ingress.*sgr-9"; then ok "a free managed group is reconciled"; else bad "a free managed group is reconciled"; fi

# --- A group someone else made (aws-asg.sh's, say) only gains rules.
setup "{$BASE,\"sg\":\"sg-asg\",\"sg_managed\":\"aws-asg\",\"sg_rules\":[[\"sgr-9\",\"tcp\",\"22\",\"22\",\"203.0.113.1/32\",\"None\"]],\"existing_copy\":\"ami-copy\"}"
run "${STD[@]}"
if [ $RC = 0 ] && ! called revoke && has "wasn't made by this script"; then ok "another script's group is left alone"; else bad "another script's group is left alone"; fi

# --- One already running: no second unless asked.
setup "{$BASE,\"existing\":{\"r2\":[[\"i-running\",\"t4g.nano\",\"running\"]]},\"existing_copy\":\"ami-copy\"}"
run "${STD[@]}"
if [ $RC = 2 ] && has "already running" && has i-running && ! grep -q '^ec2 run-instances' "$T/calls"; then ok "a duplicate isn't launched"; else bad "a duplicate isn't launched"; fi
run "${STD[@]}" --allow-another
if [ $RC = 0 ] && has LAUNCHED; then ok "--allow-another launches anyway"; else bad "--allow-another launches anyway"; fi

# --- Volumes: the image's own count.
setup "{$BASE,\"existing_copy\":\"ami-copy\",\"bdm\":[[\"/dev/xvda\",\"8\",\"True\"],[\"/dev/sdb\",\"2000\",\"True\"]]}"
run "${STD[@]}"
if [ $RC = 2 ] && has "2008 GB of volumes" && ! grep -q '^ec2 run-instances' "$T/calls"; then ok "a large image needs --large-disk"; else bad "a large image needs --large-disk"; fi
setup "{$BASE,\"existing_copy\":\"ami-copy\",\"bdm\":[[\"/dev/xvda\",\"8\",\"True\"],[\"/dev/sdb\",\"50\",\"False\"]]}"
run "${STD[@]}"
if [ $RC = 2 ] && has "outlives the instance"; then ok "a volume that outlives the instance is refused"; else bad "a volume that outlives the instance is refused"; fi
run "${STD[@]}" --keep-volumes
if [ $RC = 0 ] && has LAUNCHED; then ok "--keep-volumes allows it"; else bad "--keep-volumes allows it"; fi

# --- Ctrl-C and a closed terminal while the launch call runs: it finishes, and the result is known.
for sig in INT HUP; do
  setup "{$BASE,\"existing_copy\":\"ami-copy\",\"launch_sleep\":3}"
  ( run "${STD[@]}"; echo "$RC" > "$T/rc" ) &
  waiter=$!
  for _ in $(seq 50); do grep -q '^ec2 run-instances.*--client-token' "$T/calls" 2>/dev/null && break; sleep 0.1; done
  sleep 0.5
  pkill -"$sig" -f "[l]aunch-cheapest-spot.sh --name test-box" 2>/dev/null
  wait "$waiter"; RC=$(cat "$T/rc")
  logf=$(ls "$T"/xdg/openhydra-launch/*.log 2>/dev/null | head -1)
  if [ "$RC" = 130 ] && has "LAUNCHED i-fake" && [ -n "$logf" ] && grep -q "LAUNCHED i-fake" "$logf"; then
    ok "SIG$sig during the launch call: launched instance reported (and logged)"
  else bad "SIG$sig during the launch call: launched instance reported (and logged)"; fi
done

# --- An error after EC2 took the request: found by its token, on the 3rd look.
setup "{$BASE,\"existing_copy\":\"ami-copy\",\"launch\":\"InternalError\",\"token_found\":\"i-late\",\"token_found_after\":3}"
run "${STD[@]}"
if [ $RC = 0 ] && has "LAUNCHED i-late" && has "reported an error, but the instance exists"; then ok "an instance behind an error is found by token"; else bad "an instance behind an error is found by token"; fi

# --- An error and no instance found: stop, keep that region's copy, say how to check.
setup "{$BASE,\"launch\":\"InternalError\"}"
run "${STD[@]}"
if [ $RC = 2 ] && has "may have started an instance" && has "client-token" && ! called "deregister-image --region r1"; then
  ok "an unknown outcome keeps the region's resources"; else bad "an unknown outcome keeps the region's resources"; fi

# --- Prices sort by number in any locale.
setup "{\"regions\":[\"r1\"],\"prices\":{\"r1\":[[\"0.0100\",\"r1a\",\"t4g.nano\"],[\"0.0031\",\"r1b\",\"t4g.nano\"],[\"0.2\",\"r1c\",\"t4g.nano\"]]},\"existing_copy\":\"ami-copy\"}"
LANG=de_DE.UTF-8 LC_NUMERIC=de_DE.UTF-8 run "${STD[@]}"
if [ $RC = 0 ] && grep -m1 "^trying" "$T/out" | grep -q r1b; then ok "cheapest first under a German locale"; else bad "cheapest first under a German locale"; fi

# --- A failed copy rules the region out (one copy attempt, not one per zone).
setup "{\"regions\":[\"r1\",\"r2\"],\"prices\":{\"r1\":[[\"0.001\",\"r1a\",\"t4g.nano\"],[\"0.002\",\"r1b\",\"t4g.nano\"]],\"r2\":[[\"0.003\",\"r2a\",\"t4g.nano\"]]},\"copy_fails\":true}"
run "${STD[@]}"
n=$(grep -c '^ec2 copy-image --region r1' "$T/calls")
if [ "$n" = 1 ] && has "skipping r1 from now on: no usable image copy"; then ok "a failed copy skips its region"; else bad "a failed copy skips its region (copies in r1: $n)"; fi

# --- --ipv6 where a subnet has none: skipped, not fatal.
setup "{$BASE,\"existing_copy\":\"ami-copy\",\"subnet_v6\":0}"
run "${STD[@]}" --ipv6
if [ $RC = 1 ] && has "no IPv6 in the default subnet" && has "NOTHING LAUNCHED"; then ok "--ipv6 skips subnets without IPv6"; else bad "--ipv6 skips subnets without IPv6"; fi

# --- An older CLI: the copy's snapshots are deleted after the image.
setup "{$BASE,\"cli_new\":false,\"launch\":\"InsufficientInstanceCapacity\",\"prices\":{\"r1\":[[\"0.004\",\"r1a\",\"t4g.nano\"]]},\"regions\":[\"r1\"]}"
run "${STD[@]}"
if called "deregister-image --region r1 --image-id ami-copy$" && called "delete-snapshot --region r1 --snapshot-id snap-1"; then
  ok "older CLI: image and snapshots removed one by one"; else bad "older CLI: image and snapshots removed one by one"; fi

# --- Names with odd characters are refused before any AWS call.
setup "{$BASE}"
run "${STD[@]}" --purpose 'x,y=z'
if [ $RC = 2 ] && has "may only use" && [ ! -s "$T/calls" ]; then ok "an unsafe --purpose is refused up front"; else bad "an unsafe --purpose is refused up front"; fi

# --- --instance-profile reaches the launch request.
setup "{$BASE,\"existing_copy\":\"ami-copy\"}"
run "${STD[@]}" --instance-profile openhydra-seed-test
if [ $RC = 0 ] && called "run-instances .*--iam-instance-profile Name=openhydra-seed-test .*--client-token"; then ok "--instance-profile is attached"; else bad "--instance-profile is attached"; fi

# --- A closed output pipe (`| tee`, then Ctrl-C) doesn't stop the clean-up.
setup "{\"regions\":[\"r1\",\"r2\"],\"prices\":{\"r1\":[[\"0.001\",\"r1a\",\"t4g.nano\"]],\"r2\":[[\"0.002\",\"r2a\",\"t4g.nano\"]]},\"launch\":\"InsufficientInstanceCapacity\"}"
PATH="$T/bin:$PATH" FAKE_AWS_SCENARIO="$T/scenario.json" FAKE_AWS_LOG="$T/calls" FAKE_AWS_STATE="$T/state" \
  MYIP=198.51.100.7 TOKEN_WAITS="0 0 0" XDG_STATE_HOME="$T/xdg" AWS_PROFILE=test \
  "$LAUNCH" "${STD[@]}" 2>&1 | head -c 1 > /dev/null
n=$(grep -c '^ec2 deregister-image --region' "$T/calls")
if [ "$n" = 2 ]; then ok "a closed output pipe doesn't cut the clean-up short"; else bad "a closed output pipe doesn't cut the clean-up short (copies removed: $n of 2)"; fi

# --- A snapshot the CLI didn't delete is reported, not passed over.
setup "{$BASE,\"launch\":\"InsufficientInstanceCapacity\",\"snapshot_skipped\":true,\"prices\":{\"r1\":[[\"0.004\",\"r1a\",\"t4g.nano\"]]},\"regions\":[\"r1\"]}"
run "${STD[@]}"
if has "COULDN'T remove the image copy ami-copy" && has "not its snapshot"; then ok "a skipped snapshot is reported"; else bad "a skipped snapshot is reported"; fi

# --- A duplicate check that fails stops the run (it isn't "nothing running").
setup "{$BASE,\"existing_copy\":\"ami-copy\",\"existing_fail\":[\"r2\"]}"
run "${STD[@]}"
if [ $RC = 2 ] && has "couldn't check for a running test-box in r2" && ! grep -q '^ec2 run-instances.*--client-token' "$T/calls"; then
  ok "a failed duplicate check stops the run"; else bad "a failed duplicate check stops the run"; fi

# --- --check with no copy yet checks your image's volumes, not the stand-in's.
setup "{$BASE,\"source_bdm\":[[\"/dev/xvda\",\"8\",\"True\"],[\"/dev/sdb\",\"50\",\"False\"]]}"
run "${STD[@]}" --check
if [ $RC = 2 ] && has "outlives the instance" && ! has "CHECK PASSED"; then ok "--check checks the source image's volumes"; else bad "--check checks the source image's volumes"; fi

# --- An image whose volumes are refused is refused before it's copied anywhere (a copy is billed).
setup "{$BASE,\"source_bdm\":[[\"/dev/xvda\",\"8\",\"True\"],[\"/dev/sdb\",\"50\",\"False\"]]}"
run "${STD[@]}"
if [ $RC = 2 ] && has "outlives the instance" && ! called '^ec2 copy-image'; then ok "volumes are checked before a copy"; else bad "volumes are checked before a copy"; fi

echo "passed $PASS, failed $FAIL"
[ "$FAIL" = 0 ]
