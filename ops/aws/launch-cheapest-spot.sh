#!/bin/bash
# Launch one spot instance in the cheapest place that has capacity.
#
# Looks up the current spot price of every instance type given, in every zone of every region,
# sorts them cheapest first, and tries each in turn until one launches. (Rule: always the cheapest
# first; on no capacity or an interruption, the next cheapest. After an interruption, run it again,
# or use an Auto Scaling group, as the network probe does: see ops/monitor/aws-asg.sh.)
#
# What moves it on, and what stops it:
#   - no capacity in a zone, or the type isn't offered there   -> the next candidate
#   - a region this account may not use, a regional quota, or no usable image or subnet there
#                                                               -> every other candidate in that region is skipped
#   - anything else wrong with the request (user data, disk, key, an unknown error) -> it stops at
#     once, so a mistake that would fail everywhere isn't tried everywhere. Each region gets a full
#     `--dry-run` of the real request before anything launches there.
# Before launching anything it looks for a running instance with the same name in every candidate
# region, and stops if there is one (--allow-another to launch a second anyway): running it again
# after a dropped terminal doesn't start a duplicate.
#
# Whatever it created and didn't use (image copies, security groups) is removed when it exits,
# including on an error or Ctrl-C, and anything it couldn't remove is printed with the command to
# remove it. Nothing is removed in a region where an instance may have started. A Ctrl-C or a
# closed terminal while the launch request is in flight takes effect once AWS has answered, so the
# script always knows whether an instance started. Every result line also goes to a log file (its
# path is printed first), which survives a closed terminal.
#
# The instance is one-time spot, terminated on interruption or on `shutdown -h`, with IMDSv2 only
# (hop limit 1, no IPv6 metadata endpoint). Its security group allows SSH only from this machine's
# public IPv4, plus the --open ports from anywhere. A group this script created (tagged
# openhydra-managed-by=launch-cheapest-spot) is kept to exactly those rules, so SSH rules for old
# addresses don't pile up, but only while no instance uses it: a group in use, or one someone else
# made, only gets the missing rules added, so a run never takes a port away from a live instance.
#
# usage: launch-cheapest-spot.sh --name NAME --types "TYPE ..." AMI [options]
#   AMI is one of:
#     --ami-id ID --ami-region REGION   an image of yours; copied (once) to the region it launches in
#     --ami-name PATTERN [--ami-owner OWNER]   the newest matching image for the type's architecture
#                                              in each region (default owner amazon)
#   --purpose TAG         value of the `purpose` tag (default: the name)
#   --user-data FILE      first-boot script (at most 16 KB)
#   --disk GB             root volume size (default: the image's)
#   --large-disk          allow more than 1000 GB of volumes in all (the image's own volumes count),
#                         so a typo can't launch an expensive volume
#   --keep-volumes        allow an image with a volume that outlives the instance
#                         (DeleteOnTermination=false); refused by default, as it orphans a billing
#                         volume after a spot interruption
#   --sg-name NAME        security group to use or create (default: the name)
#   --open PROTO:PORT     also open this port to everyone, IPv4 and IPv6 (repeatable), e.g. udp:4001
#   --ipv6                give the instance an IPv6 address (zones whose subnet has none are skipped)
#   --key NAME            key pair (default openhydra-seed); --pubkey FILE imports it where missing
#   --instance-profile NAME   attach this IAM instance profile (a seed reading its key from SSM:
#                         see ops/seeds/render.sh)
#   --regions "R ..."     regions to consider (default: every region enabled for the account)
#   --allow-another       launch even if an instance with this name is already running
#   --dry-run             only print the candidates, cheapest first
#   --check               change nothing: go through every step up to a dry run of the real request
#                         in the cheapest usable zone, reporting what a launch would create, copy or
#                         change, then stop (for testing a request)
set -uo pipefail
# A closed output pipe (`… | tee log`, then Ctrl-C) must not kill the script mid clean-up: writes to
# it just fail (review 2026-10-29: SIGPIPE stopped the clean-up and left billable image copies).
trap '' PIPE
export AWS_PROFILE=${AWS_PROFILE:-openhydra}
export LC_ALL=C   # prices sort by number whatever the locale's decimal separator
MANAGED_TAG=openhydra-managed-by
MANAGED_BY=launch-cheapest-spot

NAME= TYPES= AMI_ID= AMI_REGION= AMI_NAME= AMI_OWNER=amazon PURPOSE= USERDATA= DISK= SG_NAME=
KEY=openhydra-seed PUBKEY= PROFILE= REGIONS= IPV6=0 DRY=0 CHECK=0 LARGE_DISK=0 KEEP_VOLUMES=0 ALLOW_ANOTHER=0 OPEN=()
die() { say "launch: $*" >&2; exit 2; }
# Every result line goes to the terminal and the log (see LOG below); a closed terminal only loses
# the first.
LOG=
say() {
  [ -z "$LOG" ] || printf '%s\n' "$*" >> "$LOG" 2>/dev/null
  printf '%s\n' "$*" 2>/dev/null || true
}
while [ $# -gt 0 ]; do
  [ $# -ge 2 ] || case "$1" in --ipv6|--dry-run|--check|--large-disk|--keep-volumes|--allow-another) ;; *) die "$1 needs a value";; esac
  case "$1" in
    --name) NAME=$2; shift 2;;            --types) TYPES=$2; shift 2;;
    --ami-id) AMI_ID=$2; shift 2;;        --ami-region) AMI_REGION=$2; shift 2;;
    --ami-name) AMI_NAME=$2; shift 2;;    --ami-owner) AMI_OWNER=$2; shift 2;;
    --purpose) PURPOSE=$2; shift 2;;      --user-data) USERDATA=$2; shift 2;;
    --disk) DISK=$2; shift 2;;            --sg-name) SG_NAME=$2; shift 2;;
    --open) OPEN+=("$2"); shift 2;;       --ipv6) IPV6=1; shift;;
    --key) KEY=$2; shift 2;;              --pubkey) PUBKEY=$2; shift 2;;
    --instance-profile) PROFILE=$2; shift 2;;
    --regions) REGIONS=$2; shift 2;;      --dry-run) DRY=1; shift;;
    --check) CHECK=1; shift;;             --large-disk) LARGE_DISK=1; shift;;
    --keep-volumes) KEEP_VOLUMES=1; shift;;  --allow-another) ALLOW_ANOTHER=1; shift;;
    *) die "unknown option $1 (see the header of $0)";;
  esac
done

# --- Checks that need no AWS call: a mistake here would otherwise fail in every zone.
[ -n "$NAME" ] && [ -n "$TYPES" ] || die "--name and --types are required"
PURPOSE=${PURPOSE:-$NAME}; SG_NAME=${SG_NAME:-$NAME}
# Names go into tag and filter strings unquoted, so they're kept to safe characters.
for pair in "name:$NAME" "purpose:$PURPOSE" "sg-name:$SG_NAME" "key:$KEY" ${PROFILE:+"instance-profile:$PROFILE"}; do
  case "${pair#*:}" in ''|*[!A-Za-z0-9._-]*) die "--${pair%%:*} may only use letters, digits, dots, dashes and underscores";; esac
done
case "$TYPES" in *[!a-z0-9.\ -]*) die "--types takes instance type names separated by spaces";; esac
if [ -n "$AMI_NAME" ]; then
  [ -z "$AMI_ID" ] || die "give --ami-name or --ami-id, not both"
else
  [ -n "$AMI_ID" ] && [ -n "$AMI_REGION" ] || die "give --ami-name, or --ami-id with --ami-region"
fi
if [ -n "$USERDATA" ]; then
  [ -r "$USERDATA" ] || die "can't read $USERDATA"
  [ "$(wc -c < "$USERDATA")" -le 16384 ] || die "$USERDATA is over EC2's 16 KB user-data limit"
fi
[ -z "$PUBKEY" ] || [ -r "$PUBKEY" ] || die "can't read $PUBKEY"
[ -z "$DISK" ] || case "$DISK" in ''|*[!0-9]*|0*) die "--disk takes a whole number of GB";; esac
[ -z "$DISK" ] || [ "$DISK" -le 1000 ] || [ "$LARGE_DISK" = 1 ] || die "--disk $DISK is over 1000 GB; add --large-disk if you mean it"
for o in ${OPEN[@]+"${OPEN[@]}"}; do
  p=${o%%:*}; n=${o#*:}
  case "$p:$n" in tcp:*|udp:*) ;; *) die "--open takes tcp:PORT or udp:PORT, not $o";; esac
  case "$n" in ''|*[!0-9]*|0*) die "--open: $n isn't a port number";; esac
  [ "$n" -le 65535 ] || die "--open: $n isn't a port number"
done

RUN_ID=$(date -u +%Y%m%d%H%M%S)-$$
LOG_DIR=${XDG_STATE_HOME:-$HOME/.local/state}/openhydra-launch
if mkdir -p "$LOG_DIR" 2>/dev/null && chmod 700 "$LOG_DIR" 2>/dev/null && : > "$LOG_DIR/$RUN_ID.log" 2>/dev/null; then
  chmod 600 "$LOG_DIR/$RUN_ID.log"; LOG=$LOG_DIR/$RUN_ID.log
  say "log: $LOG"
else
  say "  note: couldn't create a log in $LOG_DIR; results go to the terminal only" >&2
fi
say "run $RUN_ID: --name $NAME --types \"$TYPES\" ${AMI_ID:+--ami-id $AMI_ID --ami-region $AMI_REGION}${AMI_NAME:+--ami-name $AMI_NAME}$([ "$CHECK" = 1 ] && echo ' --check')"

aws sts get-caller-identity >/dev/null 2>&1 || die "not signed in to AWS (profile $AWS_PROFILE)"
[ -n "$REGIONS" ] || REGIONS=$(aws ec2 describe-regions --region us-east-1 --query 'Regions[].RegionName' --output text) \
  || die "couldn't list the regions"

WORK=$(mktemp -d)
LAUNCHED_REGION= LAUNCHED_SG=
: > "$WORK/copies"; : > "$WORK/groups"

# The AWS error code in a CLI error message, e.g. InsufficientInstanceCapacity.
err_code() { grep -oE '\([A-Za-z0-9.]+\) when calling' "$1" | head -1 | sed -E 's/^\(([^)]*)\).*/\1/'; }

# Whether this AWS CLI can delete an image's snapshots with it (added in 2025); otherwise they're
# deleted one by one after the image. Asked of the CLI itself, offline.
if aws ec2 deregister-image --generate-cli-skeleton 2>/dev/null | grep -q DeleteAssociatedSnapshots; then
  DEREG_SNAPSHOTS=1
else
  DEREG_SNAPSHOTS=0
fi

# Remove image $2 in region $1 with its snapshots. Prints nothing on success; the error otherwise.
remove_image() {
  local r=$1 id=$2 snaps out
  if [ "$DEREG_SNAPSHOTS" = 1 ]; then
    # It succeeds even when a snapshot isn't deleted; each one's outcome is in the response.
    out=$(aws ec2 deregister-image --region "$r" --image-id "$id" --delete-associated-snapshots \
      --query 'DeleteSnapshotResults[?ReturnCode!=`success`].[SnapshotId,ReturnCode]' --output text 2>&1) || { echo "$out"; return 1; }
    if [ -n "$out" ] && [ "$out" != None ]; then
      echo "image removed, but not its snapshot(s): $(echo "$out" | tr '\t\n' ': ') (remove with: aws ec2 delete-snapshot --region $r --snapshot-id <id>)"
      return 1
    fi
    return 0
  fi
  snaps=$(aws ec2 describe-images --region "$r" --image-ids "$id" \
    --query 'Images[0].BlockDeviceMappings[].Ebs.SnapshotId' --output text 2>&1) || { echo "$snaps"; return 1; }
  out=$(aws ec2 deregister-image --region "$r" --image-id "$id" 2>&1) || { echo "$out"; return 1; }
  for s in $snaps; do
    case "$s" in snap-*) out=$(aws ec2 delete-snapshot --region "$r" --snapshot-id "$s" 2>&1) || { echo "$out"; return 1; };; esac
  done
}
remove_image_cmd() {
  if [ "$DEREG_SNAPSHOTS" = 1 ]; then
    echo "aws ec2 deregister-image --region $1 --image-id $2 --delete-associated-snapshots"
  else
    echo "aws ec2 describe-images --region $1 --image-ids $2 --query 'Images[0].BlockDeviceMappings[].Ebs.SnapshotId' (note them), then aws ec2 deregister-image --region $1 --image-id $2, then aws ec2 delete-snapshot --region $1 --snapshot-id <each>"
  fi
}

# Look up the instance launched with token $2 in region $1, patiently (describe calls are only
# eventually consistent): about a minute in all. Prints its id, or nothing.
find_by_token() {
  local r=$1 token=$2 id w
  for w in ${TOKEN_WAITS:-0 2 4 8 16 30}; do   # (a test may shorten them)
    sleep "$w"
    id=$(aws ec2 describe-instances --region "$r" --filters "Name=client-token,Values=$token" \
      --query 'Reservations[].Instances[0].InstanceId | [0]' --output text 2>/dev/null)
    case "$id" in i-*) echo "$id"; return 0;; esac
  done
  return 1
}

# Remove what this run created and didn't end up using. Runs on every exit.
cleanup() {
  local r id g out token maybe=
  trap '' INT TERM HUP   # finish the clean-up, whatever happens to the terminal
  # A launch request that was in flight when the script was stopped: it may have started an
  # instance. Find out; either way, nothing in that region is removed.
  # An image copy requested when the script was stopped, before its id came back: find it by name.
  if [ -s "$WORK/copying" ]; then
    read -r r name < "$WORK/copying"
    id=$(aws ec2 describe-images --region "$r" --owners self --filters "Name=name,Values=$name" \
      --query 'Images[0].ImageId' --output text 2>/dev/null)
    case "$id" in ami-*) grep -q " $id\$" "$WORK/copies" || echo "$r $id" >> "$WORK/copies";;
      *) say "  an image copy ($name) may have been started in $r; check: aws ec2 describe-images --region $r --owners self --filters Name=name,Values=$name" >&2;; esac
  fi
  if [ -s "$WORK/inflight" ] && [ -z "$LAUNCHED_REGION" ]; then
    read -r r token < "$WORK/inflight"
    maybe=$r
    if id=$(find_by_token "$r" "$token"); then
      say "LAUNCHED $id in $r (the request went through before the script stopped)"
      say "  if you don't want it: aws ec2 terminate-instances --region $r --instance-ids $id"
    else
      say "  the launch request in $r may have started an instance; check with:" >&2
      say "    aws ec2 describe-instances --region $r --filters Name=client-token,Values=$token" >&2
    fi
  fi
  while read -r r id; do
    [ -n "$r" ] || continue
    [ "$r" = "$LAUNCHED_REGION" ] || [ "$r" = "$maybe" ] && continue
    if out=$(remove_image "$r" "$id"); then
      say "  removed the unused image copy $id in $r"
    else
      say "  COULDN'T remove the image copy $id in $r ($(echo "$out" | tail -1 | cut -c1-120))." >&2
      say "    Remove it with: $(remove_image_cmd "$r" "$id")" >&2
    fi
  done < "$WORK/copies"
  while read -r r g; do
    [ -n "$r" ] || continue
    [ "$r $g" = "$LAUNCHED_REGION $LAUNCHED_SG" ] || [ "$r" = "$maybe" ] && continue
    if out=$(aws ec2 delete-security-group --region "$r" --group-id "$g" 2>&1); then
      say "  removed the unused security group $g in $r"
    else
      say "  COULDN'T remove the security group $g in $r ($(echo "$out" | tail -1 | cut -c1-120))." >&2
      say "    Remove it with: aws ec2 delete-security-group --region $r --group-id $g" >&2
    fi
  done < "$WORK/groups"
  rm -rf "$WORK"
}
trap cleanup EXIT
on_signal() { say "launch: interrupted" >&2; exit 130; }
trap on_signal INT TERM HUP

# --- Current prices: one line "price region zone type" per pool, cheapest first.
for r in $REGIONS; do
  aws ec2 describe-spot-price-history --region "$r" --instance-types $TYPES --product-descriptions Linux/UNIX \
    --start-time "$(date -u +%Y-%m-%dT%H:%M:%S)" --query 'SpotPriceHistory[].[SpotPrice,AvailabilityZone,InstanceType]' \
    --output text 2>"$WORK/perr-$r" | awk -v r="$r" 'NF==3 {print $1, r, $2, $3}' > "$WORK/price-$r" &
done
wait
NOPERM=
for r in $REGIONS; do
  [ -s "$WORK/perr-$r" ] || continue
  case "$(err_code "$WORK/perr-$r")" in
    UnauthorizedOperation|AccessDenied*|AuthFailure|OptInRequired) NOPERM="$NOPERM $r";;
    *) say "  note: couldn't read prices in $r, so it isn't a candidate: $(tail -1 "$WORK/perr-$r" | cut -c1-140)" >&2;;
  esac
done
[ -z "$NOPERM" ] || say "  not usable by this account:$NOPERM"
cat "$WORK"/price-* 2>/dev/null | sort -g -k1,1 > "$WORK/candidates"
[ -s "$WORK/candidates" ] || die "no spot prices found for: $TYPES"
say "candidates, cheapest first (USD/hour):"; head -15 "$WORK/candidates" | while read -r l; do say "  $l"; done
[ "$DRY" = 1 ] && exit 0

# --- One of this name already running? Then a launch would be a second one (a rerun after a lost
# terminal, say). Looked up by the Name tag, which every version of this script set.
if [ "$ALLOW_ANOTHER" = 0 ]; then
  for r in $(awk '{print $2}' "$WORK/candidates" | sort -u); do
    { aws ec2 describe-instances --region "$r" --filters "Name=tag:Name,Values=$NAME" \
        Name=instance-state-name,Values=pending,running \
        --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name]' --output text 2>"$WORK/existerr-$r" \
        || echo failed > "$WORK/existfail-$r"; } | awk -v r="$r" 'NF {print r, $0}' > "$WORK/existing-$r" &
  done
  wait
  # A lookup that failed (throttled, denied) isn't "nothing running".
  for f in "$WORK"/existfail-*; do
    [ -e "$f" ] || continue
    r=${f##*/existfail-}
    die "couldn't check for a running $NAME in $r ($(tail -1 "$WORK/existerr-$r" | cut -c1-120)); add --allow-another to launch without checking"
  done
  if cat "$WORK"/existing-* 2>/dev/null | grep -q .; then
    say "an instance named $NAME is already running:"
    cat "$WORK"/existing-* | while read -r l; do say "  $l"; done
    die "not launching another (add --allow-another if you mean to)"
  fi
fi

# A dotted-quad IPv4 address, each part 0-255, not 0.x.x.x: the only thing the SSH rule may be built
# from (it's put into --ip-permissions as is; anything else, a captive portal's page or an injected
# value, could widen it, review 2026-10-29).
valid_ipv4() {
  case "$1" in ""|*[!0-9.]*|.*|*.|*..*) return 1;; esac
  local IFS=. part n=0
  for part in $1; do
    n=$((n + 1))
    [ "${#part}" -le 3 ] && [ "$((10#$part))" -le 255 ] || return 1
  done
  [ "$n" -eq 4 ] && [ "$((10#${1%%.*}))" -ne 0 ]
}
# (MYIP may be set beforehand, e.g. by a test.)
MYIP=${MYIP:-$(curl -fsS -4 --max-time 10 https://checkip.amazonaws.com | tr -d '\n')} || true
valid_ipv4 "$MYIP" || die "couldn't find this machine's public IPv4 (for the SSH rule); got '${MYIP:0:40}'"
SOURCE_ARCH=
if [ -n "$AMI_ID" ]; then
  read -r SOURCE_ARCH SOURCE_STATE < <(aws ec2 describe-images --region "$AMI_REGION" --image-ids "$AMI_ID" \
    --query 'Images[0].[Architecture,State]' --output text 2>/dev/null) || true
  [ "${SOURCE_STATE:-}" = available ] || die "image $AMI_ID in $AMI_REGION isn't there or isn't available"
fi

# --- Per-region state lives in files, as bash 3.2 has no associative arrays.
#   $WORK/skip-R   why region R is out (absent: usable)       $WORK/sg-R  its security group
skip_region() { echo "$2" > "$WORK/skip-$1"; say "  skipping $1 from now on: $2"; }
# In --check, what a launch would do instead of doing it.
would() { say "  (check) would $*"; }

# The security group for region $1, VPC $2: found or made, with its rules put right. Prints its id.
# In --check it changes nothing: it reports, and prints the group to dry-run with.
ensure_group() {
  local r=$1 vpc=$2 sg managed rule id key p f t c4 c6 o out in_use
  sg=$(aws ec2 describe-security-groups --region "$r" --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$vpc" \
    --query 'SecurityGroups[0].GroupId' --output text) || return 1
  if [ "$sg" = None ] || [ -z "$sg" ]; then
    if [ "$CHECK" = 1 ]; then
      would "create security group $SG_NAME in $r (SSH from $MYIP${OPEN[*]+, ${OPEN[*]}}); dry-running with the VPC's default group" >&2
      sg=$(aws ec2 describe-security-groups --region "$r" --filters Name=group-name,Values=default "Name=vpc-id,Values=$vpc" \
        --query 'SecurityGroups[0].GroupId' --output text) || return 1
      echo "$sg"; return 0
    fi
    sg=$(aws ec2 create-security-group --region "$r" --group-name "$SG_NAME" --description "$NAME (launch-cheapest-spot)" \
      --vpc-id "$vpc" --query GroupId --output text \
      --tag-specifications "ResourceType=security-group,Tags=[{Key=Name,Value=$SG_NAME},{Key=purpose,Value=$PURPOSE},{Key=$MANAGED_TAG,Value=$MANAGED_BY}]") || return 1
    echo "$r $sg" >> "$WORK/groups"
  fi
  managed=$(aws ec2 describe-security-groups --region "$r" --group-ids "$sg" \
    --query "SecurityGroups[0].Tags[?Key=='$MANAGED_TAG'].Value | [0]" --output text) || return 1
  # A group an instance uses only gains rules here: taking a rule away could cut off a live
  # instance (a seed's port, say) because this run was started with other options.
  in_use=$(aws ec2 describe-network-interfaces --region "$r" --filters "Name=group-id,Values=$sg" \
    --query 'length(NetworkInterfaces)' --output text) || return 1
  # The rules it should have, one per line: "protocol from to cidr".
  { echo "tcp 22 22 $MYIP/32"
    for o in ${OPEN[@]+"${OPEN[@]}"}; do
      echo "${o%%:*} ${o#*:} ${o#*:} 0.0.0.0/0"; echo "${o%%:*} ${o#*:} ${o#*:} ::/0"
    done; } > "$WORK/want"
  aws ec2 describe-security-group-rules --region "$r" --filters "Name=group-id,Values=$sg" \
    --query 'SecurityGroupRules[?IsEgress==`false`].[SecurityGroupRuleId,IpProtocol,FromPort,ToPort,CidrIpv4,CidrIpv6]' \
    --output text > "$WORK/have" || return 1
  : > "$WORK/present"
  while read -r id p f t c4 c6; do
    [ -n "$id" ] || continue
    key="$p $f $t $( [ "$c4" != None ] && echo "$c4" || echo "$c6" )"
    if grep -qxF "$key" "$WORK/want"; then echo "$key" >> "$WORK/present"
    elif [ "$managed" = "$MANAGED_BY" ] && [ "$in_use" = 0 ]; then
      if [ "$CHECK" = 1 ]; then would "remove the rule $key from $sg (not wanted)" >&2; continue; fi
      out=$(aws ec2 revoke-security-group-ingress --region "$r" --group-id "$sg" --security-group-rule-ids "$id" 2>&1) \
        || { say "  couldn't remove rule $key from $sg: $out" >&2; return 1; }
      say "  removed the rule $key from $sg (not wanted)" >&2
    elif [ "$managed" = "$MANAGED_BY" ]; then
      say "  note: $sg is in use by $in_use network interface(s); leaving its rule $key as it is" >&2
    else
      say "  note: $sg wasn't made by this script; leaving its rule $key as it is" >&2
    fi
  done < "$WORK/have"
  while read -r p f t c; do
    grep -qxF "$p $f $t $c" "$WORK/present" && continue
    if [ "$CHECK" = 1 ]; then would "add the rule $p $f $t $c to $sg" >&2; continue; fi
    case "$c" in *:*) rule="IpProtocol=$p,FromPort=$f,ToPort=$t,Ipv6Ranges=[{CidrIpv6=$c}]";;
                 *) rule="IpProtocol=$p,FromPort=$f,ToPort=$t,IpRanges=[{CidrIp=$c}]";; esac
    out=$(aws ec2 authorize-security-group-ingress --region "$r" --group-id "$sg" --ip-permissions "$rule" 2>&1) \
      || case "$out" in *InvalidPermission.Duplicate*) ;; *) say "  couldn't add rule $p $f $t $c to $sg: $out" >&2; return 1;; esac
  done < "$WORK/want"
  echo "$sg"
}

# Get region $1 ready (once): permission, default VPC, key pair, security group.
# Returns 0 when usable; otherwise records why in $WORK/skip-R.
prepare_region() {
  local r=$1 vpc out sg
  [ -e "$WORK/skip-$r" ] && return 1
  [ -e "$WORK/sg-$r" ] && return 0
  # Permission is checked with a dry run of an action that names no other resource, so nothing
  # else (an image id, a subnet) can fail first and hide the answer.
  out=$(aws ec2 create-security-group --region "$r" --dry-run --group-name "$SG_NAME" --description x 2>&1)
  case "$out" in *DryRunOperation*) ;;
    *UnauthorizedOperation*|*AccessDenied*|*AuthFailure*|*OptInRequired*) skip_region "$r" "this account may not use it"; return 1;;
    *) skip_region "$r" "permission check failed: $(echo "$out" | tail -1 | cut -c1-120)"; return 1;;
  esac
  vpc=$(aws ec2 describe-vpcs --region "$r" --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text 2>&1)
  case "$vpc" in vpc-*) ;; *) skip_region "$r" "no default VPC"; return 1;; esac
  echo "$vpc" > "$WORK/vpc-$r"
  if ! aws ec2 describe-key-pairs --region "$r" --key-names "$KEY" >/dev/null 2>&1; then
    [ -n "$PUBKEY" ] || { skip_region "$r" "no key pair $KEY (give --pubkey to import it)"; return 1; }
    if [ "$CHECK" = 1 ]; then
      would "import the key pair $KEY into $r from $PUBKEY; dry-running without a key pair"
      : > "$WORK/nokey-$r"
    else
      out=$(aws ec2 import-key-pair --region "$r" --key-name "$KEY" --public-key-material "fileb://$PUBKEY" 2>&1) \
        || { skip_region "$r" "couldn't import key pair $KEY: $(echo "$out" | tail -1 | cut -c1-120)"; return 1; }
    fi
  fi
  sg=$(ensure_group "$r" "$vpc") || die "couldn't set up the security group in $r (see above)"
  echo "$sg" > "$WORK/sg-$r"
}

# The image to launch $2 with in region $1; prints its id. Copies ours there when needed; returns 1
# when there's no usable image (the caller skips the type), 2 when the region can't have one at all
# (a copy failed or timed out: the caller skips the region), 3 when the image's volumes are refused
# (the reason in $WORK/volerr; checked before a copy, which is billed and has the same volumes).
image_for() {
  local r=$1 type=$2 arch id state n a
  arch=$(aws ec2 describe-instance-types --region "$r" --instance-types "$type" \
    --query 'InstanceTypes[0].ProcessorInfo.SupportedArchitectures' --output text) || return 1
  if [ -n "$AMI_NAME" ]; then
    for a in $arch; do
      id=$(aws ec2 describe-images --region "$r" --owners "$AMI_OWNER" --filters "Name=name,Values=$AMI_NAME" \
        Name=state,Values=available "Name=architecture,Values=$a" --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
      case "$id" in ami-*) echo "$id"; return 0;; esac
    done
    say "  no image matching $AMI_NAME for $type ($arch) in $r" >&2; return 1
  fi
  echo "$arch" | grep -qw "$SOURCE_ARCH" || { say "  $type can't run a $SOURCE_ARCH image" >&2; return 1; }
  [ "$r" = "$AMI_REGION" ] && { echo "$AMI_ID"; return 0; }
  [ -e "$WORK/noimage-$r" ] && return 2
  # A copy made earlier (this run or another) that is ready or on its way; never a broken one.
  id=$(aws ec2 describe-images --region "$r" --owners self --filters "Name=tag:openhydra-source-image,Values=$AMI_ID" \
    Name=state,Values=available,pending --query 'Images[0].ImageId' --output text) || return 1
  if [ "$id" = None ] || [ -z "$id" ]; then
    if [ "$CHECK" = 1 ]; then
      # A stand-in of the same architecture, so the rest of the request still gets its dry run.
      would "copy image $AMI_ID from $AMI_REGION to $r; dry-running with a current Amazon Linux image instead (the image itself isn't checked)" >&2
      id=$(aws ec2 describe-images --region "$r" --owners amazon --filters "Name=name,Values=al2023-ami-2023.*-kernel-*" \
        Name=state,Values=available "Name=architecture,Values=$SOURCE_ARCH" --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
      case "$id" in ami-*) : > "$WORK/standin-$r"; echo "$id"; return 0;; *) return 1;; esac
    fi
    : > "$WORK/volerr"
    check_volumes "$AMI_REGION" "$AMI_ID" >/dev/null || { [ -s "$WORK/volerr" ] && return 3; return 1; }
    say "  copying image $AMI_ID to $r (a few minutes)" >&2
    # Noted first, so a copy started just as the script is stopped is still found and removed.
    echo "$r $NAME-$RUN_ID" > "$WORK/copying"
    id=$(aws ec2 copy-image --region "$r" --source-region "$AMI_REGION" --source-image-id "$AMI_ID" --name "$NAME-$RUN_ID" \
      --tag-specifications "ResourceType=image,Tags=[{Key=Name,Value=$NAME},{Key=purpose,Value=$PURPOSE},{Key=openhydra-source-image,Value=$AMI_ID},{Key=$MANAGED_TAG,Value=$MANAGED_BY}]" \
        "ResourceType=snapshot,Tags=[{Key=Name,Value=$NAME},{Key=purpose,Value=$PURPOSE},{Key=openhydra-source-image,Value=$AMI_ID},{Key=$MANAGED_TAG,Value=$MANAGED_BY}]" \
      --query ImageId --output text) || { : > "$WORK/noimage-$r"; return 2; }
    echo "$r $id" >> "$WORK/copies"
    : > "$WORK/copying"
  fi
  # `aws ec2 wait image-available` gives up after 10 minutes; a copy can take longer.
  n=0
  while [ $n -lt 270 ]; do
    state=$(aws ec2 describe-images --region "$r" --image-ids "$id" --query 'Images[0].State' --output text 2>&1) \
      || { say "  lost access while waiting for the copy (signed out?): $state" >&2; return 1; }
    case "$state" in
      available) echo "$id"; return 0;;
      pending) ;;
      *) say "  the copy $id ended up $state" >&2; : > "$WORK/noimage-$r"; return 2;;
    esac
    sleep 10; n=$((n + 1))
  done
  say "  the copy didn't finish in 45 minutes" >&2; : > "$WORK/noimage-$r"; return 2
}

# The volumes image $2 in region $1 would launch with, checked: over 1000 GB in all needs
# --large-disk, and one that outlives the instance needs --keep-volumes. Prints the root device;
# returns 3 with the reason in $WORK/volerr when they're refused.
check_volumes() {
  local r=$1 ami=$2 root dev size keep total=0
  root=$(aws ec2 describe-images --region "$r" --image-ids "$ami" --query 'Images[0].RootDeviceName' --output text) || return 1
  aws ec2 describe-images --region "$r" --image-ids "$ami" \
    --query 'Images[0].BlockDeviceMappings[?Ebs].[DeviceName,Ebs.VolumeSize,Ebs.DeleteOnTermination]' --output text > "$WORK/bdm" || return 1
  while read -r dev size keep; do
    [ -n "$dev" ] || continue
    case "$size" in ''|None|*[!0-9]*) size=0;; esac
    if [ "$dev" = "$root" ] && [ -n "$DISK" ]; then
      [ "$DISK" -ge "$size" ] || { echo "--disk $DISK is smaller than the image's $size GB root volume" > "$WORK/volerr"; return 3; }
      size=$DISK; keep=True   # the override below sets DeleteOnTermination=true
    fi
    total=$((total + size))
    if [ "$keep" = False ] && [ "$KEEP_VOLUMES" = 0 ]; then
      echo "image $ami has a volume ($dev, $size GB) that outlives the instance (DeleteOnTermination=false): after a spot interruption it keeps billing. Add --keep-volumes if you mean it" > "$WORK/volerr"
      return 3
    fi
  done < "$WORK/bdm"
  [ "$total" -le 1000 ] || [ "$LARGE_DISK" = 1 ] \
    || { echo "image $ami would launch with $total GB of volumes; add --large-disk if you mean it" > "$WORK/volerr"; return 3; }
  echo "$root"
}

# Run the launch request (arguments) so that a Ctrl-C or a closed terminal can't cut it short: the
# call itself ignores those signals, and this waits for its answer before acting on them. Output in
# $WORK/out and $WORK/err; returns the call's exit status.
INTERRUPTED=0
launch_call() {
  local pid rc
  INTERRUPTED=0
  trap 'INTERRUPTED=1' INT TERM HUP
  # Ignored signals stay ignored across exec, so the CLI never sees them. The exit status goes to a
  # file, which works the same on every bash.
  ( trap '' INT TERM HUP; aws ec2 run-instances "$@" > "$WORK/out" 2> "$WORK/err"; echo $? > "$WORK/rc" ) &
  pid=$!
  # A trapped signal makes `wait` return early: wait again until the call is really done.
  while kill -0 "$pid" 2>/dev/null; do wait "$pid"; done
  trap on_signal INT TERM HUP
  rc=$(cat "$WORK/rc" 2>/dev/null) || rc=255
  return "${rc:-255}"
}

N=0 SEEN=0
while read -r PRICE R AZ TYPE; do
  SEEN=$((SEEN + 1))
  [ -e "$WORK/skip-$R" ] && continue
  say "trying $TYPE in $AZ at \$$PRICE/h"
  prepare_region "$R" || continue
  SG=$(cat "$WORK/sg-$R")
  read -r SUB V6 < <(aws ec2 describe-subnets --region "$R" --filters Name=default-for-az,Values=true "Name=availability-zone,Values=$AZ" \
    --query 'Subnets[0].[SubnetId,length(Ipv6CidrBlockAssociationSet || `[]`)]' --output text) || true
  case "${SUB:-}" in subnet-*) ;; *) say "  skipped: no default subnet in $AZ"; continue;; esac
  if [ "$IPV6" = 1 ] && [ "${V6:-0}" = 0 ]; then say "  skipped: no IPv6 in the default subnet of $AZ"; continue; fi
  AMI=$(image_for "$R" "$TYPE"); rc=$?
  case $rc in
    0) ;;
    2) skip_region "$R" "no usable image copy there"; continue;;
    3) die "$(cat "$WORK/volerr")";;
    *) say "  skipped: no usable image for $TYPE in $R"; continue;;
  esac
  : > "$WORK/volerr"
  # With a stand-in image (--check, no copy there yet), the volumes checked are your image's.
  if [ -e "$WORK/standin-$R" ]; then
    check_volumes "$AMI_REGION" "$AMI_ID" >/dev/null || { [ -s "$WORK/volerr" ] && die "$(cat "$WORK/volerr")"; die "couldn't read the volumes of image $AMI_ID in $AMI_REGION"; }
    : > "$WORK/volerr"
  fi
  ROOT=$(check_volumes "$R" "$AMI") || { [ -s "$WORK/volerr" ] && die "$(cat "$WORK/volerr")"; die "couldn't read the volumes of image $AMI in $R"; }
  NET="DeviceIndex=0,SubnetId=$SUB,Groups=$SG,AssociatePublicIpAddress=true"
  [ "$IPV6" = 1 ] && NET="$NET,Ipv6AddressCount=1"
  TAGS="Tags=[{Key=Name,Value=$NAME},{Key=purpose,Value=$PURPOSE},{Key=$MANAGED_TAG,Value=$MANAGED_BY}]"
  REQ=(--region "$R" --image-id "$AMI" --instance-type "$TYPE" --count 1
    --network-interfaces "$NET"
    --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}'
    --instance-initiated-shutdown-behavior terminate
    --metadata-options 'HttpTokens=required,HttpPutResponseHopLimit=1,HttpProtocolIpv6=disabled'
    --tag-specifications "ResourceType=instance,$TAGS" "ResourceType=volume,$TAGS" "ResourceType=network-interface,$TAGS" "ResourceType=spot-instances-request,$TAGS")
  [ -e "$WORK/nokey-$R" ] || REQ+=(--key-name "$KEY")
  [ -z "$PROFILE" ] || REQ+=(--iam-instance-profile "Name=$PROFILE")
  [ -z "$USERDATA" ] || REQ+=(--user-data "fileb://$USERDATA")
  [ -z "$DISK" ] || REQ+=(--block-device-mappings "DeviceName=$ROOT,Ebs={VolumeSize=$DISK,VolumeType=gp3,DeleteOnTermination=true}")

  # The whole request, checked without launching (capacity is the one thing a dry run can't check).
  aws ec2 run-instances "${REQ[@]}" --dry-run >/dev/null 2>"$WORK/err"
  CODE=$(err_code "$WORK/err")
  case "$CODE" in
    DryRunOperation) ;;
    Unsupported|InvalidAvailabilityZone) say "  skipped: $TYPE isn't offered in $AZ"; continue;;
    UnauthorizedOperation|AuthFailure|OptInRequired) skip_region "$R" "this account may not launch there"; continue;;
    InvalidParameterValue|InvalidParameterCombination)
      if [ "$IPV6" = 1 ] && grep -qi ipv6 "$WORK/err"; then skip_region "$R" "IPv6 isn't available there: $(tail -1 "$WORK/err" | cut -c1-160)"; continue; fi
      die "the request itself is wrong, so it would fail everywhere; stopping: $(tail -1 "$WORK/err" | cut -c1-300)";;
    *) die "the request itself is wrong, so it would fail everywhere; stopping: $(tail -1 "$WORK/err" | cut -c1-300)";;
  esac
  if [ "$CHECK" = 1 ]; then
    say "CHECK PASSED: the request would be accepted for $TYPE in $AZ at \$$PRICE/h (nothing launched, nothing changed)"; exit 0
  fi

  N=$((N + 1)); TOKEN="$(echo "$NAME" | cut -c1-30)-$RUN_ID-$N"
  echo "$R $TOKEN" > "$WORK/inflight"
  launch_call "${REQ[@]}" --client-token "$TOKEN" --query 'Instances[0].InstanceId' --output text
  OUT=$(cat "$WORK/out" 2>/dev/null)
  case "$OUT" in i-*) ID=$OUT;; *)
    CODE=$(err_code "$WORK/err")
    case "$CODE" in
      InsufficientInstanceCapacity|InsufficientCapacity|InsufficientHostCapacity|SpotMaxPriceTooLow|Unsupported|InsufficientFreeAddressesInSubnet)
        : > "$WORK/inflight"; say "  no capacity ($CODE)"
        [ "$INTERRUPTED" = 1 ] && on_signal
        continue;;
      MaxSpotInstanceCountExceeded|VcpuLimitExceeded|InstanceLimitExceeded|SpotInstanceLimitExceeded)
        : > "$WORK/inflight"; skip_region "$R" "a quota is full ($CODE)"
        [ "$INTERRUPTED" = 1 ] && on_signal
        continue;;
    esac
    # Anything else may have failed after EC2 accepted the request: look for it by its token.
    if ID=$(find_by_token "$R" "$TOKEN"); then
      say "  (the launch reported an error, but the instance exists)"
    else
      # Not found, which doesn't prove it didn't start: the clean-up keeps this region's things and
      # prints how to check.
      die "launch failed with an error that isn't about capacity; stopping: $(tail -1 "$WORK/err" | cut -c1-300)"
    fi;;
  esac
  LAUNCHED_REGION=$R LAUNCHED_SG=$SG
  : > "$WORK/inflight"
  say "LAUNCHED $ID $TYPE $R $AZ \$$PRICE/h (security group $SG, image $AMI)"
  if [ "$INTERRUPTED" = 1 ]; then
    say "  (stopped while launching: not waiting for it to run)"
    say "  if you don't want it: aws ec2 terminate-instances --region $R --instance-ids $ID"
    exit 130
  fi
  if aws ec2 wait instance-running --region "$R" --instance-ids "$ID" 2>/dev/null; then
    say "  running at $(aws ec2 describe-instances --region "$R" --instance-ids "$ID" \
      --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"
  else
    say "  it didn't reach running within 10 minutes; check: aws ec2 describe-instances --region $R --instance-ids $ID" >&2
  fi
  exit 0
done < "$WORK/candidates"
if ls "$WORK"/skip-* >/dev/null 2>&1; then
  say "regions left out:"; for f in "$WORK"/skip-*; do say "  ${f##*/skip-}: $(cat "$f")"; done
fi
say "NOTHING LAUNCHED: none of the $SEEN candidates could be used (reasons above)"; exit 1
