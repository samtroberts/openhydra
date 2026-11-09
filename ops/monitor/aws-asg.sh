#!/usr/bin/env bash
# The AWS network probe (see README.md) as a one-instance spot Auto Scaling group, so it comes back
# by itself when AWS takes the instance away, and a person is emailed if it can't.
#
# - The group keeps one instance of t4g.nano or t4g.micro, in every zone of the region offering both,
#   with the price-capacity-optimized spot strategy: among the lowest-priced pools, the one least
#   likely to be taken away (Sam's choice, 2026-10-29, after the lowest-price pool was taken away
#   twice in 20 minutes; worst case micro, about $0.22 a month more). Capacity Rebalancing starts a
#   replacement as soon as AWS warns an instance is at risk.
# - Every instance boots from an image of a set-up probe box (binary, settings, mail password), so
#   nothing secret passes through this script. The image is copied into the region once.
# - SNS emails ALERT_TO when the group fails to launch an instance, and when it has had no instance
#   in service for 15 minutes (any cause: no capacity in the whole region, a bad image, the group
#   gone). The address must confirm the subscription once, from the email AWS sends.
# - The instance allows SSH only from this machine; it needs nothing inbound otherwise.
#
# Idempotent: run it again to change the region's setup or to pick up a new source image (a new
# launch-template version; the next instance uses it). To move to another region, run it there,
# check, then remove the old group with --delete in the old region.
#
# usage: aws-asg.sh [--region R] [--delete [--images]]
#   --delete   remove the group (its instance terminates), alarm, alert topic (and its email
#              subscription) and launch template; the images and security group stay (an image is
#              the recovery point), unless --images: then this group's image copies and their
#              snapshots go too, except SOURCE_IMAGE itself
#   env: ALERT_TO (default sam@openhydra.co), SOURCE_IMAGE + SOURCE_REGION (the probe box image),
#        AWS_PROFILE (default openhydra), MYIP (the IPv4 SSH is allowed from; default this machine's)
set -euo pipefail
export AWS_PROFILE=${AWS_PROFILE:-openhydra}
R=ap-south-2 DELETE=0 IMAGES=0
while [ $# -gt 0 ]; do
  case "$1" in
    --region) R=${2:?--region needs a value}; shift 2;;
    --delete) DELETE=1; shift;;
    --images) IMAGES=1; shift;;
    *) echo "usage: aws-asg.sh [--region R] [--delete [--images]]" >&2; exit 2;;
  esac
done
[ "$IMAGES" = 0 ] || [ "$DELETE" = 1 ] || { echo "aws-asg: --images goes with --delete" >&2; exit 2; }
NAME=openhydra-monitor
TYPES=(t4g.nano t4g.micro)
# The probe box image, made from a set-up instance on 2026-10-28 (its first copy, from us-east-1,
# whose original was then removed).
SOURCE_IMAGE=${SOURCE_IMAGE:-ami-0b157fd332d9ac1d8} SOURCE_REGION=${SOURCE_REGION:-ap-south-2}
ALERT_TO=${ALERT_TO:-sam@openhydra.co}
KEY=openhydra-seed
MANAGED_TAG=openhydra-managed-by MANAGED_BY=aws-asg
TOPIC_NAME=$NAME-alerts ALARM=$NAME-down
die() { echo "aws-asg: $*" >&2; exit 1; }
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
sha256() { if command -v sha256sum >/dev/null; then sha256sum; else shasum -a 256; fi; }
# Remove image $1 in $R and its snapshots, saying so; any snapshot that stays is said too (the
# image-and-snapshots call succeeds even when one isn't deleted). Older CLIs: one by one.
remove_image() {
  local img=$1 snaps left
  if aws ec2 deregister-image --generate-cli-skeleton 2>/dev/null | grep -q DeleteAssociatedSnapshots; then
    left=$(aws ec2 deregister-image --region "$R" --image-id "$img" --delete-associated-snapshots \
      --query 'DeleteSnapshotResults[?ReturnCode!=`success`].[SnapshotId,ReturnCode]' --output text)
  else
    snaps=$(aws ec2 describe-images --region "$R" --image-ids "$img" --query 'Images[0].BlockDeviceMappings[].Ebs.SnapshotId' --output text)
    aws ec2 deregister-image --region "$R" --image-id "$img"
    left=
    for snap in $snaps; do
      aws ec2 delete-snapshot --region "$R" --snapshot-id "$snap" || left="$left $snap"
    done
  fi
  if [ -n "$left" ] && [ "$left" != None ]; then
    echo "removed image $img, but NOT its snapshot(s): $(echo "$left" | tr '\t\n' ': '); remove with: aws ec2 delete-snapshot --region $R --snapshot-id <id>" >&2
  else
    echo "removed image $img and its snapshots"
  fi
}
aws sts get-caller-identity >/dev/null 2>&1 || die "not signed in: run  aws sso login --profile $AWS_PROFILE"

if [ "$DELETE" = 1 ]; then
  # The group and its instance, its alarm and notification; the image and security group stay.
  # (delete-alarms succeeds for a name that doesn't exist.)
  aws cloudwatch delete-alarms --region "$R" --alarm-names "$ALARM"
  if out=$(aws autoscaling delete-auto-scaling-group --region "$R" --auto-scaling-group-name "$NAME" --force-delete 2>&1); then
    echo "group $NAME in $R is being deleted (its instance terminates)"
  else
    case "$out" in
      *ValidationError*"not found"*|*"AutoScalingGroup name not found"*) echo "no group $NAME in $R";;
      *) die "couldn't delete the group $NAME in $R (nothing else removed after the alarm): $out";;
    esac
  fi
  if out=$(aws ec2 delete-launch-template --region "$R" --launch-template-name "$NAME" 2>&1); then
    echo "launch template removed"
  else
    case "$out" in *NotFound*) ;; *) die "couldn't delete the launch template $NAME in $R: $out";; esac
  fi
  TOPIC=$(aws sns list-topics --region "$R" --query "Topics[?ends_with(TopicArn, ':$TOPIC_NAME')].TopicArn | [0]" --output text)
  case "$TOPIC" in arn:*) aws sns delete-topic --region "$R" --topic-arn "$TOPIC" && echo "alert topic and its subscription removed";; esac
  if [ "$IMAGES" = 1 ]; then
    # Only the copies this script made (its tag): the launcher tags its own copies, of any image and
    # possibly under the same name, as its own (review 2026-10-29). (Each list is read before the
    # loop: a failed call inside `for … in $(…)` would go unnoticed under set -e.)
    imgs=$(aws ec2 describe-images --region "$R" --owners self --filters "Name=tag:$MANAGED_TAG,Values=$MANAGED_BY" \
      Name=tag-key,Values=openhydra-source-image --query 'Images[].ImageId' --output text) || die "couldn't list this script's image copies in $R"
    for img in $imgs; do
      # Never the source image: in its own region it may be the only copy of the set-up box.
      if [ "$img" = "$SOURCE_IMAGE" ]; then echo "kept $img: it's SOURCE_IMAGE"; continue; fi
      remove_image "$img"
    done
    # Copies made before this script tagged them: said, not removed (nothing tells them apart from
    # a launcher copy named the same).
    old=$(aws ec2 describe-images --region "$R" --owners self --filters "Name=tag:Name,Values=$NAME" \
      Name=tag-key,Values=openhydra-source-image --query "Images[?!(Tags[?Key=='$MANAGED_TAG'])].ImageId" --output text) \
      || die "couldn't list the older image copies in $R"
    for img in $old; do
      [ "$img" = "$SOURCE_IMAGE" ] && continue
      echo "note: image $img (named $NAME) has no $MANAGED_TAG tag, so it was left; if it's this script's, remove it with:"
      echo "  aws ec2 deregister-image --region $R --image-id $img --delete-associated-snapshots"
    done
  fi
  exit 0
fi

# Its volumes: nothing that outlives the instance (it would keep billing after every replacement),
# and nothing big (a probe box needs a few GB). Checked on the source image, before any (billed)
# copy, which has the same volumes. (Read before the loop: a failed call feeding `done < <(…)`
# would go unnoticed under set -e and the check would pass with nothing checked.)
BDM=$(aws ec2 describe-images --region "$SOURCE_REGION" --image-ids "$SOURCE_IMAGE" \
  --query 'Images[0].BlockDeviceMappings[?Ebs].[DeviceName,Ebs.VolumeSize,Ebs.DeleteOnTermination]' --output text) \
  || die "couldn't read the volumes of image $SOURCE_IMAGE in $SOURCE_REGION"
while read -r dev size keep; do
  [ -n "$dev" ] || continue
  [ "$keep" != False ] || die "image $SOURCE_IMAGE has a volume ($dev) that outlives the instance (DeleteOnTermination=false)"
  case "$size" in ''|None|*[!0-9]*) ;; *) [ "$size" -le 100 ] || die "image $SOURCE_IMAGE has a $size GB volume ($dev); a probe box needs a few GB";; esac
done <<< "$BDM"

# 1. The image, in this region.
if [ "$R" = "$SOURCE_REGION" ]; then AMI=$SOURCE_IMAGE
else
  # A copy this script made (or one from before it tagged them), never the launcher's: those are
  # removed when the launcher's run ends, which would leave the group booting a deleted image.
  AMI=$(aws ec2 describe-images --region "$R" --owners self --filters "Name=tag:openhydra-source-image,Values=$SOURCE_IMAGE" \
    Name=state,Values=available,pending \
    --query "sort_by(Images[?!(Tags[?Key=='$MANAGED_TAG']) || Tags[?Key=='$MANAGED_TAG' && Value=='$MANAGED_BY']],&CreationDate)[-1].ImageId" --output text)
  [ -n "$AMI" ] || AMI=None
  if [ "$AMI" = None ]; then
    echo "copying image $SOURCE_IMAGE from $SOURCE_REGION (a few minutes)"
    AMI=$(aws ec2 copy-image --region "$R" --source-region "$SOURCE_REGION" --source-image-id "$SOURCE_IMAGE" \
      --name "$NAME-$(date -u +%Y%m%d%H%M%S)" --query ImageId --output text \
      --tag-specifications "ResourceType=image,Tags=[{Key=Name,Value=$NAME},{Key=purpose,Value=network-probe},{Key=openhydra-source-image,Value=$SOURCE_IMAGE},{Key=$MANAGED_TAG,Value=$MANAGED_BY}]" \
        "ResourceType=snapshot,Tags=[{Key=Name,Value=$NAME},{Key=purpose,Value=network-probe},{Key=openhydra-source-image,Value=$SOURCE_IMAGE},{Key=$MANAGED_TAG,Value=$MANAGED_BY}]")
  fi
  for _ in $(seq 270); do
    state=$(aws ec2 describe-images --region "$R" --image-ids "$AMI" --query 'Images[0].State' --output text)
    case "$state" in available) break;; pending) sleep 10;; *) die "image $AMI is $state";; esac
  done
  [ "$state" = available ] || die "image $AMI still isn't ready after 45 minutes"
fi
echo "image: $AMI"

# 2. Network: the default VPC's subnets in the zones that offer every type (a zone lacking one of
#    them fails the whole launch request there).
VPC=$(aws ec2 describe-vpcs --region "$R" --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text)
[ "$VPC" != None ] || die "no default VPC in $R"
ZONES=
AZS=$(aws ec2 describe-availability-zones --region "$R" --filters Name=state,Values=available --query 'AvailabilityZones[].ZoneName' --output text) \
  || die "couldn't list the zones of $R"
for z in $AZS; do
  n=$(aws ec2 describe-instance-type-offerings --region "$R" --location-type availability-zone --filters "Name=location,Values=$z" \
    "Name=instance-type,Values=$(IFS=,; echo "${TYPES[*]}")" --query 'length(InstanceTypeOfferings)' --output text)
  [ "$n" -eq ${#TYPES[@]} ] && ZONES="$ZONES,$z"
done
[ -n "$ZONES" ] || die "no zone in $R offers all of ${TYPES[*]}"
SUBNETS=$(aws ec2 describe-subnets --region "$R" --filters "Name=vpc-id,Values=$VPC" Name=default-for-az,Values=true \
  "Name=availability-zone,Values=${ZONES#,}" --query 'Subnets[].SubnetId' --output text | tr '\t' ',')
echo "zones: ${ZONES#,}"
aws ec2 describe-key-pairs --region "$R" --key-names "$KEY" >/dev/null 2>&1 \
  || die "no key pair $KEY in $R; import one first: aws ec2 import-key-pair --region $R --key-name $KEY --public-key-material fileb://KEY.pub"

# 3. Security group: SSH from this machine only, nothing else inbound. A group this script made is
#    kept to exactly that, so rules for old addresses don't pile up.
# MYIP overrides the address SSH is allowed from (on a temporary network, keep the usual one).
MYIP=${MYIP:-$(curl -fsS -4 -m 10 https://checkip.amazonaws.com | tr -d '\n')}
valid_ipv4 "$MYIP" || die "couldn't find this machine's public IPv4 (for the SSH rule); got '${MYIP:0:40}'"
SG=$(aws ec2 describe-security-groups --region "$R" --filters "Name=group-name,Values=$NAME" "Name=vpc-id,Values=$VPC" \
  --query 'SecurityGroups[0].GroupId' --output text)
if [ "$SG" = None ]; then
  SG=$(aws ec2 create-security-group --region "$R" --group-name "$NAME" --description "OpenHydra network probe: SSH from operator only" \
    --vpc-id "$VPC" --query GroupId --output text \
    --tag-specifications "ResourceType=security-group,Tags=[{Key=Name,Value=$NAME},{Key=purpose,Value=network-probe},{Key=$MANAGED_TAG,Value=$MANAGED_BY}]")
  echo "security group created: $SG"
fi
managed=$(aws ec2 describe-security-groups --region "$R" --group-ids "$SG" \
  --query "SecurityGroups[0].Tags[?Key=='$MANAGED_TAG'].Value | [0]" --output text)
want="tcp 22 22 $MYIP/32" have=0
# (Read first, for the same reason as the volumes above.)
RULES=$(aws ec2 describe-security-group-rules --region "$R" --filters "Name=group-id,Values=$SG" \
  --query 'SecurityGroupRules[?IsEgress==`false`].[SecurityGroupRuleId,IpProtocol,FromPort,ToPort,CidrIpv4,CidrIpv6]' --output text) \
  || die "couldn't read the rules of $SG"
while read -r id p f t c4 c6; do
  [ -n "$id" ] || continue
  rule="$p $f $t $([ "$c4" != None ] && echo "$c4" || echo "$c6")"
  if [ "$rule" = "$want" ]; then have=1
  elif [ "$managed" = "$MANAGED_BY" ]; then
    aws ec2 revoke-security-group-ingress --region "$R" --group-id "$SG" --security-group-rule-ids "$id" >/dev/null
    echo "removed rule $rule from $SG"
  else echo "note: $SG wasn't made by this script; leaving its rule $rule"
  fi
done <<< "$RULES"
[ "$have" = 1 ] || aws ec2 authorize-security-group-ingress --region "$R" --group-id "$SG" \
  --ip-permissions "IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=$MYIP/32,Description=operator}]" >/dev/null
echo "security group: $SG (SSH from $MYIP)"

# 4. Launch template. A new version only when its contents change (the hash is in the version
#    description); the group always uses the latest.
SHORT=$(echo "$R" | sed -E 's/^([a-z]{2})-([a-z])[a-z]*-([0-9]+)$/\1\2\3/')   # ap-south-2 -> aps2
UD=$(printf '#cloud-config\n# Set at every boot: a cloned image would otherwise get a name from its address.\nhostname: %s\npreserve_hostname: false\n' "$NAME-$SHORT" | base64 | tr -d '\n')
LTDATA=$(cat <<JSON
{"ImageId":"$AMI","KeyName":"$KEY",
 "NetworkInterfaces":[{"DeviceIndex":0,"AssociatePublicIpAddress":true,"Groups":["$SG"],"DeleteOnTermination":true}],
 "MetadataOptions":{"HttpTokens":"required","HttpPutResponseHopLimit":1,"HttpEndpoint":"enabled","HttpProtocolIpv6":"disabled"},
 "TagSpecifications":[{"ResourceType":"instance","Tags":[{"Key":"Name","Value":"$NAME"},{"Key":"purpose","Value":"network-probe"}]},
                      {"ResourceType":"volume","Tags":[{"Key":"Name","Value":"$NAME"},{"Key":"purpose","Value":"network-probe"}]},
                      {"ResourceType":"network-interface","Tags":[{"Key":"Name","Value":"$NAME"},{"Key":"purpose","Value":"network-probe"}]}],
 "UserData":"$UD"}
JSON
)
HASH=$(printf '%s' "$LTDATA" | sha256 | cut -c1-16)
if aws ec2 describe-launch-templates --region "$R" --launch-template-names "$NAME" >/dev/null 2>&1; then
  CUR=$(aws ec2 describe-launch-template-versions --region "$R" --launch-template-name "$NAME" --versions '$Latest' \
    --query 'LaunchTemplateVersions[0].VersionDescription' --output text)
  if [ "$CUR" = "sha:$HASH" ]; then echo "launch template unchanged"
  else
    V=$(aws ec2 create-launch-template-version --region "$R" --launch-template-name "$NAME" --version-description "sha:$HASH" \
      --launch-template-data "$LTDATA" --query 'LaunchTemplateVersion.VersionNumber' --output text)
    echo "launch template version $V (the next instance uses it)"
  fi
else
  aws ec2 create-launch-template --region "$R" --launch-template-name "$NAME" --version-description "sha:$HASH" \
    --launch-template-data "$LTDATA" \
    --tag-specifications "ResourceType=launch-template,Tags=[{Key=Name,Value=$NAME},{Key=purpose,Value=network-probe}]" >/dev/null
  echo "launch template created"
fi

# 5. The group: exactly one instance, spot only, the cheapest pool least likely to be taken away.
OVR=$(printf '{"InstanceType":"%s"},' "${TYPES[@]}"); OVR="[${OVR%,}]"
MIP="{\"LaunchTemplate\":{\"LaunchTemplateSpecification\":{\"LaunchTemplateName\":\"$NAME\",\"Version\":\"\$Latest\"},\"Overrides\":$OVR},
 \"InstancesDistribution\":{\"OnDemandBaseCapacity\":0,\"OnDemandPercentageAboveBaseCapacity\":0,\"SpotAllocationStrategy\":\"price-capacity-optimized\"}}"
if [ "$(aws autoscaling describe-auto-scaling-groups --region "$R" --auto-scaling-group-names "$NAME" --query 'length(AutoScalingGroups)' --output text)" = 1 ]; then
  aws autoscaling update-auto-scaling-group --region "$R" --auto-scaling-group-name "$NAME" --min-size 1 --max-size 1 --desired-capacity 1 \
    --capacity-rebalance --vpc-zone-identifier "$SUBNETS" --mixed-instances-policy "$MIP"
  echo "group updated"
else
  aws autoscaling create-auto-scaling-group --region "$R" --auto-scaling-group-name "$NAME" --min-size 1 --max-size 1 --desired-capacity 1 \
    --capacity-rebalance --vpc-zone-identifier "$SUBNETS" --mixed-instances-policy "$MIP" \
    --tags "Key=purpose,Value=network-probe,PropagateAtLaunch=true"
  echo "group created: it launches its instance now"
fi
aws autoscaling enable-metrics-collection --region "$R" --auto-scaling-group-name "$NAME" --granularity 1Minute --metrics GroupInServiceInstances

# Older probe images in this region the group no longer boots (it always uses the launch template's
# latest version; older versions keep naming the images they had, so they don't count): said, with
# how to remove them; never removed here, as one may be the only copy of a set-up box someone still
# wants. SOURCE_IMAGE is never listed.
USED=$(aws ec2 describe-launch-template-versions --region "$R" --launch-template-name "$NAME" --versions '$Latest' \
  --query 'LaunchTemplateVersions[0].LaunchTemplateData.ImageId' --output text)
IMGS=$(aws ec2 describe-images --region "$R" --owners self --filters "Name=tag:Name,Values=$NAME" --query 'Images[].ImageId' --output text) \
  || die "couldn't list the probe images in $R"
for img in $IMGS; do
  case " $USED $AMI $SOURCE_IMAGE " in *" $img "*) continue;; esac
  echo "note: image $img isn't what the group boots any more; remove it with:"
  echo "  aws ec2 deregister-image --region $R --image-id $img --delete-associated-snapshots"
done

# 6. Alerts: an SNS topic emailing ALERT_TO, told about failed launches and a group with nothing in service.
TOPIC=$(aws sns create-topic --region "$R" --name "$TOPIC_NAME" --tags "Key=purpose,Value=network-probe" --query TopicArn --output text)
SUB=$(aws sns list-subscriptions-by-topic --region "$R" --topic-arn "$TOPIC" --query "Subscriptions[?Endpoint=='$ALERT_TO'].SubscriptionArn | [0]" --output text)
case "$SUB" in
  arn:*) ;;
  PendingConfirmation)
    # Not confirmed yet: send the email again (its link expires after 3 days).
    aws sns subscribe --region "$R" --topic-arn "$TOPIC" --protocol email --notification-endpoint "$ALERT_TO" >/dev/null
    echo "note: $ALERT_TO hasn't confirmed the subscription; the confirmation email was sent again";;
  *)
    aws sns subscribe --region "$R" --topic-arn "$TOPIC" --protocol email --notification-endpoint "$ALERT_TO" >/dev/null
    echo "subscribed $ALERT_TO: CONFIRM it from the email AWS just sent, or no alert will arrive";;
esac
aws autoscaling put-notification-configuration --region "$R" --auto-scaling-group-name "$NAME" --topic-arn "$TOPIC" \
  --notification-types autoscaling:EC2_INSTANCE_LAUNCH_ERROR
aws cloudwatch put-metric-alarm --region "$R" --alarm-name "$ALARM" \
  --alarm-description "The OpenHydra network probe group in $R has had no instance in service for 15 minutes: nothing is watching the bootstraps." \
  --namespace AWS/AutoScaling --metric-name GroupInServiceInstances --dimensions "Name=AutoScalingGroupName,Value=$NAME" \
  --statistic Minimum --period 300 --evaluation-periods 3 --datapoints-to-alarm 3 --threshold 1 --comparison-operator LessThanThreshold \
  --treat-missing-data breaching --alarm-actions "$TOPIC" --ok-actions "$TOPIC"
echo "alerts: $TOPIC -> $ALERT_TO (launch errors; no instance for 15 minutes)"
aws autoscaling describe-auto-scaling-groups --region "$R" --auto-scaling-group-names "$NAME" \
  --query 'AutoScalingGroups[0].Instances[].[InstanceId,InstanceType,AvailabilityZone,LifecycleState]' --output text | sed 's/^/instance: /'
