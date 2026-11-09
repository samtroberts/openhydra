#!/usr/bin/env bash
# Render a seed's first-boot script from hyperqwen-seed.sh.in.
#
# The seed's identity reaches the instance one of two ways:
#   --ssm-param NAME --region R   (recommended) the instance reads it at first boot from that SSM
#       Parameter Store SecureString, with an instance role allowed to read only that parameter. The
#       rendered script holds no key. This prints the commands that store the key and make the role;
#       run them yourself, signed in to AWS (they're not run here), then launch with
#       --instance-profile.
#   --embed-key   the key is written into the rendered script. Its user-data then holds the key for as
#       long as the instance lives, readable by root and through the instance's metadata. The output
#       is written mode 600 and belongs in ~/.openhydra-seeds, never in a git checkout.
#
# usage: render.sh NAME IDENTITY_FILE OUT (--ssm-param NAME --region R | --embed-key) [--prompt-scoring]
#   --prompt-scoring  answer authenticity challenges (vLLM prompt log-probs)
set -euo pipefail
usage() { echo "usage: render.sh NAME IDENTITY_FILE OUT (--ssm-param NAME --region R | --embed-key) [--prompt-scoring]" >&2; exit 1; }
[ $# -ge 3 ] || usage
NAME=$1; IDENTITY=$2; OUT=$3; shift 3
FLAG= PARAM= REGION= EMBED=0
while [ $# -gt 0 ]; do
  case "$1" in
    --prompt-scoring) FLAG="--prompt-scoring vllm"; shift;;
    --ssm-param) PARAM=${2:-}; shift 2 || usage;;
    --region) REGION=${2:-}; shift 2 || usage;;
    --embed-key) EMBED=1; shift;;
    *) echo "render: unknown option $1" >&2; usage;;
  esac
done
if [ -n "$PARAM" ]; then
  [ "$EMBED" = 0 ] || { echo "render: give --ssm-param or --embed-key, not both" >&2; exit 1; }
  case "$PARAM" in /*) ;; *) echo "render: --ssm-param must be a path, e.g. /openhydra/seeds/$NAME/identity" >&2; exit 1;; esac
  case "$PARAM" in *[!A-Za-z0-9/_.-]*) echo "render: --ssm-param may only use letters, digits and / _ . -" >&2; exit 1;; esac
  case "$REGION" in ''|*[!a-z0-9-]*) echo "render: --ssm-param needs --region (the region the seed launches in)" >&2; exit 1;; esac
elif [ "$EMBED" = 0 ]; then
  echo "render: say where the key goes: --ssm-param NAME --region R (recommended), or --embed-key" >&2; exit 1
fi
case "$OUT" in */) echo "render: OUT must be a file path, not a directory" >&2; exit 1;; esac
[ -d "$OUT" ] && { echo "render: $OUT is a directory; give a file path" >&2; exit 1; }
TEMPLATE="$(cd "$(dirname "$0")" && pwd -P)/hyperqwen-seed.sh.in"
case "$NAME" in *[!a-z0-9-]*|"") echo "render: NAME must be lowercase letters, digits and dashes" >&2; exit 1;; esac
[ -f "$IDENTITY" ] && [ -s "$IDENTITY" ] || { echo "render: no identity (or an empty one) at $IDENTITY" >&2; exit 1; }
OUT_DIR="$(cd "$(dirname "$OUT")" && pwd -P)"
# Never into a git checkout, decided without trusting git's exit code (an error such as "dubious
# ownership" used to count as "not a checkout", review 2026-10-29): any directory from OUT's up to
# the root holding a `.git` (a directory, or a worktree's file) refuses, and so does git saying so.
d=$OUT_DIR
while :; do
  [ -e "$d/.git" ] && { echo "render: refusing to write into a git checkout ($d)" >&2; exit 1; }
  [ "$d" = / ] && break
  d=$(dirname "$d")
done
if command -v git >/dev/null; then
  inside=$(git -C "$OUT_DIR" rev-parse --is-inside-work-tree 2>/dev/null) && [ "$inside" = true ] \
    && { echo "render: refusing to write into a git checkout ($OUT_DIR)" >&2; exit 1; }
fi
[ -L "$OUT" ] && { echo "render: $OUT is a symlink; refusing to write through it" >&2; exit 1; }
# A standalone assignment, so a failed read stops the script (as a prefix assignment it wouldn't).
B64="$(base64 < "$IDENTITY" | tr -d '\n')"
[ -n "$B64" ] || { echo "render: couldn't read $IDENTITY" >&2; exit 1; }
if [ "$EMBED" = 1 ]; then
  SETUP="echo \"$B64\" | base64 -d > /etc/openhydra/identity.json"
else
  # Root reads the parameter with the instance role's credentials, once: on later boots (and
  # restarts) the key is already on disk.
  SETUP="if [ ! -s /etc/openhydra/identity.json ]; then
  command -v aws >/dev/null || { echo \"no aws CLI on this image: can't read the seed key from SSM\"; exit 1; }
  aws ssm get-parameter --region $REGION --name $PARAM --with-decryption --query Parameter.Value --output text \\
    | base64 -d > /etc/openhydra/identity.json.new
  [ -s /etc/openhydra/identity.json.new ] || { echo \"the seed key from SSM ($PARAM) is empty\"; exit 1; }
  mv /etc/openhydra/identity.json.new /etc/openhydra/identity.json
fi"
fi
umask 077
# Written to a new private file beside OUT, then moved into place: never through an existing file.
TMP="$(mktemp "$OUT_DIR/.render.XXXXXX")"
trap 'rm -f "$TMP"' EXIT
# Values go to awk through the environment, not a command line (where `ps` would show the key).
SEED_NAME="$NAME" PROMPT_SCORING="$FLAG" IDENTITY_SETUP="$SETUP" \
  awk '{ gsub(/@SEED_NAME@/, ENVIRON["SEED_NAME"]); gsub(/@PROMPT_SCORING@/, ENVIRON["PROMPT_SCORING"]);
         if ($0 == "@IDENTITY_SETUP@") { print ENVIRON["IDENTITY_SETUP"]; next } print }' "$TEMPLATE" > "$TMP"
if grep -q '@[A-Z0-9_]*@' "$TMP"; then
  echo "render: a placeholder was left unfilled" >&2; exit 1
fi
if [ "$EMBED" = 1 ]; then
  grep -q 'echo "[A-Za-z0-9+/=]\{16,\}" | base64 -d' "$TMP" || { echo "render: the identity didn't make it into the script" >&2; exit 1; }
else
  grep -q "aws ssm get-parameter --region $REGION --name $PARAM" "$TMP" || { echo "render: the SSM read didn't make it into the script" >&2; exit 1; }
  # The key through the environment, not grep's command line (where `ps` would show it).
  ! B64="$B64" awk 'index($0, ENVIRON["B64"]) { found = 1 } END { exit !found }' "$TMP" \
    || { echo "render: the key ended up in the script" >&2; exit 1; }
fi
mv -f "$TMP" "$OUT"
echo "render: wrote $OUT (seed $NAME${FLAG:+, $FLAG}$([ "$EMBED" = 1 ] && echo ', key embedded' || echo ", key from SSM $PARAM"))"
if [ "$EMBED" = 0 ]; then
  ROLE="openhydra-seed-$NAME"
  cat <<EOF

Before launching, signed in to AWS, run these once (<account id>: yours):
  # 1. The key, as a SecureString with the AWS-managed key (aws/ssm). Read from the file, so it never
  #    appears on a command line:
  base64 < $IDENTITY | tr -d '\\n' | aws ssm put-parameter --region $REGION --name $PARAM --type SecureString --value file:///dev/stdin
  # 2. A role the instance assumes, allowed to read that one parameter and nothing else:
  aws iam create-role --role-name $ROLE --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
  aws iam put-role-policy --role-name $ROLE --policy-name read-seed-key --policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":"ssm:GetParameter","Resource":"arn:aws:ssm:$REGION:<account id>:parameter$PARAM"}]}'
  aws iam create-instance-profile --instance-profile-name $ROLE
  aws iam add-role-to-instance-profile --instance-profile-name $ROLE --role-name $ROLE
  # 3. Launch with: ops/aws/launch-cheapest-spot.sh ... --user-data $OUT --instance-profile $ROLE
(No kms:Decrypt statement: the AWS-managed aws/ssm key's own policy allows decryption through SSM.)
EOF
fi
