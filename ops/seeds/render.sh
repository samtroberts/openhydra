#!/usr/bin/env bash
# Render a seed's first-boot script from hyperqwen-seed.sh.in with its identity filled in.
# The output holds the seed's private key: it is written mode 600 and belongs in
# ~/.openhydra-seeds, never in the repo.
#
# usage: render.sh NAME IDENTITY_FILE OUT [--prompt-scoring]
#   --prompt-scoring  answer authenticity challenges (vLLM prompt log-probs)
set -euo pipefail
NAME=$1; IDENTITY=$2; OUT=$3; SCORING=${4:-}
case "$NAME" in *[!a-z0-9-]*|"") echo "render: NAME must be lowercase letters, digits and dashes" >&2; exit 1;; esac
[ -f "$IDENTITY" ] || { echo "render: no identity file at $IDENTITY" >&2; exit 1; }
REPO=$(cd "$(dirname "$0")/../.." && pwd -P)
OUT_ABS="$(cd "$(dirname "$OUT")" && pwd -P)/$(basename "$OUT")"
case "$OUT_ABS" in "$REPO"/*) echo "render: refusing to write a key into the repo" >&2; exit 1;; esac
case "$SCORING" in "") FLAG="";; --prompt-scoring) FLAG="--prompt-scoring vllm";; *) echo "render: unknown option $SCORING" >&2; exit 1;; esac
B64=$(base64 < "$IDENTITY" | tr -d '\n')
umask 077
sed -e "s|@SEED_NAME@|$NAME|g" -e "s|@IDENTITY_B64@|$B64|" -e "s|@PROMPT_SCORING@|$FLAG|" \
  "$(dirname "$0")/hyperqwen-seed.sh.in" > "$OUT"
chmod 600 "$OUT"
grep -q '@[A-Z_]*@' "$OUT" && { echo "render: a placeholder was left unfilled" >&2; exit 1; }
echo "render: wrote $OUT (seed $NAME${FLAG:+, $FLAG})"
