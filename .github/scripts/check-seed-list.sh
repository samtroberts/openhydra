#!/usr/bin/env bash
# Check a signed seed list before it is published (SL1): its signature against the key the source
# pins (with OpenSSL 3, as gateways check it: Ed25519 over the seed-list domain prefix + the list
# bytes), and its sequence against the one published now (gateways only move forward).
# usage: check-seed-list.sh LIST_JSON SEED_SOURCE PINNING_SOURCE [PUBLISHED_URL]
# Exit 0 and print "publish" or "unchanged" (already published, byte for byte); else fail.
set -euo pipefail
LIST=$1; SEED_SRC=$2; PIN_SRC=$3; URL=${4:-}
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
python3 -I - "$LIST" "$SEED_SRC" "$PIN_SRC" "$W" <<'PY'
import base64, json, sys
list_path, seed_src, pin_src, w = sys.argv[1:5]
env = json.load(open(list_path))
assert set(env) == {"list", "signature", "key"}, sorted(env)
pinned = open(pin_src).read().split('PINNED_INDEX_KEY: &str = "')[1][:64]
assert env["key"].lower() == pinned, "signed by a key this source doesn't pin"
domain = open(seed_src).read().split('SEED_DOMAIN: &[u8] = b"')[1].split('"')[0].replace("\\0", "\0").encode()
body = base64.b64decode(env["list"], validate=True)
lst = json.loads(body)
assert lst["format"] == 1 and isinstance(lst["sequence"], int), "not a seed list"
for s in lst["seeds"]:
    assert len(s["public_key"]) == 64 and all(c in "0123456789abcdef" for c in s["public_key"]), s
open(f"{w}/msg", "wb").write(domain + body)
open(f"{w}/sig", "wb").write(bytes.fromhex(env["signature"]))
# Ed25519 SubjectPublicKeyInfo (DER): fixed header + the 32-byte key.
open(f"{w}/key.der", "wb").write(bytes.fromhex("302a300506032b6570032100" + pinned))
open(f"{w}/seq", "w").write(str(lst["sequence"]))
print(f"seed list #{lst['sequence']}: {len(lst['seeds'])} seeds, key {pinned[:12]}…")
PY
openssl pkeyutl -verify -pubin -inkey "$W/key.der" -keyform DER -rawin -in "$W/msg" -sigfile "$W/sig" >/dev/null \
  || { echo "bad signature" >&2; exit 1; }
echo "signature OK"
SEQ=$(cat "$W/seq")
[ -z "$URL" ] && { echo "publish"; exit 0; }
code=$(curl -sS --proto '=https,http' -o "$W/published" -w '%{http_code}' "$URL" || true)
case "$code" in
  404) echo "nothing published yet"; echo "publish"; exit 0 ;;
  200) ;;
  *) echo "couldn't read the published seed list ($code)" >&2; exit 1 ;;
esac
PUB=$(python3 -I -c 'import base64,json,sys; print(json.loads(base64.b64decode(json.load(open(sys.argv[1]))["list"]))["sequence"])' "$W/published")
if [ "$SEQ" -gt "$PUB" ]; then echo "#$SEQ replaces published #$PUB"; echo "publish"
elif cmp -s "$LIST" "$W/published"; then echo "#$SEQ is already published"; echo "unchanged"
else echo "#$SEQ is not above published #$PUB: gateways would never take it" >&2; exit 1
fi
