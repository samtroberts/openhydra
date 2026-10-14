#!/usr/bin/env bash
# Check a signed tokenizer index before it is published (review 2026-10-14, L14): its signature
# against the key the source pins (with OpenSSL 3, as gateways check it: Ed25519 over the domain
# prefix + the index bytes), and its sequence against the one published now (gateways only move
# forward, so an equal or lower one would never reach them).
# usage: check-tokenizer-index.sh INDEX_JSON PINNING_SOURCE [PUBLISHED_URL]
# Exit 0 and print "publish" or "unchanged" (already published, byte for byte); else fail.
set -euo pipefail
INDEX=$1; SRC=$2; URL=${3:-}
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
python3 -I - "$INDEX" "$SRC" "$W" <<'PY'
import base64, json, sys
index_path, src_path, w = sys.argv[1:4]
env = json.load(open(index_path))
assert set(env) == {"index", "signature", "key"}, sorted(env)
pinned = open(src_path).read().split('PINNED_INDEX_KEY: &str = "')[1][:64]
assert env["key"].lower() == pinned, "signed by a key this source doesn't pin"
domain = open(src_path).read().split('INDEX_DOMAIN: &[u8] = b"')[1].split('"')[0].replace("\\0", "\0").encode()
body = base64.b64decode(env["index"], validate=True)
idx = json.loads(body)
assert idx["format"] == 1 and isinstance(idx["sequence"], int), "not an index"
open(f"{w}/msg", "wb").write(domain + body)
open(f"{w}/sig", "wb").write(bytes.fromhex(env["signature"]))
# Ed25519 SubjectPublicKeyInfo (DER): fixed header + the 32-byte key.
open(f"{w}/key.der", "wb").write(bytes.fromhex("302a300506032b6570032100" + pinned))
open(f"{w}/seq", "w").write(str(idx["sequence"]))
print(f"index #{idx['sequence']}: {len(idx['models'])} tokenizers, key {pinned[:12]}…")
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
  *) echo "couldn't read the published index ($code)" >&2; exit 1 ;;
esac
PUB=$(python3 -I -c 'import base64,json,sys; print(json.loads(base64.b64decode(json.load(open(sys.argv[1]))["index"]))["sequence"])' "$W/published")
if [ "$SEQ" -gt "$PUB" ]; then echo "#$SEQ replaces published #$PUB"; echo "publish"
elif cmp -s "$INDEX" "$W/published"; then echo "#$SEQ is already published"; echo "unchanged"
else echo "#$SEQ is not above published #$PUB: gateways would never take it" >&2; exit 1
fi
