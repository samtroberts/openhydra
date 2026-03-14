#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-.openhydra/certs}"
mkdir -p "$OUT_DIR"

CA_KEY="$OUT_DIR/ca.key.pem"
CA_CERT="$OUT_DIR/ca.cert.pem"

if [[ ! -f "$CA_KEY" || ! -f "$CA_CERT" ]]; then
  openssl genrsa -out "$CA_KEY" 2048 >/dev/null 2>&1
  openssl req -x509 -new -nodes -key "$CA_KEY" -sha256 -days 3650 \
    -subj "/CN=OpenHydra Dev CA" -out "$CA_CERT" >/dev/null 2>&1
fi

make_cert() {
  local name="$1"
  local key="$OUT_DIR/${name}.key.pem"
  local csr="$OUT_DIR/${name}.csr.pem"
  local cert="$OUT_DIR/${name}.cert.pem"
  local ext="$OUT_DIR/${name}.ext"

  openssl genrsa -out "$key" 2048 >/dev/null 2>&1
  openssl req -new -key "$key" -subj "/CN=${name}" -out "$csr" >/dev/null 2>&1

  cat > "$ext" <<EXT
basicConstraints=CA:FALSE
subjectAltName=DNS:localhost,IP:127.0.0.1,DNS:${name}
extendedKeyUsage=serverAuth,clientAuth
keyUsage = digitalSignature, keyEncipherment
EXT

  openssl x509 -req -in "$csr" -CA "$CA_CERT" -CAkey "$CA_KEY" -CAcreateserial \
    -out "$cert" -days 825 -sha256 -extfile "$ext" >/dev/null 2>&1

  rm -f "$csr" "$ext"
}

make_cert peer-a
make_cert peer-b
make_cert peer-c
make_cert coordinator-client

echo "Generated development certificates in: $OUT_DIR"
echo "CA cert: $CA_CERT"
