# OpenHydra — Comprehensive Bug Fix & Enhancement Prompt

## Project Overview

OpenHydra is a decentralized peer-to-peer LLM inference network written in
Python 3.11+. A coordinator assembles a gRPC pipeline across peer nodes that
each hold a model shard; activations flow sequentially through the pipeline
and the final stage emits tokens. The codebase uses:

- gRPC (`grpcio`) for peer communication
- An HTTP DHT bootstrap service (`dht/bootstrap.py`, `dht/node.py`) for peer
  discovery
- A file-backed barter/HYDRA token economy (`economy/`)
- A health scorer, reputation system, and verification layer (`coordinator/`)
- Per-hop X25519 + AES-256-GCM encryption and onion routing (`peer/crypto.py`)
- PyTorch runtime for real inference (`peer/model_shard.py` `PyTorchRuntime`)

All changes must keep the existing test suite green (`pytest -q`; currently
`160 passed, 6 skipped`). Add new tests for every changed behaviour. All
Python must use `from __future__ import annotations` and typed signatures.

---

## P0 — Security Critical

### P0-a: Per-peer independent X25519 key pairs

**Problem:** `peer/crypto.py` derives every peer's private key from a single
shared `shared_secret_seed` known to the coordinator and all peers. Any node
that obtains the seed can decrypt every other peer's traffic.

**Current code path:**
```python
# peer/crypto.py
def _peer_private_key(peer_id: str, shared_secret_seed: str):
    material = hashlib.sha256(
        f"x25519:{shared_secret_seed}:{peer_id}".encode()
    ).digest()
    # clamped and returned as X25519PrivateKey
```

**Required changes:**

1. `peer/server.py` — On startup, load or generate a persistent keypair:
   - Path: `{--data-dir}/.openhydra/peer_identity/{peer_id}.key` (mode 0600)
   - If file absent, generate a new `X25519PrivateKey`, serialize with
     `serialization.Encoding.Raw` / `serialization.PrivateFormat.Raw`, write
     with `os.open(..., os.O_WRONLY|os.O_CREAT|os.O_TRUNC, 0o600)`
   - Expose the corresponding public key bytes (32 bytes, hex-encoded) as
     `peer_public_key` in the DHT announce payload and gRPC `PeerStatusResponse`

2. `dht/bootstrap.py` — Accept, store, and return `peer_public_key` in
   announce/lookup payloads without validation (it is the peer's self-asserted
   public key; trust is established via TLS + reputation, not key authority)

3. `coordinator/path_finder.py` — Add `public_key_hex: str = ""` to
   `PeerEndpoint`; populate from DHT lookup results

4. `peer/crypto.py`:
   - Add `build_activation_envelope_with_pubkey(activation, *, peer_public_key_bytes, ...)`
     that accepts a raw 32-byte public key instead of deriving it from the seed
   - Keep `build_activation_envelope` (seed-based) only as a legacy/test shim
     behind a `_LEGACY_SEED_MODE` guard — do not call it in production paths
   - Add `decrypt_activation_envelope_with_keyfile(*, keyfile_path, ...)` for
     peer-side decryption that loads the private key from the keyfile
   - `build_onion_route_envelope` must similarly accept a `List[bytes]` of
     per-peer 32-byte public keys instead of deriving from the seed

5. `coordinator/chain.py` — `InferenceChain._request_stage()` and `run()`:
   - Remove `advanced_encryption_seed` usage for key derivation
   - Use `peer.public_key_hex` (decoded from hex to bytes) for encryption
   - If `public_key_hex` is empty and encryption is enabled, raise
     `RuntimeError("encryption_required_but_no_peer_public_key")`

6. Update `scripts/gen_dev_certs.sh` to also generate dev peer identity
   keyfiles for `peer-a`, `peer-b`, `peer-c`, `coordinator-client` so
   local dev still works without a running peer daemon

7. Add tests:
   - `tests/test_peer_identity_keypair.py` — keypair generation, persistence,
     reload; verify different peer IDs produce different keys
   - `tests/test_crypto_pubkey_encrypt.py` — round-trip encrypt/decrypt using
     the new pubkey API; verify the seed-based API is not called

---

### P0-b: `openhydra_secrets.py` — restrict environment variable exposure

**Problem:** `load_secret_store` dumps the *entire* `os.environ` into the
`SecretStore`, making arbitrary env vars (e.g. `AWS_SECRET_ACCESS_KEY`, `PATH`)
queryable via `store.get()`.

**File:** `openhydra_secrets.py`

**Current code (lines 63-64):**
```python
for key, value in os.environ.items():
    values[key] = str(value)
```

**Required change:**

Define an explicit allowlist of env var names that OpenHydra is permitted to
read from the environment. Environment variables outside this list must not be
included in the `SecretStore`:

```python
_ALLOWED_ENV_KEYS: frozenset[str] = frozenset({
    "OPENHYDRA_ADVANCED_ENCRYPTION_SEED",
    "OPENHYDRA_GEO_CHALLENGE_SEED",
    "OPENHYDRA_DEPLOYMENT_PROFILE",
    "OPENHYDRA_SECRETS_FILE",
    "OPENHYDRA_DHT_BOOTSTRAP_URLS",
    "OPENHYDRA_HYDRA_LEDGER_PATH",
    "OPENHYDRA_TLS_CERT_PATH",
    "OPENHYDRA_TLS_KEY_PATH",
    "OPENHYDRA_TLS_CA_PATH",
})
```

Replace the broad env loop with:
```python
for key in _ALLOWED_ENV_KEYS:
    value = os.environ.get(key)
    if value is not None:
        values[key] = str(value)
```

File-sourced values are still overridable by the allowlisted env vars (same
precedence as before). Update `tests/test_openhydra_secrets.py` to assert
that `PATH`, `HOME`, and a fake `AWS_SECRET_ACCESS_KEY` are not present in
the returned `SecretStore`.

---

### P0-c: `generate_identity()` — produce real key material

**File:** `peer/crypto.py` (lines 32-38)

**Problem:** Returns a hex string as `private_key`, not actual key material.

**Required change:** When `cryptography` is available, generate a real
`X25519PrivateKey`, serialize it to raw bytes, hex-encode for the `Identity`
dataclass fields. When `cryptography` is unavailable, retain the sha256 stub
but mark it `_LEGACY_STUB = True` on the returned object. No caller currently
uses `generate_identity()` in a production path, but it must not silently
produce invalid key material.

---

## P1 — Correctness Bugs

### P1-a: TOCTOU race in `slash_stake`

**File:** `coordinator/ledger_bridge.py` `OpenHydraLedgerBridge.slash_stake()`

**Problem:** The lock is released before calling `_external_stake_slasher`,
then re-acquired to update `_total_burned` and `_total_slashed`. A concurrent
slash of the same peer races through this window.

**Required fix:**

1. Capture the slasher callable reference *inside* the first lock section
2. Release the lock, call the external slasher (unchanged — external I/O
   must not block the lock)
3. Re-acquire the lock for the *single* final write:
   - Update `_total_burned` by `local_slashed + min(remaining, external_slashed)`
   - Update `_total_slashed` by the same combined total
   - Do NOT re-read `acct.staked_balance` after re-acquiring — use the
     values captured in the first locked section to build the return dict

Add a test in `tests/test_hydra_ledger_bridge_bme.py` that runs two
concurrent `slash_stake` calls on the same pubkey using `threading.Thread`
and asserts that `_total_burned` equals the sum of both slash amounts with
no double-count.

---

### P1-b: `FileCreditLedger.balance()` must not write on read

**File:** `economy/barter.py`

**Problem:** `balance()` calls `self.ledger.balance()` which calls `_decay()`
and then `FileCreditLedger.balance()` calls `_save()`. A read path should
not write.

**Required fix:**

- Remove the `_save()` call from `FileCreditLedger.balance()`
- Decay is still applied in memory on every balance read (this is correct)
- The decayed state will be persisted on the next `earn()` or `spend()` call
- Add a `FileCreditLedger.flush()` method for explicit persistence (called
  on coordinator shutdown)
- Update `tests/test_barter.py` to assert that calling `balance()` 1000 times
  on a mocked filesystem results in zero file writes

---

### P1-c: `enforce_operator_caps` fallback silently defeats the cap

**File:** `coordinator/concentration_guard.py` `enforce_operator_caps()`

**Problem:** When the capped selection yields fewer peers than
`pipeline_width`, the fallback loop appends any remaining peer regardless
of operator, defeating the diversity guarantee with no signal to the caller.

**Required fix:**

- Remove the fallback loop entirely. Return the capped list even if it is
  shorter than `pipeline_width`.
- Add a return type annotation `list[PeerEndpoint]` and a docstring noting
  that the returned list may be shorter than `pipeline_width` when operator
  diversity constraints cannot be satisfied.
- Callers (`assemble_pipeline`, `coordinator/engine.py`) must handle a
  short list: if `len(result) < pipeline_width`, emit a log warning at
  `WARNING` level: `"operator_cap_enforced: pipeline assembled with
  {len(result)}/{pipeline_width} peers"` and proceed with the reduced
  pipeline (the degradation policy already handles insufficient-peer routing
  above this layer).
- Update `tests/test_concentration_guard.py` to add a case where all
  candidates are from one operator: assert the returned list length is
  `max(1, floor(pipeline_width * max_fraction))`, not `pipeline_width`.

---

### P1-d: `HealthScorer.score()` must not create entries as a side effect

**File:** `coordinator/health_scorer.py`

**Problem:** `_peer()` inserts a blank `PeerHealthStats` record when called
on an unknown `peer_id`. `score()` calls `_peer()`, so querying the score
of an unseen peer registers it in the store and triggers a disk write.

**Required fix:**

- Add a private `_peer_or_none(peer_id: str) -> PeerHealthStats | None`
  that returns `None` if the peer is unknown without inserting
- Add a private `_default_stats(peer_id: str) -> PeerHealthStats` that
  returns a detached default record (never stored)
- `score()` uses `_peer_or_none()` and falls back to `_default_stats()`
  for the calculation; it does not call `_peer()` and does not call `_save()`
- `scores()` follows the same pattern
- `record_ping`, `record_inference`, `record_verification` continue to use
  `_peer()` (write path, correct to insert)
- Update `tests/test_health_scorer.py` to assert that calling `score()` on
  an unknown peer ID results in zero disk writes and does not add the peer
  to `scorer.snapshot()`

---

### P1-e: Remove double-encode in compression telemetry

**File:** `coordinator/chain.py` `InferenceChain.run()` (around line 398)

**Problem:**
```python
compression_latent += len(self._autoencoder.encode(stage_input))
```
This re-encodes `stage_input` purely to measure latent length, wasting CPU.
The gRPC response already carries `compression_latent_dim` in the proto.

**Required fix:**

- In `_request_stage()`, read `response.compression_latent_dim` from the
  `ForwardResponse` proto and return it alongside `activation` and
  `latency_ms` (change return type to a small `_StageResult` dataclass or
  named tuple with fields `activation`, `latency_ms`, `latent_dim`)
- In `run()`, accumulate `compression_latent += stage_result.latent_dim`
  instead of re-encoding
- If `response.compression_latent_dim == 0` (non-compressing peer or old
  proto), fall back to `len(activation)` (identity — no compression)
- Update `tests/test_chain_compression.py` to mock a peer that returns a
  known `compression_latent_dim` and assert the telemetry matches without
  the autoencoder being called a second time

---

### P1-f: `DegradationDecision` — clarify `available` vs `degraded`

**File:** `coordinator/degradation.py`

**Problem:** When `allow_degradation=False` and peers are insufficient,
`degraded=False` is returned with `reason="insufficient_peers"`. Callers
cannot distinguish "serving successfully" from "cannot serve" without
string-matching on `reason`.

**Required fix:**

Add an `available: bool` field to `DegradationDecision`:
- `available=True` means the network can serve *some* model (requested or
  fallback)
- `available=False` means no viable model can be served

Update all five return sites in `DegradationPolicy.select()`:

| reason | degraded | available |
|---|---|---|
| `ok` | False | True |
| `unknown_model` | False | True (pass-through) |
| `insufficient_peers` (degradation disabled) | False | **False** |
| `insufficient_peers` (fallback found) | True | True |
| `no_viable_fallback` | False | **False** |

Update all callers in `coordinator/engine.py` that check `decision.degraded`
to also check `decision.available`; if `not decision.available`, return an
HTTP 503 with body `{"error": "no_viable_model", "reason": decision.reason}`.

Update `tests/test_degradation.py` to cover `available=False` cases.

---

## P2 — Performance

### P2-a: Batch `HealthScorer` disk writes

**File:** `coordinator/health_scorer.py`

**Problem:** `_save()` is called on every `record_ping`, `record_inference`,
and `record_verification` event — a synchronous full-JSON-serialize-and-write
on each gRPC hop.

**Required fix:**

1. Add a `_dirty: bool = False` flag to `HealthScorer`
2. Add a `_flush_interval_s: float = 5.0` constructor parameter
3. Replace all `self._save()` calls in the three `record_*` methods with
   `self._dirty = True`
4. Start a daemon background thread in `__init__` that sleeps
   `_flush_interval_s`, then calls `self._save()` if `self._dirty`, then
   clears the flag. Use a `threading.Event` for clean shutdown.
5. Add `HealthScorer.flush()` for explicit persistence (call on coordinator
   shutdown and in tests)
6. Add `HealthScorer.close()` that signals the background thread to stop,
   calls `flush()`, and joins the thread
7. Update `tests/test_health_scorer.py` to:
   - Assert that 1000 `record_ping` calls result in ≤ 2 disk writes (one
     for each flush cycle in a mocked timer scenario)
   - Test `flush()` and `close()` directly

---

### P2-b: `InMemoryDhtNode` — background pruner, lazy per-record expiry on reads

**File:** `dht/node.py`

**Problem:** `_prune()` iterates all DHT keys inside the global lock on
every `put`, `get`, `keys`, and `stats` call. Under write pressure with many
model keys this is O(all-keys) inside a mutex.

**Required fix:**

1. Remove `self._prune()` calls from `put`, `get`, `keys`, and `stats`
2. In `get()`: filter expired records at read time without iterating other
   keys — just `[r for r in self._store.get(key, []) if r.expires_at >= now]`
   (still inside the lock, but only touches the single key being read)
3. In `put()`: same lazy filter on the target key only before appending
4. Add a `start_background_pruner(interval_s: float = 30.0)` method that
   starts a daemon thread calling `_prune()` (full sweep) every
   `interval_s`. Call this from `dht/bootstrap.py` after the DHT is
   created.
5. Add `stop_background_pruner()` that sets a stop event and joins the
   thread; call from bootstrap shutdown path.
6. `keys()` and `stats()` no longer prune — they reflect the current store
   snapshot (stale records are TTL'd at read time by callers anyway)
7. Update `tests/test_dht_node.py` to verify that a `get()` after TTL
   expiry returns an empty list without having called `_prune()` explicitly.

---

## P3 — Bootstrap Infrastructure: Multiple Bootstrap URLs & Linode Setup

### P3-a: Coordinator accepts a list of bootstrap URLs

**Problem:** `EngineConfig.dht_url` is a single `str | None`. For
resilience, coordinators must be able to query multiple bootstrap nodes and
merge results.

**Files:** `coordinator/engine.py`, `coordinator/path_finder.py`,
`coordinator/api_server.py`, `coordinator/client_cli.py`

**Required changes:**

1. `EngineConfig`: rename `dht_url: str | None` to
   `dht_urls: list[str] = field(default_factory=list)`. Keep a deprecated
   `dht_url` property that returns `dht_urls[0] if dht_urls else None` for
   backwards compat with tests.

2. `coordinator/path_finder.py` `load_peers_from_dht()`:
   - Accept `dht_urls: list[str]` (or `dht_url: str` for backwards compat)
   - Query all URLs concurrently with `concurrent.futures.ThreadPoolExecutor`
     and a per-URL timeout of `dht_lookup_timeout_s`
   - Merge results, deduplicating by `peer_id`, preferring the record with
     the more recent `last_seen` timestamp
   - If all URLs fail, fall back to cached peers (existing behaviour)
   - If some URLs succeed and some fail, log a WARNING per failed URL and
     proceed with merged results

3. `coordinator/client_cli.py`: change `--dht-url` to accept multiple values
   (`nargs="+"` or comma-separated) and populate `dht_urls`. Keep
   `--dht-url` as the flag name for backwards compat.

4. `coordinator/api_server.py`: same multi-URL support.

5. `peer/server.py` and `peer/dht_announce.py`: update `--dht-url` to also
   accept multiple values; announce to all configured bootstrap URLs in
   parallel, log failures per URL without aborting the announce loop.

6. Add `tests/test_path_finder_dht.py` cases for multi-URL merge and
   partial failure.

---

### P3-b: Linode Nanode 1 GB bootstrap deployment

The Linode Nanode 1 GB ($5/mo) is used **only** as the DHT bootstrap
tracker. It runs `dht.bootstrap` and nginx for TLS termination. It does
**not** run a peer or load any model weights.

Create the following new files:

**`ops/bootstrap/bootstrap.service`** (systemd unit):
```ini
[Unit]
Description=OpenHydra DHT Bootstrap Node
After=network.target

[Service]
Type=simple
User=openhydra
WorkingDirectory=/opt/openhydra
ExecStart=/opt/openhydra/.venv/bin/python -m dht.bootstrap \
    --host 0.0.0.0 \
    --port 8468 \
    --deployment-profile prod \
    --geo-challenge-seed ${OPENHYDRA_GEO_CHALLENGE_SEED}
EnvironmentFile=/etc/openhydra/secrets.env
Restart=always
RestartSec=5
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

**`ops/bootstrap/nginx-bootstrap.conf`** (nginx vhost — TLS termination):
```nginx
server {
    listen 443 ssl http2;
    server_name bootstrap.example.com;

    ssl_certificate     /etc/letsencrypt/live/bootstrap.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/bootstrap.example.com/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;

    location / {
        proxy_pass         http://127.0.0.1:8468;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_read_timeout 10s;
    }
}
server {
    listen 80;
    server_name bootstrap.example.com;
    return 301 https://$host$request_uri;
}
```

**`ops/bootstrap/setup_nanode.sh`** — idempotent setup script:
```bash
#!/usr/bin/env bash
# Run as root on a fresh Linode Nanode Ubuntu 22.04 image.
# This machine is the DHT bootstrap tracker only — it does NOT run a peer
# or load any model weights.
# Usage: bash setup_nanode.sh bootstrap.example.com

set -euo pipefail
DOMAIN="${1:?usage: setup_nanode.sh <domain>}"

apt-get update -qq
apt-get install -y python3.11 python3.11-venv nginx certbot python3-certbot-nginx git ufw

useradd -r -s /bin/false -d /opt/openhydra openhydra || true
mkdir -p /opt/openhydra /etc/openhydra
chown openhydra:openhydra /opt/openhydra

# Clone or update repo
if [ -d /opt/openhydra/.git ]; then
    git -C /opt/openhydra pull --ff-only
else
    git clone https://github.com/your-org/openhydra.git /opt/openhydra
fi

python3.11 -m venv /opt/openhydra/.venv
# Bootstrap node only needs core deps — no torch, no transformers
/opt/openhydra/.venv/bin/pip install grpcio protobuf cryptography --quiet

# Secrets file (operator fills in values)
if [ ! -f /etc/openhydra/secrets.env ]; then
    cat > /etc/openhydra/secrets.env <<EOF
OPENHYDRA_GEO_CHALLENGE_SEED=REPLACE_WITH_STRONG_RANDOM_SECRET
EOF
    chmod 600 /etc/openhydra/secrets.env
    chown root:root /etc/openhydra/secrets.env
    echo "⚠ Edit /etc/openhydra/secrets.env before starting the service"
fi

# TLS via Let's Encrypt
certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m admin@"$DOMAIN" || true

# Firewall — bootstrap only needs 22 (SSH), 80 (redirect), 443 (HTTPS)
# Peers connect to bootstrap over HTTPS; no direct gRPC exposure needed
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

# Install and enable service
cp /opt/openhydra/ops/bootstrap/bootstrap.service /etc/systemd/system/
cp /opt/openhydra/ops/bootstrap/nginx-bootstrap.conf \
   /etc/nginx/sites-available/openhydra-bootstrap.conf
ln -sf /etc/nginx/sites-available/openhydra-bootstrap.conf \
        /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx

systemctl daemon-reload
systemctl enable openhydra-bootstrap
echo "Bootstrap node ready. Start with: systemctl start openhydra-bootstrap"
```

**`ops/bootstrap/README.md`** — operator guide covering:

- **What this machine does:** runs the DHT bootstrap tracker (`dht.bootstrap`)
  and nginx for TLS. It holds no model weights and performs no inference.
  Memory footprint: ~150 MB (nginx + Python + DHT state). A Linode Nanode
  1 GB is sufficient and is the recommended host for this role.
- Prerequisites (Linode Nanode Ubuntu 22.04, domain name pointed at the IP)
- Running `setup_nanode.sh`
- Editing `/etc/openhydra/secrets.env`
- Starting the service: `systemctl start openhydra-bootstrap`
- Health check: `curl https://bootstrap.example.com/health`
- Configuring coordinators and peers with multiple bootstrap URLs:

  ```bash
  # Coordinator querying two bootstrap nodes:
  python -m coordinator.api_server \
    --dht-url https://bootstrap1.example.com \
    --dht-url https://bootstrap2.example.com \
    --model-catalog-path ./models.catalog.json

  # Peer announcing to two bootstrap nodes:
  python -m peer.server \
    --peer-id peer-a --host <YOUR_HOST> --port 50051 \
    --shard-index 0 --total-shards 1 \
    --dht-url https://bootstrap1.example.com \
    --dht-url https://bootstrap2.example.com \
    --runtime-backend pytorch_cpu \
    --runtime-model-id Qwen/Qwen3.5-0.8B \
    --quantization-mode int4
  ```

- Horizontal scaling: run 2-3 Nanode instances for redundancy; configure
  all peers and coordinators to announce/query all of them. Each bootstrap
  is independent (no inter-bootstrap sync is needed — peers announce to all
  and coordinators merge results).

---

## P4 — SQLite Ledger for HydraCoins

Replace the file-backed JSON economy stores with SQLite across both the
barter credit ledger and the HYDRA token economy. SQLite is chosen because
HydraCoins are internal anti-abuse tokens used only within OpenHydra; the
coordinator is the trusted authority for balances. A blockchain is not
required. SQLite with WAL mode provides atomic writes, concurrent reads,
crash recovery, and automatic migration from the existing JSON files — with
no new dependencies beyond the Python stdlib.

### P4-a: SQLite barter credit ledger

**File:** `economy/barter.py`

Replace `FileCreditLedger` with `SqliteCreditLedger` (keep `FileCreditLedger`
as a deprecated alias that instantiates `SqliteCreditLedger` for one release;
all internal usages must switch):

```python
class SqliteCreditLedger:
    """
    SQLite-backed Tier 2 barter credit ledger.

    Schema:
        CREATE TABLE IF NOT EXISTS barter_credits (
            peer_id       TEXT PRIMARY KEY,
            balance       REAL NOT NULL DEFAULT 0.0,
            last_decay_ts REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS barter_meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """
```

Requirements:
- Use `sqlite3` from the stdlib (no third-party ORM)
- Enable WAL mode: `PRAGMA journal_mode=WAL`
- Enable `check_same_thread=False` with an explicit `threading.Lock` for
  write serialization (reads are concurrent-safe in WAL mode)
- Decay is applied lazily per-peer at read/write time (same algorithm as
  current `CreditLedger._decay()` but scoped to a single row)
- `earn()` and `spend()` use `BEGIN IMMEDIATE` transactions
- `balance()` is a plain `SELECT` — **no write**
- Provide `flush()` (calls `connection.commit()`) and `close()` (commits
  and closes the connection)
- Automatic schema migration: if a JSON file exists at the equivalent path
  (replacing `.db` suffix with `.json`), migrate all balances on first open
  and rename the JSON to `.json.migrated`

### P4-b: SQLite HYDRA token economy

**File:** `economy/token.py`

Apply the same pattern to `FileHydraTokenEconomy`:

```sql
CREATE TABLE IF NOT EXISTS hydra_accounts (
    peer_id         TEXT PRIMARY KEY,
    balance         REAL NOT NULL DEFAULT 0.0,
    stake           REAL NOT NULL DEFAULT 0.0,
    rewards_earned  REAL NOT NULL DEFAULT 0.0,
    slashed_total   REAL NOT NULL DEFAULT 0.0
);
CREATE TABLE IF NOT EXISTS hydra_channels (
    channel_id  TEXT PRIMARY KEY,
    payer_id    TEXT NOT NULL,
    payee_id    TEXT NOT NULL,
    escrow      REAL NOT NULL DEFAULT 0.0,
    payer_spent REAL NOT NULL DEFAULT 0.0,
    status      TEXT NOT NULL DEFAULT 'open',
    created_at  REAL NOT NULL,
    expires_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS hydra_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

Requirements:
- Same WAL + lock pattern as the barter ledger
- Channel operations use `BEGIN IMMEDIATE` transactions
- `GET /v1/account/balance` and `GET /v1/hydra/account` query the DB
  directly (no in-memory cache needed; WAL reads are fast)
- Supply cap enforcement via a single row in `hydra_meta`:
  `INSERT OR REPLACE INTO hydra_meta VALUES ('total_supply', ?)` on every
  mint/burn; read back and compare before minting
- Migrate from `.json` on first open (same rename pattern as barter)

### P4-c: Wire up in `coordinator/engine.py`

- `EngineConfig.ledger_path` — update default from
  `.openhydra/credits.json` to `.openhydra/credits.db`
- `EngineConfig.hydra_token_ledger_path` — update default from
  `.openhydra/hydra_tokens.json` to `.openhydra/hydra_tokens.db`
- Replace instantiation of `FileCreditLedger` with `SqliteCreditLedger`
- Replace instantiation of `FileHydraTokenEconomy` with the SQLite variant
- On coordinator shutdown (`api_server.py` SIGTERM handler), call
  `ledger.close()` and `token_economy.close()`

### P4-d: Tests

- `tests/test_barter.py` — add `SqliteCreditLedger` coverage for:
  earn/spend/balance, decay correctness, concurrent writes from 4 threads,
  migration from JSON
- `tests/test_ledger_store.py` — same for HYDRA token economy
- Use `tmp_path` pytest fixture for DB paths so tests are isolated

---

## P5 — Qwen3.5-0.8B as Default Model

### P5-a: Default model ID

**Files:** `peer/model_shard.py`, `peer/server.py`,
`coordinator/engine.py`, `coordinator/client_cli.py`,
`coordinator/api_server.py`

Change every default reference to the toy model where a real model will run:

| Location | Old default | New default |
|---|---|---|
| `ToyShardConfig.runtime_model_id` | `"gpt2"` | `"Qwen/Qwen3.5-0.8B"` |
| `peer/server.py` `--runtime-model-id` argparse default | `"gpt2"` | `"Qwen/Qwen3.5-0.8B"` |
| Example catalogue in README | toy placeholders | `openhydra-qwen3.5-0.8b` |

---

### P5-b: Architecture detection for Qwen

**File:** `peer/model_shard.py` `PyTorchRuntime._detect_architecture()`

Qwen models use a LLaMA-style transformer backbone (`model.layers`,
`model.embed_tokens`, `model.norm`) but have Qwen-specific attention modules.
The existing `model.layers` branch should handle the forward pass, but add
explicit detection before the generic branch:

```python
# In _detect_architecture, before the generic model.layers branch:
if hasattr(model, "model") and hasattr(model.model, "layers"):
    arch_name = type(model).__name__.lower()
    if "qwen" in arch_name or hasattr(model.model, "rotary_emb"):
        # Qwen / LLaMA family — use the existing LLaMA path
        return _DecoderArchitecture(
            family="qwen_llama",
            layers=tuple(model.model.layers),
            embed_tokens=model.model.embed_tokens,
            position_embeddings=None,
            final_norm=model.model.norm,
            rotary_emb=getattr(model.model, "rotary_emb", None),
        )
```

---

### P5-c: Tokenizer EOS detection for Qwen

**File:** `peer/model_shard.py` — wherever the EOS check occurs in the
autoregressive generation loop.

Qwen's tokenizer defines EOS as a list of token IDs, not a single integer.
Fix the EOS check to handle both:

```python
# Replace:
if next_token_id == tokenizer.eos_token_id:
    break

# With:
eos_ids = tokenizer.eos_token_id
if isinstance(eos_ids, int):
    eos_ids = {eos_ids}
else:
    eos_ids = set(eos_ids or [])
if next_token_id in eos_ids:
    break
```

Also add `trust_remote_code=True` to both `AutoModelForCausalLM.from_pretrained`
and `AutoTokenizer.from_pretrained` calls, gated on a new
`ToyShardConfig.runtime_trust_remote_code: bool` field that defaults to
`True` when `runtime_model_id` contains `"Qwen"` and `False` for `"gpt2"` and
other known-safe model IDs.

---

### P5-d: Default model catalogue

**`models.catalog.json`** (create if absent, update if present):
```json
[
  {
    "model_id": "openhydra-qwen3.5-0.8b",
    "required_peers": 1,
    "hf_model_id": "Qwen/Qwen3.5-0.8B"
  }
]
```

Add `hf_model_id: str = ""` to the `ModelAvailability` dataclass so the
coordinator can resolve a catalogue entry to its HuggingFace model ID.
Update `coordinator/engine.py` to pass `hf_model_id` through to the peer
config when building `ToyShardConfig`, so peers know which HF model to load
for a given `model_id`.

---

### P5-e: README quick-start update

Update `README.md` to replace the toy model quick-start with Qwen3.5-0.8B.
The bootstrap node (Linode Nanode) and peer nodes are always separate
machines. Peers run on contributor hardware — the operator's own machine,
a desktop with a GPU, or any cloud instance with sufficient RAM. The
Nanode is only for the tracker.

**Single-node example (contributor's own machine):**
```bash
# Start the peer on your own machine (not the Nanode):
python -m peer.server \
  --peer-id peer-a \
  --host 0.0.0.0 \
  --port 50051 \
  --shard-index 0 \
  --total-shards 1 \
  --runtime-backend pytorch_cpu \
  --runtime-model-id Qwen/Qwen3.5-0.8B \
  --quantization-mode int4 \
  --dht-url https://bootstrap1.example.com \
  --dht-url https://bootstrap2.example.com

# Start the coordinator on your own machine or a separate server:
python -m coordinator.api_server \
  --dht-url https://bootstrap1.example.com \
  --dht-url https://bootstrap2.example.com \
  --model-catalog-path ./models.catalog.json \
  --allow-degradation-default
```

**Peer hardware requirements for Qwen3.5-0.8B:**

| Quantization | Model RAM | Min host RAM | Notes |
|---|---|---|---|
| fp32 | ~3.2 GB | 6 GB | Any modern desktop/laptop |
| int8 | ~1.6 GB | 4 GB | Any modern desktop/laptop |
| int4 | ~0.8 GB | 2 GB | Raspberry Pi 5, older laptops |

Peers run on contributor-owned hardware. The Linode Nanode is the bootstrap
tracker only and has no model weight requirements.

---

### P5-f: Gated integration test

**`tests/test_real_qwen_generation.py`** — gated behind
`OPENHYDRA_RUN_REAL_TENSOR_TEST=1`:

```python
import os
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("OPENHYDRA_RUN_REAL_TENSOR_TEST", "0") != "1",
    reason="real tensor test gated",
)

MODEL_ID = os.environ.get("OPENHYDRA_PYTORCH_TEST_MODEL", "Qwen/Qwen3.5-0.8B")


def test_qwen_single_shard_generates_text():
    """Single peer, single shard, Qwen3.5-0.8B, int4 quantization.
    Asserts output is non-empty and terminates without hitting max_tokens."""
    ...


def test_qwen_eos_terminates():
    """Verify EOS detection works with Qwen's multi-token EOS id set.
    Asserts the generation loop breaks before max_tokens on a short prompt."""
    ...


def test_qwen_trust_remote_code_flag():
    """Verify trust_remote_code=True is passed for Qwen and not for gpt2."""
    ...
```

---

## Cross-cutting Requirements

1. **No regressions:** `pytest -q` must report `≥ 160 passed, 6 skipped`
   after all changes (likely more passes with new tests added).
2. **Backwards compatibility:** All existing CLI flags remain valid; new
   flags are purely additive.
3. **`from __future__ import annotations`** at the top of every modified file.
4. **Type annotations** on all new public functions and methods.
5. **Logging:** Use `logging.getLogger(__name__)` throughout — no bare
   `print()` statements.
6. **No new mandatory dependencies** beyond what is already in
   `pyproject.toml` (`sqlite3` is stdlib; `cryptography>=42` already
   pinned; `torch` and `transformers` are already optional runtime deps).
7. **`trust_remote_code=True`** must only be passed to HuggingFace when
   `ToyShardConfig.runtime_trust_remote_code` is `True`; it must default
   to `True` for Qwen model IDs and `False` for `gpt2` and other
   known-safe identifiers.
8. **Bootstrap node isolation:** No code path in `dht/bootstrap.py` or
   `ops/bootstrap/` should import `torch`, `transformers`, or any inference
   dependency. The bootstrap node must be installable with only
   `grpcio protobuf cryptography` (no ML stack).
