# Codex Prompt — P0-a / P0-c: Per-Peer Independent X25519 Keypairs

## Context

OpenHydra is a decentralised P2P LLM inference network written in Python 3.11+.
Activations flow over gRPC from peer to peer; the coordinator encrypts each hop
using X25519 ECDH + HKDF-SHA256 + AES-256-GCM (`peer/crypto.py`).

**The critical security flaw that remains unfixed:**

All peer keys are derived from a single coordinator-known `shared_secret_seed`:

```python
# peer/crypto.py  (current — INSECURE)
def _peer_private_key(peer_id: str, shared_secret_seed: str):
    material = hashlib.sha256(f"x25519:{shared_secret_seed}:{peer_id}".encode()).digest()
    ...
    return x25519.X25519PrivateKey.from_private_bytes(bytes(key_bytes))
```

This means the coordinator (or anyone who learns the seed) can reconstruct every
peer's private key, defeating the purpose of hop-by-hop encryption entirely.

**`generate_identity()` is also a stub** that returns sha256 hex strings, not real
X25519 key material:

```python
# peer/crypto.py  (current — STUB)
def generate_identity(seed: str | None = None) -> Identity:
    material = seed or secrets.token_hex(32)
    private_key = hashlib.sha256(f"priv:{material}".encode()).hexdigest()
    public_key  = hashlib.sha256(f"pub:{private_key}".encode()).hexdigest()
    peer_id = public_key[:16]
    return Identity(peer_id=peer_id, public_key=public_key, private_key=private_key)
```

---

## Goals

1. **P0-a** — Make `generate_identity()` produce real X25519 key material.
   Add helpers to persist and reload the identity from a keyfile.

2. **P0-c** — Each peer generates its own independent X25519 keypair on first
   startup, persists it to disk, and publishes the raw public key (hex) in its
   DHT announcement.  The coordinator reads the per-peer public key from the
   DHT record and uses it for encryption instead of the shared seed.

Both the seed-based functions (`_peer_private_key`, `build_activation_envelope`,
`build_onion_route_envelope`, `decrypt_activation_envelope`,
`peel_onion_route_layer`) **must be kept** as a deprecated dev/test shim so that
existing tests that do not supply a keyfile continue to pass.

---

## Cross-cutting constraints

- Python 3.11+; type-annotated throughout; `from __future__ import annotations`.
- The `cryptography>=42` package is already a dependency.
- No new third-party packages.
- All public key bytes on the wire / in DHT records are **hex-encoded raw
  32-byte X25519 public keys** (64 lower-case hex chars).
- Private-key files are written in **PEM / PKCS8 format** with mode `0600`.
  On Windows `os.chmod` is a no-op — that is acceptable.
- Keep every existing public function signature valid (add keyword-only params
  with defaults where needed).
- All new code must have unit tests; existing tests must still pass.

---

## File-by-file changes

### 1. `peer/crypto.py`

#### 1a. Fix `generate_identity()`

```python
def generate_identity(seed: str | None = None) -> Identity:
    """Generate a real X25519 Identity.

    When *seed* is supplied the private key is deterministic (HKDF from seed)
    — useful for reproducible unit tests.  Without a seed a fresh random key
    is generated.

    Falls back to the legacy sha256 stub if *cryptography* is unavailable and
    emits a DeprecationWarning.
    """
```

Implementation notes:
- If `cryptography_available()`:
  - Without seed: `private_key = x25519.X25519PrivateKey.generate()`
  - With seed: derive 32 bytes via `HKDF(SHA256, length=32, salt=b"openhydra-identity", info=b"x25519-private-key").derive(seed.encode())`, clamp to X25519, then `x25519.X25519PrivateKey.from_private_bytes(material)`
  - Extract raw public key bytes (32 bytes, `Encoding.Raw / PublicFormat.Raw`)
  - `public_key = raw_pub_bytes.hex()`  — 64 lower-case hex chars
  - `private_key = raw_priv_bytes.hex()` — derive raw private bytes via `private_key_obj.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())`
  - `peer_id = hashlib.sha256(raw_pub_bytes).hexdigest()[:16]`
- If not available:
  - `import warnings; warnings.warn("cryptography not available; using insecure sha256 identity stub", DeprecationWarning, stacklevel=2)`
  - Use existing sha256 logic (unchanged)

#### 1b. Add keyfile helpers

```python
def save_identity_keyfile(identity: Identity, path: str | Path) -> None:
    """Write the private key to *path* in PKCS8 PEM format, mode 0600.

    Creates parent directories as needed.
    Raises RuntimeError if cryptography is unavailable.
    """

def load_identity_keyfile(path: str | Path) -> Identity:
    """Load an Identity from a PKCS8 PEM private-key file written by
    save_identity_keyfile().

    Raises FileNotFoundError if *path* does not exist.
    Raises RuntimeError if cryptography is unavailable.
    """

def load_or_create_identity_keyfile(path: str | Path) -> Identity:
    """Load existing identity from *path*, or generate a fresh one and save it.

    This is the primary entry point for peer startup.
    Raises RuntimeError if cryptography is unavailable.
    """
```

Implementation notes for `save_identity_keyfile`:
```python
path = Path(path)
path.parent.mkdir(parents=True, exist_ok=True)
priv_obj = x25519.X25519PrivateKey.from_private_bytes(bytes.fromhex(identity.private_key))
pem = priv_obj.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
)
path.write_bytes(pem)
os.chmod(path, 0o600)
```

Implementation notes for `load_identity_keyfile`:
```python
path = Path(path)
pem = path.read_bytes()
priv_obj = serialization.load_pem_private_key(pem, password=None)
# Verify it is an X25519 key
if not isinstance(priv_obj, x25519.X25519PrivateKey):
    raise ValueError(f"keyfile {path} does not contain an X25519 private key")
raw_priv = priv_obj.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
raw_pub  = priv_obj.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
peer_id  = hashlib.sha256(raw_pub).hexdigest()[:16]
return Identity(
    peer_id=peer_id,
    public_key=raw_pub.hex(),
    private_key=raw_priv.hex(),
)
```

#### 1c. Add public-key-based encryption helpers

Add the following **new** functions alongside the existing seed-based ones.
The existing seed-based functions remain **unchanged** (they are the dev/test shim).

```python
def build_activation_envelope_with_pubkey(
    activation: list[float],
    *,
    raw_public_key_bytes: bytes,   # 32-byte X25519 public key
    peer_id: str,
    request_id: str,
    stage_index: int,
    level: str = "standard",
) -> ActivationEnvelope:
    """Like build_activation_envelope() but uses a raw per-peer public key
    instead of deriving it from a shared seed."""
```

Refactor the inner encrypt loop out of `build_activation_envelope()` into a
private helper `_build_activation_envelope_inner(activation, peer_pub_obj, *, peer_id, request_id, stage_index, level)` and call it from both the new function and `build_activation_envelope()` (which calls `peer_static_public_key()` as before).

```python
def decrypt_activation_envelope_with_privkey(
    *,
    ciphertext: bytes,
    nonces: list[bytes] | tuple[bytes, ...],
    ephemeral_public_keys: list[bytes] | tuple[bytes, ...],
    private_key: Any,   # x25519.X25519PrivateKey
    peer_id: str,
    request_id: str,
    stage_index: int,
) -> list[float]:
    """Like decrypt_activation_envelope() but takes an X25519PrivateKey
    object directly instead of deriving one from a shared seed."""
```

Similarly for onion routing:

```python
def build_onion_route_envelope_with_pubkeys(
    route_peer_ids: list[str] | tuple[str, ...],
    *,
    peer_public_keys: dict[str, bytes],   # peer_id → 32-byte raw pubkey
    request_id: str,
) -> OnionRouteEnvelope:
    """Like build_onion_route_envelope() but takes per-peer public keys
    instead of a shared seed.

    Raises KeyError if any peer_id in route_peer_ids is absent from
    peer_public_keys.
    """

def peel_onion_route_layer_with_privkey(
    *,
    ciphertext: bytes,
    nonces: list[bytes] | tuple[bytes, ...],
    ephemeral_public_keys: list[bytes] | tuple[bytes, ...],
    private_key: Any,   # x25519.X25519PrivateKey
    peer_id: str,
    request_id: str,
    stage_index: int,
) -> OnionRouteLayer:
    """Like peel_onion_route_layer() but takes an X25519PrivateKey directly."""
```

Add a convenience accessor:

```python
def private_key_from_identity(identity: Identity) -> Any:
    """Return an x25519.X25519PrivateKey from an Identity's private_key hex field.
    Raises RuntimeError if cryptography is unavailable."""
    _require_crypto()
    return x25519.X25519PrivateKey.from_private_bytes(bytes.fromhex(identity.private_key))
```

---

### 2. `peer/dht_announce.py`

Add one field to the `Announcement` dataclass:

```python
@dataclass
class Announcement:
    ...
    peer_public_key: str = ""   # hex-encoded 32-byte X25519 public key; "" in dev/test mode
```

No other changes to this file.

---

### 3. `peer/server.py`

#### 3a. `PeerService.__init__()`

Add two new parameters (keyword-only after the existing positional ones):

```python
peer_public_key: str = "",
peer_private_key: Any = None,   # x25519.X25519PrivateKey | None
```

Store them:

```python
self.peer_public_key: str = str(peer_public_key or "")
self._peer_private_key = peer_private_key   # may be None in dev mode
```

#### 3b. `PeerService.Forward()` — use keyfile-based decryption

Import the new crypto helpers at the top of the file:

```python
from peer.crypto import (
    cryptography_available,
    decrypt_activation_envelope,
    decrypt_activation_envelope_with_privkey,
    peel_onion_route_layer,
    peel_onion_route_layer_with_privkey,
    sign_geo_challenge,
)
```

In the `Forward()` handler, replace the decryption calls with a dispatch:

```python
# Onion route peeling
if request.onion_route_ciphertext:
    if self._peer_private_key is not None:
        onion_layer = peel_onion_route_layer_with_privkey(
            ciphertext=bytes(request.onion_route_ciphertext),
            nonces=[bytes(item) for item in request.onion_route_nonces],
            ephemeral_public_keys=[bytes(item) for item in request.onion_route_ephemeral_public_keys],
            private_key=self._peer_private_key,
            peer_id=self.peer_id,
            request_id=request.request_id,
            stage_index=int(request.stage_index),
        )
    else:
        onion_layer = peel_onion_route_layer(
            ciphertext=bytes(request.onion_route_ciphertext),
            nonces=[bytes(item) for item in request.onion_route_nonces],
            ephemeral_public_keys=[bytes(item) for item in request.onion_route_ephemeral_public_keys],
            peer_id=self.peer_id,
            request_id=request.request_id,
            stage_index=int(request.stage_index),
            shared_secret_seed=self.advanced_encryption_seed,
        )
    ...

# Activation decryption
if request.encrypted_activation:
    if self._peer_private_key is not None:
        activation_in = decrypt_activation_envelope_with_privkey(
            ciphertext=bytes(request.encrypted_activation),
            nonces=[bytes(item) for item in request.encryption_nonces],
            ephemeral_public_keys=[bytes(item) for item in request.encryption_ephemeral_public_keys],
            private_key=self._peer_private_key,
            peer_id=self.peer_id,
            request_id=request.request_id,
            stage_index=int(request.stage_index),
        )
    else:
        activation_in = decrypt_activation_envelope(
            ciphertext=bytes(request.encrypted_activation),
            nonces=[bytes(item) for item in request.encryption_nonces],
            ephemeral_public_keys=[bytes(item) for item in request.encryption_ephemeral_public_keys],
            peer_id=self.peer_id,
            request_id=request.request_id,
            stage_index=int(request.stage_index),
            shared_secret_seed=self.advanced_encryption_seed,
        )
```

#### 3c. `serve()` — generate/load keyfile at startup

Add to `serve()` signature (keyword-only, with default):

```python
data_dir: str = ".openhydra",
```

After `hardware_profile = detect_hardware_profile()` and before `PeerService(...)`:

```python
from peer.crypto import load_or_create_identity_keyfile, private_key_from_identity

peer_identity = None
peer_public_key_hex = ""
peer_priv_key_obj = None
if cryptography_available():
    keyfile_path = Path(data_dir) / "peer_identity" / f"{peer_id}.key"
    try:
        peer_identity = load_or_create_identity_keyfile(keyfile_path)
        peer_priv_key_obj = private_key_from_identity(peer_identity)
        peer_public_key_hex = peer_identity.public_key
        logging.info(
            "peer %s loaded identity from %s (pubkey=%s...)",
            peer_id, keyfile_path, peer_public_key_hex[:16],
        )
    except Exception as exc:
        logging.warning(
            "peer %s: failed to load/create identity keyfile at %s (%s); "
            "falling back to seed-based encryption",
            peer_id, keyfile_path, exc,
        )
```

Pass to `PeerService(...)`:

```python
service = PeerService(
    ...
    peer_public_key=peer_public_key_hex,
    peer_private_key=peer_priv_key_obj,
)
```

#### 3d. `_announce_loop()` — publish public key

Add `peer_public_key: str = ""` parameter to `_announce_loop()`.

In the `Announcement(...)` constructor call inside `_announce_loop()`, add:

```python
peer_public_key=peer_public_key,
```

Pass `peer_public_key=service.peer_public_key` at the call site in `serve()`.

#### 3e. CLI argparser

Add `--data-dir` argument:

```python
parser.add_argument(
    "--data-dir",
    default=".openhydra",
    help="Directory for persistent peer state (identity keyfiles, ledger, etc.)",
)
```

Pass `data_dir=args.data_dir` through to `serve()`.

---

### 4. `coordinator/path_finder.py`

#### 4a. `PeerEndpoint` dataclass

Add one field (after `geo_penalty_score`):

```python
public_key_hex: str = ""   # hex-encoded 32-byte X25519 public key; "" in dev/test mode
```

#### 4b. `load_peers_from_dht()`

In the `PeerEndpoint(...)` constructor inside the loop over `peers_payload`, add:

```python
public_key_hex=str(item.get("peer_public_key", "") or ""),
```

#### 4c. `load_peer_config()`

In the `PeerEndpoint(...)` constructor inside the loop, add:

```python
public_key_hex=str(item.get("public_key_hex", "") or ""),
```

---

### 5. `coordinator/chain.py`

#### 5a. Imports

Add to the import from `peer.crypto`:

```python
from peer.crypto import (
    build_activation_envelope,
    build_activation_envelope_with_pubkey,
    build_onion_route_envelope,
    build_onion_route_envelope_with_pubkeys,
    required_layers_for_level,
    verify_privacy_audit_tag,
)
```

#### 5b. `InferenceChain._request_stage()` — per-peer pubkey encryption

Currently the activation encryption block is:

```python
if self.advanced_encryption_enabled and wire_activation:
    envelope = build_activation_envelope(
        wire_activation,
        peer_id=peer.peer_id,
        request_id=request_id,
        stage_index=stage_index,
        shared_secret_seed=self.advanced_encryption_seed,
        level=self.advanced_encryption_level,
    )
```

Replace with:

```python
if self.advanced_encryption_enabled and wire_activation:
    peer_pubkey_hex = str(getattr(peer, "public_key_hex", "") or "")
    if peer_pubkey_hex:
        try:
            raw_pub = bytes.fromhex(peer_pubkey_hex)
        except ValueError:
            raw_pub = b""
    else:
        raw_pub = b""

    if raw_pub:
        envelope = build_activation_envelope_with_pubkey(
            wire_activation,
            raw_public_key_bytes=raw_pub,
            peer_id=peer.peer_id,
            request_id=request_id,
            stage_index=stage_index,
            level=self.advanced_encryption_level,
        )
    else:
        # Dev/test fallback: seed-based (no per-peer keyfile available)
        envelope = build_activation_envelope(
            wire_activation,
            peer_id=peer.peer_id,
            request_id=request_id,
            stage_index=stage_index,
            shared_secret_seed=self.advanced_encryption_seed,
            level=self.advanced_encryption_level,
        )
```

#### 5c. `InferenceChain.run()` — per-peer pubkey onion routing

Locate where `build_onion_route_envelope()` is called.  Replace with:

```python
route_peer_ids = [p.peer_id for p in self.pipeline]
# Attempt per-peer pubkey onion routing
peer_pubkeys: dict[str, bytes] = {}
for p in self.pipeline:
    hex_key = str(getattr(p, "public_key_hex", "") or "")
    if hex_key:
        try:
            peer_pubkeys[p.peer_id] = bytes.fromhex(hex_key)
        except ValueError:
            pass

if len(peer_pubkeys) == len(self.pipeline):
    onion_envelope = build_onion_route_envelope_with_pubkeys(
        route_peer_ids,
        peer_public_keys=peer_pubkeys,
        request_id=request_id,
    )
else:
    # Dev/test fallback: seed-based
    onion_envelope = build_onion_route_envelope(
        route_peer_ids,
        request_id=request_id,
        shared_secret_seed=self.advanced_encryption_seed,
    )
```

---

### 6. `dht/bootstrap.py`

In `DhtBootstrapHandler._normalize_peer_record()`, add one entry to the returned
dict (after `"expert_router"`):

```python
"peer_public_key": str(payload.get("peer_public_key", "") or ""),
```

Also add basic validation (optional but recommended):

```python
peer_public_key = str(payload.get("peer_public_key", "") or "")
if peer_public_key and len(peer_public_key) != 64:
    # not a valid 32-byte hex key — silently drop it
    peer_public_key = ""
```

---

## Tests to add / update

### `tests/test_crypto.py`

Add:

```python
def test_generate_identity_real_keys():
    """generate_identity() must return real X25519 material (not sha256 stubs)."""
    pytest.importorskip("cryptography")
    identity = generate_identity()
    assert len(identity.peer_id) == 16
    # public_key must be valid 64-char lowercase hex (raw 32-byte X25519 pubkey)
    assert len(identity.public_key) == 64
    assert identity.public_key == identity.public_key.lower()
    # Verify it is loadable as an X25519 key
    from cryptography.hazmat.primitives.asymmetric import x25519
    raw_pub = bytes.fromhex(identity.public_key)
    x25519.X25519PublicKey.from_public_bytes(raw_pub)  # must not raise

def test_generate_identity_deterministic_with_seed():
    pytest.importorskip("cryptography")
    a = generate_identity(seed="test-seed-abc")
    b = generate_identity(seed="test-seed-abc")
    assert a == b

def test_generate_identity_different_without_seed():
    pytest.importorskip("cryptography")
    a = generate_identity()
    b = generate_identity()
    assert a.public_key != b.public_key

def test_save_and_load_identity_keyfile(tmp_path):
    pytest.importorskip("cryptography")
    from peer.crypto import save_identity_keyfile, load_identity_keyfile
    identity = generate_identity()
    path = tmp_path / "peer.key"
    save_identity_keyfile(identity, path)
    # File must exist and be mode 0600
    import stat
    st = path.stat()
    # On non-Windows, check permissions
    if hasattr(stat, "S_IMODE"):
        assert (st.st_mode & 0o777) == 0o600 or os.name == "nt"
    loaded = load_identity_keyfile(path)
    assert loaded.peer_id == identity.peer_id
    assert loaded.public_key == identity.public_key
    assert loaded.private_key == identity.private_key

def test_load_or_create_identity_keyfile(tmp_path):
    pytest.importorskip("cryptography")
    from peer.crypto import load_or_create_identity_keyfile
    path = tmp_path / "subdir" / "peer.key"
    # First call: creates file
    a = load_or_create_identity_keyfile(path)
    assert path.exists()
    # Second call: loads existing file (same identity)
    b = load_or_create_identity_keyfile(path)
    assert a == b

def test_build_activation_envelope_with_pubkey_round_trip():
    pytest.importorskip("cryptography")
    from peer.crypto import (
        build_activation_envelope_with_pubkey,
        decrypt_activation_envelope_with_privkey,
        private_key_from_identity,
    )
    identity = generate_identity()
    raw_pub = bytes.fromhex(identity.public_key)
    priv_key = private_key_from_identity(identity)
    activation = [1.0, 2.0, 3.0]
    envelope = build_activation_envelope_with_pubkey(
        activation,
        raw_public_key_bytes=raw_pub,
        peer_id=identity.peer_id,
        request_id="req-test",
        stage_index=0,
        level="standard",
    )
    recovered = decrypt_activation_envelope_with_privkey(
        ciphertext=envelope.ciphertext,
        nonces=envelope.nonces,
        ephemeral_public_keys=envelope.ephemeral_public_keys,
        private_key=priv_key,
        peer_id=identity.peer_id,
        request_id="req-test",
        stage_index=0,
    )
    assert recovered == pytest.approx(activation)

def test_build_onion_route_with_pubkeys_round_trip():
    pytest.importorskip("cryptography")
    from peer.crypto import (
        build_onion_route_envelope_with_pubkeys,
        peel_onion_route_layer_with_privkey,
        private_key_from_identity,
    )
    peers = [generate_identity() for _ in range(3)]
    route_peer_ids = [p.peer_id for p in peers]
    peer_public_keys = {p.peer_id: bytes.fromhex(p.public_key) for p in peers}
    envelope = build_onion_route_envelope_with_pubkeys(
        route_peer_ids,
        peer_public_keys=peer_public_keys,
        request_id="req-onion",
    )
    # Peel first layer using first peer's private key
    layer = peel_onion_route_layer_with_privkey(
        ciphertext=envelope.ciphertext,
        nonces=envelope.nonces,
        ephemeral_public_keys=envelope.ephemeral_public_keys,
        private_key=private_key_from_identity(peers[0]),
        peer_id=peers[0].peer_id,
        request_id="req-onion",
        stage_index=0,
    )
    assert layer.next_peer_id == peers[1].peer_id
```

### `tests/test_dht_announce.py`

Verify `Announcement` includes `peer_public_key`:

```python
def test_announcement_has_peer_public_key_field():
    from peer.dht_announce import Announcement
    ann = Announcement(peer_id="p1", model_id="m1", host="127.0.0.1", port=9000)
    assert hasattr(ann, "peer_public_key")
    assert ann.peer_public_key == ""
```

### `tests/test_path_finder.py` (or equivalent)

Verify `PeerEndpoint` includes `public_key_hex`:

```python
def test_peer_endpoint_has_public_key_hex_field():
    from coordinator.path_finder import PeerEndpoint
    ep = PeerEndpoint(peer_id="p1", host="127.0.0.1", port=9000)
    assert hasattr(ep, "public_key_hex")
    assert ep.public_key_hex == ""
```

### `tests/test_bootstrap.py` (or equivalent)

Verify `_normalize_peer_record` passes through `peer_public_key`:

```python
def test_normalize_peer_record_passes_through_public_key():
    handler = DhtBootstrapHandler(...)
    payload = {
        "peer_id": "p1", "model_id": "m", "host": "1.2.3.4", "port": 9000,
        "peer_public_key": "a" * 64,
    }
    record = handler._normalize_peer_record(payload)
    assert record["peer_public_key"] == "a" * 64

def test_normalize_peer_record_drops_invalid_public_key():
    handler = DhtBootstrapHandler(...)
    payload = {
        "peer_id": "p1", "model_id": "m", "host": "1.2.3.4", "port": 9000,
        "peer_public_key": "short",
    }
    record = handler._normalize_peer_record(payload)
    assert record["peer_public_key"] == ""
```

---

## Rollout / backward compatibility

| Scenario | Behaviour |
|---|---|
| Peer without `--data-dir` + no keyfile | Key generated, saved to `.openhydra/peer_identity/{peer_id}.key`; public key published in DHT |
| Peer with pre-existing keyfile | Key loaded on startup; same identity across restarts |
| Coordinator talking to new peer (has `public_key_hex`) | Uses per-peer pubkey encryption |
| Coordinator talking to old/dev peer (`public_key_hex=""`) | Falls back to seed-based encryption |
| `cryptography` package absent (edge case) | `generate_identity()` emits `DeprecationWarning`, uses sha256 stub; peer logs warning and skips keyfile |
| Existing tests using seed-based `build_activation_envelope()` | Unchanged — seed-based path kept as dev shim |

---

## What NOT to change

- Do **not** remove or rename `_peer_private_key()`, `peer_static_public_key()`,
  `build_activation_envelope()`, `decrypt_activation_envelope()`,
  `build_onion_route_envelope()`, or `peel_onion_route_layer()`.
  They remain as the seed-based dev/test shim.
- Do **not** change the gRPC proto definitions — the wire format is unchanged;
  only how the key material is sourced changes.
- Do **not** add `torch` or `transformers` imports anywhere in this change set.
- The `geo_challenge` and `privacy_audit` HMAC functions (`sign_geo_challenge`,
  `verify_geo_challenge`, `build_privacy_audit_tag`, `verify_privacy_audit_tag`)
  remain seed-based — they are not part of this fix.
