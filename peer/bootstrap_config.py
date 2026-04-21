# Copyright 2026 OpenHydra contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Zero-config bootstrap helpers — Phase 2 ``ConfigResolver``.

Three concerns collected in one module so :mod:`coordinator.node` stays thin:

1.  **Stable peer-id**:  derive ``peer_id`` from a persistent Ed25519 pubkey
    instead of the hostname.  Guarantees survival across Docker/container
    restarts where ``gethostname()`` is unstable or randomised.
2.  **Port auto-retry**:  probe default ports (:8080/:50051/:4001) and
    increment by one on ``EADDRINUSE`` up to a bounded number of tries.
    Lets two nodes share a machine without manual port juggling.
3.  **Persistent ``peers.local.json``**:  ``~/.openhydra/peers.local.json``
    caches the resolved identity, ports, and advertise host so the same
    node comes back up on the same wires after a restart.

All three functions are independent — a node that wants only the peer-id
derivation can ignore the rest.  Manual CLI flags always take precedence
over these defaults (enforced by :mod:`coordinator.node`, not here).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import errno
import json
import logging
import os
import socket
import tempfile
import time
from pathlib import Path
from typing import Any

from peer.identity import load_or_create_identity


logger = logging.getLogger(__name__)


# Schema version for ``peers.local.json``.  Incremented whenever the on-disk
# shape changes in a way old readers would choke on.  When a mismatch is
# detected, :func:`load_peers_local_config` returns ``None`` so the caller
# falls back to a fresh derive-and-persist cycle.
PEERS_LOCAL_SCHEMA_VERSION = 1

# Default locations — relative to cwd to match the existing ``--identity-path``
# default in :mod:`coordinator.node` (``.openhydra/identity.key``).
DEFAULT_OPENHYDRA_DIR = ".openhydra"
DEFAULT_BOOTSTRAP_IDENTITY_PATH = ".openhydra/bootstrap_identity.json"
DEFAULT_PEERS_LOCAL_PATH = ".openhydra/peers.local.json"

# Peer-id derivation defaults.
# ``oh-`` prefix disambiguates OpenHydra peer-ids from random hex strings in
# logs and from the separate base58 ``libp2p_peer_id`` (``12D3KooW...``).
# Eight nibbles of SHA256(pubkey) gives 2^32 = 4.3 billion peer-ids — plenty
# of collision headroom for a swarm that'll realistically stay under 10⁶.
DEFAULT_PEER_ID_PREFIX = "oh-"
DEFAULT_PEER_ID_NIBBLES = 8

# Port auto-retry defaults.
DEFAULT_PORT_PROBE_HOST = "127.0.0.1"
DEFAULT_PORT_MAX_TRIES = 10


class PortResolutionError(RuntimeError):
    """Raised when :func:`resolve_port` cannot find a free port within
    ``max_tries`` attempts starting from the requested default."""


# ─── 1. Peer-id derivation ───────────────────────────────────────────────────


def derive_persistent_peer_id(
    identity_path: str = DEFAULT_BOOTSTRAP_IDENTITY_PATH,
    *,
    prefix: str = DEFAULT_PEER_ID_PREFIX,
    nibbles: int = DEFAULT_PEER_ID_NIBBLES,
) -> str:
    """Return a stable peer-id derived from a persistent Ed25519 pubkey.

    The first call generates a keypair at ``identity_path`` (0600 perms) via
    :func:`peer.identity.load_or_create_identity` and caches it.  Subsequent
    calls read the same file so the peer-id is stable across reboots.

    Shape: ``{prefix}{first N hex chars of SHA256(pubkey)}``.
    Default: ``oh-a1b2c3d4`` (prefix ``oh-`` + 8 nibbles = 11 chars total).

    The underlying 16-char hex is the same value produced by
    :func:`peer.identity.peer_id_from_public_key` so Python and Rust agree
    on the canonical peer-id for any given pubkey.
    """
    identity = load_or_create_identity(identity_path)
    hex_peer_id = str(identity["peer_id"])
    n = max(1, int(nibbles))
    return f"{prefix}{hex_peer_id[:n]}"


# ─── 2. Port auto-retry ──────────────────────────────────────────────────────


def _try_bind(host: str, port: int) -> bool:
    """Return True if we can briefly bind ``(host, port)`` for SOCK_STREAM.

    The socket is closed immediately; the caller races with the real server's
    ``listen()`` but in practice the race window is microseconds and the real
    server reports ``EADDRINUSE`` gracefully on the unlikely collision.

    Non-bind errors (permission denied, bad address) propagate as exceptions
    so they surface during boot rather than silently looping.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        sock.bind((host, port))
        return True
    except OSError as exc:
        if exc.errno in (errno.EADDRINUSE, errno.EACCES):
            return False
        raise
    finally:
        sock.close()


def resolve_port(
    *,
    default: int,
    service_name: str,
    host: str = DEFAULT_PORT_PROBE_HOST,
    max_tries: int = DEFAULT_PORT_MAX_TRIES,
) -> int:
    """Return the first free port in the range [default, default+max_tries).

    Logs ``port_auto_migrated`` at INFO level when the default was taken.
    Raises :class:`PortResolutionError` if every probed port is busy —
    otherwise inevitable on a locked-down host with hundreds of sidecars,
    and the caller should surface the error rather than fall through.

    This function is advisory — it does NOT reserve the port.  The real
    server must bind to the returned value.  The race window is tiny but
    non-zero; a second bind attempt with ``SO_REUSEADDR=0`` will reject
    collisions loudly, which is the desired failure mode.
    """
    if max_tries < 1:
        raise ValueError(f"max_tries must be >= 1, got {max_tries}")

    for offset in range(max_tries):
        port = default + offset
        try:
            if _try_bind(host, port):
                if offset > 0:
                    logger.info(
                        "port_auto_migrated: service=%s default=%d assigned=%d "
                        "(original was in use)",
                        service_name, default, port,
                    )
                return port
        except OSError as exc:
            # Fatal socket error (e.g. EACCES on :80) — re-raise with context.
            raise PortResolutionError(
                f"resolve_port: fatal socket error on {host}:{port} "
                f"for service={service_name!r}: {exc}"
            ) from exc

    raise PortResolutionError(
        f"resolve_port: no free port in [{default}, {default + max_tries}) "
        f"for service={service_name!r} on host={host!r}"
    )


# ─── 3. Persistent ``peers.local.json`` ──────────────────────────────────────


@dataclass
class PeersLocalConfig:
    """Persistent cache of this node's bootstrap state.

    Lives at :const:`DEFAULT_PEERS_LOCAL_PATH` (``~/.openhydra/peers.local.json``
    by default).  Written atomically (``tempfile`` + ``os.replace``) so a
    crash mid-write never produces a truncated file.

    ``known_peers`` is a free-form list of dicts — Phase 3 will define a
    strict schema once ``SwarmNegotiator`` needs it.
    """

    peer_id: str
    libp2p_peer_id: str = ""
    advertise_host: str = ""
    ports: dict[str, int] = field(default_factory=dict)
    last_updated_unix_ms: int = 0
    known_peers: list[dict[str, Any]] = field(default_factory=list)
    schema_version: int = PEERS_LOCAL_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path = DEFAULT_PEERS_LOCAL_PATH) -> None:
        """Atomically persist to ``path``.  Creates parent dirs with 0700 perms."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Best-effort tighten of parent perms — ignore on Windows where chmod
        # is essentially a no-op for user dirs.
        try:
            os.chmod(target.parent, 0o700)
        except OSError:
            pass

        payload = dict(self.to_dict())
        payload["last_updated_unix_ms"] = int(time.time() * 1000)

        # Atomic write: write to a sibling tempfile, fsync, then os.replace.
        # ``os.replace`` is atomic on POSIX and Windows (Python ≥3.3).
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.flush()
                try:
                    os.fsync(handle.fileno())
                except OSError:
                    # Some filesystems (tmpfs, exotic FUSE mounts) don't
                    # support fsync — data still survives a clean shutdown.
                    pass
            os.replace(tmp_path, target)
            try:
                os.chmod(target, 0o600)
            except OSError:
                pass
        except Exception:
            # Clean up the tempfile on any write failure so we don't leave
            # orphan ``.tmp`` files behind.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        # Update the in-memory copy's timestamp too so callers see the
        # persisted value on subsequent reads.
        object.__setattr__(self, "last_updated_unix_ms", int(payload["last_updated_unix_ms"]))


def load_peers_local_config(
    path: str | Path = DEFAULT_PEERS_LOCAL_PATH,
) -> PeersLocalConfig | None:
    """Return the cached config at ``path``, or ``None`` on any failure.

    Failure modes (all → ``None``, logged at DEBUG):
        * File does not exist.
        * File is corrupt / not valid JSON.
        * ``schema_version`` mismatch.
        * Required fields missing.

    Callers should treat ``None`` as "first boot on this host" and rebuild
    the config from scratch via :func:`derive_persistent_peer_id` +
    :func:`resolve_port`, then persist via :meth:`PeersLocalConfig.save`.
    """
    target = Path(path)
    if not target.exists():
        return None
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.debug("peers_local_config_unreadable: path=%s err=%s", target, exc)
        return None

    if not isinstance(raw, dict):
        logger.debug("peers_local_config_not_object: path=%s", target)
        return None

    schema = int(raw.get("schema_version", 0))
    if schema != PEERS_LOCAL_SCHEMA_VERSION:
        logger.debug(
            "peers_local_config_schema_mismatch: path=%s file=%d expected=%d",
            target, schema, PEERS_LOCAL_SCHEMA_VERSION,
        )
        return None

    peer_id = str(raw.get("peer_id") or "")
    if not peer_id:
        logger.debug("peers_local_config_missing_peer_id: path=%s", target)
        return None

    try:
        ports_raw = raw.get("ports") or {}
        ports = {str(k): int(v) for k, v in ports_raw.items()}
    except (TypeError, ValueError):
        ports = {}

    known_peers_raw = raw.get("known_peers") or []
    known_peers = [dict(p) for p in known_peers_raw if isinstance(p, dict)]

    return PeersLocalConfig(
        schema_version=schema,
        peer_id=peer_id,
        libp2p_peer_id=str(raw.get("libp2p_peer_id") or ""),
        advertise_host=str(raw.get("advertise_host") or ""),
        ports=ports,
        last_updated_unix_ms=int(raw.get("last_updated_unix_ms", 0) or 0),
        known_peers=known_peers,
    )


# ─── 4. End-to-end helper used by coordinator.node ───────────────────────────


@dataclass(frozen=True)
class ResolvedBootstrap:
    """Immutable result bundle returned by :func:`resolve_bootstrap`.

    Fields:
        ``peer_id``          — final OpenHydra peer-id (CLI override or derived).
        ``peer_id_source``   — one of ``"cli_override"``, ``"persisted"``, ``"derived"``.
        ``api_port``         — resolved HTTP API port (CLI value or post-auto-retry).
        ``grpc_port``        — resolved peer gRPC port.
        ``p2p_port``         — resolved libp2p listen port.
        ``ports_migrated``   — True if any port was auto-migrated from its default.
        ``peers_local_path`` — absolute path where :func:`PeersLocalConfig.save`
                               will persist the config.
    """

    peer_id: str
    peer_id_source: str
    api_port: int
    grpc_port: int
    p2p_port: int
    ports_migrated: bool
    peers_local_path: str


def resolve_bootstrap(
    *,
    cli_peer_id: str | None,
    cli_api_port: int,
    cli_grpc_port: int,
    cli_p2p_port: int = 4001,
    default_api_port: int = 8080,
    default_grpc_port: int = 50051,
    default_p2p_port: int = 4001,
    identity_path: str = DEFAULT_BOOTSTRAP_IDENTITY_PATH,
    peers_local_path: str = DEFAULT_PEERS_LOCAL_PATH,
    probe_host: str = DEFAULT_PORT_PROBE_HOST,
    max_port_tries: int = DEFAULT_PORT_MAX_TRIES,
) -> ResolvedBootstrap:
    """Resolve peer-id + ports for a zero-config boot.

    Precedence for peer-id:
        1. ``cli_peer_id`` (non-empty) — user intent wins.
        2. Persisted ``peer_id`` from ``peers.local.json`` — same node, same id.
        3. Derived from Ed25519 pubkey at ``identity_path`` — first boot on host.

    Precedence for ports:  if the caller passed a CLI value EQUAL to the
    documented default, we probe and auto-migrate on collision.  If the CLI
    value differs from the default, we assume the user meant it literally
    and do not probe (manual flags always take precedence).
    """
    # ─── Peer-id resolution ─────────────────────────────────────────────
    cli_pid = (cli_peer_id or "").strip()
    persisted = load_peers_local_config(peers_local_path)

    if cli_pid:
        final_peer_id = cli_pid
        peer_id_source = "cli_override"
    elif persisted is not None and persisted.peer_id:
        final_peer_id = persisted.peer_id
        peer_id_source = "persisted"
    else:
        final_peer_id = derive_persistent_peer_id(identity_path)
        peer_id_source = "derived"

    # ─── Port resolution ────────────────────────────────────────────────
    # Only auto-retry when the CLI value equals the documented default —
    # that way ``--api-port 9090`` is treated as a literal user intent
    # and never silently migrated.
    def _maybe_resolve(service: str, cli_value: int, default_value: int) -> tuple[int, bool]:
        if int(cli_value) != int(default_value):
            return int(cli_value), False
        resolved = resolve_port(
            default=int(default_value),
            service_name=service,
            host=probe_host,
            max_tries=max_port_tries,
        )
        return int(resolved), bool(resolved != int(default_value))

    api_port, api_migrated = _maybe_resolve("api", cli_api_port, default_api_port)
    grpc_port, grpc_migrated = _maybe_resolve("grpc", cli_grpc_port, default_grpc_port)
    p2p_port, p2p_migrated = _maybe_resolve("libp2p", cli_p2p_port, default_p2p_port)

    return ResolvedBootstrap(
        peer_id=final_peer_id,
        peer_id_source=peer_id_source,
        api_port=api_port,
        grpc_port=grpc_port,
        p2p_port=p2p_port,
        ports_migrated=(api_migrated or grpc_migrated or p2p_migrated),
        peers_local_path=str(Path(peers_local_path).resolve()),
    )
