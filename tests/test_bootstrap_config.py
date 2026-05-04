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

"""Unit tests for peer.bootstrap_config (Phase 2 ConfigResolver)."""

from __future__ import annotations

import json
import os
import socket
import stat
import tempfile
from pathlib import Path

import pytest

from peer.bootstrap_config import (
    DEFAULT_PEER_ID_NIBBLES,
    DEFAULT_PEER_ID_PREFIX,
    PEERS_LOCAL_SCHEMA_VERSION,
    PeersLocalConfig,
    PortResolutionError,
    ResolvedBootstrap,
    derive_persistent_peer_id,
    load_peers_local_config,
    resolve_bootstrap,
    resolve_port,
)
from peer.identity import load_or_create_identity, peer_id_from_public_key


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _occupy_port(port: int, host: str = "127.0.0.1") -> socket.socket:
    """Bind a socket to ``(host, port)`` to simulate a port collision.
    Caller is responsible for closing the returned socket."""
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
    sock.bind((host, port))
    sock.listen(1)
    return sock


def _free_port_base(reserve_count: int = 3) -> int:
    """Pick a port range [base, base+reserve_count) that's currently free.

    We bind-and-release ``reserve_count`` consecutive ports to find a safe
    test window; the underlying :func:`resolve_port` then probes inside
    this window without colliding with other tests on the host.
    """
    # Ask the kernel for an ephemeral port, then check that (port + N) are
    # also likely free.  We start from ``port + 1000`` to reduce the chance
    # of racing with the OS's own ephemeral allocation during the test run.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        base = s.getsockname()[1]
    # Offset to avoid OS ephemeral allocation, but clamp so the test
    # window (which may span up to base + 25 for bootstrap tests)
    # stays within the valid port range (0-65535).
    candidate = base + 1000
    max_port = 65535 - max(reserve_count, 30)
    if candidate > max_port:
        candidate = max(1024, base - 1000)
    return candidate


# ─── 1. derive_persistent_peer_id ────────────────────────────────────────────


def test_derive_persistent_peer_id_is_stable_across_calls(tmp_path: Path) -> None:
    """Calling twice against the same identity file returns the same peer-id."""
    identity_path = str(tmp_path / "identity.json")
    first = derive_persistent_peer_id(identity_path)
    second = derive_persistent_peer_id(identity_path)
    assert first == second


def test_derive_persistent_peer_id_has_default_shape(tmp_path: Path) -> None:
    """Default output looks like ``oh-XXXXXXXX`` (11 chars total)."""
    identity_path = str(tmp_path / "identity.json")
    peer_id = derive_persistent_peer_id(identity_path)
    assert peer_id.startswith(DEFAULT_PEER_ID_PREFIX)
    assert len(peer_id) == len(DEFAULT_PEER_ID_PREFIX) + DEFAULT_PEER_ID_NIBBLES
    # All trailing chars are lowercase hex.
    trailing = peer_id[len(DEFAULT_PEER_ID_PREFIX):]
    assert all(c in "0123456789abcdef" for c in trailing)


def test_derive_persistent_peer_id_honours_custom_prefix_and_nibbles(
    tmp_path: Path,
) -> None:
    identity_path = str(tmp_path / "identity.json")
    peer_id = derive_persistent_peer_id(identity_path, prefix="swarm-", nibbles=4)
    assert peer_id.startswith("swarm-")
    assert len(peer_id) == len("swarm-") + 4


def test_derive_persistent_peer_id_matches_identity_peer_id(tmp_path: Path) -> None:
    """Derivation reuses the exact SHA256-truncated peer_id that
    :mod:`peer.identity` writes to disk — same canonical peer-id across
    Python and Rust."""
    identity_path = str(tmp_path / "identity.json")
    # Create identity first, then derive.
    identity = load_or_create_identity(identity_path)
    expected = DEFAULT_PEER_ID_PREFIX + identity["peer_id"][:DEFAULT_PEER_ID_NIBBLES]

    derived = derive_persistent_peer_id(identity_path)
    assert derived == expected


def test_derive_persistent_peer_id_different_files_different_ids(
    tmp_path: Path,
) -> None:
    """Two distinct identity files → two distinct peer-ids."""
    idp_a = str(tmp_path / "a.json")
    idp_b = str(tmp_path / "b.json")
    assert derive_persistent_peer_id(idp_a) != derive_persistent_peer_id(idp_b)


def test_derive_persistent_peer_id_creates_parent_dir(tmp_path: Path) -> None:
    """Identity path with a missing parent dir should be created."""
    identity_path = str(tmp_path / "nested" / "deep" / "identity.json")
    assert not Path(identity_path).parent.exists()
    peer_id = derive_persistent_peer_id(identity_path)
    assert Path(identity_path).exists()
    assert peer_id.startswith(DEFAULT_PEER_ID_PREFIX)


# ─── 2. resolve_port ─────────────────────────────────────────────────────────


def test_resolve_port_returns_default_when_free() -> None:
    base = _free_port_base()
    assert resolve_port(default=base, service_name="test") == base


def test_resolve_port_increments_when_default_occupied() -> None:
    base = _free_port_base()
    occupy = _occupy_port(base)
    try:
        resolved = resolve_port(default=base, service_name="test")
        assert resolved == base + 1
    finally:
        occupy.close()


def test_resolve_port_increments_multiple_times() -> None:
    base = _free_port_base()
    occupies = [_occupy_port(base), _occupy_port(base + 1), _occupy_port(base + 2)]
    try:
        resolved = resolve_port(default=base, service_name="test")
        assert resolved == base + 3
    finally:
        for s in occupies:
            s.close()


def test_resolve_port_raises_when_all_probes_fail() -> None:
    base = _free_port_base(reserve_count=5)
    occupies = [_occupy_port(base + offset) for offset in range(5)]
    try:
        with pytest.raises(PortResolutionError):
            resolve_port(default=base, service_name="test", max_tries=5)
    finally:
        for s in occupies:
            s.close()


def test_resolve_port_rejects_invalid_max_tries() -> None:
    with pytest.raises(ValueError):
        resolve_port(default=5000, service_name="test", max_tries=0)


# ─── 3. PeersLocalConfig + load_peers_local_config ──────────────────────────


def test_peers_local_config_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "peers.local.json"
    cfg = PeersLocalConfig(
        peer_id="oh-a1b2c3d4",
        libp2p_peer_id="12D3KooWTEST",
        advertise_host="10.0.0.1",
        ports={"api": 8080, "grpc": 50051, "libp2p": 4001},
        known_peers=[{"peer_id": "oh-ffff0000", "host": "10.0.0.2"}],
    )
    cfg.save(path)

    loaded = load_peers_local_config(path)
    assert loaded is not None
    assert loaded.peer_id == "oh-a1b2c3d4"
    assert loaded.libp2p_peer_id == "12D3KooWTEST"
    assert loaded.advertise_host == "10.0.0.1"
    assert loaded.ports == {"api": 8080, "grpc": 50051, "libp2p": 4001}
    assert loaded.known_peers == [{"peer_id": "oh-ffff0000", "host": "10.0.0.2"}]
    assert loaded.schema_version == PEERS_LOCAL_SCHEMA_VERSION
    assert loaded.last_updated_unix_ms > 0


def test_peers_local_config_save_sets_600_perms(tmp_path: Path) -> None:
    path = tmp_path / "peers.local.json"
    PeersLocalConfig(peer_id="oh-deadbeef").save(path)
    perms = stat.S_IMODE(os.stat(path).st_mode)
    assert perms == 0o600, oct(perms)


def test_peers_local_config_save_atomic_tempfile_cleanup(tmp_path: Path) -> None:
    """After a successful save there must be no orphan ``.tmp`` files."""
    path = tmp_path / "peers.local.json"
    PeersLocalConfig(peer_id="oh-deadbeef").save(path)
    leftovers = [
        p.name for p in tmp_path.iterdir()
        if p.name.startswith("peers.local.json.") and p.name.endswith(".tmp")
    ]
    assert leftovers == []


def test_load_peers_local_config_missing_path_returns_none(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.json"
    assert load_peers_local_config(missing) is None


def test_load_peers_local_config_corrupt_json_returns_none(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.json"
    path.write_text("{not valid json")
    assert load_peers_local_config(path) is None


def test_load_peers_local_config_wrong_schema_returns_none(tmp_path: Path) -> None:
    path = tmp_path / "peers.local.json"
    path.write_text(json.dumps({"schema_version": 99, "peer_id": "oh-x"}))
    assert load_peers_local_config(path) is None


def test_load_peers_local_config_missing_peer_id_returns_none(tmp_path: Path) -> None:
    path = tmp_path / "peers.local.json"
    path.write_text(json.dumps({"schema_version": PEERS_LOCAL_SCHEMA_VERSION}))
    assert load_peers_local_config(path) is None


def test_peers_local_config_survives_interrupted_tempfile(tmp_path: Path) -> None:
    """Stray ``.tmp`` sibling files (simulating a crash mid-save) must not
    confuse the loader on its next read of the real file."""
    path = tmp_path / "peers.local.json"
    PeersLocalConfig(peer_id="oh-deadbeef").save(path)
    # Plant an orphan tempfile that looks like it was from a prior attempt.
    (tmp_path / "peers.local.json.orphan.tmp").write_text("{corrupt")
    loaded = load_peers_local_config(path)
    assert loaded is not None
    assert loaded.peer_id == "oh-deadbeef"


# ─── 4. resolve_bootstrap ────────────────────────────────────────────────────


def test_resolve_bootstrap_derives_on_first_boot(tmp_path: Path) -> None:
    """With no persisted config and no CLI peer-id, peer-id is derived from
    the identity pubkey."""
    idp = str(tmp_path / "identity.json")
    plp = str(tmp_path / "peers.local.json")

    base = _free_port_base()
    rb = resolve_bootstrap(
        cli_peer_id=None,
        cli_api_port=base, cli_grpc_port=base + 1, cli_p2p_port=base + 2,
        default_api_port=base, default_grpc_port=base + 1, default_p2p_port=base + 2,
        identity_path=idp, peers_local_path=plp,
    )
    assert rb.peer_id_source == "derived"
    assert rb.peer_id.startswith(DEFAULT_PEER_ID_PREFIX)
    assert rb.api_port == base
    assert rb.ports_migrated is False


def test_resolve_bootstrap_reuses_persisted(tmp_path: Path) -> None:
    """A prior ``peers.local.json`` wins over deriving a fresh peer-id."""
    idp = str(tmp_path / "identity.json")
    plp = str(tmp_path / "peers.local.json")

    # Persist a specific peer-id.
    PeersLocalConfig(peer_id="oh-cached01").save(plp)

    base = _free_port_base()
    rb = resolve_bootstrap(
        cli_peer_id=None,
        cli_api_port=base, cli_grpc_port=base + 1, cli_p2p_port=base + 2,
        default_api_port=base, default_grpc_port=base + 1, default_p2p_port=base + 2,
        identity_path=idp, peers_local_path=plp,
    )
    assert rb.peer_id_source == "persisted"
    assert rb.peer_id == "oh-cached01"


def test_resolve_bootstrap_cli_peer_id_wins(tmp_path: Path) -> None:
    """An explicit ``--peer-id`` overrides both persisted and derived sources."""
    idp = str(tmp_path / "identity.json")
    plp = str(tmp_path / "peers.local.json")
    PeersLocalConfig(peer_id="oh-cached01").save(plp)

    base = _free_port_base()
    rb = resolve_bootstrap(
        cli_peer_id="my-custom-peer",
        cli_api_port=base, cli_grpc_port=base + 1, cli_p2p_port=base + 2,
        default_api_port=base, default_grpc_port=base + 1, default_p2p_port=base + 2,
        identity_path=idp, peers_local_path=plp,
    )
    assert rb.peer_id_source == "cli_override"
    assert rb.peer_id == "my-custom-peer"


def test_resolve_bootstrap_auto_migrates_default_port(tmp_path: Path) -> None:
    """CLI port == default and default is occupied → auto-migrate to the
    next free port."""
    idp = str(tmp_path / "identity.json")
    plp = str(tmp_path / "peers.local.json")

    base = _free_port_base()
    occupy = _occupy_port(base)
    try:
        rb = resolve_bootstrap(
            cli_peer_id="test",
            cli_api_port=base, cli_grpc_port=base + 10, cli_p2p_port=base + 20,
            default_api_port=base, default_grpc_port=base + 10, default_p2p_port=base + 20,
            identity_path=idp, peers_local_path=plp,
        )
        assert rb.api_port == base + 1
        assert rb.ports_migrated is True
    finally:
        occupy.close()


def test_resolve_bootstrap_respects_non_default_port(tmp_path: Path) -> None:
    """If the CLI port DIFFERS from the documented default, treat it as a
    literal user intent — do not probe / migrate."""
    idp = str(tmp_path / "identity.json")
    plp = str(tmp_path / "peers.local.json")

    base = _free_port_base()
    occupy = _occupy_port(base + 5)  # CLI value is occupied
    try:
        # CLI=base+5 but default=base+10 → no probing, return base+5 verbatim.
        rb = resolve_bootstrap(
            cli_peer_id="test",
            cli_api_port=base + 5, cli_grpc_port=base + 11, cli_p2p_port=base + 21,
            default_api_port=base + 10, default_grpc_port=base + 11, default_p2p_port=base + 21,
            identity_path=idp, peers_local_path=plp,
        )
        assert rb.api_port == base + 5  # user value, not probed
        assert rb.ports_migrated is False
    finally:
        occupy.close()
