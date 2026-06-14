# Copyright 2026 OpenHydra contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""M1.3 FFI smoke test for P2PNode.resolve_and_route (protocol.md §5).

Single-process, no swarm. The routing *math* (resolve → filter → degrade → rank →
route-failover → no-provider) is covered exhaustively by the 12 Rust unit tests in
network/src/router.rs. This test proves only the **Python↔Rust FFI boundary**: that
the PyO3 method maps its arguments, drives the background swarm command, and returns
a clean ``no_provider`` RuntimeError when the local DHT is empty. We deliberately do
NOT stand up a 2-node localhost swarm — flaky async network tests are CI poison.
"""

from __future__ import annotations

import pytest

ohn = pytest.importorskip("openhydra_network")


def _make_node(tmp_path):
    return ohn.P2PNode(
        identity_key_path=str(tmp_path / "smoke_identity.key"),
        # Ephemeral, loopback-only addrs: no fixed ports, no LAN exposure.
        listen_addrs=["/ip4/127.0.0.1/tcp/0", "/ip4/127.0.0.1/udp/0/quic-v1"],
    )


def test_resolve_and_route_no_provider_on_empty_dht(tmp_path):
    node = _make_node(tmp_path)
    node.start()
    try:
        with pytest.raises(RuntimeError, match="no_provider"):
            node.resolve_and_route(
                # (model_id, request_canonical_id) candidates, request bytes, tier.
                [("openhydra-smoke-nonexistent-xyz", "smoke/1b/fp16/0000000000000000")],
                b"hello",
                2,
            )
    finally:
        node.stop()


def test_resolve_and_route_degrades_through_all_candidates(tmp_path):
    # Multiple candidate models (the graceful-degradation list) — with an empty DHT
    # every one yields no providers, so the FFI still returns a single clean
    # no_provider rather than hanging or partially failing.
    node = _make_node(tmp_path)
    node.start()
    try:
        with pytest.raises(RuntimeError, match="no_provider"):
            node.resolve_and_route(
                [
                    ("openhydra-smoke-9b", "smoke/9b/*/*"),
                    ("openhydra-smoke-2b", "smoke/2b/*/*"),
                ],
                b"hello",
                1,
            )
    finally:
        node.stop()
