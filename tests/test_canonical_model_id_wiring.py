# Copyright 2026 OpenHydra contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Integration test: canonical model id end-to-end (protocol.md §4, M1.1).

Proves the wiring across real components and the PyO3 boundary:

    load-site resolution → Announcement carrier → PeerEndpoint parse → discovery filter

Uses the real `openhydra_network` Rust functions, the real `Announcement` /
`PeerEndpoint` dataclasses, and the real `filter_compatible_peers` discovery gate.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest

from coordinator.path_finder import PeerEndpoint, filter_compatible_peers
from peer.canonical_id import resolve_canonical_model_id
from peer.dht_announce import Announcement

ohn = pytest.importorskip("openhydra_network")

TPL_A = "<|im_start|>{{role}}\n{{content}}<|im_end|>"
TPL_B = TPL_A + "  (variant B)"


class _FakeTok:
    def __init__(self, tpl: str) -> None:
        self.chat_template = tpl


class _FakeRuntime:
    def __init__(self, tpl: str) -> None:
        self._tokenizer = _FakeTok(tpl)


class _FakeShard:
    """Mimics a loaded shard whose runtime exposes a tokenizer + chat template."""

    def __init__(self, tpl: str) -> None:
        self._runtime = _FakeRuntime(tpl)


def _profile(quant: str = "fp16", hf: str = "Qwen/Qwen3.5-2B") -> dict:
    return {"runtime_model_id": hf, "quantization_mode": quant}


# --- step 1: canonical id is resolved at the model-load site ---


def test_resolve_canonical_model_id_at_load_site():
    cid = resolve_canonical_model_id(_FakeShard(TPL_A), "openhydra-qwen3.5-2b", _profile())
    assert cid == f"qwen3.5/2b/fp16/{ohn.chat_template_hash(TPL_A)}"


def test_resolve_returns_empty_without_tokenizer():
    # A ToyRuntime-style shard with no tokenizer → "" (not an error).
    class _NoTokShard:
        _runtime = object()

    assert resolve_canonical_model_id(_NoTokShard(), "m", _profile()) == ""


def test_resolve_uses_runtime_quant_not_recommendation():
    a = resolve_canonical_model_id(_FakeShard(TPL_A), "openhydra-qwen3.5-2b", _profile(quant="fp16"))
    b = resolve_canonical_model_id(_FakeShard(TPL_A), "openhydra-qwen3.5-2b", _profile(quant="int4"))
    assert "/fp16/" in a and "/int4/" in b and a != b


# --- step 2: Announcement carries it; PeerEndpoint parses it back ---


def test_canonical_id_survives_announcement_roundtrip():
    cid = resolve_canonical_model_id(_FakeShard(TPL_A), "openhydra-qwen3.5-2b", _profile())
    ann = Announcement(
        peer_id="p1", model_id="openhydra-qwen3.5-2b", host="h", port=1, canonical_model_id=cid
    )
    record = asdict(ann)  # the dumb carrier serialises every field
    assert record["canonical_model_id"] == cid
    ep = PeerEndpoint.from_dict(record)
    assert ep.canonical_model_id == cid


def test_dumb_carrier_default_is_empty():
    ann = Announcement(peer_id="p1", model_id="m", host="h", port=1)
    assert asdict(ann)["canonical_model_id"] == ""


# --- step 3: discovery refuses incompatible providers ---


def _ep(peer_id: str, cid: str) -> PeerEndpoint:
    return PeerEndpoint.from_dict(
        {
            "peer_id": peer_id,
            "host": "h",
            "port": 1,
            "model_id": "openhydra-qwen3.5-2b",
            "canonical_model_id": cid,
        }
    )


def test_discovery_refuses_incompatible_provider():
    cid_a = resolve_canonical_model_id(_FakeShard(TPL_A), "openhydra-qwen3.5-2b", _profile())
    cid_b = resolve_canonical_model_id(_FakeShard(TPL_B), "openhydra-qwen3.5-2b", _profile())
    assert cid_a != cid_b  # different chat template → different canonical id

    peers = [_ep("a", cid_a), _ep("b", cid_b), _ep("legacy", "")]
    # Request provider A's exact canonical id → B (different template) is refused;
    # the legacy peer (no advertised id) is kept (backward-compatible).
    kept = {p.peer_id for p in filter_compatible_peers(peers, cid_a)}
    assert kept == {"a", "legacy"}


def test_discovery_wildcard_request_keeps_same_family():
    cid_a = resolve_canonical_model_id(_FakeShard(TPL_A), "openhydra-qwen3.5-2b", _profile())
    cid_b = resolve_canonical_model_id(_FakeShard(TPL_B), "openhydra-qwen3.5-2b", _profile())
    peers = [_ep("a", cid_a), _ep("b", cid_b)]
    kept = {p.peer_id for p in filter_compatible_peers(peers, "qwen3.5/2b/*/*")}
    assert kept == {"a", "b"}  # both qwen3.5/2b regardless of template


def test_discovery_no_request_id_is_passthrough():
    peers = [_ep("a", "qwen3.5/2b/fp16/" + "a" * 16), _ep("b", "")]
    assert len(filter_compatible_peers(peers, None)) == 2
    assert len(filter_compatible_peers(peers, "")) == 2
