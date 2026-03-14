from __future__ import annotations

import time

from dht.node import InMemoryDhtNode


def test_dht_put_replaces_same_peer_id():
    dht = InMemoryDhtNode(ttl_seconds=60)
    dht.put("model:x", {"peer_id": "p1", "host": "a"})
    dht.put("model:x", {"peer_id": "p1", "host": "b"})
    records = dht.get("model:x")

    assert len(records) == 1
    assert records[0]["host"] == "b"


def test_dht_get_lazily_expires_without_full_prune(monkeypatch):
    dht = InMemoryDhtNode(ttl_seconds=1)
    def _unexpected_prune() -> None:
        raise RuntimeError("unexpected_prune")

    monkeypatch.setattr(dht, "_prune", _unexpected_prune)
    dht.put("model:x", {"peer_id": "p1"}, ttl_seconds=1)
    assert len(dht.get("model:x")) == 1

    time.sleep(1.1)
    assert dht.get("model:x") == []


def test_dht_background_pruner_cleans_stale_keys():
    dht = InMemoryDhtNode(ttl_seconds=1)
    dht.put("model:x", {"peer_id": "p1"}, ttl_seconds=1)
    dht.start_background_pruner(interval_s=0.1)
    try:
        time.sleep(1.3)
        assert dht.keys() == []
    finally:
        dht.stop_background_pruner()
