from dataclasses import asdict
import json

from peer.dht_announce import Announcement


def test_announcement_has_peer_public_key_field():
    ann = Announcement(peer_id="p1", model_id="m1", host="127.0.0.1", port=9000)
    assert hasattr(ann, "peer_public_key")
    assert ann.peer_public_key == ""


# ─── Phase 1.5: node_persona / upstream field tests ──────────────────────────


def test_announcement_defaults_include_node_persona():
    """New fields all have safe defaults so existing callers don't need to
    change.  Default persona is native_shard; upstream fields are empty."""
    ann = Announcement(peer_id="p1", model_id="m1", host="127.0.0.1", port=9000)
    assert ann.node_persona == "native_shard"
    assert ann.upstream_kind == ""
    assert ann.upstream_url == ""


def test_announcement_with_atomic_fields_roundtrips_json():
    """Atomic worker peers populate the three new fields; they must survive
    asdict → json.dumps → json.loads unchanged.  Old coordinators treat
    these as unknown keys (ignored) so forward-compat is preserved."""
    ann = Announcement(
        peer_id="ollama-bridge",
        model_id="openhydra-qwen3.5-2b",
        host="127.0.0.1",
        port=50051,
        node_persona="atomic_worker",
        upstream_kind="ollama",
        upstream_url="http://localhost:11434",
    )
    serialised = json.dumps(asdict(ann))
    parsed = json.loads(serialised)
    assert parsed["node_persona"] == "atomic_worker"
    assert parsed["upstream_kind"] == "ollama"
    assert parsed["upstream_url"] == "http://localhost:11434"


def test_announcement_defaults_survive_json_roundtrip():
    """Default (native_shard, empty upstream) values round-trip identically."""
    ann = Announcement(peer_id="p1", model_id="m1", host="127.0.0.1", port=9000)
    parsed = json.loads(json.dumps(asdict(ann)))
    assert parsed["node_persona"] == "native_shard"
    assert parsed["upstream_kind"] == ""
    assert parsed["upstream_url"] == ""
