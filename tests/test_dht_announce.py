from peer.dht_announce import Announcement


def test_announcement_has_peer_public_key_field():
    ann = Announcement(peer_id="p1", model_id="m1", host="127.0.0.1", port=9000)
    assert hasattr(ann, "peer_public_key")
    assert ann.peer_public_key == ""
