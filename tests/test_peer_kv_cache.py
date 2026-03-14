from peer import peer_pb2
from peer.server import PeerService


def _service() -> PeerService:
    return PeerService(
        peer_id="peer-a",
        model_id="openhydra-toy-345m",
        shard_index=0,
        total_shards=2,
        daemon_mode="polite",
        broken=False,
        advanced_encryption_enabled=False,
        kv_cache_max_entries=8,
    )


def test_peer_service_stores_and_reuses_kv_cache():
    service = _service()
    initial = peer_pb2.ForwardRequest(
        request_id="req-1",
        prompt="hello",
        activation=[],
        stage_index=0,
        total_stages=2,
        max_tokens=4,
        kv_session_id="session-1",
        kv_store_activation=True,
    )

    first = service.Forward(initial, None)
    assert first.error == ""
    assert first.kv_cache_hit is False
    assert list(first.activation)

    cached = peer_pb2.ForwardRequest(
        request_id="req-2",
        prompt="",
        activation=[],
        stage_index=0,
        total_stages=2,
        max_tokens=4,
        kv_session_id="session-1",
        kv_store_activation=True,
        kv_use_cached_activation=True,
    )

    second = service.Forward(cached, None)
    assert second.error == ""
    assert second.kv_cache_hit is True
    assert list(second.activation)


def test_peer_service_returns_error_on_kv_cache_miss():
    service = _service()
    request = peer_pb2.ForwardRequest(
        request_id="req-miss",
        prompt="",
        activation=[],
        stage_index=0,
        total_stages=2,
        max_tokens=4,
        kv_session_id="missing-session",
        kv_use_cached_activation=True,
    )

    response = service.Forward(request, None)
    assert "kv_cache_miss" in response.error
    assert response.kv_cache_hit is False
