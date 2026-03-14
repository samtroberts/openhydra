import pytest

from coordinator.chain import InferenceChain
from coordinator.path_finder import PeerEndpoint


def test_chain_retries_stage_with_failover(monkeypatch):
    primary = PeerEndpoint(peer_id="peer-primary", host="127.0.0.1", port=5001)
    backup = PeerEndpoint(peer_id="peer-backup", host="127.0.0.1", port=5002)

    chain = InferenceChain([primary], timeout_ms=1000)
    calls: list[str] = []

    def fake_request(self, peer, request_id, prompt, activation, stage_index, total_stages, max_tokens, **kwargs):
        calls.append(peer.peer_id)
        if peer.peer_id == "peer-primary":
            raise RuntimeError("boom")
        return [0.1, -0.2, 0.3], 1.5

    monkeypatch.setattr(InferenceChain, "_request_stage", fake_request)

    result = chain.run(
        "test",
        max_tokens=6,
        failover_pool=[backup],
        max_failovers_per_stage=1,
    )

    assert calls == ["peer-primary", "peer-backup"]
    assert result.traces[0].peer_id == "peer-backup"
    assert result.traces[0].attempt == 2
    assert result.traces[0].failed_peer_id == "peer-primary"


def test_chain_fails_if_retries_exhausted(monkeypatch):
    primary = PeerEndpoint(peer_id="peer-primary", host="127.0.0.1", port=5001)
    backup = PeerEndpoint(peer_id="peer-backup", host="127.0.0.1", port=5002)
    chain = InferenceChain([primary], timeout_ms=1000)

    def fail_request(self, peer, request_id, prompt, activation, stage_index, total_stages, max_tokens, **kwargs):
        raise RuntimeError(f"{peer.peer_id}-down")

    monkeypatch.setattr(InferenceChain, "_request_stage", fail_request)

    with pytest.raises(RuntimeError, match="stage 0 failed"):
        chain.run("test", failover_pool=[backup], max_failovers_per_stage=1)
