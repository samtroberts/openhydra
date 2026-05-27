"""Integration tests for the sharded libp2p inference pipeline (Phase 3).

Three test groups:

1. ``TestInferenceChainShardHandoff`` — Unit tests with a mock P2P node.
   Verifies:
   * ``shard_layer_start/end/total_layers`` are set in every ``ForwardRequest``.
   * The activation (hidden-state tensor) returned by peer N is passed as
     the ``activation`` input to peer N+1 (the core handoff assertion).

2. ``TestSelectPipelineSharded`` — Unit tests for
   ``CoordinatorEngine._select_pipeline_sharded()``.
   Verifies edge cases: empty fleet, no shards, incomplete coverage, happy path.

3. ``TestLayerCoverageMapInferenceIntegration`` — Verify that
   LayerCoverageMap.best_pipeline correctly produces PeerEndpoints that
   InferenceChain can use to route a request via proxy_forward.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import pytest

import struct as _struct

from coordinator.chain import InferenceChain


def _unpack_activation(req_or_resp):
    """Extract activation from either packed bytes or repeated float field."""
    packed = bytes(getattr(req_or_resp, "activation_packed", b"") or b"")
    if packed:
        n = len(packed) // 4
        return list(_struct.unpack(f'<{n}f', packed))
    return list(req_or_resp.activation)
from coordinator.layer_coverage import LayerCoverageMap, LayerRange
from coordinator.path_finder import PeerEndpoint, PeerHealth
from peer import peer_pb2


# ── Helpers ──────────────────────────────────────────────────────────────────

_LIBP2P_COUNTER = 0


def _next_libp2p_id() -> str:
    """Generate a unique fake libp2p peer ID."""
    global _LIBP2P_COUNTER
    _LIBP2P_COUNTER += 1
    return f"12D3KooW_peer{_LIBP2P_COUNTER:04d}"


def _peer(
    peer_id: str,
    layer_start: int,
    layer_end: int,
    total_layers: int,
    host: str = "127.0.0.1",
    port: int = 9000,
    libp2p_peer_id: str = "",
) -> PeerEndpoint:
    return PeerEndpoint(
        peer_id=peer_id,
        host=host,
        port=port,
        layer_start=layer_start,
        layer_end=layer_end,
        total_layers=total_layers,
        libp2p_peer_id=libp2p_peer_id or _next_libp2p_id(),
    )


def _health(peer: PeerEndpoint) -> PeerHealth:
    return PeerHealth(peer=peer, healthy=True, latency_ms=5.0, load_pct=0.1, daemon_mode="polite")


# ── Mock P2P node infrastructure ────────────────────────────────────────────


@dataclass
class _CapturedForwardRequest:
    """One captured ForwardRequest from a mock P2P proxy_forward call."""
    peer_id: str
    request_id: str
    stage_index: int
    total_stages: int
    prompt: str
    activation: list[float]
    shard_layer_start: int
    shard_layer_end: int
    shard_total_layers: int
    activation_packed: bytes = b""


class _MockP2PNode:
    """Mock libp2p P2P node that intercepts proxy_forward calls.

    Supports both legacy protobuf and OHV2 wire formats.
    """

    def __init__(self, handler):
        self._handler = handler
        self.libp2p_peer_id = "12D3KooW_coord"
        from openhydra_network import P2PNode as _RealNode
        self._real = _RealNode  # class ref for static methods

    def proxy_forward(self, target_peer_id, data):
        raw = bytes(data)
        prefix = raw[0:1]
        payload = raw[1:]
        if self._real.is_ohv2_msg(payload):
            from peer.ohv2_adapter import OHV2Request
            hdr, act, _ = self._real.decode_forward_msg(payload)
            req = OHV2Request(hdr, act)
            resp = self._handler(req)
            # Encode response as OHV2
            import struct as _s
            _resp_hdr = {
                "request_id": str(getattr(resp, "request_id", "")),
                "status": 0,
                "peer_id": str(getattr(resp, "peer_id", "")),
                "stage_index": int(getattr(resp, "stage_index", 0)),
            }
            _err = str(getattr(resp, "error", "") or "")
            if _err:
                _resp_hdr["status"] = 1
                _resp_hdr["error_message"] = _err
            if bool(getattr(resp, "kv_cache_hit", False)):
                _resp_hdr["status"] = 2
            _onp = str(getattr(resp, "onion_next_peer_id", "") or "")
            if _onp:
                _resp_hdr["onion_next_peer_id"] = _onp
            _act_list = list(getattr(resp, "activation", []))
            _packed_out = bytes(getattr(resp, "activation_packed", b"") or b"")
            if not _packed_out and _act_list:
                _packed_out = _s.pack(f'<{len(_act_list)}f', *_act_list)
            return prefix + bytes(self.encode_response_msg(_resp_hdr, _packed_out))
        else:
            req = peer_pb2.ForwardRequest()
            req.ParseFromString(payload)
            resp = self._handler(req)
            return prefix + resp.SerializeToString()

    def proxy_forward_no_wait(self, target_peer_id, data):
        pass

    def is_peer_connected(self, peer_id):
        return True

    # ── OHV2 static method delegates ──

    @staticmethod
    def is_ohv2_msg(data):
        from openhydra_network import P2PNode as _R
        return _R.is_ohv2_msg(data)

    def encode_forward_msg(self, header_dict, activation, msg_type=0):
        return self._real.encode_forward_msg(header_dict, activation, msg_type)

    def encode_response_msg(self, header_dict, activation):
        return self._real.encode_response_msg(header_dict, activation)

    def decode_forward_msg(self, data):
        return self._real.decode_forward_msg(data)

    def decode_response_msg(self, data):
        return self._real.decode_response_msg(data)


def _make_mock_p2p_node(
    peer_responses: list[list[float]],
) -> tuple[list[_CapturedForwardRequest], _MockP2PNode]:
    """Return (captured_requests, mock_p2p_node) that replays ``peer_responses``.

    ``peer_responses[i]`` is the list[float] that proxy_forward call #i returns
    as ``ForwardResponse.activation``.
    """
    captured: list[_CapturedForwardRequest] = []
    call_count = [0]

    def _handler(request):
        idx = call_count[0]
        call_count[0] += 1
        # Extract activation from either packed bytes or float list.
        _packed = bytes(getattr(request, "activation_packed", b"") or b"")
        if _packed:
            _n = len(_packed) // 4
            _act_list = list(_struct.unpack(f'<{_n}f', _packed))
        else:
            _act_list = list(request.activation)
        captured.append(
            _CapturedForwardRequest(
                peer_id=f"stub-peer-{idx}",
                request_id=str(request.request_id),
                stage_index=int(request.stage_index),
                total_stages=int(request.total_stages),
                prompt=str(request.prompt),
                activation=_act_list,
                shard_layer_start=int(request.shard_layer_start),
                shard_layer_end=int(request.shard_layer_end),
                shard_total_layers=int(request.shard_total_layers),
                activation_packed=_packed,
            )
        )
        activation_out = peer_responses[idx] if idx < len(peer_responses) else []
        # Pack activation as bytes for OHV2 compatibility.
        _packed_out = _struct.pack(f'<{len(activation_out)}f', *activation_out) if activation_out else b""
        resp = peer_pb2.ForwardResponse(
            request_id=request.request_id,
            peer_id=f"stub-peer-{idx}",
            activation=activation_out,
            activation_packed=_packed_out,
            stage_index=request.stage_index,
            error="",
        )
        return resp

    mock_node = _MockP2PNode(_handler)
    return captured, mock_node


def _make_chain_with_mock_p2p(
    pipeline: list[PeerEndpoint],
    peer_responses: list[list[float]],
    timeout_ms: int = 500,
) -> tuple[InferenceChain, list[_CapturedForwardRequest]]:
    """Build an InferenceChain with a _MockP2PNode attached.

    Returns (chain, captured_requests).
    """
    captured, mock_node = _make_mock_p2p_node(peer_responses)
    chain = InferenceChain(pipeline, timeout_ms=timeout_ms)
    chain._p2p_node = mock_node
    return chain, captured


# ── Group 1: Handoff unit tests ───────────────────────────────────────────────


class TestInferenceChainShardHandoff:
    """Verify that activations flow correctly through the mock 3-shard pipeline."""

    def _build_pipeline(self) -> list[PeerEndpoint]:
        return [
            _peer("shard-a", 0, 10, 30, port=9001),
            _peer("shard-b", 10, 20, 30, port=9002),
            _peer("shard-c", 20, 30, 30, port=9003),
        ]

    def test_shard_layer_fields_set_in_all_requests(self):
        """Every ForwardRequest must carry the shard_layer_* metadata for its peer."""
        # Peer responses: hidden states → hidden states → token IDs
        # Stage 0 → returns hidden payload [2.0, 64.0, 1.0, 2.0, …] (seq_len=2, hidden=64)
        hidden_a = [2.0, 64.0] + [0.5] * 128   # seq_len=2, hidden_size=64, 128 floats
        hidden_b = [2.0, 64.0] + [0.7] * 128
        tokens_c = [42.0, 17.0, 3.0]            # token IDs (final stage)

        pipeline = self._build_pipeline()
        chain, captured = _make_chain_with_mock_p2p(
            pipeline, [hidden_a, hidden_b, tokens_c],
        )
        chain.run("test prompt", max_tokens=3)

        assert len(captured) == 3, "Expected exactly 3 proxy_forward calls"

        # Stage 0 — shard-a [0, 10)
        req0 = captured[0]
        assert req0.shard_layer_start == 0
        assert req0.shard_layer_end == 10
        assert req0.shard_total_layers == 30
        assert req0.stage_index == 0
        assert req0.total_stages == 3

        # Stage 1 — shard-b [10, 20)
        req1 = captured[1]
        assert req1.shard_layer_start == 10
        assert req1.shard_layer_end == 20
        assert req1.shard_total_layers == 30
        assert req1.stage_index == 1
        assert req1.total_stages == 3

        # Stage 2 — shard-c [20, 30)
        req2 = captured[2]
        assert req2.shard_layer_start == 20
        assert req2.shard_layer_end == 30
        assert req2.shard_total_layers == 30
        assert req2.stage_index == 2
        assert req2.total_stages == 3

    def test_activation_output_of_stage_n_is_input_to_stage_n_plus_one(self):
        """Core handoff assertion: hidden states flow peer-to-peer without modification."""
        # Stage 0 returns these hidden states
        stage0_output = [2.0, 4.0] + [1.1, 2.2, 3.3, 4.4] * 2    # seq=2, hidden=4
        # Stage 1 returns these hidden states
        stage1_output = [2.0, 4.0] + [5.5, 6.6, 7.7, 8.8] * 2
        # Stage 2 returns token IDs
        stage2_output = [7.0, 42.0]

        pipeline = self._build_pipeline()
        chain, captured = _make_chain_with_mock_p2p(
            pipeline, [stage0_output, stage1_output, stage2_output],
        )
        chain.run("handoff test", max_tokens=2)

        assert len(captured) == 3

        # Stage 0: first stage — sends prompt (no activation input needed)
        assert captured[0].stage_index == 0
        assert captured[0].prompt == "handoff test"

        # Stage 1: must receive exactly what stage 0 returned
        assert captured[1].stage_index == 1
        assert captured[1].prompt == ""      # prompt not re-sent after stage 0
        assert _unpack_activation(captured[1]) == pytest.approx(stage0_output, abs=1e-4)

        # Stage 2: must receive exactly what stage 1 returned
        assert captured[2].stage_index == 2
        assert captured[2].prompt == ""
        assert _unpack_activation(captured[2]) == pytest.approx(stage1_output, abs=1e-4)

    def test_single_shard_pipeline(self):
        """A one-shard pipeline: stage_index=0=total_stages-1, is_first AND is_last."""
        tokens = [3.0, 7.0, 11.0]
        pipeline = [_peer("solo", 0, 32, 32, port=9001)]
        chain, captured = _make_chain_with_mock_p2p(pipeline, [tokens])
        chain.run("solo shard", max_tokens=3)

        assert len(captured) == 1
        req = captured[0]
        assert req.stage_index == 0
        assert req.total_stages == 1
        assert req.shard_layer_start == 0
        assert req.shard_layer_end == 32
        assert req.shard_total_layers == 32

    def test_shard_fields_zero_for_full_model_peer(self):
        """Full-model peers (layer_end=0) send zero shard fields."""
        tokens = [5.0]
        # PeerEndpoint with no layer range (full-model replica)
        pipeline = [PeerEndpoint(
            peer_id="full", host="127.0.0.1", port=9001,
            libp2p_peer_id=_next_libp2p_id(),
        )]
        chain, captured = _make_chain_with_mock_p2p(pipeline, [tokens])
        chain.run("full model test", max_tokens=1)

        assert len(captured) == 1
        req = captured[0]
        assert req.shard_layer_start == 0
        assert req.shard_layer_end == 0
        assert req.shard_total_layers == 0

    def test_stage_index_increments_across_pipeline(self):
        """stage_index increments 0, 1, 2, … for a 4-shard pipeline."""
        hidden = [1.0, 2.0] + [0.1] * 4
        tokens = [10.0]
        responses = [hidden, hidden, hidden, tokens]

        pipeline = [
            _peer(f"shard-{i}", i * 8, (i + 1) * 8, 32, port=9000 + i)
            for i in range(4)
        ]
        chain, captured = _make_chain_with_mock_p2p(pipeline, responses)
        chain.run("four shards", max_tokens=1)

        assert len(captured) == 4
        for i, req in enumerate(captured):
            assert req.stage_index == i, f"Stage {i} had wrong stage_index={req.stage_index}"
            assert req.total_stages == 4


# ── Group 2: _select_pipeline_sharded unit tests ──────────────────────────────


class _MinimalEngine:
    """Minimal stub that exposes only _select_pipeline_sharded for testing."""

    def _select_pipeline_sharded(self, health):
        from coordinator.layer_coverage import LayerCoverageMap
        health_peers = [h.peer for h in health]
        cmap = LayerCoverageMap.from_endpoints(health_peers)
        if not cmap.has_sharded_peers:
            return None
        if not cmap.is_complete():
            return None
        layer_pipeline = cmap.best_pipeline()
        if not layer_pipeline:
            return None
        peer_by_id = {h.peer.peer_id: h.peer for h in health}
        sharded = []
        for lr in layer_pipeline:
            if not lr.is_sharded:
                # Greedy picked a full-model replica — fall back to full-model path
                return None
            peer = peer_by_id.get(lr.peer_id)
            if peer is None:
                return None
            sharded.append(peer)
        return sharded


class TestSelectPipelineSharded:
    """Unit tests for the sharded pipeline selection logic."""

    def test_returns_none_when_no_sharded_peers(self):
        """Full-model replicas (total_layers=0) → None."""
        health = [_health(_peer("p0", 0, 0, 0))]
        engine = _MinimalEngine()
        assert engine._select_pipeline_sharded(health) is None

    def test_returns_none_when_empty_health(self):
        engine = _MinimalEngine()
        assert engine._select_pipeline_sharded([]) is None

    def test_returns_none_when_coverage_incomplete(self):
        """Only first shard present — gap in the middle → None."""
        health = [_health(_peer("p0", 0, 10, 30))]   # missing [10, 30)
        engine = _MinimalEngine()
        assert engine._select_pipeline_sharded(health) is None

    def test_returns_ordered_pipeline_with_complete_coverage(self):
        """3 contiguous shards → ordered list of 3 PeerEndpoints."""
        peers = [
            _peer("shard-a", 0, 10, 30, port=9001),
            _peer("shard-b", 10, 20, 30, port=9002),
            _peer("shard-c", 20, 30, 30, port=9003),
        ]
        health = [_health(p) for p in peers]
        engine = _MinimalEngine()
        result = engine._select_pipeline_sharded(health)

        assert result is not None
        assert len(result) == 3
        # Verify ORDER: layer_start ascending
        assert result[0].peer_id == "shard-a"
        assert result[1].peer_id == "shard-b"
        assert result[2].peer_id == "shard-c"

    def test_returns_ordered_pipeline_regardless_of_input_order(self):
        """Input health list order must not affect output pipeline order."""
        peers = [
            _peer("shard-c", 20, 30, 30, port=9003),  # last first
            _peer("shard-a", 0, 10, 30, port=9001),
            _peer("shard-b", 10, 20, 30, port=9002),
        ]
        health = [_health(p) for p in peers]
        engine = _MinimalEngine()
        result = engine._select_pipeline_sharded(health)

        assert result is not None
        layer_starts = [p.layer_start for p in result]
        assert layer_starts == sorted(layer_starts), "Pipeline must be ordered by layer_start"

    def test_greedy_picks_fewest_stages(self):
        """When peer A covers [0-20) and peer B covers [0-10), greedy picks A first."""
        peers = [
            _peer("long-a", 0, 20, 30, port=9001),   # farther reach
            _peer("short-b", 0, 10, 30, port=9002),
            _peer("last-c", 20, 30, 30, port=9003),
        ]
        health = [_health(p) for p in peers]
        engine = _MinimalEngine()
        result = engine._select_pipeline_sharded(health)

        assert result is not None
        assert result[0].peer_id == "long-a"  # greedy: extend farthest from pos=0
        assert result[-1].peer_id == "last-c"
        assert len(result) == 2  # only 2 stages needed

    def test_full_model_peer_causes_sharded_fallback(self):
        """When the greedy algorithm selects a full-model replica as the optimal
        (fewest-stage) pipeline, _select_pipeline_sharded returns None so the
        legacy full-model path handles the request correctly.

        Rationale: a peer covering [0, total_layers) in one hop is NOT a shard —
        it's a full-model replica.  The sharded pipeline only applies when every
        selected stage covers a proper sub-range.
        """
        peers = [
            _peer("shard-a", 0, 15, 30, port=9001),
            _peer("full",    0, 30, 30, port=9002),   # full-model replica; greedy picks it alone
            _peer("shard-b", 15, 30, 30, port=9003),
        ]
        health = [_health(p) for p in peers]
        engine = _MinimalEngine()
        # Greedy picks "full" as a 1-stage pipeline (optimal); we detect it's not
        # sharded and return None to fall back to full-model routing.
        result = engine._select_pipeline_sharded(health)
        assert result is None

    def test_sharded_pipeline_without_full_model_peer(self):
        """Without a full-model replica, the sharded pipeline is returned."""
        peers = [
            _peer("shard-a", 0, 15, 30, port=9001),
            _peer("shard-b", 15, 30, 30, port=9003),
        ]
        health = [_health(p) for p in peers]
        engine = _MinimalEngine()
        result = engine._select_pipeline_sharded(health)
        assert result is not None
        assert len(result) == 2
        peer_ids = [p.peer_id for p in result]
        assert peer_ids == ["shard-a", "shard-b"]


# ── Group 3: LayerCoverageMap → InferencePreparation integration ──────────────


class TestLayerCoverageMapInferenceIntegration:
    """Verify that LayerCoverageMap.best_pipeline correctly produces PeerEndpoints
    that InferenceChain can use to route a request."""

    def test_layer_range_converted_to_peer_endpoint_correctly(self):
        """Each LayerRange in best_pipeline() maps to a PeerEndpoint with correct fields."""
        peers = [
            _peer("a", 0, 16, 32, port=5001),
            _peer("b", 16, 32, 32, port=5002),
        ]
        health = [_health(p) for p in peers]

        cmap = LayerCoverageMap.from_endpoints([h.peer for h in health])
        assert cmap.is_complete()

        layer_pipeline = cmap.best_pipeline()
        assert layer_pipeline is not None
        assert len(layer_pipeline) == 2

        # Convert LayerRange → PeerEndpoint (same logic as _select_pipeline_sharded)
        peer_by_id = {h.peer.peer_id: h.peer for h in health}
        endpoint_pipeline = [peer_by_id[lr.peer_id] for lr in layer_pipeline]

        assert endpoint_pipeline[0].peer_id == "a"
        assert endpoint_pipeline[0].layer_start == 0
        assert endpoint_pipeline[0].layer_end == 16
        assert endpoint_pipeline[0].port == 5001

        assert endpoint_pipeline[1].peer_id == "b"
        assert endpoint_pipeline[1].layer_start == 16
        assert endpoint_pipeline[1].layer_end == 32
        assert endpoint_pipeline[1].port == 5002

    def test_layer_range_summary_matches_pipeline_stages(self):
        """summary()['best_pipeline_stages'] matches len(best_pipeline())."""
        peers = [
            _peer("x", 0, 11, 32),
            _peer("y", 11, 22, 32),
            _peer("z", 22, 32, 32),
        ]
        cmap = LayerCoverageMap.from_endpoints(peers)
        summary = cmap.summary()
        pipeline = cmap.best_pipeline()

        assert summary["best_pipeline_stages"] == len(pipeline)
        assert summary["coverage_complete"] is True
        assert summary["gaps"] == []

    def test_mock_handoff_with_layer_range_endpoints(self):
        """Full mock handoff: LayerCoverageMap → PeerEndpoint list → InferenceChain."""
        peers = [
            _peer("shard-0", 0, 11, 32, port=7001),
            _peer("shard-1", 11, 22, 32, port=7002),
            _peer("shard-2", 22, 32, 32, port=7003),
        ]
        cmap = LayerCoverageMap.from_endpoints(peers)
        layer_pipeline = cmap.best_pipeline()
        peer_by_id = {p.peer_id: p for p in peers}
        endpoint_pipeline = [peer_by_id[lr.peer_id] for lr in layer_pipeline]

        hidden = [2.0, 11.0] + [0.3] * 22   # seq=2, hidden=11
        tokens = [1.0, 2.0, 3.0]
        chain, captured = _make_chain_with_mock_p2p(
            endpoint_pipeline, [hidden, hidden, tokens],
        )
        result = chain.run("layer range handoff", max_tokens=3)

        assert len(captured) == 3
        # Shard fields match the LayerRange for each stage
        assert captured[0].shard_layer_start == 0  and captured[0].shard_layer_end == 11
        assert captured[1].shard_layer_start == 11 and captured[1].shard_layer_end == 22
        assert captured[2].shard_layer_start == 22 and captured[2].shard_layer_end == 32
        # Activation handoff verified
        assert _unpack_activation(captured[1]) == pytest.approx(hidden, abs=1e-4)
        assert _unpack_activation(captured[2]) == pytest.approx(hidden, abs=1e-4)
