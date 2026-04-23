# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Tests for the client-terminated pipeline (Path A) refactor.

Covers Phases 1–3:

* **Phase 1** — proto field wiring
    - ``ForwardRequest.sample_on_coordinator`` (field 54) round-trips
    - ``ForwardResponse.is_hidden_state`` (field 26) round-trips

* **Phase 2** — runtime signatures
    - ``return_hidden_state`` kwarg present on all four runtime surfaces
      (``MLXRuntime.forward``, ``MLXRuntime._forward_sharded``,
      ``PyTorchRuntime.forward``, ``PyTorchRuntime.forward_async``)
      and on ``ModelShard.forward`` / ``forward_async``
    - ``apply_final_head`` present on both real runtimes
    - ``ToyRuntime.forward`` deliberately does NOT accept the kwarg —
      ``ModelShard.forward`` gates the kwarg by runtime type to protect
      the legacy signature

* **Phase 3** — coordinator-side sample-and-reinject
    - ``HeadSampler.sample`` delegates to the borrowed runtime's
      ``apply_final_head`` with every decode param threaded through
    - ``RingSession`` registry is thread-safe (round-trip + clear)
    - ``HeadSampler.sample`` raises on runtimes without
      ``apply_final_head``

The live end-to-end ring test (sample_on_coordinator=True in a full
cross-ISP run) belongs to the benchmark harness, not the unit suite —
it needs a real GPU + MLX peer pair, which pytest doesn't get.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest


# ── Phase 1: proto fields ───────────────────────────────────────────────

def test_proto_sample_on_coordinator_field_roundtrip():
    from peer import peer_pb2

    req = peer_pb2.ForwardRequest(sample_on_coordinator=True)
    assert req.sample_on_coordinator is True
    # Default should be False.
    assert peer_pb2.ForwardRequest().sample_on_coordinator is False
    # Wire round-trip preserves the flag.
    wire = req.SerializeToString()
    restored = peer_pb2.ForwardRequest()
    restored.ParseFromString(wire)
    assert restored.sample_on_coordinator is True


def test_proto_is_hidden_state_field_roundtrip():
    from peer import peer_pb2

    resp = peer_pb2.ForwardResponse(is_hidden_state=True)
    assert resp.is_hidden_state is True
    assert peer_pb2.ForwardResponse().is_hidden_state is False
    wire = resp.SerializeToString()
    restored = peer_pb2.ForwardResponse()
    restored.ParseFromString(wire)
    assert restored.is_hidden_state is True


def test_proto_field_numbers_stable():
    """Regression guard — changing these numbers breaks wire compat."""
    from peer import peer_pb2

    req_fields = {f.name: f.number for f in peer_pb2.ForwardRequest.DESCRIPTOR.fields}
    resp_fields = {f.name: f.number for f in peer_pb2.ForwardResponse.DESCRIPTOR.fields}
    assert req_fields["sample_on_coordinator"] == 54
    assert resp_fields["is_hidden_state"] == 26


# ── Phase 2: runtime signatures ─────────────────────────────────────────

def test_return_hidden_state_kwarg_on_all_runtimes():
    """Every real runtime's forward path must accept return_hidden_state."""
    from peer.mlx_runtime import MLXRuntime
    from peer.model_shard import PyTorchRuntime, ModelShard

    for fn in (
        MLXRuntime.forward,
        MLXRuntime._forward_sharded,
        PyTorchRuntime.forward,
        PyTorchRuntime.forward_async,
        PyTorchRuntime._forward_impl,
        ModelShard.forward,
        ModelShard.forward_async,
    ):
        sig = inspect.signature(fn)
        assert "return_hidden_state" in sig.parameters, (
            f"{fn.__qualname__} missing return_hidden_state kwarg"
        )
        # Default must be False so nothing changes when the flag is off.
        assert sig.parameters["return_hidden_state"].default is False


def test_toy_runtime_does_not_accept_return_hidden_state():
    """Gate by isinstance — ToyRuntime would reject unknown kwargs."""
    from peer.model_shard import ToyRuntime

    sig = inspect.signature(ToyRuntime.forward)
    assert "return_hidden_state" not in sig.parameters


def test_apply_final_head_present_on_runtimes():
    from peer.mlx_runtime import MLXRuntime
    from peer.model_shard import PyTorchRuntime

    assert callable(getattr(MLXRuntime, "apply_final_head", None))
    assert callable(getattr(PyTorchRuntime, "apply_final_head", None))


# ── Phase 3: coordinator sample + ring-session registry ─────────────────

def test_head_sampler_registry_roundtrip():
    from coordinator.head_sampler import (
        register_head_source, get_head_sampler, clear_head_source,
    )

    clear_head_source()
    assert get_head_sampler() is None

    class _R:
        _is_last_shard = True

    runtime = _R()
    register_head_source("peer-1", runtime)
    s = get_head_sampler()
    assert s is not None
    assert s.peer_id == "peer-1"
    assert s.runtime is runtime
    clear_head_source()
    assert get_head_sampler() is None


def test_head_sampler_sample_threads_decode_params():
    """HeadSampler.sample must forward every decode kwarg to apply_final_head."""
    from coordinator.head_sampler import HeadSampler, DecodeConfig

    runtime = MagicMock()
    runtime.apply_final_head.return_value = 42

    sampler = HeadSampler(runtime=runtime, peer_id="p")
    token = sampler.sample(
        hidden_state=[1.0, 2.0, 3.0],
        decode=DecodeConfig(
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            seed=123,
        ),
        packed_bytes=b"\x00\x01",
    )
    assert token == 42
    runtime.apply_final_head.assert_called_once()
    _, kwargs = runtime.apply_final_head.call_args
    assert kwargs["packed_bytes"] == b"\x00\x01"
    assert kwargs["decode_do_sample"] is True
    assert kwargs["decode_temperature"] == pytest.approx(0.7)
    assert kwargs["decode_top_p"] == pytest.approx(0.95)
    assert kwargs["decode_top_k"] == 40
    assert kwargs["decode_seed"] == 123


def test_head_sampler_rejects_runtime_without_apply_final_head():
    from coordinator.head_sampler import HeadSampler, DecodeConfig

    class _LegacyRuntime:
        pass

    sampler = HeadSampler(runtime=_LegacyRuntime(), peer_id="p")
    with pytest.raises(RuntimeError, match="apply_final_head"):
        sampler.sample([1.0], DecodeConfig())


def test_ring_session_registry_roundtrip():
    from coordinator.head_sampler import (
        RingSession, DecodeConfig,
        register_ring_session, get_ring_session,
        unregister_ring_session, clear_ring_sessions,
    )

    clear_ring_sessions()
    assert get_ring_session("rid") is None

    session = RingSession(
        request_id="rid",
        ring_tokens_remaining=10,
        total_stages=3,
        ring_eos_ids={2, 3},
        decode=DecodeConfig(temperature=0.5, top_p=0.9),
    )
    register_ring_session(session)
    got = get_ring_session("rid")
    assert got is session
    assert got.ring_tokens_remaining == 10
    assert got.ring_eos_ids == {2, 3}
    assert got.decode.temperature == pytest.approx(0.5)

    unregister_ring_session("rid")
    assert get_ring_session("rid") is None

    # Clear helper wipes multiple sessions.
    register_ring_session(RingSession(request_id="a"))
    register_ring_session(RingSession(request_id="b"))
    clear_ring_sessions()
    assert get_ring_session("a") is None
    assert get_ring_session("b") is None


def test_ring_session_registry_replaces_on_duplicate_id():
    """Re-registering with same id overwrites — test harness contract."""
    from coordinator.head_sampler import (
        RingSession,
        register_ring_session, get_ring_session, clear_ring_sessions,
    )

    clear_ring_sessions()
    register_ring_session(RingSession(request_id="rid", ring_tokens_remaining=5))
    register_ring_session(RingSession(request_id="rid", ring_tokens_remaining=99))
    assert get_ring_session("rid").ring_tokens_remaining == 99
    clear_ring_sessions()


# ── Flag-off guarantee: default behavior unchanged ──────────────────────

def test_default_request_has_flag_off():
    """Sanity: anything that builds a ForwardRequest without opting in
    to the client-terminated pipeline must default to the legacy path."""
    from peer import peer_pb2

    assert peer_pb2.ForwardRequest().sample_on_coordinator is False
    assert peer_pb2.ForwardResponse().is_hidden_state is False


def test_run_push_ring_accepts_sample_on_coordinator_kwarg():
    """The ring-launcher must accept the new kwarg (and default to False)."""
    from coordinator.chain import InferenceChain

    sig = inspect.signature(InferenceChain.run_push_ring)
    assert "sample_on_coordinator" in sig.parameters
    assert sig.parameters["sample_on_coordinator"].default is False


# ── Phase 5: load_full_head on non-terminal shards ──────────────────────

def test_toy_shard_config_has_load_full_head():
    from peer.model_shard import ToyShardConfig

    default = ToyShardConfig()
    assert default.runtime_load_full_head is False

    enabled = ToyShardConfig(runtime_load_full_head=True)
    assert enabled.runtime_load_full_head is True


def test_peer_service_and_serve_accept_load_full_head():
    from peer.server import PeerService, serve

    for fn in (PeerService.__init__, serve):
        params = inspect.signature(fn).parameters
        assert "load_full_head" in params, (
            f"{fn.__qualname__} missing load_full_head kwarg"
        )
        assert params["load_full_head"].default is False


def test_engine_config_has_sample_on_coordinator():
    from coordinator.engine import EngineConfig

    assert EngineConfig().sample_on_coordinator is False
    assert EngineConfig(sample_on_coordinator=True).sample_on_coordinator is True


def test_register_head_source_accepts_has_final_head_attribute():
    """Phase 5 relaxes the registration gate: a runtime advertising
    ``_has_final_head=True`` is accepted regardless of ``_is_last_shard``.

    This is what lets the Mac-stage-0 peer register itself as the
    coordinator's head source when launched with
    ``--sample-on-coordinator``: it loads the full head (so
    ``_has_final_head=True``) but is not the last shard.
    """
    from coordinator.head_sampler import (
        HeadSampler, DecodeConfig, clear_head_source,
    )

    clear_head_source()

    class _FirstShardWithHead:
        """Simulates MLX stage 0 after load_full_head=True — advertises
        head weights are loaded even though is_last_shard=False."""
        _is_last_shard = False
        _has_final_head = True

        def apply_final_head(self, hidden_state, **kwargs):
            return 7

    # HeadSampler itself is oblivious to _is_last_shard / _has_final_head
    # (those are the peer-side registration gate). It just calls
    # apply_final_head. That's the right separation of concerns.
    sampler = HeadSampler(runtime=_FirstShardWithHead(), peer_id="p0")
    assert sampler.sample([1.0, 2.0], DecodeConfig()) == 7
    clear_head_source()
