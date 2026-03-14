from __future__ import annotations

from concurrent import futures
import os

import grpc
import pytest

from coordinator.chain import InferenceChain
from coordinator.path_finder import PeerEndpoint
from peer.server import GRPC_SERVER_OPTIONS, PeerService
from peer import peer_pb2_grpc


pytestmark = pytest.mark.skipif(
    os.environ.get("OPENHYDRA_RUN_REAL_TENSOR_TEST", "0") != "1",
    reason="set OPENHYDRA_RUN_REAL_TENSOR_TEST=1 to run real PyTorch tensor routing validation",
)


def _start_pytorch_peer(peer_id: str, shard_index: int, model_name: str) -> tuple[grpc.Server, int, PeerService]:
    service = PeerService(
        peer_id=peer_id,
        model_id=model_name,
        shard_index=shard_index,
        total_shards=2,
        daemon_mode="polite",
        broken=False,
        runtime_backend="pytorch_cpu",
        runtime_target="cpu",
        quantization_mode="fp32",
        runtime_model_id=model_name,
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=GRPC_SERVER_OPTIONS)
    peer_pb2_grpc.add_PeerServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    if port == 0:
        raise RuntimeError("grpc_listener_unavailable")
    server.start()
    return server, port, service


def test_real_tensor_routing_between_two_local_nodes():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    model_name = os.environ.get("OPENHYDRA_PYTORCH_TEST_MODEL", "sshleifer/tiny-gpt2")
    try:
        started = [
            _start_pytorch_peer("peer-pt-a", 0, model_name),
            _start_pytorch_peer("peer-pt-b", 1, model_name),
        ]
    except Exception as exc:
        pytest.skip(f"unable to initialize pytorch peers for model '{model_name}': {exc}")

    servers = [item[0] for item in started]
    ports = [item[1] for item in started]
    services = [item[2] for item in started]

    try:
        pipeline = [
            PeerEndpoint(peer_id="peer-pt-a", host="127.0.0.1", port=ports[0]),
            PeerEndpoint(peer_id="peer-pt-b", host="127.0.0.1", port=ports[1]),
        ]
        chain = InferenceChain(pipeline, timeout_ms=120000)
        result = chain.run("OpenHydra real tensor routing validation.", max_tokens=4)

        assert len(result.traces) == 2
        assert result.activation
        assert len(result.activation) == 1
        assert int(round(float(result.activation[0]))) >= 0

        for service in services:
            assert str(service.runtime_profile.get("backend", "")).startswith("pytorch")
            assert service.last_request_thread_id is not None
            assert service.last_inference_thread_id is not None
            assert service.last_inference_thread_id != service.last_request_thread_id
    finally:
        for server in servers:
            server.stop(grace=0)
