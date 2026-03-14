from __future__ import annotations

from concurrent import futures
import json
import os

import grpc
import pytest

from coordinator.engine import CoordinatorEngine, EngineConfig
from peer.server import GRPC_SERVER_OPTIONS, PeerService
from peer import peer_pb2_grpc


pytestmark = pytest.mark.skipif(
    os.environ.get("OPENHYDRA_RUN_REAL_TENSOR_TEST", "0") != "1",
    reason="set OPENHYDRA_RUN_REAL_TENSOR_TEST=1 to run real PyTorch tensor compression validation",
)


def _start_pytorch_peer(
    peer_id: str,
    shard_index: int,
    model_name: str,
    latent_dim: int,
) -> tuple[grpc.Server, int, PeerService]:
    service = PeerService(
        peer_id=peer_id,
        model_id="openhydra-toy-345m",
        shard_index=shard_index,
        total_shards=2,
        daemon_mode="polite",
        broken=False,
        runtime_backend="pytorch_cpu",
        runtime_target="cpu",
        quantization_mode="fp32",
        runtime_model_id=model_name,
        tensor_autoencoder_enabled=True,
        tensor_autoencoder_latent_dim=max(1, int(latent_dim)),
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=GRPC_SERVER_OPTIONS)
    peer_pb2_grpc.add_PeerServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    if port == 0:
        raise RuntimeError("grpc_listener_unavailable")
    server.start()
    return server, port, service


def test_real_tensor_compression_with_pytorch_runtime(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    model_name = os.environ.get("OPENHYDRA_PYTORCH_TEST_MODEL", "sshleifer/tiny-gpt2")
    default_latent = 256 if model_name == "gpt2" else 1
    latent_dim = int(os.environ.get("OPENHYDRA_PYTORCH_TEST_LATENT_DIM", str(default_latent)))
    try:
        started = [
            _start_pytorch_peer("peer-cmpr-a", 0, model_name, latent_dim),
            _start_pytorch_peer("peer-cmpr-b", 1, model_name, latent_dim),
        ]
    except Exception as exc:
        pytest.skip(f"unable to initialize pytorch compression peers for model '{model_name}': {exc}")

    servers = [item[0] for item in started]
    ports = [item[1] for item in started]
    services = [item[2] for item in started]

    peers_config = tmp_path / "peers.compressed.json"
    peers_config.write_text(
        json.dumps(
            [
                {
                    "peer_id": "peer-cmpr-a",
                    "host": "127.0.0.1",
                    "port": int(ports[0]),
                    "model_id": "openhydra-toy-345m",
                    "runtime_backend": "pytorch_cpu",
                    "runtime_target": "cpu",
                    "quantization_mode": "fp32",
                },
                {
                    "peer_id": "peer-cmpr-b",
                    "host": "127.0.0.1",
                    "port": int(ports[1]),
                    "model_id": "openhydra-toy-345m",
                    "runtime_backend": "pytorch_cpu",
                    "runtime_target": "cpu",
                    "quantization_mode": "fp32",
                },
            ],
            indent=2,
        )
    )

    try:
        engine = CoordinatorEngine(
            EngineConfig(
                peers_config_path=str(peers_config),
                ledger_path=str(tmp_path / "credits.json"),
                health_store_path=str(tmp_path / "health.json"),
                audit_rate=0.0,
                redundant_exec_rate=0.0,
                grounding_use_network=False,
                required_replicas=1,
                pytorch_generation_model_id=model_name,
                barter_decay_per_day=0.0,
            )
        )
        payload = engine.infer_stream(
            prompt="The capital of France is",
            max_tokens=4,
            grounding=False,
            pipeline_width=2,
            session_id="compression-session",
        )
        chunks = list(payload["stream"])
        text = "".join(chunks)

        assert payload["streaming"]["mode"] in {"pytorch_autoregressive", "pytorch_speculative_decode"}
        assert payload["streaming"]["pytorch"]["enabled"] is True
        assert text.strip() != ""
        assert any(char.isalpha() for char in text)

        first_peer = services[0].shard
        second_peer = services[1].shard
        assert first_peer.compression_enabled is True
        assert second_peer.compression_enabled is True
        total_encoded = int(first_peer.compression_encoded_payloads) + int(second_peer.compression_encoded_payloads)
        total_decoded = int(first_peer.compression_decoded_payloads) + int(second_peer.compression_decoded_payloads)
        assert total_encoded >= 1
        assert total_decoded >= 1
    finally:
        for server in servers:
            server.stop(grace=0)
