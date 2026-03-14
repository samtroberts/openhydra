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
    reason="set OPENHYDRA_RUN_REAL_TENSOR_TEST=1 to run real PyTorch onion-routing validation",
)


def _start_pytorch_peer(
    peer_id: str,
    shard_index: int,
    model_name: str,
    *,
    seed: str,
    noise_variance: float,
) -> tuple[grpc.Server, int, PeerService]:
    service = PeerService(
        peer_id=peer_id,
        model_id="openhydra-toy-345m",
        shard_index=shard_index,
        total_shards=3,
        daemon_mode="polite",
        broken=False,
        runtime_backend="pytorch_cpu",
        runtime_target="cpu",
        quantization_mode="fp32",
        runtime_model_id=model_name,
        advanced_encryption_enabled=True,
        advanced_encryption_seed=seed,
        privacy_noise_variance=max(0.0, float(noise_variance)),
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=GRPC_SERVER_OPTIONS)
    peer_pb2_grpc.add_PeerServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    if port == 0:
        raise RuntimeError("grpc_listener_unavailable")
    server.start()
    return server, port, service


def test_real_onion_routing_with_privacy_noise(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    model_name = os.environ.get("OPENHYDRA_PYTORCH_TEST_MODEL", "sshleifer/tiny-gpt2")
    seed = "real-onion-seed"
    noise_variance = float(os.environ.get("OPENHYDRA_PRIVACY_NOISE_VARIANCE", "1e-6"))
    try:
        started = [
            _start_pytorch_peer("peer-onion-a", 0, model_name, seed=seed, noise_variance=noise_variance),
            _start_pytorch_peer("peer-onion-b", 1, model_name, seed=seed, noise_variance=noise_variance),
            _start_pytorch_peer("peer-onion-c", 2, model_name, seed=seed, noise_variance=noise_variance),
        ]
    except Exception as exc:
        pytest.skip(f"unable to initialize pytorch onion-routing peers for model '{model_name}': {exc}")

    servers = [item[0] for item in started]
    ports = [item[1] for item in started]
    services = [item[2] for item in started]

    peers_config = tmp_path / "peers.onion.json"
    peers_config.write_text(
        json.dumps(
            [
                {
                    "peer_id": "peer-onion-a",
                    "host": "127.0.0.1",
                    "port": int(ports[0]),
                    "model_id": "openhydra-toy-345m",
                    "runtime_backend": "pytorch_cpu",
                    "runtime_target": "cpu",
                    "quantization_mode": "fp32",
                },
                {
                    "peer_id": "peer-onion-b",
                    "host": "127.0.0.1",
                    "port": int(ports[1]),
                    "model_id": "openhydra-toy-345m",
                    "runtime_backend": "pytorch_cpu",
                    "runtime_target": "cpu",
                    "quantization_mode": "fp32",
                },
                {
                    "peer_id": "peer-onion-c",
                    "host": "127.0.0.1",
                    "port": int(ports[2]),
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
                advanced_encryption_enabled=True,
                advanced_encryption_seed=seed,
                advanced_encryption_level="enhanced",
                barter_decay_per_day=0.0,
            )
        )
        payload = engine.infer_stream(
            prompt="The capital of France is",
            max_tokens=4,
            grounding=False,
            pipeline_width=3,
            session_id="onion-session",
        )
        chunks = list(payload["stream"])
        text = "".join(chunks)

        assert payload["encryption"]["enabled"] is True
        assert payload["encryption"]["level"] == "enhanced"
        assert int(payload["encryption"]["layers_per_hop"]) >= 2
        assert payload["streaming"]["mode"] in {"pytorch_autoregressive", "pytorch_speculative_decode"}
        assert payload["streaming"]["pytorch"]["enabled"] is True
        assert text.strip() != ""
        assert any(char.isalpha() for char in text)

        service_by_id = {service.peer_id: service for service in services}
        pipeline_ids = [str(item) for item in payload["pipeline"]]
        assert len(pipeline_ids) == 3
        for idx, peer_id in enumerate(pipeline_ids):
            service = service_by_id[peer_id]
            expected_next = pipeline_ids[idx + 1] if idx + 1 < len(pipeline_ids) else ""
            assert service.onion_layers_peeled >= 1
            assert expected_next in service.onion_next_peer_history

        if len(pipeline_ids) >= 2:
            sender_a = service_by_id[pipeline_ids[0]]
            sender_b = service_by_id[pipeline_ids[1]]
            assert sender_a.shard.privacy_noise_variance == pytest.approx(noise_variance, abs=1e-12)
            assert sender_b.shard.privacy_noise_variance == pytest.approx(noise_variance, abs=1e-12)
            assert sender_a.shard.privacy_noise_payloads >= 1
            assert sender_b.shard.privacy_noise_payloads >= 1
    finally:
        for server in servers:
            server.stop(grace=0)
