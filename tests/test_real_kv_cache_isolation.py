from __future__ import annotations

from concurrent import futures
import json
import os

import grpc
import pytest

from coordinator.engine import CoordinatorEngine, EngineConfig
from coordinator.path_finder import PeerEndpoint
from peer.server import GRPC_SERVER_OPTIONS, PeerService
from peer import peer_pb2_grpc


pytestmark = pytest.mark.skipif(
    os.environ.get("OPENHYDRA_RUN_REAL_TENSOR_TEST", "0") != "1",
    reason="set OPENHYDRA_RUN_REAL_TENSOR_TEST=1 to run real PyTorch KV-cache isolation validation",
)


def _start_pytorch_peer(peer_id: str, shard_index: int, model_name: str) -> tuple[grpc.Server, int, PeerService]:
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
        kv_cache_max_entries=8,
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=GRPC_SERVER_OPTIONS)
    peer_pb2_grpc.add_PeerServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    if port == 0:
        raise RuntimeError("grpc_listener_unavailable")
    server.start()
    return server, port, service


def test_real_kv_cache_isolation_across_two_pytorch_nodes(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    model_name = os.environ.get("OPENHYDRA_PYTORCH_TEST_MODEL", "sshleifer/tiny-gpt2")
    try:
        started = [
            _start_pytorch_peer("peer-kv-a", 0, model_name),
            _start_pytorch_peer("peer-kv-b", 1, model_name),
        ]
    except Exception as exc:
        pytest.skip(f"unable to initialize pytorch kv-isolation peers for model '{model_name}': {exc}")

    servers = [item[0] for item in started]
    ports = [item[1] for item in started]
    services = [item[2] for item in started]

    peers_config = tmp_path / "peers.kv.json"
    peers_config.write_text(
        json.dumps(
            [
                {
                    "peer_id": "peer-kv-a",
                    "host": "127.0.0.1",
                    "port": int(ports[0]),
                    "model_id": "openhydra-toy-345m",
                    "runtime_backend": "pytorch_cpu",
                    "runtime_target": "cpu",
                    "quantization_mode": "fp32",
                },
                {
                    "peer_id": "peer-kv-b",
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

    session_id = "kv-iso-session"
    prompt = "The capital of France is"
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
            prompt=prompt,
            max_tokens=4,
            grounding=False,
            pipeline_width=2,
            session_id=session_id,
        )
        chunks = list(payload["stream"])
        text = "".join(chunks)

        assert payload["streaming"]["mode"] == "pytorch_autoregressive"
        assert payload["streaming"]["pytorch"]["enabled"] is True
        assert payload["streaming"]["kv_data_plane"]["cache_updated"] is True
        assert text.strip() != ""
        assert any(char.isalpha() for char in text)

        # If generation stopped early (e.g., immediate EOS), force a deterministic cache-hit round.
        if int(payload["streaming"]["kv_data_plane"]["peer_cache_hits"]) < 1:
            tokenizer = engine._load_generation_tokenizer(model_name)
            token_ids = [int(token) for token in tokenizer.encode(prompt, add_special_tokens=True)]
            if not token_ids:
                token_ids = [int(tokenizer.eos_token_id or 0)]
            pipeline = [
                PeerEndpoint(peer_id="peer-kv-a", host="127.0.0.1", port=ports[0]),
                PeerEndpoint(peer_id="peer-kv-b", host="127.0.0.1", port=ports[1]),
            ]
            prefill = engine._run_chain(
                "",
                pipeline,
                pipeline,
                max_tokens=1,
                initial_activation=[float(token) for token in token_ids],
                kv_session_id=session_id,
                kv_store_activation=True,
                kv_use_cached_activation=False,
                kv_cache_all_stages=True,
            )
            decode = engine._run_chain(
                "",
                pipeline,
                pipeline,
                max_tokens=1,
                initial_activation=[float(int(round(prefill.activation[0])))],
                kv_session_id=session_id,
                kv_store_activation=True,
                kv_use_cached_activation=True,
                kv_cache_all_stages=True,
            )
            assert bool((decode.kv or {}).get("cache_hit")) is True

        for service in services:
            assert service.shard.kv_cache_size >= 1
            assert session_id in service.shard.kv_cache_sessions
    finally:
        for server in servers:
            server.stop(grace=0)
