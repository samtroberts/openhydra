from __future__ import annotations

from concurrent import futures
import json
import os

import grpc
import pytest

from coordinator.engine import CoordinatorEngine, EngineConfig
from peer.server import GRPC_SERVER_OPTIONS, PeerService
from peer import peer_pb2_grpc
from peer.model_shard import _default_trust_remote_code


pytestmark = pytest.mark.skipif(
    os.environ.get("OPENHYDRA_RUN_REAL_TENSOR_TEST", "0") != "1",
    reason="real tensor test gated",
)


MODEL_ID = os.environ.get("OPENHYDRA_PYTORCH_TEST_MODEL", "Qwen/Qwen3.5-0.8B")


def _start_qwen_peer(peer_id: str, model_name: str) -> tuple[grpc.Server, int]:
    service = PeerService(
        peer_id=peer_id,
        model_id="openhydra-qwen3.5-0.8b",
        shard_index=0,
        total_shards=1,
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
    return server, port


def _build_engine(tmp_path, port: int, model_name: str) -> CoordinatorEngine:
    peers_config = tmp_path / "peers.qwen.json"
    peers_config.write_text(
        json.dumps(
            [
                {
                    "peer_id": "peer-qwen-a",
                    "host": "127.0.0.1",
                    "port": int(port),
                    "model_id": "openhydra-qwen3.5-0.8b",
                    "runtime_backend": "pytorch_cpu",
                    "runtime_target": "cpu",
                    "runtime_model_id": model_name,
                    "quantization_mode": "fp32",
                }
            ],
            indent=2,
        )
    )
    return CoordinatorEngine(
        EngineConfig(
            peers_config_path=str(peers_config),
            default_model="openhydra-qwen3.5-0.8b",
            required_replicas=1,
            timeout_ms=15000,
            ledger_path=str(tmp_path / "credits.json"),
            health_store_path=str(tmp_path / "health.json"),
            grounding_use_network=False,
            audit_rate=0.0,
            redundant_exec_rate=0.0,
            barter_decay_per_day=0.0,
            pytorch_generation_model_id=model_name,
        )
    )


def test_qwen_single_shard_generates_text(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    try:
        server, port = _start_qwen_peer("peer-qwen-a", MODEL_ID)
    except Exception as exc:
        pytest.skip(f"unable to initialize qwen peer for model '{MODEL_ID}': {exc}")

    try:
        engine = _build_engine(tmp_path, port, MODEL_ID)
        payload = engine.infer_stream(
            prompt="Five facts about Bangalore",
            max_tokens=128,
            grounding=False,
            pipeline_width=1,
            model_id="openhydra-qwen3.5-0.8b",
        )
        chunks = list(payload["stream"])
        text = "".join(chunks).strip()
        assert payload["streaming"]["mode"] == "pytorch_autoregressive"
        assert payload["streaming"]["pytorch"]["enabled"] is True
        assert text != ""
        assert len(text) > 20, f"Expected coherent response, got: {text!r}"
        print(f"\n{'=' * 60}")
        print(f"  Bangalore Generation (PyTorch gRPC path)")
        print(f"{'=' * 60}")
        print(f"  Tokens: {len(chunks)}")
        print(f"  Text:\n{text}")
        print(f"{'=' * 60}")
    finally:
        server.stop(grace=0)


def test_qwen_eos_metadata_supports_multi_token_ids(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    try:
        server, port = _start_qwen_peer("peer-qwen-a", MODEL_ID)
    except Exception as exc:
        pytest.skip(f"unable to initialize qwen peer for model '{MODEL_ID}': {exc}")

    try:
        engine = _build_engine(tmp_path, port, MODEL_ID)
        payload = engine.infer_stream(
            prompt="Hello",
            max_tokens=2,
            grounding=False,
            pipeline_width=1,
            model_id="openhydra-qwen3.5-0.8b",
        )
        list(payload["stream"])
        eos_token_ids = payload["streaming"]["pytorch"]["eos_token_ids"]
        assert isinstance(eos_token_ids, list)
        assert all(isinstance(item, int) for item in eos_token_ids)
        assert payload["streaming"]["pytorch"]["tokenizer_model_id"] == MODEL_ID
    finally:
        server.stop(grace=0)


def test_qwen_trust_remote_code_flag():
    assert _default_trust_remote_code("Qwen/Qwen3.5-0.8B") is True
    assert _default_trust_remote_code("gpt2") is False
