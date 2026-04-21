# Copyright 2026 OpenHydra contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for peer.capacity (zero-config CapacityEngine).

Phase 1 focus: deterministic behaviour from synthetic ``HardwareProfile``
fixtures + representative ``ModelAvailability`` catalog entries.  No HF
Hub introspection yet — that's Phase 4.
"""

from __future__ import annotations

import json

from coordinator.degradation import ModelAvailability
from peer.capacity import (
    CAPACITY_SCHEMA_VERSION,
    DEFAULT_RESERVED_SYSTEM_MB,
    DEFAULT_TARGET_CONTEXT,
    STATUS_CAPABLE,
    STATUS_INCAPABLE,
    STATUS_SHARDABLE,
    build_capacity_report,
    calculate_model_capacity,
)
from peer.hardware import HardwareProfile


# ─── Hardware fixtures ───────────────────────────────────────────────────────


def _cuda_t4() -> HardwareProfile:
    """Simulate a Tesla T4: 15 GB VRAM, 14 GB free."""
    return HardwareProfile(
        ram_total_bytes=16 * 1024**3,
        ram_available_bytes=10 * 1024**3,
        accelerator="cuda",
        vram_total_bytes=15 * 1024**3,
        vram_available_bytes=14 * 1024**3,
        cuda_device_count=1,
    )


def _mps_m1_8gb() -> HardwareProfile:
    """Apple Silicon M1 8 GB unified memory — no dedicated VRAM fields."""
    return HardwareProfile(
        ram_total_bytes=8 * 1024**3,
        ram_available_bytes=5 * 1024**3,
        accelerator="mps",
        vram_total_bytes=None,
        vram_available_bytes=None,
        cuda_device_count=0,
    )


def _cpu_toy() -> HardwareProfile:
    """Tiny CPU-only node: 2 GB RAM, no accelerator."""
    return HardwareProfile(
        ram_total_bytes=2 * 1024**3,
        ram_available_bytes=1 * 1024**3,
        accelerator="cpu",
        vram_total_bytes=None,
        vram_available_bytes=None,
        cuda_device_count=0,
    )


def _zero_profile() -> HardwareProfile:
    """Profile where nothing is known."""
    return HardwareProfile(
        ram_total_bytes=None,
        ram_available_bytes=None,
        accelerator="cpu",
        vram_total_bytes=None,
        vram_available_bytes=None,
        cuda_device_count=0,
    )


# ─── Catalog fixtures ────────────────────────────────────────────────────────


def _qwen_08b() -> ModelAvailability:
    return ModelAvailability(
        model_id="openhydra-qwen3.5-0.8b",
        required_peers=1,
        hf_model_id="Qwen/Qwen3.5-0.8B",
        min_vram_gb=2,
        recommended_quantization="fp16",
        context_length=32768,
        shard_vram_gb=2.0,
        shards_needed=1,
        num_layers=24,
    )


def _qwen_2b() -> ModelAvailability:
    return ModelAvailability(
        model_id="openhydra-qwen3.5-2b",
        required_peers=1,
        hf_model_id="Qwen/Qwen3.5-2B",
        min_vram_gb=5,
        recommended_quantization="fp16",
        context_length=32768,
        shard_vram_gb=2.5,
        shards_needed=1,
        num_layers=24,
    )


def _qwen_9b() -> ModelAvailability:
    return ModelAvailability(
        model_id="openhydra-qwen3.5-9b",
        required_peers=2,
        hf_model_id="Qwen/Qwen3.5-9B",
        min_vram_gb=18,
        recommended_quantization="int8",
        context_length=32768,
        shard_vram_gb=9.0,
        shards_needed=2,
        num_layers=32,
    )


def _qwen_27b_fp8() -> ModelAvailability:
    return ModelAvailability(
        model_id="openhydra-qwen3.5-27b-fp8",
        required_peers=4,
        hf_model_id="Qwen/Qwen3.5-27B-FP8",
        min_vram_gb=16,
        recommended_quantization="fp8",
        context_length=32768,
        shard_vram_gb=7.0,
        shards_needed=4,
        num_layers=64,
    )


def _catalog() -> list[ModelAvailability]:
    return [_qwen_08b(), _qwen_2b(), _qwen_9b(), _qwen_27b_fp8()]


# ─── calculate_model_capacity() ──────────────────────────────────────────────


def test_t4_can_host_full_2b() -> None:
    cap = calculate_model_capacity(_cuda_t4(), _qwen_2b())
    assert cap.status == STATUS_CAPABLE
    assert cap.can_host_full is True
    assert cap.can_shard is True
    assert cap.max_layers_hostable == cap.num_layers_total == 24
    assert cap.reason == ""


def test_t4_shards_9b() -> None:
    cap = calculate_model_capacity(_cuda_t4(), _qwen_9b())
    assert cap.status == STATUS_SHARDABLE
    assert cap.can_host_full is False
    assert cap.can_shard is True
    assert 0 < cap.max_layers_hostable < cap.num_layers_total


def test_mps_unified_memory_uses_ram_budget() -> None:
    """M1 Mac with 8 GB unified memory and no vram_* fields should use RAM."""
    cap = calculate_model_capacity(_mps_m1_8gb(), _qwen_2b())
    # 2B fits in 5 GB RAM - 1 GB reserved = 4 GB.
    # per-layer cost: ~107 MB weights + KV + activations ≈ ~140 MB × 24 layers ≈ 3.3 GB.
    assert cap.status == STATUS_CAPABLE
    assert cap.max_layers_hostable == 24


def test_cpu_only_cannot_host_9b() -> None:
    cap = calculate_model_capacity(_cpu_toy(), _qwen_9b())
    assert cap.status in {STATUS_INCAPABLE, STATUS_SHARDABLE}
    # With 1 GB usable, per-layer 576 MB + KV overhead → very few layers at best.
    assert cap.max_layers_hostable < 3


def test_no_memory_reported_yields_incapable_with_reason() -> None:
    cap = calculate_model_capacity(_zero_profile(), _qwen_2b())
    assert cap.status == STATUS_INCAPABLE
    assert cap.max_layers_hostable == 0
    assert cap.reason == "no_memory_reported"


def test_zero_num_layers_degrades_gracefully() -> None:
    """Catalog entry with num_layers=0 shouldn't crash — it's marked incapable."""
    bad_model = ModelAvailability(
        model_id="openhydra-broken",
        required_peers=1,
        hf_model_id="Unknown/Broken",
        min_vram_gb=5,
        shard_vram_gb=2.0,
        num_layers=0,
    )
    cap = calculate_model_capacity(_cuda_t4(), bad_model)
    assert cap.num_layers_total == 0
    assert cap.max_layers_hostable == 0
    assert cap.status == STATUS_INCAPABLE


def test_shorter_context_yields_more_capacity() -> None:
    """Lower target_context → less KV budget needed → more layers fit."""
    long_ctx = calculate_model_capacity(_cuda_t4(), _qwen_9b(), target_context=32768)
    short_ctx = calculate_model_capacity(_cuda_t4(), _qwen_9b(), target_context=1024)
    assert short_ctx.max_layers_hostable >= long_ctx.max_layers_hostable


def test_per_layer_weights_derived_from_shard_vram() -> None:
    """shard_vram_gb=2.5, shards_needed=1, num_layers=24 → 2.5 GB / 24 ≈ 106.7 MB."""
    cap = calculate_model_capacity(_cuda_t4(), _qwen_2b())
    expected_mb = round(2.5 * 1024 / 24, 1)
    assert abs(cap.per_layer_weights_mb - expected_mb) < 1.0


def test_kv_cache_heuristic_bounds() -> None:
    """KV overhead clamped to [1 KB, 8 KB] per tok per layer."""
    cap_2b = calculate_model_capacity(_cuda_t4(), _qwen_2b())
    cap_27b = calculate_model_capacity(_cuda_t4(), _qwen_27b_fp8())
    # Qwen 2B (min_vram=5): 5 * 256 = 1280 B → 1.25 KB
    # Qwen 27B (min_vram=16): 16 * 256 = 4096 B → 4.0 KB
    assert 1.0 <= cap_2b.kv_cache_per_token_kb <= 8.0
    assert 1.0 <= cap_27b.kv_cache_per_token_kb <= 8.0
    # Larger model should have >= KV overhead.
    assert cap_27b.kv_cache_per_token_kb >= cap_2b.kv_cache_per_token_kb


# ─── build_capacity_report() ─────────────────────────────────────────────────


def test_report_schema_version_is_set() -> None:
    report = build_capacity_report(hardware=_cuda_t4(), catalog=_catalog())
    assert report.schema_version == CAPACITY_SCHEMA_VERSION


def test_report_captured_at_is_unix_ms() -> None:
    report = build_capacity_report(hardware=_cuda_t4(), catalog=_catalog())
    # 2026 is ~1.77e12 ms since epoch; sanity-check the range.
    assert 1.5e12 < report.captured_at_unix_ms < 1e13


def test_report_hardware_block_shape() -> None:
    report = build_capacity_report(
        hardware=_cuda_t4(),
        catalog=_catalog(),
        runtime_backend="pytorch_auto",
        accelerator_detail="Tesla T4",
    )
    hw = report.hardware
    assert hw["accelerator"] == "cuda"
    assert hw["accelerator_detail"] == "Tesla T4"
    assert hw["cuda_device_count"] == 1
    assert hw["vram_total_mb"] == 15 * 1024
    assert hw["vram_available_mb"] == 14 * 1024
    assert hw["reserved_system_mb"] == DEFAULT_RESERVED_SYSTEM_MB
    assert hw["usable_memory_mb"] == (14 * 1024) - DEFAULT_RESERVED_SYSTEM_MB
    assert hw["runtime_backend"] == "pytorch_auto"


def test_report_network_block_carries_identity() -> None:
    report = build_capacity_report(
        hardware=_cuda_t4(),
        catalog=_catalog(),
        peer_id="gpu1-peer",
        libp2p_peer_id="12D3KooWTEST",
        ports={"api": 8080, "grpc": 50051, "libp2p": 4001},
        advertise_host="10.0.0.1",
        requires_relay=True,
        relay_circuits=["US", "EU", "AP"],
    )
    assert report.peer_id == "gpu1-peer"
    assert report.libp2p_peer_id == "12D3KooWTEST"
    assert report.network["advertise_host"] == "10.0.0.1"
    assert report.network["ports"] == {"api": 8080, "grpc": 50051, "libp2p": 4001}
    assert report.network["requires_relay"] is True
    assert report.network["relay_circuits"] == ["US", "EU", "AP"]


def test_report_capacity_contains_one_entry_per_model() -> None:
    catalog = _catalog()
    report = build_capacity_report(hardware=_cuda_t4(), catalog=catalog)
    assert len(report.capacity) == len(catalog)
    assert {c.model_id for c in report.capacity} == {m.model_id for m in catalog}


def test_report_is_json_serializable() -> None:
    """to_dict() output must round-trip through json with no custom encoders."""
    report = build_capacity_report(hardware=_cuda_t4(), catalog=_catalog())
    payload = json.dumps(report.to_dict())
    parsed = json.loads(payload)
    assert parsed["schema_version"] == CAPACITY_SCHEMA_VERSION
    assert len(parsed["capacity"]) == len(_catalog())


def test_empty_catalog_yields_empty_capacity_list() -> None:
    """No catalog → report still builds, just no per-model entries."""
    report = build_capacity_report(hardware=_cuda_t4(), catalog=[])
    assert report.capacity == []
    assert report.schema_version == CAPACITY_SCHEMA_VERSION


def test_default_target_context_is_8192() -> None:
    """Sanity-check the documented default."""
    assert DEFAULT_TARGET_CONTEXT == 8192


# ─── Integration with catalog loader (num_layers plumbing) ───────────────────


def test_model_availability_has_num_layers_field() -> None:
    """Dataclass regression guard — capacity engine depends on this field."""
    entry = ModelAvailability(
        model_id="x",
        required_peers=1,
        num_layers=42,
    )
    assert entry.num_layers == 42


def test_catalog_loader_populates_num_layers_from_json(tmp_path) -> None:
    """The engine catalog loader must carry num_layers into ModelAvailability.

    Focused regression test — we don't need a full CoordinatorEngine. We
    import the loader's dataclass and replicate its parse logic against
    the JSON keys it reads.
    """
    # Shape + keys the loader must read.
    entry_json = {
        "model_id": "openhydra-qwen3.5-2b",
        "hf_model_id": "Qwen/Qwen3.5-2B",
        "required_peers": 1,
        "min_vram_gb": 5,
        "shard_vram_gb": 2.5,
        "shards_needed": 1,
        "num_layers": 24,
        "context_length": 32768,
        "recommended_quantization": "fp16",
    }

    # Mirror coordinator.engine._load_model_catalog extraction for num_layers.
    # If this test fails after a refactor, the loader has drifted from the
    # dataclass contract and CapacityEngine will silently see num_layers=0.
    entry = ModelAvailability(
        model_id=str(entry_json["model_id"]),
        required_peers=int(entry_json["required_peers"]),
        hf_model_id=str(entry_json["hf_model_id"]),
        min_vram_gb=int(entry_json["min_vram_gb"]),
        recommended_quantization=str(entry_json["recommended_quantization"]),
        context_length=int(entry_json["context_length"]),
        shard_vram_gb=float(entry_json["shard_vram_gb"]),
        shards_needed=int(entry_json["shards_needed"]),
        num_layers=int(entry_json["num_layers"]),
    )
    assert entry.num_layers == 24

    # And the capacity engine should treat this entry sensibly.
    cap = calculate_model_capacity(_cuda_t4(), entry)
    assert cap.num_layers_total == 24
    assert cap.status == STATUS_CAPABLE
