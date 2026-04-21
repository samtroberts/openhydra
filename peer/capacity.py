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

"""Ollama-style capacity engine for zero-config peer bootstrap.

Given a :class:`peer.hardware.HardwareProfile` and the loaded model catalogue
(:class:`coordinator.degradation.ModelAvailability`), compute how many layers
of each catalogued model the local peer can host.

The output is a JSON payload consumed by:

1. The ``/v1/internal/capacity`` HTTP endpoint (debug UI, Phase 2 visualiser).
2. The ``SwarmNegotiator`` (Phase 3) to decide which gap to fill.
3. DHT announcements (``Announcement.capacity_json``) so coordinators see
   each peer's declared capability at discovery time.

Design notes (Phase 1 — keep it conservative):
    * KV cache size is estimated via a **Qwen-calibrated heuristic** — we do
      not introspect HuggingFace ``config.json`` yet (that's Phase 4).
    * Activation overhead is a flat ~20% of per-layer weight bytes.
    * We apply a ``reserved_system_mb`` floor (default 1 GiB) for OS/driver
      headroom before dividing the remaining VRAM across layers.
    * MLX peers currently report ``vram_total_bytes=None`` — we fall back to
      ``ram_total_bytes`` because Apple Silicon unified memory shares the
      pool between CPU and GPU.
    * CPU-only peers (``accelerator="cpu"``) use RAM as the capacity budget
      (they can serve small models like Qwen3.5-0.8B — just slower).

See :doc:`docs/architecture/zero_config_bootstrap.md` for the full design.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time
from typing import Any

from coordinator.degradation import ModelAvailability
from peer.hardware import HardwareProfile


# Schema version — bump whenever the output JSON shape changes in a way
# downstream consumers (UI, coordinator) need to react to.
CAPACITY_SCHEMA_VERSION = 1

# Default capacity computation parameters. Callers can override.
DEFAULT_TARGET_CONTEXT = 8192
DEFAULT_RESERVED_SYSTEM_MB = 1024
DEFAULT_ACTIVATION_OVERHEAD_RATIO = 0.20

# KV cache heuristic bounds (bytes per token per layer).
# Calibrated against Qwen 3.5 architecture:
#   2B (GQA, num_kv_heads=2, head_dim=128, bf16) = 1024 B/tok/layer
#   9B (GQA, num_kv_heads=4, head_dim=128, bf16) = 2048 B/tok/layer
# We scale by ``min_vram_gb`` to stay conservative for larger models.
KV_BYTES_PER_TOK_PER_LAYER_FLOOR = 1024
KV_BYTES_PER_TOK_PER_LAYER_CEILING = 8192
KV_BYTES_PER_TOK_PER_LAYER_PER_GB = 256


# Status strings used in the output JSON.
STATUS_CAPABLE = "capable"       # Can host every layer of this model.
STATUS_SHARDABLE = "shardable"   # Can host some layers — useful as a shard.
STATUS_INCAPABLE = "incapable"   # Cannot host even one layer.


@dataclass(frozen=True)
class ModelCapacity:
    """Per-model capacity result — one entry per catalogue model."""

    model_id: str
    num_layers_total: int
    max_layers_hostable: int
    per_layer_weights_mb: float
    kv_cache_per_token_kb: float
    activation_overhead_mb: float
    target_context: int
    recommended_quantization: str
    can_host_full: bool
    can_shard: bool
    status: str
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CapacityReport:
    """Full capacity report emitted by :func:`build_capacity_report`."""

    schema_version: int
    peer_id: str
    libp2p_peer_id: str
    captured_at_unix_ms: int
    hardware: dict[str, Any]
    network: dict[str, Any]
    capacity: list[ModelCapacity] = field(default_factory=list)
    throughput: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "peer_id": self.peer_id,
            "libp2p_peer_id": self.libp2p_peer_id,
            "captured_at_unix_ms": self.captured_at_unix_ms,
            "hardware": dict(self.hardware),
            "network": dict(self.network),
            "capacity": [c.to_dict() for c in self.capacity],
            "throughput": dict(self.throughput),
        }


# ─── Internal helpers ────────────────────────────────────────────────────────


def _usable_memory_bytes(
    hw: HardwareProfile, reserved_system_mb: int
) -> int | None:
    """Return the budget (bytes) we can spend on model weights + KV + activations.

    Priority:
        1. GPU VRAM (if ``vram_available_bytes`` is known).
        2. Full VRAM (``vram_total_bytes``) if we only know the total.
        3. RAM (for Apple Silicon unified-memory or CPU-only peers).

    Subtracts ``reserved_system_mb`` for OS/driver overhead. Returns ``None``
    if nothing is known.
    """
    reserved_bytes = max(0, reserved_system_mb) * 1024 * 1024

    if hw.vram_available_bytes is not None and hw.vram_available_bytes > 0:
        return max(0, int(hw.vram_available_bytes) - reserved_bytes)
    if hw.vram_total_bytes is not None and hw.vram_total_bytes > 0:
        return max(0, int(hw.vram_total_bytes) - reserved_bytes)
    # MLX / CPU fallback — unified memory or pure CPU.
    if hw.ram_available_bytes is not None and hw.ram_available_bytes > 0:
        return max(0, int(hw.ram_available_bytes) - reserved_bytes)
    if hw.ram_total_bytes is not None and hw.ram_total_bytes > 0:
        return max(0, int(hw.ram_total_bytes) - reserved_bytes)
    return None


def _estimate_kv_bytes_per_tok_per_layer(model: ModelAvailability) -> int:
    """Conservative Qwen-calibrated KV cache bytes per token per layer.

    Heuristic:
        max(FLOOR, min(CEILING, 256 * min_vram_gb))

    Rationale (calibration points):
        Qwen 3.5 2B  (min_vram=5  GB) → 1280 B  (real ~1024, slightly conservative)
        Qwen 3.5 9B  (min_vram=18 GB) → 4608 B  (real ~2048, very conservative)
        Qwen 3.5 27B (min_vram=16 GB) → 4096 B  (matches rough expectation)

    When ``min_vram_gb`` is unknown (0), fall back to the floor.
    """
    if model.min_vram_gb <= 0:
        return KV_BYTES_PER_TOK_PER_LAYER_FLOOR
    scaled = int(model.min_vram_gb) * KV_BYTES_PER_TOK_PER_LAYER_PER_GB
    return max(
        KV_BYTES_PER_TOK_PER_LAYER_FLOOR,
        min(scaled, KV_BYTES_PER_TOK_PER_LAYER_CEILING),
    )


def _per_layer_weight_bytes(model: ModelAvailability) -> float:
    """Derive per-layer weight bytes from catalogue metadata.

    Preferred source: ``shard_vram_gb * 1GB / (num_layers / shards_needed)``
    — the catalogue already encodes "this is how much VRAM one shard of this
    model needs", and a shard covers ``num_layers / shards_needed`` layers.

    Fallback: if ``shard_vram_gb`` is not populated, divide the whole-model
    ``min_vram_gb`` evenly across all layers. This over-estimates slightly
    because it bundles embed + lm_head into per-layer accounting, but stays
    conservative for Phase 1.
    """
    layers_total = max(1, model.num_layers)

    if model.shard_vram_gb > 0 and model.shards_needed > 0:
        layers_per_shard = max(1, layers_total / max(1, model.shards_needed))
        return float(model.shard_vram_gb) * (1024**3) / layers_per_shard

    if model.min_vram_gb > 0:
        return float(model.min_vram_gb) * (1024**3) / layers_total

    # Catalogue entry has no VRAM hints at all — return a pessimistic default
    # that will make the caller mark the model as incapable on small nodes.
    return 512.0 * 1024 * 1024  # 512 MB per layer


# ─── Public API ──────────────────────────────────────────────────────────────


def calculate_model_capacity(
    hardware: HardwareProfile,
    model: ModelAvailability,
    *,
    target_context: int = DEFAULT_TARGET_CONTEXT,
    reserved_system_mb: int = DEFAULT_RESERVED_SYSTEM_MB,
    activation_overhead_ratio: float = DEFAULT_ACTIVATION_OVERHEAD_RATIO,
) -> ModelCapacity:
    """Compute how many layers of ``model`` this peer can host.

    Formula:
        per_layer_total = weights + (kv_per_tok_per_layer * target_context)
                                  + (weights * activation_overhead_ratio)
        max_layers = floor(usable_memory / per_layer_total)

    ``max_layers`` is capped at the model's actual layer count.
    """
    usable_bytes = _usable_memory_bytes(hardware, reserved_system_mb)
    weights_per_layer = _per_layer_weight_bytes(model)
    kv_per_tok_per_layer = _estimate_kv_bytes_per_tok_per_layer(model)
    activation_per_layer = weights_per_layer * max(0.0, activation_overhead_ratio)

    per_layer_total = (
        weights_per_layer
        + (kv_per_tok_per_layer * max(1, int(target_context)))
        + activation_per_layer
    )

    if usable_bytes is None or per_layer_total <= 0:
        return ModelCapacity(
            model_id=model.model_id,
            num_layers_total=int(model.num_layers),
            max_layers_hostable=0,
            per_layer_weights_mb=weights_per_layer / (1024**2),
            kv_cache_per_token_kb=kv_per_tok_per_layer / 1024,
            activation_overhead_mb=activation_per_layer / (1024**2),
            target_context=int(target_context),
            recommended_quantization=str(model.recommended_quantization or "fp32"),
            can_host_full=False,
            can_shard=False,
            status=STATUS_INCAPABLE,
            reason=(
                "no_memory_reported"
                if usable_bytes is None
                else "invalid_per_layer_cost"
            ),
        )

    raw_max = int(usable_bytes // per_layer_total)
    # Cap at the model's layer count (can't host more than exist).
    max_layers = max(0, min(raw_max, int(model.num_layers)))

    if max_layers >= int(model.num_layers) and model.num_layers > 0:
        status = STATUS_CAPABLE
        reason = ""
    elif max_layers > 0:
        status = STATUS_SHARDABLE
        reason = ""
    else:
        status = STATUS_INCAPABLE
        reason = (
            f"per_layer_cost {per_layer_total / (1024**2):.0f} MB exceeds "
            f"available {usable_bytes / (1024**2):.0f} MB budget"
        )

    return ModelCapacity(
        model_id=model.model_id,
        num_layers_total=int(model.num_layers),
        max_layers_hostable=int(max_layers),
        per_layer_weights_mb=round(weights_per_layer / (1024**2), 1),
        kv_cache_per_token_kb=round(kv_per_tok_per_layer / 1024, 2),
        activation_overhead_mb=round(activation_per_layer / (1024**2), 1),
        target_context=int(target_context),
        recommended_quantization=str(model.recommended_quantization or "fp32"),
        can_host_full=bool(status == STATUS_CAPABLE),
        can_shard=bool(max_layers > 0),
        status=status,
        reason=reason,
    )


def build_capacity_report(
    *,
    hardware: HardwareProfile,
    catalog: list[ModelAvailability],
    peer_id: str = "",
    libp2p_peer_id: str = "",
    ports: dict[str, int] | None = None,
    advertise_host: str = "",
    requires_relay: bool = False,
    relay_circuits: list[str] | None = None,
    runtime_backend: str = "",
    accelerator_detail: str = "",
    target_context: int = DEFAULT_TARGET_CONTEXT,
    reserved_system_mb: int = DEFAULT_RESERVED_SYSTEM_MB,
    throughput_summary: dict[str, Any] | None = None,
) -> CapacityReport:
    """Build the full JSON-ready capacity payload for this peer.

    All arguments except ``hardware`` and ``catalog`` are optional and
    default to empty/zero so callers who only have partial metadata
    (e.g. during boot before the P2P layer has announced) can still
    produce a report.
    """
    ports = dict(ports or {})
    relay_circuits = list(relay_circuits or [])

    capacity = [
        calculate_model_capacity(
            hardware,
            model,
            target_context=target_context,
            reserved_system_mb=reserved_system_mb,
        )
        for model in catalog
    ]

    usable_bytes = _usable_memory_bytes(hardware, reserved_system_mb)

    hardware_dict: dict[str, Any] = {
        "accelerator": str(hardware.accelerator),
        "accelerator_detail": str(accelerator_detail or ""),
        "cuda_device_count": int(hardware.cuda_device_count),
        "ram_total_mb": _bytes_to_mb(hardware.ram_total_bytes),
        "ram_available_mb": _bytes_to_mb(hardware.ram_available_bytes),
        "vram_total_mb": _bytes_to_mb(hardware.vram_total_bytes),
        "vram_available_mb": _bytes_to_mb(hardware.vram_available_bytes),
        "reserved_system_mb": int(reserved_system_mb),
        "usable_memory_mb": _bytes_to_mb(usable_bytes),
        "runtime_backend": str(runtime_backend or ""),
    }

    network_dict: dict[str, Any] = {
        "advertise_host": str(advertise_host or ""),
        "ports": ports,
        "requires_relay": bool(requires_relay),
        "relay_circuits": relay_circuits,
    }

    throughput_dict: dict[str, Any] = dict(throughput_summary or {})

    return CapacityReport(
        schema_version=CAPACITY_SCHEMA_VERSION,
        peer_id=str(peer_id or ""),
        libp2p_peer_id=str(libp2p_peer_id or ""),
        captured_at_unix_ms=int(time.time() * 1000),
        hardware=hardware_dict,
        network=network_dict,
        capacity=capacity,
        throughput=throughput_dict,
    )


def _bytes_to_mb(value: int | None) -> int | None:
    if value is None:
        return None
    return int(int(value) // (1024 * 1024))
