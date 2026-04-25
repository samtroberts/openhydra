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

from __future__ import annotations

import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import logging
import math
import os
import threading
from typing import Any

from peer.pytorch_activation_compressor import PyTorchActivationCompressor
from peer.privacy import PyTorchDifferentialPrivacyNoise

def _is_model_cached_locally(model_id: str) -> bool:
    """Check if a HuggingFace model is already in the local cache.

    When True, callers can pass ``local_files_only=True`` to
    ``from_pretrained()`` to avoid hitting the HF Hub on every startup.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
        # Check for config.json — if it's cached, the model is downloaded.
        result = try_to_load_from_cache(model_id, "config.json")
        return result is not None and isinstance(result, str)
    except Exception:
        return False


_TINYLLAMA_MODEL_ID = "nickypro/tinyllama-15M"
_TINYLLAMA_CACHE_DIR = os.path.expanduser("~/.cache/openhydra/models/tinyllama-15M")
_TINYLLAMA_MLX_CACHE_DIR = os.path.expanduser("~/.cache/openhydra/models/tinyllama-15M-mlx")

_QUANTIZATION_BITS = {
    "fp32": 0,
    "int8": 8,
    "int4": 4,
}
_QUANTIZATION_ALIASES = {
    "none": "fp32",
    "8bit": "int8",
    "4bit": "int4",
}


def _normalize_quantization_mode(mode: str) -> str:
    normalized = str(mode or "fp32").strip().lower()
    normalized = _QUANTIZATION_ALIASES.get(normalized, normalized)
    if normalized not in _QUANTIZATION_BITS:
        return "fp32"
    return normalized


def _apply_quantization(values: list[float], quantization_bits: int) -> list[float]:
    bits = int(quantization_bits)
    if bits <= 0 or not values:
        return list(values)
    levels = (1 << bits) - 1
    step = 2.0 / float(levels)
    quantized: list[float] = []
    for value in values:
        clamped = max(-1.0, min(1.0, float(value)))
        bucket = round((clamped + 1.0) / step)
        reconstructed = (bucket * step) - 1.0
        quantized.append(max(-1.0, min(1.0, reconstructed)))
    return quantized


def _gpu_available_hint() -> bool:
    visible_devices = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    if not visible_devices:
        return False
    if visible_devices in {"-1", "none", "None"}:
        return False
    return True


def _default_trust_remote_code(model_id: str) -> bool:
    normalized = str(model_id or "").strip().lower()
    if not normalized:
        return False
    return "qwen" in normalized


def _tokenizer_eos_ids(tokenizer: Any) -> tuple[set[int], int]:
    eos_raw = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_raw, int):
        eos_ids = {int(eos_raw)}
    elif eos_raw is None:
        eos_ids = set()
    else:
        try:
            eos_ids = {int(item) for item in list(eos_raw)}
        except TypeError:
            eos_ids = set()
    eos_ids = {item for item in eos_ids if item >= 0}
    primary = min(eos_ids) if eos_ids else 0
    return eos_ids, primary


@dataclass(frozen=True)
class ToyShardConfig:
    model_id: str = "tinyllama-15M"
    shard_index: int = 0
    total_shards: int = 1
    broken: bool = False
    runtime_backend: str = "toy_auto"
    runtime_target: str = "auto"
    quantization_mode: str = "fp32"
    runtime_model_id: str = "nickypro/tinyllama-15M"
    runtime_trust_remote_code: bool | None = None
    runtime_layer_indices: tuple[int, ...] = ()
    runtime_max_context_tokens: int = 64
    runtime_kv_cache_max_entries: int = 256
    # Path A Phase 5: force loading of ``final_norm`` + ``lm_head`` (or the
    # tied-embed linear) on EVERY shard, not just the last. Enables a
    # non-last-shard peer to serve as the coordinator's head source when
    # ``--sample-on-coordinator`` is set and the coordinator is co-located
    # with a non-terminal stage (e.g. Mac-coordinator on stage 0, GPU on
    # the last stage). Default off — preserves today's memory footprint
    # on peers that don't need head weights.
    runtime_load_full_head: bool = False
    runtime_tensor_autoencoder_enabled: bool = False
    runtime_tensor_autoencoder_latent_dim: int = 1024
    runtime_privacy_noise_variance: float = 0.0
    runtime_privacy_audit_seed: str = "openhydra-tier3-dev-seed"
    runtime_peer_id: str = ""

    # ── KV cache compaction (Phases 1-4) ────────────────────────────────────
    # Phase 1/2/3/4: enable AM-based KV cache compaction.
    runtime_kv_compaction_enabled: bool = False
    # Key-selection algorithm: "hak" (fast) or "omp" (more accurate).
    runtime_kv_compaction_method: str = "hak"
    # Uniform fraction of tokens to keep (overridden by head_budget_path).
    runtime_kv_compaction_ratio: float = 0.10
    # Phase 2: fit β biases and Cv (requires patched attention layers).
    runtime_kv_compaction_beta: bool = False
    # Phase 3: path to a JSON file with per-layer / per-head budgets.
    runtime_kv_compaction_head_budget_path: str = ""
    # Phase 4: compact mid-trajectory when stored seq len exceeds this.
    runtime_kv_compaction_online: bool = False
    runtime_kv_compaction_online_max_tokens: int = 512

    # Auto mode (6.1): three-position toggle — "off" | "auto" | "on".
    # "off" disables compaction entirely (default).
    # "on" always compacts (same as runtime_kv_compaction_enabled=True).
    # "auto" compacts only when stored seq_len > runtime_kv_compaction_auto_threshold.
    runtime_kv_compaction_mode: str = "off"
    runtime_kv_compaction_auto_threshold: int = 512

    # Phase H: radix (longest-prefix) KV cache for cross-session prefix reuse.
    runtime_kv_radix_cache_enabled: bool = False
    runtime_kv_radix_cache_max_entries: int = 128
    runtime_kv_radix_cache_min_prefix_len: int = 16

    # Phase W: warmup — run a single dummy forward pass during __init__ to
    # JIT-compile GPU kernels.  On Apple MPS this moves ~30 s of Metal shader
    # compilation from the first real request to peer startup, where it is
    # expected and acceptable.  Has no effect on ToyRuntime.
    runtime_warmup_on_start: bool = False

    # ── Phase 2a: async pipeline depth ──────────────────────────────────────
    # Number of concurrent forward executions the runtime executor accepts.
    # Default 1 = today's serial behavior (one forward at a time per peer).
    # 2+ enables the coord-side pipeline_depth >= 2 path: multiple ring
    # tokens may be in-flight on the same peer simultaneously, with their
    # forward passes parallelised across executor workers. Sized via
    # ``coordinator/node.py --pipeline-depth`` (threaded through
    # ``peer.serve()`` into ``ToyShardConfig``). The executor's
    # ``max_workers`` is set to ``max(1, runtime_pipeline_depth)``.
    runtime_pipeline_depth: int = 1

    # MLX watchdog: maximum seconds for any single MLX computation (mx.eval,
    # stream_generate loop, forward_batch decode loop).  If exceeded, the
    # computation is treated as a GPU hang and a TimeoutError is raised.
    # Default raised from 30s to 120s to support 8 GB machines under memory pressure.
    runtime_mlx_eval_timeout_s: float = 120.0

    # ── Tokenizer alignment (heterogeneous MLX ↔ PyTorch rings) ───────────
    # Canonical HuggingFace model id (e.g. "Qwen/Qwen3.5-2B") used to load
    # the *tokenizer* across every backend. When set and
    # ``runtime_mlx_force_hf_tokenizer`` is True, MLXRuntime discards the
    # tokenizer bundled with the mlx-community checkpoint and loads the HF
    # tokenizer instead, guaranteeing that token IDs encoded on an MLX peer
    # are valid indices into a PyTorch peer's HF embedding table downstream.
    runtime_hf_model_id: str = ""
    runtime_mlx_force_hf_tokenizer: bool = True
    # Startup guard: refuse to advertise if tokenizer.vocab_size does not
    # match the loaded embedding table's num_embeddings.
    runtime_tokenizer_vocab_guard: bool = True


@dataclass(frozen=True)
class RuntimeProfile:
    backend: str
    target: str
    quantization_mode: str
    quantization_bits: int
    gpu_available: bool
    estimated_tokens_per_sec: float
    estimated_memory_mb: int
    runtime_model_id: str = "openhydra-toy-345m"
    layer_start: int = 0
    layer_end: int = 0
    # Phase 3: total transformer layers in the full model (0 = unknown / not sharded).
    total_layers: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _DecoderArchitecture:
    family: str
    layers: tuple[Any, ...]
    embed_tokens: Any
    position_embeddings: Any | None = None
    final_norm: Any | None = None
    rotary_emb: Any | None = None
    # ── Gemma 4 sharded adapter extras (Phase 4) ───────────────────────
    # Populated only when ``family == "gemma4"``; ignored by every other
    # branch. These capture the layer-type-aware rotary + causal mask +
    # per-layer-input plumbing that Gemma 4 needs but llama / qwen_llama
    # don't. Carrying them inside ``_DecoderArchitecture`` keeps the
    # architecture detection output self-contained — no extra attributes
    # bolted onto ``PyTorchRuntime``.
    layer_types: tuple[str, ...] = ()        # config.layer_types[i] → "full_attention" | "sliding_attention"
    per_layer_embed: Any | None = None        # model.language_model.embed_tokens_per_layer
    per_layer_proj: Any | None = None         # model.language_model.per_layer_model_projection
    per_layer_norm: Any | None = None         # model.language_model.per_layer_projection_norm
    hidden_size_per_layer: int = 0            # config.hidden_size_per_layer_input
    text_model: Any | None = None             # Gemma4TextModel instance (for get_per_layer_inputs/project_per_layer_inputs)


def _count_model_layers(model_name: str) -> int:
    """Read num_hidden_layers from a model's config.json without loading weights.

    For multimodal models (Gemma 4, Qwen 3.5 VL), the text decoder config
    is nested under ``text_config``.
    """
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=False,
            local_files_only=_is_model_cached_locally(model_name),
        )
        n = int(getattr(config, "num_hidden_layers", 0))
        if n > 0:
            return n
        # Multimodal models: layers count is inside text_config
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            return int(getattr(text_config, "num_hidden_layers", 0))
        return 0
    except Exception:
        return 0


def _detect_layer_prefix(model_name: str) -> str:
    """Device-map key prefix for transformer layers.

    All supported families load into a text-only ``*ForCausalLM`` class that
    exposes ``model.layers.N`` as the decoder module path. This holds for:

    - LLaMA / Qwen 2.x / Qwen 3 / Gemma 2-3 — native text-only checkpoints
    - Qwen 3.5 — ``AutoModelForCausalLM`` picks ``Qwen3_5ForCausalLM`` which
      unwraps the multimodal checkpoint's ``model.language_model.*`` keys to
      the text-only tree automatically
    - Gemma 4 — takes a different path entirely (multimodal strip) because
      its auto-class is ``Gemma4ForConditionalGeneration`` with vision/audio
      towers that cannot be cleanly device-mapped.
    """
    return "model.layers"


# Gemma 4 requires the CPU-strip-and-move path because the checkpoint keys
# under ``model.language_model.embed_tokens_per_layer`` and the per-layer
# projection cannot be cleanly device-mapped via accelerate. Qwen 3.5 has a
# simpler multimodal wrapper — text decoder at ``model.language_model.*``,
# vision at ``model.visual.*``, optional ``mtp`` head — and CAN be handled
# via selective device_map without loading everything into CPU RAM.
_MULTIMODAL_MODEL_TYPES = frozenset({"gemma4"})

_MULTIMODAL_AUX_ATTRS = (
    "vision_tower",
    "audio_tower",
    "embed_vision",
    "embed_audio",
    "multi_modal_projector",
)


def _is_multimodal_model_type(model_name: str) -> bool:
    """True for model types whose checkpoints contain a vision/audio tower
    alongside the text decoder (Gemma 4, Qwen 3.5 VL, ...).

    Safe to call with just a local path — reads only ``config.json``.
    """
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=False,
            local_files_only=_is_model_cached_locally(model_name),
        )
        mt = str(getattr(config, "model_type", "")).strip().lower()
        return mt in _MULTIMODAL_MODEL_TYPES
    except Exception:
        return False


def _find_decoder_layer_list(model: Any) -> Any | None:
    """Locate the decoder layers ``nn.ModuleList`` on a HuggingFace model.

    Supports all structures OpenHydra currently handles:

    - Standard ``Qwen3_5ForCausalLM`` / Qwen / LLaMA: ``model.model.layers``
    - Gemma 4 ``ForConditionalGeneration``: ``model.model.language_model.layers``
    - Single-level wrapper: ``model.language_model.layers``

    Returns the module list (so callers can mutate entries in place) or
    ``None`` if no known path matches.
    """
    # Path 1: model.model.layers (standard text-only CausalLM)
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return inner.layers

    # Path 2: model.model.language_model.layers (Gemma 4 multimodal wrapper)
    if inner is not None:
        lm = getattr(inner, "language_model", None)
        if lm is not None and hasattr(lm, "layers"):
            return lm.layers

    # Path 3: model.language_model.layers (single-level wrapper)
    lm = getattr(model, "language_model", None)
    if lm is not None and hasattr(lm, "layers"):
        return lm.layers

    return None


def _replace_offloaded_layers_with_identity(
    model: Any,
    kept_layer_indices: tuple[int, ...],
) -> int:
    """Replace every decoder layer whose index is **not** in
    ``kept_layer_indices`` with ``nn.Identity()`` in the model's ``ModuleList``.

    Why this exists
    ---------------
    For sharded peers we build an accelerate ``device_map`` that sends
    unused layers to the ``"disk"`` device. This leaves meta-device
    placeholder modules in the real model's ``nn.ModuleList``. ANY code
    path that iterates the full list (``model.model.layers[:]``) and
    touches those placeholders triggers the torch dispatcher error
    ``GET was unable to find an engine to execute this computation``.

    Concrete reproduction: with a Qwen3.5-9B peer running layers 0-15,
    the KV-cache-aware decode path (streaming ``infer_chat_stream``)
    passes ``use_cache=True`` and transformers' ``DynamicCache`` /
    ``_update_causal_mask`` internals iterate all 32 layer slots,
    eventually dispatching an op on a meta tensor → crash.

    Replacing the out-of-shard entries with ``nn.Identity()`` is safe:

    - Indexing (``layers[layer_idx]``) stays valid; the block's
      ``layer_idx`` attribute (set at model construction) still matches
      its position in the list.
    - ``nn.Identity(*args, **kwargs)`` returns its first positional
      argument unchanged, so any ``block(hidden, ...)`` call on a
      replaced slot is a pass-through (not used by OpenHydra's
      ``_run_layers``, which only iterates ``self._selected_layers``,
      but harmless if anything else touches it).
    - Identity modules have zero parameters and live on no device, so
      iteration through ``model.modules()`` / ``model.parameters()``
      no longer yields meta tensors.

    Returns the number of layers that were replaced.
    """
    from torch import nn

    layers = _find_decoder_layer_list(model)
    if layers is None:
        return 0

    kept = set(int(i) for i in kept_layer_indices)
    replaced = 0
    for i in range(len(layers)):
        if i in kept:
            continue
        existing = layers[i]
        # Don't replace if it's already an Identity (idempotent) or if
        # it's None (already cleared by legacy ``self._blocks[idx] = None``
        # cleanup — the ModuleList still holds the original reference).
        if isinstance(existing, nn.Identity):
            continue
        layers[i] = nn.Identity()
        replaced += 1

    if replaced > 0:
        import gc
        gc.collect()

    return replaced


def _strip_multimodal_components(
    model: Any,
    layer_indices: tuple[int, ...],
    target_device: int | str,
    dtype: Any,
) -> int:
    """In-place strip auxiliary multimodal components + unused text layers.

    Replaces ``vision_tower``/``audio_tower``/``embed_*``/``multi_modal_projector``
    on the outer model with ``None`` to drop their parameters, replaces unused
    text-decoder layers with ``nn.Identity()`` placeholders via
    ``_replace_offloaded_layers_with_identity``, and moves the remaining text
    decoder + ``lm_head`` to ``target_device`` in the requested ``dtype``.

    Returns the count of layers that were replaced with Identity placeholders.
    """
    import torch
    from torch import nn

    inner_model = getattr(model, "model", None)  # Gemma4Model / Qwen3_5Model
    if inner_model is None:
        return 0

    # Text decoder lives at model.model.language_model (Gemma4TextModel,
    # Qwen3_5TextModel, ...). Falls back to direct language_model if absent.
    text_decoder = getattr(inner_model, "language_model", None)
    if text_decoder is None:
        text_decoder = getattr(model, "language_model", None)
    if text_decoder is None or not hasattr(text_decoder, "layers"):
        return 0

    # Drop aux towers on the outer wrapper — each is a few GB.
    for attr in _MULTIMODAL_AUX_ATTRS:
        if hasattr(inner_model, attr) and isinstance(
            getattr(inner_model, attr), nn.Module
        ):
            setattr(inner_model, attr, None)

    # Replace out-of-shard layers with Identity so the ModuleList indexing
    # stays valid. Shared helper with the standard sharded path.
    replaced = _replace_offloaded_layers_with_identity(model, tuple(layer_indices))

    # Move the surviving text decoder + lm_head to the target device.
    if target_device == "cpu":
        text_decoder.to(dtype=dtype)
        if hasattr(model, "lm_head") and model.lm_head is not None:
            model.lm_head.to(dtype=dtype)
    else:
        device_str = f"cuda:{target_device}" if isinstance(target_device, int) else target_device
        text_decoder.to(device=device_str, dtype=dtype)
        if hasattr(model, "lm_head") and model.lm_head is not None:
            model.lm_head.to(device=device_str, dtype=dtype)

    return replaced


def _build_selective_device_map(
    model_name: str,
    layer_indices: tuple[int, ...],
    target_device: int | str = 0,
) -> dict[str, int | str]:
    """Build a device_map that loads only the assigned layers into real memory.

    Unassigned transformer layers are mapped to ``"disk"`` which causes
    accelerate to offload them (effectively zero resident memory).
    Only the assigned layers + shared components (embeddings, norm, lm_head)
    are loaded onto the target device.

    Requires ``accelerate>=1.13.0``.
    """
    total = _count_model_layers(model_name)
    if total <= 0:
        return {"": target_device}  # Fallback: load everything

    layer_set = set(int(i) for i in layer_indices)
    dm: dict[str, int | str] = {
        "model.embed_tokens": target_device,
        "model.norm": target_device,
        "model.rotary_emb": target_device,
        "lm_head": target_device,
    }
    for i in range(total):
        dm[f"model.layers.{i}"] = target_device if i in layer_set else "disk"

    loaded = len(layer_set & set(range(total)))
    logging.info(
        "selective_device_map: %d/%d layers on device %s, %d offloaded",
        loaded, total, target_device, total - loaded,
    )
    return dm


_tinyllama_cache: dict[str, Any] = {}


def _get_tinyllama_cached() -> tuple[Any, Any]:
    """Load tinyllama-15M once per process and cache in module global."""
    if "model" in _tinyllama_cache:
        return _tinyllama_cache["model"], _tinyllama_cache["tokenizer"]

    cache_dir = _TINYLLAMA_CACHE_DIR
    safetensors_path = os.path.join(cache_dir, "model.safetensors")
    if not os.path.exists(safetensors_path):
        logging.info("tinyllama: downloading %s to %s", _TINYLLAMA_MODEL_ID, cache_dir)
        from huggingface_hub import snapshot_download
        snapshot_download(_TINYLLAMA_MODEL_ID, local_dir=cache_dir)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cache_dir, local_files_only=True,
        trust_remote_code=_default_trust_remote_code(_TINYLLAMA_MODEL_ID),
    )
    model = AutoModelForCausalLM.from_pretrained(
        cache_dir, local_files_only=True,
        trust_remote_code=_default_trust_remote_code(_TINYLLAMA_MODEL_ID),
    )
    model.eval()

    _tinyllama_cache["model"] = model
    _tinyllama_cache["tokenizer"] = tokenizer
    return model, tokenizer


class ToyRuntime:
    """Lightweight real LLM runtime using tinyllama-15M (15M params, 29 MB).

    Replaces the legacy hash-based mock with a real LLaMA model that
    generates coherent text.  Loads from a local HuggingFace cache on
    first use (~3s download if not cached).  Uses MLX on macOS Apple
    Silicon (~860 tok/s) with PyTorch CPU fallback (~50 tok/s).

    Model: nickypro/tinyllama-15M (LLaMA architecture, safetensors)
    """

    def __init__(self, config: ToyShardConfig):
        self.config = config
        self.last_forward_thread_id: int | None = None
        self._use_mlx = False
        self._mlx_model = None
        self._mlx_tokenizer = None

        # Load model via module-level cache (one load per process)
        import torch
        self._torch = torch
        self._model, self._tokenizer = _get_tinyllama_cached()

        self._runtime_profile = self._build_runtime_profile(config, use_mlx=False)

    @classmethod
    def _build_runtime_profile(cls, config: ToyShardConfig, use_mlx: bool = False) -> RuntimeProfile:
        gpu_available = _gpu_available_hint()
        quantization_mode = _normalize_quantization_mode(config.quantization_mode)
        quantization_bits = _QUANTIZATION_BITS[quantization_mode]

        return RuntimeProfile(
            backend="tinyllama_mlx" if use_mlx else "tinyllama",
            target="metal" if use_mlx else "cpu",
            quantization_mode=quantization_mode,
            quantization_bits=quantization_bits,
            gpu_available=gpu_available,
            estimated_tokens_per_sec=860.0 if use_mlx else 50.0,
            estimated_memory_mb=60,
            runtime_model_id=_TINYLLAMA_MODEL_ID,
            layer_start=int(config.shard_index),
            layer_end=int(config.shard_index + 1),
        )

    def runtime_profile(self) -> dict[str, Any]:
        return self._runtime_profile.to_dict()

    def reshard(self, new_layer_start: int, new_layer_end: int, total_layers: int) -> bool:
        """Update shard metadata (no physical resharding for ToyRuntime)."""
        self._runtime_profile = RuntimeProfile(
            backend=self._runtime_profile.backend,
            target=self._runtime_profile.target,
            quantization_mode=self._runtime_profile.quantization_mode,
            quantization_bits=self._runtime_profile.quantization_bits,
            gpu_available=self._runtime_profile.gpu_available,
            estimated_tokens_per_sec=self._runtime_profile.estimated_tokens_per_sec,
            estimated_memory_mb=self._runtime_profile.estimated_memory_mb,
            runtime_model_id=self._runtime_profile.runtime_model_id,
            layer_start=int(new_layer_start),
            layer_end=int(new_layer_end),
            total_layers=int(total_layers),
        )
        return True

    def encode_prompt(self, prompt: str, max_tokens: int) -> list[float]:
        """Tokenize prompt and return token IDs as floats."""
        ids = self._tokenizer.encode(str(prompt or ""), add_special_tokens=False)
        return [float(t) for t in ids[:max(1, max_tokens)]]

    def forward(
        self,
        prompt: str,
        activation: list[float],
        max_tokens: int,
        stage_index: int = 0,
        total_stages: int = 1,
        kv_session_id: str | None = None,
        kv_store_activation: bool = False,
        kv_use_cached_activation: bool = False,
        request_id: str | None = None,
        decode_do_sample: bool | None = None,
        decode_temperature: float | None = None,
        decode_top_p: float | None = None,
        decode_top_k: int | None = None,
        decode_seed: int | None = None,
    ) -> list[float]:
        """Run real inference and return generated token IDs as floats."""
        self.last_forward_thread_id = threading.get_ident()
        gen_count = max(1, min(int(max_tokens), 256))

        # ── MLX fast path (Apple Silicon GPU) ─────────────────────────────
        if self._use_mlx:
            return self._forward_mlx(prompt, activation, gen_count, decode_temperature)

        # ── PyTorch CPU path ──────────────────────────────────────────────
        text = str(prompt or "")
        if not text and activation:
            input_ids = self._torch.tensor(
                [[max(0, int(round(v))) for v in activation]],
                dtype=self._torch.long,
            )
        else:
            input_ids = self._tokenizer(text, return_tensors="pt")["input_ids"]

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": gen_count,
            "do_sample": bool(decode_do_sample) if decode_do_sample is not None else False,
        }
        if decode_temperature is not None and float(decode_temperature) > 0:
            gen_kwargs["temperature"] = float(decode_temperature)
            gen_kwargs["do_sample"] = True
        if decode_top_p is not None and float(decode_top_p) > 0:
            gen_kwargs["top_p"] = float(decode_top_p)
        if decode_top_k is not None and int(decode_top_k) > 0:
            gen_kwargs["top_k"] = int(decode_top_k)

        # Intentional corruption for Mystery Shopper tests
        if self.config.broken:
            gen_kwargs["temperature"] = 99.0
            gen_kwargs["do_sample"] = True

        with self._torch.no_grad():
            output = self._model.generate(input_ids=input_ids, **gen_kwargs)

        # Return only the NEW tokens (exclude prompt tokens)
        prompt_len = input_ids.shape[1]
        new_tokens = output[0][prompt_len:].tolist()
        return [float(t) for t in new_tokens]

    def _forward_mlx(
        self,
        prompt: str,
        activation: list[float],
        max_tokens: int,
        temperature: float | None = None,
    ) -> list[float]:
        """MLX fast path: generate all tokens via stream_generate on Metal GPU."""
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        text = str(prompt or "")
        if not text and activation:
            # Decode activation token IDs back to text for continuation
            ids = [max(0, int(round(v))) for v in activation]
            text = self._mlx_tokenizer.decode(ids, skip_special_tokens=True)

        temp = float(temperature) if temperature and float(temperature) > 0 else 0.0
        sampler = make_sampler(temp=temp)

        tokens: list[int] = []
        for resp in stream_generate(
            self._mlx_model,
            self._mlx_tokenizer,
            prompt=text,
            max_tokens=max_tokens,
            sampler=sampler,
        ):
            tokens.append(int(resp.token))
            if resp.finish_reason is not None:
                break

        return [float(t) for t in tokens]

    def forward_batch(self, items: list[Any]) -> list[list[float]]:
        """Sequential forward for each item (CPU model, no batching benefit)."""
        return [
            list(
                self.forward(
                    item.prompt,
                    item.activation,
                    item.max_tokens,
                    stage_index=item.stage_index,
                    total_stages=item.total_stages,
                    request_id=item.request_id,
                    decode_do_sample=item.decode_do_sample,
                    decode_temperature=item.decode_temperature,
                    decode_top_p=item.decode_top_p,
                    decode_top_k=item.decode_top_k,
                    decode_seed=item.decode_seed,
                )
            )
            for item in items
        ]

    def compaction_stats(self) -> "dict[str, int | float]":
        """Stub — ToyRuntime does not perform KV cache compaction."""
        return {
            "compact_calls": 0,
            "compact_tokens_before": 0,
            "compact_tokens_after": 0,
            "compact_tokens_saved": 0,
            "compact_latency_s": 0.0,
            "kv_cache_hits": 0,
            "kv_cache_misses": 0,
        }


class PyTorchRuntime:
    def __init__(self, config: ToyShardConfig):
        self.config = config
        self.last_forward_thread_id: int | None = None
        self.last_kv_cache_hit: bool = False
        self._compressor: PyTorchActivationCompressor | None = None
        self._privacy_noise: PyTorchDifferentialPrivacyNoise | None = None
        self.quantization_mode = _normalize_quantization_mode(config.quantization_mode)
        self.quantization_bits = _QUANTIZATION_BITS[self.quantization_mode]
        self.model_name = str(config.runtime_model_id or "gpt2").strip() or "gpt2"
        requested_trust_remote_code = config.runtime_trust_remote_code
        if requested_trust_remote_code is None:
            self._trust_remote_code = _default_trust_remote_code(self.model_name)
        else:
            self._trust_remote_code = bool(requested_trust_remote_code)
        self.max_context_tokens = max(8, int(config.runtime_max_context_tokens))
        # Phase 2a: pipeline depth controls executor concurrency. depth=1
        # preserves today's strict serial single-worker behavior;
        # depth>=2 lets concurrent ForwardRequests (different slot_ids on
        # the same ring) execute their forward passes in parallel.
        _pt_workers = max(1, int(getattr(config, "runtime_pipeline_depth", 1) or 1))
        self._executor = ThreadPoolExecutor(
            max_workers=_pt_workers,
            thread_name_prefix=f"pytorch-shard-{int(config.shard_index)}",
        )
        self._kv_cache_max_entries = max(1, int(config.runtime_kv_cache_max_entries))
        self._kv_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._last_noise_applied = False
        self._last_noise_observed_variance = 0.0
        self._last_noise_observed_std = 0.0
        self._last_noise_payload_index = 0
        self._last_noise_audit_tag = ""

        # ── Compaction SLO counters (Phase D) ────────────────────────────────
        self._compact_calls: int = 0
        self._compact_tokens_before: int = 0
        self._compact_tokens_after: int = 0
        self._compact_latency_s: float = 0.0
        self._compact_kv_cache_hits: int = 0
        self._compact_kv_cache_misses: int = 0
        self._compact_lock = threading.Lock()
        # ── Auto-mode VRAM-aware counters (Pass 6) ────────────────────────
        self._auto_skip_count: int = 0
        self._auto_trigger_count: int = 0

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - dependency controlled in integration tests.
            raise RuntimeError(
                "pytorch_runtime_unavailable: install optional deps 'torch' and 'transformers'"
            ) from exc

        self._torch = torch
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        requested_target = str(config.runtime_target or "auto").strip().lower()
        if requested_target not in {"auto", "cpu", "cuda", "mps"}:
            requested_target = "auto"
        gpu_available = bool(torch.cuda.is_available())
        mps_available = bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        target = requested_target
        if target == "auto":
            if gpu_available:
                target = "cuda"
            elif mps_available:
                target = "mps"
            else:
                target = "cpu"
        if target == "cuda" and not gpu_available:
            target = "cpu"
        if target == "mps" and not mps_available:
            target = "cpu"

        requested_backend = str(config.runtime_backend or "pytorch_auto").strip().lower()
        if requested_backend not in {"pytorch_auto", "pytorch_cpu", "pytorch_cuda", "pytorch_mps", "pytorch"}:
            requested_backend = "pytorch_auto"
        if requested_backend in {"pytorch_cpu"}:
            target = "cpu"
        elif requested_backend in {"pytorch_cuda"} and gpu_available:
            target = "cuda"
        elif requested_backend in {"pytorch_mps"} and mps_available:
            target = "mps"
        backend = (
            "pytorch_cuda" if target == "cuda"
            else "pytorch_mps" if target == "mps"
            else "pytorch_cpu"
        )

        # Skip HuggingFace Hub API calls when the model is already cached.
        # Saves 2-5s of startup time and avoids network dependency.
        _local_only = _is_model_cached_locally(self.model_name)
        if _local_only:
            logging.info("model %s found in local cache — skipping HF Hub checks", self.model_name)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self._trust_remote_code,
            local_files_only=_local_only,
        )
        quantization_config = None
        if self.quantization_bits in {4, 8}:
            if target != "cuda" or not gpu_available:
                logging.warning(
                    "PyTorch quantization (%s) requested for model '%s' without CUDA; falling back to cpu/fp32",
                    self.quantization_mode,
                    self.model_name,
                )
                target = "cpu"
                backend = "pytorch_cpu"
                self.quantization_mode = "fp32"
                self.quantization_bits = 0
            else:
                try:
                    from transformers import BitsAndBytesConfig

                    if self.quantization_bits == 4:
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True,
                        )
                    else:
                        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                except Exception as exc:
                    logging.warning(
                        "PyTorch quantization (%s) unavailable for model '%s': %s; falling back to cpu/fp32",
                        self.quantization_mode,
                        self.model_name,
                        exc,
                    )
                    target = "cpu"
                    backend = "pytorch_cpu"
                    self.quantization_mode = "fp32"
                    self.quantization_bits = 0
                    quantization_config = None

        if target == "cuda":
            self._device = torch.device("cuda")
            self._dtype = torch.float16
        elif target == "mps":
            self._device = torch.device("mps")
            self._dtype = torch.float16  # MPS supports fp16 on all Apple Silicon
        else:
            self._device = torch.device("cpu")
            self._dtype = torch.float32

        load_kwargs: dict[str, Any] = {
            "low_cpu_mem_usage": True,
        }
        quantized_weights_loaded = False
        # For sharded deployments (total_shards > 1), force float16 + device_map
        # to minimize peak memory on constrained devices (1GB nanodes).
        _is_sharded = int(config.total_shards) > 1 or bool(config.runtime_layer_indices)
        # Probe the checkpoint's native dtype — if the model was trained in
        # bfloat16 and has bf16 buffers we can't convert (e.g. Qwen 3.5's
        # linear_attn state-space params), forcing fp16 causes matmul dtype
        # mismatches partway through the forward pass. Prefer the native
        # checkpoint dtype; fall back to fp16.
        _native_dtype = torch.float16
        try:
            from transformers import AutoConfig as _AC
            _cfg_probe = _AC.from_pretrained(
                self.model_name, trust_remote_code=False,
                local_files_only=_local_only,
            )
            _probe_dtype = getattr(_cfg_probe, "torch_dtype", None)
            if _probe_dtype is None and hasattr(_cfg_probe, "text_config"):
                _probe_dtype = getattr(_cfg_probe.text_config, "torch_dtype", None)
            if isinstance(_probe_dtype, str):
                _probe_dtype = getattr(torch, _probe_dtype, None)
            if _probe_dtype in (torch.bfloat16, torch.float16, torch.float32):
                _native_dtype = _probe_dtype
        except Exception:
            pass
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"
            load_kwargs["torch_dtype"] = _native_dtype
        elif _is_sharded:
            self._dtype = _native_dtype  # Override fp32 default for memory-constrained shards
            load_kwargs["torch_dtype"] = _native_dtype
            # Selective weight loading: compute which layers this shard needs
            # BEFORE loading the model, then map unneeded layers to "disk"
            # device (zero memory). This reduces peak memory from full-model
            # to shard-size + embeddings + lm_head.
            _pre_layer_indices = self._resolve_layer_indices(
                total_layers=_count_model_layers(self.model_name),
                shard_index=max(0, int(config.shard_index)),
                total_shards=max(1, int(config.total_shards)),
                explicit_indices=tuple(config.runtime_layer_indices),
            )

            _is_multimodal = _is_multimodal_model_type(self.model_name)

            # Detect if the checkpoint has built-in quantization (FP8, etc.)
            # that causes dequantization during loading, inflating peak VRAM
            # beyond what selective device_map can handle on small GPUs.
            _has_builtin_quant = False
            try:
                from transformers import AutoConfig as _QAC
                _qcfg = _QAC.from_pretrained(
                    self.model_name, trust_remote_code=False,
                    local_files_only=_local_only,
                )
                _qc = getattr(_qcfg, "quantization_config", None)
                if isinstance(_qc, dict) and _qc.get("quant_method") in ("fp8", "gptq", "awq"):
                    _has_builtin_quant = True
            except Exception:
                pass

            if _pre_layer_indices and (_is_multimodal or _has_builtin_quant):
                # Multimodal checkpoints (Gemma 4) and FP8-quantized checkpoints
                # need the CPU-first load path. Multimodal because the text
                # decoder lives at ``model.model.language_model`` and selective
                # device_map chokes on the layout. FP8 because dequantization
                # to fp16 during loading inflates peak VRAM — 16 layers of a
                # 27B model at fp16 + embeddings can exceed 15GB T4 VRAM.
                # Strategy: load everything into CPU RAM, then strip unused
                # layers before moving the shard onto the target device.
                load_kwargs["device_map"] = {"": "cpu"}
                load_kwargs["low_cpu_mem_usage"] = True
                load_kwargs["torch_dtype"] = _native_dtype
                if _is_multimodal:
                    self._multimodal_strip_layers = tuple(_pre_layer_indices)
                    self._multimodal_strip_dtype = _native_dtype
                else:
                    # Reuse the MPS strip path but target CUDA/CPU as needed.
                    # The post-load handler moves layers to the correct device.
                    self._cpu_first_strip_layers = tuple(_pre_layer_indices)
                    self._cpu_first_strip_dtype = _native_dtype
                    self._cpu_first_strip_device = target  # "cuda" or "cpu"
            elif _pre_layer_indices and target == "mps":
                # MPS sharding: accelerate's ``device_map`` doesn't support
                # MPS as a target device. Load everything to CPU first, then
                # after load move ONLY the kept layers + shared components
                # (embed_tokens, norm, lm_head) onto MPS. Mirrors the Gemma 4
                # multimodal strip path. Phase 2's Identity swap cleans up
                # the unused layers before the move.
                load_kwargs["device_map"] = {"": "cpu"}
                load_kwargs["low_cpu_mem_usage"] = True
                load_kwargs["torch_dtype"] = _native_dtype
                self._mps_strip_layers = tuple(_pre_layer_indices)
                self._mps_strip_dtype = _native_dtype
            elif _pre_layer_indices:
                # Use device index 0 for CUDA, "cpu" string for CPU-only.
                _sel_device: int | str = 0 if target == "cuda" else "cpu"
                load_kwargs["device_map"] = _build_selective_device_map(
                    model_name=self.model_name,
                    layer_indices=_pre_layer_indices,
                    target_device=_sel_device,
                )
                # offload_folder required for "disk"-mapped layers
                import tempfile as _tempfile
                self._offload_dir = _tempfile.mkdtemp(prefix="openhydra-offload-")
                load_kwargs["offload_folder"] = self._offload_dir
            else:
                load_kwargs["device_map"] = {"": "cpu"}  # Fallback
        else:
            load_kwargs["torch_dtype"] = self._dtype

        try:
            # Skip caching_allocator_warmup for selective loading — it tries
            # to pre-allocate ALL real-device parameters as one contiguous
            # block, which OOMs on memory-constrained devices even though the
            # actual per-layer allocations would fit fine.
            _patched_warmup = False
            if _is_sharded and "device_map" in load_kwargs:
                try:
                    import transformers.modeling_utils as _tmu
                    _orig_warmup = getattr(_tmu, "caching_allocator_warmup", None)
                    if _orig_warmup is not None:
                        _tmu.caching_allocator_warmup = lambda *a, **kw: None
                        _patched_warmup = True
                except Exception:
                    pass

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=self._trust_remote_code,
                local_files_only=_local_only,
                **load_kwargs,
            )
            quantized_weights_loaded = quantization_config is not None

            if _patched_warmup and _orig_warmup is not None:
                _tmu.caching_allocator_warmup = _orig_warmup
        except Exception as exc:
            if quantization_config is None:
                raise
            logging.warning(
                "PyTorch quantized load failed for model '%s': %s; retrying with cpu/fp32",
                self.model_name,
                exc,
            )
            self.quantization_mode = "fp32"
            self.quantization_bits = 0
            backend = "pytorch_cpu"
            target = "cpu"
            self._device = torch.device("cpu")
            self._dtype = torch.float32
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                low_cpu_mem_usage=True,
                torch_dtype=self._dtype,
                trust_remote_code=self._trust_remote_code,
                local_files_only=_local_only,
            )
            quantized_weights_loaded = False

        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Multimodal strip path: drop aux towers + unused layers, then move
        # only the text decoder onto the target device. Skips the generic
        # .to(device) below because we've already placed the kept modules.
        _multimodal_stripped = False
        _strip_layer_indices = getattr(self, "_multimodal_strip_layers", None)
        if _strip_layer_indices:
            _strip_target: int | str = (
                0 if target == "cuda"
                else "mps" if target == "mps"
                else "cpu"
            )
            try:
                _strip_dtype = getattr(self, "_multimodal_strip_dtype", None) or torch.float16
                _replaced = _strip_multimodal_components(
                    self._model,
                    layer_indices=_strip_layer_indices,
                    target_device=_strip_target,
                    dtype=_strip_dtype,
                )
                logging.info(
                    "multimodal_strip: model=%s kept=%d replaced=%d target=%s",
                    self.model_name, len(_strip_layer_indices), _replaced, _strip_target,
                )
                _multimodal_stripped = True
            except Exception as _strip_exc:
                logging.warning(
                    "multimodal_strip_failed: model=%s err=%s — falling back to full .to()",
                    self.model_name, _strip_exc,
                )
            self._multimodal_strip_layers = None

        # MPS sharded post-load: move kept layers + shared components from
        # CPU → MPS after Identity-swapping the unused layers. This mirrors
        # the multimodal strip flow but for any model family on Apple Silicon.
        _mps_stripped = False
        _mps_layer_indices = getattr(self, "_mps_strip_layers", None)
        if _mps_layer_indices:
            try:
                _mps_dtype = getattr(self, "_mps_strip_dtype", None) or torch.float16
                # Phase 2 Identity swap: eliminate meta-device modules
                _id_replaced = _replace_offloaded_layers_with_identity(
                    self._model, tuple(_mps_layer_indices),
                )
                # Move surviving layers to MPS
                _decoder_layers = _find_decoder_layer_list(self._model)
                if _decoder_layers is not None:
                    for _li in _mps_layer_indices:
                        if _li < len(_decoder_layers):
                            _decoder_layers[_li].to(device="mps", dtype=_mps_dtype)
                # Move shared components (embed_tokens, norm, rotary_emb, lm_head)
                _inner = getattr(self._model, "model", self._model)
                # For Qwen 3.5, AutoModelForCausalLM picks the text-only class,
                # so _inner is the text model with .layers/.embed_tokens/.norm.
                # For models with a .language_model wrapper, look there too.
                _lm_inner = getattr(_inner, "language_model", None) or _inner
                for _attr in ("embed_tokens", "norm", "rotary_emb"):
                    _mod = getattr(_lm_inner, _attr, None)
                    if _mod is not None and isinstance(_mod, torch.nn.Module):
                        _mod.to(device="mps", dtype=_mps_dtype)
                if hasattr(self._model, "lm_head") and self._model.lm_head is not None:
                    self._model.lm_head.to(device="mps", dtype=_mps_dtype)
                import gc as _gc
                _gc.collect()
                logging.info(
                    "mps_shard_strip: model=%s kept=%d identity_replaced=%d device=mps",
                    self.model_name, len(_mps_layer_indices), _id_replaced,
                )
                _mps_stripped = True
            except Exception as _mps_exc:
                logging.warning(
                    "mps_shard_strip_failed: model=%s err=%s — falling back to CPU",
                    self.model_name, _mps_exc,
                )
            self._mps_strip_layers = None

        # CPU-first strip for FP8/quantized CUDA shards: load to CPU, swap
        # unused layers with Identity, then move kept layers to CUDA.
        _cpu_first_layer_indices = getattr(self, "_cpu_first_strip_layers", None)
        if _cpu_first_layer_indices:
            try:
                _cf_dtype = getattr(self, "_cpu_first_strip_dtype", None) or torch.float16
                _cf_device = getattr(self, "_cpu_first_strip_device", "cuda")
                _id_replaced = _replace_offloaded_layers_with_identity(
                    self._model, tuple(_cpu_first_layer_indices),
                )
                _target_dev = f"cuda:0" if _cf_device == "cuda" else _cf_device
                _decoder_layers = _find_decoder_layer_list(self._model)
                if _decoder_layers is not None:
                    for _li in _cpu_first_layer_indices:
                        if _li < len(_decoder_layers):
                            _decoder_layers[_li].to(device=_target_dev, dtype=_cf_dtype)
                _inner = getattr(self._model, "model", self._model)
                _lm_inner = getattr(_inner, "language_model", None) or _inner
                for _attr in ("embed_tokens", "norm", "rotary_emb"):
                    _mod = getattr(_lm_inner, _attr, None)
                    if _mod is not None and isinstance(_mod, torch.nn.Module):
                        _mod.to(device=_target_dev, dtype=_cf_dtype)
                if hasattr(self._model, "lm_head") and self._model.lm_head is not None:
                    self._model.lm_head.to(device=_target_dev, dtype=_cf_dtype)
                import gc as _gc2
                _gc2.collect()
                if _cf_device == "cuda":
                    torch.cuda.empty_cache()
                logging.info(
                    "cpu_first_shard_strip: model=%s kept=%d identity_replaced=%d device=%s",
                    self.model_name, len(_cpu_first_layer_indices), _id_replaced, _target_dev,
                )
            except Exception as _cf_exc:
                logging.warning(
                    "cpu_first_shard_strip_failed: model=%s err=%s",
                    self.model_name, _cf_exc,
                )
            self._cpu_first_strip_layers = None

        # Skip .to() when using selective device_map (accelerate manages placement)
        # or when quantized weights are already on device.
        _dm = load_kwargs.get("device_map")
        _has_selective_map = False
        if isinstance(_dm, dict):
            _dm_values = set(_dm.values())
            _has_selective_map = bool(_dm_values & {"disk", "meta"})
        elif isinstance(_dm, str) and _dm == "auto":
            # device_map="auto" means accelerate placed everything; don't move
            _has_selective_map = True
        # _cpu_first_strip_layers is set to None after stripping completes.
        # If it was originally set (not "sentinel"), the strip ran.
        _cpu_first_stripped = getattr(self, "_cpu_first_strip_layers", "sentinel") is None
        if (
            not quantized_weights_loaded
            and not _has_selective_map
            and not _multimodal_stripped
            and not _mps_stripped
            and not _cpu_first_stripped
        ):
            self._model.to(self._device)
        self._model.eval()

        # Phase 6 workaround (T4 + bf16 grouped-conv1d + small seq_len):
        # Qwen 3.5's ``linear_attn`` layers contain a depthwise
        # ``nn.Conv1d(C, C, kernel_size=4, groups=C, padding=3)`` module.
        # cuDNN on T4 has no kernel for this exact shape in bfloat16 when
        # ``seq_len < 4`` — it raises ``RuntimeError: GET was unable to
        # find an engine to execute this computation``. This only matters
        # for KV-aware decoding (where every decode step ships a single
        # token through the layer); prefill with a long sequence works
        # fine because seq_len >= 4.
        #
        # Proper fix: install ``causal-conv1d`` on the studio — it
        # provides the fast kernel Qwen 3.5 prefers and sidesteps cuDNN
        # entirely. Until that lands on every deployment target, we
        # wrap every grouped ``nn.Conv1d`` whose input channels match
        # its groups (= depthwise) with a fallback that retries the
        # offending call with cuDNN disabled. The outer wrapper is a
        # no-op when cuDNN succeeds, so prefill keeps the fast path.
        try:
            self._patch_depthwise_conv1d_t4_fallback()
        except Exception as _patch_exc:
            logging.debug("conv1d_t4_patch_skipped: %s", _patch_exc)

        decoder_arch = self._detect_decoder_architecture(self._model)
        self._decoder_family = str(decoder_arch.family)
        self._embed_tokens = decoder_arch.embed_tokens
        self._position_embeddings = decoder_arch.position_embeddings
        self._final_norm = decoder_arch.final_norm
        self._rotary_emb = decoder_arch.rotary_emb
        self._blocks = list(decoder_arch.layers)
        # Gemma 4 extras — only populated when family == "gemma4"
        self._layer_types: tuple[str, ...] = tuple(decoder_arch.layer_types or ())
        self._per_layer_embed = decoder_arch.per_layer_embed
        self._per_layer_proj = decoder_arch.per_layer_proj
        self._per_layer_norm = decoder_arch.per_layer_norm
        self._hidden_size_per_layer = int(decoder_arch.hidden_size_per_layer or 0)
        self._gemma4_text_model = decoder_arch.text_model
        # Re-entrant scratch for a single forward pass — the per-layer
        # inputs tensor derived from the prompt's input_ids. Populated
        # by ``_compute_gemma4_per_layer_inputs`` and consumed by the
        # ``family == "gemma4"`` branch in ``_run_layers``. Cleared at
        # the top of each ``_forward_impl`` call.
        self._pending_per_layer_inputs: Any | None = None
        self.total_layers = len(self._blocks)
        self.layer_indices = self._resolve_layer_indices(
            total_layers=self.total_layers,
            shard_index=max(0, int(config.shard_index)),
            total_shards=max(1, int(config.total_shards)),
            explicit_indices=tuple(config.runtime_layer_indices),
        )
        self._selected_layers = [self._blocks[idx] for idx in self.layer_indices]

        # Free unused layers to reduce memory on constrained devices (1GB nanodes).
        # PyTorch loads the entire model, then we select only our shard's layers.
        # Deleting unused blocks frees ~40 MB/layer for Qwen2.5-0.5B.
        if len(self.layer_indices) < self.total_layers:
            used_set = set(self.layer_indices)
            freed = 0
            for idx in range(self.total_layers):
                if idx not in used_set:
                    self._blocks[idx] = None
                    freed += 1
            if freed > 0:
                import gc as _gc
                _gc.collect()
                logging.info(
                    "pytorch_layer_cleanup: freed %d/%d unused layers",
                    freed, self.total_layers,
                )

            # Phase 2B fix (sharded KV-cache meta-tensor leak): the
            # ``self._blocks[idx] = None`` loop above only drops OUR local
            # reference — the real ``nn.ModuleList`` at
            # ``model.model.layers`` (or wherever ``_find_decoder_layer_list``
            # locates it) still holds the original meta-device module that
            # accelerate mapped to the ``"disk"`` device. Any code path that
            # iterates the full ModuleList — DynamicCache internals,
            # transformers' ``_update_causal_mask``, model-level
            # ``parameters()`` iteration — then dispatches an aten op on a
            # meta tensor and crashes with
            # ``GET was unable to find an engine to execute this computation``.
            # Replace every out-of-shard slot with ``nn.Identity()`` so the
            # ModuleList no longer contains meta tensors.
            if not _multimodal_stripped:
                try:
                    _id_replaced = _replace_offloaded_layers_with_identity(
                        self._model, tuple(self.layer_indices),
                    )
                    if _id_replaced > 0:
                        logging.info(
                            "pytorch_layer_identity_swap: %d offloaded layers replaced with nn.Identity",
                            _id_replaced,
                        )
                except Exception as _id_exc:
                    logging.warning(
                        "pytorch_layer_identity_swap_failed: %s — KV-cache paths may still crash on meta tensors",
                        _id_exc,
                    )

        # Remove accelerate dispatch hooks from kept modules.
        # When using selective device_map with "disk" offloading, accelerate
        # wraps EVERY module (including real on-device layers) with dispatch
        # hooks that check on each forward() whether to load from disk.
        # This adds 10-50ms overhead per layer per forward call — crippling
        # for GPU inference where the actual compute is <5ms/layer.
        # Removing hooks from kept modules restores native PyTorch speed.
        if _has_selective_map:
            try:
                from accelerate.hooks import remove_hook_from_module
                _hooks_removed = 0
                for module in self._model.modules():
                    if hasattr(module, "_hf_hook"):
                        remove_hook_from_module(module)
                        _hooks_removed += 1
                if _hooks_removed > 0:
                    logging.info(
                        "accelerate_hooks_removed: %d modules unhooked for native speed",
                        _hooks_removed,
                    )
            except Exception as exc:
                logging.debug("accelerate_hook_removal_failed: %s", exc)

        # Hidden size — for multimodal wrapper configs (Gemma 4,
        # Qwen 3.5 VL) the top-level ``config.hidden_size`` is ``None``
        # because the decoder dim lives inside ``text_config``. Probe the
        # nested config before falling back to the 768 default, otherwise
        # stage-to-stage hidden payloads fail with
        # ``invalid_hidden_payload:hidden_size`` when the peer defaults
        # to 768 while the real tensor is 2560 (E4B-it).
        _cfg = self._model.config
        _text_cfg = getattr(_cfg, "text_config", None)
        self._hidden_size = int(
            getattr(_cfg, "hidden_size", 0) or 0
            or (getattr(_text_cfg, "hidden_size", 0) or 0 if _text_cfg is not None else 0)
            or getattr(_cfg, "n_embd", 0) or 0
            or getattr(_cfg, "d_model", 0) or 0
            or 768
        )
        lm_head = getattr(self._model, "lm_head", None)
        if lm_head is None and hasattr(self._model, "get_output_embeddings"):
            lm_head = self._model.get_output_embeddings()
        if lm_head is None:
            raise RuntimeError("unsupported_model_architecture: missing_lm_head")
        self._lm_head = lm_head
        if bool(config.runtime_tensor_autoencoder_enabled):
            latent_dim = max(1, int(config.runtime_tensor_autoencoder_latent_dim))
            self._compressor = PyTorchActivationCompressor(
                torch_module=self._torch,
                hidden_size=self._hidden_size,
                latent_dim=latent_dim,
                device=self._device,
                dtype=self._dtype,
                seed=11 + int(config.shard_index),
            )
        noise_variance = max(0.0, float(config.runtime_privacy_noise_variance))
        if noise_variance > 0.0:
            self._privacy_noise = PyTorchDifferentialPrivacyNoise(
                torch_module=self._torch,
                variance=noise_variance,
                seed=1009 + int(config.shard_index),
            )

        param_count = int(sum(p.numel() for p in self._model.parameters()))
        bytes_per_param = 2 if self._dtype == torch.float16 else 4
        quantization_factor = 1.0
        if self.quantization_bits == 8:
            quantization_factor = 0.5
        elif self.quantization_bits == 4:
            quantization_factor = 0.25
        estimated_memory_mb = max(
            256,
            int((param_count * bytes_per_param * quantization_factor) / (1024 * 1024)),
        )

        # Phase D: measure actual throughput instead of using static estimates.
        # Falls back to a conservative static estimate if benchmarking fails.
        estimated_tps = 0.0
        try:
            from peer.throughput_bench import benchmark_and_cache
            _bench = benchmark_and_cache(
                model=self._model,
                tokenizer=self._tokenizer,
                model_id=self.model_name,
                device=target,
                layer_count=len(self.layer_indices),
                quantization=self.quantization_mode,
            )
            estimated_tps = _bench.compute_tps
        except Exception as exc:
            logging.debug("throughput_bench_skipped: %s", exc)
        if estimated_tps <= 0:
            # Static fallback (legacy formula)
            base_tps = 12.0 if target == "cpu" else 42.0
            if self.quantization_bits == 8:
                base_tps *= 1.2
            elif self.quantization_bits == 4:
                base_tps *= 1.45
            layer_fraction = (len(self.layer_indices) / float(max(1, self.total_layers))) if self.total_layers else 1.0
            layer_penalty = max(0.35, layer_fraction)
            estimated_tps = round(base_tps / layer_penalty, 3)
            logging.info("throughput_static_fallback: tps=%.1f", estimated_tps)

        self._runtime_profile = RuntimeProfile(
            backend=backend,
            target=target,
            quantization_mode=self.quantization_mode,
            quantization_bits=self.quantization_bits,
            gpu_available=gpu_available,
            estimated_tokens_per_sec=estimated_tps,
            estimated_memory_mb=estimated_memory_mb,
            runtime_model_id=self.model_name,
            layer_start=(self.layer_indices[0] if self.layer_indices else 0),
            layer_end=((self.layer_indices[-1] + 1) if self.layer_indices else 0),
            total_layers=self.total_layers,
        )

        # ── KV cache compaction (Phases 1-4) ────────────────────────────────
        self._compaction_config: Any = None
        self._compaction_budgets: Any = None
        # Auto-mode (6.1): resolve three-position toggle.
        # Priority: runtime_kv_compaction_mode > runtime_kv_compaction_enabled
        _kv_mode = str(getattr(config, "runtime_kv_compaction_mode", "off") or "off").strip().lower()
        # Backward compat: runtime_kv_compaction_enabled=True promotes "off" → "on"
        if bool(getattr(config, "runtime_kv_compaction_enabled", False)) and _kv_mode == "off":
            _kv_mode = "on"
        if _kv_mode in {"auto", "on"}:
            try:
                from peer.kv_compaction import CompactionConfig, _load_head_budgets
                budget_path: str = str(getattr(config, "runtime_kv_compaction_head_budget_path", "") or "").strip()
                self._compaction_config = CompactionConfig(
                    enabled=True,
                    method=str(getattr(config, "runtime_kv_compaction_method", "hak") or "hak"),
                    target_ratio=max(0.01, min(1.0, float(getattr(config, "runtime_kv_compaction_ratio", 0.10)))),
                    beta_enabled=bool(getattr(config, "runtime_kv_compaction_beta", False)),
                    head_budget_path=budget_path or None,
                    online_enabled=bool(getattr(config, "runtime_kv_compaction_online", False)),
                    online_max_tokens=max(4, int(getattr(config, "runtime_kv_compaction_online_max_tokens", 512))),
                    mode=_kv_mode,
                    auto_threshold=max(1, int(getattr(config, "runtime_kv_compaction_auto_threshold", 512))),
                )
                if budget_path:
                    self._compaction_budgets = _load_head_budgets(budget_path)
                # Phase 2: patch model attention layers for β injection
                if self._compaction_config.beta_enabled:
                    from peer.kv_compaction._beta_inject import (
                        patch_model_for_beta_injection,
                        detect_model_family,
                    )
                    family = detect_model_family(self.model_name)
                    patch_model_for_beta_injection(self._model, family)
                    logging.info(
                        "kv_compaction: beta injection enabled for model=%s family=%s",
                        self.model_name, family,
                    )
                logging.info(
                    "kv_compaction: enabled mode=%s method=%s ratio=%.2f beta=%s online=%s",
                    _kv_mode,
                    self._compaction_config.method,
                    self._compaction_config.target_ratio,
                    self._compaction_config.beta_enabled,
                    self._compaction_config.online_enabled,
                )
            except Exception as exc:
                logging.warning("kv_compaction_init_failed: %s — compaction disabled", exc)
                self._compaction_config = None

        # ── Radix prefix cache (Phase H) ─────────────────────────────────────
        self._radix_cache: Any = None
        if bool(getattr(config, "runtime_kv_radix_cache_enabled", False)):
            try:
                from peer.kv_compaction import RadixKVCache
                self._radix_cache = RadixKVCache(
                    max_entries=max(1, int(getattr(config, "runtime_kv_radix_cache_max_entries", 128))),
                    min_prefix_len=max(1, int(getattr(config, "runtime_kv_radix_cache_min_prefix_len", 16))),
                )
                logging.info(
                    "kv_radix_cache: enabled max_entries=%d min_prefix_len=%d",
                    self._radix_cache._max_entries, self._radix_cache._min_prefix_len,
                )
            except Exception as exc:
                logging.warning("kv_radix_cache_init_failed: %s — radix cache disabled", exc)
                self._radix_cache = None

        # ── Model warmup (Phase W) ────────────────────────────────────────────
        if bool(getattr(config, "runtime_warmup_on_start", False)):
            self._warmup()

    def _warmup(self) -> None:
        """JIT-compile GPU kernels by running one tiny forward pass at startup.

        On Apple MPS the first real inference triggers Metal shader compilation
        (~30 s).  Calling this during ``__init__`` moves that cost to peer
        startup so the first user request sees normal latency (~1-3 s).

        The warmup uses a single EOS token — minimal memory pressure, exercises
        the full embed → transformer layers → lm_head path.  Errors are caught
        and logged as warnings rather than failing startup.
        """
        import time as _time

        _t0 = _time.perf_counter()
        try:
            logging.info(
                "runtime_warmup: starting model=%s device=%s",
                self.model_name, str(self._device),
            )
            with self._torch.no_grad():
                eos_id = int(getattr(self._tokenizer, "eos_token_id", None) or 0)
                input_ids = self._torch.tensor(
                    [[eos_id]],
                    dtype=self._torch.long,
                    device=self._device,
                )
                position_ids = self._build_position_ids(seq_len=1, past_len=0)
                hidden = self._embed_from_input_ids(input_ids, position_ids)
                hidden, _ = self._run_layers(hidden)
                if self._lm_head is not None:
                    _ = self._hidden_to_next_token_payload(hidden)
        except Exception as exc:
            logging.warning(
                "runtime_warmup_failed: %s — first inference will be slow", exc,
            )
        else:
            elapsed = _time.perf_counter() - _t0
            logging.info("runtime_warmup: complete in %.1f s", elapsed)

    def compaction_stats(self) -> "dict[str, int | float]":
        """Return a snapshot of KV compaction SLO counters."""
        with self._compact_lock:
            return {
                "compact_calls":         self._compact_calls,
                "compact_tokens_before": self._compact_tokens_before,
                "compact_tokens_after":  self._compact_tokens_after,
                "compact_tokens_saved":  max(0, self._compact_tokens_before - self._compact_tokens_after),
                "compact_latency_s":     round(self._compact_latency_s, 6),
                "kv_cache_hits":         self._compact_kv_cache_hits,
                "kv_cache_misses":       self._compact_kv_cache_misses,
                "auto_skip_count":       self._auto_skip_count,
                "auto_trigger_count":    self._auto_trigger_count,
            }

    def _vram_usage_pct(self) -> float:
        """Return current GPU VRAM utilisation as a fraction [0.0, 1.0].

        CUDA: ``torch.cuda.memory_allocated() / total_mem``.
        CPU/other: returns ``0.0`` (compaction never triggers on VRAM pressure).
        """
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0)
                total = torch.cuda.get_device_properties(0).total_mem
                if total > 0:
                    return allocated / total
        except Exception:
            pass
        return 0.0

    def _patch_depthwise_conv1d_t4_fallback(self) -> None:
        """Wrap every depthwise ``nn.Conv1d`` in the model with a fallback
        that disables cuDNN when the default dispatcher raises on small
        inputs.

        Context: Qwen 3.5 ``linear_attn`` layers contain a depthwise
        ``Conv1d(C, C, kernel_size=4, groups=C, padding=3)`` that's fine
        during prefill (``seq_len >= 4``) but hits a missing cuDNN kernel
        on T4 during single-token decode (``seq_len == 1``) with bf16.
        The exact error is
        ``RuntimeError: GET was unable to find an engine to execute this
        computation``.

        The wrapper replaces ``conv._conv_forward`` with a version that
        first tries the native dispatcher; if that raises, it retries
        inside a ``torch.backends.cudnn.flags(enabled=False)`` context.
        The retry hits a slower but supported ``aten::_slow_conv_forward``
        path which works for every shape.

        Impact is surgical: only the tiny fraction of decode steps where
        cuDNN fails pay the slow-path cost. Prefill and any seq_len >= 4
        call keeps the fast path.
        """
        import torch
        from torch import nn

        _patched = 0
        for mod in self._model.modules():
            if not isinstance(mod, nn.Conv1d):
                continue
            # Only patch grouped depthwise convs (this is the Qwen 3.5
            # linear_attn shape we care about; leaves regular convs alone).
            if mod.groups != mod.in_channels:
                continue

            _original = mod._conv_forward

            def _fallback_conv_forward(
                input: Any, weight: Any, bias: Any,
                _orig=_original,
            ):
                try:
                    return _orig(input, weight, bias)
                except RuntimeError as exc:
                    msg = str(exc)
                    if "GET was unable to find an engine" not in msg:
                        raise
                    # Retry with cuDNN disabled — torch's native
                    # implementation has kernels for every shape.
                    logging.debug(
                        "conv1d_t4_cudnn_fallback: shape=%s dtype=%s",
                        tuple(input.shape), input.dtype,
                    )
                    with torch.backends.cudnn.flags(enabled=False):
                        return _orig(input, weight, bias)

            mod._conv_forward = _fallback_conv_forward  # type: ignore[assignment]
            _patched += 1

        if _patched > 0:
            logging.info(
                "conv1d_t4_fallback_patched: %d depthwise Conv1d modules wrapped",
                _patched,
            )

    @staticmethod
    def _detect_decoder_architecture(model: Any) -> _DecoderArchitecture:
        # Multimodal models (Gemma 4, Qwen 3.5 VL) wrap the text decoder
        # inside a `language_model` attribute.  The wrapping depth varies:
        #   Gemma 4: model.model.language_model.layers
        #   Others:  model.language_model.model.layers
        # Try both paths.
        _language_model = getattr(model, "language_model", None)
        if _language_model is None:
            # Gemma 4: model.model.language_model
            _model_attr = getattr(model, "model", None)
            if _model_attr is not None:
                _language_model = getattr(_model_attr, "language_model", None)
        if _language_model is not None:
            # language_model may have layers directly, or inside .model
            _inner = _language_model
            if not hasattr(_inner, "layers"):
                _inner = getattr(_language_model, "model", None)
            if _inner is not None and hasattr(_inner, "layers") and hasattr(_inner, "embed_tokens"):
                arch_name = type(model).__name__.lower()
                _cfg = getattr(model, "config", None)
                model_type = str(getattr(_cfg, "model_type", "")).strip().lower()
                _family = "llama"
                _layer_types: tuple[str, ...] = ()
                _per_layer_embed: Any | None = None
                _per_layer_proj: Any | None = None
                _per_layer_norm: Any | None = None
                _hidden_size_per_layer: int = 0
                _text_model: Any | None = None
                if "gemma4" in arch_name or model_type == "gemma4":
                    # Gemma 4 needs layer-type-aware rotary + per-layer
                    # input plumbing. Capture the text-config layer_types
                    # list and the small per-layer modules so the
                    # ``family="gemma4"`` branch in ``_run_layers`` can
                    # dispatch correctly. These modules live on the text
                    # decoder (`model.model.language_model`) which
                    # ``_strip_multimodal_components`` moved onto the
                    # target device alongside the real layers.
                    _family = "gemma4"
                    _text_model = _inner  # Gemma4TextModel — owns get_per_layer_inputs / project_per_layer_inputs
                    # Gemma 4 config nests the text decoder config under
                    # ``text_config`` (when loaded via the multimodal
                    # ``Gemma4ForConditionalGeneration`` wrapper).
                    _text_cfg = getattr(_cfg, "text_config", None) or _cfg
                    _raw_types = getattr(_text_cfg, "layer_types", None)
                    if _raw_types is not None:
                        try:
                            _layer_types = tuple(str(t) for t in _raw_types)
                        except TypeError:
                            _layer_types = ()
                    _per_layer_embed = getattr(_inner, "embed_tokens_per_layer", None)
                    _per_layer_proj = getattr(_inner, "per_layer_model_projection", None)
                    _per_layer_norm = getattr(_inner, "per_layer_projection_norm", None)
                    _hidden_size_per_layer = int(
                        getattr(_text_cfg, "hidden_size_per_layer_input", 0) or 0
                    )
                elif "gemma" in arch_name or "gemma" in model_type:
                    _family = "llama"
                elif "qwen" in arch_name or "qwen" in model_type:
                    _family = "qwen_llama"
                return _DecoderArchitecture(
                    family=_family,
                    layers=tuple(list(_inner.layers)),
                    embed_tokens=_inner.embed_tokens,
                    position_embeddings=None,
                    final_norm=getattr(_inner, "norm", None),
                    rotary_emb=getattr(_inner, "rotary_emb", None),
                    layer_types=_layer_types,
                    per_layer_embed=_per_layer_embed,
                    per_layer_proj=_per_layer_proj,
                    per_layer_norm=_per_layer_norm,
                    hidden_size_per_layer=_hidden_size_per_layer,
                    text_model=_text_model,
                )

        transformer = getattr(model, "transformer", None)
        if transformer is not None and hasattr(transformer, "h") and hasattr(transformer, "wte"):
            return _DecoderArchitecture(
                family="gpt",
                layers=tuple(list(transformer.h)),
                embed_tokens=transformer.wte,
                position_embeddings=getattr(transformer, "wpe", None),
                final_norm=getattr(transformer, "ln_f", None),
                rotary_emb=None,
            )

        llama_model = getattr(model, "model", None)
        arch_name = type(model).__name__.lower()
        model_type = str(getattr(getattr(model, "config", None), "model_type", "")).strip().lower()
        if (
            llama_model is not None
            and hasattr(llama_model, "layers")
            and hasattr(llama_model, "embed_tokens")
            and ("qwen" in arch_name or "qwen" in model_type)
        ):
            return _DecoderArchitecture(
                family="qwen_llama",
                layers=tuple(list(llama_model.layers)),
                embed_tokens=llama_model.embed_tokens,
                position_embeddings=None,
                final_norm=getattr(llama_model, "norm", None),
                rotary_emb=getattr(llama_model, "rotary_emb", None),
            )

        if (
            llama_model is not None
            and hasattr(llama_model, "layers")
            and hasattr(llama_model, "embed_tokens")
            and ("gemma" in arch_name or "gemma" in model_type)
        ):
            return _DecoderArchitecture(
                family="llama",  # Gemma uses the same layer structure as LLaMA
                layers=tuple(list(llama_model.layers)),
                embed_tokens=llama_model.embed_tokens,
                position_embeddings=None,
                final_norm=getattr(llama_model, "norm", None),
                rotary_emb=getattr(llama_model, "rotary_emb", None),
            )

        if llama_model is not None and hasattr(llama_model, "layers") and hasattr(llama_model, "embed_tokens"):
            return _DecoderArchitecture(
                family="llama",
                layers=tuple(list(llama_model.layers)),
                embed_tokens=llama_model.embed_tokens,
                position_embeddings=None,
                final_norm=getattr(llama_model, "norm", None),
                rotary_emb=getattr(llama_model, "rotary_emb", None),
            )

        raise RuntimeError(
            "unsupported_model_architecture: expected GPT-style transformer.h "
            "or LLaMA-style model.layers blocks"
        )

    @staticmethod
    def _resolve_layer_indices(
        *,
        total_layers: int,
        shard_index: int,
        total_shards: int,
        explicit_indices: tuple[int, ...],
    ) -> tuple[int, ...]:
        if total_layers <= 0:
            return ()

        sanitized = sorted({idx for idx in explicit_indices if 0 <= int(idx) < total_layers})
        if sanitized:
            return tuple(int(idx) for idx in sanitized)

        shard = min(max(0, int(shard_index)), max(0, total_shards - 1))
        base = total_layers // total_shards
        remainder = total_layers % total_shards
        start = shard * base + min(shard, remainder)
        count = base + (1 if shard < remainder else 0)
        end = min(total_layers, start + max(0, count))
        if start >= end:
            return ()
        return tuple(range(start, end))

    def runtime_profile(self) -> dict[str, Any]:
        return self._runtime_profile.to_dict()

    def reshard(self, new_layer_start: int, new_layer_end: int, total_layers: int) -> bool:
        """Reshard the PyTorchRuntime to cover [new_layer_start, new_layer_end).

        Recomputes ``_selected_layers`` from the model's existing blocks.
        Returns ``True`` on success, ``False`` if the requested range is
        invalid.
        """
        new_start = max(0, int(new_layer_start))
        new_end = min(int(total_layers), int(new_layer_end))
        if new_start >= new_end or new_end > len(self._blocks):
            logging.warning(
                "pytorch_reshard_invalid: requested [%d, %d) but model has %d blocks",
                new_start, new_end, len(self._blocks),
            )
            return False

        new_indices = tuple(range(new_start, new_end))
        self.layer_indices = new_indices
        self._selected_layers = [self._blocks[idx] for idx in self.layer_indices]
        self.total_layers = int(total_layers)

        # Rebuild the runtime profile.
        self._runtime_profile = RuntimeProfile(
            backend=self._runtime_profile.backend,
            target=self._runtime_profile.target,
            quantization_mode=self._runtime_profile.quantization_mode,
            quantization_bits=self._runtime_profile.quantization_bits,
            gpu_available=self._runtime_profile.gpu_available,
            estimated_tokens_per_sec=self._runtime_profile.estimated_tokens_per_sec,
            estimated_memory_mb=self._runtime_profile.estimated_memory_mb,
            runtime_model_id=self._runtime_profile.runtime_model_id,
            layer_start=new_start,
            layer_end=new_end,
            total_layers=int(total_layers),
        )

        # Clear KV caches since they're invalidated by shard change.
        self._kv_cache.clear()

        logging.info(
            "pytorch_reshard: [%d, %d) total=%d layers_selected=%d",
            new_start, new_end, total_layers, len(self._selected_layers),
        )
        return True

    @property
    def kv_cache_size(self) -> int:
        return len(self._kv_cache)

    @property
    def kv_cache_sessions(self) -> tuple[str, ...]:
        return tuple(self._kv_cache.keys())

    @property
    def compression_enabled(self) -> bool:
        return self._compressor is not None

    @property
    def compression_latent_dim(self) -> int:
        if self._compressor is None:
            return int(self._hidden_size)
        return int(self._compressor.latent_dim)

    @property
    def compression_encoded_payloads(self) -> int:
        if self._compressor is None:
            return 0
        return int(self._compressor.stats().encoded_payloads)

    @property
    def compression_decoded_payloads(self) -> int:
        if self._compressor is None:
            return 0
        return int(self._compressor.stats().decoded_payloads)

    @property
    def privacy_noise_variance(self) -> float:
        if self._privacy_noise is None:
            return 0.0
        return float(self._privacy_noise.variance)

    @property
    def privacy_noise_payloads(self) -> int:
        if self._privacy_noise is None:
            return 0
        return int(self._privacy_noise.stats().applied_payloads)

    @property
    def privacy_noise_observed_variance_ema(self) -> float:
        if self._privacy_noise is None:
            return 0.0
        return float(self._privacy_noise.stats().observed_variance_ema)

    @property
    def privacy_noise_last_observed_variance(self) -> float:
        return float(self._last_noise_observed_variance)

    @property
    def privacy_noise_last_observed_std(self) -> float:
        return float(self._last_noise_observed_std)

    @property
    def privacy_noise_last_audit_tag(self) -> str:
        return str(self._last_noise_audit_tag)

    @property
    def privacy_noise_last_payload_index(self) -> int:
        return int(self._last_noise_payload_index)

    @property
    def privacy_noise_last_applied(self) -> bool:
        return bool(self._last_noise_applied)

    def _kv_cache_get(self, session_id: str | None) -> dict[str, Any] | None:
        key = str(session_id or "").strip()
        if not key:
            return None
        cached = self._kv_cache.get(key)
        if cached is None:
            with self._compact_lock:
                self._compact_kv_cache_misses += 1
            return None
        self._kv_cache.move_to_end(key, last=True)
        with self._compact_lock:
            self._compact_kv_cache_hits += 1
        return cached

    def _kv_cache_set(
        self,
        session_id: str | None,
        past_key_values: Any,
        q_ref_per_layer: Any = None,   # Option A: real W_q-projected Q from AttentionQueryCapture
    ) -> None:
        key = str(session_id or "").strip()
        if not key or past_key_values is None:
            return
        # ── Phase 1-4 + Pass 6: optionally compact before storing ────────────
        if self._compaction_config is not None:
            # ── Pass 6: VRAM-aware auto mode ─────────────────────────────
            _mode = getattr(self._compaction_config, "mode", "on")
            if _mode == "auto":
                _vram_pct = self._vram_usage_pct()
                _seq_len = self._past_sequence_length(past_key_values)
                _threshold = getattr(self._compaction_config, "auto_threshold", 512)
                if _vram_pct < 0.75 and _seq_len <= _threshold:
                    # VRAM comfortable + short sequence → skip compaction
                    with self._compact_lock:
                        self._auto_skip_count += 1
                    self._kv_cache.pop(str(session_id or "").strip(), None)
                    self._kv_cache[str(session_id or "").strip()] = {"past_key_values": past_key_values}
                    while len(self._kv_cache) > self._kv_cache_max_entries:
                        self._kv_cache.popitem(last=False)
                    return
                with self._compact_lock:
                    self._auto_trigger_count += 1
            try:
                import time as _time
                from peer.kv_compaction import compact_past_key_values
                _tokens_before = self._past_sequence_length(past_key_values)
                _t0 = _time.perf_counter()
                compacted = compact_past_key_values(
                    past_key_values,
                    self._compaction_config,
                    budgets_data=self._compaction_budgets,
                    q_ref_per_layer=q_ref_per_layer,   # Option A
                )
                _latency = _time.perf_counter() - _t0
                if compacted is not None:
                    _tokens_after = self._past_sequence_length(compacted)
                    past_key_values = compacted
                else:
                    _tokens_after = _tokens_before
                with self._compact_lock:
                    self._compact_calls += 1
                    self._compact_tokens_before += _tokens_before
                    self._compact_tokens_after += _tokens_after
                    self._compact_latency_s += _latency
            except Exception as exc:
                logging.debug("kv_compaction_store_failed session=%s: %s", key, exc)
        # ─────────────────────────────────────────────────────────────────────
        self._kv_cache.pop(key, None)
        self._kv_cache[key] = {"past_key_values": past_key_values}
        while len(self._kv_cache) > self._kv_cache_max_entries:
            self._kv_cache.popitem(last=False)

    def _past_sequence_length(self, past_key_values: Any) -> int:
        if not past_key_values:
            return 0
        get_seq_length = getattr(past_key_values, "get_seq_length", None)
        if callable(get_seq_length):
            try:
                return int(get_seq_length() or 0)
            except Exception:
                pass
        if isinstance(past_key_values, (tuple, list)) and past_key_values:
            first_item = past_key_values[0]
            first_get_seq = getattr(first_item, "get_seq_length", None)
            if callable(first_get_seq):
                try:
                    return int(first_get_seq() or 0)
                except Exception:
                    pass
        try:
            first = past_key_values[0]
            if not first or len(first) < 1:
                return 0
            keys = first[0]
            if keys is None:
                return 0
            return int(keys.shape[-2])
        except Exception:
            return 0

    def _build_position_ids(self, *, seq_len: int, past_len: int) -> Any:
        return self._torch.arange(
            int(past_len),
            int(past_len) + max(0, int(seq_len)),
            device=self._device,
            dtype=self._torch.long,
        ).unsqueeze(0)

    def _embed_from_input_ids(self, input_ids: Any, position_ids: Any) -> Any:
        hidden = self._embed_tokens(input_ids)
        if self._decoder_family == "gpt" and self._position_embeddings is not None:
            hidden = hidden + self._position_embeddings(position_ids)
        return hidden

    def _apply_final_norm(self, hidden: Any) -> Any:
        if self._final_norm is None:
            return hidden
        return self._final_norm(hidden)

    def _prompt_to_hidden(self, prompt: str, max_tokens: int):
        token_budget = max(4, min(self.max_context_tokens, max(1, int(max_tokens)) * 4))
        encoded = self._tokenizer(
            prompt or "",
            return_tensors="pt",
            truncation=True,
            max_length=token_budget,
            add_special_tokens=True,
        )
        input_ids = encoded.get("input_ids")
        if input_ids is None or int(input_ids.numel()) == 0:
            _, eos_token = _tokenizer_eos_ids(self._tokenizer)
            input_ids = self._torch.tensor([[eos_token]], dtype=self._torch.long)

        input_ids = input_ids.to(self._device)
        position_ids = self._build_position_ids(seq_len=int(input_ids.shape[1]), past_len=0)
        hidden = self._embed_from_input_ids(input_ids, position_ids)
        return hidden

    def _activation_to_hidden(
        self,
        activation: list[float],
        packed_bytes: bytes | None = None,
    ):
        # Zero-copy path: ``activation_packed`` bytes from the previous
        # stage. Matches the MLX runtime's symmetric decoder so an MLX
        # peer can ship straight into a PyTorch peer (and vice versa).
        # Wire format: ``openhydra_network.encode_activation`` output —
        # header-less fp32 buffer prefixed by a ``[seq_len, hidden_size]``
        # RustTensor shape.
        if packed_bytes is not None and len(packed_bytes) >= 8:
            try:
                import openhydra_network
                rust_tensor = openhydra_network.decode_activation(packed_bytes)
                decoded = self._torch.from_dlpack(rust_tensor)
                # RustTensor shape is [1, seq_len, hidden_size].
                seq_len = int(rust_tensor.shape[1])
                hidden_size = int(rust_tensor.shape[2])
                if hidden_size != int(self._hidden_size):
                    raise RuntimeError(
                        f"invalid_hidden_payload:hidden_size "
                        f"(wire={hidden_size}, expected={self._hidden_size})"
                    )
                hidden = decoded.reshape(1, seq_len, hidden_size).to(self._device)
                if self._dtype != self._torch.float32:
                    hidden = hidden.to(dtype=self._dtype)
                return hidden
            except Exception as _dlpack_exc:
                # Rust wheel missing or malformed bytes — fall through to
                # the legacy list-float path so the peer stays useful on
                # minimal installs.
                logging.debug(
                    "pytorch_packed_fallback: %s — using list path", _dlpack_exc,
                )
        values = [float(item) for item in activation]
        if len(values) < 3:
            raise RuntimeError("invalid_hidden_payload:too_short")
        seq_len = int(round(values[0]))
        hidden_size = int(round(values[1]))
        compressed_payload = seq_len < 0
        if compressed_payload:
            seq_len = abs(seq_len)
        if seq_len <= 0 or hidden_size <= 0:
            raise RuntimeError("invalid_hidden_payload:shape")
        if seq_len > (self.max_context_tokens * 8):
            raise RuntimeError("invalid_hidden_payload:seq_len")
        if compressed_payload:
            if len(values) < 4:
                raise RuntimeError("invalid_hidden_payload:compressed_header")
            original_hidden_size = int(round(values[2]))
            if original_hidden_size != int(self._hidden_size):
                raise RuntimeError("invalid_hidden_payload:compressed_hidden_size")
            payload = values[3:]
            required = seq_len * hidden_size
            if len(payload) != required:
                raise RuntimeError("invalid_hidden_payload:length")
            latent = self._torch.tensor(payload, dtype=self._torch.float32, device=self._device).view(1, seq_len, hidden_size)
            if self._dtype != self._torch.float32:
                latent = latent.to(dtype=self._dtype)
            if self._compressor is None:
                raise RuntimeError("compression_decoder_unavailable")
            hidden = self._compressor.decode(latent)
            return hidden

        if hidden_size != int(self._hidden_size):
            raise RuntimeError("invalid_hidden_payload:hidden_size")
        payload = values[2:]
        required = seq_len * hidden_size
        if len(payload) != required:
            raise RuntimeError("invalid_hidden_payload:length")
        hidden = self._torch.tensor(payload, dtype=self._torch.float32, device=self._device).view(1, seq_len, hidden_size)
        if self._dtype != self._torch.float32:
            hidden = hidden.to(dtype=self._dtype)
        return hidden

    def _activation_to_input_ids(self, activation: list[float], prompt: str, max_tokens: int):
        if activation:
            ids = [max(0, int(round(float(item)))) for item in activation]
            if not ids:
                _, eos_token = _tokenizer_eos_ids(self._tokenizer)
                ids = [eos_token]
            input_ids = self._torch.tensor([ids], dtype=self._torch.long, device=self._device)
            return input_ids

        token_budget = max(4, min(self.max_context_tokens, max(1, int(max_tokens)) * 4))
        encoded = self._tokenizer(
            prompt or "",
            return_tensors="pt",
            truncation=True,
            max_length=token_budget,
            add_special_tokens=True,
        )
        input_ids = encoded.get("input_ids")
        if input_ids is None or int(input_ids.numel()) == 0:
            _, eos_token = _tokenizer_eos_ids(self._tokenizer)
            input_ids = self._torch.tensor([[eos_token]], dtype=self._torch.long)
        return input_ids.to(self._device)

    def _compute_gemma4_per_layer_inputs(
        self,
        prompt_token_ids: list[int] | tuple[int, ...] | None,
    ) -> Any | None:
        """Recompute the Gemma 4 per-layer inputs tensor from prompt token IDs.

        Gemma 4 decoder blocks require a ``per_layer_input`` tensor of shape
        ``[B, S, num_hidden_layers, hidden_size_per_layer_input]`` that is
        derived from ``embed_tokens_per_layer(input_ids) → reshape → norm``
        and then combined with a projection of ``inputs_embeds``. The layer
        multiplies its slice (``per_layer_inputs[:, :, layer_idx, :]``) into
        the hidden state — **it cannot be None or zero**, otherwise every
        output collapses to garbage.

        For sharded inference, every stage needs the same per_layer_inputs
        tensor. We make the coordinator ship the original prompt token IDs
        in the new ``prompt_token_ids`` field of ``ForwardRequest`` and each
        peer recomputes the tensor locally — the small per-layer modules
        (``embed_tokens_per_layer`` / ``per_layer_model_projection`` /
        ``per_layer_projection_norm``) were all moved onto the target device
        by ``_strip_multimodal_components`` at load time, so this is cheap.

        Returns ``None`` for non-Gemma-4 models or when the necessary
        modules aren't available. Callers in the ``family == "gemma4"``
        branch of ``_run_layers`` must treat ``None`` as an error.
        """
        if self._decoder_family != "gemma4":
            return None
        if not prompt_token_ids:
            return None
        text_model = self._gemma4_text_model
        if text_model is None:
            return None
        if self._per_layer_embed is None or self._per_layer_proj is None:
            return None
        try:
            ids_list = [int(t) for t in prompt_token_ids]
            if not ids_list:
                return None
            input_ids = self._torch.tensor(
                [ids_list],
                dtype=self._torch.long,
                device=self._device,
            )
            # Embed lookup + Gemma4-specific scaling lives inside the
            # ``Gemma4TextScaledWordEmbedding`` module, so calling the
            # embedding directly gives already-scaled outputs.
            inputs_embeds = self._embed_tokens(input_ids)
            per_layer_raw = text_model.get_per_layer_inputs(input_ids, inputs_embeds)
            per_layer_inputs = text_model.project_per_layer_inputs(
                inputs_embeds, per_layer_raw,
            )
            return per_layer_inputs
        except Exception as exc:
            logging.warning(
                "gemma4_per_layer_inputs_failed: %s — sharded Gemma 4 forward will error",
                exc,
            )
            return None

    def _run_layers_gemma4(
        self,
        hidden,
        *,
        past_key_values: Any | None = None,
        use_cache: bool = False,
        position_ids: Any | None = None,
        attention_mask: Any | None = None,
        cache_position: Any | None = None,
    ) -> tuple[Any, Any | None]:
        """Gemma 4 sharded decode branch — layer-type-aware rotary +
        per-layer-input multiplication + full/sliding causal masks.

        The standard llama / qwen_llama branch assumes a single rotary
        (cos, sin) pair and treats the causal mask as either None or a
        plain boolean mask. Gemma 4 alternates ``full_attention`` and
        ``sliding_attention`` layers with DIFFERENT rotary frequencies
        per type AND different causal mask shapes, and every block
        multiplies its output by ``per_layer_input`` — a tensor derived
        from the prompt's token IDs and indexed per-layer.

        This branch mirrors ``Gemma4TextModel.forward`` layer-by-layer
        but restricts the loop to ``self._selected_layers`` (our shard's
        slice). It creates a local ``DynamicCache`` so the KV-shared
        layers (config.num_kv_shared_layers > 0) can read the stored
        full-length K/V from the non-shared layers earlier in the same
        shard. This works cleanly when the shard range covers BOTH
        "full-length KV storage" layers (i.e. ``store_full_length_kv``
        is True, typically the last full_attention and the last
        sliding_attention before ``first_kv_shared_layer_idx``) AND any
        downstream shared layers. When the shard boundary splits these
        — e.g. peer A runs layers 0..K where K < first_kv_shared_layer_idx
        and peer B runs the rest — the cache state from peer A would
        need to be serialized onto the wire and reconstructed on peer B
        before its shared layers could run. That cross-peer cache
        propagation is a separate Phase 5 effort; for now this branch
        only works correctly when one shard holds the complete KV
        chain. See ``plans/sharded-inference-fixes.md::Phase 4`` for the
        full story.

        KV-cache reuse across decode steps is intentionally
        unimplemented for the first cut: Gemma 4 always re-prefills per
        decode step until Phase 5 adds KV-aware pipelining.
        """
        text_model = self._gemma4_text_model
        if text_model is None:
            raise RuntimeError(
                "gemma4_runtime_without_text_model_reference: "
                "_detect_decoder_architecture did not capture Gemma4TextModel"
            )

        per_layer_inputs = self._pending_per_layer_inputs
        if per_layer_inputs is None:
            raise RuntimeError(
                "gemma4_missing_per_layer_inputs: coordinator must pass "
                "prompt_token_ids in ForwardRequest so each stage can "
                "recompute per_layer_inputs locally"
            )

        # ── KV sharing + KV reuse plumbing ────────────────────────────
        # Gemma 4 models with ``num_kv_shared_layers > 0`` (true for
        # E4B-it which shares layers 24..41 with layers 22..23) REQUIRE
        # a ``past_key_values`` object so that:
        #   (1) non-shared attention layers with ``store_full_length_kv``
        #       True can deposit their full-length K/V into
        #       ``past_key_values.shared_layers``, and
        #   (2) KV-shared attention layers can later read that K/V back.
        # Passing ``past_key_values=None`` is the "works-for-test-models"
        # default but produces garbage logits on E4B-it because layers
        # 24..41 read random uninitialised state.
        #
        # Phase 7B: when the caller PROVIDES a ``past_key_values`` object
        # (i.e. the peer's outer ``_forward_impl`` is doing KV-aware
        # decoding and persists the cache across calls), we reuse it
        # instead of creating a fresh one. The caller is responsible for
        # storing the returned cache object via ``_kv_cache_set`` so the
        # next decode step picks it up. Without this Gemma 4 always
        # re-prefills per token (no decode speedup); with this each
        # decode step ships a single token and the existing
        # ``cache.shared_layers`` state stays valid for KV-shared layers.
        local_cache: Any | None = None
        if past_key_values is not None and not isinstance(past_key_values, (tuple, list)):
            # Caller passed an existing DynamicCache — reuse it.
            local_cache = past_key_values
        else:
            try:
                from transformers.cache_utils import DynamicCache
                local_cache = DynamicCache(config=text_model.config)
            except Exception as _cache_exc:
                logging.debug("gemma4_dynamic_cache_init_failed: %s", _cache_exc)
                local_cache = None

        _num_kv_shared_layers = int(
            getattr(text_model.config, "num_kv_shared_layers", 0) or 0
        )
        if _num_kv_shared_layers > 0:
            _first_shared = int(text_model.config.num_hidden_layers) - _num_kv_shared_layers
            _shard_indices = set(int(i) for i in self.layer_indices)
            _need_shared_kv = any(idx >= _first_shared for idx in _shard_indices)
            _have_storing_kv = any(idx < _first_shared for idx in _shard_indices)
            if _need_shared_kv and not _have_storing_kv:
                raise RuntimeError(
                    "gemma4_shard_split_breaks_kv_sharing: this shard "
                    f"contains KV-shared layers (>= {_first_shared}) but "
                    "no layers that store the shared K/V. Cross-peer KV "
                    "sharing is not yet implemented. Run Gemma 4 unsharded "
                    "via ops/bench/gemma4_direct_bench.py or keep the "
                    "shard boundary below layer "
                    f"{_first_shared} inclusive on every peer."
                )

        # Restrict the layer types we need for THIS shard's slice — avoids
        # building masks / rotary for unused types on intermediate peers.
        selected_types: set[str] = set()
        for idx in self.layer_indices:
            if idx < len(self._layer_types):
                selected_types.add(self._layer_types[idx])
            else:
                selected_types.add("full_attention")

        # Layer-type-aware rotary embeddings (per unique type)
        position_embeddings: dict[str, Any] = {}
        if self._rotary_emb is not None and position_ids is not None:
            for lt in selected_types:
                try:
                    position_embeddings[lt] = self._rotary_emb(hidden, position_ids, lt)
                except TypeError:
                    # Some rotary impls reject the layer_type kwarg — fall
                    # back to the 2-arg call. Should not happen for Gemma 4
                    # but stays defensive.
                    position_embeddings[lt] = self._rotary_emb(hidden, position_ids)

        # Full + sliding causal masks — transformers exports helper
        # functions keyed off the config. Lazy-import so the module is
        # only loaded when we actually run a Gemma 4 shard.
        causal_mask_mapping: dict[str, Any] = {}
        try:
            from transformers.models.gemma4.modeling_gemma4 import (
                create_causal_mask,
                create_sliding_window_causal_mask,
            )
        except Exception:
            create_causal_mask = None  # type: ignore[assignment]
            create_sliding_window_causal_mask = None  # type: ignore[assignment]

        # Prefer the caller's past_key_values (KV-aware decode mode,
        # future) over our local cache (prefill mode, current). The
        # local cache is the most common case today.
        _effective_pkv = past_key_values
        if _effective_pkv is None and local_cache is not None:
            _effective_pkv = local_cache
        if isinstance(_effective_pkv, (tuple, list)):
            _effective_pkv = None  # legacy tuple caches aren't compatible

        mask_kwargs = {
            "config": text_model.config,
            "inputs_embeds": hidden,
            "attention_mask": attention_mask,
            "past_key_values": _effective_pkv,
            "position_ids": position_ids,
        }
        if create_causal_mask is not None and "full_attention" in selected_types:
            try:
                causal_mask_mapping["full_attention"] = create_causal_mask(**mask_kwargs)
            except Exception as exc:
                logging.debug("gemma4_full_mask_fallback: %s", exc)
        if create_sliding_window_causal_mask is not None and "sliding_attention" in selected_types:
            try:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
            except Exception as exc:
                logging.debug("gemma4_sliding_mask_fallback: %s", exc)

        # Gemma 4 KV sharing: layers with ``store_full_length_kv=True``
        # write their K/V into this dict keyed by layer_idx. Later
        # KV-shared layers read from it via ``kv_shared_layer_index``.
        # The model's own forward() creates this dict at line ~1615 of
        # modeling_gemma4.py. Without it: TypeError 'NoneType' does not
        # support item assignment.
        _shared_kv_states: dict[int, Any] = {}

        output = hidden
        for idx, block in zip(self.layer_indices, self._selected_layers):
            layer_type = (
                self._layer_types[idx] if idx < len(self._layer_types) else "full_attention"
            )
            pli_slice = None
            if (
                per_layer_inputs is not None
                and hasattr(per_layer_inputs, "shape")
                and len(per_layer_inputs.shape) >= 3
                and idx < per_layer_inputs.shape[2]
            ):
                pli_slice = per_layer_inputs[:, :, idx, :]

            block_kwargs: dict[str, Any] = {
                "position_embeddings": position_embeddings.get(layer_type),
                "attention_mask": causal_mask_mapping.get(layer_type, attention_mask),
                "position_ids": position_ids,
                "past_key_values": _effective_pkv,
                "shared_kv_states": _shared_kv_states,
            }
            # Gemma 4 attention always needs ``use_cache=True`` when a
            # cache is present because the KV-sharing branch reads from
            # ``past_key_values.shared_layers``; without it the cache
            # doesn't get populated.
            if _effective_pkv is not None:
                block_kwargs["use_cache"] = True
            elif use_cache:
                block_kwargs["use_cache"] = True
            if cache_position is not None:
                block_kwargs["cache_position"] = cache_position

            block_out = block(output, pli_slice, **block_kwargs)
            if isinstance(block_out, tuple):
                output = block_out[0]
            else:
                output = block_out

        # Phase 7B: return the (possibly mutated) cache so the caller's
        # ``_forward_impl`` can persist it via ``_kv_cache_set`` for the
        # next decode step. Returning ``None`` here would force a fresh
        # cache on every call → no decode speedup vs Phase 1 stateless.
        return output, _effective_pkv

    def _run_layers(
        self,
        hidden,
        *,
        past_key_values: Any | None = None,
        use_cache: bool = False,
        position_ids: Any | None = None,
        attention_mask: Any | None = None,
        cache_position: Any | None = None,
    ):
        # Dispatch to the Gemma 4 adapter when applicable. Keeps the
        # standard llama / qwen_llama branch untouched.
        if self._decoder_family == "gemma4":
            return self._run_layers_gemma4(
                hidden,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_ids=position_ids,
                attention_mask=attention_mask,
                cache_position=cache_position,
            )

        output = hidden
        present_key_values: list[Any] = []
        shared_cache = past_key_values if (past_key_values is not None and not isinstance(past_key_values, (tuple, list))) else None

        # Phase 6 fix: when the caller requests ``use_cache=True`` but
        # didn't provide a ``past_key_values`` object, create a fresh
        # ``DynamicCache`` here. Without this, each block runs with
        # ``past_key_values=None`` and transformers' caching logic never
        # sees a container to write K/V into — the function ends up
        # returning a tuple of ``None``s, the caller persists garbage,
        # and the next decode step silently produces wrong logits (not
        # an exception, a plain numerical miss). Native ``Qwen3_5TextModel
        # .forward`` does exactly this same "create cache on first call"
        # step; we're just replicating it at the layer-loop level so our
        # sharded ``_run_layers`` can drive the same state machine.
        if use_cache and shared_cache is None:
            try:
                _cfg = getattr(self._model, "config", None)
                # Qwen3.5 hybrid Mamba architecture needs Qwen3_5DynamicCache
                # which carries conv_states + ssm_states alongside KV cache.
                # Plain DynamicCache crashes with "has no attribute 'conv_states'".
                _used_qwen3_5_cache = False
                if _cfg is not None:
                    _model_type = str(getattr(_cfg, "model_type", "")).lower()
                    if "qwen3_5" in _model_type:
                        try:
                            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache
                            shared_cache = Qwen3_5DynamicCache(config=_cfg)
                            _used_qwen3_5_cache = True
                        except Exception:
                            pass
                if not _used_qwen3_5_cache:
                    from transformers.cache_utils import DynamicCache
                    if _cfg is not None:
                        try:
                            shared_cache = DynamicCache(config=_cfg)
                        except TypeError:
                            shared_cache = DynamicCache()
                    else:
                        shared_cache = DynamicCache()
                    # Non-Qwen3.5: patch has_previous_state for Mamba compat.
                    # Must be callable — Qwen3.5 model code calls cache.has_previous_state()
                    # and cache.has_previous_state(layer_idx).
                    if not hasattr(shared_cache, "has_previous_state"):
                        _has_state = (past_key_values is not None)
                        shared_cache.has_previous_state = lambda *_args, **_kw: _has_state
            except Exception as _cache_exc:
                logging.debug(
                    "run_layers_dynamic_cache_init_failed: %s — falling back to cache-less forward",
                    _cache_exc,
                )
                shared_cache = None

        for idx, block in enumerate(self._selected_layers):
            layer_past = None
            if shared_cache is not None:
                layer_past = shared_cache
            elif past_key_values is not None:
                try:
                    if idx < len(past_key_values):
                        layer_past = past_key_values[idx]
                except Exception:
                    layer_past = past_key_values
            block_kwargs: dict[str, Any] = {
                "past_key_values": layer_past,
                "use_cache": use_cache,
            }
            if attention_mask is not None:
                block_kwargs["attention_mask"] = attention_mask
            if cache_position is not None:
                block_kwargs["cache_position"] = cache_position
            if position_ids is not None:
                block_kwargs["position_ids"] = position_ids
            if self._decoder_family in {"llama", "qwen_llama"} and self._rotary_emb is not None and position_ids is not None:
                # Compute rotary position embeddings (cos, sin) for each layer.
                # This is REQUIRED for correct output — without it, transformer
                # blocks receive position_embeddings=None and produce garbage.
                block_kwargs["position_embeddings"] = self._rotary_emb(output, position_ids)

            block_out = block(output, **block_kwargs)
            if isinstance(block_out, tuple):
                output = block_out[0]
                if use_cache:
                    present = block_out[1] if len(block_out) > 1 else None
                    if shared_cache is None:
                        present_key_values.append(present if present is not None else layer_past)
            else:
                output = block_out
                if use_cache:
                    if shared_cache is None:
                        present_key_values.append(layer_past)
        if not use_cache:
            return output, None
        if shared_cache is not None:
            # Mark that we now have state — next decode step's linear_attn
            # blocks will read from the cache instead of initializing fresh.
            # Qwen3_5DynamicCache has has_previous_state as a property (auto-detects
            # from conv_states) — skip the set for property-based caches.
            try:
                if hasattr(shared_cache, "has_previous_state") and not isinstance(
                    type(shared_cache).__dict__.get("has_previous_state"), property
                ):
                    # Must stay callable — model code calls cache.has_previous_state()
                    shared_cache.has_previous_state = lambda *_args, **_kw: True
            except (AttributeError, TypeError):
                pass  # Property or read-only — auto-detects state
            return output, shared_cache
        return output, tuple(present_key_values)

    def _hidden_to_payload(
        self,
        hidden,
        *,
        request_id: str | None,
        stage_index: int,
    ) -> list[float]:
        tensor = hidden
        self._last_noise_applied = False
        self._last_noise_observed_variance = 0.0
        self._last_noise_observed_std = 0.0
        self._last_noise_audit_tag = ""
        self._last_noise_payload_index = int(self.privacy_noise_payloads)
        if self._privacy_noise is not None:
            tensor = self._privacy_noise.apply(
                tensor,
                peer_id=str(self.config.runtime_peer_id or ""),
                request_id=str(request_id or ""),
                stage_index=int(stage_index),
                shared_secret_seed=str(self.config.runtime_privacy_audit_seed or ""),
            )
            noise_stats = self._privacy_noise.stats()
            self._last_noise_applied = True
            self._last_noise_observed_variance = float(noise_stats.last_observed_variance)
            self._last_noise_observed_std = float(noise_stats.last_observed_std)
            self._last_noise_audit_tag = str(noise_stats.last_audit_tag)
            self._last_noise_payload_index = int(noise_stats.applied_payloads)
        if self._compressor is not None:
            tensor = self._compressor.encode(tensor)
        tensor = tensor.detach().to(device="cpu", dtype=self._torch.float32)
        seq_len = int(tensor.shape[1])
        hidden_size = int(tensor.shape[2])
        flattened = tensor.contiguous().view(-1).tolist()
        if self._compressor is not None:
            out: list[float] = [float(-seq_len), float(hidden_size), float(self._hidden_size)]
        else:
            out = [float(seq_len), float(hidden_size)]
        out.extend(float(item) for item in flattened)
        return out

    def _hidden_to_packed_bytes(
        self,
        hidden,
        *,
        request_id: str | None,
        stage_index: int,
    ) -> bytes:
        """Convert hidden state tensor to packed activation bytes via Rust zero-copy.

        Same as ``_hidden_to_payload`` but returns binary-packed bytes using
        ``openhydra_network.encode_activation()`` (single memcpy, ~12x faster
        than ``.tolist()`` + ``struct.pack``).  Falls back to the Python path
        if the Rust module is unavailable.
        """
        tensor = hidden
        self._last_noise_applied = False
        self._last_noise_observed_variance = 0.0
        self._last_noise_observed_std = 0.0
        self._last_noise_audit_tag = ""
        self._last_noise_payload_index = int(self.privacy_noise_payloads)
        if self._privacy_noise is not None:
            tensor = self._privacy_noise.apply(
                tensor,
                peer_id=str(self.config.runtime_peer_id or ""),
                request_id=str(request_id or ""),
                stage_index=int(stage_index),
                shared_secret_seed=str(self.config.runtime_privacy_audit_seed or ""),
            )
            noise_stats = self._privacy_noise.stats()
            self._last_noise_applied = True
            self._last_noise_observed_variance = float(noise_stats.last_observed_variance)
            self._last_noise_observed_std = float(noise_stats.last_observed_std)
            self._last_noise_audit_tag = str(noise_stats.last_audit_tag)
            self._last_noise_payload_index = int(noise_stats.applied_payloads)
        if self._compressor is not None:
            tensor = self._compressor.encode(tensor)
        tensor = tensor.detach().to(device="cpu", dtype=self._torch.float32).contiguous()
        try:
            import openhydra_network
            return openhydra_network.encode_activation(tensor)
        except (ImportError, Exception):
            # Fallback to Python path
            import struct
            payload = self._hidden_to_payload(
                hidden, request_id=request_id, stage_index=stage_index,
            )
            return struct.pack(f'<{len(payload)}f', *payload)

    def apply_final_head(
        self,
        hidden_state: Any,
        *,
        packed_bytes: bytes | None = None,
        decode_do_sample: bool | None = None,
        decode_temperature: float | None = None,
        decode_top_p: float | None = None,
        decode_top_k: int | None = None,
        decode_seed: int | None = None,
    ) -> int:
        """Coordinator-side final head + sampler (Path A).

        Symmetric with ``MLXRuntime.apply_final_head``. Accepts either a
        flat ``[seq_f, hidden_f, v0, ...]`` list from ``_hidden_to_payload``
        or zero-copy ``packed_bytes`` produced by the Rust encoder, decodes
        to a PyTorch tensor shaped ``[1, seq, hidden]`` via
        ``_activation_to_hidden``, applies ``final_norm`` + ``lm_head``,
        then samples exactly as ``_logits_to_next_token_payload`` does on
        the peer side.

        Only callable on a PyTorch shard that owns the last layer
        (``self._lm_head is not None``). Raises ``RuntimeError`` otherwise.
        """
        if self._lm_head is None:
            raise RuntimeError(
                "apply_final_head: this PyTorch shard does not own the "
                "last layer; lm_head weights are on a different peer"
            )
        hidden = self._activation_to_hidden(
            hidden_state, packed_bytes=packed_bytes,
        )
        normed = self._apply_final_norm(hidden)
        logits = self._lm_head(normed)
        tokens = self._logits_to_next_token_payload(
            logits,
            token_count=1,
            decode_do_sample=bool(decode_do_sample),
            decode_temperature=max(1e-5, float(decode_temperature or 1.0)),
            decode_top_p=max(0.0, min(1.0, float(decode_top_p or 1.0))),
            decode_top_k=max(0, int(decode_top_k or 0)),
            decode_seed=(
                max(1, int(decode_seed))
                if (decode_seed is not None and int(decode_seed) > 0)
                else None
            ),
        )
        return int(round(float(tokens[0]))) if tokens else 0

    def _hidden_to_next_token_payload(
        self,
        hidden,
        token_count: int = 1,
        *,
        decode_do_sample: bool = False,
        decode_temperature: float = 1.0,
        decode_top_p: float = 1.0,
        decode_top_k: int = 0,
        decode_seed: int | None = None,
    ) -> list[float]:
        normed = self._apply_final_norm(hidden)
        logits = self._lm_head(normed)
        return self._logits_to_next_token_payload(
            logits,
            token_count=token_count,
            decode_do_sample=decode_do_sample,
            decode_temperature=decode_temperature,
            decode_top_p=decode_top_p,
            decode_top_k=decode_top_k,
            decode_seed=decode_seed,
        )

    def _logits_to_next_token_payload(
        self,
        logits,
        token_count: int = 1,
        *,
        decode_do_sample: bool = False,
        decode_temperature: float = 1.0,
        decode_top_p: float = 1.0,
        decode_top_k: int = 0,
        decode_seed: int | None = None,
    ) -> list[float]:
        count = max(1, min(int(token_count), int(logits.shape[1])))
        token_logits = logits[:, -count:, :]

        # Preserve deterministic greedy verification when validating multiple draft tokens.
        should_sample = bool(decode_do_sample and count == 1)
        if should_sample:
            work_logits = token_logits[:, -1, :]
            temperature = max(1e-5, float(decode_temperature))
            work_logits = work_logits / temperature
            top_k = max(0, int(decode_top_k))
            if top_k > 0:
                k = min(top_k, int(work_logits.shape[-1]))
                topk_vals, topk_idx = self._torch.topk(work_logits, k, dim=-1)
                filtered = self._torch.full_like(work_logits, float("-inf"))
                filtered.scatter_(1, topk_idx, topk_vals)
                work_logits = filtered
            top_p = max(0.0, min(1.0, float(decode_top_p)))
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = self._torch.sort(work_logits, descending=True, dim=-1)
                sorted_probs = self._torch.softmax(sorted_logits, dim=-1)
                cumulative_probs = self._torch.cumsum(sorted_probs, dim=-1)
                sorted_mask = cumulative_probs > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False
                sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
                unsorted_logits = self._torch.full_like(work_logits, float("-inf"))
                unsorted_logits.scatter_(1, sorted_indices, sorted_logits)
                work_logits = unsorted_logits
            probs = self._torch.softmax(work_logits, dim=-1)
            generator = None
            if decode_seed is not None:
                generator = self._torch.Generator(device=work_logits.device)
                generator.manual_seed(max(1, int(decode_seed)))
            sampled = self._torch.multinomial(probs, num_samples=1, generator=generator)
            next_tokens = sampled.squeeze(0).tolist()
        else:
            next_tokens = self._torch.argmax(token_logits, dim=-1).squeeze(0).tolist()
        if isinstance(next_tokens, int):
            next_tokens = [next_tokens]
        return [float(max(0, int(token))) for token in list(next_tokens)]

    def _forward_impl(
        self,
        prompt: str,
        activation: list[float],
        max_tokens: int,
        stage_index: int,
        total_stages: int,
        kv_session_id: str | None = None,
        kv_store_activation: bool = False,
        kv_use_cached_activation: bool = False,
        request_id: str | None = None,
        decode_do_sample: bool | None = None,
        decode_temperature: float | None = None,
        decode_top_p: float | None = None,
        decode_top_k: int | None = None,
        decode_seed: int | None = None,
        prompt_token_ids: list[int] | tuple[int, ...] | None = None,
        packed_bytes: bytes | None = None,
        return_hidden_state: bool = False,
    ) -> list[float]:
        self.last_forward_thread_id = threading.get_ident()
        self.last_kv_cache_hit = False
        self._last_noise_applied = False
        self._last_noise_observed_variance = 0.0
        self._last_noise_observed_std = 0.0
        self._last_noise_audit_tag = ""
        self._last_noise_payload_index = int(self.privacy_noise_payloads)
        stage = max(0, int(stage_index))
        stages = max(1, int(total_stages))
        is_first = stage == 0
        is_last = stage == (stages - 1)
        session_id = str(kv_session_id or "").strip()
        cache_requested = bool(session_id and kv_use_cached_activation)
        cache_entry = self._kv_cache_get(session_id) if cache_requested else None
        cached_past = (cache_entry or {}).get("past_key_values")
        if cache_requested and cached_past is None:
            logging.warning(
                "kv_cache_miss: session=%s cache_keys=%s store=%s use=%s",
                session_id[:20] if session_id else "",
                list(self._kv_cache.keys())[:5],
                kv_store_activation,
                kv_use_cached_activation,
            )
            raise RuntimeError("kv_cache_miss")
        self.last_kv_cache_hit = bool(cache_requested and cached_past is not None)
        cache_enabled = bool(session_id and (kv_store_activation or cache_requested))
        next_past = None

        # Gemma 4 sharded support: recompute the per-layer input sidecar
        # from the prompt's token IDs. Both the first stage AND later
        # stages need this tensor — stage 0 can derive it from its own
        # activation payload, but every later stage needs the ORIGINAL
        # prompt_token_ids shipped in the ForwardRequest since its
        # activation is a hidden state, not token IDs. Clear the scratch
        # field on every call so a previous request's tensor doesn't
        # leak forward.
        self._pending_per_layer_inputs = None
        if self._decoder_family == "gemma4":
            _gemma4_ids: list[int] | tuple[int, ...] | None = prompt_token_ids
            if (not _gemma4_ids) and is_first and activation:
                # Stage 0 fallback: the coordinator packed the token IDs
                # into ``activation`` directly (legacy code path before
                # the proto field existed). Use them as the canonical
                # source so new coordinators and old ones both work.
                try:
                    _gemma4_ids = [
                        max(0, int(round(float(v)))) for v in activation
                    ]
                except (TypeError, ValueError):
                    _gemma4_ids = None
            if _gemma4_ids:
                self._pending_per_layer_inputs = self._compute_gemma4_per_layer_inputs(
                    _gemma4_ids,
                )

        with self._torch.no_grad():
            full_model_stage = bool(
                is_first
                and is_last
                and self.total_layers > 0
                and len(self.layer_indices) == self.total_layers
            )
            if full_model_stage:
                input_ids = self._activation_to_input_ids(activation, prompt, max_tokens)
                past_len = self._past_sequence_length(cached_past)
                seq_len = int(input_ids.shape[1])
                model_kwargs: dict[str, Any] = {
                    "input_ids": input_ids,
                    "use_cache": cache_enabled,
                }
                if cache_requested:
                    model_kwargs["past_key_values"] = cached_past
                total_len = max(1, int(past_len + seq_len))
                if self._decoder_family in {"llama", "qwen_llama"}:
                    model_kwargs["attention_mask"] = self._torch.ones(
                        (1, total_len),
                        dtype=self._torch.bool,
                        device=self._device,
                    )
                elif past_len > 0:
                    model_kwargs["attention_mask"] = self._torch.ones(
                        (1, total_len),
                        dtype=self._torch.long,
                        device=self._device,
                    )
                # ── Phase H: radix (longest-prefix) prefix cache lookup ───────
                # Before the forward pass we check whether any earlier request stored
                # a KV cache whose token sequence is a prefix of the current one.
                # If so, we trim input_ids to the unseen suffix and inject the prefix
                # KV cache as past_key_values, avoiding redundant computation.
                _full_token_seq: tuple[int, ...] = ()
                _radix_prefix_len: int = 0
                if self._radix_cache is not None and input_ids is not None:
                    try:
                        _full_token_seq = tuple(input_ids.squeeze(0).tolist())
                        _prefix_kv, _radix_prefix_len = self._radix_cache.lookup(_full_token_seq)
                        if _radix_prefix_len > 0 and _prefix_kv is not None:
                            from peer.kv_compaction import _slice_kv_prefix
                            _sliced = _slice_kv_prefix(_prefix_kv, _radix_prefix_len)
                            if _sliced is not None:
                                # Trim input_ids to the suffix not covered by the prefix cache
                                input_ids = input_ids[:, _radix_prefix_len:]
                                model_kwargs["input_ids"] = input_ids
                                model_kwargs["past_key_values"] = _sliced
                                # Update attention_mask to cover prefix + suffix tokens
                                new_total_len = max(1, int(_radix_prefix_len + input_ids.shape[1]))
                                if self._decoder_family in {"llama", "qwen_llama"}:
                                    model_kwargs["attention_mask"] = self._torch.ones(
                                        (1, new_total_len),
                                        dtype=self._torch.bool,
                                        device=self._device,
                                    )
                                elif new_total_len > input_ids.shape[1]:
                                    model_kwargs["attention_mask"] = self._torch.ones(
                                        (1, new_total_len),
                                        dtype=self._torch.long,
                                        device=self._device,
                                    )
                                logging.debug(
                                    "radix_cache_hit session=%s prefix_len=%d",
                                    session_id, _radix_prefix_len,
                                )
                            else:
                                _radix_prefix_len = 0
                    except Exception as exc:
                        logging.debug("radix_lookup_failed: %s", exc)
                        _radix_prefix_len = 0
                # ─────────────────────────────────────────────────────────────

                # Option A: capture real Q = W_q(hidden) for improved compaction quality.
                # When compaction is active, wrap the forward pass in AttentionQueryCapture
                # to capture hidden_states per layer.  compute_q_ref() is called after the
                # forward (hooks removed) and the resulting Q tensors replace the proxy-K
                # heuristic in compact_past_key_values, giving semantically meaningful
                # key-selection scores.  Falls back transparently to proxy Q on any error.
                _q_per_layer: Any = None
                _model_out_set = False
                if (
                    self._compaction_config is not None
                    and bool(session_id and kv_store_activation)
                ):
                    try:
                        from peer.kv_compaction import AttentionQueryCapture
                        _qc = AttentionQueryCapture(
                            self._model,
                            n_ref=self._compaction_config.n_ref_queries,
                        )
                        with _qc:
                            model_out = self._model(**model_kwargs)
                        _model_out_set = True
                        _q_per_layer = _qc.compute_q_ref()
                    except Exception as exc:
                        logging.debug("query_capture_failed: %s — proxy Q will be used", exc)
                        _q_per_layer = None
                        if not _model_out_set:
                            model_out = self._model(**model_kwargs)
                else:
                    model_out = self._model(**model_kwargs)
                logits = getattr(model_out, "logits", None)
                if logits is None and isinstance(model_out, tuple) and model_out:
                    logits = model_out[0]
                if logits is None:
                    raise RuntimeError("pytorch_forward_missing_logits")
                next_past = getattr(model_out, "past_key_values", None)
                if next_past is None and isinstance(model_out, tuple) and len(model_out) > 1:
                    next_past = model_out[1]
                if bool(session_id and kv_store_activation):
                    self._kv_cache_set(session_id, next_past, q_ref_per_layer=_q_per_layer)
                    # Phase H: insert full-sequence KV into radix cache
                    if self._radix_cache is not None and _full_token_seq and next_past is not None:
                        try:
                            self._radix_cache.insert(_full_token_seq, next_past)
                        except Exception as exc:
                            logging.debug("radix_insert_failed: %s", exc)
                output_count = 1
                if cache_requested and activation and int(max_tokens) > 1:
                    output_count = min(int(max_tokens), len(activation))
                output = self._logits_to_next_token_payload(
                    logits,
                    token_count=output_count,
                    decode_do_sample=bool(decode_do_sample),
                    decode_temperature=max(1e-5, float(decode_temperature or 1.0)),
                    decode_top_p=max(0.0, min(1.0, float(decode_top_p or 1.0))),
                    decode_top_k=max(0, int(decode_top_k or 0)),
                    decode_seed=(
                        max(1, int(decode_seed))
                        if (decode_seed is not None and int(decode_seed) > 0)
                        else None
                    ),
                )
            else:
                if is_first:
                    input_ids = self._activation_to_input_ids(activation, prompt, max_tokens)
                    past_len = self._past_sequence_length(cached_past)
                    seq_len = int(input_ids.shape[1])
                    position_ids = self._build_position_ids(seq_len=seq_len, past_len=past_len)
                    hidden = self._embed_from_input_ids(input_ids, position_ids)
                else:
                    if not activation and not packed_bytes:
                        raise RuntimeError("missing_hidden_payload")
                    hidden = self._activation_to_hidden(
                        activation, packed_bytes=packed_bytes,
                    )
                    seq_len = int(hidden.shape[1])
                    past_len = self._past_sequence_length(cached_past)
                    position_ids = self._build_position_ids(seq_len=seq_len, past_len=past_len)
                cache_position = position_ids.squeeze(0)
                attention_mask = None
                if self._decoder_family in {"llama", "qwen_llama", "gemma4"}:
                    # All three families need an explicit 2D bool mask
                    # so ``create_causal_mask`` / ``_update_causal_mask``
                    # can derive per-query causal masks downstream.
                    # Gemma 4 additionally passes this to
                    # ``create_sliding_window_causal_mask`` in the
                    # ``_run_layers_gemma4`` branch.
                    total_len = max(1, int(past_len + seq_len))
                    attention_mask = self._torch.ones((1, total_len), dtype=self._torch.bool, device=self._device)
                elif past_len > 0:
                    total_len = max(1, int(past_len + seq_len))
                    attention_mask = self._torch.ones((1, total_len), dtype=self._torch.long, device=self._device)
                hidden, next_past = self._run_layers(
                    hidden,
                    past_key_values=(cached_past if cache_requested else None),
                    use_cache=cache_enabled,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                )
                if bool(session_id and kv_store_activation):
                    logging.debug(
                        "kv_cache_store: session=%s next_past_type=%s is_none=%s",
                        session_id[:20] if session_id else "",
                        type(next_past).__name__,
                        next_past is None,
                    )
                    self._kv_cache_set(session_id, next_past)
                if is_last and not return_hidden_state:
                    output_count = 1
                    if cache_requested and activation and int(max_tokens) > 1:
                        output_count = min(int(max_tokens), len(activation))
                    output = self._hidden_to_next_token_payload(
                        hidden,
                        token_count=output_count,
                        decode_do_sample=bool(decode_do_sample),
                        decode_temperature=max(1e-5, float(decode_temperature or 1.0)),
                        decode_top_p=max(0.0, min(1.0, float(decode_top_p or 1.0))),
                        decode_top_k=max(0, int(decode_top_k or 0)),
                        decode_seed=(
                            max(1, int(decode_seed))
                            if (decode_seed is not None and int(decode_seed) > 0)
                            else None
                        ),
                    )
                else:
                    # Path A (client-terminated pipeline): when
                    # ``return_hidden_state=True`` is set on the last shard,
                    # fall through to ``_hidden_to_payload`` so the
                    # coordinator can apply ``final_norm`` + ``lm_head``
                    # and sample. Non-last shards always take this path.
                    output = self._hidden_to_payload(
                        hidden,
                        request_id=request_id,
                        stage_index=stage,
                    )
        if is_last and not return_hidden_state:
            return [float(max(0, int(round(item)))) for item in output]
        return _apply_quantization(output, self.quantization_bits)

    def forward(
        self,
        prompt: str,
        activation: list[float],
        max_tokens: int,
        stage_index: int = 0,
        total_stages: int = 1,
        kv_session_id: str | None = None,
        kv_store_activation: bool = False,
        kv_use_cached_activation: bool = False,
        request_id: str | None = None,
        decode_do_sample: bool | None = None,
        decode_temperature: float | None = None,
        decode_top_p: float | None = None,
        decode_top_k: int | None = None,
        decode_seed: int | None = None,
        prompt_token_ids: list[int] | tuple[int, ...] | None = None,
        packed_bytes: bytes | None = None,
        return_hidden_state: bool = False,
    ) -> list[float]:
        return self._forward_impl(
            prompt,
            activation,
            max_tokens,
            stage_index=stage_index,
            total_stages=total_stages,
            kv_session_id=kv_session_id,
            kv_store_activation=kv_store_activation,
            kv_use_cached_activation=kv_use_cached_activation,
            request_id=request_id,
            decode_do_sample=decode_do_sample,
            decode_temperature=decode_temperature,
            decode_top_p=decode_top_p,
            decode_top_k=decode_top_k,
            decode_seed=decode_seed,
            prompt_token_ids=prompt_token_ids,
            packed_bytes=packed_bytes,
            return_hidden_state=return_hidden_state,
        )

    # Symmetric with ``MLXRuntime._forward_sharded``. The dispatcher in
    # ``ModelShard.forward`` gates ``packed_bytes`` pass-through on
    # ``hasattr(self._runtime, "_forward_sharded")`` — without this
    # alias, a PyTorch peer at stage 1+ never receives the MLX peer's
    # zero-copy packed bytes, forcing the slow list[float] path even
    # when both sides could DLPack-decode.
    _forward_sharded = _forward_impl

    async def forward_async(
        self,
        prompt: str,
        activation: list[float],
        max_tokens: int,
        stage_index: int = 0,
        total_stages: int = 1,
        kv_session_id: str | None = None,
        kv_store_activation: bool = False,
        kv_use_cached_activation: bool = False,
        request_id: str | None = None,
        decode_do_sample: bool | None = None,
        decode_temperature: float | None = None,
        decode_top_p: float | None = None,
        decode_top_k: int | None = None,
        decode_seed: int | None = None,
        prompt_token_ids: list[int] | tuple[int, ...] | None = None,
        packed_bytes: bytes | None = None,
        return_hidden_state: bool = False,
    ) -> list[float]:
        loop = asyncio.get_running_loop()
        # ``run_in_executor`` doesn't take **kwargs — pass positional args only
        # and let ``_forward_impl`` receive ``prompt_token_ids`` / ``packed_bytes``
        # via its own parameters at the tail of the signature.
        return await loop.run_in_executor(
            self._executor,
            self._forward_impl,
            prompt,
            activation,
            max_tokens,
            stage_index,
            total_stages,
            kv_session_id,
            kv_store_activation,
            kv_use_cached_activation,
            request_id,
            decode_do_sample,
            decode_temperature,
            decode_top_p,
            decode_top_k,
            decode_seed,
            prompt_token_ids,
            packed_bytes,
            return_hidden_state,
        )

    def forward_batch(self, items: list[Any]) -> list[list[float]]:
        """True tensor-batched forward pass for PyTorchRuntime.

        Submits to the shard's single-thread executor to ensure thread safety,
        matching the isolation contract of forward_async().  The actual work is
        done by _forward_batch_impl() which physically concatenates tensors along
        the batch dimension for a single GPU kernel launch.
        """
        return self._executor.submit(self._forward_batch_impl, items).result()

    def _forward_batch_impl(self, items: list[Any]) -> list[list[float]]:
        """Run a batch of forward requests with true tensor concatenation.

        Three dispatch cases based on stage position:

        Case C — Full model or first shard (is_first):
            Tokenise all prompts → right-pad → cat → model(batched_ids) or
            embed+layers → slice logits/hidden per item.

        Case A — Intermediate shard (not first, not last):
            Decode hidden states → cat [B, seq, d] → _run_layers → slice
            batched_out[i:i+1, :seq_i, :] per item.

        Case B — Last shard (not first, is last):
            Same as Case A but feed hidden_i into _hidden_to_next_token_payload.
        """
        self.last_forward_thread_id = threading.get_ident()
        B = len(items)
        if B == 0:
            return []

        # All items in one batch share the same stage/pipeline config.
        stage = max(0, int(items[0].stage_index))
        stages = max(1, int(items[0].total_stages))
        is_first = stage == 0
        is_last = stage == (stages - 1)
        full_model_stage = bool(
            is_first
            and is_last
            and self.total_layers > 0
            and len(self.layer_indices) == self.total_layers
        )

        results: list[list[float]] = []

        with self._torch.no_grad():
            # ── Case C: Full model or first shard ─────────────────────────────
            if is_first:
                input_ids_list = [
                    self._activation_to_input_ids(item.activation, item.prompt, item.max_tokens)
                    for item in items
                ]
                orig_seq_lens = [int(t.shape[1]) for t in input_ids_list]
                max_seq = max(orig_seq_lens)

                if len(set(orig_seq_lens)) == 1:
                    batched_ids = self._torch.cat(input_ids_list, dim=0)  # [B, seq]
                else:
                    # Right-pad shorter sequences to max_seq with pad_token_id.
                    pad_id = int(getattr(self._tokenizer, "pad_token_id", None) or 0)
                    padded_ids: list[Any] = []
                    for t in input_ids_list:
                        if t.shape[1] < max_seq:
                            pad_col = self._torch.full(
                                (1, max_seq - t.shape[1]),
                                pad_id,
                                dtype=self._torch.long,
                                device=t.device,
                            )
                            t = self._torch.cat([t, pad_col], dim=1)
                        padded_ids.append(t)
                    batched_ids = self._torch.cat(padded_ids, dim=0)  # [B, max_seq]

                if full_model_stage:
                    # Full model: batched_ids → model → logits [B, seq, vocab] → slice per item.
                    model_out = self._model(input_ids=batched_ids, use_cache=False)
                    logits = getattr(model_out, "logits", None)
                    if logits is None and isinstance(model_out, tuple) and model_out:
                        logits = model_out[0]
                    if logits is None:
                        raise RuntimeError("pytorch_forward_batch_missing_logits")
                    for i, item in enumerate(items):
                        logits_i = logits[i : i + 1]  # [1, seq, vocab]
                        output = self._logits_to_next_token_payload(
                            logits_i,
                            token_count=1,
                            decode_do_sample=bool(item.decode_do_sample),
                            decode_temperature=max(1e-5, float(item.decode_temperature or 1.0)),
                            decode_top_p=max(0.0, min(1.0, float(item.decode_top_p or 1.0))),
                            decode_top_k=max(0, int(item.decode_top_k or 0)),
                            decode_seed=(
                                max(1, int(item.decode_seed))
                                if (item.decode_seed is not None and int(item.decode_seed) > 0)
                                else None
                            ),
                        )
                        results.append([float(max(0, int(round(t)))) for t in output])
                else:
                    # First shard: embed → cat → _run_layers → hidden payload per item.
                    seq_len = batched_ids.shape[1]
                    position_ids = self._build_position_ids(seq_len=seq_len, past_len=0)
                    if B > 1:
                        position_ids = position_ids.expand(B, -1)
                    hidden = self._embed_from_input_ids(batched_ids, position_ids)  # [B, seq, d]
                    batched_out, _ = self._run_layers(
                        hidden, position_ids=position_ids
                    )  # [B, seq, d]
                    for i, item in enumerate(items):
                        hidden_i = batched_out[i : i + 1, : orig_seq_lens[i], :]
                        payload = self._hidden_to_payload(
                            hidden_i,
                            request_id=item.request_id,
                            stage_index=stage,
                        )
                        results.append(_apply_quantization(payload, self.quantization_bits))

            else:
                # ── Cases A & B: Intermediate or last shard ───────────────────
                # Decode hidden states from activation payloads.
                tensors = [self._activation_to_hidden(item.activation) for item in items]
                seq_lens = [int(t.shape[1]) for t in tensors]
                max_seq = max(seq_lens)

                if len(set(seq_lens)) == 1:
                    batched = self._torch.cat(tensors, dim=0)  # [B, seq, d_model]
                else:
                    # Rare: different upstream seq lengths — zero-pad to max_seq.
                    padded_tensors: list[Any] = []
                    for t in tensors:
                        if t.shape[1] < max_seq:
                            pad_t = self._torch.zeros(
                                1,
                                max_seq - t.shape[1],
                                t.shape[2],
                                dtype=t.dtype,
                                device=t.device,
                            )
                            t = self._torch.cat([t, pad_t], dim=1)
                        padded_tensors.append(t)
                    batched = self._torch.cat(padded_tensors, dim=0)  # [B, max_seq, d_model]

                seq_len = batched.shape[1]
                position_ids = self._build_position_ids(seq_len=seq_len, past_len=0)
                if B > 1:
                    position_ids = position_ids.expand(B, -1)

                # ONE GPU kernel call for the entire batch.
                batched_out, _ = self._run_layers(
                    batched, position_ids=position_ids
                )  # [B, seq, d]

                for i, item in enumerate(items):
                    hidden_i = batched_out[i : i + 1, : seq_lens[i], :]  # [1, orig_seq, d]
                    if is_last:
                        output = self._hidden_to_next_token_payload(
                            hidden_i,
                            token_count=1,
                            decode_do_sample=bool(item.decode_do_sample),
                            decode_temperature=max(1e-5, float(item.decode_temperature or 1.0)),
                            decode_top_p=max(0.0, min(1.0, float(item.decode_top_p or 1.0))),
                            decode_top_k=max(0, int(item.decode_top_k or 0)),
                            decode_seed=(
                                max(1, int(item.decode_seed))
                                if (item.decode_seed is not None and int(item.decode_seed) > 0)
                                else None
                            ),
                        )
                        results.append([float(max(0, int(round(t)))) for t in output])
                    else:
                        payload = self._hidden_to_payload(
                            hidden_i,
                            request_id=item.request_id,
                            stage_index=stage,
                        )
                        results.append(_apply_quantization(payload, self.quantization_bits))

        return results


class ModelShard:
    """Runtime strategy facade for toy and optional PyTorch backends."""

    def __init__(self, config: ToyShardConfig):
        self.config = config
        requested_backend = str(config.runtime_backend or "toy_auto").strip().lower()
        if requested_backend.startswith("pytorch") or requested_backend == "torch":
            self._runtime = PyTorchRuntime(config)
        elif requested_backend == "mlx":
            from peer.mlx_runtime import MLXRuntime  # optional dep: mlx, mlx-lm
            self._runtime = MLXRuntime(config)
        else:
            self._runtime = ToyRuntime(config)

    def runtime_profile(self) -> dict[str, Any]:
        return dict(self._runtime.runtime_profile())

    def reshard(self, new_layer_start: int, new_layer_end: int, total_layers: int) -> bool:
        """Reshard the underlying runtime to cover [new_layer_start, new_layer_end).

        Delegates to the runtime's ``reshard()`` method.  MLX runtimes log a
        warning and return ``False`` (full-model only, cannot reshard).

        Returns ``True`` on success.
        """
        reshard_fn = getattr(self._runtime, "reshard", None)
        if callable(reshard_fn):
            return bool(reshard_fn(new_layer_start, new_layer_end, total_layers))
        logging.warning(
            "reshard_unsupported: runtime %s does not support resharding",
            type(self._runtime).__name__,
        )
        return False

    def encode_prompt(self, prompt: str, max_tokens: int) -> list[float]:
        encoder = getattr(self._runtime, "encode_prompt", None)
        if callable(encoder):
            return list(encoder(prompt, max_tokens))
        return []

    def forward(
        self,
        prompt: str,
        activation: list[float],
        max_tokens: int,
        stage_index: int = 0,
        total_stages: int = 1,
        kv_session_id: str | None = None,
        kv_store_activation: bool = False,
        kv_use_cached_activation: bool = False,
        request_id: str | None = None,
        decode_do_sample: bool | None = None,
        decode_temperature: float | None = None,
        decode_top_p: float | None = None,
        decode_top_k: int | None = None,
        decode_seed: int | None = None,
        prompt_token_ids: list[int] | tuple[int, ...] | None = None,
        packed_bytes: bytes | None = None,
        return_hidden_state: bool = False,
    ) -> list[float]:
        # Only forward prompt_token_ids to runtimes that accept it. ToyRuntime
        # and MLXRuntime ignore the Gemma 4 sidecar; PyTorchRuntime consumes
        # it in its ``family == "gemma4"`` branch.
        _kwargs: dict[str, Any] = {
            "stage_index": stage_index,
            "total_stages": total_stages,
            "kv_session_id": kv_session_id,
            "kv_store_activation": kv_store_activation,
            "kv_use_cached_activation": kv_use_cached_activation,
            "request_id": request_id,
            "decode_do_sample": decode_do_sample,
            "decode_temperature": decode_temperature,
            "decode_top_p": decode_top_p,
            "decode_top_k": decode_top_k,
            "decode_seed": decode_seed,
        }
        if prompt_token_ids is not None and isinstance(self._runtime, PyTorchRuntime):
            _kwargs["prompt_token_ids"] = prompt_token_ids
        if packed_bytes is not None and hasattr(self._runtime, '_forward_sharded'):
            _kwargs["packed_bytes"] = packed_bytes
        # Path A (client-terminated pipeline): thread the flag through only
        # to runtimes that understand it. Older runtimes (ToyRuntime) would
        # raise TypeError on unknown kwargs — gate by attribute check.
        if return_hidden_state and isinstance(
            self._runtime, (PyTorchRuntime,)
        ):
            _kwargs["return_hidden_state"] = True
        elif return_hidden_state and hasattr(self._runtime, "_forward_sharded"):
            # MLXRuntime accepts it via _forward_sharded; forward() also
            # threads it through (see mlx_runtime.py). ToyRuntime is
            # excluded because it would reject the kwarg.
            _kwargs["return_hidden_state"] = True
        return list(
            self._runtime.forward(prompt, activation, max_tokens, **_kwargs)
        )

    async def forward_async(
        self,
        prompt: str,
        activation: list[float],
        max_tokens: int,
        stage_index: int = 0,
        total_stages: int = 1,
        kv_session_id: str | None = None,
        kv_store_activation: bool = False,
        kv_use_cached_activation: bool = False,
        request_id: str | None = None,
        decode_do_sample: bool | None = None,
        decode_temperature: float | None = None,
        decode_top_p: float | None = None,
        decode_top_k: int | None = None,
        decode_seed: int | None = None,
        prompt_token_ids: list[int] | tuple[int, ...] | None = None,
        return_hidden_state: bool = False,
    ) -> list[float]:
        async_forward = getattr(self._runtime, "forward_async", None)
        _kwargs: dict[str, Any] = {
            "stage_index": stage_index,
            "total_stages": total_stages,
            "kv_session_id": kv_session_id,
            "kv_store_activation": kv_store_activation,
            "kv_use_cached_activation": kv_use_cached_activation,
            "request_id": request_id,
            "decode_do_sample": decode_do_sample,
            "decode_temperature": decode_temperature,
            "decode_top_p": decode_top_p,
            "decode_top_k": decode_top_k,
            "decode_seed": decode_seed,
        }
        if prompt_token_ids is not None and isinstance(self._runtime, PyTorchRuntime):
            _kwargs["prompt_token_ids"] = prompt_token_ids
        if return_hidden_state and isinstance(self._runtime, PyTorchRuntime):
            _kwargs["return_hidden_state"] = True
        if callable(async_forward):
            return list(
                await async_forward(prompt, activation, max_tokens, **_kwargs)
            )
        # asyncio.to_thread fallback for runtimes without a native async
        # forward (ToyRuntime). These runtimes ignore prompt_token_ids.
        _legacy_kwargs = {k: v for k, v in _kwargs.items() if k != "prompt_token_ids"}
        return list(
            await asyncio.to_thread(
                self._runtime.forward,
                prompt,
                activation,
                max_tokens,
                _legacy_kwargs["stage_index"],
                _legacy_kwargs["total_stages"],
                _legacy_kwargs["kv_session_id"],
                _legacy_kwargs["kv_store_activation"],
                _legacy_kwargs["kv_use_cached_activation"],
                _legacy_kwargs["request_id"],
                _legacy_kwargs["decode_do_sample"],
                _legacy_kwargs["decode_temperature"],
                _legacy_kwargs["decode_top_p"],
                _legacy_kwargs["decode_top_k"],
                _legacy_kwargs["decode_seed"],
            )
        )

    def forward_batch(self, items: list[Any]) -> list[list[float]]:
        """Dispatch a batch of items to the underlying runtime's forward_batch().

        Falls back to sequential forward() calls if the runtime predates batching.
        This method is the bridge between BatchingQueue._dispatch() and the
        runtime-specific tensor-concatenation implementations.
        """
        fb = getattr(self._runtime, "forward_batch", None)
        if callable(fb):
            return [list(r) for r in fb(items)]
        # Fallback for legacy runtimes that lack forward_batch().
        return [
            list(
                self._runtime.forward(
                    item.prompt,
                    item.activation,
                    item.max_tokens,
                    stage_index=item.stage_index,
                    total_stages=item.total_stages,
                    request_id=item.request_id,
                    decode_do_sample=item.decode_do_sample,
                    decode_temperature=item.decode_temperature,
                    decode_top_p=item.decode_top_p,
                    decode_top_k=item.decode_top_k,
                    decode_seed=item.decode_seed,
                )
            )
            for item in items
        ]

    @property
    def uses_pytorch_runtime(self) -> bool:
        backend = str(self.runtime_profile().get("backend", "")).strip().lower()
        return backend.startswith("pytorch")

    @property
    def last_forward_thread_id(self) -> int | None:
        thread_id = getattr(self._runtime, "last_forward_thread_id", None)
        if thread_id is None:
            return None
        return int(thread_id)

    @property
    def last_kv_cache_hit(self) -> bool:
        return bool(getattr(self._runtime, "last_kv_cache_hit", False))

    def compaction_stats(self) -> "dict[str, int | float]":
        """Return compaction SLO counters from the underlying runtime."""
        fn = getattr(self._runtime, "compaction_stats", None)
        return dict(fn()) if callable(fn) else {}

    @property
    def kv_cache_size(self) -> int:
        return int(getattr(self._runtime, "kv_cache_size", 0))

    @property
    def kv_cache_sessions(self) -> tuple[str, ...]:
        values = getattr(self._runtime, "kv_cache_sessions", ())
        return tuple(str(item) for item in values)

    @property
    def compression_enabled(self) -> bool:
        return bool(getattr(self._runtime, "compression_enabled", False))

    @property
    def compression_latent_dim(self) -> int:
        return int(getattr(self._runtime, "compression_latent_dim", 0))

    @property
    def compression_encoded_payloads(self) -> int:
        return int(getattr(self._runtime, "compression_encoded_payloads", 0))

    @property
    def compression_decoded_payloads(self) -> int:
        return int(getattr(self._runtime, "compression_decoded_payloads", 0))

    @property
    def privacy_noise_variance(self) -> float:
        return float(getattr(self._runtime, "privacy_noise_variance", 0.0))

    @property
    def privacy_noise_payloads(self) -> int:
        return int(getattr(self._runtime, "privacy_noise_payloads", 0))

    @property
    def privacy_noise_observed_variance_ema(self) -> float:
        return float(getattr(self._runtime, "privacy_noise_observed_variance_ema", 0.0))

    @property
    def privacy_noise_last_observed_variance(self) -> float:
        return float(getattr(self._runtime, "privacy_noise_last_observed_variance", 0.0))

    @property
    def privacy_noise_last_observed_std(self) -> float:
        return float(getattr(self._runtime, "privacy_noise_last_observed_std", 0.0))

    @property
    def privacy_noise_last_audit_tag(self) -> str:
        return str(getattr(self._runtime, "privacy_noise_last_audit_tag", ""))

    @property
    def privacy_noise_last_payload_index(self) -> int:
        return int(getattr(self._runtime, "privacy_noise_last_payload_index", 0))

    @property
    def privacy_noise_last_applied(self) -> bool:
        return bool(getattr(self._runtime, "privacy_noise_last_applied", False))

    @staticmethod
    def _load_decode_tokenizer(model_id: str):
        normalized = str(model_id or "").strip()
        if not normalized:
            return None

        cache_name = "_openhydra_decode_tokenizer_cache"
        cache = getattr(ModelShard, cache_name, None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(ModelShard, cache_name, cache)
        if normalized in cache:
            return cache[normalized]

        try:
            from transformers import AutoTokenizer
        except Exception:
            return None

        try:
            # Local-first to avoid HF Hub HEAD requests per decode call.
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    normalized,
                    trust_remote_code=_default_trust_remote_code(normalized),
                    local_files_only=True,
                )
            except OSError:
                tokenizer = AutoTokenizer.from_pretrained(
                    normalized,
                    trust_remote_code=_default_trust_remote_code(normalized),
                )
        except Exception:
            return None
        cache[normalized] = tokenizer
        return tokenizer

    @staticmethod
    def decode_tokens(
        activation: list[float],
        max_tokens: int,
        tokenizer_model_id: str | None = None,
    ) -> list[str]:
        if not activation:
            return []

        count = max(1, min(max_tokens, 48))
        tokenizer_model = str(tokenizer_model_id or "").strip()
        if tokenizer_model:
            token_count = max(1, min(count, len(activation)))
            token_ids: list[int] = []
            for value in activation[:token_count]:
                try:
                    token_ids.append(max(0, int(round(float(value)))))
                except (TypeError, ValueError):
                    token_ids = []
                    break
            if token_ids:
                tokenizer = ModelShard._load_decode_tokenizer(tokenizer_model)
                if tokenizer is not None:
                    special_ids_raw = getattr(tokenizer, "all_special_ids", None)
                    special_ids: set[int] = set()
                    if special_ids_raw is not None:
                        try:
                            special_ids = {int(item) for item in list(special_ids_raw)}
                        except TypeError:
                            special_ids = set()
                    pieces: list[str] = []
                    for token_id in token_ids:
                        if token_id in special_ids:
                            continue
                        try:
                            piece = tokenizer.decode(
                                [token_id],
                                clean_up_tokenization_spaces=False,
                            )
                        except TypeError:
                            piece = tokenizer.decode([token_id])
                        pieces.append(str(piece))
                    if any(piece for piece in pieces):
                        return pieces
                    return []

        # Fallback: try tinyllama tokenizer for ToyRuntime outputs
        token_count = max(1, min(count, len(activation)))
        token_ids_fallback: list[int] = []
        for value in activation[:token_count]:
            try:
                token_ids_fallback.append(max(0, int(round(float(value)))))
            except (TypeError, ValueError):
                token_ids_fallback = []
                break
        if token_ids_fallback:
            fallback_tokenizer = ModelShard._load_decode_tokenizer(_TINYLLAMA_MODEL_ID)
            if fallback_tokenizer is None:
                fallback_tokenizer = ModelShard._load_decode_tokenizer(_TINYLLAMA_CACHE_DIR)
            if fallback_tokenizer is not None:
                try:
                    pieces = [fallback_tokenizer.decode([tid], skip_special_tokens=True) for tid in token_ids_fallback]
                    if any(p for p in pieces):
                        return pieces
                except Exception:
                    pass
        # Last resort: debug representation
        return [f"tok{int(round(float(v)))}" for v in activation[:count]]

    @staticmethod
    def render_text(tokens: list[str]) -> str:
        if not tokens:
            return ""

        text = " ".join(tokens)
        return text[0].upper() + text[1:] + "."

    @classmethod
    def decode_text(
        cls,
        activation: list[float],
        max_tokens: int,
        tokenizer_model_id: str | None = None,
    ) -> str:
        words = cls.decode_tokens(
            activation,
            max_tokens=max_tokens,
            tokenizer_model_id=tokenizer_model_id,
        )
        if not words:
            return ""
        if tokenizer_model_id:
            # NOTE: do NOT ``.strip()`` here. A first-token generation of
            # ``"\n\n"`` (common for Qwen 3.5 / other "thinking" models that
            # emit a whitespace preamble before their real content) would
            # otherwise collapse to "" and look like an empty response. The
            # tokenizer already emits clean pieces; trimming is the caller's
            # responsibility if they actually want it.
            return "".join(words)
        return cls.render_text(words)
