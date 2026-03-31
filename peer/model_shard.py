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

ACTIVATION_SIZE = 64

_VOCAB = [
    "hydra", "swarm", "tensor", "latency", "peer", "pipeline", "decode", "prefill",
    "routing", "reputation", "redundant", "verify", "bandwidth", "cache", "token",
    "context", "throughput", "resilient", "distributed", "inference", "model", "layer",
    "session", "fallback", "degrade", "secure", "signal", "quorum", "consensus",
    "operator", "diversity", "fairness", "compute", "queue", "scheduler", "checkpoint",
    "vector", "attention", "entropy", "stable", "adaptive", "transparent", "daemon",
    "polite", "credits", "governance", "network", "uplink", "response", "prompt",
    "quality", "balance", "retry", "health", "fault", "audit", "score", "cluster",
    "edge", "stream", "orchestrate", "shard", "fabric", "trust", "mesh",
]

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
    model_id: str = "openhydra-toy-345m"
    shard_index: int = 0
    total_shards: int = 1
    broken: bool = False
    runtime_backend: str = "toy_auto"
    runtime_target: str = "auto"
    quantization_mode: str = "fp32"
    runtime_model_id: str = "Qwen/Qwen3.5-0.8B"
    runtime_trust_remote_code: bool | None = None
    runtime_layer_indices: tuple[int, ...] = ()
    runtime_max_context_tokens: int = 64
    runtime_kv_cache_max_entries: int = 256
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

    # MLX watchdog: maximum seconds for any single MLX computation (mx.eval,
    # stream_generate loop, forward_batch decode loop).  If exceeded, the
    # computation is treated as a GPU hang and a TimeoutError is raised.
    # Default raised from 30s to 120s to support 8 GB machines under memory pressure.
    runtime_mlx_eval_timeout_s: float = 120.0


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


class ToyRuntime:
    def __init__(self, config: ToyShardConfig):
        self.config = config
        salt_material = f"{config.model_id}:{config.shard_index}:{config.total_shards}".encode()
        self._salt = hashlib.sha256(salt_material).digest()
        self._runtime_profile = self._build_runtime_profile(config)
        self.last_forward_thread_id: int | None = None

    @staticmethod
    def _digest(text: str) -> bytes:
        return hashlib.sha256(text.encode("utf-8")).digest()

    @classmethod
    def _build_runtime_profile(cls, config: ToyShardConfig) -> RuntimeProfile:
        gpu_available = _gpu_available_hint()

        requested_target = str(config.runtime_target or "auto").strip().lower()
        if requested_target not in {"auto", "cpu", "cuda"}:
            requested_target = "auto"
        target = requested_target
        if target == "auto":
            target = "cuda" if gpu_available else "cpu"
        if target == "cuda" and not gpu_available:
            target = "cpu"

        requested_backend = str(config.runtime_backend or "toy_auto").strip().lower()
        if requested_backend not in {"toy_auto", "toy_cpu", "toy_gpu_sim"}:
            requested_backend = "toy_auto"
        if requested_backend == "toy_auto":
            backend = "toy_gpu_sim" if target == "cuda" else "toy_cpu"
        else:
            backend = requested_backend
        if backend == "toy_gpu_sim" and target != "cuda":
            backend = "toy_cpu"

        quantization_mode = _normalize_quantization_mode(config.quantization_mode)
        quantization_bits = _QUANTIZATION_BITS[quantization_mode]

        base_tps = 96.0
        if backend == "toy_gpu_sim":
            base_tps *= 2.75
        if quantization_bits == 8:
            base_tps *= 1.3
        elif quantization_bits == 4:
            base_tps *= 1.75
        shard_penalty = max(0.65, 1.0 - (max(0, int(config.shard_index)) * 0.03))
        est_tps = round(max(8.0, base_tps * shard_penalty), 3)

        base_mem = 1400 if backend == "toy_cpu" else 2200
        if quantization_bits == 8:
            base_mem = int(round(base_mem * 0.58))
        elif quantization_bits == 4:
            base_mem = int(round(base_mem * 0.37))
        est_mem = max(128, int(base_mem))

        return RuntimeProfile(
            backend=backend,
            target=target,
            quantization_mode=quantization_mode,
            quantization_bits=quantization_bits,
            gpu_available=gpu_available,
            estimated_tokens_per_sec=est_tps,
            estimated_memory_mb=est_mem,
            runtime_model_id=str(config.model_id),
            layer_start=int(config.shard_index),
            layer_end=int(config.shard_index + 1),
        )

    def runtime_profile(self) -> dict[str, Any]:
        return self._runtime_profile.to_dict()

    def reshard(self, new_layer_start: int, new_layer_end: int, total_layers: int) -> bool:
        """Reshard the ToyRuntime to cover [new_layer_start, new_layer_end).

        ToyRuntime is a mock runtime — resharding simply updates the
        runtime profile metadata.  Returns ``True`` on success.
        """
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
        logging.info(
            "toy_reshard: [%d, %d) total=%d",
            new_layer_start, new_layer_end, total_layers,
        )
        return True

    def encode_prompt(self, prompt: str, max_tokens: int) -> list[float]:
        digest = self._digest(f"{prompt}|{max_tokens}")
        activation: list[float] = []
        for i in range(ACTIVATION_SIZE):
            b = digest[i % len(digest)]
            c = self._salt[i % len(self._salt)]
            centered = ((b ^ c) - 127) / 127.0
            activation.append(centered)
        return activation

    def _mix(self, activation: list[float]) -> list[float]:
        out: list[float] = []
        length = len(activation)
        if length == 0:
            return []

        step = (self.config.shard_index + 1) % length
        for i, val in enumerate(activation):
            neighbor = activation[(i + step) % length]
            bias = (self._salt[(i + self.config.shard_index) % len(self._salt)] - 127) / 1200.0
            mixed = math.tanh(0.77 * val + 0.21 * neighbor + bias)
            out.append(mixed)

        if self.config.broken:
            # Intentional deterministic corruption for Mystery Shopper tests.
            out = [math.tanh(v * 3.0 + 0.35) for v in out]

        return out

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
        self.last_forward_thread_id = threading.get_ident()
        base = activation or self.encode_prompt(prompt, max_tokens)
        quantized_base = _apply_quantization(base, self._runtime_profile.quantization_bits)
        mixed = self._mix(quantized_base)
        return _apply_quantization(mixed, self._runtime_profile.quantization_bits)

    def forward_batch(self, items: list[Any]) -> list[list[float]]:
        """Sequential forward batch for ToyRuntime.

        ToyRuntime has no GPU, so batching provides no performance benefit.
        This method exists to satisfy the forward_batch() contract expected
        by BatchingQueue._dispatch(), allowing tests to spy on it.
        """
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
        """Stub — ToyRuntime performs no real compaction."""
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
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"pytorch-shard-{int(config.shard_index)}")
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
        if requested_target not in {"auto", "cpu", "cuda"}:
            requested_target = "auto"
        gpu_available = bool(torch.cuda.is_available())
        target = requested_target
        if target == "auto":
            target = "cuda" if gpu_available else "cpu"
        if target == "cuda" and not gpu_available:
            target = "cpu"

        requested_backend = str(config.runtime_backend or "pytorch_auto").strip().lower()
        if requested_backend not in {"pytorch_auto", "pytorch_cpu", "pytorch_cuda", "pytorch"}:
            requested_backend = "pytorch_auto"
        if requested_backend in {"pytorch_cpu"}:
            target = "cpu"
        elif requested_backend in {"pytorch_cuda"} and gpu_available:
            target = "cuda"
        backend = "pytorch_cuda" if target == "cuda" else "pytorch_cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self._trust_remote_code,
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

        self._device = torch.device("cuda" if target == "cuda" else "cpu")
        self._dtype = torch.float16 if target == "cuda" else torch.float32

        load_kwargs: dict[str, Any] = {
            "low_cpu_mem_usage": True,
        }
        quantized_weights_loaded = False
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"
            load_kwargs["torch_dtype"] = torch.float16
        else:
            load_kwargs["torch_dtype"] = self._dtype

        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=self._trust_remote_code,
                **load_kwargs,
            )
            quantized_weights_loaded = quantization_config is not None
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
            )
            quantized_weights_loaded = False

        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        if not quantized_weights_loaded:
            self._model.to(self._device)
        self._model.eval()

        decoder_arch = self._detect_decoder_architecture(self._model)
        self._decoder_family = str(decoder_arch.family)
        self._embed_tokens = decoder_arch.embed_tokens
        self._position_embeddings = decoder_arch.position_embeddings
        self._final_norm = decoder_arch.final_norm
        self._rotary_emb = decoder_arch.rotary_emb
        self._blocks = list(decoder_arch.layers)
        self.total_layers = len(self._blocks)
        self.layer_indices = self._resolve_layer_indices(
            total_layers=self.total_layers,
            shard_index=max(0, int(config.shard_index)),
            total_shards=max(1, int(config.total_shards)),
            explicit_indices=tuple(config.runtime_layer_indices),
        )
        self._selected_layers = [self._blocks[idx] for idx in self.layer_indices]
        self._hidden_size = int(
            getattr(self._model.config, "hidden_size", 0)
            or getattr(self._model.config, "n_embd", 0)
            or getattr(self._model.config, "d_model", 0)
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

        base_tps = 12.0 if target == "cpu" else 42.0
        if self.quantization_bits == 8:
            base_tps *= 1.2
        elif self.quantization_bits == 4:
            base_tps *= 1.45
        layer_fraction = (len(self.layer_indices) / float(max(1, self.total_layers))) if self.total_layers else 1.0
        layer_penalty = max(0.35, layer_fraction)
        estimated_tps = round(base_tps / layer_penalty, 3)

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

    @staticmethod
    def _detect_decoder_architecture(model: Any) -> _DecoderArchitecture:
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

    def _activation_to_hidden(self, activation: list[float]):
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
        output = hidden
        present_key_values: list[Any] = []
        shared_cache = past_key_values if (past_key_values is not None and not isinstance(past_key_values, (tuple, list))) else None
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
                try:
                    block_kwargs["position_embeddings"] = self._rotary_emb(output, position_ids)
                except Exception:
                    # Keep runtime resilient across transformer minor-version signature changes.
                    pass

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
            raise RuntimeError("kv_cache_miss")
        self.last_kv_cache_hit = bool(cache_requested and cached_past is not None)
        cache_enabled = bool(session_id and (kv_store_activation or cache_requested))
        next_past = None

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
                    if not activation:
                        raise RuntimeError("missing_hidden_payload")
                    hidden = self._activation_to_hidden(activation)
                    seq_len = int(hidden.shape[1])
                    past_len = self._past_sequence_length(cached_past)
                    position_ids = self._build_position_ids(seq_len=seq_len, past_len=past_len)
                cache_position = position_ids.squeeze(0)
                attention_mask = None
                if self._decoder_family in {"llama", "qwen_llama"}:
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
                    self._kv_cache_set(session_id, next_past)
                if is_last:
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
                    output = self._hidden_to_payload(
                        hidden,
                        request_id=request_id,
                        stage_index=stage,
                    )
        if is_last:
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
    ) -> list[float]:
        loop = asyncio.get_running_loop()
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
    ) -> list[float]:
        return list(
            self._runtime.forward(
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
            )
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
    ) -> list[float]:
        async_forward = getattr(self._runtime, "forward_async", None)
        if callable(async_forward):
            return list(
                await async_forward(
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
                )
            )
        return list(
            await asyncio.to_thread(
                self._runtime.forward,
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

        words: list[str] = []
        length = len(activation)
        for i in range(count):
            val = activation[i % length]
            idx = int(((val + 1.0) / 2.0) * (len(_VOCAB) - 1))
            idx = max(0, min(len(_VOCAB) - 1, idx))
            words.append(_VOCAB[idx])
        return words

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
            return "".join(words).strip()
        return cls.render_text(words)
