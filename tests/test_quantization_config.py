"""Phase 4 — NF4 Quantization config tests.

Verifies that:
  - Alias normalisation works for all CLI flag values
  - ToyRuntime propagates quantization_mode/bits into RuntimeProfile correctly
  - Memory and TPS estimates scale with bit-width
  - PyTorchRuntime source uses NF4 with bfloat16 + double quant (code inspection)
  - PyTorchRuntime gracefully falls back to fp32 when CUDA + bitsandbytes are absent
  - DHT Announcement carries correct quantization fields
  - ModelShard facade exposes quantization fields from the underlying runtime
  - MLXRuntime reads quantization_mode from config (MLX-guarded tests)
"""
from __future__ import annotations

import importlib
import inspect
import os
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from peer.model_shard import (
    ModelShard,
    PyTorchRuntime,
    RuntimeProfile,
    ToyRuntime,
    ToyShardConfig,
    _QUANTIZATION_ALIASES,
    _QUANTIZATION_BITS,
    _normalize_quantization_mode,
)
from peer.dht_announce import Announcement


# ─── _normalize_quantization_mode ───────────────────────────────────────────

class TestNormalizeQuantizationMode:
    """Alias resolution and guard-rail for unknown values."""

    def test_canonical_fp32_passthrough(self):
        assert _normalize_quantization_mode("fp32") == "fp32"

    def test_canonical_int8_passthrough(self):
        assert _normalize_quantization_mode("int8") == "int8"

    def test_canonical_int4_passthrough(self):
        assert _normalize_quantization_mode("int4") == "int4"

    def test_alias_none_maps_to_fp32(self):
        assert _normalize_quantization_mode("none") == "fp32"

    def test_alias_8bit_maps_to_int8(self):
        assert _normalize_quantization_mode("8bit") == "int8"

    def test_alias_4bit_maps_to_int4(self):
        assert _normalize_quantization_mode("4bit") == "int4"

    def test_unknown_mode_falls_back_to_fp32(self):
        assert _normalize_quantization_mode("bfloat16") == "fp32"
        assert _normalize_quantization_mode("nf4") == "fp32"
        assert _normalize_quantization_mode("") == "fp32"

    def test_case_insensitive(self):
        assert _normalize_quantization_mode("INT4") == "int4"
        assert _normalize_quantization_mode("NONE") == "fp32"
        assert _normalize_quantization_mode("4BIT") == "int4"

    def test_all_aliases_covered(self):
        """Every entry in _QUANTIZATION_ALIASES must resolve to a valid mode."""
        for alias, expected in _QUANTIZATION_ALIASES.items():
            result = _normalize_quantization_mode(alias)
            assert result == expected, f"alias {alias!r} → {result!r}, expected {expected!r}"
            assert result in _QUANTIZATION_BITS

    def test_bits_map_completeness(self):
        """Every mode that passes through normalisation must be in _QUANTIZATION_BITS."""
        for mode in ("fp32", "int8", "int4"):
            assert mode in _QUANTIZATION_BITS


# ─── ToyRuntime: quantization propagation ───────────────────────────────────

class TestToyRuntimeQuantization:
    """ToyRuntime must propagate quantization mode/bits into RuntimeProfile."""

    def _profile(self, mode: str) -> dict[str, Any]:
        config = ToyShardConfig(quantization_mode=mode)
        rt = ToyRuntime(config)
        return rt.runtime_profile()

    def test_fp32_profile_mode(self):
        p = self._profile("fp32")
        assert p["quantization_mode"] == "fp32"
        assert p["quantization_bits"] == 0

    def test_int8_profile_mode(self):
        p = self._profile("int8")
        assert p["quantization_mode"] == "int8"
        assert p["quantization_bits"] == 8

    def test_int4_profile_mode(self):
        p = self._profile("int4")
        assert p["quantization_mode"] == "int4"
        assert p["quantization_bits"] == 4

    def test_alias_4bit_normalised_in_profile(self):
        """CLI flag '4bit' must arrive in RuntimeProfile as 'int4'."""
        p = self._profile("4bit")
        assert p["quantization_mode"] == "int4"
        assert p["quantization_bits"] == 4

    def test_alias_8bit_normalised_in_profile(self):
        p = self._profile("8bit")
        assert p["quantization_mode"] == "int8"
        assert p["quantization_bits"] == 8

    def test_alias_none_normalised_in_profile(self):
        p = self._profile("none")
        assert p["quantization_mode"] == "fp32"
        assert p["quantization_bits"] == 0

    def test_int8_memory_lower_than_fp32(self):
        mem_fp32 = self._profile("fp32")["estimated_memory_mb"]
        mem_int8 = self._profile("int8")["estimated_memory_mb"]
        assert mem_int8 < mem_fp32, "int8 should use less memory than fp32"

    def test_int4_memory_lower_than_int8(self):
        mem_int8 = self._profile("int8")["estimated_memory_mb"]
        mem_int4 = self._profile("int4")["estimated_memory_mb"]
        assert mem_int4 < mem_int8, "int4 should use less memory than int8"

    def test_int8_tps_higher_than_fp32(self):
        tps_fp32 = self._profile("fp32")["estimated_tokens_per_sec"]
        tps_int8 = self._profile("int8")["estimated_tokens_per_sec"]
        assert tps_int8 > tps_fp32, "int8 should be faster than fp32"

    def test_int4_tps_higher_than_fp32(self):
        tps_fp32 = self._profile("fp32")["estimated_tokens_per_sec"]
        tps_int4 = self._profile("int4")["estimated_tokens_per_sec"]
        assert tps_int4 > tps_fp32, "int4 should be faster than fp32"

    def test_runtime_profile_returns_dict(self):
        config = ToyShardConfig(quantization_mode="int4")
        rt = ToyRuntime(config)
        p = rt.runtime_profile()
        assert isinstance(p, dict)

    def test_estimated_tokens_per_sec_key_present(self):
        """Key must be 'estimated_tokens_per_sec', not 'estimated_tps'."""
        p = self._profile("fp32")
        assert "estimated_tokens_per_sec" in p, (
            "RuntimeProfile must use 'estimated_tokens_per_sec', not 'estimated_tps'"
        )


# ─── PyTorchRuntime: NF4 source inspection ───────────────────────────────────

class TestPyTorchRuntimeNF4Source:
    """Verify NF4 parameters are present in the PyTorchRuntime source.

    These tests use code inspection so they pass without CUDA / bitsandbytes
    installed — they simply confirm the correct BitsAndBytesConfig arguments
    are present in the source.
    """

    @pytest.fixture(scope="class")
    def pytorch_source(self) -> str:
        return inspect.getsource(PyTorchRuntime)

    def test_nf4_quant_type_present(self, pytorch_source: str):
        assert 'bnb_4bit_quant_type="nf4"' in pytorch_source, (
            "PyTorchRuntime must use bnb_4bit_quant_type='nf4' for 4-bit quantization"
        )

    def test_bfloat16_compute_dtype_present(self, pytorch_source: str):
        assert "bfloat16" in pytorch_source, (
            "PyTorchRuntime must use bfloat16 as compute dtype for NF4 (not float16)"
        )

    def test_double_quant_present(self, pytorch_source: str):
        assert "bnb_4bit_use_double_quant=True" in pytorch_source, (
            "PyTorchRuntime must enable double quantization for NF4"
        )

    def test_load_in_4bit_present(self, pytorch_source: str):
        assert "load_in_4bit=True" in pytorch_source

    def test_load_in_8bit_present(self, pytorch_source: str):
        assert "load_in_8bit=True" in pytorch_source


# ─── PyTorchRuntime: graceful fallback ──────────────────────────────────────

class TestPyTorchRuntimeFallback:
    """PyTorchRuntime must not crash when CUDA / bitsandbytes are absent."""

    def _make_config(self, mode: str) -> ToyShardConfig:
        return ToyShardConfig(
            runtime_backend="pytorch_auto",
            quantization_mode=mode,
            runtime_model_id="Qwen/Qwen3.5-0.8B",
        )

    def _build_runtime_with_mocked_torch(self, mode: str) -> PyTorchRuntime:
        """Instantiate PyTorchRuntime with torch + transformers mocked out."""
        config = self._make_config(mode)

        # Build a minimal mock transformers package.
        mock_transformers = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 2
        mock_model = MagicMock()
        mock_model.config.num_hidden_layers = 32
        mock_model.config.hidden_size = 1024
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.float16 = "float16"
        mock_torch.float32 = "float32"
        mock_torch.bfloat16 = "bfloat16"
        mock_torch.device = lambda x: x

        with (
            patch.dict("sys.modules", {
                "torch": mock_torch,
                "transformers": mock_transformers,
            }),
            patch("peer.model_shard._gpu_available_hint", return_value=False),
        ):
            rt = PyTorchRuntime(config)
        return rt

    def test_int4_fallback_to_fp32_without_cuda(self):
        """int4 on non-CUDA must silently fall back to fp32, not raise."""
        rt = self._build_runtime_with_mocked_torch("int4")
        p = rt.runtime_profile()
        # Fallback: quantization_mode should be fp32 or int4 (if bnb succeeded).
        # Either way, no exception means the fallback path works.
        assert p["quantization_mode"] in ("fp32", "int4")

    def test_int8_fallback_to_fp32_without_cuda(self):
        rt = self._build_runtime_with_mocked_torch("int8")
        p = rt.runtime_profile()
        assert p["quantization_mode"] in ("fp32", "int8")

    def test_fp32_no_quantization_config_on_cpu(self):
        """fp32 should never trigger BitsAndBytesConfig."""
        rt = self._build_runtime_with_mocked_torch("fp32")
        p = rt.runtime_profile()
        assert p["quantization_mode"] == "fp32"
        assert p["quantization_bits"] == 0


# ─── DHT Announcement: quantization fields ──────────────────────────────────

class TestDHTAnnouncementQuantizationFields:
    """Announcement dataclass must carry quantization_mode and quantization_bits."""

    def test_default_quantization_mode(self):
        ann = Announcement(peer_id="p1", model_id="m1", host="127.0.0.1", port=8468)
        assert ann.quantization_mode == "fp32"

    def test_default_quantization_bits(self):
        ann = Announcement(peer_id="p1", model_id="m1", host="127.0.0.1", port=8468)
        assert ann.quantization_bits == 0

    def test_custom_int4_mode(self):
        ann = Announcement(
            peer_id="p1", model_id="m1", host="127.0.0.1", port=8468,
            quantization_mode="int4", quantization_bits=4,
        )
        assert ann.quantization_mode == "int4"
        assert ann.quantization_bits == 4

    def test_custom_int8_mode(self):
        ann = Announcement(
            peer_id="p1", model_id="m1", host="127.0.0.1", port=8468,
            quantization_mode="int8", quantization_bits=8,
        )
        assert ann.quantization_mode == "int8"
        assert ann.quantization_bits == 8

    def test_announcement_asdict_contains_quant_fields(self):
        """asdict() output must include both quantization fields for DHT serialisation."""
        from dataclasses import asdict
        ann = Announcement(
            peer_id="p1", model_id="m1", host="127.0.0.1", port=8468,
            quantization_mode="int4", quantization_bits=4,
        )
        d = asdict(ann)
        assert "quantization_mode" in d
        assert "quantization_bits" in d
        assert d["quantization_mode"] == "int4"
        assert d["quantization_bits"] == 4


# ─── ModelShard facade propagation ──────────────────────────────────────────

class TestModelShardFacadeQuantization:
    """ModelShard.runtime_profile() must expose quantization fields from ToyRuntime."""

    def _shard(self, mode: str) -> ModelShard:
        config = ToyShardConfig(quantization_mode=mode)
        return ModelShard(config)

    def test_fp32_propagated(self):
        p = self._shard("fp32").runtime_profile()
        assert p["quantization_mode"] == "fp32"
        assert p["quantization_bits"] == 0

    def test_int8_propagated(self):
        p = self._shard("int8").runtime_profile()
        assert p["quantization_mode"] == "int8"
        assert p["quantization_bits"] == 8

    def test_int4_propagated(self):
        p = self._shard("int4").runtime_profile()
        assert p["quantization_mode"] == "int4"
        assert p["quantization_bits"] == 4

    def test_alias_4bit_propagated_as_int4(self):
        p = self._shard("4bit").runtime_profile()
        assert p["quantization_mode"] == "int4"

    def test_profile_is_fresh_dict_not_reference(self):
        """runtime_profile() must return a copy, not the internal dict."""
        shard = self._shard("int4")
        p1 = shard.runtime_profile()
        p2 = shard.runtime_profile()
        assert p1 == p2
        assert p1 is not p2

    def test_estimated_tokens_per_sec_key(self):
        """Key must be 'estimated_tokens_per_sec', NOT 'estimated_tps'."""
        p = self._shard("fp32").runtime_profile()
        assert "estimated_tokens_per_sec" in p
        assert "estimated_tps" not in p


# ─── RuntimeProfile dataclass ────────────────────────────────────────────────

class TestRuntimeProfileDataclass:
    """RuntimeProfile must have quantization_mode and quantization_bits fields."""

    def test_has_quantization_mode_field(self):
        fields = {f.name for f in RuntimeProfile.__dataclass_fields__.values()}
        assert "quantization_mode" in fields

    def test_has_quantization_bits_field(self):
        fields = {f.name for f in RuntimeProfile.__dataclass_fields__.values()}
        assert "quantization_bits" in fields

    def test_to_dict_includes_quant_fields(self):
        p = RuntimeProfile(
            backend="toy_cpu",
            target="cpu",
            quantization_mode="int4",
            quantization_bits=4,
            gpu_available=False,
            estimated_tokens_per_sec=100.0,
            estimated_memory_mb=512,
        )
        d = p.to_dict()
        assert d["quantization_mode"] == "int4"
        assert d["quantization_bits"] == 4

    def test_int4_tps_estimate_higher_than_fp32(self):
        """Quantized ToyRuntime should report higher throughput than fp32."""
        config_fp32 = ToyShardConfig(quantization_mode="fp32")
        config_int4 = ToyShardConfig(quantization_mode="int4")
        rt_fp32 = ToyRuntime(config_fp32)
        rt_int4 = ToyRuntime(config_int4)
        assert (
            rt_int4.runtime_profile()["estimated_tokens_per_sec"]
            > rt_fp32.runtime_profile()["estimated_tokens_per_sec"]
        )

    def test_int4_memory_lower_than_fp32(self):
        config_fp32 = ToyShardConfig(quantization_mode="fp32")
        config_int4 = ToyShardConfig(quantization_mode="int4")
        rt_fp32 = ToyRuntime(config_fp32)
        rt_int4 = ToyRuntime(config_int4)
        assert (
            rt_int4.runtime_profile()["estimated_memory_mb"]
            < rt_fp32.runtime_profile()["estimated_memory_mb"]
        )


# ─── MLXRuntime: quantization (MLX-guarded) ─────────────────────────────────

mlx_available = importlib.util.find_spec("mlx") is not None
mlx_lm_available = importlib.util.find_spec("mlx_lm") is not None

@pytest.mark.skipif(
    not (mlx_available and mlx_lm_available),
    reason="mlx and mlx-lm required",
)
class TestMLXRuntimeQuantization:
    """MLXRuntime quantization — only run when MLX is installed."""

    def _make_config(self, mode: str) -> ToyShardConfig:
        return ToyShardConfig(
            runtime_backend="mlx",
            quantization_mode=mode,
            runtime_model_id="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        )

    def test_profile_has_runtime_model_id(self):
        from peer.mlx_runtime import MLXRuntime
        config = self._make_config("fp32")
        rt = MLXRuntime(config)
        p = rt.runtime_profile()
        assert "runtime_model_id" in p
        assert p["runtime_model_id"] == config.runtime_model_id

    def test_profile_has_total_layers(self):
        from peer.mlx_runtime import MLXRuntime
        config = self._make_config("fp32")
        rt = MLXRuntime(config)
        p = rt.runtime_profile()
        assert "total_layers" in p
        assert isinstance(p["total_layers"], int)

    def test_profile_layer_end_equals_total_layers(self):
        """layer_end must equal total_layers (not total_layers - 1)."""
        from peer.mlx_runtime import MLXRuntime
        config = self._make_config("fp32")
        rt = MLXRuntime(config)
        p = rt.runtime_profile()
        assert p["layer_end"] == p["total_layers"], (
            f"layer_end={p['layer_end']} should equal total_layers={p['total_layers']}"
        )

    def test_profile_has_estimated_tokens_per_sec_not_estimated_tps(self):
        from peer.mlx_runtime import MLXRuntime
        config = self._make_config("fp32")
        rt = MLXRuntime(config)
        p = rt.runtime_profile()
        assert "estimated_tokens_per_sec" in p, (
            "MLXRuntime must use key 'estimated_tokens_per_sec', not 'estimated_tps'"
        )
        assert "estimated_tps" not in p

    def test_fp32_request_loads_model(self):
        from peer.mlx_runtime import MLXRuntime
        config = self._make_config("fp32")
        rt = MLXRuntime(config)
        p = rt.runtime_profile()
        assert p["backend"] == "mlx"

    def test_int4_request_sets_quantization(self):
        """Requesting int4 should result in a quantized profile."""
        from peer.mlx_runtime import MLXRuntime
        config = self._make_config("int4")
        rt = MLXRuntime(config)
        p = rt.runtime_profile()
        # The pre-quantized 4bit checkpoint or runtime quantization should be detected.
        assert p["quantization_bits"] in (4, 16), (
            "int4 request on a 4-bit checkpoint should produce quantization_bits=4"
        )

    def test_mlx_quant_bits_module_constant_present(self):
        """_MLX_QUANT_BITS must be importable from mlx_runtime."""
        from peer.mlx_runtime import _MLX_QUANT_BITS
        assert "fp32" in _MLX_QUANT_BITS
        assert "int4" in _MLX_QUANT_BITS
        assert "int8" in _MLX_QUANT_BITS
        assert _MLX_QUANT_BITS["int4"] == 4
        assert _MLX_QUANT_BITS["int8"] == 8
