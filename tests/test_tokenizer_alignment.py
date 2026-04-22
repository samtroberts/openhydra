# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Tokenizer alignment across MLX ↔ PyTorch heterogeneous peers.

The canonical tokenizer for a model is whatever
``hf_model_id`` in ``models.catalog.json`` says. Every backend — PyTorch
or MLX — must encode and decode through that HF tokenizer so token IDs
produced on one peer are valid indices into any other peer's embedding
table. These tests cover:

* :func:`peer.model_catalog.resolve_hf_model_id` — catalog lookup +
  sensible fallbacks, including explicit rejection of ``mlx-community/``
  mirrors as a fallback source.
* :class:`peer.mlx_runtime.MLXRuntime` — the hot-path override that
  discards the mlx-community tokenizer and loads the HF one when
  ``runtime_hf_model_id`` is set.
* :class:`peer.model_shard.ToyShardConfig` — the new alignment fields.

Run:  ``pytest tests/test_tokenizer_alignment.py -v``
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ─── catalog resolver ────────────────────────────────────────────────────────


class TestResolveHfModelId:
    def test_catalog_hit_returns_hf_model_id(self, tmp_path: Path):
        catalog = tmp_path / "models.catalog.json"
        catalog.write_text(json.dumps([
            {"model_id": "openhydra-qwen3.5-2b", "hf_model_id": "Qwen/Qwen3.5-2B"},
        ]))
        from peer.model_catalog import resolve_hf_model_id
        assert resolve_hf_model_id(
            "openhydra-qwen3.5-2b", catalog_path=catalog,
        ) == "Qwen/Qwen3.5-2B"

    def test_catalog_miss_falls_back_to_runtime_model_id(self, tmp_path: Path):
        catalog = tmp_path / "models.catalog.json"
        catalog.write_text(json.dumps([]))
        from peer.model_catalog import resolve_hf_model_id
        assert resolve_hf_model_id(
            "unknown-model",
            catalog_path=catalog,
            runtime_model_id="meta-llama/Llama-3-8B",
        ) == "meta-llama/Llama-3-8B"

    def test_mlx_community_runtime_id_is_rejected_as_fallback(self, tmp_path: Path):
        """An mlx-community mirror must NEVER be used as the HF tokenizer
        source — that's the exact bug this module was built to prevent."""
        catalog = tmp_path / "missing.json"  # does not exist
        from peer.model_catalog import resolve_hf_model_id
        # Runtime id is mlx-community/... → rejected.
        # model_id is a plain slug → also rejected (no "/").
        # → empty string.
        assert resolve_hf_model_id(
            "openhydra-qwen3.5-2b",
            catalog_path=catalog,
            runtime_model_id="mlx-community/Qwen3.5-2B-4bit",
        ) == ""

    def test_model_id_used_as_last_resort_when_hf_shaped(self, tmp_path: Path):
        catalog = tmp_path / "empty.json"
        catalog.write_text("[]")
        from peer.model_catalog import resolve_hf_model_id
        assert resolve_hf_model_id(
            "Qwen/Qwen3.5-2B", catalog_path=catalog,
        ) == "Qwen/Qwen3.5-2B"

    def test_missing_catalog_file_does_not_raise(self, tmp_path: Path):
        from peer.model_catalog import resolve_hf_model_id
        # Non-existent catalog + unresolvable fallbacks → empty string.
        assert resolve_hf_model_id(
            "plain-slug", catalog_path=tmp_path / "nope.json",
        ) == ""

    def test_malformed_catalog_is_non_fatal(self, tmp_path: Path):
        catalog = tmp_path / "bad.json"
        catalog.write_text("not-json")
        from peer.model_catalog import resolve_hf_model_id
        # Should log + fall through, not raise.
        assert resolve_hf_model_id(
            "plain-slug", catalog_path=catalog,
        ) == ""

    def test_entry_with_empty_hf_model_id_falls_through(self, tmp_path: Path):
        catalog = tmp_path / "c.json"
        catalog.write_text(json.dumps([
            {"model_id": "my-model", "hf_model_id": ""},
        ]))
        from peer.model_catalog import resolve_hf_model_id
        # Entry exists but hf is empty → try runtime_model_id next.
        assert resolve_hf_model_id(
            "my-model",
            catalog_path=catalog,
            runtime_model_id="Qwen/Qwen3.5-0.8B",
        ) == "Qwen/Qwen3.5-0.8B"


# ─── ToyShardConfig fields ───────────────────────────────────────────────────


class TestToyShardConfigAlignmentFields:
    def test_defaults_favour_alignment_on(self):
        from peer.model_shard import ToyShardConfig
        cfg = ToyShardConfig()
        assert cfg.runtime_hf_model_id == ""
        assert cfg.runtime_mlx_force_hf_tokenizer is True
        assert cfg.runtime_tokenizer_vocab_guard is True

    def test_fields_are_overridable(self):
        from peer.model_shard import ToyShardConfig
        cfg = ToyShardConfig(
            runtime_hf_model_id="Qwen/Qwen3.5-2B",
            runtime_mlx_force_hf_tokenizer=False,
            runtime_tokenizer_vocab_guard=False,
        )
        assert cfg.runtime_hf_model_id == "Qwen/Qwen3.5-2B"
        assert cfg.runtime_mlx_force_hf_tokenizer is False
        assert cfg.runtime_tokenizer_vocab_guard is False


# ─── MLXRuntime tokenizer override (heavily mocked) ──────────────────────────


def _install_fake_mlx() -> dict[str, types.ModuleType]:
    """Inject minimal fake mlx.* + mlx_lm modules into ``sys.modules``.

    Returns the fake modules so callers can reach into them. The fakes are
    only rich enough for the tokenizer-override path in
    ``MLXRuntime.__init__`` — anything past the tokenizer block is
    exercised separately in the real-MLX tests.
    """
    fake = {}
    # mlx.core + mlx.nn
    mx = types.ModuleType("mlx.core")
    mx.eval = lambda *a, **kw: None
    mx.array = lambda *a, **kw: MagicMock(name="mx.array")
    mx.argmax = lambda *a, **kw: MagicMock()
    fake["mlx.core"] = mx
    mlx_pkg = types.ModuleType("mlx")
    mlx_pkg.core = mx
    fake["mlx"] = mlx_pkg
    nn = types.ModuleType("mlx.nn")
    fake["mlx.nn"] = nn
    utils = types.ModuleType("mlx.utils")
    utils.tree_flatten = lambda _: []
    fake["mlx.utils"] = utils
    # mlx_lm
    mlx_lm = types.ModuleType("mlx_lm")
    fake["mlx_lm"] = mlx_lm
    return fake


class _FakeMlxTokenizer:
    vocab_size = 151643  # mlx-community Qwen variant — different on purpose
    name = "mlx-bundled"


class _FakeHfTokenizer:
    vocab_size = 151936  # canonical HF Qwen3.5-2B
    name = "hf-canonical"


class _FakeMlxModel:
    class _LanguageModel:
        class _Inner:
            class _Embed:
                # Power-of-two-ish padded vocab ≥ tokenizer.vocab_size.
                class _Weight:
                    shape = (151936, 2048)
                weight = _Weight()
            embed_tokens = _Embed()
            layers = []
        model = _Inner()
    language_model = _LanguageModel()
    # For non-sharded path, MLXRuntime reads model.model.layers.
    model = _LanguageModel._Inner

    def parameters(self):
        return {}


@pytest.fixture
def fake_mlx_env(monkeypatch):
    """Install fake mlx / mlx_lm / transformers modules for this test."""
    fakes = _install_fake_mlx()
    for name, mod in fakes.items():
        monkeypatch.setitem(sys.modules, name, mod)

    # Fake mlx_lm.load — returns (model, tokenizer) tuple.
    def _fake_load(model_name):
        return _FakeMlxModel(), _FakeMlxTokenizer()
    sys.modules["mlx_lm"].load = _fake_load

    # Fake transformers.AutoTokenizer.from_pretrained.
    transformers = types.ModuleType("transformers")
    hf_calls: list[tuple[str, dict]] = []
    def _fake_from_pretrained(name, **kwargs):
        hf_calls.append((name, dict(kwargs)))
        return _FakeHfTokenizer()
    transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=_fake_from_pretrained,
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    yield hf_calls


@pytest.fixture
def minimal_mlx_config():
    from peer.model_shard import ToyShardConfig
    return ToyShardConfig(
        runtime_backend="mlx",
        runtime_model_id="mlx-community/Qwen3.5-2B-4bit",
        runtime_hf_model_id="Qwen/Qwen3.5-2B",
        runtime_mlx_force_hf_tokenizer=True,
        runtime_tokenizer_vocab_guard=True,
        quantization_mode="fp32",
        shard_index=0,
        total_shards=1,
    )


class TestMLXRuntimeTokenizerOverride:
    def test_override_replaces_bundled_tokenizer_with_hf(
        self, fake_mlx_env, minimal_mlx_config,
    ):
        # Reload mlx_runtime so its `from transformers import …` picks up the fake.
        sys.modules.pop("peer.mlx_runtime", None)
        from peer.mlx_runtime import MLXRuntime
        rt = MLXRuntime(minimal_mlx_config)
        # The tokenizer the runtime actually uses is the HF one.
        assert isinstance(rt._tokenizer, _FakeHfTokenizer)
        assert rt._tokenizer.name == "hf-canonical"
        # HF loader was invoked with the canonical id.
        assert any(
            call[0] == "Qwen/Qwen3.5-2B" for call in fake_mlx_env
        ), f"expected HF load for Qwen/Qwen3.5-2B, got {fake_mlx_env}"

    def test_runtime_profile_advertises_hf_model_id(
        self, fake_mlx_env, minimal_mlx_config,
    ):
        sys.modules.pop("peer.mlx_runtime", None)
        from peer.mlx_runtime import MLXRuntime
        rt = MLXRuntime(minimal_mlx_config)
        profile = rt.runtime_profile()
        # Downstream peers resolve their tokenizer from runtime_model_id.
        # When the override is active that value MUST be the HF id, so
        # _resolve_pipeline_runtime_model_id lands on an HF-compatible
        # tokenizer regardless of pipeline order.
        assert profile["runtime_model_id"] == "Qwen/Qwen3.5-2B"
        assert profile["runtime_mlx_model_id"] == "mlx-community/Qwen3.5-2B-4bit"
        assert profile["runtime_hf_model_id"] == "Qwen/Qwen3.5-2B"
        assert profile["tokenizer_vocab_size"] == _FakeHfTokenizer.vocab_size

    def test_flag_off_keeps_bundled_tokenizer(
        self, fake_mlx_env, minimal_mlx_config,
    ):
        # Rebuild config with the override disabled.
        from peer.model_shard import ToyShardConfig
        cfg = ToyShardConfig(
            **{**minimal_mlx_config.__dict__, "runtime_mlx_force_hf_tokenizer": False}
        )
        sys.modules.pop("peer.mlx_runtime", None)
        from peer.mlx_runtime import MLXRuntime
        rt = MLXRuntime(cfg)
        assert isinstance(rt._tokenizer, _FakeMlxTokenizer)
        # HF loader must not have been invoked.
        assert fake_mlx_env == []

    def test_hf_load_failure_falls_back_to_bundled_tokenizer(
        self, fake_mlx_env, minimal_mlx_config, monkeypatch,
    ):
        """If ``AutoTokenizer.from_pretrained`` raises (no network, no
        cache, …) we must degrade gracefully to the bundled tokenizer
        rather than crashing peer startup — the peer is still useful
        inside a same-backend ring."""
        def _always_fails(name, **kwargs):
            raise OSError("simulated no-network")
        sys.modules["transformers"].AutoTokenizer.from_pretrained = _always_fails
        sys.modules.pop("peer.mlx_runtime", None)
        from peer.mlx_runtime import MLXRuntime
        # Disable the guard for this test — with MLX bundled vocab
        # 151643 < embed 151936, guard is actually satisfied, but make
        # the test robust to fixture changes.
        from peer.model_shard import ToyShardConfig
        cfg = ToyShardConfig(
            **{**minimal_mlx_config.__dict__, "runtime_tokenizer_vocab_guard": False}
        )
        rt = MLXRuntime(cfg)
        assert isinstance(rt._tokenizer, _FakeMlxTokenizer)

    def test_vocab_guard_rejects_oversized_tokenizer(
        self, fake_mlx_env, minimal_mlx_config,
    ):
        # Patch HF tokenizer to claim more tokens than embed_tokens has.
        class _TooBigTokenizer:
            vocab_size = 999999
        sys.modules["transformers"].AutoTokenizer.from_pretrained = (
            lambda name, **kwargs: _TooBigTokenizer()
        )
        sys.modules.pop("peer.mlx_runtime", None)
        from peer.mlx_runtime import MLXRuntime
        with pytest.raises(RuntimeError, match="tokenizer_vocab_mismatch"):
            MLXRuntime(minimal_mlx_config)

    def test_vocab_guard_tolerates_padded_embedding(
        self, fake_mlx_env, minimal_mlx_config,
    ):
        """Models pad their embedding to a round number > tokenizer.vocab_size.
        Guard must treat that as OK, not a mismatch."""
        # _FakeHfTokenizer.vocab_size == 151936, embed.weight.shape[0] == 151936.
        # Equal is fine; < is fine; only > should raise. Already covered by
        # default fixture — this test just ensures default path constructs
        # successfully.
        sys.modules.pop("peer.mlx_runtime", None)
        from peer.mlx_runtime import MLXRuntime
        rt = MLXRuntime(minimal_mlx_config)
        # If we got here, the guard accepted the load.
        assert rt._tokenizer.vocab_size <= 151936
