import sys
import types

import pytest

from peer.model_shard import (
    ModelShard,
    PyTorchRuntime,
    ToyShardConfig,
    _default_trust_remote_code,
    _tokenizer_eos_ids,
)


def test_model_shard_is_deterministic():
    shard = ModelShard(ToyShardConfig(model_id="x", shard_index=0, total_shards=3))
    a1 = shard.forward("hello hydra", [], 16)
    a2 = shard.forward("hello hydra", [], 16)
    assert a1 == a2
    tokens = ModelShard.decode_tokens(a1, 12)
    assert len(tokens) == 12
    text = ModelShard.decode_text(a1, 12)
    assert text == ModelShard.render_text(tokens)
    assert text.endswith(".")
    assert len(text) > 5


def test_model_shard_reports_runtime_profile_and_quantization(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    int4_shard = ModelShard(
        ToyShardConfig(
            model_id="x",
            shard_index=0,
            total_shards=1,
            runtime_backend="toy_auto",
            runtime_target="auto",
            quantization_mode="int4",
        )
    )
    fp32_shard = ModelShard(
        ToyShardConfig(
            model_id="x",
            shard_index=0,
            total_shards=1,
            runtime_backend="toy_auto",
            runtime_target="auto",
            quantization_mode="fp32",
        )
    )

    int4_profile = int4_shard.runtime_profile()
    fp32_profile = fp32_shard.runtime_profile()
    assert int4_profile["backend"] == "toy_gpu_sim"
    assert int4_profile["target"] == "cuda"
    assert int4_profile["quantization_mode"] == "int4"
    assert int4_profile["quantization_bits"] == 4
    assert int4_profile["gpu_available"] is True
    assert int4_profile["estimated_memory_mb"] < fp32_profile["estimated_memory_mb"]

    activation = int4_shard.forward("runtime profile check", [], 8)
    assert activation
    assert all(-1.0 <= value <= 1.0 for value in activation)


def test_detect_decoder_architecture_gpt_style():
    class _Transformer:
        def __init__(self):
            self.h = [object(), object()]
            self.wte = object()
            self.wpe = object()
            self.ln_f = object()

    class _Model:
        def __init__(self):
            self.transformer = _Transformer()

    arch = PyTorchRuntime._detect_decoder_architecture(_Model())
    assert arch.family == "gpt"
    assert len(arch.layers) == 2
    assert arch.position_embeddings is not None


def test_detect_decoder_architecture_llama_style():
    class _LlamaCore:
        def __init__(self):
            self.layers = [object(), object(), object()]
            self.embed_tokens = object()
            self.norm = object()
            self.rotary_emb = object()

    class _Model:
        def __init__(self):
            self.model = _LlamaCore()

    arch = PyTorchRuntime._detect_decoder_architecture(_Model())
    assert arch.family == "llama"
    assert len(arch.layers) == 3
    assert arch.rotary_emb is not None
    assert arch.position_embeddings is None


def test_detect_decoder_architecture_rejects_unknown_model():
    class _Model:
        pass

    with pytest.raises(RuntimeError, match="unsupported_model_architecture"):
        PyTorchRuntime._detect_decoder_architecture(_Model())


def test_default_trust_remote_code_policy():
    assert _default_trust_remote_code("Qwen/Qwen3.5-0.8B") is True
    assert _default_trust_remote_code("Qwen2.5-7B-Instruct") is True
    assert _default_trust_remote_code("gpt2") is False
    assert _default_trust_remote_code("EleutherAI/gpt-neo-125m") is False


def test_tokenizer_eos_ids_supports_int_and_list():
    class _IntTokenizer:
        eos_token_id = 42

    class _ListTokenizer:
        eos_token_id = [99, 7, 99]

    int_ids, int_primary = _tokenizer_eos_ids(_IntTokenizer())
    list_ids, list_primary = _tokenizer_eos_ids(_ListTokenizer())
    assert int_ids == {42}
    assert int_primary == 42
    assert list_ids == {7, 99}
    assert list_primary == 7


def test_load_decode_tokenizer_uses_trust_remote_code_policy(monkeypatch):
    class _AutoTokenizer:
        calls: list[tuple[str, dict[str, object]]] = []

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            cls.calls.append((str(model_id), dict(kwargs)))
            return object()

    fake_module = types.SimpleNamespace(AutoTokenizer=_AutoTokenizer)
    monkeypatch.setitem(sys.modules, "transformers", fake_module)
    monkeypatch.setattr(ModelShard, "_openhydra_decode_tokenizer_cache", {}, raising=False)

    ModelShard._load_decode_tokenizer("Qwen/Qwen3.5-0.8B")
    ModelShard._load_decode_tokenizer("gpt2")

    qwen_call = next(call for call in _AutoTokenizer.calls if call[0] == "Qwen/Qwen3.5-0.8B")
    gpt_call = next(call for call in _AutoTokenizer.calls if call[0] == "gpt2")
    assert qwen_call[1]["trust_remote_code"] is True
    assert gpt_call[1]["trust_remote_code"] is False


def test_decode_text_uses_tokenizer_for_pytorch_token_ids(monkeypatch):
    class _Tokenizer:
        def decode(self, token_ids, clean_up_tokenization_spaces=False):
            token_id = int(token_ids[0])
            if token_id == 11751:
                return " Paris"
            return f" <{token_id}>"

    monkeypatch.setattr(
        ModelShard,
        "_load_decode_tokenizer",
        staticmethod(lambda model_id: _Tokenizer()),
    )

    text = ModelShard.decode_text(
        [11751.0],
        max_tokens=6,
        tokenizer_model_id="Qwen/Qwen3.5-0.8B",
    )
    assert text == "Paris"


def test_decode_text_filters_special_token_ids(monkeypatch):
    class _Tokenizer:
        all_special_ids = [151645]

        def decode(self, token_ids, clean_up_tokenization_spaces=False):
            token_id = int(token_ids[0])
            if token_id == 151645:
                return "<|im_end|>"
            return f" <{token_id}>"

    monkeypatch.setattr(
        ModelShard,
        "_load_decode_tokenizer",
        staticmethod(lambda model_id: _Tokenizer()),
    )

    text = ModelShard.decode_text(
        [151645.0],
        max_tokens=4,
        tokenizer_model_id="Qwen/Qwen3.5-0.8B",
    )
    assert text == ""
