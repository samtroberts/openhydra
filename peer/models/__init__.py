"""peer.models — patched HuggingFace model attention layers for β injection.

Phase 2 of KV cache compaction requires each attention layer to add per-token
scalar biases (β) to the attention logits before softmax.  This is achieved
at runtime via monkey-patching rather than static file replacement, so it is
compatible with any version of transformers that exposes the standard
``model.model.layers[i].self_attn.forward`` interface.

The actual patching logic lives in ``peer.kv_compaction._beta_inject``.
This package exists to document the model-specific constants and GQA ratios
used during patching.
"""
