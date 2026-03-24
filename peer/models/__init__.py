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
