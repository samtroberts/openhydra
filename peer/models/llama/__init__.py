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

"""Llama model-family constants for KV cache compaction.

GQA configuration (Llama-3.1-8B-Instruct):
    n_query_heads  = 32
    n_kv_heads     = 8
    n_groups (GQA) = 4

The β expansion in _beta_inject.py repeats each kv-head β tensor 4 times to
align with the 32 query heads (``repeat_interleave(n_groups, dim=0)``).
"""

LLAMA3_8B_N_LAYERS: int = 32
LLAMA3_8B_N_KV_HEADS: int = 8
LLAMA3_8B_N_QUERY_HEADS: int = 32
LLAMA3_8B_HEAD_DIM: int = 128
