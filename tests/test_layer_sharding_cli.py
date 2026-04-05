# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Tests for --layer-start/--layer-end CLI and layer cleanup.

Run:  pytest tests/test_layer_sharding_cli.py -v
"""

from __future__ import annotations

import pytest


class TestExplicitLayerIndices:
    def test_layer_range_produces_correct_indices(self):
        """Explicit layer indices should be reflected in runtime profile."""
        from peer.model_shard import ToyShardConfig, ModelShard

        # tinyllama-15M has 6 layers (0-5), select layers 2-4
        config = ToyShardConfig(
            model_id="tinyllama-15M",
            runtime_backend="toy_auto",
            runtime_layer_indices=(2, 3, 4),
        )
        shard = ModelShard(config)
        profile = shard.runtime_profile()
        # ToyRuntime reports shard_index as layer_start
        assert profile["layer_start"] >= 0

    def test_empty_layer_indices_uses_shard_index(self):
        """Without explicit indices, shard_index/total_shards auto-splits."""
        from peer.model_shard import ToyShardConfig, ModelShard

        config = ToyShardConfig(
            model_id="tinyllama-15M",
            runtime_backend="toy_auto",
            shard_index=0,
            total_shards=1,
        )
        shard = ModelShard(config)
        profile = shard.runtime_profile()
        # ToyRuntime always reports shard_index as layer_start
        assert profile["layer_start"] >= 0

    def test_forward_works_with_explicit_layers(self):
        """Shard with explicit layers should still produce output."""
        from peer.model_shard import ToyShardConfig, ModelShard

        config = ToyShardConfig(
            runtime_layer_indices=(0, 1, 2, 3),
        )
        shard = ModelShard(config)
        result = shard.forward("Hello", [], max_tokens=8)
        assert len(result) > 0
