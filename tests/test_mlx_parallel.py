"""Tests for peer/mlx_parallel.py — Phase 4A + 4B.

All tests mock mx.distributed to avoid requiring a real Apple Metal cluster.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from peer.mlx_parallel import PipelineParallelMLX, assign_layers


def _mock_model(num_layers: int = 32, layer_fn=None) -> MagicMock:
    """Create a mock MLX model with ``model.model.layers``."""
    layers = []
    for i in range(num_layers):
        layer = MagicMock(name=f"layer_{i}")
        if layer_fn is not None:
            layer.side_effect = layer_fn
        layers.append(layer)
    inner = MagicMock()
    inner.layers = layers
    model = MagicMock()
    model.model = inner
    return model


# ── assign_layers math ───────────────────────────────────────────────────────


class TestAssignLayers:
    def test_even_split(self):
        assert assign_layers(32, 4) == [(0, 8), (8, 16), (16, 24), (24, 32)]

    def test_odd_layers_3_ranks(self):
        assert assign_layers(7, 3) == [(0, 3), (3, 5), (5, 7)]

    def test_odd_layers_2_ranks(self):
        assert assign_layers(5, 2) == [(0, 3), (3, 5)]

    def test_single_rank(self):
        assert assign_layers(32, 1) == [(0, 32)]

    def test_more_ranks_than_layers(self):
        result = assign_layers(5, 8)
        assert len(result) == 8
        for i in range(5):
            assert result[i][1] - result[i][0] == 1
        for i in range(5, 8):
            assert result[i][1] - result[i][0] == 0

    def test_full_coverage(self):
        for total in [1, 7, 13, 32, 48, 64]:
            for ws in [1, 2, 3, 4, 5, 7, 8]:
                result = assign_layers(total, ws)
                assert len(result) == ws
                assert result[0][0] == 0
                assert result[-1][1] == total
                for i in range(len(result) - 1):
                    assert result[i][1] == result[i + 1][0]

    def test_invalid_world_size(self):
        with pytest.raises(ValueError, match="world_size"):
            assign_layers(32, 0)

    def test_invalid_total_layers(self):
        with pytest.raises(ValueError, match="total_layers"):
            assign_layers(0, 4)

    def test_one_layer_one_rank(self):
        assert assign_layers(1, 1) == [(0, 1)]

    def test_equal_distribution_12_layers_4_ranks(self):
        assert assign_layers(12, 4) == [(0, 3), (3, 6), (6, 9), (9, 12)]

    def test_large_world_size(self):
        result = assign_layers(128, 64)
        assert len(result) == 64
        assert result[0][0] == 0
        assert result[-1][1] == 128
        assert all(result[i][1] == result[i + 1][0] for i in range(63))


# ── PipelineParallelMLX construction ─────────────────────────────────────────


class TestPipelineParallelMLXInit:
    def test_single_device_gets_all_layers(self):
        pp = PipelineParallelMLX(_mock_model(32), world_size=1, rank=0)
        assert pp.layer_start == 0
        assert pp.layer_end == 32
        assert pp.num_local_layers == 32
        assert pp.is_first is True
        assert pp.is_last is True

    def test_four_ranks_even_split(self):
        model = _mock_model(32)
        for rank in range(4):
            pp = PipelineParallelMLX(model, world_size=4, rank=rank)
            assert pp.num_local_layers == 8
            assert pp.layer_start == rank * 8
            assert pp.layer_end == (rank + 1) * 8
            assert pp.is_first == (rank == 0)
            assert pp.is_last == (rank == 3)

    def test_rank_clamped_to_valid_range(self):
        pp = PipelineParallelMLX(_mock_model(32), world_size=4, rank=99)
        assert pp.rank == 3

    def test_layer_range_property(self):
        pp = PipelineParallelMLX(_mock_model(32), world_size=2, rank=1)
        assert pp.layer_range == (16, 32)


# ── Forward (world_size=1) ───────────────────────────────────────────────────


class TestForwardSingleDevice:
    def test_forward_cpu_processes_all_layers(self):
        model = _mock_model(4, layer_fn=lambda h, mask=None: h + 1)
        pp = PipelineParallelMLX(model, world_size=1, rank=0)
        result = pp._forward_layers_cpu(0)
        assert result == 4

    def test_forward_cpu_with_mask(self):
        model = _mock_model(2, layer_fn=lambda h, mask=None: h + 1)
        pp = PipelineParallelMLX(model, world_size=1, rank=0)
        result = pp._forward_layers_cpu(0, mask="test_mask")
        assert result == 2


# ── Overlapped async_eval (Phase 4B) ────────────────────────────────────────


class TestAsyncEvalOverlap:
    def test_async_eval_default_true(self):
        pp = PipelineParallelMLX(_mock_model(0), world_size=1, rank=0)
        assert pp.async_eval is True

    def test_async_eval_can_be_disabled(self):
        pp = PipelineParallelMLX(_mock_model(0), world_size=1, rank=0, async_eval=False)
        assert pp.async_eval is False

    def test_async_eval_flag_preserved(self):
        pp = PipelineParallelMLX(_mock_model(4), world_size=1, rank=0, async_eval=True)
        assert pp.async_eval is True


# ── Distributed communication roles ─────────────────────────────────────────


class TestDistributedRoles:
    def test_first_rank(self):
        pp = PipelineParallelMLX(_mock_model(4), world_size=2, rank=0)
        assert pp.is_first is True
        assert pp.is_last is False

    def test_last_rank(self):
        pp = PipelineParallelMLX(_mock_model(4), world_size=2, rank=1)
        assert pp.is_first is False
        assert pp.is_last is True

    def test_middle_rank(self):
        pp = PipelineParallelMLX(_mock_model(6), world_size=3, rank=1)
        assert pp.is_first is False
        assert pp.is_last is False


# ── world_size=1 no overhead ─────────────────────────────────────────────────


class TestWorldSizeOneNoOverhead:
    def test_single_device_full_layers(self):
        pp = PipelineParallelMLX(_mock_model(32), world_size=1, rank=0)
        assert pp.world_size == 1
        assert pp.num_local_layers == 32

    def test_cpu_fallback_single_device(self):
        model = _mock_model(1, layer_fn=lambda h, mask=None: h + 10)
        pp = PipelineParallelMLX(model, world_size=1, rank=0)
        assert pp._forward_layers_cpu(0) == 10


# ── Edge cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_model_no_crash(self):
        pp = PipelineParallelMLX(_mock_model(0), world_size=1, rank=0)
        assert pp.total_layers == 0
        assert pp.num_local_layers == 0

    def test_find_transformer_blocks_model_model_layers(self):
        model = _mock_model(3)
        blocks = PipelineParallelMLX._find_transformer_blocks(model)
        assert len(blocks) == 3

    def test_find_transformer_blocks_model_layers(self):
        model = MagicMock(spec=[])
        model.layers = [1, 2]
        assert len(PipelineParallelMLX._find_transformer_blocks(model)) == 2

    def test_find_transformer_blocks_empty(self):
        model = MagicMock(spec=[])
        assert PipelineParallelMLX._find_transformer_blocks(model) == []
