"""Global pytest configuration.

When OPENHYDRA_USE_REAL_MODEL=1 is set:
  1. Auto-enables OPENHYDRA_RUN_REAL_TENSOR_TEST=1 (un-skips test_real_*.py tests)
  2. Provides a session-scoped ``real_mlx_runtime`` fixture (loads model ONCE per batch)
"""
from __future__ import annotations

import os

import pytest

_USE_REAL = os.environ.get("OPENHYDRA_USE_REAL_MODEL", "0") == "1"

if _USE_REAL:
    os.environ["OPENHYDRA_RUN_REAL_TENSOR_TEST"] = "1"


@pytest.fixture(scope="session")
def real_mlx_runtime():
    """Session-scoped MLXRuntime with Qwen/Qwen3.5-0.8B — loads once per batch."""
    pytest.importorskip("mlx.core")
    from peer.mlx_runtime import MLXRuntime
    from peer.model_shard import ToyShardConfig

    rt = MLXRuntime(
        ToyShardConfig(
            runtime_model_id="Qwen/Qwen3.5-0.8B",
            runtime_warmup_on_start=True,
        )
    )
    yield rt
    # Process exit handles cleanup (Metal frees memory)
