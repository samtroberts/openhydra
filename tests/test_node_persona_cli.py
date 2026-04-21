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

"""CLI validation tests for the Phase 1.5 ``--node-persona`` flag.

The validation block in ``coordinator/node.py`` runs immediately after
``argparse.parse_args()`` — before any model loading or P2P startup — so
these subprocess tests finish in well under a second.

We invoke ``python -m coordinator.node`` in a child process, pass the
offending argument combinations, and assert:
  * exit code 2 (``SystemExit(2)`` from the validation block)
  * a descriptive error on stderr or stdout (logger output may be routed
    to either; we check combined output)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_node_cli(*args: str, timeout_s: float = 10.0) -> subprocess.CompletedProcess:
    """Invoke coordinator.node with the given extra args. No network side-effects
    because we always pair the test with `--no-p2p-enabled` and an invalid config
    that fails validation before model load."""
    cmd = [sys.executable, "-m", "coordinator.node", *args]
    env = dict(os.environ)
    # Ensure the test repo root is first on PYTHONPATH so our local edits win.
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )


def _combined(proc: subprocess.CompletedProcess) -> str:
    return (proc.stdout or "") + "\n" + (proc.stderr or "")


# ─── Positive: a bare CLI parses without the validation error ────────────────
# We can't spin up a real node (no openhydra_network wheel + no model), but
# we *can* assert the persona-validation code path doesn't fire on legal args.


def test_help_flag_exits_zero_and_lists_persona():
    """--help should list the new flags and exit zero."""
    proc = _run_node_cli("--help", timeout_s=10.0)
    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    combined = _combined(proc)
    assert "--node-persona" in combined
    assert "--upstream-kind" in combined
    assert "--upstream-url" in combined
    assert "--hosted-model-ids" in combined


# ─── Negative: atomic_worker missing upstream-* flags ────────────────────────


def test_atomic_worker_without_upstream_flags_fails_fast():
    """--node-persona=atomic_worker with no upstream config exits 2 with
    a "requires" message that names the missing flags."""
    proc = _run_node_cli(
        "--node-persona", "atomic_worker",
        "--peer-id", "test",
        "--no-p2p-enabled",
    )
    assert proc.returncode == 2, f"stdout+stderr: {_combined(proc)}"
    combined = _combined(proc)
    assert "atomic_worker_cli_validation_failed" in combined
    assert "requires" in combined
    assert "--upstream-kind" in combined
    assert "--upstream-url" in combined
    assert "--hosted-model-ids" in combined


def test_atomic_worker_partial_upstream_flags_fails_fast():
    """Only --upstream-url is set — still missing kind + hosted."""
    proc = _run_node_cli(
        "--node-persona", "atomic_worker",
        "--upstream-url", "http://localhost:11434",
        "--peer-id", "test",
        "--no-p2p-enabled",
    )
    assert proc.returncode == 2, f"stdout+stderr: {_combined(proc)}"
    combined = _combined(proc)
    assert "--upstream-kind" in combined
    assert "--hosted-model-ids" in combined
    assert "--upstream-url" not in combined.split("requires")[1] \
        if "requires" in combined else True


# ─── Negative: atomic_worker + sharding flags conflict ───────────────────────


def test_atomic_worker_cannot_use_layer_start():
    """Atomic workers always serve layer 0→N — any non-zero --layer-start /
    --layer-end is forbidden.  (Explicit zero values are semantically identical
    to unset and are allowed, which is why we pass 4/12 here.)"""
    proc = _run_node_cli(
        "--node-persona", "atomic_worker",
        "--upstream-kind", "ollama",
        "--upstream-url", "http://localhost:11434",
        "--hosted-model-ids", "openhydra-qwen3.5-2b",
        "--layer-start", "4",
        "--layer-end", "12",
        "--peer-id", "test",
        "--no-p2p-enabled",
    )
    assert proc.returncode == 2, f"stdout+stderr: {_combined(proc)}"
    combined = _combined(proc)
    assert "atomic_worker_cli_validation_failed" in combined
    assert "cannot shard" in combined
    assert "--layer-start" in combined
    assert "--layer-end" in combined


def test_atomic_worker_cannot_use_total_shards():
    proc = _run_node_cli(
        "--node-persona", "atomic_worker",
        "--upstream-kind", "ollama",
        "--upstream-url", "http://localhost:11434",
        "--hosted-model-ids", "openhydra-qwen3.5-2b",
        "--total-shards", "2",
        "--peer-id", "test",
        "--no-p2p-enabled",
    )
    assert proc.returncode == 2, f"stdout+stderr: {_combined(proc)}"
    combined = _combined(proc)
    assert "cannot shard" in combined
    assert "--total-shards>1" in combined


# ─── Negative: native_shard with upstream-only flags ─────────────────────────


def test_native_shard_rejects_upstream_kind():
    """Default persona (native_shard) cannot accept atomic-worker-only flags."""
    proc = _run_node_cli(
        "--upstream-kind", "ollama",
        "--peer-id", "test",
        "--no-p2p-enabled",
    )
    assert proc.returncode == 2, f"stdout+stderr: {_combined(proc)}"
    combined = _combined(proc)
    assert "native_shard_cli_validation_failed" in combined
    assert "--upstream-kind" in combined


def test_native_shard_rejects_hosted_model_ids():
    proc = _run_node_cli(
        "--hosted-model-ids", "openhydra-qwen3.5-2b",
        "--peer-id", "test",
        "--no-p2p-enabled",
    )
    assert proc.returncode == 2, f"stdout+stderr: {_combined(proc)}"
    combined = _combined(proc)
    assert "native_shard_cli_validation_failed" in combined
    assert "--hosted-model-ids" in combined


def test_native_shard_rejects_upstream_url():
    proc = _run_node_cli(
        "--upstream-url", "http://localhost:11434",
        "--peer-id", "test",
        "--no-p2p-enabled",
    )
    assert proc.returncode == 2, f"stdout+stderr: {_combined(proc)}"
    combined = _combined(proc)
    assert "native_shard_cli_validation_failed" in combined
    assert "--upstream-url" in combined


# ─── Negative: bad --node-persona value ──────────────────────────────────────


def test_invalid_node_persona_value_rejected_by_argparse():
    """argparse's `choices=[...]` rejects typos with exit code 2."""
    proc = _run_node_cli(
        "--node-persona", "potato_worker",
        "--peer-id", "test",
        "--no-p2p-enabled",
    )
    assert proc.returncode == 2
    combined = _combined(proc)
    assert "invalid choice" in combined.lower() or "potato_worker" in combined


def test_invalid_upstream_kind_value_rejected_by_argparse():
    proc = _run_node_cli(
        "--node-persona", "atomic_worker",
        "--upstream-kind", "bogus_runtime",
        "--upstream-url", "http://localhost:11434",
        "--hosted-model-ids", "openhydra-qwen3.5-2b",
        "--peer-id", "test",
        "--no-p2p-enabled",
    )
    assert proc.returncode == 2
    combined = _combined(proc)
    assert "invalid choice" in combined.lower() or "bogus_runtime" in combined
