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

"""
openhydra-node — unified node daemon.

Starts a peer gRPC server (background daemon thread) and a local
coordinator HTTP API server (main thread) from a single concise CLI.

Every participant runs their own local node — there is no shared central
coordinator. The HTTP API defaults to 127.0.0.1 (local-only); pass
--api-host 0.0.0.0 for testnet or Docker deployments.

Architecture
------------
                   openhydra-node process
  ┌────────────────────────────────────────────────────┐
  │  peer thread (daemon)      main thread             │
  │  ┌──────────────────┐      ┌──────────────────┐    │
  │  │ peer.server      │      │ coordinator      │    │
  │  │ gRPC :50051      │      │ HTTP API :8080   │    │
  │  └────────┬─────────┘      └────────┬─────────┘    │
  └───────────┼────────────────────────┼───────────────┘
              │ announces               │ queries
              └──────────► DHT ◄───────┘

Quickstart
----------
    openhydra-node --peer-id my-node

    # Local dev with explicit DHT:
    openhydra-dht --host 127.0.0.1 --port 8468 &
    openhydra-node --peer-id my-node \\
        --dht-url http://127.0.0.1:8468 \\
        --api-host 0.0.0.0
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import threading
import time

import peer.server as peer_server
from coordinator.api_server import serve as coordinator_serve
from coordinator.engine import EngineConfig
from openhydra_defaults import PRODUCTION_BOOTSTRAP_URLS
from openhydra_logging import configure_logging

logger = logging.getLogger(__name__)


def _parse_dht_urls(raw: list[str] | None) -> list[str]:
    """Flatten comma-separated and repeated --dht-url values, preserving order."""
    out: list[str] = []
    seen: set[str] = set()
    for item in list(raw or []):
        for token in str(item).split(","):
            value = token.strip()
            if value and value not in seen:
                seen.add(value)
                out.append(value)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="openhydra-node",
        description=(
            "OpenHydra unified node daemon. "
            "Runs a peer (gRPC) and a local coordinator (HTTP API) together. "
            "Every participant runs their own node — there is no shared central coordinator."
        ),
    )

    # --- Identity / model ---
    parser.add_argument(
        "--peer-id", required=True,
        help="Unique identifier for this node (required).",
    )
    parser.add_argument("--model-id", default="openhydra-qwen3.5-0.8b",
                        help="Model ID this node serves (default: openhydra-qwen3.5-0.8b).")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Shard index within the inference pipeline (default: 0).")
    parser.add_argument("--total-shards", type=int, default=1,
                        help="Total pipeline shards (default: 1).")

    # --- Network ---
    parser.add_argument("--grpc-port", type=int, default=50051,
                        help="Port for the peer gRPC server (default: 50051).")
    parser.add_argument("--api-port", type=int, default=8080,
                        help="Port for the local coordinator HTTP API (default: 8080).")
    parser.add_argument(
        "--api-host", default="127.0.0.1",
        help=(
            "Bind address for the coordinator HTTP API. "
            "Default 127.0.0.1 (local-only). "
            "Use 0.0.0.0 for Docker/testnet deployments."
        ),
    )
    parser.add_argument(
        "--advertise-host", default=None,
        help=(
            "Hostname advertised to DHT for inbound gRPC connections from other nodes. "
            "Required in Docker when the container hostname differs from the routable name. "
            "Default: auto-detected by the peer."
        ),
    )
    parser.add_argument(
        "--dht-url", action="append", default=None, metavar="URL",
        help=(
            "DHT bootstrap URL. Repeat or comma-separate for multiple. "
            "Defaults to the production OpenHydra bootstrap nodes when omitted."
        ),
    )

    # --- Runtime ---
    parser.add_argument("--runtime-backend", default="auto",
                        help="Inference runtime backend (default: auto — detects mlx on Mac, pytorch_auto on CUDA/CPU).")
    parser.add_argument("--daemon-mode", default="polite",
                        choices=["polite", "aggressive", "passive"],
                        help="Peer resource policy (default: polite).")
    parser.add_argument("--deployment-profile", default="dev",
                        choices=["dev", "prod"],
                        help="Deployment profile — dev or prod (default: dev).")

    # --- MLX parallelism ---
    parser.add_argument("--mlx-world-size", type=int, default=1,
                        help="Number of MLX devices for pipeline parallelism (default: 1 = single device).")
    parser.add_argument("--mlx-rank", type=int, default=0,
                        help="This device's rank in the MLX pipeline (0-indexed, default: 0).")
    parser.add_argument("--mlx-eval-timeout", type=float, default=120.0,
                        help="Maximum seconds for a single MLX computation before the watchdog "
                             "kills it (default: 120). Increase on memory-constrained machines.")

    # --- Auth ---
    parser.add_argument("--api-key", default=None,
                        help="Bearer token for the HTTP API. Also read from OPENHYDRA_API_KEY.")

    # --- Observability ---
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # --- Static peer discovery ---
    parser.add_argument("--peers-config", default=None, metavar="PATH",
                        help="Path to peers.local.json for static peer discovery (optional).")
    parser.add_argument(
        "--identity-path",
        default=".openhydra/identity.key",
        help="Path to Ed25519 identity keypair file (created on first run, mode 0600).",
    )

    args = parser.parse_args()

    # Logging must be configured before any other module emits records.
    configure_logging(
        level=args.log_level,
        json_logs=(args.deployment_profile == "prod"),
    )

    # Resolve DHT URLs; fall back to production bootstrap nodes if none given.
    dht_urls: list[str] = _parse_dht_urls(args.dht_url) or list(PRODUCTION_BOOTSTRAP_URLS)

    # Resolve API key: explicit flag > env var > None (auth disabled).
    api_key: str | None = (
        str(args.api_key).strip() if args.api_key
        else (os.environ.get("OPENHYDRA_API_KEY", "").strip() or None)
    )

    # Auto-detect the best runtime backend for this machine.
    # Supported platforms: Apple Silicon (MLX), NVIDIA CUDA, AMD ROCm.
    if args.runtime_backend == "auto":
        import platform
        if platform.system() == "Darwin":
            args.runtime_backend = "mlx"
            logger.info("auto_backend: detected macOS — using MLX (Metal acceleration)")
        else:
            try:
                import torch
                _is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
                if _is_rocm and torch.cuda.is_available():
                    args.runtime_backend = "pytorch_auto"
                    _gpu_name = torch.cuda.get_device_name(0)
                    logger.info("auto_backend: detected AMD ROCm GPU (%s) — using pytorch_auto", _gpu_name)
                elif torch.cuda.is_available():
                    args.runtime_backend = "pytorch_auto"
                    _gpu_name = torch.cuda.get_device_name(0)
                    logger.info("auto_backend: detected NVIDIA CUDA GPU (%s) — using pytorch_auto", _gpu_name)
                else:
                    args.runtime_backend = "pytorch_auto"
                    logger.warning(
                        "auto_backend: no supported GPU detected. "
                        "OpenHydra is designed for Apple Silicon (MLX), NVIDIA CUDA, or AMD ROCm GPUs. "
                        "Falling back to pytorch_auto (CPU) — expect significantly slower inference."
                    )
            except ImportError:
                logger.error(
                    "auto_backend: PyTorch is not installed. "
                    "Install PyTorch with GPU support: https://pytorch.org/get-started/locally/ "
                    "(NVIDIA: pip install torch, AMD ROCm: pip install torch --index-url https://download.pytorch.org/whl/rocm6.2)"
                )
                raise SystemExit(1)

    logger.info(
        "openhydra_node_starting peer_id=%s model=%s grpc_port=%d api=%s:%d dht=%s backend=%s",
        args.peer_id, args.model_id, args.grpc_port,
        args.api_host, args.api_port, dht_urls, args.runtime_backend,
    )

    # Start the peer gRPC server in a background daemon thread.
    # daemon=True ensures it is reaped automatically when the main thread exits
    # (the coordinator's SIGTERM handler on the main thread handles clean shutdown).
    peer_thread = threading.Thread(
        target=peer_server.serve,
        kwargs={
            "host": "0.0.0.0",
            "port": args.grpc_port,
            "peer_id": args.peer_id,
            "model_id": args.model_id,
            "shard_index": args.shard_index,
            "total_shards": args.total_shards,
            "dht_urls": dht_urls,
            "daemon_mode": args.daemon_mode,
            "runtime_backend": args.runtime_backend,
            "advertise_host": args.advertise_host,
            "identity_path": args.identity_path,
            "mlx_eval_timeout_s": args.mlx_eval_timeout,
            # All other peer params use peer/server.py defaults.
        },
        name="openhydra-peer",
        daemon=True,
    )
    peer_thread.start()
    logger.info(
        "peer_thread_started grpc_port=%d; waiting 1s for gRPC bind", args.grpc_port,
    )
    time.sleep(1.0)

    # Auto-generate a local peers config so the coordinator can always find
    # its co-located peer — no --peers-config needed for single-node usage.
    peers_config_path = args.peers_config
    if not peers_config_path:
        _local_peer = [{
            "peer_id": args.peer_id,
            "host": "127.0.0.1",
            "port": args.grpc_port,
            "model_id": args.model_id,
            "operator_id": "local",
            "runtime_backend": args.runtime_backend,
        }]
        _tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="openhydra_peers_",
            delete=False,
        )
        json.dump(_local_peer, _tmp)
        _tmp.close()
        peers_config_path = _tmp.name
        logger.info("auto_peers_config: %s (local peer %s on :%d)",
                     peers_config_path, args.peer_id, args.grpc_port)

    # Resolve model catalog path — look for models.catalog.json in cwd,
    # then fall back to None (uses engine defaults).
    _catalog_path: str | None = "models.catalog.json"
    if not os.path.exists(_catalog_path):
        _catalog_path = None

    # Build the coordinator engine config with only the fields we override;
    # EngineConfig is a frozen dataclass and all other fields carry their defaults.
    engine_config = EngineConfig(
        dht_urls=dht_urls,
        deployment_profile=args.deployment_profile,
        peers_config_path=peers_config_path,
        model_catalog_path=_catalog_path,
        required_replicas=1,
        pipeline_width=1,
        timeout_ms=60000,
        max_latency_ms=60000,
    )

    # Start the coordinator HTTP API on the main thread (blocking).
    # Signal handling (SIGTERM, SIGINT) is already wired inside coordinator_serve.
    coordinator_serve(
        host=args.api_host,
        port=args.api_port,
        config=engine_config,
        api_key=api_key,
    )


if __name__ == "__main__":
    main()
