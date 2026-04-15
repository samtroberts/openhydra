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
from openhydra_defaults import PRODUCTION_BOOTSTRAP_URLS, PRODUCTION_LIBP2P_BOOTSTRAP_PEERS
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
        "--peer-id", default=None,
        help="Unique identifier for this node (default: auto-generated from hostname).",
    )
    parser.add_argument("--model-id", default="openhydra-qwen3.5-2b",
                        help="Model ID this node serves (default: openhydra-qwen3.5-2b).")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Shard index within the inference pipeline (default: 0).")
    parser.add_argument("--total-shards", type=int, default=1,
                        help="Total pipeline shards (default: 1).")
    parser.add_argument("--layer-start", type=int, default=None,
                        help="First transformer layer (inclusive). Overrides shard-index-based auto-split.")
    parser.add_argument("--layer-end", type=int, default=None,
                        help="One past the last transformer layer (exclusive). Use with --layer-start.")
    parser.add_argument("--runtime-model-id", default=None,
                        help="HuggingFace model ID or local path for the runtime (overrides --model-id for weight loading).")

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

    # --- P2P (Rust libp2p) ---
    parser.add_argument(
        "--p2p-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable the Rust libp2p networking layer for Kademlia DHT, "
            "Circuit Relay v2, DCUtR hole-punching, AutoNAT, and mDNS. "
            "Requires `pip install openhydra-network`."
        ),
    )
    parser.add_argument(
        "--p2p-listen", action="append", default=None, metavar="MULTIADDR",
        help=(
            "libp2p listen multiaddr. Repeat for multiple. "
            "Default: /ip4/0.0.0.0/tcp/4001"
        ),
    )
    parser.add_argument(
        "--p2p-bootstrap", action="append", default=None, metavar="MULTIADDR",
        help=(
            "libp2p bootstrap peer multiaddr (with /p2p/ suffix). "
            "Repeat for multiple."
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
    parser.add_argument("--specpipe", action="store_true", default=False,
                        help="Enable SpecPipe pipeline-filling speculation (P1-A).")
    parser.add_argument(
        "--autoregressive-sharded",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the outer autoregressive decode loop for non-streaming sharded PyTorch "
             "pipelines (default: True). Set --no-autoregressive-sharded to restore the legacy "
             "single-chain-call behavior while debugging.",
    )
    parser.add_argument("--chunked-prefill", action="store_true", default=False,
                        help="Enable chunked prefill for long prompts (P1-B).")
    parser.add_argument("--push-mode", action="store_true", default=False,
                        help="Enable server-to-server push mode (Petals parity Phase A).")
    parser.add_argument("--push-callback-address", default="",
                        help="Host:port where PushResult RPC arrives (usually this node's gRPC address).")
    parser.add_argument("--rebalance-enabled", action="store_true", default=False,
                        help="Enable autonomous dynamic rebalancing (peers decide their own layers).")
    parser.add_argument("--rebalance-interval", type=int, default=6,
                        help="Check rebalance every N announce cycles (default 6 = ~60s).")
    parser.add_argument("--rebalance-min-improvement", type=float, default=1.15,
                        help="Minimum improvement ratio to trigger rebalance (default 1.15 = 15%%).")
    parser.add_argument("--rebalance-cooldown", type=int, default=300,
                        help="Seconds to wait after rebalance before checking again (default 300).")

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

    # Auto-generate peer-id from hostname if not provided.
    if not args.peer_id:
        import socket
        _hostname = socket.gethostname().lower().replace(" ", "-").replace(".", "-")
        args.peer_id = f"{_hostname}-peer"
        logger.info("auto_peer_id: %s", args.peer_id)

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

    # On macOS MLX, auto-upgrade to pre-quantized 4-bit checkpoints.
    # Pre-quantized models load as QuantizedLinear layers — no runtime
    # quantization overhead, ~4x memory savings, ~2x faster generation.
    _MLX_4BIT_MAP = {
        "Qwen/Qwen3.5-0.8B": "mlx-community/Qwen3.5-0.8B-4bit",
    }
    _runtime_model_id = args.runtime_model_id or args.model_id
    if args.runtime_backend == "mlx" and _runtime_model_id in _MLX_4BIT_MAP:
        _runtime_model_id = _MLX_4BIT_MAP[_runtime_model_id]
        logger.info("mlx_4bit_upgrade: %s -> %s", args.model_id, _runtime_model_id)

    # Compute explicit layer indices from --layer-start/--layer-end
    _explicit_layer_indices: tuple[int, ...] = ()
    if args.layer_start is not None and args.layer_end is not None:
        _explicit_layer_indices = tuple(range(args.layer_start, args.layer_end))
        logger.info(
            "explicit_layer_range: [%d, %d) = %d layers",
            args.layer_start, args.layer_end, len(_explicit_layer_indices),
        )

    logger.info(
        "openhydra_node_starting peer_id=%s model=%s grpc_port=%d api=%s:%d dht=%s backend=%s",
        args.peer_id, args.model_id, args.grpc_port,
        args.api_host, args.api_port, dht_urls, args.runtime_backend,
    )

    # ── P2P Node (Rust libp2p) ──
    _p2p_node = None
    if getattr(args, "p2p_enabled", False):
        try:
            from openhydra_network import P2PNode
            _p2p_listen = getattr(args, "p2p_listen", None) or ["/ip4/0.0.0.0/tcp/4001"]
            _p2p_bootstrap = getattr(args, "p2p_bootstrap", None) or list(PRODUCTION_LIBP2P_BOOTSTRAP_PEERS)
            _p2p_node = P2PNode(
                identity_key_path=args.identity_path,
                listen_addrs=_p2p_listen,
                bootstrap_peers=_p2p_bootstrap,
            )
            _p2p_node.start()
            logger.info(
                "p2p_node_started libp2p_peer_id=%s openhydra_peer_id=%s listen=%s",
                _p2p_node.libp2p_peer_id,
                _p2p_node.openhydra_peer_id,
                _p2p_listen,
            )
        except ImportError:
            logger.warning(
                "p2p_enabled but openhydra_network not installed — "
                "run: cd network && maturin build --release && pip install target/wheels/*.whl"
            )
        except Exception as _p2p_err:
            logger.warning("p2p_node_start_failed: %s", _p2p_err)

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
            "runtime_model_id": _runtime_model_id,
            "shard_index": args.shard_index,
            "total_shards": args.total_shards,
            "expert_layer_indices": list(_explicit_layer_indices),
            "dht_urls": dht_urls,
            "daemon_mode": args.daemon_mode,
            "runtime_backend": args.runtime_backend,
            "advertise_host": args.advertise_host,
            "identity_path": args.identity_path,
            "mlx_eval_timeout_s": args.mlx_eval_timeout,
            "rebalance_enabled": bool(getattr(args, "rebalance_enabled", False)),
            "rebalance_interval": int(getattr(args, "rebalance_interval", 6)),
            "rebalance_min_improvement": float(getattr(args, "rebalance_min_improvement", 1.15)),
            "rebalance_cooldown_s": float(getattr(args, "rebalance_cooldown", 300)),
            "p2p_node": _p2p_node,
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
        # Add layer range to the local peer so the coordinator's
        # LayerCoverageMap sees it as a sharded peer from the first
        # request (before DHT propagation). Without this, the local
        # peer appears unsharded → coordinator falls back to full_model
        # mode even when a remote sharded peer is discovered via DHT.
        if _explicit_layer_indices:
            _local_peer[0]["layer_start"] = int(_explicit_layer_indices[0])
            _local_peer[0]["layer_end"] = int(_explicit_layer_indices[-1]) + 1
            # total_layers must be the FULL model depth (e.g. 24 for
            # Qwen3.5-2B), NOT this shard's layer_end. Otherwise the
            # coordinator thinks a 12-layer shard covers a "12-layer model"
            # and uses full_model mode instead of assembling a sharded
            # pipeline. Compute from total_shards × shard_size.
            _shard_size = len(_explicit_layer_indices)
            _total_layers = int(_shard_size * max(1, int(args.total_shards)))
            _local_peer[0]["total_layers"] = _total_layers
            _local_peer[0]["runtime_model_id"] = str(_runtime_model_id)
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

    # Auto-detect LAN IP for push mode callback (last peer sends result here).
    _push_callback_addr = f"127.0.0.1:{args.grpc_port}"
    try:
        import socket as _sock
        _s = _sock.socket(_sock.AF_INET, _sock.SOCK_DGRAM)
        _s.connect(("8.8.8.8", 80))
        _lan_ip = _s.getsockname()[0]
        _s.close()
        if _lan_ip and not _lan_ip.startswith("127."):
            _push_callback_addr = f"{_lan_ip}:{args.grpc_port}"
    except Exception:
        pass
    logger.info("push_callback_address=%s", _push_callback_addr)

    # Build the coordinator engine config with only the fields we override;
    # EngineConfig is a frozen dataclass and all other fields carry their defaults.
    engine_config = EngineConfig(
        dht_urls=dht_urls,
        deployment_profile=args.deployment_profile,
        peers_config_path=peers_config_path,
        model_catalog_path=_catalog_path,
        required_replicas=1,
        pipeline_width=2,
        timeout_ms=60000,
        max_latency_ms=600000,  # 10 min — stateless sharded decode is slow per token
        audit_rate=0.0,
        redundant_exec_rate=0.0,
        auditor_rate=0.0,
        # Grounding disabled by default: the fallback path injects dummy
        # "Context about {word}" snippets that waste tokens + confuse the
        # model. Users can still enable per-request via "grounding": true.
        grounding_fallback_enabled=False,
        grounding_use_network=False,
        specpipe_enabled=bool(getattr(args, "specpipe", False)),
        autoregressive_sharded_enabled=bool(getattr(args, "autoregressive_sharded", True)),
        chunked_prefill_enabled=bool(getattr(args, "chunked_prefill", False)),
        push_mode_enabled=True,  # peer-to-peer forwarding (skip coordinator round-trip)
        push_callback_address=str(getattr(args, "push_callback_address", "") or "")
            or _push_callback_addr,
    )

    # Start the coordinator HTTP API on the main thread (blocking).
    # Signal handling (SIGTERM, SIGINT) is already wired inside coordinator_serve.
    coordinator_serve(
        host=args.api_host,
        port=args.api_port,
        config=engine_config,
        api_key=api_key,
        p2p_node=_p2p_node,
    )


if __name__ == "__main__":
    main()
