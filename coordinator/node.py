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
    parser.add_argument("--hf-model-id", default=None,
                        help=(
                            "Canonical HuggingFace model id used for the tokenizer across every "
                            "backend. When unset, resolved from models.catalog.json for the given "
                            "--model-id. Forces MLX peers onto the HF tokenizer so heterogeneous "
                            "MLX ↔ PyTorch rings share one vocab."
                        ))
    parser.add_argument("--mlx-force-hf-tokenizer",
                        dest="mlx_force_hf_tokenizer",
                        action="store_true", default=True,
                        help="Force MLX runtime to use the canonical HF tokenizer (default: on).")
    parser.add_argument("--no-mlx-force-hf-tokenizer",
                        dest="mlx_force_hf_tokenizer", action="store_false",
                        help="Revert MLX runtime to the bundled mlx-community tokenizer.")
    parser.add_argument("--tokenizer-vocab-guard",
                        dest="tokenizer_vocab_guard",
                        action="store_true", default=True,
                        help="Fail startup if tokenizer vocab > embed_tokens size (default: on).")
    parser.add_argument("--no-tokenizer-vocab-guard",
                        dest="tokenizer_vocab_guard", action="store_false",
                        help="Disable the tokenizer/embedding vocab-size guard.")

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

    # --- Node persona (Phase 1.5 zero-config bootstrap) ---
    # A node is either a ``native_shard`` (tensor-passing libp2p peer — the
    # default) or an ``atomic_worker`` (wrapper around a third-party runtime
    # like Ollama/Exo that accepts full text prompts).  Atomic workers never
    # shard — they serve layer 0→N as one atomic unit.
    parser.add_argument(
        "--node-persona",
        choices=["native_shard", "atomic_worker"],
        default="native_shard",
        help=(
            "Node role. ``native_shard`` (default) is a tensor-passing libp2p peer. "
            "``atomic_worker`` is a wrapper around an external runtime (Ollama, Exo, "
            "llama.cpp, OpenAI-compatible) that handles full text prompts. "
            "Atomic workers require --upstream-kind / --upstream-url / --hosted-model-ids."
        ),
    )
    parser.add_argument(
        "--upstream-kind",
        choices=["", "ollama", "exo", "openai_compat", "llama_cpp"],
        default="",
        help=(
            "External runtime kind wrapped by this node. "
            "Only meaningful when --node-persona=atomic_worker."
        ),
    )
    parser.add_argument(
        "--upstream-url",
        default="",
        help=(
            "Base URL of the external runtime (e.g. http://localhost:11434 for Ollama). "
            "Only meaningful when --node-persona=atomic_worker."
        ),
    )
    parser.add_argument(
        "--hosted-model-ids",
        default="",
        help=(
            "Comma-separated OpenHydra model_ids served by this atomic_worker's "
            "upstream runtime (e.g. openhydra-qwen3.5-2b,openhydra-qwen3.5-9b). "
            "Declared at boot — no automatic probing of the upstream in Phase 1.5."
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
    parser.add_argument(
        "--sample-on-coordinator",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Client-terminated pipeline (Path A): last peer returns a hidden "
            "state; coordinator applies final_norm + lm_head + sampling and "
            "re-injects the token into stage 0. Eliminates the per-token "
            "ring-loopback hop. Requires the coordinator to be co-located "
            "with the last-shard peer (the common --p2p-enabled setup). "
            "Default off — reversible; opt in per run."
        ),
    )
    parser.add_argument(
        "--no-local-peer",
        action="store_true",
        default=False,
        help=(
            "Phase 6 (true Petals topology): run as a PURE coordinator "
            "with no local peer thread. Requires --sample-on-coordinator "
            "AND --runtime-model-id (the model id used to load the "
            "coordinator's standalone head weights). Use when the "
            "coordinator should orchestrate + sample but run no transformer "
            "layers — e.g. a Mac coordinator driving GPU peers in the swarm, "
            "or a cheap Linux coordinator sitting alongside GPU peers in a VPC."
        ),
    )
    parser.add_argument(
        "--standalone-head-backend",
        choices=("auto", "mlx", "pytorch"),
        default="auto",
        help=(
            "Backend for the pure-coordinator standalone head. "
            "'auto' (default) prefers MLX when importable (Apple Silicon), "
            "otherwise PyTorch. 'mlx' / 'pytorch' force a specific backend. "
            "Linux coordinators typically want 'pytorch'."
        ),
    )
    parser.add_argument(
        "--standalone-head-device",
        default="cpu",
        help=(
            "PyTorch backend only — device to place the head weights on. "
            "Default 'cpu' (sensible for a Linux coordinator without a GPU). "
            "Use 'cuda' / 'cuda:0' / 'mps' to place on accelerator if available."
        ),
    )
    parser.add_argument(
        "--standalone-head-dtype",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
        help=(
            "PyTorch backend only — weight dtype for the standalone head. "
            "Default 'float32' (safest for CPU). 'bfloat16' halves memory "
            "and is typically fine for matmul accuracy on modern CPUs."
        ),
    )
    parser.add_argument("--rebalance-enabled", action="store_true", default=False,
                        help="Enable autonomous dynamic rebalancing (peers decide their own layers).")
    parser.add_argument("--rebalance-interval", type=int, default=6,
                        help="Check rebalance every N announce cycles (default 6 = ~60s).")
    parser.add_argument("--rebalance-min-improvement", type=float, default=1.15,
                        help="Minimum improvement ratio to trigger rebalance (default 1.15 = 15%%).")
    parser.add_argument("--rebalance-cooldown", type=int, default=300,
                        help="Seconds to wait after rebalance before checking again (default 300).")

    # --- Phase 4: Continuous re-negotiation ---
    parser.add_argument(
        "--negotiation-interval-s",
        type=float, default=60.0,
        help=(
            "Continuous-re-negotiation tick cadence in seconds (default 60). "
            "Every tick the NegotiationLoop refreshes the CapacityReport and "
            "consults SwarmNegotiator; if the peer is idle and a better shard "
            "assignment is available, it re-claims.  Floor: 5s."
        ),
    )
    # --- Track B / B3: ReshardExecutor feature flag (default OFF) ---
    parser.add_argument(
        "--reshard-executor-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When the NegotiationLoop observes a changed shard assignment, "
            "run the B3 drain→unload→reload FSM to *actually* apply it to "
            "the running PeerService. Default OFF — failures leave the peer "
            "in a degraded LOADING_FAILED state (stay-degraded policy; no "
            "execv). Flip on once you're ready to bake the reload path "
            "against CUDA / MPS / Metal memory-leak surfaces."
        ),
    )
    parser.add_argument(
        "--reshard-drain-timeout-s",
        type=float, default=120.0,
        help=(
            "Max wall-clock the B3 FSM waits for in-flight requests to "
            "complete before proceeding with UNLOADING. Default 120s "
            "matches the master plan's 'do not penalise users for a "
            "network reshard event' policy."
        ),
    )

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

    # ── Phase 1.5: node persona CLI validation ─────────────────────────────
    # atomic_worker requires upstream config and cannot shard.
    # native_shard forbids upstream config and --hosted-model-ids.
    _node_persona = str(getattr(args, "node_persona", "native_shard") or "native_shard")
    _upstream_kind = str(getattr(args, "upstream_kind", "") or "")
    _upstream_url = str(getattr(args, "upstream_url", "") or "")
    _hosted_model_ids_raw = str(getattr(args, "hosted_model_ids", "") or "")
    _hosted_model_ids: list[str] = [
        m.strip() for m in _hosted_model_ids_raw.split(",") if m.strip()
    ]

    if _node_persona == "atomic_worker":
        _missing: list[str] = []
        if not _upstream_kind:
            _missing.append("--upstream-kind")
        if not _upstream_url:
            _missing.append("--upstream-url")
        if not _hosted_model_ids:
            _missing.append("--hosted-model-ids")
        if _missing:
            logger.error(
                "atomic_worker_cli_validation_failed: --node-persona=atomic_worker "
                "requires %s",
                ", ".join(_missing),
            )
            raise SystemExit(2)
        # Atomic workers cannot shard — they always serve layer 0→N as an
        # opaque text API.  Refuse any shard-related flag that would imply
        # partial coverage.
        _shard_violations: list[str] = []
        if int(getattr(args, "layer_start", 0) or 0) != 0:
            _shard_violations.append("--layer-start")
        if int(getattr(args, "layer_end", 0) or 0) != 0:
            _shard_violations.append("--layer-end")
        if int(getattr(args, "shard_index", 0) or 0) != 0:
            _shard_violations.append("--shard-index")
        if int(getattr(args, "total_shards", 1) or 1) > 1:
            _shard_violations.append("--total-shards>1")
        if _shard_violations:
            logger.error(
                "atomic_worker_cli_validation_failed: atomic workers cannot shard — "
                "remove %s (they always serve the entire model)",
                ", ".join(_shard_violations),
            )
            raise SystemExit(2)
    else:
        # native_shard default — reject upstream-only flags.
        _forbidden: list[str] = []
        if _upstream_kind:
            _forbidden.append("--upstream-kind")
        if _upstream_url:
            _forbidden.append("--upstream-url")
        if _hosted_model_ids:
            _forbidden.append("--hosted-model-ids")
        if _forbidden:
            logger.error(
                "native_shard_cli_validation_failed: --node-persona=native_shard (default) "
                "does not accept %s — pass --node-persona=atomic_worker to use them",
                ", ".join(_forbidden),
            )
            raise SystemExit(2)

    # ── Phase 2 ConfigResolver: headless bootstrap ───────────────────────
    # Resolve peer-id + ports in one shot.  Manual CLI flags take precedence:
    #   * ``--peer-id foo`` → used verbatim
    #   * ``--peer-id`` omitted → persistent peer-id from
    #     ``.openhydra/peers.local.json`` if present, else derived from the
    #     Ed25519 pubkey at ``.openhydra/bootstrap_identity.json``.
    #   * Ports: if the CLI value equals the documented default we probe and
    #     auto-increment on collision; a non-default CLI value is taken
    #     literally (the user meant that port).
    from peer.bootstrap_config import (
        DEFAULT_BOOTSTRAP_IDENTITY_PATH,
        DEFAULT_PEERS_LOCAL_PATH,
        PeersLocalConfig,
        resolve_bootstrap,
    )

    # First libp2p listen port — extract from --p2p-listen if present,
    # else default to 4001 (matches coordinator/node.py default).
    _p2p_default_port = 4001
    _p2p_listen_from_cli = args.p2p_listen or []
    if _p2p_listen_from_cli:
        _first = str(_p2p_listen_from_cli[0])
        # Extract /tcp/<port> from the multiaddr; fall back to default on parse errors.
        import re as _re
        _m = _re.search(r"/tcp/(\d+)", _first)
        _p2p_default_port = int(_m.group(1)) if _m else 4001

    _resolved = resolve_bootstrap(
        cli_peer_id=args.peer_id,
        cli_api_port=int(args.api_port),
        cli_grpc_port=int(args.grpc_port),
        cli_p2p_port=int(_p2p_default_port),
        identity_path=DEFAULT_BOOTSTRAP_IDENTITY_PATH,
        peers_local_path=DEFAULT_PEERS_LOCAL_PATH,
    )
    args.peer_id = _resolved.peer_id
    args.api_port = _resolved.api_port
    args.grpc_port = _resolved.grpc_port
    # If the libp2p listen port changed due to auto-retry, rewrite any
    # ``--p2p-listen`` multiaddrs that still reference the original default.
    # Multi-multiaddr cases (repeated --p2p-listen flags) are left intact —
    # the user clearly wanted specific addresses.
    if _resolved.p2p_port != _p2p_default_port and len(_p2p_listen_from_cli) <= 1:
        args.p2p_listen = [f"/ip4/0.0.0.0/tcp/{_resolved.p2p_port}"]
    logger.info(
        "bootstrap_resolved: peer_id=%s source=%s api=%d grpc=%d libp2p=%d migrated=%s",
        _resolved.peer_id, _resolved.peer_id_source,
        _resolved.api_port, _resolved.grpc_port, _resolved.p2p_port,
        _resolved.ports_migrated,
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

    # Resolve the canonical HF model id for the tokenizer — the authority
    # MLX ↔ PyTorch peers use to agree on a shared vocab. See
    # ``peer.model_catalog.resolve_hf_model_id`` for resolution order.
    _hf_model_id = str(getattr(args, "hf_model_id", "") or "").strip()
    if not _hf_model_id:
        from peer.model_catalog import resolve_hf_model_id as _resolve_hf
        _hf_model_id = _resolve_hf(
            args.model_id,
            catalog_path=getattr(args, "model_catalog_path", None) or "models.catalog.json",
            runtime_model_id=_runtime_model_id,
        )
    if _hf_model_id:
        logger.info(
            "hf_tokenizer_id: model=%s hf_model_id=%s (force=%s, guard=%s)",
            args.model_id, _hf_model_id,
            bool(getattr(args, "mlx_force_hf_tokenizer", True)),
            bool(getattr(args, "tokenizer_vocab_guard", True)),
        )
    else:
        logger.warning(
            "hf_model_id_unresolved: model=%s — MLX runtime will fall back to bundled tokenizer",
            args.model_id,
        )

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

    # ── B1 rendezvous: gossipsub client + hole-punch responder ──
    # Opt-in, non-fatal: if P2P is disabled or gossipsub publish fails
    # (InsufficientPeers right after boot), we continue without.
    # The responder side listens for inbound ``REQUEST_HOLE_PUNCH``
    # events targeting this node and issues an active ``dial_peer``
    # toward the requester, giving DCUtR a simultaneous-dial window
    # to punch through symmetric NAT.
    _gossip_client = None
    if _p2p_node is not None:
        try:
            from peer.gossip_client import (
                GossipClient as _GossipClient,
                attach_hole_punch_responder as _attach_hp,
            )
            _self_libp2p_id = str(getattr(_p2p_node, "libp2p_peer_id", "") or "")
            _gossip_client = _GossipClient(
                p2p_node=_p2p_node,
                self_libp2p_peer_id=_self_libp2p_id,
                poll_interval_s=0.1,
                peer_dead_debounce_s=1.0,
                hole_punch_debounce_s=5.0,
            )
            _attach_hp(
                _gossip_client,
                p2p_node=_p2p_node,
                self_libp2p_peer_id=_self_libp2p_id,
            )
            _gossip_client.start(thread_name="openhydra-gossip")
            logger.info(
                "gossip_client_started libp2p_peer_id=%s topic=openhydra/swarm/v1/events",
                _self_libp2p_id,
            )
        except Exception as _gossip_err:
            logger.warning("gossip_client_start_failed: %s", _gossip_err)
            _gossip_client = None

    # ── Phase 3 zero-config bootstrap: SwarmNegotiator + capacity snapshot ──
    # Build the CapacityReport NOW (before peer announce) so we can:
    #   1. Self-assign a shard via SwarmNegotiator when the user didn't
    #      pass explicit --layer-start / --layer-end / --shard-index.
    #   2. Serialise it as capacity_json and hand it to the announce loop.
    # Any failure here is non-fatal — we fall back to whatever the user
    # passed on the CLI (manual shard flags always win anyway).
    _capacity_json_str: str = ""
    _capacity_schema_version: int = 0
    _negotiator_assignment = None
    try:
        import json as _negotiator_json
        from coordinator.engine import EngineConfig as _EC
        from coordinator.degradation import ModelAvailability as _MA
        from peer.capacity import (
            CAPACITY_SCHEMA_VERSION as _CSV,
            NODE_PERSONA_ATOMIC_WORKER as _NP_ATOMIC,
            NODE_PERSONA_NATIVE_SHARD as _NP_NATIVE,
            UpstreamConfig as _UC,
            build_capacity_report as _build_report,
        )
        from peer.hardware import detect_hardware_profile as _detect_hw
        from peer.swarm_negotiator import (
            PeerClaim as _PeerClaim,
            SwarmNegotiator as _Negotiator,
        )

        # Load the catalogue independently of the CoordinatorEngine —
        # the engine itself spins up only later inside coordinator_serve.
        _phase3_catalog_path: str | None = "models.catalog.json"
        if not os.path.exists(_phase3_catalog_path or ""):
            _phase3_catalog_path = None
        _phase3_catalog: list = []
        if _phase3_catalog_path:
            try:
                _phase3_raw = _negotiator_json.loads(
                    open(_phase3_catalog_path).read()
                )
                _seen: set[str] = set()
                for _entry in _phase3_raw:
                    _mid = str(_entry.get("model_id", "")).strip()
                    if not _mid or _mid in _seen:
                        continue
                    _seen.add(_mid)
                    _phase3_catalog.append(_MA(
                        model_id=_mid,
                        required_peers=int(_entry.get("required_peers", 1)),
                        hf_model_id=str(_entry.get("hf_model_id", "")),
                        min_vram_gb=max(0, int(_entry.get("min_vram_gb", 0))),
                        recommended_quantization=str(_entry.get("recommended_quantization", "fp32")),
                        context_length=max(0, int(_entry.get("context_length", 4096))),
                        shard_vram_gb=max(0.0, float(_entry.get("shard_vram_gb", 0))),
                        shards_needed=max(1, int(_entry.get("shards_needed", 1))),
                        quality_tier=str(_entry.get("quality_tier", "standard")),
                        num_layers=max(0, int(_entry.get("num_layers", 0))),
                    ))
            except Exception as _cat_err:
                logger.warning("phase3_catalog_load_failed: %s", _cat_err)
                _phase3_catalog = []

        _hw = _detect_hw()
        _upstream_cfg = None
        if _node_persona == "atomic_worker":
            _upstream_cfg = _UC(
                kind=_upstream_kind,
                url=_upstream_url,
                hosted_model_ids=tuple(_hosted_model_ids),
            )
        _report = _build_report(
            hardware=_hw,
            catalog=_phase3_catalog,
            peer_id=str(args.peer_id or ""),
            libp2p_peer_id=(
                str(getattr(_p2p_node, "libp2p_peer_id", "") or "")
                if _p2p_node is not None else ""
            ),
            ports={
                "api": int(args.api_port),
                "grpc": int(args.grpc_port),
                "libp2p": int(_resolved.p2p_port),
            },
            advertise_host=str(args.advertise_host or ""),
            runtime_backend=str(args.runtime_backend or ""),
            node_persona=_node_persona,
            upstream=_upstream_cfg,
        )
        _capacity_json_str = _negotiator_json.dumps(_report.to_dict())
        _capacity_schema_version = _CSV

        # Skip negotiation if the user passed an explicit shard range —
        # manual CLI intent always wins (Phase 2 design principle).
        _manual_shard = (
            bool(_explicit_layer_indices)
            or int(getattr(args, "total_shards", 1) or 1) > 1
            or int(getattr(args, "shard_index", 0) or 0) != 0
        )
        if _manual_shard:
            logger.info(
                "swarm_negotiate_skipped: manual shard flags present "
                "(--layer-start/--layer-end/--shard-index/--total-shards)"
            )
        else:
            # Build a DHT scan function.  When we have a live p2p_node,
            # query Kademlia; otherwise return empty (first-boot-on-empty-swarm).
            def _scan_dht_for_model(mid: str) -> list:
                claims: list = []
                if _p2p_node is None:
                    return claims
                try:
                    discovered = _p2p_node.discover(mid) or []
                except Exception as _disc_err:
                    logger.debug(
                        "phase3_discover_failed: model=%s err=%s", mid, _disc_err,
                    )
                    return claims
                for rec in discovered:
                    try:
                        if not isinstance(rec, dict):
                            continue
                        claims.append(_PeerClaim(
                            libp2p_peer_id=str(rec.get("libp2p_peer_id") or ""),
                            model_id=str(rec.get("model_id") or mid),
                            layer_start=int(rec.get("layer_start", 0) or 0),
                            layer_end=int(rec.get("layer_end", 0) or 0),
                            total_layers=int(rec.get("total_layers", 0) or 0),
                            available_vram_mb=int(rec.get("available_vram_mb", 0) or 0),
                        ))
                    except (TypeError, ValueError):
                        continue
                return claims

            _negotiator = _Negotiator(
                capacity_report=_report,
                libp2p_peer_id=(
                    str(getattr(_p2p_node, "libp2p_peer_id", "") or "")
                    if _p2p_node is not None else ""
                ),
                dht_scan=_scan_dht_for_model,
                preferred_model_order=(args.model_id,) if args.model_id else (),
            )
            _negotiator_assignment = _negotiator.negotiate()
            if _negotiator_assignment is not None:
                # Override the shard args so the peer thread picks up the
                # negotiated range instead of the CLI defaults (shard_index=0,
                # total_shards=1, empty layer_start/layer_end).
                args.model_id = _negotiator_assignment.model_id
                args.layer_start = int(_negotiator_assignment.layer_start)
                args.layer_end = int(_negotiator_assignment.layer_end)
                _explicit_layer_indices = tuple(
                    range(args.layer_start, args.layer_end)
                )
                logger.info(
                    "swarm_negotiate_result: model=%s layers=[%d, %d) total=%d source=%s",
                    _negotiator_assignment.model_id,
                    _negotiator_assignment.layer_start,
                    _negotiator_assignment.layer_end,
                    _negotiator_assignment.total_layers,
                    _negotiator_assignment.source,
                )
            else:
                logger.info(
                    "swarm_negotiate_no_assignment: falling back to CLI defaults"
                )
    except Exception as _phase3_err:
        logger.warning("phase3_negotiate_failed: %s", _phase3_err)

    # ── Phase 4 zero-config bootstrap: build a LoopSnapshot + negotiation
    # loop factory.  The factory is invoked inside peer_server.serve()
    # once the live PeerService is available (so ``is_busy_fn`` can call
    # ``service.inflight_count() > 0``).  If anything here fails, we
    # degrade to Phase 3 one-shot behaviour gracefully. ────────────────
    _capacity_snapshot_ref = None
    _negotiation_loop_factory_fn = None
    try:
        from peer.negotiation_loop import (
            DEFAULT_NEGOTIATION_INTERVAL_S as _P4_INTERVAL,
            LoopSnapshot as _P4_LoopSnapshot,
            NegotiationLoop as _P4_NegotiationLoop,
        )

        _capacity_snapshot_ref = _P4_LoopSnapshot.build(
            capacity_json=_capacity_json_str,
            capacity_schema_version=_capacity_schema_version,
            current_assignment=_negotiator_assignment,
        )

        # Closure: rebuild the CapacityReport with fresh HardwareProfile.
        # Keeps static metadata (peer_id, ports, persona, upstream)
        # captured at boot — they don't drift.
        def _p4_build_report():
            # Local imports to keep boot-time module graph lean.
            from peer.capacity import (
                UpstreamConfig as _P4_UC,
                build_capacity_report as _P4_build,
            )
            from peer.hardware import detect_hardware_profile as _P4_hw
            _p4_upstream = None
            if _node_persona == "atomic_worker":
                _p4_upstream = _P4_UC(
                    kind=_upstream_kind,
                    url=_upstream_url,
                    hosted_model_ids=tuple(_hosted_model_ids),
                )
            return _P4_build(
                hardware=_P4_hw(),
                catalog=_phase3_catalog,
                peer_id=str(args.peer_id or ""),
                libp2p_peer_id=(
                    str(getattr(_p2p_node, "libp2p_peer_id", "") or "")
                    if _p2p_node is not None else ""
                ),
                ports={
                    "api": int(args.api_port),
                    "grpc": int(args.grpc_port),
                    "libp2p": int(_resolved.p2p_port),
                },
                advertise_host=str(args.advertise_host or ""),
                runtime_backend=str(args.runtime_backend or ""),
                node_persona=_node_persona,
                upstream=_p4_upstream,
            )

        # Closure: make a fresh SwarmNegotiator bound to a fresh report.
        def _p4_make_negotiator(report):
            from peer.swarm_negotiator import (
                PeerClaim as _P4_PeerClaim,
                SwarmNegotiator as _P4_Negotiator,
            )

            def _scan(mid: str) -> list:
                """Combined DHT scan for the negotiator.

                The libp2p Kademlia discover has a key-schema mismatch
                between ``handle_announce`` (which writes at
                ``/openhydra/model/<mid>/<peer_id>``) and
                ``handle_discover`` (which reads at ``/openhydra/model/
                <mid>``). Until that's fixed in Rust, we fall back to
                the HTTP DHT on port 8468 — the same source the chain's
                PathFinder uses to build live pipelines — which is
                known to work reliably in production. On an empty
                Kademlia result we merge the HTTP DHT peer list so the
                SwarmNegotiator's conflict_split heuristic can still
                see the other peers.
                """
                claims: list = []
                # Kademlia path (primary — will find peers once the key
                # schema is repaired on the Rust side).
                if _p2p_node is not None:
                    try:
                        discovered = _p2p_node.discover(mid) or []
                    except Exception:
                        discovered = []
                    for rec in discovered:
                        try:
                            if not isinstance(rec, dict):
                                continue
                            claims.append(_P4_PeerClaim(
                                libp2p_peer_id=str(rec.get("libp2p_peer_id") or ""),
                                model_id=str(rec.get("model_id") or mid),
                                layer_start=int(rec.get("layer_start", 0) or 0),
                                layer_end=int(rec.get("layer_end", 0) or 0),
                                total_layers=int(rec.get("total_layers", 0) or 0),
                                available_vram_mb=int(rec.get("available_vram_mb", 0) or 0),
                            ))
                        except (TypeError, ValueError):
                            continue
                # HTTP DHT fallback — always runs so a failing Kademlia
                # doesn't silently block the negotiator.
                try:
                    from coordinator.path_finder import load_peers_from_dht as _lpf
                    http_peers = _lpf(
                        dht_urls=list(dht_urls or ()),
                        model_id=str(mid or ""),
                        timeout_s=2.0,
                    ) or []
                except Exception:
                    http_peers = []
                seen_libp2p = {c.libp2p_peer_id for c in claims if c.libp2p_peer_id}
                for peer in http_peers:
                    try:
                        pid = str(getattr(peer, "libp2p_peer_id", "") or "")
                        # Skip self + already-known peers.
                        if not pid:
                            continue
                        if pid in seen_libp2p:
                            continue
                        total = int(getattr(peer, "total_layers", 0) or 0)
                        start = int(getattr(peer, "layer_start", 0) or 0)
                        end = int(getattr(peer, "layer_end", 0) or 0)
                        # Skip records that don't advertise a layer range
                        # at all — they predate the sharding schema.
                        if total <= 0 or end <= start:
                            continue
                        claims.append(_P4_PeerClaim(
                            libp2p_peer_id=pid,
                            model_id=str(getattr(peer, "model_id", "") or mid),
                            layer_start=start,
                            layer_end=end,
                            total_layers=total,
                            available_vram_mb=0,  # HTTP DHT doesn't track VRAM yet
                        ))
                        seen_libp2p.add(pid)
                    except (TypeError, ValueError):
                        continue
                return claims

            return _P4_Negotiator(
                capacity_report=report,
                libp2p_peer_id=(
                    str(getattr(_p2p_node, "libp2p_peer_id", "") or "")
                    if _p2p_node is not None else ""
                ),
                dht_scan=_scan,
                preferred_model_order=(args.model_id,) if args.model_id else (),
            )

        _p4_interval = float(
            getattr(args, "negotiation_interval_s", _P4_INTERVAL) or _P4_INTERVAL
        )

        # Factory: peer_server.serve() calls this once with ``(is_busy_fn,
        # service)``. The ``service`` arg gates the B3 ReshardExecutor;
        # when the ``--reshard-executor-enabled`` flag is set and a
        # ``service`` reference is available, the loop's tick hands
        # changed assignments to the executor which runs the drain →
        # unload → reload FSM. Feature-flagged off by default; on
        # failure the executor stays degraded (stay-degraded policy).
        _b3_enabled = bool(getattr(args, "reshard_executor_enabled", False))
        _b3_drain_timeout = float(
            getattr(args, "reshard_drain_timeout_s", 120.0) or 120.0
        )

        def _negotiation_loop_factory_impl(is_busy_fn, service=None):
            _reshard_fn = None
            if _b3_enabled and service is not None:
                try:
                    from peer.reshard_executor import ReshardExecutor as _ReExec
                    # Gossip publish hook so RESHARD_ANNOUNCE goes out on
                    # successful reload. ``_gossip_client`` was set up
                    # back in the P2P bootstrap block above.
                    _pub_fn = None
                    if _gossip_client is not None:
                        _pub_fn = lambda evt_type, data: bool(
                            _gossip_client.publish(evt_type, data)
                        )
                    _executor = _ReExec(
                        service=service,
                        drain_timeout_s=_b3_drain_timeout,
                        gossip_publish_fn=_pub_fn,
                    )
                    _executor.set_initial_assignment(_negotiator_assignment)
                    _reshard_fn = _executor.propose
                    logger.info(
                        "reshard_executor_enabled: drain_timeout=%.1fs gossip=%s",
                        _b3_drain_timeout,
                        "on" if _pub_fn is not None else "off",
                    )
                except Exception as _re_err:
                    logger.warning(
                        "reshard_executor_wire_failed: %s — falling back "
                        "to Phase-4 log-only behaviour", _re_err,
                    )
            return _P4_NegotiationLoop(
                build_capacity_report_fn=_p4_build_report,
                make_negotiator_fn=_p4_make_negotiator,
                snapshot=_capacity_snapshot_ref,
                initial_assignment=_negotiator_assignment,
                is_busy_fn=is_busy_fn,
                interval_s=_p4_interval,
                reshard_executor_fn=_reshard_fn,
            )

        _negotiation_loop_factory_fn = _negotiation_loop_factory_impl
        logger.info(
            "phase4_negotiation_loop_configured: interval=%.1fs",
            _p4_interval,
        )
    except Exception as _phase4_err:
        logger.warning(
            "phase4_negotiation_loop_setup_failed: %s — "
            "falling back to Phase 3 one-shot negotiation",
            _phase4_err,
        )
        _capacity_snapshot_ref = None
        _negotiation_loop_factory_fn = None

    # ── Phase 6: pure-coordinator mode (no local peer thread) ───────────
    # When ``--no-local-peer`` is set, skip the peer thread entirely and
    # construct a StandaloneHead so the coordinator can still sample
    # via Path A. Used for the true Petals topology: client / coordinator
    # on a laptop, transformer layers on remote GPU peers.
    _no_local_peer = bool(getattr(args, "no_local_peer", False))
    _sample_on_coord = bool(getattr(args, "sample_on_coordinator", False))
    if _no_local_peer:
        # Guardrail #1: --no-local-peer is meaningless without --sample-on-coordinator
        # (the coordinator would have no work to do — peers would orchestrate
        # themselves but no one would emit tokens to the HTTP queue).
        if not _sample_on_coord:
            parser.error(
                "--no-local-peer requires --sample-on-coordinator. Without "
                "Path A enabled, the coordinator has no work to do — peer "
                "ring loops back among themselves and the HTTP client never "
                "receives tokens."
            )
        # Guardrail #2: standalone head needs an HF repo id (MLX or
        # PyTorch), not one of the catalog's logical names like
        # ``openhydra-qwen3.5-2b`` which neither mlx_lm.load nor
        # transformers.AutoModel can resolve. Any string without a
        # ``user/repo`` slash is rejected.
        _rmi = str(_runtime_model_id or "").strip()
        if not _rmi or "/" not in _rmi:
            parser.error(
                "--no-local-peer requires --runtime-model-id with an "
                "HF repo id like ``Qwen/Qwen3.5-2B`` (PyTorch backend) or "
                "``mlx-community/Qwen3.5-2B-MLX-8bit`` (MLX backend). "
                f"Got {_rmi!r}, which is not a valid HF repo id "
                "(missing the ``user/repo`` slash)."
            )
        logger.info(
            "pure_coordinator_mode: --no-local-peer set → skipping peer "
            "thread; loading standalone head from %s", _runtime_model_id,
        )
        try:
            from coordinator.standalone_head import load_standalone_head
            from coordinator.head_sampler import register_head_source
            _sh_backend = str(getattr(args, "standalone_head_backend", "auto") or "auto")
            _sh_device = str(getattr(args, "standalone_head_device", "cpu") or "cpu")
            _sh_dtype = str(getattr(args, "standalone_head_dtype", "float32") or "float32")
            _standalone_head = load_standalone_head(
                str(_runtime_model_id),
                backend=_sh_backend,
                pytorch_device=_sh_device,
                pytorch_dtype=_sh_dtype,
            )
            register_head_source(
                peer_id="coordinator-standalone-head",
                runtime=_standalone_head,
            )
            logger.info(
                "standalone_head_registered: backend=%s hf_model_id=%s "
                "vocab=%d hidden=%d tie=%s",
                _standalone_head.backend,
                _standalone_head.hf_model_id,
                _standalone_head.vocab_size,
                _standalone_head.hidden_size,
                _standalone_head.tie_word_embeddings,
            )
        except Exception as exc:
            logger.error(
                "standalone_head_load_failed: %s — pure-coordinator mode "
                "cannot serve requests",
                exc,
            )
            raise
        peer_thread = None  # No local peer thread in pure-coordinator mode.
        # Start the coord-only proxy handler so inbound libp2p proxy
        # requests (specifically PROXY_METHOD_PUSH_RESULT from the
        # last peer in a Path A ring) get drained. Without this the
        # Rust request_response handler times out on every inbound
        # request because nothing calls p2p_node.poll_proxy_request.
        if _p2p_node is not None:
            from peer.server import _coordinator_proxy_handler_loop
            _coord_proxy_stop = threading.Event()
            _coord_proxy_thread = threading.Thread(
                target=_coordinator_proxy_handler_loop,
                kwargs={
                    "stop_event": _coord_proxy_stop,
                    "p2p_node": _p2p_node,
                },
                name="openhydra-coord-proxy",
                daemon=True,
            )
            _coord_proxy_thread.start()
            logger.info(
                "coordinator_proxy_handler_started: handles "
                "PROXY_METHOD_PUSH_RESULT for pure-coordinator Path A"
            )
    else:
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
                "hf_model_id": _hf_model_id,
                "mlx_force_hf_tokenizer": bool(getattr(args, "mlx_force_hf_tokenizer", True)),
                "tokenizer_vocab_guard": bool(getattr(args, "tokenizer_vocab_guard", True)),
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
                # Phase 3 zero-config: capacity snapshot attached to every
                # DHT announcement so the rest of the swarm sees this node's
                # per-model capacity without a follow-up probe.
                "capacity_json": _capacity_json_str,
                "capacity_schema_version": _capacity_schema_version,
                # Phase 4 zero-config: continuous re-negotiation plumbing.
                # ``capacity_snapshot_ref`` is updated every tick by the
                # NegotiationLoop; the announce loop reads from it on each
                # heartbeat so fresh capacity + assignment propagate without
                # any additional RPCs.
                "capacity_snapshot_ref": _capacity_snapshot_ref,
                "negotiation_loop_factory": _negotiation_loop_factory_fn,
                # Path A Phase 5: tell the local peer to load final_norm +
                # lm_head on every shard (not just the last) so the
                # coordinator — co-located in this same process — can borrow
                # the head weights via HeadSampler even when the local peer
                # is stage 0 rather than the terminal stage. Gated on the
                # ``--sample-on-coordinator`` opt-in so default deployments
                # see no extra memory usage.
                "load_full_head": bool(getattr(args, "sample_on_coordinator", False)),
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
    # In pure-coordinator mode there's NO local peer to register; the
    # coordinator relies entirely on remote peers discovered via the DHT.
    # An empty peers config triggers the existing DHT-discovery path in
    # the chain builder.
    peers_config_path = args.peers_config
    if not peers_config_path and _no_local_peer:
        # Write an empty list so downstream code that opens peers_config
        # finds a parseable file. Discovery happens entirely via DHT.
        _tmp_empty = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="openhydra_peers_pure_",
            delete=False,
        )
        json.dump([], _tmp_empty)
        _tmp_empty.close()
        peers_config_path = _tmp_empty.name
        logger.info(
            "auto_peers_config_pure_coordinator: %s (empty — DHT discovery only)",
            peers_config_path,
        )
    elif not peers_config_path:
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
        # Add libp2p peer ID for cross-ISP relay routing in push mode.
        if _p2p_node is not None:
            _local_peer[0]["libp2p_peer_id"] = str(getattr(_p2p_node, "libp2p_peer_id", "") or "")
            _local_peer[0]["requires_relay"] = True  # Conservative: assume relay needed
        _tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="openhydra_peers_",
            delete=False,
        )
        json.dump(_local_peer, _tmp)
        _tmp.close()
        peers_config_path = _tmp.name
        logger.info("auto_peers_config: %s (local peer %s on :%d)",
                     peers_config_path, args.peer_id, args.grpc_port)

    # ── Phase 2: persist peers.local.json so next boot reuses the same
    # peer-id, advertise host, and ports without any CLI flags ────────────
    try:
        _persisted = PeersLocalConfig(
            peer_id=str(args.peer_id),
            libp2p_peer_id=(
                str(getattr(_p2p_node, "libp2p_peer_id", "") or "")
                if _p2p_node is not None else ""
            ),
            advertise_host=str(args.advertise_host or ""),
            ports={
                "api": int(args.api_port),
                "grpc": int(args.grpc_port),
                "libp2p": int(_resolved.p2p_port),
            },
        )
        _persisted.save(_resolved.peers_local_path)
        logger.info("peers_local_saved: %s", _resolved.peers_local_path)
    except Exception as _persist_err:
        # Non-fatal — persistence is a convenience, not a correctness
        # requirement.  Log and continue.
        logger.warning("peers_local_save_failed: %s", _persist_err)

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
        sample_on_coordinator=bool(getattr(args, "sample_on_coordinator", False)),
    )

    # Zero-config bootstrap Phase 1: forward identity + network metadata to
    # the HTTP API so the /v1/internal/capacity endpoint can emit a complete
    # CapacityReport without re-deriving any of this. All fields default to
    # empty strings / zeros if the underlying value isn't available yet.
    _node_meta: dict[str, Any] = {
        "peer_id": str(args.peer_id or ""),
        "libp2p_peer_id": (
            str(getattr(_p2p_node, "libp2p_peer_id", "") or "")
            if _p2p_node is not None else ""
        ),
        "ports": {
            "api": int(args.api_port),
            "grpc": int(args.grpc_port),
            "libp2p": 4001,
        },
        "advertise_host": str(args.advertise_host or ""),
        "runtime_backend": str(args.runtime_backend or ""),
        # Phase 1.5: hybrid persona metadata for /v1/internal/capacity.
        # ``upstream`` is a dict (not an UpstreamConfig yet) to keep node.py
        # free of the peer.capacity import — the API handler reifies it.
        "node_persona": _node_persona,
        "upstream": (
            {
                "kind": _upstream_kind,
                "url": _upstream_url,
                "hosted_model_ids": list(_hosted_model_ids),
            }
            if _node_persona == "atomic_worker" else None
        ),
    }

    # Start the coordinator HTTP API on the main thread (blocking).
    # Signal handling (SIGTERM, SIGINT) is already wired inside coordinator_serve.
    coordinator_serve(
        host=args.api_host,
        port=args.api_port,
        config=engine_config,
        api_key=api_key,
        p2p_node=_p2p_node,
        node_meta=_node_meta,
        gossip_client=_gossip_client,
        capacity_snapshot_ref=_capacity_snapshot_ref,
    )


if __name__ == "__main__":
    main()
