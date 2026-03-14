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
import logging
import os
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
    parser.add_argument("--runtime-backend", default="toy_auto",
                        help="Inference runtime backend (default: toy_auto).")
    parser.add_argument("--daemon-mode", default="polite",
                        choices=["polite", "aggressive", "passive"],
                        help="Peer resource policy (default: polite).")
    parser.add_argument("--deployment-profile", default="dev",
                        choices=["dev", "prod"],
                        help="Deployment profile — dev or prod (default: dev).")

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

    logger.info(
        "openhydra_node_starting peer_id=%s model=%s grpc_port=%d api=%s:%d dht=%s",
        args.peer_id, args.model_id, args.grpc_port,
        args.api_host, args.api_port, dht_urls,
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

    # Build the coordinator engine config with only the fields we override;
    # EngineConfig is a frozen dataclass and all other fields carry their defaults.
    engine_config = EngineConfig(
        dht_urls=dht_urls,
        deployment_profile=args.deployment_profile,
        peers_config_path=args.peers_config,
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
