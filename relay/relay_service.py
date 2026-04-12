# Copyright 2026 OpenHydra contributors — Apache 2.0

"""gRPC relay service for NAT traversal.

Runs on each DHT bootstrap server (port 50052 by default). NATted peers
connect OUTBOUND to this service and register their gRPC channel. When a
coordinator needs to call ``Forward()`` on a NATted peer, it routes to
this relay address and the relay proxies through the peer's registered
channel.

Usage::

    python -m relay.relay_service --port 50052 --peer-id relay-us

Architecture::

    NATted peer ──(outbound)──> Relay :50052
         ^                         |
         |                         v
    Coordinator ──(Forward())──> Relay ──(proxy)──> NATted peer's channel
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from concurrent import futures
from typing import Any

import grpc

from coordinator.relay import RelayServer
from peer import peer_pb2, peer_pb2_grpc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


class RelayServicer(peer_pb2_grpc.PeerServicer):
    """gRPC servicer that proxies ``Forward()`` calls through registered
    NATted peer channels.

    Peer registration happens implicitly: when a NATted peer calls
    ``Ping()`` with its ``peer_id`` in the request metadata, the relay
    captures the calling channel context for future proxying. This avoids
    needing a separate ``Register`` RPC — the peer just pings and stays
    connected.
    """

    def __init__(self, relay: RelayServer, relay_peer_id: str):
        self._relay = relay
        self._relay_peer_id = relay_peer_id
        # Map of peer_id → (stub, channel) for reverse-proxying.
        # Populated when NATted peers call RegisterRelay().
        self._peer_stubs: dict[str, Any] = {}
        self._lock = threading.Lock()

    def Ping(self, request, context):
        """Health check + peer registration.

        When a NATted peer calls Ping(), we extract its identity from
        the request metadata and store the reverse channel for proxying.
        """
        # Extract peer metadata from the gRPC context
        metadata = dict(context.invocation_metadata() or [])
        peer_id = metadata.get("x-openhydra-peer-id", "")
        model_id = metadata.get("x-openhydra-model-id", "")

        if peer_id:
            # The peer sent its ID — register it for reverse proxying.
            # We need the peer's address to create an outbound channel back.
            peer_addr = metadata.get("x-openhydra-peer-address", "")
            if peer_addr:
                try:
                    channel = grpc.insecure_channel(
                        peer_addr,
                        options=[
                            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
                            ("grpc.max_send_message_length", 100 * 1024 * 1024),
                            ("grpc.keepalive_time_ms", 30000),
                            ("grpc.keepalive_timeout_ms", 10000),
                        ],
                    )
                    stub = peer_pb2_grpc.PeerStub(channel)
                    self._relay.register(
                        peer_id=peer_id,
                        channel=channel,
                        stub=stub,
                        model_id=model_id,
                    )
                    logger.info(
                        "relay_peer_registered: peer=%s addr=%s model=%s",
                        peer_id, peer_addr, model_id,
                    )
                except Exception as exc:
                    logger.warning(
                        "relay_peer_register_failed: peer=%s err=%s",
                        peer_id, exc,
                    )
            else:
                # Heartbeat without registration (peer already registered)
                self._relay.heartbeat(peer_id)

        return peer_pb2.PingResponse(
            peer_id=self._relay_peer_id,
            received_unix_ms=int(time.time() * 1000),
        )

    def Forward(self, request, context):
        """Proxy a Forward() call to a registered NATted peer.

        The coordinator sends the request to this relay because the peer
        announced ``relay_address`` pointing here. We look up the peer's
        registered channel and forward the request.
        """
        # The request's peer_id field tells us which peer to forward to.
        # But the request doesn't have a target peer_id — the coordinator
        # sends to the relay_address and expects it to reach the right peer.
        # We use the request's shard_layer_start/end to find the peer, or
        # fall back to the first registered peer for the model.
        target_peer_id = ""

        # Try metadata first (coordinator can set x-target-peer-id)
        metadata = dict(context.invocation_metadata() or [])
        target_peer_id = metadata.get("x-target-peer-id", "")

        if not target_peer_id:
            # Fallback: find any registered peer
            registered = self._relay.registered_peers()
            if registered:
                target_peer_id = registered[0]

        if not target_peer_id:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("relay: no registered peers")
            return peer_pb2.ForwardResponse(
                request_id=request.request_id,
                peer_id=self._relay_peer_id,
                error="relay: no registered peers",
            )

        try:
            response = self._relay.forward_to_peer(
                peer_id=target_peer_id,
                request=request,
                timeout_s=120.0,
            )
            return response
        except Exception as exc:
            logger.warning(
                "relay_forward_failed: target=%s err=%s",
                target_peer_id, exc,
            )
            return peer_pb2.ForwardResponse(
                request_id=request.request_id,
                peer_id=self._relay_peer_id,
                error=f"relay_forward_failed: {exc}",
            )

    def PeerStatus(self, request, context):
        """Return relay status including registered peer count."""
        registered = self._relay.registered_peers()
        return peer_pb2.PeerStatusResponse(
            peer_id=self._relay_peer_id,
            model_id=f"relay ({len(registered)} peers)",
            status="ok",
            uptime_seconds=0,
            load_pct=0.0,
        )


def serve(host: str = "0.0.0.0", port: int = 50052, peer_id: str = "relay"):
    """Start the relay gRPC service."""
    relay = RelayServer(max_registrations=1024, heartbeat_timeout_s=300.0)
    servicer = RelayServicer(relay, relay_peer_id=peer_id)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=32),
        options=[
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
        ],
    )
    peer_pb2_grpc.add_PeerServicer_to_server(servicer, server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    logger.info("relay_service_started: %s:%d peer_id=%s", host, port, peer_id)

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("relay_service_shutting_down")
        server.stop(grace=5)


def main():
    parser = argparse.ArgumentParser(description="OpenHydra gRPC relay for NAT traversal")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50052)
    parser.add_argument("--peer-id", default="relay")
    args = parser.parse_args()
    serve(host=args.host, port=args.port, peer_id=args.peer_id)


if __name__ == "__main__":
    main()
