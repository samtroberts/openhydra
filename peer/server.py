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

from __future__ import annotations

import argparse
import asyncio
from concurrent import futures
import logging
from pathlib import Path
import threading
import time
from typing import Any

from openhydra_defaults import PRODUCTION_BOOTSTRAP_URLS
from openhydra_logging import configure_logging

logger = logging.getLogger(__name__)

# Production Hivemind DHT signpost nodes (EU / US / AP).
# Peers connect to these to join the Kademlia DHT for decentralised discovery.
# Peer IDs are stable across restarts (identity key persisted on each Nanode).
# To update: deploy with identity_path, then read peer_id from journalctl.
_DEFAULT_HIVEMIND_SIGNPOSTS: list[str] = [
    # EU — 172.105.69.49
    "/ip4/172.105.69.49/tcp/38751/p2p/QmaEBYaG3gm8W1neRMvyyuUmeYgM8cKVe54Sy4wPE4XhBY",
    # US — 45.79.190.172
    "/ip4/45.79.190.172/tcp/38751/p2p/QmPMFTzpJ5NE1FsSCjdMPgRYMhe488sXmrGfGV7tJ1ykc2",
    # AP — 172.104.164.98
    "/ip4/172.104.164.98/tcp/38751/p2p/QmPWABM7D1j41UzCtyk3r3X2yH3HPZf1UZPDJ9qTyQYYqG",
]

import grpc

from compression.autoencoder import CompressionProfile, TensorAutoencoder
from peer.daemon_monitor import (
    DaemonController,
    DaemonMode,
    MonitorConfig,
    ResourceBudget,
)
from peer.crypto import (
    cryptography_available,
    decrypt_activation_envelope,
    decrypt_activation_envelope_with_privkey,
    load_or_create_identity_keyfile,
    peel_onion_route_layer,
    peel_onion_route_layer_with_privkey,
    private_key_from_identity,
    sign_geo_challenge,
)
from peer.batching import BatchingQueue
from peer.dht_announce import Announcement, announce_http_many
from peer.p2p_model_cache import P2PModelCache
from peer.seeder_http import ModelSeedServer
from peer.hardware import detect_hardware_profile
from peer.model_shard import ModelShard, ToyShardConfig
from peer.tls import load_server_credentials
from peer import peer_pb2
from peer import peer_pb2_grpc
from openhydra_secrets import is_insecure_secret_value, load_secret_store
try:
    from torrent.session import SessionBootstrapConfig, TorrentSessionManager
    from torrent.seeder import ArbitrationConfig
except ImportError:
    SessionBootstrapConfig = None  # type: ignore[assignment,misc]
    TorrentSessionManager = None  # type: ignore[assignment,misc]
    ArbitrationConfig = None  # type: ignore[assignment,misc]

GRPC_MAX_MESSAGE_BYTES = 100 * 1024 * 1024
GRPC_SERVER_OPTIONS: list[tuple[str, int]] = [
    ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_BYTES),
    ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_BYTES),
]
_QUANTIZATION_FLAG_TO_MODE = {
    "none": "fp32",
    "8bit": "int8",
    "4bit": "int4",
}


def _exponential_backoff_delay(
    attempt: int,
    *,
    base_seconds: float = 1.0,
    cap_seconds: float = 60.0,
) -> float:
    clamped = max(0, min(10, int(attempt)))
    return min(float(cap_seconds), float(base_seconds) * (2.0 ** clamped))


def _derive_relay_addresses(
    dht_urls: list[str],
    relay_port: int = 50052,
) -> list[str]:
    """Derive relay gRPC addresses from DHT bootstrap HTTP URLs.

    The relay service runs on the same hosts as the DHT bootstrap but on
    a different port. This helper extracts hostnames from URLs like
    ``http://bootstrap-us.openhydra.co:8468`` and returns
    ``["bootstrap-us.openhydra.co:50052", ...]``.
    """
    from urllib.parse import urlparse
    addrs: list[str] = []
    for url in dht_urls:
        try:
            parsed = urlparse(url)
            host = parsed.hostname
            if host:
                addrs.append(f"{host}:{relay_port}")
        except Exception:
            continue
    return addrs


# Proxy method prefix bytes for demultiplexing.
PROXY_METHOD_FORWARD = b'\x01'      # ForwardRequest → call Forward(), block for response
PROXY_METHOD_PUSH_RESULT = b'\x02'  # ForwardResponse → call PushResult()
PROXY_METHOD_FIRE_FORGET = b'\x03'  # ForwardRequest → ACK immediately, Forward() in background


def _proxy_handler_loop(
    *,
    stop_event: threading.Event,
    p2p_node: Any,
    service: Any,
) -> None:
    """Receive inbound proxy requests from libp2p and demux to Forward/PushResult.

    Messages carry a 1-byte method prefix:
      0x01 = ForwardRequest  → service.Forward()
      0x02 = ForwardResponse → service.PushResult()
    Legacy messages without prefix are treated as ForwardRequest (backward compat).
    """
    from peer import peer_pb2
    logging.info("proxy_handler_loop started")
    while not stop_event.is_set():
        try:
            pending = p2p_node.poll_proxy_request(timeout_ms=500)
            if pending is None:
                continue
            req_id, raw_bytes = pending
            raw = bytes(raw_bytes)
            try:
                # Demultiplex by method prefix byte.
                if raw and raw[0:1] == PROXY_METHOD_FIRE_FORGET:
                    # Fire-and-forget: ACK immediately, run Forward() in background.
                    # Decouples relay circuit lifetime from inference duration.
                    p2p_node.respond_proxy(request_id=req_id, data=PROXY_METHOD_FIRE_FORGET)
                    _ff_request = peer_pb2.ForwardRequest()
                    _ff_request.ParseFromString(raw[1:])

                    # ── Ring token emission on coordinator node ──
                    # The ring handler on the last shard emits tokens and adds them
                    # to ring_generated_ids. When the loop-back arrives here (stage 0,
                    # ring_mode=True), the coordinator should emit the latest token
                    # to its local ring queue so the inference_service can drain it.
                    if (bool(getattr(_ff_request, "ring_mode", False))
                            and _ff_request.stage_index == 0
                            and _ff_request.ring_generated_ids):
                        try:
                            from coordinator.push_receiver import emit_ring_token
                            _ring_cb = str(getattr(_ff_request, "final_callback_request_id", "") or "")
                            _latest_token = int(_ff_request.ring_generated_ids[-1])
                            emit_ring_token(_ring_cb, _latest_token)
                            logging.info("ring_token_emitted_on_coordinator: token=%d remaining=%d",
                                         _latest_token, int(_ff_request.ring_tokens_remaining))

                            # If ring is done (remaining==0 or EOS), emit sentinel.
                            _ring_eos = set(int(e) for e in _ff_request.ring_eos_ids)
                            if _ff_request.ring_tokens_remaining <= 0 or _latest_token in _ring_eos:
                                emit_ring_token(_ring_cb, None)
                                logging.info("ring_complete_on_coordinator: tokens=%d",
                                             len(_ff_request.ring_generated_ids))
                                # Don't process further — ring is done.
                                p2p_node.respond_proxy(
                                    request_id=req_id, data=PROXY_METHOD_FIRE_FORGET)
                                continue
                        except Exception as _emit_exc:
                            logging.warning("ring_emit_coordinator_failed: %s", _emit_exc)

                    def _ff_process(_req=_ff_request):
                        try:
                            logging.info("ASYNC_THREAD_START: req=%s stage=%d ring=%s",
                                         _req.request_id, _req.stage_index,
                                         bool(getattr(_req, "ring_mode", False)))
                            service.Forward(_req, context=None)
                            logging.info("ASYNC_THREAD_DONE: req=%s stage=%d",
                                         _req.request_id, _req.stage_index)
                        except Exception as _ff_exc:
                            logging.error("ASYNC_THREAD_CRASH: req=%s stage=%d err=%s",
                                          _req.request_id, _req.stage_index, _ff_exc,
                                          exc_info=True)
                    threading.Thread(target=_ff_process, daemon=True).start()
                elif raw and raw[0:1] == PROXY_METHOD_PUSH_RESULT:
                    # PushResult path: ForwardResponse → PushResult RPC.
                    push_resp = peer_pb2.ForwardResponse()
                    push_resp.ParseFromString(raw[1:])
                    ack = service.PushResult(push_resp, context=None)
                    p2p_node.respond_proxy(
                        request_id=req_id,
                        data=PROXY_METHOD_PUSH_RESULT + ack.SerializeToString(),
                    )
                else:
                    # Forward path (0x01 prefix or legacy no-prefix).
                    payload = raw[1:] if raw and raw[0:1] == PROXY_METHOD_FORWARD else raw
                    request = peer_pb2.ForwardRequest()
                    request.ParseFromString(payload)
                    response = service.Forward(request, context=None)
                    p2p_node.respond_proxy(
                        request_id=req_id,
                        data=PROXY_METHOD_FORWARD + response.SerializeToString(),
                    )
            except Exception as e:
                logging.warning("proxy_handler_error: req=%s err=%s", req_id, e)
                err_resp = peer_pb2.ForwardResponse(
                    error=f"proxy_handler_failed: {e}",
                )
                p2p_node.respond_proxy(
                    request_id=req_id,
                    data=PROXY_METHOD_FORWARD + err_resp.SerializeToString(),
                )
        except Exception as e:
            if not stop_event.is_set():
                logging.warning("proxy_handler_poll_error: %s", e)
                time.sleep(1.0)
    logging.info("proxy_handler_loop stopped")


def _coordinator_proxy_handler_loop(
    *,
    stop_event: threading.Event,
    p2p_node: Any,
    pipeline_depth: int = 1,
) -> None:
    """Minimal proxy handler for pure-coordinator mode (``--no-local-peer``).

    Runs when the coordinator has no local peer (and therefore no
    PeerService) but DOES need to receive inbound libp2p proxy requests
    — specifically ``PROXY_METHOD_PUSH_RESULT`` messages delivering
    hidden states from the last peer in a Path A ring.

    Dispatches:
      * ``PROXY_METHOD_PUSH_RESULT`` → sample on coordinator
        (HeadSampler + RingSession + emit_ring_token + re-inject).
      * Any other method byte → ACK with an error response so the
        sender's proxy_forward unblocks and surfaces the mismatch in
        its own log.

    Concurrency (Phase 2a):
        Under ``pipeline_depth >= 2``, each inbound PushResult is
        dispatched to a worker in ``ThreadPoolExecutor(max_workers=
        max(2, pipeline_depth))`` instead of being processed in-line.
        That way two concurrent PushResults for different slots of
        the same ring don't queue serially behind each other's
        ~10 ms head-sample step. The workers themselves serialise
        the per-ring compound state-transition op via
        ``RingSession.lock`` (see ``_coordinator_handle_push_result``).

        Under ``pipeline_depth == 1`` (default) the worker pool is
        skipped entirely and dispatch happens in-line on the polling
        thread — byte-identical to the pre-Phase-2a single-threaded
        loop.

    This is the coord-only analog of ``_proxy_handler_loop``. Without
    it, libp2p ``request_response`` times out on the inbound side
    because nothing in the Python process calls
    ``p2p_node.poll_proxy_request`` and the corresponding
    ``respond_proxy``. Observed as
    ``proxy inbound failure error=Timeout`` on Mac and
    ``push_result_failed: proxy outbound: Timeout`` on the remote peer.
    """
    from peer import peer_pb2

    # Worker pool (Phase 2a) — only spawned when pipeline_depth >= 2.
    # ``max(2, pipeline_depth)`` because under depth=2 two PushResults
    # can be in flight; the +1 of headroom doesn't hurt and keeps the
    # pool small (each worker handles a ~10 ms sample + ~30 ms libp2p
    # fire so 2-4 workers covers any realistic depth).
    _worker_pool: Any = None
    if int(pipeline_depth) >= 2:
        _worker_pool = futures.ThreadPoolExecutor(
            max_workers=max(2, int(pipeline_depth)),
            thread_name_prefix="oh-coord-pushresult",
        )
        logging.info(
            "coordinator_proxy_handler_loop started: pipeline_depth=%d "
            "worker_pool=%d", pipeline_depth, max(2, int(pipeline_depth)),
        )
    else:
        logging.info(
            "coordinator_proxy_handler_loop started: serial mode "
            "(pipeline_depth=1)"
        )

    def _dispatch_push_result(req_id: str, raw: bytes) -> None:
        """Handle one PROXY_METHOD_PUSH_RESULT message.

        Runs either in-line (depth=1) or on a worker (depth>=2). The
        worker version is what gives us the read-decide-write race
        ``RingSession.lock`` was designed to prevent.
        """
        try:
            push_resp = peer_pb2.ForwardResponse()
            push_resp.ParseFromString(raw[1:])
            ack = _coordinator_handle_push_result(
                response=push_resp,
                p2p_node=p2p_node,
            )
            p2p_node.respond_proxy(
                request_id=req_id,
                data=PROXY_METHOD_PUSH_RESULT + ack.SerializeToString(),
            )
        except Exception as exc:  # pragma: no cover — defensive
            import traceback as _tb
            logging.warning(
                "coord_proxy_worker_failed: req=%s err=%s\n%s",
                req_id, exc, _tb.format_exc(),
            )
            try:
                err = peer_pb2.PushAck(
                    request_id="", ok=False, error=f"dispatch_failed: {exc}",
                )
                p2p_node.respond_proxy(
                    request_id=req_id,
                    data=PROXY_METHOD_PUSH_RESULT + err.SerializeToString(),
                )
            except Exception:
                pass

    try:
        while not stop_event.is_set():
            try:
                pending = p2p_node.poll_proxy_request(timeout_ms=500)
                if pending is None:
                    continue
                req_id, raw_bytes = pending
                raw = bytes(raw_bytes)
                try:
                    if raw and raw[0:1] == PROXY_METHOD_PUSH_RESULT:
                        # Pipelined dispatch: hand off to a worker so the
                        # poll loop can drain the next message immediately.
                        # Serial dispatch: process in-line, byte-identical
                        # to pre-Phase-2a behaviour.
                        if _worker_pool is not None:
                            _worker_pool.submit(_dispatch_push_result, req_id, raw)
                        else:
                            _dispatch_push_result(req_id, raw)
                    else:
                        # Unknown method in pure-coord mode — the coordinator
                        # doesn't serve Forward() (no peer). Reply with a
                        # small error and move on.
                        err = peer_pb2.PushAck(
                            request_id="",
                            ok=False,
                            error="pure_coordinator_unsupported_proxy_method",
                        )
                        logging.warning(
                            "coord_proxy_unsupported_method: byte=%s req=%s",
                            raw[0:1].hex() if raw else "empty", req_id,
                        )
                        p2p_node.respond_proxy(
                            request_id=req_id,
                            data=PROXY_METHOD_PUSH_RESULT + err.SerializeToString(),
                        )
                except Exception as exc:  # pragma: no cover — defensive
                    import traceback as _tb
                    logging.warning(
                        "coord_proxy_dispatch_failed: req=%s err=%s\n%s",
                        req_id, exc, _tb.format_exc(),
                    )
                    try:
                        err = peer_pb2.PushAck(
                            request_id="", ok=False, error=f"dispatch_failed: {exc}",
                        )
                        p2p_node.respond_proxy(
                            request_id=req_id,
                            data=PROXY_METHOD_PUSH_RESULT + err.SerializeToString(),
                        )
                    except Exception:
                        pass
            except Exception as exc:
                if not stop_event.is_set():
                    logging.warning("coord_proxy_poll_error: %s", exc)
                    time.sleep(1.0)
    finally:
        if _worker_pool is not None:
            try:
                _worker_pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
    logging.info("coordinator_proxy_handler_loop stopped")


def _coordinator_handle_push_result(
    *,
    response: Any,
    p2p_node: Any,
) -> Any:
    """Shared PushResult handler — used by both
    :class:`PeerService.PushResult` and the pure-coordinator proxy
    handler loop. Kept as a free function so the coord-only path
    doesn't need a PeerService instance.

    Returns a ``PushAck`` reflecting whether the sample + re-inject
    succeeded.

    Concurrency model (Phase 2a):
        Called from N concurrent worker threads under
        ``pipeline_depth >= 2`` (the coord-side proxy handler pool
        dispatches each inbound PushResult to a fresh worker so two
        slots can be sampled in parallel).

        The compound op
            (state transition + token append + remaining decrement +
             EOS check + in-flight count + next_slot_id reservation)
        runs under ``session.lock`` to keep it atomic. The fire-and-
        forget re-inject (a slow ~20-50 ms libp2p proxy_forward) and
        the ``emit_ring_token`` queue write happen AFTER lock release
        so they never serialise concurrent threads on different slots
        of the same ring. Idempotent on duplicate PushResults — late
        arrivals for already-sampled slots short-circuit silently.

        Under ``pipeline_depth == 1`` (default) the lock is acquired
        but uncontended (~10 ns); the slots dict stays empty (legacy
        path tracks no per-slot state); behaviour is byte-identical
        to the pre-Phase-2a serial ring.
    """
    from peer import peer_pb2
    from coordinator.head_sampler import (
        get_head_sampler, get_ring_session, unregister_ring_session,
        SlotState,
        SLOT_STATE_AWAITING_SAMPLE, SLOT_STATE_SAMPLED,
        SLOT_STATES_FINAL, SLOT_STATES_IN_FLIGHT,
    )
    from coordinator.push_receiver import emit_ring_token

    _is_hidden = bool(getattr(response, "is_hidden_state", False))
    _slot_id = int(getattr(response, "slot_id", 0) or 0)
    _block_size = int(getattr(response, "block_size", 0) or 0)
    _block_index = int(getattr(response, "block_index", 0) or 0)
    logging.info(
        "coord_push_result_received: req=%s slot=%d is_hidden_state=%s "
        "block_size=%d block_index=%d from=%s",
        response.request_id, _slot_id, _is_hidden,
        _block_size, _block_index, response.peer_id,
    )
    if not _is_hidden:
        # Pure-coord mode only handles hidden-state responses. A
        # non-hidden PushResult in this mode is unexpected — ACK and
        # move on.
        return peer_pb2.PushAck(
            request_id=response.request_id, ok=True, error="",
        )

    # ── Phase 2b live-bench Binding #2: block-verify routing ────────
    # When block_size > 0, the response is the multi-position hidden
    # state block from a DFlash verify pass. Route the activation_packed
    # bytes to the per-(request_id, block_index) queue the multi-peer
    # transport is waiting on. Short-circuit the per-token sampler —
    # block-verify outcomes are sampled by HeadSampler.verify_block on
    # the transport-receiving thread, not here.
    if _block_size > 0:
        from coordinator.push_receiver import emit_dflash_block_response
        _packed = bytes(getattr(response, "activation_packed", b"") or b"")
        if not _packed:
            logging.error(
                "coord_block_verify_empty_payload: req=%s block=%d",
                response.request_id, _block_index,
            )
            return peer_pb2.PushAck(
                request_id=response.request_id, ok=False,
                error="block_verify_empty_payload",
            )
        delivered = emit_dflash_block_response(
            request_id=str(response.request_id),
            block_index=_block_index,
            activation_packed=_packed,
            block_size=_block_size,
        )
        if not delivered:
            # No transport waiting — late arrival or stray response.
            # Drop silently; the transport's timeout already fired.
            logging.warning(
                "coord_block_verify_no_queue: req=%s block=%d "
                "(late arrival or transport timed out)",
                response.request_id, _block_index,
            )
        return peer_pb2.PushAck(
            request_id=response.request_id, ok=True, error="",
        )

    sampler = get_head_sampler()
    if sampler is None:
        logging.error(
            "coord_push_result_no_sampler: req=%s", response.request_id,
        )
        return peer_pb2.PushAck(
            request_id=response.request_id, ok=False,
            error="no_head_sampler_registered",
        )

    session = get_ring_session(str(response.request_id))
    if session is None:
        logging.error(
            "coord_push_result_no_session: req=%s", response.request_id,
        )
        return peer_pb2.PushAck(
            request_id=response.request_id, ok=False,
            error="no_ring_session_registered",
        )

    # ── Pipelined-mode idempotency check (no-op when pipeline_depth == 1) ──
    # Late-arriving / duplicate PushResults for an already-sampled slot
    # are dropped silently. Done OUTSIDE the network/sampler work so a
    # duplicate doesn't waste a head-matmul cycle.
    if session.pipeline_depth >= 2 and _slot_id in session.slots:
        with session.lock:
            _existing = session.slots.get(_slot_id)
            if _existing is not None and _existing.state in SLOT_STATES_FINAL:
                logging.info(
                    "coord_push_result_duplicate_dropped: req=%s slot=%d state=%s",
                    response.request_id, _slot_id, _existing.state,
                )
                return peer_pb2.PushAck(
                    request_id=response.request_id, ok=True, error="",
                )

    _packed = bytes(getattr(response, "activation_packed", b"") or b"")
    _activation_list: list[float] | None = None
    if not _packed:
        _activation_list = list(response.activation)
        if not _activation_list:
            return peer_pb2.PushAck(
                request_id=response.request_id, ok=False,
                error="empty_hidden_state",
            )

    # ── Sampler runs OUTSIDE session.lock ───────────────────────────
    # Sampling on coord CPU is ~10 ms (norm + lm_head + softmax + arg-
    # max). Holding ``session.lock`` here would serialise all worker
    # threads sampling for the SAME session, defeating the point of
    # the worker pool. The sampler is read-only on session state
    # (``session.decode`` is immutable after registration).
    try:
        token_id = int(sampler.sample(
            _activation_list if _activation_list is not None else [],
            session.decode,
            packed_bytes=_packed if _packed else None,
        ))
    except Exception as exc:
        import traceback as _tb
        logging.error(
            "coord_push_result_sample_failed: req=%s slot=%d err=%s\n%s",
            response.request_id, _slot_id, exc, _tb.format_exc(),
        )
        emit_ring_token(session.callback_request_id, None)
        unregister_ring_session(response.request_id)
        return peer_pb2.PushAck(
            request_id=response.request_id, ok=False,
            error=f"sample_failed:{exc}",
        )

    # ── ATOMIC: state mutation + termination check + reinject reservation ──
    # Decide-then-act: under the lock we (a) record the sampled token,
    # (b) check termination, (c) compute in-flight count, (d) reserve
    # the next slot_id if we're going to fire. Network ops + queue
    # writes happen AFTER releasing the lock, taking only the small
    # ``reinject_args`` snapshot the closure needs.
    _now_ms = time.monotonic() * 1000.0

    with session.lock:
        # Slot bookkeeping (skipped when pipeline_depth == 1 to keep
        # the legacy path byte-identical).
        if session.pipeline_depth >= 2:
            slot = session.slots.get(_slot_id)
            if slot is None:
                # PushResult arrived for an unknown slot — could happen if
                # the initial fire never registered (race during first
                # send) or if the proto field defaulted to 0 on a peer
                # that didn't propagate it. Fall back to creating an
                # implicit slot rather than crashing; logged for diagnosis.
                slot = SlotState(
                    slot_id=_slot_id,
                    state=SLOT_STATE_AWAITING_SAMPLE,
                    dispatched_at_ms=_now_ms,
                    last_update_ms=_now_ms,
                )
                session.slots[_slot_id] = slot
                logging.warning(
                    "coord_push_result_implicit_slot: req=%s slot=%d "
                    "(no prior dispatch record — backfilled)",
                    response.request_id, _slot_id,
                )
            elif slot.state in SLOT_STATES_FINAL:
                # Race: another worker finalised this slot between the
                # idempotency check above and lock acquisition. Drop.
                logging.info(
                    "coord_push_result_duplicate_inside_lock: req=%s slot=%d "
                    "state=%s", response.request_id, _slot_id, slot.state,
                )
                return peer_pb2.PushAck(
                    request_id=response.request_id, ok=True, error="",
                )
            slot.state = SLOT_STATE_SAMPLED
            slot.token_id = token_id
            slot.last_update_ms = _now_ms

        session.ring_generated_ids.append(token_id)
        session.ring_tokens_remaining = max(0, session.ring_tokens_remaining - 1)

        # Termination INSIDE the lock so two threads can't both decide
        # "remaining > 0, fire next" when remaining == 1.
        _is_eos = token_id in session.ring_eos_ids
        is_done = (session.ring_tokens_remaining <= 0) or _is_eos

        # In-flight count INSIDE the lock — sees the just-recorded
        # state transition above so this slot isn't double-counted as
        # in-flight.
        if session.pipeline_depth >= 2:
            in_flight = sum(
                1 for s in session.slots.values()
                if s.state in SLOT_STATES_IN_FLIGHT
            )
            fire_next = (not is_done) and (in_flight < session.pipeline_depth)
        else:
            # Serial mode — fire iff not done.
            fire_next = not is_done
            in_flight = 0  # for logging only

        # Reserve next slot_id INSIDE the lock so no two threads claim
        # the same id. The reinject ITSELF fires after lock release.
        next_slot_id_reserved: Optional[int] = None
        if fire_next and session.pipeline_depth >= 2:
            next_slot_id_reserved = session.next_slot_id
            session.next_slot_id += 1
            session.slots[next_slot_id_reserved] = SlotState(
                slot_id=next_slot_id_reserved,
                state="dispatched",  # SLOT_STATE_DISPATCHED
                dispatched_at_ms=_now_ms,
                last_update_ms=_now_ms,
            )

        # Snapshot the few fields the reinject closure needs so the
        # closure can run after lock release. The fields are immutable
        # after RingSession construction (route, callback ids, decode
        # config) OR are intentionally captured-by-value here
        # (ring_tokens_remaining, ring_generated_ids).
        if fire_next:
            reinject_snapshot = {
                "request_id": session.request_id,
                "token_id": token_id,
                "next_slot_id": next_slot_id_reserved if next_slot_id_reserved is not None else 0,
                "pipeline_depth": session.pipeline_depth,
                "total_stages": int(session.total_stages),
                "kv_session_id": session.kv_session_id,
                "decode": session.decode,
                "ring_tokens_remaining": int(session.ring_tokens_remaining),
                "ring_generated_ids": list(session.ring_generated_ids),
                "ring_eos_ids": list(session.ring_eos_ids),
                "ring_first_hop_address": session.ring_first_hop_address,
                "ring_first_hop_peer_id": session.ring_first_hop_peer_id,
                "ring_first_hop_libp2p_id": session.ring_first_hop_libp2p_id,
                "ring_full_route": list(session.ring_full_route),
                "final_callback_address": session.final_callback_address,
                "callback_request_id": session.callback_request_id,
                "final_callback_libp2p_peer_id": session.final_callback_libp2p_peer_id,
                "stage0_layer_start": int(session.stage0_layer_start),
                "stage0_layer_end": int(session.stage0_layer_end),
                "stage0_total_layers": int(session.stage0_total_layers),
            }
        else:
            reinject_snapshot = None
    # ── lock released ──────────────────────────────────────────────

    # Network ops + queue writes happen WITHOUT the lock. emit_ring_token
    # is thread-safe (queue.Queue) and is_done check above already
    # serialised termination so two threads can't both emit a sentinel.
    emit_ring_token(session.callback_request_id, token_id)
    logging.info(
        "coord_ring_sampled: req=%s slot=%d token=%d remaining=%d in_flight=%d",
        response.request_id, _slot_id, token_id,
        session.ring_tokens_remaining, in_flight,
    )

    if is_done:
        emit_ring_token(session.callback_request_id, None)
        unregister_ring_session(response.request_id)
        return peer_pb2.PushAck(
            request_id=response.request_id, ok=True, error="",
        )

    if reinject_snapshot is None:
        # Pipeline depth saturated — another in-flight slot will close
        # the loop. Do not fire here.
        return peer_pb2.PushAck(
            request_id=response.request_id, ok=True, error="",
        )

    # ── Re-inject: build + fire-and-forget the next stage-0 request ──
    try:
        snap = reinject_snapshot
        route = snap["ring_full_route"]
        _next_next = route[1].address if len(route) > 1 else ""
        _next_next_id = route[1].peer_id if len(route) > 1 else ""
        req = peer_pb2.ForwardRequest(
            request_id=snap["request_id"],
            activation=[float(snap["token_id"])],
            stage_index=0,
            total_stages=snap["total_stages"],
            max_tokens=1,
            kv_session_id=snap["kv_session_id"],
            kv_store_activation=True,
            kv_use_cached_activation=True,
            decode_do_sample=snap["decode"].do_sample,
            decode_temperature=float(snap["decode"].temperature or 0.0),
            decode_top_p=float(snap["decode"].top_p or 0.0),
            decode_top_k=int(snap["decode"].top_k or 0),
            decode_seed=int(snap["decode"].seed or 0),
            shard_layer_start=snap["stage0_layer_start"],
            shard_layer_end=snap["stage0_layer_end"],
            shard_total_layers=snap["stage0_total_layers"],
            push_mode=True,
            ring_mode=True,
            sample_on_coordinator=True,
            ring_tokens_remaining=snap["ring_tokens_remaining"],
            ring_generated_ids=snap["ring_generated_ids"],
            ring_eos_ids=snap["ring_eos_ids"],
            ring_first_hop_address=snap["ring_first_hop_address"],
            ring_first_hop_peer_id=snap["ring_first_hop_peer_id"],
            ring_first_hop_libp2p_id=snap["ring_first_hop_libp2p_id"],
            ring_full_route=route,
            next_hop_address=_next_next,
            next_hop_peer_id=_next_next_id,
            final_callback_address=snap["final_callback_address"],
            final_callback_request_id=snap["callback_request_id"],
            final_callback_libp2p_peer_id=snap["final_callback_libp2p_peer_id"],
            remaining_route=route[1:],
            slot_id=snap["next_slot_id"],
            pipeline_depth=snap["pipeline_depth"],
        )
        _first_libp2p = snap["ring_first_hop_libp2p_id"]
        _req_id_log = snap["request_id"]
        _slot_id_log = snap["next_slot_id"]

        def _fire(_rreq=req, _libp2p=_first_libp2p,
                  _rid=_req_id_log, _sid=_slot_id_log):
            try:
                if p2p_node is not None and _libp2p:
                    p2p_node.proxy_forward(
                        target_peer_id=_libp2p,
                        data=PROXY_METHOD_FIRE_FORGET + _rreq.SerializeToString(),
                    )
                    logging.info(
                        "coord_reinject_done: req=%s slot=%d via_relay",
                        _rid, _sid,
                    )
                else:
                    logging.error(
                        "coord_reinject_no_route: req=%s slot=%d", _rid, _sid,
                    )
            except Exception as exc:
                logging.error(
                    "coord_reinject_crash: req=%s slot=%d err=%s",
                    _rid, _sid, exc, exc_info=True,
                )

        threading.Thread(target=_fire, daemon=True).start()
    except Exception as exc:
        import traceback as _tb
        logging.error(
            "coord_push_result_reinject_failed: req=%s err=%s\n%s",
            response.request_id, exc, _tb.format_exc(),
        )
        emit_ring_token(session.callback_request_id, None)
        unregister_ring_session(response.request_id)
        return peer_pb2.PushAck(
            request_id=response.request_id, ok=False,
            error=f"reinject_failed:{exc}",
        )

    return peer_pb2.PushAck(
        request_id=response.request_id, ok=True, error="",
    )


class _CoordinatorOnlyPushResultServicer(peer_pb2_grpc.PeerServicer):
    """Minimal gRPC servicer for pure-coordinator mode.

    Implements ONLY the ``PushResult`` RPC — the coord has no shard so
    ``Forward`` / ``Ping`` / etc. are not applicable. Every other method
    returns ``UNIMPLEMENTED``. ``PushResult`` dispatches to the shared
    ``_coordinator_handle_push_result`` helper.
    """

    def PushResult(
        self,
        request: Any,
        context: Any,
    ) -> Any:
        # Use the p2p_node attached to the servicer (set at server
        # start) for the reinject fire-and-forget path.
        return _coordinator_handle_push_result(
            response=request,
            p2p_node=getattr(self, "_p2p_node", None),
        )

    def Forward(self, request, context):  # pragma: no cover
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("pure-coordinator: Forward not served (no shard)")
        return peer_pb2.ForwardResponse(
            error="pure_coordinator_forward_unimplemented",
        )

    def Ping(self, request, context):  # pragma: no cover
        from peer import peer_pb2 as _pp
        return _pp.PingResponse(
            peer_id="coordinator-standalone-head", ok=True, load_pct=0.0,
            daemon_mode="coordinator", geo_nonce_signature="",
        )

    def GetPeerStatus(self, request, context):  # pragma: no cover
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("pure-coordinator: not a peer")
        return peer_pb2.PeerStatusResponse(peer_id="coordinator-standalone-head")


def start_coordinator_grpc_server(
    *,
    host: str = "0.0.0.0",
    port: int = 50050,
    p2p_node: Any = None,
) -> Any:
    """Start a minimal gRPC server for pure-coordinator Path A.

    Binds ``host:port`` and serves ONLY the ``PushResult`` RPC (via
    :class:`_CoordinatorOnlyPushResultServicer`). Used when
    ``--no-local-peer`` is set: the peer thread is skipped but
    LAN-reachable last peers still need a gRPC target for
    ``final_callback_address`` direct pushes.

    Returns the ``grpc.Server`` instance so the caller can stop it on
    SIGTERM.
    """
    from concurrent import futures
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=16),
        options=[
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
        ],
    )
    servicer = _CoordinatorOnlyPushResultServicer()
    servicer._p2p_node = p2p_node  # stash for reinject fire-and-forget
    peer_pb2_grpc.add_PeerServicer_to_server(servicer, server)
    _addr = f"{host}:{port}"
    server.add_insecure_port(_addr)
    server.start()
    logging.info(
        "coordinator_grpc_server_bound: %s — PushResult only",
        _addr,
    )
    return server


def _relay_heartbeat_loop(
    *,
    stop_event: threading.Event,
    relay_channel: Any,
    peer_id: str,
    interval_s: float = 120.0,
) -> None:
    """Send periodic heartbeats to the relay to keep registration alive.

    The RelayServer expires registrations after 300s without activity.
    This thread sends a Ping every 120s — well within the timeout.
    """
    from peer import peer_pb2, peer_pb2_grpc
    stub = peer_pb2_grpc.PeerStub(relay_channel)
    while not stop_event.is_set():
        try:
            stub.Ping(
                peer_pb2.PingRequest(sent_unix_ms=int(time.time() * 1000)),
                timeout=10.0,
                metadata=[("x-openhydra-peer-id", peer_id)],
            )
        except Exception as exc:
            logging.warning("relay_heartbeat_failed: peer=%s err=%s", peer_id, exc)
        stop_event.wait(interval_s)


def _resolve_quantization_mode(quantization: str, quantization_mode: str | None) -> str:
    legacy = str(quantization_mode or "").strip().lower()
    if legacy in {"fp32", "int8", "int4"}:
        return legacy
    normalized = str(quantization or "none").strip().lower()
    return _QUANTIZATION_FLAG_TO_MODE.get(normalized, "fp32")


def _resolve_deployment_security_settings(parser: argparse.ArgumentParser, args: argparse.Namespace) -> dict[str, str]:
    profile = str(getattr(args, "deployment_profile", "dev") or "dev").strip().lower()
    if profile not in {"dev", "prod"}:
        parser.error("unsupported deployment profile")

    try:
        secret_store = load_secret_store(getattr(args, "secrets_file", None))
    except RuntimeError as exc:
        parser.error(str(exc))

    advanced_seed = str(getattr(args, "advanced_encryption_seed", "") or "").strip()
    geo_seed = str(getattr(args, "geo_challenge_seed", "") or "").strip()

    if is_insecure_secret_value(advanced_seed):
        advanced_seed = str(secret_store.get("OPENHYDRA_ADVANCED_ENCRYPTION_SEED", advanced_seed) or "").strip()
    if is_insecure_secret_value(geo_seed):
        geo_seed = str(secret_store.get("OPENHYDRA_GEO_CHALLENGE_SEED", geo_seed) or "").strip()

    if profile == "prod":
        if not bool(getattr(args, "tls_enable", False)):
            parser.error("prod profile requires --tls-enable")
        if not bool(getattr(args, "tls_require_client_auth", False)):
            parser.error("prod profile requires --tls-require-client-auth")
        if not getattr(args, "tls_cert_path", None):
            parser.error("prod profile requires --tls-cert-path")
        if not getattr(args, "tls_key_path", None):
            parser.error("prod profile requires --tls-key-path")
        if not getattr(args, "tls_client_ca_path", None):
            parser.error("prod profile requires --tls-client-ca-path")
        if is_insecure_secret_value(advanced_seed):
            parser.error(
                "prod profile requires a strong encryption seed via "
                "--advanced-encryption-seed or OPENHYDRA_ADVANCED_ENCRYPTION_SEED"
            )
        if is_insecure_secret_value(geo_seed):
            parser.error(
                "prod profile requires a strong geo challenge seed via "
                "--geo-challenge-seed or OPENHYDRA_GEO_CHALLENGE_SEED"
            )

    return {
        "deployment_profile": profile,
        "advanced_encryption_seed": advanced_seed or str(getattr(args, "advanced_encryption_seed")),
        "geo_challenge_seed": geo_seed or str(getattr(args, "geo_challenge_seed")),
    }


class PeerService(peer_pb2_grpc.PeerServicer):
    def __init__(
        self,
        peer_id: str,
        model_id: str,
        shard_index: int,
        total_shards: int,
        daemon_mode: str,
        broken: bool,
        initial_resource_budget: ResourceBudget | None = None,
        advanced_encryption_enabled: bool = False,
        advanced_encryption_seed: str = "openhydra-tier3-dev-seed",
        kv_cache_max_entries: int = 1024,
        runtime_backend: str = "toy_auto",
        runtime_target: str = "auto",
        quantization_mode: str = "fp32",
        runtime_model_id: str = "Qwen/Qwen3.5-0.8B",
        hf_model_id: str = "",
        mlx_force_hf_tokenizer: bool = True,
        tokenizer_vocab_guard: bool = True,
        tensor_autoencoder_enabled: bool = False,
        tensor_autoencoder_latent_dim: int = 1024,
        privacy_noise_variance: float = 0.0,
        geo_challenge_seed: str = "openhydra-geo-dev-seed",
        expert_tags: tuple[str, ...] = (),
        expert_layer_indices: tuple[int, ...] = (),
        expert_router: bool = False,
        peer_public_key: str = "",
        peer_private_key: Any = None,
        kv_compaction_enabled: bool = False,
        kv_compaction_method: str = "hak",
        kv_compaction_ratio: float = 0.10,
        kv_compaction_beta: bool = False,
        kv_compaction_head_budget_path: str = "",
        kv_compaction_online: bool = False,
        kv_compaction_online_max_tokens: int = 512,
        kv_compaction_mode: str = "off",
        kv_compaction_auto_threshold: int = 512,
        kv_radix_cache_enabled: bool = False,
        kv_radix_cache_max_entries: int = 128,
        kv_radix_cache_min_prefix_len: int = 16,
        warmup_on_start: bool = False,
        mlx_eval_timeout_s: float = 120.0,
        batch_window_ms: float = 50.0,
        max_batch_size: int = 8,
        p2p_node: Any | None = None,
        load_full_head: bool = False,
        pipeline_depth: int = 1,
    ):
        self.peer_id = peer_id
        self._p2p_node = p2p_node
        self.model_id = model_id
        self.shard_index = shard_index
        self.total_shards = total_shards
        self.daemon_mode = daemon_mode
        self.advanced_encryption_enabled = bool(advanced_encryption_enabled)
        self.advanced_encryption_seed = str(advanced_encryption_seed)
        self.geo_challenge_seed = str(geo_challenge_seed)
        if self.advanced_encryption_enabled and not cryptography_available():
            raise RuntimeError("cryptography_not_available: install 'cryptography>=42'")
        self.shard = ModelShard(
            ToyShardConfig(
                model_id=model_id,
                shard_index=shard_index,
                total_shards=total_shards,
                broken=broken,
                runtime_backend=str(runtime_backend),
                runtime_target=str(runtime_target),
                quantization_mode=str(quantization_mode),
                runtime_model_id=str(runtime_model_id),
                runtime_hf_model_id=str(hf_model_id or ""),
                runtime_mlx_force_hf_tokenizer=bool(mlx_force_hf_tokenizer),
                runtime_tokenizer_vocab_guard=bool(tokenizer_vocab_guard),
                runtime_layer_indices=tuple(expert_layer_indices),
                runtime_kv_cache_max_entries=max(1, int(kv_cache_max_entries)),
                runtime_kv_compaction_enabled=bool(kv_compaction_enabled),
                runtime_kv_compaction_method=str(kv_compaction_method or "hak"),
                runtime_kv_compaction_ratio=max(0.01, min(1.0, float(kv_compaction_ratio))),
                runtime_kv_compaction_beta=bool(kv_compaction_beta),
                runtime_kv_compaction_head_budget_path=str(kv_compaction_head_budget_path or ""),
                runtime_kv_compaction_online=bool(kv_compaction_online),
                runtime_kv_compaction_online_max_tokens=max(4, int(kv_compaction_online_max_tokens)),
                runtime_kv_compaction_mode=str(kv_compaction_mode),
                runtime_kv_compaction_auto_threshold=max(1, int(kv_compaction_auto_threshold)),
                runtime_kv_radix_cache_enabled=bool(kv_radix_cache_enabled),
                runtime_kv_radix_cache_max_entries=max(1, int(kv_radix_cache_max_entries)),
                runtime_kv_radix_cache_min_prefix_len=max(1, int(kv_radix_cache_min_prefix_len)),
                runtime_warmup_on_start=bool(warmup_on_start),
                runtime_mlx_eval_timeout_s=max(1.0, float(mlx_eval_timeout_s)),
                runtime_tensor_autoencoder_enabled=bool(tensor_autoencoder_enabled),
                runtime_tensor_autoencoder_latent_dim=max(1, int(tensor_autoencoder_latent_dim)),
                runtime_privacy_noise_variance=max(0.0, float(privacy_noise_variance)),
                runtime_privacy_audit_seed=str(advanced_encryption_seed),
                runtime_peer_id=str(peer_id),
                runtime_load_full_head=bool(load_full_head),
                runtime_pipeline_depth=max(1, int(pipeline_depth or 1)),
            )
        )
        # Persist boot-time ToyShardConfig inputs so ``reload_shard`` can
        # rebuild a configured-identical shard with just the layer range
        # rotated. Without this, every reshard silently resets the runtime
        # to ``toy_auto`` (because ``runtime_backend or "toy_auto"`` in the
        # ModelShard dispatcher kicks in on the empty fallback), swapping
        # the production PyTorch/MLX backend for tinyllama-15M — the bug
        # that surfaced as ``IndexError: index out of range in self`` on
        # GPU1 stage 0 during the April 22 cross-ISP benchmark.
        self._boot_runtime_backend = str(runtime_backend)
        self._boot_runtime_target = str(runtime_target)
        self._boot_quantization_mode = str(quantization_mode)
        self._boot_runtime_model_id = str(runtime_model_id)
        self._boot_hf_model_id = str(hf_model_id or "")
        self._boot_mlx_force_hf_tokenizer = bool(mlx_force_hf_tokenizer)
        self._boot_tokenizer_vocab_guard = bool(tokenizer_vocab_guard)
        self._boot_kv_cache_max_entries = max(1, int(kv_cache_max_entries))
        self._boot_warmup_on_start = bool(warmup_on_start)
        self._boot_mlx_eval_timeout_s = max(1.0, float(mlx_eval_timeout_s))
        self._boot_load_full_head = bool(load_full_head)
        # Phase 2a: persist pipeline_depth so reload_shard preserves the
        # async-pipeline executor sizing across resharding events.
        self._boot_pipeline_depth = max(1, int(pipeline_depth or 1))
        self.runtime_profile = dict(self.shard.runtime_profile())
        # Path A (client-terminated pipeline): register this peer's runtime
        # with the coordinator-side HeadSampler when the shard owns the
        # last transformer layer. Idempotent — safe to re-register on
        # reshard. When coordinator and peer share a process (the common
        # setup), the HeadSampler borrows weights in place with no copy.
        self._maybe_register_head_source()
        self.batch_queue = BatchingQueue(
            self.shard,
            batch_window_ms=float(batch_window_ms),
            max_batch_size=max(1, int(max_batch_size)),
        )
        self.expert_tags = tuple(str(tag).strip().lower() for tag in expert_tags if str(tag).strip())
        self.expert_layer_indices = tuple(sorted({int(idx) for idx in expert_layer_indices if int(idx) >= 0}))
        self.expert_router = bool(expert_router)
        self.peer_public_key = str(peer_public_key or "")
        self._peer_private_key = peer_private_key
        self._inflight = 0
        self._lock = threading.Lock()
        self._resource_budget = initial_resource_budget or ResourceBudget(
            vram_fraction=1.0,
            cpu_fraction=1.0,
            should_yield=False,
            reason="default",
        )
        self.kv_cache_max_entries = max(1, int(kv_cache_max_entries))
        # Phase 2A: Server-to-server RTT measurements.
        # Populated after Forward() calls by pinging the next hop.
        # Keyed by downstream peer_id → measured RTT in ms.
        self._next_hop_rtts: dict[str, float] = {}
        self._next_hop_rtts_lock = threading.Lock()
        self._kv_cache: dict[str, list[float]] = {}
        self.last_request_thread_id: int | None = None
        self.last_inference_thread_id: int | None = None
        self.onion_layers_peeled = 0
        self.last_onion_next_peer_id: str | None = None
        self.onion_next_peer_history: list[str] = []

    # ── Path A: coordinator-side HeadSampler registration ────────────
    def _maybe_register_head_source(self) -> None:
        """Register this peer's runtime as the coordinator's head source
        when the shard owns the last transformer layer.

        Safe to call multiple times (idempotent — last caller wins).
        No-op when the shard is not the last one in the pipeline; the
        coordinator falls back to today's sample-on-peer path in that case.
        """
        try:
            shard = self.shard
            if shard is None:
                return
            runtime = getattr(shard, "_runtime", None)
            if runtime is None:
                return
            # Heuristic: accept any runtime that exposes apply_final_head
            # and reports it owns the last layer. Both MLXRuntime and
            # PyTorchRuntime gained apply_final_head in Phase 2.
            if not hasattr(runtime, "apply_final_head"):
                return
            # Phase 5: prefer ``_has_final_head`` — a runtime-level
            # advertisement that the head weights (norm + lm_head / tied
            # embed) are actually loaded on this shard. This is True for
            # (a) last-shard peers (today's behaviour) AND (b) any peer
            # launched with ``runtime_load_full_head=True`` (Path A
            # Phase 5 — lets the Mac-stage-0 coordinator borrow head
            # weights from its co-located first-shard peer).
            if hasattr(runtime, "_has_final_head"):
                if not bool(getattr(runtime, "_has_final_head", False)):
                    return
            elif hasattr(runtime, "_is_last_shard"):
                # Runtimes without the new attribute fall back to the
                # pre-Phase-5 gate.
                if not bool(getattr(runtime, "_is_last_shard", False)):
                    return
            else:
                # PyTorch runtime: check the DecoderArchitecture's lm_head.
                arch = getattr(runtime, "_model", None)
                if arch is not None and hasattr(arch, "_lm_head"):
                    if getattr(arch, "_lm_head", None) is None:
                        return
            from coordinator.head_sampler import register_head_source
            register_head_source(peer_id=str(self.peer_id), runtime=runtime)
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("head_sampler_register_skipped: %s", exc)

    def inflight_count(self) -> int:
        with self._lock:
            return self._inflight

    def set_resource_budget(self, budget: ResourceBudget) -> None:
        with self._lock:
            self._resource_budget = budget

    # ── B3 ReshardExecutor hooks ──────────────────────────────────────
    #
    # Both methods are synchronous and intentionally minimal. The
    # executor is responsible for ordering (drain before unload, etc.);
    # these methods just do the work.

    def teardown_shard(self) -> None:
        """Drop the current ``self.shard`` and free its memory.

        PyTorch path: ``del shard`` then ``torch.cuda.empty_cache()`` if
        CUDA is available.
        MLX path: ``mx.metal.clear_cache()`` if MLX is present.

        The ``self.shard`` attribute is set to ``None`` on exit; the
        caller must invoke :meth:`reload_shard` before accepting any
        new ``Forward`` requests.
        """
        with self._lock:
            old = self.shard
            self.shard = None  # type: ignore[assignment]
        # Drop the Python reference *outside* the lock so GC can proceed
        # without blocking other calls.
        del old
        try:
            import torch  # noqa: F401
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            import mlx.core as mx  # noqa: F401
            mx.metal.clear_cache()
        except Exception:
            pass
        # Also drop any lingering KV-cache session data so the new shard
        # starts fresh (session ids keyed on the old shard shape would
        # otherwise poison the first inference).
        with self._lock:
            if hasattr(self, "_kv_cache"):
                try:
                    self._kv_cache.clear()
                except Exception:
                    pass

    def reload_shard(self, assignment: Any) -> None:
        """Rebuild ``self.shard`` with a new layer-range ``assignment``.

        ``assignment`` must expose ``model_id``, ``layer_start``,
        ``layer_end``, ``total_layers`` — the
        :class:`peer.swarm_negotiator.ShardAssignment` shape.

        Raises any exception from :class:`ModelShard` construction so
        :class:`peer.reshard_executor.ReshardExecutor` can catch it and
        transition to ``LOADING_FAILED`` (stay-degraded policy).
        """
        # Compose a new ToyShardConfig that mirrors the one used at
        # ``__init__`` time but with the new layer indices.
        new_layer_indices = tuple(
            range(int(assignment.layer_start), int(assignment.layer_end))
        )
        total_layers = int(assignment.total_layers)
        # Derive shard_index / total_shards from the layer range so
        # peers with mixed shard sizes still self-describe correctly.
        slice_size = max(1, int(assignment.layer_end - assignment.layer_start))
        total_shards = max(1, total_layers // slice_size)
        shard_index = int(assignment.layer_start) // slice_size

        new_shard = ModelShard(
            ToyShardConfig(
                model_id=str(assignment.model_id or getattr(self, "model_id", "")),
                shard_index=shard_index,
                total_shards=total_shards,
                # Preserve the original runtime wiring — without this the
                # ModelShard dispatcher falls back to ``toy_auto`` (tinyllama-15M)
                # on every reshard, silently swapping out the production
                # PyTorch / MLX runtime. See the ``_boot_*`` fields on
                # ``PeerService.__init__``.
                runtime_backend=str(getattr(self, "_boot_runtime_backend", "") or ""),
                runtime_target=str(getattr(self, "_boot_runtime_target", "auto") or "auto"),
                quantization_mode=str(getattr(self, "_boot_quantization_mode", "fp32") or "fp32"),
                runtime_model_id=str(getattr(self, "_boot_runtime_model_id", "") or ""),
                runtime_hf_model_id=str(getattr(self, "_boot_hf_model_id", "") or ""),
                runtime_mlx_force_hf_tokenizer=bool(
                    getattr(self, "_boot_mlx_force_hf_tokenizer", True)
                ),
                runtime_tokenizer_vocab_guard=bool(
                    getattr(self, "_boot_tokenizer_vocab_guard", True)
                ),
                runtime_kv_cache_max_entries=int(
                    getattr(self, "_boot_kv_cache_max_entries", 1024)
                ),
                runtime_warmup_on_start=bool(getattr(self, "_boot_warmup_on_start", False)),
                runtime_mlx_eval_timeout_s=float(
                    getattr(self, "_boot_mlx_eval_timeout_s", 120.0)
                ),
                runtime_layer_indices=new_layer_indices,
                runtime_peer_id=str(self.peer_id),
                runtime_load_full_head=bool(
                    getattr(self, "_boot_load_full_head", False)
                ),
                runtime_pipeline_depth=max(
                    1, int(getattr(self, "_boot_pipeline_depth", 1) or 1)
                ),
            )
        )
        with self._lock:
            self.shard = new_shard
            # Also update the service's advertised shard metadata so the
            # announce loop picks up the new layer range on the next tick.
            try:
                self.shard_index = shard_index
                self.total_shards = total_shards
            except Exception:
                pass
            # B3 follow-up: the ``shard_layer_mismatch`` validator on the
            # Forward handler reads ``runtime_profile['layer_start']`` /
            # ``runtime_profile['layer_end']``. The assignment's actual
            # layer range is the single source of truth — any derived
            # values from ``new_shard.runtime_profile()`` tend to encode
            # ``shard_index`` / ``shard_index+1`` rather than the real
            # contiguous layer range, which then causes the validator
            # to reject the coordinator's (correct) request.
            try:
                self.runtime_profile["layer_start"] = int(assignment.layer_start)
                self.runtime_profile["layer_end"] = int(assignment.layer_end)
                self.runtime_profile["total_layers"] = int(assignment.total_layers)
            except Exception:
                pass
            # Path A: refresh the coordinator-side HeadSampler after reshard.
            # The new shard may or may not own the last layer — either way,
            # the register-or-skip heuristic handles both cases.
            try:
                from coordinator.head_sampler import unregister_head_source
                unregister_head_source(str(self.peer_id))
            except Exception:
                pass
            self._maybe_register_head_source()

    def resource_budget(self) -> ResourceBudget:
        with self._lock:
            return ResourceBudget(
                vram_fraction=self._resource_budget.vram_fraction,
                cpu_fraction=self._resource_budget.cpu_fraction,
                should_yield=self._resource_budget.should_yield,
                reason=self._resource_budget.reason,
            )

    def _kv_cache_get(self, session_id: str) -> list[float] | None:
        if not session_id:
            return None
        with self._lock:
            cached = self._kv_cache.get(session_id)
            if not cached:
                return None
            # Reinsert key to approximate LRU behavior in insertion-ordered dicts.
            self._kv_cache.pop(session_id, None)
            self._kv_cache[session_id] = list(cached)
            return list(cached)

    def _kv_cache_set(self, session_id: str, activation: list[float]) -> None:
        if not session_id or not activation:
            return
        values = [float(item) for item in activation]
        with self._lock:
            self._kv_cache.pop(session_id, None)
            self._kv_cache[session_id] = values
            while len(self._kv_cache) > self.kv_cache_max_entries:
                oldest_key = next(iter(self._kv_cache))
                self._kv_cache.pop(oldest_key, None)

    def _load_pct(self) -> float:
        with self._lock:
            inflight = self._inflight
            budget = self._resource_budget

        base = min(100.0, inflight * 12.5)
        if budget.should_yield:
            return max(95.0, base)

        cpu_fraction = max(0.0, min(1.0, budget.cpu_fraction))
        budget_pressure = (1.0 - cpu_fraction) * 15.0
        return min(100.0, base + budget_pressure)

    def compaction_stats(self) -> dict:
        """Return KV compaction statistics for DHT announcement.

        Returns an empty dict when compaction is not active or not
        supported by the current runtime backend.  The announce loop
        uses ``compact_tokens_saved`` and ``compact_latency_s`` keys.
        """
        shard = getattr(self, "shard", None)
        if shard is None:
            return {}
        # PyTorchRuntime exposes compaction stats via kv_compaction attr
        runtime = getattr(shard, "_runtime", None)
        stats_fn = getattr(runtime, "compaction_stats", None)
        if callable(stats_fn):
            try:
                return dict(stats_fn())
            except Exception:
                return {}
        return {}

    def Ping(self, request: peer_pb2.PingRequest, context: grpc.ServicerContext) -> peer_pb2.PingResponse:
        geo_nonce = str(getattr(request, "geo_nonce", "") or "").strip()
        geo_claimed_region = str(getattr(request, "geo_claimed_region", "") or "")
        geo_nonce_signature = ""
        if geo_nonce:
            geo_nonce_signature = sign_geo_challenge(
                peer_id=self.peer_id,
                nonce=geo_nonce,
                claimed_region=geo_claimed_region,
                shared_secret_seed=self.geo_challenge_seed,
            )
        return peer_pb2.PingResponse(
            peer_id=self.peer_id,
            ok=True,
            load_pct=self._load_pct(),
            daemon_mode=self.daemon_mode,
            geo_nonce_signature=geo_nonce_signature,
        )

    def Forward(self, request: peer_pb2.ForwardRequest, context: grpc.ServicerContext) -> peer_pb2.ForwardResponse:
        self.last_request_thread_id = threading.get_ident()
        with self._lock:
            self._inflight += 1

        try:
            max_tokens = int(request.max_tokens or 24)
            decode_do_sample = bool(getattr(request, "decode_do_sample", False))
            decode_temperature = float(getattr(request, "decode_temperature", 0.0) or 0.0)
            decode_top_p = float(getattr(request, "decode_top_p", 0.0) or 0.0)
            decode_top_k = int(getattr(request, "decode_top_k", 0) or 0)
            decode_seed = int(getattr(request, "decode_seed", 0) or 0)
            kv_session_id = str(getattr(request, "kv_session_id", "") or "").strip()
            kv_store_activation = bool(getattr(request, "kv_store_activation", False) and kv_session_id)
            kv_use_cached_activation = bool(getattr(request, "kv_use_cached_activation", False) and kv_session_id)

            # ── Phase 2b §5: inline KV rollback ─────────────────────
            # Honour ``request.kv_rollback_to`` BEFORE the forward.
            # Race-free: applied inline against the same Forward call
            # that carries the request, so a separate SwarmCommand
            # cannot race the next ForwardRequest. NO FALLBACK to
            # drop-and-reprefill — a 4 K context re-prefill destroys
            # the speculative-decoding TPS advantage. Any error here
            # is session-fatal and propagates to the gRPC response.
            _kv_rollback_to = int(getattr(request, "kv_rollback_to", 0) or 0)
            if _kv_rollback_to > 0 and kv_session_id:
                _apply_kv_rb = getattr(self.shard, "apply_kv_rollback", None)
                if not callable(_apply_kv_rb):
                    raise RuntimeError(
                        "apply_kv_rollback unavailable on this shard — "
                        "cannot honour kv_rollback_to without it. Phase "
                        "2b requires runtime upgrade."
                    )
                _apply_kv_rb(
                    session_id=kv_session_id,
                    target_len=_kv_rollback_to,
                )
                logger.info(
                    "kv_rollback_applied: req=%s session=%s target_len=%d",
                    request.request_id, kv_session_id, _kv_rollback_to,
                )

            # DSD: batch verification of draft tokens (P0-A)
            verify_batch = int(getattr(request, "verify_batch_size", 0) or 0)
            if verify_batch > 0 and request.draft_token_ids:
                verified = self._verify_draft_tokens(
                    prompt=request.prompt,
                    draft_token_ids=[int(t) for t in request.draft_token_ids[:verify_batch]],
                    decode_temperature=decode_temperature,
                    decode_top_p=decode_top_p,
                    decode_top_k=decode_top_k,
                    decode_seed=decode_seed,
                )
                return peer_pb2.ForwardResponse(
                    request_id=request.request_id,
                    peer_id=self.peer_id,
                    activation=[],
                    stage_index=request.stage_index,
                    error="",
                    verified_token_ids=verified,
                )

            kv_cache_hit = False
            onion_route_ciphertext = b""
            onion_route_nonces: list[bytes] = []
            onion_route_ephemeral_public_keys: list[bytes] = []
            onion_route_suite = ""
            onion_route_layers = 0
            onion_next_peer_id = ""
            with self._lock:
                self.last_onion_next_peer_id = None

            if request.onion_route_ciphertext:
                if self._peer_private_key is not None:
                    onion_layer = peel_onion_route_layer_with_privkey(
                        ciphertext=bytes(request.onion_route_ciphertext),
                        nonces=[bytes(item) for item in request.onion_route_nonces],
                        ephemeral_public_keys=[bytes(item) for item in request.onion_route_ephemeral_public_keys],
                        private_key=self._peer_private_key,
                        peer_id=self.peer_id,
                        request_id=request.request_id,
                        stage_index=int(request.stage_index),
                    )
                else:
                    onion_layer = peel_onion_route_layer(
                        ciphertext=bytes(request.onion_route_ciphertext),
                        nonces=[bytes(item) for item in request.onion_route_nonces],
                        ephemeral_public_keys=[bytes(item) for item in request.onion_route_ephemeral_public_keys],
                        peer_id=self.peer_id,
                        request_id=request.request_id,
                        stage_index=int(request.stage_index),
                        shared_secret_seed=self.advanced_encryption_seed,
                    )
                onion_route_ciphertext = onion_layer.remaining_ciphertext
                onion_route_nonces = list(onion_layer.remaining_nonces)
                onion_route_ephemeral_public_keys = list(onion_layer.remaining_ephemeral_public_keys)
                onion_route_suite = onion_layer.remaining_suite
                onion_route_layers = max(0, int(onion_layer.remaining_layers))
                onion_next_peer_id = str(onion_layer.next_peer_id or "")
                with self._lock:
                    self.onion_layers_peeled += 1
                    self.last_onion_next_peer_id = onion_next_peer_id
                    self.onion_next_peer_history.append(onion_next_peer_id)
                    if len(self.onion_next_peer_history) > 64:
                        self.onion_next_peer_history = self.onion_next_peer_history[-64:]

            # ── Phase 3: shard layer-range validation ─────────────────────────
            # The coordinator embeds the layer range it expects this peer to run
            # in shard_layer_start / shard_layer_end / shard_total_layers.
            # When those fields are non-zero we verify they match our startup
            # config and log the stage routing for observability.
            req_shard_start = int(getattr(request, "shard_layer_start", 0) or 0)
            req_shard_end = int(getattr(request, "shard_layer_end", 0) or 0)
            req_shard_total = int(getattr(request, "shard_total_layers", 0) or 0)
            if req_shard_end > 0:
                my_layer_start = int(self.runtime_profile.get("layer_start", 0) or 0)
                my_layer_end = int(self.runtime_profile.get("layer_end", 0) or 0)
                if my_layer_end > 0 and (
                    req_shard_start != my_layer_start or req_shard_end != my_layer_end
                ):
                    raise RuntimeError(
                        f"shard_layer_mismatch: coordinator expects "
                        f"[{req_shard_start},{req_shard_end}) "
                        f"but peer covers [{my_layer_start},{my_layer_end})"
                    )
                logger.debug(
                    "shard_forward: peer=%s stage=%d/%d layers=[%d,%d) total=%d",
                    self.peer_id,
                    int(request.stage_index),
                    int(request.total_stages),
                    req_shard_start,
                    req_shard_end,
                    req_shard_total,
                )
            # ──────────────────────────────────────────────────────────────────

            if request.encrypted_activation:
                if self._peer_private_key is not None:
                    activation_in = decrypt_activation_envelope_with_privkey(
                        ciphertext=bytes(request.encrypted_activation),
                        nonces=[bytes(item) for item in request.encryption_nonces],
                        ephemeral_public_keys=[bytes(item) for item in request.encryption_ephemeral_public_keys],
                        private_key=self._peer_private_key,
                        peer_id=self.peer_id,
                        request_id=request.request_id,
                        stage_index=int(request.stage_index),
                    )
                else:
                    activation_in = decrypt_activation_envelope(
                        ciphertext=bytes(request.encrypted_activation),
                        nonces=[bytes(item) for item in request.encryption_nonces],
                        ephemeral_public_keys=[bytes(item) for item in request.encryption_ephemeral_public_keys],
                        peer_id=self.peer_id,
                        request_id=request.request_id,
                        stage_index=int(request.stage_index),
                        shared_secret_seed=self.advanced_encryption_seed,
                    )
            else:
                if self.advanced_encryption_enabled and int(request.stage_index) > 0:
                    raise RuntimeError("encrypted_activation_required")
                # Check for INT8 quantized or binary-packed activation first.
                _quant_mode = str(getattr(request, "activation_quantization", "") or "").strip()
                _packed = bytes(getattr(request, "activation_packed", b"") or b"")
                if _quant_mode == "int8" and request.quantized_activation:
                    import struct as _struct_q
                    from peer.activation_codec import dequantize_int8
                    _raw = bytes(request.quantized_activation)
                    # First 8 bytes are the preserved [seq_len, hidden_size]
                    # header (2 × float32, exact). Rest is INT8 payload.
                    _header = list(_struct_q.unpack('<2f', _raw[:8]))
                    _payload = dequantize_int8(_raw[8:], list(request.quantized_scales))
                    activation_in = _header + _payload
                elif _packed:
                    # PR-1: vectorised unpack via numpy. unpack_fp32 falls
                    # back to struct.unpack when numpy is unavailable.
                    from peer.activation_codec import unpack_fp32 as _unpack_fp32
                    activation_in = _unpack_fp32(_packed)
                else:
                    activation_in = list(request.activation)

            compression_codec = str(request.compression_codec or "").strip()
            if compression_codec:
                if compression_codec != "tensor_autoencoder_mean_pool":
                    raise RuntimeError(f"unsupported_compression_codec:{compression_codec}")
                original_dim = int(request.compression_original_dim or 0)
                if original_dim <= 0:
                    raise RuntimeError("invalid_compression_original_dim")
                latent_dim = int(request.compression_latent_dim or max(1, len(activation_in)))
                decoder = TensorAutoencoder(CompressionProfile(latent_dim=max(1, latent_dim)))
                activation_in = decoder.decode(activation_in, target_dim=original_dim)

            logger.info("forward_dispatch: peer=%s stage=%d/%d backend=%s",
                       self.peer_id, int(request.stage_index), int(request.total_stages),
                       "pytorch" if self.shard.uses_pytorch_runtime else "batch_queue")

            # ── Phase 2b live-bench Binding #2: block-verify dispatch ────
            # When the request arrives with ``draft_block=True``, route
            # to the block-verify forward path instead of the per-token
            # decode loop. Hidden states for the B+1 verify positions
            # (prefix_len-1 .. prefix_len+B-1) get returned to the coord
            # via the existing PushResult path with block_size > 0.
            if bool(getattr(request, "draft_block", False)):
                _bv_response = self._handle_block_verify_request(request)
                # The handler returns a fully-formed response. Routing
                # depends on the response's role:
                #
                #  * is_hidden_state=True (terminal stage / single-peer):
                #    push to coord via _push_final_result so the dflash
                #    queue picks it up.
                #  * is_hidden_state=False (intermediate): push to next
                #    hop so the next shard in the ring runs its layer
                #    slice on top of these hidden states.
                #
                # Errors (response.error non-empty) ride either path —
                # the receiver is expected to handle the error code in
                # the response body.
                if bool(getattr(request, "push_mode", False)):
                    if bool(getattr(_bv_response, "is_hidden_state", False)):
                        # Terminal — return to coord.
                        callback_addr = str(getattr(request, "final_callback_address", "") or "").strip()
                        if callback_addr:
                            self._push_final_result(
                                response=_bv_response,
                                callback_address=callback_addr,
                                callback_request_id=str(getattr(request, "final_callback_request_id", "") or ""),
                                callback_libp2p_peer_id=str(getattr(request, "final_callback_libp2p_peer_id", "") or ""),
                                pipeline_depth=int(getattr(request, "pipeline_depth", 1) or 1),
                                slot_id=int(getattr(request, "slot_id", 0) or 0),
                            )
                    else:
                        # Intermediate — push to next hop. Reuse the
                        # existing per-token push machinery; it copies
                        # remaining_route, draft_block, block_index,
                        # prompt_token_ids, etc. through to the next
                        # ForwardRequest. Final-callback fields stay
                        # unchanged so the LAST peer eventually returns
                        # to the same coord callback.
                        next_addr = str(getattr(request, "next_hop_address", "") or "").strip()
                        if next_addr:
                            remaining = list(getattr(request, "remaining_route", []))
                            self._push_to_next_hop(
                                request=request,
                                response=_bv_response,
                                next_address=next_addr,
                                remaining_route=remaining,
                            )
                return _bv_response

            # Phase 4 (Gemma 4 sharded adapter): read the ``prompt_token_ids``
            # sidecar. Every stage needs these to recompute the per-layer
            # input tensor locally — without them, Gemma 4 layer forward
            # multiplies by None and produces zero output. Non-Gemma-4
            # families ignore the field entirely.
            _prompt_token_ids: list[int] | None = None
            if len(request.prompt_token_ids) > 0:
                _prompt_token_ids = [int(t) for t in request.prompt_token_ids]

            # ── Client-terminated pipeline (Path A, flag-gated) ────────────
            # When ``sample_on_coordinator=True`` AND this peer is the last
            # stage, skip final_norm + lm_head + sampling and return the
            # hidden state. Coordinator's HeadSampler applies the head.
            # Flag ignored on non-last stages (they always return hidden).
            _sample_on_coord = bool(getattr(request, "sample_on_coordinator", False))
            _is_last_stage = (
                int(request.total_stages) > 0
                and int(request.stage_index) == int(request.total_stages) - 1
            )
            _return_hidden = bool(_sample_on_coord and _is_last_stage)

            if self.shard.uses_pytorch_runtime:
                # Offload PyTorch matrix compute to a worker thread to protect async control planes.
                activation = asyncio.run(
                    self.shard.forward_async(
                        request.prompt,
                        activation_in,
                        max_tokens,
                        stage_index=int(request.stage_index),
                        total_stages=int(request.total_stages),
                        kv_session_id=kv_session_id,
                        kv_store_activation=kv_store_activation,
                        kv_use_cached_activation=kv_use_cached_activation,
                        request_id=str(request.request_id),
                        decode_do_sample=decode_do_sample,
                        decode_temperature=decode_temperature,
                        decode_top_p=decode_top_p,
                        decode_top_k=decode_top_k,
                        decode_seed=(decode_seed if decode_seed > 0 else None),
                        prompt_token_ids=_prompt_token_ids,
                        return_hidden_state=_return_hidden,
                    )
                )
                kv_cache_hit = bool(self.shard.last_kv_cache_hit)
            else:
                # MLX sharded path manages KV cache internally (in the
                # runtime's _kv_cache dict). Pass kv flags through to
                # shard.forward() instead of intercepting here.
                _backend = str(self.runtime_profile.get("backend", "")).lower()
                _mlx_sharded = (int(request.total_stages) > 1 and _backend == "mlx")
                if kv_use_cached_activation and not _mlx_sharded:
                    cached = self._kv_cache_get(kv_session_id)
                    if not cached:
                        raise RuntimeError("kv_cache_miss")
                    activation_in = cached
                    kv_cache_hit = True
                # Fast path: bypass BatchingQueue when inflight count is
                # low (single-peer / no concurrent requests), OR when the
                # request is sharded (forward_batch doesn't support sharding).
                if self.inflight_count() <= 1 or _mlx_sharded:
                    _fwd_kwargs: dict = dict(
                        stage_index=int(request.stage_index),
                        total_stages=int(request.total_stages),
                        request_id=str(request.request_id),
                        decode_do_sample=decode_do_sample,
                        decode_temperature=decode_temperature,
                        decode_top_p=decode_top_p,
                        decode_top_k=decode_top_k,
                        decode_seed=(decode_seed if decode_seed > 0 else None),
                    )
                    if _mlx_sharded:
                        _fwd_kwargs["kv_session_id"] = kv_session_id
                        _fwd_kwargs["kv_store_activation"] = kv_store_activation
                        _fwd_kwargs["kv_use_cached_activation"] = kv_use_cached_activation
                        # Pass raw packed bytes for zero-copy DLPack path.
                        if _packed:
                            _fwd_kwargs["packed_bytes"] = _packed
                    if _return_hidden:
                        _fwd_kwargs["return_hidden_state"] = True
                    activation = list(self.shard.forward(
                        request.prompt,
                        activation_in,
                        max_tokens,
                        **_fwd_kwargs,
                    ))
                else:
                    activation = self.batch_queue.forward(
                        request.prompt,
                        activation_in,
                        max_tokens,
                        stage_index=int(request.stage_index),
                        total_stages=int(request.total_stages),
                        request_id=str(request.request_id),
                        decode_do_sample=decode_do_sample,
                        decode_temperature=decode_temperature,
                        decode_top_p=decode_top_p,
                        decode_top_k=decode_top_k,
                        decode_seed=(decode_seed if decode_seed > 0 else None),
                    )
                if kv_store_activation:
                    self._kv_cache_set(kv_session_id, activation)
            self.last_inference_thread_id = self.shard.last_forward_thread_id

            # TOPLOC: compute activation hash for integrity verification (P2-B)
            _act_hash = b""
            try:
                from verification.toploc import activation_hash
                _act_hash = activation_hash(activation)
            except Exception:
                pass

            # Zero-copy encode: pack activation as bytes via Rust (12x faster).
            # Only for intermediate stages (hidden states); last stage returns
            # token IDs which are short and don't benefit from packing.
            _activation_packed_resp = b""
            _activation_for_proto = activation
            if activation and len(activation) > 10:
                try:
                    # PR-1: vectorised pack (numpy). Falls back to struct.pack
                    # inside pack_fp32 when numpy is absent.
                    from peer.activation_codec import pack_fp32 as _pack_fp32
                    _activation_packed_resp = _pack_fp32(activation)
                    _activation_for_proto = []  # Clear repeated field when packed is set
                except Exception:
                    pass

            response = peer_pb2.ForwardResponse(
                request_id=request.request_id,
                peer_id=self.peer_id,
                activation=_activation_for_proto,
                activation_packed=_activation_packed_resp,
                stage_index=request.stage_index,
                error="",
                kv_cache_hit=kv_cache_hit,
                onion_route_ciphertext=onion_route_ciphertext,
                onion_route_nonces=onion_route_nonces,
                onion_route_ephemeral_public_keys=onion_route_ephemeral_public_keys,
                onion_route_suite=onion_route_suite,
                onion_route_layers=onion_route_layers,
                onion_next_peer_id=onion_next_peer_id,
                dp_noise_applied=bool(self.shard.privacy_noise_last_applied),
                dp_noise_configured_variance=float(self.shard.privacy_noise_variance),
                dp_noise_observed_variance=float(self.shard.privacy_noise_last_observed_variance),
                dp_noise_observed_std=float(self.shard.privacy_noise_last_observed_std),
                dp_noise_payload_index=int(self.shard.privacy_noise_last_payload_index),
                dp_noise_audit_tag=str(self.shard.privacy_noise_last_audit_tag),
                compression_latent_dim=max(0, int(getattr(request, "compression_latent_dim", 0) or 0)),
                activation_hash=_act_hash,
                is_hidden_state=_return_hidden,
                # Phase 2b live-bench Binding #2: echo block-verify
                # metadata on the response so the coord's PushResult
                # handler routes it to the right per-(req_id, block_idx)
                # queue. Default 0 = single-token response (Phase 2a
                # path); set when this Forward processed a draft block.
                block_size=(
                    (len(list(getattr(request, "prompt_token_ids", []) or [])) + 1)
                    if bool(getattr(request, "draft_block", False))
                    else 0
                ),
                block_index=int(getattr(request, "block_index", 0) or 0),
            )

            # ── Push mode: forward to next peer or return to coordinator ──
            if bool(getattr(request, "push_mode", False)):
                next_addr = str(getattr(request, "next_hop_address", "") or "").strip()
                callback_addr = str(getattr(request, "final_callback_address", "") or "").strip()
                remaining = list(getattr(request, "remaining_route", []))

                if next_addr:
                    # Forward activation to next peer in chain
                    self._push_to_next_hop(
                        request=request,
                        response=response,
                        next_address=next_addr,
                        remaining_route=remaining,
                    )
                    return peer_pb2.ForwardResponse(
                        request_id=request.request_id,
                        peer_id=self.peer_id,
                        stage_index=request.stage_index,
                        error="",
                    )
                elif bool(getattr(request, "ring_mode", False)) and not next_addr:
                    # ── Path A: coordinator-terminated ring ───────────────
                    # When sample_on_coordinator=True, the last peer has
                    # already returned a hidden state in ``response`` (with
                    # ``is_hidden_state=True``). Skip the peer-side
                    # loop-back entirely — ship the hidden state to the
                    # coordinator via PushResult and let its HeadSampler
                    # do the sampling + stage-0 re-injection. This removes
                    # the last-peer→first-peer hop that motivated Path A.
                    if bool(getattr(request, "sample_on_coordinator", False)):
                        _cb_addr = str(getattr(request, "final_callback_address", "") or "").strip()
                        _cb_libp2p = str(getattr(request, "final_callback_libp2p_peer_id", "") or "").strip()
                        if _cb_addr or _cb_libp2p:
                            self._push_final_result(
                                response=response,
                                callback_address=_cb_addr,
                                callback_request_id=str(
                                    getattr(request, "final_callback_request_id", "") or ""
                                ),
                                callback_libp2p_peer_id=_cb_libp2p,
                                pipeline_depth=int(
                                    getattr(request, "pipeline_depth", 1) or 1
                                ),
                                slot_id=int(getattr(request, "slot_id", 0) or 0),
                            )
                            return peer_pb2.ForwardResponse(
                                request_id=request.request_id,
                                peer_id=self.peer_id,
                                stage_index=request.stage_index,
                                error="",
                            )
                        logger.warning(
                            "sample_on_coordinator_set_but_no_callback: "
                            "req=%s — falling back to legacy ring loopback",
                            request.request_id,
                        )
                    # ── Ring autoregressive (legacy): last shard loops back ──
                    _ring_activation = list(response.activation)
                    _ring_token = int(round(float(_ring_activation[0]))) if _ring_activation else 0
                    _ring_generated = list(request.ring_generated_ids) + [_ring_token]
                    _ring_remaining = int(request.ring_tokens_remaining) - 1
                    _ring_eos = set(int(e) for e in request.ring_eos_ids)
                    _ring_cb_id = str(getattr(request, "final_callback_request_id", "") or "")

                    # Emit token to coordinator's async queue (same process, zero-cost).
                    from coordinator.push_receiver import emit_ring_token
                    emit_ring_token(_ring_cb_id, _ring_token)
                    logger.info("ring_token_emitted: token=%d remaining=%d", _ring_token, _ring_remaining)

                    # Always loop back to coordinator — the coordinator's fire-and-forget
                    # handler emits tokens + sentinel. Even when remaining==0, the
                    # coordinator needs the final token delivered via the loop-back.
                    if True:
                        # Loop back to first peer with the new token.
                        _ring_route = list(request.ring_full_route)
                        _ring_next_addr = str(request.ring_first_hop_address)
                        _ring_next_next = _ring_route[1].address if len(_ring_route) > 1 else ""
                        _ring_next_next_id = _ring_route[1].peer_id if len(_ring_route) > 1 else ""
                        _ring_req = peer_pb2.ForwardRequest(
                            request_id=request.request_id,
                            activation=[float(_ring_token)],
                            stage_index=0,
                            total_stages=request.total_stages,
                            max_tokens=1,
                            kv_session_id=request.kv_session_id,
                            kv_store_activation=True,
                            kv_use_cached_activation=True,
                            decode_do_sample=request.decode_do_sample,
                            decode_temperature=request.decode_temperature,
                            decode_top_p=request.decode_top_p,
                            decode_top_k=request.decode_top_k,
                            decode_seed=request.decode_seed,
                            shard_layer_start=_ring_route[0].shard_layer_start if _ring_route else 0,
                            shard_layer_end=_ring_route[0].shard_layer_end if _ring_route else 0,
                            shard_total_layers=_ring_route[0].shard_total_layers if _ring_route else 0,
                            push_mode=True,
                            ring_mode=True,
                            ring_tokens_remaining=_ring_remaining,
                            ring_generated_ids=_ring_generated,
                            ring_eos_ids=list(request.ring_eos_ids),
                            ring_first_hop_address=request.ring_first_hop_address,
                            ring_first_hop_peer_id=request.ring_first_hop_peer_id,
                            ring_first_hop_libp2p_id=request.ring_first_hop_libp2p_id,
                            ring_full_route=_ring_route,
                            next_hop_address=_ring_next_next,
                            next_hop_peer_id=_ring_next_next_id,
                            final_callback_address=callback_addr,
                            final_callback_request_id=_ring_cb_id,
                            final_callback_libp2p_peer_id=str(getattr(request, "final_callback_libp2p_peer_id", "") or ""),
                            remaining_route=_ring_route[1:],
                            prompt_token_ids=list(request.prompt_token_ids),
                        )
                        # Fire-and-forget: ring loop-back in background thread.
                        # IMPORTANT: Do NOT use _push_to_next_hop here — it rebuilds
                        # the request with stage_index+1 and wrong shard layers.
                        # The ring loop-back sends the pre-built _ring_req directly.
                        import threading as _ring_threading
                        _ring_first_libp2p = str(request.ring_first_hop_libp2p_id or "")
                        def _ring_loop_back(_rreq=_ring_req, _raddr=_ring_next_addr,
                                            _libp2p=_ring_first_libp2p):
                            try:
                                logger.info("RING_LOOPBACK_START: req=%s remaining=%d -> %s (libp2p=%s)",
                                            _rreq.request_id, _rreq.ring_tokens_remaining,
                                            _raddr, _libp2p[:20] if _libp2p else "none")
                                # LAN-first: if we can reach the first
                                # peer directly via gRPC on a shared /16,
                                # bypass the libp2p relay path.
                                from peer.lan_routing import (
                                    is_reachable_lan as _ir, parse_host_from_address as _ph,
                                )
                                _lh = _ph(_raddr)
                                _lan_ok = bool(_lh) and _ir(_lh)
                                if _lan_ok and _raddr:
                                    import grpc as _rl_grpc
                                    _rl_ch = _rl_grpc.insecure_channel(
                                        _raddr,
                                        options=[
                                            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
                                            ("grpc.max_send_message_length", 100 * 1024 * 1024),
                                        ],
                                    )
                                    try:
                                        _rl_stub = peer_pb2_grpc.PeerStub(_rl_ch)
                                        _rl_stub.Forward(_rreq, timeout=60.0)
                                    finally:
                                        _rl_ch.close()
                                    logger.info("RING_LOOPBACK_DONE: req=%s remaining=%d via_lan",
                                                _rreq.request_id, _rreq.ring_tokens_remaining)
                                elif self._p2p_node is not None and _libp2p:
                                    self._p2p_node.proxy_forward(
                                        target_peer_id=_libp2p,
                                        data=PROXY_METHOD_FIRE_FORGET + _rreq.SerializeToString(),
                                    )
                                    logger.info("RING_LOOPBACK_DONE: req=%s remaining=%d via_relay",
                                                _rreq.request_id, _rreq.ring_tokens_remaining)
                                elif _raddr:
                                    import grpc as _rl_grpc
                                    _rl_ch = _rl_grpc.insecure_channel(
                                        _raddr,
                                        options=[
                                            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
                                            ("grpc.max_send_message_length", 100 * 1024 * 1024),
                                        ],
                                    )
                                    _rl_stub = peer_pb2_grpc.PeerStub(_rl_ch)
                                    _rl_stub.Forward(_rreq, timeout=60.0)
                                    _rl_ch.close()
                                    logger.info("RING_LOOPBACK_DONE: req=%s remaining=%d via_grpc",
                                                _rreq.request_id, _rreq.ring_tokens_remaining)
                                else:
                                    logger.error("RING_LOOPBACK_NO_ROUTE: req=%s", _rreq.request_id)
                            except Exception as _rl_exc:
                                logger.error("RING_LOOPBACK_CRASH: req=%s remaining=%d err=%s",
                                             _rreq.request_id, _rreq.ring_tokens_remaining, _rl_exc,
                                             exc_info=True)
                        _ring_threading.Thread(target=_ring_loop_back, daemon=True).start()

                elif callback_addr:
                    # Last peer: send result back to coordinator (non-ring push)
                    self._push_final_result(
                        response=response,
                        callback_address=callback_addr,
                        callback_request_id=str(getattr(request, "final_callback_request_id", "") or ""),
                        callback_libp2p_peer_id=str(getattr(request, "final_callback_libp2p_peer_id", "") or ""),
                        pipeline_depth=int(getattr(request, "pipeline_depth", 1) or 1),
                        slot_id=int(getattr(request, "slot_id", 0) or 0),
                    )
                    return peer_pb2.ForwardResponse(
                        request_id=request.request_id,
                        peer_id=self.peer_id,
                        stage_index=request.stage_index,
                        error="",
                    )

            return response
        except Exception as exc:  # pragma: no cover
            # Log the full traceback so we can debug peer-side failures —
            # the gRPC response only carries ``str(exc)`` which omits the
            # stack and hides subtle issues (e.g. the Phase 6 KV-cache
            # meta-tensor dispatcher path).
            import traceback as _tb
            logger.error(
                "Forward_failed peer=%s stage=%d req=%s: %s\n%s",
                self.peer_id, int(request.stage_index), request.request_id,
                exc, _tb.format_exc(),
            )
            return peer_pb2.ForwardResponse(
                request_id=request.request_id,
                peer_id=self.peer_id,
                activation=[],
                stage_index=request.stage_index,
                error=str(exc),
                kv_cache_hit=False,
                onion_route_ciphertext=b"",
                onion_route_nonces=[],
                onion_route_ephemeral_public_keys=[],
                onion_route_suite="",
                onion_route_layers=0,
                onion_next_peer_id="",
                dp_noise_applied=False,
                dp_noise_configured_variance=0.0,
                dp_noise_observed_variance=0.0,
                dp_noise_observed_std=0.0,
                dp_noise_payload_index=0,
                dp_noise_audit_tag="",
                compression_latent_dim=max(0, int(getattr(request, "compression_latent_dim", 0) or 0)),
            )
        finally:
            with self._lock:
                self._inflight = max(0, self._inflight - 1)

    # ── Push mode: server-to-server forwarding (Petals parity) ─────────────

    def _handle_block_verify_request(
        self,
        request: peer_pb2.ForwardRequest,
    ) -> peer_pb2.ForwardResponse:
        """Phase 2b live-bench Binding #2 — peer-side block-verify dispatch.

        When the coord fires a ForwardRequest with ``draft_block=True``,
        the request carries:

          * ``prompt_token_ids`` — the B-token draft block.
          * ``activation_packed`` — the ``[prefix_len_floats, ...]``
            payload encoding the prefix tokens (initial_activation in
            chain.py::run_push_ring is the prefix token-id list, packed
            as floats).
          * ``draft_block`` = True
          * ``block_index`` — the coord-assigned per-session counter.
          * ``kv_rollback_to`` — applied per Phase 2b §5 before this
            forward (already handled at the top of Forward).

        On a single-peer / full-model deployment (this peer owns the
        full target), the handler runs ``forward_block_for_verify``
        directly and packages the [B+1, hidden] hidden states into a
        ForwardResponse with ``is_hidden_state=True`` and
        ``block_size=B+1``.

        On a multi-peer SHARDED deployment, the handler currently
        emits a structured error response (block-verify-sharded).
        Sharded block-verify across a layer-sliced ring requires each
        peer to run its slice over B+1 positions in parallel and
        forward the hidden states down the chain — a follow-up that
        reuses the existing prefill multi-position code path.
        """
        block_drafts = list(request.prompt_token_ids or [])
        block_index = int(getattr(request, "block_index", 0) or 0)
        block_size_response = len(block_drafts) + 1   # B+1 verify positions

        stage_index = int(getattr(request, "stage_index", 0) or 0)
        total_stages = int(getattr(request, "total_stages", 1) or 1)
        is_first_stage = (stage_index == 0)
        is_last_stage = (stage_index == total_stages - 1)

        if not block_drafts:
            return peer_pb2.ForwardResponse(
                request_id=request.request_id,
                peer_id=self.peer_id,
                stage_index=request.stage_index,
                error="block_verify_empty_drafts",
                is_hidden_state=False,
                block_size=0,
                block_index=block_index,
            )

        runtime = getattr(self.shard, "_runtime", None)
        # ── Phase 2b live-bench Binding #3 — sharded block-verify ────
        # Layer-sliced peers route through forward_block_layer_slice
        # which runs THIS peer's layer slice over the multi-position
        # input. Stage 0 embeds [prefix + drafts]; intermediate / last
        # stages take incoming hidden states from the previous shard.
        # The terminal stage slices the last B+1 positions before
        # returning so only the verify positions ride the libp2p relay
        # back to coord.
        #
        # Single-peer / full-model deployments (total_stages == 1) keep
        # the simpler forward_block_for_verify path which was
        # exhaustively unit-tested in Binding #1.
        fwd_block_full = getattr(runtime, "forward_block_for_verify", None)
        fwd_block_slice = getattr(runtime, "forward_block_layer_slice", None)

        if total_stages <= 1 and callable(fwd_block_full):
            # Single-peer fast path — preserves the Binding #1 contract.
            prefix_token_ids: list[int] = []
            try:
                packed = bytes(getattr(request, "activation_packed", b"") or b"")
                if packed:
                    import struct
                    n = len(packed) // 4
                    if n > 0:
                        floats = list(struct.unpack(f"<{n}f", packed))
                        prefix_token_ids = [int(round(f)) for f in floats]
                elif list(getattr(request, "activation", []) or []):
                    prefix_token_ids = [
                        int(round(f)) for f in list(request.activation)
                    ]
            except Exception as exc:
                logger.error(
                    "block_verify_decode_prefix_failed: req=%s err=%s",
                    request.request_id, exc, exc_info=True,
                )
                return peer_pb2.ForwardResponse(
                    request_id=request.request_id,
                    peer_id=self.peer_id,
                    stage_index=request.stage_index,
                    error=f"block_verify_decode_prefix_failed: {exc}",
                    is_hidden_state=False,
                    block_size=0,
                    block_index=block_index,
                )
            if not prefix_token_ids:
                return peer_pb2.ForwardResponse(
                    request_id=request.request_id,
                    peer_id=self.peer_id,
                    stage_index=request.stage_index,
                    error="block_verify_empty_prefix",
                    is_hidden_state=False,
                    block_size=0,
                    block_index=block_index,
                )
            try:
                hidden_block = fwd_block_full(prefix_token_ids, block_drafts)
            except Exception as exc:
                logger.error(
                    "block_verify_forward_failed: req=%s block=%d err=%s",
                    request.request_id, block_index, exc, exc_info=True,
                )
                return peer_pb2.ForwardResponse(
                    request_id=request.request_id,
                    peer_id=self.peer_id,
                    stage_index=request.stage_index,
                    error=f"block_verify_forward_failed: {exc}",
                    is_hidden_state=False,
                    block_size=0,
                    block_index=block_index,
                )
            # Single-peer always behaves as last_stage from the coord's
            # perspective: it returns the verify-position hidden states.
            is_last_stage = True
            forward_output = hidden_block
        elif callable(fwd_block_slice):
            # Sharded path — Binding #3.
            try:
                if is_first_stage:
                    # Decode prefix from activation_packed (token IDs as fp32).
                    prefix_token_ids = []
                    packed = bytes(getattr(request, "activation_packed", b"") or b"")
                    if packed:
                        import struct
                        n = len(packed) // 4
                        if n > 0:
                            floats = list(struct.unpack(f"<{n}f", packed))
                            prefix_token_ids = [int(round(f)) for f in floats]
                    elif list(getattr(request, "activation", []) or []):
                        prefix_token_ids = [
                            int(round(f)) for f in list(request.activation)
                        ]
                    if not prefix_token_ids:
                        return peer_pb2.ForwardResponse(
                            request_id=request.request_id,
                            peer_id=self.peer_id,
                            stage_index=request.stage_index,
                            error="block_verify_empty_prefix",
                            is_hidden_state=False,
                            block_size=0,
                            block_index=block_index,
                        )
                    forward_output = fwd_block_slice(
                        is_first_stage=True,
                        is_last_stage=is_last_stage,
                        prefix_token_ids=prefix_token_ids,
                        draft_token_ids=block_drafts,
                    )
                else:
                    # Intermediate or last stage: incoming activation
                    # is the previous shard's pre-norm hidden state
                    # block as activation_packed bytes.
                    incoming_packed = bytes(
                        getattr(request, "activation_packed", b"") or b""
                    )
                    if not incoming_packed:
                        return peer_pb2.ForwardResponse(
                            request_id=request.request_id,
                            peer_id=self.peer_id,
                            stage_index=request.stage_index,
                            error="block_verify_empty_incoming_hidden",
                            is_hidden_state=False,
                            block_size=0,
                            block_index=block_index,
                        )
                    # Decode hidden states via runtime helper.
                    incoming_hidden = runtime._activation_to_hidden(
                        None, packed_bytes=incoming_packed,
                    )
                    forward_output = fwd_block_slice(
                        is_first_stage=False,
                        is_last_stage=is_last_stage,
                        draft_token_ids=block_drafts,
                        incoming_hidden=incoming_hidden,
                    )
            except Exception as exc:
                logger.error(
                    "block_verify_sharded_forward_failed: req=%s "
                    "stage=%d/%d block=%d err=%s",
                    request.request_id, stage_index, total_stages,
                    block_index, exc, exc_info=True,
                )
                return peer_pb2.ForwardResponse(
                    request_id=request.request_id,
                    peer_id=self.peer_id,
                    stage_index=request.stage_index,
                    error=f"block_verify_sharded_forward_failed: {exc}",
                    is_hidden_state=False,
                    block_size=0,
                    block_index=block_index,
                )
        else:
            logger.warning(
                "block_verify_runtime_unsupported: peer=%s runtime=%s "
                "lacks forward_block_for_verify and "
                "forward_block_layer_slice",
                self.peer_id,
                type(runtime).__name__ if runtime is not None else "None",
            )
            return peer_pb2.ForwardResponse(
                request_id=request.request_id,
                peer_id=self.peer_id,
                stage_index=request.stage_index,
                error="block_verify_runtime_unsupported",
                is_hidden_state=False,
                block_size=0,
                block_index=block_index,
            )

        hidden_block = forward_output

        # Pack the hidden state block to wire bytes. Reuse the
        # runtime's existing serialiser when available; fall back to
        # a generic float-pack.
        packer = getattr(runtime, "_hidden_to_packed_bytes", None)
        try:
            if callable(packer):
                packed_bytes = bytes(packer(
                    hidden_block,
                    request_id=request.request_id,
                    stage_index=stage_index,
                ))
            else:
                # Generic fallback: flatten + struct.pack as fp32.
                import struct
                if hasattr(hidden_block, "detach"):
                    flat = hidden_block.detach().to("cpu").float().contiguous().view(-1).tolist()
                else:
                    # MLX path: convert to numpy then flatten.
                    import numpy as _np
                    flat = list(_np.asarray(hidden_block).astype("float32").reshape(-1))
                packed_bytes = struct.pack(f"<{len(flat)}f", *flat)
        except Exception as exc:
            logger.error(
                "block_verify_pack_failed: req=%s block=%d err=%s",
                request.request_id, block_index, exc, exc_info=True,
            )
            return peer_pb2.ForwardResponse(
                request_id=request.request_id,
                peer_id=self.peer_id,
                stage_index=request.stage_index,
                error=f"block_verify_pack_failed: {exc}",
                is_hidden_state=False,
                block_size=0,
                block_index=block_index,
            )

        # Phase 2b live-bench Binding #3 — response shape depends on
        # this peer's role in the ring:
        #
        #  * Last stage (or single-peer): is_hidden_state=True,
        #    block_size=B+1 so the coord-side PushResult handler routes
        #    to the dflash queue.
        #  * Intermediate stage: is_hidden_state=False, block_size=0 so
        #    _push_to_next_hop forwards the rebuilt request (with our
        #    activation_packed = our shard's hidden output) to the next
        #    peer rather than to coord.
        if is_last_stage:
            logger.info(
                "block_verify_done_terminal: peer=%s req=%s stage=%d/%d "
                "block=%d drafts=%d packed_bytes=%d",
                self.peer_id, request.request_id,
                stage_index, total_stages, block_index,
                len(block_drafts), len(packed_bytes),
            )
            return peer_pb2.ForwardResponse(
                request_id=request.request_id,
                peer_id=self.peer_id,
                stage_index=request.stage_index,
                activation_packed=packed_bytes,
                is_hidden_state=True,
                block_size=block_size_response,
                block_index=block_index,
            )

        logger.info(
            "block_verify_done_intermediate: peer=%s req=%s stage=%d/%d "
            "block=%d drafts=%d packed_bytes=%d (forwarding to stage %d)",
            self.peer_id, request.request_id,
            stage_index, total_stages, block_index,
            len(block_drafts), len(packed_bytes), stage_index + 1,
        )
        return peer_pb2.ForwardResponse(
            request_id=request.request_id,
            peer_id=self.peer_id,
            stage_index=request.stage_index,
            activation_packed=packed_bytes,
            is_hidden_state=False,
            block_size=0,
            block_index=block_index,
        )

    def _push_to_next_hop(
        self,
        request: peer_pb2.ForwardRequest,
        response: peer_pb2.ForwardResponse,
        next_address: str,
        remaining_route: list,
    ) -> None:
        """Forward activation to the next peer in the push chain."""
        try:
            from coordinator.transport import create_channel

            # Build next request: carry the activation + remaining route
            next_route = remaining_route[1:] if len(remaining_route) > 1 else []
            next_next_addr = ""
            next_next_id = ""
            if next_route:
                next_next_addr = str(next_route[0].address)
                next_next_id = str(next_route[0].peer_id)

            # Use activation_packed from response if available (zero-copy path).
            # Falls back to re-packing the repeated float field.
            _push_packed = bytes(getattr(response, 'activation_packed', b'') or b'')
            _push_activation: list[float] = []
            if not _push_packed:
                _push_activation = list(response.activation)
                if _push_activation:
                    import struct as _push_struct
                    _push_packed = _push_struct.pack(f'<{len(_push_activation)}f', *_push_activation)
                    _push_activation = []

            next_req = peer_pb2.ForwardRequest(
                request_id=request.request_id,
                prompt="",  # Only stage 0 gets the prompt
                activation=_push_activation,
                activation_packed=_push_packed,
                stage_index=request.stage_index + 1,
                total_stages=request.total_stages,
                max_tokens=request.max_tokens,
                kv_session_id=request.kv_session_id,
                kv_store_activation=request.kv_store_activation,
                kv_use_cached_activation=request.kv_use_cached_activation,
                decode_do_sample=request.decode_do_sample,
                decode_temperature=request.decode_temperature,
                decode_top_p=request.decode_top_p,
                decode_top_k=request.decode_top_k,
                decode_seed=request.decode_seed,
                shard_layer_start=remaining_route[0].shard_layer_start if remaining_route else 0,
                shard_layer_end=remaining_route[0].shard_layer_end if remaining_route else 0,
                shard_total_layers=remaining_route[0].shard_total_layers if remaining_route else 0,
                push_mode=True,
                next_hop_address=next_next_addr,
                next_hop_peer_id=next_next_id,
                final_callback_address=request.final_callback_address,
                final_callback_request_id=request.final_callback_request_id,
                final_callback_libp2p_peer_id=str(getattr(request, "final_callback_libp2p_peer_id", "") or ""),
                remaining_route=next_route,
                # Gemma 4 needs prompt_token_ids at every stage for per-layer inputs.
                prompt_token_ids=list(request.prompt_token_ids),
                # Ring mode fields — must carry forward for the ring to function.
                ring_mode=bool(getattr(request, "ring_mode", False)),
                ring_tokens_remaining=int(getattr(request, "ring_tokens_remaining", 0)),
                ring_generated_ids=list(getattr(request, "ring_generated_ids", [])),
                ring_eos_ids=list(getattr(request, "ring_eos_ids", [])),
                ring_first_hop_address=str(getattr(request, "ring_first_hop_address", "") or ""),
                ring_first_hop_peer_id=str(getattr(request, "ring_first_hop_peer_id", "") or ""),
                ring_first_hop_libp2p_id=str(getattr(request, "ring_first_hop_libp2p_id", "") or ""),
                ring_full_route=list(getattr(request, "ring_full_route", [])),
                # Path A: carry sample_on_coordinator forward so the LAST
                # peer skips final_norm + lm_head + sampling and routes
                # the hidden state to the coordinator. Without this the
                # legacy ring-loopback path fires on the last peer and
                # the Path A code never runs.
                sample_on_coordinator=bool(getattr(request, "sample_on_coordinator", False)),
                # Phase 2a: forward per-ring slot_id verbatim through every
                # hop. The terminal peer echoes it on PushResult so the
                # coord can match the response back to its in-flight
                # SlotState (out-of-order safe under pipeline_depth >= 2).
                slot_id=int(getattr(request, "slot_id", 0) or 0),
                pipeline_depth=int(getattr(request, "pipeline_depth", 1) or 1),
                # Phase 2b live-bench Binding #3: propagate block-verify
                # routing fields verbatim. Each shard runs its slice
                # over the SAME drafts / kv_rollback_to / block_index;
                # only ``activation_packed`` mutates as hidden states
                # cascade down the ring.
                draft_block=bool(getattr(request, "draft_block", False)),
                block_index=int(getattr(request, "block_index", 0) or 0),
                kv_rollback_to=int(getattr(request, "kv_rollback_to", 0) or 0),
            )

            # State-aware routing: ask the Rust bridge if a direct (non-relayed)
            # connection exists. Zero blocking — instant check.
            _next_hop_libp2p_id = ""
            if remaining_route:
                _next_hop_libp2p_id = str(getattr(remaining_route[0], 'libp2p_peer_id', '') or '').strip()

            # ── LAN-first routing (2026-04-24) ─────────────────────────
            # When ``next_address`` is a private IP that we can reach via
            # a local interface in the same /16, prefer direct gRPC
            # unconditionally — even when libp2p is available. This
            # collapses cross-VPC libp2p relay hops (which break under
            # symmetric NAT, even with DCUtR) into single-RTT LAN gRPC.
            from peer.lan_routing import is_reachable_lan, parse_host_from_address
            _next_host = parse_host_from_address(next_address)
            _lan_reachable = bool(_next_host) and is_reachable_lan(_next_host)
            if _lan_reachable and next_address:
                logger.info(
                    "push_forwarded_via_lan: req=%s stage=%d -> %s "
                    "(LAN-first; bypassing libp2p_id=%s)",
                    request.request_id, request.stage_index, next_address,
                    _next_hop_libp2p_id[:20] if _next_hop_libp2p_id else "none",
                )
                # Phase 2a: with pipeline_depth >= 2 the gRPC handler
                # thread cannot afford to block on the next-hop send —
                # the caller is about to start computing the NEXT slot's
                # forward pass and we don't want that pass serialised
                # behind a 20-50 ms direct-LAN dial. Fire-and-forget
                # via a daemon thread; correctness is preserved because
                # the receiving peer's Forward handler is idempotent
                # under retry and no caller depends on the response
                # body (Path A push mode).
                _depth_ff = max(1, int(getattr(request, "pipeline_depth", 1) or 1))
                if _depth_ff >= 2:
                    self._dispatch_direct_grpc_async(
                        next_address, next_req, label="lan",
                        request_id=str(request.request_id),
                    )
                else:
                    channel = grpc.insecure_channel(
                        next_address,
                        options=[
                            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
                            ("grpc.max_send_message_length", 100 * 1024 * 1024),
                        ],
                    )
                    try:
                        stub = peer_pb2_grpc.PeerStub(channel)
                        stub.Forward(next_req, timeout=60.0)
                    finally:
                        channel.close()
            elif self._p2p_node is not None and _next_hop_libp2p_id:
                # B1 rendezvous: we're about to route through a circuit
                # relay. Before we do, publish REQUEST_HOLE_PUNCH so the
                # remote peer dials us back within the ~100 ms gossip
                # propagation window — the narrow NAT-binding overlap
                # DCUtR needs to punch through symmetric NAT. Inline
                # 5 s per-(me,target) debounce: the ``proxy_forward``
                # path fires ~once per token so we guard the publish.
                try:
                    _my_libp2p = str(getattr(self._p2p_node, "libp2p_peer_id", "") or "")
                    if _my_libp2p and _my_libp2p != _next_hop_libp2p_id:
                        import time as _t
                        now_mono = _t.monotonic()
                        _last = getattr(self, "_b1_last_pub", None) or {}
                        key = (_my_libp2p, _next_hop_libp2p_id)
                        if now_mono - _last.get(key, 0.0) >= 5.0:
                            import json as _json
                            _env = {
                                "type": "REQUEST_HOLE_PUNCH",
                                "data": {
                                    "from_peer_id": _my_libp2p,
                                    "to_peer_id": _next_hop_libp2p_id,
                                },
                                "observed_by": _my_libp2p,
                                "unix_ms": int(_t.time() * 1000),
                            }
                            try:
                                self._p2p_node.publish_event(
                                    _json.dumps(_env, separators=(",", ":"))
                                    .encode("utf-8")
                                )
                                _last[key] = now_mono
                                self._b1_last_pub = _last
                                logger.info(
                                    "b1_rendezvous_published_push: target=%s",
                                    _next_hop_libp2p_id[:14],
                                )
                            except Exception as _pub_exc:
                                logger.debug(
                                    "b1_rendezvous_publish_push_failed: %s",
                                    _pub_exc,
                                )
                except Exception:  # pragma: no cover — never derail the push
                    pass

                # Fire-and-forget: ACK instantly, inference runs async on receiver.
                self._p2p_node.proxy_forward(
                    target_peer_id=_next_hop_libp2p_id,
                    data=PROXY_METHOD_FIRE_FORGET + next_req.SerializeToString(),
                )
                logger.info(
                    "push_forwarded_via_relay: req=%s stage=%d -> %s (libp2p=%s)",
                    request.request_id, request.stage_index, next_address, _next_hop_libp2p_id[:20],
                )
            elif next_address:
                # No P2P node — direct gRPC only (LAN/VPC path).
                _depth_ff = max(1, int(getattr(request, "pipeline_depth", 1) or 1))
                if _depth_ff >= 2:
                    self._dispatch_direct_grpc_async(
                        next_address, next_req, label="direct",
                        request_id=str(request.request_id),
                    )
                else:
                    channel = grpc.insecure_channel(
                        next_address,
                        options=[
                            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
                            ("grpc.max_send_message_length", 100 * 1024 * 1024),
                        ],
                    )
                    stub = peer_pb2_grpc.PeerStub(channel)
                    stub.Forward(next_req, timeout=60.0)
                    channel.close()
                logger.info(
                    "push_forwarded: req=%s stage=%d -> %s",
                    request.request_id, request.stage_index, next_address,
                )
            else:
                logger.warning(
                    "push_forward_no_route: req=%s stage=%d (no direct, no relay, no address)",
                    request.request_id, request.stage_index,
                )
        except Exception as exc:
            logger.error("PUSH_FORWARD_CRASH: %s -> %s: %s", self.peer_id, next_address, exc,
                         exc_info=True)

    def _push_final_result(
        self,
        response: peer_pb2.ForwardResponse,
        callback_address: str,
        callback_request_id: str,
        callback_libp2p_peer_id: str = "",
        pipeline_depth: int = 1,
        slot_id: int = 0,
    ) -> None:
        """Send final activation back to the coordinator via PushResult RPC.

        Phase 2a: when ``pipeline_depth >= 2``, both direct-gRPC paths
        (LAN-first and last-resort) are dispatched on a daemon thread so
        the gRPC handler returns immediately and the peer can begin
        computing the next slot's forward without waiting for the
        coord-bound send to complete. Libp2p relay is already
        fire-and-forget via ``PROXY_METHOD_PUSH_RESULT``.
        """
        try:
            if callback_request_id:
                # Preserve ``activation_packed`` and ``is_hidden_state`` —
                # required for Path A (client-terminated pipeline), where
                # the last peer returns a packed hidden state and the
                # coordinator's PushResult handler routes based on the flag.
                response = peer_pb2.ForwardResponse(
                    request_id=callback_request_id,
                    peer_id=response.peer_id,
                    activation=list(response.activation),
                    activation_packed=bytes(getattr(response, "activation_packed", b"") or b""),
                    is_hidden_state=bool(getattr(response, "is_hidden_state", False)),
                    stage_index=response.stage_index,
                    error=response.error,
                    kv_cache_hit=response.kv_cache_hit,
                    activation_hash=response.activation_hash,
                    # Phase 2a: echo slot_id back so the coord-side
                    # _coordinator_handle_push_result can match this
                    # response to its in-flight SlotState (pipeline_depth>=2).
                    slot_id=int(slot_id or 0),
                    # Phase 2b live-bench Binding #2: preserve block-verify
                    # routing fields if the inbound response carried them.
                    block_size=int(getattr(response, "block_size", 0) or 0),
                    block_index=int(getattr(response, "block_index", 0) or 0),
                )
            _cb_libp2p = str(callback_libp2p_peer_id or '').strip()
            # LAN-first: if the callback address is a private IP we can
            # reach via a local interface (same /16), bypass libp2p
            # entirely. Caller is the coordinator's PushResult endpoint;
            # when coord and last-peer share a VPC subnet this saves a
            # transcontinental relay round-trip.
            from peer.lan_routing import is_reachable_lan, parse_host_from_address
            _cb_host = parse_host_from_address(callback_address)
            _cb_lan_reachable = bool(_cb_host) and is_reachable_lan(_cb_host)

            # Note: previous versions branched on
            # ``_p2p_node.is_peer_connected(_cb_libp2p)`` before picking
            # between direct gRPC and libp2p. That heuristic is wrong
            # for PushResult — libp2p-connected says nothing about
            # whether ``callback_address`` is network-reachable from
            # here. We now rely purely on (a) LAN reachability for
            # direct gRPC, (b) libp2p presence for relay, (c) last-
            # resort direct gRPC. The ``is_peer_connected`` probe is
            # still useful for the outbound Push-to-next-peer path
            # (different call site); dropped only here.

            _depth_ff = max(1, int(pipeline_depth or 1))
            if _cb_lan_reachable and callback_address:
                # LAN-direct gRPC to coordinator — fastest path.
                if _depth_ff >= 2:
                    self._dispatch_push_result_async(
                        callback_address, response, label="lan",
                    )
                else:
                    channel = grpc.insecure_channel(
                        callback_address,
                        options=[
                            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
                            ("grpc.max_send_message_length", 100 * 1024 * 1024),
                        ],
                    )
                    try:
                        stub = peer_pb2_grpc.PeerStub(channel)
                        stub.PushResult(response, timeout=10.0)
                    finally:
                        channel.close()
                logger.info(
                    "push_result_sent_via_lan: req=%s -> %s "
                    "(LAN-first; bypassing libp2p)",
                    response.request_id, callback_address,
                )
            elif self._p2p_node is not None and _cb_libp2p:
                # No direct connection — route PushResult through relay.
                self._p2p_node.proxy_forward(
                    target_peer_id=_cb_libp2p,
                    data=PROXY_METHOD_PUSH_RESULT + response.SerializeToString(),
                )
                logger.info(
                    "push_result_sent_via_relay: req=%s -> %s (libp2p=%s)",
                    response.request_id, callback_address, _cb_libp2p[:20],
                )
            else:
                # No P2P node — direct gRPC only.
                if _depth_ff >= 2:
                    self._dispatch_push_result_async(
                        callback_address, response, label="direct",
                    )
                else:
                    channel = grpc.insecure_channel(
                        callback_address,
                        options=[
                            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
                            ("grpc.max_send_message_length", 100 * 1024 * 1024),
                        ],
                    )
                    stub = peer_pb2_grpc.PeerStub(channel)
                    stub.PushResult(response, timeout=10.0)
                    channel.close()
                logger.info(
                    "push_result_sent: req=%s -> %s",
                    response.request_id, callback_address,
                )
        except Exception as exc:
            logger.warning("push_result_failed: %s: %s", callback_address, exc)

    # ── Phase 2a: fire-and-forget direct-gRPC dispatch helpers ───────
    def _dispatch_direct_grpc_async(
        self,
        next_address: str,
        next_req: peer_pb2.ForwardRequest,
        *,
        label: str,
        request_id: str,
    ) -> None:
        """Send a ForwardRequest on a daemon thread and return instantly.

        Used by ``_push_to_next_hop`` under ``pipeline_depth >= 2`` so
        the gRPC handler thread isn't blocked by the next-hop send. The
        receiving peer's Forward handler is idempotent under retry and
        Path A push mode doesn't depend on the response body.
        """
        import threading as _ff_threading

        def _send():
            try:
                ch = grpc.insecure_channel(
                    next_address,
                    options=[
                        ("grpc.max_receive_message_length", 100 * 1024 * 1024),
                        ("grpc.max_send_message_length", 100 * 1024 * 1024),
                    ],
                )
                try:
                    stub = peer_pb2_grpc.PeerStub(ch)
                    stub.Forward(next_req, timeout=60.0)
                finally:
                    ch.close()
            except Exception as exc:
                logger.warning(
                    "push_forward_async_failed: req=%s label=%s -> %s: %s",
                    request_id, label, next_address, exc,
                )

        _ff_threading.Thread(
            target=_send, daemon=True,
            name=f"oh-push-fwd-{label}",
        ).start()

    def _dispatch_push_result_async(
        self,
        callback_address: str,
        response: peer_pb2.ForwardResponse,
        *,
        label: str,
    ) -> None:
        """Send a PushResult on a daemon thread and return instantly.

        Used by ``_push_final_result`` under ``pipeline_depth >= 2`` so
        the last peer's gRPC handler isn't blocked by the coord-bound
        send — frees it to start computing the next slot's forward
        immediately.
        """
        import threading as _ff_threading

        def _send():
            try:
                ch = grpc.insecure_channel(
                    callback_address,
                    options=[
                        ("grpc.max_receive_message_length", 100 * 1024 * 1024),
                        ("grpc.max_send_message_length", 100 * 1024 * 1024),
                    ],
                )
                try:
                    stub = peer_pb2_grpc.PeerStub(ch)
                    stub.PushResult(response, timeout=10.0)
                finally:
                    ch.close()
            except Exception as exc:
                logger.warning(
                    "push_result_async_failed: req=%s label=%s -> %s: %s",
                    response.request_id, label, callback_address, exc,
                )

        _ff_threading.Thread(
            target=_send, daemon=True,
            name=f"oh-push-result-{label}",
        ).start()

    # ── Path A: coordinator-side sample-and-reinject ─────────────────
    def _handle_hidden_state_push_result(
        self,
        response: peer_pb2.ForwardResponse,
    ) -> peer_pb2.PushAck | None:
        """Sample a hidden state returned by the last peer, emit the
        token to the ring queue, and re-inject the next stage-0 request.

        Returns a ``PushAck`` on success or a terminal error. Returns
        ``None`` when no HeadSampler or RingSession is registered, so
        the caller falls through to the legacy push-receiver future.

        Phase 2b: block-verify responses (block_size > 0) are routed
        to the DFlash block queue instead of the per-token sampler.
        """
        # ── Phase 2b: route block-verify responses to DFlash queue ──
        _block_size = int(getattr(response, "block_size", 0) or 0)
        _block_index = int(getattr(response, "block_index", 0) or 0)
        if _block_size > 0:
            from coordinator.push_receiver import emit_dflash_block_response
            _packed = bytes(getattr(response, "activation_packed", b"") or b"")
            if not _packed:
                logger.error(
                    "block_verify_empty_payload_in_push: req=%s block=%d",
                    response.request_id, _block_index,
                )
                return peer_pb2.PushAck(
                    request_id=response.request_id, ok=False,
                    error="block_verify_empty_payload",
                )
            delivered = emit_dflash_block_response(
                request_id=str(response.request_id),
                block_index=_block_index,
                activation_packed=_packed,
                block_size=_block_size,
            )
            logger.info(
                "block_verify_routed_to_dflash: req=%s block=%d "
                "packed_bytes=%d delivered=%s",
                response.request_id, _block_index,
                len(_packed), delivered,
            )
            return peer_pb2.PushAck(
                request_id=response.request_id, ok=True, error="",
            )

        from coordinator.head_sampler import (
            get_head_sampler, get_ring_session, unregister_ring_session,
        )
        from coordinator.push_receiver import emit_ring_token

        sampler = get_head_sampler()
        if sampler is None:
            logger.error(
                "push_result_hidden_state_no_sampler: req=%s — "
                "last peer set is_hidden_state=True but no co-located "
                "HeadSampler is registered on the coordinator",
                response.request_id,
            )
            return peer_pb2.PushAck(
                request_id=response.request_id,
                ok=False,
                error="no_head_sampler_registered",
            )

        session = get_ring_session(str(response.request_id))
        if session is None:
            logger.error(
                "push_result_hidden_state_no_session: req=%s — "
                "HeadSampler present but RingSession missing (state was "
                "not registered at ring-launch time)",
                response.request_id,
            )
            return peer_pb2.PushAck(
                request_id=response.request_id,
                ok=False,
                error="no_ring_session_registered",
            )

        # ── Decode hidden-state payload ─────────────────────────────
        # Prefer zero-copy packed bytes; fall back to repeated-float.
        _packed = bytes(getattr(response, "activation_packed", b"") or b"")
        _activation_list: list[float] | None = None
        if not _packed:
            _activation_list = list(response.activation)
            if not _activation_list:
                logger.error(
                    "push_result_hidden_state_empty_payload: req=%s",
                    response.request_id,
                )
                return peer_pb2.PushAck(
                    request_id=response.request_id, ok=False,
                    error="empty_hidden_state",
                )

        # ── Apply final head + sample ────────────────────────────────
        try:
            token_id = int(sampler.sample(
                _activation_list if _activation_list is not None else [],
                session.decode,
                packed_bytes=_packed if _packed else None,
            ))
        except Exception as exc:
            import traceback as _tb
            logger.error(
                "push_result_sample_failed: req=%s err=%s\n%s",
                response.request_id, exc, _tb.format_exc(),
            )
            emit_ring_token(session.callback_request_id, None)
            unregister_ring_session(response.request_id)
            return peer_pb2.PushAck(
                request_id=response.request_id, ok=False,
                error=f"sample_failed:{exc}",
            )

        # ── Emit to ring queue + accounting ──────────────────────────
        session.ring_generated_ids.append(token_id)
        session.ring_tokens_remaining = max(0, session.ring_tokens_remaining - 1)
        emit_ring_token(session.callback_request_id, token_id)
        logger.info(
            "coordinator_ring_sampled: req=%s token=%d remaining=%d eos_hit=%s",
            response.request_id, token_id, session.ring_tokens_remaining,
            (token_id in session.ring_eos_ids),
        )

        # ── Termination check ────────────────────────────────────────
        _is_eos = token_id in session.ring_eos_ids
        if session.ring_tokens_remaining <= 0 or _is_eos:
            emit_ring_token(session.callback_request_id, None)
            unregister_ring_session(response.request_id)
            return peer_pb2.PushAck(
                request_id=response.request_id, ok=True, error="",
            )

        # ── Re-inject into stage 0 ───────────────────────────────────
        try:
            self._coordinator_reinject_ring_step(session, token_id)
        except Exception as exc:
            import traceback as _tb
            logger.error(
                "push_result_reinject_failed: req=%s err=%s\n%s",
                response.request_id, exc, _tb.format_exc(),
            )
            emit_ring_token(session.callback_request_id, None)
            unregister_ring_session(response.request_id)
            return peer_pb2.PushAck(
                request_id=response.request_id, ok=False,
                error=f"reinject_failed:{exc}",
            )

        return peer_pb2.PushAck(
            request_id=response.request_id, ok=True, error="",
        )

    def _coordinator_reinject_ring_step(
        self,
        session: "RingSession",  # type: ignore[name-defined]
        token_id: int,
    ) -> None:
        """Build and fire a new ForwardRequest for the next ring cycle.

        Fire-and-forget — mirrors the existing peer-side ring-loopback
        but runs on the coordinator. The request carries a single-token
        activation (``[float(token_id)]``) and enters at stage 0. The
        ``sample_on_coordinator`` flag is preserved so the next cycle
        also terminates on the coordinator.
        """
        route = list(session.ring_full_route)
        _next_next = route[1].address if len(route) > 1 else ""
        _next_next_id = route[1].peer_id if len(route) > 1 else ""
        req = peer_pb2.ForwardRequest(
            request_id=session.request_id,
            activation=[float(token_id)],
            stage_index=0,
            total_stages=int(session.total_stages),
            max_tokens=1,
            kv_session_id=session.kv_session_id,
            kv_store_activation=True,
            kv_use_cached_activation=True,
            decode_do_sample=session.decode.do_sample,
            decode_temperature=float(session.decode.temperature or 0.0),
            decode_top_p=float(session.decode.top_p or 0.0),
            decode_top_k=int(session.decode.top_k or 0),
            decode_seed=int(session.decode.seed or 0),
            shard_layer_start=int(session.stage0_layer_start),
            shard_layer_end=int(session.stage0_layer_end),
            shard_total_layers=int(session.stage0_total_layers),
            push_mode=True,
            ring_mode=True,
            sample_on_coordinator=True,
            ring_tokens_remaining=int(session.ring_tokens_remaining),
            ring_generated_ids=list(session.ring_generated_ids),
            ring_eos_ids=list(session.ring_eos_ids),
            ring_first_hop_address=session.ring_first_hop_address,
            ring_first_hop_peer_id=session.ring_first_hop_peer_id,
            ring_first_hop_libp2p_id=session.ring_first_hop_libp2p_id,
            ring_full_route=route,
            next_hop_address=_next_next,
            next_hop_peer_id=_next_next_id,
            final_callback_address=session.final_callback_address,
            final_callback_request_id=session.callback_request_id,
            final_callback_libp2p_peer_id=session.final_callback_libp2p_peer_id,
            remaining_route=route[1:],
        )

        _first_addr = session.ring_first_hop_address
        _first_libp2p = session.ring_first_hop_libp2p_id

        def _fire(_rreq=req, _addr=_first_addr, _libp2p=_first_libp2p):
            try:
                logger.info(
                    "COORD_REINJECT_START: req=%s remaining=%d -> %s (libp2p=%s)",
                    _rreq.request_id, _rreq.ring_tokens_remaining,
                    _addr, _libp2p[:20] if _libp2p else "none",
                )
                # LAN-first: when the coordinator and stage-0 peer share
                # a /16, skip libp2p entirely. Most likely irrelevant for
                # the pure-coordinator case (Mac coord ↔ remote VPC peer)
                # but matches the routing rule applied symmetrically on
                # other hops.
                from peer.lan_routing import (
                    is_reachable_lan as _ir, parse_host_from_address as _ph,
                )
                _lh = _ph(_addr)
                _lan_ok = bool(_lh) and _ir(_lh)
                if _lan_ok and _addr:
                    _ch = grpc.insecure_channel(
                        _addr,
                        options=[
                            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
                            ("grpc.max_send_message_length", 100 * 1024 * 1024),
                        ],
                    )
                    try:
                        _stub = peer_pb2_grpc.PeerStub(_ch)
                        _stub.Forward(_rreq, timeout=60.0)
                    finally:
                        _ch.close()
                    logger.info(
                        "COORD_REINJECT_DONE: req=%s via_lan",
                        _rreq.request_id,
                    )
                elif self._p2p_node is not None and _libp2p:
                    self._p2p_node.proxy_forward(
                        target_peer_id=_libp2p,
                        data=PROXY_METHOD_FIRE_FORGET + _rreq.SerializeToString(),
                    )
                    logger.info(
                        "COORD_REINJECT_DONE: req=%s via_relay",
                        _rreq.request_id,
                    )
                elif _addr:
                    _ch = grpc.insecure_channel(
                        _addr,
                        options=[
                            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
                            ("grpc.max_send_message_length", 100 * 1024 * 1024),
                        ],
                    )
                    _stub = peer_pb2_grpc.PeerStub(_ch)
                    _stub.Forward(_rreq, timeout=60.0)
                    _ch.close()
                    logger.info(
                        "COORD_REINJECT_DONE: req=%s via_grpc",
                        _rreq.request_id,
                    )
                else:
                    logger.error(
                        "COORD_REINJECT_NO_ROUTE: req=%s", _rreq.request_id,
                    )
            except Exception as exc:
                logger.error(
                    "COORD_REINJECT_CRASH: req=%s err=%s",
                    _rreq.request_id, exc, exc_info=True,
                )

        threading.Thread(target=_fire, daemon=True).start()

    def PushResult(self, request: peer_pb2.ForwardResponse, context: grpc.ServicerContext) -> peer_pb2.PushAck:
        """Receive final result from last peer in push chain.

        This handler is primarily used when this node is the coordinator.
        The push_receiver module registers a callback for the request_id.
        """
        _is_hidden = bool(getattr(request, "is_hidden_state", False))
        logger.info(
            "push_result_received: req=%s from=%s is_hidden_state=%s",
            request.request_id, request.peer_id, _is_hidden,
        )
        # Client-terminated pipeline (Path A, flag-gated): when the last
        # peer returns a hidden state instead of a sampled token, route to
        # the coordinator-side HeadSampler. Phase 1 only performs the
        # dispatch — the actual sample-and-reinject loop lands in Phase 3.
        # Until then, calls that arrive here with is_hidden_state=True will
        # surface NotImplementedError from HeadSampler.sample(), which is
        # the intended fail-loud behaviour while the flag is default-off.
        if _is_hidden:
            try:
                _ack = self._handle_hidden_state_push_result(request)
                if _ack is not None:
                    return _ack
            except Exception as exc:  # pragma: no cover — defensive
                import traceback as _tb
                logger.error(
                    "push_result_hidden_state_crash: req=%s err=%s\n%s",
                    request.request_id, exc, _tb.format_exc(),
                )
                # Fall through to the legacy push-receiver future below so
                # any awaiting caller sees a terminal response.
        # Dispatch to the push receiver if registered (existing behaviour —
        # covers both (a) is_hidden_state=False (today's token payload) and
        # (b) the Phase-1 hidden-state stub path above).
        try:
            from coordinator.push_receiver import _PUSH_RESULTS
            future = _PUSH_RESULTS.pop(str(request.request_id), None)
            if future is not None:
                future.set_result(request)
        except ImportError:
            pass
        return peer_pb2.PushAck(request_id=request.request_id, ok=True, error="")

    # ── Phase 3A: Bidirectional streaming ──────────────────────────────────

    def _verify_draft_tokens(
        self,
        prompt: str,
        draft_token_ids: list[int],
        decode_temperature: float = 0.0,
        decode_top_p: float = 0.0,
        decode_top_k: int = 0,
        decode_seed: int = 0,
    ) -> list[int]:
        """Verify K draft tokens by running each through the model.

        For each draft token position, the model produces its own
        next-token prediction.  These are compared against the draft
        tokens by the coordinator's accept/reject logic.

        Returns:
            List of K verified token IDs (the model's actual predictions).
        """
        verified: list[int] = []
        for draft_id in draft_token_ids:
            # Run forward with the draft token as activation input
            result = self.shard.forward(
                prompt=prompt if not verified else "",
                activation=[float(draft_id)],
                max_tokens=1,
                stage_index=0,
                total_stages=1,
                decode_temperature=decode_temperature,
                decode_top_p=decode_top_p,
                decode_top_k=decode_top_k,
                decode_seed=(decode_seed if decode_seed > 0 else None),
            )
            if result:
                verified.append(int(round(result[0])))
            else:
                verified.append(0)
        return verified

    def ForwardStream(self, request_iterator, context: grpc.ServicerContext):
        """Bidirectional streaming RPC — persistent session with KV reuse.

        Processes an incoming stream of ForwardRequests, maintaining KV cache
        state across the stream's lifetime.  Each request yields a response.
        The stream auto-closes after ``_stream_idle_timeout_s`` of inactivity.

        Args:
            request_iterator: Iterator of ForwardRequest messages.
            context: gRPC servicer context.

        Yields:
            ForwardResponse for each incoming request.
        """
        import time as _time

        idle_timeout = getattr(self, "_stream_idle_timeout_s", 30.0)
        session_kv_key = ""
        last_activity = _time.monotonic()

        logger.info("forward_stream_opened: peer=%s", self.peer_id)

        try:
            for request in request_iterator:
                now = _time.monotonic()
                if now - last_activity > idle_timeout:
                    logger.info(
                        "forward_stream_idle_timeout: peer=%s timeout=%.1fs",
                        self.peer_id, idle_timeout,
                    )
                    break

                last_activity = now

                # Enable KV reuse across the stream lifetime.
                if not session_kv_key and request.kv_session_id:
                    session_kv_key = str(request.kv_session_id)

                # Delegate to the unary Forward handler for each message.
                response = self.Forward(request, context)
                yield response
        except Exception as exc:
            logger.warning("forward_stream_error: peer=%s err=%s", self.peer_id, exc)
            yield peer_pb2.ForwardResponse(
                request_id="",
                peer_id=self.peer_id,
                activation=[],
                stage_index=0,
                error=f"stream_error: {exc}",
            )
        finally:
            logger.info("forward_stream_closed: peer=%s", self.peer_id)

    def record_next_hop_rtt(self, downstream_peer_id: str, rtt_ms: float) -> None:
        """Record the measured RTT to a downstream peer for S2S routing.

        Called after Forward() completes to store the latency to the next
        hop in the pipeline.  This data is broadcast in the next DHT
        announce cycle so coordinators can build S2S-aware pipelines.

        Args:
            downstream_peer_id: The peer_id of the downstream hop.
            rtt_ms: Measured round-trip time in milliseconds.
        """
        if not downstream_peer_id or rtt_ms <= 0:
            return
        with self._next_hop_rtts_lock:
            self._next_hop_rtts[downstream_peer_id] = round(rtt_ms, 2)

    def get_next_hop_rtts_json(self) -> str:
        """Return JSON-encoded S2S RTT dict for DHT announcement."""
        with self._next_hop_rtts_lock:
            if not self._next_hop_rtts:
                return ""
            import json as _json
            return _json.dumps(self._next_hop_rtts)

    def GetPeerStatus(
        self,
        request: peer_pb2.PeerStatusRequest,
        context: grpc.ServicerContext,
    ) -> peer_pb2.PeerStatusResponse:
        return peer_pb2.PeerStatusResponse(
            peer_id=self.peer_id,
            model_id=self.model_id,
            shard_index=self.shard_index,
            total_shards=self.total_shards,
            load_pct=self._load_pct(),
            healthy=True,
            daemon_mode=self.daemon_mode,
            dp_noise_configured_variance=float(self.shard.privacy_noise_variance),
            dp_noise_payloads=int(self.shard.privacy_noise_payloads),
            dp_noise_observed_variance_ema=float(self.shard.privacy_noise_observed_variance_ema),
            dp_noise_last_audit_tag=str(self.shard.privacy_noise_last_audit_tag),
        )


def _seeding_loop(
    *,
    stop_event: threading.Event,
    service: PeerService,
    session_manager: TorrentSessionManager,
    advertised_bandwidth_mbps: float,
    update_interval_sec: int,
) -> None:
    while not stop_event.is_set():
        inference_active = service.inflight_count() > 0
        observed = 0.0
        if advertised_bandwidth_mbps > 0:
            observed = advertised_bandwidth_mbps * (service._load_pct() / 100.0)

        session_manager.update(
            inference_active=inference_active,
            inference_observed_mbps=observed,
        )
        stop_event.wait(max(1, update_interval_sec))


def _daemon_budget_loop(
    *,
    stop_event: threading.Event,
    service: PeerService,
    controller: DaemonController,
    refresh_interval_sec: int,
) -> None:
    while not stop_event.is_set():
        try:
            budget = controller.refresh()
            service.set_resource_budget(budget)
        except Exception as exc:  # pragma: no cover
            logging.debug("daemon monitor refresh failed: %s", exc)
        stop_event.wait(max(1, refresh_interval_sec))


def _probe_available_vram_mb() -> int:
    """Return free GPU VRAM in MB.  Returns 0 when undeterminable.

    Priority:
      1. CUDA:  ``torch.cuda.mem_get_info`` (exact free bytes)
      2. Metal: ``mlx.metal.device_info`` recommendedMaxWorkingSetSize
                (safe-to-use total; used as a proxy since MLX allocates lazily)
      3. Fallback: 0 (coordinator treats this as "assume capable")
    """
    try:
        import torch
        if torch.cuda.is_available():
            free_bytes, _total = torch.cuda.mem_get_info(0)
            return int(free_bytes // (1024 * 1024))
    except Exception:
        pass
    try:
        import mlx.core as mx
        info = mx.metal.device_info()
        return int(info.get("recommendedMaxWorkingSetSize", 0) // (1024 * 1024))
    except Exception:
        pass
    return 0


def _sign_announce_safe(private_key: object, peer_id: str, host: str, port: int, model_id: str) -> str:
    """Sign an announce payload; returns empty string on any error."""
    try:
        from peer.identity import sign_announce as _sign_announce
        return _sign_announce(private_key, peer_id, host, port, model_id)  # type: ignore[arg-type]
    except Exception:
        return ""


def _announce_loop(
    *,
    stop_event: threading.Event,
    service: PeerService,
    dht_urls: list[str] | tuple[str, ...],
    advertise_host: str,
    port: int,
    operator_id: str | None,
    region: str | None,
    bandwidth_mbps: float,
    announce_interval_sec: int,
    announce_ttl_sec: int,
    session_manager: TorrentSessionManager | None,
    announced_reputation_score: float,
    announced_staked_balance: float,
    peer_public_key: str = "",
    announce_private_key: object = None,
    announce_public_key_hex: str = "",
    seeder_http_port: int = 0,
    p2p_cache: P2PModelCache | None = None,
    local_fast_path_port: int = 0,
    hivemind_adapter: Any = None,
    p2p_node: object | None = None,
    # Phase 3 zero-config: capacity snapshot attached to every announcement.
    capacity_json: str = "",
    capacity_schema_version: int = 0,
    # Phase 4 zero-config: live snapshot written by NegotiationLoop.  When
    # provided, each iteration reads fresh capacity_json + assignment from
    # this object, superseding the static kwargs above.
    capacity_snapshot_ref: Any = None,
    # Phase 2b: when True, operator specified explicit --layer-start/end
    # and the negotiator's assignment must NOT override the announced
    # layer range (the assignment reflects "desired" not "loaded").
    manual_shard: bool = False,
) -> None:
    announce_interval = max(1.0, float(announce_interval_sec))
    announced_once = False
    consecutive_failures = 0
    outage_window_s = 0.0
    while not stop_event.is_set():
        seeding_snapshot: dict[str, Any] = (
            session_manager.snapshot() if session_manager is not None else {
                "seeding_enabled": False,
                "arbitration": {
                    "effective_seed_upload_limit_mbps": 0.0,
                    "target_seed_upload_limit_mbps": 0.0,
                    "inference_active": False,
                },
            }
        )
        arbitration = dict(seeding_snapshot.get("arbitration") or {})
        runtime_profile = dict(service.runtime_profile or {})
        _cstats = service.compaction_stats()

        # ── Phase 4: read fresh capacity + assignment from the live
        # NegotiationLoop snapshot if one is wired in.  Falls back to
        # the static kwargs (Phase 3 path) when no snapshot is present.
        _effective_capacity_json = capacity_json
        _effective_capacity_schema_version = capacity_schema_version
        _effective_layer_start = int(runtime_profile.get("layer_start", 0))
        _effective_layer_end = int(runtime_profile.get("layer_end", 0))
        _effective_total_layers = int(runtime_profile.get("total_layers", 0))
        _effective_model_id = service.model_id
        if capacity_snapshot_ref is not None:
            try:
                _snap_json, _snap_ver, _snap_assignment, _ = (
                    capacity_snapshot_ref.snapshot()
                )
                if _snap_json:
                    _effective_capacity_json = _snap_json
                    _effective_capacity_schema_version = int(_snap_ver or 0)
                if _snap_assignment is not None and not manual_shard:
                    # Live assignment overrides the static runtime_profile
                    # layer range so the announce tells neighbours where
                    # we'd *like* to be — even before the next reshard lands.
                    # Skipped when manual_shard=True because the operator
                    # explicitly chose a layer range via --layer-start/end
                    # and the negotiator's "desired" range doesn't reflect
                    # the actually loaded layers.
                    _effective_layer_start = int(_snap_assignment.layer_start)
                    _effective_layer_end = int(_snap_assignment.layer_end)
                    _effective_total_layers = int(_snap_assignment.total_layers)
                    _effective_model_id = str(_snap_assignment.model_id or service.model_id)
            except Exception as _snap_err:
                logging.debug(
                    "announce_loop_snapshot_read_failed: %s — "
                    "falling back to static capacity_json",
                    _snap_err,
                )

        announcement = Announcement(
            peer_id=service.peer_id,
            model_id=str(_effective_model_id),
            host=advertise_host,
            port=port,
            operator_id=operator_id,
            region=region,
            load_pct=service._load_pct(),
            daemon_mode=service.daemon_mode,
            bandwidth_mbps=bandwidth_mbps,
            seeding_enabled=bool(seeding_snapshot.get("seeding_enabled", False)),
            seed_upload_limit_mbps=float(arbitration.get("effective_seed_upload_limit_mbps", 0.0)),
            seed_target_upload_limit_mbps=float(arbitration.get("target_seed_upload_limit_mbps", 0.0)),
            seed_inference_active=bool(arbitration.get("inference_active", False)),
            runtime_backend=str(runtime_profile.get("backend", "toy_cpu")),
            runtime_target=str(runtime_profile.get("target", "cpu")),
            runtime_model_id=str(runtime_profile.get("runtime_model_id", "")),
            quantization_mode=str(runtime_profile.get("quantization_mode", "fp32")),
            quantization_bits=int(runtime_profile.get("quantization_bits", 0)),
            runtime_gpu_available=bool(runtime_profile.get("gpu_available", False)),
            runtime_estimated_tokens_per_sec=float(runtime_profile.get("estimated_tokens_per_sec", 0.0)),
            runtime_estimated_memory_mb=int(runtime_profile.get("estimated_memory_mb", 0)),
            privacy_noise_variance=float(service.shard.privacy_noise_variance),
            privacy_noise_payloads=int(service.shard.privacy_noise_payloads),
            privacy_noise_observed_variance_ema=float(service.shard.privacy_noise_observed_variance_ema),
            privacy_noise_last_audit_tag=str(service.shard.privacy_noise_last_audit_tag),
            reputation_score=max(0.0, float(announced_reputation_score)),
            staked_balance=max(0.0, float(announced_staked_balance)),
            expert_tags=tuple(service.expert_tags),
            expert_layer_indices=tuple(service.expert_layer_indices),
            expert_router=bool(service.expert_router),
            peer_public_key=str(peer_public_key or ""),
            public_key=str(announce_public_key_hex or ""),
            signature=(_sign_announce_safe(announce_private_key, service.peer_id, advertise_host, port, service.model_id) if announce_private_key is not None else ""),
            available_vram_mb=_probe_available_vram_mb(),
            available_kv_slots=max(0, service.kv_cache_max_entries - len(getattr(service, '_kv_cache', {}))),
            next_hop_rtts_json=service.get_next_hop_rtts_json(),
            compact_tokens_saved_total=int(_cstats.get("compact_tokens_saved", 0)),
            compact_latency_total_ms=round(float(_cstats.get("compact_latency_s", 0.0)) * 1000, 1),
            layer_start=_effective_layer_start,
            layer_end=_effective_layer_end,
            total_layers=_effective_total_layers,
            seeder_http_port=int(seeder_http_port),
            cached_model_ids=tuple(p2p_cache.announce_cached_models() if p2p_cache is not None else []),
            local_fast_path_port=int(local_fast_path_port),
            nat_type=str(getattr(service, '_nat_type', 'unknown')),
            requires_relay=bool(getattr(service, '_requires_relay', False)),
            relay_peer_id=str(getattr(service, '_relay_peer_id', '')),
            relay_address=str(getattr(service, '_relay_address', '')),
            libp2p_peer_id=str(getattr(p2p_node, 'libp2p_peer_id', '') if p2p_node is not None else ''),
            # Phase 3/4 zero-config: capacity snapshot for swarm negotiation.
            # When a live NegotiationLoop is wired in via capacity_snapshot_ref
            # these values reflect the latest tick; otherwise they are the
            # static boot-time values passed as kwargs.  Empty string means
            # "no capacity info available" and is ignored by readers.
            capacity_json=str(_effective_capacity_json or ""),
            capacity_schema_version=int(_effective_capacity_schema_version or 0),
        )
        try:
            # HTTP DHT announce (legacy path).
            if dht_urls:
                successes, failures = announce_http_many(
                    announcement,
                    dht_urls=dht_urls,
                    ttl_seconds=announce_ttl_sec,
                    heartbeat=announced_once,
                )
                if not successes:
                    sample_error = next(iter(failures.values())) if failures else RuntimeError("dht_announce_no_targets")
                    raise RuntimeError(sample_error)
                for failed_url, exc in failures.items():
                    logging.warning("DHT announce partial failure for %s via %s (%s)", service.peer_id, failed_url, exc)
                if consecutive_failures > 0:
                    logging.info(
                        "DHT announce recovered for %s after %s retries",
                        service.peer_id,
                        consecutive_failures,
                    )
                if not announced_once:
                    logging.info("peer %s announced to %s DHT endpoints", service.peer_id, len(successes))

            # Hivemind DHT announce (dual-stack path).
            if hivemind_adapter is not None:
                try:
                    from dataclasses import asdict as _asdict
                    _hm_payload = _asdict(announcement)
                    _hm_payload["updated_unix_ms"] = int(time.time() * 1000)
                    hivemind_adapter.announce(_hm_payload, ttl_seconds=announce_ttl_sec)
                except Exception as hm_exc:
                    logging.debug("hivemind_announce_error: %s", hm_exc)

            # Rust Kademlia DHT announce (libp2p path).
            if p2p_node is not None:
                try:
                    from dataclasses import asdict as _asdict_p2p
                    _p2p_record = _asdict_p2p(announcement)
                    _p2p_record["updated_unix_ms"] = int(time.time() * 1000)
                    # Add libp2p PeerId for Rust-side addressing.
                    _p2p_record["libp2p_peer_id"] = getattr(p2p_node, "libp2p_peer_id", "")
                    p2p_node.announce(record=_p2p_record)
                    if not announced_once:
                        logging.info(
                            "peer %s announced to Kademlia DHT (libp2p)",
                            service.peer_id,
                        )
                except Exception as _p2p_exc:
                    logging.debug("p2p_announce_error: %s", _p2p_exc)

            announced_once = True
            consecutive_failures = 0
            outage_window_s = 0.0
            delay_s = announce_interval

            # Poll for layer rebalance directives.
            try:
                from coordinator.rebalancer import poll_directives_from_dht, RebalanceDirective
                rebalance_directives = poll_directives_from_dht(
                    peer_id=service.peer_id,
                    dht_urls=list(dht_urls),
                    timeout_s=2.0,
                )
                for directive in rebalance_directives:
                    if directive.is_expired:
                        continue
                    # Safety guard: only apply when idle (no inflight requests).
                    if service.inflight_count() > 0:
                        logging.info(
                            "rebalance_deferred: inflight=%d directive=[%d,%d)",
                            service.inflight_count(),
                            directive.new_layer_start,
                            directive.new_layer_end,
                        )
                        break
                    logging.info(
                        "rebalance_applying: peer=%s [%d,%d) reason=%s",
                        service.peer_id,
                        directive.new_layer_start,
                        directive.new_layer_end,
                        directive.reason,
                    )
                    # Drain guard: set load to 100% to prevent routing.
                    service.set_resource_budget(ResourceBudget(
                        vram_fraction=0.0,
                        cpu_fraction=0.0,
                        should_yield=True,
                        reason="resharding",
                    ))
                    # Brief drain period.
                    import time as _time_mod
                    _time_mod.sleep(min(5.0, max(0.5, float(announce_interval / 12.0))))

                    # Apply the reshard.
                    ok = service.shard.reshard(
                        directive.new_layer_start,
                        directive.new_layer_end,
                        directive.total_layers,
                    )
                    if ok:
                        # Update the runtime profile so next announce reflects new range.
                        service.runtime_profile = dict(service.shard.runtime_profile())
                        logging.info(
                            "rebalance_success: peer=%s new_range=[%d,%d)",
                            service.peer_id,
                            directive.new_layer_start,
                            directive.new_layer_end,
                        )
                    else:
                        logging.warning(
                            "rebalance_failed: peer=%s directive=[%d,%d)",
                            service.peer_id,
                            directive.new_layer_start,
                            directive.new_layer_end,
                        )
                    # Restore normal budget.
                    service.set_resource_budget(ResourceBudget(
                        vram_fraction=1.0,
                        cpu_fraction=1.0,
                        should_yield=False,
                        reason="default",
                    ))
                    break  # Only apply one directive per cycle.
            except Exception as exc:
                logging.debug("rebalance_poll_error: %s", exc)

            # Autonomous rebalancing: peer decides its own layer assignment.
            _rebalancer = getattr(service, '_autonomous_rebalancer', None)
            _rebalance_interval = int(getattr(service, '_rebalance_check_interval', 6))
            _announce_count = getattr(service, '_announce_count', 0)
            service._announce_count = _announce_count + 1
            if (
                _rebalancer is not None
                and _announce_count > 0
                and _announce_count % _rebalance_interval == 0
            ):
                try:
                    from peer.autonomous_rebalancer import load_swarm_snapshot
                    _rp = dict(service.runtime_profile or {})
                    _my_start = int(_rp.get("layer_start", 0))
                    _my_end = int(_rp.get("layer_end", 0))
                    _my_tps = float(_rp.get("estimated_tokens_per_sec", 0))
                    _total = int(_rp.get("total_layers", 0))
                    if _my_end > _my_start and _total > 0 and _my_tps > 0:
                        _swarm = load_swarm_snapshot(list(dht_urls), service.model_id)
                        _decision = _rebalancer.check(
                            my_peer_id=service.peer_id,
                            my_layer_start=_my_start,
                            my_layer_end=_my_end,
                            my_tps=_my_tps,
                            swarm_peers=_swarm,
                            total_layers=_total,
                        )
                        if _decision is not None:
                            _rebalancer.apply_with_jitter(
                                service, _decision, _my_start, _my_end,
                            )
                except Exception as _rebal_exc:
                    logging.debug("autonomous_rebalance_error: %s", _rebal_exc)

        except Exception as exc:  # pragma: no cover
            consecutive_failures += 1
            delay_s = _exponential_backoff_delay(
                consecutive_failures - 1,
                base_seconds=1.0,
                cap_seconds=max(announce_interval, 300.0),
            )
            outage_window_s += delay_s
            # If outage exceeds TTL, force a fresh announce instead of heartbeat on recovery.
            if outage_window_s >= max(announce_interval, float(max(1, announce_ttl_sec))):
                announced_once = False
            logging.warning(
                "DHT announce transient failure for %s (%s); retrying in %.1fs",
                service.peer_id,
                exc,
                delay_s,
            )

        stop_event.wait(delay_s)


def _build_torrent_session_manager(
    *,
    model_id: str,
    seed_cache_dir: str,
    seed_local_path: str | None,
    seed_source_url: str | None,
    seed_expected_sha256: str | None,
    seed_force_refresh: bool,
    seed_piece_bytes: int,
    seed_base_upload_mbps: float,
    seed_inference_fraction: float,
    seed_min_upload_mbps: float,
    seed_smoothing_alpha: float,
) -> TorrentSessionManager:
    bootstrap = SessionBootstrapConfig(
        model_id=model_id,
        cache_dir=seed_cache_dir,
        local_path=seed_local_path,
        source_url=seed_source_url,
        expected_sha256=seed_expected_sha256,
        force_refresh=seed_force_refresh,
        piece_bytes=max(1, seed_piece_bytes),
    )
    arbitration = ArbitrationConfig(
        base_upload_mbps=max(1.0, seed_base_upload_mbps),
        inference_seed_fraction=max(0.01, min(1.0, seed_inference_fraction)),
        min_seed_upload_mbps=max(0.1, seed_min_upload_mbps),
        smoothing_alpha=max(0.01, min(1.0, seed_smoothing_alpha)),
    )
    manager = TorrentSessionManager(bootstrap=bootstrap, arbitration=arbitration)
    manager.bootstrap()
    manager.update(inference_active=False, inference_observed_mbps=0.0)
    return manager


def serve(
    host: str,
    port: int,
    peer_id: str,
    model_id: str,
    shard_index: int,
    total_shards: int,
    daemon_mode: str = "polite",
    broken: bool = False,
    dht_url: str | None = None,
    dht_urls: list[str] | tuple[str, ...] | None = None,
    advertise_host: str | None = None,
    operator_id: str | None = None,
    region: str | None = None,
    bandwidth_mbps: float = 0.0,
    announce_interval_sec: int = 60,
    announce_ttl_sec: int = 300,
    tls_enable: bool = False,
    tls_cert_path: str | None = None,
    tls_key_path: str | None = None,
    tls_client_ca_path: str | None = None,
    tls_require_client_auth: bool = False,
    seed_enable: bool = False,
    seed_cache_dir: str = ".cache/openhydra",
    seed_local_path: str | None = None,
    seed_source_url: str | None = None,
    seed_expected_sha256: str | None = None,
    seed_force_refresh: bool = False,
    seed_piece_bytes: int = 1 * 1024 * 1024,
    seed_base_upload_mbps: float = 100.0,
    seed_inference_fraction: float = 0.10,
    seed_min_upload_mbps: float = 1.0,
    seed_smoothing_alpha: float = 0.35,
    seed_update_interval_sec: int = 1,
    daemon_idle_threshold_sec: int = 5 * 60,
    daemon_refresh_interval_sec: int = 5,
    daemon_high_load_threshold: float = 0.85,
    daemon_assume_idle_when_unknown: bool = True,
    advanced_encryption_enabled: bool = False,
    advanced_encryption_seed: str = "openhydra-tier3-dev-seed",
    kv_cache_max_entries: int = 1024,
    runtime_backend: str = "toy_auto",
    runtime_target: str = "auto",
    quantization_mode: str = "fp32",
    runtime_model_id: str = "Qwen/Qwen3.5-0.8B",
    hf_model_id: str = "",
    mlx_force_hf_tokenizer: bool = True,
    tokenizer_vocab_guard: bool = True,
    tensor_autoencoder_enabled: bool = False,
    tensor_autoencoder_latent_dim: int = 1024,
    privacy_noise_variance: float = 0.0,
    geo_challenge_seed: str = "openhydra-geo-dev-seed",
    announced_reputation_score: float = 0.0,
    announced_staked_balance: float = 0.0,
    expert_tags: tuple[str, ...] = (),
    expert_layer_indices: tuple[int, ...] = (),
    expert_router: bool = False,
    data_dir: str = ".openhydra",
    identity_path: str = ".openhydra/identity.key",
    kv_compaction_enabled: bool = False,
    kv_compaction_method: str = "hak",
    kv_compaction_ratio: float = 0.10,
    kv_compaction_beta: bool = False,
    kv_compaction_head_budget_path: str = "",
    kv_compaction_online: bool = False,
    kv_compaction_online_max_tokens: int = 512,
    kv_compaction_mode: str | None = None,
    kv_compaction_auto_threshold: int = 512,
    kv_radix_cache_enabled: bool = False,
    kv_radix_cache_max_entries: int = 128,
    kv_radix_cache_min_prefix_len: int = 16,
    warmup_on_start: bool = False,
    mlx_eval_timeout_s: float = 120.0,
    batch_window_ms: float = 50.0,
    max_batch_size: int = 8,
    load_full_head: bool = False,
    pipeline_depth: int = 1,
    p2p_enable: bool = False,
    seeder_port: int = 0,
    p2p_cache_dir: str | None = None,
    enable_local_fast_path: bool = False,
    hivemind_initial_peers: list[str] | None = None,
    rebalance_enabled: bool = False,
    rebalance_interval: int = 6,
    rebalance_min_improvement: float = 1.15,
    rebalance_cooldown_s: float = 300.0,
    relay_address: str = "",
    p2p_node: object | None = None,
    # Phase 3 zero-config bootstrap: capacity snapshot emitted alongside
    # every DHT announcement.  ``capacity_json`` is a JSON-string payload
    # produced by :func:`peer.capacity.build_capacity_report` then
    # ``json.dumps(report.to_dict())``.  Empty string = no capacity info
    # attached (backward-compatible with Phase 1/2 callers that don't
    # know about the CapacityEngine yet).
    capacity_json: str = "",
    capacity_schema_version: int = 0,
    # Phase 4 zero-config: continuous re-negotiation.
    # When ``capacity_snapshot_ref`` is provided, the announce loop reads
    # the latest capacity_json + current assignment from this live
    # :class:`peer.negotiation_loop.LoopSnapshot` object — overriding
    # ``capacity_json`` / ``capacity_schema_version`` above.  When
    # ``negotiation_loop_factory`` is also provided, serve() invokes it
    # with ``is_busy_fn=lambda: service.inflight_count() > 0`` and starts
    # the returned loop alongside the announce loop.  Either can be None
    # for backward-compat (Phase 1–3 callers).
    capacity_snapshot_ref: Any = None,
    negotiation_loop_factory: Any = None,
    # Phase 2b: when True, the operator specified an explicit layer range
    # via --layer-start/--layer-end and the swarm negotiator's assignment
    # must NOT override the DHT-announced layer range (the negotiator's
    # "desired" range can differ from the actually loaded layers).
    manual_shard: bool = False,
) -> None:
    resolved_dht_urls: list[str] = []
    seen_dht_urls: set[str] = set()
    for raw in list(dht_urls or []):
        for token in str(raw).split(","):
            value = token.strip()
            if not value or value in seen_dht_urls:
                continue
            seen_dht_urls.add(value)
            resolved_dht_urls.append(value)
    if dht_url:
        for token in str(dht_url).split(","):
            value = token.strip()
            if not value or value in seen_dht_urls:
                continue
            seen_dht_urls.add(value)
            resolved_dht_urls.append(value)

    from peer.identity import load_or_create_identity as _load_or_create_identity
    _identity = _load_or_create_identity(identity_path)
    if not peer_id or peer_id in ("peer-auto", ""):
        peer_id = _identity["peer_id"]
    _announce_private_key = _identity["private_key"]
    _announce_public_key_hex = _identity["public_key_hex"]

    hardware_profile = detect_hardware_profile()
    logging.info("peer %s hardware profile: %s", peer_id, hardware_profile.to_dict())

    # ── The Great Pruning: enforce GPU accelerator for production ──────────
    # Allow CPU for sharded mode (layer sharding on nanodes) and toy backend.
    _is_toy_backend = runtime_backend.startswith("toy")
    _is_sharded = bool(expert_layer_indices) or int(total_shards) > 1
    if hardware_profile.accelerator == "cpu" and not _is_toy_backend and not _is_sharded:
        logging.critical(
            "FATAL: no GPU accelerator detected (Metal/CUDA/ROCm required). "
            "This node cannot serve real models on CPU. "
            "Use --toy for development without a GPU, or use --layer-start/--layer-end for sharded CPU inference."
        )
        raise SystemExit(1)
    if hardware_profile.accelerator == "cpu" and _is_sharded:
        logging.info("peer %s: CPU-only sharded mode (layers via --layer-start/--layer-end)", peer_id)

    peer_public_key_hex = ""
    peer_priv_key_obj = None
    if cryptography_available():
        keyfile_path = Path(data_dir) / "peer_identity" / f"{peer_id}.key"
        try:
            peer_identity = load_or_create_identity_keyfile(keyfile_path)
            peer_priv_key_obj = private_key_from_identity(peer_identity)
            peer_public_key_hex = peer_identity.public_key
            logging.info(
                "peer %s loaded identity from %s (pubkey=%s...)",
                peer_id,
                keyfile_path,
                peer_public_key_hex[:16],
            )
        except Exception as exc:
            logging.warning(
                "peer %s: failed to load/create identity keyfile at %s (%s); "
                "falling back to seed-based encryption",
                peer_id,
                keyfile_path,
                exc,
            )

    daemon_mode_enum = DaemonMode(daemon_mode)
    daemon_controller = DaemonController(
        MonitorConfig(
            mode=daemon_mode_enum,
            idle_threshold_sec=max(1, daemon_idle_threshold_sec),
            high_load_threshold=max(0.1, min(1.0, daemon_high_load_threshold)),
            assume_idle_when_unknown=daemon_assume_idle_when_unknown,
        )
    )
    initial_budget = daemon_controller.refresh()

    # Auto mode (6.1): resolve kv_compaction_mode from CLI flag or legacy bool
    _resolved_kv_mode: str = str(kv_compaction_mode or "").strip().lower()
    if _resolved_kv_mode not in {"off", "auto", "on"}:
        # kv_compaction_mode was None or unset — derive from legacy flag
        _resolved_kv_mode = "on" if bool(kv_compaction_enabled) else "off"

    # ── Phase 5: P2P model distribution ───────────────────────────────────────
    # Resolution happens BEFORE PeerService so the resolved local path can be
    # passed as runtime_model_id (HuggingFace from_pretrained already supports
    # local directory paths, so no changes to runtimes are needed).
    _p2p_seeder: ModelSeedServer | None = None
    _p2p_cache: P2PModelCache | None = None
    _actual_seeder_port: int = 0
    _effective_runtime_model_id: str = str(runtime_model_id)

    if p2p_enable:
        _p2p_cache_root = Path(p2p_cache_dir or f"{data_dir}/p2p_cache")
        _manifest_cache_dir = Path(data_dir) / "hf_manifests"
        _manifest_cache_dir.mkdir(parents=True, exist_ok=True)

        _p2p_cache = P2PModelCache(
            cache_root=_p2p_cache_root,
            manifest_cache_dir=_manifest_cache_dir,
            dht_urls=resolved_dht_urls,
        )

        # Try to find the model on a peer before falling back to HF Hub.
        _p2p_resolved = _p2p_cache.resolve(runtime_model_id)
        if _p2p_resolved is not None:
            _effective_runtime_model_id = str(_p2p_resolved)
            logging.info(
                "peer %s p2p_resolved model=%s path=%s",
                peer_id,
                runtime_model_id,
                _effective_runtime_model_id,
            )

        # Start the seeder regardless (serves any models already in cache_root).
        _p2p_seeder = ModelSeedServer(cache_root=_p2p_cache_root, port=seeder_port)
        _actual_seeder_port = _p2p_seeder.start()

    service = PeerService(
        peer_id=peer_id,
        model_id=model_id,
        shard_index=shard_index,
        total_shards=total_shards,
        daemon_mode=daemon_mode,
        broken=broken,
        initial_resource_budget=initial_budget,
        advanced_encryption_enabled=advanced_encryption_enabled,
        advanced_encryption_seed=advanced_encryption_seed,
        kv_cache_max_entries=max(1, int(kv_cache_max_entries)),
        runtime_backend=str(runtime_backend),
        runtime_target=str(runtime_target),
        quantization_mode=str(quantization_mode),
        runtime_model_id=_effective_runtime_model_id,
        hf_model_id=str(hf_model_id or ""),
        mlx_force_hf_tokenizer=bool(mlx_force_hf_tokenizer),
        tokenizer_vocab_guard=bool(tokenizer_vocab_guard),
        tensor_autoencoder_enabled=bool(tensor_autoencoder_enabled),
        tensor_autoencoder_latent_dim=max(1, int(tensor_autoencoder_latent_dim)),
        privacy_noise_variance=max(0.0, float(privacy_noise_variance)),
        geo_challenge_seed=str(geo_challenge_seed),
        expert_tags=tuple(expert_tags),
        expert_layer_indices=tuple(expert_layer_indices),
        expert_router=bool(expert_router),
        peer_public_key=peer_public_key_hex,
        peer_private_key=peer_priv_key_obj,
        kv_compaction_enabled=bool(kv_compaction_enabled),
        kv_compaction_method=str(kv_compaction_method or "hak"),
        kv_compaction_ratio=max(0.01, min(1.0, float(kv_compaction_ratio))),
        kv_compaction_beta=bool(kv_compaction_beta),
        kv_compaction_head_budget_path=str(kv_compaction_head_budget_path or ""),
        kv_compaction_online=bool(kv_compaction_online),
        kv_compaction_online_max_tokens=max(4, int(kv_compaction_online_max_tokens)),
        kv_compaction_mode=_resolved_kv_mode,
        kv_compaction_auto_threshold=max(1, int(kv_compaction_auto_threshold)),
        kv_radix_cache_enabled=bool(kv_radix_cache_enabled),
        kv_radix_cache_max_entries=max(1, int(kv_radix_cache_max_entries)),
        kv_radix_cache_min_prefix_len=max(1, int(kv_radix_cache_min_prefix_len)),
        warmup_on_start=bool(warmup_on_start),
        mlx_eval_timeout_s=max(1.0, float(mlx_eval_timeout_s)),
        batch_window_ms=float(batch_window_ms),
        max_batch_size=max(1, int(max_batch_size)),
        load_full_head=bool(load_full_head),
        pipeline_depth=max(1, int(pipeline_depth or 1)),
        p2p_node=p2p_node,
    )

    # Autonomous rebalancer (Petals parity)
    if rebalance_enabled:
        from peer.autonomous_rebalancer import AutonomousRebalancer
        service._autonomous_rebalancer = AutonomousRebalancer(
            min_improvement=rebalance_min_improvement,
            cooldown_s=rebalance_cooldown_s,
        )
        service._rebalance_check_interval = max(1, rebalance_interval)
        service._announce_count = 0
        logging.info(
            "autonomous_rebalancer: enabled interval=%d min_improvement=%.2f cooldown=%ds",
            rebalance_interval, rebalance_min_improvement, int(rebalance_cooldown_s),
        )

    session_manager: TorrentSessionManager | None = None
    if seed_enable:
        session_manager = _build_torrent_session_manager(
            model_id=model_id,
            seed_cache_dir=seed_cache_dir,
            seed_local_path=seed_local_path,
            seed_source_url=seed_source_url,
            seed_expected_sha256=seed_expected_sha256,
            seed_force_refresh=seed_force_refresh,
            seed_piece_bytes=seed_piece_bytes,
            seed_base_upload_mbps=seed_base_upload_mbps,
            seed_inference_fraction=seed_inference_fraction,
            seed_min_upload_mbps=seed_min_upload_mbps,
            seed_smoothing_alpha=seed_smoothing_alpha,
        )
        logging.info("peer %s seeded genesis artifact at %s", peer_id, session_manager.genesis_result.artifact_path)

    bind_addr = f"{host}:{port}"
    bind_attempt = 0
    lifecycle_restart_attempt = 0
    shutdown_requested = False
    while not shutdown_requested:
        server = None
        while not shutdown_requested:
            candidate = grpc.server(futures.ThreadPoolExecutor(max_workers=16), options=GRPC_SERVER_OPTIONS)
            peer_pb2_grpc.add_PeerServicer_to_server(service, candidate)
            try:
                if tls_enable:
                    if not tls_cert_path or not tls_key_path:
                        raise ValueError("tls-cert-path and tls-key-path are required when tls is enabled")
                    credentials = load_server_credentials(
                        cert_path=tls_cert_path,
                        key_path=tls_key_path,
                        client_ca_path=tls_client_ca_path,
                        require_client_auth=tls_require_client_auth,
                    )
                    bound_port = int(candidate.add_secure_port(bind_addr, credentials))
                    logging.info("peer %s TLS enabled (mTLS=%s)", peer_id, tls_require_client_auth)
                else:
                    bound_port = int(candidate.add_insecure_port(bind_addr))
                    # Also listen on IPv6 so cross-ISP peers with public
                    # v6 addresses can connect directly (bypasses relay).
                    if host in ("0.0.0.0", ""):
                        _v6_addr = f"[::]:{port}"
                        try:
                            candidate.add_insecure_port(_v6_addr)
                            logging.info("peer %s also listening on %s (IPv6)", peer_id, _v6_addr)
                        except Exception as _v6_err:
                            logging.debug("peer %s IPv6 bind skipped: %s", peer_id, _v6_err)
                if bound_port <= 0:
                    raise RuntimeError(f"bind_failed:{bind_addr}")
                candidate.start()
                bind_attempt = 0
                server = candidate
                logging.info("peer %s listening on %s", peer_id, bind_addr)
                break
            except KeyboardInterrupt:
                candidate.stop(grace=0)
                shutdown_requested = True
                break
            except Exception as exc:
                bind_attempt += 1
                delay_s = _exponential_backoff_delay(bind_attempt - 1, base_seconds=1.0, cap_seconds=60.0)
                logging.warning(
                    "peer %s failed to bind/start (%s); retrying in %.1fs",
                    peer_id,
                    exc,
                    delay_s,
                )
                candidate.stop(grace=0)
                time.sleep(delay_s)

        if shutdown_requested or server is None:
            break

        stop_event = threading.Event()
        daemon_thread = threading.Thread(
            target=_daemon_budget_loop,
            kwargs={
                "stop_event": stop_event,
                "service": service,
                "controller": daemon_controller,
                "refresh_interval_sec": max(1, daemon_refresh_interval_sec),
            },
            daemon=True,
        )
        daemon_thread.start()

        seeding_thread: threading.Thread | None = None
        if session_manager is not None:
            seeding_thread = threading.Thread(
                target=_seeding_loop,
                kwargs={
                    "stop_event": stop_event,
                    "service": service,
                    "session_manager": session_manager,
                    "advertised_bandwidth_mbps": max(0.0, bandwidth_mbps),
                    "update_interval_sec": max(1, seed_update_interval_sec),
                },
                daemon=True,
            )
            seeding_thread.start()

        # Local fast-path TCP server (Phase A).
        _fast_path_port = 0
        _fast_path_server = None
        if enable_local_fast_path:
            try:
                from peer.local_fast_path import FastPathServer

                def _fast_path_handler(activation: list[float]) -> list[float]:
                    return list(service.shard.forward(
                        prompt="",
                        activation=activation,
                        max_tokens=1,
                        stage_index=0,
                        total_stages=1,
                    ))

                _fast_path_server = FastPathServer(
                    handler=_fast_path_handler,
                    bind_host=host,
                    port=0,  # OS-assigned ephemeral port.
                )
                _fast_path_server.start()
                _fast_path_port = _fast_path_server.port
                logging.info("local_fast_path: enabled on port %d", _fast_path_port)
            except Exception as exc:
                logging.warning("local_fast_path: failed to start: %s", exc)

        # Hivemind dual-stack DHT adapter (Feature 8).
        _hivemind_adapter = None
        if hivemind_initial_peers:
            try:
                from dht.hivemind_bridge import HivemindDHTAdapter
                _hivemind_adapter = HivemindDHTAdapter(
                    initial_peers=list(hivemind_initial_peers),
                    start=True,
                )
                if _hivemind_adapter.is_alive:
                    logging.info(
                        "hivemind_dht: connected to %d signpost(s)",
                        len(hivemind_initial_peers),
                    )
                else:
                    logging.warning("hivemind_dht: failed to connect — HTTP-only mode")
                    _hivemind_adapter = None
            except Exception as exc:
                logging.warning("hivemind_dht: init failed: %s — HTTP-only mode", exc)

        # ── Phase C: NAT traversal ─────────────────────────────────
        # When a Rust P2P node is provided, AutoNAT + Circuit Relay v2
        # handle NAT detection and relay automatically inside the swarm.
        # Otherwise fall back to the legacy STUN probe + Python relay.
        _relay_channel = None
        _relay_heartbeat_thread: threading.Thread | None = None
        _nat_profile = None
        if p2p_node is not None:
            # Rust P2P path — AutoNAT runs inside the swarm.
            try:
                _nat_info = p2p_node.nat_status()
                service._nat_type = str(_nat_info.get("nat_type", "unknown"))
                service._requires_relay = service._nat_type != "open"
                # AutoNAT may report "open" when the peer is reachable through
                # its relay reservation — the external IP will be the relay's IP,
                # not the peer's real public IP.  Detect this and force relay.
                _ext_ip = str(_nat_info.get("external_ip", ""))
                _relay_ips = {"45.79.190.172", "172.105.69.49", "172.104.164.98"}
                if _ext_ip in _relay_ips:
                    service._nat_type = "relay"
                    service._requires_relay = True
                logging.info(
                    "peer %s p2p_nat: type=%s is_public=%s external_ip=%s requires_relay=%s",
                    peer_id, service._nat_type,
                    _nat_info.get("is_public", False),
                    _ext_ip,
                    service._requires_relay,
                )
                # Circuit Relay v2 is automatic — no manual relay connect needed.
                # The Rust swarm holds an outbound connection to bootstrap relays
                # and accepts inbound circuits through them.
            except Exception as _p2p_nat_exc:
                logging.warning(
                    "peer %s p2p_nat_status_failed: %s — assuming unknown",
                    peer_id, _p2p_nat_exc,
                )
                service._nat_type = "unknown"
                service._requires_relay = True
        else:
            # Legacy path — STUN probe + Python relay.
            try:
                from coordinator.stun_client import probe_nat
                import os as _os
                _force_nat = _os.environ.get("OPENHYDRA_FORCE_NAT", "").strip().lower()
                if _force_nat:
                    from coordinator.stun_client import NatProfile
                    _nat_profile = NatProfile(
                        reachable=_force_nat == "open",
                        nat_type=_force_nat,
                        requires_relay=_force_nat not in ("open", "full_cone"),
                    )
                    logging.info("peer %s nat_forced: type=%s requires_relay=%s",
                                 peer_id, _nat_profile.nat_type, _nat_profile.requires_relay)
                else:
                    _nat_profile = probe_nat()
                    logging.info(
                        "peer %s nat_probe: type=%s requires_relay=%s external=%s:%d",
                        peer_id, _nat_profile.nat_type, _nat_profile.requires_relay,
                        _nat_profile.external_ip, _nat_profile.external_port,
                    )
                service._nat_type = _nat_profile.nat_type
                service._requires_relay = _nat_profile.requires_relay

                if _nat_profile.requires_relay:
                    from coordinator.relay import connect_to_relay
                    from openhydra_defaults import DEFAULT_RELAY_PORT

                    _relay_addrs: list[str] = []
                    _explicit_relay = str(relay_address or "").strip()
                    if _explicit_relay:
                        _relay_addrs = [_explicit_relay]
                    else:
                        _relay_addrs = _derive_relay_addresses(
                            resolved_dht_urls, relay_port=DEFAULT_RELAY_PORT,
                        )

                    for _raddr in _relay_addrs:
                        try:
                            _relay_channel, _relay_peer_id = connect_to_relay(
                                relay_address=_raddr,
                                peer_id=peer_id,
                                grpc_port=port,
                                model_id=model_id,
                            )
                            service._relay_peer_id = _relay_peer_id
                            service._relay_address = _raddr
                            logging.info(
                                "peer %s relay_connected: relay=%s relay_peer=%s",
                                peer_id, _raddr, _relay_peer_id,
                            )
                            break
                        except Exception as _rexc:
                            logging.warning(
                                "peer %s relay_connect_failed: %s err=%s",
                                peer_id, _raddr, _rexc,
                            )

                    if not getattr(service, '_relay_address', ''):
                        logging.error(
                            "peer %s requires relay but all relay candidates failed — "
                            "this peer will be unreachable by remote coordinators",
                            peer_id,
                        )
                    else:
                        _relay_heartbeat_thread = threading.Thread(
                            target=_relay_heartbeat_loop,
                            kwargs={
                                "stop_event": stop_event,
                                "relay_channel": _relay_channel,
                                "peer_id": peer_id,
                                "interval_s": 120.0,
                            },
                            daemon=True,
                        )
                        _relay_heartbeat_thread.start()
            except Exception as _nat_exc:
                logging.warning(
                    "peer %s nat_probe_failed: %s — assuming open (no relay)",
                    peer_id, _nat_exc,
                )
                service._nat_type = "unknown"
                service._requires_relay = False

        announce_thread: threading.Thread | None = None
        if resolved_dht_urls or _hivemind_adapter is not None or p2p_node is not None:
            effective_host = advertise_host or ("127.0.0.1" if host in {"0.0.0.0", "::"} else host)
            # Auto-detect LAN IP when binding to 0.0.0.0 and no explicit advertise_host.
            # Without this, the peer announces 127.0.0.1 which is unreachable from other machines.
            if not advertise_host and effective_host == "127.0.0.1":
                try:
                    import socket as _sock
                    _s = _sock.socket(_sock.AF_INET, _sock.SOCK_DGRAM)
                    _s.connect(("8.8.8.8", 80))
                    _lan_ip = _s.getsockname()[0]
                    _s.close()
                    if _lan_ip and not _lan_ip.startswith("127."):
                        effective_host = _lan_ip
                        logging.info(
                            "peer %s auto-detected LAN IP as advertise_host: %s",
                            peer_id, effective_host,
                        )
                except Exception:
                    pass
            # When behind NAT, use the detected external IP so other
            # peers across the internet can reach us.
            if p2p_node is not None and not advertise_host:
                # Rust P2P path: check AutoNAT-detected external IP.
                try:
                    _p2p_nat = p2p_node.nat_status()
                    _p2p_ext_ip = str(_p2p_nat.get("external_ip", "")).strip()
                    if _p2p_ext_ip and not _p2p_nat.get("is_public", False):
                        effective_host = _p2p_ext_ip
                        logging.info(
                            "peer %s using AutoNAT external IP as advertise_host: %s",
                            peer_id, effective_host,
                        )
                except Exception:
                    pass
            elif (
                _nat_profile is not None
                and _nat_profile.external_ip
                and _nat_profile.nat_type != "open"
                and not advertise_host  # don't override explicit --advertise-host
            ):
                effective_host = _nat_profile.external_ip
                logging.info(
                    "peer %s using STUN external IP as advertise_host: %s (nat_type=%s)",
                    peer_id, effective_host, _nat_profile.nat_type,
                )
            announce_thread = threading.Thread(
                target=_announce_loop,
                kwargs={
                    "stop_event": stop_event,
                    "service": service,
                    "dht_urls": resolved_dht_urls,
                    "advertise_host": effective_host,
                    "port": port,
                    "operator_id": operator_id,
                    "region": region,
                    "bandwidth_mbps": bandwidth_mbps,
                    "announce_interval_sec": announce_interval_sec,
                    "announce_ttl_sec": announce_ttl_sec,
                    "session_manager": session_manager,
                    "announced_reputation_score": max(0.0, float(announced_reputation_score)),
                    "announced_staked_balance": max(0.0, float(announced_staked_balance)),
                    "peer_public_key": service.peer_public_key,
                    "announce_private_key": _announce_private_key,
                    "announce_public_key_hex": _announce_public_key_hex,
                    "seeder_http_port": _actual_seeder_port,
                    "p2p_cache": _p2p_cache,
                    "local_fast_path_port": _fast_path_port,
                    "hivemind_adapter": _hivemind_adapter,
                    "p2p_node": p2p_node,
                    # Phase 3 zero-config: pass capacity snapshot through to
                    # every Announcement emitted by the loop.
                    "capacity_json": str(capacity_json or ""),
                    "capacity_schema_version": int(capacity_schema_version or 0),
                    # Phase 4 zero-config: live snapshot written by the
                    # NegotiationLoop overrides the static kwargs per tick.
                    "capacity_snapshot_ref": capacity_snapshot_ref,
                    "manual_shard": manual_shard,
                },
                daemon=True,
            )
            announce_thread.start()

        # ── Phase 4: continuous re-negotiation thread ──────────────────
        # Start a NegotiationLoop if the caller supplied a factory.  The
        # loop writes fresh capacity_json + current_assignment into
        # ``capacity_snapshot_ref`` every ``interval_s`` seconds; the
        # announce loop above picks those up on its next heartbeat.
        # ``is_busy_fn`` is wired to the live PeerService so we never
        # re-negotiate while a Forward() request is in flight.
        _negotiation_loop_obj = None
        if negotiation_loop_factory is not None and not manual_shard:
            try:
                # Factory signature: legacy single-arg ``(is_busy_fn)``
                # or B3 two-arg ``(is_busy_fn, service)``. Detect
                # arity rather than break existing callers.
                import inspect as _insp
                try:
                    _sig_arity = len(
                        _insp.signature(negotiation_loop_factory).parameters
                    )
                except (TypeError, ValueError):
                    _sig_arity = 1
                _is_busy = lambda: service.inflight_count() > 0
                if _sig_arity >= 2:
                    _negotiation_loop_obj = negotiation_loop_factory(_is_busy, service)
                else:
                    _negotiation_loop_obj = negotiation_loop_factory(_is_busy)
                if _negotiation_loop_obj is not None and hasattr(
                    _negotiation_loop_obj, "start"
                ):
                    _negotiation_loop_obj.start()
            except Exception as _neg_err:
                logging.warning(
                    "negotiation_loop_start_failed: %s — "
                    "falling back to one-shot Phase 3 assignment",
                    _neg_err,
                )
                _negotiation_loop_obj = None

        # Start libp2p proxy handler thread (receives inbound proxy requests
        # from remote peers and forwards to local PeerService.Forward()).
        _proxy_handler_thread: threading.Thread | None = None
        if p2p_node is not None:
            _proxy_handler_thread = threading.Thread(
                target=_proxy_handler_loop,
                kwargs={
                    "stop_event": stop_event,
                    "p2p_node": p2p_node,
                    "service": service,
                },
                daemon=True,
            )
            _proxy_handler_thread.start()

        restart_delay_s = 0.0
        try:
            while True:
                timed_out = bool(server.wait_for_termination(timeout=1.0))
                if timed_out:
                    continue
                raise RuntimeError("grpc_server_terminated")
        except KeyboardInterrupt:
            shutdown_requested = True
        except Exception as exc:
            lifecycle_restart_attempt += 1
            restart_delay_s = _exponential_backoff_delay(
                lifecycle_restart_attempt - 1,
                base_seconds=1.0,
                cap_seconds=120.0,
            )
            logging.warning(
                "peer %s runtime interruption (%s); restarting in %.1fs",
                peer_id,
                exc,
                restart_delay_s,
            )
        finally:
            stop_event.set()
            daemon_thread.join(timeout=2.0)
            if announce_thread is not None:
                announce_thread.join(timeout=2.0)
            if seeding_thread is not None:
                seeding_thread.join(timeout=2.0)
            if _proxy_handler_thread is not None:
                _proxy_handler_thread.join(timeout=2.0)
            if _fast_path_server is not None:
                try:
                    _fast_path_server.stop()
                except Exception:
                    pass
            if _hivemind_adapter is not None:
                try:
                    _hivemind_adapter.shutdown()
                except Exception:
                    pass
            if p2p_node is not None:
                try:
                    p2p_node.stop()
                except Exception:
                    pass
            if _relay_heartbeat_thread is not None:
                _relay_heartbeat_thread.join(timeout=2.0)
            if _relay_channel is not None:
                try:
                    _relay_channel.close()
                except Exception:
                    pass
            shutdown_event = server.stop(grace=2)
            try:
                shutdown_event.wait(timeout=3.0)
            except Exception:
                pass

        if shutdown_requested:
            break

        if restart_delay_s > 0.0:
            time.sleep(restart_delay_s)


def main() -> None:
    def _parse_csv_tags(value: str) -> tuple[str, ...]:
        raw = [item.strip().lower() for item in str(value).split(",")]
        out: list[str] = []
        seen: set[str] = set()
        for item in raw:
            if not item or item in seen:
                continue
            seen.add(item)
            out.append(item)
        return tuple(out)

    def _parse_csv_ints(value: str) -> tuple[int, ...]:
        out: list[int] = []
        seen: set[int] = set()
        for token in [item.strip() for item in str(value).split(",")]:
            if not token:
                continue
            try:
                idx = int(token)
            except ValueError:
                continue
            if idx < 0 or idx in seen:
                continue
            seen.add(idx)
            out.append(idx)
        return tuple(sorted(out))

    def _parse_dht_urls(raw_values: list[str] | None) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in list(raw_values or []):
            for token in str(raw).split(","):
                value = token.strip()
                if not value or value in seen:
                    continue
                seen.add(value)
                out.append(value)
        return out

    parser = argparse.ArgumentParser(description="OpenHydra Tier 1/2 peer server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--deployment-profile", choices=["dev", "prod"], default="dev")
    parser.add_argument("--secrets-file", default=None, help="Path to KEY=VALUE secrets file (0600 permissions required)")
    parser.add_argument("--peer-id", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B",
                        help="Model ID announced to the DHT (must match --runtime-model-id)")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--total-shards", type=int, default=1)
    parser.add_argument("--daemon-mode", default="polite", choices=["polite", "power_user", "dedicated"])
    parser.add_argument("--broken", action="store_true", help="Enable deterministic bad outputs for testing")
    parser.add_argument(
        "--dht-url",
        action="append",
        default=None,
        help=(
            "DHT bootstrap URL(s) — repeat the flag or use comma-separated values. "
            "Defaults to the three production OpenHydra bootstrap nodes. "
            "Passing even one --dht-url replaces the entire default list, "
            "which lets operators run isolated private networks."
        ),
    )
    parser.add_argument("--advertise-host", default=None, help="Host/IP peers should use to reach this node")
    parser.add_argument("--operator-id", default=None)
    parser.add_argument("--region", default=None, help="Optional region tag for DHT geo-aware lookup")
    parser.add_argument("--bandwidth-mbps", type=float, default=0.0)
    parser.add_argument("--announce-interval-sec", type=int, default=60)
    parser.add_argument("--announce-ttl-sec", type=int, default=300)
    parser.add_argument("--tls-enable", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tls-cert-path", default=None)
    parser.add_argument("--tls-key-path", default=None)
    parser.add_argument("--tls-client-ca-path", default=None)
    parser.add_argument("--tls-require-client-auth", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--seed-enable", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed-cache-dir", default=".cache/openhydra")
    parser.add_argument("--seed-local-path", default=None)
    parser.add_argument("--seed-source-url", default=None)
    parser.add_argument("--seed-expected-sha256", default=None)
    parser.add_argument("--seed-force-refresh", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed-piece-bytes", type=int, default=1 * 1024 * 1024)
    parser.add_argument("--seed-base-upload-mbps", type=float, default=100.0)
    parser.add_argument("--seed-inference-fraction", type=float, default=0.10)
    parser.add_argument("--seed-min-upload-mbps", type=float, default=1.0)
    parser.add_argument("--seed-smoothing-alpha", type=float, default=0.35)
    parser.add_argument("--seed-update-interval-sec", type=int, default=1)
    parser.add_argument("--daemon-idle-threshold-sec", type=int, default=5 * 60)
    parser.add_argument("--daemon-refresh-interval-sec", type=int, default=5)
    parser.add_argument("--daemon-high-load-threshold", type=float, default=0.85)
    parser.add_argument("--daemon-assume-idle-when-unknown", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--advanced-encryption-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--advanced-encryption-seed", default="openhydra-tier3-dev-seed")
    parser.add_argument("--kv-cache-max-entries", type=int, default=1024)

    # ── KV cache compaction (Phases 1-4) ─────────────────────────────────────
    parser.add_argument(
        "--kv-compaction-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable KV cache compaction via Attention Matching (arXiv:2602.16284).",
    )
    parser.add_argument(
        "--kv-compaction-method",
        choices=["hak", "omp"],
        default="hak",
        help="Key-selection algorithm: 'hak' (Highest Attention Keys, fast) or "
             "'omp' (Orthogonal Matching Pursuit, more accurate). Default: hak.",
    )
    parser.add_argument(
        "--kv-compaction-ratio",
        type=float,
        default=0.10,
        help="Fraction of KV tokens to keep after compaction (default: 0.10 = 10%%). "
             "Overridden per-head when --kv-compaction-head-budget-path is set.",
    )
    parser.add_argument(
        "--kv-compaction-beta",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Phase 2: fit scalar log-space bias corrections (β) and refit compact "
             "values (Cv) via least-squares.  Requires scipy.",
    )
    parser.add_argument(
        "--kv-compaction-head-budget-path",
        default="",
        metavar="PATH",
        help="Phase 3: path to a JSON file with per-layer / per-kv-head token "
             "budget ratios.  Pre-built files for Qwen3-4B and Llama-3.1-8B are "
             "in peer/kv_compaction/head_budgets/.",
    )
    parser.add_argument(
        "--kv-compaction-online",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Phase 4: compact mid-trajectory (at every cache-write) when the "
             "stored sequence length exceeds --kv-compaction-online-max-tokens.",
    )
    parser.add_argument(
        "--kv-compaction-online-max-tokens",
        type=int,
        default=512,
        help="Phase 4: physical KV cache size limit for online compaction (default: 512).",
    )
    parser.add_argument(
        "--kv-compaction-mode",
        choices=["off", "auto", "on"],
        default=None,
        help="Three-position KV compaction toggle: 'off' (disabled), 'auto' (compact only "
             "when stored sequence exceeds --kv-compaction-auto-threshold tokens), "
             "'on' (always compact). Overrides --kv-compaction-enabled.",
    )
    parser.add_argument(
        "--kv-compaction-auto-threshold",
        type=int,
        default=512,
        help="Auto mode: minimum stored sequence length (tokens) before compaction "
             "activates (default: 512). Only used with --kv-compaction-mode auto.",
    )

    # ── Radix prefix cache (Phase H) ──────────────────────────────────────────
    parser.add_argument(
        "--kv-radix-cache-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Phase H: enable radix (longest-prefix) KV cache for cross-session prefix reuse.",
    )
    parser.add_argument(
        "--kv-radix-cache-max-entries",
        type=int,
        default=128,
        help="Phase H: maximum number of token sequences stored in the radix cache (default: 128).",
    )
    parser.add_argument(
        "--kv-radix-cache-min-prefix-len",
        type=int,
        default=16,
        help="Phase H: minimum prefix length (tokens) required for radix cache store/retrieve (default: 16).",
    )

    parser.add_argument(
        "--runtime-backend",
        choices=["toy_auto", "toy_cpu", "toy_gpu_sim", "pytorch_auto", "pytorch_cpu", "pytorch_cuda", "mlx"],
        default="pytorch_auto",
        help="Model runtime backend. Defaults to pytorch_auto (requires torch + transformers). "
             "Use mlx for high-throughput inference on Apple Silicon (requires mlx, mlx-lm). "
             "Use --toy for a lightweight fake-token backend during development.",
    )
    parser.add_argument(
        "--toy",
        action="store_true",
        default=False,
        help="Use the lightweight ToyRuntime (fake tokens) instead of a real model. "
             "Equivalent to --runtime-backend toy_auto. For development and testing only.",
    )
    parser.add_argument(
        "--warmup-on-start",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run a single dummy forward pass on startup to JIT-compile GPU kernels. "
            "On Apple MPS this moves ~30 s of Metal shader compilation from the first "
            "request (TTFT) to peer startup where it is expected.  Recommended for "
            "production deployments using pytorch_auto or pytorch_cuda."
        ),
    )
    parser.add_argument(
        "--mlx-eval-timeout",
        type=float,
        default=120.0,
        help=(
            "Timeout in seconds for individual MLX Metal GPU operations.  If an "
            "mx.eval() call does not complete within this deadline the watchdog "
            "raises TimeoutError and marks the runtime unhealthy.  Only relevant "
            "when runtime_backend='mlx' (default: 120).  Raised from 30 to "
            "accommodate 8 GB machines under memory pressure."
        ),
    )
    parser.add_argument(
        "--batch-window-ms",
        type=float,
        default=50.0,
        help=(
            "Milliseconds to wait for additional requests before flushing a batch. "
            "Concurrent requests that arrive within this window are coalesced into a "
            "single GPU forward pass (default: 50)."
        ),
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=8,
        help=(
            "Maximum number of requests per batch.  When this limit is reached the "
            "batch flushes immediately without waiting for the window to expire. "
            "Guards against OOM on small-VRAM nodes (default: 8)."
        ),
    )
    # ── Phase 5: P2P model distribution ───────────────────────────────────────
    parser.add_argument(
        "--p2p-enable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable P2P model distribution.  When active, the peer will try to "
            "download model weights from other peers before falling back to "
            "HuggingFace Hub, and will expose its own cache for other peers to "
            "download via HTTP Range requests."
        ),
    )
    parser.add_argument(
        "--seeder-port",
        type=int,
        default=0,
        help=(
            "HTTP port for the P2P model seeder (default: 0 = OS-assigned). "
            "Only used when --p2p-enable is set."
        ),
    )
    parser.add_argument(
        "--p2p-cache-dir",
        default=None,
        metavar="PATH",
        help=(
            "Directory for P2P-downloaded model cache "
            "(default: {--data-dir}/p2p_cache). "
            "Only used when --p2p-enable is set."
        ),
    )

    parser.add_argument(
        "--enable-local-fast-path",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable raw TCP fast-path for same-LAN tensor transfer (Phase A). "
            "When enabled, the peer binds an additional TCP socket and advertises "
            "it in the DHT so LAN neighbours can bypass gRPC for activation transfer."
        ),
    )
    parser.add_argument(
        "--hivemind-initial-peers",
        nargs="*",
        default=_DEFAULT_HIVEMIND_SIGNPOSTS,
        metavar="MADDR",
        help=(
            "Hivemind multiaddr(s) of bootstrap signpost nodes for dual-stack "
            "DHT. Defaults to the 3 production signpost nodes (EU/US/AP). "
            "Pass --hivemind-initial-peers (empty) to disable Hivemind DHT."
        ),
    )

    parser.add_argument("--runtime-target", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--relay-address", default="",
                        help="Explicit relay address (host:port) for NAT traversal. "
                             "If not set, auto-derived from DHT bootstrap URLs when NAT probe requires relay.")
    parser.add_argument("--quantization", choices=["none", "8bit", "4bit"], default="none")
    parser.add_argument(
        "--quantization-mode",
        choices=["fp32", "int8", "int4"],
        default=None,
        help="Legacy alias for --quantization; kept for backward compatibility",
    )
    parser.add_argument(
        "--runtime-model-id",
        default="Qwen/Qwen3.5-0.8B",
        help="HuggingFace model id for pytorch_* runtime backends",
    )
    parser.add_argument("--tensor-autoencoder-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tensor-autoencoder-latent-dim", type=int, default=1024)
    parser.add_argument("--privacy-noise-variance", type=float, default=0.0)
    parser.add_argument("--announced-reputation-score", type=float, default=0.0)
    parser.add_argument("--announced-staked-balance", type=float, default=0.0)
    parser.add_argument("--geo-challenge-seed", default="openhydra-geo-dev-seed")
    parser.add_argument("--expert-tags", default="", help="Comma-separated expert tags, e.g. coding,math,legal")
    parser.add_argument("--expert-layer-indices", default="", help="Comma-separated layer indices for expert placement")
    parser.add_argument("--expert-router", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--data-dir",
        default=".openhydra",
        help="Directory for persistent peer state (identity keyfiles, ledger, etc.)",
    )
    parser.add_argument(
        "--identity-path",
        default=".openhydra/identity.key",
        help="Path to Ed25519 identity keypair file (created on first run, mode 0600).",
    )

    args = parser.parse_args()
    # --toy is a convenience alias for --runtime-backend toy_auto
    if args.toy:
        args.runtime_backend = "toy_auto"
    # Fall back to the production bootstrap nodes when the user passes no
    # --dht-url flags.  Passing even one flag opts out of the defaults
    # entirely so private/test networks work without surprise extra URLs.
    dht_urls = _parse_dht_urls(args.dht_url) or list(PRODUCTION_BOOTSTRAP_URLS)
    deployment_settings = _resolve_deployment_security_settings(parser, args)
    resolved_quantization_mode = _resolve_quantization_mode(args.quantization, args.quantization_mode)

    configure_logging(json_logs=str(deployment_settings.get("deployment_profile", "dev")) == "prod")
    serve(
        host=args.host,
        port=args.port,
        peer_id=args.peer_id,
        model_id=args.model_id,
        shard_index=args.shard_index,
        total_shards=args.total_shards,
        daemon_mode=args.daemon_mode,
        broken=args.broken,
        dht_urls=dht_urls,
        dht_url=(dht_urls[0] if dht_urls else None),
        advertise_host=args.advertise_host,
        operator_id=args.operator_id,
        region=args.region,
        bandwidth_mbps=max(0.0, args.bandwidth_mbps),
        announce_interval_sec=max(5, args.announce_interval_sec),
        announce_ttl_sec=max(10, args.announce_ttl_sec),
        tls_enable=args.tls_enable,
        tls_cert_path=args.tls_cert_path,
        tls_key_path=args.tls_key_path,
        tls_client_ca_path=args.tls_client_ca_path,
        tls_require_client_auth=args.tls_require_client_auth,
        seed_enable=args.seed_enable,
        seed_cache_dir=args.seed_cache_dir,
        seed_local_path=args.seed_local_path,
        seed_source_url=args.seed_source_url,
        seed_expected_sha256=args.seed_expected_sha256,
        seed_force_refresh=args.seed_force_refresh,
        seed_piece_bytes=max(1, args.seed_piece_bytes),
        seed_base_upload_mbps=max(1.0, args.seed_base_upload_mbps),
        seed_inference_fraction=max(0.01, min(1.0, args.seed_inference_fraction)),
        seed_min_upload_mbps=max(0.1, args.seed_min_upload_mbps),
        seed_smoothing_alpha=max(0.01, min(1.0, args.seed_smoothing_alpha)),
        seed_update_interval_sec=max(1, args.seed_update_interval_sec),
        daemon_idle_threshold_sec=max(1, args.daemon_idle_threshold_sec),
        daemon_refresh_interval_sec=max(1, args.daemon_refresh_interval_sec),
        daemon_high_load_threshold=max(0.1, min(1.0, args.daemon_high_load_threshold)),
        daemon_assume_idle_when_unknown=args.daemon_assume_idle_when_unknown,
        advanced_encryption_enabled=args.advanced_encryption_enabled,
        advanced_encryption_seed=str(deployment_settings["advanced_encryption_seed"]),
        kv_cache_max_entries=max(1, args.kv_cache_max_entries),
        runtime_backend=str(args.runtime_backend),
        runtime_target=str(args.runtime_target),
        quantization_mode=resolved_quantization_mode,
        runtime_model_id=str(args.runtime_model_id),
        tensor_autoencoder_enabled=bool(args.tensor_autoencoder_enabled),
        tensor_autoencoder_latent_dim=max(1, int(args.tensor_autoencoder_latent_dim)),
        privacy_noise_variance=max(0.0, float(args.privacy_noise_variance)),
        geo_challenge_seed=str(deployment_settings["geo_challenge_seed"]),
        announced_reputation_score=max(0.0, float(args.announced_reputation_score)),
        announced_staked_balance=max(0.0, float(args.announced_staked_balance)),
        expert_tags=_parse_csv_tags(args.expert_tags),
        expert_layer_indices=_parse_csv_ints(args.expert_layer_indices),
        expert_router=bool(args.expert_router),
        data_dir=str(args.data_dir),
        identity_path=str(args.identity_path),
        kv_compaction_enabled=bool(args.kv_compaction_enabled),
        kv_compaction_method=str(args.kv_compaction_method),
        kv_compaction_ratio=max(0.01, min(1.0, float(args.kv_compaction_ratio))),
        kv_compaction_beta=bool(args.kv_compaction_beta),
        kv_compaction_head_budget_path=str(args.kv_compaction_head_budget_path or ""),
        kv_compaction_online=bool(args.kv_compaction_online),
        kv_compaction_online_max_tokens=max(4, int(args.kv_compaction_online_max_tokens)),
        kv_compaction_mode=args.kv_compaction_mode,  # None | "off" | "auto" | "on"
        kv_compaction_auto_threshold=max(1, int(args.kv_compaction_auto_threshold)),
        kv_radix_cache_enabled=bool(args.kv_radix_cache_enabled),
        kv_radix_cache_max_entries=max(1, int(args.kv_radix_cache_max_entries)),
        kv_radix_cache_min_prefix_len=max(1, int(args.kv_radix_cache_min_prefix_len)),
        warmup_on_start=bool(args.warmup_on_start),
        mlx_eval_timeout_s=float(args.mlx_eval_timeout),
        batch_window_ms=float(args.batch_window_ms),
        max_batch_size=max(1, int(args.max_batch_size)),
        load_full_head=bool(getattr(args, "sample_on_coordinator", False)),
        p2p_enable=bool(args.p2p_enable),
        seeder_port=max(0, int(args.seeder_port)),
        p2p_cache_dir=args.p2p_cache_dir or None,
        enable_local_fast_path=bool(args.enable_local_fast_path),
        hivemind_initial_peers=list(args.hivemind_initial_peers) if args.hivemind_initial_peers else None,
        relay_address=str(getattr(args, "relay_address", "") or ""),
    )


if __name__ == "__main__":
    main()
