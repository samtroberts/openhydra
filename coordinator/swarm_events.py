# Copyright 2026 OpenHydra contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Phase 2b — swarm event envelopes for Topology B + failover.

Three event types ride the existing libp2p ``publish_event`` channel
(same JSON-envelope pattern the B1 rendezvous and hole-punch publish
already use, see ``peer/server.py`` ``b1_rendezvous_published`` log
line). No new RPC surface, no new DHT keys — just a stable envelope
schema each peer can recognise.

| Event type            | Direction          | Trigger |
|---|---|---|
| ``verify_result``     | coord → stage-0   | After ``HeadSampler.verify_block`` returns under Topology B. Carries (accepted_len, bonus_token, kv_rollback_to) so stage-0 starts drafting block N+1 immediately. |
| ``register_draft_model`` | any → swarm    | Coord (Topology A) or stage-0 (Topology B) on startup. Durable in the swarm registry so a stage-0 promotion knows what draft to load. |
| ``promote_drafter``   | stage-0 → swarm   | After stage-0 detects coord absence and promotes itself. Carries (from_peer_id, to_peer_id) so other peers re-target their PushResult callbacks. |

Each event is a JSON dict on the wire:

    {
      "type":      "verify_result" | "register_draft_model" | "promote_drafter",
      "v":         1,                            # schema version
      "data":      <event-specific dict>,
      "from_peer": <emitter libp2p_id>,
      "unix_ms":   <emit timestamp>,
    }

Schema-version bump is the upgrade path: a peer that sees ``v > 1``
without a matching parser drops the message and logs a structured
warning, keeping forward-compat across phased rollouts.

The publish/subscribe helpers are designed to work with EITHER:
  * a real ``P2PNode.publish_event`` / ``poll_events`` pair (production)
  * a synchronous in-memory bus (tests + single-process deployments)

The ``SwarmEventBus`` ABC abstracts the transport so the rest of the
coord/peer code never directly touches libp2p.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
import json
import logging
import threading
import time
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "EVENT_TYPE_VERIFY_RESULT",
    "EVENT_TYPE_REGISTER_DRAFT_MODEL",
    "EVENT_TYPE_PROMOTE_DRAFTER",
    "SCHEMA_VERSION",
    "VerifyResult",
    "RegisterDraftModel",
    "PromoteDrafter",
    "SwarmEvent",
    "SwarmEventBus",
    "InMemorySwarmEventBus",
    "LibP2PSwarmEventBus",
    "encode_event",
    "decode_event",
    "EventDecodeError",
]


# ── Constants ───────────────────────────────────────────────────────────


EVENT_TYPE_VERIFY_RESULT = "verify_result"
EVENT_TYPE_REGISTER_DRAFT_MODEL = "register_draft_model"
EVENT_TYPE_PROMOTE_DRAFTER = "promote_drafter"

SCHEMA_VERSION = 1

_KNOWN_EVENT_TYPES = frozenset({
    EVENT_TYPE_VERIFY_RESULT,
    EVENT_TYPE_REGISTER_DRAFT_MODEL,
    EVENT_TYPE_PROMOTE_DRAFTER,
})


class EventDecodeError(ValueError):
    """Raised when a wire envelope cannot be decoded.

    Carries a ``reason`` taxonomy so receivers can structure their
    diagnostic logs:
      * ``"malformed_json"``  — bytes are not parseable JSON
      * ``"missing_field"``   — required envelope key is absent
      * ``"unknown_type"``    — ``type`` is not a recognised event
      * ``"future_schema"``   — ``v`` is newer than this peer supports
      * ``"data_invalid"``    — ``data`` payload fails type-specific validation
    """

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


# ── Event payload dataclasses ──────────────────────────────────────────


@dataclass(frozen=True)
class VerifyResult:
    """Coord → stage-0 — block-verify outcome under Topology B.

    Stage-0 hosts the drafter under Topology B; after every verify
    pass the coord broadcasts this so stage-0 can start drafting
    block N+1 without waiting for the next ForwardRequest to arrive.

    Fields:
        session_id:     KV session identifier (matches
                        ForwardRequest.kv_session_id used during the
                        verify ring trip).
        accepted_len:   Range [0, block_size]. Drafts 0..accepted_len-1
                        were accepted by ``HeadSampler.verify_block``.
        bonus_token:    Sampled at position ``accepted_len`` — always
                        committed regardless of acceptance pattern.
        kv_rollback_to: Absolute sequence position the next forward
                        should roll the KV cache back to before
                        processing draft block N+1. Equals
                        prefix_len_at_block_start + accepted_len + 1.
        block_index:    Monotonic block counter; lets stage-0 dedupe
                        late-arriving duplicates against its own
                        local counter.
    """

    session_id: str
    accepted_len: int
    bonus_token: int
    kv_rollback_to: int
    block_index: int

    def validate(self) -> None:
        if not self.session_id:
            raise EventDecodeError(
                "data_invalid", "VerifyResult.session_id must be non-empty",
            )
        if self.accepted_len < 0:
            raise EventDecodeError(
                "data_invalid",
                f"VerifyResult.accepted_len must be >= 0; "
                f"got {self.accepted_len}",
            )
        if self.kv_rollback_to < 0:
            raise EventDecodeError(
                "data_invalid",
                f"VerifyResult.kv_rollback_to must be >= 0; "
                f"got {self.kv_rollback_to}",
            )
        if self.block_index < 0:
            raise EventDecodeError(
                "data_invalid",
                f"VerifyResult.block_index must be >= 0; "
                f"got {self.block_index}",
            )


@dataclass(frozen=True)
class RegisterDraftModel:
    """Any → swarm — durable draft-model spec.

    Sticky in the registry so a peer that joins later can replay this
    event and learn the swarm's draft configuration, AND so that on a
    coord crash under Topology A the failover stage-0 promotion knows
    exactly what draft weights to load.

    Fields:
        target_path:  HF path or local dir of the target model.
        draft_path:   HF path or local dir of the DFlash draft.
        block_size:   Tokens per draft block (1..32).
        backend:      ``"mlx"`` | ``"pytorch"`` — the drafter backend.
    """

    target_path: str
    draft_path: str
    block_size: int
    backend: str

    def validate(self) -> None:
        if not self.target_path:
            raise EventDecodeError("data_invalid", "target_path must be non-empty")
        if not self.draft_path:
            raise EventDecodeError("data_invalid", "draft_path must be non-empty")
        if not 1 <= self.block_size <= 32:
            raise EventDecodeError(
                "data_invalid",
                f"block_size must be in [1, 32]; got {self.block_size}",
            )
        if self.backend not in {"mlx", "pytorch"}:
            raise EventDecodeError(
                "data_invalid",
                f"backend must be 'mlx' or 'pytorch'; got {self.backend!r}",
            )


@dataclass(frozen=True)
class PromoteDrafter:
    """stage-0 → swarm — failover promotion announcement.

    Emitted when stage-0 detects coord absence and takes over.
    Other peers update their PushResult callback target to
    ``to_peer_id``.

    Fields:
        from_peer_id: libp2p peer id of the previous (now-absent)
                      drafter / coord.
        to_peer_id:   libp2p peer id of the promoting peer.
        unix_ms:      Wall-clock timestamp of the promotion event;
                      used to break ties if two peers race to
                      promote (later timestamp wins).
    """

    from_peer_id: str
    to_peer_id: str
    unix_ms: int

    def validate(self) -> None:
        if not self.from_peer_id:
            raise EventDecodeError("data_invalid", "from_peer_id must be non-empty")
        if not self.to_peer_id:
            raise EventDecodeError("data_invalid", "to_peer_id must be non-empty")
        if self.unix_ms <= 0:
            raise EventDecodeError(
                "data_invalid",
                f"unix_ms must be > 0; got {self.unix_ms}",
            )


# ── Wrapped envelope ────────────────────────────────────────────────────


@dataclass(frozen=True)
class SwarmEvent:
    """Decoded envelope. Carries the typed payload plus metadata
    used for routing + dedup."""

    type: str
    payload: Any   # one of VerifyResult | RegisterDraftModel | PromoteDrafter
    from_peer: str
    unix_ms: int


# ── Encode / decode ─────────────────────────────────────────────────────


def encode_event(
    payload: Any,
    *,
    from_peer: str,
    unix_ms: Optional[int] = None,
) -> bytes:
    """Wrap a typed payload in the envelope and serialise to bytes.

    The payload's class determines ``type``. Validation is run before
    encode so a malformed payload produces a clear local error
    rather than a remote-decode failure.
    """
    if isinstance(payload, VerifyResult):
        type_str = EVENT_TYPE_VERIFY_RESULT
    elif isinstance(payload, RegisterDraftModel):
        type_str = EVENT_TYPE_REGISTER_DRAFT_MODEL
    elif isinstance(payload, PromoteDrafter):
        type_str = EVENT_TYPE_PROMOTE_DRAFTER
    else:
        raise TypeError(
            f"encode_event: unsupported payload type "
            f"{type(payload).__name__}; expected VerifyResult / "
            f"RegisterDraftModel / PromoteDrafter"
        )

    payload.validate()

    envelope = {
        "type": type_str,
        "v": SCHEMA_VERSION,
        "data": asdict(payload),
        "from_peer": str(from_peer or ""),
        "unix_ms": int(unix_ms if unix_ms is not None else time.time() * 1000),
    }
    return json.dumps(envelope, separators=(",", ":")).encode("utf-8")


def decode_event(raw: bytes | str | dict) -> SwarmEvent:
    """Parse a wire envelope. Returns ``SwarmEvent`` or raises
    ``EventDecodeError`` with a structured ``reason``.

    Accepts ``bytes`` (the libp2p wire form), ``str`` (JSON text),
    or ``dict`` (already-parsed) so callers can route both transport
    paths through the same validator.
    """
    if isinstance(raw, (bytes, bytearray)):
        try:
            envelope = json.loads(bytes(raw).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EventDecodeError("malformed_json", str(exc)) from exc
    elif isinstance(raw, str):
        try:
            envelope = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise EventDecodeError("malformed_json", str(exc)) from exc
    elif isinstance(raw, dict):
        envelope = raw
    else:
        raise EventDecodeError(
            "malformed_json",
            f"decode_event: cannot decode {type(raw).__name__}",
        )

    for key in ("type", "v", "data"):
        if key not in envelope:
            raise EventDecodeError(
                "missing_field",
                f"swarm event envelope missing key {key!r}",
            )

    type_str = str(envelope["type"])
    if type_str not in _KNOWN_EVENT_TYPES:
        raise EventDecodeError(
            "unknown_type",
            f"swarm event type {type_str!r} not recognised",
        )

    schema_v = int(envelope["v"])
    if schema_v > SCHEMA_VERSION:
        raise EventDecodeError(
            "future_schema",
            f"swarm event uses schema v={schema_v}; this peer "
            f"only supports v={SCHEMA_VERSION}",
        )

    data = envelope["data"]
    if not isinstance(data, dict):
        raise EventDecodeError(
            "data_invalid",
            f"swarm event 'data' must be a dict; got "
            f"{type(data).__name__}",
        )

    try:
        if type_str == EVENT_TYPE_VERIFY_RESULT:
            payload: Any = VerifyResult(**data)
        elif type_str == EVENT_TYPE_REGISTER_DRAFT_MODEL:
            payload = RegisterDraftModel(**data)
        elif type_str == EVENT_TYPE_PROMOTE_DRAFTER:
            payload = PromoteDrafter(**data)
        else:  # pragma: no cover — guarded by _KNOWN_EVENT_TYPES check
            raise EventDecodeError("unknown_type", type_str)
    except TypeError as exc:
        raise EventDecodeError("data_invalid", str(exc)) from exc

    payload.validate()

    return SwarmEvent(
        type=type_str,
        payload=payload,
        from_peer=str(envelope.get("from_peer", "")),
        unix_ms=int(envelope.get("unix_ms", 0) or 0),
    )


# ── Bus abstraction ─────────────────────────────────────────────────────


class SwarmEventBus(ABC):
    """ABC for swarm event transport.

    Two concrete implementations:
      * ``InMemorySwarmEventBus`` — synchronous, single-process,
        deterministic. Used for tests + co-located coord+peer setups
        where libp2p isn't needed.
      * (production, lives in ``peer/server.py``) — wraps
        ``P2PNode.publish_event`` + ``poll_events``.
    """

    @abstractmethod
    def publish(self, payload: Any, *, from_peer: str = "") -> None:
        """Encode + emit ``payload`` to all subscribers."""

    @abstractmethod
    def subscribe(
        self,
        event_type: str,
        handler: Callable[[SwarmEvent], None],
    ) -> Callable[[], None]:
        """Register ``handler`` for events of ``event_type``.

        Returns an ``unsubscribe`` callable; idempotent.
        """


class InMemorySwarmEventBus(SwarmEventBus):
    """Synchronous in-memory bus.

    Publish runs every subscribed handler inline before returning.
    Subscriber list is mutex-guarded so concurrent publish/subscribe
    don't race. Errors raised by handlers are logged and isolated:
    one handler crashing does not affect other handlers' delivery
    of the same event.
    """

    def __init__(self) -> None:
        self._subs: dict[str, list[Callable[[SwarmEvent], None]]] = {}
        self._lock = threading.Lock()

    def publish(self, payload: Any, *, from_peer: str = "") -> None:
        wire = encode_event(payload, from_peer=from_peer)
        event = decode_event(wire)
        # Snapshot the subscriber list so a handler that subscribes
        # mid-delivery doesn't see this event.
        with self._lock:
            handlers = list(self._subs.get(event.type, ()))
        for handler in handlers:
            try:
                handler(event)
            except Exception:  # pragma: no cover — logged, never raised
                logger.exception(
                    "swarm_event_handler_failed: type=%s from_peer=%s",
                    event.type, event.from_peer,
                )

    def subscribe(
        self,
        event_type: str,
        handler: Callable[[SwarmEvent], None],
    ) -> Callable[[], None]:
        if event_type not in _KNOWN_EVENT_TYPES:
            raise ValueError(
                f"subscribe: unknown event_type {event_type!r}; "
                f"expected one of {sorted(_KNOWN_EVENT_TYPES)}"
            )
        with self._lock:
            self._subs.setdefault(event_type, []).append(handler)

        def _unsubscribe() -> None:
            with self._lock:
                lst = self._subs.get(event_type)
                if lst is None:
                    return
                try:
                    lst.remove(handler)
                except ValueError:
                    pass
                if not lst:
                    self._subs.pop(event_type, None)

        return _unsubscribe


# ── libp2p adapter ──────────────────────────────────────────────────────


class LibP2PSwarmEventBus(SwarmEventBus):
    """Production adapter — rides the existing ``GossipClient``
    libp2p pub/sub channel.

    The codebase already runs one ``GossipClient`` per peer (see
    ``peer/gossip_client.py``) for PEER_DEAD / REQUEST_HOLE_PUNCH
    coordination. Phase 2b's swarm events join the same channel
    rather than spinning up a parallel poll loop. Type-string
    prefixes (``verify_result``, ``register_draft_model``,
    ``promote_drafter``) don't collide with the existing types.

    The GossipClient envelope wraps our payload's ``data`` dict;
    we put the schema version inside ``data`` so versioned
    upgrades stay backwards-compat with peers running the older
    GossipClient.

    Args:
        gossip_client: A started GossipClient instance.
        local_peer_id: This peer's libp2p id. Used for outgoing
            ``from_peer`` field on published events.
    """

    def __init__(self, gossip_client: Any, *, local_peer_id: str):
        if gossip_client is None:
            raise ValueError("LibP2PSwarmEventBus: gossip_client is required")
        if not local_peer_id:
            raise ValueError("LibP2PSwarmEventBus: local_peer_id is required")
        self._gc = gossip_client
        self._local = str(local_peer_id)
        # Track active subscriptions so close() can cleanly detach.
        self._handlers: dict[str, list[tuple[Callable, Callable]]] = {}
        self._lock = threading.Lock()

    # ── Publish ────────────────────────────────────────────────────

    def publish(self, payload: Any, *, from_peer: str = "") -> None:
        """Encode + publish via the wrapped GossipClient.

        The ``from_peer`` argument is honoured if provided (for
        operator-driven publishes that want to attribute to a
        different peer); otherwise we stamp ``self._local``.

        We use our envelope encoder to populate the GossipClient's
        ``data`` field — that puts the schema version + validated
        payload inside the gossip data dict so receivers running
        the same Phase 2b version decode the wire bytes through
        ``decode_event``.
        """
        if isinstance(payload, VerifyResult):
            type_str = EVENT_TYPE_VERIFY_RESULT
        elif isinstance(payload, RegisterDraftModel):
            type_str = EVENT_TYPE_REGISTER_DRAFT_MODEL
        elif isinstance(payload, PromoteDrafter):
            type_str = EVENT_TYPE_PROMOTE_DRAFTER
        else:
            raise TypeError(
                f"LibP2PSwarmEventBus.publish: unsupported payload "
                f"{type(payload).__name__}"
            )

        payload.validate()

        # GossipClient.publish takes (event_type, data_dict). We embed
        # the schema version + dataclass fields. Receivers reconstruct
        # the typed payload via decode_event on the dict.
        from dataclasses import asdict
        data = {"v": SCHEMA_VERSION, **asdict(payload)}

        from_peer_id = str(from_peer or self._local)
        ok = self._gc.publish(type_str, data)
        if not ok:
            logger.debug(
                "libp2p_swarm_event_publish_rejected: type=%s from_peer=%s",
                type_str, from_peer_id,
            )

    # ── Subscribe ─────────────────────────────────────────────────

    def subscribe(
        self,
        event_type: str,
        handler: Callable[[SwarmEvent], None],
    ) -> Callable[[], None]:
        if event_type not in _KNOWN_EVENT_TYPES:
            raise ValueError(
                f"subscribe: unknown event_type {event_type!r}"
            )

        # Wrap the SwarmEvent handler in a GossipMessage handler that
        # decodes the gossip-data dict back into our typed event.
        def _gossip_callback(msg: Any) -> None:
            # GossipMessage: {type, data, observed_by, unix_ms, propagation_source}
            try:
                # Reconstruct the SwarmEvent envelope shape that
                # decode_event expects, then route through the
                # validator. Schema version is inside data['v'].
                data = dict(msg.data)
                schema_v = int(data.pop("v", SCHEMA_VERSION))
                envelope = {
                    "type": str(msg.type),
                    "v": schema_v,
                    "data": data,
                    "from_peer": str(msg.observed_by or ""),
                    "unix_ms": int(msg.unix_ms or 0),
                }
                event = decode_event(envelope)
            except EventDecodeError as exc:
                logger.warning(
                    "libp2p_swarm_event_decode_failed: type=%s "
                    "reason=%s err=%s",
                    msg.type, exc.reason, exc,
                )
                return
            except Exception as exc:
                logger.warning(
                    "libp2p_swarm_event_unexpected_error: type=%s err=%s",
                    msg.type, exc,
                )
                return

            try:
                handler(event)
            except Exception:
                logger.exception(
                    "libp2p_swarm_event_handler_failed: type=%s",
                    event.type,
                )

        # Register on GossipClient.
        self._gc.on(event_type, _gossip_callback)

        with self._lock:
            self._handlers.setdefault(event_type, []).append(
                (handler, _gossip_callback),
            )

        def _unsubscribe() -> None:
            with self._lock:
                lst = self._handlers.get(event_type)
                if lst is None:
                    return
                # Find and remove (handler, _gossip_callback) pair.
                for i, (h, cb) in enumerate(lst):
                    if h is handler:
                        self._gc.off(event_type, cb)
                        lst.pop(i)
                        break
                if not lst:
                    self._handlers.pop(event_type, None)

        return _unsubscribe

    def close(self) -> None:
        """Detach every subscription. Idempotent."""
        with self._lock:
            for event_type, lst in list(self._handlers.items()):
                for _h, cb in lst:
                    try:
                        self._gc.off(event_type, cb)
                    except Exception:  # pragma: no cover
                        pass
            self._handlers.clear()
