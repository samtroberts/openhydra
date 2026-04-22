# Copyright 2026 OpenHydra contributors — Apache 2.0

"""OpenHydra swarm gossip client (PR-3 / B1).

Wraps the Rust :mod:`openhydra_network` gossip primitives
(``P2PNode.publish_event`` / ``P2PNode.poll_event``) with a
Python-friendly subscribe/dispatch API.

**Wire format** — single topic ``openhydra/swarm/v1/events``. Each
payload is a UTF-8 JSON object::

    {
      "type": "PEER_DEAD",
      "data": { ... event-specific fields ... },
      "observed_by": "12D3KooW…",    # libp2p peer id of the publisher
      "unix_ms": 1713800000000       # wall-clock at publish time
    }

Senders set ``observed_by`` and ``unix_ms`` via :meth:`GossipClient.publish`.
Receivers get the raw message plus a second identifier — the immediate
gossip propagation hop — surfaced by the Rust binding as the first tuple
element of ``poll_event``. The two are usually the same; they diverge
only when a relay node re-emits.

**Event types** — documented in :mod:`peer.gossip_events` (dataclasses).
This module doesn't hard-code a schema beyond ``type``/``data``; callers
register subscribers by event type string.

**Debounce** — outbound ``PEER_DEAD`` is automatically de-duplicated: we
refuse to publish a second ``PEER_DEAD`` for the same target peer within
``peer_dead_debounce_s`` (default 1 s). This caps the per-failure
amplification factor at 1 even when multiple local hints trigger.

**Thread model** — the poll loop runs on a daemon thread started by
:meth:`start`. Each inbound message is dispatched synchronously on that
thread; subscribers **must not block**. The call order across subscribers
for a single message is registration order; one subscriber raising does
not prevent others from running.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


# --- event-type constants ----------------------------------------------------

EVENT_PEER_DEAD = "PEER_DEAD"
EVENT_REQUEST_HOLE_PUNCH = "REQUEST_HOLE_PUNCH"

# All defined event types. Unknown types are still dispatched (to the
# wildcard handler, if any) but don't appear here.
KNOWN_EVENT_TYPES: frozenset[str] = frozenset({
    EVENT_PEER_DEAD,
    EVENT_REQUEST_HOLE_PUNCH,
})

WILDCARD = "*"


# --- public message dataclasses ---------------------------------------------

@dataclass(frozen=True)
class GossipMessage:
    """A decoded swarm event, independent of event type.

    ``data`` is the event-type-specific payload; callers can narrow it via
    :meth:`as_peer_dead` / :meth:`as_request_hole_punch` below.
    """

    type: str
    data: dict[str, Any]
    observed_by: str
    unix_ms: int
    propagation_source: str  # the immediate gossip hop; see module docstring

    def as_peer_dead(self) -> "PeerDeadEvent | None":
        if self.type != EVENT_PEER_DEAD:
            return None
        return PeerDeadEvent(
            libp2p_peer_id=str(self.data.get("libp2p_peer_id") or ""),
            reason=str(self.data.get("reason") or ""),
            observed_by=self.observed_by,
            unix_ms=self.unix_ms,
        )

    def as_request_hole_punch(self) -> "RequestHolePunchEvent | None":
        if self.type != EVENT_REQUEST_HOLE_PUNCH:
            return None
        return RequestHolePunchEvent(
            from_peer_id=str(self.data.get("from_peer_id") or ""),
            to_peer_id=str(self.data.get("to_peer_id") or ""),
            unix_ms=self.unix_ms,
        )


@dataclass(frozen=True)
class PeerDeadEvent:
    libp2p_peer_id: str
    reason: str
    observed_by: str
    unix_ms: int


@dataclass(frozen=True)
class RequestHolePunchEvent:
    from_peer_id: str
    to_peer_id: str
    unix_ms: int


# --- the client --------------------------------------------------------------

Subscriber = Callable[[GossipMessage], None]


class GossipClient:
    """Subscribe/publish façade over :class:`openhydra_network.P2PNode`.

    ``p2p_node`` must expose ``publish_event(bytes) -> None`` and
    ``poll_event() -> tuple[str, bytes] | None``. Pass ``self_libp2p_peer_id``
    so outbound messages stamp their origin; consumers use this for the
    PEER_DEAD quorum (reject claims authored by the dead peer itself).

    Poll cadence is ``poll_interval_s`` (default 0.1 s) which gives
    ~100 ms end-to-end event delivery latency end-to-end after libp2p's
    own propagation time.
    """

    def __init__(
        self,
        *,
        p2p_node: Any,
        self_libp2p_peer_id: str,
        poll_interval_s: float = 0.1,
        peer_dead_debounce_s: float = 1.0,
        hole_punch_debounce_s: float = 5.0,
        clock_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._p2p_node = p2p_node
        self._self_id = str(self_libp2p_peer_id or "")
        self._poll_interval_s = max(0.01, float(poll_interval_s))
        self._peer_dead_debounce_s = max(0.0, float(peer_dead_debounce_s))
        # B1 rendezvous: a second, longer debounce keyed by ``(from, to)``
        # peer-id pair. REQUEST_HOLE_PUNCH is a stronger signal than
        # PEER_DEAD — we actively ask a remote peer to dial us — so we
        # rate-limit it harder. 5 s matches typical NAT binding TTLs;
        # any faster and we'd waste dial capacity on a peer whose prior
        # punch attempt is still resolving.
        self._hole_punch_debounce_s = max(0.0, float(hole_punch_debounce_s))
        self._clock_fn = clock_fn
        self._subscribers: dict[str, list[Subscriber]] = defaultdict(list)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # per-peer debounce map keyed by libp2p_peer_id → last-publish mono.
        self._peer_dead_last: dict[str, float] = {}
        # per-(from, to) pair debounce for REQUEST_HOLE_PUNCH.
        self._hole_punch_last: dict[tuple[str, str], float] = {}
        self._lock = threading.Lock()
        # counters (observable via :meth:`stats` for tests + /v1/internal)
        self._published = 0
        self._received = 0
        self._dispatched = 0
        self._publish_errors = 0
        self._decode_errors = 0

    # --- subscription API ---------------------------------------------------

    def on(self, event_type: str, callback: Subscriber) -> None:
        """Register a callback for ``event_type``.

        Use :data:`WILDCARD` (``"*"``) to receive every message regardless
        of type. Callbacks are invoked in registration order on the poll
        thread; they must not block.
        """
        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._lock:
            self._subscribers[str(event_type)].append(callback)

    def off(self, event_type: str, callback: Subscriber) -> bool:
        """Remove a previously-registered callback. Returns ``True`` if a
        matching registration was removed, ``False`` otherwise."""
        with self._lock:
            handlers = self._subscribers.get(str(event_type))
            if not handlers:
                return False
            try:
                handlers.remove(callback)
                return True
            except ValueError:
                return False

    # --- publish API --------------------------------------------------------

    def publish(
        self,
        event_type: str,
        data: dict[str, Any] | None = None,
    ) -> bool:
        """Publish a swarm event.

        Returns ``True`` if the Rust layer accepted the publish, ``False``
        if it rejected (typically ``InsufficientPeers`` right after boot)
        or if the event was suppressed by the local debounce. Never
        raises — callers treat failure as retryable and keep going.
        """
        payload_data = dict(data or {})
        event_type_clean = str(event_type or "").strip()
        if not event_type_clean:
            logger.warning("gossip_publish_rejected: empty event_type")
            return False

        # PEER_DEAD debounce: skip if we just published one for the same
        # target within the debounce window.
        # REQUEST_HOLE_PUNCH: per-(from, to) pair debounce.
        if (
            event_type_clean == EVENT_REQUEST_HOLE_PUNCH
            and self._hole_punch_debounce_s > 0
        ):
            pair = (
                str(payload_data.get("from_peer_id") or ""),
                str(payload_data.get("to_peer_id") or ""),
            )
            if pair[0] and pair[1]:
                with self._lock:
                    now_mono = self._clock_fn()
                    last = self._hole_punch_last.get(pair)
                    if last is not None and now_mono - last < self._hole_punch_debounce_s:
                        return False
                    self._hole_punch_last[pair] = now_mono

        if event_type_clean == EVENT_PEER_DEAD and self._peer_dead_debounce_s > 0:
            target = str(payload_data.get("libp2p_peer_id") or "")
            if target:
                with self._lock:
                    now_mono = self._clock_fn()
                    last = self._peer_dead_last.get(target)
                    # First publish for this target always succeeds; second
                    # and subsequent publishes must wait out the debounce
                    # window from the previous publish time.
                    if last is not None and now_mono - last < self._peer_dead_debounce_s:
                        return False
                    self._peer_dead_last[target] = now_mono

        envelope = {
            "type": event_type_clean,
            "data": payload_data,
            "observed_by": self._self_id,
            "unix_ms": int(time.time() * 1000),
        }
        try:
            payload_bytes = json.dumps(envelope, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as exc:
            logger.warning("gossip_publish_encode_error: %s", exc)
            self._publish_errors += 1
            return False

        try:
            self._p2p_node.publish_event(payload_bytes)
        except Exception as exc:  # noqa: BLE001 — any binding error is retryable
            logger.debug("gossip_publish_swarm_error: %s", exc)
            self._publish_errors += 1
            return False

        self._published += 1
        return True

    # --- lifecycle ----------------------------------------------------------

    def start(self, thread_name: str = "openhydra-gossip") -> threading.Thread:
        """Start the background poll+dispatch thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return self._thread
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name=thread_name, daemon=True
        )
        self._thread.start()
        return self._thread

    def stop(self, join_timeout_s: float = 2.0) -> None:
        """Signal the poll thread to stop and wait for it."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=join_timeout_s)
        self._thread = None

    # --- one-shot tick (exposed for tests) ----------------------------------

    def tick_once(self) -> int:
        """Drain every currently-queued message and dispatch it.

        Returns the count of messages dispatched. Used by unit tests
        instead of the background thread to keep them deterministic.
        """
        count = 0
        while True:
            item = None
            try:
                item = self._p2p_node.poll_event()
            except Exception as exc:  # noqa: BLE001
                logger.debug("gossip_poll_swarm_error: %s", exc)
                break
            if item is None:
                break
            propagation_source, payload_bytes = item
            self._received += 1
            msg = self._decode(propagation_source, payload_bytes)
            if msg is None:
                continue
            self._dispatch(msg)
            count += 1
        return count

    # --- internal -----------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.tick_once()
            except Exception as exc:  # noqa: BLE001 — never die silently
                logger.exception("gossip_dispatch_thread_error: %s", exc)
            if self._stop_event.wait(self._poll_interval_s):
                break

    def _decode(
        self,
        propagation_source: str,
        payload_bytes: bytes,
    ) -> GossipMessage | None:
        try:
            text = payload_bytes.decode("utf-8")
            envelope = json.loads(text)
            if not isinstance(envelope, dict):
                raise ValueError("envelope is not a JSON object")
        except (UnicodeDecodeError, ValueError) as exc:
            self._decode_errors += 1
            logger.debug("gossip_decode_error: %s", exc)
            return None
        event_type = str(envelope.get("type") or "").strip()
        if not event_type:
            self._decode_errors += 1
            return None
        data = envelope.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        return GossipMessage(
            type=event_type,
            data=data,
            observed_by=str(envelope.get("observed_by") or ""),
            unix_ms=int(envelope.get("unix_ms") or 0),
            propagation_source=str(propagation_source or ""),
        )

    def _dispatch(self, msg: GossipMessage) -> None:
        # Dispatch to specific-type subscribers, then wildcard, in
        # registration order. A raising subscriber logs but doesn't break
        # the chain for others.
        with self._lock:
            specific = list(self._subscribers.get(msg.type, ()))
            wildcard = list(self._subscribers.get(WILDCARD, ()))
        for cb in specific + wildcard:
            try:
                cb(msg)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "gossip_subscriber_error: type=%s err=%s", msg.type, exc
                )
            else:
                self._dispatched += 1

    # --- diagnostics --------------------------------------------------------

    def stats(self) -> dict[str, int]:
        """Snapshot counters for observability (shape matches what we'd
        surface on ``/v1/internal/capacity`` in a future PR)."""
        with self._lock:
            return {
                "published": int(self._published),
                "received": int(self._received),
                "dispatched": int(self._dispatched),
                "publish_errors": int(self._publish_errors),
                "decode_errors": int(self._decode_errors),
            }


# --- convenience publishers --------------------------------------------------

def publish_peer_dead(
    client: GossipClient,
    *,
    libp2p_peer_id: str,
    reason: str = "",
) -> bool:
    """Emit a ``PEER_DEAD`` event for the given libp2p peer id."""
    return client.publish(
        EVENT_PEER_DEAD,
        {"libp2p_peer_id": str(libp2p_peer_id), "reason": str(reason)},
    )


def attach_hole_punch_responder(
    client: "GossipClient",
    *,
    p2p_node: Any,
    self_libp2p_peer_id: str,
) -> Callable[["GossipMessage"], None]:
    """Wire the passive side of the B1 rendezvous.

    Subscribes to ``REQUEST_HOLE_PUNCH`` on ``client``; when an event
    arrives with ``to_peer_id == self_libp2p_peer_id`` **and** a
    non-self ``from_peer_id``, calls
    :meth:`p2p_node.dial_peer(from_peer_id)` to issue the simultaneous
    dial that lets DCUtR punch through symmetric NAT.

    Returns the subscriber callable so callers can ``client.off(...)``
    it during shutdown. Idempotent: registering twice produces two
    subscribers, but :meth:`GossipClient.off` only removes one at a
    time — so teardown matches setup.

    Exceptions from ``dial_peer`` are logged at debug and **not**
    re-raised — the gossip dispatcher continues running.
    """
    self_id = str(self_libp2p_peer_id or "").strip()

    def _respond(msg: GossipMessage) -> None:
        if msg.type != EVENT_REQUEST_HOLE_PUNCH:
            return
        to_peer = str(msg.data.get("to_peer_id") or "").strip()
        from_peer = str(msg.data.get("from_peer_id") or "").strip()
        if not self_id or not to_peer or to_peer != self_id:
            return
        if not from_peer or from_peer == self_id:
            return
        try:
            p2p_node.dial_peer(from_peer)
            logger.info(
                "b1_hole_punch_responder: dialed %s in response to "
                "REQUEST_HOLE_PUNCH (observed_by=%s)",
                from_peer,
                msg.observed_by,
            )
        except Exception as exc:  # noqa: BLE001 — dispatcher must not die
            logger.debug(
                "b1_hole_punch_dial_failed: target=%s err=%s",
                from_peer, exc,
            )

    client.on(EVENT_REQUEST_HOLE_PUNCH, _respond)
    return _respond


def publish_request_hole_punch(
    client: GossipClient,
    *,
    from_peer_id: str,
    to_peer_id: str,
) -> bool:
    """Emit a ``REQUEST_HOLE_PUNCH`` signal asking ``to_peer_id`` to
    simultaneously dial ``from_peer_id`` — the rendezvous lever that
    actually lights up DCUtR against symmetric NATs."""
    return client.publish(
        EVENT_REQUEST_HOLE_PUNCH,
        {"from_peer_id": str(from_peer_id), "to_peer_id": str(to_peer_id)},
    )
