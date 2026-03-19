from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
from pathlib import Path
import socket as _socket
import time
from urllib import parse

import grpc
import requests as _http_requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from coordinator.transport import TransportConfig, create_channel
from peer import peer_pb2
from peer import peer_pb2_grpc

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared HTTP session for DHT bootstrap lookups.
#
# A single requests.Session is reused across all threads: urllib3's underlying
# PoolManager is thread-safe, and by sharing one session we keep TCP
# connections alive (HTTP/1.1 Keep-Alive) between /lookup calls to the same
# bootstrap node.  This avoids re-triggering the iptables NEW-state hashlimit
# counter on every request once the initial connection is established.
#
# Pool sizing: max_workers in ThreadPoolExecutor = min(8, n_sources).  Each
# worker holds at most one live connection to a given host, so pool_connections
# and pool_maxsize = 8 gives each source node its own persistent socket.
#
# C10K / idle-connection guard:
#   The bootstrap DHT server sets a 30-second keep-alive timeout (server-side),
#   but we also arm the socket with TCP-level keepalive probes on the client
#   side.  This ensures the OS actively detects and reclaims dead connections
#   (e.g. the server crashed or a NAT gateway silently dropped the session)
#   rather than waiting for the default ~2-hour OS keepalive timeout.
#   Timeline: idle 10 s → first probe; if no ACK, retry every 5 s × 3 times
#   → connection marked dead and removed from the pool after ~25 s total.
# ---------------------------------------------------------------------------
class _TcpKeepAliveAdapter(HTTPAdapter):
    """HTTPAdapter that arms every socket with TCP keepalive probes."""

    # Seconds idle before the first keepalive probe is sent.
    _KEEPIDLE = 10
    # Seconds between subsequent probes.
    _KEEPINTVL = 5
    # Number of unanswered probes before the connection is declared dead.
    _KEEPCNT = 3

    def init_poolmanager(self, *args: object, **kwargs: object) -> None:  # type: ignore[override]
        opts: list[tuple[int, int, int]] = [
            (_socket.SOL_SOCKET, _socket.SO_KEEPALIVE, 1),
        ]
        # TCP_KEEPIDLE / TCP_KEEPINTVL / TCP_KEEPCNT are Linux + macOS 10.9+.
        # Absent on older BSD / Windows; getattr guard keeps the code portable.
        for name, val in [
            ("TCP_KEEPIDLE", self._KEEPIDLE),
            ("TCP_KEEPINTVL", self._KEEPINTVL),
            ("TCP_KEEPCNT", self._KEEPCNT),
        ]:
            opt = getattr(_socket, name, None)
            if opt is not None:
                opts.append((_socket.IPPROTO_TCP, opt, val))
        kwargs["socket_options"] = opts
        super().init_poolmanager(*args, **kwargs)


def _build_dht_session() -> _http_requests.Session:
    session = _http_requests.Session()
    retry = Retry(total=0)  # coordinator handles retries via ThreadPoolExecutor
    adapter = _TcpKeepAliveAdapter(
        pool_connections=8,
        pool_maxsize=8,
        max_retries=retry,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


_DHT_SESSION: _http_requests.Session = _build_dht_session()


def _normalize_tags(raw: object) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",")]
    else:
        try:
            values = [str(item).strip() for item in list(raw)]
        except TypeError:
            values = []
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        norm = value.lower()
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(norm)
    return tuple(deduped)


def _normalize_layer_indices(raw: object) -> tuple[int, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        tokens = [part.strip() for part in raw.split(",")]
    else:
        try:
            tokens = [str(item).strip() for item in list(raw)]
        except TypeError:
            tokens = []
    out: list[int] = []
    seen: set[int] = set()
    for token in tokens:
        if not token:
            continue
        try:
            value = int(token)
        except ValueError:
            continue
        if value < 0 or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return tuple(sorted(out))


@dataclass(frozen=True)
class PeerEndpoint:
    peer_id: str
    host: str
    port: int
    model_id: str | None = None
    operator_id: str | None = None
    region: str | None = None
    bandwidth_mbps: float = 0.0
    seeding_enabled: bool = False
    seed_upload_limit_mbps: float = 0.0
    seed_target_upload_limit_mbps: float = 0.0
    seed_inference_active: bool = False
    runtime_backend: str = "toy_cpu"
    runtime_target: str = "cpu"
    runtime_model_id: str = ""
    quantization_mode: str = "fp32"
    quantization_bits: int = 0
    runtime_gpu_available: bool = False
    runtime_estimated_tokens_per_sec: float = 0.0
    runtime_estimated_memory_mb: int = 0
    privacy_noise_variance: float = 0.0
    privacy_noise_payloads: int = 0
    privacy_noise_observed_variance_ema: float = 0.0
    privacy_noise_last_audit_tag: str = ""
    reputation_score: float = 0.0
    staked_balance: float = 0.0
    expert_tags: tuple[str, ...] = ()
    expert_layer_indices: tuple[int, ...] = ()
    expert_router: bool = False
    expert_admission_approved: bool = True
    expert_admission_reason: str = "approved"
    geo_verified: bool = False
    geo_challenge_rtt_ms: float | None = None
    geo_penalty_score: float = 0.0
    public_key_hex: str = ""
    # Phase 2: free GPU VRAM in MB.  0 = CPU-only or not reported.
    available_vram_mb: int = 0
    # Phase 3: layer-range sharding.  Non-zero when peer runs a sub-range of layers.
    # Peer covers transformer layers [layer_start, layer_end); total_layers = model depth.
    layer_start: int = 0
    layer_end: int = 0
    total_layers: int = 0
    # Phase 5: P2P model distribution.  seeder_http_port > 0 means this peer runs
    # a ModelSeedServer accepting HTTP Range requests for model file downloads.
    # cached_model_ids lists HF model IDs available for peer-to-peer download.
    seeder_http_port: int = 0
    cached_model_ids: tuple[str, ...] = ()
    # Phase A: local fast-path TCP port for same-LAN raw tensor transfer.
    # 0 = disabled (no fast-path server running).
    local_fast_path_port: int = 0

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PeerEndpoint":
        """Construct a PeerEndpoint from a raw dict (DHT record, JSON config, etc.).

        Handles type coercion, default values, and expert admission gating.
        Unknown keys in *data* are silently ignored.  Accepts both
        ``public_key_hex`` and the DHT-native ``peer_public_key`` key.
        """
        from coordinator.peer_utils import normalize_layer_indices, normalize_tags

        admission = bool(data.get("expert_admission_approved", True))
        geo_rtt = data.get("geo_challenge_rtt_ms")
        return cls(
            peer_id=str(data.get("peer_id", "")).strip(),
            host=str(data.get("host", "")).strip(),
            port=int(data.get("port", 0)),
            model_id=data.get("model_id"),
            operator_id=(None if data.get("operator_id") in (None, "") else str(data["operator_id"])),
            region=(None if data.get("region") in (None, "") else str(data["region"])),
            bandwidth_mbps=float(data.get("bandwidth_mbps", 0.0)),
            seeding_enabled=bool(data.get("seeding_enabled", False)),
            seed_upload_limit_mbps=float(data.get("seed_upload_limit_mbps", 0.0)),
            seed_target_upload_limit_mbps=float(data.get("seed_target_upload_limit_mbps", 0.0)),
            seed_inference_active=bool(data.get("seed_inference_active", False)),
            runtime_backend=str(data.get("runtime_backend", "toy_cpu")),
            runtime_target=str(data.get("runtime_target", "cpu")),
            runtime_model_id=str(data.get("runtime_model_id", "")),
            quantization_mode=str(data.get("quantization_mode", "fp32")),
            quantization_bits=int(data.get("quantization_bits", 0)),
            runtime_gpu_available=bool(data.get("runtime_gpu_available", False)),
            runtime_estimated_tokens_per_sec=float(data.get("runtime_estimated_tokens_per_sec", 0.0)),
            runtime_estimated_memory_mb=int(data.get("runtime_estimated_memory_mb", 0)),
            privacy_noise_variance=float(data.get("privacy_noise_variance", 0.0)),
            privacy_noise_payloads=int(data.get("privacy_noise_payloads", 0)),
            privacy_noise_observed_variance_ema=float(data.get("privacy_noise_observed_variance_ema", 0.0)),
            privacy_noise_last_audit_tag=str(data.get("privacy_noise_last_audit_tag", "")),
            reputation_score=float(data.get("reputation_score", 0.0)),
            staked_balance=float(data.get("staked_balance", 0.0)),
            expert_tags=normalize_tags(data.get("expert_tags", [])) if admission else (),
            expert_layer_indices=normalize_layer_indices(data.get("expert_layer_indices", [])) if admission else (),
            expert_router=bool(data.get("expert_router", False)) if admission else False,
            expert_admission_approved=admission,
            expert_admission_reason=str(data.get("expert_admission_reason", "approved")),
            geo_verified=bool(data.get("geo_verified", False)),
            geo_challenge_rtt_ms=float(geo_rtt) if geo_rtt is not None else None,
            geo_penalty_score=float(data.get("geo_penalty_score", 0.0)),
            public_key_hex=str(data.get("public_key_hex", "") or data.get("peer_public_key", "") or ""),
            available_vram_mb=int(data.get("available_vram_mb", 0)),
            layer_start=int(data.get("layer_start", 0)),
            layer_end=int(data.get("layer_end", 0)),
            total_layers=int(data.get("total_layers", 0)),
            seeder_http_port=int(data.get("seeder_http_port", 0)),
            cached_model_ids=tuple(
                str(m) for m in list(data.get("cached_model_ids", []) or [])
                if str(m).strip()
            ),
            local_fast_path_port=int(data.get("local_fast_path_port", 0)),
        )

    def replace(self, **overrides: Any) -> "PeerEndpoint":
        """Return a copy with given fields replaced (frozen dataclass helper)."""
        import dataclasses as _dc
        return _dc.replace(self, **overrides)


@dataclass(frozen=True)
class PeerHealth:
    peer: PeerEndpoint
    healthy: bool
    latency_ms: float
    load_pct: float
    daemon_mode: str
    error: str = ""


def load_peer_config(path: str | Path) -> list[PeerEndpoint]:
    raw = json.loads(Path(path).read_text())
    return [PeerEndpoint.from_dict(item) for item in raw]


def _normalize_dht_urls(
    dht_url: str | None = None,
    dht_urls: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    if dht_url:
        for token in str(dht_url).split(","):
            value = token.strip()
            if not value or value in seen:
                continue
            seen.add(value)
            out.append(value)
    for item in list(dht_urls or []):
        for token in str(item).split(","):
            value = token.strip()
            if not value or value in seen:
                continue
            seen.add(value)
            out.append(value)
    return out


def _lookup_peers_payload(
    *,
    dht_url: str,
    model_id: str,
    timeout_s: float,
    preferred_region: str | None,
    limit: int | None,
    sloppy_factor: int | None,
    dsht_replicas: int | None,
) -> list[dict]:
    query_items: dict[str, str | int] = {"model_id": model_id}
    if preferred_region:
        query_items["preferred_region"] = str(preferred_region)
    if limit is not None and int(limit) > 0:
        query_items["limit"] = int(limit)
    if sloppy_factor is not None and int(sloppy_factor) >= 0:
        query_items["sloppy_factor"] = int(sloppy_factor)
    if dsht_replicas is not None and int(dsht_replicas) >= 0:
        query_items["dsht_replicas"] = int(dsht_replicas)
    query = parse.urlencode(query_items)
    url = f"{dht_url.rstrip('/')}/lookup?{query}"
    resp = _DHT_SESSION.get(url, timeout=timeout_s)
    resp.raise_for_status()
    payload = resp.json()
    peers_payload = list(payload.get("peers", []))

    rebalance = dict(payload.get("rebalance") or {})
    if bool(rebalance.get("active", False)):
        recommended = int(rebalance.get("recommended_dsht_replicas", 0) or 0)
        requested = int(dsht_replicas) if dsht_replicas is not None else 0
        if recommended > requested:
            query_items["dsht_replicas"] = recommended
            follow_url = f"{dht_url.rstrip('/')}/lookup?{parse.urlencode(query_items)}"
            try:
                follow_resp = _DHT_SESSION.get(follow_url, timeout=timeout_s)
                follow_resp.raise_for_status()
                peers_payload.extend(list(follow_resp.json().get("peers", [])))
            except Exception:
                logger.warning("dht_rebalance_follow_failed url=%s", follow_url, exc_info=True)
    return peers_payload


def load_peers_from_dht(
    dht_url: str | None = None,
    model_id: str = "",
    timeout_s: float = 2.0,
    preferred_region: str | None = None,
    limit: int | None = None,
    sloppy_factor: int | None = None,
    dsht_replicas: int | None = None,
    dht_urls: list[str] | tuple[str, ...] | None = None,
    hivemind_adapter: Any = None,
) -> list[PeerEndpoint]:
    sources = _normalize_dht_urls(dht_url=dht_url, dht_urls=dht_urls)
    if not sources and hivemind_adapter is None:
        raise ValueError("dht_url_missing")

    merged_payload: list[dict] = []
    failures: list[tuple[str, Exception]] = []
    worker_count = max(1, min(8, len(sources)))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        future_by_url = {
            pool.submit(
                _lookup_peers_payload,
                dht_url=url,
                model_id=model_id,
                timeout_s=timeout_s,
                preferred_region=preferred_region,
                limit=limit,
                sloppy_factor=sloppy_factor,
                dsht_replicas=dsht_replicas,
            ): url
            for url in sources
        }
        for future in as_completed(future_by_url):
            url = future_by_url[future]
            try:
                merged_payload.extend(future.result())
            except Exception as exc:
                failures.append((url, exc))

    if not merged_payload and failures:
        raise RuntimeError(
            "dht_lookup_all_failed:"
            + ";".join(f"{url}={type(exc).__name__}:{exc}" for url, exc in failures)
        ) from failures[-1][1]

    for failed_url, exc in failures:
        logger.warning("dht_lookup_partial_failure: %s (%s)", failed_url, exc)

    # Hivemind DHT lookup (dual-stack merge).
    if hivemind_adapter is not None:
        try:
            hivemind_peers = hivemind_adapter.lookup(
                model_id=model_id,
                timeout_s=timeout_s,
            )
            if hivemind_peers:
                logger.debug(
                    "hivemind_lookup: model=%s found %d peers",
                    model_id, len(hivemind_peers),
                )
                merged_payload.extend(hivemind_peers)
        except Exception as exc:
            logger.debug("hivemind_lookup_error: %s", exc)

    if not merged_payload and not sources:
        return []

    best_by_peer: dict[str, dict] = {}
    for item in merged_payload:
        peer_id = str(item.get("peer_id", "")).strip()
        if not peer_id:
            continue
        current = best_by_peer.get(peer_id)
        current_ts = int((current or {}).get("updated_unix_ms", 0) or 0)
        incoming_ts = int(item.get("updated_unix_ms", 0) or 0)
        if current is None or incoming_ts >= current_ts:
            best_by_peer[peer_id] = dict(item)

    peers_payload = sorted(
        list(best_by_peer.values()),
        key=lambda item: (int(item.get("updated_unix_ms", 0) or 0), str(item.get("peer_id", ""))),
        reverse=True,
    )

    peers: list[PeerEndpoint] = []
    for item in peers_payload:
        peer_id = str(item.get("peer_id", "")).strip()
        host = str(item.get("host", "")).strip()
        port = item.get("port")
        if not peer_id or not host or port is None:
            continue
        peers.append(PeerEndpoint.from_dict(item))
    return peers


def dedupe_peers(peers: list[PeerEndpoint]) -> list[PeerEndpoint]:
    seen: set[str] = set()
    out: list[PeerEndpoint] = []
    for peer in peers:
        if peer.peer_id in seen:
            continue
        seen.add(peer.peer_id)
        out.append(peer)
    return out


class PathFinder:
    """Reachability and latency discovery over configurable transport."""

    def __init__(self, timeout_ms: int = 700, transport_config: TransportConfig | None = None):
        self.timeout_s = timeout_ms / 1000.0
        self.transport_config = transport_config or TransportConfig()

    def ping(self, peer: PeerEndpoint) -> PeerHealth:
        t0 = time.perf_counter()
        try:
            with create_channel(peer.address, self.transport_config) as channel:
                stub = peer_pb2_grpc.PeerStub(channel)
                response = stub.Ping(
                    peer_pb2.PingRequest(sent_unix_ms=int(time.time() * 1000)),
                    timeout=self.timeout_s,
                )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return PeerHealth(
                peer=peer,
                healthy=bool(response.ok),
                latency_ms=latency_ms,
                load_pct=float(response.load_pct),
                daemon_mode=response.daemon_mode,
            )
        except grpc.RpcError as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return PeerHealth(
                peer=peer,
                healthy=False,
                latency_ms=latency_ms,
                load_pct=100.0,
                daemon_mode="unknown",
                error=f"{exc.code().name}: {exc.details() or 'unreachable'}",
            )

    def survey(self, peers: list[PeerEndpoint]) -> list[PeerHealth]:
        return [self.ping(peer) for peer in peers]

    def discover(self, peers: list[PeerEndpoint], max_latency_ms: float | None = None) -> list[PeerHealth]:
        health = self.survey(peers)
        filtered = [h for h in health if h.healthy]
        if max_latency_ms is not None:
            filtered = [h for h in filtered if h.latency_ms <= max_latency_ms]
        return sorted(filtered, key=lambda x: x.latency_ms)
