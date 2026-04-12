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
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import secrets
import signal
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

import grpc

from dht.node import InMemoryDhtNode
from openhydra_logging import configure_logging
from openhydra_secrets import is_insecure_secret_value, load_secret_store
from peer.crypto import verify_geo_challenge
from peer import peer_pb2
from peer import peer_pb2_grpc


@dataclass(frozen=True)
class BootstrapNode:
    host: str
    port: int


def default_bootstrap_nodes() -> list[BootstrapNode]:
    return [BootstrapNode(host="127.0.0.1", port=8468)]


def model_key(model_id: str) -> str:
    return f"model:{model_id}"


def dsht_key(model_id: str, replica_index: int) -> str:
    return f"model:{model_id}:dsht:{replica_index}"


def _normalize_tags(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = [item.strip() for item in raw.split(",")]
    else:
        try:
            parts = [str(item).strip() for item in list(raw)]
        except TypeError:
            parts = []
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if not part:
            continue
        norm = part.lower()
        if norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def _normalize_layer_indices(raw: object) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = [item.strip() for item in raw.split(",")]
    else:
        try:
            parts = [str(item).strip() for item in list(raw)]
        except TypeError:
            parts = []
    out: list[int] = []
    seen: set[int] = set()
    for part in parts:
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            continue
        if value < 0 or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return sorted(out)


def _resolve_bootstrap_profile_settings(parser: argparse.ArgumentParser, args: argparse.Namespace) -> dict[str, Any]:
    profile = str(getattr(args, "deployment_profile", "dev") or "dev").strip().lower()
    if profile not in {"dev", "prod"}:
        parser.error("unsupported deployment profile")
    try:
        secret_store = load_secret_store(getattr(args, "secrets_file", None))
    except RuntimeError as exc:
        parser.error(str(exc))

    geo_seed = str(getattr(args, "geo_challenge_seed", "") or "").strip()
    if is_insecure_secret_value(geo_seed):
        geo_seed = str(secret_store.get("OPENHYDRA_GEO_CHALLENGE_SEED", geo_seed) or "").strip()

    if profile == "prod":
        if not bool(getattr(args, "geo_challenge_enabled", True)):
            parser.error("prod profile requires geo challenge to stay enabled")
        if is_insecure_secret_value(geo_seed):
            parser.error(
                "prod profile requires strong geo challenge seed via "
                "--geo-challenge-seed or OPENHYDRA_GEO_CHALLENGE_SEED"
            )
    return {
        "deployment_profile": profile,
        "geo_challenge_seed": geo_seed or str(getattr(args, "geo_challenge_seed")),
    }


class DhtBootstrapHandler(BaseHTTPRequestHandler):
    # Use HTTP/1.1 so clients can keep TCP connections alive between requests.
    # Without this, BaseHTTPRequestHandler defaults to HTTP/1.0 which sends
    # "Connection: close" on every response, forcing a new TCP handshake per
    # lookup — 57 handshakes when a coordinator scans 19 models × 3 nodes.
    protocol_version = "HTTP/1.1"

    # Maximum seconds an idle HTTP/1.1 keep-alive connection is held open.
    #
    # C10K guard: ThreadingHTTPServer spawns one thread per *connection* (not
    # per request).  If every peer holds a persistent connection indefinitely,
    # 5 000 concurrent peers = 5 000 live threads, each consuming ~8 MB of
    # stack → ~40 GB RAM on the bootstrap node.  A 30-second idle timeout
    # bounds concurrent threads to peers_active_in_last_30s — for a 30-s peer
    # heartbeat interval that is at most ~1–2× the connected peer count rather
    # than an unbounded accumulation.  Peers reconnect transparently for any
    # request after their connection expires; the Keep-Alive benefit (avoiding
    # the SYN + TLS handshake for rapid burst traffic) is preserved for active
    # sessions while idle threads are freed promptly.
    timeout = 30

    dht: InMemoryDhtNode | None = None
    default_ttl_seconds: int = 300
    default_dsht_replicas: int = 2
    default_dsht_max_replicas: int = 32
    default_lookup_window_seconds: int = 1
    default_lookup_max_requests_per_window: int = 120
    default_geo_challenge_enabled: bool = True
    default_geo_challenge_timeout_ms: int = 800
    default_geo_max_rtt_ms: float = 50.0
    default_geo_challenge_seed: str = "openhydra-geo-dev-seed"
    default_expert_min_reputation_score: float = 60.0
    default_expert_min_staked_balance: float = 0.01
    default_expert_require_stake: bool = True
    _lookup_buckets: dict[str, dict[str, float]] = {}
    _lookup_lock = threading.Lock()
    _rebalance_hints: dict[str, dict[str, Any]] = {}
    _rebalance_lock = threading.Lock()
    # Layer rebalance directives: peer_id → list[directive_dict]
    _layer_rebalance_directives: dict[str, list[dict[str, Any]]] = {}
    _layer_rebalance_lock = threading.Lock()

    def _send_json(
        self,
        payload: dict[str, Any],
        status: HTTPStatus = HTTPStatus.OK,
        headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if headers:
            for key, value in headers.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _require_dht(self) -> InMemoryDhtNode:
        if self.dht is None:
            raise RuntimeError("dht_uninitialized")
        return self.dht

    def _normalize_peer_record(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not payload.get("peer_id"):
            raise ValueError("missing_peer_id")
        if not payload.get("model_id"):
            raise ValueError("missing_model_id")
        if not payload.get("host"):
            raise ValueError("missing_host")
        if payload.get("port") is None:
            raise ValueError("missing_port")

        operator_raw = payload.get("operator_id")
        operator_id = None if operator_raw in (None, "") else str(operator_raw)
        region_raw = payload.get("region")
        region = None if region_raw in (None, "") else str(region_raw)
        peer_public_key = str(payload.get("peer_public_key", "") or "")
        if peer_public_key and len(peer_public_key) != 64:
            peer_public_key = ""

        return {
            "peer_id": str(payload["peer_id"]),
            "model_id": str(payload["model_id"]),
            "host": str(payload["host"]),
            "port": int(payload["port"]),
            "operator_id": operator_id,
            "region": region,
            "load_pct": float(payload.get("load_pct", 0.0)),
            "daemon_mode": str(payload.get("daemon_mode", "polite")),
            "bandwidth_mbps": float(payload.get("bandwidth_mbps", 0.0)),
            "seeding_enabled": bool(payload.get("seeding_enabled", False)),
            "seed_upload_limit_mbps": float(payload.get("seed_upload_limit_mbps", 0.0)),
            "seed_target_upload_limit_mbps": float(payload.get("seed_target_upload_limit_mbps", 0.0)),
            "seed_inference_active": bool(payload.get("seed_inference_active", False)),
            "runtime_backend": str(payload.get("runtime_backend", "toy_cpu")),
            "runtime_target": str(payload.get("runtime_target", "cpu")),
            "runtime_model_id": str(payload.get("runtime_model_id", "")),
            "quantization_mode": str(payload.get("quantization_mode", "fp32")),
            "quantization_bits": int(payload.get("quantization_bits", 0)),
            "runtime_gpu_available": bool(payload.get("runtime_gpu_available", False)),
            "runtime_estimated_tokens_per_sec": float(payload.get("runtime_estimated_tokens_per_sec", 0.0)),
            "runtime_estimated_memory_mb": int(payload.get("runtime_estimated_memory_mb", 0)),
            "privacy_noise_variance": float(payload.get("privacy_noise_variance", 0.0)),
            "privacy_noise_payloads": int(payload.get("privacy_noise_payloads", 0)),
            "privacy_noise_observed_variance_ema": float(payload.get("privacy_noise_observed_variance_ema", 0.0)),
            "privacy_noise_last_audit_tag": str(payload.get("privacy_noise_last_audit_tag", "")),
            "reputation_score": float(payload.get("reputation_score", 0.0)),
            "staked_balance": float(payload.get("staked_balance", 0.0)),
            "expert_tags": _normalize_tags(payload.get("expert_tags", [])),
            "expert_layer_indices": _normalize_layer_indices(payload.get("expert_layer_indices", [])),
            "expert_router": bool(payload.get("expert_router", False)),
            "peer_public_key": peer_public_key,
            "expert_admission_approved": True,
            "expert_admission_reason": "approved",
            "updated_unix_ms": int(payload.get("updated_unix_ms", 0)),
            # Layer sharding fields — required for auto-discovery of
            # sharded pipelines. Without these, coordinators can't
            # assemble multi-peer pipelines from DHT-discovered peers.
            "layer_start": int(payload.get("layer_start", 0)),
            "layer_end": int(payload.get("layer_end", 0)),
            "total_layers": int(payload.get("total_layers", 0)),
        }

    @staticmethod
    def _claims_expert(record: dict[str, Any]) -> bool:
        tags = list(record.get("expert_tags", []))
        layers = list(record.get("expert_layer_indices", []))
        router = bool(record.get("expert_router", False))
        return bool(tags or layers or router)

    def _apply_expert_admission_controls(self, record: dict[str, Any]) -> dict[str, Any]:
        if not self._claims_expert(record):
            record["expert_admission_approved"] = True
            record["expert_admission_reason"] = "not_expert"
            return record

        reputation_score = max(0.0, float(record.get("reputation_score", 0.0)))
        staked_balance = max(0.0, float(record.get("staked_balance", 0.0)))
        min_rep = max(0.0, float(self.default_expert_min_reputation_score))
        min_stake = max(0.0, float(self.default_expert_min_staked_balance))
        require_stake = bool(self.default_expert_require_stake)

        approved = reputation_score >= min_rep
        reason = "approved"
        if not approved:
            reason = "low_reputation"
        elif require_stake and staked_balance < min_stake:
            approved = False
            reason = "unstaked_or_new"

        if approved:
            record["expert_admission_approved"] = True
            record["expert_admission_reason"] = "approved"
            return record

        # Frictionless admission: node still joins DHT, but expert claim is stripped.
        record["expert_tags"] = []
        record["expert_layer_indices"] = []
        record["expert_router"] = False
        record["expert_admission_approved"] = False
        record["expert_admission_reason"] = reason
        record["load_pct"] = min(100.0, float(record.get("load_pct", 0.0)) + 20.0)
        return record

    def _put_announcement(self, payload: dict[str, Any]) -> dict[str, Any]:
        from peer.identity import verify_announce as _verify_announce
        _identity_verified = False
        if "public_key" in payload and "signature" in payload:
            try:
                _identity_verified = _verify_announce(
                    payload["public_key"], payload.get("peer_id", ""),
                    payload.get("host", ""), int(payload.get("port", 0)),
                    payload.get("model_id", ""), payload["signature"],
                )
                if not _identity_verified:
                    logging.warning("identity_verify_failed peer_id=%s", payload.get("peer_id"))
            except Exception:
                pass
        payload["identity_verified"] = _identity_verified

        dht = self._require_dht()
        record = self._normalize_peer_record(payload)
        record = self._apply_expert_admission_controls(record)
        ttl = int(payload.get("ttl_seconds", self.default_ttl_seconds))
        geo = self._geo_verify_record(record)
        claimed_region = str(record.get("region") or "").strip() or None
        verified = bool(geo.get("verified", False))
        record["geo_verified"] = verified
        record["geo_challenge_reason"] = str(geo.get("reason", "unknown"))
        rtt_ms = geo.get("rtt_ms")
        record["geo_challenge_rtt_ms"] = (round(float(rtt_ms), 6) if rtt_ms is not None else None)
        record["geo_penalty_score"] = (0.0 if verified else 1.0)
        if claimed_region and not verified:
            record["region_claimed"] = claimed_region
            record["region"] = None
            record["load_pct"] = min(100.0, float(record.get("load_pct", 0.0)) + 25.0)

        dsht_replicas = int(payload.get("dsht_replicas", self.default_dsht_replicas))
        dsht_replicas = max(0, min(int(self.default_dsht_max_replicas), dsht_replicas))
        dht.put(model_key(record["model_id"]), record, unique_field="peer_id", ttl_seconds=ttl)
        for replica_index in range(dsht_replicas):
            dht.put(dsht_key(record["model_id"], replica_index), record, unique_field="peer_id", ttl_seconds=ttl)
        return record

    def _geo_verify_record(self, record: dict[str, Any]) -> dict[str, Any]:
        claimed_region = str(record.get("region") or "").strip()
        if not claimed_region:
            return {"verified": False, "reason": "no_region", "rtt_ms": None}
        if not bool(self.default_geo_challenge_enabled):
            return {"verified": True, "reason": "challenge_disabled", "rtt_ms": None}

        nonce = secrets.token_hex(16)
        request_timeout_s = max(0.05, float(self.default_geo_challenge_timeout_ms) / 1000.0)
        peer_addr = f"{record['host']}:{int(record['port'])}"

        try:
            t0 = time.perf_counter()
            with grpc.insecure_channel(peer_addr) as channel:
                stub = peer_pb2_grpc.PeerStub(channel)
                resp = stub.Ping(
                    peer_pb2.PingRequest(
                        sent_unix_ms=int(time.time() * 1000),
                        geo_nonce=nonce,
                        geo_claimed_region=claimed_region,
                    ),
                    timeout=request_timeout_s,
                )
            rtt_ms = (time.perf_counter() - t0) * 1000.0
        except Exception:
            return {"verified": False, "reason": "challenge_unreachable", "rtt_ms": None}

        response_peer_id = str(getattr(resp, "peer_id", "") or "").strip()
        if response_peer_id and response_peer_id != str(record["peer_id"]):
            return {"verified": False, "reason": "peer_id_mismatch", "rtt_ms": rtt_ms}

        signature = str(getattr(resp, "geo_nonce_signature", "") or "").strip()
        if not signature:
            return {"verified": False, "reason": "missing_signature", "rtt_ms": rtt_ms}

        signature_ok = verify_geo_challenge(
            peer_id=str(record["peer_id"]),
            nonce=nonce,
            claimed_region=claimed_region,
            signature=signature,
            shared_secret_seed=str(self.default_geo_challenge_seed),
        )
        if not signature_ok:
            return {"verified": False, "reason": "signature_invalid", "rtt_ms": rtt_ms}

        max_rtt_ms = max(1.0, float(self.default_geo_max_rtt_ms))
        if rtt_ms > max_rtt_ms:
            return {"verified": False, "reason": "latency_violation", "rtt_ms": rtt_ms}

        return {"verified": True, "reason": "ok", "rtt_ms": rtt_ms}

    @classmethod
    def _lookup_rate_snapshot(cls) -> dict[str, dict[str, int]]:
        now = time.time()
        window = max(1, int(cls.default_lookup_window_seconds))
        with cls._lookup_lock:
            expired = [
                key
                for key, item in cls._lookup_buckets.items()
                if (now - float(item.get("window_start", 0.0))) >= window
            ]
            for key in expired:
                cls._lookup_buckets.pop(key, None)

            return {
                key: {"count": int(item.get("count", 0))}
                for key, item in cls._lookup_buckets.items()
            }

    @classmethod
    def _lookup_rebalance_snapshot(cls) -> dict[str, dict[str, Any]]:
        now = time.time()
        with cls._rebalance_lock:
            expired = [
                key
                for key, item in cls._rebalance_hints.items()
                if now >= float(item.get("expires_at_unix", 0.0))
            ]
            for key in expired:
                cls._rebalance_hints.pop(key, None)
            return {
                key: {
                    "recommended_dsht_replicas": int(item.get("recommended_dsht_replicas", 0)),
                    "replica_indices": list(item.get("dsht_replica_indices", [])),
                    "triggered_unix_ms": int(item.get("triggered_unix_ms", 0)),
                }
                for key, item in cls._rebalance_hints.items()
            }

    @classmethod
    def _rebalance_hint_for_model(cls, model_id: str) -> dict[str, Any] | None:
        key = model_key(model_id)
        now = time.time()
        with cls._rebalance_lock:
            item = cls._rebalance_hints.get(key)
            if item is None:
                return None
            if now >= float(item.get("expires_at_unix", 0.0)):
                cls._rebalance_hints.pop(key, None)
                return None
            return {
                "active": True,
                "model_id": model_id,
                "lookup_key": key,
                "reason": "hot_key_rate_limit",
                "recommended_dsht_replicas": int(item.get("recommended_dsht_replicas", 0)),
                "dsht_replica_indices": list(item.get("dsht_replica_indices", [])),
                "triggered_unix_ms": int(item.get("triggered_unix_ms", 0)),
            }

    def _trigger_dsht_rebalance(self, model_id: str) -> dict[str, Any] | None:
        dht = self._require_dht()
        key = model_key(model_id)
        now = time.time()
        window = max(1, int(self.default_lookup_window_seconds))
        base_replicas = max(1, int(self.default_dsht_replicas))
        max_replicas = max(base_replicas, int(self.default_dsht_max_replicas))
        with self._rebalance_lock:
            existing = self._rebalance_hints.get(key)
            if existing is not None and now < float(existing.get("expires_at_unix", 0.0)):
                return self._rebalance_hint_for_model(model_id)

        start = base_replicas
        room = max(0, max_replicas - start)
        if room <= 0:
            return None
        shift = max(1, min(base_replicas, room))
        replica_indices = list(range(start, start + shift))

        peers = dht.get(model_key(model_id))
        for replica_index in range(base_replicas):
            peers.extend(dht.get(dsht_key(model_id, replica_index)))
        peers = _dedupe_peers(peers)
        ttl = max(5, int(self.default_ttl_seconds))
        for replica_index in replica_indices:
            replica_key = dsht_key(model_id, replica_index)
            for peer in peers:
                dht.put(replica_key, peer, unique_field="peer_id", ttl_seconds=ttl)

        hint = {
            "recommended_dsht_replicas": int(replica_indices[-1] + 1),
            "dsht_replica_indices": replica_indices,
            "triggered_unix_ms": int(now * 1000),
            "expires_at_unix": (now + max(1, window * 2)),
        }
        with self._rebalance_lock:
            self._rebalance_hints[key] = hint
        return self._rebalance_hint_for_model(model_id)

    def _allow_lookup(self, model_id: str) -> tuple[bool, int, dict[str, Any] | None]:
        now = time.time()
        key = model_key(model_id)
        window = max(1, int(self.default_lookup_window_seconds))
        limit = max(1, int(self.default_lookup_max_requests_per_window))
        with self._lookup_lock:
            item = self._lookup_buckets.get(key)
            if item is None or (now - float(item.get("window_start", 0.0))) >= window:
                self._lookup_buckets[key] = {"window_start": now, "count": 1.0}
                return True, 0, self._rebalance_hint_for_model(model_id)

            count = int(item.get("count", 0))
            if count >= limit:
                retry_after = max(1, int(window - (now - float(item.get("window_start", 0.0)))))
                hint = self._trigger_dsht_rebalance(model_id)
                return False, retry_after, hint

            item["count"] = float(count + 1)
            return True, 0, self._rebalance_hint_for_model(model_id)

    def do_POST(self) -> None:
        try:
            body = self._read_json()
            if self.path in {"/announce", "/heartbeat"}:
                record = self._put_announcement(body)
                self._send_json({"ok": True, "peer_id": record["peer_id"], "model_id": record["model_id"]})
                return
            if self.path == "/rebalance":
                self._handle_post_rebalance(body)
                return
            self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
        except json.JSONDecodeError:
            self._send_json({"error": "invalid_json"}, status=HTTPStatus.BAD_REQUEST)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            dht = self._require_dht()

            if parsed.path == "/lookup":
                query = parse_qs(parsed.query)
                model_id = (query.get("model_id") or [""])[0].strip()
                if not model_id:
                    self._send_json({"error": "missing_model_id"}, status=HTTPStatus.BAD_REQUEST)
                    return
                allowed, retry_after, rebalance_hint = self._allow_lookup(model_id)
                if not allowed:
                    self._send_json(
                        {
                            "error": "lookup_rate_limited",
                            "model_id": model_id,
                            "retry_after_seconds": retry_after,
                            "rebalance": (
                                rebalance_hint
                                if rebalance_hint is not None
                                else {"active": False}
                            ),
                        },
                        status=HTTPStatus.TOO_MANY_REQUESTS,
                        headers={"Retry-After": str(retry_after)},
                    )
                    return
                preferred_region = (query.get("preferred_region") or [None])[0]
                preferred_region = str(preferred_region).strip() if preferred_region else None
                limit_raw = (query.get("limit") or [None])[0]
                sloppy_raw = (query.get("sloppy_factor") or [None])[0]
                diversify_raw = (query.get("diversify") or ["1"])[0]
                try:
                    limit = int(limit_raw) if limit_raw not in (None, "") else None
                    if limit is not None and limit <= 0:
                        limit = None
                except ValueError:
                    self._send_json({"error": "invalid_limit"}, status=HTTPStatus.BAD_REQUEST)
                    return
                dsht_raw = (query.get("dsht_replicas") or [None])[0]
                try:
                    sloppy_factor = int(sloppy_raw) if sloppy_raw not in (None, "") else 3
                    sloppy_factor = max(0, sloppy_factor)
                except ValueError:
                    self._send_json({"error": "invalid_sloppy_factor"}, status=HTTPStatus.BAD_REQUEST)
                    return
                try:
                    dsht_replicas = int(dsht_raw) if dsht_raw not in (None, "") else self.default_dsht_replicas
                    dsht_replicas = max(0, min(int(self.default_dsht_max_replicas), dsht_replicas))
                except ValueError:
                    self._send_json({"error": "invalid_dsht_replicas"}, status=HTTPStatus.BAD_REQUEST)
                    return
                diversify = str(diversify_raw).strip().lower() not in {"0", "false", "no"}
                hint_replicas = int((rebalance_hint or {}).get("recommended_dsht_replicas", 0))
                effective_dsht_replicas = max(dsht_replicas, hint_replicas)
                effective_dsht_replicas = min(int(self.default_dsht_max_replicas), effective_dsht_replicas)

                peers = dht.get(model_key(model_id))
                for replica_index in range(effective_dsht_replicas):
                    peers.extend(dht.get(dsht_key(model_id, replica_index)))
                peers = _dedupe_peers(peers)
                peers = _select_lookup_peers(
                    peers,
                    preferred_region=preferred_region,
                    limit=limit,
                    sloppy_factor=sloppy_factor,
                    diversify=diversify,
                )
                self._send_json(
                    {
                        "model_id": model_id,
                        "peers": peers,
                        "count": len(peers),
                        "rebalance": (
                            rebalance_hint
                            if rebalance_hint is not None
                            else {"active": False}
                        ),
                    }
                )
                return

            if parsed.path == "/rebalance":
                query = parse_qs(parsed.query)
                peer_id = (query.get("peer_id") or [""])[0].strip()
                if not peer_id:
                    self._send_json({"error": "missing_peer_id"}, status=HTTPStatus.BAD_REQUEST)
                    return
                directives = self._get_layer_rebalance_directives(peer_id)
                self._send_json({"peer_id": peer_id, "directives": directives})
                return

            if parsed.path == "/health":
                self._send_json(
                    {
                        "ok": True,
                        "keys": dht.keys(),
                        "stats": dht.stats(),
                        "lookup_rate_limit": {
                            "window_seconds": int(self.default_lookup_window_seconds),
                            "max_requests_per_window": int(self.default_lookup_max_requests_per_window),
                            "active_windows": self._lookup_rate_snapshot(),
                        },
                        "geo_challenge": {
                            "enabled": bool(self.default_geo_challenge_enabled),
                            "timeout_ms": int(self.default_geo_challenge_timeout_ms),
                            "max_rtt_ms": float(self.default_geo_max_rtt_ms),
                        },
                        "expert_admission": {
                            "min_reputation_score": float(self.default_expert_min_reputation_score),
                            "min_staked_balance": float(self.default_expert_min_staked_balance),
                            "require_stake": bool(self.default_expert_require_stake),
                        },
                        "dsht_rebalance": self._lookup_rebalance_snapshot(),
                    }
                )
                return

            self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
        except RuntimeError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)


    def _handle_post_rebalance(self, body: dict[str, Any]) -> None:
        """Store a layer rebalance directive for a target peer."""
        target_peer_id = str(body.get("target_peer_id", "")).strip()
        if not target_peer_id:
            self._send_json({"error": "missing_target_peer_id"}, status=HTTPStatus.BAD_REQUEST)
            return
        now_ms = int(time.time() * 1000)
        expires_unix_ms = int(body.get("expires_unix_ms", 0))
        if expires_unix_ms <= 0:
            # Default: 120s from now.
            expires_unix_ms = now_ms + 120_000
        if expires_unix_ms <= now_ms:
            self._send_json({"error": "directive_already_expired"}, status=HTTPStatus.BAD_REQUEST)
            return
        directive = {
            "target_peer_id": target_peer_id,
            "new_layer_start": int(body.get("new_layer_start", 0)),
            "new_layer_end": int(body.get("new_layer_end", 0)),
            "total_layers": int(body.get("total_layers", 0)),
            "reason": str(body.get("reason", "gap_fill")),
            "issued_unix_ms": int(body.get("issued_unix_ms", now_ms)),
            "expires_unix_ms": expires_unix_ms,
        }
        with self.__class__._layer_rebalance_lock:
            existing = self.__class__._layer_rebalance_directives.setdefault(target_peer_id, [])
            existing.append(directive)
            # Cap stored directives per peer to prevent unbounded growth.
            if len(existing) > 16:
                self.__class__._layer_rebalance_directives[target_peer_id] = existing[-16:]
        self._send_json({"ok": True, "target_peer_id": target_peer_id})

    @classmethod
    def _get_layer_rebalance_directives(cls, peer_id: str) -> list[dict[str, Any]]:
        """Return non-expired directives for *peer_id* and prune expired ones."""
        now_ms = int(time.time() * 1000)
        with cls._layer_rebalance_lock:
            raw = cls._layer_rebalance_directives.get(peer_id, [])
            valid = [d for d in raw if int(d.get("expires_unix_ms", 0)) > now_ms]
            if len(valid) != len(raw):
                if valid:
                    cls._layer_rebalance_directives[peer_id] = valid
                else:
                    cls._layer_rebalance_directives.pop(peer_id, None)
            return list(valid)


def _peer_sort_key(peer: dict[str, Any]) -> tuple[float, float, float]:
    load = float(peer.get("load_pct", 0.0))
    bandwidth = float(peer.get("bandwidth_mbps", 0.0))
    updated = float(peer.get("updated_unix_ms", 0))
    return (load, -bandwidth, -updated)


def _dedupe_peers(peers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for peer in peers:
        peer_id = str(peer.get("peer_id", "")).strip()
        if not peer_id:
            continue
        deduped[peer_id] = dict(peer)
    return list(deduped.values())


def _diversify_by_operator(peers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    queues: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for peer in peers:
        operator = str(peer.get("operator_id") or peer.get("peer_id") or "")
        if operator not in queues:
            queues[operator] = []
            order.append(operator)
        queues[operator].append(peer)

    out: list[dict[str, Any]] = []
    while True:
        progressed = False
        for operator in order:
            queue = queues.get(operator) or []
            if not queue:
                continue
            out.append(queue.pop(0))
            progressed = True
        if not progressed:
            break
    return out


def _select_lookup_peers(
    peers: list[dict[str, Any]],
    *,
    preferred_region: str | None,
    limit: int | None,
    sloppy_factor: int,
    diversify: bool,
) -> list[dict[str, Any]]:
    if not peers:
        return []

    ordered = sorted(peers, key=_peer_sort_key)
    if diversify:
        ordered = _diversify_by_operator(ordered)

    if not preferred_region:
        if limit is None:
            return ordered
        return ordered[:limit]

    norm_region = preferred_region.strip().lower()
    in_region = [peer for peer in ordered if str(peer.get("region") or "").strip().lower() == norm_region]
    out_region = [peer for peer in ordered if peer not in in_region]

    if limit is None:
        return in_region + out_region

    selected: list[dict[str, Any]] = []
    selected.extend(in_region[:limit])
    if len(selected) < limit:
        relay_budget = max(0, sloppy_factor)
        if relay_budget > 0:
            selected.extend(out_region[: min(relay_budget, limit - len(selected))])
        if len(selected) < limit:
            tail = [peer for peer in out_region if peer not in selected]
            selected.extend(tail[: limit - len(selected)])
    return selected[:limit]


def _exponential_backoff_delay(
    attempt: int,
    *,
    base_seconds: float = 1.0,
    cap_seconds: float = 60.0,
) -> float:
    clamped = max(0, min(10, int(attempt)))
    return min(float(cap_seconds), float(base_seconds) * (2.0 ** clamped))


def serve(
    host: str,
    port: int,
    ttl_seconds: int = 300,
    dsht_replicas: int = 2,
    dsht_max_replicas: int = 32,
    lookup_rate_limit_window_sec: int = 1,
    lookup_rate_limit_max_requests: int = 120,
    geo_challenge_enabled: bool = True,
    geo_challenge_timeout_ms: int = 800,
    geo_max_rtt_ms: float = 50.0,
    geo_challenge_seed: str = "openhydra-geo-dev-seed",
    expert_min_reputation_score: float = 60.0,
    expert_min_staked_balance: float = 0.01,
    expert_require_stake: bool = True,
) -> None:
    dht_node = InMemoryDhtNode(ttl_seconds=ttl_seconds)
    dht_node.start_background_pruner(interval_s=30.0)
    DhtBootstrapHandler.dht = dht_node
    DhtBootstrapHandler.default_ttl_seconds = ttl_seconds
    DhtBootstrapHandler.default_dsht_max_replicas = max(1, min(128, int(dsht_max_replicas)))
    DhtBootstrapHandler.default_dsht_replicas = max(0, min(DhtBootstrapHandler.default_dsht_max_replicas, int(dsht_replicas)))
    DhtBootstrapHandler.default_lookup_window_seconds = max(1, int(lookup_rate_limit_window_sec))
    DhtBootstrapHandler.default_lookup_max_requests_per_window = max(1, int(lookup_rate_limit_max_requests))
    DhtBootstrapHandler.default_geo_challenge_enabled = bool(geo_challenge_enabled)
    DhtBootstrapHandler.default_geo_challenge_timeout_ms = max(50, int(geo_challenge_timeout_ms))
    DhtBootstrapHandler.default_geo_max_rtt_ms = max(1.0, float(geo_max_rtt_ms))
    DhtBootstrapHandler.default_geo_challenge_seed = str(geo_challenge_seed)
    DhtBootstrapHandler.default_expert_min_reputation_score = max(0.0, float(expert_min_reputation_score))
    DhtBootstrapHandler.default_expert_min_staked_balance = max(0.0, float(expert_min_staked_balance))
    DhtBootstrapHandler.default_expert_require_stake = bool(expert_require_stake)
    DhtBootstrapHandler._lookup_buckets = {}
    DhtBootstrapHandler._rebalance_hints = {}
    DhtBootstrapHandler._layer_rebalance_directives = {}
    bind_attempt = 0
    lifecycle_restart_attempt = 0
    _stop = threading.Event()
    _active_server: list[ThreadingHTTPServer] = []  # single-element mutable cell

    def _on_sigterm(signum: int, _frame: object) -> None:
        logging.info("shutdown_requested signal=%s", signal.Signals(signum).name)
        _stop.set()
        if _active_server:
            threading.Thread(
                target=_active_server[0].shutdown, daemon=True, name="sigterm-shutdown"
            ).start()

    signal.signal(signal.SIGTERM, _on_sigterm)

    try:
        while not _stop.is_set():
            server: ThreadingHTTPServer | None = None
            try:
                server = ThreadingHTTPServer((host, port), DhtBootstrapHandler)
                _active_server[:] = [server]
                bind_attempt = 0
                logging.info("DHT bootstrap listening on http://%s:%s", host, port)
                server.serve_forever()
                _active_server.clear()
                if _stop.is_set():
                    break
                # Unexpected server stop: keep daemon alive with exponential restart backoff.
                lifecycle_restart_attempt += 1
                delay_s = _exponential_backoff_delay(
                    lifecycle_restart_attempt - 1,
                    base_seconds=1.0,
                    cap_seconds=120.0,
                )
                logging.warning(
                    "DHT bootstrap stopped unexpectedly; restarting in %.1fs",
                    delay_s,
                )
                time.sleep(delay_s)
            except KeyboardInterrupt:
                break
            except OSError as exc:
                bind_attempt += 1
                delay_s = _exponential_backoff_delay(bind_attempt - 1, base_seconds=1.0, cap_seconds=60.0)
                logging.warning("DHT bootstrap bind failure (%s); retrying in %.1fs", exc, delay_s)
                time.sleep(delay_s)
            except Exception as exc:
                bind_attempt += 1
                delay_s = _exponential_backoff_delay(bind_attempt - 1, base_seconds=1.0, cap_seconds=60.0)
                logging.warning("DHT bootstrap transient failure (%s); retrying in %.1fs", exc, delay_s)
                time.sleep(delay_s)
            finally:
                if server is not None:
                    server.server_close()
    finally:
        dht_node.stop_background_pruner()
        logging.info("shutdown_complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenHydra Tier-2 DHT bootstrap service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8468)
    parser.add_argument("--deployment-profile", choices=["dev", "prod"], default="dev")
    parser.add_argument("--secrets-file", default=None, help="Path to KEY=VALUE secrets file (0600 permissions required)")
    parser.add_argument("--ttl-seconds", type=int, default=300)
    parser.add_argument("--dsht-replicas", type=int, default=2)
    parser.add_argument("--dsht-max-replicas", type=int, default=32)
    parser.add_argument("--lookup-rate-limit-window-sec", type=int, default=1)
    parser.add_argument("--lookup-rate-limit-max-requests", type=int, default=120)
    parser.add_argument("--geo-challenge-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--geo-challenge-timeout-ms", type=int, default=800)
    parser.add_argument("--geo-max-rtt-ms", type=float, default=50.0)
    parser.add_argument("--geo-challenge-seed", default="openhydra-geo-dev-seed")
    parser.add_argument("--expert-min-reputation-score", type=float, default=60.0)
    parser.add_argument("--expert-min-staked-balance", type=float, default=0.01)
    parser.add_argument("--expert-require-stake", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dht-backend",
        choices=["memory", "hivemind"],
        default="memory",
        help=(
            "DHT storage backend. 'memory' (default) uses the built-in "
            "InMemoryDhtNode. 'hivemind' starts a Hivemind Kademlia DHT "
            "signpost alongside the HTTP server for dual-stack operation."
        ),
    )
    parser.add_argument(
        "--hivemind-port",
        type=int,
        default=38751,
        help="TCP port for Hivemind Kademlia signpost (default: 38751). "
             "Only used when --dht-backend=hivemind.",
    )
    args = parser.parse_args()
    profile_settings = _resolve_bootstrap_profile_settings(parser, args)
    configure_logging(json_logs=str(profile_settings.get("deployment_profile", "dev")) == "prod")

    # Optionally start Hivemind signpost alongside HTTP bootstrap.
    _signpost_dht = None
    if str(getattr(args, "dht_backend", "memory")) == "hivemind":
        try:
            from dht.signpost import serve as _signpost_serve
            import threading as _threading
            _hivemind_port = max(1, int(getattr(args, "hivemind_port", 38751)))
            logging.info("dht_backend=hivemind: starting signpost on port %d", _hivemind_port)
            _signpost_thread = _threading.Thread(
                target=_signpost_serve,
                kwargs={"host": args.host, "port": _hivemind_port},
                daemon=True,
                name="hivemind-signpost",
            )
            _signpost_thread.start()
        except ImportError:
            logging.warning("dht_backend=hivemind: hivemind not installed, falling back to memory")

    serve(
        host=args.host,
        port=args.port,
        ttl_seconds=max(5, args.ttl_seconds),
        dsht_replicas=max(0, int(args.dsht_replicas)),
        dsht_max_replicas=max(1, int(args.dsht_max_replicas)),
        lookup_rate_limit_window_sec=max(1, int(args.lookup_rate_limit_window_sec)),
        lookup_rate_limit_max_requests=max(1, int(args.lookup_rate_limit_max_requests)),
        geo_challenge_enabled=bool(args.geo_challenge_enabled),
        geo_challenge_timeout_ms=max(50, int(args.geo_challenge_timeout_ms)),
        geo_max_rtt_ms=max(1.0, float(args.geo_max_rtt_ms)),
        geo_challenge_seed=str(profile_settings["geo_challenge_seed"]),
        expert_min_reputation_score=max(0.0, float(args.expert_min_reputation_score)),
        expert_min_staked_balance=max(0.0, float(args.expert_min_staked_balance)),
        expert_require_stake=bool(args.expert_require_stake),
    )


if __name__ == "__main__":
    main()
