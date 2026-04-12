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

"""Discovery service — extracted from CoordinatorEngine.

Handles peer discovery, DHT lookups, model catalog queries, and the
full ``_discover_for_model`` orchestration that resolves a requested model
to a ranked list of healthy candidate peers with degradation fallback.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from openhydra_defaults import PRODUCTION_BOOTSTRAP_URLS

from coordinator.degradation import DegradationDecision, DegradationPolicy, ModelAvailability
from coordinator.path_finder import (
    PathFinder,
    PeerEndpoint,
    load_peer_config,
    load_peers_from_dht,
)
from coordinator.peer_selector import ScoredPeer, rank_peers
from coordinator.transport import TransportConfig

logger = logging.getLogger(__name__)


class DiscoveryService:
    """Owns peer discovery, DHT cache, model catalog queries, and network scanning.

    Parameters
    ----------
    config:
        An ``EngineConfig`` (or duck-typed equivalent).
    health:
        A ``HealthScorer`` instance for recording ping/inference outcomes and
        querying reputation scores.
    auto_scaler:
        An ``AutoScaler | None`` for demand-driven roster management.
    _dht_peer_cache:
        Shared mutable dict for DHT peer caching (keyed by model_id).
    _dht_peer_cache_lock:
        A ``threading.Lock`` protecting ``_dht_peer_cache``.
    _active_model_roster:
        Shared mutable list of currently active model IDs.
    _request_log:
        A ``RequestLog`` for recording demand signals.
    model_catalog:
        The loaded list of ``ModelAvailability`` items.
    catalogue_by_model:
        Dict mapping model_id -> ``ModelAvailability``.
    degradation_policy:
        A ``DegradationPolicy`` for degradation decisions.
    replication_monitor:
        A ``ReplicationMonitor`` for replication status.
    transport_config:
        A ``TransportConfig`` for gRPC transport settings.
    ledger_bridge:
        An ``OpenHydraLedgerBridge`` for stake queries.
    role_thresholds:
        A ``RoleThresholds`` for bandwidth role classification.
    """

    def __init__(
        self,
        config: Any,
        health: Any,
        auto_scaler: Any,
        _dht_peer_cache: dict[str, dict[str, Any]],
        _dht_peer_cache_lock: threading.Lock,
        _active_model_roster: list[str],
        _request_log: Any,
        model_catalog: list[ModelAvailability],
        catalogue_by_model: dict[str, ModelAvailability],
        degradation_policy: DegradationPolicy,
        replication_monitor: Any,
        transport_config: TransportConfig,
        ledger_bridge: Any,
        role_thresholds: Any,
        _metrics_lock: threading.Lock,
        _dht_lookup_attempts: int = 0,
        _dht_lookup_successes: int = 0,
        _dht_lookup_failures: int = 0,
        engine: Any = None,
        p2p_node: Any = None,
    ) -> None:
        self._engine = engine
        self._p2p_node = p2p_node
        self.config = config
        self.health = health
        self.auto_scaler = auto_scaler
        self._dht_peer_cache = _dht_peer_cache
        self._dht_peer_cache_lock = _dht_peer_cache_lock
        self._active_model_roster = _active_model_roster
        self._request_log = _request_log
        self.model_catalog = model_catalog
        self.catalogue_by_model = catalogue_by_model
        self.degradation_policy = degradation_policy
        self.replication_monitor = replication_monitor
        self.transport_config = transport_config
        self.ledger_bridge = ledger_bridge
        self.role_thresholds = role_thresholds
        self._metrics_lock = _metrics_lock
        self._dht_lookup_attempts = _dht_lookup_attempts
        self._dht_lookup_successes = _dht_lookup_successes
        self._dht_lookup_failures = _dht_lookup_failures
        self._last_scored_peers: list[ScoredPeer] = []
        self._last_verification_qos: dict[str, Any] = {
            "enabled": False,
            "min_events": 0,
            "min_success_rate": 0.0,
            "requested_model_blocked": False,
            "requested_model_events": 0,
            "requested_model_success_rate": None,
        }

    # ------------------------------------------------------------------
    # Model catalog
    # ------------------------------------------------------------------

    def _load_model_catalog(self) -> list[ModelAvailability]:
        """Load the model catalog from a JSON file or create a single-model default.

        Returns:
            List of ``ModelAvailability`` entries.

        Raises:
            RuntimeError: If the catalog file is missing or contains no valid entries.
        """
        if not self.config.model_catalog_path:
            return [ModelAvailability(model_id=self.config.default_model, required_peers=self.config.required_replicas)]

        path = Path(self.config.model_catalog_path)
        if not path.exists():
            raise RuntimeError(f"model_catalog_not_found: {path}")

        raw = json.loads(path.read_text())
        if not isinstance(raw, list) or not raw:
            raise RuntimeError("invalid_model_catalog: expected non-empty JSON list")

        seen: set[str] = set()
        catalogue: list[ModelAvailability] = []
        for entry in raw:
            model_id = str(entry.get("model_id", "")).strip()
            required = int(entry.get("required_peers", self.config.required_replicas))
            hf_model_id = str(entry.get("hf_model_id", "")).strip()
            if not model_id or model_id in seen:
                continue
            seen.add(model_id)
            catalogue.append(
                ModelAvailability(
                    model_id=model_id,
                    required_peers=max(1, required),
                    hf_model_id=hf_model_id,
                    min_vram_gb=max(0, int(entry.get("min_vram_gb", 0))),
                    recommended_quantization=str(entry.get("recommended_quantization", "fp32")),
                    context_length=max(0, int(entry.get("context_length", 4096))),
                    languages=tuple(str(x) for x in entry.get("languages", [])),
                    tags=tuple(str(x) for x in entry.get("tags", [])),
                    description=str(entry.get("description", "")),
                    # Phase 2: auto-scaler capability fields
                    shard_vram_gb=max(0.0, float(entry.get("shard_vram_gb", 0))),
                    shards_needed=max(1, int(entry.get("shards_needed", required))),
                    quality_tier=str(entry.get("quality_tier", "standard")),
                )
            )

        if not catalogue:
            raise RuntimeError("invalid_model_catalog: no valid entries")

        if self.config.default_model not in seen:
            default_hf_model_id = (
                self.config.default_model
                if "/" in str(self.config.default_model)
                else ""
            )
            catalogue.append(
                ModelAvailability(
                    model_id=self.config.default_model,
                    required_peers=self.config.required_replicas,
                    hf_model_id=default_hf_model_id,
                )
            )

        return catalogue

    def _catalog_hf_model_id(self, model_id: str) -> str | None:
        """Look up the HuggingFace model ID for a catalog entry.

        Args:
            model_id: The user-facing model identifier.

        Returns:
            The HuggingFace model ID string, or ``None`` if not found.
        """
        key = str(model_id or "").strip()
        if not key:
            return None
        item = self.catalogue_by_model.get(key)
        if item is None:
            return None
        value = str(getattr(item, "hf_model_id", "") or "").strip()
        return value or None

    def _catalog_model_ids(self) -> list[str]:
        """Return all model IDs in the catalog."""
        return [item.model_id for item in self.model_catalog]

    def _required_replicas(self, model_id: str) -> int:
        """Return the required replica count for a model, falling back to config default."""
        item = self.catalogue_by_model.get(model_id)
        if item is None:
            return self.config.required_replicas
        return item.required_peers

    def _normalize_peer_model(self, peer: PeerEndpoint) -> str:
        """Return the peer's model ID, falling back to the default model."""
        return peer.model_id or self.config.default_model

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _dedupe_peer_entries(self, peers: list[PeerEndpoint]) -> list[PeerEndpoint]:
        """Deduplicate peer entries by ``(peer_id, model_id)`` key.

        When two entries exist for the same key, merge layer info from the
        more-complete entry (typically the static config) into the survivor.
        This prevents DHT peers (which may lack layer_start/layer_end) from
        overwriting sharding metadata loaded from the peers config file.

        Args:
            peers: Raw list of peer endpoints (may contain duplicates).

        Returns:
            Deduplicated list of peer endpoints.
        """
        deduped: dict[tuple[str, str], PeerEndpoint] = {}
        for peer in peers:
            model_id = self._normalize_peer_model(peer)
            key = (peer.peer_id, model_id)
            existing = deduped.get(key)
            new_peer = peer.replace(model_id=model_id)
            if existing is not None:
                # Merge: preserve layer info from whichever entry has it.
                e_has_layers = int(existing.total_layers) > 0 and int(existing.layer_end) > 0
                n_has_layers = int(new_peer.total_layers) > 0 and int(new_peer.layer_end) > 0
                if e_has_layers and not n_has_layers:
                    # Existing has layer info, new doesn't — keep existing.
                    continue
                if n_has_layers and not e_has_layers:
                    # New has layer info, existing doesn't — use new.
                    deduped[key] = new_peer
                    continue
                # Both have (or both lack) layer info — prefer the newer entry
                # but carry forward layer info from existing if new lacks it.
                if not n_has_layers and e_has_layers:
                    new_peer = new_peer.replace(
                        layer_start=existing.layer_start,
                        layer_end=existing.layer_end,
                        total_layers=existing.total_layers,
                    )
            deduped[key] = new_peer
        return list(deduped.values())

    # ------------------------------------------------------------------
    # DHT URL resolution
    # ------------------------------------------------------------------

    def _configured_dht_urls(self) -> list[str]:
        """Return the list of DHT bootstrap URLs to query.

        Merges ``config.dht_urls`` and ``config.dht_url``, falling back to
        the production bootstrap nodes when no URLs are configured.

        Returns:
            Ordered, deduplicated list of DHT URL strings.
        """
        sources: list[str] = []
        seen: set[str] = set()
        for item in list(getattr(self.config, "dht_urls", [])):
            value = str(item).strip()
            if not value or value in seen:
                continue
            seen.add(value)
            sources.append(value)
        fallback = str(getattr(self.config, "dht_url", "") or "").strip()
        if fallback and fallback not in seen:
            sources.append(fallback)
        # When no URLs are explicitly configured, use the production bootstrap
        # nodes so that coordinators work out-of-the-box without any flags.
        if not sources:
            sources = list(PRODUCTION_BOOTSTRAP_URLS)
        return sources

    # ------------------------------------------------------------------
    # Peer loading
    # ------------------------------------------------------------------

    def _load_candidate_peers(self, model_ids: list[str] | None = None) -> list[PeerEndpoint]:
        """Load candidate peers from config file and/or DHT bootstrap nodes.

        Queries each configured DHT URL for every model in the filter set,
        caching results and falling back to the cache on errors.  Results
        are deduplicated before returning.

        Args:
            model_ids: Optional list of model IDs to filter for.  Defaults to
                all catalog model IDs.

        Returns:
            Deduplicated list of discovered peer endpoints.

        Raises:
            RuntimeError: If no peers are found from any source.
        """
        peers: list[PeerEndpoint] = []
        model_filter = set(model_ids or self._catalog_model_ids())

        # Resolve load_peers_from_dht through the engine module so that
        # monkeypatching coordinator.engine.load_peers_from_dht works.
        import coordinator.engine as _engine_mod
        _load_peers_from_dht = getattr(_engine_mod, "load_peers_from_dht", load_peers_from_dht)

        if self.config.peers_config_path:
            path = Path(self.config.peers_config_path)
            if not path.exists():
                raise RuntimeError(f"peer_config_not_found: {path}")
            for peer in load_peer_config(path):
                model_id = peer.model_id or self.config.default_model
                runtime_id = peer.runtime_model_id or ""
                if model_id in model_filter or (runtime_id and runtime_id in model_filter):
                    peers.append(peer.replace(model_id=model_id))

        dht_sources = self._configured_dht_urls()
        if dht_sources:
            dht_errors: list[Exception] = []
            for model_id in model_filter:
                with self._metrics_lock:
                    self._dht_lookup_attempts += 1
                try:
                    if len(dht_sources) == 1:
                        dht_peers = _load_peers_from_dht(
                            dht_sources[0],
                            model_id=model_id,
                            timeout_s=self.config.dht_lookup_timeout_s,
                            preferred_region=self.config.dht_preferred_region,
                            limit=(self.config.dht_lookup_limit if self.config.dht_lookup_limit > 0 else None),
                            sloppy_factor=max(0, int(self.config.dht_lookup_sloppy_factor)),
                            dsht_replicas=max(0, int(self.config.dht_lookup_dsht_replicas)),
                        )
                    else:
                        dht_peers = _load_peers_from_dht(
                            model_id=model_id,
                            timeout_s=self.config.dht_lookup_timeout_s,
                            preferred_region=self.config.dht_preferred_region,
                            limit=(self.config.dht_lookup_limit if self.config.dht_lookup_limit > 0 else None),
                            sloppy_factor=max(0, int(self.config.dht_lookup_sloppy_factor)),
                            dsht_replicas=max(0, int(self.config.dht_lookup_dsht_replicas)),
                            dht_urls=dht_sources,
                        )
                    if dht_peers:
                        peers.extend(dht_peers)
                        self._cache_dht_peers(model_id=model_id, peers=dht_peers)
                    else:
                        peers.extend(self._cached_dht_peers(model_id=model_id))
                    with self._metrics_lock:
                        self._dht_lookup_successes += 1
                except Exception as exc:  # pragma: no cover
                    dht_errors.append(exc)
                    peers.extend(self._cached_dht_peers(model_id=model_id))
                    with self._metrics_lock:
                        self._dht_lookup_failures += 1

            if dht_errors and not peers:
                latest = dht_errors[-1]
                raise RuntimeError(f"dht_lookup_failed: {latest}") from latest

        # Rust Kademlia DHT discover (libp2p path).
        if self._p2p_node is not None:
            for model_id in model_filter:
                try:
                    libp2p_peers = self._p2p_node.discover(model_id=model_id)
                    for p in (libp2p_peers or []):
                        peers.append(PeerEndpoint(
                            peer_id=str(p.get("peer_id", "")),
                            host=str(p.get("host", "")),
                            port=int(p.get("port", 0)),
                            model_id=str(p.get("model_id", model_id)),
                            layer_start=int(p.get("layer_start", 0)),
                            layer_end=int(p.get("layer_end", 0)),
                            total_layers=int(p.get("total_layers", 0)),
                            nat_type=str(p.get("nat_type", "unknown")),
                            requires_relay=bool(p.get("requires_relay", False)),
                            relay_address=str(p.get("relay_address", "")),
                            runtime_backend=str(p.get("runtime_backend", "")),
                            runtime_model_id=str(p.get("runtime_model_id", "")),
                        ))
                except Exception as p2p_exc:
                    logger.debug("p2p_discover_error for %s: %s", model_id, p2p_exc)

        peers = self._dedupe_peer_entries(peers)
        if not peers:
            raise RuntimeError("no_peers_from_sources")
        return peers

    # ------------------------------------------------------------------
    # DHT cache
    # ------------------------------------------------------------------

    def _cache_dht_peers(self, *, model_id: str, peers: list[PeerEndpoint]) -> None:
        """Store a DHT peer list in the in-memory cache with a TTL.

        Args:
            model_id: The model ID to cache peers under.
            peers: The peer endpoints to cache.
        """
        normalized_model = str(model_id or "").strip()
        if not normalized_model or not peers:
            return
        ttl_s = max(1.0, float(self.config.dht_lookup_cache_ttl_s))
        expires_at = time.time() + ttl_s
        snapshot = [peer for peer in peers]
        with self._dht_peer_cache_lock:
            self._dht_peer_cache[normalized_model] = {
                "expires_at": float(expires_at),
                "peers": snapshot,
            }

    def _cached_dht_peers(self, *, model_id: str) -> list[PeerEndpoint]:
        """Retrieve cached DHT peers for a model, returning empty on miss/expiry.

        Args:
            model_id: The model ID to look up.

        Returns:
            List of cached peer endpoints, or empty list if expired or absent.
        """
        normalized_model = str(model_id or "").strip()
        if not normalized_model:
            return []
        now = time.time()
        with self._dht_peer_cache_lock:
            item = self._dht_peer_cache.get(normalized_model)
            if not item:
                return []
            expires_at = float(item.get("expires_at", 0.0))
            if now >= expires_at:
                self._dht_peer_cache.pop(normalized_model, None)
                return []
            raw_peers = list(item.get("peers", []))
        out: list[PeerEndpoint] = []
        for peer in raw_peers:
            if isinstance(peer, PeerEndpoint):
                out.append(peer)
        return out

    # ------------------------------------------------------------------
    # Network scanning
    # ------------------------------------------------------------------

    def _scan_network(self, model_ids: list[str] | None = None):
        """Load peers, ping-survey them, and return healthy peers with counts.

        Args:
            model_ids: Optional model ID filter for peer loading.

        Returns:
            Tuple of (healthy_items, available_peer_counts_by_model).
        """
        peers = self._load_candidate_peers(model_ids=model_ids)
        finder = PathFinder(
            timeout_ms=min(self.config.timeout_ms, 1200),
            transport_config=self.transport_config,
        )

        survey = finder.survey(peers)
        self._record_ping_health(survey)

        healthy = [h for h in survey if h.healthy and h.latency_ms <= self.config.max_latency_ms]
        available_peer_counts: dict[str, int] = {}
        for item in healthy:
            model_id = self._normalize_peer_model(item.peer)
            available_peer_counts[model_id] = available_peer_counts.get(model_id, 0) + 1

        return healthy, available_peer_counts

    def _record_ping_health(self, survey) -> None:
        """Record ping health outcomes for all peers in a survey result."""
        for item in survey:
            self.health.record_ping(item.peer.peer_id, healthy=item.healthy, latency_ms=item.latency_ms)

    # ------------------------------------------------------------------
    # Verification helpers
    # ------------------------------------------------------------------

    def _verification_feedback_by_model(self, health) -> dict[str, dict[str, Any]]:
        """Aggregate verification metrics for every model in the catalog."""
        out: dict[str, dict[str, Any]] = {}
        for model in self.model_catalog:
            out[model.model_id] = self._verification_metrics_for_model(model.model_id, health)
        return out

    def _verification_metrics_for_model(self, model_id: str, health) -> dict[str, Any]:
        health_snapshot = self.health.snapshot()
        model_peers = [item.peer for item in health if self._normalize_peer_model(item.peer) == model_id]
        verification_ok = 0
        verification_failed = 0
        verified_peers = 0
        peers_with_failed = 0
        for peer in model_peers:
            stats = dict(health_snapshot.get(peer.peer_id) or {})
            ok = int(stats.get("verifications_ok", 0))
            failed = int(stats.get("verifications_failed", 0))
            if ok > 0 or failed > 0:
                verified_peers += 1
            if failed > 0:
                peers_with_failed += 1
            verification_ok += ok
            verification_failed += failed

        verification_total = verification_ok + verification_failed
        return {
            "verified_peers": verified_peers,
            "peers_with_failed_verifications": peers_with_failed,
            "total_verifications_ok": verification_ok,
            "total_verifications_failed": verification_failed,
            "verification_events": verification_total,
            "verification_success_rate": (
                round((verification_ok / verification_total), 6)
                if verification_total
                else 0.0
            ),
        }

    # ------------------------------------------------------------------
    # Full discovery orchestration
    # ------------------------------------------------------------------

    def _discover_for_model(self, requested_model: str, allow_degradation: bool):
        """Full discovery orchestration for a requested model.

        Scans the network, applies verification QoS gating, evaluates
        degradation policy, ranks healthy peers by reputation and stake,
        and returns ranked candidates ready for pipeline assembly.

        Args:
            requested_model: The user-requested model identifier.
            allow_degradation: Whether to allow fallback to a different model
                when the requested one has insufficient peers.

        Returns:
            Tuple of (model_health, ranked_candidates, degradation_decision,
            available_peer_counts).

        Raises:
            RuntimeError: If no viable model or peers are available.
        """
        requested_model = str(requested_model or self.config.default_model).strip() or self.config.default_model
        scan_models = list(self._catalog_model_ids())
        if bool(self.config.allow_dynamic_model_ids) and requested_model not in scan_models:
            scan_models.append(requested_model)
        # Resolve catalog alias -> HF model ID; include both in scan for bidirectional peer matching
        _catalog_hf = self._catalog_hf_model_id(requested_model) or ""
        if _catalog_hf and _catalog_hf != requested_model and _catalog_hf not in scan_models:
            scan_models.append(_catalog_hf)
        healthy, counts = self._engine._scan_network(model_ids=scan_models)
        # Merge alias count <-> HF ID count (same physical model, different announcement forms)
        if _catalog_hf and _catalog_hf != requested_model:
            _merged = counts.get(requested_model, 0) + counts.get(_catalog_hf, 0)
            counts = {**counts, requested_model: _merged, _catalog_hf: _merged}
        verification_feedback_by_model = self._verification_feedback_by_model(healthy)
        if requested_model not in verification_feedback_by_model:
            verification_feedback_by_model[requested_model] = self._verification_metrics_for_model(requested_model, healthy)

        qos_min_events = max(1, int(self.config.verification_qos_min_events))
        qos_min_success = max(0.0, min(1.0, float(self.config.verification_qos_min_success_rate)))
        qos_enabled = qos_min_success > 0.0
        qos_blocked_models: dict[str, dict[str, Any]] = {}
        gated_counts = dict(counts)
        if qos_enabled:
            for model_id, metrics in verification_feedback_by_model.items():
                events = int(metrics.get("verification_events", 0))
                success_rate = float(metrics.get("verification_success_rate", 0.0))
                if events >= qos_min_events and success_rate < qos_min_success:
                    gated_counts[model_id] = 0
                    qos_blocked_models[model_id] = metrics

        requested_qos = qos_blocked_models.get(requested_model)
        _qos_update = {
            "enabled": qos_enabled,
            "min_events": qos_min_events,
            "min_success_rate": round(qos_min_success, 6),
            "requested_model_blocked": requested_qos is not None,
            "requested_model_events": int((requested_qos or {}).get("verification_events", 0)),
            "requested_model_success_rate": (
                float(requested_qos["verification_success_rate"]) if requested_qos is not None else None
            ),
        }
        self._last_verification_qos = _qos_update
        # Propagate to engine so InferenceService sees the same state.
        if self._engine is not None:
            self._engine._last_verification_qos.update(_qos_update)

        if requested_model in self.catalogue_by_model:
            decision = self.degradation_policy.select(
                requested_model=requested_model,
                available_peer_counts=gated_counts,
                allow_degradation=allow_degradation,
            )
        elif not bool(self.config.allow_dynamic_model_ids):
            decision = DegradationDecision(
                requested_model=requested_model,
                served_model=requested_model,
                degraded=False,
                available=True,
                reason="unknown_model",
                detail=(
                    f"model '{requested_model}' is not in catalogue and dynamic model IDs are disabled"
                ),
            )
        else:
            requested_available = int(gated_counts.get(requested_model, 0))
            if requested_available >= 1:
                decision = DegradationDecision(
                    requested_model=requested_model,
                    served_model=requested_model,
                    degraded=False,
                    available=True,
                    reason="ok_dynamic",
                    detail="requested model discovered via healthy peer registry",
                )
            elif not allow_degradation:
                decision = DegradationDecision(
                    requested_model=requested_model,
                    served_model=requested_model,
                    degraded=False,
                    available=False,
                    reason="insufficient_peers",
                    detail=(
                        "requested dynamic model has 0/1 healthy peers and degradation is disabled"
                    ),
                )
            else:
                fallback_model_id: str | None = None
                fallback_available = 0
                fallback_required = 0
                for candidate in self.model_catalog:
                    available = int(gated_counts.get(candidate.model_id, 0))
                    if available >= int(candidate.required_peers):
                        fallback_model_id = candidate.model_id
                        fallback_available = available
                        fallback_required = int(candidate.required_peers)
                        break

                if fallback_model_id is not None:
                    decision = DegradationDecision(
                        requested_model=requested_model,
                        served_model=fallback_model_id,
                        degraded=True,
                        available=True,
                        reason="dynamic_model_unavailable",
                        detail=(
                            f"requested dynamic model has 0/1 healthy peers; "
                            f"fallback {fallback_model_id} has {fallback_available}/{fallback_required}"
                        ),
                    )
                else:
                    decision = DegradationDecision(
                        requested_model=requested_model,
                        served_model=requested_model,
                        degraded=False,
                        available=False,
                        reason="no_viable_fallback",
                        detail=(
                            "requested dynamic model has 0/1 healthy peers and no fallback model met required replicas"
                        ),
                    )

        if requested_qos is not None:
            qos_prefix = (
                "verification_qos_floor_breached: "
                f"{requested_model} verification_success_rate={requested_qos['verification_success_rate']} "
                f"min_required={round(qos_min_success, 6)} "
                f"events={requested_qos['verification_events']}"
            )
            decision = DegradationDecision(
                requested_model=decision.requested_model,
                served_model=decision.served_model,
                degraded=decision.degraded,
                available=decision.available,
                reason=("verification_qos" if decision.degraded else decision.reason),
                detail=f"{qos_prefix}; {decision.detail}",
            )

        if decision.reason == "unknown_model":
            raise RuntimeError(f"unknown_model:{decision.detail}")
        if not decision.available:
            raise RuntimeError(f"no_viable_model:{decision.reason}")

        served_model = decision.served_model
        _served_hf = self._catalog_hf_model_id(served_model) or served_model
        model_health = [
            h for h in healthy
            if self._normalize_peer_model(h.peer) in {served_model, _served_hf}
        ]
        if not model_health:
            raise RuntimeError(f"served_model_unavailable:no healthy peers for {served_model}")

        peer_ids = [item.peer.peer_id for item in model_health]
        reputations = self.health.scores(peer_ids)
        stake_priority_boost = max(0.0, float(self.config.hydra_stake_priority_boost))
        if stake_priority_boost > 0.0:
            for peer_id in peer_ids:
                staked_balance = max(0.0, float(self.ledger_bridge.verify_staked_balance(peer_id)))
                if staked_balance <= 0.0:
                    continue
                # Optional stake increases routing priority but does not gate admission.
                stake_bonus = stake_priority_boost * (staked_balance / (staked_balance + 1.0))
                reputations[peer_id] = min(
                    100.0,
                    float(reputations.get(peer_id, 50.0)) + stake_bonus,
                )
        bandwidth_by_peer = {item.peer.peer_id: item.peer.bandwidth_mbps for item in model_health}

        ranked = rank_peers(
            model_health,
            tier=self.config.tier,
            reputation_by_peer=reputations,
            bandwidth_by_peer=bandwidth_by_peer,
        )
        self._last_scored_peers = ranked
        candidates = [item.peer for item in ranked]
        return model_health, candidates, decision, counts

    # ------------------------------------------------------------------
    # Convenience wrapper
    # ------------------------------------------------------------------

    def _discover(self):
        """Convenience wrapper: discover peers for the default model."""
        health, candidates, _, _ = self._engine._discover_for_model(
            requested_model=self.config.default_model,
            allow_degradation=self.config.allow_degradation_default,
        )
        return health, candidates
