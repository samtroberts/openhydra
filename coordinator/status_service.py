"""Status service — extracted from CoordinatorEngine.

Provides the ``list_models``, ``network_status``, and ``metrics_snapshot``
endpoints that expose catalog data, live network health, and operational
counters without triggering inference.
"""

from __future__ import annotations

import logging
import statistics
import threading
from typing import Any

from coordinator.bandwidth_roles import (
    RoleThresholds,
    classify_role,
    role_counts_from_bandwidth,
)
from coordinator.concentration_guard import concentration_metrics
from coordinator.degradation import ModelAvailability
from coordinator.path_finder import PeerEndpoint

logger = logging.getLogger(__name__)


class StatusService:
    """Owns the read-only status/metrics endpoints.

    Parameters
    ----------
    config:
        An ``EngineConfig`` (or duck-typed equivalent).
    health:
        A ``HealthScorer`` instance.
    auto_scaler:
        An ``AutoScaler | None``.
    _active_model_roster:
        Shared mutable list of currently active model IDs.
    _dht_peer_cache:
        Shared mutable dict for DHT peer caching.
    _dht_peer_cache_lock:
        A ``threading.Lock`` protecting ``_dht_peer_cache``.
    _request_log:
        A ``RequestLog`` for demand snapshots.
    ledger_bridge:
        An ``OpenHydraLedgerBridge`` for bridge summary.
    hydra:
        The HYDRA token economy instance.
    _metrics_lock:
        A ``threading.Lock`` protecting counter variables.
    model_catalog:
        The loaded list of ``ModelAvailability`` items.
    catalogue_by_model:
        Dict mapping model_id -> ``ModelAvailability``.
    discovery_service:
        A ``DiscoveryService`` for scanning network / catalog queries.
    replication_monitor:
        A ``ReplicationMonitor`` for replication evaluation.
    role_thresholds:
        A ``RoleThresholds`` for bandwidth role classification.
    """

    def __init__(
        self,
        config: Any,
        health: Any,
        auto_scaler: Any,
        _active_model_roster: list[str],
        _dht_peer_cache: dict[str, dict[str, Any]],
        _dht_peer_cache_lock: threading.Lock,
        _request_log: Any,
        ledger_bridge: Any,
        hydra: Any,
        _metrics_lock: threading.Lock,
        model_catalog: list[ModelAvailability],
        catalogue_by_model: dict[str, ModelAvailability],
        discovery_service: Any,
        replication_monitor: Any,
        role_thresholds: RoleThresholds,
        # Counter state — mutable references
        _dht_lookup_attempts_ref: list[int],
        _dht_lookup_successes_ref: list[int],
        _dht_lookup_failures_ref: list[int],
        _kv_store_ops_total_ref: list[int],
        _kv_retrieve_ops_total_ref: list[int],
        _inference_requests_total_ref: list[int],
        engine: Any = None,
    ) -> None:
        self._engine = engine
        self.config = config
        self.health = health
        self.auto_scaler = auto_scaler
        self._active_model_roster = _active_model_roster
        self._dht_peer_cache = _dht_peer_cache
        self._dht_peer_cache_lock = _dht_peer_cache_lock
        self._request_log = _request_log
        self.ledger_bridge = ledger_bridge
        self.hydra = hydra
        self._metrics_lock = _metrics_lock
        self.model_catalog = model_catalog
        self.catalogue_by_model = catalogue_by_model
        self.discovery_service = discovery_service
        self.replication_monitor = replication_monitor
        self.role_thresholds = role_thresholds
        # Counter refs (single-element lists for mutability)
        self._dht_lookup_attempts_ref = _dht_lookup_attempts_ref
        self._dht_lookup_successes_ref = _dht_lookup_successes_ref
        self._dht_lookup_failures_ref = _dht_lookup_failures_ref
        self._kv_store_ops_total_ref = _kv_store_ops_total_ref
        self._kv_retrieve_ops_total_ref = _kv_retrieve_ops_total_ref
        self._inference_requests_total_ref = _inference_requests_total_ref

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _catalog_model_ids(self) -> list[str]:
        return [item.model_id for item in self.model_catalog]

    def _required_replicas(self, model_id: str) -> int:
        item = self.catalogue_by_model.get(model_id)
        if item is None:
            return self.config.required_replicas
        return item.required_peers

    def _normalize_peer_model(self, peer: PeerEndpoint) -> str:
        return peer.model_id or self.config.default_model

    def _role_for_peer(self, peer: PeerEndpoint) -> str:
        return classify_role(peer.bandwidth_mbps, thresholds=self.role_thresholds)

    def _replication_dict(self, model_id: str, healthy_peers: int) -> dict[str, Any]:
        status = self.replication_monitor.evaluate(
            model_id,
            healthy_peers,
            required_replicas=self._required_replicas(model_id),
        )
        return self.replication_monitor.to_dict(status)

    # ------------------------------------------------------------------
    # list_models
    # ------------------------------------------------------------------

    def list_models(self) -> dict[str, Any]:
        """Return the model catalog with cached peer counts (no DHT scan).

        Peer counts are served from the in-memory DHT cache populated during
        inference, so they become accurate after the first request without
        blocking this endpoint.

        Returns:
            OpenAI-compatible ``{"object": "list", "data": [...]}`` response.
        """
        # Serve directly from the static catalog -- no DHT scan.
        # DHT scanning is deferred to the inference path (/v1/chat/completions),
        # where peer discovery is actually needed.  Peer counts here are served
        # from the in-memory DHT cache (populated on each inference request,
        # 120 s TTL) so they become accurate after the first inference and
        # never block the models listing endpoint.
        data = []
        for model in self.model_catalog:
            cached = self._engine._cached_dht_peers(model_id=model.model_id)
            healthy_count = len(cached)
            replication = self._replication_dict(model.model_id, healthy_count)
            data.append(
                {
                    "id": model.model_id,
                    "hf_model_id": (str(model.hf_model_id).strip() or None),
                    "object": "model",
                    "owned_by": "openhydra",
                    "healthy_peers": healthy_count,
                    "required_replicas": model.required_peers,
                    "under_replicated": replication["under_replicated"],
                    "min_vram_gb": model.min_vram_gb,
                    "recommended_quantization": model.recommended_quantization,
                    "context_length": model.context_length,
                    "languages": list(model.languages),
                    "tags": list(model.tags),
                    "description": model.description,
                }
            )

        return {"object": "list", "data": data}

    # ------------------------------------------------------------------
    # network_status
    # ------------------------------------------------------------------

    def network_status(self) -> dict[str, Any]:
        """Perform a live network scan and return comprehensive health status.

        Includes per-model replication, concentration, bandwidth roles,
        seeding, runtime profiles, expert profiles, verification feedback,
        HYDRA economy summary, auto-scaler state, and layer coverage.

        Returns:
            Dict with all network status fields.
        """
        try:
            health, counts = self._engine._scan_network(model_ids=self._catalog_model_ids())
        except RuntimeError:
            per_model = [self._replication_dict(model.model_id, 0) for model in self.model_catalog]
            concentration_by_model = {
                model.model_id: {
                    "total_peers": 0,
                    "operator_counts": {},
                    "operator_shares": {},
                    "max_operator": None,
                    "max_share": 0.0,
                    "over_cap_operators": [],
                }
                for model in self.model_catalog
            }
            role_counts = {
                model.model_id: {"prefill_capable": 0, "balanced": 0, "decode_only": 0}
                for model in self.model_catalog
            }
            return {
                "deployment_profile": str(self.config.deployment_profile),
                "healthy_peers": 0,
                "average_latency_ms": None,
                "p95_latency_ms": None,
                "replication_factor": 0,
                "replication": per_model,
                "concentration": concentration_by_model,
                "bandwidth_roles": role_counts,
                "seeding": {
                    model.model_id: {
                        "seeding_enabled_peers": 0,
                        "seed_inference_active_peers": 0,
                        "total_seed_upload_limit_mbps": 0.0,
                        "avg_seed_upload_limit_mbps": 0.0,
                        "total_seed_target_upload_limit_mbps": 0.0,
                    }
                    for model in self.model_catalog
                },
                "runtime_profiles": {
                    model.model_id: {
                        "total_peers": 0,
                        "backends": {},
                        "quantization_modes": {},
                        "gpu_available_peers": 0,
                        "avg_estimated_tokens_per_sec": 0.0,
                        "avg_estimated_memory_mb": 0.0,
                    }
                    for model in self.model_catalog
                },
                "expert_profiles": {
                    model.model_id: {
                        "total_peers": 0,
                        "expert_peers": 0,
                        "router_capable_peers": 0,
                        "tags": {},
                        "layer_coverage": {},
                    }
                    for model in self.model_catalog
                },
                "verification_feedback": {
                    model.model_id: {
                        "verified_peers": 0,
                        "peers_with_failed_verifications": 0,
                        "total_verifications_ok": 0,
                        "total_verifications_failed": 0,
                        "verification_events": 0,
                        "verification_success_rate": 0.0,
                    }
                    for model in self.model_catalog
                },
                "hydra_economy": self.hydra.summary(),
                "verification_alerts": {},
                "models": self._catalog_model_ids(),
                "alerts": ["no_healthy_peers", "under_replicated"],
                "reputation": {},
            }

        latencies = [item.latency_ms for item in health]
        peer_ids = [item.peer.peer_id for item in health]

        # -- Phase 2: Auto-scaler evaluation (every 5 minutes) --
        if self.auto_scaler is not None:
            from coordinator.auto_scaler import PeerView as _PeerView
            _scaler_peers = [
                _PeerView(
                    peer_id=h.peer.peer_id,
                    available_vram_mb=int(h.peer.available_vram_mb),
                    assigned_model_id=h.peer.model_id,
                    cpu_score=0.0,
                    disk_free_gb=0.0,
                    tps=float(h.peer.runtime_estimated_tokens_per_sec),
                )
                for h in health
            ]
            _scaler_result = self.auto_scaler.maybe_evaluate(
                _scaler_peers, self._request_log
            )
            if _scaler_result is not None:
                self._active_model_roster[:] = list(_scaler_result.active_roster)

        # -- Phase 3: Layer sharding coverage --
        from coordinator.layer_coverage import LayerCoverageMap as _LayerCoverageMap
        _layer_cmap = _LayerCoverageMap.from_endpoints([h.peer for h in health])
        _layer_coverage_summary: dict | None = (
            _layer_cmap.summary() if _layer_cmap.has_sharded_peers else None
        )

        per_model = [self._replication_dict(model.model_id, int(counts.get(model.model_id, 0))) for model in self.model_catalog]
        concentration_by_model = {}
        role_counts = {}
        seeding_by_model = {}
        runtime_by_model = {}
        expert_by_model = {}
        health_snapshot = self.health.snapshot()
        verification_feedback_by_model = {}
        for model in self.model_catalog:
            model_peers = [item.peer for item in health if self._normalize_peer_model(item.peer) == model.model_id]
            metrics = concentration_metrics(model_peers, cap_fraction=self.config.operator_cap_fraction)
            concentration_by_model[model.model_id] = {
                "total_peers": metrics.total_peers,
                "operator_counts": metrics.operator_counts,
                "operator_shares": metrics.operator_shares,
                "max_operator": metrics.max_operator,
                "max_share": round(metrics.max_share, 6),
                "over_cap_operators": metrics.over_cap_operators,
            }
            role_counts[model.model_id] = role_counts_from_bandwidth(
                [peer.bandwidth_mbps for peer in model_peers],
                thresholds=self.role_thresholds,
            )
            seeding_enabled = [peer for peer in model_peers if peer.seeding_enabled]
            total_seed_limit = sum(peer.seed_upload_limit_mbps for peer in seeding_enabled)
            total_seed_target = sum(peer.seed_target_upload_limit_mbps for peer in seeding_enabled)
            seeding_by_model[model.model_id] = {
                "seeding_enabled_peers": len(seeding_enabled),
                "seed_inference_active_peers": sum(1 for peer in seeding_enabled if peer.seed_inference_active),
                "total_seed_upload_limit_mbps": round(total_seed_limit, 6),
                "avg_seed_upload_limit_mbps": round((total_seed_limit / len(seeding_enabled)), 6) if seeding_enabled else 0.0,
                "total_seed_target_upload_limit_mbps": round(total_seed_target, 6),
            }
            backend_counts: dict[str, int] = {}
            quant_counts: dict[str, int] = {}
            gpu_available_peers = 0
            total_est_tps = 0.0
            total_est_mem = 0.0
            for peer in model_peers:
                backend = str(peer.runtime_backend or "unknown")
                quant = str(peer.quantization_mode or "unknown")
                backend_counts[backend] = backend_counts.get(backend, 0) + 1
                quant_counts[quant] = quant_counts.get(quant, 0) + 1
                if bool(peer.runtime_gpu_available):
                    gpu_available_peers += 1
                total_est_tps += float(peer.runtime_estimated_tokens_per_sec)
                total_est_mem += float(peer.runtime_estimated_memory_mb)

            runtime_by_model[model.model_id] = {
                "total_peers": len(model_peers),
                "backends": backend_counts,
                "quantization_modes": quant_counts,
                "gpu_available_peers": gpu_available_peers,
                "avg_estimated_tokens_per_sec": (
                    round((total_est_tps / len(model_peers)), 6)
                    if model_peers
                    else 0.0
                ),
                "avg_estimated_memory_mb": (
                    round((total_est_mem / len(model_peers)), 6)
                    if model_peers
                    else 0.0
                ),
            }
            tag_counts: dict[str, int] = {}
            layer_counts: dict[str, int] = {}
            router_capable = 0
            expert_peers = 0
            for peer in model_peers:
                tags = [str(tag).strip().lower() for tag in tuple(peer.expert_tags) if str(tag).strip()]
                layers = [int(idx) for idx in tuple(peer.expert_layer_indices)]
                if bool(tags) or bool(layers) or bool(peer.expert_router):
                    expert_peers += 1
                if bool(peer.expert_router):
                    router_capable += 1
                for tag in tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
                for idx in layers:
                    key = str(idx)
                    layer_counts[key] = layer_counts.get(key, 0) + 1
            expert_by_model[model.model_id] = {
                "total_peers": len(model_peers),
                "expert_peers": expert_peers,
                "router_capable_peers": router_capable,
                "tags": tag_counts,
                "layer_coverage": layer_counts,
            }

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
            verification_feedback_by_model[model.model_id] = {
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

        alerts = []
        if not health:
            alerts.append("no_healthy_peers")
        if any(item["under_replicated"] for item in per_model):
            alerts.append("under_replicated")
        if any(item["over_cap_operators"] for item in concentration_by_model.values()):
            alerts.append("operator_concentration")
        if _layer_cmap.has_sharded_peers and not _layer_cmap.is_complete():
            alerts.append("sharding_incomplete")

        min_events = max(1, int(self.config.verification_alert_min_events))
        min_success = max(0.0, min(1.0, float(self.config.verification_alert_min_success_rate)))
        verification_alerts = {
            model_id: metrics
            for model_id, metrics in verification_feedback_by_model.items()
            if int(metrics.get("verification_events", 0)) >= min_events
            and float(metrics.get("verification_success_rate", 0.0)) < min_success
        }
        if verification_alerts:
            alerts.append("verification_degraded")

        return {
            "deployment_profile": str(self.config.deployment_profile),
            "healthy_peers": len(health),
            "average_latency_ms": round(statistics.mean(latencies), 2) if latencies else None,
            "p95_latency_ms": (
                round(sorted(latencies)[int(max(0, len(latencies) * 0.95 - 1))], 2) if latencies else None
            ),
            "replication_factor": len(health),
            "replication": per_model,
            "concentration": concentration_by_model,
            "bandwidth_roles": role_counts,
            "seeding": seeding_by_model,
            "runtime_profiles": runtime_by_model,
            "expert_profiles": expert_by_model,
            "verification_feedback": verification_feedback_by_model,
            "hydra_economy": self.hydra.summary(),
            "hydra_bridge": self.ledger_bridge.summary(),
            "verification_alerts": verification_alerts,
            "models": self._catalog_model_ids(),
            "alerts": alerts,
            "reputation": {peer_id: round(self.health.score(peer_id), 2) for peer_id in peer_ids},
            # Phase 2: auto-scaler state
            "active_model_roster": list(self._active_model_roster),
            "demand_weights": self._request_log.snapshot(),
            # Phase 3: layer sharding coverage
            "layer_coverage": _layer_coverage_summary,
        }

    # ------------------------------------------------------------------
    # metrics_snapshot
    # ------------------------------------------------------------------

    def metrics_snapshot(self) -> dict[str, float | int]:
        """Return a point-in-time snapshot of operational counters.

        Includes DHT lookup stats, HYDRA bridge supply figures, and
        KV/inference proxy counters.

        Returns:
            Dict of metric name to numeric value.
        """
        with self._metrics_lock:
            dht_attempts = int(self._dht_lookup_attempts_ref[0])
            dht_successes = int(self._dht_lookup_successes_ref[0])
            dht_failures = int(self._dht_lookup_failures_ref[0])
            kv_store_ops = int(self._kv_store_ops_total_ref[0])
            kv_retrieve_ops = int(self._kv_retrieve_ops_total_ref[0])
            inference_reqs = int(self._inference_requests_total_ref[0])
        dht_lookup_success_rate = (float(dht_successes) / float(dht_attempts)) if dht_attempts else 0.0
        bridge = self.ledger_bridge.summary()
        return {
            "dht_lookup_attempts": dht_attempts,
            "dht_lookup_successes": dht_successes,
            "dht_lookup_failures": dht_failures,
            "dht_lookup_success_rate": round(dht_lookup_success_rate, 6),
            "hydra_bridge_total_minted": float(bridge.get("total_minted", 0.0)),
            "hydra_bridge_total_burned": float(bridge.get("total_burned", 0.0)),
            "hydra_bridge_total_supply": float(bridge.get("total_supply", 0.0)),
            "hydra_bridge_supply_cap": float(bridge.get("supply_cap", 0.0)),
            # Phase D: KV + inference proxy counters
            "kv_store_ops_total":       kv_store_ops,
            "kv_retrieve_ops_total":    kv_retrieve_ops,
            "inference_requests_total": inference_reqs,
        }
