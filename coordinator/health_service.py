"""Health, verification and peer-discovery reporting service.

Centralises health recording, verification feedback bookkeeping,
replication status queries, and the assembly of ``discovered_peer_rows``
for the coordinator API.
"""

from __future__ import annotations

from typing import Any, Callable

from coordinator.chain import ChainResult
from coordinator.health_scorer import HealthScorer
from coordinator.path_finder import PeerEndpoint
from coordinator.replication_monitor import ReplicationMonitor


class HealthService:
    """Extracted from ``CoordinatorEngine``.

    Parameters
    ----------
    health:
        The shared ``HealthScorer`` instance.
    config:
        An ``EngineConfig`` (or duck-typed equivalent).
    replication_monitor:
        The ``ReplicationMonitor`` used to evaluate replication status.
    ledger_bridge:
        The ``OpenHydraLedgerBridge`` for stake slash operations.
    model_catalog:
        List of ``ModelAvailability`` objects.
    normalize_peer_model:
        Callable that normalises a peer's model_id (falls back to default).
    required_replicas:
        Callable ``(model_id) -> int`` returning required replica count.
    role_for_peer:
        Callable ``(peer) -> str`` returning the bandwidth role string.
    last_scored_peers_getter:
        Callable ``() -> list`` returning the last scored peer list.
    """

    def __init__(
        self,
        health: HealthScorer,
        config: Any,
        replication_monitor: ReplicationMonitor,
        ledger_bridge: Any,
        model_catalog: list[Any],
        normalize_peer_model: Callable[[PeerEndpoint], str],
        required_replicas: Callable[[str], int],
        role_for_peer: Callable[[PeerEndpoint], str],
        last_scored_peers_getter: Callable[[], list[Any]],
        engine: Any = None,
    ):
        self._engine = engine
        self.health = health
        self.config = config
        self.replication_monitor = replication_monitor
        self.ledger_bridge = ledger_bridge
        self.model_catalog = model_catalog
        self._normalize_peer_model = normalize_peer_model
        self._required_replicas = required_replicas
        self._role_for_peer = role_for_peer
        self._last_scored_peers_getter = last_scored_peers_getter

    # ------------------------------------------------------------------
    # Ping health
    # ------------------------------------------------------------------

    def _record_ping_health(self, survey) -> None:
        for item in survey:
            self.health.record_ping(item.peer.peer_id, healthy=item.healthy, latency_ms=item.latency_ms)

    # ------------------------------------------------------------------
    # Verification metrics
    # ------------------------------------------------------------------

    def _verification_feedback_by_model(self, health) -> dict[str, dict[str, Any]]:
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
    # Verification feedback application
    # ------------------------------------------------------------------

    def _apply_verification_feedback(
        self,
        *,
        primary: ChainResult,
        secondary: ChainResult | None,
        tertiary: ChainResult | None,
        verification: Any,
    ) -> dict[str, list[str]]:
        if not verification.audited:
            return {"rewarded_peers": [], "penalized_peers": []}

        queued: dict[str, list[bool]] = {}

        def _queue(result: ChainResult | None, success: bool) -> None:
            if result is None:
                return
            for peer_id in {trace.peer_id for trace in result.traces}:
                queued.setdefault(peer_id, []).append(success)

        winner_text = primary.text.strip()
        if verification.winner == "secondary" and secondary is not None:
            winner_text = secondary.text.strip()

        _queue(primary, primary.text.strip() == winner_text)
        _queue(secondary, bool(secondary and secondary.text.strip() == winner_text))
        _queue(tertiary, bool(tertiary and tertiary.text.strip() == winner_text))

        rewarded: list[str] = []
        penalized: list[str] = []
        for peer_id, votes in queued.items():
            if all(votes):
                self.health.record_verification(peer_id, success=True)
                rewarded.append(peer_id)
            elif not any(votes):
                self.health.record_verification(peer_id, success=False)
                penalized.append(peer_id)
            # Conflicting outcomes for the same peer are treated as neutral.

        slash_amount = max(0.0, float(self.config.hydra_slash_per_failed_verification))
        if slash_amount > 0.0:
            unstaked_penalty_events = max(1, int(self.config.hydra_no_stake_penalty_events))
            for peer_id in penalized:
                staked_balance = max(0.0, float(self.ledger_bridge.verify_staked_balance(peer_id)))
                if staked_balance > 0.0:
                    self.ledger_bridge.slash_stake(peer_id, min(slash_amount, staked_balance))
                    continue
                # No stake to slash: aggressively penalize reputation to suppress malicious routing.
                for _ in range(unstaked_penalty_events):
                    self.health.record_verification(peer_id, success=False)

        return {
            "rewarded_peers": sorted(rewarded),
            "penalized_peers": sorted(penalized),
        }

    # ------------------------------------------------------------------
    # Peer discovery rows
    # ------------------------------------------------------------------

    def _discovered_peer_rows(self, health) -> list[dict[str, Any]]:
        scored_lookup = {item.peer.peer_id: item for item in self._last_scored_peers_getter()}
        rows = []
        for item in health:
            scored = scored_lookup.get(item.peer.peer_id)
            model_id = self._normalize_peer_model(item.peer)
            rows.append(
                {
                    "peer_id": item.peer.peer_id,
                    "model_id": model_id,
                    "latency_ms": round(item.latency_ms, 2),
                    "load_pct": round(item.load_pct, 2),
                    "daemon_mode": item.daemon_mode,
                    "operator_id": item.peer.operator_id,
                    "region": item.peer.region,
                    "bandwidth_mbps": round(item.peer.bandwidth_mbps, 2),
                    "bandwidth_role": self._role_for_peer(item.peer),
                    "seeding_enabled": item.peer.seeding_enabled,
                    "seed_upload_limit_mbps": round(item.peer.seed_upload_limit_mbps, 6),
                    "seed_target_upload_limit_mbps": round(item.peer.seed_target_upload_limit_mbps, 6),
                    "seed_inference_active": item.peer.seed_inference_active,
                    "runtime_backend": item.peer.runtime_backend,
                    "runtime_target": item.peer.runtime_target,
                    "runtime_model_id": str(item.peer.runtime_model_id or ""),
                    "quantization_mode": item.peer.quantization_mode,
                    "quantization_bits": int(item.peer.quantization_bits),
                    "runtime_gpu_available": bool(item.peer.runtime_gpu_available),
                    "runtime_estimated_tokens_per_sec": round(item.peer.runtime_estimated_tokens_per_sec, 6),
                    "runtime_estimated_memory_mb": int(item.peer.runtime_estimated_memory_mb),
                    "privacy_noise_variance": round(float(item.peer.privacy_noise_variance), 12),
                    "privacy_noise_payloads": int(item.peer.privacy_noise_payloads),
                    "privacy_noise_observed_variance_ema": round(float(item.peer.privacy_noise_observed_variance_ema), 12),
                    "privacy_noise_last_audit_tag": str(item.peer.privacy_noise_last_audit_tag or ""),
                    "expert_admission_approved": bool(item.peer.expert_admission_approved),
                    "expert_admission_reason": str(item.peer.expert_admission_reason or "approved"),
                    "dht_reputation_score": round(float(item.peer.reputation_score), 6),
                    "dht_staked_balance": round(float(item.peer.staked_balance), 6),
                    "expert_tags": list(item.peer.expert_tags),
                    "expert_layer_indices": list(item.peer.expert_layer_indices),
                    "expert_router": bool(item.peer.expert_router),
                    "reputation": round(scored.reputation, 2) if scored else round(self.health.score(item.peer.peer_id), 2),
                    "routing_score": round(scored.score, 6) if scored else None,
                }
            )
        return rows

    # ------------------------------------------------------------------
    # Replication
    # ------------------------------------------------------------------

    def _replication_dict(self, model_id: str, healthy_peers: int) -> dict[str, Any]:
        status = self.replication_monitor.evaluate(
            model_id,
            healthy_peers,
            required_replicas=self._required_replicas(model_id),
        )
        return self.replication_monitor.to_dict(status)
