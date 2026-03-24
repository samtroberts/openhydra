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

"""Pipeline service — extracted from CoordinatorEngine.

Handles pipeline assembly: sharded pipeline selection via LayerCoverageMap,
full-model pipeline selection via concentration guard, bandwidth-role
reordering, and KV-affinity-aware bandwidth asymmetry application.
"""

from __future__ import annotations

import logging
from typing import Any

from coordinator.bandwidth_roles import (
    RoleThresholds,
    classify_role,
    estimate_prompt_tokens,
)
from coordinator.concentration_guard import assemble_pipeline
from coordinator.path_finder import PeerEndpoint

logger = logging.getLogger(__name__)


class PipelineService:
    """Owns pipeline assembly, bandwidth asymmetry, and role classification.

    Parameters
    ----------
    config:
        An ``EngineConfig`` (or duck-typed equivalent).
    kv_affinity_service:
        A ``KvAffinityService`` for KV-cache peer affinity lookups/updates.
    role_thresholds:
        A ``RoleThresholds`` for bandwidth role classification.
    """

    def __init__(
        self,
        config: Any,
        kv_affinity_service: Any,
        role_thresholds: RoleThresholds,
        engine: Any = None,
    ) -> None:
        self._engine = engine
        self.config = config
        self.kv_affinity_service = kv_affinity_service
        self.role_thresholds = role_thresholds

    # ------------------------------------------------------------------
    # Sharded pipeline selection
    # ------------------------------------------------------------------

    def _select_pipeline_sharded(
        self,
        health: list,
    ) -> list[PeerEndpoint] | None:
        """Return an ordered layer-sharded pipeline, or ``None`` if unavailable.

        Builds a :class:`~coordinator.layer_coverage.LayerCoverageMap` from the
        healthy peers.  When health metrics are available, uses Dijkstra
        cost-optimal routing (:func:`find_optimal_pipeline`) to pick the
        cheapest path; otherwise falls back to the greedy interval algorithm.

        Returns ``None`` when:
        * No sharded peers are present (all peers are full-model replicas).
        * The sharded peers do not form complete coverage of ``[0, total_layers)``.

        Phase 3: called first in :meth:`_prepare_inference`; falls through to the
        legacy ``_select_pipeline`` path when it returns ``None``.
        """
        from coordinator.layer_coverage import (
            LayerCoverageMap as _LayerCoverageMap,
            PeerMetrics as _PeerMetrics,
        )
        health_peers = [h.peer for h in health]
        cmap = _LayerCoverageMap.from_endpoints(health_peers)
        if not cmap.has_sharded_peers:
            return None
        if not cmap.is_complete():
            logger.warning(
                "sharded_pipeline_incomplete: coverage_fraction=%.4f gaps=%s",
                cmap.coverage_fraction(),
                cmap.gaps(),
            )
            # Trigger rebalancer to compute and publish directives.
            try:
                from coordinator.rebalancer import LayerRebalancer as _LayerRebalancer
                from coordinator.rebalancer import publish_directives_to_dht as _publish_directives
                rebalancer = _LayerRebalancer()
                directives = rebalancer.compute_directives(cmap, health)
                if directives:
                    dht_urls = list(self.config.dht_urls) if self.config.dht_urls else []
                    if dht_urls:
                        ok, fail = _publish_directives(directives, dht_urls, timeout_s=2.0)
                        logger.info(
                            "rebalancer_published: directives=%d ok=%d fail=%d",
                            len(directives), ok, fail,
                        )
            except Exception as exc:
                logger.debug("rebalancer_error: %s", exc)
            return None

        # Build PeerMetrics from PeerHealth objects for Dijkstra routing.
        peer_metrics: dict[str, _PeerMetrics] = {}
        for h in health:
            p = h.peer
            peer_metrics[p.peer_id] = _PeerMetrics(
                latency_ms=float(getattr(h, "latency_ms", 0.0) or 0.0),
                estimated_tps=float(
                    getattr(p, "runtime_estimated_tokens_per_sec", 0.0) or 0.0
                ),
                reputation_score=float(
                    getattr(p, "reputation_score", 50.0) or 50.0
                ),
                load_pct=float(getattr(h, "load_pct", 0.0) or 0.0) / 100.0
                if float(getattr(h, "load_pct", 0.0) or 0.0) > 1.0
                else float(getattr(h, "load_pct", 0.0) or 0.0),
            )

        layer_pipeline = cmap.best_pipeline(peer_metrics=peer_metrics)
        if not layer_pipeline:
            return None
        peer_by_id: dict[str, PeerEndpoint] = {h.peer.peer_id: h.peer for h in health}
        sharded: list[PeerEndpoint] = []
        for lr in layer_pipeline:
            if not lr.is_sharded:
                # The algorithm picked a full-model replica as the optimal
                # pipeline.  Fall back to the full-model path so it is
                # handled correctly there.
                logger.debug(
                    "sharded_pipeline_rejected: full_model_replica_selected peer_id=%s",
                    lr.peer_id,
                )
                return None
            peer = peer_by_id.get(lr.peer_id)
            if peer is None:
                logger.warning("sharded_pipeline_peer_missing: peer_id=%s", lr.peer_id)
                return None
            sharded.append(peer)
        logger.info(
            "sharded_pipeline_selected: stages=%d peers=%s",
            len(sharded),
            [f"{p.peer_id}[{getattr(p,'layer_start',0)}-{getattr(p,'layer_end',0)})" for p in sharded],
        )
        return sharded

    # ------------------------------------------------------------------
    # Full-model pipeline selection
    # ------------------------------------------------------------------

    def _select_pipeline(self, candidates: list, pipeline_width: int | None = None) -> list:
        """Assemble a full-model inference pipeline from ranked candidates.

        Uses the concentration guard to enforce operator diversity constraints.

        Args:
            candidates: Ranked list of candidate peers.
            pipeline_width: Number of peers per pipeline (defaults to config).

        Returns:
            Ordered list of peers forming the pipeline.

        Raises:
            RuntimeError: If no peers are selected.
        """
        width = max(1, pipeline_width or self.config.pipeline_width)
        pipeline = assemble_pipeline(
            candidates,
            pipeline_width=width,
            max_fraction=self.config.operator_cap_fraction,
            enforce_diversity=self.config.enforce_pipeline_diversity,
            diversity_window=max(2, self.config.diversity_window),
            max_per_window=max(1, self.config.diversity_max_per_window),
        )
        if len(pipeline) < width:
            logger.warning(
                "operator_cap_enforced: pipeline assembled with %s/%s peers",
                len(pipeline),
                width,
            )
        if not pipeline:
            raise RuntimeError("No peers selected for pipeline")
        return pipeline

    # ------------------------------------------------------------------
    # Bandwidth role helpers
    # ------------------------------------------------------------------

    def _role_for_peer(self, peer: PeerEndpoint) -> str:
        """Classify a peer's bandwidth role (prefill_capable, balanced, decode_only)."""
        return classify_role(peer.bandwidth_mbps, thresholds=self.role_thresholds)

    def _reorder_for_decode_tail(self, peers: list[PeerEndpoint]) -> list[PeerEndpoint]:
        """Move decode-only peers to the tail of the pipeline.

        Args:
            peers: Pipeline peer list to reorder.

        Returns:
            Reordered peer list with decode-only peers at the end.
        """
        if len(peers) <= 1:
            return peers
        decode_only = [peer for peer in peers if self._role_for_peer(peer) == "decode_only"]
        others = [peer for peer in peers if self._role_for_peer(peer) != "decode_only"]
        return others + decode_only

    # ------------------------------------------------------------------
    # Bandwidth asymmetry with KV affinity
    # ------------------------------------------------------------------

    def _apply_bandwidth_asymmetry(
        self,
        pipeline: list[PeerEndpoint],
        ranked_candidates: list[PeerEndpoint],
        prompt_tokens_est: int,
        *,
        session_id: str | None = None,
        model_id: str | None = None,
    ) -> tuple[list[PeerEndpoint], dict[str, Any]]:
        """Reorder the pipeline to place the best prefill peer first.

        When the estimated prompt length exceeds the prefill threshold, selects
        a prefill-capable peer (preferring the KV-affinity sticky peer if
        available) and places it at position 0.  Decode-only peers are pushed
        to the tail.

        Args:
            pipeline: Current pipeline to reorder.
            ranked_candidates: All ranked candidate peers.
            prompt_tokens_est: Estimated prompt token count.
            session_id: Optional session ID for KV affinity lookup.
            model_id: Optional model ID for KV affinity lookup.

        Returns:
            Tuple of (reordered_pipeline, bandwidth_policy_dict).
        """
        served_model = model_id or self.config.default_model
        kv_affinity_requested = bool(self.config.kv_affinity_enabled and session_id)
        previous_prefill_peer_id = self._engine._get_kv_affinity_peer(session_id, served_model)
        kv_affinity_hit = False
        kv_cold_restart = False
        kv_affinity_updated = False

        if not pipeline:
            return pipeline, {
                "prompt_tokens_est": prompt_tokens_est,
                "prefill_required": False,
                "prefill_peer_id": None,
                "prefill_peer_role": None,
                "used_prefill_fallback": False,
                "kv_affinity_enabled": self.config.kv_affinity_enabled,
                "kv_affinity_ttl_seconds": max(1, int(self.config.kv_affinity_ttl_seconds)),
                "kv_session_id": session_id,
                "kv_affinity_requested": kv_affinity_requested,
                "kv_affinity_hit": False,
                "kv_previous_prefill_peer_id": previous_prefill_peer_id,
                "kv_cold_restart": False,
                "kv_affinity_updated": False,
            }

        prefill_required = prompt_tokens_est >= max(1, self.config.prefill_token_threshold)
        arranged = self._reorder_for_decode_tail(list(pipeline))
        chosen_prefill = arranged[0]
        used_fallback = False

        if prefill_required:
            preferred: PeerEndpoint | None = None
            if previous_prefill_peer_id:
                sticky = next(
                    (peer for peer in ranked_candidates if peer.peer_id == previous_prefill_peer_id),
                    None,
                )
                if sticky is not None:
                    preferred = sticky
                    kv_affinity_hit = True
                else:
                    kv_cold_restart = True

            if preferred is None:
                prefill_candidates = [
                    peer for peer in ranked_candidates if self._role_for_peer(peer) == "prefill_capable"
                ]
                if prefill_candidates:
                    preferred = prefill_candidates[0]
                else:
                    preferred = sorted(
                        ranked_candidates,
                        key=lambda peer: peer.bandwidth_mbps,
                        reverse=True,
                    )[0]
                    used_fallback = True

            if preferred.peer_id != arranged[0].peer_id:
                remainder = [peer for peer in arranged if peer.peer_id != preferred.peer_id]
                arranged = [preferred] + remainder
                arranged = [arranged[0]] + self._reorder_for_decode_tail(arranged[1:])

            chosen_prefill = arranged[0]
            kv_affinity_updated = self._engine._set_kv_affinity_peer(session_id, served_model, chosen_prefill.peer_id)

        policy = {
            "prompt_tokens_est": prompt_tokens_est,
            "prefill_required": prefill_required,
            "prefill_peer_id": chosen_prefill.peer_id,
            "prefill_peer_role": self._role_for_peer(chosen_prefill),
            "used_prefill_fallback": used_fallback,
            "kv_affinity_enabled": self.config.kv_affinity_enabled,
            "kv_affinity_ttl_seconds": max(1, int(self.config.kv_affinity_ttl_seconds)),
            "kv_session_id": session_id,
            "kv_affinity_requested": kv_affinity_requested,
            "kv_affinity_hit": kv_affinity_hit,
            "kv_previous_prefill_peer_id": previous_prefill_peer_id,
            "kv_cold_restart": kv_cold_restart,
            "kv_affinity_updated": kv_affinity_updated,
        }
        return arranged[: len(pipeline)], policy
