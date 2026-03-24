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

"""Mixture-of-Experts geo-sharding service.

Handles expert tag / layer-index normalisation, prompt-hint extraction,
and the MoE-aware pipeline reordering that places expert-matching peers
at the head of the inference pipeline.
"""

from __future__ import annotations

import re
from typing import Any

from coordinator.path_finder import PeerEndpoint


class MoeService:
    """Extracted from ``CoordinatorEngine``.

    Parameters
    ----------
    config:
        An ``EngineConfig`` (or duck-typed equivalent) that exposes at least
        ``moe_geo_enabled``, ``moe_geo_min_tag_matches``,
        ``moe_geo_min_layer_matches``, ``moe_geo_prompt_hints_enabled``,
        and ``dht_preferred_region``.
    """

    def __init__(self, config: Any):
        self.config = config

    # ------------------------------------------------------------------
    # Normalisation helpers (static)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_expert_tags(raw: Any) -> list[str]:
        """Normalize raw expert tag input into a deduplicated lowercase list.

        Accepts a comma-separated string, an iterable, or ``None``.

        Args:
            raw: Raw tag input (str, iterable, or None).

        Returns:
            Ordered list of unique, lowercased tag strings.
        """
        if raw is None:
            return []
        if isinstance(raw, str):
            values = [part.strip().lower() for part in raw.split(",")]
        else:
            try:
                values = [str(item).strip().lower() for item in list(raw)]
            except TypeError:
                values = []
        out: list[str] = []
        seen: set[str] = set()
        for value in values:
            if not value or value in seen:
                continue
            seen.add(value)
            out.append(value)
        return out

    @staticmethod
    def _normalize_expert_layer_indices(raw: Any) -> list[int]:
        """Normalize raw layer-index input into a sorted, deduplicated list.

        Accepts a comma-separated string, an iterable, or ``None``.
        Negative indices are silently dropped.

        Args:
            raw: Raw layer index input (str, iterable, or None).

        Returns:
            Sorted list of unique non-negative layer indices.
        """
        if raw is None:
            return []
        if isinstance(raw, str):
            values = [part.strip() for part in raw.split(",")]
        else:
            try:
                values = [str(item).strip() for item in list(raw)]
            except TypeError:
                values = []
        out: list[int] = []
        seen: set[int] = set()
        for value in values:
            if not value:
                continue
            try:
                idx = int(value)
            except ValueError:
                continue
            if idx < 0 or idx in seen:
                continue
            seen.add(idx)
            out.append(idx)
        return sorted(out)

    # ------------------------------------------------------------------
    # Prompt-hint extraction
    # ------------------------------------------------------------------

    def _extract_prompt_expert_tags(self, prompt: str) -> list[str]:
        """Extract ``expert:<tag>`` hints from the prompt text.

        Args:
            prompt: The user prompt to scan for expert hints.

        Returns:
            Normalized list of extracted expert tag strings.
        """
        if not self.config.moe_geo_prompt_hints_enabled:
            return []
        tags = re.findall(r"expert:([a-z0-9][a-z0-9_-]{0,31})", str(prompt).lower())
        return self._normalize_expert_tags(tags)

    def _extract_prompt_expert_layer_indices(self, prompt: str) -> list[int]:
        """Extract ``layer:<N>`` hints from the prompt text.

        Args:
            prompt: The user prompt to scan for layer hints.

        Returns:
            Sorted list of extracted layer indices.
        """
        if not self.config.moe_geo_prompt_hints_enabled:
            return []
        layers = re.findall(r"(?:expert[-_]?layer|layer):([0-9]{1,5})", str(prompt).lower())
        return self._normalize_expert_layer_indices(layers)

    # ------------------------------------------------------------------
    # MoE geo-sharding pipeline reordering
    # ------------------------------------------------------------------

    def _apply_moe_geo_sharding(
        self,
        pipeline: list[PeerEndpoint],
        ranked_candidates: list[PeerEndpoint],
        *,
        prompt: str,
        requested_expert_tags: list[str] | None,
        requested_expert_layer_indices: list[int] | None = None,
        locked_first_peer_id: str | None = None,
    ) -> tuple[list[PeerEndpoint], dict[str, Any]]:
        """Reorder the pipeline to prioritize expert-matching peers.

        Combines explicit expert tags/layers with prompt-hint extraction,
        scores candidate peers by tag overlap, layer overlap, router bonus,
        region match, and bandwidth, then reorders the pipeline to place the
        best-matching expert peers first.

        The first peer may be locked (e.g. by KV affinity) and will not be
        displaced.

        Args:
            pipeline: The current inference pipeline to potentially reorder.
            ranked_candidates: All ranked candidate peers available.
            prompt: The effective prompt (used for hint extraction).
            requested_expert_tags: Explicit expert tags from the request.
            requested_expert_layer_indices: Explicit layer indices.
            locked_first_peer_id: Peer ID that must stay at position 0.

        Returns:
            Tuple of (reordered_pipeline, policy_dict) where the policy dict
            describes matching details and whether reordering was applied.
        """
        enabled = bool(self.config.moe_geo_enabled)
        min_tag_matches = max(1, int(self.config.moe_geo_min_tag_matches))
        min_layer_matches = max(1, int(self.config.moe_geo_min_layer_matches))
        prompt_hint_tags = self._extract_prompt_expert_tags(prompt)
        prompt_hint_layers = self._extract_prompt_expert_layer_indices(prompt)
        explicit_tags = self._normalize_expert_tags(requested_expert_tags)
        explicit_layers = self._normalize_expert_layer_indices(requested_expert_layer_indices)
        requested_tags = explicit_tags + [tag for tag in prompt_hint_tags if tag not in explicit_tags]
        requested_layers = explicit_layers + [idx for idx in prompt_hint_layers if idx not in explicit_layers]
        requested_tag_set = set(requested_tags)
        requested_layer_set = set(requested_layers)
        policy: dict[str, Any] = {
            "enabled": enabled,
            "requested_experts": list(requested_tags),
            "requested_layer_indices": list(requested_layers),
            "prompt_hint_experts": list(prompt_hint_tags),
            "prompt_hint_layer_indices": list(prompt_hint_layers),
            "min_tag_matches": min_tag_matches,
            "min_layer_matches": min_layer_matches,
            "preferred_region": (str(self.config.dht_preferred_region).strip() if self.config.dht_preferred_region else None),
            "matching_candidates": 0,
            "expert_admission_rejections": 0,
            "matched_peer_ids": [],
            "matched_tags": [],
            "matched_layer_indices": [],
            "router_peer_ids": [],
            "locked_first_peer_id": (str(locked_first_peer_id).strip() if locked_first_peer_id else None),
            "applied": False,
            "reason": "disabled",
        }
        if not enabled:
            return pipeline, policy
        if not pipeline:
            policy["reason"] = "empty_pipeline"
            return pipeline, policy
        if not requested_tag_set and not requested_layer_set:
            policy["reason"] = "no_requested_experts"
            return pipeline, policy

        lock_id = str(locked_first_peer_id or "").strip() or None
        locked_head: list[PeerEndpoint] = []
        pipeline_tail = list(pipeline)
        if lock_id:
            locked = next((peer for peer in pipeline if peer.peer_id == lock_id), None)
            if locked is not None:
                locked_head = [locked]
                pipeline_tail = [peer for peer in pipeline if peer.peer_id != locked.peer_id]

        preferred_region = policy["preferred_region"]
        candidate_priority = {peer.peer_id: idx for idx, peer in enumerate(ranked_candidates)}
        matches: list[tuple[tuple[float, float, float, float, float, float], PeerEndpoint, set[str], set[int]]] = []
        for peer in ranked_candidates:
            if not bool(peer.expert_admission_approved):
                policy["expert_admission_rejections"] = int(policy["expert_admission_rejections"]) + 1
                continue
            peer_tags = set(str(item).strip().lower() for item in tuple(peer.expert_tags) if str(item).strip())
            peer_layers = set(int(idx) for idx in tuple(peer.expert_layer_indices))
            matched_tags = requested_tag_set.intersection(peer_tags)
            matched_layers = requested_layer_set.intersection(peer_layers)
            tag_match_ok = bool(requested_tag_set) and len(matched_tags) >= min_tag_matches
            layer_match_ok = bool(requested_layer_set) and len(matched_layers) >= min_layer_matches
            if not (tag_match_ok or layer_match_ok):
                continue
            region_match = 1.0 if (preferred_region and str(peer.region or "").strip().lower() == preferred_region.lower()) else 0.0
            router_bonus = 1.0 if bool(peer.expert_router) else 0.0
            score = (
                float(len(matched_tags)),
                float(len(matched_layers)),
                router_bonus,
                region_match,
                max(0.0, float(peer.bandwidth_mbps)),
                -float(candidate_priority.get(peer.peer_id, 0)),
            )
            matches.append((score, peer, matched_tags, matched_layers))

        if not matches:
            policy["reason"] = "no_matching_experts"
            return pipeline, policy

        policy["matching_candidates"] = len(matches)
        matches.sort(key=lambda item: item[0], reverse=True)

        used_peer_ids = {peer.peer_id for peer in locked_head}
        selected: list[PeerEndpoint] = []
        matched_tags_union: set[str] = set()
        matched_layers_union: set[int] = set()
        for _, peer, matched_tags, matched_layers in matches:
            if peer.peer_id in used_peer_ids:
                continue
            used_peer_ids.add(peer.peer_id)
            selected.append(peer)
            matched_tags_union.update(matched_tags)
            matched_layers_union.update(matched_layers)
            if len(selected) >= len(pipeline_tail):
                break

        remainder = [peer for peer in pipeline_tail if peer.peer_id not in used_peer_ids]
        arranged = (locked_head + selected + remainder)[: len(pipeline)]

        matched_peer_ids: list[str] = []
        for peer in arranged:
            peer_tags = set(str(item).strip().lower() for item in tuple(peer.expert_tags) if str(item).strip())
            peer_layers = set(int(idx) for idx in tuple(peer.expert_layer_indices))
            if requested_tag_set.intersection(peer_tags) or requested_layer_set.intersection(peer_layers):
                matched_peer_ids.append(peer.peer_id)
        router_peer_ids = [peer.peer_id for peer in arranged if bool(peer.expert_router)]

        policy.update(
            {
                "matched_peer_ids": matched_peer_ids,
                "matched_tags": sorted(matched_tags_union),
                "matched_layer_indices": sorted(matched_layers_union),
                "router_peer_ids": router_peer_ids,
                "applied": arranged != pipeline,
                "reason": ("reordered" if arranged != pipeline else "already_optimal"),
            }
        )
        return arranged, policy
