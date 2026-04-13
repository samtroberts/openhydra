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

"""Inference service — extracted from CoordinatorEngine.

Orchestrates end-to-end inference: preparation (discovery + pipeline +
bandwidth + MoE), single-shot ``infer``, streaming ``infer_stream``, and
the chat convenience wrappers.  Also owns ``_run_chain`` which builds an
``InferenceChain`` and executes a single gRPC pipeline pass.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from typing import Any

from coordinator.bandwidth_roles import (
    classify_role,
    estimate_prompt_tokens,
)
from coordinator.chain import ChainResult, InferenceChain
from coordinator.concentration_guard import concentration_metrics
from coordinator.degradation import DegradationDecision
from coordinator.path_finder import PeerEndpoint
from coordinator.speculative import (
    DraftTokenModel,
    PyTorchDraftModel,
    select_verified_token_ids,
    select_verified_tokens,
)
from peer.crypto import required_layers_for_level
from peer.model_shard import ModelShard

logger = logging.getLogger(__name__)


# Re-use the InferencePreparation dataclass from engine (or define locally).
from coordinator.engine import InferencePreparation


class InferenceService:
    """Orchestrates inference requests: preparation, execution, streaming, and chat.

    Parameters
    ----------
    config:
        An ``EngineConfig`` (or duck-typed equivalent).
    discovery_service:
        A ``DiscoveryService`` for peer discovery and model catalog queries.
    pipeline_service:
        A ``PipelineService`` for pipeline assembly and bandwidth asymmetry.
    kv_affinity_service:
        A ``KvAffinityService`` for KV-cache affinity lookups/updates.
    health:
        A ``HealthScorer`` instance.
    ledger:
        A barter credit ledger.
    hydra:
        The HYDRA token economy instance.
    ledger_bridge:
        An ``OpenHydraLedgerBridge``.
    verifier:
        A ``MysteryShopper`` for verification.
    draft_model:
        A ``DraftTokenModel`` for speculative decoding.
    grounding_client:
        A ``GroundingClient`` for RAG grounding.
    replication_monitor:
        A ``ReplicationMonitor``.
    transport_config:
        A ``TransportConfig`` for gRPC transport.
    _metrics_lock:
        A ``threading.Lock`` protecting counter variables.
    """

    def __init__(
        self,
        config: Any,
        discovery_service: Any,
        pipeline_service: Any,
        kv_affinity_service: Any,
        health: Any,
        ledger: Any,
        hydra: Any,
        ledger_bridge: Any,
        verifier: Any,
        draft_model: DraftTokenModel,
        grounding_client: Any,
        replication_monitor: Any,
        transport_config: Any,
        _metrics_lock: Any,
        _kv_store_ops_total_ref: list[int],
        _kv_retrieve_ops_total_ref: list[int],
        _inference_requests_total_ref: list[int],
        _tokenizer_cache: dict[str, Any],
        _pytorch_draft_model_cache: dict[tuple[str, str], PyTorchDraftModel],
        _last_verification_qos: dict[str, Any],
        _last_scored_peers: list,
        engine: Any = None,
    ) -> None:
        self._engine = engine
        self.config = config
        self.discovery_service = discovery_service
        self.pipeline_service = pipeline_service
        self.kv_affinity_service = kv_affinity_service
        self.health = health
        self.ledger = ledger
        self.hydra = hydra
        self.ledger_bridge = ledger_bridge
        self.verifier = verifier
        self.draft_model = draft_model
        self.grounding_client = grounding_client
        self.replication_monitor = replication_monitor
        self.transport_config = transport_config
        self._metrics_lock = _metrics_lock
        self._kv_store_ops_total_ref = _kv_store_ops_total_ref
        self._kv_retrieve_ops_total_ref = _kv_retrieve_ops_total_ref
        self._inference_requests_total_ref = _inference_requests_total_ref
        self._tokenizer_cache = _tokenizer_cache
        self._pytorch_draft_model_cache = _pytorch_draft_model_cache
        self._last_verification_qos = _last_verification_qos
        self._last_scored_peers = _last_scored_peers

    # ------------------------------------------------------------------
    # Helpers re-used from engine (thin wrappers / delegates)
    # ------------------------------------------------------------------

    def _normalize_decode_controls(
        self,
        *,
        decode_do_sample: bool | None = None,
        decode_temperature: float | None = None,
        decode_top_p: float | None = None,
        decode_top_k: int | None = None,
        decode_seed: int | None = None,
    ) -> dict[str, Any]:
        """Normalize and clamp decode control parameters into a clean dict.

        Only includes keys whose values were explicitly provided (non-None).

        Returns:
            Dict of validated decode control parameters.
        """
        out: dict[str, Any] = {}
        if decode_do_sample is not None:
            out["decode_do_sample"] = bool(decode_do_sample)
        if decode_temperature is not None:
            out["decode_temperature"] = max(1e-5, float(decode_temperature))
        if decode_top_p is not None:
            out["decode_top_p"] = max(0.0, min(1.0, float(decode_top_p)))
        if decode_top_k is not None:
            out["decode_top_k"] = max(0, int(decode_top_k))
        if decode_seed is not None and int(decode_seed) > 0:
            out["decode_seed"] = int(decode_seed)
        return out

    @staticmethod
    def _normalize_decode_kwarg_aliases(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Map short-form decode kwargs (e.g. ``temperature``) to canonical names."""
        out = dict(kwargs)
        alias_map = {
            "do_sample": "decode_do_sample",
            "temperature": "decode_temperature",
            "top_p": "decode_top_p",
            "top_k": "decode_top_k",
            "seed": "decode_seed",
        }
        for alias_key, decode_key in alias_map.items():
            if alias_key in out and decode_key not in out:
                out[decode_key] = out[alias_key]
            out.pop(alias_key, None)
        return out

    @staticmethod
    def _pipeline_uses_pytorch_runtime(pipeline: list[PeerEndpoint]) -> bool:
        """Return True if every peer uses a real model backend (PyTorch or MLX).

        The autoregressive sharded decode loop is needed for any backend
        that returns one token per forward call through a sharded pipeline.
        Both PyTorch and MLX sharded runtimes require this.
        """
        if not pipeline:
            return False
        _REAL_BACKENDS = {"pytorch", "mlx"}
        backends = [str(peer.runtime_backend or "").strip().lower() for peer in pipeline]
        return bool(backends) and all(
            any(item.startswith(prefix) for prefix in _REAL_BACKENDS)
            for item in backends
        )

    @staticmethod
    def _collect_eos_token_ids(tokenizer: Any) -> tuple[set[int], set[int]]:
        """Extract (eos_token_ids, all_special_ids) from a HuggingFace tokenizer.

        Returns two sets of non-negative ints. ``eos_token_ids`` covers single
        ``eos_token_id`` plus list-valued eos specifications. ``special_ids``
        is the tokenizer's ``all_special_ids`` used to filter out BOS/PAD/etc.
        during decode. Both paths (non-streaming ``infer()`` outer loop and
        streaming ``infer_chat_stream``) use this helper so they stay in sync.
        """
        eos_ids: set[int] = set()
        special_ids: set[int] = set()
        if tokenizer is None:
            return eos_ids, special_ids
        eos_raw = getattr(tokenizer, "eos_token_id", None)
        if isinstance(eos_raw, int):
            eos_ids = {int(eos_raw)}
        elif eos_raw is not None:
            try:
                eos_ids = {int(item) for item in list(eos_raw)}
            except TypeError:
                eos_ids = set()
        eos_ids = {item for item in eos_ids if item >= 0}
        special_raw = getattr(tokenizer, "all_special_ids", None)
        if special_raw is not None:
            try:
                special_ids = {int(item) for item in list(special_raw) if int(item) >= 0}
            except TypeError:
                special_ids = set()
        return eos_ids, special_ids

    @staticmethod
    def _rotate(values: list, offset: int) -> list:
        if not values:
            return values
        offset = offset % len(values)
        return values[offset:] + values[:offset]

    @staticmethod
    def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
        """Convert a list of chat messages to a plain ``role: content`` prompt."""
        lines = []
        for item in messages:
            role = str(item.get("role", "user")).strip() or "user"
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _default_trust_remote_code(self, model_id: str) -> bool:
        return "qwen" in str(model_id or "").strip().lower()

    def _load_generation_tokenizer(self, model_id: str) -> Any:
        normalized = str(model_id or self.config.pytorch_generation_model_id).strip() or "gpt2"
        cached = self._tokenizer_cache.get(normalized)
        if cached is not None:
            return cached
        try:
            from transformers import AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "pytorch_generation_tokenizer_unavailable: install optional dependency 'transformers'"
            ) from exc
        # Try local-first to avoid HF Hub HEAD requests (saves ~800ms/call).
        # Fall back to network if local cache doesn't have the model yet.
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                normalized,
                trust_remote_code=self._default_trust_remote_code(normalized),
                local_files_only=True,
            )
        except OSError:
            tokenizer = AutoTokenizer.from_pretrained(
                normalized,
                trust_remote_code=self._default_trust_remote_code(normalized),
            )
        self._tokenizer_cache[normalized] = tokenizer
        return tokenizer

    def _resolve_runtime_model_id(self, model_id: str) -> str:
        requested = str(model_id or "").strip()
        if not requested:
            return str(self.config.pytorch_generation_model_id or "gpt2").strip() or "gpt2"
        catalog_hf = self._engine._catalog_hf_model_id(requested)
        if catalog_hf:
            return catalog_hf
        if "/" in requested:
            return requested
        return str(self.config.pytorch_generation_model_id or "gpt2").strip() or "gpt2"

    def _resolve_pipeline_runtime_model_id(self, pipeline: list[PeerEndpoint], served_model: str) -> str:
        for peer in pipeline:
            runtime_model_id = str(getattr(peer, "runtime_model_id", "") or "").strip()
            if runtime_model_id:
                return runtime_model_id
        return self._engine._resolve_runtime_model_id(served_model)

    def _messages_to_model_prompt(
        self,
        messages: list[dict[str, Any]],
        *,
        model_id: str | None = None,
    ) -> str:
        """Convert chat messages to a model-specific prompt using chat templates.

        Attempts to use the model's tokenizer chat template for proper
        formatting; falls back to the plain ``role: content`` format.

        Args:
            messages: List of chat message dicts with ``role`` and ``content``.
            model_id: Optional model ID to resolve the tokenizer for.

        Returns:
            Formatted prompt string.
        """
        fallback = self._messages_to_prompt(messages)
        if not bool(self.config.pytorch_chat_template_enabled):
            return fallback

        requested_model = str(model_id or self.config.default_model).strip() or self.config.default_model
        runtime_model = self._engine._resolve_runtime_model_id(requested_model)
        try:
            tokenizer = self._engine._load_generation_tokenizer(runtime_model)
        except Exception:
            return fallback

        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        if not callable(apply_chat_template):
            return fallback

        templated_messages: list[dict[str, str]] = []
        for item in messages:
            role = str(item.get("role", "user")).strip().lower() or "user"
            if role not in {"system", "user", "assistant", "tool"}:
                role = "user"
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            templated_messages.append({"role": role, "content": content})
        if not templated_messages:
            return fallback

        try:
            templated = apply_chat_template(
                templated_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            try:
                templated = apply_chat_template(templated_messages, tokenize=False)
            except Exception:
                return fallback
        except Exception:
            return fallback

        rendered = str(templated or "")
        if rendered.strip():
            return rendered
        return fallback

    def _load_pytorch_draft_model(self, *, tokenizer_model_id: str) -> PyTorchDraftModel:
        draft_model_id = str(self.config.pytorch_speculative_draft_model_id or "sshleifer/tiny-gpt2").strip() or "sshleifer/tiny-gpt2"
        tokenizer_model = str(tokenizer_model_id or self.config.pytorch_generation_model_id or "gpt2").strip() or "gpt2"
        key = (draft_model_id, tokenizer_model)
        cached = self._pytorch_draft_model_cache.get(key)
        if cached is not None:
            return cached
        model = PyTorchDraftModel(
            model_id=draft_model_id,
            tokenizer_model_id=tokenizer_model,
            target="cpu",
        )
        self._pytorch_draft_model_cache[key] = model
        return model

    # ------------------------------------------------------------------
    # MoE geo-sharding helpers (copied verbatim from engine)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_expert_tags(raw: Any) -> list[str]:
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

    def _extract_prompt_expert_tags(self, prompt: str) -> list[str]:
        if not self.config.moe_geo_prompt_hints_enabled:
            return []
        tags = re.findall(r"expert:([a-z0-9][a-z0-9_-]{0,31})", str(prompt).lower())
        return self._normalize_expert_tags(tags)

    def _extract_prompt_expert_layer_indices(self, prompt: str) -> list[int]:
        if not self.config.moe_geo_prompt_hints_enabled:
            return []
        layers = re.findall(r"(?:expert[-_]?layer|layer):([0-9]{1,5})", str(prompt).lower())
        return self._normalize_expert_layer_indices(layers)

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

    # ------------------------------------------------------------------
    # _run_chain
    # ------------------------------------------------------------------

    def _run_chain(
        self,
        prompt: str,
        candidates: list,
        pipeline: list,
        max_tokens: int,
        request_id: str | None = None,
        initial_activation: list[float] | None = None,
        kv_session_id: str | None = None,
        kv_use_cached_activation: bool = False,
        kv_store_activation: bool = False,
        kv_cache_stage_index: int = 0,
        kv_cache_all_stages: bool = False,
        decode_do_sample: bool | None = None,
        decode_temperature: float | None = None,
        decode_top_p: float | None = None,
        decode_top_k: int | None = None,
        decode_seed: int | None = None,
        deadline: float | None = None,
    ):
        """Build and execute an InferenceChain through the gRPC pipeline.

        Constructs the chain with the configured transport, encryption, and
        compression settings, runs it with failover support, and records
        health outcomes for each peer.  Also increments KV and inference
        proxy counters.

        Args:
            prompt: The text prompt to send through the pipeline.
            candidates: Full candidate pool for failover.
            pipeline: Ordered peers forming the active pipeline.
            max_tokens: Maximum tokens to generate.
            request_id: Optional unique request identifier.
            initial_activation: Optional activation seed vector.
            kv_session_id: Optional KV cache session identifier.
            kv_use_cached_activation: Whether to use cached activations.
            kv_store_activation: Whether to store activations in the cache.
            kv_cache_stage_index: Pipeline stage index for KV caching.
            kv_cache_all_stages: Whether to cache at all pipeline stages.
            decode_do_sample: Enable sampling during decoding.
            decode_temperature: Sampling temperature.
            decode_top_p: Nucleus sampling threshold.
            decode_top_k: Top-k sampling parameter.
            decode_seed: Random seed for reproducible sampling.
            deadline: Absolute deadline (epoch seconds) for the request.

        Returns:
            A ``ChainResult`` with generated text, traces, and metadata.
        """
        # Resolve InferenceChain through the engine module so that
        # monkeypatching coordinator.engine.InferenceChain works in tests.
        import coordinator.engine as _engine_mod
        _InferenceChain = getattr(_engine_mod, "InferenceChain", InferenceChain)
        # Phase B: create InferenceSession for history replay on failover
        _session = None
        if self.config.streaming_sessions_enabled and kv_session_id:
            from coordinator.stream_pool import InferenceSession
            _session = InferenceSession(
                session_id=str(kv_session_id),
                model_id=str(pipeline[0].model_id or "") if pipeline else "",
                prompt=prompt,
            )

        chain = _InferenceChain(
            pipeline,
            timeout_ms=self.config.timeout_ms,
            transport_config=self.transport_config,
            tensor_autoencoder_enabled=self.config.tensor_autoencoder_enabled,
            tensor_autoencoder_latent_dim=max(1, int(self.config.tensor_autoencoder_latent_dim)),
            advanced_encryption_enabled=self.config.advanced_encryption_enabled,
            advanced_encryption_seed=str(self.config.advanced_encryption_seed),
            advanced_encryption_level=str(self.config.advanced_encryption_level),
            activation_quantization_enabled=self.config.activation_quantization_enabled,
            stream_pool=getattr(self, "_stream_pool", None),
            session=_session,
        )
        run_kwargs: dict[str, Any] = {
            "max_tokens": max_tokens,
            "request_id": request_id,
            "failover_pool": candidates,
            "max_failovers_per_stage": self.config.max_failovers_per_stage,
            "deadline": deadline,
        }
        if initial_activation:
            run_kwargs["initial_activation"] = list(initial_activation)
        if kv_session_id:
            run_kwargs.update(
                {
                    "kv_session_id": str(kv_session_id),
                    "kv_use_cached_activation": bool(kv_use_cached_activation),
                    "kv_store_activation": bool(kv_store_activation),
                    "kv_cache_stage_index": int(kv_cache_stage_index),
                    "kv_cache_all_stages": bool(kv_cache_all_stages),
                }
            )
        decode_controls = self._normalize_decode_controls(
            decode_do_sample=decode_do_sample,
            decode_temperature=decode_temperature,
            decode_top_p=decode_top_p,
            decode_top_k=decode_top_k,
            decode_seed=decode_seed,
        )
        if decode_controls:
            run_kwargs.update(decode_controls)

        # -- Phase D: increment inference + KV proxy counters --
        with self._metrics_lock:
            self._inference_requests_total_ref[0] += 1
            if kv_session_id:
                if kv_store_activation:
                    self._kv_store_ops_total_ref[0] += 1
                if kv_use_cached_activation:
                    self._kv_retrieve_ops_total_ref[0] += 1
        # --

        # Use push mode when enabled and pipeline has 2+ stages.
        # Push mode eliminates coordinator-mediated round-trips between stages.
        if (
            self.config.push_mode_enabled
            and len(pipeline) >= 2
            and self.config.push_callback_address
        ):
            result = chain.run_push(
                prompt,
                max_tokens=max_tokens,
                request_id=request_id,
                deadline=deadline,
                kv_session_id=kv_session_id,
                kv_store_activation=kv_store_activation,
                kv_use_cached_activation=kv_use_cached_activation,
                callback_address=self.config.push_callback_address,
                initial_activation=initial_activation,
                **decode_controls,
            )
        else:
            result = chain.run(
                prompt,
                **run_kwargs,
            )

        for trace in result.traces:
            self.health.record_inference(trace.peer_id, success=True, latency_ms=trace.latency_ms)
            if trace.failed_peer_id:
                self.health.record_inference(trace.failed_peer_id, success=False)

        return result

    # ------------------------------------------------------------------
    # Verification feedback
    # ------------------------------------------------------------------

    def _apply_verification_feedback(
        self,
        *,
        primary: ChainResult,
        secondary: ChainResult | None,
        tertiary: ChainResult | None,
        verification: Any,
    ) -> dict[str, list[str]]:
        """Apply verification outcomes to peer health and stake.

        Rewards peers matching the winner, penalises divergent ones with
        stake slashing or aggressive reputation penalties.

        Args:
            primary: The primary inference chain result.
            secondary: Optional secondary chain result.
            tertiary: Optional tertiary chain result.
            verification: The mystery-shopper verification outcome.

        Returns:
            Dict with ``rewarded_peers`` and ``penalized_peers`` lists.
        """
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
    # Replication + discovered peer helpers
    # ------------------------------------------------------------------

    def _replication_dict(self, model_id: str, healthy_peers: int) -> dict[str, Any]:
        """Evaluate and serialize replication status for a model."""
        status = self.replication_monitor.evaluate(
            model_id,
            healthy_peers,
            required_replicas=self._engine._required_replicas(model_id),
        )
        return self.replication_monitor.to_dict(status)

    def _discovered_peer_rows(self, health) -> list[dict[str, Any]]:
        """Build detailed per-peer row dicts for API serialization."""
        scored_lookup = {item.peer.peer_id: item for item in self._last_scored_peers}
        rows = []
        for item in health:
            scored = scored_lookup.get(item.peer.peer_id)
            model_id = self._engine._normalize_peer_model(item.peer)
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
                    "bandwidth_role": self._engine._role_for_peer(item.peer),
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
    # Grounding + model metadata
    # ------------------------------------------------------------------

    def _grounding_meta(self, *, enabled: bool, snippets: list[str], grounding_result: Any | None) -> dict[str, Any]:
        """Build the grounding metadata dict for the inference response."""
        return {
            "enabled": enabled,
            "snippets": snippets,
            "provider": (grounding_result.provider if grounding_result else "disabled"),
            "cached": (grounding_result.cached if grounding_result else False),
            "fallback_used": (grounding_result.fallback_used if grounding_result else False),
            "error": (grounding_result.error if grounding_result else None),
        }

    def _model_meta(self, decision: DegradationDecision) -> dict[str, Any]:
        """Build the model metadata dict including degradation and QoS info."""
        return {
            "requested": decision.requested_model,
            "served": decision.served_model,
            "degraded": decision.degraded,
            "available": decision.available,
            "reason": decision.reason,
            "detail": decision.detail,
            "verification_qos": dict(self._last_verification_qos),
        }

    # ------------------------------------------------------------------
    # _prepare_inference
    # ------------------------------------------------------------------

    def _prepare_inference(
        self,
        *,
        prompt: str,
        pipeline_width: int | None,
        grounding: bool,
        model_id: str | None,
        allow_degradation: bool | None,
        session_id: str | None,
        expert_tags: list[str] | None = None,
        expert_layer_indices: list[int] | None = None,
    ) -> InferencePreparation:
        """Prepare everything needed before running an inference chain.

        Performs grounding injection, peer discovery, pipeline assembly
        (sharded or full-model), bandwidth asymmetry reordering, and MoE
        geo-sharding.

        Args:
            prompt: The raw user prompt.
            pipeline_width: Desired pipeline width (peers per pipeline).
            grounding: Whether to enable RAG grounding.
            model_id: Requested model identifier.
            allow_degradation: Whether model degradation fallback is allowed.
            session_id: Optional session ID for KV affinity.
            expert_tags: Optional explicit expert tags for MoE routing.
            expert_layer_indices: Optional explicit layer indices for MoE routing.

        Returns:
            An ``InferencePreparation`` dataclass with all assembled state.
        """
        requested_model = model_id or self.config.default_model
        allow_deg = self.config.allow_degradation_default if allow_degradation is None else bool(allow_degradation)

        effective_prompt = prompt
        grounding_result = None
        snippets: list[str] = []
        if grounding:
            from grounding.client_rag import inject_grounding
            grounding_result = self.grounding_client.search(prompt, max_snippets=3)
            snippets = grounding_result.snippets
            effective_prompt = inject_grounding(prompt, snippets)

        health, candidates, decision, counts = self._engine._discover_for_model(
            requested_model=requested_model,
            allow_degradation=allow_deg,
        )
        prompt_tokens_est = estimate_prompt_tokens(effective_prompt)

        # -- Phase 3: try layer-sharded pipeline first --
        # When the healthy peer set forms complete layer coverage the coordinator
        # assembles an ordered shard pipeline and bypasses the full-model peer
        # selection, bandwidth asymmetry, and MoE geo-sharding steps (which all
        # assume every peer runs the complete model).
        _sharded_pipeline = self._engine._select_pipeline_sharded(health)
        if _sharded_pipeline is not None:
            return InferencePreparation(
                effective_prompt=effective_prompt,
                snippets=snippets,
                grounding_result=grounding_result,
                health=health,
                candidates=candidates,
                decision=decision,
                counts=counts,
                primary_pipeline=_sharded_pipeline,
                primary_bandwidth_policy={},
                primary_moe_policy={},
                pipeline_mode="sharded",
            )
        # -- Legacy: full-model peer pipeline --

        primary_pipeline = self._engine._select_pipeline(candidates, pipeline_width=pipeline_width)
        primary_pipeline, primary_bandwidth_policy = self._engine._apply_bandwidth_asymmetry(
            primary_pipeline,
            candidates,
            prompt_tokens_est=prompt_tokens_est,
            session_id=session_id,
            model_id=decision.served_model,
        )
        locked_prefill_peer_id = (
            str(primary_bandwidth_policy.get("prefill_peer_id", "")).strip()
            if bool(primary_bandwidth_policy.get("prefill_required"))
            else None
        )
        primary_pipeline, primary_moe_policy = self._engine._apply_moe_geo_sharding(
            primary_pipeline,
            candidates,
            prompt=effective_prompt,
            requested_expert_tags=expert_tags,
            requested_expert_layer_indices=expert_layer_indices,
            locked_first_peer_id=locked_prefill_peer_id,
        )

        return InferencePreparation(
            effective_prompt=effective_prompt,
            snippets=snippets,
            grounding_result=grounding_result,
            health=health,
            candidates=candidates,
            decision=decision,
            counts=counts,
            primary_pipeline=primary_pipeline,
            primary_bandwidth_policy=primary_bandwidth_policy,
            primary_moe_policy=primary_moe_policy,
            pipeline_mode="full_model",
        )

    # ------------------------------------------------------------------
    # infer (single-shot)
    # ------------------------------------------------------------------

    def infer(
        self,
        *,
        prompt: str,
        max_tokens: int = 24,
        pipeline_width: int | None = None,
        grounding: bool = True,
        priority: bool = False,
        client_id: str = "anonymous",
        model_id: str | None = None,
        allow_degradation: bool | None = None,
        session_id: str | None = None,
        expert_tags: list[str] | None = None,
        expert_layer_indices: list[int] | None = None,
        decode_do_sample: bool | None = None,
        decode_temperature: float | None = None,
        decode_top_p: float | None = None,
        decode_top_k: int | None = None,
        decode_seed: int | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute a single-shot inference request with full verification.

        Runs the primary pipeline, then triggers mystery-shopper verification
        (secondary and optionally tertiary redundant executions).  Applies
        verification feedback, earns barter credits, mints HYDRA rewards,
        and returns the complete response with pipeline traces and metadata.

        Args:
            prompt: The text prompt.
            max_tokens: Maximum tokens to generate (capped by elastic ceiling).
            pipeline_width: Number of peers per pipeline.
            grounding: Whether to enable RAG grounding.
            priority: Whether to spend priority credits.
            client_id: Client identifier for credit operations.
            model_id: Requested model identifier.
            allow_degradation: Whether model degradation is allowed.
            session_id: Optional session ID for KV affinity.
            expert_tags: Optional expert tags for MoE routing.
            expert_layer_indices: Optional layer indices for MoE routing.
            decode_do_sample: Enable sampling during decoding.
            decode_temperature: Sampling temperature.
            decode_top_p: Nucleus sampling threshold.
            decode_top_k: Top-k sampling parameter.
            decode_seed: Random seed for reproducible sampling.
            request_id: Optional unique request identifier.

        Returns:
            Dict with ``response``, ``pipeline``, ``verification``,
            ``model``, ``grounding``, and other metadata.

        Raises:
            RuntimeError: If priority credits are insufficient or no peers found.
            ValueError: If max_tokens exceeds the elastic ceiling.
        """
        request_id = request_id or str(uuid.uuid4())
        max_tokens = int(max_tokens or 1024)
        logger.info(
            "infer_start req_id=%s model=%s client=%s",
            request_id, model_id or self.config.default_model, client_id,
        )
        if priority and not self.ledger.spend(client_id, 1.0):
            raise RuntimeError("insufficient_priority_credits")

        # Phase 2: record demand signal for the auto-scaler.
        self.discovery_service._request_log.record(str(model_id or self.config.default_model))

        _t_prep_start = time.perf_counter()
        prep = self._engine._prepare_inference(
            prompt=prompt,
            pipeline_width=pipeline_width,
            grounding=grounding,
            model_id=model_id,
            allow_degradation=allow_degradation,
            session_id=session_id,
            expert_tags=expert_tags,
            expert_layer_indices=expert_layer_indices,
        )
        _t_prep_ms = (time.perf_counter() - _t_prep_start) * 1000
        logger.info("PROFILE phase_A_prepare_inference=%.1fms", _t_prep_ms)

        # Compute the absolute deadline AFTER discovery/preparation completes,
        # so the full latency budget is available for the actual inference chain.
        deadline = time.time() + self.config.max_latency_ms / 1000.0
        # Elastic output cap: 2048 floor, up to 8192 if redundancy >= 3.0
        _served = prep.decision.served_model
        _available = prep.counts.get(_served, 0)
        _required = self._engine._required_replicas(_served)
        _effective_redundancy = _available / max(_required, 1)
        _elastic_ceiling = 8192 if _effective_redundancy >= 3.0 else 2048
        if max_tokens > _elastic_ceiling:
            raise ValueError(
                f"Network redundancy for {_served} is currently too low for "
                f"extended context. Maximum allowed output is {_elastic_ceiling} tokens."
            )
        decode_controls = self._normalize_decode_controls(
            decode_do_sample=decode_do_sample,
            decode_temperature=decode_temperature,
            decode_top_p=decode_top_p,
            decode_top_k=decode_top_k,
            decode_seed=decode_seed,
        )
        # SpecPipe (P1-A): when enabled and pipeline has multiple stages,
        # use concurrent speculative dispatch instead of sequential chain.
        # Computed BEFORE chunked prefill so we can skip prefill when SpecPipe
        # is active — SpecPipe handles its own prompt continuation and KV cache.
        _specpipe_stats: dict[str, Any] = {}
        _use_specpipe = (
            self.config.specpipe_enabled
            and len(prep.primary_pipeline) > 1
            and prep.pipeline_mode == "sharded"
        )

        # Chunked prefill (P1-B): split long prompts into chunks so other
        # clients' requests can interleave between chunks.
        # Skip when SpecPipe is active — SpecPipe manages its own decode loop
        # with KV-cached continuation, making chunked prefill redundant and
        # wasteful (each chunk costs a full WAN pipeline traversal).
        _chunked_prefill_stats: dict[str, Any] = {}
        if (
            not _use_specpipe
            and self.config.chunked_prefill_enabled
            and len(prep.effective_prompt.split()) > self.config.chunked_prefill_chunk_size
        ):
            from coordinator.chunked_prefill import ChunkedPrefill, ChunkedPrefillConfig

            _cp = ChunkedPrefill(ChunkedPrefillConfig(
                chunk_size=self.config.chunked_prefill_chunk_size,
            ))

            def _cp_chain_fn(chunk_prompt, **kw):
                return list(self._engine._run_chain(
                    chunk_prompt,
                    prep.candidates,
                    prep.primary_pipeline,
                    max_tokens=1,
                    request_id=request_id,
                    deadline=deadline,
                    **{k: v for k, v in kw.items() if k in (
                        "kv_session_id", "kv_store_activation", "kv_use_cached_activation",
                    )},
                    **decode_controls,
                ).activation)

            _prefill_activation = _cp.process(
                prep.effective_prompt,
                chain_fn=_cp_chain_fn,
                session_id=f"cp-{request_id}",
            )
            _chunked_prefill_stats = _cp.stats
            logger.info("PROFILE chunked_prefill: %s", _chunked_prefill_stats)

        _t_chain_start = time.perf_counter()

        if _use_specpipe:
            from coordinator.specpipe_scheduler import SpecPipeScheduler, SpecPipeConfig

            # Build stage dispatch function from the chain
            import coordinator.engine as _engine_mod
            _InferenceChain = getattr(_engine_mod, "InferenceChain", InferenceChain)
            _chain = _InferenceChain(
                prep.primary_pipeline,
                timeout_ms=self.config.timeout_ms,
                transport_config=self.transport_config,
                activation_quantization_enabled=self.config.activation_quantization_enabled,
            )

            def _stage_fn(peer, activation, prompt="", stage_index=0,
                          total_stages=1, max_tokens=1, request_id="",
                          kv_session_id="", kv_store_activation=False,
                          kv_use_cached_activation=False,
                          decode_temperature=None, decode_do_sample=None,
                          decode_top_p=None, decode_top_k=None, **kw):
                result = _chain._request_stage(
                    peer=peer, request_id=request_id, prompt=prompt,
                    activation=activation, stage_index=stage_index,
                    total_stages=total_stages, max_tokens=max_tokens,
                    kv_session_id=kv_session_id or None,
                    kv_store_activation=kv_store_activation,
                    kv_use_cached_activation=kv_use_cached_activation,
                    decode_temperature=decode_temperature,
                    decode_do_sample=decode_do_sample,
                    decode_top_p=decode_top_p,
                    decode_top_k=decode_top_k,
                    deadline=deadline,
                )
                return list(result.activation) if hasattr(result, "activation") else list(result[0])

            # Use the SAME model (Qwen2.5-0.5B) as draft — ensures token
            # IDs are compatible with the sharded target pipeline.
            # The full model runs locally on the Mac for fast drafting
            # while the pipeline shards handle verification.
            def _draft_fn(ctx, k):
                try:
                    import torch
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    import os

                    # Module-level cache for the draft model
                    if not hasattr(_draft_fn, "_model"):
                        _cache = os.path.expanduser("~/.cache/openhydra/models/Qwen2.5-0.5B")
                        if not os.path.exists(os.path.join(_cache, "model.safetensors")):
                            from huggingface_hub import snapshot_download
                            snapshot_download("Qwen/Qwen2.5-0.5B", local_dir=_cache)
                        _draft_fn._tokenizer = AutoTokenizer.from_pretrained(
                            _cache, local_files_only=True
                        )
                        _draft_fn._model = AutoModelForCausalLM.from_pretrained(
                            _cache, local_files_only=True,
                            torch_dtype=torch.float16, low_cpu_mem_usage=True,
                        )
                        _draft_fn._model.eval()
                        logger.info("specpipe_draft: loaded Qwen2.5-0.5B as draft model")

                    # Build input from context token IDs or prompt
                    if ctx and any(abs(c) > 1 for c in ctx):
                        input_ids = torch.tensor(
                            [ctx[-min(512, len(ctx)):]],
                            dtype=torch.long,
                        )
                    else:
                        input_ids = _draft_fn._tokenizer(
                            _base_prompt, return_tensors="pt"
                        )["input_ids"]

                    with torch.no_grad():
                        out = _draft_fn._model.generate(
                            input_ids=input_ids,
                            max_new_tokens=k,
                            do_sample=False,
                        )
                    new_ids = out[0][input_ids.shape[1]:].tolist()
                    return new_ids[:k]
                except Exception as exc:
                    logger.warning("specpipe_draft_failed: %s", exc)
                    base = ctx[-1] if ctx else 0
                    return [(base + i + 1) % 50000 for i in range(k)]

            spec_config = SpecPipeConfig(
                enabled=True,
                max_speculative_depth=min(self.config.specpipe_max_depth, len(prep.primary_pipeline) - 1),
            )
            scheduler = SpecPipeScheduler(
                config=spec_config,
                pipeline=prep.primary_pipeline,
                draft_fn=_draft_fn,
                stage_fn=_stage_fn,
            )

            # Run specpipe rounds until max_tokens
            # Build the full prompt including previously generated tokens
            # so the model has context for continuation.
            _base_prompt = prep.effective_prompt
            _runtime_model = self._resolve_pipeline_runtime_model_id(
                prep.primary_pipeline, prep.decision.served_model,
            )
            _kv_sid = f"specpipe-{request_id}" if request_id else None

            # Use pipelined execution for 3+ stages (overlaps tokens
            # across stages for higher throughput).  Fall back to the
            # round-based loop for 2-stage pipelines where pipelining
            # overhead outweighs the benefit.
            if len(prep.primary_pipeline) >= 2:
                all_token_ids = scheduler.run_pipelined(
                    context_ids=[0],
                    prompt=_base_prompt,
                    max_tokens=max_tokens,
                    request_id=request_id,
                    kv_session_id=_kv_sid,
                )
            else:
                all_token_ids = []
                for _round in range(max(1, max_tokens)):
                    if _round == 0:
                        _full_prompt = _base_prompt
                    else:
                        _full_prompt = _base_prompt

                    context = all_token_ids if all_token_ids else [0]
                    accepted = scheduler.run_round(
                        context_ids=context,
                        prompt=_full_prompt,
                        max_tokens=max_tokens - len(all_token_ids),
                        request_id=request_id,
                        kv_session_id=_kv_sid,
                    )
                    all_token_ids.extend(accepted)
                    if len(all_token_ids) >= max_tokens:
                        break

            scheduler.shutdown()
            _specpipe_stats = {
                "rounds": scheduler.stats.rounds,
                "real_tokens": scheduler.stats.real_tokens,
                "spec_dispatched": scheduler.stats.speculative_dispatched,
                "spec_accepted": scheduler.stats.speculative_accepted,
                "effective_per_round": scheduler.stats.effective_tokens_per_round,
                "total_stage_calls": scheduler.stats.total_stage_calls,
            }
            logger.info("PROFILE specpipe: %s", _specpipe_stats)

            # Build a ChainResult from the specpipe output.
            # Use the coordinator's tokenizer directly for reliable decode
            # instead of ModelShard.decode_text() which has a 48-token cap
            # and falls back to a vocab table for non-HF models.
            _runtime_model = self._resolve_pipeline_runtime_model_id(
                prep.primary_pipeline, prep.decision.served_model,
            )
            logger.info(
                "specpipe_decode: token_ids=%s model=%s",
                all_token_ids[:10], _runtime_model,
            )
            try:
                _decode_tok = self._engine._load_generation_tokenizer(_runtime_model)
                output_text = _decode_tok.decode(
                    all_token_ids, skip_special_tokens=True,
                ).strip()
            except Exception:
                output_text = ModelShard.decode_text(
                    [float(t) for t in all_token_ids],
                    max_tokens=max_tokens,
                    tokenizer_model_id=_runtime_model,
                )
            primary = ChainResult(
                request_id=request_id or str(uuid.uuid4()),
                text=output_text,
                activation=[float(t) for t in all_token_ids],
                traces=[],
                latency_ms=(time.perf_counter() - _t_chain_start) * 1000,
            )
        elif (
            self.config.autoregressive_sharded_enabled
            and len(prep.primary_pipeline) > 1
            and self._pipeline_uses_pytorch_runtime(prep.primary_pipeline)
        ):
            # Phase 1A + Phase 6 fix: non-streaming sharded PyTorch pipelines
            # need an outer decode loop because the last-stage peer only
            # returns ONE token per chain.run() call (it can't autoregress
            # alone — it doesn't hold layers 0..N-1).
            #
            # Two code paths live here:
            #
            # (1) KV-aware (preferred) — prefill once with the full context
            #     and ``kv_store_activation=True``, then each subsequent
            #     decode step ships ONLY the previous token and reads the
            #     peer-side KV cache via ``kv_use_cached_activation=True``.
            #     Cost per generated token is a single layer-stack forward
            #     on [1 token] at every peer → O(1) per token regardless of
            #     context length. Enabled by default because Phase 2's
            #     Identity-replacement pass stops ``DynamicCache`` iterations
            #     from dispatching on meta tensors.
            #
            # (2) Stateless re-prefill (fallback) — if any KV-aware call
            #     raises (cache miss, unsupported cache dtype, etc.), we
            #     transparently rebuild from scratch at every decode step
            #     with the full context. This was the original Phase 1 path.
            #     It's O(N²) in output length but correct without depending
            #     on any peer-side state.
            _runtime_model = self._resolve_pipeline_runtime_model_id(
                prep.primary_pipeline, prep.decision.served_model,
            )
            try:
                _ar_tokenizer = self._engine._load_generation_tokenizer(_runtime_model)
            except Exception as _tok_exc:
                logger.warning(
                    "autoregressive_sharded_tokenizer_unavailable: %s — falling back to single chain.run()",
                    _tok_exc,
                )
                _ar_tokenizer = None

            if _ar_tokenizer is not None:
                _ar_eos_ids, _ar_special_ids = self._collect_eos_token_ids(_ar_tokenizer)
                try:
                    _ar_context_ids = [
                        int(t) for t in _ar_tokenizer.encode(
                            prep.effective_prompt, add_special_tokens=True,
                        )
                    ]
                except Exception:
                    _ar_context_ids = []
                if not _ar_context_ids:
                    _ar_context_ids = [
                        int(next(iter(_ar_eos_ids))) if _ar_eos_ids else 0
                    ]

                _ar_generated: list[int] = []
                _ar_traces: list[Any] = []
                _ar_activation_hash = b""
                _ar_last_kv: dict[str, Any] | None = None
                _ar_last_compression: dict[str, Any] | None = None
                _ar_last_encryption: dict[str, Any] | None = None
                _ar_total_latency_ms = 0.0
                _ar_target_tokens = max(1, int(max_tokens))
                # KV session ID — per-request so concurrent requests don't
                # stomp on each other's peer-side caches. Falls back to a
                # synthetic UUID if the caller didn't pass session_id.
                _ar_kv_session = str(session_id or request_id or uuid.uuid4())
                _ar_kv_mode = "kv_aware"
                logger.info(
                    "autoregressive_sharded_start: stages=%d prompt_tokens=%d target=%d mode=%s session=%s",
                    len(prep.primary_pipeline), len(_ar_context_ids),
                    _ar_target_tokens, _ar_kv_mode, _ar_kv_session[:12],
                )

                def _step_stateless(all_token_ids: list[int]):
                    """Full re-prefill fallback. Sends the full context every
                    call and asks for 1 token back. Used when KV-aware path
                    errors or when the peer doesn't support the KV session."""
                    return self._engine._run_chain(
                        "",
                        prep.candidates,
                        prep.primary_pipeline,
                        max_tokens=1,
                        request_id=request_id,
                        deadline=deadline,
                        initial_activation=[float(t) for t in all_token_ids],
                        **decode_controls,
                    )

                def _step_prefill():
                    """KV-aware prefill: ship the full context AND tell the
                    peer to store the resulting K/V cache under our session
                    ID. All subsequent calls will reuse this state."""
                    return self._engine._run_chain(
                        "",
                        prep.candidates,
                        prep.primary_pipeline,
                        max_tokens=1,
                        request_id=request_id,
                        deadline=deadline,
                        initial_activation=[float(t) for t in _ar_context_ids],
                        kv_session_id=_ar_kv_session,
                        kv_store_activation=True,
                        kv_use_cached_activation=False,
                        kv_cache_stage_index=0,
                        kv_cache_all_stages=True,
                        **decode_controls,
                    )

                def _step_decode(prev_token: int):
                    """KV-aware decode step: ship ONLY the just-generated
                    token. Peer reads its cached K/V for positions [0..N-1],
                    computes K/V for position N on the fly, appends to the
                    cache (kv_store_activation=True), returns a single
                    sampled next-token id."""
                    return self._engine._run_chain(
                        "",
                        prep.candidates,
                        prep.primary_pipeline,
                        max_tokens=1,
                        request_id=request_id,
                        deadline=deadline,
                        initial_activation=[float(prev_token)],
                        kv_session_id=_ar_kv_session,
                        kv_store_activation=True,
                        kv_use_cached_activation=True,
                        kv_cache_stage_index=0,
                        kv_cache_all_stages=True,
                        **decode_controls,
                    )

                _prefill_done = False
                _kv_consecutive_failures = 0
                _KV_MAX_RETRIES = 3  # retry KV-aware path up to 3 times before permanent fallback
                for _ar_step in range(_ar_target_tokens):
                    try:
                        if _ar_kv_mode == "kv_aware":
                            if not _prefill_done:
                                _step_result = _step_prefill()
                                _prefill_done = True
                            else:
                                # We have at least one generated token if we're here
                                # past step 0; feed the last one into the decode path.
                                _step_result = _step_decode(_ar_generated[-1])
                            _kv_consecutive_failures = 0  # reset on success
                        else:
                            _step_result = _step_stateless(_ar_context_ids + _ar_generated)
                    except RuntimeError as _kv_exc:
                        if _ar_kv_mode == "kv_aware":
                            _kv_consecutive_failures += 1
                            _is_transient = "deadline_exceeded" in str(_kv_exc).lower() or "deadline exceeded" in str(_kv_exc).lower()
                            if _is_transient and _kv_consecutive_failures < _KV_MAX_RETRIES:
                                # Transient timeout — retry KV-aware on the next
                                # step. The KV cache is still valid on the peer;
                                # only fall back to stateless if the cache is
                                # actually lost (kv_cache_miss) or we hit max retries.
                                logger.warning(
                                    "autoregressive_sharded_kv_transient: step=%d attempt=%d err=%s — retrying kv_aware",
                                    _ar_step, _kv_consecutive_failures, str(_kv_exc)[:120],
                                )
                                # Re-prefill to recover: the peer's KV cache may
                                # be stale after a timeout, so do a fresh prefill
                                # with the full context built so far, then resume
                                # KV-aware decode from the next step.
                                try:
                                    _step_result = _step_stateless(
                                        _ar_context_ids + _ar_generated
                                    )
                                    _prefill_done = False  # force re-prefill on next KV step
                                except Exception:
                                    _ar_kv_mode = "stateless"
                                    raise
                            else:
                                # Permanent failure (kv_cache_miss or too many retries).
                                logger.warning(
                                    "autoregressive_sharded_kv_fallback: step=%d err=%s — switching to stateless",
                                    _ar_step, str(_kv_exc)[:200],
                                )
                                _ar_kv_mode = "stateless"
                                try:
                                    _step_result = _step_stateless(
                                        _ar_context_ids + _ar_generated
                                    )
                                except Exception as _fallback_exc:
                                    logger.error(
                                        "autoregressive_sharded_stateless_fallback_failed: %s",
                                        _fallback_exc,
                                    )
                                    raise
                        else:
                            raise

                    _ar_total_latency_ms += float(_step_result.latency_ms or 0.0)
                    _ar_traces.extend(list(_step_result.traces or []))
                    if _step_result.kv is not None:
                        _ar_last_kv = _step_result.kv
                    if _step_result.compression is not None:
                        _ar_last_compression = _step_result.compression
                    if _step_result.encryption is not None:
                        _ar_last_encryption = _step_result.encryption
                    if _step_result.activation_hash:
                        _ar_activation_hash = _step_result.activation_hash
                    if not _step_result.activation:
                        break
                    _next_id = int(round(float(_step_result.activation[0])))
                    if _ar_eos_ids and _next_id in _ar_eos_ids:
                        break
                    _ar_generated.append(_next_id)

                try:
                    _ar_text = _ar_tokenizer.decode(
                        _ar_generated, skip_special_tokens=True,
                    )
                except Exception:
                    _ar_text = ""

                primary = ChainResult(
                    request_id=request_id or str(uuid.uuid4()),
                    text=_ar_text,
                    activation=[float(t) for t in _ar_generated],
                    traces=_ar_traces,
                    latency_ms=_ar_total_latency_ms,
                    compression=_ar_last_compression,
                    encryption=_ar_last_encryption,
                    kv=_ar_last_kv,
                    activation_hash=_ar_activation_hash,
                )
                logger.info(
                    "autoregressive_sharded_done: tokens=%d latency_ms=%.1f mode=%s text=%r",
                    len(_ar_generated), _ar_total_latency_ms, _ar_kv_mode, _ar_text[:120],
                )
            else:
                # Tokenizer unavailable — fall through to the single chain.run() path.
                primary = self._engine._run_chain(
                    prep.effective_prompt,
                    prep.candidates,
                    prep.primary_pipeline,
                    max_tokens=max_tokens,
                    request_id=request_id,
                    deadline=deadline,
                    **decode_controls,
                )
        else:
            primary = self._engine._run_chain(
                prep.effective_prompt,
                prep.candidates,
                prep.primary_pipeline,
                max_tokens=max_tokens,
                request_id=request_id,
                deadline=deadline,
                **decode_controls,
            )

        _t_chain_ms = (time.perf_counter() - _t_chain_start) * 1000
        logger.info("PROFILE phase_B_run_chain=%.1fms", _t_chain_ms)

        # Post-chain decode fix: if the chain returned ToyRuntime vocab text
        # but the peer used a real runtime, re-decode using the coordinator's
        # cached tokenizer.  This fixes the coordinator-side decode path
        # where ModelShard.decode_text() falls back to _VOCAB.
        # Post-chain decode fix: re-decode activation as real token IDs when
        # the peer used a real runtime (PyTorch/MLX).  Only triggers when
        # activations look like token IDs (values > 1, integers).
        _has_real_token_ids = (
            primary.activation
            and len(primary.activation) >= 1
            and any(abs(v) > 1.5 for v in primary.activation[:5])
        )
        if _has_real_token_ids and prep.primary_pipeline:
            _last_peer = prep.primary_pipeline[-1]
            _peer_backend = str(getattr(_last_peer, "runtime_backend", "")).strip().lower()
            if _peer_backend.startswith("pytorch") or _peer_backend == "mlx" or _peer_backend == "tinyllama":
                _runtime_model = self._resolve_pipeline_runtime_model_id(
                    prep.primary_pipeline, prep.decision.served_model,
                )
                try:
                    _tok = self._engine._load_generation_tokenizer(_runtime_model)
                    _token_ids = [max(0, int(round(float(v)))) for v in primary.activation]
                    _redecoded = _tok.decode(_token_ids, skip_special_tokens=True).strip()
                    if _redecoded:
                        primary = ChainResult(
                            request_id=primary.request_id,
                            text=_redecoded,
                            activation=primary.activation,
                            traces=primary.traces,
                            latency_ms=primary.latency_ms,
                            compression=primary.compression,
                            encryption=primary.encryption,
                            kv=primary.kv,
                        )
                except Exception:
                    pass  # Tokenizer unavailable — keep chain's decode

        prompt_tokens_est = estimate_prompt_tokens(prep.effective_prompt)

        # ── Verification bypass: skip when fewer than 2 distinct peers ──
        # On single-node deployments the "mystery shopper" re-executes on
        # the same peer — pure waste (19+ seconds).  Only verify when a
        # genuinely different peer can produce an independent result.
        _unique_peer_ids = {p.peer_id for p in prep.candidates}
        _skip_verification = len(_unique_peer_ids) < 2

        secondary_result: ChainResult | None = None
        tertiary_result: ChainResult | None = None

        # TOPLOC fast path: use activation hash when available (P2-B)
        _has_toploc_hash = bool(getattr(primary, "activation_hash", b""))
        if _skip_verification:
            logger.info(
                "PROFILE phase_C_verification=SKIPPED (single-peer topology, %d unique peers)",
                len(_unique_peer_ids),
            )
            verification = self.verifier.build_skip_result(primary)
        elif _has_toploc_hash:
            # TOPLOC: verify via hash instead of re-execution (P2-B)
            logger.info("PROFILE phase_C_verification=TOPLOC (hash-based, no re-execution)")
            verification = self.verifier.build_hash_verified_result(
                primary=primary,
                activation_hash=primary.activation_hash,
            )
        else:
            secondary_pipeline = self._engine._select_pipeline(self._rotate(prep.candidates, 1), pipeline_width=pipeline_width)
            secondary_pipeline, secondary_bandwidth_policy = self._engine._apply_bandwidth_asymmetry(
                secondary_pipeline,
                self._rotate(prep.candidates, 1),
                prompt_tokens_est=prompt_tokens_est,
            )
            secondary_pipeline, _ = self._engine._apply_moe_geo_sharding(
                secondary_pipeline,
                self._rotate(prep.candidates, 1),
                prompt=prep.effective_prompt,
                requested_expert_tags=list(prep.primary_moe_policy.get("requested_experts", [])),
                requested_expert_layer_indices=list(prep.primary_moe_policy.get("requested_layer_indices", [])),
                locked_first_peer_id=(
                    str(secondary_bandwidth_policy.get("prefill_peer_id", "")).strip()
                    if bool(secondary_bandwidth_policy.get("prefill_required"))
                    else None
                ),
            )
            tertiary_pipeline = self._engine._select_pipeline(self._rotate(prep.candidates, 2), pipeline_width=pipeline_width)
            tertiary_pipeline, tertiary_bandwidth_policy = self._engine._apply_bandwidth_asymmetry(
                tertiary_pipeline,
                self._rotate(prep.candidates, 2),
                prompt_tokens_est=prompt_tokens_est,
            )
            tertiary_pipeline, _ = self._engine._apply_moe_geo_sharding(
                tertiary_pipeline,
                self._rotate(prep.candidates, 2),
                prompt=prep.effective_prompt,
                requested_expert_tags=list(prep.primary_moe_policy.get("requested_experts", [])),
                requested_expert_layer_indices=list(prep.primary_moe_policy.get("requested_layer_indices", [])),
                locked_first_peer_id=(
                    str(tertiary_bandwidth_policy.get("prefill_peer_id", "")).strip()
                    if bool(tertiary_bandwidth_policy.get("prefill_required"))
                    else None
                ),
            )

            def run_secondary() -> ChainResult:
                nonlocal secondary_result
                secondary_result = self._engine._run_chain(
                    prep.effective_prompt,
                    self._rotate(prep.candidates, 1),
                    secondary_pipeline,
                    max_tokens=max_tokens,
                    request_id=primary.request_id,
                    deadline=deadline,
                    **decode_controls,
                )
                return secondary_result

            def run_tertiary() -> ChainResult:
                nonlocal tertiary_result
                tertiary_result = self._engine._run_chain(
                    prep.effective_prompt,
                    self._rotate(prep.candidates, 2),
                    tertiary_pipeline,
                    max_tokens=max_tokens,
                    request_id=primary.request_id,
                    deadline=deadline,
                    **decode_controls,
                )
                return tertiary_result

            _t_verify_start = time.perf_counter()
            verification = self.verifier.verify(
                primary,
                run_secondary=run_secondary,
                run_tertiary=(run_tertiary if len(prep.candidates) >= 3 else None),
            )
            _t_verify_ms = (time.perf_counter() - _t_verify_start) * 1000
            logger.info("PROFILE phase_C_verification=%.1fms", _t_verify_ms)
        verification_feedback = self._engine._apply_verification_feedback(
            primary=primary,
            secondary=secondary_result,
            tertiary=tertiary_result,
            verification=verification,
        )

        response_text = primary.text if verification.winner == "primary" else verification.secondary_text or primary.text

        # ── DSD: Decentralized Speculative Decoding (P0-A) ──────────────
        # After the primary chain pass, if DSD is enabled and the primary
        # result has token IDs in its activation, generate additional
        # tokens using speculative draft-then-verify rounds.
        _dsd_stats: dict[str, int | float] = {}
        if (
            self.config.speculative_swarm_enabled
            and primary.activation
            and max_tokens > len(primary.activation)
        ):
            try:
                from coordinator.speculative_swarm import (
                    SwarmSpeculativeDecoder,
                    SwarmSpecConfig,
                )

                spec_config = SwarmSpecConfig(
                    draft_tokens=self.config.speculative_draft_tokens,
                    adaptive_enabled=self.config.speculative_adaptive_enabled,
                    min_draft_tokens=self.config.speculative_min_draft_tokens,
                    max_draft_tokens=self.config.speculative_max_draft_tokens,
                    acceptance_low_watermark=self.config.speculative_acceptance_low_watermark,
                    acceptance_high_watermark=self.config.speculative_acceptance_high_watermark,
                )

                # Draft function: use ToyRuntime-style hash or the
                # existing PyTorchDraftModel if available
                def _draft_fn(ctx: list[int], k: int) -> list[int]:
                    try:
                        draft_model = self._engine._load_pytorch_draft_model()
                        return draft_model.propose_token_ids(ctx, k)
                    except Exception:
                        # Fallback: simple deterministic draft
                        base = ctx[-1] if ctx else 0
                        return [(base + i + 1) % 50000 for i in range(k)]

                decoder = SwarmSpeculativeDecoder(
                    config=spec_config,
                    draft_fn=_draft_fn,
                )

                # Rebuild chain for verify calls
                import coordinator.engine as _engine_mod
                _InferenceChain = getattr(_engine_mod, "InferenceChain", InferenceChain)
                verify_chain = _InferenceChain(
                    prep.primary_pipeline,
                    timeout_ms=self.config.timeout_ms,
                    transport_config=self.transport_config,
                    activation_quantization_enabled=self.config.activation_quantization_enabled,
                )

                # Extract initial token IDs from primary activation
                context_ids = [int(round(float(v))) for v in primary.activation]
                generated_ids = list(context_ids)

                for _round in range(32):  # cap at 32 DSD rounds
                    remaining = max_tokens - len(generated_ids)
                    if remaining <= 0:
                        break
                    k = min(decoder.current_k, remaining)
                    draft_ids = decoder.propose(generated_ids, k)
                    try:
                        verified_ids = verify_chain.verify_tokens(
                            prompt=prep.effective_prompt,
                            draft_token_ids=draft_ids,
                            request_id=primary.request_id,
                            deadline=deadline,
                        )
                    except Exception:
                        break  # Pipeline error — stop DSD, use what we have
                    accepted, _ = decoder.accept_reject(draft_ids, verified_ids)
                    if not accepted:
                        break
                    generated_ids.extend(accepted)

                # Re-decode the extended token sequence
                if len(generated_ids) > len(primary.activation):
                    response_text = ModelShard.decode_text(
                        [float(t) for t in generated_ids],
                        max_tokens=max_tokens,
                        tokenizer_model_id=self._resolve_pipeline_runtime_model_id(
                            prep.primary_pipeline,
                            prep.decision.served_model,
                        ),
                    )

                stats = decoder.stats
                _dsd_stats = {
                    "rounds": stats.rounds,
                    "draft_tokens": stats.draft_tokens,
                    "accepted_tokens": stats.accepted_tokens,
                    "acceptance_rate": stats.acceptance_rate or 0.0,
                    "final_k": decoder.current_k,
                }
                logger.info(
                    "PROFILE dsd_complete rounds=%d accepted=%d/%d rate=%.2f final_k=%d",
                    stats.rounds, stats.accepted_tokens, stats.draft_tokens,
                    stats.acceptance_rate or 0.0, decoder.current_k,
                )
            except ImportError:
                pass  # speculative_swarm not available — skip DSD

        hydra_reward_rate = max(0.0, float(self.config.hydra_reward_per_1k_tokens))
        for trace in primary.traces:
            self.ledger.earn(trace.peer_id, tokens_served=max_tokens)
            if hydra_reward_rate > 0.0:
                self.hydra.mint_for_inference(
                    peer_id=trace.peer_id,
                    tokens_served=max_tokens,
                    reward_per_1k_tokens=hydra_reward_rate,
                )

        replication = self._engine._replication_dict(prep.decision.served_model, len(prep.health))
        concentration = concentration_metrics(
            [item.peer for item in prep.health],
            cap_fraction=self.config.operator_cap_fraction,
        )

        return {
            "request_id": primary.request_id,
            "response": response_text,
            "primary_response": primary.text,
            "latency_ms": round(primary.latency_ms, 2),
            "pipeline": [asdict(trace) for trace in primary.traces],
            "verification": asdict(verification),
            "verification_feedback": verification_feedback,
            "compression": dict(primary.compression or {}),
            "encryption": dict(primary.encryption or {}),
            "grounding": self._grounding_meta(
                enabled=grounding,
                snippets=prep.snippets,
                grounding_result=prep.grounding_result,
            ),
            "model": self._model_meta(prep.decision),
            "available_peer_counts": prep.counts,
            "replication": replication,
            "bandwidth_policy": prep.primary_bandwidth_policy,
            "moe_geo": prep.primary_moe_policy,
            "concentration": {
                "model_id": prep.decision.served_model,
                "total_peers": concentration.total_peers,
                "operator_counts": concentration.operator_counts,
                "operator_shares": concentration.operator_shares,
                "max_operator": concentration.max_operator,
                "max_share": round(concentration.max_share, 6),
                "over_cap_operators": concentration.over_cap_operators,
            },
            "discovered_peers": self._engine._discovered_peer_rows(prep.health),
            # Phase 3: surface which pipeline path was used
            "pipeline_mode": prep.pipeline_mode,
            "dsd": _dsd_stats,
        }

    # ------------------------------------------------------------------
    # infer_stream
    # ------------------------------------------------------------------

    def infer_stream(
        self,
        *,
        prompt: str,
        max_tokens: int = 24,
        pipeline_width: int | None = None,
        grounding: bool = True,
        priority: bool = False,
        client_id: str = "anonymous",
        model_id: str | None = None,
        allow_degradation: bool | None = None,
        session_id: str | None = None,
        expert_tags: list[str] | None = None,
        expert_layer_indices: list[int] | None = None,
        decode_do_sample: bool | None = None,
        decode_temperature: float | None = None,
        decode_top_p: float | None = None,
        decode_top_k: int | None = None,
        decode_seed: int | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute a streaming inference request, yielding tokens incrementally.

        Supports four execution modes: PyTorch speculative decode, PyTorch
        autoregressive, speculative decode (toy), and iterative decode.
        Includes KV cache seeding, peer-native cache reuse, pipeline-parallel
        prefetching, and adaptive speculative batch sizing.

        Args:
            prompt: The text prompt.
            max_tokens: Maximum tokens to stream.
            pipeline_width: Number of peers per pipeline.
            grounding: Whether to enable RAG grounding.
            priority: Whether to spend priority credits.
            client_id: Client identifier for credit operations.
            model_id: Requested model identifier.
            allow_degradation: Whether model degradation is allowed.
            session_id: Optional session ID for KV affinity.
            expert_tags: Optional expert tags for MoE routing.
            expert_layer_indices: Optional layer indices for MoE routing.
            decode_do_sample: Enable sampling during decoding.
            decode_temperature: Sampling temperature.
            decode_top_p: Nucleus sampling threshold.
            decode_top_k: Top-k sampling parameter.
            decode_seed: Random seed for reproducible sampling.
            request_id: Optional unique request identifier.

        Returns:
            Dict with ``stream`` (a generator yielding token strings),
            ``model``, ``grounding``, ``streaming`` config, and metadata.

        Raises:
            RuntimeError: If priority credits are insufficient or no peers found.
            ValueError: If max_tokens exceeds the elastic ceiling.
        """
        request_id = request_id or str(uuid.uuid4())
        logger.info(
            "infer_stream_start req_id=%s model=%s client=%s",
            request_id, model_id or self.config.default_model, client_id,
        )
        if priority and not self.ledger.spend(client_id, 1.0):
            raise RuntimeError("insufficient_priority_credits")

        # Streaming makes many short gRPC round-trips (one per token), so a
        # fixed overall deadline would prematurely kill the stream.  Each hop
        # already has an independent per-stage timeout from InferenceChain.
        prep = self._engine._prepare_inference(
            prompt=prompt,
            pipeline_width=pipeline_width,
            grounding=grounding,
            model_id=model_id,
            allow_degradation=allow_degradation,
            session_id=session_id,
            expert_tags=expert_tags,
            expert_layer_indices=expert_layer_indices,
        )
        max_stream_tokens = int(max_tokens or 1024)
        # Elastic output cap: 2048 floor, up to 8192 if redundancy >= 3.0
        _served = prep.decision.served_model
        _available = prep.counts.get(_served, 0)
        _required = self._engine._required_replicas(_served)
        _effective_redundancy = _available / max(_required, 1)
        _elastic_ceiling = 8192 if _effective_redundancy >= 3.0 else 2048
        if max_stream_tokens > _elastic_ceiling:
            raise ValueError(
                f"Network redundancy for {_served} is currently too low for "
                f"extended context. Maximum allowed output is {_elastic_ceiling} tokens."
            )
        hydra_reward_rate = max(0.0, float(self.config.hydra_reward_per_1k_tokens))
        pytorch_autoregressive = self._pipeline_uses_pytorch_runtime(prep.primary_pipeline)
        requested_decode_controls = self._normalize_decode_controls(
            decode_do_sample=decode_do_sample,
            decode_temperature=decode_temperature,
            decode_top_p=decode_top_p,
            decode_top_k=decode_top_k,
            decode_seed=decode_seed,
        )
        pytorch_decode_controls = (dict(requested_decode_controls) if pytorch_autoregressive else {})
        pytorch_tokenizer_model_id = self._engine._resolve_pipeline_runtime_model_id(
            prep.primary_pipeline,
            prep.decision.served_model,
        )
        pytorch_tokenizer = (
            self._engine._load_generation_tokenizer(pytorch_tokenizer_model_id)
            if pytorch_autoregressive
            else None
        )
        pytorch_eos_token_ids, pytorch_special_token_ids = self._collect_eos_token_ids(
            pytorch_tokenizer
        )
        pytorch_eos_token_id = (min(pytorch_eos_token_ids) if pytorch_eos_token_ids else None)
        pytorch_speculative_enabled = bool(self.config.speculative_enabled and pytorch_autoregressive)
        pytorch_draft_model: PyTorchDraftModel | None = None
        if pytorch_speculative_enabled:
            try:
                pytorch_draft_model = self._engine._load_pytorch_draft_model(
                    tokenizer_model_id=pytorch_tokenizer_model_id,
                )
            except Exception:
                logger.warning("speculative_draft_model_load_failed — speculative decoding disabled", exc_info=True)
                pytorch_draft_model = None
        pytorch_speculative_active = bool(pytorch_speculative_enabled and pytorch_draft_model is not None)
        speculative_enabled = bool(self.config.speculative_enabled and not pytorch_autoregressive)
        draft_batch_size = max(1, min(16, int(self.config.speculative_draft_tokens)))
        adaptive_enabled = bool(self.config.speculative_adaptive_enabled)
        adaptive_min = max(1, min(16, int(self.config.speculative_min_draft_tokens)))
        adaptive_max = max(adaptive_min, min(16, int(self.config.speculative_max_draft_tokens)))
        adaptive_batch = max(adaptive_min, min(adaptive_max, draft_batch_size))
        acceptance_low = max(0.0, min(1.0, float(self.config.speculative_acceptance_low_watermark)))
        acceptance_high = max(0.0, min(1.0, float(self.config.speculative_acceptance_high_watermark)))
        speculative_stats = {
            "enabled": (speculative_enabled or pytorch_speculative_active),
            "adaptive_enabled": adaptive_enabled,
            "configured_draft_tokens": draft_batch_size,
            "current_draft_tokens": adaptive_batch,
            "min_draft_tokens": adaptive_min,
            "max_draft_tokens": adaptive_max,
            "rounds": 0,
            "mismatch_rounds": 0,
            "accepted_tokens": 0,
            "verified_tokens": 0,
            "acceptance_rate": None,
        }
        pipeline_parallel_enabled = bool(self.config.pipeline_parallel_enabled and not pytorch_autoregressive)
        pipeline_parallel_workers = max(1, int(self.config.pipeline_parallel_workers))
        pipeline_parallel = {
            "enabled": pipeline_parallel_enabled,
            "workers": pipeline_parallel_workers,
            "prefetch_submitted": 0,
            "prefetch_hits": 0,
            "prefetch_misses": 0,
            "prefetch_failures": 0,
            "prefetch_waits": 0,
        }
        served_model = prep.decision.served_model
        kv_cache_seed = self._engine._get_kv_affinity_activation(session_id, served_model)
        kv_cache_source_peer_id = self._engine._get_kv_affinity_activation_peer(session_id, served_model)
        prefill_peer_id = str(prep.primary_bandwidth_policy.get("prefill_peer_id", "")).strip() or None
        kv_affinity_hit = bool(prep.primary_bandwidth_policy.get("kv_affinity_hit"))
        kv_cold_restart = bool(prep.primary_bandwidth_policy.get("kv_cold_restart"))
        peer_native_cache_enabled = bool(
            self.config.kv_affinity_enabled
            and self.config.kv_peer_cache_enabled
            and session_id
            and prefill_peer_id
        )
        peer_cache_can_reuse = bool(peer_native_cache_enabled and kv_affinity_hit and not kv_cold_restart)
        kv_cache_usable = bool(kv_cache_seed and prefill_peer_id)
        cross_peer_relay = bool(
            kv_cache_usable
            and kv_cache_source_peer_id
            and prefill_peer_id
            and kv_cache_source_peer_id != prefill_peer_id
        )
        kv_data_plane = {
            "enabled": self.config.kv_affinity_enabled,
            "session_id": session_id,
            "cache_available": bool(kv_cache_seed),
            "external_cache_seeded": False,
            "cache_source_peer_id": kv_cache_source_peer_id,
            "cache_target_peer_id": prefill_peer_id,
            "cross_peer_relay": cross_peer_relay,
            "cache_used": False,
            "cache_updated": False,
            "seeded_rounds": 0,
            "peer_native_cache_enabled": peer_native_cache_enabled,
            "peer_cache_requested": False,
            "peer_cache_hits": 0,
            "peer_cache_misses": 0,
            "peer_cache_fallbacks": 0,
        }

        def stream_chunks():
            nonlocal adaptive_batch
            working_prompt = prep.effective_prompt
            generated_tokens: list[str] = []
            first = True
            activation_seed = list(kv_cache_seed) if kv_cache_usable else None
            latest_activation: list[float] | None = None
            peer_cache_warm = bool(peer_cache_can_reuse)
            prefetch_executor = ThreadPoolExecutor(max_workers=pipeline_parallel_workers) if pipeline_parallel_enabled else None
            pending_prefetch: dict[str, Any] | None = None

            if pytorch_autoregressive and not pytorch_speculative_active:
                # Phase 7 fix: use the same prefill→decode loop structure as
                # the non-streaming ``infer()`` Phase 6 path. The previous
                # verify/commit dance discarded the prefill's first sampled
                # token and instead re-sent ``context_token_ids[-1]`` (the
                # last PROMPT token) as the verify input — for Qwen 3.5 chat
                # templates that token is the assistant turn marker, so the
                # model immediately predicted EOS / a special token and the
                # SSE stream emitted nothing before ``[DONE]``.
                #
                # The loop here is straightforward: prefill once with the
                # full context, then for every decode step ship JUST the
                # previously generated token and read peer-side KV state.
                # All the same Phase 2 / Phase 6 fixes apply (Identity
                # offload, ``DynamicCache`` materialisation, T4 conv1d
                # fallback) so it benefits from the same ~3× speedup.
                if pytorch_tokenizer is None:
                    raise RuntimeError("pytorch_generation_tokenizer_unavailable")

                context_token_ids = [int(token) for token in pytorch_tokenizer.encode(working_prompt, add_special_tokens=True)]
                if not context_token_ids:
                    fallback_token = (
                        int(pytorch_eos_token_id)
                        if pytorch_eos_token_id is not None
                        else 0
                    )
                    context_token_ids = [fallback_token]
                pytorch_session_id = str(session_id or request_id)
                _stream_kv_mode = "kv_aware"

                def _stream_step_prefill():
                    return self._engine._run_chain(
                        "",
                        prep.candidates,
                        prep.primary_pipeline,
                        max_tokens=1,
                        request_id=request_id,
                        initial_activation=[float(t) for t in context_token_ids],
                        kv_session_id=pytorch_session_id,
                        kv_store_activation=True,
                        kv_use_cached_activation=False,
                        kv_cache_stage_index=0,
                        kv_cache_all_stages=True,
                        **pytorch_decode_controls,
                    )

                def _stream_step_decode(prev_token: int):
                    return self._engine._run_chain(
                        "",
                        prep.candidates,
                        prep.primary_pipeline,
                        max_tokens=1,
                        request_id=request_id,
                        initial_activation=[float(prev_token)],
                        kv_session_id=pytorch_session_id,
                        kv_store_activation=True,
                        kv_use_cached_activation=True,
                        kv_cache_stage_index=0,
                        kv_cache_all_stages=True,
                        **pytorch_decode_controls,
                    )

                def _stream_step_stateless(all_token_ids: list[int]):
                    return self._engine._run_chain(
                        "",
                        prep.candidates,
                        prep.primary_pipeline,
                        max_tokens=1,
                        request_id=request_id,
                        initial_activation=[float(t) for t in all_token_ids],
                        **pytorch_decode_controls,
                    )

                _prefill_done = False
                for _ in range(max_stream_tokens):
                    try:
                        if _stream_kv_mode == "kv_aware":
                            if not _prefill_done:
                                _step_result = _stream_step_prefill()
                                _prefill_done = True
                                kv_data_plane["cache_updated"] = True
                            else:
                                _step_result = _stream_step_decode(context_token_ids[-1])
                        else:
                            _step_result = _stream_step_stateless(context_token_ids)
                    except RuntimeError as _kv_exc:
                        if _stream_kv_mode == "kv_aware":
                            logger.warning(
                                "infer_stream_kv_fallback: err=%s — switching to stateless re-prefill",
                                str(_kv_exc)[:200],
                            )
                            _stream_kv_mode = "stateless"
                            kv_data_plane["peer_cache_misses"] = int(kv_data_plane["peer_cache_misses"]) + 1
                            kv_data_plane["peer_cache_fallbacks"] = int(kv_data_plane["peer_cache_fallbacks"]) + 1
                            _step_result = _stream_step_stateless(context_token_ids)
                        else:
                            raise

                    if not _step_result.activation:
                        break
                    next_token_id = int(round(float(_step_result.activation[0])))
                    if pytorch_eos_token_ids and next_token_id in pytorch_eos_token_ids:
                        return
                    context_token_ids.append(next_token_id)
                    latest_activation = [float(next_token_id)]
                    if next_token_id in pytorch_special_token_ids:
                        if len(generated_tokens) >= max_stream_tokens:
                            return
                        continue
                    token_text = str(
                        pytorch_tokenizer.decode(
                            [next_token_id],
                            clean_up_tokenization_spaces=False,
                        )
                    )
                    if token_text:
                        generated_tokens.append(token_text)
                        yield token_text
                    for trace in _step_result.traces:
                        self.ledger.earn(trace.peer_id, tokens_served=1)
                        if hydra_reward_rate > 0.0:
                            self.hydra.mint_for_inference(
                                peer_id=trace.peer_id,
                                tokens_served=1,
                                reward_per_1k_tokens=hydra_reward_rate,
                            )
                    if len(generated_tokens) >= max_stream_tokens:
                        return
                return

            if pytorch_autoregressive:
                # Speculative decoding path (non-streaming-Phase-6) — kept
                # as-is for the speculative+streaming combination that has
                # its own KV/draft-model logic. Only entered when
                # ``--speculative`` is on AND streaming is requested.
                if pytorch_tokenizer is None:
                    raise RuntimeError("pytorch_generation_tokenizer_unavailable")

                context_token_ids = [int(token) for token in pytorch_tokenizer.encode(working_prompt, add_special_tokens=True)]
                if not context_token_ids:
                    fallback_token = (
                        int(pytorch_eos_token_id)
                        if pytorch_eos_token_id is not None
                        else 0
                    )
                    context_token_ids = [fallback_token]
                pytorch_session_id = str(session_id or request_id)
                prefill_done = False

                for _ in range(max_stream_tokens):
                    remaining = max_stream_tokens - len(generated_tokens)
                    verify_batch = min(adaptive_batch, remaining) if (pytorch_speculative_active and remaining > 1 and adaptive_batch > 1) else 1

                    if not prefill_done:
                        self._engine._run_chain(
                            "",
                            prep.candidates,
                            prep.primary_pipeline,
                            max_tokens=1,
                            request_id=request_id,
                            initial_activation=[float(token_id) for token_id in context_token_ids],
                            kv_session_id=pytorch_session_id,
                            kv_store_activation=True,
                            kv_use_cached_activation=False,
                            kv_cache_stage_index=0,
                            kv_cache_all_stages=True,
                            **pytorch_decode_controls,
                        )
                        prefill_done = True
                        kv_data_plane["cache_updated"] = True

                    if verify_batch > 1 and pytorch_draft_model is not None:
                        draft_token_ids = pytorch_draft_model.propose_token_ids(context_token_ids, max_tokens=verify_batch)
                    else:
                        draft_token_ids = []

                    verify_input_ids = list(draft_token_ids) if draft_token_ids else [context_token_ids[-1]]
                    verify_count = len(draft_token_ids) if draft_token_ids else 1
                    kv_data_plane["peer_cache_requested"] = True
                    try:
                        verify_result = self._engine._run_chain(
                            "",
                            prep.candidates,
                            prep.primary_pipeline,
                            max_tokens=max(1, verify_count),
                            request_id=request_id,
                            initial_activation=[float(token_id) for token_id in verify_input_ids],
                            kv_session_id=pytorch_session_id,
                            kv_store_activation=False,
                            kv_use_cached_activation=True,
                            kv_cache_stage_index=0,
                            kv_cache_all_stages=True,
                            **pytorch_decode_controls,
                        )
                    except RuntimeError:
                        kv_data_plane["peer_cache_misses"] = int(kv_data_plane["peer_cache_misses"]) + 1
                        kv_data_plane["peer_cache_fallbacks"] = int(kv_data_plane["peer_cache_fallbacks"]) + 1
                        self._engine._run_chain(
                            "",
                            prep.candidates,
                            prep.primary_pipeline,
                            max_tokens=1,
                            request_id=request_id,
                            initial_activation=[float(token_id) for token_id in context_token_ids],
                            kv_session_id=pytorch_session_id,
                            kv_store_activation=True,
                            kv_use_cached_activation=False,
                            kv_cache_stage_index=0,
                            kv_cache_all_stages=True,
                            **pytorch_decode_controls,
                        )
                        prefill_done = True
                        verify_result = self._engine._run_chain(
                            "",
                            prep.candidates,
                            prep.primary_pipeline,
                            max_tokens=max(1, verify_count),
                            request_id=request_id,
                            initial_activation=[float(token_id) for token_id in verify_input_ids],
                            kv_session_id=pytorch_session_id,
                            kv_store_activation=False,
                            kv_use_cached_activation=True,
                            kv_cache_stage_index=0,
                            kv_cache_all_stages=True,
                            **pytorch_decode_controls,
                        )

                    verify_tokens = [max(0, int(round(float(token)))) for token in list(verify_result.activation)]
                    if verify_count > 0:
                        verify_tokens = verify_tokens[:verify_count]
                    if not verify_tokens:
                        break

                    kv_meta = dict(verify_result.kv or {})
                    if kv_meta.get("cache_hit"):
                        kv_data_plane["cache_used"] = True
                        kv_data_plane["peer_cache_hits"] = int(kv_data_plane["peer_cache_hits"]) + 1
                    else:
                        kv_data_plane["peer_cache_misses"] = int(kv_data_plane["peer_cache_misses"]) + 1

                    if draft_token_ids:
                        selection = select_verified_token_ids(verify_tokens, draft_token_ids)
                        accepted_token_ids = list(selection.accepted_token_ids)
                        speculative_stats["rounds"] += 1
                        if selection.mismatch:
                            speculative_stats["mismatch_rounds"] += 1
                        speculative_stats["accepted_tokens"] += len(accepted_token_ids)
                        speculative_stats["verified_tokens"] += len(verify_tokens)
                        if speculative_stats["verified_tokens"]:
                            speculative_stats["acceptance_rate"] = round(
                                speculative_stats["accepted_tokens"] / speculative_stats["verified_tokens"],
                                6,
                            )
                        if adaptive_enabled and len(verify_tokens) > 1:
                            accepted_ratio = len(accepted_token_ids) / float(len(verify_tokens))
                            if accepted_ratio <= acceptance_low:
                                adaptive_batch = max(adaptive_min, adaptive_batch - 1)
                            elif accepted_ratio >= acceptance_high:
                                adaptive_batch = min(adaptive_max, adaptive_batch + 1)
                            speculative_stats["current_draft_tokens"] = adaptive_batch
                    else:
                        accepted_token_ids = [verify_tokens[0]]

                    if not accepted_token_ids:
                        break

                    commit_result = self._engine._run_chain(
                        "",
                        prep.candidates,
                        prep.primary_pipeline,
                        max_tokens=max(1, len(accepted_token_ids)),
                        request_id=request_id,
                        initial_activation=[float(token_id) for token_id in accepted_token_ids],
                        kv_session_id=pytorch_session_id,
                        kv_store_activation=True,
                        kv_use_cached_activation=True,
                        kv_cache_stage_index=0,
                        kv_cache_all_stages=True,
                        **pytorch_decode_controls,
                    )
                    kv_data_plane["cache_updated"] = True

                    for next_token_id in accepted_token_ids:
                        if pytorch_eos_token_ids and int(next_token_id) in pytorch_eos_token_ids:
                            return
                        context_token_ids.append(next_token_id)
                        latest_activation = [float(next_token_id)]
                        if int(next_token_id) in pytorch_special_token_ids:
                            if len(generated_tokens) >= max_stream_tokens:
                                return
                            continue
                        token_text = str(
                            pytorch_tokenizer.decode(
                                [next_token_id],
                                clean_up_tokenization_spaces=False,
                            )
                        )
                        if token_text:
                            generated_tokens.append(token_text)
                            yield token_text
                        for trace in commit_result.traces:
                            self.ledger.earn(trace.peer_id, tokens_served=1)
                            if hydra_reward_rate > 0.0:
                                self.hydra.mint_for_inference(
                                    peer_id=trace.peer_id,
                                    tokens_served=1,
                                    reward_per_1k_tokens=hydra_reward_rate,
                                )

                        if len(generated_tokens) >= max_stream_tokens:
                            return
                return

            def _round_signature(
                *,
                step_prompt: str,
                verify_batch: int,
                request_peer_cache_round: bool,
                use_seed_this_round: bool,
            ) -> tuple[str, int, bool, bool]:
                return (
                    str(step_prompt),
                    int(verify_batch),
                    bool(request_peer_cache_round),
                    bool(use_seed_this_round),
                )

            try:
                while len(generated_tokens) < max_stream_tokens:
                    remaining = max_stream_tokens - len(generated_tokens)
                    use_speculative_round = speculative_enabled and adaptive_batch > 1 and remaining > 1
                    verify_batch = min(adaptive_batch, remaining) if use_speculative_round else 1
                    request_peer_cache_round = bool(peer_native_cache_enabled and peer_cache_warm and session_id)
                    use_seed_this_round = activation_seed is not None and not request_peer_cache_round
                    step_prompt = working_prompt if (not use_seed_this_round and not request_peer_cache_round) else ""
                    run_kwargs: dict[str, Any] = {}
                    if use_seed_this_round:
                        run_kwargs["initial_activation"] = list(activation_seed)
                        kv_data_plane["seeded_rounds"] = int(kv_data_plane["seeded_rounds"]) + 1
                        kv_data_plane["external_cache_seeded"] = True
                        kv_data_plane["cache_used"] = True
                    if peer_native_cache_enabled and session_id:
                        run_kwargs.update(
                            {
                                "kv_session_id": str(session_id),
                                "kv_store_activation": True,
                                "kv_use_cached_activation": request_peer_cache_round,
                                "kv_cache_stage_index": 0,
                            }
                        )
                        if request_peer_cache_round:
                            kv_data_plane["peer_cache_requested"] = True

                    current_signature = _round_signature(
                        step_prompt=step_prompt,
                        verify_batch=verify_batch,
                        request_peer_cache_round=request_peer_cache_round,
                        use_seed_this_round=use_seed_this_round,
                    )

                    step_result: ChainResult | None = None
                    if pending_prefetch is not None:
                        prefetch_signature = pending_prefetch.get("signature")
                        prefetch_future = pending_prefetch.get("future")
                        if prefetch_signature == current_signature and isinstance(prefetch_future, Future):
                            pipeline_parallel["prefetch_waits"] = int(pipeline_parallel["prefetch_waits"]) + 1
                            try:
                                step_result = prefetch_future.result()
                                pipeline_parallel["prefetch_hits"] = int(pipeline_parallel["prefetch_hits"]) + 1
                            except Exception:
                                logger.warning("pipeline_parallel_prefetch_failed — falling back to inline execution", exc_info=True)
                                pipeline_parallel["prefetch_failures"] = int(pipeline_parallel["prefetch_failures"]) + 1
                                step_result = None
                        else:
                            pipeline_parallel["prefetch_misses"] = int(pipeline_parallel["prefetch_misses"]) + 1
                            if isinstance(prefetch_future, Future) and not prefetch_future.done():
                                prefetch_future.cancel()
                        pending_prefetch = None

                    if step_result is None:
                        try:
                            step_result = self._engine._run_chain(
                                step_prompt,
                                prep.candidates,
                                prep.primary_pipeline,
                                max_tokens=verify_batch,
                                request_id=request_id,
                                **run_kwargs,
                            )
                        except RuntimeError:
                            if request_peer_cache_round and activation_seed:
                                kv_data_plane["peer_cache_misses"] = int(kv_data_plane["peer_cache_misses"]) + 1
                                kv_data_plane["peer_cache_fallbacks"] = int(kv_data_plane["peer_cache_fallbacks"]) + 1
                                fallback_kwargs = dict(run_kwargs)
                                fallback_kwargs["kv_use_cached_activation"] = False
                                fallback_kwargs["initial_activation"] = list(activation_seed)
                                kv_data_plane["seeded_rounds"] = int(kv_data_plane["seeded_rounds"]) + 1
                                kv_data_plane["external_cache_seeded"] = True
                                kv_data_plane["cache_used"] = True
                                step_result = self._engine._run_chain(
                                    "",
                                    prep.candidates,
                                    prep.primary_pipeline,
                                    max_tokens=verify_batch,
                                    request_id=request_id,
                                    **fallback_kwargs,
                                )
                            else:
                                raise

                    latest_activation = list(step_result.activation)
                    activation_seed = latest_activation if latest_activation else None
                    if latest_activation and peer_native_cache_enabled and session_id:
                        peer_cache_warm = True
                    kv_meta = dict(step_result.kv or {})
                    if kv_meta.get("cache_requested"):
                        if kv_meta.get("cache_hit"):
                            kv_data_plane["peer_cache_hits"] = int(kv_data_plane["peer_cache_hits"]) + 1
                            kv_data_plane["cache_used"] = True
                        else:
                            kv_data_plane["peer_cache_misses"] = int(kv_data_plane["peer_cache_misses"]) + 1
                    tokens = ModelShard.decode_tokens(
                        step_result.activation,
                        max_tokens=verify_batch,
                        tokenizer_model_id=pytorch_tokenizer_model_id or None,
                    )
                    if not tokens:
                        break

                    if use_speculative_round:
                        draft_tokens = self.draft_model.propose(working_prompt, max_tokens=verify_batch)
                        # Resolve through engine module for monkeypatch compatibility.
                        import coordinator.engine as _engine_mod
                        _select_verified_tokens = getattr(_engine_mod, "select_verified_tokens", select_verified_tokens)
                        selection = _select_verified_tokens(tokens, draft_tokens)
                        accepted_tokens = selection.accepted_tokens or tokens[:1]
                        speculative_stats["rounds"] += 1
                        if selection.mismatch:
                            speculative_stats["mismatch_rounds"] += 1
                        speculative_stats["accepted_tokens"] += len(accepted_tokens)
                        speculative_stats["verified_tokens"] += verify_batch
                        if speculative_stats["verified_tokens"]:
                            speculative_stats["acceptance_rate"] = round(
                                speculative_stats["accepted_tokens"] / speculative_stats["verified_tokens"],
                                6,
                            )

                        if adaptive_enabled and verify_batch > 1:
                            accepted_ratio = len(accepted_tokens) / float(verify_batch)
                            if accepted_ratio <= acceptance_low:
                                adaptive_batch = max(adaptive_min, adaptive_batch - 1)
                            elif accepted_ratio >= acceptance_high:
                                adaptive_batch = min(adaptive_max, adaptive_batch + 1)
                            speculative_stats["current_draft_tokens"] = adaptive_batch
                    else:
                        accepted_tokens = tokens[:1]

                    # Submit next seeded decode round early so it can overlap chunk emission.
                    if prefetch_executor is not None and pending_prefetch is None:
                        accepted_budget = min(len(accepted_tokens), max_stream_tokens - len(generated_tokens))
                        projected_remaining = max_stream_tokens - (len(generated_tokens) + accepted_budget)
                        if projected_remaining > 0:
                            next_use_speculative_round = speculative_enabled and adaptive_batch > 1 and projected_remaining > 1
                            next_verify_batch = min(adaptive_batch, projected_remaining) if next_use_speculative_round else 1
                            next_request_peer_cache_round = bool(peer_native_cache_enabled and peer_cache_warm and session_id)
                            next_use_seed_this_round = activation_seed is not None and not next_request_peer_cache_round
                            if next_use_seed_this_round and not next_request_peer_cache_round:
                                next_prompt = ""
                                next_kwargs: dict[str, Any] = {
                                    "initial_activation": list(activation_seed),
                                }
                                if peer_native_cache_enabled and session_id:
                                    next_kwargs.update(
                                        {
                                            "kv_session_id": str(session_id),
                                            "kv_store_activation": True,
                                            "kv_use_cached_activation": False,
                                            "kv_cache_stage_index": 0,
                                        }
                                    )
                                pending_prefetch = {
                                    "signature": _round_signature(
                                        step_prompt=next_prompt,
                                        verify_batch=next_verify_batch,
                                        request_peer_cache_round=False,
                                        use_seed_this_round=True,
                                    ),
                                    "future": prefetch_executor.submit(
                                        self._engine._run_chain,
                                        next_prompt,
                                        prep.candidates,
                                        prep.primary_pipeline,
                                        max_tokens=next_verify_batch,
                                        request_id=request_id,
                                        **next_kwargs,
                                    ),
                                }
                                pipeline_parallel["prefetch_submitted"] = int(pipeline_parallel["prefetch_submitted"]) + 1

                    for token in accepted_tokens:
                        if len(generated_tokens) >= max_stream_tokens:
                            break
                        generated_tokens.append(token)
                        for trace in step_result.traces:
                            self.ledger.earn(trace.peer_id, tokens_served=1)
                            if hydra_reward_rate > 0.0:
                                self.hydra.mint_for_inference(
                                    peer_id=trace.peer_id,
                                    tokens_served=1,
                                    reward_per_1k_tokens=hydra_reward_rate,
                                )

                        working_prompt = f"{working_prompt} {token}".strip()
                        if first:
                            chunk = token[0].upper() + token[1:] if token else token
                            first = False
                        else:
                            chunk = f" {token}"
                        yield chunk

                if generated_tokens:
                    yield "."

                if latest_activation:
                    kv_data_plane["cache_updated"] = self._engine._set_kv_affinity_activation(
                        session_id=session_id,
                        model_id=served_model,
                        activation=latest_activation,
                    )
            finally:
                if pending_prefetch is not None:
                    future = pending_prefetch.get("future")
                    if isinstance(future, Future) and not future.done():
                        future.cancel()
                if prefetch_executor is not None:
                    prefetch_executor.shutdown(wait=False)

        return {
            "request_id": request_id,
            "stream": stream_chunks(),
            "model": self._model_meta(prep.decision),
            "grounding": self._grounding_meta(
                enabled=grounding,
                snippets=prep.snippets,
                grounding_result=prep.grounding_result,
            ),
            "available_peer_counts": prep.counts,
            "bandwidth_policy": prep.primary_bandwidth_policy,
            "moe_geo": prep.primary_moe_policy,
            "pipeline": [peer.peer_id for peer in prep.primary_pipeline],
            "streaming": {
                "execution_path": True,
                "mode": (
                    "pytorch_speculative_decode"
                    if (pytorch_autoregressive and pytorch_speculative_active)
                    else (
                        "pytorch_autoregressive"
                        if pytorch_autoregressive
                        else ("speculative_decode" if speculative_enabled else "iterative_decode")
                    )
                ),
                "max_tokens": max_stream_tokens,
                "speculative_enabled": (speculative_enabled or pytorch_speculative_active),
                "speculative_draft_tokens": draft_batch_size,
                "speculative": speculative_stats,
                "moe_geo": prep.primary_moe_policy,
                "pipeline_parallel": pipeline_parallel,
                "kv_data_plane": kv_data_plane,
                "pytorch": {
                    "enabled": pytorch_autoregressive,
                    "tokenizer_model_id": (
                        str(pytorch_tokenizer_model_id)
                        if pytorch_autoregressive
                        else None
                    ),
                    "draft_model_id": (
                        str(self.config.pytorch_speculative_draft_model_id)
                        if (pytorch_autoregressive and pytorch_speculative_active)
                        else None
                    ),
                    "eos_token_id": pytorch_eos_token_id,
                    "eos_token_ids": sorted(pytorch_eos_token_ids),
                    "decode": (
                        dict(pytorch_decode_controls)
                        if pytorch_autoregressive
                        else {}
                    ),
                },
            },
            "compression": {
                "enabled": self.config.tensor_autoencoder_enabled,
                "method": ("tensor_autoencoder_mean_pool" if self.config.tensor_autoencoder_enabled else "none"),
                "latent_dim": max(1, int(self.config.tensor_autoencoder_latent_dim)),
            },
            "encryption": {
                "enabled": self.config.advanced_encryption_enabled,
                "level": (
                    str(self.config.advanced_encryption_level)
                    if self.config.advanced_encryption_enabled
                    else "off"
                ),
                "layers_per_hop": (
                    required_layers_for_level(str(self.config.advanced_encryption_level))
                    if self.config.advanced_encryption_enabled
                    else 0
                ),
                "suite": (
                    "x25519_hkdf_sha256_aes256_gcm"
                    if str(self.config.advanced_encryption_level).strip().lower() == "standard"
                    else f"x25519_hkdf_sha256_aes256_gcm_onion_{str(self.config.advanced_encryption_level).strip().lower()}"
                ) if self.config.advanced_encryption_enabled else "none",
                "onion_routing": bool(
                    self.config.advanced_encryption_enabled
                    and str(self.config.advanced_encryption_level).strip().lower() in {"enhanced", "maximum"}
                    and len(prep.primary_pipeline) > 1
                ),
                "onion_layers": (
                    len(prep.primary_pipeline)
                    if (
                        self.config.advanced_encryption_enabled
                        and str(self.config.advanced_encryption_level).strip().lower() in {"enhanced", "maximum"}
                    )
                    else 0
                ),
            },
        }

    # ------------------------------------------------------------------
    # Chat wrappers
    # ------------------------------------------------------------------

    def infer_chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        """Single-shot chat inference from a list of messages.

        Converts messages to a model-specific prompt using chat templates,
        applies default decode controls, and delegates to ``infer()``.

        Args:
            messages: List of chat message dicts with ``role`` and ``content``.
            **kwargs: Additional arguments forwarded to ``infer()``.

        Returns:
            The full inference response dict.
        """
        kwargs = self._normalize_decode_kwarg_aliases(kwargs)
        requested_model = str(kwargs.get("model_id", self.config.default_model) or self.config.default_model)
        prompt = self._engine._messages_to_model_prompt(messages, model_id=requested_model)
        if "decode_do_sample" not in kwargs:
            kwargs["decode_do_sample"] = True
        if "decode_temperature" not in kwargs:
            kwargs["decode_temperature"] = float(self.config.pytorch_decode_temperature)
        if "decode_top_p" not in kwargs:
            kwargs["decode_top_p"] = float(self.config.pytorch_decode_top_p)
        if "decode_top_k" not in kwargs:
            kwargs["decode_top_k"] = int(self.config.pytorch_decode_top_k)
        if int(self.config.pytorch_decode_seed) > 0 and "decode_seed" not in kwargs:
            kwargs["decode_seed"] = int(self.config.pytorch_decode_seed)
        return self.infer(prompt=prompt, **kwargs)

    def infer_chat_stream(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        """Streaming chat inference from a list of messages.

        Converts messages to a model-specific prompt using chat templates,
        applies default decode controls, and delegates to ``infer_stream()``.

        Args:
            messages: List of chat message dicts with ``role`` and ``content``.
            **kwargs: Additional arguments forwarded to ``infer_stream()``.

        Returns:
            The streaming inference response dict with a token generator.
        """
        kwargs = self._normalize_decode_kwarg_aliases(kwargs)
        requested_model = str(kwargs.get("model_id", self.config.default_model) or self.config.default_model)
        prompt = self._engine._messages_to_model_prompt(messages, model_id=requested_model)
        if "decode_do_sample" not in kwargs:
            kwargs["decode_do_sample"] = True
        if "decode_temperature" not in kwargs:
            kwargs["decode_temperature"] = float(self.config.pytorch_decode_temperature)
        if "decode_top_p" not in kwargs:
            kwargs["decode_top_p"] = float(self.config.pytorch_decode_top_p)
        if "decode_top_k" not in kwargs:
            kwargs["decode_top_k"] = int(self.config.pytorch_decode_top_k)
        if int(self.config.pytorch_decode_seed) > 0 and "decode_seed" not in kwargs:
            kwargs["decode_seed"] = int(self.config.pytorch_decode_seed)
        return self.infer_stream(prompt=prompt, **kwargs)
