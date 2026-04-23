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

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any

from openhydra_defaults import PRODUCTION_BOOTSTRAP_URLS

logger = logging.getLogger(__name__)

from coordinator.bandwidth_roles import RoleThresholds
from coordinator.chain import ChainResult, InferenceChain
from coordinator.concentration_guard import assemble_pipeline, concentration_metrics
from coordinator.degradation import DegradationDecision, DegradationPolicy, ModelAvailability
from coordinator.health_scorer import HealthScorer
from coordinator.moe_service import MoeService
from coordinator.mystery_shopper import MysteryShopper
from coordinator.path_finder import PathFinder, PeerEndpoint, load_peer_config, load_peers_from_dht
from coordinator.peer_selector import ScoredPeer, rank_peers
from coordinator.replication_monitor import ReplicationMonitor
from coordinator.speculative import DraftTokenModel, PyTorchDraftModel, select_verified_tokens, select_verified_token_ids
from coordinator.transport import TransportConfig
from coordinator.ledger_bridge import OpenHydraLedgerBridge
from economy.barter import SqliteCreditLedger
from economy.token import SqliteHydraTokenEconomy
from grounding.client_rag import GroundingClient, GroundingConfig, inject_grounding
from peer.crypto import required_layers_for_level
from peer.model_shard import ModelShard


def _default_trust_remote_code(model_id: str) -> bool:
    return "qwen" in str(model_id or "").strip().lower()


@dataclass(frozen=True)
class EngineConfig:
    deployment_profile: str = "dev"
    peers_config_path: str | None = None
    dht_urls: list[str] = field(default_factory=list)
    dht_url: str | None = None
    dht_lookup_timeout_s: float = 3.0
    dht_lookup_cache_ttl_s: float = 120.0
    dht_lookup_limit: int = 0
    dht_lookup_sloppy_factor: int = 3
    dht_lookup_dsht_replicas: int = 2
    dht_preferred_region: str | None = None
    tls_enabled: bool = False
    tls_root_cert_path: str | None = None
    tls_client_cert_path: str | None = None
    tls_client_key_path: str | None = None
    tls_server_name_override: str | None = None
    timeout_ms: int = 5000
    max_latency_ms: float = 5000.0
    pipeline_width: int = 3
    tier: int = 2
    audit_rate: float = 0.10
    redundant_exec_rate: float = 0.25
    auditor_rate: float = 0.0
    verification_alert_min_events: int = 10
    verification_alert_min_success_rate: float = 0.80
    verification_qos_min_events: int = 10
    verification_qos_min_success_rate: float = 0.0
    seed: int = 7
    max_failovers_per_stage: int = 1
    ledger_path: str = ".openhydra/credits.db"
    barter_decay_per_day: float = 0.05
    hydra_token_ledger_path: str = ".openhydra/hydra_tokens.db"
    hydra_reward_per_1k_tokens: float = 1.0
    hydra_slash_per_failed_verification: float = 0.0
    hydra_channel_default_ttl_seconds: int = 900
    hydra_channel_max_open_per_payer: int = 8
    hydra_channel_min_deposit: float = 0.01
    hydra_supply_cap: float = 69_000_000.0
    hydra_ledger_bridge_mock_mode: bool = True
    hydra_stake_priority_boost: float = 12.0
    hydra_no_stake_penalty_events: int = 8
    hydra_governance_daily_mint_rate: float = 250_000.0
    hydra_governance_min_slash_penalty: float = 0.1
    health_store_path: str = ".openhydra/health.json"
    database_url: str | None = None
    required_replicas: int = 3
    default_model: str = "openhydra-qwen3.5-0.8b"
    allow_dynamic_model_ids: bool = True
    model_catalog_path: str | None = None
    allow_degradation_default: bool = True
    operator_cap_fraction: float = (1.0 / 3.0)
    enforce_pipeline_diversity: bool = True
    diversity_window: int = 3
    diversity_max_per_window: int = 1
    prefill_token_threshold: int = 256
    prefill_min_bandwidth_mbps: float = 500.0
    decode_max_bandwidth_mbps: float = 50.0
    grounding_cache_path: str = ".openhydra/grounding_cache.json"
    grounding_cache_ttl_seconds: int = 900
    grounding_timeout_s: float = 3.0
    grounding_use_network: bool = True
    grounding_fallback_enabled: bool = True
    speculative_enabled: bool = False
    speculative_draft_tokens: int = 4
    speculative_seed: int = 13
    speculative_adaptive_enabled: bool = True
    speculative_min_draft_tokens: int = 2
    speculative_max_draft_tokens: int = 8
    speculative_acceptance_low_watermark: float = 0.55
    speculative_acceptance_high_watermark: float = 0.80
    pipeline_parallel_enabled: bool = False
    pipeline_parallel_workers: int = 1
    tensor_autoencoder_enabled: bool = False
    tensor_autoencoder_latent_dim: int = 1024
    # P0-A: Decentralized Speculative Decoding for Swarm Mode
    speculative_swarm_enabled: bool = False
    # P0-B: INT8 activation quantization on the wire
    activation_quantization_enabled: bool = True  # Enabled by default (Petals parity)
    # Petals parity Phase A: server-to-server push mode
    push_mode_enabled: bool = False
    push_callback_address: str = ""  # "host:port" where PushResult arrives
    # Path A (client-terminated pipeline, Petals parity): when true, the
    # last peer returns its raw post-last-layer hidden state instead of
    # sampling; the coordinator's HeadSampler applies final_norm + lm_head
    # and re-injects the token into stage 0. Eliminates the per-token
    # ring-loopback hop. Requires a co-located last-shard peer whose
    # runtime has been registered as the coordinator's head source (see
    # ``coordinator/head_sampler.py``). Default off for rollback safety.
    sample_on_coordinator: bool = False
    # Petals parity Phase B: stateful streaming sessions + history replay
    streaming_sessions_enabled: bool = False
    # P1-A: SpecPipe — pipeline-filling speculative decoding
    specpipe_enabled: bool = False
    specpipe_max_depth: int = 4
    # Non-streaming sharded decode loop — when True (default), non-streaming
    # ``infer()`` runs an outer autoregressive loop over sharded PyTorch
    # pipelines (re-prefills through every stage for each token). Set to
    # False to restore pre-fix behavior (single chain.run() → one token).
    autoregressive_sharded_enabled: bool = True
    # P1-B: Chunked prefill — split long prompts for pipeline interleaving
    chunked_prefill_enabled: bool = False
    chunked_prefill_chunk_size: int = 2048
    advanced_encryption_enabled: bool = False
    advanced_encryption_seed: str = "openhydra-tier3-dev-seed"
    advanced_encryption_level: str = "standard"
    kv_affinity_enabled: bool = True
    kv_affinity_ttl_seconds: int = 1800
    kv_peer_cache_enabled: bool = True
    moe_geo_enabled: bool = False
    moe_geo_min_tag_matches: int = 1
    moe_geo_min_layer_matches: int = 1
    moe_geo_prompt_hints_enabled: bool = True
    pytorch_generation_model_id: str = "Qwen/Qwen3.5-0.8B"
    pytorch_speculative_draft_model_id: str = "sshleifer/tiny-gpt2"
    pytorch_decode_do_sample: bool = False
    pytorch_decode_temperature: float = 0.7
    pytorch_decode_top_p: float = 0.95
    pytorch_decode_top_k: int = 40
    pytorch_decode_seed: int = 0
    pytorch_chat_template_enabled: bool = True

    def __post_init__(self) -> None:
        sources: list[str] = []
        seen: set[str] = set()
        for item in list(self.dht_urls):
            for token in str(item).split(","):
                value = token.strip()
                if not value or value in seen:
                    continue
                seen.add(value)
                sources.append(value)
        if self.dht_url:
            for token in str(self.dht_url).split(","):
                value = token.strip()
                if not value or value in seen:
                    continue
                seen.add(value)
                sources.append(value)
        object.__setattr__(self, "dht_urls", sources)
        if self.dht_url is None:
            object.__setattr__(self, "dht_url", (sources[0] if sources else None))


@dataclass(frozen=True)
class InferencePreparation:
    effective_prompt: str
    snippets: list[str]
    grounding_result: Any | None
    health: list[Any]
    candidates: list[PeerEndpoint]
    decision: DegradationDecision
    counts: dict[str, int]
    primary_pipeline: list[PeerEndpoint]
    primary_bandwidth_policy: dict[str, Any]
    primary_moe_policy: dict[str, Any]
    pipeline_mode: str = "full_model"


class CoordinatorEngine:
    """Thin facade that delegates to extracted service classes."""

    def __init__(self, config: EngineConfig):
        self.config = config

        # ── Ledger + economy core objects ──────────────────────────────────
        if config.database_url:
            from economy.postgres import PostgresCreditLedger, PostgresHydraTokenEconomy
            self.ledger = PostgresCreditLedger(config.database_url, decay_per_day=config.barter_decay_per_day)
            self.hydra = PostgresHydraTokenEconomy(
                config.database_url,
                channel_default_ttl_seconds=max(1, int(config.hydra_channel_default_ttl_seconds)),
                channel_max_open_per_payer=max(1, int(config.hydra_channel_max_open_per_payer)),
                channel_min_deposit=max(0.0, float(config.hydra_channel_min_deposit)),
                supply_cap=max(0.0, float(config.hydra_supply_cap)),
            )
            logger.info("ledger_backend=postgres dsn=***")
        else:
            self.ledger = SqliteCreditLedger(config.ledger_path, decay_per_day=config.barter_decay_per_day)
            self.hydra = SqliteHydraTokenEconomy(
                config.hydra_token_ledger_path,
                channel_default_ttl_seconds=max(1, int(config.hydra_channel_default_ttl_seconds)),
                channel_max_open_per_payer=max(1, int(config.hydra_channel_max_open_per_payer)),
                channel_min_deposit=max(0.0, float(config.hydra_channel_min_deposit)),
                supply_cap=max(0.0, float(config.hydra_supply_cap)),
            )
            logger.info("ledger_backend=sqlite")

        _recovery = self.hydra.recover()
        logger.info(
            "ledger_recovery open_channels=%d expired=%d accounts=%d minted=%.2f burned=%.2f",
            int(_recovery["open_channels"]), int(_recovery["expired_on_recovery"]),
            int(_recovery["total_accounts"]), float(_recovery["total_minted"]),
            float(_recovery["total_burned"]),
        )

        self.ledger_bridge = OpenHydraLedgerBridge(
            mock_mode=bool(config.hydra_ledger_bridge_mock_mode),
            supply_cap=max(0.0, float(config.hydra_supply_cap)),
            daily_mint_rate=max(0.0, float(config.hydra_governance_daily_mint_rate)),
            min_slash_penalty=max(0.0, float(config.hydra_governance_min_slash_penalty)),
            external_stake_resolver=self._hydra_stake_balance,
            external_stake_slasher=self._hydra_slash_stake,
        )

        # ── Infrastructure objects ─────────────────────────────────────────
        self.health = HealthScorer(config.health_store_path)
        self.replication_monitor = ReplicationMonitor(required_replicas=config.required_replicas)
        self.transport_config = TransportConfig(
            tls_enabled=config.tls_enabled, root_cert_path=config.tls_root_cert_path,
            client_cert_path=config.tls_client_cert_path, client_key_path=config.tls_client_key_path,
            server_name_override=config.tls_server_name_override,
        )
        self.model_catalog = self._load_model_catalog()
        self.catalogue_by_model = {item.model_id: item for item in self.model_catalog}
        self.degradation_policy = DegradationPolicy(self.model_catalog)
        self._last_scored_peers: list[ScoredPeer] = []
        self.role_thresholds = RoleThresholds(
            prefill_min_mbps=max(1.0, config.prefill_min_bandwidth_mbps),
            decode_max_mbps=max(0.0, config.decode_max_bandwidth_mbps),
        )
        self.grounding_client = GroundingClient(GroundingConfig(
            cache_path=config.grounding_cache_path,
            cache_ttl_seconds=max(1, config.grounding_cache_ttl_seconds),
            timeout_s=max(0.1, config.grounding_timeout_s),
            use_network=config.grounding_use_network,
            fallback_enabled=config.grounding_fallback_enabled,
        ))

        verification_rate = config.redundant_exec_rate if config.tier >= 2 else config.audit_rate
        verification_mode = "redundant_execution" if config.tier >= 2 else "mystery_shopper"
        self.verifier = MysteryShopper(
            sample_rate=verification_rate, seed=config.seed, mode=verification_mode,
            auditor_sample_rate=max(0.0, min(1.0, config.auditor_rate)),
        )
        self._last_verification_qos: dict[str, Any] = {
            "enabled": False, "min_events": 0, "min_success_rate": 0.0,
            "requested_model_blocked": False, "requested_model_events": 0,
            "requested_model_success_rate": None,
        }
        self.draft_model = DraftTokenModel(seed=config.speculative_seed)
        self._pytorch_draft_model_cache: dict[tuple[str, str], PyTorchDraftModel] = {}
        self._kv_affinity: dict[tuple[str, str], dict[str, Any]] = {}
        self._tokenizer_cache: dict[str, Any] = {}
        self._channel_provider_spend: dict[str, dict[str, float]] = {}
        self._dht_peer_cache: dict[str, dict[str, Any]] = {}
        self._dht_peer_cache_lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        self._dht_lookup_attempts = 0
        self._dht_lookup_successes = 0
        self._dht_lookup_failures = 0
        self._kv_store_ops_total: int = 0
        self._kv_retrieve_ops_total: int = 0
        self._inference_requests_total: int = 0

        # ── Auto-scaler ───────────────────────────────────────────────────
        from coordinator.request_log import RequestLog
        from coordinator.auto_scaler import AutoScaler, ModelSpec as _ScalerModelSpec
        self._request_log = RequestLog()
        _scaler_specs = [
            _ScalerModelSpec(
                model_id=m.model_id, shard_vram_mb=int(m.shard_vram_gb * 1024),
                shards_needed=max(1, m.shards_needed), quality_tier=m.quality_tier,
                required_peers=m.required_peers,
            )
            for m in self.model_catalog if m.shard_vram_gb > 0
        ]
        if _scaler_specs:
            self._auto_scaler: AutoScaler | None = AutoScaler(_scaler_specs)
            self._active_model_roster: list[str] = list(self._auto_scaler.active_roster)
        else:
            self._auto_scaler = None
            self._active_model_roster = [self.config.default_model]

        # ── Extracted services ─────────────────────────────────────────────
        from coordinator.economy_service import EconomyService
        from coordinator.kv_affinity_service import KvAffinityService
        from coordinator.tokenization_service import TokenizationService
        from coordinator.health_service import HealthService
        from coordinator.discovery_service import DiscoveryService
        from coordinator.pipeline_service import PipelineService
        from coordinator.status_service import StatusService
        from coordinator.inference_service import InferenceService

        self._economy = EconomyService(ledger=self.ledger, hydra=self.hydra, ledger_bridge=self.ledger_bridge)

        self._kv_affinity_svc = KvAffinityService(config=self.config, kv_affinity=self._kv_affinity)

        self._tokenization_svc = TokenizationService(
            config=self.config, tokenizer_cache=self._tokenizer_cache,
            draft_model_cache=self._pytorch_draft_model_cache,
            catalog_hf_model_id=self._catalog_hf_model_id,
        )

        self._health_svc = HealthService(
            health=self.health, config=self.config, replication_monitor=self.replication_monitor,
            ledger_bridge=self.ledger_bridge, model_catalog=self.model_catalog,
            normalize_peer_model=self._normalize_peer_model,
            required_replicas=self._required_replicas,
            role_for_peer=self._role_for_peer,
            last_scored_peers_getter=lambda: self._last_scored_peers,
            engine=self,
        )

        self._moe_svc = MoeService(config=self.config)

        self._discovery_svc = DiscoveryService(
            config=self.config, health=self.health, auto_scaler=self._auto_scaler,
            _dht_peer_cache=self._dht_peer_cache, _dht_peer_cache_lock=self._dht_peer_cache_lock,
            _active_model_roster=self._active_model_roster, _request_log=self._request_log,
            model_catalog=self.model_catalog, catalogue_by_model=self.catalogue_by_model,
            degradation_policy=self.degradation_policy, replication_monitor=self.replication_monitor,
            transport_config=self.transport_config, ledger_bridge=self.ledger_bridge,
            role_thresholds=self.role_thresholds, _metrics_lock=self._metrics_lock,
            _dht_lookup_attempts=self._dht_lookup_attempts,
            _dht_lookup_successes=self._dht_lookup_successes,
            _dht_lookup_failures=self._dht_lookup_failures,
            engine=self,
        )

        self._pipeline_svc = PipelineService(
            config=self.config, kv_affinity_service=self._kv_affinity_svc,
            role_thresholds=self.role_thresholds,
            engine=self,
        )

        self._status_svc = StatusService(
            config=self.config, health=self.health, auto_scaler=self._auto_scaler,
            _active_model_roster=self._active_model_roster,
            _dht_peer_cache=self._dht_peer_cache, _dht_peer_cache_lock=self._dht_peer_cache_lock,
            _request_log=self._request_log, ledger_bridge=self.ledger_bridge, hydra=self.hydra,
            _metrics_lock=self._metrics_lock, model_catalog=self.model_catalog,
            catalogue_by_model=self.catalogue_by_model, discovery_service=self._discovery_svc,
            replication_monitor=self.replication_monitor, role_thresholds=self.role_thresholds,
            _dht_lookup_attempts_ref=[0], _dht_lookup_successes_ref=[0],
            _dht_lookup_failures_ref=[0], _kv_store_ops_total_ref=[0],
            _kv_retrieve_ops_total_ref=[0], _inference_requests_total_ref=[0],
            engine=self,
        )

        self._inference_svc = InferenceService(
            config=self.config, discovery_service=self._discovery_svc,
            pipeline_service=self._pipeline_svc, kv_affinity_service=self._kv_affinity_svc,
            health=self.health, ledger=self.ledger, hydra=self.hydra,
            ledger_bridge=self.ledger_bridge, verifier=self.verifier,
            draft_model=self.draft_model, grounding_client=self.grounding_client,
            replication_monitor=self.replication_monitor, transport_config=self.transport_config,
            _metrics_lock=self._metrics_lock,
            _kv_store_ops_total_ref=[0], _kv_retrieve_ops_total_ref=[0],
            _inference_requests_total_ref=[0],
            _tokenizer_cache=self._tokenizer_cache,
            _pytorch_draft_model_cache=self._pytorch_draft_model_cache,
            _last_verification_qos=self._last_verification_qos,
            _last_scored_peers=self._last_scored_peers,
            engine=self,
        )
        # Phase B: attach StreamPool for persistent streaming connections
        if self.config.streaming_sessions_enabled:
            from coordinator.stream_pool import StreamPool
            self._inference_svc._stream_pool = StreamPool(
                idle_timeout_s=30.0, max_streams=256,
            )

    # ══════════════════════════════════════════════════════════════════════
    # Kept: helpers used by __init__ (called before services exist)
    # ══════════════════════════════════════════════════════════════════════

    def _hydra_stake_balance(self, pubkey: str) -> float:
        """Return the staked HYDRA balance for a public key (0.0 on error)."""
        try:
            snapshot = self.hydra.account_snapshot(pubkey)
        except RuntimeError:
            return 0.0
        return max(0.0, float(snapshot.get("stake", 0.0)))

    def _hydra_slash_stake(self, pubkey: str, amount: float) -> float:
        """Slash up to ``amount`` staked tokens from a peer (0.0 on error)."""
        requested = max(0.0, float(amount))
        if requested <= 0.0:
            return 0.0
        available = self._hydra_stake_balance(pubkey)
        if available <= 0.0:
            return 0.0
        slash_amount = min(available, requested)
        try:
            payload = self.hydra.slash(peer_id=pubkey, amount=slash_amount)
        except RuntimeError:
            return 0.0
        return max(0.0, float(payload.get("slashed", 0.0)))

    def _load_model_catalog(self) -> list[ModelAvailability]:
        """Load the model catalog from JSON or create a single-model default."""
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
            catalogue.append(ModelAvailability(
                model_id=model_id, required_peers=max(1, required), hf_model_id=hf_model_id,
                min_vram_gb=max(0, int(entry.get("min_vram_gb", 0))),
                recommended_quantization=str(entry.get("recommended_quantization", "fp32")),
                context_length=max(0, int(entry.get("context_length", 4096))),
                languages=tuple(str(x) for x in entry.get("languages", [])),
                tags=tuple(str(x) for x in entry.get("tags", [])),
                description=str(entry.get("description", "")),
                shard_vram_gb=max(0.0, float(entry.get("shard_vram_gb", 0))),
                shards_needed=max(1, int(entry.get("shards_needed", required))),
                quality_tier=str(entry.get("quality_tier", "standard")),
                # Phase 1 zero-config: plumb num_layers through so CapacityEngine
                # can divide shard_vram_gb across layers.
                num_layers=max(0, int(entry.get("num_layers", 0))),
            ))
        if not catalogue:
            raise RuntimeError("invalid_model_catalog: no valid entries")
        if self.config.default_model not in seen:
            catalogue.append(ModelAvailability(
                model_id=self.config.default_model,
                required_peers=self.config.required_replicas,
                hf_model_id=(self.config.default_model if "/" in str(self.config.default_model) else ""),
            ))
        return catalogue

    def _catalog_hf_model_id(self, model_id: str) -> str | None:
        """Look up the HuggingFace model ID for a catalog entry."""
        key = str(model_id or "").strip()
        if not key:
            return None
        item = self.catalogue_by_model.get(key)
        if item is None:
            return None
        value = str(getattr(item, "hf_model_id", "") or "").strip()
        return value or None

    def _normalize_peer_model(self, peer: PeerEndpoint) -> str:
        """Return the peer's model ID, falling back to the default model."""
        return peer.model_id or self.config.default_model

    def _required_replicas(self, model_id: str) -> int:
        """Return required replica count for a model from the catalog."""
        item = self.catalogue_by_model.get(model_id)
        return item.required_peers if item is not None else self.config.required_replicas

    # ══════════════════════════════════════════════════════════════════════
    # Kept: static / utility methods (no service extraction)
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _rotate(values: list, offset: int) -> list:
        if not values:
            return values
        offset = offset % len(values)
        return values[offset:] + values[:offset]

    @staticmethod
    def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
        lines = []
        for item in messages:
            role = str(item.get("role", "user")).strip() or "user"
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _normalize_decode_kwarg_aliases(kwargs: dict[str, Any]) -> dict[str, Any]:
        out = dict(kwargs)
        alias_map = {"do_sample": "decode_do_sample", "temperature": "decode_temperature",
                     "top_p": "decode_top_p", "top_k": "decode_top_k", "seed": "decode_seed"}
        for alias_key, decode_key in alias_map.items():
            if alias_key in out and decode_key not in out:
                out[decode_key] = out[alias_key]
            out.pop(alias_key, None)
        return out

    @staticmethod
    def _pipeline_uses_pytorch_runtime(pipeline: list[PeerEndpoint]) -> bool:
        if not pipeline:
            return False
        backends = [str(peer.runtime_backend or "").strip().lower() for peer in pipeline]
        return bool(backends) and all(item.startswith("pytorch") for item in backends)

    def _normalize_decode_controls(
        self, *, decode_do_sample: bool | None = None, decode_temperature: float | None = None,
        decode_top_p: float | None = None, decode_top_k: int | None = None,
        decode_seed: int | None = None,
    ) -> dict[str, Any]:
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

    # ══════════════════════════════════════════════════════════════════════
    # Kept: chat wrappers (call through self so monkeypatching works)
    # ══════════════════════════════════════════════════════════════════════

    def _messages_to_model_prompt(self, messages: list[dict[str, Any]], *, model_id: str | None = None) -> str:
        """Delegate to InferenceService for chat template rendering."""
        return self._inference_svc._messages_to_model_prompt(messages, model_id=model_id)

    def infer_chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        """Single-shot chat inference with default decode controls."""
        kwargs = self._normalize_decode_kwarg_aliases(kwargs)
        requested_model = str(kwargs.get("model_id", self.config.default_model) or self.config.default_model)
        prompt = self._messages_to_model_prompt(messages, model_id=requested_model)
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
        """Streaming chat inference with default decode controls."""
        kwargs = self._normalize_decode_kwarg_aliases(kwargs)
        requested_model = str(kwargs.get("model_id", self.config.default_model) or self.config.default_model)
        prompt = self._messages_to_model_prompt(messages, model_id=requested_model)
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

    # ══════════════════════════════════════════════════════════════════════
    # Facade delegates: TokenizationService
    # ══════════════════════════════════════════════════════════════════════

    def _load_generation_tokenizer(self, model_id: str) -> Any:
        """Delegate to TokenizationService."""
        return self._tokenization_svc._load_generation_tokenizer(model_id)

    def _load_pytorch_draft_model(self, *, tokenizer_model_id: str) -> PyTorchDraftModel:
        """Delegate to TokenizationService."""
        return self._tokenization_svc._load_pytorch_draft_model(tokenizer_model_id=tokenizer_model_id)

    # ══════════════════════════════════════════════════════════════════════
    # Facade delegates: KvAffinityService
    # ══════════════════════════════════════════════════════════════════════

    def _kv_affinity_key(self, session_id: str, model_id: str) -> tuple[str, str]:
        """Delegate to KvAffinityService."""
        return self._kv_affinity_svc._kv_affinity_key(session_id, model_id)

    def _purge_expired_kv_affinity(self) -> None:
        """Delegate to KvAffinityService."""
        return self._kv_affinity_svc._purge_expired_kv_affinity()

    def _get_kv_affinity_peer(self, session_id: str | None, model_id: str) -> str | None:
        """Delegate to KvAffinityService."""
        return self._kv_affinity_svc._get_kv_affinity_peer(session_id, model_id)

    def _set_kv_affinity_peer(self, session_id: str | None, model_id: str, peer_id: str | None) -> bool:
        """Delegate to KvAffinityService."""
        return self._kv_affinity_svc._set_kv_affinity_peer(session_id, model_id, peer_id)

    def _get_kv_affinity_activation(self, session_id: str | None, model_id: str) -> list[float] | None:
        """Delegate to KvAffinityService."""
        return self._kv_affinity_svc._get_kv_affinity_activation(session_id, model_id)

    def _get_kv_affinity_activation_peer(self, session_id: str | None, model_id: str) -> str | None:
        """Delegate to KvAffinityService."""
        return self._kv_affinity_svc._get_kv_affinity_activation_peer(session_id, model_id)

    def _set_kv_affinity_activation(self, session_id: str | None, model_id: str, activation: list[float] | None) -> bool:
        """Delegate to KvAffinityService."""
        return self._kv_affinity_svc._set_kv_affinity_activation(session_id, model_id, activation)

    # ══════════════════════════════════════════════════════════════════════
    # Facade delegates: HealthService
    # ══════════════════════════════════════════════════════════════════════

    def _record_ping_health(self, survey) -> None:
        """Delegate to HealthService."""
        return self._health_svc._record_ping_health(survey)

    def _verification_feedback_by_model(self, health) -> dict[str, dict[str, Any]]:
        """Delegate to HealthService."""
        return self._health_svc._verification_feedback_by_model(health)

    def _verification_metrics_for_model(self, model_id: str, health) -> dict[str, Any]:
        """Delegate to HealthService."""
        return self._health_svc._verification_metrics_for_model(model_id, health)

    def _apply_verification_feedback(self, *, primary: ChainResult, secondary: ChainResult | None,
                                     tertiary: ChainResult | None, verification: Any) -> dict[str, list[str]]:
        """Delegate to HealthService."""
        return self._health_svc._apply_verification_feedback(
            primary=primary, secondary=secondary, tertiary=tertiary, verification=verification)

    def _replication_dict(self, model_id: str, healthy_peers: int) -> dict[str, Any]:
        """Delegate to HealthService."""
        return self._health_svc._replication_dict(model_id, healthy_peers)

    def _discovered_peer_rows(self, health) -> list[dict[str, Any]]:
        """Delegate to HealthService."""
        return self._health_svc._discovered_peer_rows(health)

    # ══════════════════════════════════════════════════════════════════════
    # Facade delegates: MoeService
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _normalize_expert_tags(raw: Any) -> list[str]:
        """Delegate to MoeService."""
        return MoeService._normalize_expert_tags(raw)

    @staticmethod
    def _normalize_expert_layer_indices(raw: Any) -> list[int]:
        """Delegate to MoeService."""
        return MoeService._normalize_expert_layer_indices(raw)

    def _extract_prompt_expert_tags(self, prompt: str) -> list[str]:
        """Delegate to MoeService."""
        return self._moe_svc._extract_prompt_expert_tags(prompt)

    def _extract_prompt_expert_layer_indices(self, prompt: str) -> list[int]:
        """Delegate to MoeService."""
        return self._moe_svc._extract_prompt_expert_layer_indices(prompt)

    def _apply_moe_geo_sharding(self, pipeline: list[PeerEndpoint], ranked_candidates: list[PeerEndpoint], *,
                                prompt: str, requested_expert_tags: list[str] | None,
                                requested_expert_layer_indices: list[int] | None = None,
                                locked_first_peer_id: str | None = None) -> tuple[list[PeerEndpoint], dict[str, Any]]:
        """Delegate to MoeService."""
        return self._moe_svc._apply_moe_geo_sharding(
            pipeline, ranked_candidates, prompt=prompt, requested_expert_tags=requested_expert_tags,
            requested_expert_layer_indices=requested_expert_layer_indices,
            locked_first_peer_id=locked_first_peer_id)

    # ══════════════════════════════════════════════════════════════════════
    # Facade delegates: DiscoveryService
    # ══════════════════════════════════════════════════════════════════════

    def _dedupe_peer_entries(self, peers: list[PeerEndpoint]) -> list[PeerEndpoint]:
        """Delegate to DiscoveryService."""
        return self._discovery_svc._dedupe_peer_entries(peers)

    def _configured_dht_urls(self) -> list[str]:
        """Delegate to DiscoveryService."""
        return self._discovery_svc._configured_dht_urls()

    def _load_candidate_peers(self, model_ids: list[str] | None = None) -> list[PeerEndpoint]:
        """Delegate to DiscoveryService."""
        return self._discovery_svc._load_candidate_peers(model_ids=model_ids)

    def _cache_dht_peers(self, *, model_id: str, peers: list[PeerEndpoint]) -> None:
        """Delegate to DiscoveryService."""
        return self._discovery_svc._cache_dht_peers(model_id=model_id, peers=peers)

    def _cached_dht_peers(self, *, model_id: str) -> list[PeerEndpoint]:
        """Delegate to DiscoveryService."""
        return self._discovery_svc._cached_dht_peers(model_id=model_id)

    def _scan_network(self, model_ids: list[str] | None = None):
        """Delegate to DiscoveryService."""
        return self._discovery_svc._scan_network(model_ids=model_ids)

    def _discover_for_model(self, requested_model: str, allow_degradation: bool):
        """Delegate to DiscoveryService."""
        return self._discovery_svc._discover_for_model(requested_model, allow_degradation)

    def _discover(self):
        """Delegate to DiscoveryService."""
        return self._discovery_svc._discover()

    def _resolve_runtime_model_id(self, model_id: str) -> str:
        """Delegate to TokenizationService."""
        return self._tokenization_svc._resolve_runtime_model_id(model_id)

    def _resolve_pipeline_runtime_model_id(self, pipeline: list[PeerEndpoint], served_model: str) -> str:
        """Delegate to TokenizationService."""
        return self._tokenization_svc._resolve_pipeline_runtime_model_id(pipeline, served_model)

    def _catalog_model_ids(self) -> list[str]:
        """Return all model IDs in the catalog."""
        return [item.model_id for item in self.model_catalog]

    # ══════════════════════════════════════════════════════════════════════
    # Facade delegates: PipelineService
    # ══════════════════════════════════════════════════════════════════════

    def _select_pipeline_sharded(self, health: list) -> list[PeerEndpoint] | None:
        """Delegate to PipelineService."""
        return self._pipeline_svc._select_pipeline_sharded(health)

    def _select_pipeline(self, candidates: list, pipeline_width: int | None = None) -> list:
        """Delegate to PipelineService."""
        return self._pipeline_svc._select_pipeline(candidates, pipeline_width=pipeline_width)

    def _role_for_peer(self, peer: PeerEndpoint) -> str:
        """Delegate to PipelineService."""
        return self._pipeline_svc._role_for_peer(peer)

    def _reorder_for_decode_tail(self, peers: list[PeerEndpoint]) -> list[PeerEndpoint]:
        """Delegate to PipelineService."""
        return self._pipeline_svc._reorder_for_decode_tail(peers)

    def _apply_bandwidth_asymmetry(self, pipeline: list[PeerEndpoint], ranked_candidates: list[PeerEndpoint],
                                   prompt_tokens_est: int, *, session_id: str | None = None,
                                   model_id: str | None = None) -> tuple[list[PeerEndpoint], dict[str, Any]]:
        """Delegate to PipelineService."""
        return self._pipeline_svc._apply_bandwidth_asymmetry(
            pipeline, ranked_candidates, prompt_tokens_est,
            session_id=session_id, model_id=model_id)

    # ══════════════════════════════════════════════════════════════════════
    # Facade delegates: InferenceService
    # ══════════════════════════════════════════════════════════════════════

    def _grounding_meta(self, *, enabled: bool, snippets: list[str], grounding_result: Any | None) -> dict[str, Any]:
        """Delegate to InferenceService."""
        return self._inference_svc._grounding_meta(enabled=enabled, snippets=snippets, grounding_result=grounding_result)

    def _model_meta(self, decision: DegradationDecision) -> dict[str, Any]:
        """Delegate to InferenceService."""
        return self._inference_svc._model_meta(decision)

    def _run_chain(self, prompt: str, candidates: list, pipeline: list, max_tokens: int,
                   request_id: str | None = None, initial_activation: list[float] | None = None,
                   kv_session_id: str | None = None, kv_use_cached_activation: bool = False,
                   kv_store_activation: bool = False, kv_cache_stage_index: int = 0,
                   kv_cache_all_stages: bool = False, decode_do_sample: bool | None = None,
                   decode_temperature: float | None = None, decode_top_p: float | None = None,
                   decode_top_k: int | None = None, decode_seed: int | None = None,
                   deadline: float | None = None):
        """Delegate to InferenceService."""
        return self._inference_svc._run_chain(
            prompt, candidates, pipeline, max_tokens, request_id=request_id,
            initial_activation=initial_activation, kv_session_id=kv_session_id,
            kv_use_cached_activation=kv_use_cached_activation,
            kv_store_activation=kv_store_activation, kv_cache_stage_index=kv_cache_stage_index,
            kv_cache_all_stages=kv_cache_all_stages, decode_do_sample=decode_do_sample,
            decode_temperature=decode_temperature, decode_top_p=decode_top_p,
            decode_top_k=decode_top_k, decode_seed=decode_seed, deadline=deadline)

    def _prepare_inference(self, *, prompt: str, pipeline_width: int | None, grounding: bool,
                           model_id: str | None, allow_degradation: bool | None,
                           session_id: str | None, expert_tags: list[str] | None = None,
                           expert_layer_indices: list[int] | None = None) -> InferencePreparation:
        """Delegate to InferenceService."""
        return self._inference_svc._prepare_inference(
            prompt=prompt, pipeline_width=pipeline_width, grounding=grounding,
            model_id=model_id, allow_degradation=allow_degradation,
            session_id=session_id, expert_tags=expert_tags,
            expert_layer_indices=expert_layer_indices)

    def infer(self, *, prompt: str, max_tokens: int = 24, pipeline_width: int | None = None,
              grounding: bool = True, priority: bool = False, client_id: str = "anonymous",
              model_id: str | None = None, allow_degradation: bool | None = None,
              session_id: str | None = None, expert_tags: list[str] | None = None,
              expert_layer_indices: list[int] | None = None, decode_do_sample: bool | None = None,
              decode_temperature: float | None = None, decode_top_p: float | None = None,
              decode_top_k: int | None = None, decode_seed: int | None = None,
              request_id: str | None = None) -> dict[str, Any]:
        """Execute a single-shot inference request. Delegate to InferenceService."""
        return self._inference_svc.infer(
            prompt=prompt, max_tokens=max_tokens, pipeline_width=pipeline_width,
            grounding=grounding, priority=priority, client_id=client_id,
            model_id=model_id, allow_degradation=allow_degradation,
            session_id=session_id, expert_tags=expert_tags,
            expert_layer_indices=expert_layer_indices, decode_do_sample=decode_do_sample,
            decode_temperature=decode_temperature, decode_top_p=decode_top_p,
            decode_top_k=decode_top_k, decode_seed=decode_seed, request_id=request_id)

    def infer_stream(self, *, prompt: str, max_tokens: int = 24, pipeline_width: int | None = None,
                     grounding: bool = True, priority: bool = False, client_id: str = "anonymous",
                     model_id: str | None = None, allow_degradation: bool | None = None,
                     session_id: str | None = None, expert_tags: list[str] | None = None,
                     expert_layer_indices: list[int] | None = None, decode_do_sample: bool | None = None,
                     decode_temperature: float | None = None, decode_top_p: float | None = None,
                     decode_top_k: int | None = None, decode_seed: int | None = None,
                     request_id: str | None = None) -> dict[str, Any]:
        """Execute a streaming inference request. Delegate to InferenceService."""
        return self._inference_svc.infer_stream(
            prompt=prompt, max_tokens=max_tokens, pipeline_width=pipeline_width,
            grounding=grounding, priority=priority, client_id=client_id,
            model_id=model_id, allow_degradation=allow_degradation,
            session_id=session_id, expert_tags=expert_tags,
            expert_layer_indices=expert_layer_indices, decode_do_sample=decode_do_sample,
            decode_temperature=decode_temperature, decode_top_p=decode_top_p,
            decode_top_k=decode_top_k, decode_seed=decode_seed, request_id=request_id)

    # ══════════════════════════════════════════════════════════════════════
    # Facade delegates: StatusService
    # ══════════════════════════════════════════════════════════════════════

    def list_models(self) -> dict[str, Any]:
        """Return the model catalog. Delegate to StatusService."""
        if not hasattr(self, "_status_svc"):
            # Fallback for partial engines created via __new__ without __init__.
            return self._list_models_inline()
        return self._status_svc.list_models()

    def _list_models_inline(self) -> dict[str, Any]:
        """Inline list_models for partial engines without extracted services."""
        import time as _time
        data = []
        for model in self.model_catalog:
            healthy_count = 0
            if hasattr(self, "_dht_peer_cache"):
                with self._dht_peer_cache_lock:
                    item = self._dht_peer_cache.get(model.model_id)
                    if item and _time.time() < float(item.get("expires_at", 0)):
                        healthy_count = len(item.get("peers", []))
            if hasattr(self, "replication_monitor"):
                status = self.replication_monitor.evaluate(model.model_id, healthy_count, required_replicas=model.required_peers)
                replication = self.replication_monitor.to_dict(status)
            else:
                replication = {"under_replicated": False}
            data.append({
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
            })
        return {"object": "list", "data": data}

    def network_status(self) -> dict[str, Any]:
        """Return live network health status. Delegate to StatusService."""
        return self._status_svc.network_status()

    def metrics_snapshot(self) -> dict[str, float | int]:
        """Return operational counter snapshot. Delegate to StatusService."""
        return self._status_svc.metrics_snapshot()

    # ══════════════════════════════════════════════════════════════════════
    # Facade delegates: EconomyService
    # ══════════════════════════════════════════════════════════════════════

    def _record_channel_provider_spend(self, channel_id: str, provider_peer_id: str, amount: float) -> None:
        """Delegate to EconomyService."""
        self._economy._record_channel_provider_spend(channel_id, provider_peer_id, amount)

    def _set_channel_payee_spend(self, channel_id: str, payee_peer_id: str, total_spent: float) -> None:
        """Delegate to EconomyService."""
        self._economy._set_channel_payee_spend(channel_id, payee_peer_id, total_spent)

    def _bridge_settle_channel_close(self, close_payload: dict[str, Any]) -> dict[str, Any]:
        """Delegate to EconomyService."""
        return self._economy._bridge_settle_channel_close(close_payload)

    def account_balance(self, client_id: str) -> dict[str, Any]:
        """Return combined barter + HYDRA balance. Delegate to EconomyService."""
        return self._economy.account_balance(client_id)

    def hydra_status(self) -> dict[str, Any]:
        """Return HYDRA economy summary. Delegate to EconomyService."""
        return self._economy.hydra_status()

    def hydra_account(self, client_id: str) -> dict[str, Any]:
        """Return HYDRA account snapshot. Delegate to EconomyService."""
        return self._economy.hydra_account(client_id)

    def hydra_governance_params(self) -> dict[str, Any]:
        """Return governance parameters. Delegate to EconomyService."""
        return self._economy.hydra_governance_params()

    def hydra_governance_vote(self, pubkey: str, proposal_id: str, vote: str) -> dict[str, Any]:
        """Submit a governance vote. Delegate to EconomyService."""
        return self._economy.hydra_governance_vote(pubkey, proposal_id, vote)

    def hydra_transfer(self, from_client_id: str, to_client_id: str, amount: float) -> dict[str, Any]:
        """Transfer HYDRA tokens. Delegate to EconomyService."""
        return self._economy.hydra_transfer(from_client_id, to_client_id, amount)

    def hydra_stake(self, client_id: str, amount: float) -> dict[str, Any]:
        """Stake HYDRA tokens. Delegate to EconomyService."""
        return self._economy.hydra_stake(client_id, amount)

    def hydra_unstake(self, client_id: str, amount: float) -> dict[str, Any]:
        """Unstake HYDRA tokens. Delegate to EconomyService."""
        return self._economy.hydra_unstake(client_id, amount)

    def hydra_open_channel(self, channel_id: str, payer: str, payee: str, deposit: float, ttl_seconds: int | None = None) -> dict[str, Any]:
        """Open a state channel. Delegate to EconomyService."""
        return self._economy.hydra_open_channel(channel_id, payer, payee, deposit, ttl_seconds)

    def hydra_charge_channel(self, channel_id: str, amount: float, provider_peer_id: str | None = None) -> dict[str, Any]:
        """Charge a state channel. Delegate to EconomyService."""
        return self._economy.hydra_charge_channel(channel_id, amount, provider_peer_id)

    def hydra_reconcile_channel(self, channel_id: str, total_spent: float, nonce: int) -> dict[str, Any]:
        """Reconcile a state channel. Delegate to EconomyService."""
        return self._economy.hydra_reconcile_channel(channel_id, total_spent, nonce)

    def hydra_close_channel(self, channel_id: str) -> dict[str, Any]:
        """Close a state channel and settle. Delegate to EconomyService."""
        return self._economy.hydra_close_channel(channel_id)

    # ══════════════════════════════════════════════════════════════════════
    # Kept: close
    # ══════════════════════════════════════════════════════════════════════

    def close(self) -> None:
        """Shut down all persistent resources (health store, ledgers)."""
        self.health.close()
        self.ledger.close()
        self.hydra.close()
