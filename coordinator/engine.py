from __future__ import annotations

from dataclasses import asdict, dataclass, field
from concurrent.futures import Future, ThreadPoolExecutor
import json
import logging
from pathlib import Path
import re
import statistics
import threading
import time
import uuid
from typing import Any

from openhydra_defaults import PRODUCTION_BOOTSTRAP_URLS

logger = logging.getLogger(__name__)

from coordinator.bandwidth_roles import (
    RoleThresholds,
    classify_role,
    estimate_prompt_tokens,
    role_counts_from_bandwidth,
)
from coordinator.chain import ChainResult, InferenceChain
from coordinator.concentration_guard import assemble_pipeline, concentration_metrics
from coordinator.degradation import DegradationDecision, DegradationPolicy, ModelAvailability
from coordinator.health_scorer import HealthScorer
from coordinator.mystery_shopper import MysteryShopper
from coordinator.path_finder import PathFinder, PeerEndpoint, load_peer_config, load_peers_from_dht
from coordinator.peer_selector import ScoredPeer, rank_peers
from coordinator.replication_monitor import ReplicationMonitor
from coordinator.speculative import (
    DraftTokenModel,
    PyTorchDraftModel,
    select_verified_token_ids,
    select_verified_tokens,
)
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
    # Phase 3: "sharded" when LayerCoverageMap assembled the pipeline;
    # "full_model" for the legacy full-model-replica path.
    pipeline_mode: str = "full_model"


class CoordinatorEngine:
    def __init__(self, config: EngineConfig):
        self.config = config
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
            int(_recovery["open_channels"]),
            int(_recovery["expired_on_recovery"]),
            int(_recovery["total_accounts"]),
            float(_recovery["total_minted"]),
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
        self.health = HealthScorer(config.health_store_path)
        self.replication_monitor = ReplicationMonitor(required_replicas=config.required_replicas)
        self.transport_config = TransportConfig(
            tls_enabled=config.tls_enabled,
            root_cert_path=config.tls_root_cert_path,
            client_cert_path=config.tls_client_cert_path,
            client_key_path=config.tls_client_key_path,
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
        self.grounding_client = GroundingClient(
            GroundingConfig(
                cache_path=config.grounding_cache_path,
                cache_ttl_seconds=max(1, config.grounding_cache_ttl_seconds),
                timeout_s=max(0.1, config.grounding_timeout_s),
                use_network=config.grounding_use_network,
                fallback_enabled=config.grounding_fallback_enabled,
            )
        )
        verification_rate = config.redundant_exec_rate if config.tier >= 2 else config.audit_rate
        verification_mode = "redundant_execution" if config.tier >= 2 else "mystery_shopper"
        self.verifier = MysteryShopper(
            sample_rate=verification_rate,
            seed=config.seed,
            mode=verification_mode,
            auditor_sample_rate=max(0.0, min(1.0, config.auditor_rate)),
        )
        self._last_verification_qos = {
            "enabled": False,
            "min_events": 0,
            "min_success_rate": 0.0,
            "requested_model_blocked": False,
            "requested_model_events": 0,
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
        # ── KV + inference proxy counters (Phase D) ───────────────────────────
        self._kv_store_ops_total: int = 0
        self._kv_retrieve_ops_total: int = 0
        self._inference_requests_total: int = 0
        # ── Phase 2: Auto-scaler + demand tracking ─────────────────────────────
        from coordinator.request_log import RequestLog
        from coordinator.auto_scaler import AutoScaler, ModelSpec as _ScalerModelSpec
        self._request_log = RequestLog()
        _scaler_specs: list[_ScalerModelSpec] = []
        for _item in self.model_catalog:
            if _item.shard_vram_gb > 0:
                _scaler_specs.append(
                    _ScalerModelSpec(
                        model_id=_item.model_id,
                        shard_vram_mb=int(_item.shard_vram_gb * 1024),
                        shards_needed=max(1, _item.shards_needed),
                        quality_tier=_item.quality_tier,
                        required_peers=_item.required_peers,
                    )
                )
        if _scaler_specs:
            self._auto_scaler: AutoScaler | None = AutoScaler(_scaler_specs)
            self._active_model_roster: list[str] = list(self._auto_scaler.active_roster)
        else:
            self._auto_scaler = None
            self._active_model_roster = [self.config.default_model]

    def _hydra_stake_balance(self, pubkey: str) -> float:
        try:
            snapshot = self.hydra.account_snapshot(pubkey)
        except RuntimeError:
            return 0.0
        return max(0.0, float(snapshot.get("stake", 0.0)))

    def _hydra_slash_stake(self, pubkey: str, amount: float) -> float:
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

    def _messages_to_model_prompt(
        self,
        messages: list[dict[str, Any]],
        *,
        model_id: str | None = None,
    ) -> str:
        fallback = self._messages_to_prompt(messages)
        if not bool(self.config.pytorch_chat_template_enabled):
            return fallback

        requested_model = str(model_id or self.config.default_model).strip() or self.config.default_model
        runtime_model = self._resolve_runtime_model_id(requested_model)
        try:
            tokenizer = self._load_generation_tokenizer(runtime_model)
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

    def _normalize_decode_controls(
        self,
        *,
        decode_do_sample: bool | None = None,
        decode_temperature: float | None = None,
        decode_top_p: float | None = None,
        decode_top_k: int | None = None,
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

    @staticmethod
    def _normalize_decode_kwarg_aliases(kwargs: dict[str, Any]) -> dict[str, Any]:
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
        if not pipeline:
            return False
        backends = [str(peer.runtime_backend or "").strip().lower() for peer in pipeline]
        return bool(backends) and all(item.startswith("pytorch") for item in backends)

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
        tokenizer = AutoTokenizer.from_pretrained(
            normalized,
            trust_remote_code=_default_trust_remote_code(normalized),
        )
        self._tokenizer_cache[normalized] = tokenizer
        return tokenizer

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

    def _load_model_catalog(self) -> list[ModelAvailability]:
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
        key = str(model_id or "").strip()
        if not key:
            return None
        item = self.catalogue_by_model.get(key)
        if item is None:
            return None
        value = str(getattr(item, "hf_model_id", "") or "").strip()
        return value or None

    def _resolve_runtime_model_id(self, model_id: str) -> str:
        requested = str(model_id or "").strip()
        if not requested:
            return str(self.config.pytorch_generation_model_id or "gpt2").strip() or "gpt2"
        catalog_hf = self._catalog_hf_model_id(requested)
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
        return self._resolve_runtime_model_id(served_model)

    def _catalog_model_ids(self) -> list[str]:
        return [item.model_id for item in self.model_catalog]

    def _required_replicas(self, model_id: str) -> int:
        item = self.catalogue_by_model.get(model_id)
        if item is None:
            return self.config.required_replicas
        return item.required_peers

    def _normalize_peer_model(self, peer: PeerEndpoint) -> str:
        return peer.model_id or self.config.default_model

    def _record_ping_health(self, survey) -> None:
        for item in survey:
            self.health.record_ping(item.peer.peer_id, healthy=item.healthy, latency_ms=item.latency_ms)

    def _dedupe_peer_entries(self, peers: list[PeerEndpoint]) -> list[PeerEndpoint]:
        deduped: dict[tuple[str, str], PeerEndpoint] = {}
        for peer in peers:
            model_id = self._normalize_peer_model(peer)
            key = (peer.peer_id, model_id)
            deduped[key] = PeerEndpoint(
                peer_id=peer.peer_id,
                host=peer.host,
                port=peer.port,
                model_id=model_id,
                operator_id=peer.operator_id,
                region=peer.region,
                bandwidth_mbps=peer.bandwidth_mbps,
                seeding_enabled=peer.seeding_enabled,
                seed_upload_limit_mbps=peer.seed_upload_limit_mbps,
                seed_target_upload_limit_mbps=peer.seed_target_upload_limit_mbps,
                seed_inference_active=peer.seed_inference_active,
                runtime_backend=peer.runtime_backend,
                runtime_target=peer.runtime_target,
                runtime_model_id=peer.runtime_model_id,
                quantization_mode=peer.quantization_mode,
                quantization_bits=peer.quantization_bits,
                runtime_gpu_available=peer.runtime_gpu_available,
                runtime_estimated_tokens_per_sec=peer.runtime_estimated_tokens_per_sec,
                runtime_estimated_memory_mb=peer.runtime_estimated_memory_mb,
                privacy_noise_variance=peer.privacy_noise_variance,
                privacy_noise_payloads=peer.privacy_noise_payloads,
                privacy_noise_observed_variance_ema=peer.privacy_noise_observed_variance_ema,
                privacy_noise_last_audit_tag=peer.privacy_noise_last_audit_tag,
                reputation_score=peer.reputation_score,
                staked_balance=peer.staked_balance,
                expert_tags=tuple(peer.expert_tags),
                expert_layer_indices=tuple(peer.expert_layer_indices),
                expert_router=peer.expert_router,
                expert_admission_approved=peer.expert_admission_approved,
                expert_admission_reason=peer.expert_admission_reason,
                )
        return list(deduped.values())

    def _configured_dht_urls(self) -> list[str]:
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

    def _load_candidate_peers(self, model_ids: list[str] | None = None) -> list[PeerEndpoint]:
        peers: list[PeerEndpoint] = []
        model_filter = set(model_ids or self._catalog_model_ids())

        if self.config.peers_config_path:
            path = Path(self.config.peers_config_path)
            if not path.exists():
                raise RuntimeError(f"peer_config_not_found: {path}")
            for peer in load_peer_config(path):
                model_id = peer.model_id or self.config.default_model
                runtime_id = peer.runtime_model_id or ""
                if model_id in model_filter or (runtime_id and runtime_id in model_filter):
                    peers.append(
                        PeerEndpoint(
                            peer_id=peer.peer_id,
                            host=peer.host,
                            port=peer.port,
                            model_id=model_id,
                            operator_id=peer.operator_id,
                            region=peer.region,
                            bandwidth_mbps=peer.bandwidth_mbps,
                            seeding_enabled=peer.seeding_enabled,
                            seed_upload_limit_mbps=peer.seed_upload_limit_mbps,
                            seed_target_upload_limit_mbps=peer.seed_target_upload_limit_mbps,
                            seed_inference_active=peer.seed_inference_active,
                            runtime_backend=peer.runtime_backend,
                            runtime_target=peer.runtime_target,
                            runtime_model_id=peer.runtime_model_id,
                            quantization_mode=peer.quantization_mode,
                            quantization_bits=peer.quantization_bits,
                            runtime_gpu_available=peer.runtime_gpu_available,
                            runtime_estimated_tokens_per_sec=peer.runtime_estimated_tokens_per_sec,
                            runtime_estimated_memory_mb=peer.runtime_estimated_memory_mb,
                            privacy_noise_variance=peer.privacy_noise_variance,
                            privacy_noise_payloads=peer.privacy_noise_payloads,
                            privacy_noise_observed_variance_ema=peer.privacy_noise_observed_variance_ema,
                            privacy_noise_last_audit_tag=peer.privacy_noise_last_audit_tag,
                            reputation_score=peer.reputation_score,
                            staked_balance=peer.staked_balance,
                            expert_tags=tuple(peer.expert_tags),
                            expert_layer_indices=tuple(peer.expert_layer_indices),
                            expert_router=peer.expert_router,
                            expert_admission_approved=peer.expert_admission_approved,
                            expert_admission_reason=peer.expert_admission_reason,
                        )
                    )

        dht_sources = self._configured_dht_urls()
        if dht_sources:
            dht_errors: list[Exception] = []
            for model_id in model_filter:
                with self._metrics_lock:
                    self._dht_lookup_attempts += 1
                try:
                    if len(dht_sources) == 1:
                        dht_peers = load_peers_from_dht(
                            dht_sources[0],
                            model_id=model_id,
                            timeout_s=self.config.dht_lookup_timeout_s,
                            preferred_region=self.config.dht_preferred_region,
                            limit=(self.config.dht_lookup_limit if self.config.dht_lookup_limit > 0 else None),
                            sloppy_factor=max(0, int(self.config.dht_lookup_sloppy_factor)),
                            dsht_replicas=max(0, int(self.config.dht_lookup_dsht_replicas)),
                        )
                    else:
                        dht_peers = load_peers_from_dht(
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

        peers = self._dedupe_peer_entries(peers)
        if not peers:
            raise RuntimeError("no_peers_from_sources")
        return peers

    def _cache_dht_peers(self, *, model_id: str, peers: list[PeerEndpoint]) -> None:
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

    def _scan_network(self, model_ids: list[str] | None = None):
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

    def _discover_for_model(self, requested_model: str, allow_degradation: bool):
        requested_model = str(requested_model or self.config.default_model).strip() or self.config.default_model
        scan_models = list(self._catalog_model_ids())
        if bool(self.config.allow_dynamic_model_ids) and requested_model not in scan_models:
            scan_models.append(requested_model)
        # Resolve catalog alias → HF model ID; include both in scan for bidirectional peer matching
        _catalog_hf = self._catalog_hf_model_id(requested_model) or ""
        if _catalog_hf and _catalog_hf != requested_model and _catalog_hf not in scan_models:
            scan_models.append(_catalog_hf)
        healthy, counts = self._scan_network(model_ids=scan_models)
        # Merge alias count ↔ HF ID count (same physical model, different announcement forms)
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
        self._last_verification_qos = {
            "enabled": qos_enabled,
            "min_events": qos_min_events,
            "min_success_rate": round(qos_min_success, 6),
            "requested_model_blocked": requested_qos is not None,
            "requested_model_events": int((requested_qos or {}).get("verification_events", 0)),
            "requested_model_success_rate": (
                float(requested_qos["verification_success_rate"]) if requested_qos is not None else None
            ),
        }

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

    def _discover(self):
        health, candidates, _, _ = self._discover_for_model(
            requested_model=self.config.default_model,
            allow_degradation=self.config.allow_degradation_default,
        )
        return health, candidates

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

    def _select_pipeline(self, candidates: list, pipeline_width: int | None = None) -> list:
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

    def _role_for_peer(self, peer: PeerEndpoint) -> str:
        return classify_role(peer.bandwidth_mbps, thresholds=self.role_thresholds)

    def _reorder_for_decode_tail(self, peers: list[PeerEndpoint]) -> list[PeerEndpoint]:
        if len(peers) <= 1:
            return peers
        decode_only = [peer for peer in peers if self._role_for_peer(peer) == "decode_only"]
        others = [peer for peer in peers if self._role_for_peer(peer) != "decode_only"]
        return others + decode_only

    def _apply_bandwidth_asymmetry(
        self,
        pipeline: list[PeerEndpoint],
        ranked_candidates: list[PeerEndpoint],
        prompt_tokens_est: int,
        *,
        session_id: str | None = None,
        model_id: str | None = None,
    ) -> tuple[list[PeerEndpoint], dict[str, Any]]:
        served_model = model_id or self.config.default_model
        kv_affinity_requested = bool(self.config.kv_affinity_enabled and session_id)
        previous_prefill_peer_id = self._get_kv_affinity_peer(session_id, served_model)
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
            kv_affinity_updated = self._set_kv_affinity_peer(session_id, served_model, chosen_prefill.peer_id)

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
        chain = InferenceChain(
            pipeline,
            timeout_ms=self.config.timeout_ms,
            transport_config=self.transport_config,
            tensor_autoencoder_enabled=self.config.tensor_autoencoder_enabled,
            tensor_autoencoder_latent_dim=max(1, int(self.config.tensor_autoencoder_latent_dim)),
            advanced_encryption_enabled=self.config.advanced_encryption_enabled,
            advanced_encryption_seed=str(self.config.advanced_encryption_seed),
            advanced_encryption_level=str(self.config.advanced_encryption_level),
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

        # ── Phase D: increment inference + KV proxy counters ─────────────────
        with self._metrics_lock:
            self._inference_requests_total += 1
            if kv_session_id:
                if kv_store_activation:
                    self._kv_store_ops_total += 1
                if kv_use_cached_activation:
                    self._kv_retrieve_ops_total += 1
        # ─────────────────────────────────────────────────────────────────────

        result = chain.run(
            prompt,
            **run_kwargs,
        )

        for trace in result.traces:
            self.health.record_inference(trace.peer_id, success=True, latency_ms=trace.latency_ms)
            if trace.failed_peer_id:
                self.health.record_inference(trace.failed_peer_id, success=False)

        return result

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

    def _replication_dict(self, model_id: str, healthy_peers: int) -> dict[str, Any]:
        status = self.replication_monitor.evaluate(
            model_id,
            healthy_peers,
            required_replicas=self._required_replicas(model_id),
        )
        return self.replication_monitor.to_dict(status)

    def _discovered_peer_rows(self, health) -> list[dict[str, Any]]:
        scored_lookup = {item.peer.peer_id: item for item in self._last_scored_peers}
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

    def _kv_affinity_key(self, session_id: str, model_id: str) -> tuple[str, str]:
        return session_id, model_id

    def _purge_expired_kv_affinity(self) -> None:
        if not self._kv_affinity:
            return
        now = time.time()
        expired = [
            key
            for key, item in self._kv_affinity.items()
            if float(item.get("expires_at", 0.0)) < now
        ]
        for key in expired:
            self._kv_affinity.pop(key, None)

    def _get_kv_affinity_peer(self, session_id: str | None, model_id: str) -> str | None:
        if not self.config.kv_affinity_enabled or not session_id:
            return None
        self._purge_expired_kv_affinity()
        item = self._kv_affinity.get(self._kv_affinity_key(session_id, model_id))
        if item is None:
            return None
        peer_id = str(item.get("prefill_peer_id", "")).strip()
        return peer_id or None

    def _set_kv_affinity_peer(self, session_id: str | None, model_id: str, peer_id: str | None) -> bool:
        if not self.config.kv_affinity_enabled or not session_id or not peer_id:
            return False
        now = time.time()
        ttl = max(1, int(self.config.kv_affinity_ttl_seconds))
        key = self._kv_affinity_key(session_id, model_id)
        previous = self._kv_affinity.get(key, {})
        activation_cache = previous.get("activation")
        activation_peer_id = previous.get("activation_peer_id")
        self._kv_affinity[key] = {
            "prefill_peer_id": peer_id,
            "updated_at": now,
            "expires_at": now + ttl,
            "activation": activation_cache,
            "activation_peer_id": activation_peer_id,
            "activation_updated_at": previous.get("activation_updated_at"),
        }
        return True

    def _get_kv_affinity_activation(self, session_id: str | None, model_id: str) -> list[float] | None:
        if not self.config.kv_affinity_enabled or not session_id:
            return None
        self._purge_expired_kv_affinity()
        item = self._kv_affinity.get(self._kv_affinity_key(session_id, model_id))
        if item is None:
            return None
        raw = item.get("activation")
        if not isinstance(raw, list) or not raw:
            return None
        out: list[float] = []
        for value in raw:
            try:
                out.append(float(value))
            except (TypeError, ValueError):
                return None
        return out or None

    def _get_kv_affinity_activation_peer(self, session_id: str | None, model_id: str) -> str | None:
        if not self.config.kv_affinity_enabled or not session_id:
            return None
        self._purge_expired_kv_affinity()
        item = self._kv_affinity.get(self._kv_affinity_key(session_id, model_id))
        if item is None:
            return None
        peer_id = str(item.get("activation_peer_id", "")).strip()
        return peer_id or None

    def _set_kv_affinity_activation(
        self,
        session_id: str | None,
        model_id: str,
        activation: list[float] | None,
    ) -> bool:
        if not self.config.kv_affinity_enabled or not session_id or not activation:
            return False
        key = self._kv_affinity_key(session_id, model_id)
        now = time.time()
        ttl = max(1, int(self.config.kv_affinity_ttl_seconds))
        item = dict(self._kv_affinity.get(key, {}))
        item["activation"] = [float(v) for v in activation]
        item["activation_updated_at"] = now
        item["activation_peer_id"] = str(item.get("prefill_peer_id", "")).strip() or None
        item["updated_at"] = now
        item["expires_at"] = now + ttl
        self._kv_affinity[key] = item
        return True

    def _grounding_meta(self, *, enabled: bool, snippets: list[str], grounding_result: Any | None) -> dict[str, Any]:
        return {
            "enabled": enabled,
            "snippets": snippets,
            "provider": (grounding_result.provider if grounding_result else "disabled"),
            "cached": (grounding_result.cached if grounding_result else False),
            "fallback_used": (grounding_result.fallback_used if grounding_result else False),
            "error": (grounding_result.error if grounding_result else None),
        }

    def _model_meta(self, decision: DegradationDecision) -> dict[str, Any]:
        return {
            "requested": decision.requested_model,
            "served": decision.served_model,
            "degraded": decision.degraded,
            "available": decision.available,
            "reason": decision.reason,
            "detail": decision.detail,
            "verification_qos": dict(self._last_verification_qos),
        }

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
        requested_model = model_id or self.config.default_model
        allow_deg = self.config.allow_degradation_default if allow_degradation is None else bool(allow_degradation)

        effective_prompt = prompt
        grounding_result = None
        snippets: list[str] = []
        if grounding:
            grounding_result = self.grounding_client.search(prompt, max_snippets=3)
            snippets = grounding_result.snippets
            effective_prompt = inject_grounding(prompt, snippets)

        health, candidates, decision, counts = self._discover_for_model(
            requested_model=requested_model,
            allow_degradation=allow_deg,
        )
        prompt_tokens_est = estimate_prompt_tokens(effective_prompt)

        # ── Phase 3: try layer-sharded pipeline first ─────────────────────────
        # When the healthy peer set forms complete layer coverage the coordinator
        # assembles an ordered shard pipeline and bypasses the full-model peer
        # selection, bandwidth asymmetry, and MoE geo-sharding steps (which all
        # assume every peer runs the complete model).
        _sharded_pipeline = self._select_pipeline_sharded(health)
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
        # ── Legacy: full-model peer pipeline ─────────────────────────────────

        primary_pipeline = self._select_pipeline(candidates, pipeline_width=pipeline_width)
        primary_pipeline, primary_bandwidth_policy = self._apply_bandwidth_asymmetry(
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
        primary_pipeline, primary_moe_policy = self._apply_moe_geo_sharding(
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
        request_id = request_id or str(uuid.uuid4())
        max_tokens = int(max_tokens or 1024)
        logger.info(
            "infer_start req_id=%s model=%s client=%s",
            request_id, model_id or self.config.default_model, client_id,
        )
        if priority and not self.ledger.spend(client_id, 1.0):
            raise RuntimeError("insufficient_priority_credits")

        # Phase 2: record demand signal for the auto-scaler.
        self._request_log.record(str(model_id or self.config.default_model))

        # Compute an absolute deadline for this request so every gRPC hop
        # respects the overall latency budget rather than getting a fresh window.
        deadline = time.time() + self.config.max_latency_ms / 1000.0

        prep = self._prepare_inference(
            prompt=prompt,
            pipeline_width=pipeline_width,
            grounding=grounding,
            model_id=model_id,
            allow_degradation=allow_degradation,
            session_id=session_id,
            expert_tags=expert_tags,
            expert_layer_indices=expert_layer_indices,
        )
        # Elastic output cap: 2048 floor, up to 8192 if redundancy >= 3.0
        _served = prep.decision.served_model
        _available = prep.counts.get(_served, 0)
        _required = self._required_replicas(_served)
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
        primary = self._run_chain(
            prep.effective_prompt,
            prep.candidates,
            prep.primary_pipeline,
            max_tokens=max_tokens,
            request_id=request_id,
            deadline=deadline,
            **decode_controls,
        )
        prompt_tokens_est = estimate_prompt_tokens(prep.effective_prompt)

        secondary_pipeline = self._select_pipeline(self._rotate(prep.candidates, 1), pipeline_width=pipeline_width)
        secondary_pipeline, secondary_bandwidth_policy = self._apply_bandwidth_asymmetry(
            secondary_pipeline,
            self._rotate(prep.candidates, 1),
            prompt_tokens_est=prompt_tokens_est,
        )
        secondary_pipeline, _ = self._apply_moe_geo_sharding(
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
        tertiary_pipeline = self._select_pipeline(self._rotate(prep.candidates, 2), pipeline_width=pipeline_width)
        tertiary_pipeline, tertiary_bandwidth_policy = self._apply_bandwidth_asymmetry(
            tertiary_pipeline,
            self._rotate(prep.candidates, 2),
            prompt_tokens_est=prompt_tokens_est,
        )
        tertiary_pipeline, _ = self._apply_moe_geo_sharding(
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

        secondary_result: ChainResult | None = None
        tertiary_result: ChainResult | None = None

        def run_secondary() -> ChainResult:
            nonlocal secondary_result
            secondary_result = self._run_chain(
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
            tertiary_result = self._run_chain(
                prep.effective_prompt,
                self._rotate(prep.candidates, 2),
                tertiary_pipeline,
                max_tokens=max_tokens,
                request_id=primary.request_id,
                deadline=deadline,
                **decode_controls,
            )
            return tertiary_result

        verification = self.verifier.verify(
            primary,
            run_secondary=run_secondary,
            run_tertiary=(run_tertiary if len(prep.candidates) >= 3 else None),
        )
        verification_feedback = self._apply_verification_feedback(
            primary=primary,
            secondary=secondary_result,
            tertiary=tertiary_result,
            verification=verification,
        )

        response_text = primary.text if verification.winner == "primary" else verification.secondary_text or primary.text

        hydra_reward_rate = max(0.0, float(self.config.hydra_reward_per_1k_tokens))
        for trace in primary.traces:
            self.ledger.earn(trace.peer_id, tokens_served=max_tokens)
            if hydra_reward_rate > 0.0:
                self.hydra.mint_for_inference(
                    peer_id=trace.peer_id,
                    tokens_served=max_tokens,
                    reward_per_1k_tokens=hydra_reward_rate,
                )

        replication = self._replication_dict(prep.decision.served_model, len(prep.health))
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
            "discovered_peers": self._discovered_peer_rows(prep.health),
            # Phase 3: surface which pipeline path was used
            "pipeline_mode": prep.pipeline_mode,
        }

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
        prep = self._prepare_inference(
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
        _required = self._required_replicas(_served)
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
        pytorch_tokenizer_model_id = self._resolve_pipeline_runtime_model_id(
            prep.primary_pipeline,
            prep.decision.served_model,
        )
        pytorch_tokenizer = (
            self._load_generation_tokenizer(pytorch_tokenizer_model_id)
            if pytorch_autoregressive
            else None
        )
        pytorch_eos_token_ids: set[int] = set()
        pytorch_special_token_ids: set[int] = set()
        if pytorch_tokenizer is not None:
            eos_raw = getattr(pytorch_tokenizer, "eos_token_id", None)
            if isinstance(eos_raw, int):
                pytorch_eos_token_ids = {int(eos_raw)}
            elif eos_raw is not None:
                try:
                    pytorch_eos_token_ids = {int(item) for item in list(eos_raw)}
                except TypeError:
                    pytorch_eos_token_ids = set()
            pytorch_eos_token_ids = {item for item in pytorch_eos_token_ids if item >= 0}
            special_raw = getattr(pytorch_tokenizer, "all_special_ids", None)
            if special_raw is not None:
                try:
                    pytorch_special_token_ids = {
                        int(item) for item in list(special_raw) if int(item) >= 0
                    }
                except TypeError:
                    pytorch_special_token_ids = set()
        pytorch_eos_token_id = (min(pytorch_eos_token_ids) if pytorch_eos_token_ids else None)
        pytorch_speculative_enabled = bool(self.config.speculative_enabled and pytorch_autoregressive)
        pytorch_draft_model: PyTorchDraftModel | None = None
        if pytorch_speculative_enabled:
            try:
                pytorch_draft_model = self._load_pytorch_draft_model(
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
        kv_cache_seed = self._get_kv_affinity_activation(session_id, served_model)
        kv_cache_source_peer_id = self._get_kv_affinity_activation_peer(session_id, served_model)
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

            if pytorch_autoregressive:
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
                        self._run_chain(
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
                        verify_result = self._run_chain(
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
                        self._run_chain(
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
                        verify_result = self._run_chain(
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

                    commit_result = self._run_chain(
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
                            step_result = self._run_chain(
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
                                step_result = self._run_chain(
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
                        selection = select_verified_tokens(tokens, draft_tokens)
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
                                        self._run_chain,
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
                    kv_data_plane["cache_updated"] = self._set_kv_affinity_activation(
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

    def infer_chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
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

    def list_models(self) -> dict[str, Any]:
        # Serve directly from the static catalog — no DHT scan.
        # DHT scanning is deferred to the inference path (/v1/chat/completions),
        # where peer discovery is actually needed.  Peer counts here are served
        # from the in-memory DHT cache (populated on each inference request,
        # 120 s TTL) so they become accurate after the first inference and
        # never block the models listing endpoint.
        data = []
        for model in self.model_catalog:
            cached = self._cached_dht_peers(model_id=model.model_id)
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

    def network_status(self) -> dict[str, Any]:
        try:
            health, counts = self._scan_network(model_ids=self._catalog_model_ids())
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

        # ── Phase 2: Auto-scaler evaluation (every 5 minutes) ─────────────────
        if self._auto_scaler is not None:
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
            _scaler_result = self._auto_scaler.maybe_evaluate(
                _scaler_peers, self._request_log
            )
            if _scaler_result is not None:
                self._active_model_roster = list(_scaler_result.active_roster)

        # ── Phase 3: Layer sharding coverage ──────────────────────────────────
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

    def _record_channel_provider_spend(self, channel_id: str, provider_peer_id: str, amount: float) -> None:
        key = str(channel_id).strip()
        provider = str(provider_peer_id).strip()
        delta = max(0.0, float(amount))
        if not key or not provider or delta <= 0.0:
            return
        bucket = self._channel_provider_spend.setdefault(key, {})
        bucket[provider] = round(float(bucket.get(provider, 0.0)) + delta, 6)

    def _set_channel_payee_spend(self, channel_id: str, payee_peer_id: str, total_spent: float) -> None:
        key = str(channel_id).strip()
        payee = str(payee_peer_id).strip()
        target = max(0.0, float(total_spent))
        if not key or not payee:
            return
        bucket = self._channel_provider_spend.setdefault(key, {})
        allocated = sum(float(value) for value in bucket.values())
        if allocated <= 0.0:
            bucket[payee] = round(target, 6)
            return
        if target >= allocated:
            bucket[payee] = round(float(bucket.get(payee, 0.0)) + (target - allocated), 6)
            return
        bucket.clear()
        bucket[payee] = round(target, 6)

    def _bridge_settle_channel_close(self, close_payload: dict[str, Any]) -> dict[str, Any]:
        channel_id = str(close_payload.get("channel_id", "")).strip()
        payer = str(close_payload.get("payer", "")).strip()
        payee = str(close_payload.get("payee", "")).strip()
        payee_amount = max(0.0, float(close_payload.get("payee_amount", 0.0)))

        spent_by_provider = {
            str(peer_id): max(0.0, float(amount))
            for peer_id, amount in dict(self._channel_provider_spend.pop(channel_id, {})).items()
            if str(peer_id).strip() and float(amount) > 0.0
        }
        if not spent_by_provider and payee and payee_amount > 0.0:
            spent_by_provider = {payee: payee_amount}

        settlement = {
            "enabled": True,
            "channel_id": channel_id,
            "payer_pubkey": payer,
            "burn_receipt": None,
            "mint_receipts": [],
            "errors": [],
        }
        if not payer or payee_amount <= 0.0:
            settlement["enabled"] = False
            return settlement

        burn_amount = payee_amount
        try:
            settlement["burn_receipt"] = self.ledger_bridge.burn_for_compute(
                payer_pubkey=payer,
                amount=burn_amount,
            )
        except RuntimeError as exc:
            settlement["errors"].append(f"burn_for_compute_failed:{exc}")
            return settlement

        minted_total = 0.0
        for provider_id, amount in spent_by_provider.items():
            if minted_total >= burn_amount:
                break
            mint_amount = min(max(0.0, float(amount)), burn_amount - minted_total)
            if mint_amount <= 0.0:
                continue
            try:
                receipt = self.ledger_bridge.mint_provider_rewards(
                    payee_pubkey=provider_id,
                    amount=mint_amount,
                )
                settlement["mint_receipts"].append(receipt)
                minted_total += mint_amount
            except RuntimeError as exc:
                settlement["errors"].append(f"mint_provider_rewards_failed:{provider_id}:{exc}")
                break

        return settlement

    def account_balance(self, client_id: str) -> dict[str, Any]:
        hydra_account = self.hydra.account_snapshot(client_id)
        return {
            "client_id": client_id,
            "priority_credits": round(self.ledger.balance(client_id), 6),
            "hydra": hydra_account,
            "hydra_bridge": self.ledger_bridge.account_snapshot(client_id),
        }

    def hydra_status(self) -> dict[str, Any]:
        bridge_summary = self.ledger_bridge.summary()
        is_mock = bool(bridge_summary.get("mock_mode", True))
        return {
            "hydra": self.hydra.summary(),
            "hydra_bridge": bridge_summary,
            # Surfaced at the top level so clients and dashboards can detect mock mode
            # without parsing the nested bridge summary.
            "mock_mode": is_mock,
            "mock_mode_warning": (
                "HYDRA bridge is running in mock mode — all token settlement is "
                "in-memory only. No real on-chain transactions occur."
                if is_mock else None
            ),
        }

    def hydra_account(self, client_id: str) -> dict[str, Any]:
        return {"hydra": self.hydra.account_snapshot(client_id)}

    def metrics_snapshot(self) -> dict[str, float | int]:
        with self._metrics_lock:
            dht_attempts = int(self._dht_lookup_attempts)
            dht_successes = int(self._dht_lookup_successes)
            dht_failures = int(self._dht_lookup_failures)
            kv_store_ops = int(self._kv_store_ops_total)
            kv_retrieve_ops = int(self._kv_retrieve_ops_total)
            inference_reqs = int(self._inference_requests_total)
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

    def hydra_governance_params(self) -> dict[str, Any]:
        return {"hydra_governance": {"params": self.ledger_bridge.governance_params()}}

    def hydra_governance_vote(self, pubkey: str, proposal_id: str, vote: str) -> dict[str, Any]:
        return {
            "hydra_governance_vote": self.ledger_bridge.submit_vote(
                pubkey=pubkey,
                proposal_id=proposal_id,
                vote=vote,
            )
        }

    def hydra_transfer(self, from_client_id: str, to_client_id: str, amount: float) -> dict[str, Any]:
        result = self.hydra.transfer(from_peer_id=from_client_id, to_peer_id=to_client_id, amount=float(amount))
        return {
            "from_client_id": from_client_id,
            "to_client_id": to_client_id,
            "amount": float(amount),
            "result": {
                "from_balance": round(float(result["from_balance"]), 6),
                "to_balance": round(float(result["to_balance"]), 6),
            },
        }

    def hydra_stake(self, client_id: str, amount: float) -> dict[str, Any]:
        result = self.hydra.stake(peer_id=client_id, amount=float(amount))
        return {
            "client_id": client_id,
            "amount": float(amount),
            "result": {
                "balance": round(float(result["balance"]), 6),
                "stake": round(float(result["stake"]), 6),
            },
        }

    def hydra_unstake(self, client_id: str, amount: float) -> dict[str, Any]:
        result = self.hydra.unstake(peer_id=client_id, amount=float(amount))
        return {
            "client_id": client_id,
            "amount": float(amount),
            "result": {
                "balance": round(float(result["balance"]), 6),
                "stake": round(float(result["stake"]), 6),
            },
        }

    def hydra_open_channel(
        self,
        channel_id: str,
        payer: str,
        payee: str,
        deposit: float,
        ttl_seconds: int | None = None,
    ) -> dict[str, Any]:
        self._channel_provider_spend.pop(str(channel_id).strip(), None)
        return {
            "hydra_channel": self.hydra.open_state_channel(
                channel_id=channel_id,
                payer=payer,
                payee=payee,
                deposit=float(deposit),
                ttl_seconds=(int(ttl_seconds) if ttl_seconds is not None else None),
            )
        }

    def hydra_charge_channel(self, channel_id: str, amount: float, provider_peer_id: str | None = None) -> dict[str, Any]:
        payload = self.hydra.charge_state_channel(channel_id=channel_id, amount=float(amount))
        provider = str(provider_peer_id or payload.get("payee") or "").strip()
        if provider:
            self._record_channel_provider_spend(channel_id, provider, float(amount))
        return {
            "hydra_channel": payload
        }

    def hydra_reconcile_channel(self, channel_id: str, total_spent: float, nonce: int) -> dict[str, Any]:
        payload = self.hydra.reconcile_state_channel(
            channel_id=channel_id,
            total_spent=float(total_spent),
            nonce=int(nonce),
        )
        payee = str(payload.get("payee", "")).strip()
        if payee:
            self._set_channel_payee_spend(channel_id, payee, float(payload.get("spent", 0.0)))
        return {
            "hydra_channel": payload
        }

    def hydra_close_channel(self, channel_id: str) -> dict[str, Any]:
        close_payload = self.hydra.close_state_channel(channel_id=channel_id)
        settlement = self._bridge_settle_channel_close(close_payload)
        return {
            "hydra_channel_close": close_payload,
            "hydra_bridge_settlement": settlement,
        }

    def close(self) -> None:
        self.health.close()
        self.ledger.close()
        self.hydra.close()
