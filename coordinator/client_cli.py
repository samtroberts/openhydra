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
import json

from coordinator.engine import CoordinatorEngine, EngineConfig


def main() -> None:
    def _parse_expert_tags(raw: str) -> list[str]:
        values = [item.strip().lower() for item in str(raw).split(",")]
        out: list[str] = []
        seen: set[str] = set()
        for value in values:
            if not value or value in seen:
                continue
            seen.add(value)
            out.append(value)
        return out

    def _parse_expert_layers(raw: str) -> list[int]:
        values = [item.strip() for item in str(raw).split(",")]
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

    def _parse_dht_urls(raw_values: list[str] | None) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in list(raw_values or []):
            for token in str(raw).split(","):
                value = token.strip()
                if not value or value in seen:
                    continue
                seen.add(value)
                out.append(value)
        return out

    parser = argparse.ArgumentParser(description="OpenHydra Tier 1/2 coordinator CLI")
    parser.add_argument("--peers", default=None, help="Path to JSON peer config")
    parser.add_argument(
        "--dht-url",
        action="append",
        default=[],
        help="Optional DHT bootstrap URL(s); repeat flag or use comma-separated values",
    )
    parser.add_argument("--dht-lookup-limit", type=int, default=0)
    parser.add_argument("--dht-lookup-sloppy-factor", type=int, default=3)
    parser.add_argument("--dht-lookup-dsht-replicas", type=int, default=2)
    parser.add_argument("--dht-preferred-region", default=None)
    parser.add_argument("--model", default=None, help="Requested model id (defaults to engine default)")
    parser.add_argument("--allow-degradation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expert-tags", default="", help="Comma-separated expert tags for MoE geo routing")
    parser.add_argument("--expert-layers", default="", help="Comma-separated expert layer indices for MoE geo routing")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--pipeline-width", type=int, default=3)
    parser.add_argument("--timeout-ms", type=int, default=500)
    parser.add_argument("--max-latency-ms", type=float, default=5000.0)
    parser.add_argument("--audit-rate", type=float, default=0.10)
    parser.add_argument("--redundant-exec-rate", type=float, default=0.25)
    parser.add_argument("--auditor-rate", type=float, default=0.0)
    parser.add_argument("--verification-alert-min-events", type=int, default=10)
    parser.add_argument("--verification-alert-min-success-rate", type=float, default=0.80)
    parser.add_argument("--verification-qos-min-events", type=int, default=10)
    parser.add_argument("--verification-qos-min-success-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tier", type=int, default=2)
    parser.add_argument("--max-failovers-per-stage", type=int, default=1)
    parser.add_argument("--ledger-path", default=".openhydra/credits.db")
    parser.add_argument("--barter-decay-per-day", type=float, default=0.05)
    parser.add_argument("--hydra-token-ledger-path", default=".openhydra/hydra_tokens.db")
    parser.add_argument("--hydra-reward-per-1k-tokens", type=float, default=1.0)
    parser.add_argument("--hydra-slash-per-failed-verification", type=float, default=0.0)
    parser.add_argument("--hydra-channel-default-ttl-seconds", type=int, default=900)
    parser.add_argument("--hydra-channel-max-open-per-payer", type=int, default=8)
    parser.add_argument("--hydra-channel-min-deposit", type=float, default=0.01)
    parser.add_argument("--hydra-supply-cap", type=float, default=69_000_000.0)
    parser.add_argument("--hydra-ledger-bridge-mock-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hydra-stake-priority-boost", type=float, default=12.0)
    parser.add_argument("--hydra-no-stake-penalty-events", type=int, default=8)
    parser.add_argument("--hydra-governance-daily-mint-rate", type=float, default=250_000.0)
    parser.add_argument("--hydra-governance-min-slash-penalty", type=float, default=0.1)
    parser.add_argument("--health-store-path", default=".openhydra/health.json")
    parser.add_argument("--required-replicas", type=int, default=3)
    parser.add_argument("--allow-dynamic-model-ids", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-catalog-path", default=None, help="JSON model catalogue path")
    parser.add_argument("--operator-cap-fraction", type=float, default=(1.0 / 3.0))
    parser.add_argument("--enforce-pipeline-diversity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diversity-window", type=int, default=3)
    parser.add_argument("--diversity-max-per-window", type=int, default=1)
    parser.add_argument("--prefill-token-threshold", type=int, default=256)
    parser.add_argument("--prefill-min-bandwidth-mbps", type=float, default=500.0)
    parser.add_argument("--decode-max-bandwidth-mbps", type=float, default=50.0)
    parser.add_argument("--grounding-cache-path", default=".openhydra/grounding_cache.json")
    parser.add_argument("--grounding-cache-ttl-seconds", type=int, default=900)
    parser.add_argument("--grounding-timeout-s", type=float, default=3.0)
    parser.add_argument("--grounding-use-network", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grounding-fallback-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--speculative-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--speculative-draft-tokens", type=int, default=4)
    parser.add_argument("--speculative-seed", type=int, default=13)
    parser.add_argument("--speculative-adaptive-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--speculative-min-draft-tokens", type=int, default=2)
    parser.add_argument("--speculative-max-draft-tokens", type=int, default=8)
    parser.add_argument("--speculative-acceptance-low-watermark", type=float, default=0.55)
    parser.add_argument("--speculative-acceptance-high-watermark", type=float, default=0.80)
    parser.add_argument("--pipeline-parallel-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pipeline-parallel-workers", type=int, default=1)
    parser.add_argument("--tensor-autoencoder-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tensor-autoencoder-latent-dim", type=int, default=1024)
    parser.add_argument("--advanced-encryption-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--advanced-encryption-seed", default="openhydra-tier3-dev-seed")
    parser.add_argument(
        "--advanced-encryption-level",
        choices=["standard", "enhanced", "maximum"],
        default="standard",
    )
    parser.add_argument("--kv-peer-cache-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--moe-geo-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--moe-geo-min-tag-matches", type=int, default=1)
    parser.add_argument("--moe-geo-min-layer-matches", type=int, default=1)
    parser.add_argument("--moe-geo-prompt-hints-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pytorch-generation-model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--pytorch-speculative-draft-model-id", default="sshleifer/tiny-gpt2")
    parser.add_argument("--tls-enable", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tls-root-cert-path", default=None)
    parser.add_argument("--tls-client-cert-path", default=None)
    parser.add_argument("--tls-client-key-path", default=None)
    parser.add_argument("--tls-server-name-override", default=None)
    parser.add_argument("--client-id", default="anonymous")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--priority", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--grounding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--json", action="store_true", help="Emit structured JSON output")
    args = parser.parse_args()

    dht_urls = _parse_dht_urls(args.dht_url)
    if not args.peers and not dht_urls:
        parser.error("at least one peer source is required: --peers or --dht-url")

    engine = CoordinatorEngine(
        EngineConfig(
            peers_config_path=args.peers,
            dht_urls=dht_urls,
            dht_url=(dht_urls[0] if dht_urls else None),
            dht_lookup_limit=max(0, int(args.dht_lookup_limit)),
            dht_lookup_sloppy_factor=max(0, int(args.dht_lookup_sloppy_factor)),
            dht_lookup_dsht_replicas=max(0, int(args.dht_lookup_dsht_replicas)),
            dht_preferred_region=(str(args.dht_preferred_region) if args.dht_preferred_region else None),
            tls_enabled=args.tls_enable,
            tls_root_cert_path=args.tls_root_cert_path,
            tls_client_cert_path=args.tls_client_cert_path,
            tls_client_key_path=args.tls_client_key_path,
            tls_server_name_override=args.tls_server_name_override,
            timeout_ms=args.timeout_ms,
            max_latency_ms=args.max_latency_ms,
            pipeline_width=args.pipeline_width,
            tier=args.tier,
            audit_rate=args.audit_rate,
            redundant_exec_rate=args.redundant_exec_rate,
            auditor_rate=max(0.0, min(1.0, args.auditor_rate)),
            verification_alert_min_events=max(1, args.verification_alert_min_events),
            verification_alert_min_success_rate=max(0.0, min(1.0, args.verification_alert_min_success_rate)),
            verification_qos_min_events=max(1, args.verification_qos_min_events),
            verification_qos_min_success_rate=max(0.0, min(1.0, args.verification_qos_min_success_rate)),
            seed=args.seed,
            max_failovers_per_stage=max(0, args.max_failovers_per_stage),
            ledger_path=args.ledger_path,
            barter_decay_per_day=max(0.0, args.barter_decay_per_day),
            hydra_token_ledger_path=args.hydra_token_ledger_path,
            hydra_reward_per_1k_tokens=max(0.0, args.hydra_reward_per_1k_tokens),
            hydra_slash_per_failed_verification=max(0.0, args.hydra_slash_per_failed_verification),
            hydra_channel_default_ttl_seconds=max(1, args.hydra_channel_default_ttl_seconds),
            hydra_channel_max_open_per_payer=max(1, args.hydra_channel_max_open_per_payer),
            hydra_channel_min_deposit=max(0.0, args.hydra_channel_min_deposit),
            hydra_supply_cap=max(0.0, float(args.hydra_supply_cap)),
            hydra_ledger_bridge_mock_mode=bool(args.hydra_ledger_bridge_mock_mode),
            hydra_stake_priority_boost=max(0.0, float(args.hydra_stake_priority_boost)),
            hydra_no_stake_penalty_events=max(1, int(args.hydra_no_stake_penalty_events)),
            hydra_governance_daily_mint_rate=max(0.0, float(args.hydra_governance_daily_mint_rate)),
            hydra_governance_min_slash_penalty=max(0.0, float(args.hydra_governance_min_slash_penalty)),
            health_store_path=args.health_store_path,
            required_replicas=max(1, args.required_replicas),
            allow_dynamic_model_ids=bool(args.allow_dynamic_model_ids),
            model_catalog_path=args.model_catalog_path,
            operator_cap_fraction=max(0.01, min(1.0, args.operator_cap_fraction)),
            enforce_pipeline_diversity=args.enforce_pipeline_diversity,
            diversity_window=max(2, args.diversity_window),
            diversity_max_per_window=max(1, args.diversity_max_per_window),
            prefill_token_threshold=max(1, args.prefill_token_threshold),
            prefill_min_bandwidth_mbps=max(1.0, args.prefill_min_bandwidth_mbps),
            decode_max_bandwidth_mbps=max(0.0, args.decode_max_bandwidth_mbps),
            grounding_cache_path=args.grounding_cache_path,
            grounding_cache_ttl_seconds=max(1, args.grounding_cache_ttl_seconds),
            grounding_timeout_s=max(0.1, args.grounding_timeout_s),
            grounding_use_network=args.grounding_use_network,
            grounding_fallback_enabled=args.grounding_fallback_enabled,
            speculative_enabled=args.speculative_enabled,
            speculative_draft_tokens=max(1, args.speculative_draft_tokens),
            speculative_seed=int(args.speculative_seed),
            speculative_adaptive_enabled=args.speculative_adaptive_enabled,
            speculative_min_draft_tokens=max(1, args.speculative_min_draft_tokens),
            speculative_max_draft_tokens=max(1, args.speculative_max_draft_tokens),
            speculative_acceptance_low_watermark=max(0.0, min(1.0, args.speculative_acceptance_low_watermark)),
            speculative_acceptance_high_watermark=max(0.0, min(1.0, args.speculative_acceptance_high_watermark)),
            pipeline_parallel_enabled=args.pipeline_parallel_enabled,
            pipeline_parallel_workers=max(1, args.pipeline_parallel_workers),
            tensor_autoencoder_enabled=args.tensor_autoencoder_enabled,
            tensor_autoencoder_latent_dim=max(1, args.tensor_autoencoder_latent_dim),
            advanced_encryption_enabled=args.advanced_encryption_enabled,
            advanced_encryption_seed=str(args.advanced_encryption_seed),
            advanced_encryption_level=str(args.advanced_encryption_level),
            kv_peer_cache_enabled=args.kv_peer_cache_enabled,
            moe_geo_enabled=args.moe_geo_enabled,
            moe_geo_min_tag_matches=max(1, args.moe_geo_min_tag_matches),
            moe_geo_min_layer_matches=max(1, args.moe_geo_min_layer_matches),
            moe_geo_prompt_hints_enabled=args.moe_geo_prompt_hints_enabled,
            pytorch_generation_model_id=str(args.pytorch_generation_model_id),
            pytorch_speculative_draft_model_id=str(args.pytorch_speculative_draft_model_id),
        )
    )

    try:
        payload = engine.infer(
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            pipeline_width=args.pipeline_width,
            grounding=args.grounding,
            priority=args.priority,
            client_id=args.client_id,
            session_id=args.session_id,
            model_id=args.model,
            allow_degradation=args.allow_degradation,
            expert_tags=_parse_expert_tags(args.expert_tags),
            expert_layer_indices=_parse_expert_layers(args.expert_layers),
        )

        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"Request ID: {payload['request_id']}")
            pipeline_ids = [item["peer_id"] for item in payload["pipeline"]]
            print(f"Pipeline: {', '.join(pipeline_ids)}")
            print(f"Latency: {payload['latency_ms']} ms")
            print(
                "Audited: "
                f"{payload['verification']['audited']} | "
                f"Match: {payload['verification']['match']} | "
                f"Mode: {payload['verification']['mode']} | "
                f"Rate: {payload['verification']['sample_rate']}"
            )
            feedback = payload.get("verification_feedback", {})
            if feedback:
                print(
                    "Verification Feedback: "
                    f"rewarded={feedback.get('rewarded_peers', [])} "
                    f"penalized={feedback.get('penalized_peers', [])}"
                )
            print("")
            print(payload["response"])
    finally:
        engine.close()


if __name__ == "__main__":
    main()
