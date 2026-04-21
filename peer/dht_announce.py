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

from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
from pathlib import Path
import time
from typing import Any
from urllib import request


@dataclass
class Announcement:
    peer_id: str
    model_id: str
    host: str
    port: int
    operator_id: str | None = None
    region: str | None = None
    load_pct: float = 0.0
    daemon_mode: str = "polite"
    bandwidth_mbps: float = 0.0
    seeding_enabled: bool = False
    seed_upload_limit_mbps: float = 0.0
    seed_target_upload_limit_mbps: float = 0.0
    seed_inference_active: bool = False
    runtime_backend: str = "toy_cpu"
    runtime_target: str = "cpu"
    runtime_model_id: str = ""
    quantization_mode: str = "fp32"
    quantization_bits: int = 0
    runtime_gpu_available: bool = False
    runtime_estimated_tokens_per_sec: float = 0.0
    runtime_estimated_memory_mb: int = 0
    privacy_noise_variance: float = 0.0
    privacy_noise_payloads: int = 0
    privacy_noise_observed_variance_ema: float = 0.0
    privacy_noise_last_audit_tag: str = ""
    reputation_score: float = 0.0
    staked_balance: float = 0.0
    expert_tags: tuple[str, ...] = ()
    expert_layer_indices: tuple[int, ...] = ()
    expert_router: bool = False
    peer_public_key: str = ""
    public_key: str = ""
    signature: str = ""
    # Phase 2: free GPU VRAM reported by the peer for auto-scaler capability checks.
    # 0 = unknown / CPU-only.  Coordinators treat 0 as "assume capable" (optimistic).
    available_vram_mb: int = 0
    # Phase 3: layer-range sharding fields.  All three must be non-zero for the
    # coordinator to treat this peer as a shard (rather than a full-model replica).
    # layer_start : first transformer layer handled by this peer (inclusive)
    # layer_end   : one past the last layer (exclusive); peer covers [start, end)
    # total_layers: total transformer depth of the full model (e.g. 32 for LLaMA-8B)
    layer_start: int = 0
    layer_end: int = 0
    total_layers: int = 0
    # Phase 5: P2P model distribution.  seeder_http_port > 0 means this peer is
    # running a ModelSeedServer that accepts Range requests for model file downloads.
    # cached_model_ids lists HuggingFace model IDs (e.g. "Qwen/Qwen3.5-0.8B") whose
    # weight files have been locally verified and are available for peer download.
    seeder_http_port: int = 0
    cached_model_ids: tuple[str, ...] = ()
    # Phase A: local fast-path TCP port for same-LAN raw tensor transfer.
    # 0 = disabled (no fast-path server running).
    local_fast_path_port: int = 0
    # Phase 2A: KV cache availability — number of free cache slots.
    # 0 = unknown or full.  Coordinators penalise 0-slot peers in routing.
    available_kv_slots: int = 0
    # Phase 2A: Server-to-server measured RTT (ms) to downstream peers.
    # JSON-encoded dict[str, float] keyed by downstream peer_id.
    # Empty string = no measurements available.
    next_hop_rtts_json: str = ""
    # Pass 6: KV compaction SLO metrics — lifetime counters.
    compact_tokens_saved_total: int = 0
    compact_latency_total_ms: float = 0.0
    # Petals parity Phase C: NAT traversal and relay.
    # nat_type: "open", "full_cone", "restricted", "symmetric", "unknown"
    nat_type: str = "unknown"
    requires_relay: bool = False
    relay_peer_id: str = ""        # peer_id of the relay handling this peer
    relay_address: str = ""        # "host:port" of the relay (for routing)
    # Cross-ISP: libp2p peer ID for proxy forwarding through Circuit Relay.
    libp2p_peer_id: str = ""
    # Zero-config bootstrap Phase 1: capacity engine payload.
    # capacity_schema_version: bumps when the JSON shape changes (see
    #   peer.capacity.CAPACITY_SCHEMA_VERSION).  0 = no capacity report attached.
    # capacity_json: stringified CapacityReport produced by
    #   peer.capacity.build_capacity_report(...).to_dict().  Coordinators can
    #   parse it lazily — old peers that never populate this remain compatible.
    capacity_schema_version: int = 0
    capacity_json: str = ""


def announce_local(announcement: Announcement, registry_file: str = ".openhydra_registry.json") -> None:
    path = Path(registry_file)
    current: list[dict[str, Any]] = []
    if path.exists():
        current = json.loads(path.read_text())

    remaining = [item for item in current if item.get("peer_id") != announcement.peer_id]
    remaining.append(asdict(announcement))
    path.write_text(json.dumps(remaining, indent=2))


def announce_http(
    announcement: Announcement,
    dht_url: str,
    ttl_seconds: int = 300,
    timeout_s: float = 2.0,
    heartbeat: bool = False,
) -> dict[str, Any]:
    endpoint = "/heartbeat" if heartbeat else "/announce"
    body = {
        **asdict(announcement),
        "updated_unix_ms": int(time.time() * 1000),
        "ttl_seconds": int(ttl_seconds),
    }
    payload = json.dumps(body).encode("utf-8")
    req = request.Request(
        url=f"{dht_url.rstrip('/')}{endpoint}",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _normalize_dht_urls(dht_urls: list[str] | tuple[str, ...] | str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    values = [dht_urls] if isinstance(dht_urls, str) else list(dht_urls)
    for raw in values:
        for token in str(raw).split(","):
            value = token.strip()
            if not value or value in seen:
                continue
            seen.add(value)
            out.append(value)
    return out


def announce_http_many(
    announcement: Announcement,
    dht_urls: list[str] | tuple[str, ...] | str,
    ttl_seconds: int = 300,
    timeout_s: float = 2.0,
    heartbeat: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, Exception]]:
    urls = _normalize_dht_urls(dht_urls)
    if not urls:
        return {}, {}

    successes: dict[str, dict[str, Any]] = {}
    failures: dict[str, Exception] = {}
    worker_count = max(1, min(8, len(urls)))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        future_by_url = {
            pool.submit(
                announce_http,
                announcement,
                dht_url=url,
                ttl_seconds=ttl_seconds,
                timeout_s=timeout_s,
                heartbeat=heartbeat,
            ): url
            for url in urls
        }
        for future in as_completed(future_by_url):
            url = future_by_url[future]
            try:
                successes[url] = dict(future.result())
            except Exception as exc:  # pragma: no cover
                failures[url] = exc
                logging.warning("dht_announce_failed: url=%s error=%s", url, exc)
    return successes, failures
