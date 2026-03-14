from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torrent.genesis import GenesisResult, bootstrap_weights
from torrent.seeder import ArbitrationConfig, BandwidthArbitrator


@dataclass(frozen=True)
class SessionBootstrapConfig:
    model_id: str
    cache_dir: str = ".cache/openhydra"
    local_path: str | None = None
    source_url: str | None = None
    expected_sha256: str | None = None
    force_refresh: bool = False
    piece_bytes: int = 1 * 1024 * 1024


class TorrentSessionManager:
    """Tracks genesis artifact and adaptive seeding upload policy."""

    def __init__(
        self,
        *,
        bootstrap: SessionBootstrapConfig,
        arbitration: ArbitrationConfig | None = None,
    ):
        self.bootstrap_config = bootstrap
        self.arbitrator = BandwidthArbitrator(arbitration)
        self.genesis_result: GenesisResult | None = None

    def bootstrap(self) -> GenesisResult:
        self.genesis_result = bootstrap_weights(
            model_id=self.bootstrap_config.model_id,
            cache_dir=self.bootstrap_config.cache_dir,
            local_path=self.bootstrap_config.local_path,
            source_url=self.bootstrap_config.source_url,
            expected_sha256=self.bootstrap_config.expected_sha256,
            force_refresh=self.bootstrap_config.force_refresh,
            piece_bytes=self.bootstrap_config.piece_bytes,
        )
        return self.genesis_result

    def update(self, *, inference_active: bool, inference_observed_mbps: float | None = None) -> dict[str, Any]:
        self.arbitrator.update(
            inference_active=inference_active,
            inference_observed_mbps=inference_observed_mbps,
        )
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        arbitration = self.arbitrator.snapshot()
        return {
            "seeding_enabled": True,
            "model_id": self.bootstrap_config.model_id,
            "genesis": (
                {
                    "artifact_path": self.genesis_result.artifact_path,
                    "artifact_sha256": self.genesis_result.artifact_sha256,
                    "artifact_bytes": self.genesis_result.artifact_bytes,
                    "manifest_path": self.genesis_result.manifest_path,
                    "torrent_meta_path": self.genesis_result.torrent_meta_path,
                }
                if self.genesis_result
                else None
            ),
            "arbitration": arbitration,
        }
