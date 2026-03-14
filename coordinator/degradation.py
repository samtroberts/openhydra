from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelAvailability:
    model_id: str
    required_peers: int
    hf_model_id: str = ""
    # Extended fields (6.3)
    min_vram_gb: int = 0
    recommended_quantization: str = "fp32"
    context_length: int = 4096
    languages: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    description: str = ""
    # Phase 2 auto-scaler fields
    shard_vram_gb: float = 0.0    # VRAM per shard in GB (0 = not sharded / unknown)
    shards_needed: int = 1        # peers required for a full pipeline
    quality_tier: str = "standard"  # "basic" | "standard" | "advanced" | "frontier"


@dataclass(frozen=True)
class DegradationDecision:
    requested_model: str
    served_model: str
    degraded: bool
    available: bool
    reason: str
    detail: str


class DegradationPolicy:
    """Graceful model fallback based on healthy peer availability."""

    def __init__(self, catalogue: list[ModelAvailability]):
        if not catalogue:
            raise ValueError("catalogue must not be empty")
        self.catalogue = catalogue
        self._index = {item.model_id: idx for idx, item in enumerate(catalogue)}

    def select(
        self,
        requested_model: str,
        available_peer_counts: dict[str, int],
        allow_degradation: bool = True,
    ) -> DegradationDecision:
        requested_idx = self._index.get(requested_model)
        if requested_idx is None:
            return DegradationDecision(
                requested_model=requested_model,
                served_model=requested_model,
                degraded=False,
                available=True,
                reason="unknown_model",
                detail=f"model '{requested_model}' is not in catalogue",
            )

        requested = self.catalogue[requested_idx]
        available_for_requested = int(available_peer_counts.get(requested.model_id, 0))
        if available_for_requested >= requested.required_peers:
            return DegradationDecision(
                requested_model=requested_model,
                served_model=requested_model,
                degraded=False,
                available=True,
                reason="ok",
                detail="requested model has sufficient healthy peers",
            )

        if not allow_degradation:
            return DegradationDecision(
                requested_model=requested_model,
                served_model=requested_model,
                degraded=False,
                available=False,
                reason="insufficient_peers",
                detail=(
                    f"requested model has {available_for_requested}/{requested.required_peers} healthy peers "
                    "and degradation is disabled"
                ),
            )

        # Degrade toward smaller/lighter models appearing later in catalogue order.
        for candidate in self.catalogue[requested_idx + 1 :]:
            available = int(available_peer_counts.get(candidate.model_id, 0))
            if available >= candidate.required_peers:
                return DegradationDecision(
                    requested_model=requested_model,
                    served_model=candidate.model_id,
                    degraded=True,
                    available=True,
                    reason="insufficient_peers",
                    detail=(
                        f"requested model has {available_for_requested}/{requested.required_peers} healthy peers; "
                        f"fallback {candidate.model_id} has {available}/{candidate.required_peers}"
                    ),
                )

        return DegradationDecision(
            requested_model=requested_model,
            served_model=requested_model,
            degraded=False,
            available=False,
            reason="no_viable_fallback",
            detail=(
                f"requested model has {available_for_requested}/{requested.required_peers} healthy peers "
                "and no fallback model met required replicas"
            ),
        )
