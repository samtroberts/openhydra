"""Support role assignment for peers that cannot serve inference.

Prevents the "gravitational collapse" failure mode where weak nodes earn
nothing and leave the network, shrinking the swarm to a handful of elite nodes.

Every peer gets a role.  Roles are ordered by earnings rate, and we assign
the most valuable role each peer qualifies for.

Earnings multipliers (relative to inference base rate of 1.0):

    inference            1.0   (1000 tokens = 1 credit)
    embedding_server     0.3
    verification_auditor 0.5
    kv_compaction        0.2
    activation_relay     0.1
    model_cache_seed     0.2

Design spec: plans/auto-scaling-policy.md § 8
"""
from __future__ import annotations

from dataclasses import dataclass


# ── Constants ─────────────────────────────────────────────────────────────────

#: Earnings multiplier per role, relative to inference = 1.0.
EARNINGS_MULTIPLIER: dict[str, float] = {
    "inference":            1.0,
    "embedding_server":     0.3,
    "verification_auditor": 0.5,
    "kv_compaction":        0.2,
    "model_cache_seed":     0.2,
    "activation_relay":     0.1,
}

_MIN_CPU_SCORE: float = 100.0   # normalised benchmark score for auditor role
_MIN_DISK_GB: float   = 10.0    # free disk for model-cache-seed role
_MIN_GPU_MB_EMBED: int = 512    # minimum VRAM for embedding-server role (MiniLM = ~80 MB)


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RoleAssignment:
    """The role assigned to one peer by the :func:`assign_role` function."""

    peer_id: str
    role: str                  # e.g. "inference:openhydra-qwen3.5-0.8b" or "embedding_server"
    model_id: str | None       # populated for inference roles; None for support roles
    earnings_multiplier: float


# ── Assignment logic ──────────────────────────────────────────────────────────

def assign_role(
    peer_id: str,
    available_vram_mb: int,
    cpu_score: float,
    disk_free_gb: float,
    model_roster: list[tuple[str, int]],
) -> RoleAssignment:
    """Return the most valuable role this peer can fill.

    Evaluation order (most valuable first):

    1. Inference for the highest-VRAM model the peer can run.
    2. Embedding server (requires ≥ 512 MB GPU or unknown VRAM).
    3. Verification auditor (requires CPU score ≥ threshold).
    4. Model-cache seed (requires ≥ 10 GB free disk).
    5. Activation relay (always available — minimum role).

    Args:
        peer_id:           Unique peer identifier.
        available_vram_mb: Free GPU VRAM in MB.  0 = CPU-only / unknown.
        cpu_score:         Normalised CPU benchmark score (0 = unknown).
        disk_free_gb:      Free disk space in GB (0 = unknown).
        model_roster:      List of (model_id, shard_vram_mb) pairs for models
                           currently on the active roster.  Caller should
                           pass the roster sorted largest-first; if not, this
                           function handles it internally.

    Returns:
        :class:`RoleAssignment` for the highest-earning role the peer qualifies for.
    """
    # ── Inference: try models largest-VRAM first ──────────────────────────────
    for model_id, shard_vram_mb in sorted(model_roster, key=lambda x: x[1], reverse=True):
        if available_vram_mb <= 0:
            # Unknown VRAM → optimistically allow; coordinator validates on use.
            return RoleAssignment(
                peer_id=peer_id,
                role=f"inference:{model_id}",
                model_id=model_id,
                earnings_multiplier=EARNINGS_MULTIPLIER["inference"],
            )
        if available_vram_mb >= shard_vram_mb:
            return RoleAssignment(
                peer_id=peer_id,
                role=f"inference:{model_id}",
                model_id=model_id,
                earnings_multiplier=EARNINGS_MULTIPLIER["inference"],
            )

    # ── Support roles ─────────────────────────────────────────────────────────
    # Embedding server: small GPU (≥512 MB) or unknown
    if available_vram_mb <= 0 or available_vram_mb >= _MIN_GPU_MB_EMBED:
        return RoleAssignment(
            peer_id=peer_id,
            role="embedding_server",
            model_id=None,
            earnings_multiplier=EARNINGS_MULTIPLIER["embedding_server"],
        )

    # Verification auditor: sufficient CPU
    if cpu_score >= _MIN_CPU_SCORE:
        return RoleAssignment(
            peer_id=peer_id,
            role="verification_auditor",
            model_id=None,
            earnings_multiplier=EARNINGS_MULTIPLIER["verification_auditor"],
        )

    # Model-cache seed: ample disk space
    if disk_free_gb >= _MIN_DISK_GB:
        return RoleAssignment(
            peer_id=peer_id,
            role="model_cache_seed",
            model_id=None,
            earnings_multiplier=EARNINGS_MULTIPLIER["model_cache_seed"],
        )

    # Activation relay: absolute fallback — every peer qualifies
    return RoleAssignment(
        peer_id=peer_id,
        role="activation_relay",
        model_id=None,
        earnings_multiplier=EARNINGS_MULTIPLIER["activation_relay"],
    )
