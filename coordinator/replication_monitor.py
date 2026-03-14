from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ReplicationStatus:
    model_id: str
    healthy_peers: int
    required_replicas: int
    under_replicated: bool
    deficit: int


class ReplicationMonitor:
    def __init__(self, required_replicas: int = 3):
        self.required_replicas = max(1, required_replicas)

    def evaluate(self, model_id: str, healthy_peers: int, required_replicas: int | None = None) -> ReplicationStatus:
        required = max(1, required_replicas if required_replicas is not None else self.required_replicas)
        deficit = max(0, required - healthy_peers)
        return ReplicationStatus(
            model_id=model_id,
            healthy_peers=healthy_peers,
            required_replicas=required,
            under_replicated=deficit > 0,
            deficit=deficit,
        )

    @staticmethod
    def to_dict(status: ReplicationStatus) -> dict:
        return asdict(status)
