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
