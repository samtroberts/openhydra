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

from dataclasses import dataclass


@dataclass
class StateChannel:
    """Simple unilateral off-chain micropayment channel scaffold."""

    payer: str
    payee: str
    deposited: float
    spent: float = 0.0
    nonce: int = 0
    channel_id: str = ""
    closed: bool = False
    created_at: float = 0.0
    expires_at: float = 0.0

    def __post_init__(self) -> None:
        self.deposited = max(0.0, float(self.deposited))
        self.spent = max(0.0, min(float(self.spent), self.deposited))
        self.nonce = max(0, int(self.nonce))
        self.channel_id = str(self.channel_id or "")
        self.closed = bool(self.closed)
        self.created_at = max(0.0, float(self.created_at))
        self.expires_at = max(0.0, float(self.expires_at))

    @property
    def remaining(self) -> float:
        return max(0.0, self.deposited - self.spent)

    def is_expired(self, now_ts: float) -> bool:
        if self.closed:
            return False
        return self.expires_at > 0.0 and float(now_ts) >= self.expires_at

    def charge(self, amount: float) -> bool:
        if self.closed:
            return False
        amount = float(amount)
        if amount <= 0.0:
            return False
        if self.spent + amount > self.deposited:
            return False
        self.spent += amount
        self.nonce += 1
        return True

    def reconcile(self, total_spent: float, nonce: int) -> bool:
        """Apply monotonic off-chain state update."""
        if self.closed:
            return False
        total_spent = float(total_spent)
        nonce = int(nonce)
        if nonce <= self.nonce:
            return False
        if total_spent < self.spent or total_spent > self.deposited:
            return False
        self.spent = total_spent
        self.nonce = nonce
        return True

    def close(self) -> tuple[float, float]:
        self.closed = True
        payee_amount = self.spent
        payer_refund = self.remaining
        return payee_amount, payer_refund

    def to_dict(self) -> dict[str, object]:
        return {
            "channel_id": self.channel_id,
            "payer": self.payer,
            "payee": self.payee,
            "deposited": self.deposited,
            "spent": self.spent,
            "nonce": self.nonce,
            "closed": self.closed,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "StateChannel":
        return cls(
            payer=str(payload.get("payer", "")),
            payee=str(payload.get("payee", "")),
            deposited=float(payload.get("deposited", 0.0)),
            spent=float(payload.get("spent", 0.0)),
            nonce=int(payload.get("nonce", 0)),
            channel_id=str(payload.get("channel_id", "")),
            closed=bool(payload.get("closed", False)),
            created_at=float(payload.get("created_at", 0.0)),
            expires_at=float(payload.get("expires_at", 0.0)),
        )
