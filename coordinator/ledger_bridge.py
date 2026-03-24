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
import threading
from typing import Callable


@dataclass
class BridgeAccount:
    pubkey: str
    balance: float = 0.0
    staked_balance: float = 0.0

    @property
    def total(self) -> float:
        return float(self.balance + self.staked_balance)


class OpenHydraLedgerBridge:
    """Mockable L1 settlement bridge with burn-and-mint equilibrium controls.

    PRODUCTION NOTE — mock_mode=True (default)
    ==========================================
    In mock mode all burn/mint/slash operations are applied to an **in-memory**
    account book.  No real on-chain transactions occur.  This is intentional for
    development and testnet use.

    To wire a real L1 (e.g. an EVM-compatible HYDRA token contract):

    1. Set ``mock_mode=False`` when constructing this class.  All methods that
       currently raise ``"hydra_bridge_live_mode_not_implemented"`` will then
       propagate to callers — replace those raises with real RPC calls.

    2. Implement ``external_stake_resolver(pubkey: str) -> float``:
       A callable that queries on-chain staked balance for a given peer pubkey.
       Example (web3.py)::

           from web3 import Web3
           w3 = Web3(Web3.HTTPProvider(RPC_URL))
           contract = w3.eth.contract(address=HYDRA_ADDR, abi=HYDRA_ABI)
           resolver = lambda pk: contract.functions.stakedBalance(pk).call() / 1e18

    3. Implement ``external_stake_slasher(pubkey: str, amount: float) -> float``:
       Submits a slash transaction and returns the amount actually slashed.
       Must be idempotent (safe to retry on failure).

    4. Replace ``burn_for_compute`` / ``mint_provider_rewards`` with signed
       EVM transactions (e.g. using a coordinator-held private key stored in
       ``openhydra_secrets.py`` / Vault).

    5. Add transaction retry / finality-waiting logic (``receipt.status == 1``).

    The interface (method signatures and return dict shapes) is deliberately
    stable — no callers need to change when you swap in the real implementation.
    """

    def __init__(
        self,
        *,
        mock_mode: bool = True,
        supply_cap: float = 69_000_000.0,
        daily_mint_rate: float = 250_000.0,
        min_slash_penalty: float = 0.10,
        external_stake_resolver: Callable[[str], float] | None = None,
        external_stake_slasher: Callable[[str, float], float] | None = None,
    ):
        self.mock_mode = bool(mock_mode)
        self.supply_cap = max(0.0, float(supply_cap))
        self.daily_mint_rate = max(0.0, float(daily_mint_rate))
        self.min_slash_penalty = max(0.0, float(min_slash_penalty))
        self._accounts: dict[str, BridgeAccount] = {}
        self._lock = threading.Lock()
        self._total_supply = 0.0
        self._total_burned = 0.0
        self._total_minted = 0.0
        self._total_slashed = 0.0
        self._governance_votes: dict[str, dict[str, str]] = {}
        self._vote_history: list[dict[str, str | int]] = []
        self._external_stake_resolver = external_stake_resolver
        self._external_stake_slasher = external_stake_slasher

    @staticmethod
    def _normalize_pubkey(pubkey: str) -> str:
        key = str(pubkey).strip()
        if not key:
            raise RuntimeError("hydra_bridge_invalid_pubkey")
        return key

    @staticmethod
    def _positive_amount(amount: float) -> float:
        value = float(amount)
        if value <= 0.0:
            raise RuntimeError("hydra_bridge_invalid_amount")
        return value

    def _account(self, pubkey: str) -> BridgeAccount:
        key = self._normalize_pubkey(pubkey)
        acct = self._accounts.get(key)
        if acct is None:
            acct = BridgeAccount(pubkey=key)
            self._accounts[key] = acct
        return acct

    def _external_stake(self, pubkey: str) -> float:
        resolver = self._external_stake_resolver
        if resolver is None:
            return 0.0
        try:
            return max(0.0, float(resolver(pubkey)))
        except Exception:
            return 0.0

    def _ensure_supply_within_cap(self, delta: float) -> None:
        projected = self._total_supply + float(delta)
        if projected > (self.supply_cap + 1e-9):
            raise RuntimeError("hydra_bridge_supply_cap_exceeded")

    def seed_account(
        self,
        pubkey: str,
        *,
        balance: float | None = None,
        staked_balance: float | None = None,
    ) -> dict[str, float]:
        """Mock utility for tests/bootstrap balances."""
        with self._lock:
            acct = self._account(pubkey)
            next_balance = acct.balance if balance is None else max(0.0, float(balance))
            next_stake = acct.staked_balance if staked_balance is None else max(0.0, float(staked_balance))
            delta = (next_balance + next_stake) - acct.total
            if delta > 0.0:
                self._ensure_supply_within_cap(delta)
                self._total_minted += delta
            elif delta < 0.0:
                self._total_burned += abs(delta)
            acct.balance = next_balance
            acct.staked_balance = next_stake
            self._total_supply += delta
            return {
                "balance": acct.balance,
                "staked_balance": acct.staked_balance,
                "total_supply": self._total_supply,
            }

    def account_snapshot(self, pubkey: str) -> dict[str, float | str]:
        with self._lock:
            acct = self._account(pubkey)
            return {
                "pubkey": acct.pubkey,
                "balance": round(float(acct.balance), 6),
                "staked_balance": round(float(acct.staked_balance), 6),
                "total": round(float(acct.total), 6),
            }

    def verify_balance(self, pubkey: str) -> float:
        with self._lock:
            acct = self._account(pubkey)
            return float(acct.balance)

    def verify_staked_balance(self, pubkey: str) -> float:
        with self._lock:
            acct = self._account(pubkey)
            local = float(acct.staked_balance)
        return max(local, self._external_stake(pubkey))

    def slash_stake(self, pubkey: str, amount: float) -> dict[str, float | str]:
        delta = self._positive_amount(amount)
        external_slashed = 0.0

        with self._lock:
            acct = self._account(pubkey)
            acct_pubkey = str(acct.pubkey)
            local_slashed = min(float(acct.staked_balance), delta)
            remaining_stake = max(0.0, float(acct.staked_balance) - local_slashed)
            acct.staked_balance = remaining_stake
            self._total_supply -= local_slashed
            total_supply_after_local = float(self._total_supply)
            remaining = max(0.0, delta - local_slashed)
            external_slasher = self._external_stake_slasher

        if remaining > 0.0 and external_slasher is not None:
            try:
                external_slashed = max(0.0, float(external_slasher(pubkey, remaining)))
            except Exception:
                external_slashed = 0.0

        total_slashed = local_slashed + min(remaining, external_slashed)
        if total_slashed > 0.0:
            with self._lock:
                self._total_burned += total_slashed
                self._total_slashed += total_slashed
        return {
            "pubkey": acct_pubkey,
            "slashed": round(float(total_slashed), 6),
            "remaining_stake": round(float(remaining_stake), 6),
            "total_supply": round(float(total_supply_after_local), 6),
        }

    def burn_for_compute(self, payer_pubkey: str, amount: float) -> dict[str, float | str]:
        if not self.mock_mode:
            raise RuntimeError("hydra_bridge_live_mode_not_implemented")

        delta = self._positive_amount(amount)
        with self._lock:
            acct = self._account(payer_pubkey)
            if acct.balance < delta:
                raise RuntimeError("hydra_bridge_insufficient_balance")
            acct.balance -= delta
            self._total_supply -= delta
            self._total_burned += delta
            return {
                "payer_pubkey": acct.pubkey,
                "burned": round(float(delta), 6),
                "balance": round(float(acct.balance), 6),
                "total_supply": round(float(self._total_supply), 6),
            }

    def mint_provider_rewards(self, payee_pubkey: str, amount: float) -> dict[str, float | str]:
        if not self.mock_mode:
            raise RuntimeError("hydra_bridge_live_mode_not_implemented")

        delta = self._positive_amount(amount)
        with self._lock:
            self._ensure_supply_within_cap(delta)
            acct = self._account(payee_pubkey)
            acct.balance += delta
            self._total_supply += delta
            self._total_minted += delta
            return {
                "payee_pubkey": acct.pubkey,
                "minted": round(float(delta), 6),
                "balance": round(float(acct.balance), 6),
                "total_supply": round(float(self._total_supply), 6),
            }

    def summary(self) -> dict[str, float | int | bool]:
        with self._lock:
            return {
                "mock_mode": bool(self.mock_mode),
                "supply_cap": round(float(self.supply_cap), 6),
                "daily_mint_rate": round(float(self.daily_mint_rate), 6),
                "min_slash_penalty": round(float(self.min_slash_penalty), 6),
                "total_supply": round(float(self._total_supply), 6),
                "total_minted": round(float(self._total_minted), 6),
                "total_burned": round(float(self._total_burned), 6),
                "total_slashed": round(float(self._total_slashed), 6),
                "accounts": len(self._accounts),
                "governance_proposals": len(self._governance_votes),
                "governance_votes": len(self._vote_history),
            }

    def governance_params(self) -> dict[str, float | bool]:
        with self._lock:
            return {
                "mock_mode": bool(self.mock_mode),
                "supply_cap": round(float(self.supply_cap), 6),
                "daily_mint_rate": round(float(self.daily_mint_rate), 6),
                "min_slash_penalty": round(float(self.min_slash_penalty), 6),
            }

    def submit_vote(self, pubkey: str, proposal_id: str, vote: str) -> dict[str, str | int | bool]:
        voter = self._normalize_pubkey(pubkey)
        proposal = str(proposal_id).strip()
        if not proposal:
            raise RuntimeError("hydra_bridge_invalid_proposal_id")
        vote_norm = str(vote or "").strip().lower()
        if vote_norm not in {"yes", "no", "abstain"}:
            raise RuntimeError("hydra_bridge_invalid_vote")
        with self._lock:
            proposal_votes = self._governance_votes.setdefault(proposal, {})
            proposal_votes[voter] = vote_norm
            payload = {
                "pubkey": voter,
                "proposal_id": proposal,
                "vote": vote_norm,
                "proposal_votes": len(proposal_votes),
                "accepted": True,
            }
            self._vote_history.append(dict(payload))
            if len(self._vote_history) > 4096:
                self._vote_history = self._vote_history[-4096:]
            return payload
