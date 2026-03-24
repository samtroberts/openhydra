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
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any

from economy.state_channel import StateChannel


@dataclass
class TokenAccount:
    peer_id: str
    balance: float = 0.0
    stake: float = 0.0
    rewards_earned: float = 0.0
    slashed_total: float = 0.0

    @property
    def total(self) -> float:
        return self.balance + self.stake

    def to_dict(self) -> dict[str, Any]:
        return {
            "peer_id": self.peer_id,
            "balance": float(self.balance),
            "stake": float(self.stake),
            "rewards_earned": float(self.rewards_earned),
            "slashed_total": float(self.slashed_total),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TokenAccount":
        return cls(
            peer_id=str(payload.get("peer_id", "")),
            balance=float(payload.get("balance", 0.0)),
            stake=float(payload.get("stake", 0.0)),
            rewards_earned=float(payload.get("rewards_earned", 0.0)),
            slashed_total=float(payload.get("slashed_total", 0.0)),
        )


class HydraTokenEconomy:
    """Tier 3 HYDRA token + state-channel scaffold."""

    def __init__(
        self,
        *,
        channel_default_ttl_seconds: int = 900,
        channel_max_open_per_payer: int = 8,
        channel_min_deposit: float = 0.01,
        now_fn=None,
    ):
        self.accounts: dict[str, TokenAccount] = {}
        self.channels: dict[str, StateChannel] = {}
        self.total_minted: float = 0.0
        self.total_burned: float = 0.0
        self.total_slashed: float = 0.0
        self.total_auto_closed: int = 0
        self.channel_default_ttl_seconds = max(1, int(channel_default_ttl_seconds))
        self.channel_max_open_per_payer = max(1, int(channel_max_open_per_payer))
        self.channel_min_deposit = max(0.0, float(channel_min_deposit))
        self._now_fn = now_fn or time.time

    def _now(self) -> float:
        return float(self._now_fn())

    def _close_channel(self, channel: StateChannel, *, reason: str) -> dict[str, Any]:
        payee_amount, payer_refund = channel.close()
        payer = self.account(channel.payer)
        payee = self.account(channel.payee)
        payer.balance += payer_refund
        payee.balance += payee_amount
        return {
            "channel_id": channel.channel_id,
            "payer": channel.payer,
            "payee": channel.payee,
            "payee_amount": payee_amount,
            "payer_refund": payer_refund,
            "close_reason": str(reason),
            "closed": True,
        }

    def _auto_expire_channels(self) -> int:
        now_ts = self._now()
        expired = 0
        for channel in self.channels.values():
            if channel.is_expired(now_ts):
                self._close_channel(channel, reason="expired_auto")
                expired += 1
        if expired:
            self.total_auto_closed += expired
        return expired

    def _open_channel_count_for_payer(self, payer: str) -> int:
        return sum(
            1
            for item in self.channels.values()
            if (not item.closed and item.payer == payer)
        )

    def account(self, peer_id: str) -> TokenAccount:
        key = str(peer_id).strip()
        if not key:
            raise RuntimeError("hydra_invalid_account")
        if key not in self.accounts:
            self.accounts[key] = TokenAccount(peer_id=key)
        return self.accounts[key]

    @staticmethod
    def _as_positive_amount(amount: float) -> float:
        value = float(amount)
        if value <= 0.0:
            raise RuntimeError("hydra_invalid_amount")
        return value

    def mint(self, peer_id: str, amount: float, *, track_rewards: bool = False) -> float:
        delta = self._as_positive_amount(amount)
        acct = self.account(peer_id)
        acct.balance += delta
        if track_rewards:
            acct.rewards_earned += delta
        self.total_minted += delta
        return acct.balance

    def burn(self, peer_id: str, amount: float) -> float:
        delta = self._as_positive_amount(amount)
        acct = self.account(peer_id)
        if acct.balance < delta:
            raise RuntimeError("hydra_insufficient_balance")
        acct.balance -= delta
        self.total_burned += delta
        return acct.balance

    def transfer(self, from_peer_id: str, to_peer_id: str, amount: float) -> dict[str, float]:
        delta = self._as_positive_amount(amount)
        sender = self.account(from_peer_id)
        receiver = self.account(to_peer_id)
        if sender.peer_id == receiver.peer_id:
            raise RuntimeError("hydra_same_account_transfer")
        if sender.balance < delta:
            raise RuntimeError("hydra_insufficient_balance")
        sender.balance -= delta
        receiver.balance += delta
        return {
            "from_balance": sender.balance,
            "to_balance": receiver.balance,
        }

    def stake(self, peer_id: str, amount: float) -> dict[str, float]:
        delta = self._as_positive_amount(amount)
        acct = self.account(peer_id)
        if acct.balance < delta:
            raise RuntimeError("hydra_insufficient_balance")
        acct.balance -= delta
        acct.stake += delta
        return {"balance": acct.balance, "stake": acct.stake}

    def unstake(self, peer_id: str, amount: float) -> dict[str, float]:
        delta = self._as_positive_amount(amount)
        acct = self.account(peer_id)
        if acct.stake < delta:
            raise RuntimeError("hydra_insufficient_stake")
        acct.stake -= delta
        acct.balance += delta
        return {"balance": acct.balance, "stake": acct.stake}

    def slash(self, peer_id: str, amount: float) -> dict[str, float]:
        delta = self._as_positive_amount(amount)
        acct = self.account(peer_id)
        from_stake = min(acct.stake, delta)
        acct.stake -= from_stake
        remaining = delta - from_stake
        from_balance = min(acct.balance, remaining)
        acct.balance -= from_balance
        slashed = from_stake + from_balance
        acct.slashed_total += slashed
        self.total_slashed += slashed
        self.total_burned += slashed
        return {
            "slashed": slashed,
            "balance": acct.balance,
            "stake": acct.stake,
        }

    def mint_for_inference(self, peer_id: str, tokens_served: int, reward_per_1k_tokens: float = 1.0) -> float:
        served = max(0, int(tokens_served))
        if served <= 0:
            return self.account(peer_id).balance
        reward_rate = max(0.0, float(reward_per_1k_tokens))
        if reward_rate <= 0.0:
            return self.account(peer_id).balance
        reward = (served / 1000.0) * reward_rate
        return self.mint(peer_id, reward, track_rewards=True)

    def open_state_channel(
        self,
        channel_id: str,
        payer: str,
        payee: str,
        deposit: float,
        ttl_seconds: int | None = None,
    ) -> StateChannel:
        self._auto_expire_channels()
        key = str(channel_id).strip()
        if not key:
            raise RuntimeError("hydra_invalid_channel_id")
        if key in self.channels:
            raise RuntimeError("hydra_channel_exists")
        payer_acct = self.account(payer)
        payee_acct = self.account(payee)
        if payer_acct.peer_id == payee_acct.peer_id:
            raise RuntimeError("hydra_invalid_channel_parties")
        value = self._as_positive_amount(deposit)
        if value < self.channel_min_deposit:
            raise RuntimeError("hydra_channel_deposit_too_small")
        if self._open_channel_count_for_payer(payer_acct.peer_id) >= self.channel_max_open_per_payer:
            raise RuntimeError("hydra_channel_limit_exceeded")
        if payer_acct.balance < value:
            raise RuntimeError("hydra_insufficient_balance")
        payer_acct.balance -= value
        ttl = self.channel_default_ttl_seconds if ttl_seconds is None else max(1, int(ttl_seconds))
        now_ts = self._now()
        channel = StateChannel(
            payer=payer_acct.peer_id,
            payee=payee_acct.peer_id,
            deposited=value,
            channel_id=key,
            created_at=now_ts,
            expires_at=(now_ts + float(ttl)),
        )
        self.channels[key] = channel
        return channel

    def channel(self, channel_id: str) -> StateChannel:
        self._auto_expire_channels()
        key = str(channel_id).strip()
        channel = self.channels.get(key)
        if channel is None:
            raise RuntimeError("hydra_unknown_channel")
        return channel

    def charge_state_channel(self, channel_id: str, amount: float) -> StateChannel:
        channel = self.channel(channel_id)
        if channel.closed:
            raise RuntimeError("hydra_channel_closed")
        if not channel.charge(amount):
            raise RuntimeError("hydra_channel_insufficient_deposit")
        return channel

    def reconcile_state_channel(self, channel_id: str, total_spent: float, nonce: int) -> StateChannel:
        channel = self.channel(channel_id)
        if channel.closed:
            raise RuntimeError("hydra_channel_closed")
        if not channel.reconcile(total_spent=total_spent, nonce=nonce):
            raise RuntimeError("hydra_invalid_channel_update")
        return channel

    def close_state_channel(self, channel_id: str) -> dict[str, Any]:
        channel = self.channel(channel_id)
        if channel.closed:
            raise RuntimeError("hydra_channel_closed")
        return self._close_channel(channel, reason="manual")

    def account_snapshot(self, peer_id: str) -> dict[str, Any]:
        self._auto_expire_channels()
        acct = self.account(peer_id)
        open_channels = [
            item.channel_id
            for item in self.channels.values()
            if (not item.closed and (item.payer == acct.peer_id or item.payee == acct.peer_id))
        ]
        return {
            "peer_id": acct.peer_id,
            "balance": round(acct.balance, 6),
            "stake": round(acct.stake, 6),
            "total": round(acct.total, 6),
            "rewards_earned": round(acct.rewards_earned, 6),
            "slashed_total": round(acct.slashed_total, 6),
            "open_channels": sorted(open_channels),
        }

    def summary(self) -> dict[str, Any]:
        self._auto_expire_channels()
        active_channels = [item for item in self.channels.values() if not item.closed]
        locked = sum(item.deposited for item in active_channels)
        circulating = sum(item.total for item in self.accounts.values())
        supply = circulating + locked
        return {
            "accounts": len(self.accounts),
            "channels_total": len(self.channels),
            "channels_open": len(active_channels),
            "locked_in_channels": round(locked, 6),
            "circulating_supply": round(circulating, 6),
            "total_supply": round(supply, 6),
            "total_minted": round(self.total_minted, 6),
            "total_burned": round(self.total_burned, 6),
            "total_slashed": round(self.total_slashed, 6),
            "auto_closed_channels": int(self.total_auto_closed),
            "channel_policy": {
                "default_ttl_seconds": int(self.channel_default_ttl_seconds),
                "max_open_per_payer": int(self.channel_max_open_per_payer),
                "min_deposit": round(float(self.channel_min_deposit), 6),
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "accounts": {peer_id: item.to_dict() for peer_id, item in self.accounts.items()},
            "channels": {channel_id: item.to_dict() for channel_id, item in self.channels.items()},
            "total_minted": float(self.total_minted),
            "total_burned": float(self.total_burned),
            "total_slashed": float(self.total_slashed),
            "total_auto_closed": int(self.total_auto_closed),
            "channel_default_ttl_seconds": int(self.channel_default_ttl_seconds),
            "channel_max_open_per_payer": int(self.channel_max_open_per_payer),
            "channel_min_deposit": float(self.channel_min_deposit),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HydraTokenEconomy":
        economy = cls(
            channel_default_ttl_seconds=int(payload.get("channel_default_ttl_seconds", 900)),
            channel_max_open_per_payer=int(payload.get("channel_max_open_per_payer", 8)),
            channel_min_deposit=float(payload.get("channel_min_deposit", 0.01)),
        )
        economy.accounts = {
            str(peer_id): TokenAccount.from_dict(dict(item))
            for peer_id, item in dict(payload.get("accounts", {})).items()
        }
        economy.channels = {
            str(channel_id): StateChannel.from_dict(dict(item))
            for channel_id, item in dict(payload.get("channels", {})).items()
        }
        economy.total_minted = float(payload.get("total_minted", 0.0))
        economy.total_burned = float(payload.get("total_burned", 0.0))
        economy.total_slashed = float(payload.get("total_slashed", 0.0))
        economy.total_auto_closed = int(payload.get("total_auto_closed", 0))
        return economy


class SqliteHydraTokenEconomy:
    """SQLite-backed HYDRA token + state-channel economy."""

    def __init__(
        self,
        path: str,
        *,
        channel_default_ttl_seconds: int = 900,
        channel_max_open_per_payer: int = 8,
        channel_min_deposit: float = 0.01,
        supply_cap: float = 69_000_000.0,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        self._supply_cap = max(0.0, float(supply_cap))
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()
        self._set_default_meta(
            channel_default_ttl_seconds=max(1, int(channel_default_ttl_seconds)),
            channel_max_open_per_payer=max(1, int(channel_max_open_per_payer)),
            channel_min_deposit=max(0.0, float(channel_min_deposit)),
        )
        self._migrate_legacy_json_if_present()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hydra_accounts (
                peer_id TEXT PRIMARY KEY,
                balance REAL NOT NULL DEFAULT 0.0,
                stake REAL NOT NULL DEFAULT 0.0,
                rewards_earned REAL NOT NULL DEFAULT 0.0,
                slashed_total REAL NOT NULL DEFAULT 0.0
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hydra_channels (
                channel_id TEXT PRIMARY KEY,
                payer_id TEXT NOT NULL,
                payee_id TEXT NOT NULL,
                escrow REAL NOT NULL DEFAULT 0.0,
                payer_spent REAL NOT NULL DEFAULT 0.0,
                status TEXT NOT NULL DEFAULT 'open',
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                nonce INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hydra_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def _meta_get(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM hydra_meta WHERE key=?", (str(key),)).fetchone()
        if row is None:
            return None
        return str(row[0])

    def _meta_set(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO hydra_meta(key, value) VALUES (?, ?)",
            (str(key), str(value)),
        )

    def _meta_float(self, key: str, default: float) -> float:
        raw = self._meta_get(key)
        if raw is None:
            return float(default)
        return float(raw)

    def _meta_int(self, key: str, default: int) -> int:
        raw = self._meta_get(key)
        if raw is None:
            return int(default)
        return int(raw)

    def _set_default_meta(
        self,
        *,
        channel_default_ttl_seconds: int,
        channel_max_open_per_payer: int,
        channel_min_deposit: float,
    ) -> None:
        defaults = {
            "total_minted": "0.0",
            "total_burned": "0.0",
            "total_slashed": "0.0",
            "total_auto_closed": "0",
            "total_supply": "0.0",
            "channel_default_ttl_seconds": str(int(channel_default_ttl_seconds)),
            "channel_max_open_per_payer": str(int(channel_max_open_per_payer)),
            "channel_min_deposit": str(float(channel_min_deposit)),
            "supply_cap": str(float(self._supply_cap)),
        }
        for key, value in defaults.items():
            if self._meta_get(key) is None:
                self._meta_set(key, value)
        self._conn.commit()

    def _legacy_json_path(self) -> Path | None:
        if self.path.suffix != ".db":
            return None
        return self.path.with_suffix(".json")

    def _migrate_legacy_json_if_present(self) -> None:
        legacy_path = self._legacy_json_path()
        if legacy_path is None or not legacy_path.exists():
            return
        account_rows = self._conn.execute("SELECT COUNT(*) FROM hydra_accounts").fetchone()
        channel_rows = self._conn.execute("SELECT COUNT(*) FROM hydra_channels").fetchone()
        if (account_rows and int(account_rows[0]) > 0) or (channel_rows and int(channel_rows[0]) > 0):
            return
        payload = json.loads(legacy_path.read_text())
        economy = HydraTokenEconomy.from_dict(dict(payload))
        with self._write_lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._persist_economy(economy)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        legacy_path.rename(legacy_path.with_suffix(".json.migrated"))

    @staticmethod
    def _compute_total_supply(economy: HydraTokenEconomy) -> float:
        active_channels = [item for item in economy.channels.values() if not item.closed]
        locked = sum(float(item.deposited) for item in active_channels)
        circulating = sum(float(item.total) for item in economy.accounts.values())
        return float(circulating + locked)

    def _enforce_supply_cap(self, economy: HydraTokenEconomy) -> None:
        cap = max(0.0, self._meta_float("supply_cap", self._supply_cap))
        total_supply = self._compute_total_supply(economy)
        if total_supply > cap + 1e-9:
            raise RuntimeError("hydra_supply_cap_exceeded")

    def _load_economy(self) -> HydraTokenEconomy:
        economy = HydraTokenEconomy(
            channel_default_ttl_seconds=max(1, self._meta_int("channel_default_ttl_seconds", 900)),
            channel_max_open_per_payer=max(1, self._meta_int("channel_max_open_per_payer", 8)),
            channel_min_deposit=max(0.0, self._meta_float("channel_min_deposit", 0.01)),
        )
        account_rows = self._conn.execute(
            """
            SELECT peer_id, balance, stake, rewards_earned, slashed_total
            FROM hydra_accounts
            """
        ).fetchall()
        economy.accounts = {
            str(peer_id): TokenAccount(
                peer_id=str(peer_id),
                balance=float(balance),
                stake=float(stake),
                rewards_earned=float(rewards_earned),
                slashed_total=float(slashed_total),
            )
            for peer_id, balance, stake, rewards_earned, slashed_total in account_rows
        }
        channel_rows = self._conn.execute(
            """
            SELECT channel_id, payer_id, payee_id, escrow, payer_spent, status, created_at, expires_at, nonce
            FROM hydra_channels
            """
        ).fetchall()
        economy.channels = {
            str(channel_id): StateChannel(
                channel_id=str(channel_id),
                payer=str(payer_id),
                payee=str(payee_id),
                deposited=float(escrow),
                spent=float(payer_spent),
                nonce=int(nonce),
                closed=(str(status).lower() != "open"),
                created_at=float(created_at),
                expires_at=float(expires_at),
            )
            for channel_id, payer_id, payee_id, escrow, payer_spent, status, created_at, expires_at, nonce in channel_rows
        }
        economy.total_minted = self._meta_float("total_minted", 0.0)
        economy.total_burned = self._meta_float("total_burned", 0.0)
        economy.total_slashed = self._meta_float("total_slashed", 0.0)
        economy.total_auto_closed = self._meta_int("total_auto_closed", 0)
        return economy

    def _persist_economy(self, economy: HydraTokenEconomy) -> None:
        self._conn.execute("DELETE FROM hydra_accounts")
        for item in economy.accounts.values():
            self._conn.execute(
                """
                INSERT INTO hydra_accounts(peer_id, balance, stake, rewards_earned, slashed_total)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(item.peer_id),
                    float(item.balance),
                    float(item.stake),
                    float(item.rewards_earned),
                    float(item.slashed_total),
                ),
            )

        self._conn.execute("DELETE FROM hydra_channels")
        for item in economy.channels.values():
            status = "closed" if item.closed else "open"
            self._conn.execute(
                """
                INSERT INTO hydra_channels(
                    channel_id, payer_id, payee_id, escrow, payer_spent, status, created_at, expires_at, nonce
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(item.channel_id),
                    str(item.payer),
                    str(item.payee),
                    float(item.deposited),
                    float(item.spent),
                    status,
                    float(item.created_at),
                    float(item.expires_at),
                    int(item.nonce),
                ),
            )

        total_supply = self._compute_total_supply(economy)
        self._meta_set("total_minted", str(float(economy.total_minted)))
        self._meta_set("total_burned", str(float(economy.total_burned)))
        self._meta_set("total_slashed", str(float(economy.total_slashed)))
        self._meta_set("total_auto_closed", str(int(economy.total_auto_closed)))
        self._meta_set("total_supply", str(float(total_supply)))
        self._meta_set("channel_default_ttl_seconds", str(int(economy.channel_default_ttl_seconds)))
        self._meta_set("channel_max_open_per_payer", str(int(economy.channel_max_open_per_payer)))
        self._meta_set("channel_min_deposit", str(float(economy.channel_min_deposit)))
        self._meta_set("supply_cap", str(float(self._supply_cap)))

    def _with_write(self, callback):
        with self._write_lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                economy = self._load_economy()
                result = callback(economy)
                self._enforce_supply_cap(economy)
                self._persist_economy(economy)
                self._conn.commit()
                return result
            except Exception:
                self._conn.rollback()
                raise

    def account_snapshot(self, peer_id: str) -> dict[str, Any]:
        return self._with_write(lambda economy: economy.account_snapshot(peer_id))

    def summary(self) -> dict[str, Any]:
        return self._with_write(lambda economy: economy.summary())

    def mint_for_inference(self, peer_id: str, tokens_served: int, reward_per_1k_tokens: float = 1.0) -> float:
        return self._with_write(
            lambda economy: economy.mint_for_inference(
                peer_id=peer_id,
                tokens_served=tokens_served,
                reward_per_1k_tokens=reward_per_1k_tokens,
            )
        )

    def transfer(self, from_peer_id: str, to_peer_id: str, amount: float) -> dict[str, float]:
        return self._with_write(
            lambda economy: economy.transfer(from_peer_id=from_peer_id, to_peer_id=to_peer_id, amount=amount)
        )

    def stake(self, peer_id: str, amount: float) -> dict[str, float]:
        return self._with_write(lambda economy: economy.stake(peer_id=peer_id, amount=amount))

    def unstake(self, peer_id: str, amount: float) -> dict[str, float]:
        return self._with_write(lambda economy: economy.unstake(peer_id=peer_id, amount=amount))

    def slash(self, peer_id: str, amount: float) -> dict[str, float]:
        return self._with_write(lambda economy: economy.slash(peer_id=peer_id, amount=amount))

    def open_state_channel(
        self,
        channel_id: str,
        payer: str,
        payee: str,
        deposit: float,
        ttl_seconds: int | None = None,
    ) -> dict[str, Any]:
        return self._with_write(
            lambda economy: economy.open_state_channel(
                channel_id=channel_id,
                payer=payer,
                payee=payee,
                deposit=deposit,
                ttl_seconds=ttl_seconds,
            ).to_dict()
        )

    def charge_state_channel(self, channel_id: str, amount: float) -> dict[str, Any]:
        return self._with_write(lambda economy: economy.charge_state_channel(channel_id=channel_id, amount=amount).to_dict())

    def reconcile_state_channel(self, channel_id: str, total_spent: float, nonce: int) -> dict[str, Any]:
        return self._with_write(
            lambda economy: economy.reconcile_state_channel(
                channel_id=channel_id,
                total_spent=total_spent,
                nonce=nonce,
            ).to_dict()
        )

    def close_state_channel(self, channel_id: str) -> dict[str, Any]:
        return self._with_write(lambda economy: economy.close_state_channel(channel_id=channel_id))

    def recover(self) -> dict[str, int | float]:
        """Run once on coordinator startup.

        Loads the persisted economy, settles any channels whose TTL expired
        while the coordinator was down, and writes the result back.  Returns a
        summary that callers (e.g. CoordinatorEngine) can log at startup.
        """

        def _run(economy: HydraTokenEconomy) -> dict[str, int | float]:
            expired_before = int(economy.total_auto_closed)
            economy._auto_expire_channels()
            expired_count = int(economy.total_auto_closed) - expired_before
            open_count = sum(1 for ch in economy.channels.values() if not ch.closed)
            return {
                "open_channels": open_count,
                "expired_on_recovery": expired_count,
                "total_accounts": len(economy.accounts),
                "total_minted": float(economy.total_minted),
                "total_burned": float(economy.total_burned),
            }

        return self._with_write(_run)

    def flush(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.commit()
        finally:
            self._conn.close()


class FileHydraTokenEconomy(SqliteHydraTokenEconomy):
    """Deprecated alias kept for backward compatibility."""

    pass
