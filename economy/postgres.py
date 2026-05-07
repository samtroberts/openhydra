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

"""
economy/postgres.py — PostgreSQL-backed ledger implementations.

These classes provide the same interface as SqliteCreditLedger and
SqliteHydraTokenEconomy but persist data in a PostgreSQL database via
psycopg2.  The entire module is guarded by _PSYCOPG2_AVAILABLE so that
importing this file never hard-fails on a machine without psycopg2.

Install the extra:  pip install openhydra-network[postgres]
"""
from __future__ import annotations

import threading
import time
from typing import Any

try:
    import psycopg2
    import psycopg2.extras
    _PSYCOPG2_AVAILABLE = True
except ImportError:
    _PSYCOPG2_AVAILABLE = False

from economy.token import HydraTokenEconomy, TokenAccount
from economy.state_channel import StateChannel


def _require_psycopg2() -> None:
    if not _PSYCOPG2_AVAILABLE:
        raise RuntimeError(
            "psycopg2 is not installed. "
            "Install it with: pip install openhydra-network[postgres]"
        )


class PostgresCreditLedger:
    """PostgreSQL-backed Tier 2 barter credit ledger.

    Implements the same interface as SqliteCreditLedger.
    Uses a single table:
        credits(peer_id TEXT PRIMARY KEY, balance REAL, updated_at REAL)

    Decay is applied on read/write using the per-row updated_at timestamp
    (analogous to the SQLite last_decay_ts column).
    """

    def __init__(self, dsn: str, decay_per_day: float = 0.05):
        _require_psycopg2()
        self._dsn = dsn
        self._decay_per_day = max(0.0, float(decay_per_day))
        self._write_lock = threading.Lock()
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = False
        self._init_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS credits (
                    peer_id   TEXT PRIMARY KEY,
                    balance   REAL NOT NULL DEFAULT 0.0,
                    updated_at REAL
                )
                """
            )
        self._conn.commit()

    @staticmethod
    def _apply_decay(balance: float, *, decay_per_day: float, last_ts: float, now_ts: float) -> float:
        elapsed_days = max(0.0, (now_ts - float(last_ts)) / 86400.0)
        if elapsed_days <= 0.0:
            return float(balance)
        factor = max(0.0, 1.0 - float(decay_per_day)) ** elapsed_days
        return float(balance) * factor

    def _read_row(self, cur, peer_id: str) -> tuple[float, float]:
        cur.execute(
            "SELECT balance, updated_at FROM credits WHERE peer_id = %s",
            (str(peer_id),),
        )
        row = cur.fetchone()
        if row is None:
            return 0.0, time.time()
        return float(row[0]), float(row[1]) if row[1] is not None else time.time()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def earn(self, peer_id: str, tokens_served: int) -> float:
        credits = max(0.0, int(tokens_served) / 1000.0)
        now_ts = time.time()
        with self._write_lock:
            try:
                with self._conn.cursor() as cur:
                    balance, last_ts = self._read_row(cur, peer_id)
                    decayed = self._apply_decay(
                        balance,
                        decay_per_day=self._decay_per_day,
                        last_ts=last_ts,
                        now_ts=now_ts,
                    )
                    new_balance = decayed + credits
                    cur.execute(
                        """
                        INSERT INTO credits (peer_id, balance, updated_at)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (peer_id) DO UPDATE SET
                            balance    = EXCLUDED.balance,
                            updated_at = EXCLUDED.updated_at
                        """,
                        (str(peer_id), float(new_balance), float(now_ts)),
                    )
                self._conn.commit()
                return float(new_balance)
            except Exception:
                self._conn.rollback()
                raise

    def spend(self, peer_id: str, credits: float) -> bool:
        amount = float(credits)
        if amount <= 0.0:
            return True
        now_ts = time.time()
        with self._write_lock:
            try:
                with self._conn.cursor() as cur:
                    balance, last_ts = self._read_row(cur, peer_id)
                    decayed = self._apply_decay(
                        balance,
                        decay_per_day=self._decay_per_day,
                        last_ts=last_ts,
                        now_ts=now_ts,
                    )
                    if decayed < amount:
                        self._conn.rollback()
                        return False
                    new_balance = decayed - amount
                    cur.execute(
                        """
                        INSERT INTO credits (peer_id, balance, updated_at)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (peer_id) DO UPDATE SET
                            balance    = EXCLUDED.balance,
                            updated_at = EXCLUDED.updated_at
                        """,
                        (str(peer_id), float(new_balance), float(now_ts)),
                    )
                self._conn.commit()
                return True
            except Exception:
                self._conn.rollback()
                raise

    def balance(self, peer_id: str) -> float:
        now_ts = time.time()
        with self._conn.cursor() as cur:
            balance, last_ts = self._read_row(cur, peer_id)
        return self._apply_decay(
            balance,
            decay_per_day=self._decay_per_day,
            last_ts=last_ts,
            now_ts=now_ts,
        )

    def flush(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.commit()
        finally:
            self._conn.close()


# ---------------------------------------------------------------------------
# PostgresHydraTokenEconomy
# ---------------------------------------------------------------------------

class PostgresHydraTokenEconomy:
    """PostgreSQL-backed HYDRA token + state-channel economy.

    Implements the same interface as SqliteHydraTokenEconomy.
    Tables:
        hydra_accounts  — one row per peer
        hydra_channels  — one row per payment channel
        hydra_meta      — key/value configuration & aggregate counters
    """

    def __init__(
        self,
        dsn: str,
        *,
        channel_default_ttl_seconds: int = 900,
        channel_max_open_per_payer: int = 8,
        channel_min_deposit: float = 0.01,
        supply_cap: float = 69_000_000.0,
    ):
        _require_psycopg2()
        self._dsn = dsn
        self._write_lock = threading.Lock()
        self._supply_cap = max(0.0, float(supply_cap))
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = False
        self._init_schema()
        self._set_default_meta(
            channel_default_ttl_seconds=max(1, int(channel_default_ttl_seconds)),
            channel_max_open_per_payer=max(1, int(channel_max_open_per_payer)),
            channel_min_deposit=max(0.0, float(channel_min_deposit)),
        )

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS hydra_accounts (
                    peer_id        TEXT PRIMARY KEY,
                    balance        REAL NOT NULL DEFAULT 0.0,
                    stake          REAL NOT NULL DEFAULT 0.0,
                    rewards_earned REAL NOT NULL DEFAULT 0.0,
                    slashed_total  REAL NOT NULL DEFAULT 0.0
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS hydra_channels (
                    channel_id  TEXT PRIMARY KEY,
                    payer_id    TEXT NOT NULL,
                    payee_id    TEXT NOT NULL,
                    escrow      REAL NOT NULL DEFAULT 0.0,
                    payer_spent REAL NOT NULL DEFAULT 0.0,
                    status      TEXT NOT NULL DEFAULT 'open',
                    created_at  REAL NOT NULL,
                    expires_at  REAL NOT NULL,
                    nonce       INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS hydra_meta (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Meta helpers
    # ------------------------------------------------------------------

    def _meta_get(self, cur, key: str) -> str | None:
        cur.execute("SELECT value FROM hydra_meta WHERE key = %s", (str(key),))
        row = cur.fetchone()
        if row is None:
            return None
        return str(row[0])

    def _meta_set(self, cur, key: str, value: str) -> None:
        cur.execute(
            """
            INSERT INTO hydra_meta (key, value)
            VALUES (%s, %s)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
            """,
            (str(key), str(value)),
        )

    def _meta_float(self, cur, key: str, default: float) -> float:
        raw = self._meta_get(cur, key)
        if raw is None:
            return float(default)
        return float(raw)

    def _meta_int(self, cur, key: str, default: int) -> int:
        raw = self._meta_get(cur, key)
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
        with self._conn.cursor() as cur:
            for key, value in defaults.items():
                if self._meta_get(cur, key) is None:
                    self._meta_set(cur, key, value)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Load / persist
    # ------------------------------------------------------------------

    def _load_economy(self, cur) -> HydraTokenEconomy:
        economy = HydraTokenEconomy(
            channel_default_ttl_seconds=max(1, self._meta_int(cur, "channel_default_ttl_seconds", 900)),
            channel_max_open_per_payer=max(1, self._meta_int(cur, "channel_max_open_per_payer", 8)),
            channel_min_deposit=max(0.0, self._meta_float(cur, "channel_min_deposit", 0.01)),
        )
        cur.execute(
            "SELECT peer_id, balance, stake, rewards_earned, slashed_total FROM hydra_accounts"
        )
        economy.accounts = {
            str(peer_id): TokenAccount(
                peer_id=str(peer_id),
                balance=float(balance),
                stake=float(stake),
                rewards_earned=float(rewards_earned),
                slashed_total=float(slashed_total),
            )
            for peer_id, balance, stake, rewards_earned, slashed_total in cur.fetchall()
        }
        cur.execute(
            """
            SELECT channel_id, payer_id, payee_id, escrow, payer_spent,
                   status, created_at, expires_at, nonce
            FROM hydra_channels
            """
        )
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
            for channel_id, payer_id, payee_id, escrow, payer_spent, status, created_at, expires_at, nonce
            in cur.fetchall()
        }
        economy.total_minted = self._meta_float(cur, "total_minted", 0.0)
        economy.total_burned = self._meta_float(cur, "total_burned", 0.0)
        economy.total_slashed = self._meta_float(cur, "total_slashed", 0.0)
        economy.total_auto_closed = self._meta_int(cur, "total_auto_closed", 0)
        return economy

    @staticmethod
    def _compute_total_supply(economy: HydraTokenEconomy) -> float:
        active_channels = [item for item in economy.channels.values() if not item.closed]
        locked = sum(float(item.deposited) for item in active_channels)
        circulating = sum(float(item.total) for item in economy.accounts.values())
        return float(circulating + locked)

    def _enforce_supply_cap(self, cur, economy: HydraTokenEconomy) -> None:
        cap = max(0.0, self._meta_float(cur, "supply_cap", self._supply_cap))
        total_supply = self._compute_total_supply(economy)
        if total_supply > cap + 1e-9:
            raise RuntimeError("hydra_supply_cap_exceeded")

    def _persist_economy(self, cur, economy: HydraTokenEconomy) -> None:
        cur.execute("DELETE FROM hydra_accounts")
        for item in economy.accounts.values():
            cur.execute(
                """
                INSERT INTO hydra_accounts (peer_id, balance, stake, rewards_earned, slashed_total)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    str(item.peer_id),
                    float(item.balance),
                    float(item.stake),
                    float(item.rewards_earned),
                    float(item.slashed_total),
                ),
            )

        cur.execute("DELETE FROM hydra_channels")
        for item in economy.channels.values():
            status = "closed" if item.closed else "open"
            cur.execute(
                """
                INSERT INTO hydra_channels (
                    channel_id, payer_id, payee_id, escrow, payer_spent,
                    status, created_at, expires_at, nonce
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
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
        self._meta_set(cur, "total_minted", str(float(economy.total_minted)))
        self._meta_set(cur, "total_burned", str(float(economy.total_burned)))
        self._meta_set(cur, "total_slashed", str(float(economy.total_slashed)))
        self._meta_set(cur, "total_auto_closed", str(int(economy.total_auto_closed)))
        self._meta_set(cur, "total_supply", str(float(total_supply)))
        self._meta_set(cur, "channel_default_ttl_seconds", str(int(economy.channel_default_ttl_seconds)))
        self._meta_set(cur, "channel_max_open_per_payer", str(int(economy.channel_max_open_per_payer)))
        self._meta_set(cur, "channel_min_deposit", str(float(economy.channel_min_deposit)))
        self._meta_set(cur, "supply_cap", str(float(self._supply_cap)))

    def _with_write(self, callback):
        with self._write_lock:
            try:
                with self._conn.cursor() as cur:
                    economy = self._load_economy(cur)
                    result = callback(economy)
                    self._enforce_supply_cap(cur, economy)
                    self._persist_economy(cur, economy)
                self._conn.commit()
                return result
            except Exception:
                self._conn.rollback()
                raise

    # ------------------------------------------------------------------
    # Public interface (matches SqliteHydraTokenEconomy)
    # ------------------------------------------------------------------

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
            lambda economy: economy.transfer(
                from_peer_id=from_peer_id,
                to_peer_id=to_peer_id,
                amount=amount,
            )
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
        return self._with_write(
            lambda economy: economy.charge_state_channel(
                channel_id=channel_id,
                amount=amount,
            ).to_dict()
        )

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

        Settles any channels whose TTL expired while the coordinator was down
        and writes the result back.  Returns a summary dict matching the one
        SqliteHydraTokenEconomy.recover() produces.
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
