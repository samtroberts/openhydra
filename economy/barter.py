from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import sqlite3
import threading
import time


@dataclass
class CreditLedger:
    decay_per_day: float = 0.05
    balances: dict[str, float] = field(default_factory=dict)
    last_decay_ts: float = field(default_factory=time.time)

    def _decay(self) -> None:
        now = time.time()
        elapsed_days = max(0.0, (now - self.last_decay_ts) / 86400.0)
        if elapsed_days <= 0:
            return
        factor = max(0.0, 1.0 - self.decay_per_day) ** elapsed_days
        for peer_id in list(self.balances.keys()):
            self.balances[peer_id] *= factor
        self.last_decay_ts = now

    def earn(self, peer_id: str, tokens_served: int) -> float:
        self._decay()
        credits = max(0.0, tokens_served / 1000.0)
        self.balances[peer_id] = self.balances.get(peer_id, 0.0) + credits
        return self.balances[peer_id]

    def spend(self, peer_id: str, credits: float) -> bool:
        self._decay()
        current = self.balances.get(peer_id, 0.0)
        if current < credits:
            return False
        self.balances[peer_id] = current - credits
        return True

    def balance(self, peer_id: str) -> float:
        self._decay()
        return self.balances.get(peer_id, 0.0)

    def to_dict(self) -> dict:
        self._decay()
        return {
            "decay_per_day": self.decay_per_day,
            "balances": self.balances,
            "last_decay_ts": self.last_decay_ts,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CreditLedger":
        return cls(
            decay_per_day=float(payload.get("decay_per_day", 0.05)),
            balances={k: float(v) for k, v in dict(payload.get("balances", {})).items()},
            last_decay_ts=float(payload.get("last_decay_ts", time.time())),
        )


class SqliteCreditLedger:
    """SQLite-backed Tier 2 barter credit ledger."""

    def __init__(self, path: str, decay_per_day: float = 0.05):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()
        self._set_decay_per_day(decay_per_day)
        self._migrate_legacy_json_if_present()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS barter_credits (
                peer_id TEXT PRIMARY KEY,
                balance REAL NOT NULL DEFAULT 0.0,
                last_decay_ts REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS barter_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def _set_decay_per_day(self, decay_per_day: float) -> None:
        value = max(0.0, float(decay_per_day))
        self._conn.execute(
            "INSERT OR REPLACE INTO barter_meta(key, value) VALUES ('decay_per_day', ?)",
            (str(value),),
        )
        self._conn.commit()

    def _decay_per_day(self) -> float:
        row = self._conn.execute("SELECT value FROM barter_meta WHERE key='decay_per_day'").fetchone()
        if row is None:
            return 0.05
        return max(0.0, float(row[0]))

    def _legacy_json_path(self) -> Path | None:
        if self.path.suffix != ".db":
            return None
        return self.path.with_suffix(".json")

    def _migrate_legacy_json_if_present(self) -> None:
        legacy_path = self._legacy_json_path()
        if legacy_path is None or not legacy_path.exists():
            return
        row = self._conn.execute("SELECT COUNT(*) FROM barter_credits").fetchone()
        if row is not None and int(row[0]) > 0:
            return

        payload = json.loads(legacy_path.read_text())
        decay = max(0.0, float(payload.get("decay_per_day", self._decay_per_day())))
        balances = dict(payload.get("balances", {}))
        last_decay_ts = float(payload.get("last_decay_ts", time.time()))

        with self._write_lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute("DELETE FROM barter_credits")
                self._conn.execute(
                    "INSERT OR REPLACE INTO barter_meta(key, value) VALUES ('decay_per_day', ?)",
                    (str(decay),),
                )
                for peer_id, balance in balances.items():
                    self._conn.execute(
                        """
                        INSERT OR REPLACE INTO barter_credits(peer_id, balance, last_decay_ts)
                        VALUES (?, ?, ?)
                        """,
                        (str(peer_id), float(balance), float(last_decay_ts)),
                    )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        legacy_path.rename(legacy_path.with_suffix(".json.migrated"))

    @staticmethod
    def _apply_decay(balance: float, *, decay_per_day: float, last_decay_ts: float, now_ts: float) -> float:
        elapsed_days = max(0.0, (now_ts - float(last_decay_ts)) / 86400.0)
        if elapsed_days <= 0.0:
            return float(balance)
        factor = max(0.0, 1.0 - float(decay_per_day)) ** elapsed_days
        return float(balance) * factor

    def _read_balance_row(self, peer_id: str) -> tuple[float, float]:
        row = self._conn.execute(
            "SELECT balance, last_decay_ts FROM barter_credits WHERE peer_id=?",
            (str(peer_id),),
        ).fetchone()
        if row is None:
            return 0.0, time.time()
        return float(row[0]), float(row[1])

    def earn(self, peer_id: str, tokens_served: int) -> float:
        credits = max(0.0, int(tokens_served) / 1000.0)
        now_ts = time.time()
        with self._write_lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                decay_per_day = self._decay_per_day()
                balance, last_decay_ts = self._read_balance_row(peer_id)
                decayed = self._apply_decay(balance, decay_per_day=decay_per_day, last_decay_ts=last_decay_ts, now_ts=now_ts)
                new_balance = decayed + credits
                self._conn.execute(
                    """
                    INSERT INTO barter_credits(peer_id, balance, last_decay_ts)
                    VALUES (?, ?, ?)
                    ON CONFLICT(peer_id) DO UPDATE SET
                        balance=excluded.balance,
                        last_decay_ts=excluded.last_decay_ts
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
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                decay_per_day = self._decay_per_day()
                balance, last_decay_ts = self._read_balance_row(peer_id)
                decayed = self._apply_decay(balance, decay_per_day=decay_per_day, last_decay_ts=last_decay_ts, now_ts=now_ts)
                if decayed < amount:
                    self._conn.rollback()
                    return False
                new_balance = decayed - amount
                self._conn.execute(
                    """
                    INSERT INTO barter_credits(peer_id, balance, last_decay_ts)
                    VALUES (?, ?, ?)
                    ON CONFLICT(peer_id) DO UPDATE SET
                        balance=excluded.balance,
                        last_decay_ts=excluded.last_decay_ts
                    """,
                    (str(peer_id), float(new_balance), float(now_ts)),
                )
                self._conn.commit()
                return True
            except Exception:
                self._conn.rollback()
                raise

    def balance(self, peer_id: str) -> float:
        decay_per_day = self._decay_per_day()
        now_ts = time.time()
        balance, last_decay_ts = self._read_balance_row(peer_id)
        return self._apply_decay(balance, decay_per_day=decay_per_day, last_decay_ts=last_decay_ts, now_ts=now_ts)

    def flush(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.commit()
        finally:
            self._conn.close()


class FileCreditLedger(SqliteCreditLedger):
    """Deprecated alias kept for backward compatibility."""

    pass
