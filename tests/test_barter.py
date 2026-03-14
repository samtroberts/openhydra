from __future__ import annotations

import json
from pathlib import Path
import threading
import time

from economy.barter import CreditLedger, SqliteCreditLedger
import pytest


def test_barter_earn_and_spend():
    ledger = CreditLedger(decay_per_day=0.0)
    balance = ledger.earn("peer-x", tokens_served=2500)
    assert balance == 2.5
    assert ledger.spend("peer-x", 1.25)
    assert round(ledger.balance("peer-x"), 5) == 1.25


def test_sqlite_credit_ledger_earn_spend_balance(tmp_path: Path):
    ledger_path = tmp_path / "credits.db"
    ledger = SqliteCreditLedger(str(ledger_path), decay_per_day=0.0)
    try:
        assert ledger.earn("peer-x", tokens_served=2500) == pytest.approx(2.5, abs=1e-9)
        assert ledger.spend("peer-x", 1.0)
        assert ledger.balance("peer-x") == pytest.approx(1.5, abs=1e-9)
    finally:
        ledger.close()


def test_sqlite_credit_ledger_decay_is_applied_lazily(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ledger_path = tmp_path / "credits.db"
    now = {"value": 100.0}
    monkeypatch.setattr(time, "time", lambda: now["value"])
    ledger = SqliteCreditLedger(str(ledger_path), decay_per_day=0.1)
    try:
        ledger.earn("peer-x", tokens_served=1000)
        now["value"] += 86400.0
        assert ledger.balance("peer-x") == pytest.approx(0.9, rel=1e-6)
    finally:
        ledger.close()


def test_sqlite_credit_ledger_supports_concurrent_writes(tmp_path: Path):
    ledger = SqliteCreditLedger(str(tmp_path / "credits.db"), decay_per_day=0.0)
    try:
        def _worker():
            for _ in range(250):
                ledger.earn("peer-x", tokens_served=1000)

        threads = [threading.Thread(target=_worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert ledger.balance("peer-x") == pytest.approx(1000.0, abs=1e-6)
    finally:
        ledger.close()


def test_sqlite_credit_ledger_migrates_from_json(tmp_path: Path):
    db_path = tmp_path / "credits.db"
    legacy_path = tmp_path / "credits.json"
    payload = {
        "decay_per_day": 0.0,
        "balances": {"peer-a": 3.25},
        "last_decay_ts": 1.0,
    }
    legacy_path.write_text(json.dumps(payload))
    ledger = SqliteCreditLedger(str(db_path), decay_per_day=0.05)
    try:
        assert ledger.balance("peer-a") == pytest.approx(3.25, abs=1e-9)
        assert not legacy_path.exists()
        assert (tmp_path / "credits.json.migrated").exists()
    finally:
        ledger.close()
