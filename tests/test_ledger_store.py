from __future__ import annotations

import json
import time

from economy.barter import SqliteCreditLedger
from economy.token import SqliteHydraTokenEconomy


def test_sqlite_credit_ledger_persists(tmp_path):
    path = tmp_path / "credits.db"

    ledger1 = SqliteCreditLedger(str(path), decay_per_day=0.0)
    ledger1.earn("peer-x", tokens_served=1500)
    ledger1.close()

    ledger2 = SqliteCreditLedger(str(path), decay_per_day=0.0)
    assert ledger2.balance("peer-x") == 1.5
    assert ledger2.spend("peer-x", 0.5)
    ledger2.close()

    ledger3 = SqliteCreditLedger(str(path), decay_per_day=0.0)
    assert ledger3.balance("peer-x") == 1.0
    ledger3.close()


def test_sqlite_hydra_store_migrates_legacy_json(tmp_path):
    db_path = tmp_path / "hydra_tokens.db"
    legacy_path = tmp_path / "hydra_tokens.json"
    legacy_payload = {
        "accounts": {
            "peer-a": {
                "peer_id": "peer-a",
                "balance": 2.0,
                "stake": 0.0,
                "rewards_earned": 2.0,
                "slashed_total": 0.0,
            }
        },
        "channels": {},
        "total_minted": 2.0,
        "total_burned": 0.0,
        "total_slashed": 0.0,
        "total_auto_closed": 0,
        "channel_default_ttl_seconds": 900,
        "channel_max_open_per_payer": 8,
        "channel_min_deposit": 0.01,
    }
    legacy_path.write_text(json.dumps(legacy_payload))

    store = SqliteHydraTokenEconomy(str(db_path))
    try:
        snap = store.account_snapshot("peer-a")
        assert snap["balance"] == 2.0
        assert not legacy_path.exists()
        assert (tmp_path / "hydra_tokens.json.migrated").exists()
    finally:
        store.close()


# ---------------------------------------------------------------------------
# recover() tests
# ---------------------------------------------------------------------------

def test_recover_on_empty_db_returns_zero_stats(tmp_path):
    """recover() on a fresh DB returns all-zero stats and does not crash."""
    store = SqliteHydraTokenEconomy(str(tmp_path / "hydra.db"))
    try:
        stats = store.recover()
        assert stats["open_channels"] == 0
        assert stats["expired_on_recovery"] == 0
        assert stats["total_accounts"] == 0
        assert stats["total_minted"] == 0.0
        assert stats["total_burned"] == 0.0
    finally:
        store.close()


def test_recover_closes_expired_channels_and_settles_balances(tmp_path):
    """After a simulated crash, recover() settles channels whose TTL lapsed."""
    db_path = str(tmp_path / "hydra.db")
    now_ts = time.time()

    # ---- Session 1: open two channels, one already expired ----
    s1 = SqliteHydraTokenEconomy(db_path, channel_min_deposit=0.001)
    try:
        # Fund payer accounts so the deposit doesn't fail the supply-cap check
        s1.mint_for_inference("alice", tokens_served=10_000, reward_per_1k_tokens=1.0)
        s1.mint_for_inference("bob", tokens_served=10_000, reward_per_1k_tokens=1.0)
        # Channel A: still active (expires far in future)
        s1.open_state_channel("ch-active", payer="alice", payee="bob", deposit=1.0, ttl_seconds=9999)
        # Channel B: already expired (set TTL to 1 second, then we'll fake the clock)
        s1.open_state_channel("ch-expired", payer="bob", payee="alice", deposit=0.5, ttl_seconds=1)
    finally:
        s1.close()

    # Directly patch the expired channel's expires_at to the past via SQLite
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE hydra_channels SET expires_at=? WHERE channel_id=?",
        (now_ts - 10.0, "ch-expired"),
    )
    conn.commit()
    conn.close()

    # ---- Session 2: recover() should settle the expired channel ----
    s2 = SqliteHydraTokenEconomy(db_path)
    try:
        stats = s2.recover()
        assert stats["expired_on_recovery"] == 1
        assert stats["open_channels"] == 1   # ch-active still open
        assert stats["total_accounts"] == 2
    finally:
        s2.close()


def test_recover_is_idempotent(tmp_path):
    """Calling recover() twice does not double-close channels."""
    db_path = str(tmp_path / "hydra.db")

    store = SqliteHydraTokenEconomy(db_path, channel_min_deposit=0.001)
    try:
        store.mint_for_inference("alice", tokens_served=10_000, reward_per_1k_tokens=1.0)
        store.open_state_channel("ch-1", payer="alice", payee="bob", deposit=1.0, ttl_seconds=9999)

        stats1 = store.recover()
        stats2 = store.recover()

        assert stats1["expired_on_recovery"] == 0
        assert stats2["expired_on_recovery"] == 0
        assert stats1["open_channels"] == stats2["open_channels"] == 1
    finally:
        store.close()
