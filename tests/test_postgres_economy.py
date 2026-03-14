"""
tests/test_postgres_economy.py

Unit tests for the PostgreSQL-backed ledger classes.

No real PostgreSQL instance is needed — psycopg2.connect() is mocked via
unittest.mock so all DB interactions are intercepted at the cursor level.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helper: build a minimal psycopg2 stub so the module can be imported even
# when psycopg2 is not installed in the test environment.
# ---------------------------------------------------------------------------

def _make_psycopg2_stub():
    """Return a minimal psycopg2 module stub."""
    stub = types.ModuleType("psycopg2")
    stub.connect = MagicMock()
    stub.extras = types.ModuleType("psycopg2.extras")
    return stub


def _install_psycopg2_stub():
    """Inject the stub into sys.modules so economy.postgres can import it."""
    if "psycopg2" not in sys.modules:
        sys.modules["psycopg2"] = _make_psycopg2_stub()
        sys.modules["psycopg2.extras"] = sys.modules["psycopg2"].extras


# ---------------------------------------------------------------------------
# Cursor / connection factory helpers
# ---------------------------------------------------------------------------

def _make_cursor(fetchone_result=None, fetchall_result=None):
    """Return a MagicMock that behaves as a psycopg2 cursor context manager."""
    cur = MagicMock()
    cur.fetchone.return_value = fetchone_result
    cur.fetchall.return_value = fetchall_result or []
    # Support `with conn.cursor() as cur:` — the context manager returns itself
    cur.__enter__ = MagicMock(return_value=cur)
    cur.__exit__ = MagicMock(return_value=False)
    return cur


def _make_connection(cursor=None):
    """Return a MagicMock that behaves as a psycopg2 connection."""
    conn = MagicMock()
    if cursor is None:
        cursor = _make_cursor()
    conn.cursor.return_value = cursor
    conn.autocommit = False
    conn.commit = MagicMock()
    conn.rollback = MagicMock()
    conn.close = MagicMock()
    return conn


# ---------------------------------------------------------------------------
# 1. PostgresCreditLedger — earn & spend issue correct SQL
# ---------------------------------------------------------------------------

class TestPostgresCreditLedgerEarnSpend:
    def test_postgres_credit_ledger_earn_spend(self):
        _install_psycopg2_stub()

        import importlib
        import economy.postgres as pg_mod
        importlib.reload(pg_mod)

        # Patch _PSYCOPG2_AVAILABLE so the guard passes
        pg_mod._PSYCOPG2_AVAILABLE = True

        conn = _make_connection()

        with patch.object(pg_mod, "_PSYCOPG2_AVAILABLE", True), \
             patch("psycopg2.connect", return_value=conn):
            ledger = pg_mod.PostgresCreditLedger("postgresql://fake/db")

        # After __init__, commit is called for schema creation
        conn.commit.assert_called()

        # Reset call history so we can track earn() specifically
        conn.reset_mock()
        cur = _make_cursor(fetchone_result=None)  # no existing row → fresh account
        conn.cursor.return_value = cur

        ledger.earn("peer-A", 2000)

        # cursor.execute should have been called: the INSERT/ON CONFLICT upsert
        execute_calls = cur.execute.call_args_list
        sql_calls = [str(c.args[0]).strip() for c in execute_calls if c.args]
        upsert_calls = [s for s in sql_calls if "INSERT INTO credits" in s]
        assert upsert_calls, "earn() must issue an INSERT INTO credits statement"

        # Verify %s placeholders are used (not SQLite's ?)
        for sql in upsert_calls:
            assert "%s" in sql, f"Expected %%s placeholder, got: {sql!r}"
            assert "?" not in sql, f"Found SQLite placeholder '?' in: {sql!r}"

        # Reset and test spend()
        conn.reset_mock()
        # fetchone returns a row with a recent timestamp so decay is negligible
        import time as _time
        cur2 = _make_cursor(fetchone_result=(100.0, _time.time()))  # balance=100, updated_at=now
        conn.cursor.return_value = cur2

        result = ledger.spend("peer-A", 10.0)
        assert result is True

        execute_calls2 = cur2.execute.call_args_list
        sql_calls2 = [str(c.args[0]).strip() for c in execute_calls2 if c.args]
        upsert_calls2 = [s for s in sql_calls2 if "INSERT INTO credits" in s]
        assert upsert_calls2, "spend() must issue an INSERT INTO credits statement"
        for sql in upsert_calls2:
            assert "%s" in sql, f"Expected %%s placeholder in spend SQL, got: {sql!r}"


# ---------------------------------------------------------------------------
# 2. PostgresCreditLedger — balance() returns correct value
# ---------------------------------------------------------------------------

class TestPostgresCreditLedgerBalance:
    def test_postgres_credit_ledger_balance(self):
        _install_psycopg2_stub()

        import importlib
        import economy.postgres as pg_mod
        importlib.reload(pg_mod)
        pg_mod._PSYCOPG2_AVAILABLE = True

        conn = _make_connection()

        with patch.object(pg_mod, "_PSYCOPG2_AVAILABLE", True), \
             patch("psycopg2.connect", return_value=conn):
            ledger = pg_mod.PostgresCreditLedger("postgresql://fake/db", decay_per_day=0.0)

        # Set up cursor to return a fixed balance row; updated_at very recent so no decay
        import time
        now = time.time()
        cur = _make_cursor(fetchone_result=(42.5, now))
        conn.cursor.return_value = cur

        bal = ledger.balance("peer-B")
        # With decay_per_day=0.0 and a very recent timestamp, balance should be ~42.5
        assert abs(bal - 42.5) < 0.01, f"Expected ~42.5, got {bal}"

        # Verify the SELECT query used %s, not ?
        execute_calls = cur.execute.call_args_list
        select_calls = [
            str(c.args[0]).strip()
            for c in execute_calls
            if c.args and "SELECT" in str(c.args[0])
        ]
        assert select_calls, "balance() must issue a SELECT query"
        for sql in select_calls:
            assert "%s" in sql or "WHERE peer_id" in sql, \
                f"Expected parameterised SELECT, got: {sql!r}"


# ---------------------------------------------------------------------------
# 3. PostgresHydraTokenEconomy — tables are created on init
# ---------------------------------------------------------------------------

class TestPostgresHydraEconomyInit:
    def test_postgres_hydra_economy_init(self):
        _install_psycopg2_stub()

        import importlib
        import economy.postgres as pg_mod
        importlib.reload(pg_mod)
        pg_mod._PSYCOPG2_AVAILABLE = True

        conn = _make_connection()
        # _set_default_meta calls _meta_get for each key → fetchone returns None (fresh DB)
        cur = _make_cursor(fetchone_result=None, fetchall_result=[])
        conn.cursor.return_value = cur

        with patch.object(pg_mod, "_PSYCOPG2_AVAILABLE", True), \
             patch("psycopg2.connect", return_value=conn):
            economy = pg_mod.PostgresHydraTokenEconomy("postgresql://fake/db")

        # At least one commit must have happened (schema + meta init)
        assert conn.commit.called, "PostgresHydraTokenEconomy.__init__ must commit"

        # Verify all three tables are created
        all_sql = " ".join(
            str(c.args[0]) for c in cur.execute.call_args_list if c.args
        )
        assert "hydra_accounts" in all_sql, "hydra_accounts table must be created"
        assert "hydra_channels" in all_sql, "hydra_channels table must be created"
        assert "hydra_meta" in all_sql, "hydra_meta table must be created"


# ---------------------------------------------------------------------------
# 4. EngineConfig with database_url=None → CoordinatorEngine uses SQLite
# ---------------------------------------------------------------------------

class TestEngineUsesSqliteWithoutDatabaseUrl:
    def test_engine_uses_sqlite_without_database_url(self):
        """EngineConfig(database_url=None) → ledger is SqliteCreditLedger."""
        import importlib

        # We need to import engine without triggering the full module init chain
        # that requires real gRPC stubs, so we patch the heavy dependencies.
        with patch("coordinator.engine.HealthScorer"), \
             patch("coordinator.engine.ReplicationMonitor"), \
             patch("coordinator.engine.GroundingClient"), \
             patch("coordinator.engine.MysteryShopper"), \
             patch("coordinator.engine.OpenHydraLedgerBridge"), \
             patch("coordinator.engine.load_peer_config"), \
             patch("coordinator.engine.load_peers_from_dht"):

            from coordinator.engine import CoordinatorEngine, EngineConfig
            from economy.barter import SqliteCreditLedger
            from economy.token import SqliteHydraTokenEconomy

            config = EngineConfig(database_url=None)

            # Patch SqliteHydraTokenEconomy.recover() to avoid file I/O
            with patch.object(SqliteHydraTokenEconomy, "recover", return_value={
                "open_channels": 0,
                "expired_on_recovery": 0,
                "total_accounts": 0,
                "total_minted": 0.0,
                "total_burned": 0.0,
            }), patch.object(SqliteHydraTokenEconomy, "_init_schema"), \
               patch.object(SqliteHydraTokenEconomy, "_set_default_meta"), \
               patch.object(SqliteHydraTokenEconomy, "_migrate_legacy_json_if_present"), \
               patch("economy.barter.SqliteCreditLedger._init_schema"), \
               patch("economy.barter.SqliteCreditLedger._set_decay_per_day"), \
               patch("economy.barter.SqliteCreditLedger._migrate_legacy_json_if_present"), \
               patch("sqlite3.connect"):

                engine = CoordinatorEngine(config)
                assert isinstance(engine.ledger, SqliteCreditLedger), (
                    f"Expected SqliteCreditLedger, got {type(engine.ledger)}"
                )
                assert isinstance(engine.hydra, SqliteHydraTokenEconomy), (
                    f"Expected SqliteHydraTokenEconomy, got {type(engine.hydra)}"
                )


# ---------------------------------------------------------------------------
# 5. EngineConfig with database_url set → CoordinatorEngine uses Postgres
# ---------------------------------------------------------------------------

class TestEngineUsesPostgresWithDatabaseUrl:
    def test_engine_uses_postgres_with_database_url(self):
        """EngineConfig(database_url='postgresql://...') → ledger is PostgresCreditLedger."""
        _install_psycopg2_stub()

        import importlib
        import economy.postgres as pg_mod
        importlib.reload(pg_mod)
        pg_mod._PSYCOPG2_AVAILABLE = True

        pg_conn = _make_connection()
        # fresh DB: fetchone returns None for meta lookups, fetchall returns []
        pg_cur = _make_cursor(fetchone_result=None, fetchall_result=[])
        pg_conn.cursor.return_value = pg_cur

        with patch.object(pg_mod, "_PSYCOPG2_AVAILABLE", True), \
             patch("psycopg2.connect", return_value=pg_conn), \
             patch("coordinator.engine.HealthScorer"), \
             patch("coordinator.engine.ReplicationMonitor"), \
             patch("coordinator.engine.GroundingClient"), \
             patch("coordinator.engine.MysteryShopper"), \
             patch("coordinator.engine.OpenHydraLedgerBridge"), \
             patch("coordinator.engine.load_peer_config"), \
             patch("coordinator.engine.load_peers_from_dht"):

            # Reload engine so it picks up the patched psycopg2
            import coordinator.engine as eng_mod
            importlib.reload(eng_mod)

            from coordinator.engine import CoordinatorEngine, EngineConfig

            config = EngineConfig(database_url="postgresql://openhydra:openhydra@localhost:5432/openhydra")

            # Patch PostgresHydraTokenEconomy.recover() to avoid real DB calls
            with patch.object(pg_mod.PostgresHydraTokenEconomy, "recover", return_value={
                "open_channels": 0,
                "expired_on_recovery": 0,
                "total_accounts": 0,
                "total_minted": 0.0,
                "total_burned": 0.0,
            }):
                engine = CoordinatorEngine(config)

            assert isinstance(engine.ledger, pg_mod.PostgresCreditLedger), (
                f"Expected PostgresCreditLedger, got {type(engine.ledger)}"
            )
            assert isinstance(engine.hydra, pg_mod.PostgresHydraTokenEconomy), (
                f"Expected PostgresHydraTokenEconomy, got {type(engine.hydra)}"
            )
