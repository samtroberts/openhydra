-- OpenHydra PostgreSQL schema
-- Applied automatically via docker-entrypoint-initdb.d

CREATE TABLE IF NOT EXISTS credits (
    peer_id    TEXT PRIMARY KEY,
    balance    REAL NOT NULL DEFAULT 0.0,
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS hydra_accounts (
    peer_id        TEXT PRIMARY KEY,
    balance        REAL NOT NULL DEFAULT 0.0,
    stake          REAL NOT NULL DEFAULT 0.0,
    rewards_earned REAL NOT NULL DEFAULT 0.0,
    slashed_total  REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS hydra_channels (
    channel_id  TEXT PRIMARY KEY,
    payer_id    TEXT NOT NULL,
    provider_id TEXT NOT NULL,
    deposit     REAL NOT NULL DEFAULT 0.0,
    spent       REAL NOT NULL DEFAULT 0.0,
    provider_spent REAL NOT NULL DEFAULT 0.0,
    status      TEXT NOT NULL DEFAULT 'open',
    created_at  REAL,
    expires_at  REAL
);

CREATE TABLE IF NOT EXISTS hydra_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
