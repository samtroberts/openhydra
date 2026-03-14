from economy.token import FileHydraTokenEconomy, HydraTokenEconomy
import pytest


def test_hydra_token_transfer_stake_and_slash():
    economy = HydraTokenEconomy()
    economy.mint("alice", 10.0)
    economy.transfer("alice", "bob", 3.0)

    snap_alice = economy.account_snapshot("alice")
    snap_bob = economy.account_snapshot("bob")
    assert snap_alice["balance"] == 7.0
    assert snap_bob["balance"] == 3.0

    economy.stake("bob", 2.0)
    stake_snap = economy.account_snapshot("bob")
    assert stake_snap["balance"] == 1.0
    assert stake_snap["stake"] == 2.0

    economy.slash("bob", 1.5)
    slashed_snap = economy.account_snapshot("bob")
    assert slashed_snap["balance"] == 1.0
    assert slashed_snap["stake"] == 0.5
    assert slashed_snap["slashed_total"] == 1.5

    summary = economy.summary()
    assert summary["total_minted"] == 10.0
    assert summary["total_burned"] == 1.5
    assert summary["total_slashed"] == 1.5


def test_hydra_state_channel_charge_reconcile_and_close():
    economy = HydraTokenEconomy()
    economy.mint("alice", 10.0)

    opened = economy.open_state_channel("ch-1", "alice", "bob", 6.0)
    assert opened.channel_id == "ch-1"
    assert economy.account_snapshot("alice")["balance"] == 4.0

    charged = economy.charge_state_channel("ch-1", 1.5)
    assert charged.spent == 1.5
    assert charged.nonce == 1

    reconciled = economy.reconcile_state_channel("ch-1", total_spent=2.0, nonce=2)
    assert reconciled.spent == 2.0
    assert reconciled.nonce == 2

    close_payload = economy.close_state_channel("ch-1")
    assert close_payload["payee_amount"] == 2.0
    assert close_payload["payer_refund"] == 4.0

    assert economy.account_snapshot("alice")["balance"] == 8.0
    assert economy.account_snapshot("bob")["balance"] == 2.0


def test_file_hydra_economy_persists_balances_and_channels(tmp_path):
    path = tmp_path / "hydra_tokens.json"

    store_a = FileHydraTokenEconomy(str(path))
    store_a.mint_for_inference("peer-a", tokens_served=2000, reward_per_1k_tokens=1.0)
    store_a.open_state_channel(channel_id="stream-1", payer="peer-a", payee="peer-b", deposit=1.0)
    store_a.charge_state_channel(channel_id="stream-1", amount=0.25)

    store_b = FileHydraTokenEconomy(str(path))
    snap_a = store_b.account_snapshot("peer-a")
    snap_b = store_b.account_snapshot("peer-b")
    summary = store_b.summary()

    assert snap_a["balance"] == 1.0
    assert snap_b["balance"] == 0.0
    assert snap_a["open_channels"] == ["stream-1"]
    assert summary["channels_open"] == 1
    assert summary["locked_in_channels"] == 1.0


def test_hydra_channel_limit_and_min_deposit_enforced():
    economy = HydraTokenEconomy(
        channel_max_open_per_payer=1,
        channel_min_deposit=0.5,
    )
    economy.mint("alice", 5.0)

    with pytest.raises(RuntimeError, match="hydra_channel_deposit_too_small"):
        economy.open_state_channel("small", "alice", "bob", 0.1)

    economy.open_state_channel("c1", "alice", "bob", 1.0)
    with pytest.raises(RuntimeError, match="hydra_channel_limit_exceeded"):
        economy.open_state_channel("c2", "alice", "carol", 1.0)


def test_hydra_channel_auto_expiry_settles_and_closes():
    now_ts = {"value": 10.0}
    economy = HydraTokenEconomy(
        channel_default_ttl_seconds=5,
        now_fn=lambda: now_ts["value"],
    )
    economy.mint("alice", 3.0)
    economy.open_state_channel("exp-1", "alice", "bob", 2.0)
    economy.charge_state_channel("exp-1", 1.0)

    now_ts["value"] = 20.0
    summary = economy.summary()
    assert summary["channels_open"] == 0
    assert summary["auto_closed_channels"] == 1

    snap_alice = economy.account_snapshot("alice")
    snap_bob = economy.account_snapshot("bob")
    assert snap_alice["balance"] == 2.0
    assert snap_bob["balance"] == 1.0

    with pytest.raises(RuntimeError, match="hydra_channel_closed"):
        economy.charge_state_channel("exp-1", 0.1)
