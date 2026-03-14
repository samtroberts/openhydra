from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from coordinator.api_server import _resolve_runtime_profile_settings
from dht.bootstrap import _resolve_bootstrap_profile_settings
from peer.server import _resolve_deployment_security_settings


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(prog="test")


def test_api_dev_profile_keeps_mock_ledger_default():
    args = SimpleNamespace(
        deployment_profile="dev",
        secrets_file=None,
        hydra_ledger_bridge_mock_mode=None,
        advanced_encryption_seed="openhydra-tier3-dev-seed",
        tls_enable=False,
        tls_root_cert_path=None,
        tls_client_cert_path=None,
        tls_client_key_path=None,
        tls_server_name_override=None,
    )
    settings = _resolve_runtime_profile_settings(_parser(), args)
    assert settings["deployment_profile"] == "dev"
    assert settings["hydra_ledger_bridge_mock_mode"] is True


def test_api_prod_profile_forces_live_ledger_and_tls(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENHYDRA_ADVANCED_ENCRYPTION_SEED", "prod-advanced-seed")
    args = SimpleNamespace(
        deployment_profile="prod",
        secrets_file=None,
        hydra_ledger_bridge_mock_mode=None,
        advanced_encryption_seed="openhydra-tier3-dev-seed",
        tls_enable=True,
        tls_root_cert_path="/certs/ca.pem",
        tls_client_cert_path="/certs/client.pem",
        tls_client_key_path="/certs/client.key",
        tls_server_name_override="peer.internal",
    )
    settings = _resolve_runtime_profile_settings(_parser(), args)
    assert settings["hydra_ledger_bridge_mock_mode"] is False
    assert settings["advanced_encryption_seed"] == "prod-advanced-seed"


def test_api_prod_profile_rejects_mock_ledger():
    args = SimpleNamespace(
        deployment_profile="prod",
        secrets_file=None,
        hydra_ledger_bridge_mock_mode=True,
        advanced_encryption_seed="prod-seed",
        tls_enable=True,
        tls_root_cert_path="/certs/ca.pem",
        tls_client_cert_path="/certs/client.pem",
        tls_client_key_path="/certs/client.key",
        tls_server_name_override="peer.internal",
    )
    with pytest.raises(SystemExit):
        _resolve_runtime_profile_settings(_parser(), args)


def test_peer_prod_profile_requires_mtls_and_secrets(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENHYDRA_ADVANCED_ENCRYPTION_SEED", "prod-advanced-seed")
    monkeypatch.setenv("OPENHYDRA_GEO_CHALLENGE_SEED", "prod-geo-seed")
    args = SimpleNamespace(
        deployment_profile="prod",
        secrets_file=None,
        advanced_encryption_seed="openhydra-tier3-dev-seed",
        geo_challenge_seed="openhydra-geo-dev-seed",
        tls_enable=True,
        tls_require_client_auth=True,
        tls_cert_path="/certs/peer.pem",
        tls_key_path="/certs/peer.key",
        tls_client_ca_path="/certs/ca.pem",
    )
    settings = _resolve_deployment_security_settings(_parser(), args)
    assert settings["advanced_encryption_seed"] == "prod-advanced-seed"
    assert settings["geo_challenge_seed"] == "prod-geo-seed"


def test_peer_prod_profile_rejects_non_mtls():
    args = SimpleNamespace(
        deployment_profile="prod",
        secrets_file=None,
        advanced_encryption_seed="prod-advanced",
        geo_challenge_seed="prod-geo",
        tls_enable=True,
        tls_require_client_auth=False,
        tls_cert_path="/certs/peer.pem",
        tls_key_path="/certs/peer.key",
        tls_client_ca_path="/certs/ca.pem",
    )
    with pytest.raises(SystemExit):
        _resolve_deployment_security_settings(_parser(), args)


def test_bootstrap_prod_profile_requires_geo_seed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENHYDRA_GEO_CHALLENGE_SEED", "prod-geo-seed")
    args = SimpleNamespace(
        deployment_profile="prod",
        secrets_file=None,
        geo_challenge_seed="openhydra-geo-dev-seed",
        geo_challenge_enabled=True,
    )
    settings = _resolve_bootstrap_profile_settings(_parser(), args)
    assert settings["geo_challenge_seed"] == "prod-geo-seed"


def test_bootstrap_prod_profile_rejects_disabled_geo_challenge():
    args = SimpleNamespace(
        deployment_profile="prod",
        secrets_file=None,
        geo_challenge_seed="prod-geo-seed",
        geo_challenge_enabled=False,
    )
    with pytest.raises(SystemExit):
        _resolve_bootstrap_profile_settings(_parser(), args)
