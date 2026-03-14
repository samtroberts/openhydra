from __future__ import annotations

from pathlib import Path

import pytest

from openhydra_secrets import is_insecure_secret_value, load_secret_store


def test_load_secret_store_reads_file_and_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    secrets_file = tmp_path / "secrets.env"
    secrets_file.write_text(
        "OPENHYDRA_ADVANCED_ENCRYPTION_SEED=file-seed\n"
        "OPENHYDRA_GEO_CHALLENGE_SEED=file-geo\n",
        encoding="utf-8",
    )
    secrets_file.chmod(0o600)

    monkeypatch.setenv("OPENHYDRA_ADVANCED_ENCRYPTION_SEED", "env-seed")
    store = load_secret_store(str(secrets_file))
    assert store.get("OPENHYDRA_ADVANCED_ENCRYPTION_SEED") == "env-seed"
    assert store.get("OPENHYDRA_GEO_CHALLENGE_SEED") == "file-geo"


def test_load_secret_store_filters_non_allowlisted_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENHYDRA_ADVANCED_ENCRYPTION_SEED", "env-seed")
    monkeypatch.setenv("PATH", "/bin:/usr/bin")
    monkeypatch.setenv("HOME", "/Users/sam")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "should-not-leak")

    store = load_secret_store(None)
    assert store.get("OPENHYDRA_ADVANCED_ENCRYPTION_SEED") == "env-seed"
    assert store.get("PATH") is None
    assert store.get("HOME") is None
    assert store.get("AWS_SECRET_ACCESS_KEY") is None


def test_load_secret_store_rejects_insecure_permissions(tmp_path: Path):
    secrets_file = tmp_path / "insecure.env"
    secrets_file.write_text("OPENHYDRA_ADVANCED_ENCRYPTION_SEED=x\n", encoding="utf-8")
    secrets_file.chmod(0o644)

    with pytest.raises(RuntimeError, match="insecure_secrets_file_permissions"):
        load_secret_store(str(secrets_file))


def test_is_insecure_secret_value_catches_dev_defaults():
    assert is_insecure_secret_value("")
    assert is_insecure_secret_value("changeme")
    assert is_insecure_secret_value("openhydra-tier3-dev-seed")
    assert is_insecure_secret_value("openhydra-geo-dev-seed")
    assert not is_insecure_secret_value("prod-entropy-seed-123")
