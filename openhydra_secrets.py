from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

_ALLOWED_ENV_KEYS: frozenset[str] = frozenset(
    {
        "OPENHYDRA_ADVANCED_ENCRYPTION_SEED",
        "OPENHYDRA_GEO_CHALLENGE_SEED",
        "OPENHYDRA_DEPLOYMENT_PROFILE",
        "OPENHYDRA_SECRETS_FILE",
        "OPENHYDRA_DHT_BOOTSTRAP_URLS",
        "OPENHYDRA_HYDRA_LEDGER_PATH",
        "OPENHYDRA_TLS_CERT_PATH",
        "OPENHYDRA_TLS_KEY_PATH",
        "OPENHYDRA_TLS_CA_PATH",
    }
)


@dataclass(frozen=True)
class SecretStore:
    values: dict[str, str]
    source_path: str | None = None

    def get(self, key: str, default: str | None = None) -> str | None:
        value = self.values.get(str(key))
        if value is None:
            return default
        value = str(value).strip()
        if not value:
            return default
        return value

    def require(self, key: str) -> str:
        value = self.get(key)
        if value is None:
            raise RuntimeError(f"missing_required_secret:{key}")
        return value


def _parse_secrets_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        out[key] = value.strip()
    return out


def _validate_secrets_file_permissions(path: Path) -> None:
    stat = path.stat()
    # Reject group/other readable/writable/executable bits for production secrets.
    if (stat.st_mode & 0o077) != 0:
        raise RuntimeError("insecure_secrets_file_permissions")


def load_secret_store(secrets_file: str | None) -> SecretStore:
    values: dict[str, str] = {}
    source_path: str | None = None
    if secrets_file:
        path = Path(str(secrets_file)).expanduser().resolve()
        if not path.exists():
            raise RuntimeError(f"secrets_file_not_found:{path}")
        _validate_secrets_file_permissions(path)
        values.update(_parse_secrets_file(path))
        source_path = str(path)
    # Environment overrides file values to support secure runtime injection.
    for key in _ALLOWED_ENV_KEYS:
        value = os.environ.get(key)
        if value is None:
            continue
        values[key] = str(value)
    return SecretStore(values=values, source_path=source_path)


def is_insecure_secret_value(value: str | None) -> bool:
    candidate = str(value or "").strip().lower()
    if not candidate:
        return True
    if candidate in {"changeme", "replace-me", "default", "dev", "test"}:
        return True
    if "openhydra-tier3-dev-seed" in candidate:
        return True
    if "openhydra-geo-dev-seed" in candidate:
        return True
    return False
