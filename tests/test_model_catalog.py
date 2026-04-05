"""Tests for 6.3 — Model Catalog Expansion.

Verifies that models.catalog.json has the extended schema and that
coordinator/engine.py correctly parses and exposes the new fields.
"""
from __future__ import annotations

import json
import pathlib
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


REPO_ROOT = pathlib.Path(__file__).parent.parent
CATALOG_PATH = REPO_ROOT / "models.catalog.json"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_catalog() -> list[dict]:
    return json.loads(CATALOG_PATH.read_text())


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestModelCatalogJson:
    """Tests for the models.catalog.json file itself."""

    def test_catalog_loads_and_has_enough_entries(self):
        """Catalog loads without error and has exactly 5 Qwen 3.5 entries."""
        catalog = _load_catalog()
        assert isinstance(catalog, list)
        assert len(catalog) >= 5, f"Expected 5 entries, got {len(catalog)}"

    def test_all_entries_have_required_fields(self):
        """Every entry has model_id, required_peers, and hf_model_id."""
        catalog = _load_catalog()
        for entry in catalog:
            assert "model_id" in entry, f"Missing model_id: {entry}"
            assert "required_peers" in entry, f"Missing required_peers: {entry}"
            assert "hf_model_id" in entry, f"Missing hf_model_id: {entry}"
            assert isinstance(entry["model_id"], str) and entry["model_id"]
            assert isinstance(entry["required_peers"], int) and entry["required_peers"] >= 1
            assert isinstance(entry["hf_model_id"], str) and entry["hf_model_id"]

    def test_extended_fields_present_in_at_least_one_entry(self):
        """At least one entry has min_vram_gb, recommended_quantization, context_length, tags, description."""
        catalog = _load_catalog()
        fields = ["min_vram_gb", "recommended_quantization", "context_length", "tags", "description"]
        for field in fields:
            has_field = any(field in entry for entry in catalog)
            assert has_field, f"No entry has field '{field}'"

    def test_multi_peer_models_exist(self):
        """At least one model requires more than 1 peer."""
        catalog = _load_catalog()
        multi = [e for e in catalog if e.get("required_peers", 1) > 1]
        assert multi, "Expected at least one model with required_peers > 1"

    def test_largest_model_has_4_peers(self):
        """The frontier 27B model requires 4 peers."""
        catalog = _load_catalog()
        large = [e for e in catalog if e.get("required_peers", 1) >= 4]
        assert large, "Expected at least one model with required_peers >= 4"
        assert large[0]["model_id"] == "openhydra-qwen3.5-27b"

    def test_all_models_have_valid_ids(self):
        """All catalog models have openhydra- prefix."""
        catalog = _load_catalog()
        for entry in catalog:
            assert entry["model_id"].startswith("openhydra-"), f"Invalid model_id: {entry['model_id']}"


class TestModelAvailabilityDataclass:
    """Tests for the extended ModelAvailability dataclass."""

    def test_model_availability_has_new_fields(self):
        """ModelAvailability accepts the 6 new extended fields."""
        from coordinator.degradation import ModelAvailability
        m = ModelAvailability(
            model_id="test-model",
            required_peers=1,
            hf_model_id="org/repo",
            min_vram_gb=8,
            recommended_quantization="int8",
            context_length=32768,
            languages=("en", "multilingual"),
            tags=("chat", "medium"),
            description="A test model.",
        )
        assert m.min_vram_gb == 8
        assert m.recommended_quantization == "int8"
        assert m.context_length == 32768
        assert m.languages == ("en", "multilingual")
        assert m.tags == ("chat", "medium")
        assert m.description == "A test model."

    def test_model_availability_default_values(self):
        """ModelAvailability new fields have sensible defaults."""
        from coordinator.degradation import ModelAvailability
        m = ModelAvailability(model_id="x", required_peers=1)
        assert m.min_vram_gb == 0
        assert m.recommended_quantization == "fp32"
        assert m.context_length == 4096
        assert m.languages == ()
        assert m.tags == ()
        assert m.description == ""


class TestLoadModelCatalog:
    """Tests for coordinator/engine._load_model_catalog()."""

    def _make_engine_with_catalog(self, catalog_data: list[dict]):
        """Return a minimal CoordinatorEngine with a temp catalog file."""
        from coordinator.engine import CoordinatorEngine, EngineConfig
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(catalog_data, f)
            catalog_path = f.name

        cfg = EngineConfig(
            peers_config_path="peers.local.json",
            model_catalog_path=catalog_path,
            default_model="openhydra-qwen3.5-0.8b",
        )
        engine = CoordinatorEngine.__new__(CoordinatorEngine)
        engine.config = cfg
        return engine

    def test_load_catalog_parses_new_fields(self):
        """_load_model_catalog() populates ModelAvailability new fields."""
        from coordinator.engine import CoordinatorEngine, EngineConfig
        catalog_data = [
            {
                "model_id": "openhydra-test-model",
                "required_peers": 2,
                "hf_model_id": "org/test-model",
                "min_vram_gb": 12,
                "recommended_quantization": "int8",
                "context_length": 16384,
                "languages": ["en", "multilingual"],
                "tags": ["chat", "medium"],
                "description": "A test model for unit tests.",
            }
        ]
        engine = self._make_engine_with_catalog(catalog_data)
        catalogue = engine._load_model_catalog()
        # Find the test model
        model = next((m for m in catalogue if m.model_id == "openhydra-test-model"), None)
        assert model is not None
        assert model.min_vram_gb == 12
        assert model.recommended_quantization == "int8"
        assert model.context_length == 16384
        assert model.languages == ("en", "multilingual")
        assert model.tags == ("chat", "medium")
        assert model.description == "A test model for unit tests."

    def test_load_real_catalog_parses_all_entries(self):
        """_load_model_catalog() can parse the real models.catalog.json."""
        from coordinator.engine import CoordinatorEngine, EngineConfig
        cfg = EngineConfig(
            peers_config_path="peers.local.json",
            model_catalog_path=str(CATALOG_PATH),
            default_model="openhydra-qwen3.5-0.8b",
        )
        engine = CoordinatorEngine.__new__(CoordinatorEngine)
        engine.config = cfg
        catalogue = engine._load_model_catalog()
        assert len(catalogue) >= 5
        # Every entry should be a ModelAvailability with the new fields
        from coordinator.degradation import ModelAvailability
        for item in catalogue:
            assert isinstance(item, ModelAvailability)
            assert hasattr(item, "min_vram_gb")
            assert hasattr(item, "tags")
            assert hasattr(item, "description")


class TestListModelsOutput:
    """Tests for list_models() returning the new fields."""

    def _make_minimal_engine(self, catalog_data: list[dict]):
        """Return a partial CoordinatorEngine for testing list_models()."""
        from coordinator.engine import CoordinatorEngine, EngineConfig
        from coordinator.degradation import DegradationPolicy, ModelAvailability

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(catalog_data, f)
            catalog_path = f.name

        cfg = EngineConfig(
            peers_config_path="peers.local.json",
            model_catalog_path=catalog_path,
            default_model="openhydra-test",
        )
        import threading

        engine = CoordinatorEngine.__new__(CoordinatorEngine)
        engine.config = cfg
        engine.model_catalog = engine._load_model_catalog()
        engine.catalogue_by_model = {m.model_id: m for m in engine.model_catalog}
        engine.degradation_policy = DegradationPolicy(engine.model_catalog)
        # list_models() reads from the DHT peer cache — initialise the
        # two attributes that __init__ would normally set.
        engine._dht_peer_cache: dict = {}
        engine._dht_peer_cache_lock = threading.Lock()
        # Stub out replication_monitor so list_models() doesn't need full init
        repl_mock = MagicMock()
        repl_mock.evaluate.return_value = MagicMock()
        repl_mock.to_dict.return_value = {"under_replicated": False}
        engine.replication_monitor = repl_mock
        return engine

    def test_list_models_includes_new_fields(self):
        """list_models() returns min_vram_gb, recommended_quantization, context_length, languages, tags, description."""
        catalog_data = [
            {
                "model_id": "openhydra-test",
                "required_peers": 1,
                "hf_model_id": "org/test",
                "min_vram_gb": 4,
                "recommended_quantization": "fp32",
                "context_length": 8192,
                "languages": ["en"],
                "tags": ["chat", "small"],
                "description": "A test model.",
            }
        ]
        engine = self._make_minimal_engine(catalog_data)

        # list_models() no longer triggers a DHT scan — it reads the static
        # catalog and the in-memory peer cache, so no patching is required.
        result = engine.list_models()

        assert result["object"] == "list"
        assert result["data"]
        model_out = result["data"][0]
        assert model_out["min_vram_gb"] == 4
        assert model_out["recommended_quantization"] == "fp32"
        assert model_out["context_length"] == 8192
        assert model_out["languages"] == ["en"]
        assert model_out["tags"] == ["chat", "small"]
        assert model_out["description"] == "A test model."
