"""Tests for PeerEndpoint.from_dict() and .replace() — v0.1.1."""

from coordinator.path_finder import PeerEndpoint
from coordinator.peer_utils import normalize_tags, normalize_layer_indices


class TestFromDict:
    def test_minimal(self):
        p = PeerEndpoint.from_dict({"peer_id": "p1", "host": "10.0.0.1", "port": 5001})
        assert p.peer_id == "p1"
        assert p.host == "10.0.0.1"
        assert p.port == 5001
        assert p.model_id is None
        assert p.layer_start == 0
        assert p.seeder_http_port == 0
        assert p.public_key_hex == ""
        assert p.cached_model_ids == ()

    def test_all_fields_round_trip(self):
        data = {
            "peer_id": "peer-full",
            "host": "192.168.1.1",
            "port": 50051,
            "model_id": "Qwen/Qwen3.5-0.8B",
            "operator_id": "op1",
            "region": "us-east",
            "bandwidth_mbps": 100.0,
            "seeding_enabled": True,
            "seed_upload_limit_mbps": 25.0,
            "seed_target_upload_limit_mbps": 30.0,
            "seed_inference_active": True,
            "runtime_backend": "mlx",
            "runtime_target": "mps",
            "runtime_model_id": "Qwen/Qwen3.5-0.8B",
            "quantization_mode": "int4",
            "quantization_bits": 4,
            "runtime_gpu_available": True,
            "runtime_estimated_tokens_per_sec": 252.0,
            "runtime_estimated_memory_mb": 1024,
            "privacy_noise_variance": 0.01,
            "privacy_noise_payloads": 5,
            "privacy_noise_observed_variance_ema": 0.009,
            "privacy_noise_last_audit_tag": "abc123",
            "reputation_score": 85.0,
            "staked_balance": 1000.0,
            "expert_tags": ["math", "code"],
            "expert_layer_indices": [0, 1, 2],
            "expert_router": True,
            "expert_admission_approved": True,
            "expert_admission_reason": "approved",
            "geo_verified": True,
            "geo_challenge_rtt_ms": 42.5,
            "geo_penalty_score": 0.1,
            "public_key_hex": "deadbeef1234",
            "available_vram_mb": 8192,
            "layer_start": 0,
            "layer_end": 16,
            "total_layers": 32,
            "seeder_http_port": 9000,
            "cached_model_ids": ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B"],
            "local_fast_path_port": 8081,
        }
        p = PeerEndpoint.from_dict(data)
        assert p.peer_id == "peer-full"
        assert p.model_id == "Qwen/Qwen3.5-0.8B"
        assert p.layer_start == 0
        assert p.layer_end == 16
        assert p.total_layers == 32
        assert p.seeder_http_port == 9000
        assert p.cached_model_ids == ("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B")
        assert p.local_fast_path_port == 8081
        assert p.geo_verified is True
        assert p.geo_challenge_rtt_ms == 42.5
        assert p.public_key_hex == "deadbeef1234"
        assert p.available_vram_mb == 8192
        assert p.expert_tags == ("math", "code")
        assert p.expert_layer_indices == (0, 1, 2)
        assert p.expert_router is True
        assert p.reputation_score == 85.0
        assert p.runtime_estimated_tokens_per_sec == 252.0

    def test_dht_key_alias_peer_public_key(self):
        """DHT records use 'peer_public_key', not 'public_key_hex'."""
        p = PeerEndpoint.from_dict({
            "peer_id": "p1", "host": "10.0.0.1", "port": 5001,
            "peer_public_key": "abcdef",
        })
        assert p.public_key_hex == "abcdef"

    def test_public_key_hex_preferred_over_dht_alias(self):
        p = PeerEndpoint.from_dict({
            "peer_id": "p1", "host": "10.0.0.1", "port": 5001,
            "public_key_hex": "preferred",
            "peer_public_key": "dht_fallback",
        })
        assert p.public_key_hex == "preferred"

    def test_admission_gating_clears_expert_fields(self):
        p = PeerEndpoint.from_dict({
            "peer_id": "p1", "host": "10.0.0.1", "port": 5001,
            "expert_admission_approved": False,
            "expert_tags": ["math", "code"],
            "expert_layer_indices": [0, 1],
            "expert_router": True,
        })
        assert p.expert_admission_approved is False
        assert p.expert_tags == ()
        assert p.expert_layer_indices == ()
        assert p.expert_router is False

    def test_unknown_keys_ignored(self):
        p = PeerEndpoint.from_dict({
            "peer_id": "p1", "host": "10.0.0.1", "port": 5001,
            "unknown_field": "should_not_crash",
            "another_future_field": 42,
        })
        assert p.peer_id == "p1"

    def test_geo_rtt_none_when_absent(self):
        p = PeerEndpoint.from_dict({"peer_id": "p1", "host": "10.0.0.1", "port": 5001})
        assert p.geo_challenge_rtt_ms is None


class TestReplace:
    def test_replace_model_id_preserves_all_fields(self):
        original = PeerEndpoint(
            peer_id="shard-peer",
            host="10.0.0.1",
            port=5001,
            model_id="old-model",
            layer_start=0,
            layer_end=16,
            total_layers=32,
            seeder_http_port=9000,
            cached_model_ids=("Qwen/Qwen3.5-0.8B",),
            public_key_hex="deadbeef",
            geo_verified=True,
            available_vram_mb=8192,
            local_fast_path_port=8081,
        )
        replaced = original.replace(model_id="new-model")
        assert replaced.model_id == "new-model"
        assert replaced.peer_id == "shard-peer"
        assert replaced.layer_start == 0
        assert replaced.layer_end == 16
        assert replaced.total_layers == 32
        assert replaced.seeder_http_port == 9000
        assert replaced.cached_model_ids == ("Qwen/Qwen3.5-0.8B",)
        assert replaced.public_key_hex == "deadbeef"
        assert replaced.geo_verified is True
        assert replaced.available_vram_mb == 8192
        assert replaced.local_fast_path_port == 8081

    def test_replace_returns_new_instance(self):
        original = PeerEndpoint(peer_id="p1", host="10.0.0.1", port=5001)
        replaced = original.replace(host="10.0.0.2")
        assert original.host == "10.0.0.1"
        assert replaced.host == "10.0.0.2"
        assert original is not replaced


class TestNormalizeUtils:
    def test_normalize_tags_from_string(self):
        assert normalize_tags("Math, CODE, math") == ("math", "code")

    def test_normalize_tags_from_list(self):
        assert normalize_tags(["Math", "Code", "math"]) == ("math", "code")

    def test_normalize_tags_none(self):
        assert normalize_tags(None) == ()

    def test_normalize_layer_indices_sorted(self):
        assert normalize_layer_indices("3,1,2,1") == (1, 2, 3)

    def test_normalize_layer_indices_rejects_negative(self):
        assert normalize_layer_indices([-1, 0, 5]) == (0, 5)

    def test_normalize_layer_indices_none(self):
        assert normalize_layer_indices(None) == ()
