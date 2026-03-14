from torrent.session import SessionBootstrapConfig, TorrentSessionManager
from torrent.seeder import ArbitrationConfig


def test_torrent_session_bootstrap_and_snapshot(tmp_path):
    manager = TorrentSessionManager(
        bootstrap=SessionBootstrapConfig(
            model_id="session-model",
            cache_dir=str(tmp_path / "cache"),
        ),
        arbitration=ArbitrationConfig(base_upload_mbps=50.0, inference_seed_fraction=0.1, smoothing_alpha=1.0),
    )

    result = manager.bootstrap()
    snap = manager.snapshot()

    assert result.model_id == "session-model"
    assert snap["seeding_enabled"] is True
    assert snap["genesis"]["artifact_path"].endswith("session-model.safetensors")
    assert snap["arbitration"]["effective_seed_upload_limit_mbps"] == 50.0


def test_torrent_session_updates_inference_policy(tmp_path):
    manager = TorrentSessionManager(
        bootstrap=SessionBootstrapConfig(
            model_id="session-model",
            cache_dir=str(tmp_path / "cache"),
        ),
        arbitration=ArbitrationConfig(base_upload_mbps=100.0, inference_seed_fraction=0.1, smoothing_alpha=1.0),
    )
    manager.bootstrap()

    updated = manager.update(inference_active=True, inference_observed_mbps=12.0)
    assert updated["arbitration"]["inference_active"] is True
    assert updated["arbitration"]["effective_seed_upload_limit_mbps"] == 10.0
