from torrent.seeder import ArbitrationConfig, BandwidthArbitrator


def test_arbitrator_throttles_seeding_when_inference_active():
    arb = BandwidthArbitrator(ArbitrationConfig(base_upload_mbps=100.0, inference_seed_fraction=0.10, smoothing_alpha=1.0))

    state = arb.update(inference_active=True)
    assert round(state.effective_upload_limit(), 4) == 10.0


def test_arbitrator_restores_seeding_when_idle():
    arb = BandwidthArbitrator(ArbitrationConfig(base_upload_mbps=120.0, inference_seed_fraction=0.10, smoothing_alpha=1.0))

    arb.update(inference_active=True)
    state = arb.update(inference_active=False)
    assert round(state.effective_upload_limit(), 4) == 120.0


def test_arbitrator_smoothing_transitions():
    arb = BandwidthArbitrator(ArbitrationConfig(base_upload_mbps=100.0, inference_seed_fraction=0.10, smoothing_alpha=0.5))

    first = arb.update(inference_active=True).effective_upload_limit()
    second = arb.update(inference_active=True).effective_upload_limit()

    assert first > 10.0
    assert second > 10.0
    assert second < first
