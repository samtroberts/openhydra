from coordinator.degradation import DegradationPolicy, ModelAvailability


def _policy() -> DegradationPolicy:
    # Ordered largest -> smallest for fallback direction.
    return DegradationPolicy(
        [
            ModelAvailability(model_id="openhydra-70b", required_peers=8),
            ModelAvailability(model_id="openhydra-8b", required_peers=3),
            ModelAvailability(model_id="openhydra-toy-345m", required_peers=1),
        ]
    )


def test_degradation_falls_back_to_smaller_model():
    policy = _policy()
    decision = policy.select(
        requested_model="openhydra-70b",
        available_peer_counts={"openhydra-70b": 2, "openhydra-8b": 3},
        allow_degradation=True,
    )
    assert decision.degraded is True
    assert decision.available is True
    assert decision.served_model == "openhydra-8b"
    assert decision.reason == "insufficient_peers"


def test_degradation_disabled_returns_insufficient_peers():
    policy = _policy()
    decision = policy.select(
        requested_model="openhydra-70b",
        available_peer_counts={"openhydra-70b": 2, "openhydra-8b": 3},
        allow_degradation=False,
    )
    assert decision.degraded is False
    assert decision.available is False
    assert decision.served_model == "openhydra-70b"
    assert decision.reason == "insufficient_peers"


def test_degradation_unknown_model():
    policy = _policy()
    decision = policy.select(
        requested_model="unknown-model",
        available_peer_counts={},
        allow_degradation=True,
    )
    assert decision.available is True
    assert decision.reason == "unknown_model"


def test_degradation_no_viable_fallback_marks_unavailable():
    policy = _policy()
    decision = policy.select(
        requested_model="openhydra-70b",
        available_peer_counts={"openhydra-70b": 1, "openhydra-8b": 2, "openhydra-toy-345m": 0},
        allow_degradation=True,
    )

    assert decision.degraded is False
    assert decision.available is False
    assert decision.reason == "no_viable_fallback"
