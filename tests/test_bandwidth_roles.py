from coordinator.bandwidth_roles import classify_role, estimate_prompt_tokens, role_counts_from_bandwidth


def test_classify_role_thresholds():
    assert classify_role(700.0) == "prefill_capable"
    assert classify_role(50.0) == "decode_only"
    assert classify_role(120.0) == "balanced"


def test_estimate_prompt_tokens_non_empty():
    assert estimate_prompt_tokens("") == 1
    assert estimate_prompt_tokens("hello world from hydra") == 4


def test_role_counts_from_bandwidth():
    counts = role_counts_from_bandwidth([700.0, 120.0, 20.0])
    assert counts == {"prefill_capable": 1, "balanced": 1, "decode_only": 1}
