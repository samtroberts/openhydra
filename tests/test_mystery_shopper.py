from coordinator.chain import ChainResult, StageTrace
from coordinator.mystery_shopper import MysteryShopper


def _result(text: str) -> ChainResult:
    return ChainResult(
        request_id="r1",
        text=text,
        activation=[0.1],
        traces=[StageTrace(peer_id="p", latency_ms=1.0, stage_index=0)],
        latency_ms=1.0,
    )


def test_mystery_shopper_match():
    shopper = MysteryShopper(sample_rate=1.0, seed=1, mode="redundant_execution")
    primary = _result("same")
    verification = shopper.verify(primary, run_secondary=lambda: _result("same"))
    assert verification.audited
    assert verification.match
    assert verification.winner == "primary"
    assert verification.mode == "redundant_execution"
    assert verification.sample_rate == 1.0
    assert verification.auditor_triggered is False


def test_mystery_shopper_mismatch_uses_tertiary_tiebreaker():
    shopper = MysteryShopper(sample_rate=1.0, seed=1)
    primary = _result("alpha")
    verification = shopper.verify(
        primary,
        run_secondary=lambda: _result("beta"),
        run_tertiary=lambda: _result("alpha"),
    )
    assert verification.audited
    assert not verification.match
    assert verification.winner == "primary"
    assert verification.mode == "mystery_shopper"
    assert verification.auditor_triggered is False


def test_mystery_shopper_auditor_spotcheck_runs_tertiary():
    shopper = MysteryShopper(sample_rate=1.0, auditor_sample_rate=1.0, seed=1)
    verification = shopper.verify(
        _result("same"),
        run_secondary=lambda: _result("same"),
        run_tertiary=lambda: _result("different"),
    )
    assert verification.audited is True
    assert verification.auditor_triggered is True
    assert verification.tertiary_text == "different"
    assert verification.winner == "primary"
    assert verification.match is False


def test_mystery_shopper_rng_advances_across_calls():
    shopper = MysteryShopper(sample_rate=0.5, seed=1)
    assert shopper.should_audit() is True
    assert shopper.should_audit() is False
