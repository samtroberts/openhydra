from verification.auditor import AuditSampler, should_audit


def test_audit_sampler_advances_sequence():
    sampler = AuditSampler(sample_rate=0.5, seed=1)
    assert sampler.should_sample() is True
    assert sampler.should_sample() is False


def test_should_audit_helper_clamps_rate():
    assert should_audit(sample_rate=-1.0, seed=1) is False
    assert should_audit(sample_rate=2.0, seed=1) is True
