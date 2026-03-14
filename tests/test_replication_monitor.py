from coordinator.replication_monitor import ReplicationMonitor


def test_replication_monitor_flags_under_replicated():
    monitor = ReplicationMonitor(required_replicas=3)
    status = monitor.evaluate("openhydra-toy-345m", healthy_peers=1)
    assert status.under_replicated
    assert status.deficit == 2


def test_replication_monitor_healthy_when_threshold_met():
    monitor = ReplicationMonitor(required_replicas=3)
    status = monitor.evaluate("openhydra-toy-345m", healthy_peers=3)
    assert not status.under_replicated
    assert status.deficit == 0
