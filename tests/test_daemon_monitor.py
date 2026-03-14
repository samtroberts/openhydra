from peer.daemon_monitor import (
    DaemonController,
    DaemonMode,
    MonitorConfig,
    RuntimeSignals,
    compute_resource_budget,
    get_resource_budget,
)


def test_get_resource_budget_polite_compat():
    active = get_resource_budget(DaemonMode.POLITE, user_idle=False)
    idle = get_resource_budget(DaemonMode.POLITE, user_idle=True)

    assert active.should_yield is True
    assert active.vram_fraction == 0.0
    assert idle.should_yield is False
    assert idle.vram_fraction == 0.5


def test_compute_resource_budget_polite_user_active():
    budget = compute_resource_budget(
        MonitorConfig(mode=DaemonMode.POLITE, idle_threshold_sec=300.0),
        RuntimeSignals(user_idle_seconds=120.0, cpu_load=0.1, fullscreen_active=False),
    )

    assert budget.should_yield is True
    assert budget.reason == "user-active"
    assert budget.cpu_fraction == 0.0


def test_compute_resource_budget_power_user_high_load():
    budget = compute_resource_budget(
        MonitorConfig(mode=DaemonMode.POWER_USER, high_load_threshold=0.85),
        RuntimeSignals(user_idle_seconds=None, cpu_load=0.9, fullscreen_active=False),
    )

    assert budget.should_yield is False
    assert budget.reason == "high-system-load"
    assert budget.vram_fraction == 0.5
    assert budget.cpu_fraction == 0.35


def test_daemon_controller_refresh_uses_probes():
    controller = DaemonController(
        MonitorConfig(mode=DaemonMode.POLITE, idle_threshold_sec=60.0),
        idle_probe=lambda: 10.0,
        cpu_load_probe=lambda: 0.2,
        fullscreen_probe=lambda: False,
    )
    budget = controller.refresh()

    assert budget.should_yield is True
    assert budget.reason == "user-active"
    assert controller.current_budget().reason == "user-active"
