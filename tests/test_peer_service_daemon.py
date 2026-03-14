import pytest

from peer.daemon_monitor import ResourceBudget
from peer.server import PeerService


def _service() -> PeerService:
    return PeerService(
        peer_id="peer-a",
        model_id="openhydra-toy-345m",
        shard_index=0,
        total_shards=3,
        daemon_mode="polite",
        broken=False,
    )


def test_peer_service_default_load_has_no_budget_pressure():
    service = _service()
    assert service._load_pct() == 0.0


def test_peer_service_yield_budget_reports_high_load():
    service = _service()
    service.set_resource_budget(
        ResourceBudget(
            vram_fraction=0.0,
            cpu_fraction=0.0,
            should_yield=True,
            reason="user-active",
        )
    )

    assert service._load_pct() == 95.0


def test_peer_service_cpu_budget_adds_load_pressure():
    service = _service()
    service.set_resource_budget(
        ResourceBudget(
            vram_fraction=0.5,
            cpu_fraction=0.5,
            should_yield=False,
            reason="test",
        )
    )

    assert service._load_pct() == pytest.approx(7.5, abs=1e-6)
