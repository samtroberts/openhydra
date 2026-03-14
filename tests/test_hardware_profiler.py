from peer import hardware


def test_hardware_profiler_cpu_profile(monkeypatch):
    monkeypatch.setattr(hardware, "_system_ram_bytes", lambda: (16_000_000_000, 8_000_000_000))
    monkeypatch.setattr(hardware, "_load_torch_module", lambda: None)

    profile = hardware.detect_hardware_profile()

    assert profile.ram_total_bytes == 16_000_000_000
    assert profile.ram_available_bytes == 8_000_000_000
    assert profile.accelerator == "cpu"
    assert profile.vram_total_bytes is None
    assert profile.vram_available_bytes is None
    assert profile.cuda_device_count == 0


def test_hardware_profiler_cuda_profile(monkeypatch):
    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 2

        @staticmethod
        def mem_get_info() -> tuple[int, int]:
            return (2_000_000_000, 12_000_000_000)

    class _FakeMps:
        @staticmethod
        def is_available() -> bool:
            return False

    class _FakeBackends:
        mps = _FakeMps()

    class _FakeTorch:
        cuda = _FakeCuda()
        backends = _FakeBackends()

    monkeypatch.setattr(hardware, "_system_ram_bytes", lambda: (32_000_000_000, 24_000_000_000))
    monkeypatch.setattr(hardware, "_load_torch_module", lambda: _FakeTorch())

    profile = hardware.detect_hardware_profile()

    assert profile.ram_total_bytes == 32_000_000_000
    assert profile.ram_available_bytes == 24_000_000_000
    assert profile.accelerator == "cuda"
    assert profile.vram_total_bytes == 12_000_000_000
    assert profile.vram_available_bytes == 2_000_000_000
    assert profile.cuda_device_count == 2
