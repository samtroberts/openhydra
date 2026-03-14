from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Any


@dataclass(frozen=True)
class HardwareProfile:
    ram_total_bytes: int | None
    ram_available_bytes: int | None
    accelerator: str
    vram_total_bytes: int | None
    vram_available_bytes: int | None
    cuda_device_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_proc_meminfo() -> tuple[int | None, int | None]:
    path = "/proc/meminfo"
    if not os.path.exists(path):
        return None, None

    total_kb: int | None = None
    available_kb: int | None = None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    available_kb = int(line.split()[1])
    except Exception:
        return None, None

    total_bytes = int(total_kb * 1024) if total_kb is not None else None
    available_bytes = int(available_kb * 1024) if available_kb is not None else None
    return total_bytes, available_bytes


def _system_ram_bytes() -> tuple[int | None, int | None]:
    total_bytes: int | None = None
    available_bytes: int | None = None
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total_pages = int(os.sysconf("SC_PHYS_PAGES"))
        total_bytes = page_size * total_pages
        try:
            available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            available_bytes = page_size * available_pages
        except (ValueError, OSError):
            available_bytes = None
    except (ValueError, OSError, AttributeError):
        total_bytes = None
        available_bytes = None

    if total_bytes is None:
        proc_total, proc_available = _read_proc_meminfo()
        total_bytes = proc_total
        available_bytes = proc_available
    elif available_bytes is None:
        _, proc_available = _read_proc_meminfo()
        if proc_available is not None:
            available_bytes = proc_available

    return total_bytes, available_bytes


def _load_torch_module():
    try:
        import torch
    except Exception:
        return None
    return torch


def detect_hardware_profile() -> HardwareProfile:
    ram_total, ram_available = _system_ram_bytes()
    accelerator = "cpu"
    vram_total: int | None = None
    vram_available: int | None = None
    cuda_device_count = 0

    torch = _load_torch_module()
    if torch is not None:
        cuda = getattr(torch, "cuda", None)
        if cuda is not None and bool(cuda.is_available()):
            accelerator = "cuda"
            try:
                cuda_device_count = int(cuda.device_count())
            except Exception:
                cuda_device_count = 0
            try:
                free_bytes, total_bytes = cuda.mem_get_info()
                vram_available = int(free_bytes)
                vram_total = int(total_bytes)
            except Exception:
                try:
                    current_device = int(cuda.current_device())
                    props = cuda.get_device_properties(current_device)
                    vram_total = int(getattr(props, "total_memory", 0)) or None
                except Exception:
                    vram_total = None
                    vram_available = None
        else:
            backends = getattr(torch, "backends", None)
            mps_backend = getattr(backends, "mps", None) if backends is not None else None
            if mps_backend is not None and bool(mps_backend.is_available()):
                accelerator = "mps"

    return HardwareProfile(
        ram_total_bytes=ram_total,
        ram_available_bytes=ram_available,
        accelerator=accelerator,
        vram_total_bytes=vram_total,
        vram_available_bytes=vram_available,
        cuda_device_count=cuda_device_count,
    )
