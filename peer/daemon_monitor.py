# Copyright 2026 OpenHydra contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
import platform
import re
import shutil
import subprocess
import threading
from typing import Callable


class DaemonMode(str, Enum):
    POLITE = "polite"
    POWER_USER = "power_user"
    DEDICATED = "dedicated"


@dataclass
class ResourceBudget:
    vram_fraction: float
    cpu_fraction: float
    should_yield: bool
    reason: str = "default"


@dataclass(frozen=True)
class RuntimeSignals:
    user_idle_seconds: float | None
    cpu_load: float | None
    fullscreen_active: bool = False


@dataclass(frozen=True)
class MonitorConfig:
    mode: DaemonMode
    idle_threshold_sec: float = 5.0 * 60.0
    high_load_threshold: float = 0.85
    assume_idle_when_unknown: bool = True


def _clamp_fraction(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def compute_resource_budget(
    config: MonitorConfig,
    signals: RuntimeSignals,
) -> ResourceBudget:
    mode = config.mode
    if mode == DaemonMode.DEDICATED:
        return ResourceBudget(
            vram_fraction=1.0,
            cpu_fraction=1.0,
            should_yield=False,
            reason="dedicated",
        )

    if mode == DaemonMode.POLITE:
        if signals.user_idle_seconds is None:
            user_idle = bool(config.assume_idle_when_unknown)
        else:
            user_idle = signals.user_idle_seconds >= config.idle_threshold_sec

        if not user_idle:
            return ResourceBudget(
                vram_fraction=0.0,
                cpu_fraction=0.0,
                should_yield=True,
                reason="user-active",
            )

        return ResourceBudget(
            vram_fraction=0.5,
            cpu_fraction=0.2,
            should_yield=False,
            reason="idle",
        )

    if signals.fullscreen_active:
        return ResourceBudget(
            vram_fraction=0.25,
            cpu_fraction=0.25,
            should_yield=True,
            reason="fullscreen-active",
        )

    if signals.cpu_load is not None and signals.cpu_load >= config.high_load_threshold:
        return ResourceBudget(
            vram_fraction=0.5,
            cpu_fraction=0.35,
            should_yield=False,
            reason="high-system-load",
        )

    return ResourceBudget(
        vram_fraction=0.7,
        cpu_fraction=0.6,
        should_yield=False,
        reason="power-user",
    )


def probe_user_idle_seconds() -> float | None:
    system = platform.system()

    if system == "Darwin":
        try:
            out = subprocess.check_output(["ioreg", "-c", "IOHIDSystem"], text=True, timeout=1.5)
            match = re.search(r'"HIDIdleTime"\s*=\s*(\d+)', out)
            if not match:
                return None
            return float(int(match.group(1))) / 1_000_000_000.0
        except Exception:
            return None

    if system == "Linux":
        if not shutil.which("xprintidle"):
            return None
        try:
            out = subprocess.check_output(["xprintidle"], text=True, timeout=1.5).strip()
            return float(out) / 1000.0
        except Exception:
            return None

    if system == "Windows":  # pragma: no cover
        try:
            import ctypes
            from ctypes import wintypes

            class LASTINPUTINFO(ctypes.Structure):
                _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]

            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32

            info = LASTINPUTINFO()
            info.cbSize = ctypes.sizeof(LASTINPUTINFO)
            if not user32.GetLastInputInfo(ctypes.byref(info)):
                return None
            tick_count = kernel32.GetTickCount()
            elapsed_ms = max(0, int(tick_count) - int(info.dwTime))
            return elapsed_ms / 1000.0
        except Exception:
            return None

    return None


def probe_system_cpu_load() -> float | None:
    try:
        one_min, _, _ = os.getloadavg()
    except Exception:
        return None

    cores = os.cpu_count() or 1
    if cores <= 0:
        cores = 1
    return _clamp_fraction(one_min / float(cores))


def probe_fullscreen_active() -> bool:
    # Cross-platform fullscreen detection is intentionally deferred.
    return False


def get_resource_budget(mode: DaemonMode, user_idle: bool) -> ResourceBudget:
    config = MonitorConfig(
        mode=mode,
        idle_threshold_sec=1.0,
        high_load_threshold=1.0,
        assume_idle_when_unknown=user_idle,
    )
    signals = RuntimeSignals(
        user_idle_seconds=2.0 if user_idle else 0.0,
        cpu_load=0.0,
        fullscreen_active=False,
    )
    return compute_resource_budget(config, signals)


class DaemonController:
    def __init__(
        self,
        config: MonitorConfig,
        *,
        idle_probe: Callable[[], float | None] = probe_user_idle_seconds,
        cpu_load_probe: Callable[[], float | None] = probe_system_cpu_load,
        fullscreen_probe: Callable[[], bool] = probe_fullscreen_active,
    ):
        self.config = config
        self._idle_probe = idle_probe
        self._cpu_load_probe = cpu_load_probe
        self._fullscreen_probe = fullscreen_probe
        self._lock = threading.Lock()
        self._budget = ResourceBudget(vram_fraction=1.0, cpu_fraction=1.0, should_yield=False, reason="initial")

    def refresh(self) -> ResourceBudget:
        signals = RuntimeSignals(
            user_idle_seconds=self._idle_probe(),
            cpu_load=self._cpu_load_probe(),
            fullscreen_active=self._fullscreen_probe(),
        )
        budget = compute_resource_budget(self.config, signals)
        with self._lock:
            self._budget = budget
        return budget

    def current_budget(self) -> ResourceBudget:
        with self._lock:
            return ResourceBudget(
                vram_fraction=self._budget.vram_fraction,
                cpu_fraction=self._budget.cpu_fraction,
                should_yield=self._budget.should_yield,
                reason=self._budget.reason,
            )
