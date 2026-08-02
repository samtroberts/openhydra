// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Host hardware probe — CPU / RAM / GPU(s), shown in the UI (like LM Studio's system panel)
//! so the user sees what they're running on and which models fit.
//!
//! Per-OS, using only stock system tools (no extra crates): macOS `sysctl` +
//! `system_profiler`; Linux `/proc` + `nvidia-smi`/`lspci`; Windows `wmic` + `nvidia-smi`.
//! The command-runners are thin; the **parsing is pure and unit-tested** against fixtures for
//! all three OSes (only the macOS path can run live in CI here).
//!
//! Apple Silicon has **unified memory** (the GPU shares system RAM), so its GPU entry is
//! flagged `unified: true` with `vram_bytes = ram_bytes` rather than a separate pool.

use std::process::Command;

use serde::Serialize;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct GpuInfo {
    pub name: String,
    /// Dedicated VRAM in bytes; `None` when the tool didn't report it. For unified memory this
    /// equals `ram_bytes`.
    pub vram_bytes: Option<u64>,
    /// True on Apple Silicon (GPU shares system RAM) — the UI labels it "unified".
    pub unified: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct SystemInfo {
    pub os: String,
    pub arch: String,
    /// Human CPU/chip name, e.g. "Apple M1" / "AMD Ryzen 9 7900X" / "13th Gen Intel Core i7".
    pub cpu: String,
    pub ram_bytes: u64,
    pub gpus: Vec<GpuInfo>,
}

/// What a Tier-2 engine (vLLM/ComfyUI/Exo) needs, probed so the installer can pick a path
/// (Docker-first → uv venv) and **block with an actionable message on a system prereq we
/// can't install for the user** (an NVIDIA driver needs sudo/reboot). Serializable so the UI
/// can show the report too.
#[derive(Debug, Clone, Serialize, PartialEq, Eq, Default)]
pub struct Prereqs {
    /// NVIDIA driver version from `nvidia-smi` (e.g. "535.104.05"); `None` = no usable GPU.
    pub nvidia_driver: Option<String>,
    pub docker: bool,
    pub uv: bool,
    /// Python version (e.g. "3.11.6") from `python3`/`python`; `None` if absent.
    pub python: Option<String>,
}

/// Probe the Tier-2 prerequisites (stock tools only; never fails).
pub fn probe_prereqs() -> Prereqs {
    Prereqs {
        nvidia_driver: run("nvidia-smi", &["--query-gpu=driver_version", "--format=csv,noheader"])
            .and_then(|s| parse_first_nonempty(&s)),
        docker: run("docker", &["--version"]).is_some(),
        uv: run("uv", &["--version"]).is_some(),
        python: run("python3", &["--version"])
            .or_else(|| run("python", &["--version"]))
            .and_then(|s| parse_python_version(&s)),
    }
}

fn parse_first_nonempty(s: &str) -> Option<String> {
    s.lines().map(str::trim).find(|l| !l.is_empty()).map(str::to_string)
}

/// "Python 3.11.6" → "3.11.6".
fn parse_python_version(s: &str) -> Option<String> {
    s.split_whitespace()
        .find(|t| t.chars().next().is_some_and(|c| c.is_ascii_digit()))
        .map(str::to_string)
}

fn run(program: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(program).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Probe the host. Never fails — unknown fields fall back to `std::env::consts` / "Unknown".
pub fn probe() -> SystemInfo {
    let os = std::env::consts::OS.to_string();
    let arch = std::env::consts::ARCH.to_string();
    match std::env::consts::OS {
        "macos" => probe_macos(os, arch),
        "linux" => probe_linux(os, arch),
        "windows" => probe_windows(os, arch),
        _ => SystemInfo { os, arch, cpu: "Unknown".into(), ram_bytes: 0, gpus: vec![] },
    }
}

// ── macOS ──

fn probe_macos(os: String, arch: String) -> SystemInfo {
    let cpu = run("sysctl", &["-n", "machdep.cpu.brand_string"])
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "Apple".into());
    let ram_bytes = run("sysctl", &["-n", "hw.memsize"])
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0);
    let gpus = run("system_profiler", &["SPDisplaysDataType"])
        .and_then(|sp| parse_macos_gpu(&sp))
        .map(|(name, cores)| {
            let name = match cores {
                Some(c) => format!("{name} ({c}-core GPU)"),
                None => name,
            };
            // Apple Silicon = unified memory; report the shared pool as VRAM.
            vec![GpuInfo { name, vram_bytes: Some(ram_bytes), unified: true }]
        })
        .unwrap_or_default();
    SystemInfo { os, arch, cpu, ram_bytes, gpus }
}

/// Pull the GPU chipset name + core count out of `system_profiler SPDisplaysDataType`.
fn parse_macos_gpu(sp: &str) -> Option<(String, Option<u32>)> {
    let name = sp
        .lines()
        .find_map(|l| l.trim().strip_prefix("Chipset Model:"))
        .map(|s| s.trim().to_string())?;
    let cores = sp
        .lines()
        .find_map(|l| l.trim().strip_prefix("Total Number of Cores:"))
        .and_then(|s| s.trim().parse::<u32>().ok());
    Some((name, cores))
}

// ── Linux ──

fn probe_linux(os: String, arch: String) -> SystemInfo {
    let cpu = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|c| parse_cpuinfo_model(&c))
        .unwrap_or_else(|| "Unknown CPU".into());
    let ram_bytes = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|m| parse_meminfo_kb(&m))
        .unwrap_or(0);
    let gpus = probe_gpus_nvidia_smi().unwrap_or_else(|| {
        // No NVIDIA tool → best-effort name from lspci, no VRAM figure.
        run("sh", &["-c", "lspci | grep -Ei 'vga|3d|display'"])
            .map(|s| parse_lspci_gpus(&s))
            .unwrap_or_default()
    });
    SystemInfo { os, arch, cpu, ram_bytes, gpus }
}

fn parse_cpuinfo_model(cpuinfo: &str) -> Option<String> {
    cpuinfo
        .lines()
        .find(|l| l.starts_with("model name"))
        .and_then(|l| l.split(':').nth(1))
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn parse_meminfo_kb(meminfo: &str) -> Option<u64> {
    meminfo
        .lines()
        .find(|l| l.starts_with("MemTotal:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|kb| kb.parse::<u64>().ok())
        .map(|kb| kb * 1024)
}

fn parse_lspci_gpus(lspci: &str) -> Vec<GpuInfo> {
    lspci
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            // "01:00.0 VGA compatible controller: NVIDIA Corporation AD102 [GeForce RTX 4090]"
            let name = l.splitn(3, ':').nth(2).unwrap_or(l).trim().to_string();
            GpuInfo { name, vram_bytes: None, unified: false }
        })
        .collect()
}

// ── Windows ──

fn probe_windows(os: String, arch: String) -> SystemInfo {
    let cpu = run("wmic", &["cpu", "get", "name"])
        .and_then(|s| parse_wmic_value(&s))
        .unwrap_or_else(|| "Unknown CPU".into());
    let ram_bytes = run("wmic", &["computersystem", "get", "TotalPhysicalMemory"])
        .and_then(|s| parse_wmic_value(&s))
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);
    let gpus = probe_gpus_nvidia_smi().unwrap_or_else(|| {
        run("wmic", &["path", "win32_VideoController", "get", "name,AdapterRAM"])
            .map(|s| parse_wmic_gpus(&s))
            .unwrap_or_default()
    });
    SystemInfo { os, arch, cpu, ram_bytes, gpus }
}

/// `wmic KEY get FIELD` prints a header line then the value; take the first non-header line.
fn parse_wmic_value(out: &str) -> Option<String> {
    out.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .nth(1) // 0 = the column header
        .map(|s| s.to_string())
}

fn parse_wmic_gpus(out: &str) -> Vec<GpuInfo> {
    // Columns: "AdapterRAM  Name" (wmic orders alphabetically). Header first, then rows.
    let mut lines = out.lines().map(str::trim).filter(|l| !l.is_empty());
    let Some(header) = lines.next() else { return vec![] };
    let name_first = header.to_ascii_lowercase().find("name").unwrap_or(usize::MAX)
        < header.to_ascii_lowercase().find("adapterram").unwrap_or(0);
    lines
        .filter_map(|row| {
            // Split on runs of ≥2 spaces (wmic pads columns).
            let cols: Vec<&str> = row.split("  ").map(str::trim).filter(|s| !s.is_empty()).collect();
            if cols.is_empty() {
                return None;
            }
            let (name, ram) = if name_first {
                (cols.first().copied()?, cols.get(1).copied())
            } else {
                (cols.get(1).copied()?, cols.first().copied())
            };
            let vram_bytes = ram.and_then(|r| r.parse::<u64>().ok()).filter(|&b| b > 0);
            Some(GpuInfo { name: name.to_string(), vram_bytes, unified: false })
        })
        .collect()
}

// ── nvidia-smi (Linux + Windows) ──

fn probe_gpus_nvidia_smi() -> Option<Vec<GpuInfo>> {
    let out = run(
        "nvidia-smi",
        &["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
    )?;
    let gpus = parse_nvidia_csv(&out);
    (!gpus.is_empty()).then_some(gpus)
}

/// Parse `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits` — rows of
/// `NAME, MiB`.
fn parse_nvidia_csv(out: &str) -> Vec<GpuInfo> {
    out.lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| {
            let mut parts = l.splitn(2, ',');
            let name = parts.next()?.trim().to_string();
            let mib = parts
                .next()
                .and_then(|s| s.trim().split_whitespace().next())
                .and_then(|s| s.parse::<u64>().ok());
            Some(GpuInfo { name, vram_bytes: mib.map(|m| m * 1024 * 1024), unified: false })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_macos_gpu_name_and_cores() {
        let sp = "Graphics/Displays:\n\n    Apple M1:\n\n      Chipset Model: Apple M1\n      Type: GPU\n      Total Number of Cores: 7\n      Vendor: Apple (0x106b)\n";
        assert_eq!(parse_macos_gpu(sp), Some(("Apple M1".to_string(), Some(7))));
        // No core count → still returns the name.
        let sp2 = "      Chipset Model: Apple M3 Max\n      Type: GPU\n";
        assert_eq!(parse_macos_gpu(sp2), Some(("Apple M3 Max".to_string(), None)));
        assert_eq!(parse_macos_gpu("no gpu here"), None);
    }

    #[test]
    fn parses_linux_cpu_and_ram() {
        let cpuinfo = "processor\t: 0\nmodel name\t: AMD Ryzen 9 7900X 12-Core Processor\nflags\t: fpu\n";
        assert_eq!(parse_cpuinfo_model(cpuinfo).as_deref(), Some("AMD Ryzen 9 7900X 12-Core Processor"));
        let meminfo = "MemTotal:       32791528 kB\nMemFree:         1000000 kB\n";
        assert_eq!(parse_meminfo_kb(meminfo), Some(32791528 * 1024));
        assert_eq!(parse_meminfo_kb("nope"), None);
    }

    #[test]
    fn parses_nvidia_smi_csv_to_bytes() {
        let out = "NVIDIA GeForce RTX 4090, 24564\nTesla T4, 15360\n";
        let gpus = parse_nvidia_csv(out);
        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[0].name, "NVIDIA GeForce RTX 4090");
        assert_eq!(gpus[0].vram_bytes, Some(24564 * 1024 * 1024));
        assert!(!gpus[0].unified);
        assert_eq!(gpus[1].name, "Tesla T4");
        assert_eq!(gpus[1].vram_bytes, Some(15360 * 1024 * 1024));
    }

    #[test]
    fn parses_lspci_gpu_names() {
        let lspci = "01:00.0 VGA compatible controller: NVIDIA Corporation AD102 [GeForce RTX 4090]\n";
        let gpus = parse_lspci_gpus(lspci);
        assert_eq!(gpus.len(), 1);
        assert!(gpus[0].name.contains("GeForce RTX 4090"));
        assert_eq!(gpus[0].vram_bytes, None);
    }

    #[test]
    fn parses_tier2_prereq_helpers() {
        assert_eq!(parse_first_nonempty("\n535.104.05\n").as_deref(), Some("535.104.05"));
        assert_eq!(parse_first_nonempty("  \n \n"), None);
        assert_eq!(parse_python_version("Python 3.11.6\n").as_deref(), Some("3.11.6"));
        assert_eq!(parse_python_version("Python 3.12.1").as_deref(), Some("3.12.1"));
        assert_eq!(parse_python_version("no version here"), None);
    }

    #[test]
    fn parses_wmic_value_and_gpus() {
        assert_eq!(parse_wmic_value("Name\nAMD Ryzen 9 7900X\n").as_deref(), Some("AMD Ryzen 9 7900X"));
        assert_eq!(parse_wmic_value("TotalPhysicalMemory\n34078572544\n").as_deref(), Some("34078572544"));
        // wmic orders columns alphabetically: AdapterRAM before Name.
        let gpu_out = "AdapterRAM     Name\n25757220864    NVIDIA GeForce RTX 4090\n";
        let gpus = parse_wmic_gpus(gpu_out);
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].name, "NVIDIA GeForce RTX 4090");
        assert_eq!(gpus[0].vram_bytes, Some(25757220864));
    }
}
