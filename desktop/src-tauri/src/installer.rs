// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Tier-1 engine installer (roadmap Part D, **v1**).
//!
//! v1 = the install recipe for each engine is **hardcoded in the app** (a `match (engine,
//! os, accel)`), so a command is as trusted as the app itself — no signed catalog yet (that
//! is the v2 upgrade). The executor drives a small, fixed set of step kinds and **streams
//! every line to the webview** as Tauri events, so the Engines view shows live progress.
//!
//! Flow (roadmap D.5): Detect → PrereqCheck → (consent, in the UI) → Run/Open →
//! HealthCheck → hand back to `detect_engines`. Idempotent: a `detect_first` hit short-circuits.
//!
//! Safety: every step kind is a **vetted, typed primitive** — the recipe picks *which*, never
//! runs arbitrary caller-supplied shell, and the app **never executes a downloaded binary**
//! (every recipe uses the vendor's own installer script / package manager, or opens the
//! vendor download page in the browser for the user to run). A signed catalog with a verified
//! `download_binary`/`download_archive` primitive is the v2 upgrade.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::io::BufRead;

use serde::Serialize;
use tauri::{AppHandle, Emitter};

/// GPU/accelerator target — chooses the right recipe variant where it matters (llama.cpp).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Accel {
    Cuda,
    Rocm,
    Metal,
    Cpu,
    /// Let the engine's own installer auto-detect (Ollama does this).
    Auto,
}

impl Accel {
    pub fn from_str_opt(s: Option<&str>) -> Self {
        match s.map(|s| s.to_ascii_lowercase()).as_deref() {
            Some("cuda") => Accel::Cuda,
            Some("rocm") => Accel::Rocm,
            Some("metal") => Accel::Metal,
            Some("cpu") => Accel::Cpu,
            _ => Accel::Auto,
        }
    }
}

/// One vetted installer step. The recipe is an ordered list of these; the catalog (v2) will
/// serialise exactly this shape. No variant carries arbitrary shell — `Run` is a fixed
/// program + args chosen by the recipe, never a user string.
#[derive(Debug, Clone)]
pub enum Step {
    /// Run a vetted command (program + args), streaming stdout+stderr to the webview.
    Run { program: String, args: Vec<String> },
    /// Download a vendor GUI installer (`url` → `install_dir/<filename>`, streamed to disk via
    /// `curl`) **and hand it to the OS to run** — `open` the `.dmg`/`.pkg` on macOS, run the
    /// signed `.exe` on Windows, `chmod +x` + launch the `.AppImage` on Linux. Integrity is the
    /// OS's own code-signature gate (Gatekeeper notarization / SmartScreen Authenticode) at
    /// launch — the right check for a moving "latest" URL (a pinned SHA is impossible + redundant
    /// with it). The user consented at the install dialog and still sees the installer's own UI.
    DownloadRun { url: String, filename: String },
}

/// How to confirm the install succeeded.
#[derive(Debug, Clone)]
pub enum Health {
    /// The engine is serving — poll an HTTP endpoint until it answers (Ollama).
    Http { url: String, timeout_s: u64 },
    /// The install placed a server binary on PATH, but it needs a model at launch before it
    /// serves (llama.cpp) — so success is "the binary is now callable", not an HTTP 200.
    ProgramOnPath { program: String },
    /// The remaining steps are the user's (a GUI installer they run, a server they enable) —
    /// there's nothing to auto-check, so succeed and show `note` as the next step (LM Studio).
    Manual { note: String },
}

/// A resolved install plan for one (engine, os, accel).
#[derive(Debug, Clone)]
pub struct Recipe {
    pub engine: &'static str,
    /// Human note shown on the consent line (what will run, from where).
    pub summary: String,
    pub steps: Vec<Step>,
    pub health: Health,
    /// Optional model to pull + warm once the engine is healthy (roadmap D.5).
    pub default_model: Option<&'static str>,
    /// True when the recipe's exact commands are verified on this OS; false = structured but
    /// needs a real-target check before it can be trusted end-to-end (surfaced to the UI).
    pub verified: bool,
}

impl Recipe {
    /// The terminal success message for a recipe with no `default_model` to pull — tailored to
    /// how "done" reads for this engine (serving vs installed-binary vs manual GUI step).
    pub fn completion_message(&self) -> String {
        match &self.health {
            Health::Http { .. } => format!("{} installed and serving", self.engine),
            Health::ProgramOnPath { .. } => {
                format!("{} installed — ready to configure", self.engine)
            }
            Health::Manual { note } => note.clone(),
        }
    }
}

/// Host OS, normalised from [`std::env::consts::OS`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Os {
    Macos,
    Linux,
    Windows,
}

impl Os {
    pub fn current() -> Option<Self> {
        match std::env::consts::OS {
            "macos" => Some(Os::Macos),
            "linux" => Some(Os::Linux),
            "windows" => Some(Os::Windows),
            _ => None,
        }
    }
}

/// Canonical engine id — the UI and `detect_engines` disagree on punctuation
/// (`llama-cpp` vs `llama.cpp`), so fold both to the crate's `detect_engines` spelling.
pub fn normalize_engine(engine: &str) -> &'static str {
    match engine {
        "ollama" => "ollama",
        "llama.cpp" | "llama-cpp" | "llama_cpp" | "llamacpp" => "llama.cpp",
        "lm-studio" | "lmstudio" | "lm_studio" => "lm-studio",
        "vllm" => "vllm",
        "comfyui" | "comfy-ui" | "comfy_ui" => "comfyui",
        "exo" => "exo",
        _ => "",
    }
}

/// The engine's standard local health endpoint — reused as the detect-first probe.
fn health_url(engine: &str) -> &'static str {
    match normalize_engine(engine) {
        "ollama" => "http://127.0.0.1:11434/api/version",
        "llama.cpp" => "http://127.0.0.1:8080/health",
        "lm-studio" => "http://127.0.0.1:1234/v1/models",
        "vllm" => "http://127.0.0.1:8000/health",
        "comfyui" => "http://127.0.0.1:8188/",
        "exo" => "http://127.0.0.1:52415/",
        _ => "",
    }
}

/// Official vLLM Docker image (CUDA + Python + vLLM baked in). A pinned digest is the v2
/// hardening; `latest` is honest for the guided Tier-2 path.
const VLLM_IMAGE: &str = "vllm/vllm-openai:latest";

/// The hardcoded v1 recipe for `(engine, os, accel)`, or an error when unsupported.
///
/// **Confidence markers** are honest: `verified: true` only where the exact command is a
/// vendor-official, cross-checked path. Ollama-on-Linux (the official `install.sh`) is the
/// one fully-verified flagship path today; the others are structured with the documented
/// method and flagged for a real-target check (v1 ships them behind the "may take minutes /
/// may need a real machine to confirm" note; v2's signed catalog replaces the guesswork).
/// Which flavour of an engine to install, where more than one exists (currently ComfyUI / Exo on
/// macOS: a desktop `.app` vs a headless CLI/source install). `Default` = the platform's preferred
/// flavour (app on macOS); `Cli` forces the headless path.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Variant {
    #[default]
    Default,
    App,
    Cli,
}

impl Variant {
    pub fn from_str_opt(s: Option<&str>) -> Variant {
        match s {
            Some("cli") => Variant::Cli,
            Some("app") => Variant::App,
            _ => Variant::Default,
        }
    }
}

/// Does `engine` offer a CLI/headless alternative to its default install on this OS? (Powers the
/// app-vs-CLI toggle in the consent UI.) Today: ComfyUI + Exo on macOS (default = app, CLI = source).
pub fn has_cli_variant(engine: &str, os: Os) -> bool {
    os == Os::Macos && matches!(normalize_engine(engine), "comfyui" | "exo")
}

pub fn recipe_for(engine: &str, os: Os, accel: Accel) -> Result<Recipe, String> {
    recipe_for_variant(engine, os, accel, Variant::Default)
}

pub fn recipe_for_variant(engine: &str, os: Os, _accel: Accel, variant: Variant) -> Result<Recipe, String> {
    match normalize_engine(engine) {
        "ollama" => Ok(ollama_recipe(os)),
        "llama.cpp" => llama_cpp_recipe(os),
        "lm-studio" => Ok(lm_studio_recipe(os)),
        "vllm" => vllm_recipe(os),
        "comfyui" => comfyui_recipe(os, variant),
        "exo" => exo_recipe(os, variant),
        _ => Err(format!("no installer for engine '{engine}'")),
    }
}

/// vLLM (Tier-2) — the **probe-then-install** flow. vLLM is Linux + NVIDIA-GPU only, and its
/// hardest prereq (a GPU driver) can't be installed for the user, so we probe first and either
/// pick a path or **block with an actionable message**:
///   1. no OS/GPU support → block ("needs Linux + NVIDIA");
///   2. **Docker present → pull the official image** (CUDA+Python+vLLM baked in — most reliable);
///   3. else **`uv` present → isolated venv + `uv pip install vllm`** (guided, may take minutes);
///   4. else → block ("install Docker or uv").
/// vLLM is a server needing a `--model` at launch, so success = "installed" (`Health::Manual`
/// with the exact launch command incl. the T4/Turing runtime flags), not an HTTP probe.
fn vllm_recipe(os: Os) -> Result<Recipe, String> {
    match os {
        Os::Linux => vllm_plan(os, &crate::hostinfo::probe_prereqs()),
        // vLLM's CPU build-from-source does NOT compile on Apple Silicon (Apple Clang rejects a
        // constexpr std::sqrt in csrc/cpu/sgl-kernels/fla.cpp — verified across C++17/20/26). The
        // working Mac path is the community `vllm-metal` plugin: MLX backend, PREBUILT wheels (no
        // source compile), so it sidesteps that wall. Verified install method vs the repo (2026-08).
        Os::Macos => vllm_metal_recipe(),
        // No native Windows build — vLLM runs under WSL2 (a Linux environment).
        Os::Windows => Err(
            "vLLM has no native Windows build — run it under WSL2 (Linux), where OpenHydra's \
             Docker/uv path then applies."
                .into(),
        ),
    }
}

/// vLLM on Apple Silicon via the community `vllm-metal` plugin (github.com/vllm-project/vllm-metal):
/// MLX backend, prebuilt wheels (no source compile), installs into `~/.venv-vllm-metal`. Needs
/// native arm64 Python 3.12; the script pulls vLLM core itself. `verified: false` — community
/// plugin; the install *method* is verified vs the repo, the outcome is user-tested.
fn vllm_metal_recipe() -> Result<Recipe, String> {
    if !has_program("curl") {
        return Err("The vllm-metal installer is fetched with curl — install curl first, then retry.".into());
    }
    Ok(Recipe {
        engine: "vllm",
        summary: "Install vLLM for Apple Silicon via the community vllm-metal plugin (MLX backend, \
                  PREBUILT wheels — no source compile, unlike the CPU build which doesn't compile on \
                  a Mac). Runs the official install.sh into ~/.venv-vllm-metal. Needs native arm64 \
                  Python 3.12. Then `source ~/.venv-vllm-metal/bin/activate` and `vllm serve <model>`."
            .into(),
        steps: vec![Step::Run {
            program: "sh".into(),
            args: vec![
                "-c".into(),
                "curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash".into(),
            ],
        }],
        health: Health::Manual {
            note: "vLLM (Metal/MLX) installed into ~/.venv-vllm-metal. Activate it \
                   (`source ~/.venv-vllm-metal/bin/activate`) and run `vllm serve <hf-model> \
                   --port 8000`; this card flips to running once it serves."
                .into(),
        },
        default_model: None,
        verified: false,
    })
}

/// Pure planner (probe → decision) — unit-tested with synthetic [`Prereqs`].
fn vllm_plan(_os: Os, p: &crate::hostinfo::Prereqs) -> Result<Recipe, String> {
    // (1) System prereq we cannot install for the user (needs sudo/reboot): the GPU driver.
    let Some(driver) = p.nvidia_driver.as_deref() else {
        return Err("vLLM needs an NVIDIA GPU with drivers, but `nvidia-smi` isn't present. \
                    Install the NVIDIA driver + CUDA toolkit (a system step we can't do for \
                    you), then retry."
            .into());
    };
    let launch_note = |port: &str| {
        format!(
            "vLLM installed (NVIDIA driver {driver}). Launch it with a model to serve an \
             OpenAI API on :{port}, e.g. `vllm serve <hf-model> --port {port}`. On a Turing/T4 \
             GPU add `--enforce-eager` and set `VLLM_ATTENTION_BACKEND=TRITON_ATTN`. This card \
             flips to running once it serves."
        )
    };
    // (2) Docker-first — one image bakes CUDA+Python+vLLM; the most reliable path.
    if p.docker {
        return Ok(Recipe {
            engine: "vllm",
            summary: format!(
                "Docker path (recommended): pull `{VLLM_IMAGE}` — bundles CUDA + Python + vLLM. \
                 NVIDIA driver {driver} + Docker detected."
            ),
            steps: vec![Step::Run {
                program: "docker".into(),
                args: vec!["pull".into(), VLLM_IMAGE.into()],
            }],
            health: Health::Manual { note: launch_note("8000") },
            default_model: None,
            verified: false,
        });
    }
    // (3) uv venv fallback.
    if !p.uv {
        return Err("vLLM needs either Docker (recommended) or `uv` (a fast Python installer). \
                    Install Docker, or `uv` from astral.sh/uv, then retry."
            .into());
    }
    let venv = install_dir("vllm").join("venv");
    let venv_s = venv.to_string_lossy().into_owned();
    let py = venv.join("bin").join("python").to_string_lossy().into_owned();
    Ok(Recipe {
        engine: "vllm",
        summary: format!(
            "uv path: create an isolated venv at {venv_s} and install vLLM. NVIDIA driver \
             {driver} + uv detected. (A hash-pinned lockfile is the reliability follow-up — \
             this is the guided \"may take minutes / may fail\" path.)"
        ),
        steps: vec![
            Step::Run { program: "uv".into(), args: vec!["venv".into(), venv_s] },
            Step::Run {
                program: "uv".into(),
                args: vec!["pip".into(), "install".into(), "--python".into(), py, "vllm".into()],
            },
        ],
        health: Health::Manual { note: launch_note("8000") },
        default_model: None,
        verified: false,
    })
}

/// LM Studio (Tier-3) — a GUI app with no headless install. "Install" here means: start a
/// **direct download** of the OS-correct official installer (the browser follows the stable
/// `lmstudio.ai/download/latest/<os>` 302 to the current build, so no version/hash pin), then
/// guide the user to run it and enable its Local Server. We deliberately never execute the
/// downloaded binary — the browser + the user's own "open installer" step are the trust
/// boundary, exactly as if they clicked Download on the site.
fn lm_studio_recipe(os: Os) -> Recipe {
    // Verified against lmstudio.ai (2026-07): darwin/arm64 .dmg, win32/x64 .exe, linux/x64 AppImage.
    let (url, size, filename) = match os {
        Os::Macos => ("https://lmstudio.ai/download/latest/darwin/arm64", "~570 MB .dmg, Apple Silicon", "LM-Studio.dmg"),
        Os::Windows => ("https://lmstudio.ai/download/latest/win32/x64", "~617 MB .exe", "LM-Studio-Setup.exe"),
        Os::Linux => ("https://lmstudio.ai/download/latest/linux/x64", "~1.1 GB AppImage, x86-64", "LM-Studio.AppImage"),
    };
    let run_verb = match os {
        Os::Macos => "installs it to /Applications and launches it",
        Os::Windows => "launches the installer",
        Os::Linux => "launches the AppImage",
    };
    Recipe {
        engine: "lm-studio",
        summary: format!(
            "Download the official LM Studio installer ({size}) and run it — the app fetches it, \
             then {run_verb}. The installer is signed (macOS Gatekeeper / Windows SmartScreen \
             verify it). LM Studio is a GUI app: complete its install, then enable Developer → \
             Local Server (:1234) and it appears here."
        ),
        steps: vec![Step::DownloadRun { url: url.to_string(), filename: filename.to_string() }],
        health: Health::Manual {
            note: "LM Studio installed. OpenHydra started its local server on :1234 (via the \
                   bundled `lms` CLI) — this card flips to running once it serves. If it doesn't, \
                   open LM Studio → Developer → Start Server."
                .into(),
        },
        default_model: None,
        // Official verified URL; the OS code-signature gate verifies the installer at launch.
        verified: true,
    }
}

fn ollama_recipe(os: Os) -> Recipe {
    let health = Health::Http { url: health_url("ollama").to_string(), timeout_s: 30 };
    match os {
        // Verified: the official installer sets up + starts the systemd service and
        // auto-detects CUDA/ROCm. HTTPS from the vendor; the script is the vendor's own.
        Os::Linux => Recipe {
            engine: "ollama",
            summary: "Install Ollama via the official script (ollama.com/install.sh) → starts on :11434".into(),
            steps: vec![Step::Run {
                program: "sh".into(),
                args: vec!["-c".into(), "curl -fsSL https://ollama.com/install.sh | sh".into()],
            }],
            health,
            default_model: Some("qwen2.5:7b"),
            verified: true,
        },
        // macOS: use the SAME official `install.sh` — Ollama's README documents it as
        // cross-platform (mac + Linux), and it manages the server (Metal auto). This replaces
        // the earlier `brew install ollama` (Homebrew is *community*, not Ollama-official, and
        // `brew install` doesn't start the server, so the :11434 health check would fail).
        // `unverified` only because it isn't run end-to-end on a mac here.
        Os::Macos => Recipe {
            engine: "ollama",
            summary: "Install Ollama via the official installer (ollama.com/install.sh, Metal auto) → serves on :11434".into(),
            steps: vec![Step::Run {
                program: "sh".into(),
                args: vec!["-c".into(), "curl -fsSL https://ollama.com/install.sh | sh".into()],
            }],
            health,
            default_model: Some("qwen2.5:7b"),
            verified: false,
        },
        // Windows: Ollama's **official PowerShell one-liner** (per its README). Replaces the
        // old download-the-exe-with-a-placeholder-hash path — a moving "latest" installer can't
        // be hash-pinned, and this is the vendor's own recommended command.
        Os::Windows => Recipe {
            engine: "ollama",
            summary: "Install Ollama via the official PowerShell installer (ollama.com/install.ps1) → serves on :11434".into(),
            steps: vec![Step::Run {
                program: "powershell".into(),
                args: vec!["-Command".into(), "irm https://ollama.com/install.ps1 | iex".into()],
            }],
            health,
            default_model: Some("qwen2.5:7b"),
            verified: false,
        },
    }
}

/// llama.cpp Tier-1 via **Homebrew** (`brew install llama.cpp`) on macOS/Linux — a vetted,
/// formula-managed install (Homebrew handles the download + digest + build, so no manual hash
/// pinning), mirroring the "use the official installer" pattern of Ollama-on-Linux.
///
/// Key difference from Ollama: llama.cpp is a **server binary, not a model manager** — after
/// install, `llama-server` is on PATH but serves nothing until launched with a GGUF model. So
/// success is "the binary is present" ([`Health::ProgramOnPath`]), there is **no**
/// `default_model` to pull, and the summary tells the user the next step. Windows has no
/// equivalent one-liner → routed to Guided install.
fn llama_cpp_recipe(os: Os) -> Result<Recipe, String> {
    match os {
        Os::Macos | Os::Linux => Ok(Recipe {
            engine: "llama.cpp",
            summary: "Install llama.cpp via Homebrew (brew install llama.cpp) — installs \
                      `llama-server`; then point it at a GGUF model to serve on :8080."
                .into(),
            steps: vec![Step::Run {
                program: "brew".into(),
                args: vec!["install".into(), "llama.cpp".into()],
            }],
            // No HTTP endpoint right after install (no model loaded) — success = the binary
            // is callable. Retried briefly for PATH propagation.
            health: Health::ProgramOnPath { program: "llama-server".into() },
            default_model: None,
            // Homebrew is authoritative, but flag pending a real-macOS/-Linux end-to-end run.
            verified: false,
        }),
        Os::Windows => Err(
            "llama.cpp has no one-click Windows installer — use the Guided install (download a \
             prebuilt release from github.com/ggml-org/llama.cpp)"
                .to_string(),
        ),
    }
}

/// User-space install root for an engine's artifacts: `~/.openhydra/engines/<id>/`.
pub fn install_dir(engine: &str) -> PathBuf {
    crate::openhydra_dir().join("engines").join(engine)
}

/// True when a probed Python version string (e.g. "3.13.1") is exactly 3.13.x — Exo pins
/// `requires-python = "==3.13.*"`, so ≥3.12 (or even 3.14) does NOT satisfy it.
fn python_is_313(v: &Option<String>) -> bool {
    let Some(v) = v else { return false };
    let mut parts = v.split('.').map(|s| s.parse::<u32>().unwrap_or(0));
    let (major, minor) = (parts.next().unwrap_or(0), parts.next().unwrap_or(0));
    major == 3 && minor == 13
}

/// ComfyUI (Tier-2) — image gen. Cross-platform via the **official `comfy-cli`**, which clones
/// ComfyUI + installs the GPU-correct PyTorch (Metal on Apple Silicon, CUDA/ROCm/CPU on Linux)
/// + ComfyUI-Manager. Probe-then-install: prefer `uv` (isolated, provides Python) via `uvx`
/// (no PATH surprises); else `pip`; block if neither Python nor uv is present. A server needing
/// a checkpoint model, so `Health::Manual`. `verified: false` (guided; big torch download).
fn comfyui_recipe(os: Os, variant: Variant) -> Result<Recipe, String> {
    match os {
        // macOS default = the official signed desktop app (robust, no toolchain). `Cli` forces the
        // headless comfy-cli source install (for running ComfyUI without the GUI). Linux/Windows
        // only have the comfy-cli path.
        Os::Macos if variant != Variant::Cli => comfyui_mac_app_recipe(),
        _ => comfyui_plan(os, &crate::hostinfo::probe_prereqs()),
    }
}

/// ComfyUI on macOS — the official desktop app. `download.comfy.org/mac/dmg/arm64` 307-redirects
/// to the current signed build (stable, like LM Studio's `/latest`), so no version pin. Bundles
/// Python + Metal PyTorch. `verified: false` — not yet e2e-verified as an OpenHydra provider.
fn comfyui_mac_app_recipe() -> Result<Recipe, String> {
    Ok(Recipe {
        engine: "comfyui",
        summary: "Download the official ComfyUI desktop app (signed .dmg, ~170 MB) and install it \
                  to /Applications — like LM Studio. It bundles Python + the Metal PyTorch build, so \
                  no toolchain needed. Then launch ComfyUI and add a checkpoint model."
            .into(),
        steps: vec![Step::DownloadRun {
            url: "https://download.comfy.org/mac/dmg/arm64".into(),
            filename: "ComfyUI.dmg".into(),
        }],
        health: Health::Manual {
            note: "ComfyUI desktop installed to /Applications. Launch it and add a checkpoint model \
                   (Stable Diffusion / Flux); this card flips to running once it serves."
                .into(),
        },
        default_model: None,
        verified: false,
    })
}

fn comfyui_plan(os: Os, p: &crate::hostinfo::Prereqs) -> Result<Recipe, String> {
    if p.python.is_none() && !p.uv {
        return Err("ComfyUI's installer (comfy-cli) needs Python 3.9+ — install Python, or `uv` \
                    (astral.sh/uv), then retry."
            .into());
    }
    let ws = install_dir("comfyui").to_string_lossy().into_owned();
    // comfy-cli requires an explicit accelerator with --skip-prompt (no interactive pick). On
    // Apple Silicon that's --m-series (Metal); elsewhere default to --cpu (Linux GPU users can
    // reinstall with --nvidia/--amd). Verified against Comfy-Org/comfy-cli.
    let gpu = match os {
        Os::Macos => "--m-series",
        _ => "--cpu",
    };
    // `uvx --from comfy-cli comfy …` runs the CLI without a global install / PATH lookup; the pip
    // fallback puts `comfy` in a user bin that may not be on PATH this session (guided caveat).
    let (via, steps) = if p.uv {
        (
            "uv",
            vec![Step::Run {
                program: "uvx".into(),
                args: vec![
                    "--from".into(), "comfy-cli".into(), "comfy".into(),
                    "--skip-prompt".into(), "--workspace".into(), ws.clone(), "install".into(), gpu.into(),
                ],
            }],
        )
    } else {
        (
            "pip",
            vec![
                Step::Run { program: "pip".into(), args: vec!["install".into(), "--user".into(), "comfy-cli".into()] },
                Step::Run { program: "comfy".into(), args: vec!["--skip-prompt".into(), "--workspace".into(), ws.clone(), "install".into(), gpu.into()] },
            ],
        )
    };
    Ok(Recipe {
        engine: "comfyui",
        summary: format!(
            "Install ComfyUI via the official comfy-cli (using {via}) → clones ComfyUI + installs \
             the GPU-correct PyTorch into {ws}. Big download. Then launch it (`comfy launch`, \
             serves :8188) and add a checkpoint model."
        ),
        steps,
        health: Health::Manual {
            note: "ComfyUI installed. Launch it (`comfy launch`, serves :8188) and add a \
                   checkpoint model (Stable Diffusion / Flux); this card flips to running once \
                   it serves."
                .into(),
        },
        default_model: None,
        verified: false,
    })
}

/// Resolve the latest GitHub release `.dmg` asset URL for `owner/repo` (e.g. `exo-explore/exo`)
/// via the releases API. Best-effort with `curl` (blocking; call from `spawn_blocking`); returns
/// None offline or if no `.dmg` asset is present. Keys off `browser_download_url` (the actual
/// download link) rather than the asset `name`/`url` fields, so it survives compact JSON.
fn latest_github_dmg_url(owner_repo: &str) -> Option<String> {
    let api = format!("https://api.github.com/repos/{owner_repo}/releases/latest");
    let out = Command::new("curl")
        .args(["-sL", "-H", "User-Agent: openhydra-desktop", "-H", "Accept: application/vnd.github+json", &api])
        .output()
        .ok()?;
    parse_github_dmg_url(&String::from_utf8_lossy(&out.stdout))
}

/// Pull the first `browser_download_url` ending in `.dmg` out of a GitHub releases JSON body.
/// Split out from the network call so it's unit-testable.
fn parse_github_dmg_url(body: &str) -> Option<String> {
    body.split("\"browser_download_url\":\"")
        .skip(1)
        .filter_map(|chunk| chunk.split('"').next())
        .find(|url| url.to_lowercase().ends_with(".dmg"))
        .map(str::to_string)
}

/// Exo on macOS — install the native EXO desktop app from its GitHub `.dmg` (like LM Studio),
/// which is far more robust than the source build. URL resolved from the latest release so it
/// never pins a stale version. `verified: false` — not yet e2e-verified as an OpenHydra provider.
fn exo_mac_app_recipe() -> Result<Recipe, String> {
    let url = latest_github_dmg_url("exo-explore/exo").ok_or(
        "Couldn't reach GitHub to find the latest EXO release — check your connection and retry. \
         (Or install Exo from source on Linux.)",
    )?;
    Ok(Recipe {
        engine: "exo",
        summary: "Download the native EXO macOS app (its official signed .dmg) and install it to \
                  /Applications — like LM Studio. Then launch EXO to expose an OpenAI API and join \
                  the P2P cluster."
            .into(),
        steps: vec![Step::DownloadRun { url, filename: "EXO.dmg".into() }],
        health: Health::Manual {
            note: "EXO installed to /Applications. Launch it to serve an OpenAI API + join the \
                   cluster; this card flips to running once it serves."
                .into(),
        },
        default_model: None,
        verified: false,
    })
}

/// Exo (Tier-2) — P2P inference sharded across your devices; runs on macOS (MLX) + Linux.
/// Installs from source (no wheels) and **requires Python ≥ 3.12**. Probe-then-install: needs
/// git; prefer `uv` (which *provides* Python 3.12), else a system Python ≥ 3.12, else block.
/// Windows → WSL2. `Health::Manual` (launch the `exo` server). `verified: false`.
fn exo_recipe(os: Os, variant: Variant) -> Result<Recipe, String> {
    match os {
        // macOS default = the native signed app (robust). `Cli` forces the headless source build
        // (git clone → uv venv 3.13 → pip) for running exo without the GUI.
        Os::Macos if variant != Variant::Cli => exo_mac_app_recipe(),
        Os::Windows => Err("Exo targets macOS + Linux — on Windows, run it under WSL2 (a Linux environment).".into()),
        // Linux, or macOS + Cli → source build.
        _ => exo_plan(os, &crate::hostinfo::probe_prereqs()),
    }
}

fn exo_plan(_os: Os, p: &crate::hostinfo::Prereqs) -> Result<Recipe, String> {
    if !has_program("git") {
        return Err("Exo installs from source — `git` is required. Install git, then retry.".into());
    }
    let dir = install_dir("exo");
    let src = dir.join("src").to_string_lossy().into_owned();
    let venv = dir.join("venv");
    let venv_s = venv.to_string_lossy().into_owned();
    let py = venv.join("bin").join("python").to_string_lossy().into_owned();
    const REPO: &str = "https://github.com/exo-explore/exo.git";
    let note = "Exo installed. Run its `exo` server to expose an OpenAI-compatible API and join \
                the P2P cluster; this card flips to running once it serves.";
    if p.uv {
        Ok(Recipe {
            engine: "exo",
            summary: format!(
                "Install Exo (shard models across your devices) from source into {}: git clone → \
                 uv venv (Python 3.13 — uv fetches it) → uv pip install. Then run `exo`.",
                dir.to_string_lossy()
            ),
            steps: vec![
                Step::Run { program: "git".into(), args: vec!["clone".into(), REPO.into(), src.clone()] },
                // Exo pins requires-python ==3.13.* — uv downloads 3.13 automatically if absent.
                Step::Run { program: "uv".into(), args: vec!["venv".into(), "--python".into(), "3.13".into(), venv_s] },
                Step::Run { program: "uv".into(), args: vec!["pip".into(), "install".into(), "--python".into(), py, src] },
            ],
            health: Health::Manual { note: note.into() },
            default_model: None,
            verified: false,
        })
    } else if python_is_313(&p.python) {
        Ok(Recipe {
            engine: "exo",
            summary: format!(
                "Install Exo from source into {} with your Python {} (git clone → venv → pip install). Then run `exo`.",
                dir.to_string_lossy(),
                p.python.as_deref().unwrap_or("3.13")
            ),
            steps: vec![
                Step::Run { program: "git".into(), args: vec!["clone".into(), REPO.into(), src.clone()] },
                Step::Run { program: "python3".into(), args: vec!["-m".into(), "venv".into(), venv_s] },
                Step::Run { program: py, args: vec!["-m".into(), "pip".into(), "install".into(), src] },
            ],
            health: Health::Manual { note: note.into() },
            default_model: None,
            verified: false,
        })
    } else {
        Err("Exo pins Python 3.13 (requires-python ==3.13.*). Install `uv` (astral.sh/uv — it \
             fetches Python 3.13 for you) or Python 3.13, then retry."
            .into())
    }
}

// ── Prereq gate (B3) ──

/// Resolve `program` to a runnable path. Returns the bare name when it's on `PATH`, else the first
/// matching absolute install dir. A GUI-launched macOS app inherits a minimal PATH
/// (`/usr/bin:/bin:/usr/sbin:/sbin`) that excludes Homebrew — so `Command::new("ollama")` fails with
/// `No such file or directory (os error 2)` even when Ollama is installed. Spawning the returned
/// absolute path sidesteps that. On Windows we only trust `where` (no fixed install dir).
pub fn resolve_program(program: &str) -> Option<PathBuf> {
    let probe = if cfg!(windows) { "where" } else { "which" };
    let on_path = Command::new(probe)
        .arg(program)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if on_path {
        return Some(PathBuf::from(program));
    }
    #[cfg(not(windows))]
    {
        for dir in ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"] {
            let cand = Path::new(dir).join(program);
            if cand.exists() {
                return Some(cand);
            }
        }
        if let Ok(home) = std::env::var("HOME") {
            let cand = Path::new(&home).join(".lmstudio/bin").join(program);
            if cand.exists() {
                return Some(cand);
            }
        }
    }
    None
}

/// Is `program` on PATH (or in a common absolute install dir)? Used to gate recipes that need a
/// package manager (macOS `brew`). See [`resolve_program`] for the GUI-PATH rationale.
pub fn has_program(program: &str) -> bool {
    resolve_program(program).is_some()
}

/// The prereqs a recipe needs, checked before any step runs. Returns the first blocking
/// problem (with an actionable message), or `None` when good to go.
pub fn prereq_blocker(engine: &str, os: Os) -> Option<String> {
    match (normalize_engine(engine), os) {
        // Ollama's official installer (macOS + Linux) needs `curl`; Windows uses PowerShell's
        // built-in `irm` (no prereq).
        ("ollama", Os::Linux | Os::Macos) => (!has_program("curl"))
            .then(|| "curl is required for the Ollama installer — install curl first".to_string()),
        ("llama.cpp", Os::Macos | Os::Linux) => (!has_program("brew")).then(|| {
            "Homebrew (brew) is required for the llama.cpp install — get it from brew.sh".to_string()
        }),
        // LM Studio: `curl` streams the installer to disk (bundled on macOS + Win10+/Linux).
        ("lm-studio", _) => (!has_program("curl"))
            .then(|| "curl is required to download the LM Studio installer — install curl first".to_string()),
        _ => None,
    }
}

/// Is the engine already present? (detect-first, so Install is idempotent). For llama.cpp
/// the *binary on PATH* counts even without a running model server; otherwise probe the port.
pub fn already_installed(engine: &str) -> bool {
    use openhydra_agent::adapter::HttpClient;
    // llama.cpp's install marker is the `llama-server` binary. Return that directly — do NOT fall
    // through to the :8080 health probe: llama-server's default port collides with OpenHydra's own
    // gateway (:8080), so the probe would answer from OpenHydra itself and false-positive.
    if normalize_engine(engine) == "llama.cpp" {
        return has_program("llama-server");
    }
    let url = health_url(engine);
    if url.is_empty() {
        return false;
    }
    openhydra_agent::ReqwestClient::new()
        .map(|c| c.get(url).is_ok())
        .unwrap_or(false)
}

// ── Executor (B1): run a recipe, streaming every line to the webview ──

/// The event payload the UI listens for on `install://progress`.
#[derive(Debug, Clone, Serialize)]
pub struct InstallEvent {
    /// Which install this belongs to (the engine id) — lets the UI route concurrent installs.
    pub engine: String,
    /// `log` (a streamed line), `phase` (a stage change), `download` (a progress tick),
    /// `done` (success) or `error`.
    pub kind: String,
    pub message: String,
    /// 0-100 on `download` ticks when the total size is known — drives a determinate bar.
    /// Absent on every other kind (and on downloads with no Content-Length).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub percent: Option<u8>,
}

fn emit(app: &AppHandle, engine: &str, kind: &str, message: impl Into<String>) {
    let _ = app.emit(
        "install://progress",
        InstallEvent {
            engine: engine.to_string(),
            kind: kind.into(),
            message: message.into(),
            percent: None,
        },
    );
}

/// Emit a download-progress tick: `percent` drives a determinate bar (None ⇒ indeterminate),
/// `message` is the "42% · 5.2 MB/s · ~18s left" line the UI shows under it.
fn emit_progress(app: &AppHandle, engine: &str, percent: Option<u8>, message: impl Into<String>) {
    let _ = app.emit(
        "install://progress",
        InstallEvent {
            engine: engine.to_string(),
            kind: "download".into(),
            message: message.into(),
            percent,
        },
    );
}

/// Emit a terminal success event (public so the command can close out a recipe with no model).
pub fn emit_done(app: &AppHandle, engine: &str, message: impl Into<String>) {
    emit(app, engine, "done", message);
}

/// Emit a terminal failure event so the UI surfaces the error inline.
pub fn emit_error(app: &AppHandle, engine: &str, message: impl Into<String>) {
    emit(app, engine, "error", message);
}

/// Full silent macOS install of a `.dmg`: mount → copy the `.app` into `/Applications` →
/// detach → launch. No drag step. The app is signed/notarized, so Gatekeeper still verifies it
/// on first launch. Always detaches, even if the copy fails.
#[cfg(target_os = "macos")]
fn install_dmg(app: &AppHandle, engine: &str, dmg: &Path) -> Result<(), String> {
    emit(app, engine, "phase", "mounting the disk image".to_string());
    let out = Command::new("hdiutil")
        .args(["attach", &dmg.to_string_lossy(), "-nobrowse", "-noverify"])
        .output()
        .map_err(|e| format!("hdiutil attach: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "could not mount the installer: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    // hdiutil prints the mount point as the last "/Volumes/…" path in its output.
    let stdout = String::from_utf8_lossy(&out.stdout);
    let mount = stdout
        .lines()
        .filter_map(|l| l.find("/Volumes/").map(|i| l[i..].trim_end().to_string()))
        .last()
        .ok_or("could not locate the mounted volume")?;
    // Copy + launch inside a closure so we ALWAYS detach afterwards.
    let installed = (|| -> Result<String, String> {
        let app_bundle = std::fs::read_dir(&mount)
            .map_err(|e| format!("read volume: {e}"))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.extension().is_some_and(|x| x == "app"))
            .ok_or("no application found in the disk image")?;
        let name = app_bundle.file_name().unwrap_or_default().to_string_lossy().into_owned();
        let dest = Path::new("/Applications").join(&name);
        emit(app, engine, "phase", format!("copying {name} → /Applications"));
        let _ = std::fs::remove_dir_all(&dest); // replace any stale copy
        // `ditto` preserves the code signature + xattrs (a plain `cp -R` can strip them).
        let cp = Command::new("ditto").arg(&app_bundle).arg(&dest).status().map_err(|e| e.to_string())?;
        if !cp.success() {
            return Err(format!(
                "couldn't copy {name} to /Applications — you may need admin rights, or drag it in manually"
            ));
        }
        Ok(name)
    })();
    let _ = Command::new("hdiutil").args(["detach", &mount, "-quiet"]).status();
    let name = installed?;
    emit(app, engine, "log", format!("installed {name} to /Applications"));
    emit(app, engine, "phase", format!("launching {name}"));
    let _ = Command::new("open").arg(Path::new("/Applications").join(&name)).status();
    Ok(())
}

fn fmt_bytes(n: u64) -> String {
    const U: [&str; 4] = ["B", "KB", "MB", "GB"];
    let (mut v, mut i) = (n as f64, 0);
    while v >= 1024.0 && i < U.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{n} B")
    } else {
        format!("{v:.1} {}", U[i])
    }
}

fn fmt_speed(bytes_per_s: f64) -> String {
    format!("{}/s", fmt_bytes(bytes_per_s.max(0.0) as u64))
}

fn fmt_eta(secs: u64) -> String {
    if secs == 0 {
        "estimating…".to_string()
    } else if secs < 60 {
        format!("~{secs}s left")
    } else {
        format!("~{}m {}s left", secs / 60, secs % 60)
    }
}

/// Download `url` to `path` with a determinate progress bar. Fetches the total via a redirect-
/// following HEAD, streams the body with `curl` in the background, and polls the growing file to
/// emit `download` ticks carrying percent + "42% · 5.2 MB/s · ~18s left". Falls back to an
/// indeterminate byte counter when the server withholds `Content-Length`. Blocking.
fn download_with_progress(app: &AppHandle, engine: &str, url: &str, path: &Path) -> Result<(), String> {
    let path_s = path.to_string_lossy().into_owned();
    // Total size from a HEAD that follows redirects (`-I -L`); some CDNs omit it → indeterminate.
    let total: Option<u64> = Command::new("curl")
        .args(["-sIL", url])
        .output()
        .ok()
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .filter(|l| l.to_ascii_lowercase().starts_with("content-length:"))
                .filter_map(|l| l.split(':').nth(1)?.trim().parse::<u64>().ok())
                .next_back()
        });
    let mut child = Command::new("curl")
        .args(["-fL", "--silent", "--show-error", url, "-o", &path_s])
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("curl {url}: {e}"))?;
    let mut last_len = 0u64;
    let mut last_at = std::time::Instant::now();
    loop {
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
            if !status.success() {
                let mut errs = String::new();
                if let Some(mut e) = child.stderr.take() {
                    use std::io::Read;
                    let _ = e.read_to_string(&mut errs);
                }
                return Err(format!("download failed ({status}) — {} {url}", errs.trim()));
            }
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(400));
        let cur = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let now = std::time::Instant::now();
        let dt = now.duration_since(last_at).as_secs_f64().max(0.001);
        let speed = cur.saturating_sub(last_len) as f64 / dt; // bytes/s over the last tick
        last_len = cur;
        last_at = now;
        match total {
            // Hold at 99% until curl actually exits, so the bar never claims done early.
            Some(t) if t > 0 => {
                let pct = ((cur as f64 / t as f64) * 100.0).clamp(0.0, 99.0) as u8;
                let eta = if speed > 1.0 { (t.saturating_sub(cur) as f64 / speed) as u64 } else { 0 };
                emit_progress(
                    app,
                    engine,
                    Some(pct),
                    format!("{pct}% · {} of {} · {} · {}", fmt_bytes(cur), fmt_bytes(t), fmt_speed(speed), fmt_eta(eta)),
                );
            }
            _ => emit_progress(app, engine, None, format!("{} downloaded · {}", fmt_bytes(cur), fmt_speed(speed))),
        }
    }
    emit_progress(app, engine, Some(100), "download complete".to_string());
    Ok(())
}

/// Run one `Step`, streaming its output. Blocking (call from `spawn_blocking`).
fn run_step(app: &AppHandle, engine: &str, step: &Step) -> Result<(), String> {
    match step {
        Step::Run { program, args } => {
            emit(app, engine, "phase", format!("running: {program} {}", args.join(" ")));
            let mut child = Command::new(program)
                .args(args)
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .map_err(|e| format!("spawn {program}: {e}"))?;
            // Stream stdout on this thread; drain stderr on another so neither pipe blocks.
            if let Some(err) = child.stderr.take() {
                let (app2, engine2) = (app.clone(), engine.to_string());
                std::thread::spawn(move || {
                    for line in std::io::BufReader::new(err).lines().map_while(Result::ok) {
                        emit(&app2, &engine2, "log", line);
                    }
                });
            }
            if let Some(out) = child.stdout.take() {
                for line in std::io::BufReader::new(out).lines().map_while(Result::ok) {
                    emit(app, engine, "log", line);
                }
            }
            let status = child.wait().map_err(|e| e.to_string())?;
            if !status.success() {
                return Err(format!("step failed ({status}): {program}"));
            }
            Ok(())
        }
        Step::DownloadRun { url, filename } => {
            let dir = install_dir(engine);
            std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
            let path = dir.join(filename);
            let path_s = path.to_string_lossy().into_owned();
            // Stream to disk with a determinate progress bar (follows the vendor 302; never buffers
            // a ~1 GB installer in RAM). `-f` fails on an HTTP error instead of saving an error page.
            emit(app, engine, "phase", "downloading".to_string());
            download_with_progress(app, engine, url, &path)?;
            // Install per OS. The installer/app is signed, so the OS verifies its signature
            // (macOS Gatekeeper notarization / Windows SmartScreen Authenticode) at run/launch.
            emit(app, engine, "phase", "installing".to_string());
            #[cfg(target_os = "macos")]
            {
                let _ = &path_s; // download used it; macOS installs from the mounted volume
                install_dmg(app, engine, &path)?;
            }
            #[cfg(target_os = "windows")]
            {
                emit(app, engine, "phase", "launching the installer".to_string());
                let run = Command::new("cmd")
                    .args(["/C", "start", "", &path_s])
                    .status()
                    .map_err(|e| format!("launch installer: {e}"))?;
                if !run.success() {
                    return Err(format!("installer exited with {run}"));
                }
                emit(app, engine, "log", "installer launched — follow its prompts".to_string());
            }
            #[cfg(all(unix, not(target_os = "macos")))]
            {
                // Linux AppImage: make it executable, then launch it.
                use std::os::unix::fs::PermissionsExt;
                if let Ok(md) = std::fs::metadata(&path) {
                    let mut perm = md.permissions();
                    perm.set_mode(0o755);
                    let _ = std::fs::set_permissions(&path, perm);
                }
                emit(app, engine, "phase", "launching the AppImage".to_string());
                let run = Command::new(&path_s)
                    .status()
                    .map_err(|e| format!("launch AppImage: {e}"))?;
                if !run.success() {
                    return Err(format!("AppImage exited with {run}"));
                }
                emit(app, engine, "log", "LM Studio launched".to_string());
            }
            Ok(())
        }
    }
}

/// Confirm the install per its [`Health`] policy: poll an HTTP endpoint (serving engines), or
/// wait for a binary to land on PATH (llama.cpp — installed but needs a model to serve).
fn wait_healthy(app: &AppHandle, engine: &str, health: &Health) -> Result<(), String> {
    use openhydra_agent::adapter::HttpClient;
    match health {
        Health::Http { url, timeout_s } => {
            emit(app, engine, "phase", format!("health-check {url}"));
            let client = openhydra_agent::ReqwestClient::new().map_err(|e| e.to_string())?;
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(*timeout_s);
            while std::time::Instant::now() < deadline {
                if client.get(url).is_ok() {
                    return Ok(());
                }
                std::thread::sleep(std::time::Duration::from_millis(750));
            }
            Err(format!("{engine} did not become healthy within {timeout_s}s"))
        }
        Health::ProgramOnPath { program } => {
            emit(app, engine, "phase", format!("confirming {program} is installed"));
            // A package manager may update PATH only for new shells — poll briefly.
            for _ in 0..8 {
                if has_program(program) {
                    return Ok(());
                }
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
            Err(format!(
                "{program} not found on PATH after install — open a new terminal, or check the \
                 install output above"
            ))
        }
        // Nothing to auto-verify (a GUI installer the user runs). The next-step note is
        // surfaced as the terminal `done` message (see `Recipe::completion_message`).
        Health::Manual { .. } => Ok(()),
    }
}

/// Locate the LM Studio `lms` CLI (bundled with the app — the first launch bootstraps it onto PATH
/// / `~/.lmstudio/bin/lms`), run `lms server start`, and confirm :1234. `wait_cli_secs` bounds how
/// long to wait for the CLI to appear (longer just after install, ~0 for an on-demand "Run").
fn start_lm_studio_core(wait_cli_secs: u64) -> Result<(), String> {
    use openhydra_agent::adapter::HttpClient;
    let candidate = std::env::var("HOME").map(|h| format!("{h}/.lmstudio/bin/lms")).unwrap_or_default();
    let mut lms: Option<String> = None;
    for _ in 0..wait_cli_secs.max(1) {
        if !candidate.is_empty() && Path::new(&candidate).exists() {
            lms = Some(candidate.clone());
            break;
        }
        if has_program("lms") {
            lms = Some("lms".to_string());
            break;
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
    let lms = lms.ok_or("LM Studio CLI (lms) not found yet — open LM Studio once, then retry.")?;
    // `lms server start` prints "Success! Server is now running on port 1234" and returns. The CLI
    // lingers after an app uninstall and will exit non-zero, so DON'T trust its exit alone — the
    // arbiter is whether :1234 actually serves.
    let out = Command::new(&lms).args(["server", "start"]).output().map_err(|e| format!("lms server start: {e}"))?;
    if let Ok(client) = openhydra_agent::ReqwestClient::new() {
        for _ in 0..12 {
            if client.get(health_url("lm-studio")).is_ok() {
                return Ok(());
            }
            std::thread::sleep(std::time::Duration::from_millis(750));
        }
    }
    let stderr = String::from_utf8_lossy(&out.stderr);
    Err(format!(
        "LM Studio server didn't come up on :1234 — is LM Studio installed? {}",
        stderr.trim()
    ))
}

/// Best-effort auto-start of LM Studio's server right after install, so install → usable is one
/// step. NEVER fails the install: on a fresh first-run where the CLI isn't ready, it just logs.
#[cfg(target_os = "macos")]
fn ensure_lm_studio_server(app: &AppHandle) {
    emit(app, "lm-studio", "phase", "starting the LM Studio local server (:1234)".to_string());
    match start_lm_studio_core(15) {
        Ok(()) => emit(app, "lm-studio", "log", "LM Studio server started on :1234.".to_string()),
        Err(e) => emit(app, "lm-studio", "log", format!("{e} Start it from Developer → Start Server.")),
    }
}

#[cfg(not(target_os = "macos"))]
fn ensure_lm_studio_server(_app: &AppHandle) {}

/// Is `engine` installed on disk (present regardless of whether its server is running)? Powers the
/// "Run" CTA for installed-but-idle engines — distinct from [`already_installed`], which probes the
/// *serving* port. Conservative: only engines with an unambiguous binary/app marker.
pub fn installed_on_disk(engine: &str) -> bool {
    // Detect a user's EXISTING install anywhere — CLI on PATH (has_program also checks Homebrew
    // dirs), a /Applications app bundle, a known venv, OR our own engines dir — so a pre-existing
    // install elsewhere isn't a false-negative. Markers are SUCCESS markers (a real binary/app),
    // never a bare dir a failed install might leave behind.
    match normalize_engine(engine) {
        "ollama" => has_program("ollama") || app_installed("Ollama"),
        "llama.cpp" => has_program("llama-server"),
        "lm-studio" => {
            // macOS: the `.app` is the canonical marker. Do NOT fall back to `has_program("lms")`
            // — that CLI in ~/.lmstudio/bin persists after the app is deleted (phantom install).
            #[cfg(target_os = "macos")]
            {
                app_installed("LM Studio")
            }
            #[cfg(not(target_os = "macos"))]
            {
                has_program("lms")
            }
        }
        // comfy-cli (source) clones ComfyUI INTO the workspace root (not a ComfyUI/ subdir), so the
        // marker is <workspace>/main.py; also accept the desktop app or a global `comfy` CLI.
        "comfyui" => {
            app_installed("ComfyUI")
                || has_program("comfy")
                || install_dir("comfyui").join("main.py").exists()
        }
        // The native app, a global `exo` CLI, or a source build's `exo` entrypoint.
        "exo" => {
            app_installed("EXO")
                || app_installed("Exo")
                || has_program("exo")
                || install_dir("exo").join("venv").join("bin").join("exo").exists()
        }
        // A global `vllm` CLI, the vllm-metal venv (~/.venv-vllm-metal), or a source venv of ours.
        "vllm" => {
            has_program("vllm")
                || std::env::var("HOME")
                    .map(|h| Path::new(&h).join(".venv-vllm-metal").join("bin").join("vllm").exists())
                    .unwrap_or(false)
                || install_dir("vllm").join("venv").join("bin").join("vllm").exists()
        }
        _ => false,
    }
}

/// Is a macOS app bundle `/Applications/<name>.app` installed? Always false off macOS.
fn app_installed(name: &str) -> bool {
    #[cfg(target_os = "macos")]
    {
        Path::new("/Applications").join(format!("{name}.app")).exists()
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = name;
        false
    }
}

/// Which engines can OpenHydra start on demand (self-serving, no model/cluster arg needed).
pub fn is_runnable(engine: &str) -> bool {
    matches!(normalize_engine(engine), "ollama" | "lm-studio")
}

/// Start an installed engine's local server so it flips from "installed" to "running". Blocking;
/// call from `spawn_blocking`. Returns a helpful error for engines that need a model/cluster arg.
pub fn run_engine(engine: &str) -> Result<(), String> {
    match normalize_engine(engine) {
        "ollama" => {
            // Resolve to an absolute path first: a GUI-launched macOS app has a minimal PATH that
            // excludes Homebrew/`/usr/local/bin`, so a bare `ollama` spawns `os error 2` even when
            // it's installed. See `resolve_program`.
            let bin = resolve_program("ollama").ok_or_else(|| {
                "Ollama isn't on this app's PATH. Open Ollama once from Applications, or install it \
                 from ollama.com, then try Run again."
                    .to_string()
            })?;
            // Idempotent: if the launch agent already serves :11434 this exits fast; harmless.
            Command::new(&bin)
                .arg("serve")
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .map_err(|e| format!("start ollama ({}): {e}", bin.display()))?;
            Ok(())
        }
        "lm-studio" => start_lm_studio_core(3),
        "llama.cpp" => Err("llama.cpp needs a model — run `llama-server -m <model.gguf>`; it appears here once it serves.".to_string()),
        "comfyui" => Err("Start ComfyUI from its folder — `comfy launch` (serves :8188).".to_string()),
        "exo" => Err("Start Exo from its folder — run `exo` in the Exo venv.".to_string()),
        "vllm" => Err("vLLM needs a model — run `vllm serve <model>`.".to_string()),
        other => Err(format!("Don't know how to start {other} automatically.")),
    }
}

/// Execute a resolved recipe end-to-end, emitting progress. Blocking.
pub fn run_recipe(app: &AppHandle, recipe: &Recipe) -> Result<(), String> {
    let engine = recipe.engine;
    for step in &recipe.steps {
        run_step(app, engine, step)?;
    }
    // LM Studio ships a GUI + a CLI server; auto-start the server so install → usable is one step.
    if engine == "lm-studio" {
        ensure_lm_studio_server(app);
    }
    wait_healthy(app, engine, &recipe.health)?;
    Ok(())
}

/// After the engine is healthy, pull + warm its `default_model` (roadmap D.5) so it shows in
/// Share immediately. Ollama-specific (its `/api/pull` + `keep_alive`); a no-op for engines
/// without a pull API. Streams pull progress as log events; best-effort warm.
pub fn pull_and_warm(app: &AppHandle, engine: &str, model: &str) -> Result<(), String> {
    if engine != "ollama" {
        return Ok(());
    }
    use openhydra_agent::adapter::HttpClient;
    let client = openhydra_agent::ReqwestClient::new().map_err(|e| e.to_string())?;
    emit(app, engine, "phase", format!("pulling {model} (first time can take minutes)"));
    // Stream the pull so a multi-GB download never trips a single-request timeout and the UI
    // sees progress. Emit only on a changed `status` so we don't flood the event channel.
    let body = serde_json::json!({ "name": model, "stream": true }).to_string();
    let lines = client
        .post_stream("http://127.0.0.1:11434/api/pull", &body)
        .map_err(|e| format!("pull {model}: {e}"))?;
    let mut last = String::new();
    for line in lines {
        let Ok(line) = line else { break };
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
            if let Some(err) = v.get("error").and_then(|e| e.as_str()) {
                return Err(format!("pull {model}: {err}"));
            }
            if let Some(status) = v.get("status").and_then(|s| s.as_str()) {
                if status != last {
                    emit(app, engine, "log", status.to_string());
                    last = status.to_string();
                }
            }
        }
    }
    // Warm the model into VRAM and pin it there (keep_alive=-1) so the first real request is fast.
    emit(app, engine, "phase", format!("warming {model}"));
    let warm = serde_json::json!({
        "model": model, "keep_alive": -1, "stream": false,
        "messages": [{ "role": "user", "content": "hi" }],
    })
    .to_string();
    let _ = client.post_json("http://127.0.0.1:11434/api/chat", &warm);
    emit(app, engine, "done", format!("{engine} ready — {model} pulled and warm"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ollama_linux_recipe_is_verified_and_pulls_a_default_model() {
        let r = recipe_for("ollama", Os::Linux, Accel::Auto).unwrap();
        assert_eq!(r.engine, "ollama");
        assert!(r.verified, "the Linux install.sh path is the verified flagship");
        assert_eq!(r.default_model, Some("qwen2.5:7b"));
        assert!(matches!(&r.health, Health::Http { url, .. } if url.contains(":11434")));
        assert!(matches!(r.steps.first(), Some(Step::Run { .. })));
    }

    #[test]
    fn llama_cpp_recipe_uses_homebrew_and_binary_health() {
        for os in [Os::Macos, Os::Linux] {
            // The UI spelling `llama-cpp` must normalise to the recipe.
            let r = recipe_for("llama-cpp", os, Accel::Auto).unwrap();
            assert_eq!(r.engine, "llama.cpp");
            assert!(r.default_model.is_none(), "llama.cpp is a server, not a model manager");
            assert!(
                matches!(&r.health, Health::ProgramOnPath { program } if program == "llama-server"),
                "success = the binary is on PATH (no model loaded yet)"
            );
            assert!(matches!(r.steps.first(), Some(Step::Run { program, .. }) if program == "brew"));
        }
        assert!(
            recipe_for("llama.cpp", Os::Windows, Accel::Auto).is_err(),
            "no one-click llama.cpp installer on Windows → guided"
        );
    }

    #[test]
    fn lm_studio_recipe_downloads_and_runs_the_verified_installer() {
        // Tier-3: "install" = download the OS-correct official installer and run it (the OS
        // code-signature gate verifies it at launch). Verified stable "latest" redirects.
        let mac = recipe_for("lm-studio", Os::Macos, Accel::Auto).unwrap();
        assert_eq!(mac.engine, "lm-studio");
        assert!(mac.default_model.is_none());
        assert!(matches!(&mac.health, Health::Manual { .. }));
        assert!(mac.verified);
        match mac.steps.first() {
            Some(Step::DownloadRun { url, filename }) => {
                assert_eq!(url, "https://lmstudio.ai/download/latest/darwin/arm64");
                assert!(filename.ends_with(".dmg"));
            }
            other => panic!("expected a DownloadRun step, got {other:?}"),
        }
        // `lmstudio`/`lm_studio` spellings normalise; Windows/Linux get their own installers.
        assert!(matches!(recipe_for("lmstudio", Os::Windows, Accel::Auto).unwrap().steps.first(),
            Some(Step::DownloadRun { url, filename }) if url.ends_with("/win32/x64") && filename.ends_with(".exe")));
        assert!(matches!(recipe_for("lm_studio", Os::Linux, Accel::Auto).unwrap().steps.first(),
            Some(Step::DownloadRun { url, filename }) if url.ends_with("/linux/x64") && filename.ends_with(".AppImage")));
        // The manual GUI next-step becomes the terminal message (not a misleading "serving").
        assert!(mac.completion_message().contains(":1234"));
    }

    #[test]
    fn vllm_plan_prefers_docker_then_uv_and_blocks_on_missing_prereqs() {
        use crate::hostinfo::Prereqs;
        let drv = || Some("535.104.05".to_string());

        // No GPU driver → hard block (a system prereq we can't install).
        let no_gpu = Prereqs { nvidia_driver: None, docker: true, uv: true, ..Default::default() };
        let e = vllm_plan(Os::Linux, &no_gpu).unwrap_err();
        assert!(e.contains("NVIDIA") && e.contains("nvidia-smi"), "actionable GPU message: {e}");

        // GPU + Docker → Docker path (image pull), Manual health w/ launch guidance.
        let docker = Prereqs { nvidia_driver: drv(), docker: true, uv: false, ..Default::default() };
        let r = vllm_plan(Os::Linux, &docker).unwrap();
        assert!(r.summary.contains("Docker"));
        assert!(matches!(r.steps.first(), Some(Step::Run { program, args })
            if program == "docker" && args.iter().any(|a| a.contains("vllm/vllm-openai"))));
        assert!(matches!(&r.health, Health::Manual { note } if note.contains("TRITON_ATTN")));
        assert!(!r.verified);

        // GPU + uv (no Docker) → uv venv + pip install path.
        let uv = Prereqs { nvidia_driver: drv(), docker: false, uv: true, ..Default::default() };
        let r = vllm_plan(Os::Linux, &uv).unwrap();
        assert!(r.summary.contains("uv") && r.summary.contains("venv"));
        assert_eq!(r.steps.len(), 2, "uv venv + uv pip install");
        assert!(matches!(&r.steps[1], Step::Run { args, .. } if args.iter().any(|a| a == "vllm")));

        // GPU but neither Docker nor uv → block asking for one of them.
        let bare = Prereqs { nvidia_driver: drv(), docker: false, uv: false, ..Default::default() };
        let e = vllm_plan(Os::Linux, &bare).unwrap_err();
        assert!(e.contains("Docker") && e.contains("uv"));
    }

    #[test]
    fn vllm_mac_uses_vllm_metal_and_windows_wsl2() {
        // macOS: the CPU source build doesn't compile on Apple Clang, so the Mac path is the
        // community vllm-metal plugin (MLX, prebuilt wheels via its install.sh). Gated on curl.
        if has_program("curl") {
            let mac = recipe_for("vllm", Os::Macos, Accel::Auto).unwrap();
            assert!(!mac.verified, "vllm-metal is community/experimental");
            assert!(
                matches!(mac.steps.first(), Some(Step::Run { args, .. }) if args.iter().any(|a| a.contains("vllm-metal"))),
                "mac vLLM installs vllm-metal"
            );
        }
        // Windows: no native build → WSL2.
        assert!(recipe_for("vllm", Os::Windows, Accel::Auto).unwrap_err().contains("WSL2"));
    }

    #[test]
    fn engine_id_normalises_across_spellings() {
        for s in ["llama.cpp", "llama-cpp", "llama_cpp", "llamacpp"] {
            assert_eq!(normalize_engine(s), "llama.cpp");
        }
        assert_eq!(normalize_engine("ollama"), "ollama");
        assert_eq!(normalize_engine("vllm"), "vllm");
        assert_eq!(normalize_engine("koboldcpp"), "", "unknown engines normalise to empty");
    }

    #[test]
    fn ollama_other_os_recipes_resolve_but_are_flagged_unverified() {
        for os in [Os::Macos, Os::Windows] {
            let r = recipe_for("ollama", os, Accel::Auto).unwrap();
            assert!(!r.verified, "{os:?} recipe must be flagged pending a real-target check");
        }
    }

    #[test]
    fn ollama_uses_official_installers_per_os() {
        // Verified against Ollama's README (2026-07): install.sh (mac+Linux), install.ps1 (Win).
        let lin = recipe_for("ollama", Os::Linux, Accel::Auto).unwrap();
        assert!(matches!(lin.steps.first(), Some(Step::Run { args, .. })
            if args.iter().any(|a| a.contains("install.sh"))));
        let mac = recipe_for("ollama", Os::Macos, Accel::Auto).unwrap();
        assert!(matches!(mac.steps.first(), Some(Step::Run { program, args })
            if program == "sh" && args.iter().any(|a| a.contains("install.sh"))),
            "macOS uses the official cross-platform install.sh, not brew");
        let win = recipe_for("ollama", Os::Windows, Accel::Auto).unwrap();
        assert!(matches!(win.steps.first(), Some(Step::Run { program, args })
            if program == "powershell" && args.iter().any(|a| a.contains("install.ps1"))),
            "Windows uses the official PowerShell installer, no placeholder-hash download");
    }

    #[test]
    fn unsupported_engine_is_an_error() {
        assert!(recipe_for("koboldcpp", Os::Linux, Accel::Auto).is_err());
        assert!(recipe_for("nonsense", Os::Linux, Accel::Auto).is_err());
    }

    #[test]
    fn parses_the_dmg_url_not_the_name_or_api_field() {
        // Compact JSON where the asset `name` and `url` also contain ".dmg"/https — the parser
        // must return the browser_download_url, not those decoys.
        let body = r#"{"assets":[{"url":"https://api.github.com/repos/exo-explore/exo/releases/assets/403520692","id":403520692,"name":"EXO-1.0.71.dmg","browser_download_url":"https://github.com/exo-explore/exo/releases/download/v1.0.71/EXO-1.0.71.dmg"}]}"#;
        assert_eq!(
            parse_github_dmg_url(body).as_deref(),
            Some("https://github.com/exo-explore/exo/releases/download/v1.0.71/EXO-1.0.71.dmg")
        );
        // No dmg asset → None.
        assert_eq!(parse_github_dmg_url(r#"{"assets":[{"browser_download_url":"https://x/app.zip"}]}"#), None);
    }

    #[test]
    fn only_self_serving_engines_are_runnable() {
        // Ollama + LM Studio start with no model/cluster arg → get a "Run" CTA.
        assert!(is_runnable("ollama"));
        assert!(is_runnable("lm-studio") && is_runnable("lmstudio"));
        // These need a model/cluster arg, so no auto-"Run".
        for e in ["llama.cpp", "vllm", "comfyui", "exo"] {
            assert!(!is_runnable(e), "{e} should not be runnable");
        }
    }

    #[test]
    fn run_engine_refuses_model_dependent_engines_with_guidance() {
        // Non-self-serving engines can't be started blind — return an actionable error, never Ok.
        for e in ["llama.cpp", "vllm", "comfyui", "exo"] {
            let r = run_engine(e);
            assert!(r.is_err(), "{e} must not silently 'start'");
            assert!(!r.unwrap_err().is_empty());
        }
    }

    #[test]
    fn python_version_gate() {
        // Exo pins ==3.13.* → only 3.13.x satisfies; 3.12 and 3.14 do NOT.
        assert!(python_is_313(&Some("3.13.1".into())));
        assert!(python_is_313(&Some("3.13.0".into())));
        assert!(!python_is_313(&Some("3.12.0".into())));
        assert!(!python_is_313(&Some("3.14.0".into())));
        assert!(!python_is_313(&Some("4.0.0".into())));
        assert!(!python_is_313(&None));
    }

    #[test]
    fn comfyui_plan_prefers_uv_then_pip_and_blocks_without_python() {
        use crate::hostinfo::Prereqs;
        // uv → a single `uvx --from comfy-cli comfy … install`.
        // macOS now installs the official desktop app (DownloadRun to the signed .dmg).
        let mac = recipe_for("comfyui", Os::Macos, Accel::Auto).unwrap();
        assert!(matches!(mac.steps.first(), Some(Step::DownloadRun { url, .. }) if url.contains("download.comfy.org")));
        // Linux uses comfy-cli via uv with the --cpu accelerator (explicit, for --skip-prompt).
        let uv = Prereqs { uv: true, python: Some("3.11.0".into()), ..Default::default() };
        let lin = comfyui_plan(Os::Linux, &uv).unwrap();
        assert!(matches!(lin.steps.first(), Some(Step::Run { program, args })
            if program == "uvx" && args.iter().any(|a| a == "comfy-cli") && args.iter().any(|a| a == "--cpu")));
        assert!(matches!(&lin.health, Health::Manual { note } if note.contains(":8188")));
        // python but no uv → pip fallback.
        let pip = Prereqs { uv: false, python: Some("3.10.0".into()), ..Default::default() };
        assert!(matches!(comfyui_plan(Os::Linux, &pip).unwrap().steps.first(),
            Some(Step::Run { program, .. }) if program == "pip"));
        // neither → block.
        assert!(comfyui_plan(Os::Linux, &Prereqs::default()).is_err());
    }

    #[test]
    fn exo_plan_gates_on_windows_git_and_python_313() {
        use crate::hostinfo::Prereqs;
        // Windows → WSL2 (short-circuits before the probe).
        assert!(recipe_for("exo", Os::Windows, Accel::Auto).unwrap_err().contains("WSL2"));
        // The remaining paths need git present on the test host.
        if has_program("git") {
            // uv → clone + `uv venv --python 3.13` + `uv pip install` (3 steps); uv fetches 3.13
            // even when the system Python is older, so python: 3.9 still takes the uv path.
            let uv = Prereqs { uv: true, python: Some("3.9.0".into()), ..Default::default() };
            let r = exo_plan(Os::Macos, &uv).unwrap();
            assert_eq!(r.steps.len(), 3);
            assert!(matches!(&r.steps[0], Step::Run { program, .. } if program == "git"));
            assert!(matches!(&r.steps[1], Step::Run { args, .. } if args.iter().any(|a| a == "3.13")));
            // no uv + Python ≠ 3.13 → block (Exo pins ==3.13.*).
            assert!(exo_plan(Os::Macos, &Prereqs { uv: false, python: Some("3.12.1".into()), ..Default::default() }).is_err());
            assert!(exo_plan(Os::Macos, &Prereqs { uv: false, python: Some("3.14.0".into()), ..Default::default() }).is_err());
            // no uv + Python 3.13.x → venv/pip path.
            let py = Prereqs { uv: false, python: Some("3.13.1".into()), ..Default::default() };
            assert!(matches!(exo_plan(Os::Macos, &py).unwrap().steps.get(1),
                Some(Step::Run { program, args }) if program == "python3" && args.iter().any(|a| a == "venv")));
        }
    }
}
