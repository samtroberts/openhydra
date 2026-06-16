# Contributing to OpenHydra

Thank you for your interest in OpenHydra! We welcome contributions of all kinds — bug reports, feature requests, documentation, operator tooling, and code.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Pull Request Guidelines](#pull-request-guidelines)
- [License Agreement](#license-agreement)
- [Good First Issues](#good-first-issues)

---

## Code of Conduct

OpenHydra follows the [Contributor Covenant](https://www.contributor-covenant.org/) Code of Conduct. Be respectful, inclusive, and constructive. Harassment of any kind will not be tolerated.

---

## How to Contribute

### Reporting Bugs

Open a [GitHub Issue](../../issues/new?template=bug_report.md) and include:

- Your OS, Rust toolchain version (`rustc --version`), and which inference engine you ran (Ollama / vLLM / llama.cpp / …)
- Steps to reproduce
- Expected vs actual behaviour
- Relevant log output — run with `RUST_LOG=info` (or `debug`) and redact any API keys or secrets

### Requesting Features

Open a [GitHub Issue](../../issues/new?template=feature_request.md) with:

- A clear description of the use case
- How it fits the decentralised inference model
- Any prior art or related work

### Submitting Code

1. Fork the repository and clone your fork
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Make your changes (see [Development Setup](#development-setup))
4. Add or update tests
5. Push and open a Pull Request against `main`

---

## Development Setup

OpenHydra is a pure-Rust workspace — three crates (`network/`, `protocol/`, `agent/`) and a single binary, `openhydra-agent`. There is no Python in any role.

### Prerequisites

- **Rust toolchain** (stable) — install via <https://rustup.rs>
- **A C/C++ compiler** — `xcode-select --install` (macOS) or `apt install build-essential libssl-dev pkg-config` (Linux)
- **An inference engine** for end-to-end testing (optional for unit tests) — e.g. [Ollama](https://ollama.com), vLLM, LM Studio, or llama.cpp

```bash
# 1. Clone
git clone https://github.com/samtroberts/openhydra.git
cd openhydra

# 2. Build the agent (links no Python)
cargo build -p openhydra-agent           # or: make build

# 3. Run the test suite
make test                                 # cargo test --workspace
```

The workspace is pure Rust — no Python toolchain is required.

---

## Running Tests

```bash
# Whole workspace
make test
# equivalently:
cargo test --workspace --no-default-features

# A single crate
cargo test -p openhydra-agent --no-default-features

# Lint (clippy, warnings as errors) and format
make clippy        # cargo clippy --workspace --no-default-features --all-targets -- -D warnings
make fmt           # cargo fmt --all
```

All pull requests must pass `make test` and `make clippy` with zero failures before review.

---

## Pull Request Guidelines

- **One concern per PR** — keep diffs focused and reviewable
- **Tests required** — new behaviour must have matching tests
- **No secrets** — never commit `.env`, `*.pem`, `*.key`, API tokens, or seed phrases
- **Descriptive commits** — prefer `feat(agent): add per-key rate-limit tiers` over `fix stuff`

### Commit style (Conventional Commits)

```
feat(agent): add an LM Studio engine adapter
fix(network): handle empty Kademlia lookup response gracefully
docs(readme): clarify the provider/gateway split
test(protocol): add receipt round-trip boundary tests
refactor(network): extract relay-reservation logic
```

Scopes map to the crates: `agent`, `network`, `protocol` (plus `docs`, `ops`, `ci`).

---

## License Agreement

By contributing to OpenHydra you agree that your contributions will be licensed under the **Apache License 2.0**.

See [LICENSE](LICENSE) for the full license text.

---

## Good First Issues

Look for issues labelled **`good first issue`** — these are scoped tasks suitable for new contributors:

- Adding a new engine adapter under `agent/src/adapters/`
- Improving error messages and `tracing` diagnostics
- Writing additional unit tests
- Documentation and example improvements

---

Questions? Join the discussion in [GitHub Discussions](../../discussions) or open an issue.
