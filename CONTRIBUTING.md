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

- Your OS, Python version, and hardware (CPU/GPU/RAM)
- Steps to reproduce
- Expected vs actual behaviour
- Relevant log output (redact any API keys or secrets)

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

### Prerequisites

- **Python 3.11+**
- **C/C++ compiler**: `xcode-select --install` (macOS) or `apt install build-essential libssl-dev` (Linux)
- **Apple Silicon**: `pip install "openhydra-network[mlx]"` for GPU acceleration
- **NVIDIA**: `pip install torch --index-url https://download.pytorch.org/whl/cu124`

```bash
# 1. Clone
git clone https://github.com/samtroberts/openhydra.git
cd openhydra

# 2. Create virtual environment
make venv
source .venv/bin/activate

# 3. Install all dependencies
make install

# 4. Compile protobuf definitions
make proto

# 5. Optional: install interactive shell extras
pip install -e ".[shell]"

# 6. Verify everything works
make test
```

### Optional extras

| Extra | Command | Purpose |
|---|---|---|
| KV compaction research | `pip install -e ".[kv-compaction]"` | transformers + scipy for Phase 1–4 |
| PostgreSQL ledger | `pip install -e ".[postgres]"` | psycopg2 for production ledger backend |
| Interactive shell | `pip install -e ".[shell]"` | prompt_toolkit for `openhydra-shell` |

---

## Running Tests

```bash
# Fast suite (~2 min, 982+ tests)
python -m pytest tests/ -q

# With coverage report
python -m pytest tests/ --cov=coordinator --cov=peer --cov=economy --cov=dht

# Full suite including real-tensor tests (requires PyTorch + HF model download)
OPENHYDRA_RUN_REAL_TENSOR_TEST=1 python -m pytest tests/ -q

# Single module
python -m pytest tests/test_kv_compaction.py -v

# Lint check (style + security rules)
pip install ruff && ruff check . --config pyproject.toml
```

All pull requests must pass the test suite with zero failures before review.

---

## Pull Request Guidelines

- **One concern per PR** — keep diffs focused and reviewable
- **Tests required** — new behaviour must have matching tests; aim for ≥80 % coverage on changed lines
- **No secrets** — never commit `.env`, `*.pem`, `*.key`, API tokens, or seed phrases
- **Descriptive commits** — prefer `feat(coordinator): add per-key rate-limit tiers` over `fix stuff`
- **Update the progress tracker** — if your PR completes a roadmap item, mark it in `OpenHydra_progress.md`

### Commit style (Conventional Commits)

```
feat(peer): add QUIC transport option
fix(dht): handle empty lookup response gracefully
docs(readme): clarify ARM deployment steps
test(kv_compaction): add auto-mode boundary tests
refactor(coordinator): extract rate-limit logic into middleware
```

---

## License Agreement

By contributing to OpenHydra you agree that your contributions will be licensed under the **Apache License 2.0**.

See [LICENSE](LICENSE) for the full license text.

---

## Good First Issues

Look for issues labelled **`good first issue`** — these are scoped tasks suitable for new contributors:

- Adding entries to `models.catalog.json`
- Improving error messages and logging
- Writing additional unit tests
- Documentation and example improvements
- SDK improvements (`sdk/python/`, `sdk/typescript/`)

---

Questions? Join the discussion in [GitHub Discussions](../../discussions) or open an issue.
