<!--
Licensed to the Apache Software Foundation (ASF) under one or more
contributor license agreements.  See the NOTICE file distributed with
this work for additional information regarding copyright ownership.
The ASF licenses this file to You under the Apache License, Version 2.0
(the "License"); you may not use this file except in compliance with
the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Contributing to Apache Mahout

Thank you for your interest in contributing to Apache Mahout!

This document describes **repository-wide** setup and workflow. For **subproject-specific** build, test, and development details, see the [Project-Specific Guides](#project-specific-guides) below.

## Table of Contents

- [Quick Start](#quick-start)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Project-Specific Guides](#project-specific-guides)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## Quick Start

### Prerequisites

- Python 3.10 (>=3.10,<3.14)
- [`uv`](https://docs.astral.sh/uv/) package manager
- Git

### Installation

1. **Install uv:**
   ```bash
   pip install uv
   ```

2. **Clone and install:**
   ```bash
   git clone https://github.com/apache/mahout.git
   cd mahout
   uv sync --group dev              # Core Qumat (no GPU required)
   # uv sync --group dev --extra qdp  # With QDP extension (requires CUDA GPU)
   ```
   Add `--extra qdp` if you need GPU-accelerated encoding or want to run QDP tests. QDP tests are auto-skipped if the extension is not installed.

3. **Set up pre-commit hooks:**
   ```bash
   pre-commit install
   ```

---

## Development Workflow

### 1. Open an Issue

Create a new issue on [GitHub](https://github.com/apache/mahout/issues) and discuss your idea with the community.

### 2. Create a Branch

```bash
git checkout -b your-feature-name
```

### 3. Make Changes

Make your changes, then commit (pre-commit hooks run automatically):

```bash
git add .
git commit -m "Description of your changes"
git push
```

### 4. Pre-commit Checks

Run pre-commit manually if needed:

```bash
pre-commit run              # Staged files only
pre-commit run --all-files  # All files
```

From the repo root you can also use:

```bash
make pre-commit
```

### 5. Create a Pull Request

Open a pull request on GitHub and follow the [pull request template](.github/PULL_REQUEST_TEMPLATE).

---

## Testing

Tests are unified under pytest in the `testing/` directory:

| Directory         | Description |
|-------------------|-------------|
| `testing/qumat/`  | Qumat quantum computing library tests |
| `testing/qdp/`    | Quantum Data Plane tests (GPU; auto-skipped if extension unavailable) |
| `testing/utils/`  | Shared test utilities and fixtures |

**Run all tests:**

```bash
make tests
```

You can also run subsets from the repo root:

| Command | Description |
|---------|-------------|
| `make test_rust` | QDP Rust unit tests (requires NVIDIA GPU; skipped if none detected) |
| `make test_python` | Python tests via pytest (syncs dev deps; builds QDP extension if GPU present, then runs full suite) |

See [testing/README.md](testing/README.md) for more options and details.

---

## Project-Specific Guides

Apache Mahout includes several subprojects. Use the root workflow above for issues, branches, and pull requests; use the guides below for **build, run, and test** in each area.

| Subproject | Guide | Description |
|------------|-------|-------------|
| **Qumat** | *(this document)* | Core Python library; use root Quick Start and [Testing](#testing). |
| **QDP** (Quantum Data Plane) | [qdp/CONTRIBUTING.md](qdp/CONTRIBUTING.md) | GPU-accelerated pipeline: Rust/CUDA, DevContainer, build, install, benchmarks, profiling. |
| **Website** | [website/CONTRIBUTING.md](website/CONTRIBUTING.md) | Docusaurus site: docs source in `/docs`, sync, versioning, deployment. |

---

## Troubleshooting

- **Pre-commit fails:** Run `pre-commit run --all-files` to see errors. Common fixes: `cargo fmt` (Rust), `cargo clippy` (Rust lint), and ensuring you use the repo venv (`uv run pre-commit` or `make pre-commit`).
- **Wrong Python or missing package:** Ensure the virtual environment is activated and you ran `uv sync --group dev` from the repo root. For QDP, see [qdp/CONTRIBUTING.md](qdp/CONTRIBUTING.md).

---

## References

- [testing/README.md](testing/README.md) — Test structure and commands
- [.github/PULL_REQUEST_TEMPLATE](.github/PULL_REQUEST_TEMPLATE) — PR description template
- [docs/](docs/) — Documentation source (used by the website)
