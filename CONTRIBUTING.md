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

# Contributing to Apache Mahout (Qumat)

Thank you for your interest in contributing to Apache Mahout!

## Table of Contents

- [Quick Start](#quick-start)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [QDP Development](#qdp-development)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## Quick Start

### Prerequisites

- Python 3.10 (>=3.10,<3.14)
- `uv` package manager
- Git

### Installation

1. **Install uv:**
   ```bash
   pip install uv
   ```
   Or follow the [uv documentation](https://docs.astral.sh/uv/).

2. **Clone and install:**
   ```bash
   git clone https://github.com/apache/mahout.git
   cd mahout
   uv sync --group dev              # Core Qumat (no GPU required)
   # uv sync --group dev --extra qdp  # With QDP extension (requires CUDA GPU)
   ```

   **Note:** Add `--extra qdp` if you need GPU-accelerated encoding or want to run QDP tests. QDP tests are automatically skipped if the extension is not installed.

3. **Set up pre-commit hooks:**
   ```bash
   pre-commit install
   ```

---

## Development Workflow

### 1. Open an Issue

Create a new issue in [GitHub](https://github.com/apache/mahout/issues) and discuss your ideas with the community.

### 2. Create a Branch

```bash
git checkout -b your-feature-name
```

### 3. Make Changes

Make your changes, then commit (pre-commit hooks will run automatically):

```bash
git add .
git commit -m "Description of your changes"
git push
```

### 4. Pre-commit Checks

Run pre-commit hooks:

```bash
pre-commit run              # Check staged files
pre-commit run --all-files  # Check all files
```

### 5. Create a Pull Request

Create a pull request on GitHub. Follow the [pull request template](.github/PULL_REQUEST_TEMPLATE) to provide a detailed description.

## Testing

The project uses a unified test workflow with pytest. Tests are organized in the `testing/` directory.

### Test Structure

- `testing/qumat/` - Tests for the Qumat quantum computing library
- `testing/qdp/` - Tests for the Quantum Data Plane (GPU-accelerated, auto-skipped if extension unavailable)
- `testing/utils/` - Shared test utilities and helpers
- `testing/conftest.py` - Pytest configuration with shared fixtures

To run all tests:
```
make tests
```

See [testing/README.md](testing/README.md) for detailed testing documentation.

## QDP Development

> **Note:** QDP development requires Linux with NVIDIA GPU and CUDA toolkit.

### Prerequisites

- Linux machine
- NVIDIA GPU with CUDA driver and toolkit installed
- Python 3.10 (>=3.10,<3.14)
- Rust & Cargo

Verify CUDA installation:
```bash
nvcc --version
```

### Setup Options

**Option 1: Local Setup**

Follow the [Quick Start](#quick-start) installation steps.

**Option 2: DevContainer (Recommended)**

1. Open the project in VS Code
2. Click "Reopen in Container" when prompted
3. Container includes: CUDA toolkit, Python 3.10, Rust, development tools, GPU access

### Building

From the `qdp/` directory:

```bash
cargo build -p qdp-core    # Build Rust core
make build                 # Or use Makefile
make build_nvtx_profile    # With NVTX observability features
```

Or run pre-commit with makefile style (that ensures you uses `pre-commit` in uv's venv)
```bash
make pre-commit
```

### 2.5 Create a Pull Request

From `qdp/qdp-python/`:

## 3. Website Development

The website is built with [Docusaurus](https://docusaurus.io/). The `/docs` directory is the source of truth for documentation.

### Local Development

```bash
cd website-new
npm install
npm run start
```

This starts a local development server at `http://localhost:3000` with hot reload.

### Building

```bash
cd website-new
npm run build
```

The build syncs documentation from `/docs` and `/qdp/docs` automatically.

### Documentation Workflow

1. Edit documentation in the `/docs` directory (not `website-new/docs`)
2. Run `npm run sync` in `website-new/` to update the website docs
3. The sync runs automatically during `npm run start` and `npm run build`

### Creating a Version Snapshot

When releasing a new version, snapshot the current documentation:

```bash
cd website-new
npm run docusaurus docs:version X.Y
```

## 4. Project Structure

- `qumat/` - Core library code
- `qdp/` - QDP (Quantum Data Plane)
- `docs/` - Documentation (source of truth)
- `examples/` - Examples and Jupyter notebooks
- `testing/` - Test files (using pytest)
- `website-new/` - Website source code (using Docusaurus)
