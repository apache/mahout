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

### Running Tests

```bash
# Run all tests
pytest

# Run specific test directory
pytest testing/qumat/
pytest testing/qdp/

# Run with verbose output
pytest -v
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

### Installing Python Package

From `qdp/qdp-python/`:

```bash
uv sync --group dev
uv run maturin develop
```

Or from `qdp/`:

```bash
make install              # Standard installation
make install_profile      # With profiling support
```

### Testing

**Unit Tests:**
```bash
# From qdp/ directory
make test        # All tests (Python + Rust)
make test_python # Python only
make test_rust   # Rust only
```

**Benchmark Tests:**
```bash
make benchmark
```

See [qdp/docs/test/README.md](qdp/docs/test/README.md) for detailed testing documentation.

### Profiling

**Rust Examples:**
```bash
make run_nvtx_profile                    # Default example
make run_nvtx_profile EXAMPLE=my_example # Custom example
```

**Python Benchmarks:**
```bash
make install_profile
nsys profile python qdp-python/benchmark/benchmark_e2e.py
```

See [qdp/docs/observability/NVTX_USAGE.md](qdp/docs/observability/NVTX_USAGE.md) for details.

## Troubleshooting

### Python Import Fails

**Problem:** `ModuleNotFoundError: No module named '_qdp'`

**Solution:**
- Verify: `python -c "import _qdp"`
- Ensure virtual environment is activated: `source .venv/bin/activate`
- Reinstall: `cd qdp/qdp-python && uv run maturin develop`

### Build Fails with CUDA Errors

**Problem:** CUDA-related build errors

**Solution:**
- Ensure `nvcc` is in PATH: `which nvcc`
- Clean and rebuild: `cargo clean && cargo build -p qdp-core`

### Runtime CUDA Errors

**Problem:** "invalid device ordinal" or "out of memory"

**Solution:**
- Check available GPUs: `nvidia-smi`
- Verify GPU visibility: `echo $CUDA_VISIBLE_DEVICES`
- Set specific GPU: `CUDA_VISIBLE_DEVICES=0 python your_script.py`

### Benchmark Tests Fail

**Problem:** Benchmark tests produce errors or unexpected results

**Solution:**
- Install dependencies: `cd qdp/qdp-python && uv sync --group benchmark`
- Check GPU memory: `nvidia-smi`
- Uninstall optional dependencies if needed: `uv pip uninstall qiskit pennylane`

### DevContainer Fails to Start

**Problem:** Container won't start or can't access GPU

**Solution:**
- Ensure Docker and NVIDIA Container Toolkit are installed
- Test GPU access: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`
- Rebuild container: VS Code Command Palette → "Dev Containers: Rebuild Container Without Cache"

## References

### Project Structure

- `qumat/` - Core library code
- `qdp/` - QDP (Quantum Data Plane)
- `docs/` - Documentation
- `examples/` - Examples and Jupyter notebooks
- `testing/` - Test files (using pytest)
- `website/` - Website source code (using Jekyll)

### Development Documents

- **Testing Documentation**: See [testing/README.md](testing/README.md)
- **QDP Test Documentation**: See [qdp/docs/test/README.md](qdp/docs/test/README.md)
- **NVTX Profiling**: See [qdp/docs/observability/NVTX_USAGE.md](qdp/docs/observability/NVTX_USAGE.md)
