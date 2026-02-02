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

# Contributing to QDP (Quantum Data Plane)

This guide covers **QDP-specific** build, test, install, and profiling. For repository-wide workflow (issues, branches, pull requests, pre-commit), see the root [CONTRIBUTING.md](../CONTRIBUTING.md).

## Prerequisites

- Linux machine (QDP currently targets Linux with NVIDIA GPU)
- NVIDIA GPU with CUDA driver and toolkit installed
- Python 3.10 (>=3.10,<3.14)
- Rust & Cargo

Verify CUDA:

```bash
nvcc --version
```

## Setup Options

### Option 1: Local Setup

From the repository root, follow [Quick Start](../CONTRIBUTING.md#quick-start) and install with QDP:

```bash
uv sync --group dev --extra qdp
```

### Option 2: DevContainer (Recommended)

1. Open the project in VS Code.
2. When prompted, click **Reopen in Container** (or use Command Palette: *Dev Containers: Reopen in Container*).
3. The container provides: CUDA toolkit, Python 3.10, Rust, development tools, and GPU access.

All commands below can be run inside the container.

## Build

From the **`qdp/`** directory:

```bash
cargo build -p qdp-core
# or
make build
```

With NVTX observability (for profiling):

```bash
make build_nvtx_profile
```

## Install as Python Package

From the **repository root** you can install the main project with QDP:

```bash
uv sync --group dev --extra qdp
```

To develop the QDP Python bindings in place (e.g. for testing or benchmarks), from **`qdp/qdp-python/`**:

```bash
uv sync --group dev
uv run maturin develop
```

Or from **`qdp/`**:

```bash
make install
```

To install with profiling support (NVTX):

```bash
make install_profile
```

## Test

### Rust unit tests

From **`qdp/`**:

```bash
make test_rust
# or
cargo test --workspace
```

### Python tests (unified suite)

From the **repository root**:

```bash
make tests
```

QDP-related tests live under `testing/qdp/` and are auto-skipped if the QDP extension or a suitable GPU is not available. See [testing/README.md](../testing/README.md).

### Benchmarks (e2e)

Benchmarks are in `qdp-python/benchmark/`. From **`qdp/`**:

```bash
make benchmark
```

This installs the package (if needed), benchmark dependencies, and runs the benchmark suite. To run only the QDP benchmarks (no qiskit/pennylane), uninstall those packages in your venv.

Manual run from **`qdp/qdp-python/benchmark/`** (after installing the package):

```bash
python benchmark_e2e.py
python benchmark_latency.py
python benchmark_throughput.py
```

## Profiling and Observability

### Rust examples with NVTX

From **`qdp/`**:

```bash
make run_nvtx_profile                    # default example: nvtx_profile
make run_nvtx_profile EXAMPLE=my_example  # custom example
```

This builds the example with observability, runs it under `nsys`, and prints profiling stats.

### Python benchmarks with NVTX

Install with profiling support, then run under nsys:

```bash
make install_profile
nsys profile python qdp-python/benchmark/benchmark_e2e.py
```

See [docs/observability/NVTX_USAGE.md](docs/observability/NVTX_USAGE.md) for details.

## Troubleshooting

| Problem | Suggestion |
|--------|------------|
| Python import fails after install | Use the same venv where the package was installed; check with `python -c "import _qdp"`. Activate the venv: `source .venv/bin/activate`. |
| Build fails with CUDA errors | Ensure CUDA toolkit is installed and `nvcc` is in PATH. Try `cargo clean` and rebuild. |
| "No CUDA installed" despite having nvcc | Run `cargo clean` and build again. |
| Runtime: "invalid device ordinal" or "out of memory" | Check GPUs with `nvidia-smi`, visibility with `echo $CUDA_VISIBLE_DEVICES`. Pin device: `CUDA_VISIBLE_DEVICES=0 python your_script.py`. |
| Benchmark failures or odd results | Install deps: `uv sync --group benchmark` (from `qdp/qdp-python`). Check GPU memory. Uninstall qiskit/pennylane if you only need QDP benchmarks. |
| Pre-commit hooks fail | Run `pre-commit run --all-files`; fix formatting with `cargo fmt`, lint with `cargo clippy`. |
| DevContainer won’t start | Ensure Docker and NVIDIA Container Toolkit are installed. Test: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`. Rebuild without cache: *Dev Containers: Rebuild Container Without Cache*. |

## References

- [qdp-python/README.md](qdp-python/README.md) — Package usage
- [docs/observability/NVTX_USAGE.md](docs/observability/NVTX_USAGE.md) — NVTX profiling
- [docs/test/README.md](docs/test/README.md) — QDP test layout and commands
