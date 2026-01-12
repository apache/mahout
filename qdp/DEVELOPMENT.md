# Development Guide

This guide explains how to develop and test mahout qdp.

## Prerequisites

> Note: Currently we only support Linux machines with NVIDIA GPU.

- Linux machine
- NVIDIA GPU with CUDA driver and toolkit installed
- Python 3.10
- Rust & Cargo

You can run the following to ensure you have successfully installed CUDA toolkit:

```sh
nvcc --version
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2025 NVIDIA Corporation
# Built on Wed_Aug_20_01:58:59_PM_PDT_2025
# Cuda compilation tools, release 13.0, V13.0.88
# Build cuda_13.0.r13.0/compiler.36424714_0
```

## Using DevContainer (Alternative Setup)

If you prefer a containerized development environment or want to avoid installing CUDA and development tools directly on your host machine, you can use the provided DevContainer configuration.

### Setup

1. Open the project in VS Code
2. When prompted, click "Reopen in Container" (or use Command Palette: `Dev Containers: Reopen in Container`)
3. VS Code will build and start the container using the configuration in [.devcontainer/devcontainer.json](.devcontainer/devcontainer.json)

The container includes:
- **Base image**: `nvidia/cuda:12.4.1-devel-ubuntu22.04` with full CUDA toolkit
- **Python 3.10**: Installed via DevContainer features
- **Rust & Cargo**: Installed automatically via [.devcontainer/setup.sh](.devcontainer/setup.sh)
- **Development tools**: uv, pre-commit hooks, and build essentials
- **GPU access**: The container has full access to all GPUs on the host
- **VS Code extensions**: Python, Rust Analyzer, and TOML support pre-installed

Once the container is running, you can proceed with the build and test steps as described in the sections below. All commands should be run inside the container terminal.

## Build

Execute the following command in the `qdp/` directory to build:

```sh
cargo build -p qdp-core
```

Or use the Makefile:

```bash
make build
```

To build with NVTX observability features enabled:

```bash
make build_nvtx_profile
```

## Profiling and Observability

### Profiling Rust Examples

To run NVTX profiling with nsys on Rust examples and view performance statistics:

```bash
make run_nvtx_profile                    # Uses default nvtx_profile example
make run_nvtx_profile EXAMPLE=my_example # Uses custom example
```

This will:
1. Build the specified example with observability features enabled
2. Run it with `nsys` to collect profiling data
3. Display profiling statistics automatically

### Profiling Python Benchmarks

To profile Python benchmarks with NVTX annotations, you need to install the package with profiling support:

```bash
make install_profile
```

This installs the Python package with observability features enabled. Then you can profile any Python script:

```bash
nsys profile python qdp-python/benchmark/benchmark_e2e.py
```

For more details on NVTX profiling, markers, and how to interpret results, please refer to [NVTX_USAGE docs](./docs/observability/NVTX_USAGE.md).

## Install as Python Package

The full documentation on how to use mahout qdp as a Python package is available in [qdp-python docs](./qdp-python/README.md). Please refer to the docs for more details on how to use the package. We will only show how to install it from source here.

First, create a Python environment with `uv`:

```bash
# add a uv python 3.11 environment
uv venv -p python3.11
source .venv/bin/activate
```

Then go to the `qdp-python/` directory and run the following commands to install mahout qdp Python package:

```bash
uv sync --group dev
uv run maturin develop
```

Alternatively, you can directly run the following command from the `qdp/` directory:

```bash
make install
```

To install the package with profiling support (includes NVTX observability features for performance analysis):

```bash
make install_profile
```

## Test

There are two types of tests in mahout qdp: unit tests and e2e tests (benchmark tests).

### Unit Tests

You can use the following make commands from the `qdp/` directory:

```bash
make test        # Run all unit tests (Python + Rust)
make test_python # Run Python tests only
make test_rust   # Run Rust tests only
```

Or follow the instructions in [test docs](./docs/test/README.md) to run unit tests manually.

### Benchmark Tests

The e2e and benchmark tests are located in the `qdp-python/benchmark` directory and are written in Python.

First, ensure you set up the Python environment and install the mahout qdp package following the [Install as Python package](#install-as-python-package) section.

To run all benchmark tests, use the make command from the `qdp/` directory:

```bash
make benchmark
```

This will:
1. Install the mahout qdp package if not already installed
2. Install benchmark dependencies (`uv sync --group benchmark`)
3. Run all benchmark tests

If you only want to run mahout qdp without running qiskit or pennylane benchmark tests, simply uninstall them:

```sh
uv pip uninstall qiskit pennylane
```

You can also run individual tests manually from the `qdp-python/benchmark/` directory:

```sh
# E2E test
python benchmark_e2e.py

# Benchmark test for Data-to-State latency
python benchmark_latency.py

# Benchmark test for dataloader throughput
python benchmark_throughput.py
```

## Troubleshooting

### Q: Python import fails after installation

A: Ensure you're using the correct Python environment where the package was installed. Verify with `python -c "import _qdp"`. Make sure you activated the virtual environment: `source .venv/bin/activate`.

### Q: Build fails with CUDA-related errors

A: Ensure CUDA toolkit is properly installed and `nvcc` is in PATH. Try `cargo clean` and rebuild.

### Q: I already installed CUDA driver and toolkit, making sure nvcc exists in PATH, but still get "no CUDA installed" warning

A: Run `cargo clean` to clean up the cache and try again.

### Q: Runtime CUDA errors like "invalid device ordinal" or "out of memory"

A: Check available GPUs with `nvidia-smi`. Verify GPU visibility with `echo $CUDA_VISIBLE_DEVICES`. If needed, specify a GPU: `CUDA_VISIBLE_DEVICES=0 python your_script.py`.

### Q: Benchmark tests fail or produce unexpected results

A: Ensure all dependencies are installed with `uv sync --group benchmark` (from `qdp/qdp-python`). Check GPU memory availability using `nvidia-smi`. If you don't need qiskit/pennylane comparisons, uninstall them as mentioned in the [E2e test section](#e2e-tests).

### Q: Pre-commit hooks fail

A: Run `pre-commit run --all-files` to see specific errors. Common fixes include running `cargo fmt` for formatting and `cargo clippy` for linting issues.

### Q: DevContainer fails to start

A: Ensure Docker and NVIDIA Container Toolkit are installed. Test with `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`. Try rebuilding without cache via Command Palette: "Dev Containers: Rebuild Container Without Cache".
