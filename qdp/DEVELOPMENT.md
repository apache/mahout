# Development Guide

This guide explains how to develop and test mahout qdp.

## Prerequisites

> Note: Currently we only support Linux machines with NVIDIA GPU.

- Linux machine
- NVIDIA GPU with CUDA driver and toolkit installed

You can run the following to ensure you have successfully installed CUDA toolkit:

```sh
nvcc --version
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2025 NVIDIA Corporation
# Built on Wed_Aug_20_01:58:59_PM_PDT_2025
# Cuda compilation tools, release 13.0, V13.0.88
# Build cuda_13.0.r13.0/compiler.36424714_0
```

## Build

Execute the following command in the `qdp/` directory to build

```sh
cargo build -p qdp-core
```

To build with NVTX enabled, please refer to [NVTX_USAGE docs](./docs/observability/NVTX_USAGE.md).

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

## Test

There are two types of tests in mahout qdp: unit tests and e2e tests (benchmark tests).

### Unit Tests

You can simply follow the instructions in [test docs](./docs/test/README.md) to run unit tests.

### E2e Tests

The e2e and benchmark tests are located in the `benchmark` directory and are written in Python. To run them, please ensure you set up the Python environment and install the mahout qdp package following the [Install as Python package](#install-as-python-package) section.

Then, go to the `benchmark/` directory, where all e2e and benchmark tests are located, and install the requirements needed for testing:

```sh
uv pip install -r requirements.txt
```

If you only want to run mahout qdp without running qiskit or pennylane benchmark tests, simply uninstall them:

```sh
uv pip uninstall qiskit pennylane
```

Then, run the tests:

```sh
# benchmark test for dataloader throughput
python benchmark_dataloader_throughput.py

# e2e test
python benchmark_e2e.py
```

## Troubleshooting

### Q: I already installed CUDA driver and toolkit, making sure nvcc exists in PATH, but still get "no CUDA installed" warning

A: Run `cargo clean` to clean up the cache and try again
