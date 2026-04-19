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

# QDP Development Guide

This is the single practical guide for QDP contributors:

- setup development environment
- verify CUDA/GPU (`nvcc`, `nvidia-smi`)
- build Python extension
- run tests
- run benchmarks
- profile with NVTX + `nsys`

## 1. Prerequisites

- Linux + NVIDIA GPU
- CUDA toolkit (must provide `nvcc`)
- Python 3.10-3.12
- Rust toolchain
- `uv`

Quick check:

```bash
python --version
uv --version
cargo --version
nvcc --version
nvidia-smi
```

If you meant "nccv", this guide assumes you meant `nvcc`.

## 2. Unified Development Environment

Use one venv at repo root (`mahout/.venv`):

```bash
cd mahout
uv sync --group dev --extra qdp
source .venv/bin/activate
```

Build QDP Python extension in editable mode:

```bash
uv run --active maturin develop --manifest-path qdp/qdp-python/Cargo.toml
```

Sanity check import:

```bash
uv run python -c "import _qdp; import qumat.qdp as qdp; print('QDP import ok')"
```

## 3. Development Loop

Rebuild extension after Rust/PyO3 changes:

```bash
uv run --active maturin develop --manifest-path qdp/qdp-python/Cargo.toml
```

Run Rust tests:

```bash
cd qdp
cargo test --workspace
cd ..
```

**Encoding / pipeline dtype:** `qdp_core::Encoding::supports_f32` gates whether
`PipelineConfig::normalize()` keeps `dtype = Float32` for the synthetic pipeline. It reflects
**which encoders implement `encode_batch_f32` today** (currently amplitude only), not every
encoding that might eventually get a batch f32 path. When angle/basis gain real batch f32
support, widen `supports_f32` and adjust tests accordingly.

Run Python tests:

```bash
uv run pytest testing/qdp -v
uv run pytest testing/qdp_python -v
```

## 4. Benchmarks

From the repo root, set up and prepare benchmarks:

```bash
make setup-benchmark
```

This will:
1. Install benchmark dependencies into the unified root venv
2. Build the QDP extension (if GPU available)
3. Display instructions for running specific benchmarks

Then run benchmark scripts:

```bash
uv run --project qdp/qdp-python python qdp/qdp-python/benchmark/benchmark_e2e.py
uv run --project qdp/qdp-python python qdp/qdp-python/benchmark/benchmark_latency.py
uv run --project qdp/qdp-python python qdp/qdp-python/benchmark/benchmark_throughput.py
```

Examples:

```bash
uv run --project qdp/qdp-python python qdp/qdp-python/benchmark/benchmark_e2e.py --frameworks all
uv run --project qdp/qdp-python python qdp/qdp-python/benchmark/benchmark_throughput.py --qubits 16 --batches 200 --batch-size 64 --prefetch 16
```

For manual setup (if `make setup-benchmark` is not available):

```bash
source .venv/bin/activate
uv sync --project qdp/qdp-python --group benchmark --active
```

See [qdp/qdp-python/benchmark/README.md](qdp-python/benchmark/README.md) for detailed benchmark documentation.

## 5. NVTX / nsys Profiling

Build extension with observability feature:

```bash
uv run --active maturin develop --manifest-path qdp/qdp-python/Cargo.toml --features observability
```

Profile a benchmark:

```bash
nsys profile --trace=cuda,nvtx --force-overwrite=true -o qdp-e2e \
  uv run --project qdp/qdp-python python qdp/qdp-python/benchmark/benchmark_e2e.py
```

Read profiling summary:

```bash
nsys stats qdp-e2e.nsys-rep
```

## 6. Common Issues

- `nvcc: command not found`
  - CUDA toolkit is not installed or not in `PATH`.
- `No CUDA installed` during build
  - run `cargo clean` in `qdp/`, then rebuild via `maturin develop`.
- Import error for `_qdp`
  - ensure you are in root `.venv` and rerun `maturin develop`.
- Wrong GPU or OOM
  - use `CUDA_VISIBLE_DEVICES=0` and reduce qubits / batch size.
