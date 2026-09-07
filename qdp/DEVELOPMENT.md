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

### Pre-push sanity: no-CUDA build

CI builds on a runner without `nvcc`. Without the toolkit no kernels are
embedded and every kernel lookup reports `KernelError::Unavailable`; the
CUDA Runtime symbols used for pinned memory and streams are stubbed. Before
pushing Rust changes, make sure that configuration still compiles:

```bash
cd qdp
QDP_NO_CUDA=1 CARGO_TARGET_DIR=target/nocuda cargo build --workspace --lib --release
cargo check --workspace --tests
cd ..
```

The separate target directory keeps the no-CUDA build from invalidating
your normal build cache.

## 4. Architecture: one encode path

QDP has three layers, and each has exactly one job:

| Layer | Directory | Owns |
|-------|-----------|------|
| Python | `qdp-python/qumat_qdp/` | Loader and benchmark API, backend selection, the PyTorch reference (`torch_ref.py`) and the Triton AMD path |
| Rust | `qdp-core/src/` | Readers, the pinned dual-stream upload pipeline, prefetching, state-vector memory, DLPack, and one small descriptor per encoding |
| CUDA | `qdp-kernels/src/*.cu` | Device code only: `extern "C" __global__` kernels that take a batch |

Every encoding is a `Kernel` (see `qdp-core/src/gpu/kernels/mod.rs`). The
engine has a single entry point, `QdpEngine::encode(input, shape, num_qubits,
encoding)`, and `input` says where the data lives: a host slice of `f64` or
`f32`, or a pointer already on the device (from a CUDA tensor, on its stream).
A single sample is a batch of one. Uploading, chunking, allocating the state
vector, converting precision and wrapping DLPack are shared and never written
per encoding.

Kernels are compiled to device-only fatbins by `qdp-kernels/build.rs`,
embedded in the crate, and loaded through the CUDA driver API on first use
(`qdp_kernels::registry`). There are no host launchers, no `extern "C"`
declarations to keep in sync by hand, and no stubs for builds without CUDA.

### Which files do I touch?

| I want to... | Edit | Language |
|--------------|------|----------|
| Add or change an encoding | `qdp-kernels/src/<name>.cu`, `qdp-core/src/gpu/kernels/<name>.rs`, `qumat_qdp/torch_ref.py` | CUDA, a little Rust, a little Python |
| Speed up a kernel | the one `.cu` file | CUDA |
| Add a data source or file format | `qdp-core/src/readers/` | Rust |
| Tune copy overlap, prefetch, pooling | `qdp-core/src/gpu/pipeline.rs`, `pipeline_runner.rs` | Rust |
| Add a loader option, benchmark, or API sugar | `qumat_qdp/` | Python |
| Support AMD for an encoding | `qumat_qdp/triton_amd.py` | Python (Triton) |
| Accept a new tensor type from Python | `qdp-python/src/input.rs` | Rust (PyO3) |

### Adding an encoding

1. **Device code.** Create `qdp-kernels/src/<name>.cu` with a batch kernel:

   ```cuda
   extern "C" __global__ void <name>_encode_batch_kernel(
       const double* __restrict__ input,      // num_samples * sample_size
       cuDoubleComplex* __restrict__ state,   // num_samples * (1 << num_qubits)
       size_t num_samples, size_t state_len, unsigned int num_qubits) { ... }
   ```

   Add an `_f32` variant (`const float*`, `cuComplex*`) if you want the
   float32 path. Add the file name to `KERNEL_SOURCES` in
   `qdp-kernels/build.rs`. Nothing else in `qdp-kernels` changes.

2. **Descriptor.** Create `qdp-core/src/gpu/kernels/<name>.rs` implementing
   `Kernel`: `name`, `sample_size(num_qubits)`, `supports(dtype)`, and
   `launch`, which names the symbol, picks a `LaunchConfig`, and passes
   arguments with `kernel_args!`. Copy `angle.rs` (about 80 lines) as the
   template. Override `validate_host` / `validate_device` only if the
   default finite check is not the right rule. Register the module in
   `kernels/mod.rs` and add a variant to `Encoding` in `types.rs`.

3. **Reference and tests.** Add `<name>_encode` to
   `qumat_qdp/torch_ref.py` and register it in `_ENCODERS`; add the name to
   `ENCODINGS` in `testing/qdp/test_parity.py`. The parity grid then checks
   every (precision, shape, input location) cell of your kernel against the
   reference. Add the encoding to `ENCODINGS` in
   `qdp-python/benchmark/baseline.py` so it is benchmarked.

Run `make check-kernels` in `qdp/` to confirm every embedded kernel symbol
resolves, and `make parity` from the repo root to run the grid.

### Guarding performance

Before an engine or pipeline change, capture a baseline; after it, compare:

```bash
uv run python qdp/qdp-python/benchmark/baseline.py capture --out /tmp/qdp-baseline.json
uv run python qdp/qdp-python/benchmark/baseline.py compare /tmp/qdp-baseline.json --tolerance 0.05
```

`compare` exits non-zero when any encoding's throughput drops or latency
rises by more than the tolerance.

## 5. Benchmarks

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

## 6. NVTX / nsys Profiling

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

## 7. Common Issues

- `nvcc: command not found`
  - CUDA toolkit is not installed or not in `PATH`.
- `No CUDA installed` during build
  - run `cargo clean` in `qdp/`, then rebuild via `maturin develop`.
- Import error for `_qdp`
  - ensure you are in root `.venv` and rerun `maturin develop`.
- Wrong GPU or OOM
  - use `CUDA_VISIBLE_DEVICES=0` and reduce qubits / batch size.
