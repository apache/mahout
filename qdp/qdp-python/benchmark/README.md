# Benchmarks

This directory contains Python benchmarks for Mahout QDP. There are three main
scripts:

- `benchmark_e2e.py`: end-to-end latency from disk to GPU VRAM (includes IO,
  normalization, encoding, transfer, and a dummy forward pass).
- `benchmark_throughput.py`: DataLoader-style throughput benchmark
  that measures vectors/sec across Mahout, PennyLane, and Qiskit.
- `benchmark_latency.py`: Data-to-State latency benchmark (CPU RAM -> GPU VRAM).

## Quick Start

From the repo root, the easiest way to set up the benchmark environment is:

```bash
make benchmark
```

This will:
1. Set up the benchmark environment in the unified root venv (`mahout/.venv`)
2. Attempt to build the QDP GPU extension (skipped on CPU-only systems)
3. Display instructions for running specific benchmarks manually

> Note: These benchmarks require an NVIDIA GPU with compatible CUDA drivers. On
> CPU-only systems, `make benchmark` will only prepare the environment and print
> instructions; the GPU benchmarks themselves will not run.

To run individual benchmarks after setup:

```bash
uv run --project qdp/qdp-python python qdp/qdp-python/benchmark/benchmark_e2e.py
uv run --project qdp/qdp-python python qdp/qdp-python/benchmark/benchmark_latency.py
uv run --project qdp/qdp-python python qdp/qdp-python/benchmark/benchmark_throughput.py
```

This keeps all benchmark dependencies in the unified repo root venv (`mahout/.venv`).

## Manual Setup

If you prefer to set up manually (or if `make benchmark` is not available):

```bash
# First-time setup: install dependencies and build QDP extension
uv sync --group dev --extra qdp
source .venv/bin/activate
uv sync --project qdp/qdp-python --group benchmark --active
unset CONDA_PREFIX && uv run --active maturin develop --manifest-path qdp/qdp-python/Cargo.toml
```

Then run benchmarks with `uv run --project qdp/qdp-python python ...`.

## E2E Benchmark (Disk -> GPU)

```bash
uv run --project qdp/qdp-python python qdp/qdp-python/benchmark/benchmark_e2e.py
```

Additional options:

```bash
uv run --project qdp/qdp-python python qdp/qdp-python/benchmark/benchmark_e2e.py --qubits 16 --samples 200 --frameworks mahout-parquet mahout-arrow
uv run --project qdp/qdp-python python qdp/qdp-python/benchmark/benchmark_e2e.py --frameworks all
uv run --project qdp/qdp-python python qdp/qdp-python/benchmark/benchmark_e2e.py --encoding-method basis
```

Notes:

- `--frameworks` accepts a space-separated list or `all`.
  Options: `mahout-parquet`, `mahout-arrow`, `pennylane`, `qiskit`.
- `--encoding-method` selects the encoding method: `amplitude` (default) or `basis`.
- The script writes `final_benchmark_data.parquet` and
  `final_benchmark_data.arrow` in the current working directory and overwrites
  them on each run.
- If multiple frameworks run, the script compares output states for
  correctness at the end.

## Data-to-State Latency Benchmark

```bash
uv run --project qdp/qdp-python python qdp/qdp-python/benchmark/benchmark_latency.py --qubits 16 --batches 200 --batch-size 64 --prefetch 16
uv run --project qdp/qdp-python python qdp/qdp-python/benchmark/benchmark_latency.py --frameworks mahout,pennylane
uv run --project qdp/qdp-python python qdp/qdp-python/benchmark/benchmark_latency.py --encoding-method basis
```

Notes:

- `--frameworks` is a comma-separated list or `all`.
  Options: `mahout`, `pennylane`, `qiskit-init`, `qiskit-statevector`.
- `--encoding-method` selects the encoding method: `amplitude` (default), `angle`, `basis`, `iqp`, or `iqp-z`.
- The latency test reports average milliseconds per vector.
- Flags:
  - `--qubits`: controls the input length together with `--encoding-method`.
    Amplitude uses `2^qubits`, angle and `iqp-z` use `qubits`, basis uses one basis index,
    and `iqp` uses `qubits + qubits*(qubits-1)/2`.
  - `--batches`: number of host-side batches to stream.
  - `--batch-size`: vectors per batch; raises total samples (`batches * batch-size`).
  - `--prefetch`: CPU queue depth; higher values help keep the pipeline fed.
- See `qdp/qdp-python/benchmark/benchmark_latency.md` for details and example output.

## DataLoader Throughput Benchmark

Simulates a typical QML training loop by continuously loading batches of 64
vectors (default). Goal: demonstrate that QDP can saturate GPU utilization and
avoid the "starvation" often seen in hybrid training loops.

See `qdp/qdp-python/benchmark/benchmark_throughput.md` for details and example
output.

```bash
uv run --project qdp/qdp-python python qdp/qdp-python/benchmark/benchmark_throughput.py --qubits 16 --batches 200 --batch-size 64 --prefetch 16
uv run --project qdp/qdp-python python qdp/qdp-python/benchmark/benchmark_throughput.py --frameworks mahout,pennylane
uv run --project qdp/qdp-python python qdp/qdp-python/benchmark/benchmark_throughput.py --encoding-method basis
```

Notes:

- `--frameworks` is a comma-separated list or `all`.
  Options: `mahout`, `pennylane`, `qiskit`.
- `--encoding-method` selects the encoding method: `amplitude` (default), `angle`, `basis`, `iqp`, or `iqp-z`.
- For synthetic inputs, amplitude uses `2^qubits`, angle and `iqp-z` use `qubits`,
  basis uses one basis index, and `iqp` uses `qubits + qubits*(qubits-1)/2`.
- Throughput is reported in vectors/sec (higher is better).

## Dependency Notes

- Qiskit and PennyLane are optional. If they are not installed, their benchmark
  legs are skipped automatically.
- For Mahout-only runs, you can uninstall the competitor frameworks:
  `uv pip uninstall qiskit pennylane`.

### We can also run benchmarks on colab notebooks(without owning a GPU)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/apache/mahout/blob/dev-qdp/qdp/qdp-python/benchmark/notebooks/mahout_benchmark.ipynb)
