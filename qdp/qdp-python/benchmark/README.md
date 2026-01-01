# Benchmarks

This directory contains Python benchmarks for Mahout QDP. There are three main
scripts:

- `benchmark_e2e.py`: end-to-end latency from disk to GPU VRAM (includes IO,
  normalization, encoding, transfer, and a dummy forward pass).
- `benchmark_throughput.py`: DataLoader-style throughput benchmark
  that measures vectors/sec across Mahout, PennyLane, and Qiskit.
- `benchmark_latency.py`: Data-to-State latency benchmark (CPU RAM -> GPU VRAM).

## Quick Start

From the repo root:

```bash
cd qdp
make benchmark
```

This installs the QDP Python package (if needed), installs benchmark
dependencies, and runs both benchmarks.

## Manual Setup

```bash
cd qdp/qdp-python
uv sync --group benchmark
```

Then run benchmarks with `uv run python ...` or activate the virtual
environment and use `python ...`.

## E2E Benchmark (Disk -> GPU)

```bash
cd qdp/qdp-python/benchmark
python benchmark_e2e.py
```

Additional options:

```bash
python benchmark_e2e.py --qubits 16 --samples 200 --frameworks mahout-parquet mahout-arrow
python benchmark_e2e.py --frameworks all
```

Notes:

- `--frameworks` accepts a space-separated list or `all`.
  Options: `mahout-parquet`, `mahout-arrow`, `pennylane`, `qiskit`.
- The script writes `final_benchmark_data.parquet` and
  `final_benchmark_data.arrow` in the current working directory and overwrites
  them on each run.
- If multiple frameworks run, the script compares output states for
  correctness at the end.

## Data-to-State Latency Benchmark

```bash
cd qdp/qdp-python/benchmark
python benchmark_latency.py --qubits 16 --batches 200 --batch-size 64 --prefetch 16
python benchmark_latency.py --frameworks mahout,pennylane
```

Notes:

- `--frameworks` is a comma-separated list or `all`.
  Options: `mahout`, `pennylane`, `qiskit-init`, `qiskit-statevector`.
- The latency test reports average milliseconds per vector.
- Flags:
  - `--qubits`: controls vector length (`2^qubits`).
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
cd qdp/qdp-python/benchmark
python benchmark_throughput.py --qubits 16 --batches 200 --batch-size 64 --prefetch 16
python benchmark_throughput.py --frameworks mahout,pennylane
```

Notes:

- `--frameworks` is a comma-separated list or `all`.
  Options: `mahout`, `pennylane`, `qiskit`.
- Throughput is reported in vectors/sec (higher is better).

## Dependency Notes

- Qiskit and PennyLane are optional. If they are not installed, their benchmark
  legs are skipped automatically.
- For Mahout-only runs, you can uninstall the competitor frameworks:
  `uv pip uninstall qiskit pennylane`.

### We can also run benchmarks on colab notebooks(without owning a GPU)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/apache/mahout/blob/dev-qdp/qdp/qdp-python/benchmark/notebooks/mahout_benchmark.ipynb)
