# Data-to-State Latency Benchmark

This benchmark isolates the "Data-to-State" pipeline (CPU RAM -> GPU VRAM) and
compares Mahout (QDP) against PennyLane and Qiskit baselines:

- Qiskit Initialize (`qiskit-init`): circuit-based state preparation.
- Qiskit Statevector (`qiskit-statevector`): raw data loading baseline.

The primary metric is average time-to-state in milliseconds (lower is better).

## Workload

- Qubits: 16 (vector length `2^16`)
- Batches: 200
- Batch size: 64
- Prefetch depth: 16 (CPU producer queue)

## Running

```bash
# Latency test (CPU RAM -> GPU VRAM)
python qdp/qdp-python/benchmark/benchmark_latency.py --qubits 16 \
  --batches 200 --batch-size 64 --prefetch 16

# Run only selected frameworks
python qdp/qdp-python/benchmark/benchmark_latency.py --frameworks mahout,pennylane
```

## Example Output

```
Generating 12800 samples of 16 qubits...
  Batch size   : 64
  Vector length: 65536
  Batches      : 200
  Prefetch     : 16
  Frameworks   : pennylane, qiskit-init, qiskit-statevector, mahout
  Generated 12800 samples
  PennyLane/Qiskit format: 6400.00 MB
  Mahout format: 6400.00 MB

======================================================================
DATA-TO-STATE LATENCY BENCHMARK: 16 Qubits, 12800 Samples
======================================================================

[PennyLane] Full Pipeline (DataLoader -> GPU)...
  Total Time: 26.1952 s (2.047 ms/vector)

[Qiskit Initialize] Full Pipeline (DataLoader -> GPU)...
  Total Time: 975.8720 s (76.243 ms/vector)

[Qiskit Statevector] Full Pipeline (DataLoader -> GPU)...
  Total Time: 115.5840 s (9.030 ms/vector)

[Mahout] Full Pipeline (DataLoader -> GPU)...
  Total Time: 11.5384 s (0.901 ms/vector)

======================================================================
LATENCY (Lower is Better)
Samples: 12800, Qubits: 16
======================================================================
Mahout             0.901 ms/vector
PennyLane          2.047 ms/vector
Qiskit Statevector 9.030 ms/vector
Qiskit Initialize  76.243 ms/vector
----------------------------------------------------------------------
Speedup vs PennyLane:       2.27x
Speedup vs Qiskit Init:    84.61x
Speedup vs Qiskit Statevec: 10.02x
```

## Notes

- Latency numbers are average milliseconds per vector across the full run.
- PennyLane and Qiskit timings include CPU-side state preparation; Mahout timing
  includes CPU->GPU encode + DLPack handoff.
- Missing frameworks are auto-skipped; use `--frameworks` to control the legs.
- Requires a CUDA-capable GPU (`torch.cuda.is_available()` must be true).
