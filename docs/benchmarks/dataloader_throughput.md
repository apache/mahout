# DataLoader Throughput Benchmark

This benchmark mirrors the `qdp-core/examples/dataloader_throughput.rs` pipeline and compares Mahout (QDP) against PennyLane and Qiskit on the same workload. It streams batches from a CPU-side producer, encodes amplitude states on GPU, and reports vectors-per-second.

## Workload

- Qubits: 16 (vector length `2^16`)
- Batches: 200
- Batch size: 64
- Prefetch depth: 16 (CPU producer queue)

## Running

```bash
# QDP-only Rust example
cargo run -p qdp-core --example dataloader_throughput --release

# Cross-framework comparison (requires deps in qdp/benchmark/requirements.txt)
python qdp/benchmark/benchmark_dataloader_throughput.py --qubits 16 --batches 200 --batch-size 64 --prefetch 16
```

## Example Output

```
Generating 12800 samples of 16 qubits...
  Batch size   : 64
  Vector length: 65536
  Batches      : 200
  Prefetch     : 16
  Generated 12800 samples
  PennyLane/Qiskit format: 6400.00 MB
  Mahout format: 6400.00 MB

======================================================================
DATALOADER THROUGHPUT BENCHMARK: 16 Qubits, 12800 Samples
======================================================================

[PennyLane] Full Pipeline (DataLoader -> GPU)...
  Total Time: 26.1952 s (488.6 vectors/sec)

[Qiskit] Full Pipeline (DataLoader -> GPU)...
  Total Time: 975.8720 s (13.1 vectors/sec)

[Mahout] Full Pipeline (DataLoader -> GPU)...
  IO + Encode Time: 115.3920 s
  Total Time: 115.5840 s (110.8 vectors/sec)

======================================================================
THROUGHPUT (Higher is Better)
Samples: 12800, Qubits: 16
======================================================================
PennyLane        488.6 vectors/sec
Mahout           110.8 vectors/sec
Qiskit            13.1 vectors/sec
----------------------------------------------------------------------
Speedup vs PennyLane:       0.23x
Speedup vs Qiskit:          8.44x
```

## Notes

- Example numbers reuse prior timings scaled to the default 12.8k vectors; re-run on target GPUs for fresh measurements.
- PennyLane/Qiskit sections include CPU-side state preparation time; Mahout timing includes IO + encode on GPU.
- Install competitor dependencies only if you plan to run their legs; the script auto-skips missing frameworks.
- Flags:
  - `--qubits`: controls vector length (`2^qubits`).
  - `--batches`: number of host-side batches to stream.
  - `--batch-size`: vectors per batch; raises total samples (`batches * batch-size`).
  - `--prefetch`: CPU queue depth; higher values help hide slow CPU-side prep (e.g., Qiskit state prep) and keep GPU fed.
