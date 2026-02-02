# QDP Python API (qumat_qdp)

Public Python API for QDP: GPU-accelerated encoding, benchmark helpers, and a batch iterator for training or evaluation loops.

## Overview

The **qumat_qdp** package wraps the native extension `_qdp` and adds:

- **Encoding:** `QdpEngine` and `QuantumTensor` for encoding classical data into quantum states and zero-copy DLPack integration.
- **Benchmark:** `QdpBenchmark` for throughput/latency runs (full pipeline in Rust, GIL released).
- **Data loader:** `QuantumDataLoader` for iterating encoded batches one at a time (`for qt in loader:`).

Import from the package:

```python
from qumat_qdp import (
    QdpEngine,
    QuantumTensor,
    QdpBenchmark,
    ThroughputResult,
    LatencyResult,
    QuantumDataLoader,
    run_throughput_pipeline_py,
)
```

**Requirements:** Linux with NVIDIA GPU (CUDA). Loader and pipeline helpers are stubs on other platforms and raise `RuntimeError`.

---

## Encoding API

### QdpEngine

GPU encoder. Constructor and main methods:

**`QdpEngine(device_id=0, precision="float32")`**

- `device_id` (int): CUDA device ID.
- `precision` (str): `"float32"` or `"float64"`.
- Raises `RuntimeError` on init failure or unsupported precision.

**`encode(data, num_qubits, encoding_method="amplitude") -> QuantumTensor`**

- `data`: list of floats, 1D/2D NumPy array (float64, C-contiguous), PyTorch tensor (CPU/CUDA), or file path (`.parquet`, `.arrow`, `.feather`, `.npy`, `.pt`, `.pth`, `.pb`).
- `num_qubits` (int): Number of qubits.
- `encoding_method` (str): `"amplitude"` | `"angle"` | `"basis"` | `"iqp"` | `"iqp-z"`.
- Returns a DLPack-compatible tensor; use `torch.from_dlpack(qtensor)`. Shape `[batch_size, 2^num_qubits]`.

**`create_synthetic_loader(total_batches, batch_size=64, num_qubits=16, encoding_method="amplitude", seed=None)`**

- Returns an iterator that yields one `QuantumTensor` per batch. GIL is released during each encode. Linux/CUDA only.

### QuantumTensor

DLPack wrapper for a GPU quantum state.

- **`__dlpack__(stream=None)`:** Returns a DLPack PyCapsule (single use).
- **`__dlpack_device__()`:** Returns `(device_type, device_id)`; CUDA is `(2, gpu_id)`.

If not consumed, memory is freed when the object is dropped; if consumed (e.g. by PyTorch), ownership transfers to the consumer.

---

## Benchmark API

Runs the full encode pipeline in Rust (warmup + timed loop) with GIL released. No Python-side loop.

### QdpBenchmark

Builder; chain methods then call `run_throughput()` or `run_latency()`.

**Constructor:** `QdpBenchmark(device_id=0)`

**Chainable methods:**

| Method | Description |
|--------|-------------|
| `qubits(n)` | Number of qubits. |
| `encoding(method)` | `"amplitude"` \| `"angle"` \| `"basis"`. |
| `batches(total, size=64)` | Total batches and batch size. |
| `prefetch(n)` | No-op (API compatibility). |
| `warmup(n)` | Warmup batch count. |

**`run_throughput() -> ThroughputResult`**

- Requires `qubits` and `batches` to be set.
- Returns `ThroughputResult` with `duration_sec`, `vectors_per_sec`.
- Raises `ValueError` if config missing; `RuntimeError` if pipeline unavailable.

**`run_latency() -> LatencyResult`**

- Same pipeline; returns `LatencyResult` with `duration_sec`, `latency_ms_per_vector`.

### Result types

| Type | Fields |
|------|--------|
| `ThroughputResult` | `duration_sec`, `vectors_per_sec` |
| `LatencyResult` | `duration_sec`, `latency_ms_per_vector` |

### Example

```python
from qumat_qdp import QdpBenchmark, ThroughputResult, LatencyResult

result = (
    QdpBenchmark(device_id=0)
    .qubits(16)
    .encoding("amplitude")
    .batches(100, size=64)
    .warmup(2)
    .run_throughput()
)
print(result.vectors_per_sec)

lat = (
    QdpBenchmark(device_id=0)
    .qubits(16)
    .encoding("amplitude")
    .batches(100, size=64)
    .run_latency()
)
print(lat.latency_ms_per_vector)
```

---

## Data Loader API

Iterate over encoded batches one at a time. Each batch is a `QuantumTensor`; encoding runs in Rust with GIL released per batch.

### QuantumDataLoader

Builder for a synthetic-data loader. Calling `iter(loader)` (or `for qt in loader`) creates the Rust-backed iterator.

**Constructor:**
`QuantumDataLoader(device_id=0, num_qubits=16, batch_size=64, total_batches=100, encoding_method="amplitude", seed=None)`

**Chainable methods:**

| Method | Description |
|--------|-------------|
| `qubits(n)` | Number of qubits. |
| `encoding(method)` | `"amplitude"` \| `"angle"` \| `"basis"`. |
| `batches(total, size=64)` | Total batches and batch size. |
| `source_synthetic(total_batches=None)` | Synthetic data (default); optional override for total batches. |
| `seed(s)` | RNG seed for reproducibility. |

**Iteration:** `for qt in loader:` yields `QuantumTensor` of shape `[batch_size, 2^num_qubits]`. Consume once per tensor, e.g. `torch.from_dlpack(qt)`.

### Example

```python
from qumat_qdp import QuantumDataLoader
import torch

loader = (
    QuantumDataLoader(device_id=0)
    .qubits(16)
    .encoding("amplitude")
    .batches(100, size=64)
    .source_synthetic()
)

for qt in loader:
    batch = torch.from_dlpack(qt)  # [batch_size, 2^16]
    # use batch ...
```

---

## Low-level: run_throughput_pipeline_py

Runs the full pipeline in Rust with GIL released. Used by `QdpBenchmark`; can be called directly.

**Signature:**
`run_throughput_pipeline_py(device_id=0, num_qubits=16, batch_size=64, total_batches=100, encoding_method="amplitude", warmup_batches=0, seed=None) -> tuple[float, float, float]`

**Returns:** `(duration_sec, vectors_per_sec, latency_ms_per_vector)`.

**Raises:** `RuntimeError` on failure or when not available (e.g. non-Linux).

---

## Backward compatibility

`benchmark/api.py` and `benchmark/loader.py` re-export from `qumat_qdp`. Prefer:

- `from qumat_qdp import QdpBenchmark, ThroughputResult, LatencyResult`
- `from qumat_qdp import QuantumDataLoader`

Benchmark scripts add the project root to `sys.path`, so from the `qdp-python` directory you can run:

```bash
uv run python benchmark/run_pipeline_baseline.py
uv run python benchmark/benchmark_loader_throughput.py
```

without setting `PYTHONPATH`.
