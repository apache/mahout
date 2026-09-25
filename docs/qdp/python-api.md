---
title: QDP Python API (qumat_qdp)
sidebar_label: Python API
---

# QDP Python API (qumat_qdp)

`qumat_qdp` is the user-facing Python package for QDP. It exposes:

- A unified encoding facade via `QdpEngine`
- Benchmark and loader builders
- Backend-selection helpers
- Optional access to native `_qdp` types when the Rust extension is installed

## Overview

Import from the package root:

```python
from qumat_qdp import (
    BACKEND,
    Backend,
    LatencyResult,
    MemoryEstimate,
    NativeQuantumTensor,
    QdpBenchmark,
    QdpEngine,
    QdpTensor,
    QuantumDataLoader,
    QuantumTensor,
    RustQdpEngine,
    ThroughputResult,
    TritonAmdEngine,
    estimate_memory,
    force_backend,
    is_triton_amd_available,
    run_throughput_pipeline_py,
)
```

Key exports:

- `QdpEngine`: unified encode entry point for `backend="cuda"` and `backend="amd"`.
- `QdpBenchmark`: benchmark builder for the Rust pipeline or explicit PyTorch reference backend.
- `QuantumDataLoader`: batch iterator builder for synthetic or file-backed inputs.
- `QdpTensor` / `QuantumTensor`: thin DLPack facade type.
- `estimate_memory` / `MemoryEstimate`: upfront host and device memory sizing for a pipeline configuration.
- `Backend`, `BACKEND`, `force_backend`: backend detection and override helpers for the `_qdp` / PyTorch-reference selection layer.
- `TritonAmdEngine`, `is_triton_amd_available`: direct AMD-route entry points.
- `RustQdpEngine`, `NativeQuantumTensor`: native `_qdp` exports when the extension is available.

`BACKEND` is the backend-selection snapshot captured when `qumat_qdp` is imported:

- `Backend.RUST_CUDA`: `_qdp` is available
- `Backend.NONE`: no auto-detected backend is available
- `Backend.PYTORCH`: only when selected explicitly before import-time evaluation

PyTorch is not used as an automatic fallback. The two backend-selection surfaces are independent:

- `force_backend(backend: Backend | None)` only affects what `get_backend()` returns (and, transitively, the cached `BACKEND` snapshot at import time). Pass `None` to restore auto-detection. It does **not** switch which implementation `QdpBenchmark` or `QuantumDataLoader` use, and it does **not** change the availability of `RustQdpEngine` / `NativeQuantumTensor` (those depend on whether `_qdp` could be imported).
- To run the PyTorch reference pipeline through the builders, call `.backend("pytorch")` on `QdpBenchmark` or `QuantumDataLoader` explicitly.

If you need the current override state after import time, call `get_backend()` from `qumat_qdp._backend` rather than relying on the exported `BACKEND` constant.

## Backend Model

`qumat_qdp` has two related backend-selection surfaces:

| Surface | Selector | Values |
|--------|----------|--------|
| Unified engine routing | `QdpEngine(..., backend=...)` | `"cuda"`, `"amd"` |
| Builder fallback/reference routing | `.backend(...)` on `QdpBenchmark` / `QuantumDataLoader` | `"rust"`, `"pytorch"` |

These selectors are intentionally different:

- `QdpEngine(..., backend="cuda")` routes to the native `_qdp` CUDA engine.
- `QdpEngine(..., backend="amd")` routes to the Triton AMD implementation.
- `QdpBenchmark(...).backend("rust")` uses the Rust pipeline.
- `QdpBenchmark(...).backend("pytorch")` uses the pure-PyTorch reference implementation.
- `QuantumDataLoader(...).backend("rust")` uses the Rust-backed iterator.
- `QuantumDataLoader(...).backend("pytorch")` uses the explicit PyTorch reference iterator.

## Encoding API

### QdpEngine

Unified encoding facade.

**Constructor**

`QdpEngine(device_id=0, precision="float32", backend="cuda")`

- `device_id` (int): GPU device ordinal. Maps to a CUDA device for `backend="cuda"` and to the ROCm/PyTorch device for `backend="amd"`.
- `precision` (str): `"float32"` or `"float64"`
- `backend` (str): `"cuda"` or `"amd"`

**Method**

`encode(data, num_qubits, encoding_method="amplitude")`

- `data`: Python list, NumPy array, PyTorch tensor, or other backend-supported input
- `num_qubits` (int): number of qubits
- `encoding_method` (str): depends on selected route

Route support summary:

| Route | Supported methods |
|--------|-------------------|
| `QdpEngine(..., backend="cuda")` | `amplitude`, `angle`, `basis`, `phase`, `iqp`, `iqp-z` |
| `QdpEngine(..., backend="amd")` | `amplitude`, `angle`, `basis` |

**CUDA tensor-input caveat:** when `data` is provided as a zero-copy CUDA `torch.Tensor`, supported encoding methods are currently limited to `amplitude`, `angle`, `basis`, `iqp`, and `iqp-z`. In that input path, `phase` is not currently supported — pass a CPU tensor, NumPy array, or Python list to use `phase` on the CUDA route.

Result type notes:

- The CUDA route returns the native `_qdp` tensor object when `_qdp` is available.
- The AMD route currently returns a `torch.Tensor` from the Triton implementation.
- `QdpTensor` / `QuantumTensor` provide a DLPack facade for backend-native tensor producers.

Direct AMD usage is also available:

```python
from qumat_qdp import QdpEngine, TritonAmdEngine

router = QdpEngine(backend="amd", device_id=0, precision="float32")
engine = TritonAmdEngine(device_id=0, precision="float32")
```

Use `is_triton_amd_available()` to check whether the Triton AMD runtime is available before selecting that route.

A pure-PyTorch reference encoder is also available at `qumat_qdp.torch_ref.encode`. It supports `amplitude`, `angle`, `basis`, and `iqp` (call `iqp_encode(..., enable_zz=False)` directly for Z-only behavior). It does not implement `phase` or `iqp-z`, and runs entirely in PyTorch — useful for cross-checking the Rust pipeline.

### QdpTensor and QuantumTensor

`QdpTensor` is a thin DLPack facade; `QuantumTensor` is an alias.

Available methods:

- `__dlpack__(stream=None)`
- `__dlpack_device__()`
- `to_torch()`

Example:

```python
import torch
from qumat_qdp import QdpEngine

qt = QdpEngine(device_id=0, backend="cuda").encode(
    [[1.0, 0.0, 0.0, 0.0]],
    num_qubits=2,
    encoding_method="amplitude",
)
tensor = torch.from_dlpack(qt)
```

## Benchmark API

`QdpBenchmark` is a builder for throughput and latency runs.

**Constructor**

`QdpBenchmark(device_id=0)`

**Chainable methods**

| Method | Description |
|--------|-------------|
| `qubits(n)` | Number of qubits. |
| `encoding(method)` | `"amplitude"` \| `"angle"` \| `"basis"` \| `"iqp"` \| `"iqp-z"`. |
| `batches(total, size=64)` | Total batches and batch size. |
| `prefetch(n)` | No-op (API compatibility). |
| `warmup(n)` | Warmup batch count. |
| `backend(name)` | Select `"rust"` or `"pytorch"`. |

Backend notes:

- `"rust"` is the default backend.
- `"pytorch"` is explicit and intended as a reference path.
- The Rust benchmark path is the native pipeline used by `run_throughput_pipeline_py`.
- The PyTorch benchmark path is implemented in Python and is useful for fallback testing and comparisons.

Encoding support summary:

| Benchmark backend | Supported methods |
|--------|-------------------|
| `"rust"` | `amplitude`, `angle`, `basis` |
| `"pytorch"` | `amplitude`, `angle`, `basis`, `iqp` |

Treat `phase` and `iqp-z` as direct-encode methods today rather than benchmark-helper features.

Example:

```python
from qumat_qdp import QdpBenchmark

rust_result = (
    QdpBenchmark(device_id=0)
    .qubits(16)
    .encoding("amplitude")
    .batches(100, size=64)
    .warmup(2)
    .run_throughput()
)

ref_result = (
    QdpBenchmark(device_id=0)
    .backend("pytorch")
    .qubits(16)
    .encoding("iqp")
    .batches(100, size=64)
    .run_throughput()
)
```

Result types:

| Type | Fields |
|--------|--------|
| `ThroughputResult` | `duration_sec`, `vectors_per_sec` |
| `LatencyResult` | `duration_sec`, `latency_ms_per_vector` |

## Data Loader API

`QuantumDataLoader` is a builder for encoded batch iteration.

**Constructor**

`QuantumDataLoader(device_id=0, num_qubits=16, batch_size=64, total_batches=100, encoding_method="amplitude", seed=None)`

**Chainable methods**

| Method | Description |
|--------|-------------|
| `qubits(n)` | Number of qubits. |
| `encoding(method)` | `"amplitude"` \| `"angle"` \| `"basis"` \| `"iqp"` \| `"iqp-z"`. |
| `batches(total, size=64)` | Total batches and batch size. |
| `source_synthetic(total_batches=None)` | Synthetic data (default); optional override for total batches. |
| `source_file(path, streaming=False)` | Use a file-backed source. |
| `seed(s)` | RNG seed for reproducibility. |
| `null_handling(policy)` | Set `"fill_zero"` or `"reject"`. |
| `backend(name)` | Select `"rust"` or `"pytorch"`. |

File source notes:

- `source_file(path, streaming=False)` accepts local paths and also accepts remote `s3://...` / `gs://...` paths when QDP is built with the optional `remote-io` feature.
- Remote URL query strings and fragments are rejected.
- `streaming=True` currently supports `.parquet` only.
- The Rust backend supports broader file formats and streaming loaders.
- The explicit PyTorch backend supports synthetic data plus `.npy`, `.pt`, and `.pth` files only.

Memory check notes:

- The Rust backend estimates how much device memory a configuration needs and rejects it when the estimate exceeds free GPU memory, rather than failing part-way through a run. The check runs when iteration starts, before any input file is read — so a `QuantumDataLoader(...)` call returns normally and the rejection surfaces at the `for` statement.
- The error names the encoding, batch size, and qubit count alongside the requested and available memory, and the remedy is to lower `batch_size` or `qubits`.
- The rejection is raised as `RuntimeError`, not `ValueError`, because the loader wraps every backend error in `RuntimeError` — catch that and match on the message. `estimate_memory()` raises `ValueError` for the same oversized configuration.
- The estimate budgets two concurrent batch state buffers, because a batch stays resident on the device until the consumer releases its tensor. A configuration that fits one buffer but not two is rejected.
- The buffer is sized by the wider of the loader's input dtype and the engine's precision, since the input dtype sets the encode path's working precision while every path converts its result to the engine precision. With the defaults — file loaders read float64, `QdpEngine` runs float32 — it is the input dtype that sizes the budget. The error reports both values.
- Basis input read from a file is always budgeted as float64, because basis values are integer state indices that the file readers load as float64 whatever `dtype` is requested. Synthetic basis data is budgeted at the requested `dtype`.
- The check is skipped whenever the CUDA runtime reports no usable device — a build without the CUDA toolkit, a host with no driver, or an empty `CUDA_VISIBLE_DEVICES` — so CPU-only environments are unaffected.
- The comparison is against free memory sampled when iteration starts; nothing is reserved. Another process or a second loader can claim that memory before the first batch is allocated, so passing the check is a fast sanity check rather than a guarantee.
- To size a configuration before building the loader, call `estimate_memory()` (see [Memory Estimation API](#memory-estimation-api)).

Iteration behavior depends on backend:

| Loader backend | Iteration yields |
|--------|------------------|
| `"rust"` | `NativeQuantumTensor` / `_qdp.QuantumTensor` (the native Rust extension type, not the `QuantumTensor` facade alias) |
| `"pytorch"` | `torch.Tensor` |

Encoding support summary:

| Loader backend | Supported methods |
|--------|-------------------|
| `"rust"` | documented for `amplitude`, `angle`, `basis` |
| `"pytorch"` | `amplitude`, `angle`, `basis`, `iqp` |

`phase` and `iqp-z` are not documented loader-first methods.

Example:

```python
from qumat_qdp import QuantumDataLoader
import torch

loader = (
    QuantumDataLoader(device_id=0)
    .backend("pytorch")
    .qubits(4)
    .encoding("iqp")
    .batches(5, size=8)
    .source_synthetic()
)

for batch in loader:
    assert isinstance(batch, torch.Tensor)
```

Rust-backed file example:

```python
from qumat_qdp import QuantumDataLoader
import torch

loader = (
    QuantumDataLoader(device_id=0)
    .qubits(8)
    .encoding("amplitude")
    .batches(10, size=32)
    .source_file("dataset.parquet", streaming=True)
    .null_handling("fill_zero")
)

for qt in loader:
    batch = torch.from_dlpack(qt)
```

## Memory Estimation API

`estimate_memory()` answers the same question the loader's memory check answers when iteration starts, but without building anything — use it to size `qubits` and `batch_size` against a memory budget instead of discovering the ceiling from a rejection.

Signature:

`estimate_memory(num_qubits, batch_size, encoding_method="amplitude", dtype="f64", prefetch_depth=16)`

Returns a `MemoryEstimate` with three fields, all in bytes:

| Field | Meaning |
|-------|---------|
| `cpu_prefetch_bytes` | Host prefetch pool: `prefetch_depth` batches of raw, unencoded input |
| `gpu_state_bytes` | Device state-vector buffer, including the two-buffer allowance the memory check applies |
| `total_bytes` | Sum of the two |

```python
from qumat_qdp import estimate_memory

est = estimate_memory(num_qubits=20, batch_size=64, dtype="f32")
print(f"{est.gpu_state_bytes / 1024**2:.0f} MiB of device state")
```

Notes:

- The function is pure configuration arithmetic: it allocates nothing and opens no device, so it works on a stub build or a host with no GPU — which is the point, since it exists to size configurations that the current machine may not be able to run.
- The device state vector holds `2**num_qubits` complex amplitudes per sample for every encoding, so `gpu_state_bytes` is identical for `amplitude` and `angle` at equal qubit counts even though their input widths differ. Only `cpu_prefetch_bytes` reflects the input width.
- `prefetch_depth` scales `cpu_prefetch_bytes` only. Device memory does not depend on it, which is why the memory check's error suggests lowering `batch_size` or `qubits` and never `prefetch_depth`.
- `gpu_state_bytes` is the figure the memory check compares against free VRAM, with one caveat: the check budgets the wider of this `dtype` and the engine's precision, and always float64 for `basis` read from a file. Pass the engine precision as `dtype` when the two differ, or the estimate will be half what the check applies.
- Every failure the estimator reports is an argument error — unknown encoding or dtype name, a `num_qubits` whose `2**n` state vector is not representable, or arithmetic overflow — and raises `ValueError`. A negative `num_qubits` or `batch_size` fails at the argument boundary first and raises `OverflowError`. `RuntimeError` is raised only when the `_qdp` extension is not installed.
- Encodings without an f32 batch path are estimated as float64 even when float32 is requested, mirroring what the pipeline does with the same request.

## Low-level Rust Pipeline Helper

`run_throughput_pipeline_py(...)` is the low-level native helper used by the Rust benchmark path.

Signature:

`run_throughput_pipeline_py(device_id=0, num_qubits=16, batch_size=64, total_batches=100, encoding_method="amplitude", warmup_batches=0, seed=None, float32_pipeline=False)`

Returns a tuple:

`(duration_sec, vectors_per_sec, latency_ms_per_vector)`

This helper is only available when `_qdp` is installed.

It builds its own engine and pipeline rather than going through `QuantumDataLoader`, so the memory check described under [Data Loader API](#data-loader-api) does not apply to it: an oversized configuration fails part-way through the run. Call `estimate_memory()` first if you are sizing a benchmark near the limits of the device.

## Backward Compatibility

`benchmark/api.py` and `benchmark/loader.py` continue to re-export the modern `qumat_qdp` API. Prefer importing from `qumat_qdp` directly.

From `qdp/qdp-python`, benchmark scripts can still be run with:

```bash
uv run python benchmark/run_pipeline_baseline.py
uv run python benchmark/benchmark_loader_throughput.py
```
