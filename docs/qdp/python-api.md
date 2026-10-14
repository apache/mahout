---
title: QDP Python API (qumat_qdp)
sidebar_label: Python API
---

# QDP Python API (qumat_qdp)

`qumat_qdp` is the user-facing Python package for QDP. It exposes:

- A unified encoding facade via `QdpEngine`
- Benchmark and loader builders
- GPU availability checks
- Optional access to native `_qdp` types when the Rust extension is installed

`qumat.qdp` re-exports `QdpEngine` and `QuantumTensor` from this package.

## Overview

Import from the package root:

```python
from qumat_qdp import (
    LatencyResult,
    NativeQdpEngine,
    NativeQuantumTensor,
    QdpBenchmark,
    QdpEngine,
    QdpTensor,
    QuantumDataLoader,
    QuantumTensor,
    ThroughputResult,
    TritonAmdEngine,
    is_native_cuda_available,
    is_triton_amd_available,
    run_throughput_pipeline_py,
)
```

Key exports:

- `QdpEngine`: unified encode entry point for `backend="cuda"` and `backend="amd"`.
- `QdpBenchmark`: benchmark builder for the CUDA and AMD backends.
- `QuantumDataLoader`: batch iterator builder for synthetic or file-backed inputs.
- `QdpTensor` / `QuantumTensor`: thin DLPack facade type.
- `TritonAmdEngine`, `is_triton_amd_available`: direct AMD-route entry points.
- `is_native_cuda_available`: whether the native engine reports a usable CUDA device.
- `NativeQdpEngine`, `NativeQuantumTensor`: the raw `_qdp` extension classes, when the extension is available.

An importable `_qdp` does not guarantee a usable GPU; check that with `is_native_cuda_available()` or `is_triton_amd_available()`.

## Backend Model

Every backend selector names the GPU backend to run on:

| Surface | Selector | Values |
|--------|----------|--------|
| Engine | `QdpEngine(..., backend=...)` | `"cuda"`, `"amd"` |
| Data loader | `QuantumDataLoader(...).backend(...)` | `"cuda"` (default), `"amd"`, `"auto"` |
| Benchmark | `QdpBenchmark(...).backend(...)` | `"cuda"` (default), `"amd"` |

- `"cuda"` routes to the native `_qdp` Rust/CUDA implementation on NVIDIA GPUs.
- `"amd"` routes to the Triton AMD implementation on ROCm GPUs.
- `"auto"` (data loader only) picks between the two for the host.

A selector raises when its GPU backend is not usable. `QuantumDataLoader(...).backend("auto")` picks the GPU backend for the host:

1. `"cuda"`, when `_qdp` reports a usable CUDA device;
2. otherwise `"amd"`, when `is_triton_amd_available()` is true;
3. otherwise iteration raises `RuntimeError`.

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
| `QdpEngine(..., backend="amd")` | `amplitude`, `angle`, `basis`, `phase`, `iqp`, `iqp-z` |

**CUDA tensor-input caveat:** when `data` is provided as a zero-copy CUDA `torch.Tensor`, supported encoding methods are currently limited to `amplitude`, `angle`, `basis`, `iqp`, and `iqp-z`. In that input path, `phase` is not currently supported — pass a CPU tensor, NumPy array, or Python list to use `phase` on the CUDA route.

Result type notes:

- Both routes return a `QdpTensor`. Convert it with `torch.from_dlpack(qt)` or `qt.to_torch()`.
- `qt.value` holds the backend-native object: the `_qdp` tensor on the CUDA route, a `torch.Tensor` on the AMD route. `qt.backend` is `"cuda"` or `"amd"`.

Direct AMD usage is also available:

```python
from qumat_qdp import QdpEngine, TritonAmdEngine

router = QdpEngine(backend="amd", device_id=0, precision="float32")
engine = TritonAmdEngine(device_id=0, precision="float32")
```

Use `is_triton_amd_available()` to check whether the Triton AMD runtime is available before selecting that route.

### QdpTensor and QuantumTensor

`QdpTensor` is a thin DLPack facade over a backend-native result; `QuantumTensor` is an alias.

Fields:

- `value`: the backend-native object
- `backend`: `"cuda"` or `"amd"`

Available methods:

- `__dlpack__(stream=None, **kwargs)`
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
| `dtype(name)` | `"f32"` (default) or `"f64"`. Encodings without an f32 kernel (`phase`, `iqp`, `iqp-z`) run in f64 on the CUDA backend. |
| `backend(name)` | Select `"cuda"` or `"amd"`. |

Backend notes:

- `"cuda"` is the default backend. It is the native pipeline used by `run_throughput_pipeline_py`.
- `"amd"` runs the Triton AMD engine in a Python loop on the selected ROCm device and raises if no ROCm runtime is available.

Encoding support summary:

| Benchmark backend | Supported methods |
|--------|-------------------|
| `"cuda"` | `amplitude`, `angle`, `basis` |
| `"amd"` | `amplitude`, `angle`, `basis`, `iqp`, `iqp-z` |

Treat `phase` and `iqp-z` as direct-encode methods today rather than benchmark-helper features.

Example:

```python
from qumat_qdp import QdpBenchmark

cuda_result = (
    QdpBenchmark(device_id=0)
    .qubits(16)
    .encoding("amplitude")
    .batches(100, size=64)
    .warmup(2)
    .run_throughput()
)

amd_result = (
    QdpBenchmark(device_id=0)
    .backend("amd")
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
| `dtype(name)` | Element precision for file sources: `"float64"` (default, lossless) or `"float32"`. Synthetic data is generated in float32. |
| `backend(name)` | Select `"cuda"`, `"amd"`, or `"auto"` (see [Backend Model](#backend-model)). |

File source notes:

- `source_file(path, streaming=False)` accepts local paths and also accepts remote `s3://...` / `gs://...` paths when QDP is built with the optional `remote-io` feature.
- Remote URL query strings and fragments are rejected.
- `streaming=True` currently supports `.parquet` only.
- The `"cuda"` backend supports broader file formats and streaming loaders.
- The `"amd"` backend loads files in memory and supports synthetic data plus `.npy`, `.pt`, and `.pth` files only.

Iteration behavior depends on backend:

| Loader backend | Iteration yields |
|--------|------------------|
| `"cuda"` | `QdpTensor` wrapping the native `_qdp.QuantumTensor` |
| `"amd"` | `QdpTensor` wrapping a `torch.Tensor` on the ROCm device |

Encoding support summary:

| Loader backend | Supported methods |
|--------|-------------------|
| `"cuda"` | documented for `amplitude`, `angle`, `basis` |
| `"amd"` | `amplitude`, `angle`, `basis`, `iqp`, `iqp-z` |

`phase` and `iqp-z` are not documented loader-first methods.

Example:

```python
from qumat_qdp import QuantumDataLoader
import torch

loader = (
    QuantumDataLoader(device_id=0)
    .backend("amd")
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

## Low-level Rust Pipeline Helper

`run_throughput_pipeline_py(...)` is the low-level native helper used by the Rust benchmark path.

Signature:

`run_throughput_pipeline_py(device_id=0, num_qubits=16, batch_size=64, total_batches=100, encoding_method="amplitude", warmup_batches=0, seed=None, dtype="f64")`

Returns a tuple:

`(duration_sec, vectors_per_sec, latency_ms_per_vector)`

This helper is only available when `_qdp` is installed.

## Backward Compatibility

`benchmark/api.py` and `benchmark/loader.py` continue to re-export the modern `qumat_qdp` API. Prefer importing from `qumat_qdp` directly.

From `qdp/qdp-python`, benchmark scripts can still be run with:

```bash
uv run python benchmark/run_pipeline_baseline.py
uv run python benchmark/benchmark_loader_throughput.py
```
