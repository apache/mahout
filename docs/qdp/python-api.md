---
title: QDP Python API (qumat_qdp)
sidebar_label: Python API
---

# QDP Python API (qumat_qdp)

`qumat_qdp` is the user-facing Python package for QDP. It exposes:

- A unified encoding facade via `QdpEngine`
- Benchmark and loader builders
- Backend-selection helpers
- CPU-side input validation and L2 norms via `Preprocessor`
- Optional access to native `_qdp` types when the Rust extension is installed

## Overview

Import from the package root:

```python
from qumat_qdp import (
    BACKEND,
    Backend,
    LatencyResult,
    NativeQuantumTensor,
    Preprocessor,
    QdpBenchmark,
    QdpEngine,
    QdpTensor,
    QuantumDataLoader,
    QuantumTensor,
    RustQdpEngine,
    ThroughputResult,
    TritonAmdEngine,
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
- `Preprocessor`: CPU-side input checks and L2 norms (the same checks the encoders run before using the GPU).
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

## Preprocessing API

`Preprocessor` exposes the CPU-side checks that the native encoders run
before any data reaches the GPU. Use it to fail fast on bad input, or to
see the norm that amplitude encoding divides by. All four methods are
static, run on the CPU, and need the `_qdp` extension but not a GPU (a
stub build without the CUDA toolkit works).

| Method | Returns | Checks |
|--------|---------|--------|
| `validate_input(data, num_qubits)` | `None` | `1 <= num_qubits <= 30`, `data` non-empty, `len(data) <= 2**num_qubits` |
| `calculate_l2_norm(data)` | `float` | norm is non-zero and finite (no NaN / Inf) |
| `validate_batch(batch_data, num_samples, sample_size, num_qubits)` | `None` | `num_samples > 0`, length is `num_samples * sample_size`, `sample_size <= 2**num_qubits` |
| `calculate_batch_l2_norms(batch_data, num_samples, sample_size)` | `numpy.ndarray` (float64, one norm per sample) | every sample norm is non-zero and finite |

Input notes:

- `data` is a 1D NumPy array or a sequence of numbers. Other dtypes are
  converted to `float64`; a C-contiguous `float64` array is used without
  a copy.
- `batch_data` is either 2D with shape `(num_samples, sample_size)` or the
  same values flattened to 1D in row-major order. Non-contiguous and
  Fortran-ordered arrays are read in logical row order.
- A failed check raises `RuntimeError` whose message starts with
  `"Invalid input:"` (the same message `QdpEngine.encode` reports). A 1D
  argument given a 2D array raises `TypeError`; values NumPy cannot
  convert to `float64` (strings, ragged lists) raise `ValueError`.

### Why the norm matters for amplitude encoding

Amplitude encoding stores a feature vector `x` as the amplitudes of an
`n`-qubit state. A quantum state must have total probability 1, so QDP
loads `x / ||x||` rather than `x`, padding with zeros up to `2**n`
entries. `calculate_l2_norm` returns that `||x||`. A vector of all zeros
has norm 0 and cannot be normalized, so it is rejected instead of being
encoded as an invalid state; NaN or infinite values are rejected for the
same reason.

```python
import numpy as np
from qumat_qdp import Preprocessor

x = np.array([3.0, 4.0])  # 2 features -> fits in 1 qubit (2**1 = 2)

Preprocessor.validate_input(x, num_qubits=1)  # raises on bad input
norm = Preprocessor.calculate_l2_norm(x)  # 5.0
amplitudes = x / norm  # [0.6, 0.8]
print(norm, amplitudes, (amplitudes**2).sum())  # probabilities sum to 1

try:
    Preprocessor.calculate_l2_norm([0.0, 0.0])
except RuntimeError as err:
    print(err)  # Invalid input: Input data has zero norm
```

`QdpEngine(...).encode(x, num_qubits=1)` on a GPU produces the state with
the same amplitudes `[0.6, 0.8]` (as complex numbers).

Batch form, for a 2D array of samples:

```python
batch = np.array([[3.0, 4.0], [1.0, 1.0]])
Preprocessor.validate_batch(batch, num_samples=2, sample_size=2, num_qubits=1)
norms = Preprocessor.calculate_batch_l2_norms(batch, 2, 2)  # [5.0, 1.414...]
states = batch / norms[:, None]
```

Naming note: these method names mirror the Rust
`qdp_core::preprocessing::Preprocessor` API and are the stable public
names. If they are ever renamed, the old names should keep working with a
`DeprecationWarning` for at least one release rather than being removed.

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

## Low-level Rust Pipeline Helper

`run_throughput_pipeline_py(...)` is the low-level native helper used by the Rust benchmark path.

Signature:

`run_throughput_pipeline_py(device_id=0, num_qubits=16, batch_size=64, total_batches=100, encoding_method="amplitude", warmup_batches=0, seed=None, dtype="f64")`

`dtype` selects the pipeline element type: `"f64"` (default) or `"f32"`.

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
