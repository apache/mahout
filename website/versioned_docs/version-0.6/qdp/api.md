---
title: API Reference - QDP
---

# API Reference

Mahout QDP (Quantum Data Plane) provides GPU-accelerated quantum state encoding.
It writes classical data directly into GPU memory and returns a DLPack-compatible
handle for zero-copy integration with downstream frameworks.

## Module: `qumat.qdp`

Prefer importing from `qumat.qdp`; the native extension is `_qdp`.

```python
import qumat.qdp as qdp
# or: from _qdp import QdpEngine, QuantumTensor
```

## Class: `QdpEngine`

### `QdpEngine(device_id=0, precision="float32", backend="cuda")`

Create a GPU encoder instance.

**Parameters**
- `device_id` (int): GPU device ordinal, default `0` (CUDA ordinal when `backend="cuda"`; ROCm/PyTorch device index when `backend="amd"`).
- `precision` (str): `"float32"` or `"float64"`.
- `backend` (str): Public Python route selector. `qumat.qdp` uses `"cuda"` for the Rust/CUDA path and `"amd"` for the Triton AMD route.

**Raises**
- `RuntimeError`: Initialization failure or unsupported precision.

### `encode(data, num_qubits, encoding_method="amplitude") -> QuantumTensor`

Encode classical input into a quantum state and return a DLPack tensor on GPU.

**Parameters**
- `data`: Supported inputs
  - `list[float]`
  - `numpy.ndarray` (1D/2D, dtype=float64, C-contiguous)
  - `torch.Tensor` (CPU): float64, contiguous
  - `torch.Tensor` (CUDA, direct-input fast path): float32 or float64, contiguous, on the same device as the engine. Limited to `amplitude`, `angle`, `basis`, `iqp`, and `iqp-z` (see Advanced Encoding Support Matrix). For `phase`, pass a CPU tensor, NumPy array, or list instead.
  - `str` / `pathlib.Path` file path
    - `.parquet`, `.arrow` / `.feather`, `.npy`, `.pt` / `.pth`, `.pb`
    - remote URL (`s3://bucket/key`, `gs://bucket/key`) when built with `remote-io`
- `num_qubits` (int): Number of qubits, range 1–30.
- `encoding_method` (str): `"amplitude" | "angle" | "basis" | "phase" | "iqp" | "iqp-z"` (lowercase).

**Returns**
- `QuantumTensor` with 2D shape:
  - single sample: `[1, 2^num_qubits]`
  - batch: `[batch_size, 2^num_qubits]`

**Notes**
- Output dtype is `complex64` (`precision="float32"`) or `complex128` (`precision="float64"`).
- Parquet streaming currently supports `"amplitude"` and `"basis"`.
- PyTorch file inputs (`.pt`, `.pth`) require building with the `pytorch` feature.
- Remote URL query/fragment is not supported (`?versionId=...`, `#...`).

### Advanced Encoding Support Matrix

For the advanced methods documented below, support currently differs by route:

| Surface | `phase` | `iqp` | `iqp-z` |
|---------|---------|-------|---------|
| `qumat.qdp.QdpEngine(..., backend="cuda").encode(...)` | ✅ | ✅ | ✅ |
| CUDA tensor direct-input fast path | ❌ | ✅ | ✅ |
| `qumat.qdp.QdpEngine(..., backend="amd")` | ❌ | ❌ | ❌ |

Notes:
- `phase` is implemented and tested on the CUDA encode path, but it is not part of the CUDA tensor direct-input fast path today.
- The AMD Triton route currently supports `amplitude`, `angle`, and `basis` only.

**Raises**
- `RuntimeError`: Invalid inputs, shapes, dtypes, or unsupported formats.

### `encode_from_tensorflow(path, num_qubits, encoding_method="amplitude") -> QuantumTensor`

Encode directly from a TensorFlow TensorProto file (`.pb`).

## Class: `QuantumTensor`

DLPack wrapper for GPU-resident quantum states.

### `__dlpack__(stream=None)`

Return a DLPack PyCapsule. This can be consumed only once.

### `__dlpack_device__()`

Return `(device_type, device_id)`; CUDA devices report `(2, gpu_id)`.

**Ownership & Lifetime**
- If not consumed, memory is freed when the object is dropped.
- If consumed, ownership transfers to the consumer (e.g., PyTorch).

## Encoding Methods

### `amplitude`

- Input length `<= 2^num_qubits`; remaining entries are zero-padded.
- L2 normalization is applied.
- Zero vectors / NaN / Inf raise errors.

### `angle`

- Each sample must provide exactly `num_qubits` angles.
- NaN / Inf raise errors.

### `basis`

- Each sample provides one integer index in `[0, 2^num_qubits)`.

### `phase`

- Expects `n` phase angles for `n = num_qubits`.
- Produces a product state of the form `(1/sqrt(2^n)) * sum_b exp(i * phase(b)) |b>`.
- All phase angles must be finite (no NaN/Inf).
- Available on the CUDA encode path; not currently exposed through the AMD route.

### `iqp`

- Expects `n + n*(n-1)/2` parameters for `n = num_qubits` (Z + ZZ terms).
- All parameters must be finite (no NaN/Inf).

### `iqp-z`

- Expects `n` parameters for `n = num_qubits` (Z terms only).
- All parameters must be finite (no NaN/Inf).

## Supported File Formats

- **Parquet**: `.parquet`
- **Arrow IPC**: `.arrow`, `.feather`
- **NumPy**: `.npy`
- **PyTorch**: `.pt`, `.pth` (requires `pytorch` feature)
- **TensorFlow**: `.pb` (TensorProto)

## Common Errors

- `num_qubits` out of range.
- Input length exceeds `2^num_qubits`.
- CPU NumPy / Torch inputs that are not contiguous `float64`.
- CUDA Torch tensors used with unsupported encoding methods (e.g. `phase`), or CUDA tensors that are not contiguous / not on the same device as the engine.
- DLPack capsule consumed more than once.

## Requirements & Deprecation

- Linux with an NVIDIA GPU (CUDA).
- CUDA driver/toolkit installed.
- Python 3.10–3.12 (`requires-python >=3.10,<3.13`).
- `qumat` installed (the QDP native extension ships as `qumat-qdp`).
- Optional: `torch` for DLPack integration, `numpy` for NumPy inputs.

No deprecations are planned for the initial PoC.
