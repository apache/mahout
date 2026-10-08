---
title: NumPy Input
sidebar_label: NumPy Input
---

# NumPy Input

QDP accepts NumPy data in two ways:

1. an in-memory `numpy.ndarray` passed to `QdpEngine.encode(...)`, and
2. a `.npy` file path passed to `QdpEngine.encode(...)` or to
   `QuantumDataLoader(...).source_file(...)`.

Both routes end up as a flat, row-major `float64` buffer that the encoder
validates and copies to the GPU. This page lists what is accepted and the
error you get otherwise.

## Quick reference

| Property | In-memory `ndarray` (CUDA route) | `.npy` file (CUDA route / Rust loader) |
|----------|----------------------------------|----------------------------------------|
| dtype | `float64` only | `float64` only (`<f8` or `>f8`) |
| Shape | 1D (one sample) or 2D `(batch, features)` | 2D `(num_samples, sample_size)` only |
| Memory layout | C-contiguous (row-major) only | C or Fortran order (converted on read) |
| Copy | Borrowed without a Python-side copy | Whole file read into host memory |
| Empty input | Rejected by the encoder | Shape with a 0 dimension rejected |
| NaN / Inf | Rejected by the encoder | Read as-is, then rejected by the encoder |

Output precision is independent of input dtype: the input is always
`float64`, and `QdpEngine(precision="float32")` (the default) or
`precision="float64"` selects `complex64` or `complex128` output.

## In-memory arrays

```python
import numpy as np
import torch
from qumat_qdp import QdpEngine

engine = QdpEngine(device_id=0, precision="float64")

# One sample: 1D array -> output shape (1, 2**num_qubits)
x = np.array([3.0, 4.0, 0.0, 0.0])
state = torch.from_dlpack(engine.encode(x, num_qubits=2))

# A batch: 2D array (batch, features) -> output (batch, 2**num_qubits)
X = np.random.default_rng(0).standard_normal((64, 16))
states = torch.from_dlpack(engine.encode(X, num_qubits=4))
```

`encode` detects NumPy input through `__array_interface__`. Strings and
`os.PathLike` objects are checked first (they are treated as file paths);
`torch.Tensor` objects do not have `__array_interface__` and take the
PyTorch path instead.

### dtype

Only `float64` is accepted. `float32`, integer, and big-endian (`>f8`)
arrays are rejected; NumPy input is never converted for you:

```text
RuntimeError: Failed to extract 1D NumPy array. Ensure dtype is float64.
RuntimeError: Failed to extract 2D NumPy array. Ensure dtype is float64.
```

Convert explicitly with `x.astype(np.float64)`. This applies to every
encoding method, including `basis` (indices must be stored as
float64 integers such as `3.0`) and `angle`.

### Shape

- **1D** `(n,)`: a single sample. The result has batch dimension 1.
- **2D** `(batch, n)`: `batch` samples of `n` features each.
- Anything else is rejected before any GPU work:

```text
RuntimeError: Unsupported array shape: 3D. Expected 1D array for single
sample encoding or 2D array (batch_size, features) for batch encoding.
```

How many features `n` each method expects (see [Concepts](./concepts)):

| Method | Features per sample |
|--------|---------------------|
| `amplitude` | 1 to `2**num_qubits` (shorter input is zero-padded) |
| `angle`, `phase`, `iqp-z` | exactly `num_qubits` |
| `iqp` | exactly `num_qubits + num_qubits*(num_qubits-1)/2` |
| `basis` | exactly 1 (the basis-state index) |

### Memory layout

The array must be C-contiguous (row-major). Slices with a step, transposes
and Fortran-ordered arrays are rejected rather than copied:

```text
RuntimeError: NumPy array must be C-contiguous (row-major). Fortran-ordered,
transposed or strided arrays are not supported
(try: data = np.ascontiguousarray(data))
```

```python
X_t = X.T                        # transposed view: not C-contiguous
engine.encode(np.ascontiguousarray(X_t), num_qubits=4)  # OK
```

Read-only arrays (for example from `np.load(..., mmap_mode="r")`) are fine;
QDP only reads the buffer.

### Values

The encoders validate values after extraction. For `amplitude`, a sample
whose L2 norm is zero, NaN or infinite is rejected (the state is
`x / ||x||`, which is undefined for a zero vector). `angle`, `phase` and
`iqp` reject non-finite values; `basis` rejects negative, non-integer or
out-of-range indices. These errors are raised as
`RuntimeError: Encoding failed: Invalid input: ...`.

## `.npy` files

```python
import numpy as np
from qumat_qdp import QdpEngine

X = np.random.default_rng(0).standard_normal((1000, 16))
np.save("features.npy", X)  # float64, 2D

engine = QdpEngine(device_id=0)
qt = engine.encode("features.npy", num_qubits=4)  # str or pathlib.Path
```

The file is read by the Rust `NumpyReader` (`qdp-core/src/readers/numpy.rs`)
using `ndarray-npy`. The path must end in `.npy`; `.npz` archives are not
supported.

Accepted files:

- dtype `float64`, either little-endian (`<f8`, the default on x86 and
  ARM) or big-endian (`>f8`).
- exactly 2D, shape `(num_samples, sample_size)`, both dimensions > 0.
  Save a single sample as `x.reshape(1, -1)`.
- `fortran_order` `True` or `False`. Fortran files are converted to row
  order on read, so each row is still one sample.

The whole array is loaded into host memory before encoding, so very large
files need enough RAM for one full copy (see issue #787 for a proposed
size check). For datasets larger than memory, use Parquet with
`source_file(path, streaming=True)`.

Errors from the reader (prefixed with `Encoding from NumPy failed:` when
raised by `QdpEngine.encode`):

| Cause | Message contains |
|-------|------------------|
| File does not exist | `I/O error: NumPy file not found: <path>` |
| float32 file | `incorrect descriptor ('<f4') for this type` |
| int64 file | `incorrect descriptor ('<i8') for this type` |
| 1D or 3D array | `ndim 1 of array did not match Dimension type with NDIM = Some(2)` |
| A zero dimension, e.g. `(0, 4)` | `Both dimensions must be > 0` |
| Truncated file | `reached EOF before reading all data` |
| Not an NPY file | `error parsing header: start does not match magic string` |

NaN and Inf values are read unchanged; the encoder then rejects them as
described under [Values](#values).

### Loader and pipeline

`QuantumDataLoader(...).source_file("data.npy")` with the default Rust
backend uses the same reader, so the same file rules apply. With
`.dtype("float32")` the loaded `float64` values are narrowed to `float32`
after reading (values outside the float32 range become ±Inf, and a
warning is logged). `basis` indices stay `float64`.

The explicit PyTorch reference backend (`.backend("pytorch")`) loads
`.npy` files with `np.load` instead and is more permissive: it accepts any
numeric dtype (converted to float64), 1D files (treated as one sample)
and any memory layout. It rejects arrays with more than two dimensions
and, unlike the Rust reader, big-endian files (`ValueError: given numpy
array has byte order different from the native byte order`).

## Example script

[`examples/qdp/numpy_input.py`](https://github.com/apache/mahout/blob/main/examples/qdp/numpy_input.py)
converts float32 and transposed arrays into the accepted form, saves a
`.npy` file, and (on a GPU) checks that the array and file give the same
states. The preparation part runs without a GPU.

## Testing

`QdpEngine` needs a CUDA device, so the in-memory and file encode paths
are covered by GPU tests. The `.npy` reader and the PyTorch reference
loader run on any machine.

CPU-only (no GPU needed):

```bash
# Rust: .npy reader behaviour (dtype, byte order, layout, shape, errors)
cargo test --manifest-path qdp/Cargo.toml -p qdp-core \
  --test numpy --test numpy_edge_cases

# Python: PyTorch reference loader reading .npy files
pytest testing/qdp_python/test_fallback.py -k npy
```

GPU (`@pytest.mark.gpu`, skipped automatically without a device):

```bash
pytest testing/qdp/test_numpy.py
```
