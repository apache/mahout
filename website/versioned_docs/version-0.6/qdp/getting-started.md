---
title: Getting Started with QDP
---

# Getting Started with QDP

QDP (Quantum Data Plane) is a GPU-accelerated library for encoding classical data into quantum states. It ships as part of the Apache Mahout `qumat` package and exposes a single `QdpEngine` facade with explicit backend selection for NVIDIA CUDA and AMD ROCm.

## Prerequisites

- Python 3.10+
- One of:
  - NVIDIA GPU with a CUDA-compatible PyTorch build (verify with `python -c "import torch; print(torch.cuda.is_available())"`)
  - AMD GPU with a ROCm-compatible PyTorch build (verify with `python -c "import torch; print(torch.version.hip)"`) plus Triton with HIP support

## Supported GPU Backends

QDP currently supports the following GPU backends:

- NVIDIA CUDA backend
  - CUDA builds target NVIDIA GPUs supported by the installed CUDA toolkit.
  - The current default CUDA architecture shortlist spans common NVIDIA
    generations from Turing through Blackwell: `sm_75`, `sm_80`, `sm_86`,
    `sm_89`, `sm_90`, `sm_100`, and `sm_120`.
  - The final generated CUDA targets depend on the local `nvcc` supported
    architectures.

- AMD ROCm backend
  - ROCm builds target AMD GPUs supported by the installed ROCm stack and the
    Triton backend used by QDP.
  - The final supported devices depend on the local ROCm environment.

## Installation

```bash
pip install qumat[qdp]
```

For development from source:

```bash
git clone https://github.com/apache/mahout.git
cd mahout
uv sync --active --project qdp/qdp-python --group dev
source .venv/bin/activate
uv run --active maturin develop --manifest-path qdp/qdp-python/Cargo.toml
```

For AMD ROCm with the Triton backend, install a ROCm-compatible PyTorch first, then sync the benchmark group (which includes `triton`):

```bash
uv sync --active --project qdp/qdp-python --group benchmark
```

You can confirm the AMD route is usable at runtime:

```python
from qumat_qdp import is_triton_amd_available
print(is_triton_amd_available())
```

## Quick Start

```python
import torch
from qumat.qdp import QdpEngine

# CUDA is the default backend.
engine = QdpEngine(device_id=0)
data = [0.5, 0.5, 0.5, 0.5]
qtensor = engine.encode(data, num_qubits=2, encoding_method="amplitude")

# Zero-copy hand-off to PyTorch via DLPack.
# On CUDA the result is a native QuantumTensor whose DLPack export is
# single-use, so pass it to only one consumer.
tensor = torch.from_dlpack(qtensor)
```

To run on AMD ROCm, pass `backend="amd"`:

```python
engine = QdpEngine(device_id=0, backend="amd")
qtensor = engine.encode(data, num_qubits=2, encoding_method="amplitude")
tensor = torch.from_dlpack(qtensor)
```

## Encoding Methods

| Method | Constraint | Example |
|--------|-----------|---------|
| `amplitude` | CUDA: up to `2^num_qubits` values per sample (zero-padded). AMD: exactly `2^num_qubits` values per sample | `engine.encode([0.5, 0.5, 0.5, 0.5], num_qubits=2, encoding_method="amplitude")` |
| `angle` | one angle per qubit (`num_qubits` values per sample) | `engine.encode([0.1, 0.2, 0.3, 0.4], num_qubits=4, encoding_method="angle")` |
| `basis` | one integer index per sample, `0 ≤ index < 2^num_qubits` | `engine.encode([13], num_qubits=4, encoding_method="basis")` (encodes basis state 13) |
| `phase` | one phase angle per qubit (`num_qubits` values per sample) | `engine.encode([0.0, 1.57], num_qubits=2, encoding_method="phase")` |
| `iqp`, `iqp-z` | data length matches the IQP parameter shape for `num_qubits` (`iqp-z`: `n`; `iqp`: `n + n*(n-1)/2`) | `engine.encode([0.1, 0.2, 0.3], num_qubits=2, encoding_method="iqp")` |

> Backend support: `amplitude`, `angle`, and `basis` work on both CUDA and AMD. `phase`, `iqp`, and `iqp-z` are CUDA-only today. See [Concepts](./concepts/) for IQP parameter layout, phase encoding semantics, and method details.

## File Inputs

File and remote-URL inputs are handled by the CUDA route. The AMD route's `encode()` accepts only array-like data (`list`, `numpy.ndarray`, `torch.Tensor`); load files into memory first, or use the CUDA route for the loader/streaming features.

On CUDA, `engine.encode(...)` accepts a path or `pathlib.Path` for supported file formats:

```python
engine.encode("data.parquet", num_qubits=10, encoding_method="amplitude")
# Other supported formats: .arrow, .feather, .npy, .pt, .pth, .pb
```

Remote object storage URLs are accepted when QDP is built with the `remote-io` feature:

```python
engine.encode("s3://my-bucket/path/data.parquet", num_qubits=10, encoding_method="amplitude")
engine.encode("gs://my-bucket/path/data.parquet", num_qubits=10, encoding_method="amplitude")
```

Notes:
- Query strings and fragments in remote URLs (e.g. `?versionId=...`, `#...`) are not supported.
- Streaming sources are limited to `.parquet`; see the [Python API](./python-api/) page for the loader/streaming surface.

## Tips

- Default `precision` is `"float32"`; pass `precision="float64"` for higher precision: `QdpEngine(device_id=0, precision="float64")`.
- NumPy inputs must be `float64` dtype. CUDA `torch.Tensor` inputs accept `float32` or `float64` for amplitude; angle and IQP methods require `float64` for batched inputs.
- Backend selection is explicit; valid values are `"cuda"` and `"amd"` (with `"triton_amd"` accepted as an alias for `"amd"`).

## Troubleshooting

| Problem | Resolution |
|---------|-----------|
| `ImportError` on `qumat.qdp` | Activate the project venv: `source mahout/.venv/bin/activate` |
| `ValueError: Unsupported backend` | Use `backend="cuda"` or `backend="amd"` |
| AMD route unavailable | Confirm `torch.version.hip` is set, then check `qumat_qdp.is_triton_amd_available()` and install the benchmark group if `triton` is missing |
| CUDA build failures (from source) | Run `cargo clean` in `qdp/` and rebuild with `maturin develop` |
| Out of memory | Reduce `num_qubits` or use `precision="float32"` |

## Next Steps

- [Concepts](./concepts/) - Quantum encoding concepts and architecture
- [API Reference](./api/) - Public `qumat.qdp` API
- [Python API](./python-api/) - `qumat_qdp` package surface, loaders, and benchmarks
- [Examples](./examples/) - Usage examples

## Help

- Mailing List: dev@mahout.apache.org
- [GitHub Issues](https://github.com/apache/mahout/issues)
