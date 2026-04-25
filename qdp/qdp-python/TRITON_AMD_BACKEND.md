# Triton AMD Backend

This document describes the Triton-based implementation used by the QDP AMD backend on ROCm.

## Prerequisites

- AMD GPU supported by ROCm
- ROCm driver/runtime installed
- PyTorch ROCm build (`torch.version.hip` is not `None`)
- Triton installed with HIP support

## Install (project environment)

```bash
uv sync --project qdp/qdp-python --group benchmark --active
```

This installs the benchmark group, including `triton`.

## Runtime capability checks

Use:

```python
from qumat_qdp import is_triton_amd_available
print(is_triton_amd_available())
```

The check validates:
- ROCm runtime is visible through PyTorch
- Triton is importable
- Triton active backend is HIP (when query is available)

## Usage

### Unified AMD routing (recommended)

```python
import torch
from qumat_qdp import QdpEngine

engine = QdpEngine(device_id=0, precision="float32", backend="amd")
x = torch.randn(64, 1024, device="cuda", dtype=torch.float32)
qt = engine.encode(x, num_qubits=10, encoding_method="amplitude")
state = torch.from_dlpack(qt)
```

The public `QdpEngine(..., backend="amd")` route goes directly to the Triton
engine and returns the Python `QuantumTensorWrapper` compatibility wrapper.
The CUDA route remains Rust-owned and returns the extension `QuantumTensor`.

### Triton implementation details

```python
import torch
from qumat_qdp.triton_amd import TritonAmdEngine

engine = TritonAmdEngine(device_id=0, precision="float32")
x = torch.randn(64, 1024, device="cuda", dtype=torch.float32)
qt = engine.encode(x, num_qubits=10, encoding_method="amplitude")
state = torch.from_dlpack(qt)
```

Supported methods:
- `amplitude`
- `angle`
- `basis`
- `iqp` (full, with ZZ entanglement)
- `iqp-z` (Z-only diagonal, no ZZ pairs)
- `phase`

## Correctness tests

Run Triton backend tests:

```bash
uv run --project qdp/qdp-python pytest qdp/qdp-python/tests/test_triton_amd_backend.py -q
uv run --project qdp/qdp-python pytest -m rocm qdp/qdp-python/tests -q
```

Tests include:
- parity against Torch reference outputs (amplitude/angle/basis/iqp)
- structural checks for `phase` (output is a unit-norm product state)
- optional parity against CUDA backend reference (when NVIDIA CUDA path is present)

## Baseline benchmark

```bash
uv run --project qdp/qdp-python python qdp/qdp-python/benchmark/benchmark_triton_amd.py \
  --qubits 12 --batch-size 64 --batches 200 --encoding-method amplitude
```
