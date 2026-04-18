# Triton AMD Backend (`triton_amd`)

This document describes how to use the QDP Triton backend on AMD GPUs via ROCm.

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

### Direct Triton backend

```python
import torch
from qumat_qdp import TritonAmdKernel

engine = TritonAmdKernel(device_id=0, precision="float32")
x = torch.randn(64, 1024, device="cuda", dtype=torch.float32)
qt = engine.encode(x, num_qubits=10, encoding_method="amplitude")
state = torch.from_dlpack(qt)
```

Supported methods:
- `amplitude`
- `angle`
- `basis`

Not supported in `triton_amd`:
- `iqp` (currently CUDA backend only)

### Unified Engine Routing (recommended)

```python
from qumat_qdp import create_encoder_engine
import torch

engine = create_encoder_engine(backend="auto", device_id=0, precision="float32")
qt = engine.encode([[1.0, 0.0, 0.0, 0.0]], num_qubits=2, encoding_method="amplitude")
state = torch.from_dlpack(qt)
# auto order: triton_amd -> cuda
```

All routed backends return a unified DLPack-compatible object (`QuantumTensor` from `qumat_qdp.backend`).

## Correctness tests

Run Triton backend tests:

```bash
uv run --project qdp/qdp-python pytest qdp/qdp-python/tests/test_triton_amd_backend.py -q
uv run --project qdp/qdp-python pytest -m rocm qdp/qdp-python/tests -q
```

Tests include:
- parity against Torch reference outputs (amplitude/angle/basis)
- optional parity against CUDA backend reference (when NVIDIA CUDA path is present)

## Baseline benchmark

```bash
uv run --project qdp/qdp-python python qdp/qdp-python/benchmark/benchmark_triton_amd.py \
  --qubits 12 --batch-size 64 --batches 200 --encoding-method amplitude
```
