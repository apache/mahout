---
title: Examples - QDP
---

# Examples

:::info Prerequisites
These examples require Linux with an NVIDIA GPU and CUDA.

- Install QDP support: `pip install qumat[qdp]`
- Optional deps used below: `torch`, `numpy`
:::

## Example 1: Basic Encode + DLPack to PyTorch

```python
import torch
import qumat.qdp as qdp

engine = qdp.QdpEngine(device_id=0)
qtensor = engine.encode(
    [0.5, 0.5, 0.5, 0.5],
    num_qubits=2,
    encoding_method="amplitude",
)

# DLPack capsule is single-consume; convert once.
tensor = torch.from_dlpack(qtensor)
assert tensor.is_cuda
assert tensor.shape == (1, 4)  # [batch_size, 2^num_qubits]
```

## Example 2: NumPy Batch + `.npy` File Input

```python
from pathlib import Path
import tempfile

import numpy as np
import torch
import qumat.qdp as qdp

engine = qdp.QdpEngine(device_id=0)
num_qubits = 2
state_len = 2**num_qubits

# In-memory NumPy batch input (dtype must be float64).
batch = np.array(
    [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]],
    dtype=np.float64,
)
qtensor_mem = engine.encode(batch, num_qubits=num_qubits, encoding_method="amplitude")
tensor_mem = torch.from_dlpack(qtensor_mem)
assert tensor_mem.shape == (2, state_len)

# File input from .npy
with tempfile.TemporaryDirectory() as tmpdir:
    npy_path = Path(tmpdir) / "batch.npy"
    np.save(npy_path, batch)
    qtensor_file = engine.encode(
        str(npy_path),
        num_qubits=num_qubits,
        encoding_method="amplitude",
    )
    tensor_file = torch.from_dlpack(qtensor_file)
    assert tensor_file.shape == (2, state_len)
```
