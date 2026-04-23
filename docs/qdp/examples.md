---
title: Examples - QDP
---

# Examples

:::info Prerequisites
These examples use the CUDA route for the advanced methods below.

- Install QDP support: `pip install qumat[qdp]`
- Optional deps used below: `torch`, `numpy`
- The AMD route currently supports `amplitude`, `angle`, and `basis`, but not `phase`, `iqp`, or `iqp-z`.
:::

## Example 1: Basic Encode + DLPack to PyTorch

```python
import torch
import qumat.qdp as qdp

engine = qdp.QdpEngine(device_id=0, backend="cuda")
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

## Example 2: Phase Encoding

```python
import torch
import qumat.qdp as qdp

engine = qdp.QdpEngine(device_id=0, backend="cuda")
qtensor = engine.encode(
    [0.0, torch.pi / 2],
    num_qubits=2,
    encoding_method="phase",
)

state = torch.from_dlpack(qtensor)
assert state.shape == (1, 4)
```

## Example 3: IQP and IQP-Z

```python
import torch
import qumat.qdp as qdp

engine = qdp.QdpEngine(device_id=0, backend="cuda")

# 2 qubits:
# - iqp-z expects n = 2 parameters
# - iqp expects n + n*(n-1)/2 = 3 parameters
state_z = torch.from_dlpack(
    engine.encode([0.1, -0.2], num_qubits=2, encoding_method="iqp-z")
)
state_full = torch.from_dlpack(
    engine.encode([0.1, -0.2, 0.3], num_qubits=2, encoding_method="iqp")
)

assert state_z.shape == (1, 4)
assert state_full.shape == (1, 4)
```
