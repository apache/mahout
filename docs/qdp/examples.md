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
