---
layout: page
title: Examples - QDP
---

# Examples

This document demonstrates practical usage of **Apache Mahout QDP (Quantum Data Plane)**
for direct quantum state preparation on GPU. These examples focus on end-to-end workflows
and typical usage patterns.

---

## Example 1: Basic Amplitude Encoding

Encode a classical vector into a GPU-backed quantum state using amplitude encoding.

```python
import mahout_qdp as qdp
import pyarrow as pa
import torch

raw_vector = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
input_data = pa.array(raw_vector, type=pa.float64())

state = qdp.encode_amplitude(input_data)
tensor = torch.utils.dlpack.from_dlpack(state.to_dlpack())

print(tensor)
print("Device:", tensor.device)
```

## Example 2: Automatic Normalization and Padding

QDP automatically normalizes input vectors and pads them to the nearest power-of-two length.

```python
import mahout_qdp as qdp
import pyarrow as pa

raw_vector = [3.0, 4.0, 5.0]
input_data = pa.array(raw_vector, type=pa.float64())

state = qdp.encode_amplitude(input_data)
```

No manual preprocessing is required.