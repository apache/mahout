---
title: Examples - QDP
---

# Examples

This document presents **practical usage examples** of
**Apache Mahout QDP (Quantum Data Plane)**.

QDP is designed to support **data-centric quantum state preparation**
with an emphasis on **efficient memory handling and accelerator execution**.
The examples below demonstrate how QDP can be integrated into
realistic data processing and machine learning workflows.

This example intentionally complements introductory API-level examples
by focusing on **system-level usage patterns** and end-to-end behavior.

---

## Unified End-to-End Example

The following program demonstrates QDP behavior across multiple scenarios
within a **single executable workflow**:

1. Preparing a quantum state from a classical feature vector  
2. Handling unnormalized and non-power-of-two inputs  
3. Illustrating circuit-free state preparation in contrast to
   circuit-based quantum frameworks  

Presenting these scenarios together avoids repeated setup code and reflects
how QDP is typically used in practice.

---

```python
"""
Unified QDP Example

This script demonstrates:
1. End-to-end feature-vector to quantum-state preparation
2. Automatic normalization and padding handled by QDP
3. Circuit-free, data-centric state preparation

Dependencies:
- mahout_qdp
- pyarrow
- numpy
- torch
"""

import numpy as np
import pyarrow as pa
import mahout_qdp as qdp
import torch

print("\n=== Apache Mahout QDP Unified Example ===\n")

# ============================================================
# EXAMPLE 1: Feature Vector to GPU-Backed Quantum State
# ============================================================

print("Example 1: Feature vector to GPU-backed quantum state")

# Simulate output from a classical model (e.g., PCA or embedding layer)
features = np.random.rand(12).astype("float64")

# Represent the data using Apache Arrow for efficient memory handling
arrow_features = pa.array(features, type=pa.float64())

# Encode the classical data into a quantum amplitude state
quantum_state_1 = qdp.encode_amplitude(arrow_features)

# Convert the quantum state into a PyTorch tensor using DLPack
gpu_tensor = torch.utils.dlpack.from_dlpack(
    quantum_state_1.to_dlpack()
)

print("  Quantum state shape :", gpu_tensor.shape)
print("  Tensor device       :", gpu_tensor.device)
print("  State norm          :", torch.linalg.norm(gpu_tensor).item())
print()

# ============================================================
# EXAMPLE 2: Handling Non-Ideal Input
# ============================================================

print("Example 2: Handling unnormalized, arbitrary-length input")

# Input vector that is not normalized and not a power of two
raw_input = [3.2, 7.1, 1.4, 9.6, 2.8, 4.3, 6.5]

arrow_raw = pa.array(raw_input, type=pa.float64())

# QDP automatically normalizes and pads the input
quantum_state_2 = qdp.encode_amplitude(arrow_raw)

print("  Raw input length    :", len(raw_input))
print("  Quantum state ready :", quantum_state_2 is not None)
print()

# ============================================================
# EXAMPLE 3: Circuit-Free State Preparation
# ============================================================

print("Example 3: Circuit-free state preparation")

# In circuit-based frameworks, amplitude encoding typically requires
# constructing a circuit and decomposing the state into gates.
#
# QDP removes this abstraction and performs direct data-to-state encoding.

simple_data = [0.25, 0.50, 0.75, 1.00]
arrow_simple = pa.array(simple_data, type=pa.float64())

quantum_state_3 = qdp.encode_amplitude(arrow_simple)

print("  Quantum state prepared directly from data")
print("  No circuit construction required")

print("\n=== End of QDP Unified Example ===\n")
