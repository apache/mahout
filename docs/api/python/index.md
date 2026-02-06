---
title: Python API Reference
sidebar_label: Python API
sidebar_position: 1
---

# Python API Reference

Auto-generated API documentation for Apache Mahout's Python libraries.

## Qumat - Quantum Circuit Abstraction

The `qumat` package provides a unified interface for quantum circuit programming across multiple backends (Qiskit, Cirq, Amazon Braket).

### Core Classes

- **[QuMat](./qumat.md)** - Main quantum circuit interface for creating and executing quantum circuits

### Quick Example

```python
from qumat import QuMat

# Initialize with Qiskit backend
qc = QuMat({
    "backend_name": "qiskit",
    "backend_options": {"simulator_type": "statevector", "shots": 1000}
})

# Create and run a Bell state circuit
qc.create_empty_circuit(2)
qc.apply_hadamard_gate(0)
qc.apply_cnot_gate(0, 1)
results = qc.execute_circuit()
```

## QDP - Quantum Data Plane

The `qumat_qdp` package provides GPU-accelerated quantum state encoding for machine learning pipelines.

:::note GPU Required
The QDP Python bindings require a CUDA-capable GPU. See the [Rust API documentation](/api/rust/qdp_core/index.html) for the underlying implementation details.
:::

### Core Classes

- **QdpEngine** - GPU-accelerated quantum state encoder
- **QuantumTensor** - DLPack-compatible tensor wrapper for zero-copy GPU data sharing
- **QuantumDataLoader** - Batch iterator for training pipelines
- **QdpBenchmark** - Performance measurement utilities

### Quick Example

```python
from qumat_qdp import QdpEngine, QuantumDataLoader
import torch

# GPU encoding
engine = QdpEngine(device_id=0, precision="float32")
tensor = engine.encode(data, num_qubits=4, encoding_method="amplitude")

# Batch loading for ML pipelines
loader = (QuantumDataLoader(device_id=0)
          .qubits(16)
          .encoding("amplitude")
          .batches(100, size=64))

for qt in loader:
    batch = torch.from_dlpack(qt)
    # ... training step
```

---

*This documentation is auto-generated from source code docstrings using [pdoc](https://pdoc.dev).*
