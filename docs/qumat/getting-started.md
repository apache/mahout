---
title: Getting Started
---

# Getting Started with Qumat

## Installation

Install Qumat from PyPI:

```bash
pip install qumat
```

Install with QDP (Quantum Data Plane) support:

```bash
pip install qumat[qdp]
```

QDP (the `qdp` extra) requires Python 3.10-3.12 and is not available on Python 3.13.

## From Source (Development)

For development or the latest changes, use [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/apache/mahout.git
cd mahout
pip install uv
uv sync                     # Core Qumat
uv sync --python '>=3.10,<3.13' --extra qdp  # With QDP (NVIDIA GPU + CUDA; QDP needs Python 3.10-3.12)
```

:::note Why uv?
The repository commits a `uv.lock`, so `uv sync` gives a reproducible development environment across Python 3.10-3.13.
:::

## Basic Usage

```python
from qumat import QuMat

qumat = QuMat({
    "backend_name": "qiskit",
    "backend_options": {"simulator_type": "aer_simulator"},
})
qumat.create_empty_circuit(num_qubits=2)
qumat.apply_hadamard_gate(0)
qumat.apply_cnot_gate(0, 1)
result = qumat.execute_circuit()
print(result)
```

## Dependencies

Prior to installation, ensure Python 3.10-3.13 is installed. Dependencies such as Qiskit, Cirq, and Amazon Braket SDK are managed automatically. QDP (the `qdp` extra) requires Python 3.10-3.12 and is not available on Python 3.13.

## Apache Release

Official source releases are available at [apache.org/dist/mahout](http://www.apache.org/dist/mahout).

To verify the integrity of a downloaded release:

```bash
gpg --import KEYS
gpg --verify mahout-qumat-0.5.zip.asc mahout-qumat-0.5.zip
```

## Examples

Refer to repository examples:

- `examples/Simple_Example.ipynb`
- `examples/Optimization_Example.ipynb`

## Next Steps

- [Basic Gates](./basic-gates)
- [API Reference](./api)
- [QDP Getting Started](../qdp/getting-started.md)
