---
title: Getting Started with QDP
---

# Getting Started with QDP

QDP (Quantum Data Plane) is a GPU-accelerated library for encoding classical data into quantum states.

## Prerequisites

- Linux with NVIDIA GPU
- CUDA toolkit installed (`nvcc --version` to verify)
- Python 3.10+

## Installation

```bash
pip install qumat[qdp]
```

For development (from source):

```bash
git clone https://github.com/apache/mahout.git
cd mahout/qdp/qdp-python
uv venv -p python3.10 && source .venv/bin/activate
uv sync --group dev && uv run maturin develop
```

## Quick Start

```python
import torch
from qumat.qdp import QdpEngine

engine = QdpEngine(0)  # GPU device 0
data = [0.5, 0.5, 0.5, 0.5]
qtensor = engine.encode(data, num_qubits=2, encoding_method="amplitude")

# Convert to PyTorch (zero-copy)
tensor = torch.from_dlpack(qtensor)  # Note: can only be consumed once
```

## Encoding Methods

| Method | Constraint | Example |
|--------|-----------|---------|
| `amplitude` | data length ≤ 2^num_qubits | `encode([0.5, 0.5, 0.5, 0.5], num_qubits=2, encoding_method="amplitude")` |
| `angle` | data length = num_qubits | `encode([0.1, 0.2, 0.3, 0.4], num_qubits=4, encoding_method="angle")` |
| `basis` | data length = num_qubits | `encode([1, 0, 1, 1], num_qubits=4, encoding_method="basis")` |

## File Inputs

```python
engine.encode("data.parquet", num_qubits=10, encoding_method="amplitude")  # also: .arrow, .npy, .pt, .pb
```

## Tips

- Use `precision="float64"` for higher precision: `QdpEngine(0, precision="float64")`
- NumPy inputs must be `float64` dtype
- Streaming only works with Parquet files

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Import fails | Activate venv: `source .venv/bin/activate` |
| CUDA errors | Run `cargo clean` in `qdp/` and rebuild |
| Out of memory | Reduce `num_qubits` or use `precision="float32"` |

## Next Steps

- [Concepts](concepts.md) - Learn about quantum encoding concepts
- [API Reference](api.md) - Detailed API documentation
- [Examples](examples.md) - More usage examples

## Help

- Mailing List: dev@mahout.apache.org
- [GitHub Issues](https://github.com/apache/mahout/issues)
