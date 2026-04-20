---
title: Getting Started with QDP
---

# Getting Started with QDP

QDP (Quantum Data Plane) is a GPU-accelerated library for encoding classical data into quantum states.

## Prerequisites

- Python 3.10+
- One of:
  - NVIDIA GPU with CUDA toolkit installed (`nvcc --version` to verify)
  - AMD GPU with ROCm and PyTorch ROCm installed (`python -c "import torch; print(torch.version.hip)"`)

## Installation

```bash
pip install qumat[qdp]
```

For development (from source):

```bash
git clone https://github.com/apache/mahout.git
cd mahout
uv sync --group dev --extra qdp
source .venv/bin/activate
uv run --active maturin develop --manifest-path qdp/qdp-python/Cargo.toml
```

For AMD ROCm with the Triton backend:

```bash
uv sync --group dev --extra qdp
pip install triton
```

## Quick Start

```python
import torch
from qumat.qdp import QdpEngine

engine = QdpEngine(device_id=0, backend="cuda")
data = [0.5, 0.5, 0.5, 0.5]
qtensor = engine.encode(data, num_qubits=2, encoding_method="amplitude")

# Convert to PyTorch (zero-copy)
tensor = torch.from_dlpack(qtensor)  # Note: can only be consumed once
```

AMD ROCm uses the same API with a different backend selector:

```python
engine = QdpEngine(device_id=0, backend="amd", precision="float32")
qtensor = engine.encode(data, num_qubits=2, encoding_method="amplitude")
tensor = torch.from_dlpack(qtensor)
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

Remote object storage URLs are supported when QDP is built with `remote-io`:

```python
engine.encode("s3://my-bucket/path/data.parquet", num_qubits=10, encoding_method="amplitude")
engine.encode("gs://my-bucket/path/data.parquet", num_qubits=10, encoding_method="amplitude")
```

Notes:
- Remote URL query/fragment is not supported (`?versionId=...`, `#...`).
- Streaming still requires `.parquet`.

## Tips

- Use `precision="float64"` for higher precision: `QdpEngine(0, precision="float64")`
- NumPy inputs must be `float64` dtype
- Streaming only works with Parquet files

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Import fails | Activate root venv: `source mahout/.venv/bin/activate` (or `cd mahout && source .venv/bin/activate`) |
| CUDA errors | Run `cargo clean` in `qdp/` and rebuild |
| AMD backend unavailable | Verify ROCm PyTorch reports `torch.version.hip` and install `triton` |
| Out of memory | Reduce `num_qubits` or use `precision="float32"` |

## Next Steps

- [Concepts](./concepts/) - Learn about quantum encoding concepts
- [API Reference](./api/) - Detailed API documentation
- [Examples](./examples/) - More usage examples

## Help

- Mailing List: dev@mahout.apache.org
- [GitHub Issues](https://github.com/apache/mahout/issues)
