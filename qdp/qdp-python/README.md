# qumat-qdp

GPU-accelerated quantum state encoding for [Apache Mahout Qumat](https://github.com/apache/mahout).

## Installation

```bash
pip install qumat[qdp]
```

Requires one of:
- NVIDIA GPU (CUDA path via `QdpEngine`)
- AMD GPU with ROCm-enabled PyTorch (ROCm path via `AmdQdpEngine`)
- AMD GPU with ROCm + Triton (ROCm Triton path via `TritonAmdEngine`)

## Usage

```python
import qumat.qdp as qdp
import torch

# Initialize engine on GPU 0
engine = qdp.QdpEngine(device_id=0)

# Encode data into quantum state
qtensor = engine.encode([1.0, 2.0, 3.0, 4.0], num_qubits=2, encoding_method="amplitude")

# Zero-copy transfer to PyTorch
tensor = torch.from_dlpack(qtensor)
print(tensor)  # Complex tensor on CUDA
```

### AMD ROCm Usage

```python
import qumat.qdp as qdp
import torch

# ROCm path (PyTorch ROCm build required: torch.version.hip is not None)
engine = qdp.AmdQdpEngine(device_id=0, precision="float32")
qtensor = engine.encode(torch.randn(8, 4, device="cuda"), 2, "amplitude")
tensor = torch.from_dlpack(qtensor)
print(tensor.device, tensor.dtype)  # cuda:0, complex64
```

### Triton AMD Backend Usage

```python
import torch
from qumat_qdp import TritonAmdEngine, create_encoder_engine

# Force Triton AMD backend
engine = TritonAmdEngine(device_id=0, precision="float32")
qt = engine.encode(torch.randn(64, 1024, device="cuda"), 10, "amplitude")
state = torch.from_dlpack(qt)

# Or route automatically:
engine_auto = create_encoder_engine(backend="auto", device_id=0, precision="float32")
qt = engine_auto.encode(torch.randn(8, 4, device="cuda"), 2, "amplitude")
state = torch.from_dlpack(qt)
```

See `qdp/qdp-python/TRITON_AMD_BACKEND.md` for setup and validation details.

## Encoding Methods

| Method | Description |
|--------|-------------|
| `amplitude` | Normalize input as quantum amplitudes |
| `angle` | Map values to rotation angles (one per qubit) |
| `basis` | Encode integer as computational basis state |
| `iqp` | IQP-style encoding with entanglement |

Backend support boundary:
- CUDA (`QdpEngine`): `amplitude`, `angle`, `basis`, `iqp`
- Triton AMD (`TritonAmdEngine` / `backend="triton_amd"`): `amplitude`, `angle`, `basis` (no `iqp` yet)

## Input Sources

```python
# Python list
qtensor = engine.encode([1.0, 2.0, 3.0, 4.0], 2, "amplitude")

# NumPy array
qtensor = engine.encode(np.array([[1, 2, 3, 4], [4, 3, 2, 1]]), 2, "amplitude")

# PyTorch tensor (CPU or CUDA)
qtensor = engine.encode(torch.tensor([1.0, 2.0, 3.0, 4.0]), 2, "amplitude")

# File formats
qtensor = engine.encode("data.parquet", 10, "amplitude")
qtensor = engine.encode("data.arrow", 10, "amplitude")
qtensor = engine.encode("data.npy", 10, "amplitude")
qtensor = engine.encode("data.pt", 10, "amplitude")

# Remote object storage URLs (requires building with remote-io feature)
qtensor = engine.encode("s3://my-bucket/data.parquet", 10, "amplitude")
qtensor = engine.encode("gs://my-bucket/data.parquet", 10, "amplitude")
```

## Links

- [Documentation](https://mahout.apache.org/)
- [GitHub](https://github.com/apache/mahout)
- [Qumat Package](https://pypi.org/project/qumat/)

## License

Apache License 2.0
