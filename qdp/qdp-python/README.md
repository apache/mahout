# qumat-qdp

GPU-accelerated quantum state encoding for [Apache Mahout Qumat](https://github.com/apache/mahout).

## Installation

```bash
pip install qumat[qdp]
```

Requires one of:
- NVIDIA GPU (CUDA path via `QdpEngine`)
- AMD GPU with ROCm (AMD path via `QdpEngine(backend="amd")`)

Recommended environment setup:

```bash
python -m venv .venv
source .venv/bin/activate

# Install the GPU runtime for your platform first:
# - NVIDIA users: CUDA-compatible torch / triton
# - AMD users: ROCm-compatible torch / triton

uv sync --active --project qdp/qdp-python --group dev
```

Use `--active` so `uv` reuses the environment that already has the correct GPU
runtime stack.

## Usage

```python
import qumat.qdp as qdp
import torch

# Initialize the unified QDP engine on GPU 0.
# Choose the backend explicitly.
engine = qdp.QdpEngine(device_id=0, backend="cuda")

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

# Unified AMD engine route
engine = qdp.QdpEngine(device_id=0, precision="float32", backend="amd")
qt = engine.encode(torch.randn(8, 4, device="cuda"), 2, "amplitude")
state = torch.from_dlpack(qt)
print(state.device, state.dtype)  # cuda:0, complex64

```

The public `QdpEngine` is a unified Python facade with explicit backend selection:
- `backend="cuda"` routes to the Rust `_qdp.QdpEngine`
- `backend="amd"` routes to the Triton AMD engine directly

See `qdp/qdp-python/TRITON_AMD_BACKEND.md` for Triton AMD setup and validation details.

## Encoding Methods

| Method | Description |
|--------|-------------|
| `amplitude` | Normalize input as quantum amplitudes |
| `angle` | Map values to rotation angles (one per qubit) |
| `basis` | Encode integer as computational basis state |
| `iqp` | IQP-style encoding with full ZZ entanglement |
| `iqp-z` | IQP encoding with Z-only diagonal (no ZZ pairs) |
| `phase` | Per-qubit phase product state via H⊗P(x_k) |

Backend support boundary:
- CUDA (`QdpEngine`): `amplitude`, `angle`, `basis`, `iqp`, `iqp-z`, `phase`
  - `phase` is currently only reachable on the CUDA path via host inputs
    (Python list / NumPy / file / CPU torch tensor). The Python extension's
    CUDA-tensor validation does not yet allowlist `phase`; cuda-resident
    torch tensors must use `.cpu()` first when targeting `phase`. Tracked as
    a follow-up.
- AMD (`QdpEngine(..., backend="amd")`): `amplitude`, `angle`, `basis`, `iqp`, `iqp-z`, `phase`

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
