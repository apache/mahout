# qumat-qdp

GPU-accelerated quantum state encoding for [Apache Mahout Qumat](https://github.com/apache/mahout).

## Installation

```bash
pip install qumat[qdp]
```

Requires CUDA-capable GPU.

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

## Encoding Methods

| Method | Description |
|--------|-------------|
| `amplitude` | Normalize input as quantum amplitudes |
| `angle` | Map values to rotation angles (one per qubit) |
| `basis` | Encode integer as computational basis state |
| `iqp` | IQP-style encoding with entanglement |

### Pipeline / loader dtype (Rust internals)

`QuantumDataLoader` and `run_throughput_pipeline` build a Rust `PipelineConfig` with an
`encoding` plus a `dtype` (float32 vs float64). The prefetch thread can only keep an
end-to-end **float32 host batch** for encodings whose GPU stack implements the batch **f32**
path (`encode_batch_f32`). **Today that is amplitude only.** Angle and basis still fall back
to float64 for that loop until their batch f32 implementations exist. The eventual full
matrix (e.g. angle/basis under `supports_f32` once kernels are wired) is broader than what
the pipeline uses today.

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
