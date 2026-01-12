# qdp-python

PyO3 Python bindings for Apache Mahout QDP.

## Usage

```python
from _qdp import QdpEngine

# Initialize on GPU 0 (defaults to float32 output)
engine = QdpEngine(0)

# Optional: request float64 output if you need higher precision
# engine = QdpEngine(0, precision="float64")

# Encode data from Python list
data = [0.5, 0.5, 0.5, 0.5]
dlpack_ptr = engine.encode(data, num_qubits=2, encoding_method="amplitude")

# Or encode from file formats
tensor_parquet = engine.encode_from_parquet("data.parquet", 10, "amplitude")
tensor_arrow = engine.encode_from_arrow_ipc("data.arrow", 10, "amplitude")
```

## Build from source
```bash
# add a uv python 3.11 environment
uv venv -p python3.11
source .venv/bin/activate
```
```bash
uv sync --group dev
uv run maturin develop
```

## Encoding methods

- `"amplitude"` - Amplitude encoding
- `"angle"` - Angle encoding
- `"basis"` - Basis encoding

## File format support

- **Parquet** - `encode_from_parquet(path, num_qubits, encoding_method)`
- **Arrow IPC** - `encode_from_arrow_ipc(path, num_qubits, encoding_method)`

## Adding new bindings

1. Add method to `#[pymethods]` in `src/lib.rs`:
```rust
#[pymethods]
impl QdpEngine {
    fn my_method(&self, arg: f64) -> PyResult<String> {
        Ok(format!("Result: {}", arg))
    }
}
```

2. Rebuild: `uv run maturin develop`

3. Use in Python:
```python
engine = QdpEngine(0)
result = engine.my_method(42.0)
```
