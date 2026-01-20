---
layout: page
title: Getting Started with QDP
---

# Getting Started

QDP (Quantum Data Plane) is the GPU-accelerated component in the Apache Mahout / Qumat stack for **encoding classical data into quantum states** at high throughput.

It provides:

- **CUDA kernels + a Rust core** optimized for data-to-state encoding
- **DLPack-based, zero-copy integration** with frameworks like PyTorch (and other DLPack consumers)
- A simple API centered around **`QdpEngine`**

This page will help you:

1. Verify requirements and install QDP
2. Run your first QDP program
3. Understand the basic usage workflow
4. Troubleshoot common issues

---

## 1. Requirements and installation

### 1.1 OS / GPU / CUDA / Python

QDP currently targets:

- **OS**: Linux
- **GPU**: NVIDIA GPU
- **CUDA**: a working CUDA driver and CUDA toolkit
- **Python**: 3.10–3.12

Quick checks:

```bash
nvidia-smi
nvcc --version
```

- If `nvidia-smi` works, your NVIDIA driver can see the GPU.
- If `nvcc --version` works, your CUDA toolkit is available (and typically on your `PATH`).

### 1.2 Install Qumat + QDP (Python users)

If you want to use QDP from Python, install Qumat with the `qdp` extra (requires a CUDA-capable NVIDIA GPU):

```bash
git clone https://github.com/apache/mahout.git
cd mahout

pip install uv
uv sync --extra qdp
```

Then in Python:

```python
import qumat.qdp as qdp
```

### 1.3 Build from source (developers / contributors)

If you're developing QDP (or building the native extension locally), you can build from the `qdp/` directory:

```bash
cd qdp

# Recommended: a Python 3.11 virtual environment
uv venv -p python3.11
source .venv/bin/activate

uv sync --group dev
uv run maturin develop
```

For a full development and testing workflow, see `qdp/DEVELOPMENT.md`.

---

## 2. Your first QDP program

This minimal example shows how to:

1. Create a `QdpEngine` on GPU 0
2. Encode a length-4 vector into a 2-qubit state via amplitude encoding (length must be \(2^{n}\) for \(n\) qubits)
3. Convert the result to a PyTorch tensor via DLPack (zero-copy)

```python
import qumat.qdp as qdp
import torch

engine = qdp.QdpEngine(device_id=0)

data = [1.0, 2.0, 3.0, 4.0]
num_qubits = 2

qtensor = engine.encode(
    data,
    num_qubits=num_qubits,
    encoding_method="amplitude",
)

torch_tensor = torch.from_dlpack(qtensor)

print(torch_tensor.shape)
print(torch_tensor)
```

If this runs without import errors or CUDA runtime errors, your QDP installation is working.

---

## 3. Basic workflow overview

At a high level, using QDP looks like this:

1. **Choose an input source**
   - Python lists / NumPy arrays
   - Files (e.g. Parquet, Arrow IPC, NumPy `.npy`)
2. **Create a `QdpEngine`**
   - Select the GPU with `device_id`
   - (Optional) choose output precision (`float32` by default; `float64` if needed)
3. **Pick an encoding method**
   - `"amplitude"`: amplitude encoding
   - `"basis"`: map an integer index to a computational basis state
   - `"angle"`: angle-based encoding (availability/behavior may vary by version)
4. **Consume the output**
   - Use DLPack to hand off to PyTorch (or other DLPack consumers) without copying

### 3.1 Encoding from files

The Python bindings also support encoding directly from files. For example, Parquet:

```python
from _qdp import QdpEngine  # alternatively: `import qumat.qdp as qdp`

engine = QdpEngine(0)
qtensor = engine.encode("data.parquet", 10, "amplitude")
```

Supported file formats depend on the bindings, but commonly include:

- `.parquet`
- `.arrow` / `.feather` (Arrow IPC)
- `.npy`

Refer to `qdp/qdp-python/README.md` for the authoritative list for your version.

---

## 4. FAQ / troubleshooting

### 4.1 `ImportError: No module named '_qdp'` (or QDP cannot be imported)

- If you're using Qumat: make sure you installed with `uv sync --extra qdp`
- If you're building from source: run `uv run maturin develop` under `qdp/`
- Ensure you're running Python in the environment where QDP was installed (virtualenv/conda, etc.)

### 4.2 CUDA errors (e.g. "no CUDA installed", "invalid device ordinal")

- Verify your GPU is visible: `nvidia-smi`
- Verify your toolkit is available: `nvcc --version`
- If you changed CUDA/driver versions, rebuild the native extension:

```bash
cd qdp
cargo clean
uv run maturin develop
```

### 4.3 Out-of-memory or poor performance

- Monitor VRAM usage with `nvidia-smi`
- Prefer batch-oriented workflows when available (instead of encoding samples one-by-one)
- For deeper profiling, use NVTX + `nsys` (see `qdp/docs/observability/NVTX_USAGE.md` and `qdp/DEVELOPMENT.md`)

### 4.4 Where to go next

- `qdp/qdp-python/README.md` (Python bindings usage)
- `qdp/DEVELOPMENT.md` (developer build/test guide)
