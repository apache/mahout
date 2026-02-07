---
title: Rust API Reference
sidebar_label: Rust API (qdp-core)
sidebar_position: 2
---

# Rust API Reference — `qdp-core`

API documentation for the `qdp-core` crate (v0.2.0), the Rust engine powering Mahout's Quantum Data Plane.

## QdpEngine

Main entry point for GPU-accelerated quantum state encoding.

```rust
use qdp_core::{QdpEngine, Precision};

let engine = QdpEngine::new(0)?;                        // GPU device 0, f32
let engine = QdpEngine::new_with_precision(0, Precision::Float64)?;

// Encode classical data → quantum state (returns DLPack pointer)
let dlpack_ptr = engine.encode(&data, num_qubits, "amplitude")?;

// Batch encode (fused kernel, most efficient)
let dlpack_ptr = engine.encode_batch(&batch, num_samples, sample_size, num_qubits, "amplitude")?;
```

### Methods

| Method | Description |
|--------|-------------|
| `new(device_id)` | Initialize engine on CUDA device (default f32 precision) |
| `new_with_precision(device_id, precision)` | Initialize with explicit precision |
| `encode(data, num_qubits, method)` | Encode single sample, returns DLPack pointer |
| `encode_batch(data, num_samples, sample_size, num_qubits, method)` | Batch encode with fused kernel |
| `encode_from_gpu_ptr(ptr, len, num_qubits, method, stream)` | Zero-copy encode from existing GPU memory |
| `device()` | Get CUDA device reference |

## QuantumEncoder Trait

Encoding strategy interface. All encoders implement this trait.

```rust
pub trait QuantumEncoder: Send + Sync {
    fn encode(&self, device: &Arc<CudaDevice>, data: &[f64], num_qubits: usize) -> Result<GpuStateVector>;
    fn encode_batch(&self, device: &Arc<CudaDevice>, batch_data: &[f64], num_samples: usize, sample_size: usize, num_qubits: usize) -> Result<GpuStateVector>;
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
}
```

### Encoding Implementations

| Encoder | Method Name | Description |
|---------|-------------|-------------|
| `AmplitudeEncoder` | `"amplitude"` | L2-normalized state injection. Data mapped to probability amplitudes. |
| `AngleEncoder` | `"angle"` | Rotation-based encoding. Each feature → rotation angle on a qubit. |
| `BasisEncoder` | `"basis"` | Computational basis encoding. Maps index → basis state `\|i⟩`. |
| `IqpEncoder` | `"iqp"` | Instantaneous Quantum Polynomial encoding for kernel methods. |

Use `get_encoder(name)` to get an encoder by name:

```rust
let encoder = qdp_core::gpu::get_encoder("amplitude")?;
```

## GpuStateVector

GPU-resident quantum state buffer with DLPack export.

| Method | Description |
|--------|-------------|
| `new(device, qubits, precision)` | Allocate state vector on GPU |
| `to_dlpack()` | Export as DLPack pointer for zero-copy PyTorch/JAX integration |
| `to_precision(device, target)` | Convert between f32/f64 precision |
| `num_qubits` | Number of qubits |
| `size_elements` | Number of complex elements (2^qubits) |

## Precision

```rust
pub enum Precision {
    Float32,  // Complex64 (2×f32 per element)
    Float64,  // Complex128 (2×f64 per element)
}
```

## Error Handling

```rust
pub enum MahoutError {
    Cuda(String),             // CUDA runtime errors
    InvalidInput(String),     // Bad input data or parameters
    MemoryAllocation(String), // GPU memory allocation failures
    KernelLaunch(String),     // CUDA kernel launch errors
    DLPack(String),           // DLPack conversion errors
    Io(String),               // File I/O errors
    NotImplemented(String),   // Feature not available
}

pub type Result<T> = std::result::Result<T, MahoutError>;
```

## Modules

| Module | Description |
|--------|-------------|
| `gpu` | GPU memory management, encoding implementations, pipeline infrastructure |
| `gpu::encodings` | Quantum encoder implementations (amplitude, angle, basis, IQP) |
| `gpu::memory` | GPU state vector allocation and DLPack export |
| `gpu::pipeline` | Dual-stream encoding pipeline (Linux/CUDA only) |
| `io` | I/O utilities — Parquet, Arrow IPC, NumPy, TensorFlow, Torch readers |
| `reader` | Generic `DataReader` and `StreamingDataReader` traits |
| `readers` | Format-specific reader implementations |
| `preprocessing` | Input data validation and normalization |
| `dlpack` | DLPack FFI types for zero-copy tensor interop |
| `tf_proto` | TensorFlow TensorProto protobuf definitions |

## Data Readers

| Reader | Format | Streaming |
|--------|--------|-----------|
| `ParquetReader` | Apache Parquet (`List<Float64>`, `FixedSizeList<Float64>`) | `ParquetStreamingReader` |
| `ArrowIPCReader` | Arrow IPC (Feather v2) | — |
| `NumpyReader` | NumPy `.npy` | — |
| `TorchReader` | PyTorch `.pt` | — |
| `TensorFlowReader` | TensorFlow TFRecord | — |

---

*Auto-generated from `qdp-core` source using `cargo doc`. See the [full rustdoc output](/api/rust/qdp_core/index.html) for complete details.*
