---
layout: page
title: Core Concepts - QDP
---

# Core Concepts

This page explains the core concepts behind QDP (Quantum Data Plane): the architecture, the supported encoding methods, GPU memory management, DLPack zero-copy integration, and key performance characteristics.

---

## 1. Architecture overview

At a high level, QDP is organized as layered components:

- **Python API (Qumat)**: `qumat.qdp` exposes a friendly Python entry point (`QdpEngine`).
- **Python native extension (PyO3)**: `qdp/qdp-python` builds the `_qdp` module, bridging Python ↔ Rust and implementing Python-side DLPack (`__dlpack__`).
- **Rust core**: `qdp/qdp-core` contains the engine (`QdpEngine`), encoder implementations, IO readers, GPU pipelines, and DLPack export.
- **CUDA kernels**: `qdp/qdp-kernels` provides the CUDA kernels invoked by the Rust encoders.

Data flow (conceptual):

```
Python (qumat.qdp)  →  _qdp (PyO3)  →  qdp-core (Rust)  →  qdp-kernels (CUDA)
        │                        │              │                 │
        └──── torch.from_dlpack(qtensor) ◄──────┴── DLPack DLManagedTensor (GPU ptr)
```

---

## 2. What QDP produces: a GPU-resident state vector

QDP encodes classical data into a **state vector** $\vert\psi\rangle$ for $n$ qubits.

- **State length**: $2^{n}$
- **Type**: complex numbers (on GPU)
- **Shape exposed via DLPack**:
  - Single sample: `[1, 2^n]` (always 2D)
  - Batch: `[batch_size, 2^n]`

QDP supports two output precisions:

- **complex64** (2×float32) when output precision is `float32`
- **complex128** (2×float64) when output precision is `float64`

---

## 3. Encoder selection and validation

QDP uses a strategy pattern for encoding methods:

- `encoding_method` is a string, e.g. `"amplitude"`, `"basis"`, `"angle"`
- QDP maps it to a concrete encoder at runtime

All encoders perform input validation (at minimum):

- $1 \le n \le 30$
- input is not empty
- for vector-based encodings: `len(data) <= 2^n`

Note: $n=30$ is already very large—just the output state for a single sample is on the order of $2^{30}$ complex numbers.

---

## 4. Encoding methods (amplitude, basis, angle)

### 4.1 Amplitude encoding

**Goal**: represent a real-valued feature vector $x$ as quantum amplitudes:

$$
\vert\psi\rangle = \sum_{i=0}^{2^{n}-1} \frac{x_i}{\|x\|_2}\,\vert i\rangle
$$

Key properties in QDP:

- **Normalization**: QDP computes $\|x\|_2$ and rejects zero-norm inputs.
- **Padding**: if `len(x) < 2^n`, the remaining amplitudes are treated as zeros.
- **GPU execution**: the normalization and write into the GPU state vector is performed by CUDA kernels.
- **Batch support**: amplitude encoding supports a batch path to reduce kernel launch / allocation overhead (recommended when encoding many samples).

When to use it:

- You have dense real-valued vectors and want a direct amplitude representation.

Trade-offs:

- Output size grows exponentially with `num_qubits` ($2^n$), so it can become memory-heavy quickly.

### 4.2 Basis encoding

**Goal**: map an integer index $i$ into a computational basis state $\vert i\rangle$.

For $n$ qubits with $0 \le i < 2^n$:

- $\psi[i] = 1$
- $\psi[j] = 0$ for all $j \ne i$

Key properties in QDP:

- **Input shape**:
  - single sample expects exactly one value: `[index]`
  - batch expects one index per sample (effectively `sample_size = 1`)
- **Range checking**: indices must be valid for the chosen `num_qubits`
- **GPU execution**: kernel sets the one-hot amplitude at the requested index

When to use it:

- Your data naturally represents discrete states (categories, token IDs, hashed features, etc.).

### 4.3 Angle encoding (planned)

Angle encoding typically maps features to rotation angles (e.g., via $R_x(\theta)$, $R_y(\theta)$, $R_z(\theta)$) and constructs a state by applying rotations across qubits.

**Current status in this codebase**:

- The `"angle"` encoder exists as a placeholder and **returns an error stating it is not implemented yet**.
- Use `"amplitude"` or `"basis"` for now.

---

## 5. GPU memory management

QDP is designed to keep the encoded states on the GPU and to avoid unnecessary allocations/copies where possible.

### 5.1 Output state vector allocation

For each encoded sample, QDP allocates a state vector of size $2^n$. Memory usage grows exponentially:

- complex128 uses 16 bytes per element
- rough output size (single sample) is:
  - $2^n \times 16$ bytes for complex128
  - $2^n \times 8$ bytes for complex64

QDP performs **pre-flight checks** before large allocations to fail fast with an OOM-aware message (e.g., suggesting smaller `num_qubits` or batch size).

### 5.2 Pinned host memory and streaming pipelines

For high-throughput IO → GPU workflows (especially streaming from Parquet), QDP uses:

- **Pinned host buffers** (page-locked memory) to speed up host↔device transfers.
- **Double buffering** (ping-pong) so one buffer can be filled while another is being processed.
- **Device staging buffers** (for streaming) so that copies and compute can overlap.

Streaming Parquet encoding is implemented as a **producer/consumer pipeline**:

- a background IO thread reads chunks into pinned host buffers
- the GPU side processes each chunk while IO continues

In the current implementation, streaming Parquet supports:

- `"amplitude"`
- `"basis"`

(`"angle"` is not supported for streaming yet.)

### 5.3 Asynchronous copy/compute overlap (dual streams)

For large workloads, QDP uses multiple CUDA streams and CUDA events to overlap:

- **H2D copies** (copy stream)
- **kernel execution** (compute stream)

This reduces time spent waiting on PCIe transfers and can improve throughput substantially for large batches.

---

## 6. DLPack zero-copy integration

QDP exposes results using the **DLPack protocol**, which allows frameworks like PyTorch to consume GPU memory **without copying**.

Conceptually:

1. Rust allocates GPU memory for the state vector.
2. Rust wraps it into a DLPack `DLManagedTensor`.
3. Python returns an object that implements `__dlpack__`.
4. PyTorch calls `torch.from_dlpack(qtensor)` and takes ownership via DLPack's deleter.

Important details:

- The returned DLPack capsule is **single-consume** (can only be used once). This prevents double-free bugs.
- Memory lifetime is managed safely via reference counting on the Rust side, and freed by the DLPack deleter when the consumer releases it.

---

## 7. Performance characteristics and practical guidance

### 7.1 What makes QDP fast

- GPU kernels replace circuit-based state preparation for the supported encodings.
- Batch APIs reduce allocation and kernel launch overhead.
- Streaming pipelines overlap IO and GPU compute.

### 7.2 Choosing parameters wisely

- **Prefer batch encoding** when encoding many samples (lower overhead, better GPU utilization).
- **Keep `num_qubits` realistic**. Output size is $2^n$ and becomes the dominant cost quickly.
- **Pick the right encoding**:
  - amplitude: dense real-valued vectors
  - basis: discrete indices / categorical states
  - angle: planned, not implemented yet in this version

### 7.3 Profiling

If you need to understand where time is spent (copy vs compute), QDP supports NVTX-based profiling. See `qdp/docs/observability/NVTX_USAGE.md`.
