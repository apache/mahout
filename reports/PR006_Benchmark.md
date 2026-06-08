# PR006: Tensor Core Acceleration for IQP Encoding

## Overview
This PR implements **Tensor Core Acceleration** for IQP (Instantaneous Quantum Polynomial) state encoding using a Matrix-Free Kronecker Product Decomposition. It leverages the mixed-precision Adaptive Ozaki Implicit Hadamard Engine to perform Fast Walsh-Hadamard Transforms (FWT) via Tensor Core GEMM operations.

## Key Optimizations
1.  **Operator Fusion (N <= 12):** Fuses phase computation, FWT, and normalization into a single shared-memory kernel, avoiding DRAM roundtrips.
2.  **Kronecker Product Decomposition (N > 12):** Transforms the FWT into a sequence of GEMM operations compatible with Tensor Cores.
3.  **Adaptive Ozaki Engine:** Uses Tensor Core INT8/S32 MMA instructions to perform high-precision double-summation for the FWT matrix-vector products.
4.  **Bank-Conflict-Free Batch Transpose:** Efficiently reorders data layouts for GEMM compatibility.

## Benchmark Results
**Environment:**
- **GPU:** NVIDIA GeForce RTX 4060 (8GB VRAM)
- **CUDA:** 13.0
- **Batch Size:** 1024 samples
- **Precision:** Float64

| Qubits (N) | FWT (Baseline) | Tensor Core Path | Speedup | Note |
| :--- | :--- | :--- | :--- | :--- |
| 8 | 0.663 ms | 0.550 ms | 1.20x | Fused Shared Memory |
| 10 | 2.432 ms | 1.891 ms | 1.29x | Fused Shared Memory |
| 12 | 13.954 ms | 8.297 ms | 1.68x | Fused Shared Memory |
| 14 | 68.750 ms | 206.926 ms | 0.33x | TC-GEMM (Decomposed) |
| 16 | 353.473 ms | 1121.369 ms | 0.32x | TC-GEMM (Decomposed) |

### Analysis
- **Fused Path (N <= 12):** Significant performance gain (up to 1.68x) by eliminating global memory accesses.
- **Tensor Core Path (N > 12):** Currently exhibits a slowdown compared to the optimized FWT baseline. This is primarily due to:
    - **Memory Allocation Overhead:** Synchronous `cudaMalloc`/`cudaFree` calls for temporary buffers inside the encoding loop.
    - **Transpose Overhead:** Multiple batch transpositions required by the Kronecker decomposition.
    - **Summation Engine Complexity:** The Adaptive Ozaki engine for double precision requires 7x prime-field accumulation stages, which adds overhead for relatively small GEMM sizes (e.g., 128x128 for N=14).

## Conclusion
PR6 establishes the foundational architecture for Tensor Core acceleration in QDP. While the fused path provides immediate gains, the decomposed GEMM path requires further optimization (e.g., buffer reuse, asynchronous allocation, and kernel tuning) to outperform the highly optimized FWT baseline on modern GPUs.

## Code Cleanup
- Removed agent-specific prefix comments.
- Fixed `ldmatrix` 16-byte alignment bugs in `ImplicitHadamardOzaki.cu`.
- Optimized `build.rs` to target modern CUDA architectures (SM 80+).
- Exposed `encode_batch_tc` API to Python.
