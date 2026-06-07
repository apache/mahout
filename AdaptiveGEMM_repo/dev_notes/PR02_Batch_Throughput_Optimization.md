### Related Issues

<!-- Closes #123 -->
N/A

### Changes

- [ ] Bug fix
- [ ] New feature
- [x] Refactoring
- [ ] Documentation
- [ ] Test
- [ ] CI/CD pipeline
- [ ] Other

### Why

As part of the IQP Encoding Optimization PR Split Plan, PR 2 focuses on "Batch throughput optimization" and lays the structural groundwork for Tensor Core (TC) acceleration (which will be fully introduced in PR 5 & 6). 

**Architectural Philosophy: Dual-Path Explicit Opt-in**
It is crucial to note that these new Tensor Core optimizations do *not* automatically replace or override the existing standard algorithms. We are adopting a **Dual-Path Architecture**:
1.  **Standard Path (`encode_batch`):** The original, hardware-agnostic FP64 FWT path is fully preserved. This ensures that users on older hardware (without Tensor Cores) or those requiring strict IEEE 754 standard FP64 behavior without any mixed-precision artifacts can continue running unmodified.
2.  **Tensor Core Path (`encode_batch_tc`):** This is a new, highly specialized API path introduced here. Because Tensor Cores utilize INT8 mixed-precision arithmetic (compensated via the Chinese Remainder Theorem later in PR 6), there are microscopic floating-point differences. In HPC and quantum simulation, auto-dispatching to mixed-precision can cause difficult-to-debug numerical artifacts. Therefore, the TC pipeline is strictly an **explicit opt-in** for advanced users seeking maximum throughput on supported hardware (Turing/Ampere/Hopper).

To prepare for this `encode_batch_tc` pipeline, we need a robust scaffolding for batch data transformation. The original code processed matrices sequentially; this refactoring introduces batched layouts and kernels required for the Kronecker-based matrix multiplication that Tensor Cores will eventually execute.

### How

- **Created `iqp_tc.cu`:** Introduced new kernels specifically designed to manage memory layout for batched operations.
- **Phase Split Kernel (`iqp_phase_split_kernel`):** Unrolls the batch and splits the initial phase computation into pure real and imaginary parts to prepare for INT8 matrix multiplication.
- **Batch Transpose Kernel (`iqp_tc_batch_transpose_kernel`):** Implemented a Shared Memory Bank-Conflict-Free matrix transpose kernel, essential for efficiently reordering data between Tensor Core FWT stages.
- **Recombine Kernel (`recombine_complex_kernel`):** Restores the split real and imaginary parts back into the standard `cuDoubleComplex` format expected by downstream processes.
- **Rust Integration:** Updated `lib.rs` and `iqp.rs` to expose and call the new `launch_iqp_encode_tc` function from Rust, laying the structural groundwork for the full Tensor Core pipeline.

## Checklist

- [x] Added or updated unit tests for all changes (Verified that existing tests pass, and batching logic doesn't break `qdp-core`)
- [x] Added or updated documentation for all changes (Added explicit comments describing the purpose of the new kernels)