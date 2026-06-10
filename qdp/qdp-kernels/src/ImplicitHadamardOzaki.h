#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <cstdint>

#include "AdaptiveOzaki.h"

namespace ozaki {

// ---------------------------------------------------------------------------
// ImplicitHadamardOzakiEngine
//
// PR009: Re-enabled Tensor Core INT8 MMA path.
//
// execute_implicit_hadamard dispatches based on matrix size:
//   - m >= TC_MIN_DIM && n >= TC_MIN_DIM && k >= 32:
//       TC path: precompute_modulo_kernel_p26_implicit + mma.sync.aligned.m16n8k32
//   - Otherwise:
//       Naive FP64 CUDA Core kernel (PR008 strategic fallback, always correct)
//
// The function is stream-aware and can be chained with other CUDA kernels
// in the same stream (e.g., from iqp_tc.cu's Kronecker product decomposition).
// ---------------------------------------------------------------------------
class ImplicitHadamardOzakiEngine {
public:
    explicit ImplicitHadamardOzakiEngine(const OzakiConfig& config) : config_(config) {}
    ~ImplicitHadamardOzakiEngine() = default;

    // Compute C = A × H_n × norm_factor  (H_n is the n×n Walsh-Hadamard matrix)
    // A: [m × k] FP64 row-major on device
    // C: [m × n] FP64 row-major on device (overwritten; must be pre-allocated)
    // k == n (Hadamard is square; caller ensures this)
    void execute_implicit_hadamard(
        const double* d_A,
        double*       d_C,
        int m, int n, int k,
        double norm_factor,
        cudaStream_t stream = 0,
        bool transpose_batch = false,
        int batch_rows = 0
    );

    void execute_fused_kronecker_hadamard(
        const double* d_X, double* d_Z, double* d_Y,
        int batch_size, int dim,
        double norm_factor, cudaStream_t stream = 0
    );

private:
    OzakiConfig config_;
};

} // namespace ozaki
