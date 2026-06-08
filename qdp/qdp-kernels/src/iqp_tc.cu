// iqp_tc.cu
#include <cuda_runtime.h>
#include <cuComplex.h>
#include "kernel_config.h"

// Phase computation (from iqp.cu)
__device__ double compute_phase_tc(
    const double* __restrict__ data,
    size_t x,
    unsigned int num_qubits,
    int enable_zz
) {
    double phase = 0.0;
    for (unsigned int i = 0; i < num_qubits; ++i) {
        phase += data[i] * (double)((x >> i) & 1U);
    }
    if (enable_zz) {
        unsigned int pair_idx = num_qubits;
        for (unsigned int i = 0; i < num_qubits; ++i) {
            for (unsigned int j = i + 1; j < num_qubits; ++j) {
                phase += data[pair_idx] * (double)(((x >> i) & 1U) & ((x >> j) & 1U));
                pair_idx++;
            }
        }
    }
    return phase;
}

// Pre-GEMM setup - Unroll Batch and compute initial Phase (split into pure real/imaginary parts)
// This prepares the data layout for the Kronecker product decomposition in upcoming PRs.
__global__ void iqp_phase_split_kernel(
    const double* __restrict__ data_batch,
    double* __restrict__ state_real,
    double* __restrict__ state_imag,
    size_t num_samples,
    size_t state_len,
    unsigned int num_qubits,
    unsigned int data_len,
    int enable_zz
) {
    const size_t total_elements = num_samples * state_len;
    const size_t stride = gridDim.x * blockDim.x;
    const size_t state_mask = state_len - 1;

    for (size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
         global_idx < total_elements;
         global_idx += stride) {
        const size_t sample_idx = global_idx >> num_qubits;
        const size_t x = global_idx & state_mask;
        const double* data = data_batch + sample_idx * data_len;

        double phase = compute_phase_tc(data, x, num_qubits, enable_zz);
        double cos_phase, sin_phase;
        sincos(phase, &sin_phase, &cos_phase);

        state_real[global_idx] = cos_phase;
        state_imag[global_idx] = sin_phase;
    }
}

#define TRANSPOSE_TILE_DIM 32
#define TRANSPOSE_BLOCK_ROWS 8

// Shared Memory Bank-Conflict-Free Batch Transpose
// Essential for reordering the data efficiently before/after Tensor Core FWT matrix multiplications.
__global__ void iqp_tc_batch_transpose_kernel(const double* __restrict__ in, double* __restrict__ out, int B, int rows, int cols) {
    // TILE_DIM x (TILE_DIM+1) pad to avoid shared memory bank conflicts
    __shared__ double tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM + 1];

    int b = blockIdx.z;
    int x = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.y;

    // Load from global memory (coalesced) into shared memory
    for (int j = 0; j < TRANSPOSE_TILE_DIM; j += TRANSPOSE_BLOCK_ROWS) {
        if (x < cols && (y + j) < rows) {
            tile[threadIdx.y + j][threadIdx.x] = in[b * rows * cols + (y + j) * cols + x];
        }
    }

    __syncthreads();

    // Transposed block coordinates
    x = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.x;
    y = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.y;

    // Store from shared memory to global memory (coalesced)
    for (int j = 0; j < TRANSPOSE_TILE_DIM; j += TRANSPOSE_BLOCK_ROWS) {
        if (x < rows && (y + j) < cols) {
            out[b * rows * cols + (y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

void iqp_tc_launch_transpose(const double* d_in, double* d_out, int B, int rows, int cols, cudaStream_t stream) {
    dim3 block(TRANSPOSE_TILE_DIM, TRANSPOSE_BLOCK_ROWS, 1);
    dim3 grid((cols + TRANSPOSE_TILE_DIM - 1) / TRANSPOSE_TILE_DIM,
              (rows + TRANSPOSE_TILE_DIM - 1) / TRANSPOSE_TILE_DIM, B);
    iqp_tc_batch_transpose_kernel<<<grid, block, 0, stream>>>(d_in, d_out, B, rows, cols);
}

// Recombine Real and Imaginary parts back into cuDoubleComplex
// This restores the memory layout after Tensor Core matrix multiplications.
__global__ void recombine_complex_kernel(
    const double* __restrict__ real_part,
    const double* __restrict__ imag_part,
    cuDoubleComplex* __restrict__ out,
    size_t total_elements
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        out[idx] = make_cuDoubleComplex(real_part[idx], imag_part[idx]);
    }
}

extern "C" int launch_iqp_encode_tc(
    const double* data_batch_d,
    void*         state_batch_d,
    size_t        num_samples,
    size_t        state_len,
    unsigned int  num_qubits,
    int           enable_zz,
    cudaStream_t  stream
) {
    // Scaffold for batch layout manipulation
    size_t total_elements = num_samples * state_len;
    
    double *d_state_real, *d_state_imag;
    cudaMalloc(&d_state_real, total_elements * sizeof(double));
    cudaMalloc(&d_state_imag, total_elements * sizeof(double));
    
    unsigned int data_len = num_qubits;
    const size_t blocks = (total_elements + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE;
    
    iqp_phase_split_kernel<<<blocks, DEFAULT_BLOCK_SIZE, 0, stream>>>(
        data_batch_d, d_state_real, d_state_imag, num_samples, state_len, num_qubits, data_len, enable_zz
    );
    
    // In future PRs, Kronecker Transpose and FWT will happen here.
    
    recombine_complex_kernel<<<blocks, DEFAULT_BLOCK_SIZE, 0, stream>>>(
        d_state_real, d_state_imag, static_cast<cuDoubleComplex*>(state_batch_d), total_elements
    );
    
    cudaFree(d_state_real);
    cudaFree(d_state_imag);
    
    return (int)cudaSuccess;
}
