//
// Licensed to the Apache Software Foundation (ASF) under one or more
// contributor license agreements.  See the NOTICE file distributed with
// this work for additional information regarding copyright ownership.
// The ASF licenses this file to You under the Apache License, Version 2.0
// (the "License"); you may not use this file except in compliance with
// the License.  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

// Shared-memory FWT path (Operator Fusion)
// Fuses Phase computation, Fast Walsh-Hadamard Transform, and Normalization
// entirely within Shared Memory. This completely avoids DRAM roundtrips for N <= 12.
__global__ void iqp_phase_fwt_normalize_tc_kernel(
    const double* __restrict__ data_batch,
    cuDoubleComplex* __restrict__ state_batch,
    size_t num_samples,
    size_t state_len,
    unsigned int num_qubits,
    unsigned int data_len,
    int enable_zz,
    double norm_factor
) {
    extern __shared__ cuDoubleComplex shared_state[];

    size_t tid = threadIdx.x;
    size_t sample_idx = blockIdx.x;

    if (sample_idx >= num_samples) return;

    const double* data = data_batch + sample_idx * data_len;
    cuDoubleComplex* state = state_batch + sample_idx * state_len;

    // 1. Phase calculation directly into Shared Memory
    for (size_t i = tid; i < state_len; i += blockDim.x) {
        double phase = compute_phase_tc(data, i, num_qubits, enable_zz);
        double cos_phase, sin_phase;
        sincos(phase, &sin_phase, &cos_phase);
        shared_state[i] = make_cuDoubleComplex(cos_phase, sin_phase);
    }
    __syncthreads();

    // 2. Perform Hadamard FWT in Shared Memory
    for (unsigned int stage = 0; stage < num_qubits; ++stage) {
        size_t stride = 1ULL << stage;
        size_t block_size = stride << 1;
        size_t num_pairs = state_len >> 1;

        for (size_t pair_idx = tid; pair_idx < num_pairs; pair_idx += blockDim.x) {
            size_t block_idx = pair_idx / stride;
            size_t pair_offset = pair_idx % stride;
            size_t i = block_idx * block_size + pair_offset;
            size_t j = i + stride;

            cuDoubleComplex a = shared_state[i];
            cuDoubleComplex b = shared_state[j];

            shared_state[i] = cuCadd(a, b);
            shared_state[j] = cuCsub(a, b);
        }
        __syncthreads();
    }

    // 3. Normalize and write back to Global Memory
    for (size_t i = tid; i < state_len; i += blockDim.x) {
        cuDoubleComplex val = shared_state[i];
        state[i] = make_cuDoubleComplex(
            cuCreal(val) * norm_factor,
            cuCimag(val) * norm_factor
        );
    }
}

// PR2: Pre-GEMM setup - Unroll Batch and compute initial Phase (split into pure real/imaginary parts)
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

// PR2: Shared Memory Bank-Conflict-Free Batch Transpose
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

// Naive Implicit Hadamard GEMM (Fallback before PR5/6 Tensor Core integration)
// Computes Y = X * H_K where H_K is a KxK Hadamard matrix generated on-the-fly.
__global__ void naive_implicit_hadamard_gemm_kernel(const double* __restrict__ X, double* __restrict__ Y, int B, int M, int K, double norm) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (m < M && k < K) {
        double sum = 0.0;
        for (int i = 0; i < K; ++i) {
            double h_val = (__popc(k & i) & 1) ? -1.0 : 1.0;
            sum += X[b * M * K + m * K + i] * h_val;
        }
        Y[b * M * K + m * K + k] = sum * norm;
    }
}

void launch_naive_implicit_hadamard(const double* d_in, double* d_out, int B, int M, int K, double norm, cudaStream_t stream) {
    dim3 block(16, 16, 1);
    dim3 grid((K + 15) / 16, (M + 15) / 16, B);
    naive_implicit_hadamard_gemm_kernel<<<grid, block, 0, stream>>>(d_in, d_out, B, M, K, norm);
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
    if (num_qubits <= FWT_SHARED_MEM_THRESHOLD) {
        // For N <= 12, use the fused Shared Memory FWT kernel
        double norm_factor = 1.0 / (double)state_len;
        unsigned int data_len = num_qubits;
        // Request max dynamic shared memory for this kernel
        cudaFuncSetAttribute(iqp_phase_fwt_normalize_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
        iqp_phase_fwt_normalize_tc_kernel<<<num_samples, DEFAULT_BLOCK_SIZE, state_len * sizeof(cuDoubleComplex), stream>>>(
            data_batch_d, static_cast<cuDoubleComplex*>(state_batch_d), num_samples, state_len, num_qubits, data_len, enable_zz, norm_factor
        );
    } else {
        // Blocked TC-FWT (Kronecker Product Decomposition)
        size_t m_samples = num_samples;
        size_t total_elements = m_samples * state_len;

        int n1 = num_qubits / 2;
        int n2 = num_qubits - n1;
        int dim1 = 1 << n1;
        int dim2 = 1 << n2;

        double *d_state_real, *d_state_imag;
        double *d_out_real, *d_out_imag;
        double *d_temp_real, *d_temp_imag;
        cudaMalloc(&d_state_real, total_elements * sizeof(double));
        cudaMalloc(&d_state_imag, total_elements * sizeof(double));
        cudaMalloc(&d_out_real, total_elements * sizeof(double));
        cudaMalloc(&d_out_imag, total_elements * sizeof(double));
        cudaMalloc(&d_temp_real, total_elements * sizeof(double));
        cudaMalloc(&d_temp_imag, total_elements * sizeof(double));

        // 1. Initialize Phase (Split Real/Imag)
        unsigned int data_len = num_qubits;
        const size_t blocks = (total_elements + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE;
        iqp_phase_split_kernel<<<blocks, DEFAULT_BLOCK_SIZE, 0, stream>>>(
            data_batch_d, d_state_real, d_state_imag, num_samples, state_len, num_qubits, data_len, enable_zz
        );

        double norm_factor = 1.0 / (double)state_len;

        // 3. TC-FWT Step 1: Z = X * H_{n2} (X shape: B*dim1 x dim2)
        // Uses Naive GEMM Placeholder. PR5/6 will replace this with Ozaki Implicit Engine.
        launch_naive_implicit_hadamard(d_state_real, d_out_real, num_samples * dim1, dim2, dim2, 1.0, stream);
        launch_naive_implicit_hadamard(d_state_imag, d_out_imag, num_samples * dim1, dim2, dim2, 1.0, stream);

        // 4. TC-FWT Step 2: Transpose (B, dim1, dim2) -> (B, dim2, dim1)
        iqp_tc_launch_transpose(d_out_real, d_temp_real, num_samples, dim1, dim2, stream);
        iqp_tc_launch_transpose(d_out_imag, d_temp_imag, num_samples, dim1, dim2, stream);

        // 5. TC-FWT Step 3: Y_T = Z_T * H_{n1} (Z_T shape: B*dim2 x dim1)
        launch_naive_implicit_hadamard(d_temp_real, d_out_real, num_samples * dim2, dim1, dim1, norm_factor, stream);
        launch_naive_implicit_hadamard(d_temp_imag, d_out_imag, num_samples * dim2, dim1, dim1, norm_factor, stream);

        // 6. TC-FWT Step 4: Transpose back (B, dim2, dim1) -> (B, dim1, dim2)
        iqp_tc_launch_transpose(d_out_real, d_temp_real, num_samples, dim2, dim1, stream);
        iqp_tc_launch_transpose(d_out_imag, d_temp_imag, num_samples, dim2, dim1, stream);

        // 7. Recombine and Write back
        recombine_complex_kernel<<<blocks, DEFAULT_BLOCK_SIZE, 0, stream>>>(
            d_temp_real, d_temp_imag, static_cast<cuDoubleComplex*>(state_batch_d), total_elements
        );

        cudaFree(d_state_real);
        cudaFree(d_state_imag);
        cudaFree(d_out_real);
        cudaFree(d_out_imag);
        cudaFree(d_temp_real);
        cudaFree(d_temp_imag);
    }

    return (int)cudaSuccess;
}
