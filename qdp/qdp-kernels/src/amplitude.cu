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

// Amplitude Encoding CUDA Kernel

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <vector_types.h>

__global__ void amplitude_encode_kernel(
    const double* __restrict__ input,
    cuDoubleComplex* __restrict__ state,
    size_t input_len,
    size_t state_len,
    double inv_norm
) {
    // We process 2 elements per thread to maximize memory bandwidth via double2
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread handles two state amplitudes (indices 2*idx and 2*idx + 1)
    size_t state_idx_base = idx * 2;

    if (state_idx_base >= state_len) return;

    double v1 = 0.0;
    double v2 = 0.0;

    // Vectorized Load Optimization:
    // If we are well within bounds, treat input as double2 to issue a single 128-bit load instruction.
    // This reduces memory transactions and improves throughput on RTX cards.
    if (state_idx_base + 1 < input_len) {
        // Reinterpret cast to load two doubles at once
        // Note: Assumes input is reasonably aligned (standard cudaMalloc provides 256-byte alignment)
        const double2* input_vec = reinterpret_cast<const double2*>(input);
        double2 loaded = input_vec[idx];
        v1 = loaded.x;
        v2 = loaded.y;
    }
    // Handle edge case: Odd input length
    else if (state_idx_base < input_len) {
        v1 = input[state_idx_base];
        // v2 remains 0.0
    }

    // Write output:
    // Apply pre-calculated reciprocal (multiplication is faster than division)
    state[state_idx_base]     = make_cuDoubleComplex(v1 * inv_norm, 0.0);

    // Check boundary for the second element (state_len is usually power of 2, but good to be safe)
    if (state_idx_base + 1 < state_len) {
        state[state_idx_base + 1] = make_cuDoubleComplex(v2 * inv_norm, 0.0);
    }
}

extern "C" {

/// Launch amplitude encoding kernel
///
/// # Arguments
/// * input_d - Device pointer to input data (already normalized by host)
/// * state_d - Device pointer to output state vector
/// * input_len - Number of input elements
/// * state_len - Target state vector size (2^num_qubits)
/// * norm - L2 norm computed by host
/// * stream - CUDA stream for async execution (nullptr = default stream)
///
/// # Returns
/// CUDA error code (0 = cudaSuccess)
int launch_amplitude_encode(
    const double* input_d,
    void* state_d,
    size_t input_len,
    size_t state_len,
    double norm,
    cudaStream_t stream
) {
    if (norm <= 0.0) {
        return cudaErrorInvalidValue;
    }

    double inv_norm = 1.0 / norm;

    cuDoubleComplex* state_complex_d = static_cast<cuDoubleComplex*>(state_d);

    const int blockSize = 256;
    // Halve the grid size because each thread now processes 2 elements
    const int gridSize = (state_len / 2 + blockSize - 1) / blockSize;

    amplitude_encode_kernel<<<gridSize, blockSize, 0, stream>>>(
        input_d,
        state_complex_d,
        input_len,
        state_len,
        inv_norm // Pass reciprocal
    );

    return (int)cudaGetLastError();
}

/// Compute inverse L2 norms for batch using tree reduction
/// Output: inv_norms[sample_idx] = 1.0 / sqrt(sum(x^2))
__global__ void compute_l2_norms_batch_kernel(
    const double* __restrict__ input_batch,
    double* __restrict__ norms_out,
    size_t num_samples,
    size_t input_len
) {
    extern __shared__ double sdata[];

    const size_t sample_idx = blockIdx.x;
    if (sample_idx >= num_samples) return;

    const size_t input_base = sample_idx * input_len;
    const size_t tid = threadIdx.x;

    // Compute sum of squares (grid-stride loop)
    double sum_sq = 0.0;
    for (size_t i = tid; i < input_len; i += blockDim.x) {
        const double val = __ldg(input_batch + input_base + i);
        sum_sq += val * val;
    }

    // Tree reduction in shared memory
    sdata[tid] = sum_sq;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write inverse norm (avoid division in encoding kernel)
    if (tid == 0) {
        const double norm = sqrt(sdata[0]);
        norms_out[sample_idx] = (norm > 0.0) ? (1.0 / norm) : 0.0;
    }
}

/// Launch kernel to compute inverse L2 norms for batch
int launch_compute_l2_norms_batch(
    const double* input_batch_d,
    double* norms_out_d,
    size_t num_samples,
    size_t input_len,
    cudaStream_t stream
) {
    if (num_samples == 0 || input_len == 0) {
        return cudaErrorInvalidValue;
    }

    const int blockSize = 256;
    const size_t sharedMemSize = blockSize * sizeof(double);
    const size_t gridSize = num_samples;  // One block per sample

    compute_l2_norms_batch_kernel<<<gridSize, blockSize, sharedMemSize, stream>>>(
        input_batch_d,
        norms_out_d,
        num_samples,
        input_len
    );

    return (int)cudaGetLastError();
}

/// Optimized batch amplitude encoding kernel
///
/// Memory Layout (row-major):
/// - input_batch: [sample0_data | sample1_data | ... | sampleN_data]
/// - state_batch: [sample0_state | sample1_state | ... | sampleN_state]
///
/// Optimizations:
/// 1. Vectorized double2 loads for 128-bit memory transactions
/// 2. Grid-stride loop for arbitrary batch sizes
/// 3. Coalesced memory access within warps
/// 4. Minimized register pressure
__global__ void amplitude_encode_batch_kernel(
    const double* __restrict__ input_batch,
    cuDoubleComplex* __restrict__ state_batch,
    const double* __restrict__ inv_norms,
    size_t num_samples,
    size_t input_len,
    size_t state_len
) {
    // Grid-stride loop pattern for flexibility
    const size_t elements_per_sample = state_len / 2;  // Each thread handles 2 elements
    const size_t total_work = num_samples * elements_per_sample;
    const size_t stride = gridDim.x * blockDim.x;

    size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process elements in grid-stride fashion
    for (size_t idx = global_idx; idx < total_work; idx += stride) {
        // Decompose linear index into (sample, element_pair)
        const size_t sample_idx = idx / elements_per_sample;
        const size_t elem_pair = idx % elements_per_sample;

        // Calculate base addresses (strength-reduced)
        const size_t input_base = sample_idx * input_len;
        const size_t state_base = sample_idx * state_len;
        const size_t elem_offset = elem_pair * 2;

        // Load inverse norm (cached by L1)
        const double inv_norm = inv_norms[sample_idx];

        // Vectorized load: read 2 doubles as double2 for 128-bit transaction
        double v1, v2;
        if (elem_offset + 1 < input_len) {
            // Aligned vectorized load
            const double2 vec_data = __ldg(reinterpret_cast<const double2*>(input_batch + input_base) + elem_pair);
            v1 = vec_data.x;
            v2 = vec_data.y;
        } else if (elem_offset < input_len) {
            // Edge case: single element load
            v1 = __ldg(input_batch + input_base + elem_offset);
            v2 = 0.0;
        } else {
            // Padding region
            v1 = v2 = 0.0;
        }

        // Normalize and write as complex numbers
        // Compiler will optimize multiplications
        const cuDoubleComplex c1 = make_cuDoubleComplex(v1 * inv_norm, 0.0);
        const cuDoubleComplex c2 = make_cuDoubleComplex(v2 * inv_norm, 0.0);

        // Write to global memory (coalesced within warp)
        state_batch[state_base + elem_offset] = c1;
        if (elem_offset + 1 < state_len) {
            state_batch[state_base + elem_offset + 1] = c2;
        }
    }
}

/// Launch optimized batch amplitude encoding kernel
///
/// # Arguments
/// * input_batch_d - Device pointer to batch input data
/// * state_batch_d - Device pointer to output batch state vectors
/// * inv_norms_d - Device pointer to inverse norms array
/// * num_samples - Number of samples in batch
/// * input_len - Elements per sample
/// * state_len - State vector size per sample (2^num_qubits)
/// * stream - CUDA stream for async execution
///
/// # Returns
/// CUDA error code (0 = cudaSuccess)
int launch_amplitude_encode_batch(
    const double* input_batch_d,
    void* state_batch_d,
    const double* inv_norms_d,
    size_t num_samples,
    size_t input_len,
    size_t state_len,
    cudaStream_t stream
) {
    if (num_samples == 0 || state_len == 0) {
        return cudaErrorInvalidValue;
    }

    cuDoubleComplex* state_complex_d = static_cast<cuDoubleComplex*>(state_batch_d);

    // Optimal configuration for modern GPUs (SM 7.0+)
    // - Block size: 256 threads (8 warps, good occupancy)
    // - Grid size: Enough blocks to saturate GPU, but not excessive
    const int blockSize = 256;
    const size_t total_work = num_samples * (state_len / 2);

    // Calculate grid size: aim for high occupancy without too many blocks
    // Limit to reasonable number of blocks to avoid scheduler overhead
    const size_t blocks_needed = (total_work + blockSize - 1) / blockSize;
    const size_t max_blocks = 2048;  // Reasonable limit for most GPUs
    const size_t gridSize = (blocks_needed < max_blocks) ? blocks_needed : max_blocks;

    amplitude_encode_batch_kernel<<<gridSize, blockSize, 0, stream>>>(
        input_batch_d,
        state_complex_d,
        inv_norms_d,
        num_samples,
        input_len,
        state_len
    );

    return (int)cudaGetLastError();
}

/// Fused kernel: compute norm and encode in one pass
/// Each block processes one sample: compute norm via reduction, then normalize and write
__global__ void fused_amplitude_encode_batch_kernel(
    const double* __restrict__ input_batch,
    cuDoubleComplex* __restrict__ state_batch,
    size_t num_samples,
    size_t input_len,
    size_t state_len
) {
    extern __shared__ double sdata[];

    const size_t sample_idx = blockIdx.x;
    if (sample_idx >= num_samples) return;

    const size_t input_base = sample_idx * input_len;
    const size_t state_base = sample_idx * state_len;
    const size_t tid = threadIdx.x;

    // Step 1: Compute sum of squares (grid-stride loop)
    double sum_sq = 0.0;
    for (size_t i = tid; i < input_len; i += blockDim.x) {
        const double val = __ldg(input_batch + input_base + i);
        sum_sq += val * val;
    }

    // Step 2: Tree reduction
    sdata[tid] = sum_sq;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Step 3: Compute inverse norm and broadcast
    double inv_norm = 0.0;
    if (tid == 0) {
        const double norm = sqrt(sdata[0]);
        inv_norm = (norm > 1e-9) ? (1.0 / norm) : 0.0;
        sdata[0] = inv_norm;
    }
    __syncthreads();
    inv_norm = sdata[0];

    // Step 4: Normalize and write (standard double reads, GPU coalescing handles efficiency)
    for (size_t i = tid; i < state_len; i += blockDim.x) {
        double val = (i < input_len) ? __ldg(input_batch + input_base + i) : 0.0;
        state_batch[state_base + i] = make_cuDoubleComplex(val * inv_norm, 0.0);
    }
}

/// Launch fused amplitude encoding kernel (norm computation + encoding in one pass)
int launch_fused_amplitude_encode_batch(
    const double* input_batch_d,
    void* state_batch_d,
    size_t num_samples,
    size_t input_len,
    size_t state_len,
    cudaStream_t stream
) {
    if (num_samples == 0 || state_len == 0) {
        return cudaErrorInvalidValue;
    }

    cuDoubleComplex* state_complex_d = static_cast<cuDoubleComplex*>(state_batch_d);

    const int blockSize = 256;
    const size_t sharedMemSize = blockSize * sizeof(double);
    const size_t gridSize = num_samples;  // One block per sample

    fused_amplitude_encode_batch_kernel<<<gridSize, blockSize, sharedMemSize, stream>>>(
        input_batch_d,
        state_complex_d,
        num_samples,
        input_len,
        state_len
    );

    return (int)cudaGetLastError();
}

// TODO: Future encoding methods:
// - launch_angle_encode (angle encoding)
// - launch_basis_encode (basis encoding)
// - launch_iqp_encode (IQP encoding)

} // extern "C"
