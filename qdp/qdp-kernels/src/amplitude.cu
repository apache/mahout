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
#include <math.h>
#include "kernel_config.h"

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
    // Use __ldg() to pull through the read-only cache; cudaMalloc aligns to 256 bytes so the
    // reinterpret_cast<double2*> load is naturally aligned.
    if (state_idx_base + 1 < input_len) {
        // Reinterpret cast to load two doubles at once
        const double2 loaded = __ldg(reinterpret_cast<const double2*>(input) + idx);
        v1 = loaded.x;
        v2 = loaded.y;
    }
    // Handle edge case: Odd input length
    else if (state_idx_base < input_len) {
        v1 = __ldg(input + state_idx_base);
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

__global__ void amplitude_encode_kernel_f32(
    const float* __restrict__ input,
    cuComplex* __restrict__ state,
    size_t input_len,
    size_t state_len,
    float inv_norm
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t state_idx_base = idx * 2;
    if (state_idx_base >= state_len) return;

    float v1 = 0.0f;
    float v2 = 0.0f;

    if (state_idx_base + 1 < input_len) {
        // Mirror the double kernel: cached vectorized load for two floats
        const float2 loaded = __ldg(reinterpret_cast<const float2*>(input) + idx);
        v1 = loaded.x;
        v2 = loaded.y;
    } else if (state_idx_base < input_len) {
        v1 = __ldg(input + state_idx_base);
    }

    state[state_idx_base] = make_cuComplex(v1 * inv_norm, 0.0f);
    if (state_idx_base + 1 < state_len) {
        state[state_idx_base + 1] = make_cuComplex(v2 * inv_norm, 0.0f);
    }
}

// Warp-level reduction for sum using shuffle instructions
__device__ __forceinline__ double warp_reduce_sum(double val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Warp-level reduction for sum using shuffle instructions (float32)
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction built on top of warp reduction
__device__ __forceinline__ double block_reduce_sum(double val) {
    __shared__ double shared[32]; // supports up to 1024 threads (32 warps)
    int lane = threadIdx.x & (warpSize - 1);
    int warp_id = threadIdx.x >> 5;

    val = warp_reduce_sum(val);
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    // Only first warp participates in final reduction
    val = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane] : 0.0;
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}


// Block-level reduction built on top of warp reduction (float32)
__device__ __forceinline__ float block_reduce_sum_f32(float val) {
    __shared__ float shared[32]; // supports up to 1024 threads (32 warps)
    int lane = threadIdx.x & (warpSize - 1);
    int warp_id = threadIdx.x >> 5;

    val = warp_reduce_sum_f32(val);
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    // Only first warp participates in final reduction
    val = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane] : 0.0f;
    if (warp_id == 0) {
        val = warp_reduce_sum_f32(val);
    }
    return val;
}

extern "C" {

/// Launch amplitude encoding kernel
///
/// # Arguments
/// * input_d - Device pointer to input data (already normalized by host)
/// * state_d - Device pointer to output state vector
/// * input_len - Number of input elements
/// * state_len - Target state vector size (2^num_qubits)
/// * inv_norm - Reciprocal L2 norm (1 / ||input||)
/// * stream - CUDA stream for async execution (nullptr = default stream)
///
/// # Returns
/// CUDA error code (0 = cudaSuccess)
int launch_amplitude_encode(
    const double* input_d,
    void* state_d,
    size_t input_len,
    size_t state_len,
    double inv_norm,
    cudaStream_t stream
) {
    if (inv_norm <= 0.0 || !isfinite(inv_norm)) {
        return cudaErrorInvalidValue;
    }

    cuDoubleComplex* state_complex_d = static_cast<cuDoubleComplex*>(state_d);

    const int blockSize = DEFAULT_BLOCK_SIZE;
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

/// Launch amplitude encoding kernel for float32
int launch_amplitude_encode_f32(
    const float* input_d,
    void* state_d,
    size_t input_len,
    size_t state_len,
    float inv_norm,
    cudaStream_t stream
) {
    if (inv_norm <= 0.0f || !isfinite(inv_norm)) {
        return cudaErrorInvalidValue;
    }

    cuComplex* state_complex_d = static_cast<cuComplex*>(state_d);

    const int blockSize = DEFAULT_BLOCK_SIZE;
    const int gridSize = (state_len / 2 + blockSize - 1) / blockSize;

    amplitude_encode_kernel_f32<<<gridSize, blockSize, 0, stream>>>(
        input_d,
        state_complex_d,
        input_len,
        state_len,
        inv_norm
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
    // - Block size: DEFAULT_BLOCK_SIZE threads (8 warps, good occupancy)
    // - Grid size: Enough blocks to saturate GPU, but not excessive
    const int blockSize = DEFAULT_BLOCK_SIZE;
    const size_t total_work = num_samples * (state_len / 2);

    // Calculate grid size: aim for high occupancy without too many blocks
    // Limit to reasonable number of blocks to avoid scheduler overhead
    const size_t blocks_needed = (total_work + blockSize - 1) / blockSize;
    const size_t max_blocks = MAX_GRID_BLOCKS;
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

/// Kernel: accumulate L2 norm using coalesced vectorized loads.
/// Each block atomically adds its partial sum to the output accumulator.
__global__ void l2_norm_kernel(
    const double* __restrict__ input,
    size_t input_len,
    double* __restrict__ out_accum
) {
    // Vectorized double2 loads for bandwidth and coalescing
    const size_t vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;

    double local_sum = 0.0;

    // Process two elements per iteration via double2
    size_t vec_offset = vec_idx;
    size_t offset = vec_offset * 2;
    while (offset + 1 < input_len) {
        const double2 v = __ldg(reinterpret_cast<const double2*>(input) + vec_offset);
        local_sum += v.x * v.x + v.y * v.y;
        vec_offset += stride;
        offset = vec_offset * 2;
    }

    // Handle tail element if input_len is odd
    if (offset < input_len) {
        const double v = __ldg(input + offset);
        local_sum += v * v;
    }

    const double block_sum = block_reduce_sum(local_sum);
    if (threadIdx.x == 0) {
        atomicAdd(out_accum, block_sum);
    }
}

/// Kernel: accumulate L2 norm using coalesced vectorized loads (float32).
/// Each block atomically adds its partial sum to the output accumulator.
__global__ void l2_norm_kernel_f32(
    const float* __restrict__ input,
    size_t input_len,
    float* __restrict__ out_accum
) {
    // Vectorized float2 loads for bandwidth and coalescing
    const size_t vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;

    float local_sum = 0.0f;

    // Process two elements per iteration via float2
    size_t vec_offset = vec_idx;
    size_t offset = vec_offset * 2;
    while (offset + 1 < input_len) {
        const float2 v = __ldg(reinterpret_cast<const float2*>(input) + vec_offset);
        local_sum += v.x * v.x + v.y * v.y;
        vec_offset += stride;
        offset = vec_offset * 2;
    }

    // Handle tail element if input_len is odd
    if (offset < input_len) {
        const float v = __ldg(input + offset);
        local_sum += v * v;
    }

    const float block_sum = block_reduce_sum_f32(local_sum);
    if (threadIdx.x == 0) {
        atomicAdd(out_accum, block_sum);
    }
}

/// Kernel: accumulate L2 norms for a batch.
/// Grid is organized as (blocks_per_sample * num_samples) blocks.
__global__ void l2_norm_batch_kernel(
    const double* __restrict__ input_batch,
    size_t num_samples,
    size_t sample_len,
    size_t blocks_per_sample,
    double* __restrict__ out_norms
) {
    const size_t sample_idx = blockIdx.x / blocks_per_sample;
    if (sample_idx >= num_samples) return;

    const size_t block_in_sample = blockIdx.x % blocks_per_sample;
    const size_t base = sample_idx * sample_len;

    const size_t vec_idx = block_in_sample * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * blocks_per_sample;

    double local_sum = 0.0;

    size_t vec_offset = vec_idx;
    size_t offset = vec_offset * 2;
    while (offset + 1 < sample_len) {
        const double2 v = __ldg(reinterpret_cast<const double2*>(input_batch + base) + vec_offset);
        local_sum += v.x * v.x + v.y * v.y;
        vec_offset += stride;
        offset = vec_offset * 2;
    }

    if (offset < sample_len) {
        const double v = __ldg(input_batch + base + offset);
        local_sum += v * v;
    }

    const double block_sum = block_reduce_sum(local_sum);
    if (threadIdx.x == 0) {
        atomicAdd(out_norms + sample_idx, block_sum);
    }
}

/// Kernel: converts accumulated sum-of-squares into inverse norms.
__global__ void finalize_inv_norm_kernel(
    double* __restrict__ norms,
    size_t count
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    double sum = norms[idx];
    // Guard against zero or NaN to avoid inf propagation
    if (sum <= 0.0 || !isfinite(sum)) {
        norms[idx] = 0.0;
    } else {
        norms[idx] = rsqrt(sum);
    }
}

/// Kernel: converts accumulated sum-of-squares into inverse norms (float32).
__global__ void finalize_inv_norm_kernel_f32(
    float* __restrict__ norms,
    size_t count
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    float sum = norms[idx];
    // Guard against zero or NaN to avoid inf propagation
    if (sum <= 0.0f || !isfinite(sum)) {
        norms[idx] = 0.0f;
    } else {
        norms[idx] = rsqrtf(sum);
    }
}

/// Launch L2 norm reduction for a single vector.
/// Writes the inverse norm (1 / ||x||) into `inv_norm_out_d`.
int launch_l2_norm(
    const double* input_d,
    size_t input_len,
    double* inv_norm_out_d,
    cudaStream_t stream
) {
    if (input_len == 0) {
        return cudaErrorInvalidValue;
    }

    cudaError_t memset_status = cudaMemsetAsync(
        inv_norm_out_d,
        0,
        sizeof(double),
        stream
    );
    if (memset_status != cudaSuccess) {
        return memset_status;
    }

    const int blockSize = DEFAULT_BLOCK_SIZE;
    const size_t elements_per_block = blockSize * 2; // double2 per thread
    size_t gridSize = (input_len + elements_per_block - 1) / elements_per_block;
    gridSize = (gridSize == 0) ? 1 : gridSize;
    const size_t maxBlocks = MAX_GRID_BLOCKS_L2_NORM;
    if (gridSize > maxBlocks) gridSize = maxBlocks;

    l2_norm_kernel<<<gridSize, blockSize, 0, stream>>>(
        input_d,
        input_len,
        inv_norm_out_d
    );

    // Finalize: convert accumulated sum to inverse norm
    finalize_inv_norm_kernel<<<1, 32, 0, stream>>>(
        inv_norm_out_d,
        1
    );

    return (int)cudaGetLastError();
}

/// Launch L2 norm reduction for a single vector (float32).
/// Writes the inverse norm (1 / ||x||) into `inv_norm_out_d`.
int launch_l2_norm_f32(
    const float* input_d,
    size_t input_len,
    float* inv_norm_out_d,
    cudaStream_t stream
) {
    if (input_len == 0) {
        return cudaErrorInvalidValue;
    }

    cudaError_t memset_status = cudaMemsetAsync(
        inv_norm_out_d,
        0,
        sizeof(float),
        stream
    );
    if (memset_status != cudaSuccess) {
        return memset_status;
    }

    const int blockSize = DEFAULT_BLOCK_SIZE;
    const size_t elements_per_block = blockSize * 2; // float2 per thread
    size_t gridSize = (input_len + elements_per_block - 1) / elements_per_block;
    gridSize = (gridSize == 0) ? 1 : gridSize;
    const size_t maxBlocks = MAX_GRID_BLOCKS_L2_NORM;
    if (gridSize > maxBlocks) gridSize = maxBlocks;

    l2_norm_kernel_f32<<<gridSize, blockSize, 0, stream>>>(
        input_d,
        input_len,
        inv_norm_out_d
    );

    // Finalize: convert accumulated sum to inverse norm
    finalize_inv_norm_kernel_f32<<<1, 32, 0, stream>>>(
        inv_norm_out_d,
        1
    );

    return (int)cudaGetLastError();
}

/// Launch L2 norm reduction for a batch of vectors.
/// Writes inverse norms for each sample into `inv_norms_out_d`.
int launch_l2_norm_batch(
    const double* input_batch_d,
    size_t num_samples,
    size_t sample_len,
    double* inv_norms_out_d,
    cudaStream_t stream
) {
    if (num_samples == 0 || sample_len == 0) {
        return cudaErrorInvalidValue;
    }

    cudaError_t memset_status = cudaMemsetAsync(
        inv_norms_out_d,
        0,
        num_samples * sizeof(double),
        stream
    );
    if (memset_status != cudaSuccess) {
        return memset_status;
    }

    const int blockSize = DEFAULT_BLOCK_SIZE;
    const size_t elements_per_block = blockSize * 2; // double2 per thread
    size_t blocks_per_sample = (sample_len + elements_per_block - 1) / elements_per_block;
    const size_t max_blocks_per_sample = MAX_BLOCKS_PER_SAMPLE;
    if (blocks_per_sample == 0) blocks_per_sample = 1;
    if (blocks_per_sample > max_blocks_per_sample) {
        blocks_per_sample = max_blocks_per_sample;
    }

    size_t gridSize = num_samples * blocks_per_sample;
    const size_t max_grid = CUDA_MAX_GRID_DIM_1D; // CUDA grid dimension limit for 1D launch
    if (gridSize > max_grid) {
        blocks_per_sample = max_grid / num_samples;
        if (blocks_per_sample == 0) {
            blocks_per_sample = 1;
        }
        gridSize = num_samples * blocks_per_sample;
    }

    l2_norm_batch_kernel<<<gridSize, blockSize, 0, stream>>>(
        input_batch_d,
        num_samples,
        sample_len,
        blocks_per_sample,
        inv_norms_out_d
    );

    const int finalizeBlock = FINALIZE_BLOCK_SIZE;
    const int finalizeGrid = (num_samples + finalizeBlock - 1) / finalizeBlock;
    finalize_inv_norm_kernel<<<finalizeGrid, finalizeBlock, 0, stream>>>(
        inv_norms_out_d,
        num_samples
    );

    return (int)cudaGetLastError();
}

/// Kernel: convert complex128 state vector to complex64.
__global__ void convert_state_to_complex64_kernel(
    const cuDoubleComplex* __restrict__ input_state,
    cuComplex* __restrict__ output_state,
    size_t len
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) return;

    const cuDoubleComplex v = input_state[idx];
    output_state[idx] = make_cuComplex((float)v.x, (float)v.y);
}

/// Launch conversion kernel from complex128 to complex64.
int convert_state_to_float(
    const cuDoubleComplex* input_state_d,
    cuComplex* output_state_d,
    size_t len,
    cudaStream_t stream
) {
    if (len == 0) {
        return cudaErrorInvalidValue;
    }

    const int blockSize = DEFAULT_BLOCK_SIZE;
    const int gridSize = (int)((len + blockSize - 1) / blockSize);

    convert_state_to_complex64_kernel<<<gridSize, blockSize, 0, stream>>>(
        input_state_d,
        output_state_d,
        len
    );

    return (int)cudaGetLastError();
}

// TODO: Future encoding methods:
// - launch_angle_encode (angle encoding)
// - launch_iqp_encode (IQP encoding)

} // extern "C"
