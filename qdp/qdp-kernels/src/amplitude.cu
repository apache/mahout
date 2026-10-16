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
#include <stdint.h>
#include "kernel_config.h"

extern "C" __global__ void amplitude_encode_kernel(
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

extern "C" __global__ void amplitude_encode_kernel_f32(
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

/// Optimized batch amplitude encoding kernel
///
/// Memory Layout (row-major):
/// - input_batch: [sample0_data | sample1_data | ... | sampleN_data]
/// - state_batch: [sample0_state | sample1_state | ... | sampleN_state]
///
/// Optimizations:
/// 1. Vectorized double2 loads for 128-bit memory transactions when aligned
/// 2. Grid-stride loop for arbitrary batch sizes
/// 3. Coalesced memory access within warps
/// 4. Scalar fallback for misaligned sample bases and odd tails
extern "C" __global__ void amplitude_encode_batch_kernel(
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

        double v1, v2;
        const double* sample_input = input_batch + input_base;
        const bool sample_input_aligned =
            (reinterpret_cast<uintptr_t>(sample_input) & (alignof(double2) - 1)) == 0;

        if (sample_input_aligned && elem_offset + 1 < input_len) {
            const double2 vec_data =
                __ldg(reinterpret_cast<const double2*>(sample_input) + elem_pair);
            v1 = vec_data.x;
            v2 = vec_data.y;
        } else if (elem_offset < input_len) {
            v1 = __ldg(sample_input + elem_offset);
            v2 = (elem_offset + 1 < input_len)
                ? __ldg(sample_input + elem_offset + 1)
                : 0.0;
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

/// Optimized batch amplitude encoding kernel (float32)
///
/// Memory Layout (row-major):
/// - input_batch: [sample0_data | sample1_data | ... | sampleN_data]
/// - state_batch: [sample0_state | sample1_state | ... | sampleN_state]
///
/// Optimizations:
/// 1. Vectorized float2 loads for 64-bit memory transactions
/// 2. Grid-stride loop for arbitrary batch sizes
/// 3. Coalesced memory access within warps
/// 4. Minimized register pressure
extern "C" __global__ void amplitude_encode_batch_kernel_f32(
    const float* __restrict__ input_batch,
    cuComplex* __restrict__ state_batch,
    const float* __restrict__ inv_norms,
    size_t num_samples,
    size_t input_len,
    size_t state_len
) {
    // Grid-stride loop pattern for flexibility
    const size_t elements_per_sample = state_len / 2;
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
        const float inv_norm = inv_norms[sample_idx];

        float v1, v2;
        const float* sample_input = input_batch + input_base;
        const bool sample_input_aligned =
            (reinterpret_cast<uintptr_t>(sample_input) & (alignof(float2) - 1)) == 0;

        if (sample_input_aligned && elem_offset + 1 < input_len) {
            const float2 vec_data =
                __ldg(reinterpret_cast<const float2*>(sample_input) + elem_pair);
            v1 = vec_data.x;
            v2 = vec_data.y;
        } else if (elem_offset < input_len) {
            v1 = __ldg(sample_input + elem_offset);
            v2 = (elem_offset + 1 < input_len)
                ? __ldg(sample_input + elem_offset + 1)
                : 0.0f;
        } else {
            v1 = v2 = 0.0f;
        }

        // Normalize and write as complex numbers
        const cuComplex c1 = make_cuComplex(v1 * inv_norm, 0.0f);
        const cuComplex c2 = make_cuComplex(v2 * inv_norm, 0.0f);

        // Write to global memory (coalesced within warp)
        state_batch[state_base + elem_offset] = c1;
        if (elem_offset + 1 < state_len) {
            state_batch[state_base + elem_offset + 1] = c2;
        }
    }
}

/// Kernel: accumulate L2 norm using coalesced vectorized loads.
/// Each block atomically adds its partial sum to the output accumulator.
extern "C" __global__ void l2_norm_kernel(
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
extern "C" __global__ void l2_norm_kernel_f32(
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
extern "C" __global__ void l2_norm_batch_kernel(
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
    const double* sample_input = input_batch + base;
    const bool sample_input_aligned =
        (reinterpret_cast<uintptr_t>(sample_input) & (alignof(double2) - 1)) == 0;

    double local_sum = 0.0;

    size_t vec_offset = vec_idx;
    size_t offset = vec_offset * 2;
    if (sample_input_aligned) {
        while (offset + 1 < sample_len) {
            const double2 v = __ldg(reinterpret_cast<const double2*>(sample_input) + vec_offset);
            local_sum += v.x * v.x + v.y * v.y;
            vec_offset += stride;
            offset = vec_offset * 2;
        }
    } else {
        while (offset + 1 < sample_len) {
            const double v1 = __ldg(sample_input + offset);
            const double v2 = __ldg(sample_input + offset + 1);
            local_sum += v1 * v1 + v2 * v2;
            vec_offset += stride;
            offset = vec_offset * 2;
        }
    }

    if (offset < sample_len) {
        const double v = __ldg(sample_input + offset);
        local_sum += v * v;
    }

    const double block_sum = block_reduce_sum(local_sum);
    if (threadIdx.x == 0) {
        atomicAdd(out_norms + sample_idx, block_sum);
    }
}

/// Kernel: accumulate L2 norms for a batch (float32).
/// Grid is organized as (blocks_per_sample * num_samples) blocks.
extern "C" __global__ void l2_norm_batch_kernel_f32(
    const float* __restrict__ input_batch,
    size_t num_samples,
    size_t sample_len,
    size_t blocks_per_sample,
    float* __restrict__ out_norms
) {
    const size_t sample_idx = blockIdx.x / blocks_per_sample;
    if (sample_idx >= num_samples) return;

    const size_t block_in_sample = blockIdx.x % blocks_per_sample;
    const size_t base = sample_idx * sample_len;

    const size_t vec_idx = block_in_sample * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * blocks_per_sample;
    const float* sample_input = input_batch + base;
    const bool sample_input_aligned =
        (reinterpret_cast<uintptr_t>(sample_input) & (alignof(float2) - 1)) == 0;

    float local_sum = 0.0f;

    size_t vec_offset = vec_idx;
    size_t offset = vec_offset * 2;
    if (sample_input_aligned) {
        while (offset + 1 < sample_len) {
            const float2 v = __ldg(reinterpret_cast<const float2*>(sample_input) + vec_offset);
            local_sum += v.x * v.x + v.y * v.y;
            vec_offset += stride;
            offset = vec_offset * 2;
        }
    } else {
        while (offset + 1 < sample_len) {
            const float v1 = __ldg(sample_input + offset);
            const float v2 = __ldg(sample_input + offset + 1);
            local_sum += v1 * v1 + v2 * v2;
            vec_offset += stride;
            offset = vec_offset * 2;
        }
    }

    if (offset < sample_len) {
        const float v = __ldg(sample_input + offset);
        local_sum += v * v;
    }

    const float block_sum = block_reduce_sum_f32(local_sum);
    if (threadIdx.x == 0) {
        atomicAdd(out_norms + sample_idx, block_sum);
    }
}

/// Kernel: converts accumulated sum-of-squares into inverse norms.
extern "C" __global__ void finalize_inv_norm_kernel(
    double* __restrict__ norms,
    size_t count,
    int* __restrict__ error_flag
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    // norms[idx] holds the sum of squares; turn it into the inverse norm.
    // A zero or non-finite sum means the sample cannot be normalised: raise
    // the flag and write 0 so the encode kernel emits a zero row rather
    // than Inf/NaN.
    const double sum = norms[idx];
    if (sum > (double)0 && isfinite(sum)) {
        norms[idx] = (double)1 / sqrt(sum);
    } else {
        norms[idx] = (double)0;
        if (error_flag) atomicOr(error_flag, 1);
    }
}

/// Kernel: converts accumulated sum-of-squares into inverse norms (float32).
extern "C" __global__ void finalize_inv_norm_kernel_f32(
    float* __restrict__ norms,
    size_t count,
    int* __restrict__ error_flag
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    // norms[idx] holds the sum of squares; turn it into the inverse norm.
    // A zero or non-finite sum means the sample cannot be normalised: raise
    // the flag and write 0 so the encode kernel emits a zero row rather
    // than Inf/NaN.
    const float sum = norms[idx];
    if (sum > (float)0 && isfinite(sum)) {
        norms[idx] = (float)1 / sqrtf(sum);
    } else {
        norms[idx] = (float)0;
        if (error_flag) atomicOr(error_flag, 1);
    }
}

/// Kernel: convert complex128 state vector to complex64.
extern "C" __global__ void convert_state_to_complex64_kernel(
    const cuDoubleComplex* __restrict__ input_state,
    cuComplex* __restrict__ output_state,
    size_t len
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) return;

    const cuDoubleComplex v = input_state[idx];
    output_state[idx] = make_cuComplex((float)v.x, (float)v.y);
}

/// Kernel: convert complex64 state vector to complex128.
extern "C" __global__ void convert_state_to_complex128_kernel(
    const cuComplex* __restrict__ input_state,
    cuDoubleComplex* __restrict__ output_state,
    size_t len
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) return;

    const cuComplex v = input_state[idx];
    output_state[idx] = make_cuDoubleComplex((double)v.x, (double)v.y);
}

// TODO: Future encoding methods:
// - launch_angle_encode (angle encoding)
// - launch_iqp_encode (IQP encoding)
