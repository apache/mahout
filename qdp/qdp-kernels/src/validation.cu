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

// Shared validation CUDA kernels.

#include <cuda_runtime.h>
#include <math.h>
#include "kernel_config.h"

__global__ void check_finite_batch_kernel_f32(
    const float* __restrict__ input_batch,
    size_t total_values,
    int* __restrict__ has_non_finite
) {
    const size_t stride = gridDim.x * blockDim.x;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_values;
         idx += stride) {
        if (!isfinite(input_batch[idx])) {
            atomicExch(has_non_finite, 1);
            return;
        }
    }
}

__global__ void check_finite_batch_kernel_f64(
    const double* __restrict__ input_batch,
    size_t total_values,
    int* __restrict__ has_non_finite
) {
    const size_t stride = gridDim.x * blockDim.x;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_values;
         idx += stride) {
        if (!isfinite(input_batch[idx])) {
            atomicExch(has_non_finite, 1);
            return;
        }
    }
}

// Validation error flags for basis indices (bitmask, written via atomicOr).
// 0 means valid.
#define BASIS_IDX_ERR_NON_FINITE   0x1
#define BASIS_IDX_ERR_NEGATIVE     0x2
#define BASIS_IDX_ERR_NON_INTEGER  0x4
#define BASIS_IDX_ERR_OUT_OF_RANGE 0x8

// Validate f32 basis indices and cast them to size_t in a single pass.
// `indices_out` receives the truncated indices (set to 0 when the sample is
// invalid to keep the downstream encode kernel bounded).
__global__ void validate_and_cast_basis_indices_kernel_f32(
    const float* __restrict__ input_batch,
    size_t num_samples,
    size_t state_len,
    size_t* __restrict__ indices_out,
    int* __restrict__ error_flags
) {
    const size_t stride = gridDim.x * blockDim.x;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < num_samples;
         idx += stride) {
        const float v = input_batch[idx];
        if (!isfinite(v)) {
            atomicOr(error_flags, BASIS_IDX_ERR_NON_FINITE);
            indices_out[idx] = 0;
            continue;
        }
        if (v < 0.0f) {
            atomicOr(error_flags, BASIS_IDX_ERR_NEGATIVE);
            indices_out[idx] = 0;
            continue;
        }
        const float truncated = truncf(v);
        if (truncated != v) {
            atomicOr(error_flags, BASIS_IDX_ERR_NON_INTEGER);
            indices_out[idx] = 0;
            continue;
        }
        if ((double)truncated >= (double)state_len) {
            atomicOr(error_flags, BASIS_IDX_ERR_OUT_OF_RANGE);
            indices_out[idx] = 0;
            continue;
        }
        indices_out[idx] = (size_t)truncated;
    }
}

// Bounds-check existing size_t basis indices against state_len.
__global__ void check_basis_indices_kernel_usize(
    const size_t* __restrict__ indices,
    size_t num_samples,
    size_t state_len,
    int* __restrict__ error_flags
) {
    const size_t stride = gridDim.x * blockDim.x;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < num_samples;
         idx += stride) {
        if (indices[idx] >= state_len) {
            atomicOr(error_flags, BASIS_IDX_ERR_OUT_OF_RANGE);
            return;
        }
    }
}

extern "C" {

/// Launch batch finite-value validation for float32 input.
///
/// Writes 1 to `has_non_finite_d` if any NaN/Inf is found, else leaves it at 0.
int launch_check_finite_batch_f32(
    const float* input_batch_d,
    size_t total_values,
    int* has_non_finite_d,
    cudaStream_t stream
) {
    if (total_values == 0 || has_non_finite_d == nullptr) {
        return cudaErrorInvalidValue;
    }

    cudaError_t memset_status = cudaMemsetAsync(
        has_non_finite_d,
        0,
        sizeof(int),
        stream
    );
    if (memset_status != cudaSuccess) {
        return memset_status;
    }

    const int blockSize = DEFAULT_BLOCK_SIZE;
    size_t gridSize = (total_values + blockSize - 1) / blockSize;
    if (gridSize == 0) {
        gridSize = 1;
    }
    if (gridSize > MAX_GRID_BLOCKS) {
        gridSize = MAX_GRID_BLOCKS;
    }

    check_finite_batch_kernel_f32<<<gridSize, blockSize, 0, stream>>>(
        input_batch_d,
        total_values,
        has_non_finite_d
    );

    return (int)cudaGetLastError();
}

/// Launch batch finite-value validation for float64 input.
///
/// Writes 1 to `has_non_finite_d` if any NaN/Inf is found, else leaves it at 0.
int launch_check_finite_batch_f64(
    const double* input_batch_d,
    size_t total_values,
    int* has_non_finite_d,
    cudaStream_t stream
) {
    if (total_values == 0 || has_non_finite_d == nullptr) {
        return cudaErrorInvalidValue;
    }

    cudaError_t memset_status = cudaMemsetAsync(
        has_non_finite_d,
        0,
        sizeof(int),
        stream
    );
    if (memset_status != cudaSuccess) {
        return memset_status;
    }

    const int blockSize = DEFAULT_BLOCK_SIZE;
    size_t gridSize = (total_values + blockSize - 1) / blockSize;
    if (gridSize == 0) {
        gridSize = 1;
    }
    if (gridSize > MAX_GRID_BLOCKS) {
        gridSize = MAX_GRID_BLOCKS;
    }

    check_finite_batch_kernel_f64<<<gridSize, blockSize, 0, stream>>>(
        input_batch_d,
        total_values,
        has_non_finite_d
    );

    return (int)cudaGetLastError();
}

/// Launch combined validation + cast for float32 basis indices.
///
/// Writes a bitmask into `error_flags_d` (0 = all valid) and casts each valid
/// sample into `indices_out_d` (a `size_t` per sample). Invalid samples have
/// their output slot zeroed so the downstream encode kernel is bounded even
/// if the caller ignores the error flag.
int launch_validate_and_cast_basis_indices_f32(
    const float* input_batch_d,
    size_t num_samples,
    size_t state_len,
    size_t* indices_out_d,
    int* error_flags_d,
    cudaStream_t stream
) {
    if (num_samples == 0 || state_len == 0
        || error_flags_d == nullptr || indices_out_d == nullptr) {
        return cudaErrorInvalidValue;
    }

    cudaError_t memset_status = cudaMemsetAsync(error_flags_d, 0, sizeof(int), stream);
    if (memset_status != cudaSuccess) {
        return memset_status;
    }

    const int blockSize = DEFAULT_BLOCK_SIZE;
    size_t gridSize = (num_samples + blockSize - 1) / blockSize;
    if (gridSize == 0) {
        gridSize = 1;
    }
    if (gridSize > MAX_GRID_BLOCKS) {
        gridSize = MAX_GRID_BLOCKS;
    }

    validate_and_cast_basis_indices_kernel_f32<<<gridSize, blockSize, 0, stream>>>(
        input_batch_d,
        num_samples,
        state_len,
        indices_out_d,
        error_flags_d
    );

    return (int)cudaGetLastError();
}

/// Launch bounds-check for size_t basis indices.
int launch_check_basis_indices_usize(
    const size_t* indices_d,
    size_t num_samples,
    size_t state_len,
    int* error_flags_d,
    cudaStream_t stream
) {
    if (num_samples == 0 || state_len == 0 || error_flags_d == nullptr) {
        return cudaErrorInvalidValue;
    }

    cudaError_t memset_status = cudaMemsetAsync(error_flags_d, 0, sizeof(int), stream);
    if (memset_status != cudaSuccess) {
        return memset_status;
    }

    const int blockSize = DEFAULT_BLOCK_SIZE;
    size_t gridSize = (num_samples + blockSize - 1) / blockSize;
    if (gridSize == 0) {
        gridSize = 1;
    }
    if (gridSize > MAX_GRID_BLOCKS) {
        gridSize = MAX_GRID_BLOCKS;
    }

    check_basis_indices_kernel_usize<<<gridSize, blockSize, 0, stream>>>(
        indices_d,
        num_samples,
        state_len,
        error_flags_d
    );

    return (int)cudaGetLastError();
}

} // extern "C"
