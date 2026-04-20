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

} // extern "C"
