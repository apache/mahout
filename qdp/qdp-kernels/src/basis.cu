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

// Basis Encoding CUDA Kernels
//
// Maps integer indices to computational basis states.
// For index i with n qubits: state[i] = 1.0, all others = 0.0
// Example: index=3 with 3 qubits → |011⟩ (state[3] = 1.0)

#include <cuda_runtime.h>
#include <cuComplex.h>
#include "kernel_config.h"

/// Single sample basis encoding kernel
///
/// Sets state[basis_index] = 1.0 + 0.0i, all others = 0.0 + 0.0i
extern "C" __global__ void basis_encode_kernel(
    size_t basis_index,
    cuDoubleComplex* __restrict__ state,
    size_t state_len
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_len) return;

    if (idx == basis_index) {
        state[idx] = make_cuDoubleComplex(1.0, 0.0);
    } else {
        state[idx] = make_cuDoubleComplex(0.0, 0.0);
    }
}

/// Batch basis encoding kernel
///
/// Each sample has its own basis index, resulting in independent basis states.
/// Memory layout:
/// - basis_indices: [idx0, idx1, ..., idxN]
/// - state_batch: [sample0_state | sample1_state | ... | sampleN_state]
extern "C" __global__ void basis_encode_batch_kernel(
    const size_t* __restrict__ basis_indices,
    cuDoubleComplex* __restrict__ state_batch,
    size_t num_samples,
    size_t state_len,
    unsigned int num_qubits
) {
    // Grid-stride loop over all elements across all samples
    const size_t total_elements = num_samples * state_len;
    const size_t stride = gridDim.x * blockDim.x;
    const size_t state_mask = state_len - 1;

    for (size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
         global_idx < total_elements;
         global_idx += stride) {
        // Decompose into (sample_idx, element_idx)
        // state_len = 2^num_qubits, so division/modulo can use shift/mask
        const size_t sample_idx = global_idx >> num_qubits;
        const size_t element_idx = global_idx & state_mask;

        // Get basis index for this sample
        const size_t basis_index = basis_indices[sample_idx];

        // Set amplitude: 1.0 at basis_index, 0.0 elsewhere
        if (element_idx == basis_index) {
            state_batch[global_idx] = make_cuDoubleComplex(1.0, 0.0);
        } else {
            state_batch[global_idx] = make_cuDoubleComplex(0.0, 0.0);
        }
    }
}

/// Single sample basis encoding kernel (F32)
///
/// Sets state[basis_index] = 1.0 + 0.0i, all others = 0.0 + 0.0i
extern "C" __global__ void basis_encode_kernel_f32(
    size_t basis_index,
    cuComplex* __restrict__ state,
    size_t state_len
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_len) return;

    if (idx == basis_index) {
        state[idx] = make_cuComplex(1.0f, 0.0f);
    } else {
        state[idx] = make_cuComplex(0.0f, 0.0f);
    }
}

/// Batch basis encoding kernel (F32)
///
/// Each sample has its own basis index, resulting in independent basis states.
/// Memory layout:
/// - basis_indices: [idx0, idx1, ..., idxN]
/// - state_batch: [sample0_state | sample1_state | ... | sampleN_state]
extern "C" __global__ void basis_encode_batch_kernel_f32(
    const size_t* __restrict__ basis_indices,
    cuComplex* __restrict__ state_batch,
    size_t num_samples,
    size_t state_len,
    unsigned int num_qubits
) {
    // Grid-stride loop over all elements across all samples
    const size_t total_elements = num_samples * state_len;
    const size_t stride = gridDim.x * blockDim.x;
    const size_t state_mask = state_len - 1;

    for (size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
         global_idx < total_elements;
         global_idx += stride) {
        // Decompose into (sample_idx, element_idx)
        // state_len = 2^num_qubits, so division/modulo can use shift/mask
        const size_t sample_idx = global_idx >> num_qubits;
        const size_t element_idx = global_idx & state_mask;

        // Get basis index for this sample
        const size_t basis_index = basis_indices[sample_idx];

        // Set amplitude: 1.0 at basis_index, 0.0 elsewhere
        if (element_idx == basis_index) {
            state_batch[global_idx] = make_cuComplex(1.0f, 0.0f);
        } else {
            state_batch[global_idx] = make_cuComplex(0.0f, 0.0f);
        }
    }
}
