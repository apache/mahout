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

/// Single sample basis encoding kernel
///
/// Sets state[basis_index] = 1.0 + 0.0i, all others = 0.0 + 0.0i
__global__ void basis_encode_kernel(
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
__global__ void basis_encode_batch_kernel(
    const size_t* __restrict__ basis_indices,
    cuDoubleComplex* __restrict__ state_batch,
    size_t num_samples,
    size_t state_len
) {
    // Grid-stride loop over all elements across all samples
    const size_t total_elements = num_samples * state_len;
    const size_t stride = gridDim.x * blockDim.x;

    for (size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
         global_idx < total_elements;
         global_idx += stride) {
        // Decompose into (sample_idx, element_idx)
        const size_t sample_idx = global_idx / state_len;
        const size_t element_idx = global_idx % state_len;

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

extern "C" {

/// Launch basis encoding kernel
///
/// # Arguments
/// * basis_index - The computational basis state index (0 to state_len-1)
/// * state_d - Device pointer to output state vector
/// * state_len - Target state vector size (2^num_qubits)
/// * stream - CUDA stream for async execution (nullptr = default stream)
///
/// # Returns
/// CUDA error code (0 = cudaSuccess)
int launch_basis_encode(
    size_t basis_index,
    void* state_d,
    size_t state_len,
    cudaStream_t stream
) {
    if (state_len == 0) {
        return cudaErrorInvalidValue;
    }

    if (basis_index >= state_len) {
        return cudaErrorInvalidValue;
    }

    cuDoubleComplex* state_complex_d = static_cast<cuDoubleComplex*>(state_d);

    const int blockSize = 256;
    const int gridSize = (state_len + blockSize - 1) / blockSize;

    basis_encode_kernel<<<gridSize, blockSize, 0, stream>>>(
        basis_index,
        state_complex_d,
        state_len
    );

    return (int)cudaGetLastError();
}

/// Launch batch basis encoding kernel
///
/// # Arguments
/// * basis_indices_d - Device pointer to array of basis indices (one per sample)
/// * state_batch_d - Device pointer to output batch state vectors
/// * num_samples - Number of samples in batch
/// * state_len - State vector size per sample (2^num_qubits)
/// * stream - CUDA stream for async execution
///
/// # Returns
/// CUDA error code (0 = cudaSuccess)
int launch_basis_encode_batch(
    const size_t* basis_indices_d,
    void* state_batch_d,
    size_t num_samples,
    size_t state_len,
    cudaStream_t stream
) {
    if (num_samples == 0 || state_len == 0) {
        return cudaErrorInvalidValue;
    }

    cuDoubleComplex* state_complex_d = static_cast<cuDoubleComplex*>(state_batch_d);

    const int blockSize = 256;
    const size_t total_elements = num_samples * state_len;
    const size_t blocks_needed = (total_elements + blockSize - 1) / blockSize;
    const size_t max_blocks = 2048;
    const size_t gridSize = (blocks_needed < max_blocks) ? blocks_needed : max_blocks;

    basis_encode_batch_kernel<<<gridSize, blockSize, 0, stream>>>(
        basis_indices_d,
        state_complex_d,
        num_samples,
        state_len
    );

    return (int)cudaGetLastError();
}

} // extern "C"
