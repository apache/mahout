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

// Angle Encoding CUDA Kernels
//
// For each qubit angle x_k, build a product state:
// |psi(x)> = ⊗_k (cos(x_k)|0> + sin(x_k)|1>)

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>

__global__ void angle_encode_kernel(
    const double* __restrict__ angles,
    cuDoubleComplex* __restrict__ state,
    size_t state_len,
    unsigned int num_qubits
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_len) return;

    double amplitude = 1.0;
    for (unsigned int bit = 0; bit < num_qubits; ++bit) {
        double angle = angles[bit];
        amplitude *= ((idx >> bit) & 1U) ? sin(angle) : cos(angle);
    }

    state[idx] = make_cuDoubleComplex(amplitude, 0.0);
}

__global__ void angle_encode_batch_kernel(
    const double* __restrict__ angles_batch,
    cuDoubleComplex* __restrict__ state_batch,
    size_t num_samples,
    size_t state_len,
    unsigned int num_qubits
) {
    const size_t total_elements = num_samples * state_len;
    const size_t stride = gridDim.x * blockDim.x;
    const size_t state_mask = state_len - 1;

    for (size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
         global_idx < total_elements;
         global_idx += stride) {
        const size_t sample_idx = global_idx >> num_qubits;
        const size_t element_idx = global_idx & state_mask;
        const double* angles = angles_batch + sample_idx * num_qubits;

        double amplitude = 1.0;
        for (unsigned int bit = 0; bit < num_qubits; ++bit) {
            double angle = angles[bit];
            amplitude *= ((element_idx >> bit) & 1U) ? sin(angle) : cos(angle);
        }

        state_batch[global_idx] = make_cuDoubleComplex(amplitude, 0.0);
    }
}

extern "C" {

/// Launch angle encoding kernel
///
/// # Arguments
/// * angles_d - Device pointer to per-qubit angles
/// * state_d - Device pointer to output state vector
/// * state_len - Target state vector size (2^num_qubits)
/// * num_qubits - Number of qubits (angles length)
/// * stream - CUDA stream for async execution (nullptr = default stream)
///
/// # Returns
/// CUDA error code (0 = cudaSuccess)
int launch_angle_encode(
    const double* angles_d,
    void* state_d,
    size_t state_len,
    unsigned int num_qubits,
    cudaStream_t stream
) {
    if (state_len == 0 || num_qubits == 0) {
        return cudaErrorInvalidValue;
    }

    cuDoubleComplex* state_complex_d = static_cast<cuDoubleComplex*>(state_d);

    const int blockSize = 256;
    const int gridSize = (state_len + blockSize - 1) / blockSize;

    angle_encode_kernel<<<gridSize, blockSize, 0, stream>>>(
        angles_d,
        state_complex_d,
        state_len,
        num_qubits
    );

    return (int)cudaGetLastError();
}

/// Launch batch angle encoding kernel
///
/// # Arguments
/// * angles_batch_d - Device pointer to batch angles (num_samples * num_qubits)
/// * state_batch_d - Device pointer to output batch state vectors
/// * num_samples - Number of samples in batch
/// * state_len - State vector size per sample (2^num_qubits)
/// * num_qubits - Number of qubits (angles length)
/// * stream - CUDA stream for async execution
///
/// # Returns
/// CUDA error code (0 = cudaSuccess)
int launch_angle_encode_batch(
    const double* angles_batch_d,
    void* state_batch_d,
    size_t num_samples,
    size_t state_len,
    unsigned int num_qubits,
    cudaStream_t stream
) {
    if (num_samples == 0 || state_len == 0 || num_qubits == 0) {
        return cudaErrorInvalidValue;
    }

    cuDoubleComplex* state_complex_d = static_cast<cuDoubleComplex*>(state_batch_d);

    const int blockSize = 256;
    const size_t total_elements = num_samples * state_len;
    const size_t blocks_needed = (total_elements + blockSize - 1) / blockSize;
    const size_t max_blocks = 2048;
    const size_t gridSize = (blocks_needed < max_blocks) ? blocks_needed : max_blocks;

    angle_encode_batch_kernel<<<gridSize, blockSize, 0, stream>>>(
        angles_batch_d,
        state_complex_d,
        num_samples,
        state_len,
        num_qubits
    );

    return (int)cudaGetLastError();
}

} // extern "C"
