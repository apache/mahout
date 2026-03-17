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

// Phase Encoding CUDA Kernels
//
// For each qubit phase x_k, build a product state:
// |psi(x)> = ⊗_k (1/√2)(|0> + e^{i x_k}|1>)
//
// Equivalently, amplitude at basis index b is:
//   state[b] = (1/√2^n) * exp(i * Σ_k x_k * b_k)
// where b_k = (b >> k) & 1 is the k-th bit of b.
//
// Circuit: H⊗N layer followed by P(x_k) per qubit.
// Depth: 2.  Input x_k ∈ (0, 2π] recommended to avoid aliasing.

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
#include "kernel_config.h"

// Precompute 1/√2^n as a compile-time-friendly inline.
// For n qubits the norm factor is pow(M_SQRT1_2, n).
__device__ __forceinline__ double phase_norm(unsigned int num_qubits) {
    // M_SQRT1_2 = 1/√2 ≈ 0.7071067811865476
    double factor = 1.0;
    for (unsigned int k = 0; k < num_qubits; ++k) {
        factor *= M_SQRT1_2;
    }
    return factor;
}

__global__ void phase_encode_kernel(
    const double* __restrict__ phases,
    cuDoubleComplex* __restrict__ state,
    size_t state_len,
    unsigned int num_qubits
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_len) return;

    // φ(idx) = Σ_k phases[k] * b_k,  b_k = (idx >> k) & 1
    double phi = 0.0;
    for (unsigned int bit = 0; bit < num_qubits; ++bit) {
        if ((idx >> bit) & 1U) {
            phi += phases[bit];
        }
    }

    double norm = phase_norm(num_qubits);
    double re, im;
    sincos(phi, &im, &re);   // re = cos(φ), im = sin(φ)

    state[idx] = make_cuDoubleComplex(norm * re, norm * im);
}

__global__ void phase_encode_batch_kernel(
    const double* __restrict__ phases_batch,
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
        const double* phases = phases_batch + sample_idx * num_qubits;

        double phi = 0.0;
        for (unsigned int bit = 0; bit < num_qubits; ++bit) {
            if ((element_idx >> bit) & 1U) {
                phi += phases[bit];
            }
        }

        double norm = phase_norm(num_qubits);
        double re, im;
        sincos(phi, &im, &re);

        state_batch[global_idx] = make_cuDoubleComplex(norm * re, norm * im);
    }
}

extern "C" {

/// Launch phase encoding kernel
///
/// Produces the product state ⊗_k (1/√2)(|0> + e^{i x_k}|1>) in the
/// computational basis.  Each amplitude state[b] is written as:
///   (1/√2^n) * (cos(φ(b)) + i*sin(φ(b))),  φ(b) = Σ_k phases[k] * b_k
///
/// # Arguments
/// * phases_d  - Device pointer to per-qubit phase angles (length num_qubits)
/// * state_d   - Device pointer to output state vector (length state_len)
/// * state_len - Target state vector size (2^num_qubits)
/// * num_qubits - Number of qubits (phases length)
/// * stream    - CUDA stream for async execution (nullptr = default stream)
///
/// # Returns
/// CUDA error code (0 = cudaSuccess)
int launch_phase_encode(
    const double* phases_d,
    void* state_d,
    size_t state_len,
    unsigned int num_qubits,
    cudaStream_t stream
) {
    if (state_len == 0 || num_qubits == 0) {
        return cudaErrorInvalidValue;
    }

    cuDoubleComplex* state_complex_d = static_cast<cuDoubleComplex*>(state_d);

    const int blockSize = DEFAULT_BLOCK_SIZE;
    const int gridSize = (state_len + blockSize - 1) / blockSize;

    phase_encode_kernel<<<gridSize, blockSize, 0, stream>>>(
        phases_d,
        state_complex_d,
        state_len,
        num_qubits
    );

    return (int)cudaGetLastError();
}

/// Launch batch phase encoding kernel
///
/// # Arguments
/// * phases_batch_d - Device pointer to batch phases (num_samples * num_qubits)
/// * state_batch_d  - Device pointer to output batch state vectors
/// * num_samples    - Number of samples in batch
/// * state_len      - State vector size per sample (2^num_qubits)
/// * num_qubits     - Number of qubits (phases length per sample)
/// * stream         - CUDA stream for async execution
///
/// # Returns
/// CUDA error code (0 = cudaSuccess)
int launch_phase_encode_batch(
    const double* phases_batch_d,
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

    const int blockSize = DEFAULT_BLOCK_SIZE;
    const size_t total_elements = num_samples * state_len;
    const size_t blocks_needed = (total_elements + blockSize - 1) / blockSize;
    const size_t max_blocks = MAX_GRID_BLOCKS;
    const size_t gridSize = (blocks_needed < max_blocks) ? blocks_needed : max_blocks;

    phase_encode_batch_kernel<<<gridSize, blockSize, 0, stream>>>(
        phases_batch_d,
        state_complex_d,
        num_samples,
        state_len,
        num_qubits
    );

    return (int)cudaGetLastError();
}

} // extern "C"
