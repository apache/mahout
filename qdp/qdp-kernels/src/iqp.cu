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

// IQP (Instantaneous Quantum Polynomial) Encoding CUDA Kernels
//
// Creates entangled quantum states via diagonal phase gates:
// |psi> = H^n * U_phase(data) * H^n |0>^n
//
// The amplitude for basis state |z> is:
// amplitude[z] = (1/2^n) * sum_x exp(i*theta(x)) * (-1)^popcount(x AND z)
//
// Two variants:
// - enable_zz=0: theta(x) = sum_i x_i * data_i  (n parameters)
// - enable_zz=1: theta(x) = sum_i x_i * data_i + sum_{i<j} x_i * x_j * data_ij
//                (n + n*(n-1)/2 parameters)

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>

// Compute phase theta(x) for a given basis state x
__device__ double compute_phase(
    const double* __restrict__ data,
    size_t x,
    unsigned int num_qubits,
    int enable_zz
) {
    double phase = 0.0;

    // Single-qubit Z terms: sum_i x_i * data[i]
    for (unsigned int i = 0; i < num_qubits; ++i) {
        if ((x >> i) & 1U) {
            phase += data[i];
        }
    }

    // Two-qubit ZZ terms (if enabled): sum_{i<j} x_i * x_j * data[n + pair_index]
    if (enable_zz) {
        unsigned int pair_idx = num_qubits;
        for (unsigned int i = 0; i < num_qubits; ++i) {
            for (unsigned int j = i + 1; j < num_qubits; ++j) {
                if (((x >> i) & 1U) && ((x >> j) & 1U)) {
                    phase += data[pair_idx];
                }
                pair_idx++;
            }
        }
    }

    return phase;
}

__global__ void iqp_encode_kernel(
    const double* __restrict__ data,
    cuDoubleComplex* __restrict__ state,
    size_t state_len,
    unsigned int num_qubits,
    int enable_zz
) {
    size_t z = blockIdx.x * blockDim.x + threadIdx.x;
    if (z >= state_len) return;

    double real_sum = 0.0;
    double imag_sum = 0.0;

    // Sum over all input basis states x
    for (size_t x = 0; x < state_len; ++x) {
        double phase = compute_phase(data, x, num_qubits, enable_zz);

        // Compute (-1)^{popcount(x AND z)} using __popcll intrinsic
        int parity = __popcll(x & z) & 1;
        double sign = (parity == 0) ? 1.0 : -1.0;

        // Accumulate: sign * exp(i*phase) = sign * (cos(phase) + i*sin(phase))
        double cos_phase, sin_phase;
        sincos(phase, &sin_phase, &cos_phase);
        real_sum += sign * cos_phase;
        imag_sum += sign * sin_phase;
    }

    // Normalize by 1/2^n (state_len = 2^n)
    double norm = 1.0 / (double)state_len;
    state[z] = make_cuDoubleComplex(real_sum * norm, imag_sum * norm);
}

__global__ void iqp_encode_batch_kernel(
    const double* __restrict__ data_batch,
    cuDoubleComplex* __restrict__ state_batch,
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
        const size_t z = global_idx & state_mask;
        const double* data = data_batch + sample_idx * data_len;

        double real_sum = 0.0;
        double imag_sum = 0.0;

        // Sum over all input basis states x
        for (size_t x = 0; x < state_len; ++x) {
            double phase = compute_phase(data, x, num_qubits, enable_zz);

            // Compute (-1)^{popcount(x AND z)}
            int parity = __popcll(x & z) & 1;
            double sign = (parity == 0) ? 1.0 : -1.0;

            double cos_phase, sin_phase;
            sincos(phase, &sin_phase, &cos_phase);
            real_sum += sign * cos_phase;
            imag_sum += sign * sin_phase;
        }

        double norm = 1.0 / (double)state_len;
        state_batch[global_idx] = make_cuDoubleComplex(real_sum * norm, imag_sum * norm);
    }
}

extern "C" {

/// Launch IQP encoding kernel
///
/// # Arguments
/// * data_d - Device pointer to encoding parameters
/// * state_d - Device pointer to output state vector
/// * state_len - Target state vector size (2^num_qubits)
/// * num_qubits - Number of qubits
/// * enable_zz - 0 for Z-only, 1 for full ZZ interactions
/// * stream - CUDA stream for async execution (nullptr = default stream)
///
/// # Returns
/// CUDA error code (0 = cudaSuccess)
int launch_iqp_encode(
    const double* data_d,
    void* state_d,
    size_t state_len,
    unsigned int num_qubits,
    int enable_zz,
    cudaStream_t stream
) {
    if (state_len == 0 || num_qubits == 0) {
        return cudaErrorInvalidValue;
    }

    cuDoubleComplex* state_complex_d = static_cast<cuDoubleComplex*>(state_d);

    const int blockSize = 256;
    const int gridSize = (state_len + blockSize - 1) / blockSize;

    iqp_encode_kernel<<<gridSize, blockSize, 0, stream>>>(
        data_d,
        state_complex_d,
        state_len,
        num_qubits,
        enable_zz
    );

    return (int)cudaGetLastError();
}

/// Launch batch IQP encoding kernel
///
/// # Arguments
/// * data_batch_d - Device pointer to batch parameters (num_samples * data_len)
/// * state_batch_d - Device pointer to output batch state vectors
/// * num_samples - Number of samples in batch
/// * state_len - State vector size per sample (2^num_qubits)
/// * num_qubits - Number of qubits
/// * data_len - Length of each sample's data
/// * enable_zz - 0 for Z-only, 1 for full ZZ interactions
/// * stream - CUDA stream for async execution
///
/// # Returns
/// CUDA error code (0 = cudaSuccess)
int launch_iqp_encode_batch(
    const double* data_batch_d,
    void* state_batch_d,
    size_t num_samples,
    size_t state_len,
    unsigned int num_qubits,
    unsigned int data_len,
    int enable_zz,
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

    iqp_encode_batch_kernel<<<gridSize, blockSize, 0, stream>>>(
        data_batch_d,
        state_complex_d,
        num_samples,
        state_len,
        num_qubits,
        data_len,
        enable_zz
    );

    return (int)cudaGetLastError();
}

} // extern "C"
