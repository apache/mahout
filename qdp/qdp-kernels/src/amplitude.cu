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

// TODO: Future encoding methods:
// - launch_angle_encode (angle encoding)
// - launch_basis_encode (basis encoding)
// - launch_iqp_encode (IQP encoding)

} // extern "C"
