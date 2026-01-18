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
// IQP encoding creates a quantum state using a diagonal unitary circuit:
// |ψ(x)⟩ = H^⊗n U_diag(x) H^⊗n |0⟩^n
//
// Mathematically equivalent to direct state preparation:
// |ψ(x)⟩ = 1/√(2^n) Σ_{z=0}^{2^n-1} exp(i·φ(z,x)) |z⟩
//
// where φ(z,x) is the phase function combining:
// - Single-qubit rotations: Σ_i x_i · z_i
// - Two-qubit entangling terms: Σ_{i<j} x_i · x_j · z_i · z_j
//
// This implementation directly computes the amplitude for each basis state,
// avoiding circuit simulation overhead for O(1) per-amplitude computation.

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
#include <stdint.h>  // For SIZE_MAX

/// IQP encoding kernel (no entanglement)
///
/// Creates state: |ψ(x)⟩ = 1/√(2^n) Σ_z exp(i·Σ_i x_i·z_i) |z⟩
///
/// Each thread computes one amplitude based on basis state index.
/// Phase = sum over active qubits (where z_i = 1) of x_i.
__global__ void iqp_encode_kernel(
    const double* __restrict__ input,
    cuDoubleComplex* __restrict__ state,
    size_t num_features,
    size_t num_qubits,
    size_t state_len,
    double norm_factor
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_len) return;

    // Compute phase for this basis state
    // Phase = Σ_i x_i * z_i where z_i is bit i of idx
    double phase = 0.0;
    size_t z = idx;

    // Sum contributions from each active qubit
    // Using __ldg() for read-only cache path (L1 texture cache)
    for (size_t i = 0; i < num_features && i < num_qubits; ++i) {
        if (z & 1) {
            phase += __ldg(&input[i]);
        }
        z >>= 1;
    }

    // Amplitude = norm_factor * exp(i * phase)
    // Using Euler's formula: exp(i*θ) = cos(θ) + i*sin(θ)
    double cos_phase, sin_phase;
    sincos(phase, &sin_phase, &cos_phase);

    state[idx] = make_cuDoubleComplex(norm_factor * cos_phase, norm_factor * sin_phase);
}

/// IQP encoding kernel with linear entanglement
///
/// Creates state with nearest-neighbor entangling terms:
/// φ(z,x) = Σ_i x_i·z_i + Σ_i x_i·x_{i+1}·z_i·z_{i+1}
///
/// The entanglement pattern connects consecutive qubits.
__global__ void iqp_encode_linear_kernel(
    const double* __restrict__ input,
    cuDoubleComplex* __restrict__ state,
    size_t num_features,
    size_t num_qubits,
    size_t state_len,
    double norm_factor
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_len) return;

    double phase = 0.0;

    // Single-qubit rotations: Σ_i x_i * z_i
    size_t z = idx;
    for (size_t i = 0; i < num_features && i < num_qubits; ++i) {
        if (z & 1) {
            phase += __ldg(&input[i]);
        }
        z >>= 1;
    }

    // Linear entanglement: Σ_i x_i * x_{i+1} * z_i * z_{i+1}
    // Only add term when both adjacent qubits are active
    for (size_t i = 0; i + 1 < num_features && i + 1 < num_qubits; ++i) {
        size_t bit_i = (idx >> i) & 1;
        size_t bit_i1 = (idx >> (i + 1)) & 1;
        if (bit_i && bit_i1) {
            phase += __ldg(&input[i]) * __ldg(&input[i + 1]);
        }
    }

    double cos_phase, sin_phase;
    sincos(phase, &sin_phase, &cos_phase);

    state[idx] = make_cuDoubleComplex(norm_factor * cos_phase, norm_factor * sin_phase);
}

/// IQP encoding kernel with full entanglement
///
/// Creates state with all-pairs entangling terms:
/// φ(z,x) = Σ_i x_i·z_i + Σ_{i<j} x_i·x_j·z_i·z_j
///
/// O(n²) interactions for expressivity at cost of computation.
__global__ void iqp_encode_full_kernel(
    const double* __restrict__ input,
    cuDoubleComplex* __restrict__ state,
    size_t num_features,
    size_t num_qubits,
    size_t state_len,
    double norm_factor
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_len) return;

    double phase = 0.0;

    // Single-qubit rotations
    size_t z = idx;
    for (size_t i = 0; i < num_features && i < num_qubits; ++i) {
        if (z & 1) {
            phase += __ldg(&input[i]);
        }
        z >>= 1;
    }

    // Full entanglement: Σ_{i<j} x_i * x_j * z_i * z_j
    size_t effective_n = (num_features < num_qubits) ? num_features : num_qubits;
    for (size_t i = 0; i < effective_n; ++i) {
        size_t bit_i = (idx >> i) & 1;
        if (!bit_i) continue;  // Skip if qubit i is not active

        for (size_t j = i + 1; j < effective_n; ++j) {
            size_t bit_j = (idx >> j) & 1;
            if (bit_j) {
                phase += __ldg(&input[i]) * __ldg(&input[j]);
            }
        }
    }

    double cos_phase, sin_phase;
    sincos(phase, &sin_phase, &cos_phase);

    state[idx] = make_cuDoubleComplex(norm_factor * cos_phase, norm_factor * sin_phase);
}

/// Batch IQP encoding kernel (no entanglement)
///
/// Grid-stride loop processes multiple samples efficiently.
/// Memory layout:
/// - input_batch: [sample0_features | sample1_features | ...]
/// - state_batch: [sample0_state | sample1_state | ...]
///
/// TODO: For large num_features, consider shared memory caching of input
/// features per sample to reduce global memory traffic (potential 10-20x gain).
__global__ void iqp_encode_batch_kernel(
    const double* __restrict__ input_batch,
    cuDoubleComplex* __restrict__ state_batch,
    size_t num_samples,
    size_t num_features,
    size_t num_qubits,
    size_t state_len,
    double norm_factor
) {
    const size_t total_elements = num_samples * state_len;
    const size_t stride = gridDim.x * blockDim.x;
    // Precompute mask for bit operations (state_len = 2^num_qubits)
    const size_t state_mask = state_len - 1;

    for (size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
         global_idx < total_elements;
         global_idx += stride) {

        // Use bit operations instead of div/mod (state_len is power of 2)
        const size_t sample_idx = global_idx >> num_qubits;
        const size_t basis_idx = global_idx & state_mask;

        const double* input = input_batch + sample_idx * num_features;

        // Compute phase for this basis state
        double phase = 0.0;
        size_t z = basis_idx;
        for (size_t i = 0; i < num_features && i < num_qubits; ++i) {
            if (z & 1) {
                phase += __ldg(&input[i]);
            }
            z >>= 1;
        }

        double cos_phase, sin_phase;
        sincos(phase, &sin_phase, &cos_phase);

        state_batch[global_idx] = make_cuDoubleComplex(
            norm_factor * cos_phase,
            norm_factor * sin_phase
        );
    }
}

/// Batch IQP encoding kernel with linear entanglement
__global__ void iqp_encode_batch_linear_kernel(
    const double* __restrict__ input_batch,
    cuDoubleComplex* __restrict__ state_batch,
    size_t num_samples,
    size_t num_features,
    size_t num_qubits,
    size_t state_len,
    double norm_factor
) {
    const size_t total_elements = num_samples * state_len;
    const size_t stride = gridDim.x * blockDim.x;
    const size_t state_mask = state_len - 1;

    for (size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
         global_idx < total_elements;
         global_idx += stride) {

        // Use bit operations instead of div/mod
        const size_t sample_idx = global_idx >> num_qubits;
        const size_t basis_idx = global_idx & state_mask;

        const double* input = input_batch + sample_idx * num_features;

        double phase = 0.0;

        // Single-qubit terms
        size_t z = basis_idx;
        for (size_t i = 0; i < num_features && i < num_qubits; ++i) {
            if (z & 1) {
                phase += __ldg(&input[i]);
            }
            z >>= 1;
        }

        // Linear entanglement
        for (size_t i = 0; i + 1 < num_features && i + 1 < num_qubits; ++i) {
            size_t bit_i = (basis_idx >> i) & 1;
            size_t bit_i1 = (basis_idx >> (i + 1)) & 1;
            if (bit_i && bit_i1) {
                phase += __ldg(&input[i]) * __ldg(&input[i + 1]);
            }
        }

        double cos_phase, sin_phase;
        sincos(phase, &sin_phase, &cos_phase);

        state_batch[global_idx] = make_cuDoubleComplex(
            norm_factor * cos_phase,
            norm_factor * sin_phase
        );
    }
}

/// Batch IQP encoding kernel with full entanglement
__global__ void iqp_encode_batch_full_kernel(
    const double* __restrict__ input_batch,
    cuDoubleComplex* __restrict__ state_batch,
    size_t num_samples,
    size_t num_features,
    size_t num_qubits,
    size_t state_len,
    double norm_factor
) {
    const size_t total_elements = num_samples * state_len;
    const size_t stride = gridDim.x * blockDim.x;
    const size_t state_mask = state_len - 1;

    for (size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
         global_idx < total_elements;
         global_idx += stride) {

        // Use bit operations instead of div/mod
        const size_t sample_idx = global_idx >> num_qubits;
        const size_t basis_idx = global_idx & state_mask;

        const double* input = input_batch + sample_idx * num_features;

        double phase = 0.0;

        // Single-qubit terms
        size_t z = basis_idx;
        for (size_t i = 0; i < num_features && i < num_qubits; ++i) {
            if (z & 1) {
                phase += __ldg(&input[i]);
            }
            z >>= 1;
        }

        // Full entanglement
        size_t effective_n = (num_features < num_qubits) ? num_features : num_qubits;
        for (size_t i = 0; i < effective_n; ++i) {
            size_t bit_i = (basis_idx >> i) & 1;
            if (!bit_i) continue;

            for (size_t j = i + 1; j < effective_n; ++j) {
                size_t bit_j = (basis_idx >> j) & 1;
                if (bit_j) {
                    phase += __ldg(&input[i]) * __ldg(&input[j]);
                }
            }
        }

        double cos_phase, sin_phase;
        sincos(phase, &sin_phase, &cos_phase);

        state_batch[global_idx] = make_cuDoubleComplex(
            norm_factor * cos_phase,
            norm_factor * sin_phase
        );
    }
}

extern "C" {

/// Entanglement type enum (must match Rust side)
/// 0 = None (single-qubit rotations only)
/// 1 = Linear (nearest-neighbor)
/// 2 = Full (all-pairs)
enum IqpEntanglement {
    IQP_NONE = 0,
    IQP_LINEAR = 1,
    IQP_FULL = 2
};

/// Launch IQP encoding kernel
///
/// # Arguments
/// * input_d - Device pointer to input features
/// * state_d - Device pointer to output state vector
/// * num_features - Number of input features
/// * num_qubits - Number of qubits (determines state_len = 2^num_qubits)
/// * state_len - State vector size (2^num_qubits)
/// * entanglement - Entanglement type (0=none, 1=linear, 2=full)
/// * stream - CUDA stream for async execution (nullptr = default stream)
///
/// # Returns
/// CUDA error code (0 = cudaSuccess)
int launch_iqp_encode(
    const double* input_d,
    void* state_d,
    size_t num_features,
    size_t num_qubits,
    size_t state_len,
    int entanglement,
    cudaStream_t stream
) {
    // Null pointer validation
    if (input_d == nullptr || state_d == nullptr) {
        return cudaErrorInvalidDevicePointer;
    }

    if (state_len == 0 || num_qubits == 0) {
        return cudaErrorInvalidValue;
    }

    // Prevent shift overflow (num_qubits > 63 causes UB in 1ULL << num_qubits)
    // Limit to 30 for practical memory constraints (2^30 = 1B elements)
    if (num_qubits > 30) {
        return cudaErrorInvalidValue;
    }

    // Verify state_len = 2^num_qubits
    if (state_len != (1ULL << num_qubits)) {
        return cudaErrorInvalidValue;
    }

    if (num_features == 0) {
        return cudaErrorInvalidValue;
    }

    cuDoubleComplex* state_complex_d = static_cast<cuDoubleComplex*>(state_d);

    // Normalization factor: 1/√(2^n)
    const double norm_factor = 1.0 / sqrt((double)state_len);

    const int blockSize = 256;
    const size_t gridSize = (state_len + blockSize - 1) / blockSize;

    switch (entanglement) {
        case IQP_NONE:
            iqp_encode_kernel<<<gridSize, blockSize, 0, stream>>>(
                input_d, state_complex_d, num_features, num_qubits, state_len, norm_factor
            );
            break;
        case IQP_LINEAR:
            iqp_encode_linear_kernel<<<gridSize, blockSize, 0, stream>>>(
                input_d, state_complex_d, num_features, num_qubits, state_len, norm_factor
            );
            break;
        case IQP_FULL:
            iqp_encode_full_kernel<<<gridSize, blockSize, 0, stream>>>(
                input_d, state_complex_d, num_features, num_qubits, state_len, norm_factor
            );
            break;
        default:
            return cudaErrorInvalidValue;
    }

    return (int)cudaGetLastError();
}

/// Launch batch IQP encoding kernel
///
/// # Arguments
/// * input_batch_d - Device pointer to batch input features
/// * state_batch_d - Device pointer to output batch state vectors
/// * num_samples - Number of samples in batch
/// * num_features - Features per sample
/// * num_qubits - Number of qubits
/// * state_len - State vector size per sample (2^num_qubits)
/// * entanglement - Entanglement type (0=none, 1=linear, 2=full)
/// * stream - CUDA stream for async execution
///
/// # Returns
/// CUDA error code (0 = cudaSuccess)
int launch_iqp_encode_batch(
    const double* input_batch_d,
    void* state_batch_d,
    size_t num_samples,
    size_t num_features,
    size_t num_qubits,
    size_t state_len,
    int entanglement,
    cudaStream_t stream
) {
    // Null pointer validation
    if (input_batch_d == nullptr || state_batch_d == nullptr) {
        return cudaErrorInvalidDevicePointer;
    }

    if (num_samples == 0 || state_len == 0 || num_qubits == 0) {
        return cudaErrorInvalidValue;
    }

    // Prevent shift overflow and memory explosion
    if (num_qubits > 30) {
        return cudaErrorInvalidValue;
    }

    if (state_len != (1ULL << num_qubits)) {
        return cudaErrorInvalidValue;
    }

    if (num_features == 0) {
        return cudaErrorInvalidValue;
    }

    // Check for integer overflow in total_elements calculation
    if (num_samples > SIZE_MAX / state_len) {
        return cudaErrorInvalidValue;
    }

    cuDoubleComplex* state_complex_d = static_cast<cuDoubleComplex*>(state_batch_d);

    const double norm_factor = 1.0 / sqrt((double)state_len);

    const int blockSize = 256;
    const size_t total_elements = num_samples * state_len;
    const size_t blocks_needed = (total_elements + blockSize - 1) / blockSize;
    const size_t max_blocks = 2048;
    const size_t gridSize = (blocks_needed < max_blocks) ? blocks_needed : max_blocks;

    switch (entanglement) {
        case IQP_NONE:
            iqp_encode_batch_kernel<<<gridSize, blockSize, 0, stream>>>(
                input_batch_d, state_complex_d, num_samples, num_features,
                num_qubits, state_len, norm_factor
            );
            break;
        case IQP_LINEAR:
            iqp_encode_batch_linear_kernel<<<gridSize, blockSize, 0, stream>>>(
                input_batch_d, state_complex_d, num_samples, num_features,
                num_qubits, state_len, norm_factor
            );
            break;
        case IQP_FULL:
            iqp_encode_batch_full_kernel<<<gridSize, blockSize, 0, stream>>>(
                input_batch_d, state_complex_d, num_samples, num_features,
                num_qubits, state_len, norm_factor
            );
            break;
        default:
            return cudaErrorInvalidValue;
    }

    return (int)cudaGetLastError();
}

} // extern "C"
