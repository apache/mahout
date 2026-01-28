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
#include "kernel_config.h"

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

// ============================================================================
// Naive O(4^n) Implementation (kept as fallback for small n and verification)
// ============================================================================

__global__ void iqp_encode_kernel_naive(
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


// ============================================================================
// FWT O(n * 2^n) Implementation
// ============================================================================

// Step 1: Compute f[x] = exp(i*theta(x)) for all x
// One thread per state, reuses existing compute_phase()
__global__ void iqp_phase_kernel(
    const double* __restrict__ data,
    cuDoubleComplex* __restrict__ state,
    size_t state_len,
    unsigned int num_qubits,
    int enable_zz
) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= state_len) return;

    double phase = compute_phase(data, x, num_qubits, enable_zz);

    double cos_phase, sin_phase;
    sincos(phase, &sin_phase, &cos_phase);
    state[x] = make_cuDoubleComplex(cos_phase, sin_phase);
}

// Step 2a: FWT butterfly stage for global memory (n > threshold)
// Each thread handles one butterfly pair per stage
// Walsh-Hadamard butterfly: (a, b) -> (a + b, a - b)
__global__ void fwt_butterfly_stage_kernel(
    cuDoubleComplex* __restrict__ state,
    size_t state_len,
    unsigned int stage  // 0 to n-1
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread processes one butterfly pair
    // For stage s, butterflies are separated by 2^s
    size_t stride = 1ULL << stage;
    size_t block_size = stride << 1;  // 2^(s+1)
    size_t num_pairs = state_len >> 1;  // state_len / 2 total pairs

    if (idx >= num_pairs) return;

    // Compute which butterfly pair this thread handles
    size_t block_idx = idx / stride;
    size_t pair_offset = idx % stride;
    size_t i = block_idx * block_size + pair_offset;
    size_t j = i + stride;

    // Load values
    cuDoubleComplex a = state[i];
    cuDoubleComplex b = state[j];

    // Butterfly: (a, b) -> (a + b, a - b)
    state[i] = cuCadd(a, b);
    state[j] = cuCsub(a, b);
}

// Step 2b: FWT using shared memory (n <= threshold)
// All stages in single kernel launch
__global__ void fwt_shared_memory_kernel(
    cuDoubleComplex* __restrict__ state,
    size_t state_len,
    unsigned int num_qubits
) {
    extern __shared__ cuDoubleComplex shared_state[];

    size_t tid = threadIdx.x;
    size_t bid = blockIdx.x;

    // For shared memory FWT, we process the entire state in one block
    // Block 0 handles the full transform
    if (bid > 0) return;

    // Load state into shared memory
    for (size_t i = tid; i < state_len; i += blockDim.x) {
        shared_state[i] = state[i];
    }
    __syncthreads();

    // Perform all FWT stages in shared memory
    for (unsigned int stage = 0; stage < num_qubits; ++stage) {
        size_t stride = 1ULL << stage;
        size_t block_size = stride << 1;
        size_t num_pairs = state_len >> 1;

        // Each thread handles multiple pairs if needed
        for (size_t pair_idx = tid; pair_idx < num_pairs; pair_idx += blockDim.x) {
            size_t block_idx = pair_idx / stride;
            size_t pair_offset = pair_idx % stride;
            size_t i = block_idx * block_size + pair_offset;
            size_t j = i + stride;

            cuDoubleComplex a = shared_state[i];
            cuDoubleComplex b = shared_state[j];

            shared_state[i] = cuCadd(a, b);
            shared_state[j] = cuCsub(a, b);
        }
        __syncthreads();
    }

    // Write back to global memory
    for (size_t i = tid; i < state_len; i += blockDim.x) {
        state[i] = shared_state[i];
    }
}

// Step 3: Normalize the state by 1/state_len (= 1/2^n)
__global__ void normalize_state_kernel(
    cuDoubleComplex* __restrict__ state,
    size_t state_len,
    double norm_factor
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_len) return;

    cuDoubleComplex val = state[idx];
    state[idx] = make_cuDoubleComplex(
        cuCreal(val) * norm_factor,
        cuCimag(val) * norm_factor
    );
}

// ============================================================================
// Naive O(4^n) Batch Implementation (kept as fallback)
// ============================================================================

__global__ void iqp_encode_batch_kernel_naive(
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


// ============================================================================
// FWT O(n * 2^n) Batch Implementation
// ============================================================================

// Step 1: Compute f[x] = exp(i*theta(x)) for all x, for all samples in batch
__global__ void iqp_phase_batch_kernel(
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
        const size_t x = global_idx & state_mask;
        const double* data = data_batch + sample_idx * data_len;

        double phase = compute_phase(data, x, num_qubits, enable_zz);

        double cos_phase, sin_phase;
        sincos(phase, &sin_phase, &cos_phase);
        state_batch[global_idx] = make_cuDoubleComplex(cos_phase, sin_phase);
    }
}

// Step 2: FWT butterfly stage for batch (global memory)
// Processes all samples in parallel
__global__ void fwt_butterfly_batch_kernel(
    cuDoubleComplex* __restrict__ state_batch,
    size_t num_samples,
    size_t state_len,
    unsigned int num_qubits,
    unsigned int stage
) {
    const size_t pairs_per_sample = state_len >> 1;
    const size_t total_pairs = num_samples * pairs_per_sample;
    const size_t grid_stride = gridDim.x * blockDim.x;

    // For stage s, butterflies are separated by 2^s
    const size_t stride = 1ULL << stage;
    const size_t block_size = stride << 1;

    for (size_t global_pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
         global_pair_idx < total_pairs;
         global_pair_idx += grid_stride) {

        // Determine which sample and which pair within that sample
        const size_t sample_idx = global_pair_idx / pairs_per_sample;
        const size_t pair_idx = global_pair_idx % pairs_per_sample;

        // Compute indices within this sample's state
        const size_t block_idx = pair_idx / stride;
        const size_t pair_offset = pair_idx % stride;
        const size_t local_i = block_idx * block_size + pair_offset;
        const size_t local_j = local_i + stride;

        // Global indices
        const size_t base = sample_idx * state_len;
        const size_t i = base + local_i;
        const size_t j = base + local_j;

        // Load values
        cuDoubleComplex a = state_batch[i];
        cuDoubleComplex b = state_batch[j];

        // Butterfly: (a, b) -> (a + b, a - b)
        state_batch[i] = cuCadd(a, b);
        state_batch[j] = cuCsub(a, b);
    }
}

// Step 3: Normalize all samples in batch
__global__ void normalize_batch_kernel(
    cuDoubleComplex* __restrict__ state_batch,
    size_t total_elements,
    double norm_factor
) {
    const size_t stride = gridDim.x * blockDim.x;

    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_elements;
         idx += stride) {
        cuDoubleComplex val = state_batch[idx];
        state_batch[idx] = make_cuDoubleComplex(
            cuCreal(val) * norm_factor,
            cuCimag(val) * norm_factor
        );
    }
}

extern "C" {

/// Launch IQP encoding kernel using FWT optimization
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
///
/// # Algorithm
/// For num_qubits >= FWT_MIN_QUBITS, uses Fast Walsh-Hadamard Transform:
///   1. Phase computation: f[x] = exp(i*theta(x)) - O(2^n)
///   2. FWT transform: WHT of phase array - O(n * 2^n)
///   3. Normalization: divide by 2^n - O(2^n)
/// Total: O(n * 2^n) vs naive O(4^n)
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
    const int blockSize = DEFAULT_BLOCK_SIZE;

    // Use naive kernel for small n (FWT overhead not worth it)
    if (num_qubits < FWT_MIN_QUBITS) {
        const int gridSize = (state_len + blockSize - 1) / blockSize;
        iqp_encode_kernel_naive<<<gridSize, blockSize, 0, stream>>>(
            data_d,
            state_complex_d,
            state_len,
            num_qubits,
            enable_zz
        );
        return (int)cudaGetLastError();
    }

    // FWT-based implementation for larger n
    const int gridSize = (state_len + blockSize - 1) / blockSize;

    // Step 1: Compute phase array f[x] = exp(i*theta(x))
    iqp_phase_kernel<<<gridSize, blockSize, 0, stream>>>(
        data_d,
        state_complex_d,
        state_len,
        num_qubits,
        enable_zz
    );

    // Step 2: Apply FWT
    if (num_qubits <= FWT_SHARED_MEM_THRESHOLD) {
        // Shared memory FWT - all stages in one kernel
        size_t shared_mem_size = state_len * sizeof(cuDoubleComplex);
        fwt_shared_memory_kernel<<<1, blockSize, shared_mem_size, stream>>>(
            state_complex_d,
            state_len,
            num_qubits
        );
    } else {
        // Global memory FWT - one kernel launch per stage
        const size_t num_pairs = state_len >> 1;
        const int fwt_grid_size = (num_pairs + blockSize - 1) / blockSize;

        for (unsigned int stage = 0; stage < num_qubits; ++stage) {
            fwt_butterfly_stage_kernel<<<fwt_grid_size, blockSize, 0, stream>>>(
                state_complex_d,
                state_len,
                stage
            );
        }
    }

    // Step 3: Normalize by 1/2^n
    double norm_factor = 1.0 / (double)state_len;
    normalize_state_kernel<<<gridSize, blockSize, 0, stream>>>(
        state_complex_d,
        state_len,
        norm_factor
    );

    return (int)cudaGetLastError();
}

/// Launch batch IQP encoding kernel using FWT optimization
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
///
/// # Algorithm
/// For num_qubits >= FWT_MIN_QUBITS, uses Fast Walsh-Hadamard Transform:
///   1. Phase computation for all samples - O(batch * 2^n)
///   2. FWT transform for all samples - O(batch * n * 2^n)
///   3. Normalization - O(batch * 2^n)
/// Total: O(batch * n * 2^n) vs naive O(batch * 4^n)
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
    const int blockSize = DEFAULT_BLOCK_SIZE;
    const size_t total_elements = num_samples * state_len;
    const size_t blocks_needed = (total_elements + blockSize - 1) / blockSize;
    const size_t gridSize = (blocks_needed < MAX_GRID_BLOCKS) ? blocks_needed : MAX_GRID_BLOCKS;

    // Use naive kernel for small n (FWT overhead not worth it)
    if (num_qubits < FWT_MIN_QUBITS) {
        iqp_encode_batch_kernel_naive<<<gridSize, blockSize, 0, stream>>>(
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

    // FWT-based implementation for larger n

    // Step 1: Compute phase array f[x] = exp(i*theta(x)) for all samples
    iqp_phase_batch_kernel<<<gridSize, blockSize, 0, stream>>>(
        data_batch_d,
        state_complex_d,
        num_samples,
        state_len,
        num_qubits,
        data_len,
        enable_zz
    );

    // Step 2: Apply FWT to all samples (global memory version for batch)
    // For batch processing, we always use global memory FWT
    // (shared memory would require processing samples one at a time)
    const size_t total_pairs = num_samples * (state_len >> 1);
    const size_t fwt_blocks_needed = (total_pairs + blockSize - 1) / blockSize;
    const size_t fwt_grid_size = (fwt_blocks_needed < MAX_GRID_BLOCKS) ? fwt_blocks_needed : MAX_GRID_BLOCKS;

    for (unsigned int stage = 0; stage < num_qubits; ++stage) {
        fwt_butterfly_batch_kernel<<<fwt_grid_size, blockSize, 0, stream>>>(
            state_complex_d,
            num_samples,
            state_len,
            num_qubits,
            stage
        );
    }

    // Step 3: Normalize by 1/2^n
    double norm_factor = 1.0 / (double)state_len;
    normalize_batch_kernel<<<gridSize, blockSize, 0, stream>>>(
        state_complex_d,
        total_elements,
        norm_factor
    );

    return (int)cudaGetLastError();
}

} // extern "C"
