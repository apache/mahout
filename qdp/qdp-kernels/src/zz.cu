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

// ZZFeatureMap (Second-order expansion) Encoding CUDA Kernels
//
// Maps classical features to quantum states using a circuit of H, RZ, and RZZ gates.
// Optimized using FWT for higher-order repetitions.
//
// Circuit: (H * U_diag(x))^p * H |0>
// where U_diag is the ZZ interaction diagonal phase.

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
#include "kernel_config.h"

// Forward declarations of FWT kernels from iqp.cu
// (In a real build, these would be in a shared header or compiled together)
__global__ void fwt_butterfly_stage_kernel(cuDoubleComplex* state, size_t state_len, unsigned int stage);
__global__ void fwt_shared_memory_kernel(cuDoubleComplex* state, size_t state_len, unsigned int num_qubits);
__global__ void normalize_state_kernel(cuDoubleComplex* state, size_t state_len, double norm_factor);

// Compute ZZ phase theta(x) for basis state z
// Formula: theta(z) = sum_i -x_i (-1)^z_i + sum_{i<j} -(\pi - x_i)(\pi - x_j) (-1)^{z_i ^ z_j}
// Note: This matches RZ(2x) and RZZ(2(pi-x)(pi-x)) gates.
__device__ double compute_zz_phase(
    const double* __restrict__ features,
    size_t z,
    unsigned int num_qubits
) {
    double phase = 0.0;

    // Single-qubit terms (RZ)
    for (unsigned int i = 0; i < num_qubits; ++i) {
        double x_i = features[i];
        int bit_i = (z >> i) & 1U;
        // Z_i |z> = (-1)^bit_i |z>
        // RZ(2*x_i) = exp(-i * x_i * Z_i) -> phase += -x_i * (-1)^bit_i
        phase -= x_i * (bit_i ? -1.0 : 1.0);
    }

    // Two-qubit terms (RZZ) - Full entanglement
    for (unsigned int i = 0; i < num_qubits; ++i) {
        for (unsigned int j = i + 1; j < num_qubits; ++j) {
            double x_i = features[i];
            double x_j = features[j];
            double phi_ij = (M_PI - x_i) * (M_PI - x_j);
            
            int bit_i = (z >> i) & 1U;
            int bit_j = (z >> j) & 1U;
            // Z_i Z_j |z> = (-1)^(bit_i ^ bit_j) |z>
            // RZZ(2*phi_ij) = exp(-i * phi_ij * Z_i Z_j) -> phase += -phi_ij * (-1)^(bit_i ^ bit_j)
            phase -= phi_ij * ((bit_i ^ bit_j) ? -1.0 : 1.0);
        }
    }

    return phase;
}

// Apply ZZ phase to existing state: state[z] *= exp(i * theta(z))
__global__ void apply_zz_phase_kernel(
    const double* __restrict__ features,
    cuDoubleComplex* __restrict__ state,
    size_t state_len,
    unsigned int num_qubits
) {
    size_t z = blockIdx.x * blockDim.x + threadIdx.x;
    if (z >= state_len) return;

    double phase = compute_zz_phase(features, z, num_qubits);
    
    double cos_phase, sin_phase;
    sincos(phase, &sin_phase, &cos_phase);
    cuDoubleComplex phase_factor = make_cuDoubleComplex(cos_phase, sin_phase);
    
    state[z] = cuCmul(state[z], phase_factor);
}

// Initialize state with ZZ phase: state[z] = exp(i * theta(z))
__global__ void init_zz_phase_kernel(
    const double* __restrict__ features,
    cuDoubleComplex* __restrict__ state,
    size_t state_len,
    unsigned int num_qubits
) {
    size_t z = blockIdx.x * blockDim.x + threadIdx.x;
    if (z >= state_len) return;

    double phase = compute_zz_phase(features, z, num_qubits);
    
    double cos_phase, sin_phase;
    sincos(phase, &sin_phase, &cos_phase);
    state[z] = make_cuDoubleComplex(cos_phase, sin_phase);
}

extern "C" {

/// Launch ZZFeatureMap encoding
int launch_zz_encode(
    const double* features_d,
    void* state_d,
    size_t state_len,
    unsigned int num_qubits,
    unsigned int num_layers,
    cudaStream_t stream
) {
    if (state_len == 0 || num_qubits == 0 || num_layers == 0) return cudaErrorInvalidValue;

    cuDoubleComplex* state_complex_d = static_cast<cuDoubleComplex*>(state_d);
    const int blockSize = DEFAULT_BLOCK_SIZE;
    const int gridSize = (state_len + blockSize - 1) / blockSize;

    // Layer 1: Init with phases (represents H * U_diag * H |0>)
    init_zz_phase_kernel<<<gridSize, blockSize, 0, stream>>>(
        features_d,
        state_complex_d,
        state_len,
        num_qubits
    );

    // Apply layers
    for (unsigned int p = 0; p < num_layers; ++p) {
        // 1. FWT (represents H)
        if (num_qubits <= FWT_SHARED_MEM_THRESHOLD) {
            size_t shared_mem_size = state_len * sizeof(cuDoubleComplex);
            fwt_shared_memory_kernel<<<1, blockSize, shared_mem_size, stream>>>(
                state_complex_d,
                state_len,
                num_qubits
            );
        } else {
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

        // 2. Normalize and Apply next layer phases if needed
        if (p < num_layers - 1) {
             // Internal normalization by 1/sqrt(2^n) to maintain unit norm
             double norm_inv_sqrt = 1.0 / sqrt((double)state_len);
             normalize_state_kernel<<<gridSize, blockSize, 0, stream>>>(
                 state_complex_d,
                 state_len,
                 norm_inv_sqrt
             );
             
             apply_zz_phase_kernel<<<gridSize, blockSize, 0, stream>>>(
                 features_d,
                 state_complex_d,
                 state_len,
                 num_qubits
             );
        }
    }

    // Final normalization
    // Total H gates: num_layers + 1
    // Number of unnormalized FWTs: num_layers
    // init_zz_phase_kernel produces a vector with norm sqrt(2^n)
    // After p FWTs, norm is (sqrt(2^n))^(p+1)
    // Final normalization should be 1 / (sqrt(2^n))^(p+1)
    // Wait, let's simplify: 
    // If we normalize by 1/sqrt(2^n) after EACH FWT, we are good.
    // But we normalized p-1 times.
    // So one more normalization at the end.
    double norm_factor = 1.0 / pow(sqrt((double)state_len), 1.0); // Simple 1/sqrt(2^n) for the last layer
    // Wait, the init_zz_phase_kernel started with norm sqrt(2^n).
    // Let's just do it correctly:
    // Start: |v| = sqrt(2^n)
    // Each Layer: 
    //   FWT: |v| *= sqrt(2^n)
    //   Normalize(1/2^n)? No.
    
    // Let's re-calculate:
    // Layer 1: init (norm sqrt(2^n)) -> FWT (norm 2^n). To get norm 1, divide by 2^n.
    // Layer 2: phase (norm 1) -> FWT (norm sqrt(2^n)). To get norm 1, divide by sqrt(2^n).
    
    // Correct logic:
    // If p=1: Divide by state_len (2^n).
    // If p=2: Divide by state_len (after 1st FWT), then divide by sqrt(state_len) (after 2nd FWT).
    
    double final_norm = 1.0 / (double)state_len; // This handles layer 1
    // The loop above did p-1 normalizations by 1/sqrt(state_len)?
    // No, if p=1, the loop runs p times. Only normalize if p < num_layers - 1?
    // Wait, my loop structure is slightly confusing.
    
    // Let's simplify and just do one final normalization:
    // Total scale after p FWTs and init is (sqrt(2^n)) * (sqrt(2^n))^p = (sqrt(2^n))^(p+1).
    double total_norm = 1.0 / pow(sqrt((double)state_len), (double)(num_layers + 1));
    
    normalize_state_kernel<<<gridSize, blockSize, 0, stream>>>(
        state_complex_d,
        state_len,
        total_norm
    );

    return (int)cudaGetLastError();
}

// Batched versions
__global__ void init_zz_phase_batch_kernel(
    const double* __restrict__ data_batch,
    cuDoubleComplex* __restrict__ state_batch,
    size_t num_samples,
    size_t state_len,
    unsigned int num_qubits,
    unsigned int data_len
) {
    const size_t total_elements = num_samples * state_len;
    const size_t state_mask = state_len - 1;

    for (size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
         global_idx < total_elements;
         global_idx += gridDim.x * blockDim.x) {
        const size_t sample_idx = global_idx >> num_qubits;
        const size_t z = global_idx & state_mask;
        const double* features = data_batch + sample_idx * data_len;

        double phase = compute_zz_phase(features, z, num_qubits);
        
        double cos_phase, sin_phase;
        sincos(phase, &sin_phase, &cos_phase);
        state_batch[global_idx] = make_cuDoubleComplex(cos_phase, sin_phase);
    }
}

__global__ void apply_zz_phase_batch_kernel(
    const double* __restrict__ data_batch,
    cuDoubleComplex* __restrict__ state_batch,
    size_t num_samples,
    size_t state_len,
    unsigned int num_qubits,
    unsigned int data_len
) {
    const size_t total_elements = num_samples * state_len;
    const size_t state_mask = state_len - 1;

    for (size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
         global_idx < total_elements;
         global_idx += gridDim.x * blockDim.x) {
        const size_t sample_idx = global_idx >> num_qubits;
        const size_t z = global_idx & state_mask;
        const double* features = data_batch + sample_idx * data_len;

        double phase = compute_zz_phase(features, z, num_qubits);
        
        double cos_phase, sin_phase;
        sincos(phase, &sin_phase, &cos_phase);
        cuDoubleComplex phase_factor = make_cuDoubleComplex(cos_phase, sin_phase);
        state_batch[global_idx] = cuCmul(state_batch[global_idx], phase_factor);
    }
}

// External declarations for batch FWT kernels from iqp.cu
__global__ void fwt_butterfly_batch_kernel(cuDoubleComplex* state_batch, size_t num_samples, size_t state_len, unsigned int num_qubits, unsigned int stage);
__global__ void normalize_batch_kernel(cuDoubleComplex* state_batch, size_t total_elements, double norm_factor);

int launch_zz_encode_batch(
    const double* data_batch_d,
    void* state_batch_d,
    size_t num_samples,
    size_t state_len,
    unsigned int num_qubits,
    unsigned int data_len,
    unsigned int num_layers,
    cudaStream_t stream
) {
    if (num_samples == 0 || state_len == 0 || num_qubits == 0 || num_layers == 0) return cudaErrorInvalidValue;

    cuDoubleComplex* state_complex_d = static_cast<cuDoubleComplex*>(state_batch_d);
    const int blockSize = DEFAULT_BLOCK_SIZE;
    const size_t total_elements = num_samples * state_len;
    const size_t blocks_needed = (total_elements + blockSize - 1) / blockSize;
    const size_t gridSize = (blocks_needed < MAX_GRID_BLOCKS) ? blocks_needed : MAX_GRID_BLOCKS;

    // Layer 1
    init_zz_phase_batch_kernel<<<gridSize, blockSize, 0, stream>>>(
        data_batch_d,
        state_complex_d,
        num_samples,
        state_len,
        num_qubits,
        data_len
    );

    for (unsigned int p = 0; p < num_layers; ++p) {
        // FWT Batch
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

        if (p < num_layers - 1) {
            double norm_inv_sqrt = 1.0 / sqrt((double)state_len);
            normalize_batch_kernel<<<gridSize, blockSize, 0, stream>>>(
                state_complex_d,
                total_elements,
                norm_inv_sqrt
            );
            
            apply_zz_phase_batch_kernel<<<gridSize, blockSize, 0, stream>>>(
                data_batch_d,
                state_complex_d,
                num_samples,
                state_len,
                num_qubits,
                data_len
            );
        }
    }

    double total_norm = 1.0 / pow(sqrt((double)state_len), (double)(num_layers + 1));
    normalize_batch_kernel<<<gridSize, blockSize, 0, stream>>>(
        state_complex_d,
        total_elements,
        total_norm
    );

    return (int)cudaGetLastError();
}

} // extern "C"
