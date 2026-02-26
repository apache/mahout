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

// ZZFeatureMap Encoding CUDA Kernels
//
// Implements ZZFeatureMap using repeated H^n and diagonal phase layers with
// configurable entanglement patterns and repetition count.

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
#include "kernel_config.h"

#define ENTANGLEMENT_FULL 0
#define ENTANGLEMENT_LINEAR 1
#define ENTANGLEMENT_CIRCULAR 2

__device__ double compute_zzfeaturemap_phase(
    const double* __restrict__ layer_data,
    size_t x,
    unsigned int num_qubits,
    int entanglement_mode
) {
    double phase = 0.0;

    for (unsigned int i = 0; i < num_qubits; ++i) {
        if ((x >> i) & 1U) {
            phase += layer_data[i];
        }
    }

    unsigned int pair_idx = num_qubits;

    switch (entanglement_mode) {
        case ENTANGLEMENT_FULL:
            for (unsigned int i = 0; i < num_qubits; ++i) {
                for (unsigned int j = i + 1; j < num_qubits; ++j) {
                    if (((x >> i) & 1U) && ((x >> j) & 1U)) {
                        phase += layer_data[pair_idx];
                    }
                    pair_idx++;
                }
            }
            break;

        case ENTANGLEMENT_LINEAR:
            for (unsigned int i = 0; i + 1 < num_qubits; ++i) {
                if (((x >> i) & 1U) && ((x >> (i + 1)) & 1U)) {
                    phase += layer_data[pair_idx];
                }
                pair_idx++;
            }
            break;

        case ENTANGLEMENT_CIRCULAR:
            for (unsigned int i = 0; i + 1 < num_qubits; ++i) {
                if (((x >> i) & 1U) && ((x >> (i + 1)) & 1U)) {
                    phase += layer_data[pair_idx];
                }
                pair_idx++;
            }
            if (num_qubits > 1) {
                if (((x >> (num_qubits - 1)) & 1U) && ((x >> 0) & 1U)) {
                    phase += layer_data[pair_idx];
                }
            }
            break;
    }

    return phase;
}

__global__ void zzfeaturemap_init_state_kernel(
    cuDoubleComplex* __restrict__ state,
    size_t state_len
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_len) return;

    if (idx == 0) {
        state[idx] = make_cuDoubleComplex(1.0, 0.0);
    } else {
        state[idx] = make_cuDoubleComplex(0.0, 0.0);
    }
}

__global__ void zzfeaturemap_phase_kernel(
    const double* __restrict__ layer_data,
    cuDoubleComplex* __restrict__ state,
    size_t state_len,
    unsigned int num_qubits,
    int entanglement_mode
) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= state_len) return;

    double phase = compute_zzfeaturemap_phase(layer_data, x, num_qubits, entanglement_mode);

    double cos_phase, sin_phase;
    sincos(phase, &sin_phase, &cos_phase);
    cuDoubleComplex phase_factor = make_cuDoubleComplex(cos_phase, sin_phase);

    state[x] = cuCmul(state[x], phase_factor);
}

__global__ void zzfeaturemap_fwt_butterfly_kernel(
    cuDoubleComplex* __restrict__ state,
    size_t state_len,
    unsigned int stage
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    size_t stride = 1ULL << stage;
    size_t block_size = stride << 1;
    size_t num_pairs = state_len >> 1;

    if (idx >= num_pairs) return;

    size_t block_idx = idx / stride;
    size_t pair_offset = idx % stride;
    size_t i = block_idx * block_size + pair_offset;
    size_t j = i + stride;

    cuDoubleComplex a = state[i];
    cuDoubleComplex b = state[j];

    state[i] = cuCadd(a, b);
    state[j] = cuCsub(a, b);
}

__global__ void zzfeaturemap_normalize_kernel(
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

extern "C" {

int launch_zzfeaturemap_encode(
    const double* data_d,
    void* state_d,
    size_t state_len,
    unsigned int num_qubits,
    unsigned int reps,
    unsigned int params_per_layer,
    int entanglement_mode,
    cudaStream_t stream
) {
    if (state_len == 0 || num_qubits == 0 || reps == 0) {
        return cudaErrorInvalidValue;
    }

    cuDoubleComplex* state_complex_d = static_cast<cuDoubleComplex*>(state_d);
    const int blockSize = DEFAULT_BLOCK_SIZE;
    const int gridSize = (state_len + blockSize - 1) / blockSize;

    zzfeaturemap_init_state_kernel<<<gridSize, blockSize, 0, stream>>>(
        state_complex_d,
        state_len
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return (int)err;

    for (unsigned int layer = 0; layer < reps; ++layer) {
        const size_t num_pairs = state_len >> 1;
        const int fwt_grid_size = (num_pairs + blockSize - 1) / blockSize;

        for (unsigned int stage = 0; stage < num_qubits; ++stage) {
            zzfeaturemap_fwt_butterfly_kernel<<<fwt_grid_size, blockSize, 0, stream>>>(
                state_complex_d,
                state_len,
                stage
            );
        }

        double h_norm = 1.0 / sqrt((double)state_len);
        zzfeaturemap_normalize_kernel<<<gridSize, blockSize, 0, stream>>>(
            state_complex_d,
            state_len,
            h_norm
        );

        const double* layer_data = data_d + layer * params_per_layer;
        zzfeaturemap_phase_kernel<<<gridSize, blockSize, 0, stream>>>(
            layer_data,
            state_complex_d,
            state_len,
            num_qubits,
            entanglement_mode
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) return (int)err;
    }

    return (int)cudaSuccess;
}

} // extern "C"
