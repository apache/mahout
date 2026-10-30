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
#include "kernel_config.h"

extern "C" __global__ void angle_encode_kernel(
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

extern "C" __global__ void angle_encode_kernel_f32(
    const float* __restrict__ angles,
    cuComplex* __restrict__ state,
    size_t state_len,
    unsigned int num_qubits
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_len) return;

    float amplitude = 1.0f;
    for (unsigned int bit = 0; bit < num_qubits; ++bit) {
        float angle = angles[bit];
        amplitude *= ((idx >> bit) & 1U) ? sinf(angle) : cosf(angle);
    }

    state[idx] = make_cuComplex(amplitude, 0.0f);
}

extern "C" __global__ void angle_encode_batch_kernel(
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

extern "C" __global__ void angle_encode_batch_kernel_f32(
    const float* __restrict__ angles_batch,
    cuComplex* __restrict__ state_batch,
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
        const float* angles = angles_batch + sample_idx * num_qubits;

        float amplitude = 1.0f;
        for (unsigned int bit = 0; bit < num_qubits; ++bit) {
            const float angle = angles[bit];
            amplitude *= ((element_idx >> bit) & 1U) ? sinf(angle) : cosf(angle);
        }

        state_batch[global_idx] = make_cuComplex(amplitude, 0.0f);
    }
}
