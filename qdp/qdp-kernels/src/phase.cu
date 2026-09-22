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

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

#include "kernel_config.h"

extern "C" __global__ void phase_encode_kernel(
    const double* __restrict__ phases,
    cuDoubleComplex* __restrict__ state,
    size_t state_len,
    unsigned int num_qubits,
    double norm_factor
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_len) return;

    // φ(idx) = Σ_k phases[k] * b_k,  b_k = (idx >> k) & 1
    double phi = 0.0;
    double norm = 1.0;
    for (unsigned int bit = 0; bit < num_qubits; ++bit) {
        phi += phases[bit] * (double)((idx >> bit) & 1U);
        norm *= M_SQRT1_2;
    }

    double re, im;
    sincos(phi, &im, &re);   // re = cos(φ), im = sin(φ)

    state[idx] = make_cuDoubleComplex(norm_factor * re, norm_factor * im);
}

extern "C" __global__ void phase_encode_batch_kernel(
    const double* __restrict__ phases_batch,
    cuDoubleComplex* __restrict__ state_batch,
    size_t num_samples,
    size_t state_len,
    unsigned int num_qubits,
    double norm_factor
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
            phi += phases[bit] * (double)((element_idx >> bit) & 1U);
        }

        double re, im;
        sincos(phi, &im, &re);

        state_batch[global_idx] = make_cuDoubleComplex(norm_factor * re, norm_factor * im);
    }
}
