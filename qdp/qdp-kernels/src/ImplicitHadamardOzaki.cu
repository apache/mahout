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

#include "ImplicitHadamardOzaki.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

namespace implicit_ozaki_kernels {

__device__ __forceinline__ void mma_m16n8k32_s8(
    int32_t* d, const uint32_t* a, const uint32_t* b, const int32_t* c) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3])
    );
}

__device__ __forceinline__ void ldmatrix_x4_int8(uint32_t* d, void* smem_ptr) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(smem_addr));
}

__device__ __forceinline__ void ldmatrix_x2_int8(uint32_t* d, void* smem_ptr) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
        : "=r"(d[0]), "=r"(d[1]) : "r"(smem_addr));
}

__device__ __forceinline__ size_t get_A8_offset(int r, int c, int m, int k) {
    int tile_m = r / 128;
    int tile_k = c / 32;
    int num_tiles_k = (k + 31) / 32;
    int in_tile_r = r % 128;
    int in_tile_c = c % 32;
    return (size_t)(tile_m * num_tiles_k + tile_k) * 4096 + in_tile_r * 32 + in_tile_c;
}

__global__ void precompute_modulo_kernel_p26_implicit(const double* __restrict__ s, int8_t* __restrict__ d, int r, int c, int m, double sh, double sl) {
    int local_k = threadIdx.x; int local_m = threadIdx.y;
    int tile_k = blockIdx.x; int tile_m = blockIdx.y;
    const int pr[7] = {127, 113, 109, 107, 103, 101, 97};
    size_t padded_size = (size_t)((r + 127) / 128) * ((c + 31) / 32) * 4096;
    int num_tiles_k = (c + 31) / 32;
    int m_idx = (tile_m / 4) * 128 + (tile_m % 4) * 32 + local_m;
    int k_idx = tile_k * 32 + local_k;
    if (m_idx < r && k_idx < c) {
        double v = s[(size_t)m_idx * c + k_idx];
        int32_t iv = (v == 0.0) ? 0 : __double2int_rn(v * sh);
        size_t out_off = (size_t)( (m_idx / 128) * num_tiles_k + tile_k ) * 4096 + (m_idx % 128) * 32 + local_k;
        for (int p = 0; p < 7; p++) {
            int32_t rem = iv % pr[p];
            if (rem < 0) rem += pr[p];
            d[p * padded_size + out_off] = (int8_t)rem;
        }
    }
}

template <bool SingleBuffer>
__device__ void implicit_ozaki_process_one_tile(
    int ct,
    const int8_t* __restrict__ A8_h,
    double* __restrict__ C,
    int m, int n, int k,
    double inv,
    double norm_factor,
    int8_t* sA8,
    int8_t* sB8,
    const int8_t* h_pos,
    const int8_t* h_neg,
    int warp_id,
    int lane_id) {

    constexpr int kTileBytes = 2048;
    constexpr int kBufferCount = SingleBuffer ? 1 : 2;
    constexpr int kPrimeStride = kTileBytes * kBufferCount;

    const int pr[7] = {127, 113, 109, 107, 103, 101, 97};
    const uint64_t M = 168897325606883ULL;
    const uint64_t f[7] = {
        147618922380819ULL, 112099994871825ULL, 134807957135769ULL,
        34726552928518ULL, 96747011755399ULL, 130435558389474ULL,
        19153304965729ULL
    };

    const int total_tiles_m = (m + 63) / 64;
    const int tile_m = (ct % total_tiles_m) * 64;
    const int tile_n = (ct / total_tiles_m) * 64;

    int32_t prime_acc[7][4][8][4];
    for (int p = 0; p < 7; p++)
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 8; j++)
                for (int r = 0; r < 4; r++)
                    prime_acc[p][i][j][r] = 0;

    for (int kk = 0; kk < k; kk += 32) {
        const int b_idx = SingleBuffer ? 0 : ((kk / 32) & 1);
        const size_t padded_mk = (size_t)((m + 127) / 128) * ((k + 31) / 32) * 4096;

        for (int p = 0; p < 7; ++p) {
            const int8_t* Ap_global = A8_h + p * padded_mk;
            int8_t* sA_p = &sA8[p * kPrimeStride + b_idx * kTileBytes];
            int8_t* sB_p = &sB8[p * kPrimeStride + b_idx * kTileBytes];

            for (int i = threadIdx.x; i < 64 * 32; i += blockDim.x) {
                int r = i / 32, c = i % 32;
                sA_p[r * 32 + c] = (tile_m + r < m && kk + c < k)
                    ? Ap_global[get_A8_offset(tile_m + r, kk + c, m, k)] : 0;
            }
            for (int i = threadIdx.x; i < 32 * 64; i += blockDim.x) {
                int r = i / 64, c = i % 64;
                int8_t val = 0;
                if (kk + r < k && tile_n + c < n) {
                    int parity = __popcll((kk + r) & (tile_n + c)) & 1;
                    val = (parity == 0) ? h_pos[p] : h_neg[p];
                }
                sB_p[r * 64 + c] = val;
            }
        }
        __syncthreads();

        const int wr = warp_id / 4;
        const int wc = warp_id % 4;

        for (int p = 0; p < 7; ++p) {
            int8_t* sA_p = &sA8[p * 4096 + b_idx * 2048];
            int8_t* sB_p = &sB8[p * 4096 + b_idx * 2048];

            for (int mt = 0; mt < 2; mt++) {
                uint32_t af[4];
                int row_a = wr * 32 + mt * 16 + (lane_id % 16);
                int col_a = (lane_id / 16) * 16;
                ldmatrix_x4_int8(af, &sA_p[row_a * 32 + col_a]);

                uint32_t bf_all[4];
                int row_b = (lane_id % 16);
                int col_b = wc * 16;
                ldmatrix_x4_int8(bf_all, &sB_p[row_b * 64 + col_b]);

                for (int nt = 0; nt < 2; nt++) {
                    uint32_t bf[2];
                    bf[0] = bf_all[nt * 2];
                    bf[1] = bf_all[nt * 2 + 1];

                    mma_m16n8k32_s8(
                        prime_acc[p][wr * 2 + mt][wc * 2 + nt], af, bf,
                        prime_acc[p][wr * 2 + mt][wc * 2 + nt]);
                }
            }
        }
        __syncthreads();
    }

    const int wr = warp_id / 4;
    const int wc = warp_id % 4;
    for (int mt = 0; mt < 2; mt++) {
        for (int nt = 0; nt < 2; nt++) {
            uint64_t final_acc[4] = {0, 0, 0, 0};
            for (int p = 0; p < 7; ++p) {
                for (int r = 0; r < 4; r++) {
                    int32_t v = prime_acc[p][wr * 2 + mt][wc * 2 + nt][r];
                    uint32_t rem = (v % pr[p] + pr[p]) % pr[p];
                    final_acc[r] += (uint64_t)rem * f[p];
                }
            }

            int r_base = tile_m + wr * 32 + mt * 16 + (lane_id % 8);
            int c_base = tile_n + wc * 16 + nt * 8 + (lane_id / 8) * 2;

            auto store_res = [&](int r, int c, uint64_t cv) {
                if (r < m && c < n) {
                    double fv = (double)(cv % M);
                    if (cv % M > M / 2) fv -= (double)M;
                    C[(size_t)r * n + c] = fv * norm_factor * inv;
                }
            };

            store_res(r_base, c_base, final_acc[0]);
            store_res(r_base + 8, c_base, final_acc[1]);
            store_res(r_base, c_base + 1, final_acc[2]);
            store_res(r_base + 8, c_base + 1, final_acc[3]);
        }
    }
}

__device__ void implicit_ozaki_init_hadamard_signs(int8_t* h_pos, int8_t* h_neg) {
    const int pr[7] = {127, 113, 109, 107, 103, 101, 97};
    if (threadIdx.x == 0) {
        for (int p = 0; p < 7; p++) {
            h_pos[p] = 1 % pr[p];
            int rem_neg = (-1) % pr[p];
            if (rem_neg < 0) rem_neg += pr[p];
            h_neg[p] = rem_neg;
        }
    }
}

// NCU / WDDM: one tile per block, static shared memory (28 KiB), single-buffer MMA path.
__global__ void implicit_hadamard_ozaki_grid_kernel_implicit(
    const int8_t* __restrict__ A8_h,
    double* __restrict__ C, int m, int n, int k, double inv,
    double norm_factor) {

    constexpr int kS8Bytes = 7 * 2048;
    __shared__ alignas(16) int8_t shared_mem[2 * kS8Bytes];
    int8_t* sA8 = &shared_mem[0];
    int8_t* sB8 = &shared_mem[kS8Bytes];

    __shared__ int8_t h_pos[7];
    __shared__ int8_t h_neg[7];
    implicit_ozaki_init_hadamard_signs(h_pos, h_neg);
    __syncthreads();

    const int ct = blockIdx.x;
    const int total_tiles = ((m + 63) / 64) * ((n + 63) / 64);
    if (ct >= total_tiles) return;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    implicit_ozaki_process_one_tile<true>(
        ct, A8_h, C, m, n, k, inv, norm_factor,
        sA8, sB8, h_pos, h_neg, warp_id, lane_id);
}

template <bool SingleBuffer>
__global__ void implicit_hadamard_ozaki_persistent_kernel_implicit(
    const int8_t* __restrict__ A8_h,
    double* __restrict__ C, int m, int n, int k, double inv, int* __restrict__ d_work_queue,
    double norm_factor) {

    constexpr int kTileBytes = 2048;
    constexpr int kBufferCount = SingleBuffer ? 1 : 2;
    constexpr int kPrimeStride = kTileBytes * kBufferCount;
    constexpr int kS8Bytes = 7 * kPrimeStride;
    extern __shared__ int8_t shared_mem[];
    int8_t* sA8 = &shared_mem[0];
    int8_t* sB8 = &shared_mem[kS8Bytes];

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    __shared__ int tile_idx_shared;

    const int total_tiles = ((m + 63) / 64) * ((n + 63) / 64);

    __shared__ int8_t h_pos[7];
    __shared__ int8_t h_neg[7];
    implicit_ozaki_init_hadamard_signs(h_pos, h_neg);
    __syncthreads();

    while (true) {
        if (threadIdx.x == 0) tile_idx_shared = atomicAdd(d_work_queue, 1);
        __syncthreads();
        const int ct = tile_idx_shared;
        if (ct >= total_tiles) break;

        implicit_ozaki_process_one_tile<SingleBuffer>(
            ct, A8_h, C, m, n, k, inv, norm_factor,
            sA8, sB8, h_pos, h_neg, warp_id, lane_id);
        __syncthreads();
    }
}

} // namespace implicit_ozaki_kernels

namespace ozaki {

void ImplicitHadamardOzakiEngine::execute_implicit_hadamard(const double* d_A, double* d_C, int m, int n, int k, double norm_factor, cudaStream_t stream) {
    size_t padded_mk = (size_t)((m + 127) / 128) * ((k + 31) / 32) * 4096;

    int8_t *dA8_h = nullptr;
    int *d_queue = nullptr;

    cudaMalloc(&dA8_h, 7ULL * padded_mk);

    double scale_A = pow(2.0, 30.0);
    double inv_A = pow(2.0, -30.0);

    dim3 pre_block(32, 32);
    dim3 pre_grid_A((k + 31) / 32, ((m + 127) / 128) * 4);
    implicit_ozaki_kernels::precompute_modulo_kernel_p26_implicit<<<pre_grid_A, pre_block, 0, stream>>>(d_A, dA8_h, m, k, 0, scale_A, 0.0);

    const bool ncu_profile = (std::getenv("OZAKI_NCU_PROFILE") != nullptr);
    if (ncu_profile) {
        // Profiling path: grid kernel (one tile per block), single-buffer smem — NCU-friendly on WDDM.
        const int total_tiles = ((m + 63) / 64) * ((n + 63) / 64);
        implicit_ozaki_kernels::implicit_hadamard_ozaki_grid_kernel_implicit<<<total_tiles, 256, 0, stream>>>(
            dA8_h, d_C, m, n, k, inv_A, norm_factor);
    } else {
        int num_sms;
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
        cudaMalloc(&d_queue, sizeof(int));
        cudaMemsetAsync(d_queue, 0, sizeof(int), stream);
        cudaFuncSetAttribute(
            implicit_ozaki_kernels::implicit_hadamard_ozaki_persistent_kernel_implicit<false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            60 * 1024);
        implicit_ozaki_kernels::implicit_hadamard_ozaki_persistent_kernel_implicit<false><<<num_sms, 256, 58 * 1024, stream>>>(
            dA8_h, d_C, m, n, k, inv_A, d_queue, norm_factor);
    }

    cudaStreamSynchronize(stream);

    cudaFree(dA8_h);
    if (d_queue) cudaFree(d_queue);
}

} // namespace ozaki
