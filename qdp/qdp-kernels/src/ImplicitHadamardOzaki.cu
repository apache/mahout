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
    const double scale[7] = {
        64.0,
        14.0 * 64.0,
        10.0 * 64.0,
        9.0 * 64.0,
        8.0 * 64.0,
        8.0 * 64.0,
        7.0 * 64.0
    };
    size_t padded_size = (size_t)((r + 127) / 128) * ((c + 31) / 32) * 4096;
    int num_tiles_k = (c + 31) / 32;
    int m_idx = (tile_m / 4) * 128 + (tile_m % 4) * 32 + local_m;
    int k_idx = tile_k * 32 + local_k;
    if (m_idx < r && k_idx < c) {
        double v = s[(size_t)m_idx * c + k_idx];
        int32_t iv = (v == 0.0) ? 0 : __double2int_rn(v * sh);
        size_t out_off = (size_t)( (m_idx / 128) * num_tiles_k + tile_k ) * 4096 + (m_idx % 128) * 32 + local_k;
        for (int p = 0; p < 7; p++) {
            int32_t rem;
            if (p == 0) { rem = iv % 127; if (rem < 0) rem += 127; }
            else if (p == 1) { rem = iv % 113; if (rem < 0) rem += 113; }
            else if (p == 2) { rem = iv % 109; if (rem < 0) rem += 109; }
            else if (p == 3) { rem = iv % 107; if (rem < 0) rem += 107; }
            else if (p == 4) { rem = iv % 103; if (rem < 0) rem += 103; }
            else if (p == 5) { rem = iv % 101; if (rem < 0) rem += 101; }
            else { rem = iv % 97; if (rem < 0) rem += 97; }
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
    double cA[7], cB[7];
    #pragma unroll
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
            int8_t* sA_p = &sA8[p * kPrimeStride + b_idx * kTileBytes];
            int8_t* sB_p = &sB8[p * kPrimeStride + b_idx * kTileBytes];

            for (int mt = 0; mt < 2; mt++) {
                uint32_t af[4];
                int Row = wr * 32 + mt * 16 + (lane_id % 16);
                int Col_start = (lane_id / 16) * 16;
                uint4 af_val = *reinterpret_cast<uint4*>(&sA_p[Row * 32 + Col_start]);
                af[0] = af_val.x; af[1] = af_val.y; af[2] = af_val.z; af[3] = af_val.w;

                for (int nt = 0; nt < 2; nt++) {
                    uint32_t bf[2];
                    int col = wc * 16 + nt * 8 + (lane_id % 8);
                    int row_start = (lane_id / 8) * 8;
                    uint32_t b0 = 0, b1 = 0;
                    for (int j = 0; j < 4; j++) {
                        b0 |= ((uint32_t)(uint8_t)sB_p[(row_start + j) * 64 + col]) << (j * 8);
                        b1 |= ((uint32_t)(uint8_t)sB_p[(row_start + 4 + j) * 64 + col]) << (j * 8);
                    }
                    bf[0] = b0; bf[1] = b1;

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
            for (int r = 0; r < 4; r++) {
                int32_t v;
                uint32_t rem;

                v = prime_acc[0][wr * 2 + mt][wc * 2 + nt][r]; rem = (v % 127 + 127) % 127; final_acc[r] += (uint64_t)rem * 147618922380819ULL;
                v = prime_acc[1][wr * 2 + mt][wc * 2 + nt][r]; rem = (v % 113 + 113) % 113; final_acc[r] += (uint64_t)rem * 112099994871825ULL;
                v = prime_acc[2][wr * 2 + mt][wc * 2 + nt][r]; rem = (v % 109 + 109) % 109; final_acc[r] += (uint64_t)rem * 134807957135769ULL;
                v = prime_acc[3][wr * 2 + mt][wc * 2 + nt][r]; rem = (v % 107 + 107) % 107; final_acc[r] += (uint64_t)rem *  34726552928518ULL;
                v = prime_acc[4][wr * 2 + mt][wc * 2 + nt][r]; rem = (v % 103 + 103) % 103; final_acc[r] += (uint64_t)rem *  96747011755399ULL;
                v = prime_acc[5][wr * 2 + mt][wc * 2 + nt][r]; rem = (v % 101 + 101) % 101; final_acc[r] += (uint64_t)rem * 130435558389474ULL;
                v = prime_acc[6][wr * 2 + mt][wc * 2 + nt][r]; rem = (v % 97  + 97)  % 97;  final_acc[r] += (uint64_t)rem *  19153304965729ULL;
            }

            int r_base = tile_m + wr * 32 + mt * 16 + (lane_id / 4);
            int c_base = tile_n + wc * 16 + nt * 8 + (lane_id % 4) * 2;

            auto store_res = [&](int r, int c, uint64_t cv) {
                if (r < m && c < n) {
                    double fv = (double)(cv % M);
                    if (cv % M > M / 2) fv -= (double)M;
                    C[(size_t)r * n + c] = fv * norm_factor * inv;
                }
            };

            store_res(r_base, c_base, final_acc[0]);
            store_res(r_base, c_base + 1, final_acc[1]);
            store_res(r_base + 8, c_base, final_acc[2]);
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

    __shared__ alignas(16) int8_t h_pos[8];
    __shared__ alignas(16) int8_t h_neg[8];
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

__global__ void naive_hadamard_ozaki_kernel(const double* A, double* C, int m, int n, int k, double norm_factor) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < m && c < n) {
        double sum = 0.0;
        for (int kk = 0; kk < k; ++kk) {
            int parity = __popcll(kk & c) & 1;
            double H_val = (parity == 0) ? 1.0 : -1.0;
            sum += A[r * k + kk] * H_val;
        }
        C[r * n + c] = sum * norm_factor;
    }
}

template <int DIM, int THREADS, int WARPS_C>
__global__ void __launch_bounds__(THREADS, 3) implicit_hadamard_ozaki_fused_batch_kernel(
    const double* __restrict__ X,
    double*       C,
    int k, double inv, double norm_factor,
    bool transpose_batch, int batch_rows
) {
    const int tile_n = blockIdx.x * 64;
    const int tile_m = blockIdx.y * 64;

    const int wid  = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int wr   = wid / WARPS_C;
    const int wc   = wid % WARPS_C;

    constexpr int kAS = 64*32;
    constexpr int kBS = 32*64;
    __shared__ alignas(128) double smem_out[64 * 64];
    int8_t* smem = (int8_t*)smem_out;

    const uint64_t M    = 168897325606883ULL;
    double scale        = inv * norm_factor;

    uint64_t sum[2][2][4];
    #pragma unroll
    for (int mt = 0; mt < 2; mt++) {
        for (int nt = 0; nt < 2; nt++) {
            for (int e = 0; e < 4; e++) {
                sum[mt][nt][e] = 0;
            }
        }
    }

    // Barrett reduction magic: inv_p[i] ≈ floor(2^32 / prime[i]) + 1
    // rem = vp - prime * ((uint64_t)vp * barrett >> 32)
    constexpr uint32_t kPrime[7]   = {127, 113, 109, 107, 103, 101, 97};
    constexpr uint64_t kBarrett[7] = {
        33818641ULL, 37991696ULL, 39408440ULL, 40131153ULL,
        41680701ULL, 42516781ULL, 44278014ULL
    };
    const uint64_t M_p[7] = {
        147618922380819ULL, 112099994871825ULL, 134807957135769ULL,
        34726552928518ULL, 96747011755399ULL, 130435558389474ULL,
        19153304965729ULL
    };

    int8_t* smem_X = smem;

    int8_t* smem_B = smem + 7 * 64 * 32;

    for (int kk = 0; kk < k; kk += 32) {
        for (int i = threadIdx.x; i < 32*64; i += blockDim.x) {
            int rr = i>>5, cc = i&31;
            int smem_idx = rr * 32 + cc;
            int gr = tile_m+rr, gc = kk+cc;
            if (gc < k) {
                float v = (float)X[(size_t)gr*k + gc];
                if (v == 0.0f) {
                smem_X[0*2048 + smem_idx] = 0; smem_X[1*2048 + smem_idx] = 0; smem_X[2*2048 + smem_idx] = 0;
                smem_X[3*2048 + smem_idx] = 0; smem_X[4*2048 + smem_idx] = 0; smem_X[5*2048 + smem_idx] = 0; smem_X[6*2048 + smem_idx] = 0;
            } else {
                int32_t sign = (v > 0.0f) ? 1 : -1;
                uint64_t u_iv = __double2ull_rn(fabs((double)v) * 2.0e9);
                uint64_t u_rem; int32_t rem;
                u_rem = u_iv % 127; rem = sign * (int32_t)u_rem; if(rem<0) rem+=127; smem_X[0*2048 + smem_idx] = (int8_t)rem;
                u_rem = u_iv % 113; rem = sign * (int32_t)u_rem; if(rem<0) rem+=113; smem_X[1*2048 + smem_idx] = (int8_t)rem;
                u_rem = u_iv % 109; rem = sign * (int32_t)u_rem; if(rem<0) rem+=109; smem_X[2*2048 + smem_idx] = (int8_t)rem;
                u_rem = u_iv % 107; rem = sign * (int32_t)u_rem; if(rem<0) rem+=107; smem_X[3*2048 + smem_idx] = (int8_t)rem;
                u_rem = u_iv % 103; rem = sign * (int32_t)u_rem; if(rem<0) rem+=103; smem_X[4*2048 + smem_idx] = (int8_t)rem;
                u_rem = u_iv % 101; rem = sign * (int32_t)u_rem; if(rem<0) rem+=101; smem_X[5*2048 + smem_idx] = (int8_t)rem;
                u_rem = u_iv % 97;  rem = sign * (int32_t)u_rem; if(rem<0) rem+=97;  smem_X[6*2048 + smem_idx] = (int8_t)rem;
            }
        } else {
            smem_X[0*2048 + smem_idx] = 0; smem_X[1*2048 + smem_idx] = 0; smem_X[2*2048 + smem_idx] = 0;
            smem_X[3*2048 + smem_idx] = 0; smem_X[4*2048 + smem_idx] = 0; smem_X[5*2048 + smem_idx] = 0; smem_X[6*2048 + smem_idx] = 0;
        }
        }
        uint32_t* smem_B_u32 = (uint32_t*)smem_B;
        for (int i = threadIdx.x; i < 64*8; i += blockDim.x) {
            int nc = i >> 3;
            int kr_u32 = i & 7;
            int kr_base = kr_u32 * 4;

            uint32_t packed = 0;
            #pragma unroll
            for(int j=0; j<4; j++) {
                int kr = kr_base + j;
                int gk = kk+kr, gn = tile_n+nc;
                bool valid = (gk<k && gn<k);
                int par = valid ? (__popcll((unsigned long long)gk & (unsigned long long)gn) & 1) : 0;
                int8_t val = valid ? (par ? -1 : 1) : 0;
                packed |= ((uint32_t)(uint8_t)val) << (j * 8);
            }
            smem_B_u32[nc * 8 + kr_u32] = packed;
        }
        __syncthreads();

        #pragma unroll
        for (int p = 0; p < 7; ++p) {
            int32_t acc[2][2][4] = {0};
            int8_t* sA = smem_X + p * 2048;

            uint32_t af[2][4];
            uint32_t bf[2][2];

            #pragma unroll
            for (int mt = 0; mt < 2; mt++) {
                int row = lane % 16;
                int col_bytes = (lane / 16) * 16;
                void* ptr_A = &sA[(wr*32 + mt*16 + row) * 32 + col_bytes];
                implicit_ozaki_kernels::ldmatrix_x4_int8(af[mt], ptr_A);
            }

            #pragma unroll
            for (int nt = 0; nt < 2; nt++) {
                int nc = (wc*16 + nt*8) + (lane % 8);
                int kr_bytes = ((lane % 16) / 8) * 16;
                void* ptr_B = &smem_B[nc * 32 + kr_bytes];
                implicit_ozaki_kernels::ldmatrix_x2_int8(bf[nt], ptr_B);
            }


            #pragma unroll
            for (int mt = 0; mt < 2; mt++) {
                #pragma unroll
                for (int nt = 0; nt < 2; nt++) {
                    implicit_ozaki_kernels::mma_m16n8k32_s8(acc[mt][nt], af[mt], bf[nt], acc[mt][nt]);
                }
            }
            __syncwarp();

            #pragma unroll
            for (int mt = 0; mt < 2; mt++) {
                #pragma unroll
                for (int nt = 0; nt < 2; nt++) {
                    #pragma unroll
                    for (int e = 0; e < 4; e++) {
                        int32_t v = acc[mt][nt][e];
                        uint64_t m_p = M_p[p];
                        switch (p) {
                            case 0: v = (v % 127 + 127) % 127; break;
                            case 1: v = (v % 113 + 113) % 113; break;
                            case 2: v = (v % 109 + 109) % 109; break;
                            case 3: v = (v % 107 + 107) % 107; break;
                            case 4: v = (v % 103 + 103) % 103; break;
                            case 5: v = (v % 101 + 101) % 101; break;
                            default:v = (v % 97  + 97 ) % 97;  break;
                        }
                        sum[mt][nt][e] += (uint64_t)v * m_p;
                    }
                }
            }
        }
    }


    #pragma unroll
    for (int mt = 0; mt < 2; mt++) {
        #pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            const int mr  = wr*32 + mt*16 + (lane>>2);
            const int nc_e0 = wc*16 + nt*8 + (lane & 3) * 2;
            const int nc_e1 = nc_e0 + 1;

            const int rows[4] = {mr,  mr,   mr+8, mr+8};
            const int cols[4] = {nc_e0, nc_e1, nc_e0, nc_e1};

            #pragma unroll
            for (int e = 0; e < 4; e++) {
                uint64_t cmod = sum[mt][nt][e] % M;
                double   fv   = (double)cmod;
                if (cmod > M/2) fv -= (double)M;

                fv *= (1.0 / 2.0e9);

                smem_out[rows[e] * 64 + cols[e]] = fv * scale;
            }
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < 4096; i += blockDim.x) {
        int r, c;
        if (transpose_batch && batch_rows > 0) {
            c = i / 64;
            r = i % 64;
        } else {
            r = i / 64;
            c = i % 64;
        }

        int global_r = tile_m + r;
        int global_c = tile_n + c;

        if (global_c < k) {
            double val = smem_out[r * 64 + c];
            if (transpose_batch && batch_rows > 0) {
                int batch_idx = global_r / batch_rows;
                int r_in_batch = global_r % batch_rows;
                C[(size_t)batch_idx * ((size_t)k * batch_rows) + global_c * batch_rows + r_in_batch] = val;
            } else {
                C[(size_t)global_r * k + global_c] = val;
            }
        }
    }
}

void ImplicitHadamardOzakiEngine::execute_implicit_hadamard(const double* d_A, double* d_C, int m, int n, int k, double norm_factor, cudaStream_t stream, bool transpose_batch, int batch_rows) {
    if (transpose_batch && batch_rows > 0 && (n == 128 || n == 64)) {
        dim3 block(256);
        dim3 grid((n + 63) / 64, (m + 63) / 64);
        implicit_hadamard_ozaki_fused_batch_kernel<64, 256, 4><<<grid, block, 0, stream>>>(d_A, d_C, k, 1.0, norm_factor, transpose_batch, batch_rows);
        return;
    }

    dim3 block(16, 16);
    dim3 grid((n + 15) / 16, (m + 15) / 16);
    naive_hadamard_ozaki_kernel<<<grid, block, 0, stream>>>(d_A, d_C, m, n, k, norm_factor);
}

void ImplicitHadamardOzakiEngine::execute_fused_kronecker_hadamard(const double* d_X, double* d_Z, double* d_Y, int batch_size, int dim, double norm_factor, cudaStream_t stream) {}

} // namespace ozaki
