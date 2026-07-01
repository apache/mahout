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

#include "AdaptiveOzaki.h"
#include <iostream>
#include <cmath>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

__device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* global_ptr) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(global_ptr));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
__device__ __forceinline__ void cp_async_wait_0() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}
__device__ __forceinline__ void cp_async_wait_1() {
    asm volatile("cp.async.wait_group 1;\n" ::);
}

__device__ __forceinline__ double pow2_int(int exp) {
    return ldexp(1.0, exp);
}

__device__ __forceinline__ size_t get_A8_offset(int m_idx, int k_idx, int m, int k) {
    int tile_m = m_idx >> 7;
    int tile_k = k_idx >> 5;
    int local_m = m_idx & 127;
    int local_k = k_idx & 31;
    int num_tiles_k = (k + 31) >> 5;
    return (size_t)(tile_m * num_tiles_k + tile_k) * 4096 + (local_m << 5) + local_k;
}

__device__ __forceinline__ size_t get_B8_offset(int n_idx, int k_idx, int n, int k) {
    int tile_n = n_idx >> 6;
    int tile_k = k_idx >> 5;
    int local_n = n_idx & 63;
    int local_k = k_idx & 31;
    int num_tiles_k = (k + 31) >> 5;
    return (size_t)(tile_n * num_tiles_k + tile_k) * 2048 + (local_n << 5) + local_k;
}

__device__ __forceinline__ double fp64_hi(double v, int split_bits) {
    double scale = pow2_int(split_bits);
    double scaled = v * scale;
    double high_scaled = static_cast<double>(__double2ll_rn(scaled));
    return high_scaled / scale;
}

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

namespace ozaki {

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

AdaptiveOzakiEngine::AdaptiveOzakiEngine(const OzakiConfig& config) : config_(config) {
}
AdaptiveOzakiEngine::~AdaptiveOzakiEngine() {
    freeWorkspace();
}

void AdaptiveOzakiEngine::allocateWorkspace(int m, int n, int k) {
    if (workspace_allocated_) freeWorkspace();
    int nm = (m + 127) / 128, nn = (n + 127) / 128;
    cudaMalloc(&dmA_h, nm * 8); cudaMalloc(&dmA_l, nm * 8);
    cudaMalloc(&dmB_h, nn * 8); cudaMalloc(&dmB_l, nn * 8);

    size_t padded_mk = (size_t)((m + 127) / 128) * ((k + 31) / 32) * 4096;
    size_t padded_kn = (size_t)((n + 63) / 64) * ((k + 31) / 32) * 2048;
    cudaMalloc(&dA8_h, 7ULL * padded_mk); cudaMalloc(&dA8_l, 7ULL * padded_mk);
    cudaMalloc(&dB8_h, 7ULL * padded_kn); cudaMalloc(&dB8_l, 7ULL * padded_kn);

    size_t mk = (size_t)m * k;
    size_t kn = (size_t)k * n;
    cudaMalloc(&dA_hi_f32, mk * sizeof(float));
    cudaMalloc(&dA_low_f32, mk * sizeof(float));
    cudaMalloc(&dB_hi_f32, kn * sizeof(float));
    cudaMalloc(&dB_low_f32, kn * sizeof(float));
    cudaMalloc(&d_global_work_queue, sizeof(int));

    workspace_allocated_ = true;
    }

    void AdaptiveOzakiEngine::freeWorkspace() {
    if (!workspace_allocated_) return;
    cudaFree(dmA_h); cudaFree(dmA_l); cudaFree(dmB_h); cudaFree(dmB_l);
    cudaFree(dA8_h); cudaFree(dA8_l); cudaFree(dB8_h); cudaFree(dB8_l);
    cudaFree(dA_hi_f32); cudaFree(dA_low_f32); cudaFree(dB_hi_f32); cudaFree(dB_low_f32);
    cudaFree(d_global_work_queue);
    workspace_allocated_ = false;
}

PreScanStats AdaptiveOzakiEngine::analyzeMatrix(const double* /*d_A*/, const double* /*d_B*/, int m, int n, int k) { return PreScanStats{1.0, 0.0, std::min(m, std::min(n, k))}; }
int AdaptiveOzakiEngine::calculateOptimalTile(int /*m*/, int /*n*/, int /*k*/) { return 128; }

__global__ void precompute_modulo_kernel(const double* __restrict__ s, int8_t* __restrict__ d, int r, int c, int m, double sh, double sl) {
    // blockIdx.x: k_tiles, blockIdx.y: m_tiles. Block size: (32, 32)
    int local_k = threadIdx.x; // 0..31
    int local_m = threadIdx.y; // 0..31
    int tile_k = blockIdx.x;
    int tile_m = blockIdx.y;

    const int pr[7] = {127, 113, 109, 107, 103, 101, 97};
    size_t padded_size = (size_t)((r + 127) / 128) * ((c + 31) / 32) * 4096;
    int num_tiles_k = (c + 31) / 32;

    // A block (32x32) processes a 32x32 sub-tile of a 128x32 tile.
    // There are 4 such sub-tiles in a 128x32 tile vertically.
    // We can just launch more blocks or loop.
    // Let's launch with grid (num_tiles_k, num_tiles_m * 4).
    int m_idx = (tile_m / 4) * 128 + (tile_m % 4) * 32 + local_m;
    int k_idx = tile_k * 32 + local_k;

    if (m_idx < r && k_idx < c) {
        double v = s[(size_t)m_idx * c + k_idx];
        int32_t iv = 0;
        if (v != 0.0) {
            if (m == 0) iv = __double2int_rn(v * sh);
            else iv = __double2int_rn((v - (double)__double2ll_rn(v * sh) / sh) * sl);
        }

        size_t out_off = (size_t)( (m_idx / 128) * num_tiles_k + tile_k ) * 4096 + (m_idx % 128) * 32 + local_k;
        for (int p = 0; p < 7; p++) {
            int32_t rem = iv % pr[p];
            if (rem < 0) rem += pr[p];
            d[p * padded_size + out_off] = (int8_t)rem;
        }
    }
}
__global__ void precompute_modulo_kernel_B(const double* __restrict__ s, int8_t* __restrict__ d, int r, int c, int m, double sh, double sl) {
    // blockIdx.x: n_tiles, blockIdx.y: k_tiles. Block size: (32, 32)
    // s is k x n (r x c).
    int local_k = threadIdx.x; // 0..31
    int local_n = threadIdx.y; // 0..31
    int tile_n = blockIdx.x;
    int tile_k = blockIdx.y;

    const int pr[7] = {127, 113, 109, 107, 103, 101, 97};
    size_t padded_size = (size_t)((c + 63) / 64) * ((r + 31) / 32) * 2048;
    int num_tiles_k = (r + 31) / 32;

    // A block (32x32) processes a 32x32 sub-tile of a 64x32 tile.
    // There are 2 such sub-tiles in a 64x32 tile vertically (n direction).
    // Launch with grid (num_tiles_n * 2, num_tiles_k).
    int n_idx = (tile_n / 2) * 64 + (tile_n % 2) * 32 + local_n;
    int k_idx = tile_k * 32 + local_k;

    if (k_idx < r && n_idx < c) {
        double v = s[(size_t)k_idx * c + n_idx];
        int32_t iv = 0;
        if (v != 0.0) {
            if (m == 0) iv = __double2int_rn(v * sh);
            else iv = __double2int_rn((v - (double)__double2ll_rn(v * sh) / sh) * sl);
        }

        size_t out_off = (size_t)( (n_idx / 64) * num_tiles_k + tile_k ) * 2048 + (n_idx % 64) * 32 + local_k;
        for (int p = 0; p < 7; p++) {
            int32_t rem = iv % pr[p];
            if (rem < 0) rem += pr[p];
            d[p * padded_size + out_off] = (int8_t)rem;
        }
    }
}

__global__ void compute_slice_max_kernel(const double* __restrict__ A, const double* __restrict__ B, uint64_t* __restrict__ mA, uint64_t* __restrict__ mB, int m, int n, int k, int modA, int modB, double sh, double sl) {
    int bx = blockIdx.x;
    uint64_t lm = 0;
    if (blockIdx.y == 0) {
        if (!mA) return;
        int rb = bx * 128; if (rb >= m) return;
        for (int r = 0; r < 128; r++) {
            if (rb + r < m) {
                for (int c = threadIdx.x * 2; c < k; c += blockDim.x * 2) {
                    if (c + 1 < k) {
                        double2 v2 = *(const double2*)&A[(size_t)(rb + r) * k + c];
                        if (v2.x != 0.0) {
                            int64_t iv = (modA == 0) ? __double2ll_rn(v2.x * sh) : __double2ll_rn((v2.x - (double)__double2ll_rn(v2.x * sh) / sh) * sl);
                            uint64_t av = iv >= 0 ? iv : -iv; if (av > lm) lm = av;
                        }
                        if (v2.y != 0.0) {
                            int64_t iv = (modA == 0) ? __double2ll_rn(v2.y * sh) : __double2ll_rn((v2.y - (double)__double2ll_rn(v2.y * sh) / sh) * sl);
                            uint64_t av = iv >= 0 ? iv : -iv; if (av > lm) lm = av;
                        }
                    } else {
                        double v = A[(size_t)(rb + r) * k + c];
                        if (v != 0.0) {
                            int64_t iv = (modA == 0) ? __double2ll_rn(v * sh) : __double2ll_rn((v - (double)__double2ll_rn(v * sh) / sh) * sl);
                            uint64_t av = iv >= 0 ? iv : -iv; if (av > lm) lm = av;
                        }
                    }
                }
            }
        }
    } else {
        if (!mB) return;
        int cb = bx * 128; if (cb >= n) return;
        int c_limit = (cb + 128 <= n) ? 128 : (n - cb);
        if (c_limit == 128) {
            int total_items = k * 64;
            for (int idx = threadIdx.x; idx < total_items; idx += blockDim.x) {
                int r = idx >> 6;
                int c = (idx & 63) * 2;
                double2 v2 = *(const double2*)&B[(size_t)r * n + cb + c];
                if (v2.x != 0.0) {
                    int64_t iv = (modB == 0) ? __double2ll_rn(v2.x * sh) : __double2ll_rn((v2.x - (double)__double2ll_rn(v2.x * sh) / sh) * sl);
                    uint64_t av = iv >= 0 ? iv : -iv; if (av > lm) lm = av;
                }
                if (v2.y != 0.0) {
                    int64_t iv = (modB == 0) ? __double2ll_rn(v2.y * sh) : __double2ll_rn((v2.y - (double)__double2ll_rn(v2.y * sh) / sh) * sl);
                    uint64_t av = iv >= 0 ? iv : -iv; if (av > lm) lm = av;
                }
            }
        } else {
            int c_limit_half = (c_limit + 1) / 2;
            int total_items = k * c_limit_half;
            for (int idx = threadIdx.x; idx < total_items; idx += blockDim.x) {
                int r = idx / c_limit_half;
                int c_half = idx % c_limit_half;
                int c = c_half * 2;

                if (c + 1 < c_limit) {
                    double2 v2 = *(const double2*)&B[(size_t)r * n + cb + c];
                    if (v2.x != 0.0) {
                        int64_t iv = (modB == 0) ? __double2ll_rn(v2.x * sh) : __double2ll_rn((v2.x - (double)__double2ll_rn(v2.x * sh) / sh) * sl);
                        uint64_t av = iv >= 0 ? iv : -iv; if (av > lm) lm = av;
                    }
                    if (v2.y != 0.0) {
                        int64_t iv = (modB == 0) ? __double2ll_rn(v2.y * sh) : __double2ll_rn((v2.y - (double)__double2ll_rn(v2.y * sh) / sh) * sl);
                        uint64_t av = iv >= 0 ? iv : -iv; if (av > lm) lm = av;
                    }
                } else if (c < c_limit) {
                    double v = B[(size_t)r * n + cb + c];
                    if (v != 0.0) {
                        int64_t iv = (modB == 0) ? __double2ll_rn(v * sh) : __double2ll_rn((v - (double)__double2ll_rn(v * sh) / sh) * sl);
                        uint64_t av = iv >= 0 ? iv : -iv; if (av > lm) lm = av;
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        uint64_t remote = __shfl_down_sync(0xffffffff, lm, offset);
        if (remote > lm) lm = remote;
    }
    __shared__ uint64_t sm[8];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) sm[warp_id] = lm;
    __syncthreads();
    if (warp_id == 0) {
        lm = (lane < (blockDim.x / 32)) ? sm[lane] : 0;
        #pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            uint64_t remote = __shfl_down_sync(0xffffffff, lm, offset);
            if (remote > lm) lm = remote;
        }
        if (lane == 0) {
            if (blockIdx.y == 0) mA[bx] = lm;
            else mB[bx] = lm;
        }
    }
}

__global__ void bitshift_to_int8_kernel(const double* src, int8_t* dst, int count, int shift) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        double v = src[idx] * pow2_int(shift);
        int64_t iv = __double2ll_rn(v);
        if (iv > 127) iv = 127;
        if (iv < -127) iv = -127;
        dst[idx] = static_cast<int8_t>(iv);
    }
}

__global__ void residual_kernel(const double* A, const double* B, const double* C, double* R, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        double sum = 0.0;
        for (int kk = 0; kk < k; ++kk) {
            sum += A[(size_t)row * k + kk] * B[(size_t)kk * n + col];
        }
        R[(size_t)row * n + col] = sum - C[(size_t)row * n + col];
    }
}

__global__ void accumulate_sliced_kernel(const int8_t* A8, const int8_t* B8, double* C, int m, int n, int k, double scale) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        int32_t sum = 0;
        for (int kk = 0; kk < k; ++kk) {
            sum += static_cast<int32_t>(A8[(size_t)row * k + kk]) * static_cast<int32_t>(B8[(size_t)kk * n + col]);
        }
        C[(size_t)row * n + col] += static_cast<double>(sum) * scale;
    }
}

__global__ void accumulate_fused_kernel(const double* A, const double* B, double* C, int m, int n, int k, double scale) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        double sum = 0.0;
        for (int kk = 0; kk < k; ++kk) {
            sum += A[(size_t)row * k + kk] * B[(size_t)kk * n + col];
        }
        atomicAdd(&C[(size_t)row * n + col], sum * scale);
    }
}

__global__ void accumulate_fused_kernel_masked(const double* A, const double* B, double* C, int m, int n, int k, double scale, int row_stride, int row_limit) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        if ((row % row_stride) >= row_limit) return;
        double sum = 0.0;
        for (int kk = 0; kk < k; ++kk) {
            sum += A[(size_t)row * k + kk] * B[(size_t)kk * n + col];
        }
        atomicAdd(&C[(size_t)row * n + col], sum * scale);
    }
}

__global__ void accumulate_cross_terms_tf32_kernel(const float* __restrict__ A, const float* __restrict__ B, double* __restrict__ C, int m, int n, int k, double scale) {
    using namespace nvcuda;
    int rb = blockIdx.y * 16;
    int cb = blockIdx.x * 16;
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c;

    __shared__ float sa[16][8];
    __shared__ float sb[16][8];
    __shared__ float sc[16][16];

    int lane = threadIdx.x;

    // Chunking to prevent FP32 accumulator swamping
    const int CHUNK_SIZE = 256;

    for (int chunk_start = 0; chunk_start < k; chunk_start += CHUNK_SIZE) {
        wmma::fill_fragment(c, 0.0f);
        int chunk_end = min(chunk_start + CHUNK_SIZE, k);

        for (int ck = chunk_start; ck < chunk_end; ck += 8) {
            for (int i = 0; i < 4; ++i) {
                int idx = i * 32 + lane;
                int row_a = idx / 8;
                int col_a = idx % 8;
                if (rb + row_a < m && ck + col_a < k) sa[row_a][col_a] = A[(size_t)(rb + row_a) * k + ck + col_a];
                else sa[row_a][col_a] = 0.0f;

                int row_b = idx / 16;
                int col_b = idx % 16;
                if (ck + row_b < k && cb + col_b < n) sb[col_b][row_b] = B[(size_t)(ck + row_b) * n + cb + col_b];
                else sb[col_b][row_b] = 0.0f;
            }
            __syncthreads();

            wmma::load_matrix_sync(a, &sa[0][0], 8);
            wmma::load_matrix_sync(b, &sb[0][0], 8);
            wmma::mma_sync(c, a, b, c);
            __syncthreads();
        }

        wmma::store_matrix_sync(&sc[0][0], c, 16, wmma::mem_row_major);
        __syncthreads();

        for (int i = 0; i < 8; ++i) {
            int idx = i * 32 + lane;
            int r = idx / 16;
            int c_idx = idx % 16;
            if (rb + r < m && cb + c_idx < n) {
                atomicAdd(&C[(size_t)(rb + r) * n + cb + c_idx], static_cast<double>(sc[r][c_idx]) * scale);
            }
        }
        __syncthreads();
    }
}

__global__ void accumulate_cross_terms_fp32_cc_kernel(const double* A, const double* B, double* C, int m, int n, int k, double scale) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        double sum = 0.0;
        for (int kk = 0; kk < k; ++kk) {
            sum += static_cast<double>(static_cast<float>(A[(size_t)row * k + kk])) *
                   static_cast<double>(static_cast<float>(B[(size_t)kk * n + col]));
        }
        atomicAdd(&C[(size_t)row * n + col], sum * scale);
    }
}
__global__ void split_high_low_kernel(const double* src, double* high, double* low, float* high_f32, float* low_f32, int count, int shift) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        double scale = pow2_int(shift);
        double scaled = src[idx] * scale;
        double high_scaled = static_cast<double>(__double2ll_rn(scaled));
        double hi = high_scaled / scale;
        double lo = src[idx] - hi;
        if (high) high[idx] = hi;
        if (low) low[idx] = lo;
        if (high_f32) high_f32[idx] = static_cast<float>(hi);
        if (low_f32) low_f32[idx] = static_cast<float>(lo);
    }
}

__global__ void add_residual_kernel(double* C, const double* R, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        C[idx] += R[idx];
    }
}

__global__ void add_residual_kernel_masked(double* C, const double* R, int m, int n, int row_stride, int row_limit) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        if ((row % row_stride) >= row_limit) return;
        size_t idx = (size_t)row * n + col;
        C[idx] += R[idx];
    }
}

__global__ void microbench_fp64_kernel(double* d_out) {
    double a = 1.000001 + threadIdx.x;
    double b = 1.000002;
    double sum = 0;
    #pragma unroll(1)
    for (int i = 0; i < 10000; ++i) {
        sum = fma(a, b, sum);
        a += 0.000001; b -= 0.000001;
    }
    if (threadIdx.x == 0) d_out[0] = sum;
}

__global__ void microbench_fp32_kernel(float* d_out) {
    float a = 1.000001f + threadIdx.x;
    float b = 1.000002f;
    float sum = 0;
    #pragma unroll(1)
    for (int i = 0; i < 10000; ++i) {
        sum = fmaf(a, b, sum);
        a += 0.000001f; b -= 0.000001f;
    }
    if (threadIdx.x == 0) d_out[0] = sum;
}

__global__ void microbench_fp16_tc_kernel(float* d_out) {
    using namespace nvcuda;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(a, __float2half(1.0f));
    wmma::fill_fragment(b, __float2half(1.0f));
    wmma::fill_fragment(c, 0.0f);
    #pragma unroll(1)
    for (int i = 0; i < 10000; ++i) {
        wmma::mma_sync(c, a, b, c);
    }
    if (threadIdx.x == 0) d_out[0] = c.x[0];
}

__global__ void microbench_tc_kernel(int* d_out) {
    uint32_t a[4] = {1, 1, 1, 1};
    uint32_t b[2] = {1, 1};
    int32_t c[4] = {0, 0, 0, 0};
    #pragma unroll(1)
    for (int i = 0; i < 10000; ++i) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
    if (threadIdx.x == 0) d_out[0] = c[0];
}

__global__ void microbench_tf32_tc_kernel(float* d_out) {
    using namespace nvcuda;
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c;
    wmma::fill_fragment(a, 1.0f);
    wmma::fill_fragment(b, 1.0f);
    wmma::fill_fragment(c, 0.0f);
    #pragma unroll(1)
    for (int i = 0; i < 10000; ++i) {
        wmma::mma_sync(c, a, b, c);
    }
    if (threadIdx.x == 0) d_out[0] = c.x[0];
}

__global__ void microbench_int32_kernel(int* d_out) {
    int a = 100 + threadIdx.x;
    int b = 3;
    int sum = 0;
    #pragma unroll(1)
    for (int i = 0; i < 10000; ++i) {
        sum += (a % b);
        a++;
    }
    if (threadIdx.x == 0) d_out[0] = sum;
}

void AdaptiveOzakiEngine::profileHardwareRatio() {
    if (ratio_fp64_tc > 0) return;

    double* d_fp64;
    int* d_tc;
    float* d_fp32;
    float* d_fp16;
    float* d_tf32;
    int* d_int32;
    cudaMalloc(&d_fp64, sizeof(double));
    cudaMalloc(&d_tc, sizeof(int));
    cudaMalloc(&d_fp32, sizeof(float));
    cudaMalloc(&d_fp16, sizeof(float));
    cudaMalloc(&d_tf32, sizeof(float));
    cudaMalloc(&d_int32, sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    microbench_fp64_kernel<<<1, 32>>>(d_fp64);
    microbench_fp32_kernel<<<1, 32>>>(d_fp32);
    microbench_fp16_tc_kernel<<<1, 32>>>(d_fp16);
    microbench_tf32_tc_kernel<<<1, 32>>>(d_tf32);
    microbench_int32_kernel<<<1, 32>>>(d_int32);
    microbench_tc_kernel<<<1, 32>>>(d_tc);
    cudaDeviceSynchronize();

    float ms_fp64 = 0, ms_fp32 = 0, ms_fp16 = 0, ms_tf32 = 0, ms_int32 = 0, ms_tc = 0;

    cudaEventRecord(start);
    for(int i=0; i<5; i++) microbench_fp64_kernel<<<1, 32>>>(d_fp64);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_fp64, start, stop);

    cudaEventRecord(start);
    for(int i=0; i<5; i++) microbench_fp32_kernel<<<1, 32>>>(d_fp32);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_fp32, start, stop);

    cudaEventRecord(start);
    for(int i=0; i<5; i++) microbench_fp16_tc_kernel<<<1, 32>>>(d_fp16);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_fp16, start, stop);

    cudaEventRecord(start);
    for(int i=0; i<5; i++) microbench_tf32_tc_kernel<<<1, 32>>>(d_tf32);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_tf32, start, stop);

    cudaEventRecord(start);
    for(int i=0; i<5; i++) microbench_int32_kernel<<<1, 32>>>(d_int32);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_int32, start, stop);

    cudaEventRecord(start);
    for(int i=0; i<5; i++) microbench_tc_kernel<<<1, 32>>>(d_tc);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_tc, start, stop);

    // FP64 FMA: 32 ops/iter
    // FP32 FMA: 32 ops/iter
    // INT32 Modulo: 32 ops/iter
    // FP16 WMMA (16x16x16): 16*16*16 = 4096 ops/iter
    // TF32 WMMA (16x16x8): 16*16*8 = 2048 ops/iter
    // INT8 MMA (16x8x32): 16*8*32 = 4096 ops/iter

    double throughput_tc   = 4096.0 / ms_tc;
    double throughput_fp16 = 4096.0 / ms_fp16;
    double throughput_tf32 = 2048.0 / ms_tf32;
    double throughput_fp32 = 32.0 / ms_fp32;
    double throughput_int32 = 32.0 / ms_int32;
    double throughput_fp64 = 32.0 / ms_fp64;

    ratio_fp64_tc = throughput_fp64 / throughput_tc;
    ratio_fp32_tc = throughput_fp32 / throughput_tc;
    ratio_fp16_tc = throughput_fp16 / throughput_tc;
    ratio_tf32_tc = throughput_tf32 / throughput_tc;
    ratio_int32_tc = throughput_int32 / throughput_tc;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_fp64);
    cudaFree(d_tc);
    cudaFree(d_fp32);
    cudaFree(d_fp16);
    cudaFree(d_tf32);
    cudaFree(d_int32);

    std::cout << "[AdaptiveOzaki Phase 24] Hardware Profile (vs INT8 TC):\n";
    std::cout << "  - FP16 TC: " << (1.0 / ratio_fp16_tc) << "x slower (Alpha: " << ratio_fp16_tc << ")\n";
    std::cout << "  - TF32 TC: " << (1.0 / ratio_tf32_tc) << "x slower (Alpha: " << ratio_tf32_tc << ")\n";
    std::cout << "  - INT32 CC: " << (1.0 / ratio_int32_tc) << "x slower (Alpha: " << ratio_int32_tc << ")\n";
    std::cout << "  - FP32 CC: " << (1.0 / ratio_fp32_tc) << "x slower (Alpha: " << ratio_fp32_tc << ")\n";
    std::cout << "  - FP64 CC: " << (1.0 / ratio_fp64_tc) << "x slower (Alpha: " << ratio_fp64_tc << ")\n";
}

__constant__ int d_primes[7] = {127, 113, 109, 107, 103, 101, 97};
__constant__ uint64_t d_M_arr[8] = {0ULL, 127ULL, 14351ULL, 1564259ULL, 167375713ULL, 17239698439ULL, 1741209542339ULL, 168897325606883ULL};
__constant__ uint64_t d_coeffs_flat[29] = {0, 1, 1017, 13335, 775971, 27686, 760603, 71167626, 111090075, 42995596, 109498130, 12624346101, 7475621447, 10755041228, 5800272372, 15063814170, 1357320824343, 662584162129, 734822375666, 1643571624077, 980486926754, 1586052256388, 147618922380819, 112099994871825, 134807957135769, 34726552928518, 96747011755399, 130435558389474, 19153304965729};
__constant__ int d_coeff_offsets[8] = {0, 1, 2, 4, 7, 11, 16, 22};

struct KernelHeteroConfig {
    int enable_fp64;
    int enable_residual;
    int split_fp64_bits;
    int split_tc_bits;
    int split_residual_bits;
    int warp_fp64;
    int warp_tc;
    int warp_residual;
};

__device__ __forceinline__ void crt_pass_kernel_body(const int8_t* __restrict__ A8, const int8_t* __restrict__ B8, double* __restrict__ C, const uint64_t* mA, const uint64_t* mB, int m, int n, int k, double inv, KernelHeteroConfig cfg) {
    int rb = blockIdx.y * 128, cb = blockIdx.x * 64, tid = threadIdx.x, lane = tid % 32, wid = tid / 32;
    int wm = wid / 2, ms = wm * 32, ns = (wid % 2) * 32;
    uint64_t bma = mA[blockIdx.y], bmb = mB[blockIdx.x / 2], M = 0; int nl = 0;
    int tc_begin = cfg.warp_fp64;
    int tc_end = cfg.warp_fp64 + cfg.warp_tc;
    bool tc_active = (cfg.warp_tc <= 0) ? true : (wid >= tc_begin && wid < tc_end);
    if (bma > 0 && bmb > 0) {
        uint64_t th = (uint64_t)k * bma * bmb;
        if (th < 127ULL) nl = 1; else if (th < 14351ULL) nl = 2; else if (th < 1564259ULL) nl = 3; else if (th < 167375713ULL) nl = 4; else if (th < 17239698439ULL) nl = 5; else if (th < 1741209542339ULL) nl = 6; else nl = 7;
    }
    bool cap_tc_bits = ((cfg.enable_fp64 && cfg.warp_fp64 > 0) || (cfg.enable_residual && cfg.warp_residual > 0));
    if (cap_tc_bits && cfg.split_tc_bits > 0) {
        double target = pow2_int(cfg.split_tc_bits);
        int nl_cap = 0;
        for (int i = 1; i <= 7; ++i) {
            if (static_cast<double>(d_M_arr[i]) >= target) {
                nl_cap = i;
                break;
            }
        }
        if (nl_cap == 0) nl_cap = 7;
        if (nl > nl_cap) nl = nl_cap;
    }
    if (nl == 0) return;

    __shared__ alignas(16) int8_t sa[2][128][48], sb[2][64][48];
    uint64_t final_weighted_sum[2][4][4];
    for (int i = 0; i < 2; i++) for (int j = 0; j < 4; j++) for (int r = 0; r < 4; r++) final_weighted_sum[i][j][r] = 0;

    M = d_M_arr[nl]; int off = d_coeff_offsets[nl];

    size_t padded_mk = (size_t)((m + 127) / 128) * ((k + 31) / 32) * 4096;
    size_t padded_kn = (size_t)((n + 63) / 64) * ((k + 31) / 32) * 2048;

    for (int p = 0; p < nl; p++) {
        int32_t cf[2][4][4];
        for (int i = 0; i < 2; i++) for (int j = 0; j < 4; j++) for (int r = 0; r < 4; r++) cf[i][j][r] = 0;

        // Load sa: 128 rows x 32 cols. 256 threads * 16 bytes = 4096 bytes. (128*32 = 4096).
        int m_idx = tid / 2, k_idx = (tid % 2) * 16;
        const int8_t* ptr_A8 = &A8[p * padded_mk + get_A8_offset(rb + m_idx, k_idx, m, k)];
        if (rb + m_idx < m && k_idx < k) cp_async_16(&sa[0][m_idx][k_idx], ptr_A8); else *(int4*)&sa[0][m_idx][k_idx] = make_int4(0, 0, 0, 0);

        // Load sb: 64 rows x 32 cols. 128 loads of 16 bytes.
        int n_idx = tid / 2, kb_idx = (tid % 2) * 16;
        const int8_t* ptr_B8 = &B8[p * padded_kn + get_B8_offset(cb + n_idx, kb_idx, n, k)];
        if (tid < 128) {
            if (cb + n_idx < n && kb_idx < k) cp_async_16(&sb[0][n_idx][kb_idx], ptr_B8); else *(int4*)&sb[0][n_idx][kb_idx] = make_int4(0, 0, 0, 0);
        }
        cp_async_commit();

        for (int ck = 0; ck < k; ck += 32) {
            ptr_A8 += 4096;
            ptr_B8 += 2048;
            int cbuf = (ck / 32) % 2, fbuf = 1 - cbuf;
            if (ck + 32 < k) {
                if (rb + m_idx < m && ck + 32 + k_idx < k) cp_async_16(&sa[fbuf][m_idx][k_idx], ptr_A8); else *(int4*)&sa[fbuf][m_idx][k_idx] = make_int4(0, 0, 0, 0);
                if (tid < 128) {
                    if (cb + n_idx < n && ck + 32 + kb_idx < k) cp_async_16(&sb[fbuf][n_idx][kb_idx], ptr_B8); else *(int4*)&sb[fbuf][n_idx][kb_idx] = make_int4(0, 0, 0, 0);
                }
                cp_async_commit(); cp_async_wait_1();
            } else {
                cp_async_wait_0();
            }
            __syncthreads();

            if (tc_active) {
                uint32_t bf[4][2];
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    uint32_t addr_b = __cvta_generic_to_shared(&sb[cbuf][ns + j * 8 + (lane % 8)][(lane / 8 % 2) * 16]);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];" : "=r"(bf[j][0]), "=r"(bf[j][1]) : "r"(addr_b));
                }

                #pragma unroll
                for (int i = 0; i < 2; i++) {
                    uint32_t af[4];
                    int a_row = (lane / 8 % 2) * 8 + (lane % 8);
                    int a_col = (lane / 16) * 16;
                    uint32_t addr_a = __cvta_generic_to_shared(&sa[cbuf][ms + i * 16 + a_row][a_col]);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(af[0]), "=r"(af[1]), "=r"(af[2]), "=r"(af[3]) : "r"(addr_a));

                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+r"(cf[i][j][0]), "+r"(cf[i][j][1]), "+r"(cf[i][j][2]), "+r"(cf[i][j][3])
                            : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(bf[j][0]), "r"(bf[j][1]));
                    }
                }
            }
            __syncthreads(); // Ensure all warps finished consuming current tile before loading next one
        }
        __syncthreads();
        uint64_t cp = d_coeffs_flat[off + p];
        if (tc_active) {
            for (int i = 0; i < 2; i++) for (int j = 0; j < 4; j++) for (int r = 0; r < 4; r++) {
                int32_t v = cf[i][j][r] % d_primes[p]; if (v < 0) v += d_primes[p];
                final_weighted_sum[i][j][r] = (final_weighted_sum[i][j][r] + ((uint64_t)v * cp) % M) % M;
            }
        }
    }

    int r_off[4] = {lane / 4, lane / 4, (lane / 4) + 8, (lane / 4) + 8};
    int c_off[4] = {(lane % 4) * 2, (lane % 4) * 2 + 1, (lane % 4) * 2, (lane % 4) * 2 + 1};
    if (tc_active) {
        for (int i = 0; i < 2; i++) for (int j = 0; j < 4; j++) for (int r = 0; r < 4; r++) {
            int gm = rb + ms + i * 16 + r_off[r], gn = cb + ns + j * 8 + c_off[r];
            if (gm < m && gn < n) {
                uint64_t cv = final_weighted_sum[i][j][r] % M;
                double fv = (cv > M / 2) ? (double)((int64_t)cv - (int64_t)M) : (double)cv;
                atomicAdd(&C[(size_t)gm * n + gn], fv * inv);
            }
        }
    }
}

__global__ __launch_bounds__(256, 3)
void decoupled_crt_pass_kernel(const int8_t* __restrict__ A8, const int8_t* __restrict__ B8, double* __restrict__ C, const uint64_t* mA, const uint64_t* mB, int m, int n, int k, double inv) {
    KernelHeteroConfig cfg;
    cfg.enable_fp64 = 0;
    cfg.enable_residual = 0;
    cfg.split_fp64_bits = 0;
    cfg.split_tc_bits = 0;
    cfg.split_residual_bits = 0;
    cfg.warp_fp64 = 0;
    cfg.warp_tc = 0;
    cfg.warp_residual = 0;
    crt_pass_kernel_body(A8, B8, C, mA, mB, m, n, k, inv, cfg);
}

__global__ __launch_bounds__(256, 2)
void hetero_crt_pass_kernel(const int8_t* __restrict__ A8, const int8_t* __restrict__ B8, const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C, const uint64_t* mA, const uint64_t* mB, int m, int n, int k, double inv, KernelHeteroConfig cfg, int pass, int fp64_kernel, int split_bits) {
    crt_pass_kernel_body(A8, B8, C, mA, mB, m, n, k, inv, cfg);
    if (fp64_kernel && pass == 0) {
        int rb = blockIdx.y * 128, cb = blockIdx.x * 64; // Note cb is * 64 now (since C tile is 128x64) Wait! In hetero_crt_pass_kernel it was cb = blockIdx.x * 128. Let me fix the grid.
        int tid = threadIdx.x, lane = tid % 32, wid = tid / 32;
        int wm = wid / 2, wn = wid % 2, ms = wm * 16, ns = wn * 32;
        // Only let the designated fp64 warps perform FP64 hi*hi accumulation
        bool fp64_active = (cfg.warp_fp64 > 0) ? (wid >= 0 && wid < cfg.warp_fp64) : true;
        if (fp64_active) {
            // Each lane computes its own outputs and writes them directly.
            int row = rb + ms + lane % 16;
            double sum0 = 0.0, sum1 = 0.0;
            if (row < m) {
                int col0 = cb + ns + (lane / 16) * 16 + (lane % 8) * 2;
                int col1 = col0 + 1;
                for (int kk = 0; kk < k; ++kk) {
                    double a_hi = fp64_hi(A[(size_t)row * k + kk], split_bits);
                    if (col0 < n) {
                        double b_hi0 = fp64_hi(B[(size_t)kk * n + col0], split_bits);
                        sum0 += a_hi * b_hi0;
                    }
                    if (col1 < n) {
                        double b_hi1 = fp64_hi(B[(size_t)kk * n + col1], split_bits);
                        sum1 += a_hi * b_hi1;
                    }
                }
                // write back directly; ownership of (row,col) is unique within this block
                if (col0 < n) C[(size_t)row * n + col0] += sum0;
                if (col1 < n) C[(size_t)row * n + col1] += sum1;
            }
        }
    }
}

static KernelHeteroConfig make_kernel_config(const OzakiConfig& config) {
    KernelHeteroConfig cfg;
    cfg.enable_fp64 = config.enable_fp64 ? 1 : 0;
    cfg.enable_residual = config.enable_residual ? 1 : 0;
    cfg.split_fp64_bits = config.split_fp64_bits;
    cfg.split_tc_bits = config.split_tc_bits;
    cfg.split_residual_bits = config.split_residual_bits;
    cfg.warp_fp64 = config.enable_fp64 ? config.warp_fp64 : 0;
    cfg.warp_tc = config.warp_tc;
    cfg.warp_residual = config.enable_residual ? config.warp_residual : 0;
    if (config.mode == ExecutionMode::Phase23Hetero) {
        int total = cfg.warp_fp64 + cfg.warp_tc + cfg.warp_residual;
        bool valid_partition = (total == 8 && cfg.warp_tc > 0);
        bool require_full_tc = !(config.enable_residual && cfg.warp_residual > 0);
        if (!valid_partition || (require_full_tc && cfg.warp_tc != 8)) {
            cfg.warp_fp64 = 0;
            cfg.warp_tc = 8;
            cfg.warp_residual = 0;
        }
    }
    return cfg;
}


__global__ void precompute_modulo_kernel_p26(const double* __restrict__ s, int8_t* __restrict__ d, int r, int c, int m, double sh, double sl) {
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
        for (int p = 0; p < 7; p++) { int32_t rem = iv % pr[p]; if (rem < 0) rem += pr[p]; d[p * padded_size + out_off] = (int8_t)rem; }
    }
}

__global__ void precompute_modulo_kernel_B_p26(const double* __restrict__ s, int8_t* __restrict__ d, int r, int c, int m, double sh, double sl) {
    int local_k = threadIdx.x; int local_n = threadIdx.y;
    int tile_n = blockIdx.x; int tile_k = blockIdx.y;
    const int pr[7] = {127, 113, 109, 107, 103, 101, 97};
    size_t padded_size = (size_t)((c + 63) / 64) * ((r + 31) / 32) * 2048;
    int num_tiles_k = (r + 31) / 32;
    int n_idx = (tile_n / 2) * 64 + (tile_n % 2) * 32 + local_n;
    int k_idx = tile_k * 32 + local_k;
    if (k_idx < r && n_idx < c) {
        double v = s[(size_t)k_idx * c + n_idx];
        int32_t iv = (v == 0.0) ? 0 : __double2int_rn(v * sh);
        size_t out_off = (size_t)( (n_idx / 64) * num_tiles_k + tile_k ) * 2048 + (n_idx % 64) * 32 + local_k;
        for (int p = 0; p < 7; p++) { int32_t rem = iv % pr[p]; if (rem < 0) rem += pr[p]; d[p * padded_size + out_off] = (int8_t)rem; }
    }
}

__global__ void hybrid_ozaki_persistent_kernel(
    const int8_t* __restrict__ A8_h, const int8_t* __restrict__ B8_h,
    const float* __restrict__ A_hi_f32, const float* __restrict__ A_low_f32,
    const float* __restrict__ B_hi_f32, const float* __restrict__ B_low_f32,
    double* __restrict__ C, int m, int n, int k, double inv, int* __restrict__ d_work_queue) {
    extern __shared__ int8_t shared_mem[];
    int8_t* sA8 = &shared_mem[0];
    int8_t* sB8 = &sA8[28672];
    float* sTF32 = (float*)&sB8[28672];

    int warp_id = threadIdx.x / 32, lane_id = threadIdx.x % 32;
    __shared__ int tile_idx;
    int total_tiles_m = (m + 63) / 64, total_tiles_n = (n + 63) / 64, total_tiles = total_tiles_m * total_tiles_n;

    const int pr[7] = {127, 113, 109, 107, 103, 101, 97};
    const uint64_t M = 168897325606883ULL;
    const uint64_t f[7] = {
        147618922380819ULL, 112099994871825ULL, 134807957135769ULL,
        34726552928518ULL, 96747011755399ULL, 130435558389474ULL,
        19153304965729ULL
    };

    while (true) {
        if (threadIdx.x == 0) tile_idx = atomicAdd(d_work_queue, 1);
        __syncthreads();
        int ct = tile_idx; if (ct >= total_tiles) break;
        int tile_m = (ct % total_tiles_m) * 64, tile_n = (ct / total_tiles_m) * 64;

        uint64_t final_acc[8][4];
        for(int i=0; i<8; i++) for(int j=0; j<4; j++) final_acc[i][j] = 0;

        int32_t prime_acc[7][8][4];
        for(int p=0; p<7; p++) for(int i=0; i<8; i++) for(int j=0; j<4; j++) prime_acc[p][i][j] = 0;

        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 8, float> c_frag_tf32[2][2];
        if (warp_id >= 4) {
            for(int i=0; i<2; i++) for(int j=0; j<2; j++) nvcuda::wmma::fill_fragment(c_frag_tf32[i][j], 0.0f);
        }

        for (int kk = 0; kk < k; kk += 32) {
            int b_idx = (kk / 32) % 2;
            int k_size = min(32, k - kk);
            size_t padded_mk = (size_t)((m + 127) / 128) * ((k + 31) / 32) * 4096;
            size_t padded_kn = (size_t)((n + 63) / 64) * ((k + 31) / 32) * 2048;

            for (int p = 0; p < 7; ++p) {
                const int8_t* Ap = A8_h + p * padded_mk;
                const int8_t* Bp = B8_h + p * padded_kn;

                for (int i = threadIdx.x; i < 64 * k_size; i += 256) {
                    int r = i / k_size, c = i % k_size;
                    int8_t val = (tile_m + r < m && kk + c < k) ? Ap[get_A8_offset(tile_m + r, kk + c, m, k)] : 0;
                    sA8[p * 4096 + b_idx * 2048 + r * 32 + c] = val;
                }
                for (int i = threadIdx.x; i < k_size * 64; i += 256) {
                    int r = i / 64, c = i % 64;
                    int8_t val = (kk + r < k && tile_n + c < n) ? Bp[get_B8_offset(tile_n + c, kk + r, n, k)] : 0;
                    sB8[p * 4096 + b_idx * 2048 + r * 64 + c] = val;
                }
            }
            for (int i = threadIdx.x; i < 64 * k_size; i += 256) {
                int r = i / k_size, c = i % k_size;
                sTF32[b_idx * 4096 + r * 32 + c] = (tile_m + r < m && kk + c < k) ? A_hi_f32[(size_t)(tile_m + r) * k + kk + c] : 0.0f;
            }
            for (int i = threadIdx.x; i < k_size * 64; i += 256) {
                int r = i / 64, c = i % 64;
                sTF32[2048 + b_idx * 4096 + r * 64 + c] = (kk + r < k && tile_n + c < n) ? B_hi_f32[(size_t)(kk + r) * n + (tile_n + c)] : 0.0f;
            }
            __syncthreads();

            if (warp_id < 4) {
                int wr = warp_id / 2;
                int wc = warp_id % 2;

                for (int p = 0; p < 7; ++p) {
                    int8_t* pA = &sA8[p * 4096 + b_idx * 2048];
                    int8_t* pB = &sB8[p * 4096 + b_idx * 2048];

                    #pragma unroll
                    for (int k_s = 0; k_s < 32; k_s += 32) {
                        #pragma unroll
                        for (int mt = 0; mt < 2; ++mt) {
                            #pragma unroll
                            for (int nt = 0; nt < 4; ++nt) {
                                uint32_t ra[4], rb[2];

                                int r_a_0 = lane_id / 4;
                                int r_a_8 = r_a_0 + 8;
                                int k_base_a = (lane_id % 4) * 8;
                                int r0 = wr * 32 + mt * 16 + r_a_0;
                                int r8 = wr * 32 + mt * 16 + r_a_8;
                                int c_a = k_s + k_base_a;

                                uint32_t final_va0 = 0, final_va1 = 0, final_va2 = 0, final_va3 = 0;
                                final_va0 |= ((uint32_t)(uint8_t)pA[r0 * 32 + c_a + 0]) << 0;
                                final_va0 |= ((uint32_t)(uint8_t)pA[r0 * 32 + c_a + 1]) << 8;
                                final_va0 |= ((uint32_t)(uint8_t)pA[r0 * 32 + c_a + 2]) << 16;
                                final_va0 |= ((uint32_t)(uint8_t)pA[r0 * 32 + c_a + 3]) << 24;

                                final_va1 |= ((uint32_t)(uint8_t)pA[r8 * 32 + c_a + 0]) << 0;
                                final_va1 |= ((uint32_t)(uint8_t)pA[r8 * 32 + c_a + 1]) << 8;
                                final_va1 |= ((uint32_t)(uint8_t)pA[r8 * 32 + c_a + 2]) << 16;
                                final_va1 |= ((uint32_t)(uint8_t)pA[r8 * 32 + c_a + 3]) << 24;

                                final_va2 |= ((uint32_t)(uint8_t)pA[r0 * 32 + c_a + 4]) << 0;
                                final_va2 |= ((uint32_t)(uint8_t)pA[r0 * 32 + c_a + 5]) << 8;
                                final_va2 |= ((uint32_t)(uint8_t)pA[r0 * 32 + c_a + 6]) << 16;
                                final_va2 |= ((uint32_t)(uint8_t)pA[r0 * 32 + c_a + 7]) << 24;

                                final_va3 |= ((uint32_t)(uint8_t)pA[r8 * 32 + c_a + 4]) << 0;
                                final_va3 |= ((uint32_t)(uint8_t)pA[r8 * 32 + c_a + 5]) << 8;
                                final_va3 |= ((uint32_t)(uint8_t)pA[r8 * 32 + c_a + 6]) << 16;
                                final_va3 |= ((uint32_t)(uint8_t)pA[r8 * 32 + c_a + 7]) << 24;

                                ra[0] = final_va0; ra[1] = final_va1; ra[2] = final_va2; ra[3] = final_va3;

                                int cb_base = wc * 32 + nt * 8;
                                int n_col = lane_id / 4;
                                int k_base = (lane_id % 4) * 8;
                                int c0 = cb_base + n_col;

                                uint32_t final_vb0 = 0, final_vb1 = 0;
                                final_vb0 |= ((uint32_t)(uint8_t)pB[(k_s + k_base + 0) * 64 + c0]) << 0;
                                final_vb0 |= ((uint32_t)(uint8_t)pB[(k_s + k_base + 1) * 64 + c0]) << 8;
                                final_vb0 |= ((uint32_t)(uint8_t)pB[(k_s + k_base + 2) * 64 + c0]) << 16;
                                final_vb0 |= ((uint32_t)(uint8_t)pB[(k_s + k_base + 3) * 64 + c0]) << 24;

                                final_vb1 |= ((uint32_t)(uint8_t)pB[(k_s + k_base + 4) * 64 + c0]) << 0;
                                final_vb1 |= ((uint32_t)(uint8_t)pB[(k_s + k_base + 5) * 64 + c0]) << 8;
                                final_vb1 |= ((uint32_t)(uint8_t)pB[(k_s + k_base + 6) * 64 + c0]) << 16;
                                final_vb1 |= ((uint32_t)(uint8_t)pB[(k_s + k_base + 7) * 64 + c0]) << 24;
                                rb[0] = final_vb0; rb[1] = final_vb1;

                                mma_m16n8k32_s8(prime_acc[p][mt * 4 + nt], ra, rb, prime_acc[p][mt * 4 + nt]);
                            }
                        }
                    }
                }
            }

            if (warp_id >= 4) {
                int w_row = ((warp_id - 4) / 2) * 32, w_col = ((warp_id - 4) % 2) * 32;
                for (int k_s = 0; k_s < 32; k_s += 8) {
                    if (kk + k_s < k) {
                        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> a_hi[2], a_lo[2];
                        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> b_hi[2], b_lo[2];

                        for(int i=0; i<2; i++) {
                            int r_idx = tile_m + w_row + i*16;
                            if (r_idx < m) {
                                nvcuda::wmma::load_matrix_sync(a_hi[i], &A_hi_f32[(size_t)r_idx * k + kk + k_s], k);
                                nvcuda::wmma::load_matrix_sync(a_lo[i], &A_low_f32[(size_t)r_idx * k + kk + k_s], k);
                            } else {
                                nvcuda::wmma::fill_fragment(a_hi[i], 0.0f);
                                nvcuda::wmma::fill_fragment(a_lo[i], 0.0f);
                            }
                        }
                        for(int j=0; j<2; j++) {
                            int c_idx = tile_n + w_col + j*16;
                            if (c_idx < n) {
                                nvcuda::wmma::load_matrix_sync(b_hi[j], &B_hi_f32[(size_t)(kk + k_s) * n + c_idx], n);
                                nvcuda::wmma::load_matrix_sync(b_lo[j], &B_low_f32[(size_t)(kk + k_s) * n + c_idx], n);
                            } else {
                                nvcuda::wmma::fill_fragment(b_hi[j], 0.0f);
                                nvcuda::wmma::fill_fragment(b_lo[j], 0.0f);
                            }
                        }

                        for(int i=0; i<2; i++) for(int j=0; j<2; j++) {
                            nvcuda::wmma::mma_sync(c_frag_tf32[i][j], a_hi[i], b_lo[j], c_frag_tf32[i][j]);
                            nvcuda::wmma::mma_sync(c_frag_tf32[i][j], a_lo[i], b_hi[j], c_frag_tf32[i][j]);
                        }
                    }
                }
            }
            __syncthreads();
        }

        if (warp_id < 4) {
            for (int p = 0; p < 7; ++p) {
                for(int i=0; i<8; i++) {
                    for(int j=0; j<4; j++) {
                        uint32_t rem = (prime_acc[p][i][j] % pr[p] + pr[p]) % pr[p];
                        final_acc[i][j] += (uint64_t)rem * f[p];
                    }
                }
            }
        }

        if (warp_id < 4) {
            int wr = warp_id / 2, wc = warp_id % 2;
            for (int i = 0; i < 8; i++) {
                int mt = i / 4, nt = i % 4;
                int r_base = tile_m + wr * 32 + mt * 16 + (lane_id / 4);
                int c_base = tile_n + wc * 32 + nt * 8 + (lane_id % 4) * 2;

                uint64_t cv0 = final_acc[i][0];
                uint64_t cv1 = final_acc[i][1];
                uint64_t cv2 = final_acc[i][2];
                uint64_t cv3 = final_acc[i][3];

                auto store_res = [&](int r, int c, uint64_t cv) {
                    if (r < m && c < n) {
                        double fv = (double)(cv % M);
                        if (cv % M > M / 2) fv -= (double)M;
                        atomicAdd(&C[(size_t)r * n + c], fv * inv);
                    }
                };

                store_res(r_base, c_base, cv0);
                store_res(r_base, c_base + 1, cv1);
                store_res(r_base + 8, c_base, cv2);
                store_res(r_base + 8, c_base + 1, cv3);
            }
        }

        if (warp_id >= 4) {
            int w_row = ((warp_id - 4) / 2) * 32, w_col = ((warp_id - 4) % 2) * 32;
            for(int i=0; i<2; i++) for(int j=0; j<2; j++) {
                int r_c = tile_m + w_row + i*16, c_c = tile_n + w_col + j*16;
                float temp[16*16];
                nvcuda::wmma::store_matrix_sync(temp, c_frag_tf32[i][j], 16, nvcuda::wmma::mem_row_major);
                for(int r=0; r<16; r++) for(int c=0; c<16; c++) {
                    if(r_c + r < m && c_c + c < n) atomicAdd(&C[(size_t)(r_c+r)*n + c_c+c], (double)temp[r*16+c]);
                }
            }
        }
        __syncthreads();
    }
}

void AdaptiveOzakiEngine::execute(const double* dA, const double* dB, double* dC, int m, int n, int k) {

    if (config_.mode == ExecutionMode::Phase26HybridOzaki) {
        profileHardwareRatio();
        if (!workspace_allocated_) allocateWorkspace(m, n, k);
        cudaMemset(dC, 0, (size_t)m * n * 8);
        cudaStream_t st; cudaStreamCreate(&st);
        split_high_low_kernel<<<((size_t)m*k+255)/256, 256, 0, st>>>(dA, nullptr, nullptr, dA_hi_f32, dA_low_f32, (int)(m*k), config_.split_fp64_bits);
        split_high_low_kernel<<<((size_t)k*n+255)/256, 256, 0, st>>>(dB, nullptr, nullptr, dB_hi_f32, dB_low_f32, (int)(k*n), config_.split_fp64_bits);

        dim3 pre_block(32, 32);
        dim3 pre_grid_A((k + 31) / 32, ((m + 127) / 128) * 4);
        dim3 pre_grid_B(((n + 63) / 64) * 2, (k + 31) / 32);
        precompute_modulo_kernel_p26<<<pre_grid_A, pre_block, 0, st>>>(dA, dA8_h, m, k, 0, pow(2.0,15.0), pow(2.0,30.0));
        precompute_modulo_kernel_B_p26<<<pre_grid_B, pre_block, 0, st>>>(dB, dB8_h, k, n, 0, pow(2.0,15.0), pow(2.0,30.0));

        int num_sms; cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
        cudaMemsetAsync(d_global_work_queue, 0, sizeof(int), st);
        cudaFuncSetAttribute(hybrid_ozaki_persistent_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 96*1024);
        hybrid_ozaki_persistent_kernel<<<num_sms, 256, 96*1024, st>>>(dA8_h, dB8_h, dA_hi_f32, dA_low_f32, dB_hi_f32, dB_low_f32, dC, m, n, k, pow(2.0,-30.0), d_global_work_queue);
        cudaStreamSynchronize(st); cudaStreamDestroy(st);
        return;
    }
    profileHardwareRatio();

    OzakiConfig local_cfg = config_;
    int max_fp64_cells = (int)(8192.0 / 7.0 * ratio_fp64_tc);
    bool do_fp64_kernel = (local_cfg.mode == ExecutionMode::Phase23Hetero && local_cfg.enable_fp64 && local_cfg.enable_kernel_fp64);

    if (do_fp64_kernel && max_fp64_cells < 32) {
        static bool printed_dispatch = false;
        if (!printed_dispatch) {
            std::cout << "[AdaptiveOzaki] Smart Dispatch: Alpha (" << ratio_fp64_tc << ") too low. Max FP64 cells allowed per warp: " << max_fp64_cells << " < 32.\n";
            std::cout << "[AdaptiveOzaki] Smart Dispatch: Disabling kernel fusion to prevent Tensor Core stalls. Falling back to Host-side stream overlap.\n";
            printed_dispatch = true;
        }
        local_cfg.enable_kernel_fp64 = false;
        local_cfg.warp_fp64 = 0;
        local_cfg.warp_tc = 8;
        do_fp64_kernel = false;
    }

    bool local_alloc = false;
    if (!workspace_allocated_) { allocateWorkspace(m, n, k); local_alloc = true; }

    cudaMemset(dC, 0, (size_t)m * n * 8);
    double sh = pow(2.0, 17.0), sl = pow(2.0, 34.0);
    int nm = (m + 127) / 128;
    int nn_tc = (n + 63) / 64; // TC tile is 128x64
    int nn_max = (n + 127) / 128; // Max scan still uses 128

    const double* A_tc = dA;
    const double* B_tc = dB;
    double* dA_hi = nullptr;
    double* dA_low = nullptr;
    double* dB_hi = nullptr;
    double* dB_low = nullptr;

    bool is_phase24 = (local_cfg.mode == ExecutionMode::Phase24ExtremeMix);
    bool do_fp64 = ((local_cfg.mode == ExecutionMode::Phase23Hetero || is_phase24) && local_cfg.enable_fp64);
    if (local_cfg.mode == ExecutionMode::Phase23Hetero && !local_cfg.enable_kernel_fp64 && config_.enable_fp64) do_fp64 = true;
    bool do_residual = ((local_cfg.mode == ExecutionMode::Phase23Hetero || is_phase24) && local_cfg.enable_residual);
    bool do_cross_terms = do_fp64 && !do_residual;

    // Stream Setup
    cudaStream_t stream_tc = 0; // Or Stream 1 (INT8)
    cudaStream_t stream_fp64 = 0; // Stream 4
    cudaStream_t stream_fp16 = 0; // Stream 2
    cudaStream_t stream_fp32 = 0; // Stream 3
    cudaEvent_t split_done = nullptr;
    cudaEvent_t p24_sync_event = nullptr;

    bool use_streams = do_fp64 && !do_fp64_kernel;
    if (use_streams || is_phase24) {
        cudaStreamCreate(&stream_tc);
        cudaStreamCreate(&stream_fp64);
        cudaEventCreate(&split_done);
        if (is_phase24) {
            cudaStreamCreate(&stream_fp16);
            cudaStreamCreate(&stream_fp32);
            cudaEventCreate(&p24_sync_event);
        }
    }
    if (do_fp64) {
        size_t mk = (size_t)m * k;
        size_t kn = (size_t)k * n;
        cudaMalloc(&dA_hi, mk * sizeof(double));
        cudaMalloc(&dA_low, mk * sizeof(double));
        cudaMalloc(&dB_hi, kn * sizeof(double));
        cudaMalloc(&dB_low, kn * sizeof(double));
        int threads = 256;
        int blocks_a = (int)((mk + threads - 1) / threads);
        int blocks_b = (int)((kn + threads - 1) / threads);
        split_high_low_kernel<<<blocks_a, threads, 0, use_streams ? stream_fp64 : 0>>>(dA, dA_hi, dA_low, dA_hi_f32, dA_low_f32, (int)mk, local_cfg.split_fp64_bits);
        split_high_low_kernel<<<blocks_b, threads, 0, use_streams ? stream_fp64 : 0>>>(dB, dB_hi, dB_low, dB_hi_f32, dB_low_f32, (int)kn, local_cfg.split_fp64_bits);
        if (use_streams) {
            cudaEventRecord(split_done, stream_fp64);
            cudaStreamWaitEvent(stream_tc, split_done, 0);
            if (is_phase24) {
                cudaStreamWaitEvent(stream_fp32, split_done, 0);
                cudaStreamWaitEvent(stream_fp16, split_done, 0);
            }
        }

        if (!do_fp64_kernel) {
            dim3 block(16, 16);
            dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
            if (do_residual && local_cfg.warp_fp64 > 0 && local_cfg.warp_fp64 < 8) {
                accumulate_fused_kernel_masked<<<grid, block, 0, stream_fp64>>>(dA_hi, dB_hi, dC, m, n, k, 1.0, 8, local_cfg.warp_fp64);
            } else {
                accumulate_fused_kernel<<<grid, block, 0, stream_fp64>>>(dA_hi, dB_hi, dC, m, n, k, 1.0);
            }
            if (do_cross_terms) {
                if (is_phase24) {
                    // Use TF32 WMMA Kernel instead of FP32 CC!
                    dim3 tc_block(32);
                    dim3 tc_grid((n + 15) / 16, (m + 15) / 16);
                    accumulate_cross_terms_tf32_kernel<<<tc_grid, tc_block, 0, stream_fp16>>>(dA_hi_f32, dB_low_f32, dC, m, n, k, 1.0);
                    accumulate_cross_terms_tf32_kernel<<<tc_grid, tc_block, 0, stream_fp16>>>(dA_low_f32, dB_hi_f32, dC, m, n, k, 1.0);
                } else {
                    accumulate_fused_kernel<<<grid, block, 0, stream_fp64>>>(dA_hi, dB_low, dC, m, n, k, 1.0);
                    accumulate_fused_kernel<<<grid, block, 0, stream_fp64>>>(dA_low, dB_hi, dC, m, n, k, 1.0);
                }
            }
        }
        A_tc = dA_low;
        B_tc = dB_low;
    }

    dim3 pre_block(32, 32);
    dim3 pre_grid_A((k + 31) / 32, ((m + 127) / 128) * 4);
    dim3 pre_grid_B(((n + 63) / 64) * 2, (k + 31) / 32);

    precompute_modulo_kernel<<<pre_grid_A, pre_block, 0, stream_tc>>>(A_tc, dA8_h, m, k, 0, sh, sl);
    precompute_modulo_kernel<<<pre_grid_A, pre_block, 0, stream_tc>>>(A_tc, dA8_l, m, k, 1, sh, sl);
    precompute_modulo_kernel_B<<<pre_grid_B, pre_block, 0, stream_tc>>>(B_tc, dB8_h, k, n, 0, sh, sl);
    precompute_modulo_kernel_B<<<pre_grid_B, pre_block, 0, stream_tc>>>(B_tc, dB8_l, k, n, 1, sh, sl);

    compute_slice_max_kernel<<<dim3(nm, 1), 256, 0, stream_tc>>>(A_tc, nullptr, dmA_h, nullptr, m, n, k, 0, 0, sh, sl);
    compute_slice_max_kernel<<<dim3(nm, 1), 256, 0, stream_tc>>>(A_tc, nullptr, dmA_l, nullptr, m, n, k, 1, 0, sh, sl);
    compute_slice_max_kernel<<<dim3(nn_max, 2, 1), 256, 0, stream_tc>>>(nullptr, B_tc, nullptr, dmB_h, m, n, k, 0, 0, sh, sl);
    compute_slice_max_kernel<<<dim3(nn_max, 2, 1), 256, 0, stream_tc>>>(nullptr, B_tc, nullptr, dmB_l, m, n, k, 0, 1, sh, sl);

    int mA_map[4] = {0, 0, 1, 1}, mB_map[4] = {0, 1, 0, 1};
    int8_t *A8_ptrs[2] = {dA8_h, dA8_l}, *B8_ptrs[2] = {dB8_h, dB8_l};
    uint64_t *mA_ptrs[2] = {dmA_h, dmA_l}, *mB_ptrs[2] = {dmB_h, dmB_l};
    double inv[4] = {pow(2.0, -34.0), pow(2.0, -51.0), pow(2.0, -51.0), pow(2.0, -68.0)};

    KernelHeteroConfig kcfg = make_kernel_config(local_cfg);
    for (int pass = 0; pass < 4; pass++) {
        if (local_cfg.mode == ExecutionMode::Phase23Hetero) {
            hetero_crt_pass_kernel<<<dim3(nn_tc, nm), 256, 0, stream_tc>>>(
                A8_ptrs[mA_map[pass]], B8_ptrs[mB_map[pass]],
                dA, dB, dC,
                mA_ptrs[mA_map[pass]], mB_ptrs[mB_map[pass]],
                m, n, k, inv[pass], kcfg, pass, do_fp64_kernel ? 1 : 0, local_cfg.split_fp64_bits);
        } else {
            decoupled_crt_pass_kernel<<<dim3(nn_tc, nm), 256, 0, stream_tc>>>(A8_ptrs[mA_map[pass]], B8_ptrs[mB_map[pass]], dC, mA_ptrs[mA_map[pass]], mB_ptrs[mB_map[pass]], m, n, k, inv[pass]);
        }
    }

    if (do_residual) {
        bool do_partial_residual = local_cfg.enable_partial_residual;
        if (use_streams) {
            cudaStreamSynchronize(stream_fp64);
            cudaStreamSynchronize(stream_tc);
        }
        size_t mn = (size_t)m * n;
        double* dR = nullptr;
        cudaMalloc(&dR, mn * sizeof(double));
        computeResidualGradedRing(dA, dB, dC, dR, m, n, k);
        if (do_partial_residual && local_cfg.warp_residual > 0 && local_cfg.warp_residual < 8) {
            dim3 block(16, 16);
            dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
            add_residual_kernel_masked<<<grid, block>>>(dC, dR, m, n, 8, local_cfg.warp_residual);
        } else {
            int threads = 256;
            int blocks = (int)((mn + threads - 1) / threads);
            add_residual_kernel<<<blocks, threads>>>(dC, dR, (int)mn);
        }
        cudaFree(dR);
    }

    if (use_streams) {
        cudaStreamSynchronize(stream_fp64);
        cudaStreamSynchronize(stream_tc);
        if (is_phase24) {
            cudaStreamSynchronize(stream_fp16);
            cudaStreamSynchronize(stream_fp32);
        }
        cudaEventDestroy(split_done);
        cudaStreamDestroy(stream_fp64);
        cudaStreamDestroy(stream_tc);
        if (is_phase24) {
            cudaStreamDestroy(stream_fp16);
            cudaStreamDestroy(stream_fp32);
            cudaEventDestroy(p24_sync_event);
        }
    }

    if (dA_hi) cudaFree(dA_hi);
    if (dA_low) cudaFree(dA_low);
    if (dB_hi) cudaFree(dB_hi);
    if (dB_low) cudaFree(dB_low);

    if (local_alloc) freeWorkspace();
}

void AdaptiveOzakiEngine::bitShiftToINT8(const double* d_src, int8_t* d_dst, int count, int shift) {
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    bitshift_to_int8_kernel<<<blocks, threads>>>(d_src, d_dst, count, shift);
}

void AdaptiveOzakiEngine::computeResidualGradedRing(const double* d_A, const double* d_B, const double* d_C, double* d_R, int m, int n, int k) {
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    residual_kernel<<<grid, block>>>(d_A, d_B, d_C, d_R, m, n, k);
}

void AdaptiveOzakiEngine::accumulateSlicedProduct(const int8_t* d_A, const int8_t* d_B, double* d_C, int m, int n, int k, int shift) {
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    double scale = ldexp(1.0, shift);
    accumulate_sliced_kernel<<<grid, block>>>(d_A, d_B, d_C, m, n, k, scale);
}

void AdaptiveOzakiEngine::accumulateFusedSlicedProductWMMA(const double* d_A, const double* d_B, double* d_C, int m, int n, int k, int sA, int sB) {
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    int total_shift = sA + sB;
    double scale = ldexp(1.0, -total_shift);
    accumulate_fused_kernel<<<grid, block>>>(d_A, d_B, d_C, m, n, k, scale);
}

} // namespace ozaki

// PR6: Hook up AdaptiveOzakiEngine for mixed-precision graded-ring Tensor Core
// operations on *non-Hadamard* (arbitrary matrix) logic. This complements the
// ImplicitHadamardOzakiEngine (specialized for +/-1 Hadamard structure in IQP FWT)
// and finalizes the full TC-accelerated pipeline for general GEMM use cases in QDP.
extern "C" int launch_adaptive_ozaki_gemm(
    const double* dA,
    const double* dB,
    double* dC,
    int m, int n, int k,
    cudaStream_t stream
) {
    // Default to hybrid mode for best mixed FP64/INT8 TC performance on modern GPUs (sm_80+).
    // Caller can extend later to pass config.
    ozaki::OzakiConfig config;
    config.mode = ozaki::ExecutionMode::Phase26HybridOzaki;

    ozaki::AdaptiveOzakiEngine engine(config);
    // execute() manages its own workspace and internal streams for hybrid path.
    // For stream forwarding in future: extend execute() signature or use events.
    engine.execute(dA, dB, dC, m, n, k);

    // Best-effort: if a stream was provided by caller, honor a sync for safety
    // (engine may have used internal streams). Real production path should refine this.
    if (stream) {
        cudaStreamSynchronize(stream);
    }

    return static_cast<int>(cudaGetLastError());
}
