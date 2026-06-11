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
//
// Copyright (c) 2026 Advanced Micro Devices, Inc.
// Author: Jeff Daily <jeff.daily@amd.com>

// HIP forwarding shim for <cuComplex.h> (HIP build path only; see
// cuda_runtime.h in this directory for how it is selected). hipcc does not
// ship a <cuComplex.h>; HIP's <hip/hip_complex.h> provides the same complex
// layout and helpers under hip* names. The aliases below let the .cu sources
// keep their cuComplex / cuDoubleComplex / make_cu* / cuC* spellings unchanged.

#pragma once
#include <hip/hip_complex.h>

typedef hipDoubleComplex cuDoubleComplex;
typedef hipFloatComplex  cuComplex;

#define make_cuDoubleComplex make_hipDoubleComplex
#define make_cuComplex       make_hipFloatComplex

// The kernels call cuCreal/cuCimag/cuCadd/cuCsub only on cuDoubleComplex, so
// alias to HIP's double-precision helpers (hipC*), not the float (hipC*f) set.
#define cuCreal cuCreal_double
#define cuCimag cuCimag_double
#define cuCadd  cuCadd_double
#define cuCsub  cuCsub_double
#define cuCmul  cuCmul_double
#define cuConj  cuConj_double

static __host__ __device__ inline double cuCreal_double(hipDoubleComplex z) { return hipCreal(z); }
static __host__ __device__ inline double cuCimag_double(hipDoubleComplex z) { return hipCimag(z); }
static __host__ __device__ inline hipDoubleComplex cuCadd_double(hipDoubleComplex a, hipDoubleComplex b) { return hipCadd(a, b); }
static __host__ __device__ inline hipDoubleComplex cuCsub_double(hipDoubleComplex a, hipDoubleComplex b) { return hipCsub(a, b); }
static __host__ __device__ inline hipDoubleComplex cuCmul_double(hipDoubleComplex a, hipDoubleComplex b) { return hipCmul(a, b); }
static __host__ __device__ inline hipDoubleComplex cuConj_double(hipDoubleComplex z) { return hipConj(z); }
