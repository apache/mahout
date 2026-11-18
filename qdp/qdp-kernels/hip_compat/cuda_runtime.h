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

// HIP forwarding shim for <cuda_runtime.h>.
//
// This file exists ONLY on the HIP build path: qdp-kernels/build.rs adds the
// hip_compat/ directory to the include search path exclusively when compiling
// with hipcc, so a CUDA build never sees it and pulls the real toolkit header
// instead. The .cu sources keep their original `#include <cuda_runtime.h>`
// spelling; this header maps the small set of cuda* runtime symbols the
// kernels reference to their hip* equivalents (HIP error codes match CUDA's
// numerically for these codes).

#pragma once
#include <hip/hip_runtime.h>

// MSVC <math.h> does not define POSIX math constants unless _USE_MATH_DEFINES
// is set before the first system include. Provide the one the kernels use.
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.7071067811865475244008443621
#endif

#define cudaError_t              hipError_t
#define cudaSuccess              hipSuccess
#define cudaErrorInvalidValue    hipErrorInvalidValue
#define cudaStream_t             hipStream_t
#define cudaGetLastError         hipGetLastError
#define cudaGetDevice            hipGetDevice
#define cudaDeviceGetAttribute   hipDeviceGetAttribute
#define cudaDevAttrMaxGridDimX   hipDeviceAttributeMaxGridDimX
#define cudaMemsetAsync          hipMemsetAsync
#define cudaMalloc               hipMalloc
