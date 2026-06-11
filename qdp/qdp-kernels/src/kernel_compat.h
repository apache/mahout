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

// Cross-vendor kernel compatibility shims.
//
// Included by the kernel TUs that use warp intrinsics (amplitude.cu). On CUDA
// it is inert. On HIP it supplies the one warp-intrinsic difference that does
// not translate 1:1: the full-warp lane mask for __shfl_*_sync.

#ifndef KERNEL_COMPAT_H
#define KERNEL_COMPAT_H

#if defined(__HIP_PLATFORM_AMD__)
// ROCm's __shfl_*_sync static_asserts a 64-bit mask (sizeof(MaskT) == 8): the
// 32-bit literal 0xffffffff every CUDA warp-sync uses fails to COMPILE,
// independent of the active wave width. Use an all-lanes 64-bit mask.
#define QDP_FULL_WARP_MASK 0xffffffffffffffffULL
#else
#define QDP_FULL_WARP_MASK 0xffffffffu
#endif

#endif // KERNEL_COMPAT_H
