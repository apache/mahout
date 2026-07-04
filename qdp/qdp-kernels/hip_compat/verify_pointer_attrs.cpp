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

// Build-time verification that the hand-rolled Rust mirror of
// hipPointerAttribute_t / hipMemoryType in qdp-core/src/gpu/cuda_ffi.rs matches
// the installed ROCm headers.
//
// cudaPointerGetAttributes (HIP path) calls hipPointerGetAttributes, which fills
// a Rust-side #[repr(C)] struct whose layout is assumed here, and reads the
// hipMemoryType enum by numeric value. Those field offsets and enum values have
// differed across ROCm major versions, so a silent header change would corrupt
// the pointer-attribute reads validate_cuda_input_ptr depends on (either
// rejecting valid device pointers or, worse, waving a host pointer through).
//
// hipcc compiles this translation unit as part of the kernel build; if the
// installed hip_runtime_api.h ever diverges from what the Rust FFI assumes, the
// static_asserts below fail and the build stops with a clear message instead of
// miscompiling. The port targets ROCm >= 6.0 (see DEVELOPMENT.md); the `type`
// field name and the Device=2 / Managed=3 enum values are that convention.

#include <hip/hip_runtime.h>
#include <cstddef>

static_assert(static_cast<int>(hipMemoryTypeDevice) == 2,
              "hipMemoryTypeDevice changed: update HIP_MEMORY_TYPE_DEVICE in qdp-core/src/gpu/cuda_ffi.rs");
static_assert(static_cast<int>(hipMemoryTypeManaged) == 3,
              "hipMemoryTypeManaged changed: update HIP_MEMORY_TYPE_MANAGED in qdp-core/src/gpu/cuda_ffi.rs");

static_assert(offsetof(hipPointerAttribute_t, type) == 0,
              "hipPointerAttribute_t::type moved: update HipPointerAttributes mirror in qdp-core/src/gpu/cuda_ffi.rs");
static_assert(offsetof(hipPointerAttribute_t, device) == 4,
              "hipPointerAttribute_t::device moved: update HipPointerAttributes mirror in qdp-core/src/gpu/cuda_ffi.rs");
static_assert(offsetof(hipPointerAttribute_t, devicePointer) == 8,
              "hipPointerAttribute_t::devicePointer moved: update HipPointerAttributes mirror in qdp-core/src/gpu/cuda_ffi.rs");
static_assert(offsetof(hipPointerAttribute_t, hostPointer) == 16,
              "hipPointerAttribute_t::hostPointer moved: update HipPointerAttributes mirror in qdp-core/src/gpu/cuda_ffi.rs");
static_assert(offsetof(hipPointerAttribute_t, isManaged) == 24,
              "hipPointerAttribute_t::isManaged moved: update HipPointerAttributes mirror in qdp-core/src/gpu/cuda_ffi.rs");
static_assert(offsetof(hipPointerAttribute_t, allocationFlags) == 28,
              "hipPointerAttribute_t::allocationFlags moved: update HipPointerAttributes mirror in qdp-core/src/gpu/cuda_ffi.rs");
static_assert(sizeof(hipPointerAttribute_t) <= 32,
              "hipPointerAttribute_t larger than the 32-byte Rust HipPointerAttributes buffer: hipPointerGetAttributes would overflow it");
