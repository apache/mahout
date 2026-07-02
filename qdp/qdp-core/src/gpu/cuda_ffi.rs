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

//! Centralized GPU runtime API FFI declarations.
//!
//! These are the runtime entry points the pinned-memory pool, OOM guard, and
//! dual-stream pipeline call directly (outside the cudarc/`gpu_rt` slice). The
//! public function names keep their `cuda*` spelling so call sites are
//! unchanged across backends. Exactly one implementation is selected at compile
//! time:
//!   * `cuda_rt` -- binds libcudart directly (default `cuda` feature, CUDA
//!     Toolkit present);
//!   * `no_cuda_stubs` -- returns a non-zero sentinel so the crate still links
//!     without the toolkit (`qdp_no_cuda`, set by the build script);
//!   * `hip_rt` -- thin wrappers over the matching libamdhip64 entry points
//!     (the `hip` feature), which are 1:1 with the CUDA runtime and whose status
//!     codes match CUDA's numerically for the codes used here.

use std::ffi::c_void;

pub(crate) const CUDA_MEMCPY_HOST_TO_DEVICE: u32 = 1;
#[allow(dead_code)]
pub(crate) const CUDA_MEMCPY_DEVICE_TO_HOST: u32 = 2;
pub(crate) const CUDA_EVENT_DISABLE_TIMING: u32 = 0x02;
pub(crate) const CUDA_EVENT_DEFAULT: u32 = 0x00;
#[allow(dead_code)]
pub(crate) const CUDA_MEMORY_TYPE_DEVICE: i32 = 2;
#[allow(dead_code)]
pub(crate) const CUDA_MEMORY_TYPE_MANAGED: i32 = 3;

#[allow(dead_code)]
#[repr(C)]
pub(crate) struct CudaPointerAttributes {
    pub memory_type: i32,
    pub device: i32,
    pub device_pointer: *mut c_void,
    pub host_pointer: *mut c_void,
    pub is_managed: i32,
    pub allocation_flags: u32,
}

// CUDA/HIP error codes (numerically identical for the codes used).
pub(crate) const CUDA_SUCCESS: i32 = 0;
#[allow(dead_code)]
pub(crate) const CUDA_ERROR_NOT_READY: i32 = 34;

// ---- CUDA backend: bind libcudart directly (toolkit present) ----
#[cfg(all(feature = "cuda", not(feature = "hip"), not(qdp_no_cuda)))]
pub(crate) use cuda_rt::*;

#[cfg(all(feature = "cuda", not(feature = "hip"), not(qdp_no_cuda)))]
mod cuda_rt {
    use super::CudaPointerAttributes;
    use std::ffi::c_void;

    unsafe extern "C" {
        pub(crate) fn cudaHostAlloc(pHost: *mut *mut c_void, size: usize, flags: u32) -> i32;
        pub(crate) fn cudaFreeHost(ptr: *mut c_void) -> i32;

        #[allow(dead_code)]
        pub(crate) fn cudaPointerGetAttributes(
            attributes: *mut CudaPointerAttributes,
            ptr: *const c_void,
        ) -> i32;

        pub(crate) fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;

        pub(crate) fn cudaMemcpyAsync(
            dst: *mut c_void,
            src: *const c_void,
            count: usize,
            kind: u32,
            stream: *mut c_void,
        ) -> i32;

        #[allow(dead_code)]
        pub(crate) fn cudaMemcpy(
            dst: *mut c_void,
            src: *const c_void,
            count: usize,
            kind: u32,
        ) -> i32;

        pub(crate) fn cudaEventCreateWithFlags(event: *mut *mut c_void, flags: u32) -> i32;
        pub(crate) fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
        pub(crate) fn cudaEventDestroy(event: *mut c_void) -> i32;
        pub(crate) fn cudaStreamWaitEvent(
            stream: *mut c_void,
            event: *mut c_void,
            flags: u32,
        ) -> i32;
        pub(crate) fn cudaStreamSynchronize(stream: *mut c_void) -> i32;

        pub(crate) fn cudaMemsetAsync(
            devPtr: *mut c_void,
            value: i32,
            count: usize,
            stream: *mut c_void,
        ) -> i32;

        #[allow(dead_code)]
        pub(crate) fn cudaEventQuery(event: *mut c_void) -> i32;
        pub(crate) fn cudaEventSynchronize(event: *mut c_void) -> i32;
        pub(crate) fn cudaEventElapsedTime(
            ms: *mut f32,
            start: *mut c_void,
            end: *mut c_void,
        ) -> i32;
    }
}

// ---------------------------------------------------------------------------
// Stub implementations when building the CUDA backend without the toolkit
// (`qdp_no_cuda`, set by the build script when nvcc is absent). Stubs return a
// non-zero sentinel so the crate links but any actual GPU call surfaces as a
// runtime error through the existing `if ret != 0 { Err(...) }` paths.
//
// Wrapped in a private module so a single `#[allow(non_snake_case)]` covers all
// stub names (they are camelCase to match the real CUDA Runtime API).
// ---------------------------------------------------------------------------
#[cfg(all(feature = "cuda", not(feature = "hip"), qdp_no_cuda))]
pub(crate) use no_cuda_stubs::*;

#[cfg(all(feature = "cuda", not(feature = "hip"), qdp_no_cuda))]
#[allow(non_snake_case)]
mod no_cuda_stubs {
    use super::CudaPointerAttributes;
    use std::ffi::c_void;

    /// Sentinel error code returned by stub CUDA Runtime calls.
    ///
    /// Matches the "999" convention used by qdp-kernels' kernel-launcher stubs.
    const QDP_CUDA_UNAVAILABLE: i32 = 999;

    pub(crate) unsafe fn cudaHostAlloc(_pHost: *mut *mut c_void, _size: usize, _flags: u32) -> i32 {
        QDP_CUDA_UNAVAILABLE
    }

    pub(crate) unsafe fn cudaFreeHost(_ptr: *mut c_void) -> i32 {
        QDP_CUDA_UNAVAILABLE
    }

    #[allow(dead_code)]
    pub(crate) unsafe fn cudaPointerGetAttributes(
        _attributes: *mut CudaPointerAttributes,
        _ptr: *const c_void,
    ) -> i32 {
        QDP_CUDA_UNAVAILABLE
    }

    pub(crate) unsafe fn cudaMemGetInfo(_free: *mut usize, _total: *mut usize) -> i32 {
        QDP_CUDA_UNAVAILABLE
    }

    pub(crate) unsafe fn cudaMemcpyAsync(
        _dst: *mut c_void,
        _src: *const c_void,
        _count: usize,
        _kind: u32,
        _stream: *mut c_void,
    ) -> i32 {
        QDP_CUDA_UNAVAILABLE
    }

    #[allow(dead_code)]
    pub(crate) unsafe fn cudaMemcpy(
        _dst: *mut c_void,
        _src: *const c_void,
        _count: usize,
        _kind: u32,
    ) -> i32 {
        QDP_CUDA_UNAVAILABLE
    }

    pub(crate) unsafe fn cudaEventCreateWithFlags(_event: *mut *mut c_void, _flags: u32) -> i32 {
        QDP_CUDA_UNAVAILABLE
    }

    pub(crate) unsafe fn cudaEventRecord(_event: *mut c_void, _stream: *mut c_void) -> i32 {
        QDP_CUDA_UNAVAILABLE
    }

    pub(crate) unsafe fn cudaEventDestroy(_event: *mut c_void) -> i32 {
        QDP_CUDA_UNAVAILABLE
    }

    pub(crate) unsafe fn cudaStreamWaitEvent(
        _stream: *mut c_void,
        _event: *mut c_void,
        _flags: u32,
    ) -> i32 {
        QDP_CUDA_UNAVAILABLE
    }

    pub(crate) unsafe fn cudaStreamSynchronize(_stream: *mut c_void) -> i32 {
        QDP_CUDA_UNAVAILABLE
    }

    pub(crate) unsafe fn cudaMemsetAsync(
        _devPtr: *mut c_void,
        _value: i32,
        _count: usize,
        _stream: *mut c_void,
    ) -> i32 {
        QDP_CUDA_UNAVAILABLE
    }

    #[allow(dead_code)]
    pub(crate) unsafe fn cudaEventQuery(_event: *mut c_void) -> i32 {
        QDP_CUDA_UNAVAILABLE
    }

    pub(crate) unsafe fn cudaEventSynchronize(_event: *mut c_void) -> i32 {
        QDP_CUDA_UNAVAILABLE
    }

    pub(crate) unsafe fn cudaEventElapsedTime(
        _ms: *mut f32,
        _start: *mut c_void,
        _end: *mut c_void,
    ) -> i32 {
        QDP_CUDA_UNAVAILABLE
    }
}

// ---- HIP backend: bind libamdhip64, expose the same cuda* names ----
#[cfg(feature = "hip")]
pub(crate) use hip_rt::*;

// The wrapper functions deliberately keep the cuda* spelling so call sites are
// vendor-agnostic; suppress the snake_case lint for that intentional naming.
#[cfg(feature = "hip")]
#[allow(non_snake_case)]
mod hip_rt {
    use super::{CUDA_MEMORY_TYPE_DEVICE, CUDA_MEMORY_TYPE_MANAGED, CudaPointerAttributes};
    use std::ffi::c_void;

    // hipMemoryType enum values are NOT guaranteed equal to CUDA's across ROCm
    // releases (older HIP used Host=0/Device=1; the hip_runtime_api.h note flags
    // this explicitly). So we read the real hipPointerAttribute_t and compare its
    // `type` field against the named hipMemoryType* constants rather than a magic
    // number, then translate to the CUDA convention the caller expects.
    //
    // Both these enum values and the HipPointerAttributes layout below are
    // verified against the installed ROCm headers at build time by
    // qdp-kernels/hip_compat/verify_pointer_attrs.cpp (compiled by hipcc), which
    // static_asserts hipMemoryTypeDevice/Managed and the hipPointerAttribute_t
    // field offsets/size. If a future ROCm release changes either, the build
    // fails loudly instead of silently misreading pointer attributes. The port
    // targets ROCm >= 6.0 (see DEVELOPMENT.md), where these are the convention.
    const HIP_MEMORY_TYPE_DEVICE: i32 = 2; // hipMemoryTypeDevice
    const HIP_MEMORY_TYPE_MANAGED: i32 = 3; // hipMemoryTypeManaged

    // Mirror of hipPointerAttribute_t (ROCm hip_runtime_api.h): the leading
    // `type` field is the hipMemoryType enum read by cudaPointerGetAttributes.
    // Layout guarded by the build-time static_assert check noted above.
    #[repr(C)]
    struct HipPointerAttributes {
        memory_type: i32,
        device: i32,
        device_pointer: *mut c_void,
        host_pointer: *mut c_void,
        is_managed: i32,
        allocation_flags: u32,
    }

    unsafe extern "C" {
        fn hipHostMalloc(ptr: *mut *mut c_void, size: usize, flags: u32) -> i32;
        fn hipHostFree(ptr: *mut c_void) -> i32;
        fn hipPointerGetAttributes(attributes: *mut c_void, ptr: *const c_void) -> i32;
        fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
        fn hipMemcpyAsync(
            dst: *mut c_void,
            src: *const c_void,
            count: usize,
            kind: u32,
            stream: *mut c_void,
        ) -> i32;
        fn hipMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: u32) -> i32;
        fn hipEventCreateWithFlags(event: *mut *mut c_void, flags: u32) -> i32;
        fn hipEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
        fn hipEventDestroy(event: *mut c_void) -> i32;
        fn hipStreamWaitEvent(stream: *mut c_void, event: *mut c_void, flags: u32) -> i32;
        fn hipStreamSynchronize(stream: *mut c_void) -> i32;
        fn hipMemsetAsync(dst: *mut c_void, value: i32, count: usize, stream: *mut c_void) -> i32;
        fn hipEventQuery(event: *mut c_void) -> i32;
        fn hipEventSynchronize(event: *mut c_void) -> i32;
        fn hipEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> i32;
    }

    // hipHostMallocDefault == 0, matching cudaHostAllocDefault used by callers.
    pub(crate) unsafe fn cudaHostAlloc(p: *mut *mut c_void, size: usize, flags: u32) -> i32 {
        unsafe { hipHostMalloc(p, size, flags) }
    }
    pub(crate) unsafe fn cudaFreeHost(ptr: *mut c_void) -> i32 {
        unsafe { hipHostFree(ptr) }
    }

    #[allow(dead_code)]
    pub(crate) unsafe fn cudaPointerGetAttributes(
        attributes: *mut CudaPointerAttributes,
        ptr: *const c_void,
    ) -> i32 {
        let mut hip_attrs = HipPointerAttributes {
            memory_type: 0,
            device: 0,
            device_pointer: std::ptr::null_mut(),
            host_pointer: std::ptr::null_mut(),
            is_managed: 0,
            allocation_flags: 0,
        };
        let ret = unsafe { hipPointerGetAttributes(&mut hip_attrs as *mut _ as *mut c_void, ptr) };
        if ret != 0 {
            return ret;
        }
        // Translate the hipMemoryType enum to the CUDA convention the caller
        // checks against, comparing the named hipMemoryType* values explicitly
        // (do not assume the numeric enum equals CUDA's). Anything else stays
        // verbatim so the caller's "not device memory" branch still fires.
        let memory_type = match hip_attrs.memory_type {
            HIP_MEMORY_TYPE_DEVICE => CUDA_MEMORY_TYPE_DEVICE,
            HIP_MEMORY_TYPE_MANAGED => CUDA_MEMORY_TYPE_MANAGED,
            other => other,
        };
        unsafe {
            *attributes = CudaPointerAttributes {
                memory_type,
                device: hip_attrs.device,
                device_pointer: hip_attrs.device_pointer,
                host_pointer: hip_attrs.host_pointer,
                is_managed: hip_attrs.is_managed,
                allocation_flags: hip_attrs.allocation_flags,
            };
        }
        0
    }

    pub(crate) unsafe fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32 {
        unsafe { hipMemGetInfo(free, total) }
    }

    // hipMemcpyAsync is the exact 1:1 of cudaMemcpyAsync: it enqueues on the
    // stream and returns without blocking the host, preserving the dual-stream
    // H2D/compute overlap. (hipMemcpyWithStream would synchronize the stream
    // before returning, serializing the pipeline.)
    pub(crate) unsafe fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: u32,
        stream: *mut c_void,
    ) -> i32 {
        unsafe { hipMemcpyAsync(dst, src, count, kind, stream) }
    }

    #[allow(dead_code)]
    pub(crate) unsafe fn cudaMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: u32,
    ) -> i32 {
        unsafe { hipMemcpy(dst, src, count, kind) }
    }

    pub(crate) unsafe fn cudaEventCreateWithFlags(event: *mut *mut c_void, flags: u32) -> i32 {
        unsafe { hipEventCreateWithFlags(event, flags) }
    }
    pub(crate) unsafe fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32 {
        unsafe { hipEventRecord(event, stream) }
    }
    pub(crate) unsafe fn cudaEventDestroy(event: *mut c_void) -> i32 {
        unsafe { hipEventDestroy(event) }
    }
    pub(crate) unsafe fn cudaStreamWaitEvent(
        stream: *mut c_void,
        event: *mut c_void,
        flags: u32,
    ) -> i32 {
        unsafe { hipStreamWaitEvent(stream, event, flags) }
    }
    pub(crate) unsafe fn cudaStreamSynchronize(stream: *mut c_void) -> i32 {
        unsafe { hipStreamSynchronize(stream) }
    }
    pub(crate) unsafe fn cudaMemsetAsync(
        dev_ptr: *mut c_void,
        value: i32,
        count: usize,
        stream: *mut c_void,
    ) -> i32 {
        unsafe { hipMemsetAsync(dev_ptr, value, count, stream) }
    }

    #[allow(dead_code)]
    pub(crate) unsafe fn cudaEventQuery(event: *mut c_void) -> i32 {
        unsafe { hipEventQuery(event) }
    }
    pub(crate) unsafe fn cudaEventSynchronize(event: *mut c_void) -> i32 {
        unsafe { hipEventSynchronize(event) }
    }
    pub(crate) unsafe fn cudaEventElapsedTime(
        ms: *mut f32,
        start: *mut c_void,
        end: *mut c_void,
    ) -> i32 {
        unsafe { hipEventElapsedTime(ms, start, end) }
    }
}
