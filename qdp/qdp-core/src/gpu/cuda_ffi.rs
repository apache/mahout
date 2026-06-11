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
// Portions Copyright (c) 2026 Advanced Micro Devices, Inc.
// Author: Jeff Daily <jeff.daily@amd.com>

//! Centralized GPU runtime API FFI declarations.
//!
//! These are the runtime entry points the pinned-memory pool, OOM guard, and
//! dual-stream pipeline call directly (outside the cudarc/`gpu_rt` slice). The
//! public function names keep their `cuda*` spelling so call sites are
//! unchanged across vendors. On the default `cuda` feature they bind libcudart
//! directly; on the `hip` feature they are thin wrappers over the matching
//! libamdhip64 entry points (which are 1:1, and whose status codes match
//! CUDA's numerically for the codes used here).

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

// ---- CUDA backend: bind libcudart directly ----
#[cfg(all(feature = "cuda", not(feature = "hip")))]
pub(crate) use cuda_rt::*;

#[cfg(all(feature = "cuda", not(feature = "hip")))]
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

// ---- HIP backend: bind libamdhip64, expose the same cuda* names ----
#[cfg(feature = "hip")]
pub(crate) use hip_rt::*;

// The wrapper functions deliberately keep the cuda* spelling so call sites are
// vendor-agnostic; suppress the snake_case lint for that intentional naming.
#[cfg(feature = "hip")]
#[allow(non_snake_case)]
mod hip_rt {
    use super::CudaPointerAttributes;
    use std::ffi::c_void;

    // hipPointerAttribute_t has the same first fields we read (memoryType,
    // device, devicePointer, hostPointer); a #[repr(C)] alias suffices since we
    // only read memory_type/device.
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
        // hipPointerAttribute_t leads with the same memoryType/device/pointers
        // fields we read; reinterpret the destination accordingly.
        unsafe { hipPointerGetAttributes(attributes as *mut c_void, ptr) }
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
