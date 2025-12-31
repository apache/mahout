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

//! Centralized CUDA Runtime API FFI declarations.

use std::ffi::c_void;

pub(crate) const CUDA_MEMCPY_HOST_TO_DEVICE: u32 = 1;
pub(crate) const CUDA_EVENT_DISABLE_TIMING: u32 = 0x02;

unsafe extern "C" {
    pub(crate) fn cudaHostAlloc(pHost: *mut *mut c_void, size: usize, flags: u32) -> i32;
    pub(crate) fn cudaFreeHost(ptr: *mut c_void) -> i32;

    pub(crate) fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;

    pub(crate) fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: u32,
        stream: *mut c_void,
    ) -> i32;

    pub(crate) fn cudaEventCreateWithFlags(event: *mut *mut c_void, flags: u32) -> i32;
    pub(crate) fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    pub(crate) fn cudaEventDestroy(event: *mut c_void) -> i32;
    pub(crate) fn cudaStreamWaitEvent(stream: *mut c_void, event: *mut c_void, flags: u32) -> i32;
    pub(crate) fn cudaStreamSynchronize(stream: *mut c_void) -> i32;

    pub(crate) fn cudaMemsetAsync(
        devPtr: *mut c_void,
        value: i32,
        count: usize,
        stream: *mut c_void,
    ) -> i32;
}
