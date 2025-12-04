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

// FFI interface for CUDA kernels
// Kernels in .cu files, compiled via build.rs
// Dummy implementations provided for non-CUDA platforms

use std::ffi::c_void;

// Complex number (matches CUDA's cuDoubleComplex)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuDoubleComplex {
    pub x: f64,  // Real part
    pub y: f64,  // Imaginary part
}

// Implement DeviceRepr for cudarc compatibility
#[cfg(target_os = "linux")]
unsafe impl cudarc::driver::DeviceRepr for CuDoubleComplex {}

// Also implement ValidAsZeroBits for alloc_zeros support
#[cfg(target_os = "linux")]
unsafe impl cudarc::driver::ValidAsZeroBits for CuDoubleComplex {}

// CUDA kernel FFI (Linux only, dummy on other platforms)
#[cfg(target_os = "linux")]
unsafe extern "C" {
    /// Launch amplitude encoding kernel
    /// Returns CUDA error code (0 = success)
    ///
    /// # Safety
    /// Requires valid GPU pointers, must sync before freeing
    pub fn launch_amplitude_encode(
        input_d: *const f64,
        state_d: *mut c_void,
        input_len: usize,
        state_len: usize,
        norm: f64,
        stream: *mut c_void,
    ) -> i32;

    // TODO: launch_angle_encode, launch_basis_encode
}

// Dummy implementation for non-Linux (allows compilation)
#[cfg(not(target_os = "linux"))]
#[unsafe(no_mangle)]
pub extern "C" fn launch_amplitude_encode(
    _input_d: *const f64,
    _state_d: *mut c_void,
    _input_len: usize,
    _state_len: usize,
    _norm: f64,
    _stream: *mut c_void,
) -> i32 {
    999 // Error: CUDA unavailable
}
