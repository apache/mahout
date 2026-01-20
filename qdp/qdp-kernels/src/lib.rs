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
    pub x: f64, // Real part
    pub y: f64, // Imaginary part
}

// Implement DeviceRepr for cudarc compatibility
#[cfg(target_os = "linux")]
unsafe impl cudarc::driver::DeviceRepr for CuDoubleComplex {}

// Also implement ValidAsZeroBits for alloc_zeros support
#[cfg(target_os = "linux")]
unsafe impl cudarc::driver::ValidAsZeroBits for CuDoubleComplex {}

// Complex number (matches CUDA's cuComplex / cuFloatComplex)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuComplex {
    pub x: f32, // Real part
    pub y: f32, // Imaginary part
}

// Implement DeviceRepr for cudarc compatibility
#[cfg(target_os = "linux")]
unsafe impl cudarc::driver::DeviceRepr for CuComplex {}

// Also implement ValidAsZeroBits for alloc_zeros support
#[cfg(target_os = "linux")]
unsafe impl cudarc::driver::ValidAsZeroBits for CuComplex {}

// CUDA kernel FFI (Linux only; stubbed when built without nvcc/CUDA)
#[cfg(all(target_os = "linux", not(qdp_no_cuda)))]
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
        inv_norm: f64,
        stream: *mut c_void,
    ) -> i32;

    /// Launch amplitude encoding kernel (float32 input/output)
    /// Returns CUDA error code (0 = success)
    ///
    /// # Safety
    /// Requires valid GPU pointers, must sync before freeing
    pub fn launch_amplitude_encode_f32(
        input_d: *const f32,
        state_d: *mut c_void,
        input_len: usize,
        state_len: usize,
        inv_norm: f32,
        stream: *mut c_void,
    ) -> i32;

    /// Launch batch amplitude encoding kernel
    /// Returns CUDA error code (0 = success)
    ///
    /// # Safety
    /// Requires valid GPU pointers, must sync before freeing
    pub fn launch_amplitude_encode_batch(
        input_batch_d: *const f64,
        state_batch_d: *mut c_void,
        inv_norms_d: *const f64,
        num_samples: usize,
        input_len: usize,
        state_len: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Launch L2 norm reduction (returns inverse norm)
    /// Returns CUDA error code (0 = success)
    ///
    /// # Safety
    /// Pointers must reference valid device memory on the provided stream.
    pub fn launch_l2_norm(
        input_d: *const f64,
        input_len: usize,
        inv_norm_out_d: *mut f64,
        stream: *mut c_void,
    ) -> i32;

    /// Launch batched L2 norm reduction (returns inverse norms per sample)
    /// Returns CUDA error code (0 = success)
    ///
    /// # Safety
    /// Pointers must reference valid device memory on the provided stream.
    pub fn launch_l2_norm_batch(
        input_batch_d: *const f64,
        num_samples: usize,
        sample_len: usize,
        inv_norms_out_d: *mut f64,
        stream: *mut c_void,
    ) -> i32;

    /// Convert a complex128 state vector to complex64 on GPU.
    /// Returns CUDA error code (0 = success).
    ///
    /// # Safety
    /// Pointers must reference valid device memory on the provided stream.
    pub fn convert_state_to_float(
        input_state_d: *const CuDoubleComplex,
        output_state_d: *mut CuComplex,
        len: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Launch basis encoding kernel
    /// Maps an integer index to a computational basis state.
    /// Returns CUDA error code (0 = success)
    ///
    /// # Safety
    /// Requires valid GPU pointer, must sync before freeing
    pub fn launch_basis_encode(
        basis_index: usize,
        state_d: *mut c_void,
        state_len: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Launch batch basis encoding kernel
    /// Returns CUDA error code (0 = success)
    ///
    /// # Safety
    /// Requires valid GPU pointers, must sync before freeing
    pub fn launch_basis_encode_batch(
        basis_indices_d: *const usize,
        state_batch_d: *mut c_void,
        num_samples: usize,
        state_len: usize,
        num_qubits: u32,
        stream: *mut c_void,
    ) -> i32;

    // TODO: launch_angle_encode
}

// Dummy implementation for non-Linux and Linux builds without CUDA (allows linking)
#[cfg(any(not(target_os = "linux"), qdp_no_cuda))]
#[unsafe(no_mangle)]
pub extern "C" fn launch_amplitude_encode(
    _input_d: *const f64,
    _state_d: *mut c_void,
    _input_len: usize,
    _state_len: usize,
    _inv_norm: f64,
    _stream: *mut c_void,
) -> i32 {
    999 // Error: CUDA unavailable
}

#[cfg(any(not(target_os = "linux"), qdp_no_cuda))]
#[unsafe(no_mangle)]
pub extern "C" fn launch_amplitude_encode_f32(
    _input_d: *const f32,
    _state_d: *mut c_void,
    _input_len: usize,
    _state_len: usize,
    _inv_norm: f32,
    _stream: *mut c_void,
) -> i32 {
    999
}

#[cfg(any(not(target_os = "linux"), qdp_no_cuda))]
#[unsafe(no_mangle)]
pub extern "C" fn launch_amplitude_encode_batch(
    _input_batch_d: *const f64,
    _state_batch_d: *mut c_void,
    _inv_norms_d: *const f64,
    _num_samples: usize,
    _input_len: usize,
    _state_len: usize,
    _stream: *mut c_void,
) -> i32 {
    999
}

#[cfg(any(not(target_os = "linux"), qdp_no_cuda))]
#[unsafe(no_mangle)]
pub extern "C" fn launch_l2_norm(
    _input_d: *const f64,
    _input_len: usize,
    _inv_norm_out_d: *mut f64,
    _stream: *mut c_void,
) -> i32 {
    999
}

#[cfg(any(not(target_os = "linux"), qdp_no_cuda))]
#[unsafe(no_mangle)]
pub extern "C" fn launch_l2_norm_batch(
    _input_batch_d: *const f64,
    _num_samples: usize,
    _sample_len: usize,
    _inv_norms_out_d: *mut f64,
    _stream: *mut c_void,
) -> i32 {
    999
}

#[cfg(any(not(target_os = "linux"), qdp_no_cuda))]
#[unsafe(no_mangle)]
pub extern "C" fn convert_state_to_float(
    _input_state_d: *const CuDoubleComplex,
    _output_state_d: *mut CuComplex,
    _len: usize,
    _stream: *mut c_void,
) -> i32 {
    999
}

#[cfg(any(not(target_os = "linux"), qdp_no_cuda))]
#[unsafe(no_mangle)]
pub extern "C" fn launch_basis_encode(
    _basis_index: usize,
    _state_d: *mut c_void,
    _state_len: usize,
    _stream: *mut c_void,
) -> i32 {
    999
}

#[cfg(any(not(target_os = "linux"), qdp_no_cuda))]
#[unsafe(no_mangle)]
pub extern "C" fn launch_basis_encode_batch(
    _basis_indices_d: *const usize,
    _state_batch_d: *mut c_void,
    _num_samples: usize,
    _state_len: usize,
    _num_qubits: u32,
    _stream: *mut c_void,
) -> i32 {
    999
}
