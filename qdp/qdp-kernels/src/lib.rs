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

/// Return code of every no-CUDA stub ("CUDA unavailable"); qdp-core's
/// `cuda_error_to_string` maps it to a readable message.
#[cfg(any(not(target_os = "linux"), qdp_no_cuda))]
const CUDA_UNAVAILABLE: i32 = 999;

/// Declares one CUDA launcher (or an f64/f32 pair) for *both* build modes.
///
/// * CUDA build (`target_os = "linux"` and not `qdp_no_cuda`): an
///   `unsafe extern "C"` declaration resolved against the nvcc-built
///   `libkernels.a`.
/// * No-CUDA build: a safe `#[unsafe(no_mangle)] extern "C"` stub with the
///   same symbol name that returns [`CUDA_UNAVAILABLE`], so dependents link.
///
/// Forms:
/// * `pub fn name(args);` - a single symbol written with concrete types
///   (f64-only kernels such as IQP and phase, converters, validators).
/// * `pub fn f64_name / f32_name as Alias<Real>(args);` - a precision pair.
///   `Real` in `args` is `f64` for the first symbol and `f32` for the second,
///   and `Alias<T>` names the shared `unsafe extern "C"` fn-pointer type
///   (used by [`KernelElem`]).
/// * `pub fn f64_name / f32_name as Alias(args);` - a pair whose signature
///   does not mention `Real` (basis encoding: only the output differs).
macro_rules! define_stub {
    (
        $(#[$doc:meta])*
        pub fn $f64:ident / $f32:ident as $alias:ident<Real>(
            $($arg:ident: $ty:ty),* $(,)?
        );
    ) => {
        define_stub!(@pair [$(#[$doc])*] $f64 / $f32 ($($arg: $ty),*));
        #[doc = concat!("Signature shared by [`", stringify!($f64), "`] (`Real = f64`) and [`", stringify!($f32), "`] (`Real = f32`).")]
        pub type $alias<Real> = unsafe extern "C" fn($($ty),*) -> i32;
    };
    (
        $(#[$doc:meta])*
        pub fn $f64:ident / $f32:ident as $alias:ident(
            $($arg:ident: $ty:ty),* $(,)?
        );
    ) => {
        define_stub!(@pair [$(#[$doc])*] $f64 / $f32 ($($arg: $ty),*));
        #[doc = concat!("Signature shared by [`", stringify!($f64), "`] and [`", stringify!($f32), "`].")]
        pub type $alias = unsafe extern "C" fn($($ty),*) -> i32;
    };
    (
        $(#[$doc:meta])*
        pub fn $name:ident($($arg:ident: $ty:ty),* $(,)?);
    ) => {
        #[cfg(all(target_os = "linux", not(qdp_no_cuda)))]
        unsafe extern "C" {
            $(#[$doc])*
            pub fn $name($($arg: $ty),*) -> i32;
        }

        #[cfg(any(not(target_os = "linux"), qdp_no_cuda))]
        $(#[$doc])*
        #[unsafe(no_mangle)]
        #[allow(unused_variables)]
        pub extern "C" fn $name($($arg: $ty),*) -> i32 {
            $crate::CUDA_UNAVAILABLE
        }
    };
    (@pair [$(#[$doc:meta])*] $f64:ident / $f32:ident ($($arg:ident: $ty:ty),*)) => {
        define_stub!(@real f64, [$(#[$doc])*] $f64($($arg: $ty),*));
        define_stub!(
            @real f32,
            [$(#[$doc])* #[doc = ""] #[doc = "Single-precision (`f32` / `CuComplex`) variant."]]
            $f32($($arg: $ty),*)
        );
    };
    // Each symbol lives in a private module so `Real` can be aliased to a
    // different type per symbol; the fn is re-exported at the crate root,
    // so the Rust path and the (unmangled) C symbol name are unchanged.
    (@real $real:ty, [$(#[$doc:meta])*] $name:ident($($arg:ident: $ty:ty),*)) => {
        mod $name {
            #[allow(unused_imports)]
            use super::*;
            #[allow(dead_code)]
            type Real = $real;
            define_stub! {
                $(#[$doc])*
                pub fn $name($($arg: $ty),*);
            }
        }
        pub use self::$name::$name;
    };
}

mod elem;
pub use elem::KernelElem;

define_stub! {
    /// Launch amplitude encoding kernel.
    /// Returns CUDA error code (0 = success).
    ///
    /// # Safety
    /// Requires valid GPU pointers, must sync before freeing.
    pub fn launch_amplitude_encode / launch_amplitude_encode_f32 as AmplitudeEncodeFn<Real>(
        input_d: *const Real,
        state_d: *mut c_void,
        input_len: usize,
        state_len: usize,
        inv_norm: Real,
        stream: *mut c_void,
    );
}

define_stub! {
    /// Launch batch amplitude encoding kernel.
    /// Returns CUDA error code (0 = success).
    ///
    /// # Safety
    /// Requires valid GPU pointers, must sync before freeing.
    pub fn launch_amplitude_encode_batch / launch_amplitude_encode_batch_f32
        as AmplitudeEncodeBatchFn<Real>(
        input_batch_d: *const Real,
        state_batch_d: *mut c_void,
        inv_norms_d: *const Real,
        num_samples: usize,
        input_len: usize,
        state_len: usize,
        stream: *mut c_void,
    );
}

define_stub! {
    /// Launch L2 norm reduction (returns inverse norm).
    /// Returns CUDA error code (0 = success).
    ///
    /// # Safety
    /// Pointers must reference valid device memory on the provided stream.
    pub fn launch_l2_norm / launch_l2_norm_f32 as L2NormFn<Real>(
        input_d: *const Real,
        input_len: usize,
        inv_norm_out_d: *mut Real,
        stream: *mut c_void,
    );
}

define_stub! {
    /// Launch batched L2 norm reduction (returns inverse norms per sample).
    /// Returns CUDA error code (0 = success).
    ///
    /// # Safety
    /// Pointers must reference valid device memory on the provided stream.
    pub fn launch_l2_norm_batch / launch_l2_norm_batch_f32 as L2NormBatchFn<Real>(
        input_batch_d: *const Real,
        num_samples: usize,
        sample_len: usize,
        inv_norms_out_d: *mut Real,
        stream: *mut c_void,
    );
}

define_stub! {
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
    );
}

define_stub! {
    /// Convert a complex64 state vector to complex128 on GPU.
    /// Returns CUDA error code (0 = success).
    ///
    /// # Safety
    /// Pointers must reference valid device memory on the provided stream.
    pub fn convert_state_to_double(
        input_state_d: *const CuComplex,
        output_state_d: *mut CuDoubleComplex,
        len: usize,
        stream: *mut c_void,
    );
}

define_stub! {
    /// Launch basis encoding kernel.
    /// Maps an integer index to a computational basis state.
    /// Returns CUDA error code (0 = success).
    ///
    /// # Safety
    /// Requires valid GPU pointer; must sync before freeing.
    pub fn launch_basis_encode / launch_basis_encode_f32 as BasisEncodeFn(
        basis_index: usize,
        state_d: *mut c_void,
        state_len: usize,
        stream: *mut c_void,
    );
}

define_stub! {
    /// Launch batch basis encoding kernel.
    /// Returns CUDA error code (0 = success).
    ///
    /// # Safety
    /// Requires valid GPU pointers; must sync before freeing.
    pub fn launch_basis_encode_batch / launch_basis_encode_batch_f32 as BasisEncodeBatchFn(
        basis_indices_d: *const usize,
        state_batch_d: *mut c_void,
        num_samples: usize,
        state_len: usize,
        num_qubits: u32,
        stream: *mut c_void,
    );
}

define_stub! {
    /// Launch angle encoding kernel.
    /// Returns CUDA error code (0 = success).
    ///
    /// # Safety
    /// Requires valid GPU pointers; must sync before freeing.
    pub fn launch_angle_encode / launch_angle_encode_f32 as AngleEncodeFn<Real>(
        angles_d: *const Real,
        state_d: *mut c_void,
        state_len: usize,
        num_qubits: u32,
        stream: *mut c_void,
    );
}

define_stub! {
    /// Launch batch angle encoding kernel.
    /// Returns CUDA error code (0 = success).
    ///
    /// # Safety
    /// Requires valid GPU pointers; must sync before freeing.
    pub fn launch_angle_encode_batch / launch_angle_encode_batch_f32
        as AngleEncodeBatchFn<Real>(
        angles_batch_d: *const Real,
        state_batch_d: *mut c_void,
        num_samples: usize,
        state_len: usize,
        num_qubits: u32,
        stream: *mut c_void,
    );
}

define_stub! {
    /// Launch batch finite-value validation (NaN/Inf check).
    /// Returns CUDA error code (0 = success).
    ///
    /// # Safety
    /// Requires valid GPU pointers, must sync before reading the output flag.
    pub fn launch_check_finite_batch_f64 / launch_check_finite_batch_f32
        as CheckFiniteBatchFn<Real>(
        input_batch_d: *const Real,
        total_values: usize,
        has_non_finite_d: *mut i32,
        stream: *mut c_void,
    );
}

define_stub! {
    /// Launch combined validation + cast for float32 basis indices.
    /// `error_flags_d` receives a bitmask (0 = all indices valid); `indices_out_d`
    /// receives the truncated `size_t` indices (0 for invalid slots).
    ///
    /// # Safety
    /// Requires valid GPU pointers, must sync before reading the output flag.
    pub fn launch_validate_and_cast_basis_indices_f32(
        input_batch_d: *const f32,
        num_samples: usize,
        state_len: usize,
        indices_out_d: *mut usize,
        error_flags_d: *mut i32,
        stream: *mut c_void,
    );
}

define_stub! {
    /// Launch bounds-check for an existing `size_t` basis-index buffer.
    ///
    /// # Safety
    /// Requires valid GPU pointers, must sync before reading the output flag.
    pub fn launch_check_basis_indices_usize(
        indices_d: *const usize,
        num_samples: usize,
        state_len: usize,
        error_flags_d: *mut i32,
        stream: *mut c_void,
    );
}

// IQP and phase encoding only have f64 kernels.

define_stub! {
    /// Launch IQP encoding kernel.
    /// Returns CUDA error code (0 = success).
    ///
    /// # Safety
    /// Requires valid GPU pointers, must sync before freeing.
    pub fn launch_iqp_encode(
        data_d: *const f64,
        state_d: *mut c_void,
        state_len: usize,
        num_qubits: u32,
        enable_zz: i32,
        stream: *mut c_void,
    );
}

define_stub! {
    /// Launch batch IQP encoding kernel.
    /// Returns CUDA error code (0 = success).
    ///
    /// # Safety
    /// Requires valid GPU pointers, must sync before freeing.
    pub fn launch_iqp_encode_batch(
        data_batch_d: *const f64,
        state_batch_d: *mut c_void,
        num_samples: usize,
        state_len: usize,
        num_qubits: u32,
        data_len: u32,
        enable_zz: i32,
        stream: *mut c_void,
    );
}

define_stub! {
    /// Launch phase encoding kernel.
    /// Returns CUDA error code (0 = success).
    ///
    /// # Safety
    /// Requires valid GPU pointers, must sync before freeing.
    pub fn launch_phase_encode(
        phases_d: *const f64,
        state_d: *mut c_void,
        state_len: usize,
        num_qubits: u32,
        stream: *mut c_void,
    );
}

define_stub! {
    /// Launch batch phase encoding kernel.
    /// Returns CUDA error code (0 = success).
    ///
    /// # Safety
    /// Requires valid GPU pointers, must sync before freeing.
    pub fn launch_phase_encode_batch(
        phases_batch_d: *const f64,
        state_batch_d: *mut c_void,
        num_samples: usize,
        state_len: usize,
        num_qubits: u32,
        stream: *mut c_void,
    );
}
