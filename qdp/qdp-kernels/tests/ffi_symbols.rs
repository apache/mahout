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

//! Pins the public FFI surface of `qdp-kernels`.
//!
//! Every exported kernel launcher is bound *by name* to a function pointer
//! with its exact C signature. If a symbol is renamed, dropped, or its
//! signature drifts, this file stops compiling. It compiles in both cfg
//! branches: with CUDA the names resolve to `unsafe extern "C"`
//! declarations; without CUDA (`qdp_no_cuda`) they resolve to safe stubs,
//! which coerce to the same `unsafe extern "C" fn` pointer type.

use qdp_kernels::*;
use std::ffi::c_void;

type V = *mut c_void;

/// Every exported symbol paired with its C name.
fn ffi_surface() -> Vec<(&'static str, usize)> {
    macro_rules! sym {
        ($name:ident: fn($($ty:ty),* $(,)?)) => {{
            let f: unsafe extern "C" fn($($ty),*) -> i32 = $name;
            (stringify!($name), f as usize)
        }};
    }
    vec![
        sym!(launch_amplitude_encode: fn(*const f64, V, usize, usize, f64, V)),
        sym!(launch_amplitude_encode_f32: fn(*const f32, V, usize, usize, f32, V)),
        sym!(launch_amplitude_encode_batch: fn(*const f64, V, *const f64, usize, usize, usize, V)),
        sym!(launch_amplitude_encode_batch_f32: fn(*const f32, V, *const f32, usize, usize, usize, V)),
        sym!(launch_l2_norm: fn(*const f64, usize, *mut f64, V)),
        sym!(launch_l2_norm_f32: fn(*const f32, usize, *mut f32, V)),
        sym!(launch_l2_norm_batch: fn(*const f64, usize, usize, *mut f64, V)),
        sym!(launch_l2_norm_batch_f32: fn(*const f32, usize, usize, *mut f32, V)),
        sym!(convert_state_to_float: fn(*const CuDoubleComplex, *mut CuComplex, usize, V)),
        sym!(convert_state_to_double: fn(*const CuComplex, *mut CuDoubleComplex, usize, V)),
        sym!(launch_basis_encode: fn(usize, V, usize, V)),
        sym!(launch_basis_encode_f32: fn(usize, V, usize, V)),
        sym!(launch_basis_encode_batch: fn(*const usize, V, usize, usize, u32, V)),
        sym!(launch_basis_encode_batch_f32: fn(*const usize, V, usize, usize, u32, V)),
        sym!(launch_angle_encode: fn(*const f64, V, usize, u32, V)),
        sym!(launch_angle_encode_f32: fn(*const f32, V, usize, u32, V)),
        sym!(launch_angle_encode_batch: fn(*const f64, V, usize, usize, u32, V)),
        sym!(launch_angle_encode_batch_f32: fn(*const f32, V, usize, usize, u32, V)),
        sym!(launch_check_finite_batch_f32: fn(*const f32, usize, *mut i32, V)),
        sym!(launch_check_finite_batch_f64: fn(*const f64, usize, *mut i32, V)),
        sym!(launch_validate_and_cast_basis_indices_f32: fn(*const f32, usize, usize, *mut usize, *mut i32, V)),
        sym!(launch_check_basis_indices_usize: fn(*const usize, usize, usize, *mut i32, V)),
        sym!(launch_iqp_encode: fn(*const f64, V, usize, u32, i32, V)),
        sym!(launch_iqp_encode_batch: fn(*const f64, V, usize, usize, u32, u32, i32, V)),
        sym!(launch_phase_encode: fn(*const f64, V, usize, u32, V)),
        sym!(launch_phase_encode_batch: fn(*const f64, V, usize, usize, u32, V)),
    ]
}

#[test]
fn ffi_surface_has_26_distinct_symbols() {
    let surface = ffi_surface();
    assert_eq!(surface.len(), 26);
    let mut names: Vec<_> = surface.iter().map(|(n, _)| *n).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), 26, "duplicate symbol name in surface list");
}

/// In a no-CUDA build every launcher is a safe stub that must report
/// "CUDA unavailable" (999) without touching its (null) arguments.
#[test]
#[cfg(qdp_no_cuda)]
fn stubs_return_cuda_unavailable() {
    let n = std::ptr::null_mut::<c_void>();
    assert_eq!(
        launch_amplitude_encode(std::ptr::null(), n, 0, 0, 1.0, n),
        999
    );
    assert_eq!(
        launch_amplitude_encode_f32(std::ptr::null(), n, 0, 0, 1.0, n),
        999
    );
    assert_eq!(launch_basis_encode(0, n, 0, n), 999);
    assert_eq!(launch_basis_encode_f32(0, n, 0, n), 999);
    assert_eq!(
        launch_angle_encode_batch(std::ptr::null(), n, 0, 0, 0, n),
        999
    );
    assert_eq!(
        launch_angle_encode_batch_f32(std::ptr::null(), n, 0, 0, 0, n),
        999
    );
    assert_eq!(
        launch_check_finite_batch_f64(std::ptr::null(), 0, std::ptr::null_mut(), n),
        999
    );
    assert_eq!(
        launch_check_finite_batch_f32(std::ptr::null(), 0, std::ptr::null_mut(), n),
        999
    );
    assert_eq!(
        convert_state_to_float(std::ptr::null(), std::ptr::null_mut(), 0, n),
        999
    );
    assert_eq!(
        launch_iqp_encode_batch(std::ptr::null(), n, 0, 0, 0, 0, 1, n),
        999
    );
    assert_eq!(launch_phase_encode(std::ptr::null(), n, 0, 0, n), 999);
}
