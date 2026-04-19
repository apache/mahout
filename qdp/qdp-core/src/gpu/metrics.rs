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

//! Fidelity and trace distance metrics for quantum state validation.
//!
//! These utilities compare quantum states encoded at different precisions
//! or by different implementations.  All computations download GPU data
//! to the host and run on the CPU to produce a single scalar per sample.
//! They are intended for **testing and validation**, not the hot path.

#[cfg(target_os = "linux")]
use cudarc::driver::CudaDevice;
#[cfg(target_os = "linux")]
use std::sync::Arc;

#[cfg(target_os = "linux")]
use qdp_kernels::{CuComplex, CuDoubleComplex};

use crate::error::{MahoutError, Result};

/// Compute the state fidelity |⟨ψ|φ⟩|² between two complex state vectors
/// given as interleaved (re, im) f64 slices of equal length.
///
/// Both slices must have length `2 * state_dim` (re0, im0, re1, im1, ...).
/// Inputs must be normalized; the result is clamped to `[0, 1]` to absorb
/// floating-point rounding error.  Fidelity == 1 means identical states (up
/// to global phase).
pub fn fidelity_f64(state_a: &[f64], state_b: &[f64]) -> Result<f64> {
    if state_a.len() != state_b.len() {
        return Err(MahoutError::InvalidInput(format!(
            "fidelity: length mismatch ({} vs {})",
            state_a.len(),
            state_b.len()
        )));
    }
    if !state_a.len().is_multiple_of(2) {
        return Err(MahoutError::InvalidInput(
            "fidelity: length must be even (interleaved re/im pairs)".to_string(),
        ));
    }

    // ⟨ψ|φ⟩ = Σ_i conj(a_i) * b_i
    let mut re_acc = 0.0_f64;
    let mut im_acc = 0.0_f64;
    for i in (0..state_a.len()).step_by(2) {
        let a_re = state_a[i];
        let a_im = state_a[i + 1];
        let b_re = state_b[i];
        let b_im = state_b[i + 1];
        // conj(a) * b = (a_re - i*a_im)(b_re + i*b_im)
        //             = (a_re*b_re + a_im*b_im) + i*(a_re*b_im - a_im*b_re)
        re_acc += a_re * b_re + a_im * b_im;
        im_acc += a_re * b_im - a_im * b_re;
    }

    Ok((re_acc * re_acc + im_acc * im_acc).clamp(0.0, 1.0))
}

/// Compute fidelity from interleaved f32 data (promoted to f64 for accumulation).
pub fn fidelity_f32(state_a: &[f32], state_b: &[f32]) -> Result<f64> {
    if state_a.len() != state_b.len() {
        return Err(MahoutError::InvalidInput(format!(
            "fidelity_f32: length mismatch ({} vs {})",
            state_a.len(),
            state_b.len()
        )));
    }
    if !state_a.len().is_multiple_of(2) {
        return Err(MahoutError::InvalidInput(
            "fidelity_f32: length must be even (interleaved re/im pairs)".to_string(),
        ));
    }

    let mut re_acc = 0.0_f64;
    let mut im_acc = 0.0_f64;
    for i in (0..state_a.len()).step_by(2) {
        let a_re = state_a[i] as f64;
        let a_im = state_a[i + 1] as f64;
        let b_re = state_b[i] as f64;
        let b_im = state_b[i + 1] as f64;
        re_acc += a_re * b_re + a_im * b_im;
        im_acc += a_re * b_im - a_im * b_re;
    }

    Ok((re_acc * re_acc + im_acc * im_acc).clamp(0.0, 1.0))
}

/// Cross-precision fidelity: compare an f32 state against an f64 reference.
/// Both are interleaved (re, im) with the same number of complex elements.
pub fn fidelity_cross_precision(state_f32: &[f32], state_f64: &[f64]) -> Result<f64> {
    if state_f32.len() != state_f64.len() {
        return Err(MahoutError::InvalidInput(format!(
            "fidelity_cross_precision: length mismatch ({} vs {})",
            state_f32.len(),
            state_f64.len()
        )));
    }
    if !state_f32.len().is_multiple_of(2) {
        return Err(MahoutError::InvalidInput(
            "fidelity_cross_precision: length must be even".to_string(),
        ));
    }

    let mut re_acc = 0.0_f64;
    let mut im_acc = 0.0_f64;
    for i in (0..state_f32.len()).step_by(2) {
        let a_re = state_f32[i] as f64;
        let a_im = state_f32[i + 1] as f64;
        let b_re = state_f64[i];
        let b_im = state_f64[i + 1];
        re_acc += a_re * b_re + a_im * b_im;
        im_acc += a_re * b_im - a_im * b_re;
    }

    Ok((re_acc * re_acc + im_acc * im_acc).clamp(0.0, 1.0))
}

/// Trace distance between two pure states: √(1 − |⟨ψ|φ⟩|²).
/// Returns a value in [0, 1].  0 means identical, 1 means orthogonal.
pub fn trace_distance_f64(state_a: &[f64], state_b: &[f64]) -> Result<f64> {
    let f = fidelity_f64(state_a, state_b)?;
    Ok((1.0 - f.clamp(0.0, 1.0)).sqrt())
}

/// Trace distance for f32 states.
pub fn trace_distance_f32(state_a: &[f32], state_b: &[f32]) -> Result<f64> {
    let f = fidelity_f32(state_a, state_b)?;
    Ok((1.0 - f.clamp(0.0, 1.0)).sqrt())
}

/// Trace distance cross-precision (f32 vs f64).
pub fn trace_distance_cross_precision(state_f32: &[f32], state_f64: &[f64]) -> Result<f64> {
    let f = fidelity_cross_precision(state_f32, state_f64)?;
    Ok((1.0 - f.clamp(0.0, 1.0)).sqrt())
}

// ── GPU readback helpers (Linux/CUDA only) ──────────────────────────────

/// Download f64 complex GPU data to host as interleaved (re, im) f64 vec.
///
/// `gpu_ptr` must point to `num_elements` `CuDoubleComplex` values on device.
#[cfg(target_os = "linux")]
pub fn download_complex_f64(
    device: &Arc<CudaDevice>,
    gpu_ptr: *const CuDoubleComplex,
    num_elements: usize,
) -> Result<Vec<f64>> {
    if gpu_ptr.is_null() {
        return Err(MahoutError::InvalidInput(
            "download_complex_f64: null GPU pointer".to_string(),
        ));
    }

    let byte_count = num_elements * std::mem::size_of::<CuDoubleComplex>();
    let mut host_buf = vec![0.0_f64; num_elements * 2]; // interleaved re, im

    unsafe {
        let ret = cudarc::driver::sys::lib().cuMemcpyDtoH_v2(
            host_buf.as_mut_ptr() as *mut _,
            gpu_ptr as u64,
            byte_count,
        );
        if ret != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(MahoutError::Cuda(format!(
                "cuMemcpyDtoH failed during f64 download: {:?}",
                ret
            )));
        }
    }
    let _ = device; // keep device alive
    Ok(host_buf)
}

/// Download f32 complex GPU data to host as interleaved (re, im) f32 vec.
#[cfg(target_os = "linux")]
pub fn download_complex_f32(
    device: &Arc<CudaDevice>,
    gpu_ptr: *const CuComplex,
    num_elements: usize,
) -> Result<Vec<f32>> {
    if gpu_ptr.is_null() {
        return Err(MahoutError::InvalidInput(
            "download_complex_f32: null GPU pointer".to_string(),
        ));
    }

    let byte_count = num_elements * std::mem::size_of::<CuComplex>();
    let mut host_buf = vec![0.0_f32; num_elements * 2];

    unsafe {
        let ret = cudarc::driver::sys::lib().cuMemcpyDtoH_v2(
            host_buf.as_mut_ptr() as *mut _,
            gpu_ptr as u64,
            byte_count,
        );
        if ret != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(MahoutError::Cuda(format!(
                "cuMemcpyDtoH failed during f32 download: {:?}",
                ret
            )));
        }
    }
    let _ = device;
    Ok(host_buf)
}
