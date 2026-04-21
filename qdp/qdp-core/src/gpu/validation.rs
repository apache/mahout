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

//! GPU-side validation helpers shared across encoders.
//!
//! These wrap the atomic-flag validation kernels in `qdp-kernels/src/validation.cu`.
//! They allocate a 1-element i32 flag buffer, launch the relevant kernel, and copy
//! the flag back to the host, converting non-zero states into `MahoutError::InvalidInput`.

#![allow(unused_unsafe)]

use crate::error::{MahoutError, Result, cuda_error_to_string};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtrMut};
use std::ffi::c_void;
use std::sync::Arc;

// Error bitmask flags emitted by `check_basis_indices_kernel_*`.
// Must stay in sync with `qdp-kernels/src/validation.cu`.
const BASIS_IDX_ERR_NON_FINITE: i32 = 0x1;
const BASIS_IDX_ERR_NEGATIVE: i32 = 0x2;
const BASIS_IDX_ERR_NON_INTEGER: i32 = 0x4;
const BASIS_IDX_ERR_OUT_OF_RANGE: i32 = 0x8;

/// Assert that every value in a device-resident f32 buffer is finite.
///
/// # Safety
/// `input_d` must point to at least `total_values` `f32`s on `device`, and `stream` must
/// be either null or a valid CUDA stream associated with `device`.
pub unsafe fn assert_all_finite_f32(
    device: &Arc<CudaDevice>,
    input_d: *const f32,
    total_values: usize,
    stream: *mut c_void,
    context: &str,
) -> Result<()> {
    if total_values == 0 {
        return Ok(());
    }
    let mut flag = device.alloc_zeros::<i32>(1).map_err(|e| {
        MahoutError::MemoryAllocation(format!(
            "Failed to allocate finite-check flag buffer: {:?}",
            e
        ))
    })?;
    let ret = unsafe {
        qdp_kernels::launch_check_finite_batch_f32(
            input_d,
            total_values,
            *flag.device_ptr_mut() as *mut i32,
            stream,
        )
    };
    if ret != 0 {
        return Err(MahoutError::KernelLaunch(format!(
            "{}: finite validation kernel (f32) failed: {} ({})",
            context,
            ret,
            cuda_error_to_string(ret)
        )));
    }
    let host_flags = device.dtoh_sync_copy(&flag).map_err(|e| {
        MahoutError::Cuda(format!(
            "{}: failed to copy finite validation flag: {:?}",
            context, e
        ))
    })?;
    if host_flags.first().copied().unwrap_or_default() != 0 {
        return Err(MahoutError::InvalidInput(format!(
            "{}: batch contains non-finite values (NaN or Inf)",
            context
        )));
    }
    Ok(())
}

/// Assert that every value in a device-resident f64 buffer is finite.
///
/// # Safety
/// See `assert_all_finite_f32`.
pub unsafe fn assert_all_finite_f64(
    device: &Arc<CudaDevice>,
    input_d: *const f64,
    total_values: usize,
    stream: *mut c_void,
    context: &str,
) -> Result<()> {
    if total_values == 0 {
        return Ok(());
    }
    let mut flag = device.alloc_zeros::<i32>(1).map_err(|e| {
        MahoutError::MemoryAllocation(format!(
            "Failed to allocate finite-check flag buffer: {:?}",
            e
        ))
    })?;
    let ret = unsafe {
        qdp_kernels::launch_check_finite_batch_f64(
            input_d,
            total_values,
            *flag.device_ptr_mut() as *mut i32,
            stream,
        )
    };
    if ret != 0 {
        return Err(MahoutError::KernelLaunch(format!(
            "{}: finite validation kernel (f64) failed: {} ({})",
            context,
            ret,
            cuda_error_to_string(ret)
        )));
    }
    let host_flags = device.dtoh_sync_copy(&flag).map_err(|e| {
        MahoutError::Cuda(format!(
            "{}: failed to copy finite validation flag: {:?}",
            context, e
        ))
    })?;
    if host_flags.first().copied().unwrap_or_default() != 0 {
        return Err(MahoutError::InvalidInput(format!(
            "{}: batch contains non-finite values (NaN or Inf)",
            context
        )));
    }
    Ok(())
}

fn basis_index_error_message(flags: i32, state_len: usize) -> String {
    let mut reasons: Vec<&'static str> = Vec::new();
    if flags & BASIS_IDX_ERR_NON_FINITE != 0 {
        reasons.push("non-finite");
    }
    if flags & BASIS_IDX_ERR_NEGATIVE != 0 {
        reasons.push("negative");
    }
    if flags & BASIS_IDX_ERR_NON_INTEGER != 0 {
        reasons.push("non-integer");
    }
    if flags & BASIS_IDX_ERR_OUT_OF_RANGE != 0 {
        reasons.push("out of range");
    }
    format!(
        "Basis index batch contains invalid values ({}); valid indices must be finite, \
         non-negative integers in [0, {})",
        reasons.join(", "),
        state_len
    )
}

/// Validate a device-resident f32 basis-index buffer and cast it to `size_t`.
///
/// Returns a newly allocated device buffer holding the truncated indices.
/// Synchronizes on `stream` to make the error flag observable.
///
/// # Safety
/// See `assert_all_finite_f32`.
pub unsafe fn validate_and_cast_basis_indices_f32(
    device: &Arc<CudaDevice>,
    input_d: *const f32,
    num_samples: usize,
    state_len: usize,
    stream: *mut c_void,
) -> Result<CudaSlice<usize>> {
    if num_samples == 0 {
        return Err(MahoutError::InvalidInput(
            "Number of samples cannot be zero".into(),
        ));
    }
    let mut flag = device.alloc_zeros::<i32>(1).map_err(|e| {
        MahoutError::MemoryAllocation(format!(
            "Failed to allocate basis-index flag buffer: {:?}",
            e
        ))
    })?;
    let mut indices_out = device.alloc_zeros::<usize>(num_samples).map_err(|e| {
        MahoutError::MemoryAllocation(format!(
            "Failed to allocate basis-index cast buffer: {:?}",
            e
        ))
    })?;
    let ret = unsafe {
        qdp_kernels::launch_validate_and_cast_basis_indices_f32(
            input_d,
            num_samples,
            state_len,
            *indices_out.device_ptr_mut() as *mut usize,
            *flag.device_ptr_mut() as *mut i32,
            stream,
        )
    };
    if ret != 0 {
        return Err(MahoutError::KernelLaunch(format!(
            "Basis index validate+cast kernel (f32) failed: {} ({})",
            ret,
            cuda_error_to_string(ret)
        )));
    }
    let host_flags = device.dtoh_sync_copy(&flag).map_err(|e| {
        MahoutError::Cuda(format!(
            "Failed to copy basis-index validation flag: {:?}",
            e
        ))
    })?;
    let bits = host_flags.first().copied().unwrap_or_default();
    if bits != 0 {
        return Err(MahoutError::InvalidInput(basis_index_error_message(
            bits, state_len,
        )));
    }
    Ok(indices_out)
}

/// Assert that every value in a device-resident `usize` buffer is a valid
/// basis index (i.e. strictly less than `state_len`).
///
/// # Safety
/// See `assert_all_finite_f32`.
pub unsafe fn assert_basis_indices_in_range_usize(
    device: &Arc<CudaDevice>,
    indices_d: *const usize,
    num_samples: usize,
    state_len: usize,
    stream: *mut c_void,
) -> Result<()> {
    if num_samples == 0 {
        return Ok(());
    }
    let mut flag = device.alloc_zeros::<i32>(1).map_err(|e| {
        MahoutError::MemoryAllocation(format!(
            "Failed to allocate basis-index flag buffer: {:?}",
            e
        ))
    })?;
    let ret = unsafe {
        qdp_kernels::launch_check_basis_indices_usize(
            indices_d,
            num_samples,
            state_len,
            *flag.device_ptr_mut() as *mut i32,
            stream,
        )
    };
    if ret != 0 {
        return Err(MahoutError::KernelLaunch(format!(
            "Basis index bounds-check kernel failed: {} ({})",
            ret,
            cuda_error_to_string(ret)
        )));
    }
    let host_flags = device.dtoh_sync_copy(&flag).map_err(|e| {
        MahoutError::Cuda(format!(
            "Failed to copy basis-index validation flag: {:?}",
            e
        ))
    })?;
    let bits = host_flags.first().copied().unwrap_or_default();
    if bits != 0 {
        return Err(MahoutError::InvalidInput(basis_index_error_message(
            bits, state_len,
        )));
    }
    Ok(())
}
