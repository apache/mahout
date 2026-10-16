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

//! Device-side input validation.
//!
//! Each check queues a small kernel that raises a flag. Nothing here waits:
//! the flag is read back by the returned [`Pending`] once the caller has
//! synchronised the stream, so a check never breaks copy/compute overlap.
//!
//! Device code: `qdp-kernels/src/validation.cu`.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DevicePtrMut, sys};
use qdp_kernels::{LaunchConfig, config, kernel_args};

use crate::error::{MahoutError, Result};
use crate::gpu::kernels::{LaunchCtx, Pending, Real};

const BASIS_FLAG_NON_FINITE: i32 = 1 << 0;
const BASIS_FLAG_NEGATIVE: i32 = 1 << 1;
const BASIS_FLAG_NON_INTEGER: i32 = 1 << 2;
const BASIS_FLAG_OUT_OF_RANGE: i32 = 1 << 3;

/// A one-int flag, cleared on the launch stream so the check kernel that
/// follows on that stream sees the zero.
pub(crate) fn alloc_flag(ctx: &LaunchCtx, what: &str) -> Result<CudaSlice<i32>> {
    let mut flag = ctx.device.alloc_zeros::<i32>(1).map_err(|e| {
        MahoutError::MemoryAllocation(format!("Failed to allocate {what} flag buffer: {:?}", e))
    })?;
    // SAFETY: `flag` is a live 4-byte device allocation.
    unsafe {
        cudarc::driver::result::memset_d8_async(
            *flag.device_ptr_mut(),
            0,
            std::mem::size_of::<i32>(),
            ctx.stream as sys::CUstream,
        )
        .map_err(|e| MahoutError::Cuda(format!("Failed to clear {what} flag: {e}")))?;
    }
    Ok(flag)
}

/// Copy a flag back (after the stream has been synchronised).
pub(crate) fn read_flag(
    device: &Arc<CudaDevice>,
    flag: &CudaSlice<i32>,
    what: &str,
) -> Result<i32> {
    let host = device
        .dtoh_sync_copy(flag)
        .map_err(|e| MahoutError::Cuda(format!("Failed to copy {what} flag: {:?}", e)))?;
    Ok(host.first().copied().unwrap_or_default())
}

fn basis_index_error_message(flags: i32, state_len: usize) -> String {
    let mut reasons: Vec<&'static str> = Vec::new();
    if flags & BASIS_FLAG_NON_FINITE != 0 {
        reasons.push("non-finite");
    }
    if flags & BASIS_FLAG_NEGATIVE != 0 {
        reasons.push("negative");
    }
    if flags & BASIS_FLAG_NON_INTEGER != 0 {
        reasons.push("non-integer");
    }
    if flags & BASIS_FLAG_OUT_OF_RANGE != 0 {
        reasons.push("out of range");
    }
    format!(
        "Basis index batch contains invalid values ({}); valid indices must be finite, \
         non-negative integers in [0, {})",
        reasons.join(", "),
        state_len
    )
}

/// Queue a check that `input` holds no NaN or Inf.
///
/// # Safety
/// `input` must hold `total_values` elements valid on `ctx.stream`.
pub unsafe fn check_all_finite<T: Real>(
    ctx: &LaunchCtx,
    input: *const T,
    total_values: usize,
    context: &'static str,
) -> Result<Pending> {
    let mut pending = Pending::new();
    if total_values == 0 {
        return Ok(pending);
    }
    let mut flag = alloc_flag(ctx, "finite-check")?;
    let flag_ptr = *flag.device_ptr_mut() as *mut i32;
    // SAFETY: argument list matches `check_finite_batch_kernel_{f32,f64}`.
    unsafe {
        ctx.launch(
            "validation",
            crate::gpu::kernels::symbol(
                T::PRECISION,
                "check_finite_batch_kernel_f64",
                "check_finite_batch_kernel_f32",
            ),
            LaunchConfig::grid_stride(total_values, config::MAX_GRID_BLOCKS),
            &mut kernel_args![input, total_values, flag_ptr],
        )?;
    }
    pending.check(move |device| {
        if read_flag(device, &flag, "finite-check")? != 0 {
            return Err(MahoutError::InvalidInput(format!(
                "{}: batch contains non-finite values (NaN or Inf)",
                context
            )));
        }
        Ok(())
    });
    Ok(pending)
}

/// Queue validation and cast of float basis indices to `usize`. The returned
/// buffer is valid once the stream reaches the kernel; the flag verdict is
/// raised by the [`Pending`].
///
/// # Safety
/// `input` must hold `num_samples` elements valid on `ctx.stream`.
pub unsafe fn cast_basis_indices<T: Real>(
    ctx: &LaunchCtx,
    input: *const T,
    num_samples: usize,
    state_len: usize,
) -> Result<(CudaSlice<usize>, Pending)> {
    if num_samples == 0 {
        return Err(MahoutError::InvalidInput(
            "Number of samples cannot be zero".into(),
        ));
    }
    let mut flag = alloc_flag(ctx, "basis-index")?;
    let flag_ptr = *flag.device_ptr_mut() as *mut i32;
    let mut indices = ctx.device.alloc_zeros::<usize>(num_samples).map_err(|e| {
        MahoutError::MemoryAllocation(format!(
            "Failed to allocate basis-index cast buffer: {:?}",
            e
        ))
    })?;
    let indices_ptr = *indices.device_ptr_mut() as *mut usize;
    // SAFETY: both cast kernels share the (input, num_samples, state_len,
    // indices_out, error_flags) signature; only the input element type differs.
    unsafe {
        ctx.launch(
            "validation",
            crate::gpu::kernels::symbol(
                T::PRECISION,
                "validate_and_cast_basis_indices_kernel_f64",
                "validate_and_cast_basis_indices_kernel_f32",
            ),
            LaunchConfig::grid_stride(num_samples, config::MAX_GRID_BLOCKS),
            &mut kernel_args![input, num_samples, state_len, indices_ptr, flag_ptr],
        )?;
    }
    let mut pending = Pending::new();
    pending.check(move |device| {
        let bits = read_flag(device, &flag, "basis-index")?;
        if bits != 0 {
            return Err(MahoutError::InvalidInput(basis_index_error_message(
                bits, state_len,
            )));
        }
        Ok(())
    });
    Ok((indices, pending))
}

/// Queue a range check of integer basis indices against `state_len`.
///
/// # Safety
/// `indices` must hold `num_samples` elements valid on `ctx.stream`.
pub unsafe fn check_basis_indices_in_range(
    ctx: &LaunchCtx,
    indices: *const usize,
    num_samples: usize,
    state_len: usize,
) -> Result<Pending> {
    if num_samples == 0 {
        return Err(MahoutError::InvalidInput(
            "Number of samples cannot be zero".into(),
        ));
    }
    let mut flag = alloc_flag(ctx, "basis-index")?;
    let flag_ptr = *flag.device_ptr_mut() as *mut i32;
    // SAFETY: argument list matches `check_basis_indices_kernel_usize`.
    unsafe {
        ctx.launch(
            "validation",
            "check_basis_indices_kernel_usize",
            LaunchConfig::grid_stride(num_samples, config::MAX_GRID_BLOCKS),
            &mut kernel_args![indices, num_samples, state_len, flag_ptr],
        )?;
    }
    let mut pending = Pending::new();
    pending.check(move |device| {
        let bits = read_flag(device, &flag, "basis-index")?;
        if bits != 0 {
            return Err(MahoutError::InvalidInput(basis_index_error_message(
                bits, state_len,
            )));
        }
        Ok(())
    });
    Ok(pending)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_message_lists_every_failure() {
        let msg = basis_index_error_message(BASIS_FLAG_NON_FINITE | BASIS_FLAG_NEGATIVE, 4);
        assert!(msg.contains("non-finite, negative"), "{msg}");
        assert!(msg.contains("[0, 4)"), "{msg}");
        assert!(basis_index_error_message(BASIS_FLAG_NON_INTEGER, 4).contains("non-integer"));
        assert!(basis_index_error_message(BASIS_FLAG_OUT_OF_RANGE, 4).contains("out of range"));
    }
}
